# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-friendly hash table with thread-safe insertion and atomic max value updates.

This module provides a simple open-addressing hash table that can be used in Warp kernels.
The hash table supports:
- Thread-safe insertion of (key, value) pairs where both are uint64
- When a key already exists, the value is atomically maxed with the existing value
- Open addressing with linear probing for collision resolution

Example usage:

    # Create the hash table
    ht = HashTable(capacity=1024, device="cuda:0")

    # Use in a kernel
    @wp.kernel
    def my_kernel(
        keys: wp.array(dtype=wp.uint64),
        values: wp.array(dtype=wp.uint64),
        ht_keys: wp.array(dtype=wp.uint64),
        ht_values: wp.array(dtype=wp.uint64),
        ht_capacity: int,
    ):
        tid = wp.tid()
        hashtable_insert(keys[tid], values[tid], ht_keys, ht_values, ht_capacity)

    # Launch the kernel
    wp.launch(my_kernel, dim=n, inputs=[keys, values, ht.keys, ht.values, ht.capacity])

    # Read results back
    entries = ht.get_entries()  # Returns list of (key, value) tuples
"""

from __future__ import annotations

import warp as wp

# Sentinel value for empty slots (max uint64 value, unlikely to be a valid key)
_HASHTABLE_EMPTY_KEY_VALUE = 0xFFFFFFFFFFFFFFFF
HASHTABLE_EMPTY_KEY = wp.constant(wp.uint64(_HASHTABLE_EMPTY_KEY_VALUE))


def _next_power_of_two(n: int) -> int:
    """Round up to the next power of two.

    Args:
        n: The input value (must be positive)

    Returns:
        The smallest power of two >= n
    """
    if n <= 0:
        return 1
    # Handle the case where n is already a power of two
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@wp.func
def _hashtable_hash(key: wp.uint64, capacity_mask: int) -> int:
    """Compute hash index for a key using a simple but effective hash function.

    Uses the FNV-1a inspired mixing to distribute keys well across the table.

    Args:
        key: The uint64 key to hash
        capacity_mask: The capacity mask (capacity - 1) for fast modulo via bitwise AND

    Returns:
        The hash index in range [0, capacity)
    """
    # FNV-1a style mixing for good distribution
    h = key
    h = h ^ (h >> wp.uint64(33))
    h = h * wp.uint64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> wp.uint64(33))
    h = h * wp.uint64(0xC4CEB9FE1A85EC53)
    h = h ^ (h >> wp.uint64(33))
    return int(h) & capacity_mask


@wp.func
def hashtable_insert(
    key: wp.uint64,
    value: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
) -> bool:
    """Insert or update a key-value pair in the hash table.

    This function is thread-safe and can be called from multiple threads concurrently.
    If the key doesn't exist, it inserts the new key-value pair.
    If the key already exists, it atomically updates the value to max(existing, new).

    Uses open addressing with linear probing for collision resolution.
    The keys/values arrays must have a power-of-two length.

    Args:
        key: The uint64 key to insert
        value: The uint64 value to insert or max with
        keys: The hash table keys array (length must be power of two)
        values: The hash table values array (same length as keys)

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    # Linear probing with a maximum of 'capacity' attempts
    for _i in range(capacity):
        # Try to claim this slot with our key
        old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

        if old_key == HASHTABLE_EMPTY_KEY:
            # We claimed an empty slot - write our value
            wp.atomic_max(values, idx, value)
            return True
        elif old_key == key:
            # Key already exists - atomic max the value
            wp.atomic_max(values, idx, value)
            return True

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return False


@wp.func
def hashtable_insert_with_index(
    key: wp.uint64,
    value: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
) -> bool:
    """Insert or update a key-value pair, tracking active slot indices.

    Same as hashtable_insert, but also maintains a compact list of active slot indices.
    When a NEW key is inserted (not an update), the slot index is added to active_slots.
    The last element of active_slots (at index = capacity) stores the count.

    Args:
        key: The uint64 key to insert
        value: The uint64 value to insert or max with
        keys: The hash table keys array (length must be power of two)
        values: The hash table values array (same length as keys)
        active_slots: Array of size (capacity + 1) tracking active slot indices.
                      active_slots[capacity] is the count of active slots.

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    # Linear probing with a maximum of 'capacity' attempts
    for _i in range(capacity):
        # Try to claim this slot with our key
        old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

        if old_key == HASHTABLE_EMPTY_KEY:
            # We claimed an empty slot - this is a NEW entry
            # Add to active slots list
            active_idx = wp.atomic_add(active_slots, capacity, 1)
            if active_idx < capacity:
                active_slots[active_idx] = idx
            # Write our value
            wp.atomic_max(values, idx, value)
            return True
        elif old_key == key:
            # Key already exists - just atomic max the value (no new entry)
            wp.atomic_max(values, idx, value)
            return True

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return False


@wp.func
def hashtable_insert_slot(
    key: wp.uint64,
    slot_id: int,
    value: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    values_per_key: int,
) -> bool:
    """Insert or update a value in a specific slot for a key.

    Each key has `values_per_key` value slots. This function writes to the
    specified slot_id within the entry. Different threads can write to different
    slots of the same key concurrently.

    Args:
        key: The uint64 key to insert
        slot_id: Which value slot to write to (0 to values_per_key-1)
        value: The uint64 value to insert or max with
        keys: The hash table keys array (length must be power of two)
        values: The hash table values array (length = keys.length * values_per_key)
        active_slots: Array of size (capacity + 1) tracking active entry indices.
                      active_slots[capacity] is the count of active entries.
        values_per_key: Number of value slots per key

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    # Linear probing with a maximum of 'capacity' attempts
    for _i in range(capacity):
        # Try to claim this slot with our key
        old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

        if old_key == HASHTABLE_EMPTY_KEY:
            # We claimed an empty slot - this is a NEW entry
            # Add to active slots list
            active_idx = wp.atomic_add(active_slots, capacity, 1)
            if active_idx < capacity:
                active_slots[active_idx] = idx
            # Write to the specific value slot
            value_idx = idx * values_per_key + slot_id
            wp.atomic_max(values, value_idx, value)
            return True
        elif old_key == key:
            # Key already exists - write to the specific value slot
            value_idx = idx * values_per_key + slot_id
            wp.atomic_max(values, value_idx, value)
            return True

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return False


@wp.func
def hashtable_lookup(
    key: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
) -> wp.uint64:
    """Look up a key in the hash table.

    The keys/values arrays must have a power-of-two length.

    Args:
        key: The uint64 key to look up
        keys: The hash table keys array (length must be power of two)
        values: The hash table values array (same length as keys)

    Returns:
        The value associated with the key, or HASHTABLE_EMPTY_KEY if not found
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    for _i in range(capacity):
        stored_key = keys[idx]

        if stored_key == key:
            return values[idx]

        if stored_key == HASHTABLE_EMPTY_KEY:
            # Empty slot means key not in table
            return HASHTABLE_EMPTY_KEY

        # Linear probe
        idx = (idx + 1) & capacity_mask

    return HASHTABLE_EMPTY_KEY


@wp.kernel
def _hashtable_clear_active_kernel(
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
    values_per_key: int,
):
    """Kernel to clear only the active slots in the hash table.

    Reads count from GPU memory (active_slots[capacity]) to avoid CPU sync.
    Each thread checks if its index is within the active count.
    """
    tid = wp.tid()

    # Read count from GPU - stored at active_slots[capacity]
    count = active_slots[capacity]

    # Only clear if this thread is within the active range
    if tid < count:
        slot_idx = active_slots[tid]
        keys[slot_idx] = HASHTABLE_EMPTY_KEY
        # Clear all value slots for this entry
        value_base = slot_idx * values_per_key
        for i in range(values_per_key):
            values[value_base + i] = wp.uint64(0)


class HashTable:
    """A GPU-friendly hash table with thread-safe insertion and atomic max value updates.

    This hash table uses open addressing with linear probing. It's designed for
    use in Warp GPU kernels where multiple threads may insert entries concurrently.

    When inserting a key that already exists, the value is atomically maxed with
    the existing value (i.e., the table stores the maximum value seen for each key).

    The capacity is automatically rounded up to the next power of two for
    efficient modulo operations via bitwise AND.

    Supports multiple values per key via the `values_per_key` parameter. When using
    multiple values per key, use `hashtable_insert_slot()` to write to specific slots.

    Attributes:
        capacity: The maximum number of entries the table can hold (power of two)
        capacity_mask: The capacity mask (capacity - 1) for fast modulo
        values_per_key: Number of value slots per key (default 1)
        keys: Warp array storing the keys
        values: Warp array storing the values (length = capacity * values_per_key)
        active_slots: Compact array tracking active slot indices. Size is (capacity + 1).
                      active_slots[0:count] contains the indices of occupied slots.
                      active_slots[capacity] is the count of active entries.
        device: The device where the table is allocated
    """

    def __init__(self, capacity: int, device: str | None = None, values_per_key: int = 1):
        """Initialize an empty hash table.

        Args:
            capacity: Maximum number of unique keys the table can store.
                      Will be rounded up to the next power of two.
                      Should be larger than the expected number of entries
                      (recommended: 2x expected entries for good performance).
            device: The Warp device to allocate on (e.g., "cuda:0", "cpu").
                    If None, uses the default device.
            values_per_key: Number of value slots per key. Use with hashtable_insert_slot()
                           for multi-value entries. Default is 1.
        """
        # Round up to next power of two for efficient modulo via bitwise AND
        self.capacity = _next_power_of_two(capacity)
        self.capacity_mask = self.capacity - 1
        self.values_per_key = values_per_key
        self.device = device

        # Allocate arrays and initialize keys to empty sentinel
        self.keys = wp.zeros(self.capacity, dtype=wp.uint64, device=device)
        # Values array is capacity * values_per_key
        self.values = wp.zeros(self.capacity * values_per_key, dtype=wp.uint64, device=device)

        # Active slots array: indices of occupied slots + count at the end
        # Size is capacity + 1, where active_slots[capacity] is the count
        self.active_slots = wp.zeros(self.capacity + 1, dtype=wp.int32, device=device)

        # Fill keys with empty sentinel
        self.clear()

    def clear(self):
        """Clear the hash table, removing all entries.

        This clears all slots. For sparse tables, use clear_active() instead
        which only clears the slots that were actually used.
        """
        self.keys.fill_(_HASHTABLE_EMPTY_KEY_VALUE)
        self.values.zero_()
        self.active_slots.zero_()

    def clear_active(self):
        """Clear only the active entries in the hash table.

        This is more efficient than clear() when the table is sparsely populated,
        as it only resets the slots that were actually used.

        This method is CUDA graph capture compatible - no CPU synchronization.
        """
        # Launch kernel with capacity threads - kernel reads count from GPU and checks bounds
        wp.launch(
            _hashtable_clear_active_kernel,
            dim=self.capacity,
            inputs=[self.keys, self.values, self.active_slots, self.capacity, self.values_per_key],
            device=self.device,
        )

        # Reset the active_slots array (including count at the end)
        self.active_slots.zero_()

    def get_active_count(self) -> int:
        """Get the number of active entries in the hash table.

        This is O(1) when using hashtable_insert_with_index.

        Returns:
            The number of active entries.
        """
        return int(self.active_slots.numpy()[self.capacity])

    def get_num_entries(self) -> int:
        """Count the number of entries in the hash table.

        Returns:
            The number of non-empty entries.
        """
        keys_np = self.keys.numpy()
        return int((keys_np != _HASHTABLE_EMPTY_KEY_VALUE).sum())

    def get_entries(self) -> list[tuple[int, int]]:
        """Extract all entries from the hash table.

        Returns:
            A list of (key, value) tuples for all non-empty entries.
        """
        keys_np = self.keys.numpy()
        values_np = self.values.numpy()

        entries = []
        for i in range(self.capacity):
            if keys_np[i] != _HASHTABLE_EMPTY_KEY_VALUE:
                entries.append((int(keys_np[i]), int(values_np[i])))

        return entries

    def get_entries_arrays(self) -> tuple[wp.array, wp.array, int]:
        """Extract all entries as compact Warp arrays.

        This is useful when you need to process the entries in another kernel.

        Returns:
            Tuple of (keys_array, values_array, num_entries) where the arrays
            contain only the non-empty entries.
        """
        keys_np = self.keys.numpy()
        values_np = self.values.numpy()

        mask = keys_np != _HASHTABLE_EMPTY_KEY_VALUE
        compact_keys = keys_np[mask]
        compact_values = values_np[mask]

        if len(compact_keys) == 0:
            return (
                wp.empty(0, dtype=wp.uint64, device=self.device),
                wp.empty(0, dtype=wp.uint64, device=self.device),
                0,
            )

        return (
            wp.array(compact_keys, dtype=wp.uint64, device=self.device),
            wp.array(compact_values, dtype=wp.uint64, device=self.device),
            len(compact_keys),
        )


# Kernel for batch insertion (useful for testing and simple use cases)
@wp.kernel
def _hashtable_batch_insert_kernel(
    input_keys: wp.array(dtype=wp.uint64),
    input_values: wp.array(dtype=wp.uint64),
    table_keys: wp.array(dtype=wp.uint64),
    table_values: wp.array(dtype=wp.uint64),
):
    """Kernel to batch insert key-value pairs into the hash table."""
    tid = wp.tid()
    hashtable_insert(input_keys[tid], input_values[tid], table_keys, table_values)


def batch_insert(
    table: HashTable,
    keys: wp.array,
    values: wp.array,
):
    """Batch insert key-value pairs into a hash table.

    This is a convenience function for inserting multiple entries at once.

    Args:
        table: The HashTable to insert into
        keys: Array of uint64 keys to insert
        values: Array of uint64 values to insert
    """
    assert keys.shape[0] == values.shape[0], "Keys and values must have the same length"
    n = keys.shape[0]
    if n == 0:
        return

    wp.launch(
        _hashtable_batch_insert_kernel,
        dim=n,
        inputs=[keys, values, table.keys, table.values],
        device=table.device,
    )

