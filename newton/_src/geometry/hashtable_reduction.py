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

"""GPU-friendly hash table optimized for contact reduction.

This module provides a specialized hash table for tracking best contacts
during global contact reduction. Key features:
- Thread-safe insertion with atomic max for selecting best contacts
- Multiple values per key (one per reduction slot)
- Active slot tracking for efficient clearing
- Power-of-two capacity for fast modulo via bitwise AND
"""

from __future__ import annotations

import warp as wp

# Sentinel value for empty slots
_HASHTABLE_EMPTY_KEY_VALUE = 0xFFFFFFFFFFFFFFFF
HASHTABLE_EMPTY_KEY = wp.constant(wp.uint64(_HASHTABLE_EMPTY_KEY_VALUE))


def _next_power_of_two(n: int) -> int:
    """Round up to the next power of two."""
    if n <= 0:
        return 1
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
    """Compute hash index using a simplified mixer."""
    h = key
    h = h ^ (h >> wp.uint64(33))
    h = h * wp.uint64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> wp.uint64(33))
    return int(h) & capacity_mask


@wp.func
def hashtable_find_or_insert(
    key: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
) -> int:
    """Find or insert a key and return the entry index.

    This function locates an existing entry or creates a new one for the key.
    Once you have the entry index, use hashtable_update_slot_direct() to
    write values to specific slots without repeated hash lookups.

    Args:
        key: The uint64 key to find or insert
        keys: The hash table keys array (length must be power of two)
        active_slots: Array of size (capacity + 1) tracking active entry indices.
                      active_slots[capacity] is the count of active entries.

    Returns:
        Entry index (>= 0) if successful, -1 if the table is full
    """
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    # Linear probing with a maximum of 'capacity' attempts
    for _i in range(capacity):
        # Read first to check if key exists (keys only transition EMPTY -> KEY)
        stored_key = keys[idx]

        if stored_key == key:
            # Key already exists - return its index
            return idx

        if stored_key == HASHTABLE_EMPTY_KEY:
            # Try to claim this slot
            old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

            if old_key == HASHTABLE_EMPTY_KEY:
                # We claimed an empty slot - this is a NEW entry
                # Add to active slots list
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                return idx
            elif old_key == key:
                # Another thread just inserted the same key - use it
                return idx
            # else: Another thread claimed with different key - continue probing

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return -1


@wp.func
def hashtable_update_slot_direct(
    entry_idx: int,
    slot_id: int,
    value: wp.uint64,
    values: wp.array(dtype=wp.uint64),
    values_per_key: int,
):
    """Update a value slot directly using the entry index.

    Use this after hashtable_find_or_insert() to write multiple values
    to the same entry without repeated hash lookups.

    Args:
        entry_idx: Entry index from hashtable_find_or_insert()
        slot_id: Which value slot to write to (0 to values_per_key-1)
        value: The uint64 value to max with existing value
        values: The hash table values array
        values_per_key: Number of value slots per key
    """
    value_idx = entry_idx * values_per_key + slot_id
    # Check before atomic to reduce contention
    if values[value_idx] < value:
        wp.atomic_max(values, value_idx, value)


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
    slots of the same key concurrently using atomic_max.

    For inserting multiple values to the same key, prefer using
    hashtable_find_or_insert() + hashtable_update_slot_direct() to avoid
    repeated hash lookups.

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
    entry_idx = hashtable_find_or_insert(key, keys, active_slots)
    if entry_idx < 0:
        return False
    hashtable_update_slot_direct(entry_idx, slot_id, value, values, values_per_key)
    return True


@wp.kernel
def _hashtable_clear_active_kernel(
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
    values_per_key: int,
    num_threads: int,
):
    """Kernel to clear only the active slots in the hash table.

    Uses grid-stride loop for efficient thread utilization.
    Reads count from GPU memory - works because all threads read before any writes.
    """
    tid = wp.tid()

    # Read count from GPU - stored at active_slots[capacity]
    # All threads read this value before any modifications happen
    count = active_slots[capacity]

    # Grid-stride loop: each thread processes multiple entries if needed
    i = tid
    while i < count:
        slot_idx = active_slots[i]
        keys[slot_idx] = HASHTABLE_EMPTY_KEY
        # Clear all value slots for this entry
        value_base = slot_idx * values_per_key
        for j in range(values_per_key):
            values[value_base + j] = wp.uint64(0)
        i += num_threads


@wp.kernel
def _zero_count_kernel(
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Zero the count element after clearing."""
    active_slots[capacity] = 0


class ReductionHashTable:
    """Hash table optimized for contact reduction.

    Uses open addressing with linear probing. Designed for GPU kernels
    where many threads insert concurrently. Supports multiple values per key
    for storing reduction slots (one per direction/beta combination).

    Attributes:
        capacity: Maximum number of unique keys (power of two)
        values_per_key: Number of value slots per key
        keys: Warp array storing the keys
        values: Warp array storing the values (length = capacity * values_per_key)
        active_slots: Array tracking active slot indices (size = capacity + 1)
        device: The device where the table is allocated
    """

    def __init__(self, capacity: int, values_per_key: int, device: str | None = None):
        """Initialize an empty hash table.

        Args:
            capacity: Maximum number of unique keys. Rounded up to power of two.
            values_per_key: Number of value slots per key.
            device: Warp device (e.g., "cuda:0", "cpu").
        """
        self.capacity = _next_power_of_two(capacity)
        self.values_per_key = values_per_key
        self.device = device

        # Allocate arrays
        self.keys = wp.zeros(self.capacity, dtype=wp.uint64, device=device)
        self.values = wp.zeros(self.capacity * values_per_key, dtype=wp.uint64, device=device)
        self.active_slots = wp.zeros(self.capacity + 1, dtype=wp.int32, device=device)

        self.clear()

    def clear(self):
        """Clear all entries in the hash table."""
        self.keys.fill_(_HASHTABLE_EMPTY_KEY_VALUE)
        self.values.zero_()
        self.active_slots.zero_()

    def clear_active(self):
        """Clear only the active entries. CUDA graph capture compatible.

        Uses two kernel launches:
        1. Clear all active hashtable entries (keys + values) using grid-stride loop
        2. Zero the count element

        The two-kernel approach is needed to avoid race conditions on CPU where
        threads execute sequentially.
        """
        # Use fixed thread count for efficient GPU utilization
        # Grid-stride loop handles any number of active entries
        num_threads = min(1024, self.capacity)
        wp.launch(
            _hashtable_clear_active_kernel,
            dim=num_threads,
            inputs=[self.keys, self.values, self.active_slots, self.capacity, self.values_per_key, num_threads],
            device=self.device,
        )
        # Zero the count in a separate kernel to avoid CPU race condition
        wp.launch(
            _zero_count_kernel,
            dim=1,
            inputs=[self.active_slots, self.capacity],
            device=self.device,
        )

