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
        # Optimization: Read first to avoid atomic if key already exists
        # This is safe because keys only transition from EMPTY -> KEY, never change afterwards
        stored_key = keys[idx]

        if stored_key == key:
            # Key matches - write value
            value_idx = idx * values_per_key + slot_id
            if values[value_idx] < value:
                wp.atomic_max(values, value_idx, value)
            return True

        if stored_key == HASHTABLE_EMPTY_KEY:
            # Try to claim
            old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)

            if old_key == HASHTABLE_EMPTY_KEY:
                # We claimed an empty slot - this is a NEW entry
                # Add to active slots list
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                # Write to the specific value slot (no check needed - slot is fresh)
                value_idx = idx * values_per_key + slot_id
                wp.atomic_max(values, value_idx, value)
                return True
            elif old_key == key:
                # Key already exists - check if our value is larger before atomic
                value_idx = idx * values_per_key + slot_id
                if values[value_idx] < value:
                    wp.atomic_max(values, value_idx, value)
                return True

        # Collision with different key - linear probe to next slot
        idx = (idx + 1) & capacity_mask

    # Table is full
    return False


@wp.kernel
def _hashtable_clear_active_kernel(
    keys: wp.array(dtype=wp.uint64),
    values: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
    values_per_key: int,
):
    """Kernel to clear only the active slots in the hash table."""
    tid = wp.tid()

    # Read count from GPU - stored at active_slots[capacity]
    count = active_slots[capacity]

    if tid < count:
        slot_idx = active_slots[tid]
        keys[slot_idx] = HASHTABLE_EMPTY_KEY
        # Clear all value slots for this entry
        value_base = slot_idx * values_per_key
        for i in range(values_per_key):
            values[value_base + i] = wp.uint64(0)


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
        """Clear only the active entries. CUDA graph capture compatible."""
        wp.launch(
            _hashtable_clear_active_kernel,
            dim=self.capacity,
            inputs=[self.keys, self.values, self.active_slots, self.capacity, self.values_per_key],
            device=self.device,
        )
        self.active_slots.zero_()

