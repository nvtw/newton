#!/usr/bin/env python
"""Tests for the reduction hash table."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.hashtable_reduction import (
    HASHTABLE_EMPTY_KEY,
    ReductionHashTable,
    hashtable_insert_slot,
)


class TestReductionHashTable(unittest.TestCase):
    """Test cases for the ReductionHashTable class."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_basic_creation(self):
        """Test creating an empty hash table."""
        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        self.assertGreaterEqual(ht.capacity, 64)
        self.assertEqual(ht.values_per_key, 13)
        # Check that keys are initialized to empty
        keys_np = ht.keys.numpy()
        self.assertTrue(np.all(keys_np == 0xFFFFFFFFFFFFFFFF))

    def test_power_of_two_rounding(self):
        """Test that capacity is rounded to power of two."""
        ht1 = ReductionHashTable(capacity=100, values_per_key=1, device="cpu")
        self.assertEqual(ht1.capacity, 128)  # Next power of 2

        ht2 = ReductionHashTable(capacity=64, values_per_key=1, device="cpu")
        self.assertEqual(ht2.capacity, 64)  # Already power of 2

        ht3 = ReductionHashTable(capacity=1, values_per_key=1, device="cpu")
        self.assertEqual(ht3.capacity, 1)

    def test_values_array_size(self):
        """Test that values array is capacity * values_per_key."""
        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        self.assertEqual(ht.values.shape[0], ht.capacity * 13)

    def test_insert_single_slot(self):
        """Test inserting values into different slots of the same key."""

        @wp.kernel
        def insert_test_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            # Insert into slot 0
            hashtable_insert_slot(
                wp.uint64(123), 0, wp.uint64(100),
                keys, values, active_slots, values_per_key
            )
            # Insert into slot 5
            hashtable_insert_slot(
                wp.uint64(123), 5, wp.uint64(200),
                keys, values, active_slots, values_per_key
            )
            # Insert into slot 12
            hashtable_insert_slot(
                wp.uint64(123), 12, wp.uint64(300),
                keys, values, active_slots, values_per_key
            )

        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        wp.launch(
            insert_test_kernel,
            dim=1,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Find the entry
        keys_np = ht.keys.numpy()
        values_np = ht.values.numpy()

        entry_idx = np.where(keys_np == 123)[0]
        self.assertEqual(len(entry_idx), 1)
        idx = entry_idx[0]

        # Check values at each slot
        self.assertEqual(values_np[idx * 13 + 0], 100)
        self.assertEqual(values_np[idx * 13 + 5], 200)
        self.assertEqual(values_np[idx * 13 + 12], 300)

    def test_atomic_max_behavior(self):
        """Test that atomic max correctly keeps the maximum value."""

        @wp.kernel
        def atomic_max_test_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            # All threads try to write to same key and slot
            # Values are 1, 2, 3, ..., 100
            hashtable_insert_slot(
                wp.uint64(999), 0, wp.uint64(tid + 1),
                keys, values, active_slots, values_per_key
            )

        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        wp.launch(
            atomic_max_test_kernel,
            dim=100,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Find the entry
        keys_np = ht.keys.numpy()
        values_np = ht.values.numpy()

        entry_idx = np.where(keys_np == 999)[0]
        self.assertEqual(len(entry_idx), 1)
        idx = entry_idx[0]

        # The maximum value should be 100
        self.assertEqual(values_np[idx * 13 + 0], 100)

    def test_multiple_keys(self):
        """Test inserting multiple different keys."""

        @wp.kernel
        def multi_key_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            key = wp.uint64(tid + 1)  # Keys 1, 2, 3, ...
            value = wp.uint64((tid + 1) * 10)  # Values 10, 20, 30, ...
            hashtable_insert_slot(key, 0, value, keys, values, active_slots, values_per_key)

        ht = ReductionHashTable(capacity=256, values_per_key=1, device="cpu")
        wp.launch(
            multi_key_kernel,
            dim=100,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Check that we have 100 entries
        keys_np = ht.keys.numpy()
        non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
        self.assertEqual(np.sum(non_empty), 100)

        # Check active slots count
        active_count = ht.active_slots.numpy()[ht.capacity]
        self.assertEqual(active_count, 100)

    def test_clear(self):
        """Test clearing the hash table."""

        @wp.kernel
        def insert_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            hashtable_insert_slot(
                wp.uint64(tid + 1), 0, wp.uint64(tid * 10),
                keys, values, active_slots, values_per_key
            )

        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")

        # Insert some data
        wp.launch(
            insert_kernel,
            dim=50,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Verify data exists
        keys_np = ht.keys.numpy()
        non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
        self.assertEqual(np.sum(non_empty), 50)

        # Clear
        ht.clear()

        # Verify table is empty
        keys_np = ht.keys.numpy()
        self.assertTrue(np.all(keys_np == 0xFFFFFFFFFFFFFFFF))
        self.assertTrue(np.all(ht.values.numpy() == 0))
        self.assertTrue(np.all(ht.active_slots.numpy() == 0))

    def test_clear_active(self):
        """Test clearing only active entries."""

        @wp.kernel
        def insert_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            hashtable_insert_slot(
                wp.uint64(tid + 1), 0, wp.uint64(tid * 10),
                keys, values, active_slots, values_per_key
            )

        ht = ReductionHashTable(capacity=256, values_per_key=13, device="cpu")

        # Insert some data (sparse - only 20 entries in a 256-capacity table)
        wp.launch(
            insert_kernel,
            dim=20,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Verify data exists
        active_count = ht.active_slots.numpy()[ht.capacity]
        self.assertEqual(active_count, 20)

        # Clear active
        ht.clear_active()

        # Verify table is empty
        keys_np = ht.keys.numpy()
        # After clear_active, the keys that were used should be empty
        non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
        self.assertEqual(np.sum(non_empty), 0)

        # Active count should be 0
        active_count = ht.active_slots.numpy()[ht.capacity]
        self.assertEqual(active_count, 0)

    def test_high_collision(self):
        """Test with many threads competing for same keys."""

        @wp.kernel
        def collision_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            # Only 10 unique keys, but 1000 threads
            key = wp.uint64(tid % 10)
            slot = tid % 13
            value = wp.uint64(tid)
            hashtable_insert_slot(key, slot, value, keys, values, active_slots, values_per_key)

        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        wp.launch(
            collision_kernel,
            dim=1000,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Should have exactly 10 unique keys
        keys_np = ht.keys.numpy()
        non_empty = keys_np != 0xFFFFFFFFFFFFFFFF
        self.assertEqual(np.sum(non_empty), 10)

        # Active count should be 10
        active_count = ht.active_slots.numpy()[ht.capacity]
        self.assertEqual(active_count, 10)

    def test_early_exit_optimization(self):
        """Test that the early exit optimization works correctly.
        
        When a smaller value tries to update a slot that already has a larger value,
        it should skip the atomic operation but still return True.
        """

        @wp.kernel
        def insert_descending_kernel(
            keys: wp.array(dtype=wp.uint64),
            values: wp.array(dtype=wp.uint64),
            active_slots: wp.array(dtype=wp.int32),
            values_per_key: int,
        ):
            tid = wp.tid()
            # Insert values in descending order: 999, 998, 997, ...
            value = wp.uint64(999 - tid)
            hashtable_insert_slot(
                wp.uint64(1), 0, value,
                keys, values, active_slots, values_per_key
            )

        ht = ReductionHashTable(capacity=64, values_per_key=13, device="cpu")
        wp.launch(
            insert_descending_kernel,
            dim=1000,
            inputs=[ht.keys, ht.values, ht.active_slots, ht.values_per_key],
            device="cpu",
        )
        wp.synchronize()

        # Find the entry
        keys_np = ht.keys.numpy()
        values_np = ht.values.numpy()

        entry_idx = np.where(keys_np == 1)[0]
        self.assertEqual(len(entry_idx), 1)
        idx = entry_idx[0]

        # The maximum value should be 999 (first insertion)
        self.assertEqual(values_np[idx * 13 + 0], 999)


if __name__ == "__main__":
    unittest.main()

