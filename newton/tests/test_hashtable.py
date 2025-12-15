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

"""Tests for the GPU hash table implementation."""

import unittest

import numpy as np
import warp as wp

from newton._src.core.hashtable import (
    HASHTABLE_EMPTY_KEY,
    HashTable,
    batch_insert,
    hashtable_insert,
)


class TestHashTable(unittest.TestCase):
    """Test cases for the HashTable class."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once for all tests."""
        wp.init()

    def test_basic_insertion(self):
        """Test basic insertion of unique keys."""
        ht = HashTable(capacity=64, device="cpu")

        keys = wp.array([1, 2, 3, 4, 5], dtype=wp.uint64, device="cpu")
        values = wp.array([10, 20, 30, 40, 50], dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)

        entries = ht.get_entries()
        self.assertEqual(len(entries), 5)

        entries_dict = dict(entries)
        self.assertEqual(entries_dict[1], 10)
        self.assertEqual(entries_dict[2], 20)
        self.assertEqual(entries_dict[3], 30)
        self.assertEqual(entries_dict[4], 40)
        self.assertEqual(entries_dict[5], 50)

    def test_max_value_update(self):
        """Test that duplicate keys get their values maxed."""
        ht = HashTable(capacity=64, device="cpu")

        # Insert same key multiple times with different values
        keys = wp.array([1, 1, 1, 1], dtype=wp.uint64, device="cpu")
        values = wp.array([10, 50, 30, 20], dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)

        entries = ht.get_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0], 1)
        self.assertEqual(entries[0][1], 50)  # Max of 10, 50, 30, 20

    def test_mixed_insertion(self):
        """Test insertion with both unique keys and duplicates."""
        ht = HashTable(capacity=64, device="cpu")

        # Mix of unique keys and duplicates
        keys = wp.array([1, 2, 1, 3, 2, 1], dtype=wp.uint64, device="cpu")
        values = wp.array([10, 20, 30, 40, 50, 5], dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)

        entries = ht.get_entries()
        self.assertEqual(len(entries), 3)

        entries_dict = dict(entries)
        self.assertEqual(entries_dict[1], 30)  # Max of 10, 30, 5
        self.assertEqual(entries_dict[2], 50)  # Max of 20, 50
        self.assertEqual(entries_dict[3], 40)  # Only one value

    def test_clear(self):
        """Test that clear removes all entries."""
        ht = HashTable(capacity=64, device="cpu")

        keys = wp.array([1, 2, 3], dtype=wp.uint64, device="cpu")
        values = wp.array([10, 20, 30], dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)
        self.assertEqual(ht.get_num_entries(), 3)

        ht.clear()
        self.assertEqual(ht.get_num_entries(), 0)
        self.assertEqual(len(ht.get_entries()), 0)

    def test_large_scale_concurrent(self):
        """Test with many concurrent insertions to verify thread safety."""
        ht = HashTable(capacity=4096, device="cpu")

        # Create 1000 insertions with 100 unique keys (10 duplicates each)
        n_keys = 100
        n_duplicates = 10
        n_total = n_keys * n_duplicates

        np.random.seed(42)
        keys_np = np.repeat(np.arange(n_keys, dtype=np.uint64), n_duplicates)
        values_np = np.random.randint(1, 1000, size=n_total, dtype=np.uint64)

        # Shuffle to simulate random insertion order
        indices = np.random.permutation(n_total)
        keys_np = keys_np[indices]
        values_np = values_np[indices]

        keys = wp.array(keys_np, dtype=wp.uint64, device="cpu")
        values = wp.array(values_np, dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)

        entries = ht.get_entries()
        self.assertEqual(len(entries), n_keys)

        # Verify each key has the max value
        entries_dict = dict(entries)
        for k in range(n_keys):
            mask = keys_np == k
            expected_max = values_np[mask].max()
            self.assertEqual(entries_dict[k], expected_max)

    def test_get_entries_arrays(self):
        """Test getting entries as compact Warp arrays."""
        ht = HashTable(capacity=64, device="cpu")

        keys = wp.array([1, 2, 3], dtype=wp.uint64, device="cpu")
        values = wp.array([10, 20, 30], dtype=wp.uint64, device="cpu")

        batch_insert(ht, keys, values)

        out_keys, out_values, num_entries = ht.get_entries_arrays()

        self.assertEqual(num_entries, 3)
        self.assertEqual(out_keys.shape[0], 3)
        self.assertEqual(out_values.shape[0], 3)

        # Check the entries match
        entries_dict = {}
        for i in range(num_entries):
            k = int(out_keys.numpy()[i])
            v = int(out_values.numpy()[i])
            entries_dict[k] = v

        self.assertEqual(entries_dict[1], 10)
        self.assertEqual(entries_dict[2], 20)
        self.assertEqual(entries_dict[3], 30)

    def test_empty_table(self):
        """Test operations on empty table."""
        ht = HashTable(capacity=64, device="cpu")

        self.assertEqual(ht.get_num_entries(), 0)
        self.assertEqual(len(ht.get_entries()), 0)

        out_keys, out_values, num_entries = ht.get_entries_arrays()
        self.assertEqual(num_entries, 0)

    def test_custom_kernel_insertion(self):
        """Test using the hashtable_insert function directly in a custom kernel."""

        @wp.kernel
        def custom_insert_kernel(
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
        ):
            tid = wp.tid()
            # Insert (tid, tid * 10)
            key = wp.uint64(tid)
            value = wp.uint64(tid * 10)
            hashtable_insert(key, value, ht_keys, ht_values)

        ht = HashTable(capacity=128, device="cpu")

        wp.launch(
            custom_insert_kernel,
            dim=50,
            inputs=[ht.keys, ht.values],
            device="cpu",
        )

        entries = ht.get_entries()
        self.assertEqual(len(entries), 50)

        entries_dict = dict(entries)
        for i in range(50):
            self.assertEqual(entries_dict[i], i * 10)

    def test_power_of_two_rounding(self):
        """Test that capacity is rounded up to power of two."""
        # Test various input capacities
        ht1 = HashTable(capacity=100, device="cpu")
        self.assertEqual(ht1.capacity, 128)  # Next power of two after 100
        self.assertEqual(ht1.capacity_mask, 127)

        ht2 = HashTable(capacity=64, device="cpu")
        self.assertEqual(ht2.capacity, 64)  # Already power of two
        self.assertEqual(ht2.capacity_mask, 63)

        ht3 = HashTable(capacity=1, device="cpu")
        self.assertEqual(ht3.capacity, 1)
        self.assertEqual(ht3.capacity_mask, 0)

        ht4 = HashTable(capacity=1000, device="cpu")
        self.assertEqual(ht4.capacity, 1024)
        self.assertEqual(ht4.capacity_mask, 1023)

    def test_stress_high_collision(self):
        """Stress test with 20000+ threads writing highly colliding values.

        This tests thread safety under extreme contention where many threads
        compete to write to the same small set of keys.
        """
        n_threads = 25000
        n_unique_keys = 50  # Only 50 unique keys -> 500 threads per key on average

        # Each thread writes to key = tid % n_unique_keys
        # Value = tid (so we can verify max is correct)
        ht = HashTable(capacity=128, device="cpu")

        @wp.kernel
        def stress_insert_kernel(
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
            n_keys: int,
        ):
            tid = wp.tid()
            # Highly colliding: all threads map to one of n_keys buckets
            key = wp.uint64(tid % n_keys)
            value = wp.uint64(tid)
            hashtable_insert(key, value, ht_keys, ht_values)

        wp.launch(
            stress_insert_kernel,
            dim=n_threads,
            inputs=[ht.keys, ht.values, n_unique_keys],
            device="cpu",
        )

        entries = ht.get_entries()
        self.assertEqual(len(entries), n_unique_keys)

        # Verify each key has the maximum thread id that wrote to it
        entries_dict = dict(entries)
        for k in range(n_unique_keys):
            # Threads that wrote to key k: k, k+n_unique_keys, k+2*n_unique_keys, ...
            # Max thread id for key k
            max_tid_for_key = k + ((n_threads - 1 - k) // n_unique_keys) * n_unique_keys
            self.assertEqual(
                entries_dict[k],
                max_tid_for_key,
                f"Key {k}: expected max {max_tid_for_key}, got {entries_dict[k]}",
            )

    def test_stress_extreme_collision_single_key(self):
        """Extreme stress test: all 20000 threads write to the SAME key.

        This is the worst-case scenario for contention.
        """
        n_threads = 20000

        ht = HashTable(capacity=16, device="cpu")

        @wp.kernel
        def single_key_stress_kernel(
            ht_keys: wp.array(dtype=wp.uint64),
            ht_values: wp.array(dtype=wp.uint64),
        ):
            tid = wp.tid()
            # ALL threads write to key 42
            key = wp.uint64(42)
            value = wp.uint64(tid)
            hashtable_insert(key, value, ht_keys, ht_values)

        wp.launch(
            single_key_stress_kernel,
            dim=n_threads,
            inputs=[ht.keys, ht.values],
            device="cpu",
        )

        entries = ht.get_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0], 42)
        self.assertEqual(entries[0][1], n_threads - 1)  # Max tid is n_threads - 1


if __name__ == "__main__":
    unittest.main()

