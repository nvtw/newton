# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check that bit-limited int64 sorts reproduce the full-width permutation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.helpers.scan_and_sort import active_prefix_sort_levels, sort_variable_length_int64


def _sort(keys: np.ndarray, active: int, end_bit: int | None) -> tuple[np.ndarray, np.ndarray]:
    capacity = len(keys)
    key_array = wp.zeros(2 * capacity, dtype=wp.int64, device="cuda:0")
    value_array = wp.zeros(2 * capacity, dtype=wp.int32, device="cuda:0")
    key_array.assign(np.concatenate([keys, np.zeros(capacity, dtype=np.int64)]))
    value_array.assign(np.arange(2 * capacity, dtype=np.int32))
    count = wp.array([active], dtype=wp.int32, device="cuda:0")
    sort_variable_length_int64(key_array, value_array, count, end_bit=end_bit)
    return key_array.numpy()[:capacity], value_array.numpy()[:capacity]


@unittest.skipUnless(wp.is_cuda_available(), "CUDA radix sort")
class TestSortEndBit(unittest.TestCase):
    def test_body_pair_and_interaction_keys_match_full_width(self):
        """Duplicates, the largest in-range key and the masked tail keep their order."""
        rng = np.random.default_rng(7)
        capacity = 4096
        active = 3000
        num_bodies = 13741
        body_keys = rng.integers(0, num_bodies**2, capacity, dtype=np.int64)
        body_keys[::7] = body_keys[0]
        body_keys[5] = num_bodies**2 - 1
        nodes = rng.integers(0, num_bodies, capacity, dtype=np.int64)
        partitions = rng.integers(-3, 40000, capacity, dtype=np.int64) & 0xFFFFFFFF
        graph_keys = (nodes << 32) | partitions
        graph_keys[9] = ((num_bodies - 1) << 32) | 0xFFFFFFFF
        cases = (
            (body_keys, (num_bodies**2).bit_length()),
            (graph_keys, 32 + num_bodies.bit_length()),
        )
        for keys, end_bit in cases:
            self.assertLess(end_bit, 64)
            expected = _sort(keys, active, None)
            actual = _sort(keys, active, end_bit)
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[0][:active], expected[0][:active])

    def test_active_prefix_sort_matches_full_sort(self):
        """Every captured prefix branch matches a full sort and restores the sentinel tail."""
        capacity = 1 << 20
        levels = active_prefix_sort_levels(capacity)
        self.assertEqual(levels, 5)
        rng = np.random.default_rng(11)
        source = rng.integers(0, 1 << 40, capacity, dtype=np.int64)
        source[::5] = source[3]
        device = "cuda:0"
        keys = wp.zeros(2 * capacity, dtype=wp.int64, device=device)
        values = wp.zeros(2 * capacity, dtype=wp.int32, device=device)
        source_keys = wp.array(source, dtype=wp.int64, device=device)
        source_values = wp.array(np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        flags = wp.zeros(levels, dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.copy(keys, source_keys, count=capacity)
            wp.copy(values, source_values, count=capacity)
            # Fill the ping-pong half with stale keys a prefix sort must not expose.
            wp.copy(keys, source_keys, dest_offset=capacity, count=capacity)
            sort_variable_length_int64(keys, values, count, end_bit=41, prefix_flags=flags)
        sentinel = np.iinfo(np.int64).max
        for active in (0, 1, 4097, 65535, 65536, 131071, 131072, 200001, 524287, 524288, capacity):
            with self.subTest(active=active):
                count.fill_(active)
                wp.capture_launch(capture.graph)
                order = np.argsort(source[:active], kind="stable")
                sorted_keys = keys.numpy()[:capacity]
                np.testing.assert_array_equal(sorted_keys[:active], source[:active][order])
                np.testing.assert_array_equal(values.numpy()[:active], order.astype(np.int32))
                np.testing.assert_array_equal(sorted_keys[active:], sentinel)
                self.assertEqual(int(flags.numpy().sum()), 1)


if __name__ == "__main__":
    unittest.main()
