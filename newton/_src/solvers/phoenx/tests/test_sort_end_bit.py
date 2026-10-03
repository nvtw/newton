# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check that bit-limited int64 sorts reproduce the full-width permutation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.helpers.scan_and_sort import sort_variable_length_int64


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


if __name__ == "__main__":
    unittest.main()
