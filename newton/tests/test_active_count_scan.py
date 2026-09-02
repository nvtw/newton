# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.utils import ACTIVE_SCAN_CHUNK, ActiveCountScan
from newton.tests.unittest_utils import get_test_devices


class TestActiveCountScan(unittest.TestCase):
    def test_matches_exclusive_scan_over_live_prefix(self) -> None:
        """Scan results equal a full exclusive scan for every live index."""
        rng = np.random.default_rng(7)
        capacity = 3 * ACTIVE_SCAN_CHUNK + 123
        counts_np = rng.integers(0, 9, size=capacity, dtype=np.int32)
        for device in get_test_devices():
            scan = ActiveCountScan(capacity, device=device)
            counts = wp.array(counts_np, dtype=wp.int32, device=device)
            for n in (
                0,
                1,
                5,
                ACTIVE_SCAN_CHUNK - 1,
                ACTIVE_SCAN_CHUNK,
                ACTIVE_SCAN_CHUNK + 1,
                2 * ACTIVE_SCAN_CHUNK + 17,
                capacity,
            ):
                with self.subTest(device=device, n=n):
                    prefix = wp.full(capacity, -1, dtype=wp.int32, device=device)
                    num_elements = wp.array([n], dtype=wp.int32, device=device)
                    scan.scan(counts, prefix, num_elements)
                    result = prefix.numpy()
                    expected = np.concatenate(([0], np.cumsum(counts_np[:n], dtype=np.int64)[:-1])) if n > 0 else []
                    np.testing.assert_array_equal(result[:n], np.asarray(expected, dtype=np.int32))
                    # Chunks entirely beyond the live prefix are left untouched.
                    live_end = -(-n // ACTIVE_SCAN_CHUNK) * ACTIVE_SCAN_CHUNK
                    self.assertTrue(np.all(result[live_end:] == -1))

    def test_clamps_live_count_to_capacity(self) -> None:
        """A live count above the capacity behaves like the full buffer."""
        capacity = ACTIVE_SCAN_CHUNK + 5
        counts_np = np.arange(capacity, dtype=np.int32) % 4
        for device in get_test_devices():
            with self.subTest(device=device):
                scan = ActiveCountScan(capacity, device=device)
                counts = wp.array(counts_np, dtype=wp.int32, device=device)
                prefix = wp.zeros(capacity, dtype=wp.int32, device=device)
                scan.scan(counts, prefix, wp.array([capacity + 1000], dtype=wp.int32, device=device))
                expected = np.concatenate(([0], np.cumsum(counts_np)[:-1])).astype(np.int32)
                np.testing.assert_array_equal(prefix.numpy(), expected)

    def test_rejects_mismatched_capacity(self) -> None:
        for device in get_test_devices():
            with self.subTest(device=device):
                scan = ActiveCountScan(16, device=device)
                counts = wp.zeros(32, dtype=wp.int32, device=device)
                with self.assertRaises(ValueError):
                    scan.scan(counts, counts, wp.array([1], dtype=wp.int32, device=device))


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
