# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp-aggregated histogram increments must match per-thread atomics."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import warp_aggregated_increment


@wp.kernel(enable_backward=False)
def _increment_kernel(keys: wp.array[wp.int32], counts: wp.array[wp.int32], slots: wp.array[wp.int32]):
    tid = wp.tid()
    key = keys[tid]
    if key < wp.int32(0):
        return
    slots[tid] = warp_aggregated_increment(counts, key)


class TestWarpAggregatedIncrement(unittest.TestCase):
    def test_counts_and_unique_slots(self):
        """Mixed sorted, shuffled and inactive keys get exact counts and dense slots."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = "cuda:0"
        key_count = 40
        rng = np.random.default_rng(7)
        keys = np.sort(rng.integers(-1, key_count, 100_003)).astype(np.int32)
        # Divergent warps: unsorted keys and inactive lanes share warps.
        rng.shuffle(keys[:5000])
        keys_device = wp.array(keys, dtype=wp.int32, device=device)
        counts = wp.zeros(key_count, dtype=wp.int32, device=device)
        slots = wp.full(len(keys), -1, dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            counts.zero_()
            wp.launch(_increment_kernel, dim=len(keys), inputs=[keys_device, counts, slots], device=device)
        for _ in range(2):
            wp.capture_launch(capture.graph)
        slots_host = slots.numpy()
        np.testing.assert_array_equal(counts.numpy(), np.bincount(keys[keys >= 0], minlength=key_count))
        np.testing.assert_array_equal(slots_host[keys < 0], -1)
        for key in range(key_count):
            key_slots = np.sort(slots_host[keys == key])
            np.testing.assert_array_equal(key_slots, np.arange(len(key_slots)))


if __name__ == "__main__":
    unittest.main()
