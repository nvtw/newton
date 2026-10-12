# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prevent large-world integer overflow in contact lane allocation."""

import unittest

import warp as wp

from newton._src.solvers.phoenx.simulation_kernels import _rigid_singleton_lane_allocation


@wp.kernel(enable_backward=False)
def allocate(
    single: wp.array[wp.int32],
    multi: wp.array[wp.int32],
    lanes: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    tid = wp.tid()
    output[tid] = _rigid_singleton_lane_allocation(single[tid], multi[tid], lanes[tid])


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSingletonLaneAllocation(unittest.TestCase):
    def test_large_grid_uses_many_warp_aligned_lanes(self):
        # 385024 * 40000 exceeds signed int32. The original expression
        # wrapped negative, clamped to one warp, and serialized all 40k
        # singleton columns through 32 lanes.
        single = wp.array([40000, 40000, 0, 1], dtype=wp.int32, device="cuda:0")
        multi = wp.array([20000, 0, 20000, 1], dtype=wp.int32, device="cuda:0")
        lanes = wp.array([385024, 385024, 385024, 64], dtype=wp.int32, device="cuda:0")
        output = wp.zeros(4, dtype=wp.int32, device="cuda:0")
        wp.launch(allocate, dim=4, inputs=[single, multi, lanes, output], device="cuda:0")
        result = output.numpy()
        self.assertGreater(int(result[0]), 385024 // 4)
        self.assertLess(int(result[0]), 385024 // 2)
        self.assertEqual(int(result[0]) % 32, 0)
        self.assertEqual(int(result[1]), 385024)
        self.assertEqual(int(result[2]), 0)
        self.assertEqual(int(result[3]), 32)


if __name__ == "__main__":
    unittest.main()
