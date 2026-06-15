# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Allocation + invariant tests for :class:`CopyStateContainer`."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_NONE
from newton._src.solvers.phoenx.mass_splitting.copy_state import (
    CopyStateContainer,
    copy_state_container_zeros,
)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX mass-splitting tests are CUDA-only per feedback_phoenx_tests_capture_only.",
)
class TestCopyStateContainer(unittest.TestCase):
    def test_zero_alloc_shapes_and_defaults(self):
        device = wp.get_preferred_device()
        capacity = 32
        num_nodes = 7
        cs = copy_state_container_zeros(capacity=capacity, num_nodes=num_nodes, device=device)

        # Shapes.
        self.assertEqual(cs.position.shape[0], capacity)
        self.assertEqual(cs.orientation.shape[0], capacity)
        self.assertEqual(cs.velocity.shape[0], capacity)
        self.assertEqual(cs.angular_velocity.shape[0], capacity)
        self.assertEqual(cs.access_mode.shape[0], capacity)
        self.assertEqual(cs.partition_list.shape[0], capacity)
        self.assertEqual(cs.section_end.shape[0], num_nodes)
        self.assertEqual(cs.highest_index_in_use.shape[0], 1)

        # Defaults: all zero except access_mode = ACCESS_MODE_NONE.
        np.testing.assert_array_equal(cs.position.numpy(), 0)
        np.testing.assert_array_equal(cs.orientation.numpy(), 0)
        np.testing.assert_array_equal(cs.velocity.numpy(), 0)
        np.testing.assert_array_equal(cs.angular_velocity.numpy(), 0)
        np.testing.assert_array_equal(cs.access_mode.numpy(), int(ACCESS_MODE_NONE))
        np.testing.assert_array_equal(cs.partition_list.numpy(), 0)
        np.testing.assert_array_equal(cs.section_end.numpy(), 0)
        np.testing.assert_array_equal(cs.highest_index_in_use.numpy(), 0)

    def test_struct_is_warp_compatible(self):
        # Sanity: passing the struct into a trivial kernel must compile.
        # If wp.struct misdetected any field type, the kernel below
        # would fail at JIT.
        device = wp.get_preferred_device()
        cs = copy_state_container_zeros(capacity=4, num_nodes=2, device=device)

        @wp.kernel(enable_backward=False)
        def _probe(c: CopyStateContainer, out: wp.array[wp.int32]):
            if wp.tid() == 0:
                out[0] = c.section_end.shape[0]

        out = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_probe, dim=1, inputs=[cs, out], device=device)
        self.assertEqual(int(out.numpy()[0]), 2)

    def test_capacity_validation(self):
        with self.assertRaises(ValueError):
            copy_state_container_zeros(capacity=0, num_nodes=1)
        with self.assertRaises(ValueError):
            copy_state_container_zeros(capacity=1, num_nodes=0)


if __name__ == "__main__":
    unittest.main()
