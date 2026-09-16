# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check independent-group coverage and intra-group ordering for both grids."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.dispatch import color_groups as groups
from newton._src.solvers.phoenx.solver_phoenx_kernels import (
    BodyContainer,
    ConstraintContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
)


@wp.func
def record(
    constraints: ConstraintContainer,
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    cc: ContactContainer,
    contacts: ContactViews,
    copies: CopyStateContainer,
    num_joints: wp.int32,
    previous: wp.array[wp.int32],
    triangles: wp.int32,
    bending: wp.int32,
    tetrahedra: wp.int32,
    hexahedra: wp.int32,
    num_bodies: wp.int32,
    idt: wp.float32,
    sor: wp.float32,
    cid: wp.int32,
    slot: wp.int32,
    group: wp.int32,
):
    wp.atomic_add(cc.impulses, 0, cid, 1.0)
    cc.impulses[1, cid] = wp.float32(group)
    depth = wp.float32(1.0)
    if previous[cid] >= 0:
        depth += cc.impulses[2, previous[cid]]
    cc.impulses[2, cid] = depth


@unittest.skipUnless(wp.is_cuda_available(), "Group dispatch requires CUDA")
class TestColorGroupGrid(unittest.TestCase):
    def test_ragged_groups_cover_rows_once_and_keep_color_order(self):
        """Exercise empty groups, extra lane work, and groups beyond either grid."""
        with patch.object(groups, "_make_singleworld_dispatch_func", return_value=(record, None)):
            kernels = {blocks: groups.get_sweep_kernel("coverage_test", False, blocks) for blocks in (32, 64)}
        for group_count in (0, 1, 33, 65, 131):
            width = 3
            colors = max(0, group_count * width - 1)
            sizes = np.arange(colors, dtype=np.int32) % 35 + 1
            starts = np.r_[0, np.cumsum(sizes)].astype(np.int32)
            count = int(starts[-1])
            previous = np.full(count, -1, dtype=np.int32)
            expected_group = np.zeros(count, dtype=np.float32)
            expected_depth = np.zeros(count, dtype=np.float32)
            for color in range(colors):
                begin, end = starts[color : color + 2]
                expected_group[begin:end] = color // width
                expected_depth[begin:end] = color % width + 1
                if color % width:
                    previous[begin:end] = starts[color - 1]
            results = []
            for blocks, kernel in kernels.items():
                with self.subTest(groups=group_count, blocks=blocks):
                    cc = ContactContainer()
                    cc.impulses = wp.zeros((3, count), dtype=wp.float32, device="cuda:0")
                    wp.launch(
                        kernel,
                        (blocks, 32),
                        [
                            ConstraintContainer(),
                            ContactColumnContainer(),
                            BodyContainer(),
                            ParticleContainer(),
                            cc,
                            ContactViews(),
                            CopyStateContainer(),
                            0,
                            wp.array(previous, dtype=wp.int32, device="cuda:0"),
                            0,
                            1.0,
                            wp.array(np.arange(count), dtype=wp.int32, device="cuda:0"),
                            wp.array(starts, dtype=wp.int32, device="cuda:0"),
                            wp.array([colors], dtype=wp.int32, device="cuda:0"),
                            width,
                        ],
                        block_dim=32,
                        device="cuda:0",
                    )
                    actual = cc.impulses.numpy()
                    np.testing.assert_array_equal(actual[0], np.ones(count))
                    np.testing.assert_array_equal(actual[1], expected_group)
                    np.testing.assert_array_equal(actual[2], expected_depth)
                    results.append(actual.tobytes())
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
