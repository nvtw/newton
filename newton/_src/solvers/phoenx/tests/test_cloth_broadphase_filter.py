# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the phoenx cloth broadphase filter callback.

The filter drops cloth-triangle pair candidates where both shapes
are triangles and the two triangles share at least one node. Rigid
shapes (index ``< num_rigid_shapes``) always pass through.

Triangle topology lives in the cloth-triangle rows of a phoenx
:class:`ConstraintContainer` (the real source of truth in the
solver) -- the test stamps a couple of rows by hand via the schema
setters and hands the container to the filter.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.solvers.phoenx.cloth_collision.broadphase_filter import (
    PhoenxBroadphaseFilterData,
    phoenx_broadphase_filter,
)
from newton._src.solvers.phoenx.constraints.constraint_cloth_triangle import (
    cloth_triangle_set_body1,
    cloth_triangle_set_body2,
    cloth_triangle_set_body3,
    cloth_triangle_set_type,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@wp.kernel(enable_backward=False)
def _stamp_cloth_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    indices: wp.array[wp.int32],  # length 3 * T
):
    """Test-only kernel that writes one cloth-triangle row per thread,
    setting the type tag + the three particle-index slots."""
    t = wp.tid()
    cid = cid_offset + t
    cloth_triangle_set_type(constraints, cid)
    cloth_triangle_set_body1(constraints, cid, indices[t * 3 + 0])
    cloth_triangle_set_body2(constraints, cid, indices[t * 3 + 1])
    cloth_triangle_set_body3(constraints, cid, indices[t * 3 + 2])


def _make_constraints_with_cloth(rows: list[tuple[int, int, int]], device) -> tuple:
    """Build a phoenx ConstraintContainer with the given cloth-triangle
    rows starting at cid 0. Returns (constraints, cid_offset = 0)."""
    T = len(rows)
    num_dwords = PhoenXWorld.required_constraint_dwords(num_joints=0, num_cloth_triangles=T)
    constraints = constraint_container_zeros(num_constraints=max(1, T), num_dwords=num_dwords, device=device)
    flat = np.array(rows, dtype=np.int32).reshape(-1)
    indices = wp.array(flat, dtype=wp.int32, device=device)
    wp.launch(
        _stamp_cloth_rows_kernel,
        dim=T,
        inputs=[constraints, wp.int32(0), indices],
        device=device,
    )
    return constraints, 0


def _overlap_all_aabbs(n: int, device) -> tuple[wp.array, wp.array]:
    lower = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    upper = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    return lower, upper


def _read_pairs(candidate: wp.array, count: wp.array) -> set[tuple[int, int]]:
    n = int(count.numpy()[0])
    if n == 0:
        return set()
    arr = candidate.numpy()
    return {(int(min(p[0], p[1])), int(max(p[0], p[1]))) for p in arr[:n]}


@unittest.skipUnless(wp.is_cuda_available(), "Filter test runs on CUDA")
class TestPhoenxBroadphaseFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.device = wp.get_device()

    def _run_filter(self, n: int, fdata: PhoenxBroadphaseFilterData) -> set[tuple[int, int]]:
        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=phoenx_broadphase_filter)
        lower, upper = _overlap_all_aabbs(n, self.device)
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(max(1, n * (n - 1) // 2), dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)
        bp.launch(
            shape_lower=lower,
            shape_upper=upper,
            shape_gap=None,
            shape_collision_group=groups,
            shape_world=worlds,
            shape_count=n,
            candidate_pair=candidate,
            candidate_pair_count=count,
            device=self.device,
            filter_data=fdata,
        )
        return _read_pairs(candidate, count)

    def test_rigid_only_passes_through(self) -> None:
        """No triangles in the shape space -> filter is a no-op."""
        n = 4
        # Empty cloth-row block.  The filter still receives a
        # valid ConstraintContainer (sentinel) but will never read
        # from it because every shape index is below the threshold.
        constraints, cid_offset = _make_constraints_with_cloth([(0, 1, 2)], self.device)
        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = n  # all rigid
        fdata.constraints = constraints
        fdata.cloth_cid_offset = cid_offset

        pairs = self._run_filter(n, fdata)
        # All C(4, 2) = 6 rigid-rigid pairs should survive.
        self.assertEqual(len(pairs), 6)

    def test_drops_adjacent_triangles(self) -> None:
        """Two cloth triangles sharing a vertex are dropped; the rigid
        shape's pairs against both triangles survive."""
        # 1 rigid shape (idx 0), 2 cloth triangles (idx 1, 2).
        # Triangle 0 -> particles (10, 11, 12)
        # Triangle 1 -> particles (12, 13, 14) shares node 12.
        S = 1
        n = S + 2
        constraints, cid_offset = _make_constraints_with_cloth([(10, 11, 12), (12, 13, 14)], self.device)
        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = S
        fdata.constraints = constraints
        fdata.cloth_cid_offset = cid_offset

        pairs = self._run_filter(n, fdata)
        # Expected: (0, 1) and (0, 2) survive (rigid vs each triangle).
        # Dropped: (1, 2) -- both triangles share node 12.
        self.assertEqual(pairs, {(0, 1), (0, 2)})

    def test_keeps_disjoint_triangles(self) -> None:
        """Two cloth triangles with no shared nodes both survive."""
        S = 0
        n = 2
        constraints, cid_offset = _make_constraints_with_cloth([(10, 11, 12), (20, 21, 22)], self.device)
        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = S
        fdata.constraints = constraints
        fdata.cloth_cid_offset = cid_offset

        pairs = self._run_filter(n, fdata)
        self.assertEqual(pairs, {(0, 1)})


if __name__ == "__main__":
    unittest.main()
