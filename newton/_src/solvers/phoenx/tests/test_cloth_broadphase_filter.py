# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the phoenx cloth broadphase filter callback.

The filter drops cloth-triangle pair candidates where both shapes
are triangles and the two triangles share at least one node. Rigid
shapes (index ``< num_rigid_shapes``) always pass through.
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

    def test_rigid_only_passes_through(self) -> None:
        """No triangles in the shape space -> filter is a no-op."""
        n = 4  # all rigid
        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=phoenx_broadphase_filter)

        lower, upper = _overlap_all_aabbs(n, self.device)
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(n * (n - 1) // 2, dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Empty triangle table; threshold equals shape_count so nothing is a triangle.
        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = n
        fdata.tri_indices = wp.zeros(1, dtype=wp.vec3i, device=self.device)

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
        pairs = _read_pairs(candidate, count)
        # All C(4, 2) = 6 rigid-rigid pairs should survive.
        self.assertEqual(len(pairs), 6)

    def test_drops_adjacent_triangles(self) -> None:
        """Two cloth triangles sharing a vertex are dropped; the rigid
        shape's pairs against both triangles survive."""
        # Layout: 1 rigid shape (idx 0), 2 cloth triangles (idx 1, 2).
        # Triangle 1: nodes (10, 11, 12)
        # Triangle 2: nodes (12, 13, 14)   <- shares node 12 with triangle 1
        S = 1
        T = 2
        n = S + T
        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=phoenx_broadphase_filter)

        lower, upper = _overlap_all_aabbs(n, self.device)
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(n * (n - 1) // 2, dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        tri_indices_np = np.array([[10, 11, 12], [12, 13, 14]], dtype=np.int32)
        tri_indices = wp.array(tri_indices_np, dtype=wp.vec3i, device=self.device)

        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = S
        fdata.tri_indices = tri_indices

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
        pairs = _read_pairs(candidate, count)
        # Expected: (0,1) and (0,2) survive (rigid vs each triangle).
        # Dropped: (1,2) — both triangles, share node 12.
        self.assertEqual(pairs, {(0, 1), (0, 2)})

    def test_keeps_disjoint_triangles(self) -> None:
        """Two cloth triangles with no shared nodes both survive."""
        S = 0
        T = 2
        n = S + T
        shape_world = np.zeros(n, dtype=np.int32)
        bp = BroadPhaseAllPairs(shape_world, device=self.device, filter_func=phoenx_broadphase_filter)

        lower, upper = _overlap_all_aabbs(n, self.device)
        groups = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=self.device)
        worlds = wp.array(shape_world, dtype=wp.int32, device=self.device)
        candidate = wp.zeros(max(1, n * (n - 1) // 2), dtype=wp.vec2i, device=self.device)
        count = wp.zeros(1, dtype=wp.int32, device=self.device)

        tri_indices_np = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
        tri_indices = wp.array(tri_indices_np, dtype=wp.vec3i, device=self.device)

        fdata = PhoenxBroadphaseFilterData()
        fdata.num_rigid_shapes = S
        fdata.tri_indices = tri_indices

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
        pairs = _read_pairs(candidate, count)
        # Only (0, 1); they don't share any node.
        self.assertEqual(pairs, {(0, 1)})


if __name__ == "__main__":
    unittest.main()
