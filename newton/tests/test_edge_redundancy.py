# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the edge-redundancy broad-phase prototype.

The :func:`find_redundant_edges` helper builds an oriented bounding box around
every manifold edge of a mesh and reports which edges are fully contained
inside another edge's box.
"""

import math
import unittest

import numpy as np

import newton
from newton._src.geometry.edge_redundancy import (
    EdgeRedundancyResult,
    EdgeResolutionResult,
    find_redundant_edges,
    resolve_edge_removals,
)


def _single_triangle_mesh() -> newton.Mesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.int32)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _empty_mesh() -> newton.Mesh:
    return newton.Mesh(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), compute_inertia=False)


def _swallowing_mesh() -> newton.Mesh:
    """Two parallel manifold edges in one plane: a long one and a short one slightly offset.

    The long edge ``A0-A1`` sits in the y=0 strip; the short edge ``B0-B1`` sits
    at y = 0.001 (a tiny offset) and lies fully within the X-range of the long
    edge. Each edge is wrapped by top and bottom triangles so it is manifold.
    With generous box extents the long edge's oriented box is expected to
    fully contain the short edge's segment.
    """
    vertices = np.array(
        [
            # Long edge stack: y in {-1, 0, 1} at x in {0, 10}.
            [0.0, -1.0, 0.0],  # 0  bot_a0
            [10.0, -1.0, 0.0],  # 1  bot_a1
            [0.0, 0.0, 0.0],  # 2  A0
            [10.0, 0.0, 0.0],  # 3  A1
            [0.0, 1.0, 0.0],  # 4  top_a0
            [10.0, 1.0, 0.0],  # 5  top_a1
            # Short edge stack: y offset slightly so vertices don't coincide with the long stack.
            [3.0, -0.999, 0.0],  # 6  bot_b0
            [7.0, -0.999, 0.0],  # 7  bot_b1
            [3.0, 0.001, 0.0],  # 8  B0
            [7.0, 0.001, 0.0],  # 9  B1
            [3.0, 1.001, 0.0],  # 10 top_b0
            [7.0, 1.001, 0.0],  # 11 top_b1
        ],
        dtype=np.float32,
    )
    # Triangles wrap each manifold edge with one tri above and one below.
    indices = np.array(
        [
            # Long edge A0-A1 sandwiched between bot and top.
            0, 1, 2,
            1, 3, 2,
            2, 3, 4,
            3, 5, 4,
            # Short edge B0-B1 sandwiched between bot and top.
            6, 7, 8,
            7, 9, 8,
            8, 9, 10,
            9, 11, 10,
        ],
        dtype=np.int32,
    )  # fmt: skip
    return newton.Mesh(vertices, indices, compute_inertia=False)


class TestEdgeRedundancyEdgeCases(unittest.TestCase):
    def test_empty_mesh_returns_empty_result(self):
        result = find_redundant_edges(_empty_mesh())
        self.assertIsInstance(result, EdgeRedundancyResult)
        self.assertEqual(result.edge_indices.shape, (0, 2))
        self.assertEqual(result.candidate_for_removal.shape, (0,))
        self.assertEqual(result.broad_phase_pair_count, 0)
        self.assertEqual(int(result.swallowed_offsets[-1]), 0)

    def test_single_triangle_has_no_manifold_edges(self):
        result = find_redundant_edges(_single_triangle_mesh())
        # All three edges are boundary (one adjacent triangle); none are manifold.
        self.assertEqual(result.edge_indices.shape, (0, 2))
        self.assertEqual(result.candidate_for_removal.shape, (0,))


class TestEdgeRedundancyCube(unittest.TestCase):
    def test_cube_default_extents_no_candidates(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        result = find_redundant_edges(mesh)
        # 12 silhouette + 6 face diagonals = 18 manifold edges on a cube.
        self.assertEqual(len(result.edge_indices), 18)
        self.assertEqual(int(result.candidate_for_removal.sum()), 0)
        self.assertEqual(int(result.swallow_count_per_box.sum()), 0)
        self.assertEqual(int(result.swallowed_offsets[-1]), 0)

    def test_cube_oversized_extents_produce_candidates(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        # Half-extents larger than the cube edge length -> sister edges get swallowed.
        result = find_redundant_edges(mesh, half_height=2.0, half_width=2.0)
        self.assertGreater(int(result.candidate_for_removal.sum()), 0)
        # CSR consistency: total swallowed entries equals sum over box counts.
        self.assertEqual(int(result.swallow_count_per_box.sum()), int(result.swallowed_offsets[-1]))


class TestEdgeRedundancySwallowing(unittest.TestCase):
    def test_long_edge_box_swallows_short_edge(self):
        mesh = _swallowing_mesh()
        # Generous box so the long edge's oriented box covers the short edge.
        result = find_redundant_edges(mesh, half_height=0.5, half_width=0.5)

        # Locate the long and short edges in the manifold-edge subset.
        rows = [tuple(sorted((int(a), int(b)))) for a, b in result.edge_indices]
        long_edge = (2, 3)  # A0-A1
        short_edge = (8, 9)  # B0-B1
        self.assertIn(long_edge, rows)
        self.assertIn(short_edge, rows)
        long_idx = rows.index(long_edge)
        short_idx = rows.index(short_edge)

        self.assertTrue(bool(result.candidate_for_removal[short_idx]))
        self.assertGreaterEqual(int(result.num_containers_per_edge[short_idx]), 1)
        self.assertGreaterEqual(int(result.swallow_count_per_box[long_idx]), 1)

        # The CSR slice for the long edge's box must list the short edge's index.
        lo = int(result.swallowed_offsets[long_idx])
        hi = int(result.swallowed_offsets[long_idx + 1])
        swallowed = {int(x) for x in result.swallowed_indices[lo:hi]}
        self.assertIn(short_idx, swallowed)

    def test_csr_offsets_are_monotonic(self):
        mesh = _swallowing_mesh()
        result = find_redundant_edges(mesh, half_height=0.5, half_width=0.5)
        offsets = result.swallowed_offsets
        # Strictly non-decreasing and ends with the total entry count.
        self.assertTrue(bool(np.all(np.diff(offsets) >= 0)))
        self.assertEqual(int(offsets[-1]), int(result.swallow_count_per_box.sum()))


def _make_synthetic_result(
    *,
    edge_indices: np.ndarray,
    dihedral_angles: np.ndarray,
    swallow_lists: list[list[int]],
) -> EdgeRedundancyResult:
    """Build an EdgeRedundancyResult by hand, bypassing the GPU path."""
    n = len(edge_indices)
    swallow_count = np.array([len(s) for s in swallow_lists], dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(swallow_count)
    indices = (
        np.concatenate([np.array(s, dtype=np.int32) for s in swallow_lists])
        if any(swallow_lists)
        else np.zeros(0, dtype=np.int32)
    )
    num_containers = np.zeros(n, dtype=np.int32)
    for swallowed in swallow_lists:
        for e in swallowed:
            num_containers[e] += 1
    candidate = num_containers > 0
    return EdgeRedundancyResult(
        edge_indices=np.asarray(edge_indices, dtype=np.int32).reshape(-1, 2),
        dihedral_angles=np.asarray(dihedral_angles, dtype=np.float32),
        candidate_for_removal=candidate,
        num_containers_per_edge=num_containers,
        swallow_count_per_box=swallow_count,
        swallowed_offsets=offsets,
        swallowed_indices=indices,
        broad_phase_pair_count=int(swallow_count.sum()),
        aabb_diagonal=1.0,
        half_height=1.0,
        half_width=1.0,
    )


class TestEdgeRemovalResolution(unittest.TestCase):
    def test_swallowing_mesh_default_threshold_removes_short_edge(self):
        mesh = _swallowing_mesh()
        result = find_redundant_edges(mesh, half_height=0.5, half_width=0.5)
        resolution = resolve_edge_removals(result)
        self.assertIsInstance(resolution, EdgeResolutionResult)
        rows = [tuple(sorted((int(a), int(b)))) for a, b in result.edge_indices]
        long_idx = rows.index((2, 3))
        short_idx = rows.index((8, 9))
        self.assertTrue(bool(resolution.kept[long_idx]))
        self.assertTrue(bool(resolution.to_remove[short_idx]))
        self.assertFalse(bool(resolution.kept[short_idx]))
        self.assertFalse(bool(resolution.to_remove[long_idx]))
        # The two masks are always disjoint.
        self.assertEqual(int((resolution.kept & resolution.to_remove).sum()), 0)

    def test_threshold_zero_removes_nothing(self):
        mesh = _swallowing_mesh()
        result = find_redundant_edges(mesh, half_height=0.5, half_width=0.5)
        # angle_threshold == 0 means no swallowed edge qualifies (angle < 0 is False everywhere).
        resolution = resolve_edge_removals(result, angle_threshold_rad=0.0)
        self.assertEqual(int(resolution.to_remove.sum()), 0)

    def test_cube_oversized_extents_kept_and_removed_are_disjoint(self):
        mesh = newton.Mesh.create_box(0.5, compute_inertia=False)
        result = find_redundant_edges(mesh, half_height=2.0, half_width=2.0)
        # Cube silhouette edges have a 90 deg dihedral and must never be
        # removed at the default 10 deg threshold; only the 6 face diagonals
        # (0 deg dihedral) qualify. Both masks must always be disjoint.
        resolution = resolve_edge_removals(result, angle_threshold_rad=math.radians(10.0))
        self.assertEqual(int((resolution.kept & resolution.to_remove).sum()), 0)
        # With a high threshold even silhouette edges become eligible; still
        # disjoint.
        resolution_loose = resolve_edge_removals(result, angle_threshold_rad=math.radians(120.0))
        self.assertEqual(int((resolution_loose.kept & resolution_loose.to_remove).sum()), 0)

    def test_skip_when_container_already_removed(self):
        # Three edges. L1 swallows {L2, S}, L2 swallows {S}. With descending
        # sort, L1 (count=2) is processed first: it keeps L1, removes L2 and S.
        # Then L2 is reached: its container (L2) is already removed, so the
        # iteration is skipped. The final state must have L1 kept, L2 and S
        # removed, and no double-counting on S.
        result = _make_synthetic_result(
            edge_indices=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),  # L1=0, L2=1, S=2
            dihedral_angles=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            swallow_lists=[[1, 2], [2], []],
        )
        resolution = resolve_edge_removals(result, angle_threshold_rad=math.radians(10.0))
        self.assertTrue(bool(resolution.kept[0]))
        self.assertFalse(bool(resolution.kept[1]))
        self.assertTrue(bool(resolution.to_remove[1]))
        self.assertTrue(bool(resolution.to_remove[2]))
        self.assertFalse(bool(resolution.to_remove[0]))
        # L1 must come before L2 in the order (L1's count is larger).
        order = [int(x) for x in resolution.order]
        self.assertLess(order.index(0), order.index(1))

    def test_definitely_keep_protects_from_later_removal(self):
        # Two boxes both want to remove edge K. Box A is processed first and
        # promotes K to "kept" (it is A's container). Box B then tries to
        # remove K via its swallowed list -> must be ignored.
        # Setup:
        #   edge 0 = "A" with swallow list [2]  (count 1, processed first as
        #              max count is 2 from box 1; tie broken by stable sort).
        # We make A swallow more so it sorts first:
        #   edge 0 = "A": swallows {2, 3}        (count 2)
        #   edge 1 = "B": swallows {0, 4}        (count 2; tie -> stable sort
        #                                         picks lower index 0 first)
        #   edges 2, 3, 4: swallow nothing
        result = _make_synthetic_result(
            edge_indices=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.int32),
            dihedral_angles=np.zeros(5, dtype=np.float32),
            swallow_lists=[[2, 3], [0, 4], [], [], []],
        )
        resolution = resolve_edge_removals(result, angle_threshold_rad=math.radians(10.0))
        # A is processed first, promotes edge 0 to kept and removes 2 and 3.
        self.assertTrue(bool(resolution.kept[0]))
        self.assertTrue(bool(resolution.to_remove[2]))
        self.assertTrue(bool(resolution.to_remove[3]))
        # B is processed next; its container (edge 1) is NOT yet removed, so
        # B promotes edge 1 to kept. B then tries to remove edges 0 and 4.
        # Edge 0 is already in kept -> protected. Edge 4 is removed.
        self.assertTrue(bool(resolution.kept[1]))
        self.assertFalse(bool(resolution.to_remove[0]))
        self.assertTrue(bool(resolution.to_remove[4]))


if __name__ == "__main__":
    unittest.main()
