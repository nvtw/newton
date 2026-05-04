# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the broad-phase contact filter callback.

The filter is wired in :class:`SolverPhoenX` to drop cloth-triangle
pairs that share a vertex.  These tests exercise the filter directly
on :class:`BroadPhaseAllPairs` so they don't depend on a full solver
build.

CUDA + graph-capture-only per repo policy.  All scenarios share a
single class-level warm-up so kernel modules are compiled and cached
once; per-test work is a graph replay against fresh input arrays.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.solvers.phoenx.solver import (
    PhoenXClothFilterData,
    phoenx_cloth_share_vertex_filter,
)


def _make_overlapping_aabbs(num_shapes: int, device) -> tuple[wp.array, wp.array]:
    """All shapes are unit cubes centred at the origin -> every AABB
    pair overlaps, so the broad phase emits every shape combination
    unless the filter drops it."""
    lower = np.full((num_shapes, 3), -0.5, dtype=np.float32)
    upper = np.full((num_shapes, 3), 0.5, dtype=np.float32)
    return (
        wp.array(lower, dtype=wp.vec3, device=device),
        wp.array(upper, dtype=wp.vec3, device=device),
    )


def _make_tri_indices(tri_indices_np: np.ndarray, device) -> wp.array:
    """Pack ``(N, 3)`` int32 connectivity into a ``vec4i`` array
    (4th component unused)."""
    n = len(tri_indices_np)
    quads = np.empty((n, 4), dtype=np.int32)
    if n > 0:
        quads[:, :3] = tri_indices_np
        quads[:, 3] = -1
    return wp.array(quads, dtype=wp.vec4i, device=device)


def _run_pairs(
    bp: BroadPhaseAllPairs,
    num_shapes: int,
    *,
    filter_data: PhoenXClothFilterData | None,
    device,
) -> set[tuple[int, int]]:
    """Launch ``bp`` once and return the unordered set of canonical
    ``(i, j)`` pairs it emitted."""
    lower, upper = _make_overlapping_aabbs(num_shapes, device)
    collision_group = wp.array(np.ones(num_shapes, dtype=np.int32), dtype=wp.int32, device=device)
    shape_world = wp.array(np.zeros(num_shapes, dtype=np.int32), dtype=wp.int32, device=device)

    max_pairs = max(1, num_shapes * (num_shapes - 1) // 2)
    candidate_pair = wp.zeros(max_pairs, dtype=wp.vec2i, device=device)
    candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=device)

    launch_kwargs = {}
    if filter_data is not None:
        launch_kwargs["filter_data"] = filter_data

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        bp.launch(
            lower,
            upper,
            None,
            collision_group,
            shape_world,
            num_shapes,
            candidate_pair,
            candidate_pair_count,
            device=device,
            **launch_kwargs,
        )
    wp.capture_launch(capture.graph)

    n = int(candidate_pair_count.numpy()[0])
    pairs = candidate_pair.numpy()[:n]
    return {(int(p[0]), int(p[1])) for p in pairs}


@unittest.skipUnless(wp.is_cuda_available(), "Broad-phase filter tests require CUDA")
class TestBroadPhaseFilter(unittest.TestCase):
    """Default + cloth-filter variants share one class-level warm-up
    so each test method only pays for one graph capture / replay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = wp.get_device("cuda:0")
        # Sized for the largest scenario (mixed scene: 2 rigid + 3 triangles).
        cls._max_shapes = 5

        # Warm-up launches outside graph capture so the kernel modules
        # are compiled and cached before the per-test scoped captures.
        for filter_func, filter_data_type in (
            (None, None),
            (phoenx_cloth_share_vertex_filter, PhoenXClothFilterData),
        ):
            shape_world = wp.array(np.zeros(cls._max_shapes, dtype=np.int32), dtype=wp.int32, device=cls.device)
            kwargs = {}
            if filter_func is not None:
                kwargs["filter_func"] = filter_func
                kwargs["filter_data_type"] = filter_data_type
            bp = BroadPhaseAllPairs(shape_world, device=cls.device, **kwargs)
            lower, upper = _make_overlapping_aabbs(cls._max_shapes, cls.device)
            collision_group = wp.array(np.ones(cls._max_shapes, dtype=np.int32), dtype=wp.int32, device=cls.device)
            cand = wp.zeros(cls._max_shapes * cls._max_shapes, dtype=wp.vec2i, device=cls.device)
            count = wp.zeros(1, dtype=wp.int32, device=cls.device)
            launch_kwargs = {}
            if filter_func is not None:
                fd = PhoenXClothFilterData()
                fd.num_rigid_shapes = cls._max_shapes
                fd.tri_indices = _make_tri_indices(np.zeros((0, 3), dtype=np.int32), cls.device)
                launch_kwargs["filter_data"] = fd
            bp.launch(
                lower,
                upper,
                None,
                collision_group,
                shape_world,
                cls._max_shapes,
                cand,
                count,
                device=cls.device,
                **launch_kwargs,
            )

    def _make_filtered_bp(
        self, num_rigid_shapes: int, tri_indices_np: np.ndarray
    ) -> tuple[BroadPhaseAllPairs, PhoenXClothFilterData, int]:
        num_triangles = len(tri_indices_np)
        num_shapes = num_rigid_shapes + num_triangles

        shape_world = wp.array(np.zeros(num_shapes, dtype=np.int32), dtype=wp.int32, device=self.device)
        bp = BroadPhaseAllPairs(
            shape_world,
            device=self.device,
            filter_func=phoenx_cloth_share_vertex_filter,
            filter_data_type=PhoenXClothFilterData,
        )

        filter_data = PhoenXClothFilterData()
        filter_data.num_rigid_shapes = int(num_rigid_shapes)
        filter_data.tri_indices = _make_tri_indices(tri_indices_np, self.device)

        return bp, filter_data, num_shapes

    def test_no_filter_keeps_all_pairs(self) -> None:
        with wp.ScopedDevice(self.device):
            num_shapes = 4
            shape_world = wp.array(np.zeros(num_shapes, dtype=np.int32), dtype=wp.int32)
            bp = BroadPhaseAllPairs(shape_world, device=self.device)
            pairs = _run_pairs(bp, num_shapes, filter_data=None, device=self.device)
        expected = {(i, j) for i in range(num_shapes) for j in range(i + 1, num_shapes)}
        self.assertEqual(pairs, expected)

    def test_tt_shared_vertex_pair_is_dropped(self) -> None:
        with wp.ScopedDevice(self.device):
            tris = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)  # share vertex 0
            bp, filter_data, num_shapes = self._make_filtered_bp(0, tris)
            pairs = _run_pairs(bp, num_shapes, filter_data=filter_data, device=self.device)
        self.assertEqual(pairs, set())

    def test_tt_disjoint_pair_is_kept(self) -> None:
        with wp.ScopedDevice(self.device):
            tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)  # no shared vertex
            bp, filter_data, num_shapes = self._make_filtered_bp(0, tris)
            pairs = _run_pairs(bp, num_shapes, filter_data=filter_data, device=self.device)
        self.assertEqual(pairs, {(0, 1)})

    def test_mixed_scene(self) -> None:
        """Two rigids + three triangles: covers RR, RT, TT-shared, TT-disjoint
        in one graph replay.  Tri0/tri1 share vertex 0, tri2 is disjoint."""
        with wp.ScopedDevice(self.device):
            tris = np.array(
                [
                    [0, 1, 2],  # tri0
                    [0, 3, 4],  # tri1: shares vertex 0 with tri0
                    [5, 6, 7],  # tri2: disjoint
                ],
                dtype=np.int32,
            )
            bp, filter_data, num_shapes = self._make_filtered_bp(2, tris)
            pairs = _run_pairs(bp, num_shapes, filter_data=filter_data, device=self.device)

        # Shape indices: 0,1 rigid; 2,3,4 triangles 0,1,2.
        expected = {
            (0, 1),  # RR kept
            (0, 2),
            (0, 3),
            (0, 4),  # rigid 0 vs each triangle (RT kept)
            (1, 2),
            (1, 3),
            (1, 4),  # rigid 1 vs each triangle (RT kept)
            (2, 4),  # tri0 vs tri2 disjoint -> kept
            (3, 4),  # tri1 vs tri2 disjoint -> kept
            # (2, 3) -- tri0/tri1 share vertex 0 -> dropped
        }
        self.assertEqual(pairs, expected)


if __name__ == "__main__":
    unittest.main()
