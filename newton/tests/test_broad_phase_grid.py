# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pair-set equivalence between uniform-grid and SAP broad phases."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_grid import BroadPhaseGrid
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton._src.geometry.flags import ShapeFlags


class TestBroadPhaseGrid(unittest.TestCase):
    def test_grid_matches_sap_with_moving_and_large_shapes(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(17)
        shape_count = 128
        world_np = rng.choice(np.array([-1, 0, 1], dtype=np.int32), shape_count)
        world_np[:3] = 0
        group_np = rng.choice(np.array([-2, -1, 0, 1, 2], dtype=np.int32), shape_count)
        group_np[:3] = 1
        flags_np = np.full(shape_count, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        flags_np[::11] = 0
        flags_np[:3] = int(ShapeFlags.COLLIDE_SHAPES)
        world = wp.array(world_np, dtype=wp.int32, device=device)
        group = wp.array(group_np, dtype=wp.int32, device=device)
        flags = wp.array(flags_np, dtype=wp.int32, device=device)
        gap = wp.array(rng.uniform(0.0, 0.04, shape_count).astype(np.float32), dtype=wp.float32, device=device)
        displacement = wp.array(
            rng.uniform(-0.12, 0.12, (shape_count, 3)).astype(np.float32), dtype=wp.vec3, device=device
        )
        lower = wp.empty(shape_count, dtype=wp.vec3, device=device)
        upper = wp.empty(shape_count, dtype=wp.vec3, device=device)
        excluded = wp.array(np.array([[1, 2], [20, 31]], dtype=np.int32), dtype=wp.vec2i, device=device)

        sap = BroadPhaseSAP(world, shape_flags=flags, direction_search=False, device=device)
        # One slot per shape forces the overflow path. Shape zero also spans
        # many cells and must use the large-shape side pass.
        grids = [
            BroadPhaseGrid(world, shape_flags=flags, capacity_factor=1, pair_mode=mode, device=device)
            for mode in ("scalar", "warp_deterministic")
        ]
        pairs_sap = wp.empty(shape_count * shape_count, dtype=wp.vec2i, device=device)
        pairs_grid = wp.empty(shape_count * shape_count, dtype=wp.vec2i, device=device)
        count_sap = wp.zeros(1, dtype=wp.int32, device=device)
        count_grid = wp.zeros(1, dtype=wp.int32, device=device)

        def launch(broad_phase, pairs, count):
            broad_phase.launch(
                lower,
                upper,
                gap,
                group,
                world,
                shape_count,
                pairs,
                count,
                device=device,
                filter_pairs=excluded,
                shape_displacement=displacement,
            )

        def pair_set(pairs, count):
            num_pairs = int(count.numpy()[0])
            actual = {tuple(pair) for pair in pairs.numpy()[:num_pairs]}
            self.assertEqual(len(actual), num_pairs, "duplicate grid pairs")
            return actual

        for _ in range(3):
            centers = rng.uniform(-1.0, 1.0, (shape_count, 3)).astype(np.float32)
            half_sizes = rng.uniform(0.03, 0.25, (shape_count, 3)).astype(np.float32)
            low = centers - half_sizes
            high = centers + half_sizes
            low[0] = (-5.0, -5.0, -5.0)
            high[0] = (5.0, 5.0, 5.0)
            # A second pair touches exactly at a face after gap expansion.
            low[1] = (0.0, 0.0, 0.0)
            high[1] = (0.1, 0.1, 0.1)
            low[2] = (0.1, 0.0, 0.0)
            high[2] = (0.2, 0.1, 0.1)
            lower.assign(low)
            upper.assign(high)
            launch(sap, pairs_sap, count_sap)
            expected = pair_set(pairs_sap, count_sap)
            for grid in grids:
                launch(grid, pairs_grid, count_grid)
                self.assertEqual(expected, pair_set(pairs_grid, count_grid))
                if grid.pair_mode == "warp_deterministic":
                    emitted = pairs_grid.numpy()[: int(count_grid.numpy()[0])].copy()
                    launch(grid, pairs_grid, count_grid)
                    np.testing.assert_array_equal(emitted, pairs_grid.numpy()[: len(emitted)])

        for grid in grids:
            count_grid.zero_()
            with wp.ScopedCapture(device=device) as capture:
                launch(grid, pairs_grid, count_grid)
            wp.capture_launch(capture.graph)
            self.assertEqual(pair_set(pairs_sap, count_sap), pair_set(pairs_grid, count_grid))


if __name__ == "__main__":
    unittest.main()
