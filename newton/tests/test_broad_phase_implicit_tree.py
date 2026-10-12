# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact candidate pairs and graph replay for the experimental implicit BVH."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_implicit_tree import BroadPhaseImplicitTree
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton._src.geometry.flags import ShapeFlags


class TestBroadPhaseImplicitTree(unittest.TestCase):
    def test_swept_pairs_without_pair_queues_match_all_pairs(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(42)
        n = 96
        world_np = rng.choice(np.array([-1, 0, 1], dtype=np.int32), n)
        world_np[:3] = 0
        group_np = rng.choice(np.array([-2, -1, 0, 1, 2], dtype=np.int32), n)
        group_np[:3] = 1
        flags_np = np.full(n, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        flags_np[::13] = 0
        flags_np[:3] = int(ShapeFlags.COLLIDE_SHAPES)
        world = wp.array(world_np, dtype=wp.int32, device=device)
        group = wp.array(group_np, dtype=wp.int32, device=device)
        flags = wp.array(flags_np, dtype=wp.int32, device=device)
        gap = wp.array(rng.uniform(0.0, 0.04, n).astype(np.float32), dtype=wp.float32, device=device)
        displacement = wp.array(rng.uniform(-0.2, 0.2, (n, 3)).astype(np.float32), dtype=wp.vec3, device=device)
        lower = wp.empty(n, dtype=wp.vec3, device=device)
        upper = wp.empty(n, dtype=wp.vec3, device=device)
        excluded = wp.array(np.array([[1, 2], [20, 31]], dtype=np.int32), dtype=wp.vec2i, device=device)
        sap = BroadPhaseSAP(world, shape_flags=flags, direction_search=False, device=device)
        oracle = BroadPhaseAllPairs(world, shape_flags=flags, device=device)
        tree = BroadPhaseImplicitTree(world, shape_flags=flags, queue_capacity_factor=64, device=device)
        overflow_tree = BroadPhaseImplicitTree(world, shape_flags=flags, queue_capacity_factor=1, device=device)
        arrays = []
        for value in vars(tree).values():
            if isinstance(value, wp.array):
                arrays.append(value)
            elif isinstance(value, list):
                arrays.extend(item for item in value if isinstance(item, wp.array))
        self.assertLessEqual(sum(array.capacity for array in arrays), 400 * n)
        pairs = [wp.empty(n * n, dtype=wp.vec2i, device=device) for _ in range(3)]
        counts = [wp.zeros(1, dtype=wp.int32, device=device) for _ in range(3)]
        oracle_pairs = wp.empty(n * n, dtype=wp.vec2i, device=device)
        oracle_count = wp.zeros(1, dtype=wp.int32, device=device)

        def launch(bp, out, count):
            bp.launch(
                lower,
                upper,
                gap,
                group,
                world,
                n,
                out,
                count,
                device=device,
                filter_pairs=excluded,
                shape_displacement=displacement,
            )

        def pair_set(out, count):
            size = int(count.numpy()[0])
            result = {tuple(p) for p in out.numpy()[:size]}
            self.assertEqual(len(result), size)
            return result

        for shift in (0.0, 2.0e6, -2.0e6):
            centers = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
            half = rng.uniform(0.03, 0.4, (n, 3)).astype(np.float32)
            lo = centers - half
            hi = centers + half
            lo[0] = (-5.0, -5.0, -5.0)
            hi[0] = (5.0, 5.0, 5.0)
            lo[1] = (0.0, 0.0, 0.0)
            hi[1] = (0.1, 0.1, 0.1)
            lo[2] = (0.1, 0.0, 0.0)
            hi[2] = (0.2, 0.1, 0.1)
            lo += shift
            hi += shift
            lower.assign(lo)
            upper.assign(hi)
            launch(oracle, oracle_pairs, oracle_count)
            for i, bp in enumerate((sap, tree, overflow_tree)):
                launch(bp, pairs[i], counts[i])
            expected = pair_set(oracle_pairs, oracle_count)
            if shift == 0.0:
                self.assertEqual(pair_set(pairs[0], counts[0]), expected)
            self.assertEqual(pair_set(pairs[1], counts[1]), expected)
            self.assertEqual(pair_set(pairs[2], counts[2]), expected)

        small_pairs = wp.empty(8, dtype=wp.vec2i, device=device)
        small_count = wp.zeros(1, dtype=wp.int32, device=device)
        launch(tree, small_pairs, small_count)
        self.assertGreater(len(expected), len(small_pairs))
        self.assertEqual(int(small_count.numpy()[0]), len(expected))
        self.assertTrue({tuple(pair) for pair in small_pairs.numpy()}.issubset(expected))

        with wp.ScopedCapture(device=device) as capture:
            launch(tree, pairs[1], counts[1])
            launch(overflow_tree, pairs[2], counts[2])
        for _ in range(2):
            wp.capture_launch(capture.graph)
            self.assertEqual(pair_set(pairs[1], counts[1]), expected)
            self.assertEqual(pair_set(pairs[2], counts[2]), expected)

        for i, bp in ((1, tree), (2, overflow_tree)):
            counts[i].assign(np.array([1], dtype=np.int32))
            bp.launch(
                lower,
                upper,
                gap,
                group,
                world,
                n,
                pairs[i],
                counts[i],
                device=device,
                filter_pairs=excluded,
                shape_displacement=displacement,
                skip_count_zero=True,
            )
            self.assertEqual(int(counts[i].numpy()[0]), len(expected) + 1)
            self.assertEqual({tuple(p) for p in pairs[i].numpy()[1 : len(expected) + 1]}, expected)


if __name__ == "__main__":
    unittest.main()
