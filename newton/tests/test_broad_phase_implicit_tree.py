# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact candidate pairs and graph replay for the experimental implicit BVH."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_common import build_pair_exclusion_index
from newton._src.geometry.broad_phase_implicit_tree import BroadPhaseImplicitTree
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton._src.geometry.flags import ShapeFlags


@wp.struct
class _ShapeActivity:
    active: wp.array[int]


@wp.func
def _passive(shape: int, data: _ShapeActivity) -> bool:
    return data.active[shape] == 0


@wp.func
def _has_active_endpoint(pair: wp.vec2i, data: _ShapeActivity) -> int:
    return int(not (_passive(pair[0], data) and _passive(pair[1], data)))


class TestBroadPhaseImplicitTree(unittest.TestCase):
    def test_active_shape_finds_passive_neighbors_of_any_size(self):
        """Keep mixed pairs when the active endpoint changes or the filter is removed."""
        device = wp.get_device("cpu")
        world = wp.zeros(3, dtype=int, device=device)
        group = wp.ones_like(world)
        half = np.array([[0.1] * 3, [1.0] * 3, [2.0e38] * 3], dtype=np.float32)
        lower = wp.array(-half, dtype=wp.vec3, device=device)
        upper = wp.array(half, dtype=wp.vec3, device=device)
        pairs = wp.empty(3, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=int, device=device)
        tree = BroadPhaseImplicitTree(
            world,
            device=device,
            filter_func=_has_active_endpoint,
            filter_data_type=_ShapeActivity,
            passive_shape=_passive,
        )
        data = _ShapeActivity()
        data.active = wp.zeros_like(world)
        for active in ([1, 0, 0], [0, 0, 1], [0, 0, 0]):
            data.active.assign(np.array(active, dtype=np.int32))
            tree.launch(lower, upper, None, group, world, 3, pairs, count, device=device, filter_data=data)
            actual = sorted(map(tuple, pairs.numpy()[: int(count.numpy()[0])]))
            self.assertEqual(actual, [(a, b) for a, b in [(0, 1), (0, 2), (1, 2)] if active[a] or active[b]])

        tree._set_filter_kernels(None, None)
        tree.launch(lower, upper, None, group, world, 3, pairs, count, device=device)
        self.assertEqual(sorted(map(tuple, pairs.numpy()[: int(count.numpy()[0])])), [(0, 1), (0, 2), (1, 2)])

    def test_runtime_rows_pair_only_while_their_flags_collide(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(7)
        n, reserved = 80, 24
        dynamic_rows = np.arange(n) >= n - reserved
        centers = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        half = rng.uniform(0.05, 0.4, (n, 3)).astype(np.float32)
        lower = wp.array(centers - half, dtype=wp.vec3, device=device)
        upper = wp.array(centers + half, dtype=wp.vec3, device=device)
        world = wp.zeros(n, dtype=wp.int32, device=device)
        group = wp.full(n, 1, dtype=wp.int32, device=device)
        flags_np = np.where(dynamic_rows, 0, int(ShapeFlags.COLLIDE_SHAPES)).astype(np.int32)
        flags = wp.array(flags_np, dtype=wp.int32, device=device)
        tree = BroadPhaseImplicitTree(world, shape_flags=flags, device=device, dynamic_rows=dynamic_rows)
        pairs = wp.empty(n * n, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as capture:
            tree.launch(lower, upper, None, group, world, n, pairs, count, device=device)

        def expected(current: np.ndarray) -> set:
            oracle = BroadPhaseAllPairs(world, shape_flags=current, device=device)
            out = wp.empty(n * n, dtype=wp.vec2i, device=device)
            size = wp.zeros(1, dtype=wp.int32, device=device)
            oracle.launch(lower, upper, None, group, world, n, out, size, device=device)
            return {tuple(pair) for pair in out.numpy()[: int(size.numpy()[0])]}

        sizes = []
        # No runtime shape, some, others, none again: the captured launch follows the flags.
        for active in (np.zeros(n, bool), dynamic_rows & (np.arange(n) % 2 == 0), dynamic_rows, np.zeros(n, bool)):
            current = np.where(active, int(ShapeFlags.COLLIDE_SHAPES), flags_np).astype(np.int32)
            flags.assign(current)
            wp.capture_launch(capture.graph)
            found = {tuple(pair) for pair in pairs.numpy()[: int(count.numpy()[0])]}
            self.assertEqual(found, expected(current))
            sizes.append(len(found))
        self.assertGreater(sizes[2], sizes[1])
        self.assertGreater(sizes[1], sizes[0])
        self.assertEqual(sizes[3], sizes[0])

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

    def test_sparse_active_leaves_with_strided_query_match_all_pairs(self):
        """Every sibling level of every active leaf is walked, also when lanes take several work items."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(7)
        n = 3000
        world = wp.zeros(n, dtype=wp.int32, device=device)
        group = wp.ones(n, dtype=wp.int32, device=device)
        centers = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
        half = rng.uniform(0.01, 0.15, (n, 3)).astype(np.float32)
        # A few large boxes span many subtrees, as static scenery does.
        half[:5] = 3.0
        lower = wp.array(centers - half, dtype=wp.vec3, device=device)
        upper = wp.array(centers + half, dtype=wp.vec3, device=device)
        data = _ShapeActivity()
        # Mostly passive shapes with sparse active ones, like a sleeping scene.
        active = (rng.uniform(size=n) < 0.1).astype(np.int32)
        active[:2] = (1, 0)
        data.active = wp.array(active, dtype=wp.int32, device=device)
        tree = BroadPhaseImplicitTree(
            world,
            device=device,
            filter_func=_has_active_endpoint,
            filter_data_type=_ShapeActivity,
            passive_shape=_passive,
        )
        oracle = BroadPhaseAllPairs(world, device=device)
        pairs = wp.empty(n * 64, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        oracle_pairs = wp.empty(n * 64, dtype=wp.vec2i, device=device)
        oracle_count = wp.zeros(1, dtype=wp.int32, device=device)
        oracle.launch(lower, upper, None, group, world, n, oracle_pairs, oracle_count, device=device)
        expected = {
            tuple(p) for p in oracle_pairs.numpy()[: int(oracle_count.numpy()[0])] if active[p[0]] or active[p[1]]
        }
        self.assertGreater(len(expected), 100)
        # A launch narrower than the work lets each lane loop over several (leaf, level) items.
        for threads in (tree._query_threads, 97):
            tree._query_threads = threads
            with wp.ScopedCapture(device=device) as capture:
                tree.launch(lower, upper, None, group, world, n, pairs, count, device=device, filter_data=data)
            wp.capture_launch(capture.graph)
            size = int(count.numpy()[0])
            found = {tuple(p) for p in pairs.numpy()[:size]}
            self.assertEqual(len(found), size)
            self.assertEqual(found, expected)

    def test_reused_pairs_match_all_pairs_on_every_launch(self):
        """Kept pairs give the exact candidates while shapes move, change flags, wake, and fall asleep."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(3)
        n = 600
        world = wp.zeros(n, dtype=wp.int32, device=device)
        group = wp.ones(n, dtype=wp.int32, device=device)
        flags_np = np.full(n, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        flags_np[::17] = 0
        flags = wp.array(flags_np, dtype=wp.int32, device=device)
        centers = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        half = rng.uniform(0.02, 0.12, (n, 3)).astype(np.float32)
        velocity = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
        lower = wp.empty(n, dtype=wp.vec3, device=device)
        upper = wp.empty(n, dtype=wp.vec3, device=device)
        displacement = wp.empty(n, dtype=wp.vec3, device=device)
        gap = wp.array(rng.uniform(0.0, 0.01, n).astype(np.float32), dtype=wp.float32, device=device)
        active = (rng.uniform(size=n) < 0.7).astype(np.int32)
        passive_at_build = np.flatnonzero(active == 0)
        data = _ShapeActivity()
        data.active = wp.array(active, dtype=wp.int32, device=device)
        tree = BroadPhaseImplicitTree(
            world,
            shape_flags=flags,
            dynamic_rows=np.ones(n, dtype=bool),
            device=device,
            filter_func=_has_active_endpoint,
            filter_data_type=_ShapeActivity,
            passive_shape=_passive,
            pair_reuse_slack=0.001,
        )
        oracle = BroadPhaseAllPairs(world, device=device)
        pairs = wp.empty(n * 64, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        oracle_pairs = wp.empty(n * 64, dtype=wp.vec2i, device=device)
        oracle_count = wp.zeros(1, dtype=wp.int32, device=device)
        dt = 0.01

        def launch(bp, out, size, **kwargs):
            bp.launch(
                lower, upper, gap, group, world, n, out, size, device=device, shape_displacement=displacement, **kwargs
            )

        with wp.ScopedCapture(device=device) as capture:
            launch(tree, pairs, count, filter_data=data)

        def step(expect_rebuild):
            builds = int(tree._rebuild.numpy()[1])
            lower.assign(centers - half)
            upper.assign(centers + half)
            displacement.assign(velocity * dt)
            flags.assign(flags_np)
            data.active.assign(active)
            wp.capture_launch(capture.graph)
            launch(oracle, oracle_pairs, oracle_count)
            # The oracle read the flags once; apply the current ones.
            expected = {
                tuple(p)
                for p in oracle_pairs.numpy()[: int(oracle_count.numpy()[0])]
                if (active[p[0]] or active[p[1]]) and flags_np[p[0]] and flags_np[p[1]]
            }
            size = int(count.numpy()[0])
            found = {tuple(p) for p in pairs.numpy()[:size]}
            self.assertEqual(len(found), size)
            self.assertEqual(found, expected)
            self.assertEqual(int(tree._rebuild.numpy()[1]) - builds, int(expect_rebuild))
            return len(expected)

        self.assertGreater(step(True), 200)
        for _ in range(3):
            # Steady motion stays within the bounds that cover the next pass.
            centers += velocity * dt
            step(False)
            centers += velocity * dt
            step(True)
        # A shape that falls asleep only loses pairs; one that stops colliding too.
        active[np.flatnonzero(active)[:50]] = 0
        flags_np[1::19] = 0
        centers += velocity * dt
        step(False)
        # A shape that wakes up, starts colliding, or jumps needs a rebuild.
        for change in ("wake", "collide", "jump"):
            if change == "wake":
                active[next(i for i in passive_at_build if flags_np[i])] = 1
            elif change == "collide":
                flags_np[0] = int(ShapeFlags.COLLIDE_SHAPES)
            else:
                centers[5] += 0.05
            step(True)
            step(False)

    def test_indexed_exclusions_match_sorted_prefix(self):
        model_shapes, shapes = 12, 15
        # Include sparse rows, a missing first shape, and extra runtime shape ids.
        exclusions = np.array(
            [(a, b) for a in range(model_shapes) if a % 3 != 1 for b in range(a + 1, model_shapes)], dtype=np.int32
        )

        def check(device):
            lower = wp.full(shapes, wp.vec3(-1.0), dtype=wp.vec3, device=device)
            upper = wp.full(shapes, wp.vec3(1.0), dtype=wp.vec3, device=device)
            worlds = wp.zeros(shapes, dtype=wp.int32, device=device)
            groups = wp.ones(shapes, dtype=wp.int32, device=device)
            storage = wp.empty(shapes * shapes, dtype=wp.vec2i, device=device)
            count = wp.zeros(1, dtype=wp.int32, device=device)
            tree = BroadPhaseImplicitTree(worlds, device=device)
            index = build_pair_exclusion_index(exclusions, model_shapes, device)
            flat = wp.array(exclusions, dtype=wp.vec2i, device=device)

            def visit(filter_pairs, size):
                tree.launch(
                    lower,
                    upper,
                    None,
                    groups,
                    worlds,
                    shapes,
                    storage,
                    count,
                    filter_pairs=filter_pairs,
                    num_filter_pairs=size,
                    device=device,
                )

            def pairs():
                size = int(count.numpy()[0])
                found = {tuple(pair) for pair in storage.numpy()[:size]}
                self.assertEqual(len(found), size)
                return found

            for size in (0, len(exclusions) // 2, None):
                visit(flat, size)
                expected = pairs()
                visit(index, size)
                self.assertEqual(pairs(), expected)
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as captured:
                        visit(index, size)
                    wp.capture_launch(captured.graph)
                    self.assertEqual(pairs(), expected)

        for device in wp.get_devices():
            with self.subTest(device=device):
                check(device)


if __name__ == "__main__":
    unittest.main()
