# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare independent island blocks to the original color-ordered CUDA sweep."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.islands.iteration_schedule import _index_colors, _key_island_order
from newton._src.solvers.phoenx.tests.test_parallel_prepare_sleeping import _build_model


class TestIslandIterationSchedule(unittest.TestCase):
    def test_color_index_covers_the_entire_island(self):
        """Index every live slot with its color and the end of that color's run."""
        device = "cpu"
        keys = wp.array([0, 0, 0, 0, 1, 2], dtype=int, device=device)
        slots = wp.array([0, 1, 3, 4, 7, 8], dtype=int, device=device)
        island_ends = wp.array([4, 5], dtype=int, device=device)
        starts = wp.array([0, 2, 3, 5, 8], dtype=int, device=device)
        num_colors = wp.array([4], dtype=int, device=device)
        families = wp.empty(0, dtype=int, device=device)
        colors = wp.full(6, -1, dtype=int, device=device)
        ends = wp.full_like(colors, -1)
        wp.launch(
            _index_colors,
            dim=len(keys),
            inputs=[keys, slots, island_ends, 2, starts, num_colors, families, False, colors, ends, families],
            device=device,
        )
        np.testing.assert_array_equal(colors.numpy(), [0, 0, 2, 2, 3, -1])
        np.testing.assert_array_equal(ends.numpy(), [2, -1, 4, -1, 5, -1])

    def test_islands_are_ordered_longest_first(self):
        """Order every island once by its color span, longest first, ties in index order.

        Blocks deal the ordered islands out in alternating directions, so the
        longest serial color chains are spread over different blocks.
        """
        device = "cpu"
        roots = wp.array([3, 1, 0, 4, -1, -1], dtype=int, device=device)
        count = wp.array([4], dtype=int, device=device)
        first_color = wp.array([0, 2, 0, 5, 1], dtype=int, device=device)
        last_color = wp.array([1, 9, 0, 6, 9], dtype=int, device=device)
        keys = wp.empty(2 * len(roots), dtype=int, device=device)
        order = wp.empty(2 * len(roots), dtype=int, device=device)
        wp.launch(
            _key_island_order,
            dim=len(roots),
            inputs=[roots, count, first_color, last_color, keys, order],
            device=device,
        )
        wp.utils.radix_sort_pairs(keys, order, len(roots), end_bit=17)
        # Spans: island 4 -> 9, 1 -> 8, 3 -> 2, 0 -> 2 (ties keep their index order).
        np.testing.assert_array_equal(order.numpy()[:4], [4, 1, 3, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
    def test_mixed_islands_in_captured_sweeps(self):
        """Preserve poses and impulses through multiple warm starts on the same island schedule."""
        # The stack and anchored pendulum share the static world body. Their
        # contacts and joints must form independent writable components.
        with wp.ScopedDevice("cuda:0"):
            model = _build_model()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")

            def run(independent, giant=False, temporal=False):
                solver = newton.solvers.SolverPhoenX(
                    model,
                    collision_pipeline=pipeline,
                    step_layout="single_world",
                    joint_mode="maximal_pgs",
                    mass_splitting=False,
                    substeps=2,
                    parallel_contact_prepare=True,
                    solver_iterations=3,
                    island_parallel_iterations=independent,
                    temporal_iterations=temporal,
                )
                world = solver.world
                # Island blocks solve every color with the head kernel. Without a
                # fused tail, the color-ordered sweep uses the same arithmetic.
                world._fuse_threshold = 0
                if giant:
                    # Every island is solved by the whole grid, one color at a time.
                    schedule = world._dispatcher._island_schedule
                    schedule.merge_width = 0
                    # Block 0 walks the narrow colors, the grid the wide ones.
                    schedule.fuse_columns = 1
                state, control, contacts = model.state(), model.control(), pipeline.contacts()

                def step():
                    pipeline.collide(state, contacts)
                    solver.step(state, state, control, contacts, 1 / 240)

                step()
                with wp.ScopedCapture() as capture:
                    step()
                history = []
                for _ in range(8):
                    wp.capture_launch(capture.graph)
                    history.append(
                        (state.body_q.numpy(), state.body_qd.numpy(), world._contact_container.impulses.numpy())
                    )
                if independent and not giant:
                    schedule = world._dispatcher._island_schedule
                    self.assertGreaterEqual(int(schedule.count.numpy()[0]), 2)
                    self.assertGreater(int(world._partitioner.num_colors.numpy()[0]), 1)
                return history

            for temporal in (False, True):
                reference = run(False, temporal=temporal)
                for actual in (run(True, temporal=temporal), run(True, giant=True, temporal=temporal)):
                    for expected_step, actual_step in zip(reference, actual, strict=True):
                        for expected, result in zip(expected_step, actual_step, strict=True):
                            np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))


if __name__ == "__main__":
    unittest.main()
