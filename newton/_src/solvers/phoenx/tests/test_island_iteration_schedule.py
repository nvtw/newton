# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare independent island blocks to the original color-ordered CUDA sweep."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.islands.iteration_schedule import _index_colors
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
        count = wp.array([2, 0], dtype=int, device=device)
        wp.launch(
            _index_colors,
            dim=len(keys),
            inputs=[keys, slots, island_ends, 2, starts, num_colors, families, False, colors, ends, families, count],
            device=device,
        )
        np.testing.assert_array_equal(colors.numpy(), [0, 0, 2, 2, 3, -1])
        np.testing.assert_array_equal(ends.numpy(), [2, -1, 4, -1, 5, -1])

    @unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
    def test_mixed_islands_in_captured_sweeps(self):
        """Preserve poses and impulses through multiple warm starts on the same island schedule."""
        # The stack and anchored pendulum share the static world body. Their
        # contacts and joints must form independent writable components.
        with wp.ScopedDevice("cuda:0"):
            model = _build_model()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")

            def run(independent):
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
                )
                world = solver.world
                # Island blocks solve every color with the head kernel. Without a
                # fused tail, the color-ordered sweep uses the same arithmetic.
                world._fuse_threshold = 0
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
                if independent:
                    schedule = world._dispatcher._island_schedule
                    self.assertGreaterEqual(int(schedule.count.numpy()[0]), 2)
                    self.assertGreater(int(world._partitioner.num_colors.numpy()[0]), 1)
                return history

            for reference, actual in zip(run(False), run(True), strict=True):
                for expected, result in zip(reference, actual, strict=True):
                    np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))


if __name__ == "__main__":
    unittest.main()
