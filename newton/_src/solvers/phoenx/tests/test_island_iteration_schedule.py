# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare independent island blocks to the original color-ordered CUDA sweep."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_parallel_prepare_sleeping import _build_model


@unittest.skipUnless(wp.is_cuda_available(), "Island iteration scheduling requires CUDA")
class TestIslandIterationSchedule(unittest.TestCase):
    def test_mixed_islands_in_captured_sweeps(self):
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
                    substeps=1,
                    solver_iterations=3,
                    island_parallel_iterations=independent,
                )
                world = solver.world
                # Exercise large and small colors on a scene small enough for a unit test.
                world._fuse_threshold = 1
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
                    starts = world._partitioner.color_starts.numpy()[
                        : int(world._partitioner.num_colors.numpy()[0]) + 1
                    ]
                    self.assertGreater(int(np.diff(starts).max()), world._fuse_threshold)
                return history

            for reference, actual in zip(run(False), run(True), strict=True):
                for expected, result in zip(reference, actual, strict=True):
                    np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))


if __name__ == "__main__":
    unittest.main()
