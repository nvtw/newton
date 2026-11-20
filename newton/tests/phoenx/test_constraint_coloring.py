# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify PhoenX partitioning includes all physical constraints."""

import unittest

import numpy as np
import warp as wp

import newton


class TestCpuWorldColoring(unittest.TestCase):
    def test_jp_coloring_supports_every_sphere(self):
        """The per-world JP fallback must color every ground contact on CPU."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            radius = 0.1
            for x in range(4):
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, radius), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=radius)

            state = self._settle(builder, 24, jp_coloring=True)
            np.testing.assert_allclose(state.body_q.numpy()[:, 2], radius, atol=1.0e-4)
            np.testing.assert_allclose(state.body_qd.numpy()[:, 2], 0.0, atol=1.0e-4)

    def test_settled_stack_is_independent_of_body_order(self):
        """Every contact color must finish before any lane starts the next sweep."""
        for scheduler in ("fast_tail", "block_world"):
            with self.subTest(scheduler=scheduler), wp.ScopedDevice("cpu"):
                heights = []
                for order in (range(6), reversed(range(6))):
                    builder = newton.ModelBuilder()
                    builder.add_ground_plane()
                    for level in order:
                        center = wp.vec3(0.0, 0.0, 0.1 + 0.2 * level)
                        body = builder.add_body(xform=wp.transform(center, wp.quat_identity()))
                        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.1)
                    state = self._settle(builder, 96, scheduler=scheduler)
                    heights.append(np.sort(state.body_q.numpy()[:, 2]))

                np.testing.assert_allclose(heights[0], heights[1], atol=1.0e-4, rtol=0.0)

    @staticmethod
    def _settle(builder, steps, *, jp_coloring=False, scheduler="fast_tail"):
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            step_layout="multi_world",
            multi_world_scheduler=scheduler,
            substeps=1,
            solver_iterations=8,
            velocity_iterations=1,
        )
        if jp_coloring:
            solver.world._use_greedy_coloring = False
        state, next_state = model.state(), model.state()
        contacts = pipeline.contacts()
        control = model.control()
        for _ in range(steps):
            solver.collide(state, contacts)
            solver.step(state, next_state, control, contacts, 1.0 / 240.0)
            state, next_state = next_state, state
        return state
