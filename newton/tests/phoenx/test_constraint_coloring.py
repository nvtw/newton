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

            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                step_layout="multi_world",
                multi_world_scheduler="fast_tail",
                substeps=1,
            )
            solver.world._use_greedy_coloring = False
            state, next_state = model.state(), model.state()
            contacts = pipeline.contacts()
            control = model.control()
            for _ in range(24):
                solver.collide(state, contacts)
                solver.step(state, next_state, control, contacts, 1.0 / 240.0)
                state, next_state = next_state, state

            np.testing.assert_allclose(state.body_q.numpy()[:, 2], radius, atol=1.0e-4)
            np.testing.assert_allclose(state.body_qd.numpy()[:, 2], 0.0, atol=1.0e-4)
