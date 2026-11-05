# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conveyor surfaces: PhoenX friction drives contacts to the mesh surface velocity."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


@unittest.skipUnless(wp.is_cuda_available(), "conveyor tests use CUDA graph capture")
class TestConveyorSurface(unittest.TestCase):
    def test_box_follows_belt_and_stops_with_it(self):
        vertices = np.array([[-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [3.0, 3.0, 0.0], [-3.0, 3.0, 0.0]], dtype=np.float32)
        belt = newton.Mesh(
            vertices, np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), compute_inertia=False, enable_surface_velocity=True
        )
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(body=-1, mesh=belt, cfg=cfg)
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, step_layout="single_world", substeps=4)
        state_0, state_1 = model.state(), model.state()
        contacts = model._collision_pipeline.contacts()

        with wp.ScopedCapture() as capture:
            solver.collide(state_0, contacts)
            solver.step(state_0, state_1, None, contacts, 1.0 / 60.0)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        for surface_velocity in ((1.0, 0.0, 0.0), (0.0, -0.5, 0.0), (0.0, 0.0, 0.0)):
            belt.mesh.velocities.fill_(wp.vec3(*surface_velocity))
            for _ in range(60):
                wp.capture_launch(capture.graph)
            qd = state_0.body_qd.numpy()[body]
            np.testing.assert_allclose(qd[:3], surface_velocity, atol=2.0e-2)
            np.testing.assert_allclose(qd[3:], 0.0, atol=5.0e-2)
            self.assertAlmostEqual(float(state_0.body_q.numpy()[body][2]), 0.1, delta=3.0e-3)


if __name__ == "__main__":
    unittest.main()
