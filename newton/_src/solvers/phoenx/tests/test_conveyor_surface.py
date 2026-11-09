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


@unittest.skipUnless(wp.is_cuda_available(), "conveyor tests use CUDA graph capture")
class TestConveyorContactModes(unittest.TestCase):
    def test_spatial_vertex_velocities_in_contact_modes(self):
        """Preserve interpolated slip through contact grouping and patch fallback."""
        for friction in ("point", "patch"):
            for grouping in (False, True):
                for sleeping in (0.0, 0.02):
                    with self.subTest(friction=friction, grouping=grouping, sleeping=sleeping):
                        builder = newton.ModelBuilder()
                        cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
                        bodies = []
                        # Add boxes before the mesh so canonical body ordering can
                        # reverse the narrow phase's shape endpoints during grouping.
                        for y in (-1.0, 1.0):
                            body = builder.add_body(xform=wp.transform((0.0, y, 0.1), wp.quat_identity()))
                            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                            bodies.append(body)
                        stationary = builder.add_body(xform=wp.transform((0.0, 5.0, 0.1), wp.quat_identity()))
                        builder.add_shape_box(stationary, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                        builder.add_shape_box(
                            -1,
                            hx=1.0,
                            hy=0.5,
                            hz=0.1,
                            xform=wp.transform((0.0, 5.0, -0.1), wp.quat_identity()),
                            cfg=cfg,
                        )
                        vertices = np.array(
                            [[-4.0, -3.0, 0.0], [4.0, -3.0, 0.0], [4.0, 3.0, 0.0], [-4.0, 3.0, 0.0]], dtype=np.float32
                        )
                        mesh = newton.Mesh(
                            vertices,
                            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
                            compute_inertia=False,
                            enable_surface_velocity=True,
                        )
                        builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
                        model = builder.finalize()
                        solver = newton.solvers.SolverPhoenX(
                            model,
                            step_layout="single_world",
                            substeps=4,
                            contact_friction_model=friction,
                            enable_body_pair_grouping=grouping,
                            sleeping_velocity_threshold=sleeping,
                        )
                        state, next_state = model.state(), model.state()
                        contacts = model._collision_pipeline.contacts()
                        with wp.ScopedCapture() as capture:
                            solver.collide(state, contacts)
                            solver.step(state, next_state, None, contacts, 1.0 / 60.0)
                            wp.copy(state.body_q, next_state.body_q)
                            wp.copy(state.body_qd, next_state.body_qd)
                        # An affine field needs only two triangles: tessellation must
                        # not replace per-contact interpolation with one mesh speed.
                        for scale in (0.0, 1.0, -1.0, 0.0, 1.0) if sleeping else (1.0, -1.0, 0.0):
                            velocities = np.zeros_like(vertices)
                            velocities[:, 0] = scale * (0.4 + 0.2 * vertices[:, 1])
                            mesh.mesh.velocities.assign(velocities)
                            for _ in range(90):
                                wp.capture_launch(capture.graph)
                            poses = state.body_q.numpy()[bodies]
                            velocity = state.body_qd.numpy()[bodies]
                            expected = scale * (0.4 + 0.2 * poses[:, 1])
                            np.testing.assert_allclose(velocity[:, 0], expected, atol=0.03)
                            np.testing.assert_allclose(velocity[:, 1:3], 0.0, atol=0.03)
                            np.testing.assert_allclose(poses[:, 2], 0.1, atol=0.004)
                            np.testing.assert_allclose(state.body_qd.numpy()[stationary], 0.0, atol=0.004)
                            np.testing.assert_allclose(
                                state.body_q.numpy()[stationary, :3], (0.0, 5.0, 0.1), atol=0.004
                            )

                            if sleeping:
                                roots = solver.world.bodies.island_root.numpy()
                                self.assertGreaterEqual(roots[stationary + 1], 0)
                                self.assertTrue(np.all(roots[np.asarray(bodies) + 1] < 0))


if __name__ == "__main__":
    unittest.main()
