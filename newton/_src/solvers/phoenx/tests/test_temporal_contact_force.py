# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Temporal force readout must reproduce the actual contact impulse wrench."""

import unittest

import numpy as np
import warp as wp

import newton


@unittest.skipUnless(wp.is_cuda_available(), "Temporal contacts require CUDA")
class TestTemporalContactForce(unittest.TestCase):
    def test_contact_wrench_matches_momentum_change(self):
        """Include anchor friction, COM torque, outer-step scaling and reset."""
        self._check_wrenches(False)

    def test_dynamic_contact_wrench_matches_momentum_change(self):
        """Include paired dynamic-anchor forces and moments under graph replay."""
        self._check_wrenches(True)

    def _check_wrenches(self, dynamic_support):
        builder = newton.ModelBuilder()
        if not dynamic_support:
            builder.add_ground_plane()
        body = builder.add_body(
            xform=wp.transform((1.2, -0.7, 0.099), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01),
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6))
        if dynamic_support:
            support = builder.add_body(
                xform=wp.transform((1.2, -0.7, -0.1), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(0.02, 0, 0, 0, 0.02, 0, 0, 0, 0.02),
            )
            builder.add_shape_box(
                support, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6)
            )
        model = builder.finalize(device="cuda:0")
        model.request_contact_attributes("force")
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            solver_scheme="tgs",
            joint_solver="block_pgs",
            articulation_mode="maximal",
            step_layout="single_world",
            substeps=4,
            solver_iterations=1,
            velocity_iterations=1,
            mass_splitting=True,
            mass_splitting_color_group_size=4,
            prepare_refresh_stride=1,
            sor_boost=1.0,
        )
        state = model.state()
        initial = np.zeros((model.body_count, 6), dtype=np.float32)
        initial[body] = [0.7, 0.2, -0.1, 0.1, 0.3, 0.2]
        state.body_qd.assign(initial)
        masses = model.body_mass.numpy()
        dt = 1.0 / 120
        shape_body = model.shape_body.numpy()
        graph = None

        def advance():
            pipeline.collide(state, contacts)
            solver.step(state, state, None, contacts, dt)
            solver.update_contacts(contacts, state)

        for frame in range(6):
            q0 = state.body_q.numpy()[:, :3].astype(float)
            v0 = state.body_qd.numpy().astype(float)
            if frame == 1:
                with wp.ScopedCapture(device=model.device) as capture:
                    advance()
                graph = capture.graph
            if graph is None:
                advance()
            else:
                wp.capture_launch(graph)
            q1 = state.body_q.numpy()[:, :3].astype(float)
            v1 = state.body_qd.numpy().astype(float)
            impulse = np.zeros((model.body_count, 3))
            moment = np.zeros_like(impulse)
            count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
            self.assertGreater(count, 0)
            shapes0 = contacts.rigid_contact_shape0.numpy()
            shapes1 = contacts.rigid_contact_shape1.numpy()
            for k, wrench in enumerate(contacts.force.numpy()[:count].astype(float) * dt):
                b0, b1 = shape_body[shapes0[k]], shape_body[shapes1[k]]
                origin = q1[b0] if b0 >= 0 else np.zeros(3)
                world_moment = wrench[3:] + np.cross(origin, wrench[:3])
                for endpoint, sign in ((b0, 1), (b1, -1)):
                    if endpoint >= 0:
                        impulse[endpoint] += sign * wrench[:3]
                        moment[endpoint] += sign * world_moment
            gravity = masses[:, None] * np.array([0, 0, -9.81]) * dt
            np.testing.assert_allclose(impulse, masses[:, None] * (v1[:, :3] - v0[:, :3]) - gravity, atol=3e-6)
            expected = masses[:, None] * (
                0.01 * (v1[:, 3:] - v0[:, 3:]) + np.cross(q1, v1[:, :3]) - np.cross(q0, v0[:, :3])
            )
            expected -= np.cross(q0, gravity)
            np.testing.assert_allclose(moment, expected, atol=5e-6)
            if frame == 0:
                self.assertGreater(np.linalg.norm(impulse[body, :2]), 1e-5)

        lifted = state.body_q.numpy()
        lifted[0, 2] = 1.0
        state.body_q.assign(lifted)
        wp.capture_launch(graph)
        np.testing.assert_array_equal(contacts.force.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
