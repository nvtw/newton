# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Anisotropic contact response must not add kinetic energy."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.articulations.reduced_forest import ReducedForestContactSystem
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_coupled_velocity_update_no_soft_pd,
    contact_project_friction_metric,
    contact_project_velocity_update_no_soft_pd,
)
from newton._src.solvers.phoenx.tests.test_reduced_articulation import _total_momentum


@wp.kernel(enable_backward=False)
def _solve_contact(cc: ContactContainer, coupled: wp.bool, impulse: wp.array[wp.vec3]):
    n = wp.vec3(1.0, 0.0, 0.0)
    t1 = wp.vec3(0.0, 1.0, 0.0)
    t2 = wp.vec3(0.0, 0.0, 1.0)
    if coupled:
        impulse[0] = contact_project_coupled_velocity_update_no_soft_pd(
            cc,
            0,
            n,
            t1,
            t2,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            10.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.9,
            0.9,
            0.9,
        )
    else:
        impulse[0] = contact_project_velocity_update_no_soft_pd(
            cc,
            0,
            n,
            t1,
            t2,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            10.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        )


@wp.kernel(enable_backward=False)
def _solve_rank_one_contact(cc: ContactContainer, impulse: wp.array[wp.vec3]):
    impulse[0] = contact_project_coupled_velocity_update_no_soft_pd(
        cc,
        0,
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@wp.kernel(enable_backward=False)
def _solve_sliding_contact(cc: ContactContainer, impulse: wp.array[wp.vec3]):
    impulse[0] = contact_project_coupled_velocity_update_no_soft_pd(
        cc,
        0,
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        0.0,
        -1.0,
        -0.86,
        1.0,
        0.01,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


@wp.kernel(enable_backward=False)
def _solve_metric_batch(
    mobility: wp.array[wp.vec3],
    velocity: wp.array[wp.vec2],
    old_impulse: wp.array[wp.vec2],
    result: wp.array[wp.vec2],
):
    i = wp.tid()
    k = mobility[i]
    v = velocity[i]
    old = old_impulse[i]
    result[i] = contact_project_friction_metric(k[0], k[1], k[2], v[0], v[1], old[0], old[1], 1.0, 1.0)


class TestContactCoupling(unittest.TestCase):
    def test_cross_articulation_friction_does_not_add_energy(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        inertia = wp.mat33(0.5005, 0.4995, 0.0, 0.4995, 0.5005, 0.0, 0.0, 0.0, 1.0)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=10.0, gap=0.001)
        for center in (0.5, 1.5000001):
            body = builder.add_link(mass=1.0, inertia=inertia)
            builder.add_shape_sphere(
                body, radius=0.5, xform=wp.transform(wp.vec3(center, 1.0, -2.0), wp.quat_identity()), cfg=cfg
            )
            builder.add_articulation([builder.add_joint_free(body)])
        model = builder.finalize(device="cuda:0")
        state = model.state()
        speed = state.joint_qd.numpy()
        speed[:3] = 0.5
        speed[6:9] = -0.5
        state.joint_qd.assign(speed)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8, contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="reduced",
            step_layout="single_world",
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            sor_boost=1.0,
        )

        def energy():
            q = state.body_q.numpy()
            qd = state.body_qd.numpy()
            total = 0.0
            for index in range(2):
                omega = np.asarray(wp.quat_rotate_inv(wp.quat(*q[index, 3:]), wp.vec3(*qd[index, 3:])))
                total += 0.5 * (qd[index, :3] @ qd[index, :3] + omega @ np.asarray(inertia).reshape(3, 3) @ omega)
            return total

        if getattr(self, "_use_forest", False):
            solver._reduced_articulation.forest_contact_system = ReducedForestContactSystem(
                solver._reduced_articulation, 8
            )
        before = energy()
        momentum_before = _total_momentum(model, state)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state, state, model.control(), contacts, 1.0e-4)
        self.assertGreater(int(solver._reduced_articulation.contact_block_system.fallback_count.numpy()[0]), 0)
        self.assertLessEqual(energy(), before + 1.0e-7)
        np.testing.assert_allclose(_total_momentum(model, state), momentum_before, rtol=0.0, atol=2.0e-6)
        self._last_velocity = state.body_qd.numpy()

    def test_metric_friction_handles_rotation_scale_and_small_mobility(self):
        cases = []
        for angle in (0.0, 0.7):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for scale in (1.0e-5, 1.0, 1.0e5):
                for condition in (1.0, 100.0, 250.0, 998.0, 10000.0):
                    cases.append(
                        (
                            scale * rotation @ np.diag([condition, 1.0]) @ rotation.T,
                            scale * rotation @ np.array([-1.0, -0.86]),
                            rotation @ np.array([0.99, 0.14]),
                        )
                    )
        cases.append((np.diag([1.0e-8, 1.0]), np.array([-1.0e-9, 0.0]), np.array([0.7, 0.7])))
        matrices = np.asarray([item[0] for item in cases], dtype=np.float32)
        velocities = np.asarray([item[1] for item in cases], dtype=np.float32)
        old = np.asarray([item[2] for item in cases], dtype=np.float32)
        packed = np.stack([matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1]], axis=1)
        result = wp.zeros(len(cases), dtype=wp.vec2, device="cpu")
        wp.launch(
            _solve_metric_batch,
            dim=len(cases),
            inputs=[
                wp.array(packed, dtype=wp.vec3, device="cpu"),
                wp.array(velocities, dtype=wp.vec2, device="cpu"),
                wp.array(old, dtype=wp.vec2, device="cpu"),
            ],
            outputs=[result],
            device="cpu",
        )
        for raw_matrix, raw_velocity, previous, friction in zip(matrices, velocities, old, result.numpy(), strict=True):
            matrix = raw_matrix.astype(np.float64)
            velocity = raw_velocity.astype(np.float64)
            after_velocity = velocity + matrix @ (friction.astype(np.float64) - previous)
            self.assertAlmostEqual(float(np.linalg.norm(friction)), 1.0, places=5)
            slip_length = np.linalg.norm(after_velocity)
            self.assertLessEqual(float(friction @ after_velocity), 1.0e-6 * slip_length)
            cross = friction[0] * after_velocity[1] - friction[1] * after_velocity[0]
            roundoff = (
                8.0
                * np.finfo(np.float32).eps
                * (np.linalg.norm(velocity) + np.linalg.norm(matrix) * np.linalg.norm(friction - previous))
            )
            self.assertLess(
                abs(cross), 2.0e-5 * slip_length + roundoff, msg=str((matrix, velocity, previous, friction))
            )
            before = 0.5 * velocity @ np.linalg.solve(matrix, velocity)
            after = 0.5 * after_velocity @ np.linalg.solve(matrix, after_velocity)
            self.assertLessEqual(after, before + 1.0e-6 * max(before, 1.0e-12))

    def test_anisotropic_sliding_friction_dissipates_energy(self):
        mobility = np.diag([100.0, 1.0])
        velocity = np.array([-1.0, -0.86])
        cc = ContactContainer()
        cc.impulses = wp.array(np.array([[1.0], [0.99], [0.14]], dtype=np.float32), device="cpu")
        impulse = wp.zeros(1, dtype=wp.vec3, device="cpu")
        wp.launch(_solve_sliding_contact, dim=1, inputs=[cc], outputs=[impulse], device="cpu")
        after_velocity = velocity + mobility @ impulse.numpy()[0, 1:]
        before = 0.5 * velocity @ np.linalg.solve(mobility, velocity)
        after = 0.5 * after_velocity @ np.linalg.solve(mobility, after_velocity)
        self.assertLessEqual(after, before + 1.0e-6)
        friction = cc.impulses.numpy()[1:, 0]
        self.assertAlmostEqual(float(np.linalg.norm(friction)), 1.0, places=5)
        self.assertLess(float(friction @ after_velocity), 0.0)
        self.assertAlmostEqual(float(friction[0] * after_velocity[1] - friction[1] * after_velocity[0]), 0.0, places=5)

    def test_post_integration_contact_response_conserves_momentum(self):
        self._check_self_contact_momentum(steps=1, radius=0.2, at_rest=False)

    def test_sustained_self_contact_and_motor_conserve_momentum(self):
        center_distance = np.sqrt(0.2**2 + 0.15**2 + 2.0 * 0.2 * 0.15 * np.cos(0.3))
        self._check_self_contact_momentum(steps=200, radius=float(0.5 * center_distance + 5.0e-6), at_rest=True)

    def _check_self_contact_momentum(self, *, steps, radius, at_rest):
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        root = builder.add_link()
        middle = builder.add_link(mass=1.0, inertia=wp.mat33(0.003, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.003))
        tip = builder.add_link()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0)
        builder.add_shape_sphere(root, radius=radius, cfg=cfg)
        builder.add_shape_sphere(tip, radius=radius, cfg=cfg)
        joints = [builder.add_joint_free(root)]
        joints.append(
            builder.add_joint_revolute(
                root, middle, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())
            )
        )
        joints.append(
            builder.add_joint_revolute(
                middle, tip, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity())
            )
        )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        q = state.joint_q.numpy()
        q[-2:] = [0.3, -0.6]
        state.joint_q.assign(q)
        qd = state.joint_qd.numpy()
        if not at_rest:
            qd[:3] = [0.4, -0.15, 0.0]
            qd[5:] = [0.2, 0.3, -0.25]
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        joint_force = control.joint_f.numpy()
        joint_force[-2 if at_rest else -1] = 0.1
        control.joint_f.assign(joint_force)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            articulation_mode="reduced",
            step_layout="single_world",
            substeps=1,
            solver_iterations=8,
            sor_boost=1.0,
        )
        if getattr(self, "_use_forest", False):
            solver._reduced_articulation.forest_contact_system = ReducedForestContactSystem(
                solver._reduced_articulation, 32
            )
        contacts = pipeline.contacts()
        before = _total_momentum(model, state)
        with wp.ScopedCapture(device=model.device) as capture:
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, 1.0 / 1000.0)
        peak_friction = 0.0
        for step in range(steps):
            wp.capture_launch(capture.graph)
            if step % 20 == 0:
                after = _total_momentum(model, state)
                np.testing.assert_allclose(after, before, rtol=0.0, atol=2.0e-4)
                impulses = solver.world._contact_container.impulses.numpy()
                peak_friction = max(peak_friction, float(np.max(np.abs(impulses[1:3]))))
        if steps > 1:
            self.assertGreater(peak_friction, 1.0e-6)
        np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0.0, atol=2.0e-4)

    def test_rank_one_tangent_mobility_retains_friction(self):
        cc = ContactContainer()
        cc.impulses = wp.zeros((3, 1), dtype=wp.float32, device="cpu")
        impulse = wp.zeros(1, dtype=wp.vec3, device="cpu")
        wp.launch(_solve_rank_one_contact, dim=1, inputs=[cc], outputs=[impulse], device="cpu")
        np.testing.assert_allclose(impulse.numpy()[0], [1.0, 1.0, 0.0], atol=1.0e-6)

    def test_friction_contact_does_not_add_kinetic_energy(self):
        # This positive definite mobility has strongly coupled contact axes.
        mobility = np.full((3, 3), 0.9)
        np.fill_diagonal(mobility, 1.0)
        velocity = -np.ones(3)
        before = 0.5 * velocity @ np.linalg.solve(mobility, velocity)
        cc = ContactContainer()
        cc.impulses = wp.zeros((3, 1), dtype=wp.float32, device="cpu")
        impulse = wp.zeros(1, dtype=wp.vec3, device="cpu")
        wp.launch(_solve_contact, dim=1, inputs=[cc, True], outputs=[impulse], device="cpu")
        after_velocity = velocity + mobility @ impulse.numpy()[0]
        after = 0.5 * after_velocity @ np.linalg.solve(mobility, after_velocity)
        self.assertLessEqual(after, before + 1.0e-6)
        np.testing.assert_allclose(after_velocity[1:], np.zeros(2), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
