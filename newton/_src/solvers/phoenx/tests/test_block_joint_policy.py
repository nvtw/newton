# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public block joint policy: bounded motors and property refresh lifecycle."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests.test_contact_coupling import _total_momentum
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture, _make_revolute


def make_model(kp):
    return _make_revolute(
        two_body=True,
        inertia=0.8,
        armature=0.0,
        gear=1.0,
        passive_damping=0.0,
        kp=kp,
        kd=0.0,
        target_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def make_solver(model, **options):
    return newton.solvers.SolverPhoenX(
        model,
        joint_mode="maximal_pgs",
        step_layout="single_world",
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        sor_boost=1.0,
        **options,
    )


@unittest.skipUnless(_cuda_with_graph_capture(), "Block policy tests require CUDA")
class TestBlockJointPolicy(unittest.TestCase):
    def test_bounded_internal_motor_preserves_momentum(self):
        model = make_model(100.0)
        limit = 0.2
        dt = 0.01
        model.joint_effort_limit.assign(np.asarray([limit], dtype=np.float32))
        solver = make_solver(model, mass_splitting=True, max_colored_partitions=0)
        state = model.state()
        control = model.control()
        control.joint_target_q.assign(np.asarray([1.0], dtype=np.float32))
        momentum = _total_momentum(model, state)
        previous = state.body_qd.numpy()
        for frame in range(8):
            state.clear_forces()
            solver.step(state, state, control, None, dt)
            velocity = state.body_qd.numpy()
            impulse = 0.8 * (velocity[1, 5] - previous[1, 5])
            self.assertAlmostEqual(float(impulse), limit * dt, delta=2.0e-7)
            self.assertAlmostEqual(float(velocity[0, 5]), -float(velocity[1, 5]), delta=2.0e-7)
            np.testing.assert_allclose(
                _total_momentum(model, state),
                momentum,
                rtol=0.0,
                atol=2.0e-6,
                err_msg=f"Internal drive changed momentum at step {frame}",
            )
            previous = velocity

    def test_overflow_only_joint_preserves_momentum_and_replays(self):
        """A D6 hinge forced into overflow must initialize its own copies."""
        model = make_model(0.0)
        initial = model.state()
        qd = initial.body_qd.numpy()
        qd[0, :3] = (0.2, -0.1, 0.3)
        qd[1, :3] = (-0.3, 0.4, -0.2)
        qd[0, 3:] = (0.1, -0.2, 0.4)
        qd[1, 3:] = (-0.3, 0.1, -0.5)
        initial.body_qd.assign(qd)

        def run():
            solver = make_solver(
                model,
                mass_splitting=True,
                mass_splitting_unrolled=True,
                mass_splitting_overflow_only=True,
                max_colored_partitions=0,
                mass_splitting_batch_size=1,
            )
            state = model.state()
            state.body_qd.assign(qd)
            before = _total_momentum(model, state)
            for _ in range(4):
                state.clear_forces()
                solver.step(state, state, model.control(), None, 0.005)
                np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0, atol=3e-6)
            self.assertEqual(solver.step_report().overflow_size, 1)
            return state.body_q.numpy(), state.body_qd.numpy()

        first = run()
        second = run()
        for actual, expected in zip(first, second, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_overflow_only_regular_joint_shares_overflow_body(self):
        """Regular D6 rows must ignore overflow slot zero on a shared body."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        links = [
            builder.add_link(
                xform=wp.transform_identity(), mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            )
            for _ in range(3)
        ]
        joints = [
            builder.add_joint_revolute(parent=links[i], child=links[i + 1], axis=(0.0, 0.0, 1.0)) for i in range(2)
        ]
        builder.add_articulation(joints)
        model = builder.finalize()

        def run():
            solver = make_solver(
                model,
                mass_splitting=True,
                mass_splitting_unrolled=True,
                mass_splitting_overflow_only=True,
                max_colored_partitions=1,
                mass_splitting_batch_size=1,
            )
            state = model.state()
            qd = state.body_qd.numpy()
            qd[:, :3] = ((0.3, -0.1, 0.2), (-0.2, 0.4, -0.3), (0.1, 0.2, -0.4))
            qd[:, 3:] = ((0.1, -0.2, 0.3), (-0.3, 0.1, -0.2), (0.2, 0.3, -0.1))
            state.body_qd.assign(qd)
            before = _total_momentum(model, state)
            for _ in range(4):
                state.clear_forces()
                solver.step(state, state, model.control(), None, 0.005)
                np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0, atol=3e-6)
            self.assertEqual(solver.step_report().overflow_size, 1)
            return state.body_q.numpy(), state.body_qd.numpy()

        first = run()
        second = run()
        for actual, expected in zip(first, second, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_remote_block_joint_does_not_change_overflow_contacts(self):
        """Local D6 preparation must not write one contact copy into every body."""

        def run(with_joint):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for x in (-0.95, 0.0, 0.95):
                body = builder.add_body(
                    xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                    mass=1.0,
                    inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
                )
                builder.add_shape_box(
                    body,
                    hx=0.5,
                    hy=0.5,
                    hz=0.5,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.001),
                )
            if with_joint:
                remote = [
                    builder.add_link(
                        xform=wp.transform(wp.vec3(100.0, 0.0, 0.0), wp.quat_identity()),
                        mass=1.0,
                        inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2),
                    )
                    for _ in range(2)
                ]
                hinge = builder.add_joint_revolute(parent=remote[0], child=remote[1], axis=(0.0, 0.0, 1.0))
                builder.add_articulation([hinge])
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverPhoenX(
                model,
                collision_pipeline=pipeline,
                joint_mode="maximal_pgs",
                step_layout="single_world",
                substeps=1,
                mass_splitting=True,
                mass_splitting_unrolled=True,
                mass_splitting_overflow_only=True,
                max_colored_partitions=0,
                mass_splitting_batch_size=1,
                solver_iterations=2,
                velocity_iterations=1,
                sor_boost=1.0,
            )
            state = model.state()
            qd = state.body_qd.numpy()
            qd[0, :3] = (0.3, 0.0, 0.0)
            qd[2, :3] = (-0.3, 0.0, 0.0)
            state.body_qd.assign(qd)
            for _ in range(4):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, state, model.control(), contacts, 0.005)
            copies = solver.world._copy_state.count_per_node.numpy()
            self.assertGreaterEqual(int(copies[2]), 2)
            return state.body_q.numpy()[:3], state.body_qd.numpy()[:3]

        contact_only = run(False)
        with_remote_joint = run(True)
        for actual, expected in zip(with_remote_joint, contact_only, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_joint_property_refresh_rebinds_dynamic_rows(self):
        model = make_model(0.0)
        model.joint_effort_limit.assign(np.asarray([1000.0], dtype=np.float32))
        solver = make_solver(model)
        control = model.control()
        control.joint_target_q.assign(np.asarray([1.0], dtype=np.float32))
        dt = 0.01
        expected_rows = (5, 6, 6, 5, 6)
        for kp, row_count in zip((0.0, 40.0, 80.0, 0.0, 20.0), expected_rows, strict=True):
            model.joint_target_ke.assign(np.asarray([kp], dtype=np.float32))
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            direct = solver._direct_equality_system
            block = solver.world.constraints.bilateral
            self.assertEqual(int(block.row_count.numpy()[0]), row_count)
            self.assertEqual(block.row_dynamic.ptr, direct.row_dynamic.ptr)
            self.assertEqual(block.accumulated.ptr, direct.accumulated_impulse.ptr)
            state = model.state()
            state.clear_forces()
            solver.step(state, state, control, None, dt)
            velocity = state.body_qd.numpy()
            expected_relative = dt * kp / (0.4 + dt * dt * kp)
            self.assertAlmostEqual(
                float(velocity[1, 5] - velocity[0, 5]),
                expected_relative,
                delta=2.0e-5,
                msg=f"Stale joint rows after changing drive stiffness to {kp}",
            )
            np.testing.assert_allclose(_total_momentum(model, state), 0.0, rtol=0.0, atol=2.0e-6)

    def test_unsupported_policy_combinations_are_rejected(self):
        model = make_model(10.0)
        for options in (
            {"joint_mode": "unknown"},
            {"joint_mode": "maximal_pgs", "step_layout": "multi_world"},
            {"joint_mode": "maximal_pgs", "contact_friction_model": "patch", "step_layout": "single_world"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                newton.solvers.SolverPhoenX(model, **options)


if __name__ == "__main__":
    unittest.main()
