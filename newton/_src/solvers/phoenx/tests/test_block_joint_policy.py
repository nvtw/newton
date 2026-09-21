# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public block joint policy: bounded motors and property refresh lifecycle."""

import unittest

import numpy as np

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
