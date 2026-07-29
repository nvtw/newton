# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for direct maximal-coordinate joint drives."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _cuda_with_graph_capture() -> bool:
    device = wp.get_preferred_device()
    return bool(device.is_cuda and wp.is_mempool_enabled(device))


def _make_revolute(
    *,
    two_body: bool,
    inertia: float,
    armature: float,
    gear: float,
    passive_damping: float,
    kp: float,
    kd: float,
    target_mode: newton.JointTargetMode,
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    parent = -1
    if two_body:
        parent = builder.add_link(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
        )
    child = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
    )
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        target_ke=kp,
        target_kd=kd,
        actuator_mode=target_mode,
        armature=armature,
        damping=passive_damping,
        gear_ratio=gear,
        effort_limit=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize()


def _make_prismatic(
    *,
    mass: float,
    armature: float,
    passive_damping: float,
    kp: float,
    kd: float,
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    parent = builder.add_link(
        xform=wp.transform_identity(),
        mass=mass,
        inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    child = builder.add_link(
        xform=wp.transform_identity(),
        mass=mass,
        inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    joint = builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        target_ke=kp,
        target_kd=kd,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        armature=armature,
        damping=passive_damping,
        effort_limit=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize()


def _step_scalar(
    model: newton.Model,
    *,
    q: float,
    qd: float,
    target_q: float,
    target_qd: float,
    dt: float,
    mass_splitting: bool = False,
) -> tuple[float, float, newton.solvers.SolverPhoenX]:
    state_0 = model.state()
    state_1 = model.state()
    state_0.joint_q.assign(np.asarray([q], dtype=np.float32))
    state_0.joint_qd.assign(np.asarray([qd], dtype=np.float32))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    control.joint_target_q.assign(np.asarray([target_q], dtype=np.float32))
    control.joint_target_qd.assign(np.asarray([target_qd], dtype=np.float32))
    solver = newton.solvers.SolverPhoenX(
        model,
        substeps=1,
        solver_iterations=1,
        velocity_iterations=0,
        articulation_mode="maximal",
        joint_equality_solver="direct",
        mass_splitting=mass_splitting,
        step_layout="single_world" if mass_splitting else "multi_world",
    )

    def step() -> None:
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)

    with wp.ScopedCapture(device=model.device) as capture:
        step()
    wp.capture_launch(capture.graph)
    joint_q = wp.zeros(1, dtype=wp.float32, device=model.device)
    joint_qd = wp.zeros(1, dtype=wp.float32, device=model.device)
    newton.eval_ik(model, state_1, joint_q, joint_qd)
    return float(joint_q.numpy()[0]), float(joint_qd.numpy()[0]), solver


def _implicit_velocity(
    *,
    physical_inertia: float,
    q: float,
    qd: float,
    target_q: float,
    target_qd: float,
    dt: float,
    armature: float,
    passive_damping: float,
    kp: float,
    kd: float,
) -> float:
    numerator = (physical_inertia + armature) * qd
    numerator += dt * (kp * (target_q - q) + kd * target_qd)
    denominator = physical_inertia + armature + dt * (passive_damping + kd) + dt * dt * kp
    return numerator / denominator


@unittest.skipUnless(_cuda_with_graph_capture(), "Direct-drive analytical tests require CUDA graph capture")
class TestDirectDriveAnalytical(unittest.TestCase):
    def test_mass_splitting_drive_matches_implicit_euler(self) -> None:
        model = _make_revolute(
            two_body=True,
            inertia=0.8,
            armature=0.25,
            gear=1.0,
            passive_damping=0.1,
            kp=20.0,
            kd=3.0,
            target_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        q = -0.15
        qd = 0.3
        target_q = 0.5
        target_qd = -0.2
        dt = 0.02
        q_after, qd_after, _solver = _step_scalar(
            model,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
            mass_splitting=True,
        )
        expected_qd = _implicit_velocity(
            physical_inertia=0.4,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
            armature=0.25,
            passive_damping=0.1,
            kp=20.0,
            kd=3.0,
        )
        self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
        self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)

    def test_multi_world_drives_match_independent_implicit_euler_steps(self) -> None:
        inertia = 0.9
        armature = 0.25
        passive_damping = 0.2
        kp = 30.0
        kd = 3.0
        dt = 0.01
        q = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
        qd = np.asarray([-0.1, 0.2, -0.3], dtype=np.float64)
        target_q = np.asarray([0.4, 0.1, -0.2], dtype=np.float64)
        target_qd = np.asarray([0.05, -0.1, 0.2], dtype=np.float64)

        template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        child = template.add_link(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
        )
        joint = template.add_joint_revolute(
            parent=-1,
            child=child,
            axis=(0.0, 0.0, 1.0),
            target_ke=kp,
            target_kd=kd,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            armature=armature,
            damping=passive_damping,
            effort_limit=100.0,
        )
        template.add_articulation([joint])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        builder.replicate(template, 3)
        model = builder.finalize()
        state_0 = model.state()
        state_1 = model.state()
        state_0.joint_q.assign(q.astype(np.float32))
        state_0.joint_qd.assign(qd.astype(np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        control.joint_target_q.assign(target_q.astype(np.float32))
        control.joint_target_qd.assign(target_qd.astype(np.float32))
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
            joint_equality_solver="direct",
            step_layout="multi_world",
        )
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        result_q = wp.zeros(3, dtype=wp.float32, device=model.device)
        result_qd = wp.zeros(3, dtype=wp.float32, device=model.device)
        newton.eval_ik(model, state_1, result_q, result_qd)

        expected_qd = np.asarray(
            [
                _implicit_velocity(
                    physical_inertia=inertia,
                    q=q[index],
                    qd=qd[index],
                    target_q=target_q[index],
                    target_qd=target_qd[index],
                    dt=dt,
                    armature=armature,
                    passive_damping=passive_damping,
                    kp=kp,
                    kd=kd,
                )
                for index in range(3)
            ]
        )
        self.assertEqual(model.world_count, 3)
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (6, 6, 6))
        np.testing.assert_array_equal(
            solver._direct_equality_system.direct_drive_joint_mask,
            np.ones(3, dtype=bool),
        )
        np.testing.assert_array_equal(
            solver.world._joint_pgs_enabled.numpy()[: solver.world.num_joints],
            np.zeros(3, dtype=np.int32),
        )
        np.testing.assert_allclose(result_qd.numpy(), expected_qd, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(result_q.numpy(), q + dt * expected_qd, rtol=2.0e-5, atol=2.0e-5)

    def test_coupled_three_joint_mechanism_matches_matrix_implicit_euler(self) -> None:
        inertia = np.asarray([0.8, 1.1, 0.6], dtype=np.float64)
        armature = np.asarray([0.2, 0.4, 0.1], dtype=np.float64)
        passive_damping = np.asarray([0.3, 0.1, 0.2], dtype=np.float64)
        kp = np.asarray([25.0, 40.0, 15.0], dtype=np.float64)
        kd = np.asarray([2.0, 3.5, 1.0], dtype=np.float64)
        q = np.asarray([0.15, -0.2, 0.1], dtype=np.float64)
        qd = np.asarray([-0.1, 0.25, -0.3], dtype=np.float64)
        target_q = np.asarray([0.5, 0.1, -0.25], dtype=np.float64)
        target_qd = np.asarray([0.2, -0.15, 0.05], dtype=np.float64)
        dt = 0.0125

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        bodies = [
            builder.add_link(
                xform=wp.transform_identity(),
                mass=1.0,
                inertia=((value, 0.0, 0.0), (0.0, value, 0.0), (0.0, 0.0, value)),
            )
            for value in inertia
        ]
        joints = []
        for index in range(3):
            joints.append(
                builder.add_joint_revolute(
                    parent=-1 if index == 0 else bodies[index - 1],
                    child=bodies[index],
                    axis=(0.0, 0.0, 1.0),
                    target_ke=float(kp[index]),
                    target_kd=float(kd[index]),
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                    armature=float(armature[index]),
                    damping=float(passive_damping[index]),
                    effort_limit=0.0,
                )
            )
        builder.add_articulation(joints)
        model = builder.finalize()
        state_0 = model.state()
        state_1 = model.state()
        state_0.joint_q.assign(q.astype(np.float32))
        state_0.joint_qd.assign(qd.astype(np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        control.joint_target_q.assign(target_q.astype(np.float32))
        control.joint_target_qd.assign(target_qd.astype(np.float32))
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=1,
            solver_iterations=1,
            velocity_iterations=0,
            articulation_mode="maximal",
            joint_equality_solver="direct",
        )
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)

        result_q = wp.zeros(3, dtype=wp.float32, device=model.device)
        result_qd = wp.zeros(3, dtype=wp.float32, device=model.device)
        newton.eval_ik(model, state_1, result_q, result_qd)

        mass = np.empty((3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                mass[row, column] = np.sum(inertia[max(row, column) :])
        impedance = armature + dt * (passive_damping + kd) + dt * dt * kp
        rhs = mass @ qd + armature * qd
        rhs += dt * (kp * (target_q - q) + kd * target_qd)
        expected_qd = np.linalg.solve(mass + np.diag(impedance), rhs)

        np.testing.assert_allclose(result_qd.numpy(), expected_qd, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(result_q.numpy(), q + dt * expected_qd, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_array_equal(solver._direct_equality_system.direct_drive_joint_mask, np.ones(3, dtype=bool))

    def test_anchored_revolute_target_modes_match_implicit_euler(self) -> None:
        inertia = 0.7
        armature = 0.3
        gear = 2.0
        passive_damping = 0.4
        q = 0.2
        qd = -0.35
        target_q = 0.8
        target_qd = 0.45
        dt = 0.01
        for target_mode, expected_kp, expected_kd, expected_target_qd in (
            (newton.JointTargetMode.POSITION, 30.0, 4.0, 0.0),
            (newton.JointTargetMode.VELOCITY, 0.0, 4.0, target_qd),
            (newton.JointTargetMode.POSITION_VELOCITY, 30.0, 4.0, target_qd),
        ):
            with self.subTest(target_mode=target_mode):
                model = _make_revolute(
                    two_body=False,
                    inertia=inertia,
                    armature=armature,
                    gear=gear,
                    passive_damping=passive_damping,
                    kp=30.0,
                    kd=4.0,
                    target_mode=target_mode,
                )
                q_after, qd_after, solver = _step_scalar(
                    model,
                    q=q,
                    qd=qd,
                    target_q=target_q,
                    target_qd=target_qd,
                    dt=dt,
                )
                expected_qd = _implicit_velocity(
                    physical_inertia=inertia,
                    q=q,
                    qd=qd,
                    target_q=target_q,
                    target_qd=expected_target_qd,
                    dt=dt,
                    armature=gear * gear * armature,
                    passive_damping=passive_damping,
                    kp=expected_kp,
                    kd=expected_kd,
                )
                self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
                self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
                self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)

    def test_free_two_body_revolute_matches_reduced_inertia(self) -> None:
        inertia = 0.8
        armature = 0.25
        kp = 20.0
        kd = 3.0
        q = -0.15
        qd = 0.3
        target_q = 0.5
        target_qd = -0.2
        dt = 0.02
        model = _make_revolute(
            two_body=True,
            inertia=inertia,
            armature=armature,
            gear=1.0,
            passive_damping=0.1,
            kp=kp,
            kd=kd,
            target_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        q_after, qd_after, _solver = _step_scalar(
            model,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
        )
        expected_qd = _implicit_velocity(
            physical_inertia=0.5 * inertia,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
            armature=armature,
            passive_damping=0.1,
            kp=kp,
            kd=kd,
        )
        self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
        self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)

    def test_free_two_body_prismatic_matches_reduced_mass(self) -> None:
        mass = 1.4
        armature = 0.6
        passive_damping = 0.2
        kp = 35.0
        kd = 5.0
        q = 0.1
        qd = -0.25
        target_q = 0.4
        target_qd = 0.3
        dt = 0.015
        model = _make_prismatic(
            mass=mass,
            armature=armature,
            passive_damping=passive_damping,
            kp=kp,
            kd=kd,
        )
        q_after, qd_after, _solver = _step_scalar(
            model,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
        )
        expected_qd = _implicit_velocity(
            physical_inertia=0.5 * mass,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
            armature=armature,
            passive_damping=passive_damping,
            kp=kp,
            kd=kd,
        )
        self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
        self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)

    def test_unsaturated_finite_effort_drive_matches_implicit_euler(self) -> None:
        model = _make_revolute(
            two_body=False,
            inertia=1.0,
            armature=0.2,
            gear=1.0,
            passive_damping=0.3,
            kp=10.0,
            kd=1.0,
            target_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        model.joint_effort_limit.assign(np.asarray([100.0], dtype=np.float32))
        q = 0.1
        qd = -0.2
        target_q = 0.15
        target_qd = 0.05
        dt = 0.01
        q_after, qd_after, solver = _step_scalar(
            model,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
        )
        expected_qd = _implicit_velocity(
            physical_inertia=1.0,
            q=q,
            qd=qd,
            target_q=target_q,
            target_qd=target_qd,
            dt=dt,
            armature=0.2,
            passive_damping=0.3,
            kp=10.0,
            kd=1.0,
        )
        self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
        self.assertTrue(solver._direct_equality_system.bounded_drive_joint_mask[0])
        self.assertFalse(bool(solver.world._joint_pgs_enabled.numpy()[0]))
        dynamic_rows = solver._direct_equality_system.topology.row_dynamic
        self.assertFalse(bool(np.any(solver._direct_equality_system.drive_saturated.numpy()[dynamic_rows])))
        self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
        self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)

    def test_saturated_finite_effort_drive_matches_constant_torque_step(self) -> None:
        inertia = 1.0
        armature = 0.2
        gear = 1.5
        passive_damping = 0.3
        effort_limit = 2.0
        q = 0.0
        qd = 0.1
        dt = 0.01
        model = _make_revolute(
            two_body=False,
            inertia=inertia,
            armature=armature,
            gear=gear,
            passive_damping=passive_damping,
            kp=1000.0,
            kd=10.0,
            target_mode=newton.JointTargetMode.POSITION,
        )
        model.joint_effort_limit.assign(np.asarray([effort_limit], dtype=np.float32))
        q_after, qd_after, solver = _step_scalar(
            model,
            q=q,
            qd=qd,
            target_q=1.0,
            target_qd=0.0,
            dt=dt,
        )
        reflected_armature = gear * gear * armature
        reflected_effort_limit = gear * effort_limit
        expected_qd = ((inertia + reflected_armature) * qd + dt * reflected_effort_limit) / (
            inertia + reflected_armature + dt * passive_damping
        )
        self.assertTrue(solver._direct_equality_system.direct_drive_joint_mask[0])
        self.assertTrue(solver._direct_equality_system.bounded_drive_joint_mask[0])
        self.assertFalse(bool(solver.world._joint_pgs_enabled.numpy()[0]))
        dynamic_rows = solver._direct_equality_system.topology.row_dynamic
        self.assertTrue(bool(np.all(solver._direct_equality_system.drive_saturated.numpy()[dynamic_rows])))
        self.assertAlmostEqual(qd_after, expected_qd, delta=2.0e-5)
        self.assertAlmostEqual(q_after, q + dt * expected_qd, delta=2.0e-5)


if __name__ == "__main__":
    unittest.main()
