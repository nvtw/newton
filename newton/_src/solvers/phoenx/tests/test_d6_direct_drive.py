# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical direct-drive tests for maximal-coordinate D6 joints."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

SUBSTEPS = 5
DT = 0.02
INERTIA = 0.8
ARMATURE = 0.25
KP = 20.0
KD = 3.0
TARGET = 0.25


def _implicit_scalar_step() -> tuple[float, float]:
    """Integrate the expected scalar implicit PD response over five substeps."""
    q = 0.0
    qd = 0.0
    h = DT / SUBSTEPS
    mass = INERTIA + ARMATURE
    for _ in range(SUBSTEPS):
        qd = (mass * qd + h * KP * (TARGET - q)) / (mass + h * KD + h * h * KP)
        q += h * qd
    return q, qd


def _implicit_cartesian_step() -> tuple[float, float]:
    """Integrate the Cartesian implicit PD response over five substeps."""
    q = 0.0
    qd = 0.0
    h = DT / SUBSTEPS
    mass = 1.0 + ARMATURE
    for _ in range(SUBSTEPS):
        qd = (mass * qd + h * KP * (TARGET - q)) / (mass + h * KD + h * h * KP)
        q += h * qd
    return q, qd


def _make_cartesian(axis_index: int) -> newton.Model:
    """Build a three-axis Cartesian D6 with one active PD drive."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    linear_axes = []
    for index, axis in enumerate(axes):
        active = index == axis_index
        linear_axes.append(
            newton.ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=-1.0e10,
                limit_upper=1.0e10,
                target_ke=KP if active else 0.0,
                target_kd=KD if active else 0.0,
                armature=ARMATURE if active else 0.0,
                effort_limit=0.0,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY if active else newton.JointTargetMode.NONE,
            )
        )
    joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=linear_axes)
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device())


def _make_gimbal(axis_index: int | None, *, left_handed: bool, effort_limit: float = 0.0) -> newton.Model:
    """Build a three-axis D6 with one active PD drive."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0 if left_handed else 1.0))
    angular_axes = []
    for index, axis in enumerate(axes):
        active = axis_index is None or index == axis_index
        angular_axes.append(
            newton.ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
                target_ke=KP if active else 0.0,
                target_kd=KD if active else 0.0,
                armature=ARMATURE if active else 0.0,
                effort_limit=effort_limit if active else 0.0,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY if active else newton.JointTargetMode.NONE,
            )
        )
    joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=angular_axes)
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device())


def _make_reduced_d6(mode: str, axis_index: int) -> newton.Model:
    """Build a universal, cylindrical, or planar D6 with one drive."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((INERTIA, 0.0, 0.0), (0.0, INERTIA, 0.0), (0.0, 0.0, INERTIA)),
    )

    def config(axis: tuple[float, float, float], index: int) -> newton.ModelBuilder.JointDofConfig:
        active = index == axis_index
        return newton.ModelBuilder.JointDofConfig(
            axis=axis,
            limit_lower=-1.0e10,
            limit_upper=1.0e10,
            target_ke=KP if active else 0.0,
            target_kd=KD if active else 0.0,
            armature=ARMATURE if active else 0.0,
            effort_limit=0.0,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY if active else newton.JointTargetMode.NONE,
        )

    if mode == "universal":
        linear_axes = []
        angular_axes = [config((1.0, 0.0, 0.0), 0), config((0.0, 1.0, 0.0), 1)]
    elif mode == "cylindrical":
        linear_axes = [config((0.0, 0.0, 1.0), 0)]
        angular_axes = [config((0.0, 0.0, 1.0), 1)]
    elif mode == "generic":
        linear_axes = [config((1.0, 0.0, 0.0), 0)]
        angular_axes = [config((0.0, 1.0, 0.0), 1), config((0.0, 0.0, 1.0), 2)]
    else:
        linear_axes = [config((1.0, 0.0, 0.0), 0), config((0.0, 1.0, 0.0), 1)]
        angular_axes = [config((0.0, 0.0, 1.0), 2)]
    joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=linear_axes, angular_axes=angular_axes)
    builder.add_articulation([joint])
    return builder.finalize(device=wp.get_preferred_device())


def _implicit_reduced_d6_step(physical_mass: float) -> tuple[float, float]:
    """Integrate one scalar reduced-D6 drive over five substeps."""
    q = 0.0
    qd = 0.0
    h = DT / SUBSTEPS
    mass = physical_mass + ARMATURE
    for _ in range(SUBSTEPS):
        qd = (mass * qd + h * KP * (TARGET - q)) / (mass + h * KD + h * h * KP)
        q += h * qd
    return q, qd


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 drive tests require CUDA graphs.")
class TestD6DirectDrive(unittest.TestCase):
    """Compare each gimbal-axis drive against its scalar analytical solution."""

    def test_gimbal_rows_recover_coordinate_rates(self) -> None:
        """Use the reciprocal or minimum-norm Euler-rate map in drive rows."""
        frame = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -2.0, 0.5)), 0.6)
        rotation = np.asarray(wp.quat_to_matrix(frame), dtype=np.float64).reshape(3, 3)
        cases = ((False, 0.7), (True, 1.4), (False, np.pi / 2.0), (True, -np.pi / 2.0))
        for left_handed, pitch in cases:
            with self.subTest(left_handed=left_handed, pitch=pitch):
                angles = (0.4, pitch, -0.3)
                model = _make_gimbal(None, left_handed=left_handed)
                model.joint_X_p.assign([wp.transform(wp.vec3(), frame)])
                solver = newton.solvers.SolverPhoenX(model, articulation_mode="maximal")
                sign = -1.0 if left_handed else 1.0
                orientation = (
                    frame
                    * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angles[0])
                    * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angles[1])
                    * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, sign), angles[2])
                )
                orientations = solver.world.bodies.orientation.numpy()
                orientations[1] = np.asarray(orientation)
                solver.world.bodies.orientation.assign(orientations)
                system = solver._direct_equality_system
                system.refresh_geometry(wp.float32(60.0))
                dynamic_rows = np.flatnonzero(system.topology.row_dynamic)
                self.assertEqual(len(dynamic_rows), 3)
                coordinates = system.dynamic_coordinate.numpy()[dynamic_rows]
                parent = system.row_wrench0.numpy()[0, 3:6, 3:].astype(np.float64)
                child = system.row_wrench1.numpy()[0, 3:6, 3:].astype(np.float64)
                q0, q1 = (float(coordinates[0]), float(coordinates[1]))
                sx, cx = np.sin(q0), np.cos(q0)
                sy, cy = np.sin(q1), np.cos(q1)
                motion = rotation @ np.asarray(
                    ((1.0, 0.0, sign * sy), (0.0, cx, -sign * sx * cy), (0.0, sx, sign * cx * cy))
                )
                expected = np.linalg.pinv(motion, rcond=1.0e-4)
                np.testing.assert_allclose(child, expected, atol=4.0e-5, rtol=0.0)
                np.testing.assert_allclose(parent + child, 0.0, atol=3.0e-6, rtol=0.0)

    def test_inverse_kinematics_returns_minimum_norm_gimbal_rates(self) -> None:
        """Keep D6 inverse kinematics finite and defined at gimbal lock."""
        for left_handed in (False, True):
            with self.subTest(left_handed=left_handed):
                model = _make_gimbal(None, left_handed=left_handed)
                coordinates = np.asarray((0.4, np.pi / 2.0, -0.3), dtype=np.float32)
                model.joint_q.assign(coordinates)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                sign = -1.0 if left_handed else 1.0
                omega = np.asarray((0.8, -0.4, 0.6), dtype=np.float64)
                body_qd = np.zeros((1, 6), dtype=np.float32)
                body_qd[0, 3:] = omega
                state.body_qd.assign(body_qd)
                joint_q = wp.zeros_like(model.joint_q)
                joint_qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, joint_q, joint_qd)
                recovered = joint_q.numpy()
                sx, cx = np.sin(recovered[0]), np.cos(recovered[0])
                sy, cy = np.sin(recovered[1]), np.cos(recovered[1])
                motion = np.asarray(
                    ((1.0, 0.0, sign * sy), (0.0, cx, -sign * sx * cy), (0.0, sx, sign * cx * cy)),
                    dtype=np.float64,
                )
                expected = np.linalg.pinv(motion, rcond=1.0e-4) @ omega
                rates = joint_qd.numpy()
                self.assertTrue(np.isfinite(rates).all())
                np.testing.assert_allclose(rates, expected, atol=4.0e-5, rtol=0.0)

    def test_gimbal_drive_equilibrium_at_singular_pitch(self) -> None:
        """Keep a driven gimbal finite and stationary at either singular pitch."""
        for left_handed in (False, True):
            for pitch in (-np.pi / 2.0, np.pi / 2.0):
                with self.subTest(left_handed=left_handed, pitch=pitch):
                    model = _make_gimbal(None, left_handed=left_handed)
                    coordinates = np.asarray((0.4, pitch, -0.3), dtype=np.float32)
                    model.joint_q.assign(coordinates)
                    state = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    initial = state.body_q.numpy().copy()
                    control = model.control()
                    control.joint_target_q.assign(coordinates)
                    solver = newton.solvers.SolverPhoenX(
                        model, substeps=5, solver_iterations=2, articulation_mode="maximal"
                    )
                    with wp.ScopedCapture(model.device) as capture:
                        state.clear_forces()
                        solver.step(state, state, control, None, 1.0 / 60.0)
                    for _ in range(20):
                        wp.capture_launch(capture.graph)
                    pose = state.body_q.numpy()
                    velocity = state.body_qd.numpy()
                    self.assertTrue(np.isfinite(pose).all())
                    self.assertTrue(np.isfinite(velocity).all())
                    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-4, rtol=0.0)
                    np.testing.assert_allclose(pose[:, :3], initial[:, :3], atol=1.0e-6, rtol=0.0)
                    self.assertAlmostEqual(abs(float(np.dot(pose[0, 3:], initial[0, 3:]))), 1.0, delta=2.0e-6)

    def test_single_axis_pd_matches_implicit_euler(self) -> None:
        """Match right- and left-handed D6 PD rows over five implicit substeps."""
        expected_q, expected_qd = _implicit_scalar_step()
        for left_handed in (False, True):
            for axis_index in range(3):
                with self.subTest(left_handed=left_handed, axis=axis_index):
                    model = _make_gimbal(axis_index, left_handed=left_handed)
                    state = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    control = model.control()
                    target = np.zeros(3, dtype=np.float32)
                    target[axis_index] = TARGET
                    control.joint_target_q.assign(target)
                    solver = newton.solvers.SolverPhoenX(
                        model,
                        substeps=SUBSTEPS,
                        solver_iterations=1,
                        velocity_iterations=0,
                        articulation_mode="maximal",
                    )
                    with wp.ScopedCapture(model.device) as capture:
                        state.clear_forces()
                        solver.step(state, state, control, None, DT)
                    wp.capture_launch(capture.graph)
                    q = wp.zeros_like(model.joint_q)
                    qd = wp.zeros_like(model.joint_qd)
                    newton.eval_ik(model, state, q, qd)
                    direct = solver._direct_equality_system
                    self.assertEqual(direct.topology.dimensions, (4,))
                    self.assertTrue(direct.direct_drive_joint_mask[0])
                    self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                    self.assertAlmostEqual(float(q.numpy()[axis_index]), expected_q, delta=2.0e-5)
                    self.assertAlmostEqual(float(qd.numpy()[axis_index]), expected_qd, delta=2.0e-5)

    def test_cartesian_pd_matches_implicit_euler(self) -> None:
        """Match each Cartesian D6 drive against implicit Euler."""
        expected_q, expected_qd = _implicit_cartesian_step()
        for axis_index in range(3):
            with self.subTest(axis=axis_index):
                model = _make_cartesian(axis_index)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                target = np.zeros(3, dtype=np.float32)
                target[axis_index] = TARGET
                control.joint_target_q.assign(target)
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=SUBSTEPS,
                    solver_iterations=1,
                    velocity_iterations=0,
                    articulation_mode="maximal",
                )
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    solver.step(state, state, control, None, DT)
                wp.capture_launch(capture.graph)
                q = wp.zeros_like(model.joint_q)
                qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, q, qd)
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (4,))
                self.assertTrue(direct.direct_drive_joint_mask[0])
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                self.assertAlmostEqual(float(q.numpy()[axis_index]), expected_q, delta=2.0e-5)
                self.assertAlmostEqual(float(qd.numpy()[axis_index]), expected_qd, delta=2.0e-5)

    def test_reduced_d6_free_axis_drives_match_implicit_euler(self) -> None:
        """Drive every free axis of reduced-pattern and generic D6 joints directly."""
        cases = (
            ("universal", (INERTIA, INERTIA), 5),
            ("cylindrical", (1.0, INERTIA), 5),
            ("planar", (1.0, 1.0, INERTIA), 4),
            ("generic", (1.0, INERTIA, INERTIA), 4),
        )
        for mode, physical_masses, expected_dimension in cases:
            for axis_index, physical_mass in enumerate(physical_masses):
                with self.subTest(mode=mode, axis=axis_index):
                    model = _make_reduced_d6(mode, axis_index)
                    state = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    control = model.control()
                    target = np.zeros(len(physical_masses), dtype=np.float32)
                    target[axis_index] = TARGET
                    control.joint_target_q.assign(target)
                    solver = newton.solvers.SolverPhoenX(
                        model,
                        substeps=SUBSTEPS,
                        solver_iterations=2,
                        velocity_iterations=1,
                        articulation_mode="maximal",
                    )
                    with wp.ScopedCapture(model.device) as capture:
                        state.clear_forces()
                        solver.step(state, state, control, None, DT)
                    wp.capture_launch(capture.graph)
                    q = wp.zeros_like(model.joint_q)
                    qd = wp.zeros_like(model.joint_qd)
                    newton.eval_ik(model, state, q, qd)
                    expected_q, expected_qd = _implicit_reduced_d6_step(physical_mass)
                    direct = solver._direct_equality_system
                    self.assertEqual(direct.topology.dimensions, (expected_dimension,))
                    self.assertTrue(direct.direct_drive_joint_mask[0])
                    self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                    self.assertAlmostEqual(float(q.numpy()[axis_index]), expected_q, delta=3.0e-5)
                    self.assertAlmostEqual(float(qd.numpy()[axis_index]), expected_qd, delta=3.0e-5)

    def test_finite_effort_gimbal_drive_saturates_analytically(self) -> None:
        """Match a finite-effort D6 drive against constant-torque integration."""
        effort_limit = 0.5
        expected_qd = DT * effort_limit / (INERTIA + ARMATURE)
        expected_q = (DT / SUBSTEPS) * expected_qd * (SUBSTEPS + 1) / 2.0
        for left_handed in (False, True):
            with self.subTest(left_handed=left_handed):
                model = _make_gimbal(2, left_handed=left_handed, effort_limit=effort_limit)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                control.joint_target_q.assign(np.asarray((0.0, 0.0, TARGET), dtype=np.float32))
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=SUBSTEPS,
                    solver_iterations=1,
                    velocity_iterations=0,
                    articulation_mode="maximal",
                )
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    solver.step(state, state, control, None, DT)
                wp.capture_launch(capture.graph)
                q = wp.zeros_like(model.joint_q)
                qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, q, qd)
                self.assertTrue(bool(solver._direct_equality_system.drive_saturated.numpy()[-1]))
                self.assertAlmostEqual(float(q.numpy()[2]), expected_q, delta=2.0e-5)
                self.assertAlmostEqual(float(qd.numpy()[2]), expected_qd, delta=2.0e-5)

    def test_coupled_gimbal_pd_tracks_target(self) -> None:
        """Track all three transported-axis targets without PGS joint rows."""
        target = np.asarray((0.35, -0.25, 0.2), dtype=np.float32)
        for left_handed in (False, True):
            with self.subTest(left_handed=left_handed):
                model = _make_gimbal(None, left_handed=left_handed)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                control.joint_target_q.assign(target)
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=SUBSTEPS,
                    solver_iterations=1,
                    velocity_iterations=0,
                    articulation_mode="maximal",
                )
                with wp.ScopedCapture(model.device) as capture:
                    state.clear_forces()
                    solver.step(state, state, control, None, DT)
                for _ in range(360):
                    wp.capture_launch(capture.graph)
                q = wp.zeros_like(model.joint_q)
                qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, q, qd)
                direct = solver._direct_equality_system
                self.assertEqual(direct.topology.dimensions, (6,))
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
                np.testing.assert_allclose(q.numpy(), target, rtol=0.0, atol=2.0e-3)
                np.testing.assert_allclose(qd.numpy(), 0.0, rtol=0.0, atol=2.0e-3)


if __name__ == "__main__":
    unittest.main()
