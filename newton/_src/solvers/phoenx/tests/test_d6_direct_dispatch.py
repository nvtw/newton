# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct-solver dispatch for supported maximal-coordinate D6 joints."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_GENERIC_D6,
    joint_constraint_clear_reset_worlds,
)


def _make_body(builder: newton.ModelBuilder) -> int:
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return body


def _assert_direct_limits(test: unittest.TestCase, solver, dofs, lower, upper) -> None:
    """Check that the exact mechanism, not an inequality row, owns these joint bounds."""
    direct = solver._direct_equality_system
    np.testing.assert_array_equal(np.flatnonzero(direct.direct_limit_dof_mask), dofs)
    rows = np.flatnonzero(direct.row_limited.numpy())
    np.testing.assert_array_equal(np.asarray(direct.topology.row_dof)[rows], dofs)
    np.testing.assert_allclose(direct.row_limit_lower.numpy()[rows], lower, atol=1.0e-6)
    np.testing.assert_allclose(direct.row_limit_upper.numpy()[rows], upper, atol=1.0e-6)
    test.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 0)


def _mode_for(model: newton.Model) -> int:
    solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")
    return int(solver._joint_constraints.joint_mode.numpy()[0])


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX direct D6 dispatch tests run on CUDA only")
class TestD6DirectDispatch(unittest.TestCase):
    def test_free_rotation_d6_uses_compact_rows(self) -> None:
        """Bound row storage while the compact rows still lock translation."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        joints = []
        for _ in range(3):
            joints.append(
                builder.add_joint_d6(
                    parent=-1,
                    child=_make_body(builder),
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig.create_unlimited(axis)
                        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
                    ],
                )
            )
        builder.add_articulation(joints)
        model = builder.finalize(device=wp.get_preferred_device())
        solver = newton.solvers.SolverPhoenX(model, joint_mode="maximal_direct", velocity_iterations=1)
        names = ("row_wrench0", "row_wrench1", "row_bias", "row_error", "row_stiffness", "row_damping")
        direct = solver._direct_equality_system
        # Two spatial wrenches and four scalar fields cost 64 bytes per row.
        # Three locked translations must not reserve three unused rotations.
        row_bytes = sum(getattr(direct, name).numpy().nbytes for name in names)
        self.assertLessEqual(row_bytes, 64 * len(direct.topology.row_joint))
        state = model.state()
        start = state.body_q.numpy()[:, :3].copy()
        velocities = state.body_qd.numpy()
        velocities[:, :3] = (0.2, 0.4, 0.6)
        velocities[:, 3:] = (0.3, -0.2, 0.1)
        state.body_qd.assign(velocities)
        for _ in range(4):
            state.clear_forces()
            solver.step(state, state, model.control(), None, 1.0 / 120.0)
        # The anchors sit at the body origins, so the locked translations
        # hold the origins in place while the free rotations keep spinning.
        np.testing.assert_allclose(state.body_q.numpy()[:, :3], start, atol=1.0e-6)
        np.testing.assert_allclose(state.body_qd.numpy()[:, :3], 0.0, atol=1.0e-5)
        self.assertGreater(float(np.linalg.norm(state.body_qd.numpy()[:, 3:], axis=1).min()), 0.3)

    def test_direct_drive_reports_constraint_wrench(self) -> None:
        """Report direct D6 reactions through the common diagnostics path."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = _make_body(builder)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            target_vel=2.0,
            target_kd=4.0,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            friction=0.5,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_direct",
            substeps=2,
            solver_iterations=2,
            velocity_iterations=1,
        )
        state = model.state()
        state.clear_forces()
        solver.step(state, state, model.control(), None, 1.0 / 120.0)

        direct = solver._direct_equality_system
        impulses = direct.accumulated_impulse.numpy()
        wrenches = direct.row_wrench1.numpy()
        structural = int(direct.joint_to_structural.numpy()[joint])
        expected = np.zeros(6, dtype=np.float32)
        for row, (row_joint, local_row) in enumerate(
            zip(direct.topology.row_joint, direct.topology.row_local, strict=True)
        ):
            if row_joint == joint:
                expected += impulses[row] * wrenches[structural, local_row] / solver.world.substep_dt
        friction_impulses = direct.friction_impulse.numpy()
        self.assertGreater(float(np.linalg.norm(friction_impulses)), 1.0e-6)
        for row, friction_joint in enumerate(direct.friction_joints.numpy()):
            if friction_joint == joint:
                expected += friction_impulses[row] / solver.world.substep_dt

        reported = wp.zeros(solver.world.num_constraints, dtype=wp.spatial_vector, device=model.device)
        solver.world.gather_constraint_wrenches(reported)
        self.assertGreater(float(np.linalg.norm(expected)), 1.0e-3)
        np.testing.assert_allclose(reported.numpy()[0], expected, rtol=2.0e-6, atol=2.0e-5)

    def test_projected_d6_limits_use_common_rows(self) -> None:
        """Keep D6 inequalities available without the direct equality system."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = _make_body(builder)
        locked = [
            newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=1.0, limit_upper=-1.0)
            for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
        ]
        angular = [
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, limit_lower=-0.2, limit_upper=0.2),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Z,
                limit_lower=1.0,
                limit_upper=-1.0,
            ),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=locked, angular_axes=angular)
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        coordinates = model.joint_q.numpy()
        coordinates[3] = 0.5
        model.joint_q.assign(coordinates)

        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_projected",
            substeps=4,
            solver_iterations=2,
            velocity_iterations=1,
        )

        self.assertIsNone(solver._direct_equality_system)
        data = solver.world.constraints.d6
        self.assertEqual(int(data.row_count.numpy()[0]), 1)
        self.assertEqual(int(data.row_axis.numpy()[0, 0]), 3)
        self.assertAlmostEqual(float(data.lower.numpy()[0, 0]), -0.2, delta=1.0e-6)
        self.assertAlmostEqual(float(data.upper.numpy()[0, 0]), 0.2, delta=1.0e-6)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        state.clear_forces()
        solver.step(state, state, model.control(), None, 1.0 / 120.0)
        self.assertLessEqual(abs(float(state.joint_q.numpy()[3])), 0.2005)

        impulses = data.lower_impulse.numpy()[0] + data.upper_impulse.numpy()[0] + data.friction_impulse.numpy()[0]
        expected_wrench = np.sum(data.wrench1.numpy()[0] * impulses[:, None], axis=0) / solver.world.substep_dt
        reported_wrench = wp.zeros(solver.world.num_constraints, dtype=wp.spatial_vector, device=model.device)
        solver.world.gather_constraint_wrenches(reported_wrench)
        np.testing.assert_allclose(reported_wrench.numpy()[0], expected_wrench, rtol=2.0e-6, atol=2.0e-5)

        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        lower[3] = -1.0e6
        upper[3] = 1.0e6
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 0)

    def test_projected_prismatic_d6_retains_free_axis(self) -> None:
        """Keep one-axis D6 motion and limits intact in the maximal projector."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axis = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.X,
            limit_lower=-0.1,
            limit_upper=0.1,
        )
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=[axis])
        builder.add_articulation([joint])
        model = builder.finalize()
        model.joint_q.assign(np.asarray((0.25,), dtype=np.float32))
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_projected",
            substeps=4,
            solver_iterations=2,
            velocity_iterations=1,
        )

        state.clear_forces()
        solver.step(state, state, model.control(), None, 1.0 / 120.0)
        self.assertLessEqual(abs(float(state.joint_q.numpy()[0])), 0.1005)

    def test_ball_limits_use_common_d6_rows(self) -> None:
        """Keep native BALL limits on the common D6 inequality path."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        joint = builder.add_joint_ball(parent=-1, child=body)
        builder.add_articulation([joint])
        model = builder.finalize()

        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        lower[:3] = [-0.4, -0.5, -0.6]
        upper[:3] = [0.4, 0.5, 0.6]
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)

        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")
        data = solver.world.constraints.d6
        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        self.assertEqual(int(data.row_count.numpy()[0]), 3)
        np.testing.assert_array_equal(data.row_axis.numpy()[0, :3], [0, 1, 2])
        np.testing.assert_allclose(data.lower.numpy()[0, :3], lower[:3], atol=1.0e-6)
        np.testing.assert_allclose(data.upper.numpy()[0, :3], upper[:3], atol=1.0e-6)
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 1)

    def test_angular_three_axis_d6_uses_common_rows_with_limits(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0, limit_upper=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-1.0, limit_upper=1.0),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")

        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        _assert_direct_limits(self, solver, [0, 1, 2], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    def test_angular_two_axis_d6_uses_common_rows_with_limits(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.8, limit_upper=0.8),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.3, limit_upper=0.5),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])

        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")

        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        _assert_direct_limits(self, solver, [0, 1], [-0.8, -1.3], [0.8, 0.5])

    def test_angular_two_axis_d6_limit_row_reacts_to_violation(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-0.25, limit_upper=0.25),
            newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, 0.0))
        solver = newton.solvers.SolverPhoenX(model, substeps=5, solver_iterations=8, joint_mode="maximal_direct")

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.65)
        body_q[0, 3:7] = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        state_0.body_q.assign(body_q)
        state_0.body_qd.assign(np.zeros((1, 6), dtype=np.float32))

        # An exact mechanism recovers a violated bound at its drift rate.
        for _ in range(20):
            state_0.clear_forces()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()[0]
        body_qd = state_0.body_qd.numpy()[0]
        angle_x = 2.0 * np.arctan2(float(body_q[3]), float(body_q[6]))
        self.assertAlmostEqual(angle_x, 0.25, delta=5.0e-4)
        self.assertAlmostEqual(float(body_qd[3]), 0.0, delta=1.0e-4)

    def test_one_axis_d6_limits_use_prepared_common_rows(self) -> None:
        """Prepare reduced revolute and prismatic D6 limits through common rows."""
        for angular in (False, True):
            with self.subTest(angular=angular):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = _make_body(builder)
                axis = newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.X,
                    limit_lower=-0.1,
                    limit_upper=0.1,
                )
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    linear_axes=[] if angular else [axis],
                    angular_axes=[axis] if angular else [],
                )
                builder.add_articulation([joint])
                model = builder.finalize()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=5,
                    solver_iterations=8,
                    velocity_iterations=1,
                    joint_mode="maximal_direct",
                )
                _assert_direct_limits(self, solver, [0], [-0.1], [0.1])
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                poses = state.body_q.numpy()
                if angular:
                    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.25)
                    poses[0, 3:] = (rotation[0], rotation[1], rotation[2], rotation[3])
                else:
                    poses[0, 0] = 0.25
                state.body_q.assign(poses)
                state.body_qd.zero_()
                # An exact mechanism recovers a violated bound at its drift rate.
                for _ in range(10):
                    state.clear_forces()
                    solver.step(state, state, model.control(), None, 1.0 / 60.0)
                joint_q = wp.zeros_like(model.joint_q)
                joint_qd = wp.zeros_like(model.joint_qd)
                newton.eval_ik(model, state, joint_q, joint_qd)
                self.assertLessEqual(abs(float(joint_q.numpy()[0])), 0.1005)
                self.assertLessEqual(float(joint_q.numpy()[0] * joint_qd.numpy()[0]), 2.0e-4)

    def test_angular_one_axis_d6_uses_common_rows(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=(0.0, 1.0, 0.0),
                limit_lower=-0.25,
                limit_upper=0.5,
                target_pos=0.1,
                target_ke=10.0,
                target_kd=1.0,
            )
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")

        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        self.assertEqual(int(solver._direct_equality_system.generic_linear_count.numpy()[0]), 3)
        self.assertEqual(int(solver._direct_equality_system.generic_angular_count.numpy()[0]), 2)
        _assert_direct_limits(self, solver, [0], [-0.25], [0.5])

    def test_linear_one_axis_d6_uses_common_rows(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.2, limit_upper=0.3)]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])

        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")
        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        self.assertEqual(int(solver._direct_equality_system.generic_linear_count.numpy()[0]), 2)
        self.assertEqual(int(solver._direct_equality_system.generic_angular_count.numpy()[0]), 3)
        _assert_direct_limits(self, solver, [0], [-0.2], [0.3])

    def test_two_axis_cartesian_d6_uses_common_rows(self) -> None:
        """Preserve only the two authored Cartesian translation directions."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited((1.0, 0.0, 0.0)),
            newton.ModelBuilder.JointDofConfig.create_unlimited((0.0, 1.0, 0.0)),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=2,
            joint_mode="maximal_direct",
        )
        direct = solver._direct_equality_system

        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        self.assertEqual(direct.topology.dimensions, (4,))
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        initial_qd = np.array([[1.25, -0.75, 2.0, 3.0, -2.0, 1.0]], dtype=np.float32)
        state_0.body_qd.assign(initial_qd)
        with wp.ScopedCapture(device=model.device) as capture:
            state_0.clear_forces()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)

        final_qd = state_1.body_qd.numpy()[0]
        np.testing.assert_allclose(final_qd[:2], initial_qd[0, :2], rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(final_qd[2:], 0.0, rtol=0.0, atol=2.0e-4)

    def test_three_axis_cartesian_d6_uses_common_rows(self) -> None:
        """Lock rotation while leaving all Cartesian translations free."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited((1.0, 0.0, 0.0)),
            newton.ModelBuilder.JointDofConfig.create_unlimited((0.0, 1.0, 0.0)),
            newton.ModelBuilder.JointDofConfig.create_unlimited((0.0, 0.0, 1.0)),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, solver_iterations=2, joint_mode="maximal_direct")

        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        self.assertEqual(solver._direct_equality_system.topology.dimensions, (3,))
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)

    def test_mixed_d6_uses_complement_direct_rows(self) -> None:
        """Constrain only the complement of mixed translational and rotational freedoms."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = _make_body(builder)
        linear_axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X)]
        angular_axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        builder.add_joint_d6(
            parent=-1,
            child=body,
            linear_axes=linear_axes,
            angular_axes=angular_axes,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=2,
            joint_mode="maximal_direct",
        )

        self.assertEqual(solver._direct_equality_system.topology.dimensions, (3,))
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)

        state = model.state()
        initial_qd = np.asarray(((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),), dtype=np.float32)
        state.body_qd.assign(initial_qd)
        with wp.ScopedCapture(model.device) as capture:
            state.clear_forces()
            solver.step(state, state, model.control(), None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)

        final_qd = state.body_qd.numpy()[0]
        # The first free axis stays on the parent while the second rotates with the child.
        orientation = wp.quat(*state.body_q.numpy()[0, 3:])
        child_z = np.asarray(wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0)))
        locked = np.cross((0.0, 1.0, 0.0), child_z)
        omega = final_qd[3:]
        free_speed = (final_qd[0], omega[1], np.dot(omega, child_z))
        locked_speed = (final_qd[1], final_qd[2], np.dot(omega, locked))
        np.testing.assert_allclose(free_speed, initial_qd[0, [0, 4, 5]], rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(locked_speed, 0.0, rtol=0.0, atol=2.0e-4)

    def test_sliding_d6_rows_match_rotated_frame_derivative(self) -> None:
        """Match the sliding Jacobian and conserve each floating-pair row's momentum."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        orientation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
        origin = np.asarray((0.3, -0.2, 0.5), dtype=np.float32)
        separation = np.asarray(wp.quat_rotate(orientation, wp.vec3(0.4, 0.1, -0.2)))
        positions = (origin, origin + separation)
        links = []
        for index, position in enumerate(positions):
            body = builder.add_link(
                xform=wp.transform(wp.vec3(*position), orientation),
                mass=float(index + 1),
                inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.02),
            )
            builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=builder.ShapeConfig(density=0.0))
            links.append(body)
        builder.add_joint_d6(
            parent=links[0],
            child=links[1],
            linear_axes=[
                builder.JointDofConfig(
                    axis=newton.Axis.X, limit_lower=-1.0e10, limit_upper=1.0e10, target_ke=20.0, target_kd=2.0
                )
            ],
            angular_axes=[
                builder.JointDofConfig.create_unlimited(newton.Axis.Y),
                builder.JointDofConfig.create_unlimited(newton.Axis.Z),
            ],
        )
        builder.add_articulation(list(range(builder.joint_count)))
        solver = newton.solvers.SolverPhoenX(builder.finalize(), joint_mode="maximal_direct")
        system = solver._direct_equality_system
        # Articulation initialization evaluates its zero coordinates; set the
        # intended separated geometry explicitly before preparing the rows.
        body_positions = solver.world.bodies.position.numpy()
        body_orientations = solver.world.bodies.orientation.numpy()
        for body, position in zip(links, positions, strict=True):
            body_positions[body + 1] = position
            body_orientations[body + 1] = np.asarray(orientation)
        solver.world.bodies.position.assign(body_positions)
        solver.world.bodies.orientation.assign(body_orientations)
        system.refresh_geometry(wp.float32(60.0))
        wrench0 = system.row_wrench0.numpy()[0, :4].astype(np.float64)
        wrench1 = system.row_wrench1.numpy()[0, :4].astype(np.float64)
        self.assertEqual(int(np.count_nonzero(system.topology.row_dynamic)), 1)
        self.assertAlmostEqual(float(np.linalg.norm(wrench1[3, :3])), 1.0, delta=2.0e-6)
        # Independently differentiate n(parent orientation) dot (p1 - p0).
        epsilon = 1.0e-4
        for row in (0, 1, 3):
            direction = wrench1[row, :3]
            for axis in np.eye(3):
                cross = np.cross(axis, direction)
                parallel = axis * np.dot(axis, direction)
                plus = direction * np.cos(epsilon) + cross * np.sin(epsilon) + parallel * (1 - np.cos(epsilon))
                minus = direction * np.cos(epsilon) - cross * np.sin(epsilon) + parallel * (1 - np.cos(epsilon))
                derivative = np.dot(plus - minus, separation) / (2 * epsilon)
                self.assertAlmostEqual(float(np.dot(wrench0[row, 3:], axis)), float(derivative), delta=2.0e-6)
            np.testing.assert_allclose(wrench1[row, 3:], 0.0, atol=2.0e-6, rtol=0.0)
        # Each locked or driven row impulse must have zero net force and world torque.
        np.testing.assert_allclose(wrench0[:, :3] + wrench1[:, :3], 0.0, atol=2.0e-6, rtol=0.0)
        torque = (
            wrench0[:, 3:]
            + np.cross(positions[0], wrench0[:, :3])
            + wrench1[:, 3:]
            + np.cross(positions[1], wrench1[:, :3])
        )
        np.testing.assert_allclose(torque, 0.0, atol=2.0e-6, rtol=0.0)

    def test_cartesian_d6_finite_limit_pushes_back(self) -> None:
        """Push a violated finite Cartesian limit back through the exact mechanism row."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-0.1, limit_upper=0.1),
            newton.ModelBuilder.JointDofConfig.create_unlimited((0.0, 1.0, 0.0)),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            joint_mode="maximal_direct",
        )
        self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
        _assert_direct_limits(self, solver, [0], [-0.1], [0.1])

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        body_q[0, 0] = 0.25
        state_0.body_q.assign(body_q)
        state_0.body_qd.zero_()
        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        position = float(state_1.body_q.numpy()[0, 0])
        self.assertLess(position, 0.25, msg=f"finite D6 limit did not reduce the violation: x={position}")
        # An exact mechanism recovers a violated bound at its drift rate.
        for _ in range(9):
            state_1.clear_forces()
            solver.step(state_1, state_1, model.control(), None, 1.0 / 60.0)

        position = float(state_1.body_q.numpy()[0, 0])
        velocity = state_1.body_qd.numpy()[0]
        self.assertAlmostEqual(position, 0.1, delta=5.0e-4)
        self.assertAlmostEqual(float(velocity[0]), 0.0, delta=1.0e-4)
        np.testing.assert_allclose(velocity[1:], 0.0, rtol=0.0, atol=2.0e-4)

    def test_six_axis_d6_limits_share_common_rows(self) -> None:
        """Prepare and solve all six bounded D6 axes through one representation."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = _make_body(builder)

        def bounded(axis):
            return newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=-0.1, limit_upper=0.1)

        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            linear_axes=[bounded(newton.Axis.X), bounded(newton.Axis.Y), bounded(newton.Axis.Z)],
            angular_axes=[bounded(newton.Axis.X), bounded(newton.Axis.Y), bounded(newton.Axis.Z)],
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        data = solver.world.constraints.d6
        self.assertEqual(int(data.row_count.numpy()[0]), 6)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        poses = state.body_q.numpy()
        poses[0, :3] = (0.2, -0.2, 0.2)
        rotation = wp.quat_rpy(0.2, -0.2, 0.2)
        poses[0, 3:] = (rotation[0], rotation[1], rotation[2], rotation[3])
        state.body_q.assign(poses)
        state.body_qd.zero_()
        joint_q = wp.zeros_like(model.joint_q)
        joint_qd = wp.zeros_like(model.joint_qd)

        state.clear_forces()
        solver.step(state, state, model.control(), None, 1.0 / 60.0)
        newton.eval_ik(model, state, joint_q, joint_qd)
        first_coordinates = joint_q.numpy()[:6]
        first_rates = joint_qd.numpy()[:6]
        self.assertLessEqual(float(np.max(first_coordinates * first_rates)), 2.0e-3)

        state.clear_forces()
        solver.step(state, state, model.control(), None, 1.0 / 60.0)
        newton.eval_ik(model, state, joint_q, joint_qd)
        coordinates = joint_q.numpy()[:6]
        self.assertTrue(np.isfinite(coordinates).all())
        self.assertLessEqual(float(np.max(np.abs(coordinates))), 0.1005)
        rates = joint_qd.numpy()[:6]
        self.assertTrue(np.isfinite(rates).all())
        self.assertLessEqual(
            float(np.max(coordinates * rates)),
            2.0e-4,
            msg=f"outward D6 rate: coordinates={coordinates}, rates={rates}",
        )

    def test_three_axis_d6_limits_remain_finite_at_gimbal_singularity(self) -> None:
        """Use the bounded reciprocal-axis fallback at singular pitch."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, limit_lower=-0.2, limit_upper=0.2),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, limit_lower=-2.0, limit_upper=2.0),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, limit_lower=-0.2, limit_upper=0.2),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        coordinates = np.asarray((0.0, np.pi / 2.0, 0.0), dtype=np.float32)
        model.joint_q.assign(coordinates)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        for _ in range(20):
            state.clear_forces()
            solver.step(state, state, model.control(), None, 1.0 / 60.0)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state.body_qd.numpy()).all())

    def test_d6_warm_start_clears_on_world_reset(self) -> None:
        """Clear every common D6 impulse when its world resets."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.X,
                limit_lower=-0.1,
                limit_upper=0.1,
                friction=0.5,
            ),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")
        data = solver.world.constraints.d6
        data.lower_impulse.fill_(1.0)
        data.upper_impulse.fill_(-2.0)
        data.friction_impulse.fill_(3.0)
        joint_constraint_clear_reset_worlds(
            solver.world.constraints,
            solver.bodies,
            1,
            wp.ones(1, dtype=wp.float32, device=model.device),
            device=model.device,
        )
        np.testing.assert_array_equal(data.lower_impulse.numpy(), 0.0)
        np.testing.assert_array_equal(data.upper_impulse.numpy(), 0.0)
        np.testing.assert_array_equal(data.friction_impulse.numpy(), 0.0)

    def test_live_d6_limit_refreshes_common_rows(self) -> None:
        """Rebuild common D6 rows and PGS ownership after live property edits."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = _make_body(builder)
        axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=5, joint_mode="maximal_direct")
        self.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 0)
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
        self.assertFalse(np.any(solver._direct_equality_system.direct_limit_dof_mask))

        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        lower[0], upper[0] = -0.1, 0.1
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _assert_direct_limits(self, solver, [0], [-0.1], [0.1])
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)

        lower[0], upper[0] = -1.0e10, 1.0e10
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 0)
        self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)
        self.assertFalse(np.any(solver._direct_equality_system.direct_limit_dof_mask))

    def test_cartesian_d6_drive_respects_common_limit(self) -> None:
        """Keep a direct D6 drive and its unilateral bound active together."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = _make_body(builder)
        linear_axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.X,
                limit_lower=-0.1,
                limit_upper=0.1,
                target_pos=0.5,
                target_ke=200.0,
                target_kd=20.0,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            ),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        joint = builder.add_joint_d6(parent=-1, child=body, linear_axes=linear_axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        direct = solver._direct_equality_system
        self.assertTrue(direct.enabled)
        self.assertTrue(bool(direct.direct_drive_joint_mask[0]))
        _assert_direct_limits(self, solver, [0], [-0.1], [0.1])

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        control = model.control()
        control.joint_target_q.assign(np.asarray((0.5, 0.0, 0.0), dtype=np.float32))
        for _ in range(120):
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 120.0)
        position = float(state.body_q.numpy()[0, 0])
        velocity = float(state.body_qd.numpy()[0, 0])
        self.assertLessEqual(position, 0.101, msg=f"direct drive escaped its D6 limit: x={position}")
        self.assertLessEqual(velocity, 1.0e-3, msg=f"direct drive pushed outward at its D6 limit: vx={velocity}")

    def test_fully_free_d6_friction_conserves_pair_momentum(self) -> None:
        """Damp relative D6 motion with equal and opposite impulses."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = _make_body(builder)
        child = _make_body(builder)
        linear_axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.X,
                limit_lower=-1.0e10,
                limit_upper=1.0e10,
                friction=0.5,
            ),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        angular_axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        joint = builder.add_joint_d6(parent=parent, child=child, linear_axes=linear_axes, angular_axes=angular_axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        self.assertFalse(solver._direct_equality_system.enabled)
        self.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 1)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[:, 0] = (-1.0, 1.0)
        state.body_qd.assign(velocity)
        for _ in range(30):
            state.clear_forces()
            solver.step(state, state, model.control(), None, 1.0 / 120.0)
        velocity = state.body_qd.numpy()
        np.testing.assert_allclose(np.sum(velocity[:, :3], axis=0), 0.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(np.sum(0.01 * velocity[:, 3:], axis=0), 0.0, rtol=0.0, atol=2.0e-6)
        self.assertAlmostEqual(abs(float(velocity[1, 0] - velocity[0, 0])), 1.75, delta=2.0e-3)

    def test_offset_hinge_limit_converges_under_impact(self) -> None:
        """Keep a two-sided hinge limit convergent when the pivot carries most of the inertia."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        # With the center of mass off the hinge, the pivot reaction more than
        # doubles the inertia a bare torque would see.
        body = builder.add_link(
            xform=wp.transform_identity(),
            mass=0.35,
            com=wp.vec3(0.1, 0.0, 0.0),
            inertia=((0.0005, 0.0, 0.0), (0.0, 0.0025, 0.0), (0.0, 0.0, 0.002)),
        )
        joint = builder.add_joint_revolute(
            parent=-1, child=body, axis=(0.0, 0.0, 1.0), limit_lower=-0.09, limit_upper=0.09
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_preferred_device())
        solver = newton.solvers.SolverPhoenX(
            model, joint_mode="maximal_direct", solver_iterations=8, velocity_iterations=1
        )
        state = model.state()
        for step in range(10):
            state.clear_forces()
            if step == 0:
                # An impact drives the hinge through the narrow limit window.
                forces = state.body_f.numpy()
                forces[body, 5] = 50.0
                state.body_f.assign(forces)
            solver.step(state, state, model.control(), None, 1.0 / 120.0)
            self.assertLess(float(np.abs(state.body_qd.numpy()[body, 3:]).max()), 50.0)
        orientation = state.body_q.numpy()[body, 3:]
        angle = 2.0 * np.arctan2(orientation[2], orientation[3])
        self.assertLessEqual(abs(float(angle)), 0.09 + 1.0e-3)

    def test_fully_free_d6_limit_conserves_pair_momentum(self) -> None:
        """Resolve a limit-only six-axis D6 joint with paired linear and angular momentum."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = _make_body(builder)
        child = _make_body(builder)
        linear_axes = [
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, limit_lower=-0.1, limit_upper=0.1),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        angular_axes = [
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Y),
            newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.Z),
        ]
        joint = builder.add_joint_d6(parent=parent, child=child, linear_axes=linear_axes, angular_axes=angular_axes)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(
            model,
            substeps=5,
            solver_iterations=8,
            velocity_iterations=1,
            joint_mode="maximal_direct",
        )
        self.assertFalse(solver._direct_equality_system.enabled)
        self.assertEqual(int(solver.world.constraints.d6.row_count.numpy()[0]), 1)

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        poses = state_0.body_q.numpy()
        poses[child, 0] = 0.25
        state_0.body_q.assign(poses)
        state_0.body_qd.zero_()
        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        poses = state_1.body_q.numpy()
        velocities = state_1.body_qd.numpy()
        separation = poses[child, :3] - poses[parent, :3]
        self.assertLessEqual(float(separation[0]), 0.1005)
        masses = np.asarray((1.0, 1.0), dtype=np.float64)
        linear_momentum = np.sum(masses[:, None] * velocities[:, :3], axis=0)
        angular_momentum = np.sum(
            0.01 * velocities[:, 3:] + np.cross(poses[:, :3], masses[:, None] * velocities[:, :3]),
            axis=0,
        )
        np.testing.assert_allclose(linear_momentum, 0.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(angular_momentum, 0.0, rtol=0.0, atol=2.0e-6)

        wrench0 = solver.world.constraints.d6.wrench0.numpy()[0, 0].astype(np.float64)
        wrench1 = solver.world.constraints.d6.wrench1.numpy()[0, 0].astype(np.float64)
        np.testing.assert_allclose(wrench0[:3] + wrench1[:3], 0.0, rtol=0.0, atol=2.0e-6)
        body_positions = solver.world.bodies.position.numpy()[[parent + 1, child + 1]].astype(np.float64)
        world_torque = (
            wrench0[3:]
            + np.cross(body_positions[0], wrench0[:3])
            + wrench1[3:]
            + np.cross(body_positions[1], wrench1[:3])
        )
        np.testing.assert_allclose(world_torque, 0.0, rtol=0.0, atol=2.0e-6)

    def test_three_axis_gimbals_use_direct_translation_rows(self) -> None:
        """Keep both gimbal handedness variants rotationally free through direct rows."""
        for label, third_axis in (("right", (0.0, 0.0, 1.0)), ("left", (0.0, 0.0, -1.0))):
            with self.subTest(handedness=label):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
                body = _make_body(builder)
                axes = [
                    newton.ModelBuilder.JointDofConfig.create_unlimited((1.0, 0.0, 0.0)),
                    newton.ModelBuilder.JointDofConfig.create_unlimited((0.0, 1.0, 0.0)),
                    newton.ModelBuilder.JointDofConfig.create_unlimited(third_axis),
                ]
                joint = builder.add_joint_d6(parent=-1, child=body, angular_axes=axes)
                builder.add_articulation([joint])
                model = builder.finalize()
                solver = newton.solvers.SolverPhoenX(
                    model,
                    substeps=5,
                    solver_iterations=1,
                    velocity_iterations=1,
                    joint_mode="maximal_direct",
                )
                direct = solver._direct_equality_system
                self.assertEqual(int(solver._joint_constraints.joint_mode.numpy()[0]), int(JOINT_MODE_GENERIC_D6))
                self.assertEqual(direct.topology.dimensions, (3,))
                self.assertTrue(bool(direct.joint_mask[0]))
                self.assertEqual(int(solver.world._joint_pgs_enabled.numpy()[0]), 0)

                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                initial = np.asarray([0.4, -0.3, 0.2, 0.7, -0.6, 0.5], dtype=np.float32)
                state.body_qd.assign(initial.reshape(1, 6))
                with wp.ScopedCapture(model.device) as capture:
                    solver.step(state, state, model.control(), None, 1.0 / 60.0)
                wp.capture_launch(capture.graph)
                velocity = state.body_qd.numpy()[0]
                np.testing.assert_allclose(velocity[:3], 0.0, rtol=0.0, atol=2.0e-4)
                np.testing.assert_allclose(velocity[3:], initial[3:], rtol=2.0e-4, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
