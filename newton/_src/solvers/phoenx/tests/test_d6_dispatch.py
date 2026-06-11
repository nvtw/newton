# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""D6 auto-dispatch tests for :class:`SolverPhoenX`.

PhoenX's :func:`model_adapter.build_adbs_init_arrays` detects D6 joint
configurations whose per-DoF lock pattern matches a specialized mode
(FIXED / BALL / REVOLUTE / PRISMATIC / UNIVERSAL / CYLINDRICAL / PLANAR)
and routes them to the corresponding constraint kernels. The first
four route to existing modes (Phase 1); UNIVERSAL (Phase 2b),
CYLINDRICAL (Phase 2a), and PLANAR (Phase 2c) are new modes built by
composing existing helpers with different subsets of constraint rows.

Coverage (all CUDA + graph-captured):

* :class:`TestD6Detection` -- adapter-only check.
* :class:`TestD6Revolute` -- end-to-end equivalence with native revolute.
* :class:`TestD6Universal` -- twist-lock invariant.
* :class:`TestD6Cylindrical` -- 2 free DoFs along the shared axis,
  4 perpendicular DoFs locked.
* :class:`TestD6Planar` -- 2 in-plane lin + 1 about-normal ang free,
  3 perpendicular DoFs locked.
* :class:`TestD6Unsupported` -- non-parallel-axes "cylindrical" and
  "planar" configurations still raise (= Phase 3 generic D6 territory).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    DRIVE_MODE_OFF,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)

_FPS = 240
_DT = 1.0 / _FPS


def _make_solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=20,
        velocity_iterations=2,
    )


def _build_d6_pendulum_revolute_equivalent(
    *,
    mass: float = 1.0,
    length: float = 0.5,
    inertia: float = 1.0e-4,
    init_angle: float = 0.0,
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    free_angular_index: int = 1,
    free_damping: float = 0.0,
) -> newton.Model:
    """D6 joint with the revolute pattern: 3 linear locked, 2 angular
    locked, 1 angular free along ``axis``.

    ``free_angular_index`` is 0/1/2 -- which of the three angular axes
    is the free one (the other two get a LOCKED limit window). Lets us
    test that the adapter picks the correct ``effective_qd`` regardless
    of where the free axis sits in the angular sublist.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    bob = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    builder.add_shape_box(bob, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    # Three linear axes: all locked. limit_lower > limit_upper is the
    # Newton sentinel for LOCKED.
    lin_axes = [
        newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0),
        newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=1.0, limit_upper=-1.0),
        newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=1.0, limit_upper=-1.0),
    ]
    # Three angular axes: two locked, one free along ``axis``. The free
    # one goes into slot ``free_angular_index``; the other two pick
    # arbitrary perpendicular axes (any LOCKED axis is fine since the
    # adapter doesn't read locked axes' direction).
    ang_axes = []
    for k in range(3):
        if k == free_angular_index:
            ang_axes.append(
                newton.ModelBuilder.JointDofConfig(
                    axis=axis,
                    limit_lower=-1.0e6,
                    limit_upper=1.0e6,
                    damping=free_damping,
                )
            )
        else:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0))

    j = builder.add_joint_d6(
        parent=-1,
        child=bob,
        linear_axes=lin_axes,
        angular_axes=ang_axes,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    if init_angle != 0.0:
        # The free angular DoF lives at qd_start + 3 + free_angular_index.
        qd_arr = np.zeros(model.joint_dof_count, dtype=np.float32)
        q_arr = np.zeros(model.joint_coord_count, dtype=np.float32)
        # joint_q layout matches joint_qd for D6 (scalar coords only).
        q_arr[3 + free_angular_index] = float(init_angle)
        model.joint_q.assign(q_arr)
        model.joint_qd.assign(qd_arr)
    return model


def _build_native_revolute_pendulum(
    *,
    mass: float = 1.0,
    length: float = 0.5,
    inertia: float = 1.0e-4,
    init_angle: float = 0.0,
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> newton.Model:
    """Same pendulum but constructed natively via ``add_joint_revolute``.
    Reference for the D6-revolute equivalence test."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    bob = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    builder.add_shape_box(bob, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    j = builder.add_joint_revolute(
        parent=-1,
        child=bob,
        axis=axis,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))
    if init_angle != 0.0:
        model.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    return model


def _run_pendulum(model: newton.Model, *, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Captures (clear, step) into a CUDA graph and replays. Returns
    ``(joint_q_traj, joint_qd_traj)`` per frame."""
    device = wp.get_device()
    assert device.is_cuda, "PhoenX D6 tests require CUDA"

    solver = _make_solver(model)
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    def _frame() -> None:
        s0.clear_forces()
        solver.step(s0, s1, model.control(), None, _DT)
        newton.eval_ik(model, s1, jq, jqd)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)

    q_traj = np.empty((frames, model.joint_coord_count), dtype=np.float32)
    qd_traj = np.empty((frames, model.joint_dof_count), dtype=np.float32)

    _frame()
    q_traj[0] = jq.numpy()
    qd_traj[0] = jqd.numpy()
    if frames == 1:
        return q_traj, qd_traj

    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph
    for i in range(1, frames):
        wp.capture_launch(graph)
        q_traj[i] = jq.numpy()
        qd_traj[i] = jqd.numpy()
    return q_traj, qd_traj


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Detection(unittest.TestCase):
    """Adapter-level check: the right ``joint_mode`` is picked for each
    known pattern. Reads ``solver._adbs.joint_mode`` directly so the test
    catches dispatch bugs without running physics."""

    def _adbs_mode_for(self, model: newton.Model) -> int:
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        return int(solver._adbs.joint_mode.numpy()[0])

    def test_d6_all_locked_dispatches_to_fixed(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_FIXED))

    def test_d6_ball_pattern_dispatches_to_ball_socket(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_BALL_SOCKET))

    def test_d6_angular_only_ball_dispatches_to_ball_socket(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_BALL_SOCKET))

    def test_d6_ball_passive_damping_is_packed(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6, damping=1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6, damping=3.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=-1.0e6, limit_upper=1.0e6, damping=2.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_BALL_SOCKET))
        self.assertEqual(int(solver._adbs.drive_mode.numpy()[0]), int(DRIVE_MODE_OFF))
        self.assertAlmostEqual(float(solver._adbs.damping_drive.numpy()[0]), 3.0)

    def test_d6_revolute_passive_damping_uses_velocity_row(self) -> None:
        model = _build_d6_pendulum_revolute_equivalent(free_angular_index=1, free_damping=4.0)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_REVOLUTE))
        self.assertEqual(int(solver._adbs.drive_mode.numpy()[0]), int(DRIVE_MODE_VELOCITY))
        self.assertAlmostEqual(float(solver._adbs.damping_drive.numpy()[0]), 4.0)

    def test_d6_angular_only_universal_dispatches_to_universal(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_UNIVERSAL))

    def test_d6_universal_passive_damping_is_packed(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6, damping=2.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6, damping=4.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_UNIVERSAL))
        self.assertAlmostEqual(float(solver._adbs.damping_drive.numpy()[0]), 4.0)

    def test_d6_universal_free_axis_limits_are_packed(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-0.25, limit_upper=0.30),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_UNIVERSAL))
        self.assertEqual(int(solver._adbs.d6_limit_count.numpy()[0]), 1)
        np.testing.assert_allclose(solver._adbs.d6_limit_lower.numpy()[0], [-0.25, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(solver._adbs.d6_limit_upper.numpy()[0], [0.30, 0.0, 0.0], atol=1.0e-6)

    def test_d6_revolute_pattern_dispatches_to_revolute(self) -> None:
        model = _build_d6_pendulum_revolute_equivalent(free_angular_index=1)
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_REVOLUTE))

    def test_d6_revolute_pattern_picks_correct_free_axis(self) -> None:
        """The free angular axis can be at any of the 3 angular-sublist
        positions. The adapter must pick the right qd and the right axis
        vector regardless. Tests offset 0, 1, 2."""
        for free_idx in (0, 1, 2):
            with self.subTest(free_idx=free_idx):
                model = _build_d6_pendulum_revolute_equivalent(free_angular_index=free_idx)
                solver = newton.solvers.SolverPhoenX(model, substeps=1)
                self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_REVOLUTE))
                # joint_idx_to_dof_start must point at the free DoF.
                # joint_qd_start == 0 for the only joint; free axis at
                # offset (3 lin + free_idx).
                expected_dof_start = 3 + free_idx
                self.assertEqual(int(solver._adbs.joint_idx_to_dof_start.numpy()[0]), expected_dof_start)

    def test_coord_layout_drive_uses_target_q_index(self) -> None:
        prev = newton.use_coord_layout_targets
        try:
            newton.use_coord_layout_targets = True
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            root = builder.add_link(xform=wp.transform_identity(), mass=1.0)
            child = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5), q=wp.quat_identity()), mass=1.0)
            free = builder.add_joint_free(child=root)
            revolute = builder.add_joint_revolute(
                parent=root,
                child=child,
                axis=(0.0, 1.0, 0.0),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
                target_pos=0.7,
                target_ke=10.0,
                target_kd=1.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
            builder.add_articulation([free, revolute])
            model = builder.finalize()
        finally:
            newton.use_coord_layout_targets = prev

        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.drive_dof_start.numpy()[0]), 6)
        self.assertEqual(int(solver._adbs.drive_target_q_index.numpy()[0]), 7)
        self.assertAlmostEqual(float(solver._adbs.target.numpy()[0]), 0.7, places=6)

    def test_d6_prismatic_pattern_dispatches_to_prismatic(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        self.assertEqual(self._adbs_mode_for(model), int(JOINT_MODE_PRISMATIC))


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Revolute(unittest.TestCase):
    """End-to-end equivalence: a D6 dispatched to revolute must simulate
    identically to a native revolute joint with the same geometry, mass,
    and initial conditions. Any divergence flags an adapter bug.

    Tolerance is tight (1e-3 rad in q over 60 frames of free swing)
    because the constraint kernels are bit-identical -- the only
    differences could come from the adapter setting up slightly
    different `eff_inv` or `joint_q_at_init` values."""

    def test_d6_revolute_matches_native_revolute_under_gravity(self) -> None:
        kwargs = {"mass": 1.0, "length": 0.5, "inertia": 1.0e-4, "init_angle": 0.3, "axis": (0.0, 1.0, 0.0)}
        model_native = _build_native_revolute_pendulum(**kwargs)
        model_d6 = _build_d6_pendulum_revolute_equivalent(**kwargs, free_angular_index=1)

        q_native, qd_native = _run_pendulum(model_native, frames=60)
        q_d6, qd_d6 = _run_pendulum(model_d6, frames=60)

        # native joint_q is scalar [q]. D6 joint_q is [0,0,0, q_x,q_y,q_z]
        # (3 zero linear + 3 angular with only the free one non-zero).
        # Free axis index is 1 (Y), so q_d6[:, 4] is the equivalent of
        # the native q_native[:, 0].
        np.testing.assert_allclose(
            q_d6[:, 3 + 1],
            q_native[:, 0],
            atol=1.0e-3,
            err_msg="D6-revolute and native-revolute joint angles must match under identical gravity",
        )
        np.testing.assert_allclose(
            qd_d6[:, 3 + 1],
            qd_native[:, 0],
            atol=1.0e-2,
            err_msg="D6-revolute and native-revolute joint velocities must match",
        )


def _build_d6_cylindrical_piston(
    *,
    mass: float = 1.0,
    inertia: float = 1.0e-2,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    free_lin_index: int = 2,
    free_ang_index: int = 2,
    init_slide: float = 0.0,
    init_spin: float = 0.0,
) -> newton.Model:
    """Cylindrical joint: piston that can slide along ``axis`` and spin
    about ``axis``. The other 2 linear axes (perpendicular) and 2 angular
    axes (perpendicular) are locked.

    The free linear and angular axes share ``axis`` (parallel) -- this
    is the requirement for the cylindrical pattern to be detected. The
    other two linear / angular axes are populated with arbitrary
    perpendicular directions (their direction is irrelevant since
    they're locked)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    lin_axes = []
    for k in range(3):
        if k == free_lin_index:
            lin_axes.append(newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=-1.0e6, limit_upper=1.0e6))
        else:
            lin_axes.append(newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0))
    ang_axes = []
    for k in range(3):
        if k == free_ang_index:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=-1.0e6, limit_upper=1.0e6))
        else:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0))

    j = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=lin_axes,
        angular_axes=ang_axes,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j])
    model = builder.finalize()
    # No gravity for the kinematic tests below; we test the constraint
    # behavior in isolation.
    model.set_gravity((0.0, 0.0, 0.0))

    # Seed via joint_q so eval_fk produces a consistent body_q.
    if init_slide != 0.0 or init_spin != 0.0:
        q_arr = np.zeros(model.joint_coord_count, dtype=np.float32)
        q_arr[free_lin_index] = float(init_slide)
        q_arr[3 + free_ang_index] = float(init_spin)
        model.joint_q.assign(q_arr)
    return model


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Cylindrical(unittest.TestCase):
    """Cylindrical joint: 2 lin locked + 1 lin free + 2 ang locked + 1
    ang free, with the free linear and angular axes parallel. The body
    must:

    * Slide and spin freely along/about the shared axis (2 free DoFs).
    * Stay locked in the 4 perpendicular DoFs (no perpendicular
      translation, no off-axis rotation)."""

    def test_dispatches_to_cylindrical_mode(self) -> None:
        model = _build_d6_cylindrical_piston()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_CYLINDRICAL))

    def test_body_translates_and_rotates_only_along_axis(self) -> None:
        """Seed a body with linear velocity along Z and angular velocity
        about Z. After many frames, the body must:

        * Have moved along Z (translation is free) and rotated about Z
          (rotation is free).
        * Stayed at x = y = 0 (perpendicular translation is locked).
        * Have angular velocity perpendicular to Z still ~0 (perpendicular
          rotation is locked).
        """
        model = _build_d6_cylindrical_piston(axis=(0.0, 0.0, 1.0))
        device = wp.get_device()
        solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=20)

        s0 = model.state()
        s1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        # Seed body_qd: linear along Z (1 m/s) and angular about Z (2 rad/s).
        # Other components zero.
        init_qd = np.zeros((1, 6), dtype=np.float32)
        init_qd[0, 2] = 1.0  # v_z
        init_qd[0, 5] = 2.0  # omega_z
        s0.body_qd.assign(init_qd)

        n_frames = 120

        def _frame() -> None:
            s0.clear_forces()
            solver.step(s0, s1, model.control(), None, _DT)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)

        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph
        for _ in range(n_frames - 1):
            wp.capture_launch(graph)

        body_q = s0.body_q.numpy()[0]
        body_qd = s0.body_qd.numpy()[0]
        # Position: lateral (x, y) must be ~0 (locked); z arbitrary (free).
        self.assertLess(abs(float(body_q[0])), 1e-3, f"lateral X drift = {body_q[0]}")
        self.assertLess(abs(float(body_q[1])), 1e-3, f"lateral Y drift = {body_q[1]}")
        # The body must actually have translated along Z by ~1 m (1 m/s * 0.5 s).
        self.assertGreater(abs(float(body_q[2])), 0.3, f"Z translation = {body_q[2]} -- slide didn't happen")
        # Velocity: perpendicular components (vx, vy, omega_x, omega_y) ~0.
        self.assertLess(abs(float(body_qd[0])), 0.05, f"v_x = {body_qd[0]}")
        self.assertLess(abs(float(body_qd[1])), 0.05, f"v_y = {body_qd[1]}")
        self.assertLess(abs(float(body_qd[3])), 0.05, f"omega_x = {body_qd[3]}")
        self.assertLess(abs(float(body_qd[4])), 0.05, f"omega_y = {body_qd[4]}")
        # Free components retained: v_z still ~1, omega_z still ~2 (no friction).
        self.assertAlmostEqual(float(body_qd[2]), 1.0, delta=0.05, msg=f"v_z = {body_qd[2]}")
        self.assertAlmostEqual(float(body_qd[5]), 2.0, delta=0.05, msg=f"omega_z = {body_qd[5]}")


def _build_d6_universal_pendulum(
    *,
    mass: float = 1.0,
    length: float = 0.5,
    inertia: float = 1.0e-2,
    locked_angular_index: int = 2,
    locked_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    init_swing_axis: int = 1,
    init_swing_angle: float = 0.3,
) -> newton.Model:
    """A pendulum hung from a universal joint: 3 lin locked + 1 ang
    locked + 2 ang free. The pendulum can swing in 2 axes; the locked
    axis (typically vertical) prevents twist about the suspension line.

    ``locked_axis`` is the world-frame axis that's locked; the two
    perpendicular axes are free to swing. ``locked_angular_index``
    picks which of the 3 angular slots in the D6 spec holds the locked
    axis (the other two get an arbitrary perpendicular direction; their
    direction is irrelevant since they're not constrained)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    bob = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    builder.add_shape_box(bob, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    lin_axes = [
        newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0),
        newton.ModelBuilder.JointDofConfig(axis=(0.0, 1.0, 0.0), limit_lower=1.0, limit_upper=-1.0),
        newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=1.0, limit_upper=-1.0),
    ]
    ang_axes = []
    for k in range(3):
        if k == locked_angular_index:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=locked_axis, limit_lower=1.0, limit_upper=-1.0))
        else:
            ang_axes.append(
                newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0e6, limit_upper=1.0e6)
            )

    j = builder.add_joint_d6(
        parent=-1,
        child=bob,
        linear_axes=lin_axes,
        angular_axes=ang_axes,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    # Seed body angular velocity to swing about an axis perpendicular to
    # the locked axis. We use direct body_qd seeding (rather than
    # joint_qd) because for the D6 universal pattern, joint_qd[locked]
    # would have to stay zero, and seeding only joint_qd of a free axis
    # via eval_fk requires careful index management. body_qd is the
    # ground truth for PhoenX's import path anyway.
    if init_swing_angle != 0.0:
        # Approximate: spin about ``init_swing_axis`` so the body starts
        # with non-zero angular velocity perpendicular to the locked axis.
        qd_arr = np.zeros((1, 6), dtype=np.float32)
        qd_arr[0, 3 + init_swing_axis] = float(init_swing_angle)  # angular component
        # Linear velocity at COM for consistency with the joint anchor
        # (anchor must stay at world origin -> v_com = cross(omega, r)
        # where r = anchor - com = (0, 0, +length)).
        omega = np.zeros(3, dtype=np.float32)
        omega[init_swing_axis] = float(init_swing_angle)
        r = np.array([0.0, 0.0, float(length)], dtype=np.float32)
        v_com = np.cross(omega, r)
        qd_arr[0, 0:3] = v_com
        # Done in the state directly in the test runner.
        model._init_body_qd = qd_arr  # stash for test runner
    return model


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Universal(unittest.TestCase):
    """A D6 with 3 lin locked + 1 ang locked + 2 ang free dispatches to
    JOINT_MODE_UNIVERSAL. The simulated body must:

    * Stay attached at the joint anchor (3 lin locked is enforced).
    * Stay un-twisted about the locked axis (1 ang locked is enforced
      via the rigid Box2D path on the axial row).
    * Swing freely about the 2 perpendicular angular axes (no rotational
      constraint there).
    """

    def test_universal_pattern_dispatches_to_universal(self) -> None:
        model = _build_d6_universal_pendulum(locked_angular_index=2)
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_UNIVERSAL))

    def test_universal_locked_axis_stays_zero(self) -> None:
        """A bob suspended from a universal joint, released with an
        angular velocity perpendicular to the locked axis, must not
        accumulate twist about the locked axis. Twist drift > a few
        milliradians flags a failure of the 1-row angular lock."""
        # Locked axis = Z (vertical). Bob hangs below; swings in X/Y.
        model = _build_d6_universal_pendulum(
            locked_angular_index=2, locked_axis=(0.0, 0.0, 1.0), init_swing_axis=1, init_swing_angle=0.5
        )

        # Set up and run with seeded angular velocity.
        device = wp.get_device()
        solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=20)
        s0 = model.state()
        s1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        if hasattr(model, "_init_body_qd"):
            s0.body_qd.assign(model._init_body_qd)

        n_frames = 60

        def _frame() -> None:
            s0.clear_forces()
            solver.step(s0, s1, model.control(), None, _DT)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)

        # Warm-up + graph capture
        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph
        for _ in range(n_frames - 1):
            wp.capture_launch(graph)

        # The body's orientation should remain such that its local Z
        # axis projects close to (0, 0, 1) under the locked-axis (world Z)
        # constraint. Equivalently: a vector that started along world Z
        # in the body frame should still align with world Z (twist about
        # Z is forbidden).
        body_q = s0.body_q.numpy()[0]
        body_rot = body_q[3:7]  # quaternion (x, y, z, w)
        # Body-local Z axis rotated into world.
        z_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x, y, z, w = body_rot
        # Quaternion-rotate z_local: t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
        qxyz = np.array([x, y, z], dtype=np.float32)
        _t_vec_z = 2.0 * np.cross(qxyz, z_local)
        # Universal allows swing in 2 axes, so the body's "up" axis can
        # rotate away from world Z -- that's the whole point. What
        # CAN'T happen is rotation ABOUT world Z (twist). Check the twist
        # by projecting the body's local X axis onto the X-Y plane: a
        # universal joint with locked twist keeps this projection fixed
        # in direction (only its magnitude changes as the body tilts).
        # At rest the body's X axis points along world X; after swinging,
        # if no twist, the X axis projected to X-Y plane should still
        # point along world X (modulo the projection shortening it).
        x_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t_vec_x = 2.0 * np.cross(qxyz, x_local)
        x_world = x_local + w * t_vec_x + np.cross(qxyz, t_vec_x)
        x_world_in_plane = np.array([x_world[0], x_world[1]], dtype=np.float32)
        plane_len = float(np.linalg.norm(x_world_in_plane))
        if plane_len > 1e-3:
            x_world_in_plane = x_world_in_plane / plane_len
            twist_angle = abs(float(np.arctan2(x_world_in_plane[1], x_world_in_plane[0])))
        else:
            twist_angle = 0.0
        # 50 mrad tolerance: the angular lock uses the rigid Box2D path
        # (hertz = 1e9) which converges to ~ms-scale on a typical PGS
        # iteration budget. A broken lock (no constraint applied) would
        # produce arbitrary twist as the body swings around.
        self.assertLess(
            twist_angle,
            0.05,
            msg=f"twist about locked axis = {twist_angle:.4f} rad; should be ~0 (universal joint locks twist)",
        )

    def test_universal_free_axis_limit_stops_rotation(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(
            xform=wp.transform_identity(),
            mass=1.0,
            inertia=((1.0e-2, 0, 0), (0, 1.0e-2, 0), (0, 0, 1.0e-2)),
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-0.25, limit_upper=0.25),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, 0.0))

        device = wp.get_device()
        solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=20)
        s0 = model.state()
        s1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        body_qd = np.zeros((1, 6), dtype=np.float32)
        body_qd[0, 3] = 4.0
        s0.body_qd.assign(body_qd)
        jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

        def _frame() -> None:
            s0.clear_forces()
            solver.step(s0, s1, model.control(), None, _DT)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)

        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph
        for _ in range(119):
            wp.capture_launch(graph)

        newton.eval_ik(model, s0, jq, jqd)
        q = jq.numpy()
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertLess(abs(float(q[0])), 0.40)


def _build_d6_planar_puck(
    *,
    mass: float = 1.0,
    inertia: float = 1.0e-2,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    locked_lin_index: int = 2,
    free_ang_index: int = 2,
) -> newton.Model:
    """Planar joint: puck constrained to a plane perpendicular to
    ``normal``. Can translate in the plane (2 in-plane lin axes free)
    and rotate about ``normal`` (1 ang axis free).

    ``locked_lin_index`` is the slot in the linear sublist that holds
    the locked direction (= the plane normal). ``free_ang_index`` is
    the slot in the angular sublist that holds the free direction
    (= the same plane normal). The adapter checks that the locked-lin
    axis and free-ang axis are parallel."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    body = builder.add_link(
        xform=wp.transform_identity(),
        mass=float(mass),
        inertia=((inertia, 0, 0), (0, inertia, 0), (0, 0, inertia)),
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    lin_axes = []
    for k in range(3):
        if k == locked_lin_index:
            lin_axes.append(newton.ModelBuilder.JointDofConfig(axis=normal, limit_lower=1.0, limit_upper=-1.0))
        else:
            lin_axes.append(
                newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=-1.0e6, limit_upper=1.0e6)
            )
    ang_axes = []
    for k in range(3):
        if k == free_ang_index:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=normal, limit_lower=-1.0e6, limit_upper=1.0e6))
        else:
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), limit_lower=1.0, limit_upper=-1.0))

    j = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=lin_axes,
        angular_axes=ang_axes,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Planar(unittest.TestCase):
    """Planar joint: 1 lin locked + 2 lin free + 2 ang locked + 1 ang
    free, with locked-lin and free-ang axes parallel. The body must:

    * Translate freely in the plane (2 in-plane DoFs unactuated).
    * Rotate freely about the plane normal (1 angular DoF unactuated).
    * Stay locked normal to the plane (out-of-plane translation = 0).
    * Stay locked in pitch / roll (rotation perpendicular to the normal = 0).
    """

    def test_dispatches_to_planar_mode(self) -> None:
        model = _build_d6_planar_puck()
        solver = newton.solvers.SolverPhoenX(model, substeps=1)
        self.assertEqual(int(solver._adbs.joint_mode.numpy()[0]), int(JOINT_MODE_PLANAR))

    def test_body_stays_in_plane_with_free_in_plane_motion(self) -> None:
        """Seed a puck with linear velocity in the plane (X+Y) and
        angular velocity about the normal (Z). After replay, the puck
        must have translated in X/Y, rotated about Z, and stayed at
        z = 0 with no off-axis rotation."""
        model = _build_d6_planar_puck(normal=(0.0, 0.0, 1.0))
        device = wp.get_device()
        solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=20)

        s0 = model.state()
        s1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        # In-plane translation: v_x = 0.5, v_y = 0.3. About-normal spin: omega_z = 1.5.
        init_qd = np.zeros((1, 6), dtype=np.float32)
        init_qd[0, 0] = 0.5
        init_qd[0, 1] = 0.3
        init_qd[0, 5] = 1.5
        s0.body_qd.assign(init_qd)

        n_frames = 120

        def _frame() -> None:
            s0.clear_forces()
            solver.step(s0, s1, model.control(), None, _DT)
            wp.copy(s0.body_q, s1.body_q)
            wp.copy(s0.body_qd, s1.body_qd)

        _frame()
        with wp.ScopedCapture(device=device) as capture:
            _frame()
        graph = capture.graph
        for _ in range(n_frames - 1):
            wp.capture_launch(graph)

        body_q = s0.body_q.numpy()[0]
        body_qd = s0.body_qd.numpy()[0]
        # z-position locked to ~0 (the plane).
        self.assertLess(abs(float(body_q[2])), 1e-3, f"z-drift = {body_q[2]}")
        # In-plane motion: x and y should have advanced (~ vt for v=0.5/0.3, t=0.5s).
        self.assertGreater(abs(float(body_q[0])), 0.15, f"X translation = {body_q[0]}")
        self.assertGreater(abs(float(body_q[1])), 0.1, f"Y translation = {body_q[1]}")
        # Velocity components: v_z and omega_x, omega_y locked.
        self.assertLess(abs(float(body_qd[2])), 0.05, f"v_z = {body_qd[2]}")
        self.assertLess(abs(float(body_qd[3])), 0.05, f"omega_x = {body_qd[3]}")
        self.assertLess(abs(float(body_qd[4])), 0.05, f"omega_y = {body_qd[4]}")
        # Free velocities retained (no friction, no gravity).
        self.assertAlmostEqual(float(body_qd[0]), 0.5, delta=0.05, msg=f"v_x = {body_qd[0]}")
        self.assertAlmostEqual(float(body_qd[1]), 0.3, delta=0.05, msg=f"v_y = {body_qd[1]}")
        self.assertAlmostEqual(float(body_qd[5]), 1.5, delta=0.05, msg=f"omega_z = {body_qd[5]}")


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX D6 tests run on CUDA only")
class TestD6Unsupported(unittest.TestCase):
    """Configurations outside Phase 1's pattern set must raise a
    descriptive ``NotImplementedError`` rather than silently misroute."""

    def _make_d6(self, lin_locks: list[bool], ang_locks: list[bool]) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        # Axis 0 -> X, 1 -> Y, 2 -> Z. LOCKED uses limit_lower > limit_upper.
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        lin = [
            newton.ModelBuilder.JointDofConfig(
                axis=axes[i],
                limit_lower=1.0 if lin_locks[i] else -1.0e6,
                limit_upper=-1.0 if lin_locks[i] else 1.0e6,
            )
            for i in range(3)
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(
                axis=axes[i],
                limit_lower=1.0 if ang_locks[i] else -1.0e6,
                limit_upper=-1.0 if ang_locks[i] else 1.0e6,
            )
            for i in range(3)
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        return builder.finalize()

    # Cylindrical (1 lin + 1 ang free along same axis) and universal
    # (1 ang locked + 2 ang free) are now supported via JOINT_MODE_CYLINDRICAL
    # and JOINT_MODE_UNIVERSAL respectively -- see TestD6Cylindrical and
    # TestD6Universal classes for their end-to-end coverage.

    def test_cylindrical_with_non_parallel_axes_raises(self) -> None:
        """A "cylindrical-shaped" lock pattern (2 lin + 1 lin free,
        2 ang + 1 ang free) with the free axes NOT parallel is not a
        physical cylindrical joint -- it's a more general D6. The
        adapter must reject it with a Phase 2+ message."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        # Linear free along Z, angular free along X -- perpendicular.
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=-1.0e6, limit_upper=1.0e6),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        with self.assertRaisesRegex(NotImplementedError, "Generic D6"):
            newton.solvers.SolverPhoenX(model, substeps=1)

    # Planar pattern with parallel locked-lin and free-ang axes is now
    # supported via JOINT_MODE_PLANAR -- see :class:`TestD6Planar`.

    def test_planar_pattern_non_parallel_axes_raises(self) -> None:
        """A planar-shaped lock pattern with the locked-lin axis NOT
        parallel to the free-ang axis is not a physical planar joint
        and must be rejected."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        # Lock the Z translation; free the Y rotation (perpendicular to Z).
        lin = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        ang = [
            newton.ModelBuilder.JointDofConfig(axis=(1, 0, 0), limit_lower=1.0, limit_upper=-1.0),
            newton.ModelBuilder.JointDofConfig(axis=(0, 1, 0), limit_lower=-1.0e6, limit_upper=1.0e6),
            newton.ModelBuilder.JointDofConfig(axis=(0, 0, 1), limit_lower=1.0, limit_upper=-1.0),
        ]
        j = builder.add_joint_d6(parent=-1, child=body, linear_axes=lin, angular_axes=ang)
        builder.add_articulation([j])
        model = builder.finalize()
        with self.assertRaisesRegex(NotImplementedError, "Generic D6"):
            newton.solvers.SolverPhoenX(model, substeps=1)


if __name__ == "__main__":
    unittest.main()
