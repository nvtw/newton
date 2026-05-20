# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""D6 auto-dispatch tests for :class:`SolverPhoenX`.

PhoenX's :func:`model_adapter.build_adbs_init_arrays` detects D6 joint
configurations whose per-DoF lock pattern matches a specialized mode
(FIXED / BALL / REVOLUTE / PRISMATIC / UNIVERSAL) and routes them to
the corresponding constraint kernels. The first four route to existing
modes (Phase 1); UNIVERSAL is a new mode (Phase 2b) composed from the
BALL_SOCKET 3-row positional lock plus a 1-row angular twist lock that
reuses the existing axial drive/limit machinery in rigid Box2D mode.

Coverage (all CUDA + graph-captured):

* :class:`TestD6Detection` -- adapter-only check. Each pattern routes
  to the right ``joint_mode``; free DoF is picked correctly.

* :class:`TestD6Revolute` -- end-to-end equivalence: a D6 dispatched
  to REVOLUTE simulates identically to a native ``add_joint_revolute``.

* :class:`TestD6Universal` -- universal joint dispatch + the twist-lock
  invariant: a pendulum with a locked Z-axis must not accumulate
  rotation about Z even when swinging freely in X/Y.

* :class:`TestD6Unsupported` -- cylindrical and planar still raise
  ``NotImplementedError`` (their kernel implementations are Phase 2a /
  Phase 2c follow-ups).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_FIXED,
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
            ang_axes.append(newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=-1.0e6, limit_upper=1.0e6))
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

    def test_cylindrical_pattern_raises(self) -> None:
        """1 linear free + 1 angular free along same axis -- needs the
        Phase 2a cylindrical mode."""
        model = self._make_d6(lin_locks=[True, True, False], ang_locks=[True, True, False])
        with self.assertRaisesRegex(NotImplementedError, "Phase 2"):
            newton.solvers.SolverPhoenX(model, substeps=1)

    # Universal (1 angular axis locked, 2 angular free) is now supported
    # via JOINT_MODE_UNIVERSAL -- see :class:`TestD6Universal` below.

    def test_planar_pattern_raises(self) -> None:
        """2 linear + 1 angular free (planar joint) -- needs Phase 2c."""
        model = self._make_d6(lin_locks=[False, False, True], ang_locks=[True, True, False])
        with self.assertRaisesRegex(NotImplementedError, "Phase 2"):
            newton.solvers.SolverPhoenX(model, substeps=1)


if __name__ == "__main__":
    unittest.main()
