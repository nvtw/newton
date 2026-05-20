# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint Coulomb-friction tests for :class:`SolverPhoenX`.

PhoenX implements per-joint axial friction as a saturated soft
constraint on the same scalar row as the drive / limit, matching
MuJoCo's ``dof_frictionloss + actuator`` decomposition: the total
axial impulse per substep is the sum of the clamped PD drive term and
a clamped friction term capped at ``±μ * dt``. The friction iterate is
the same regularized PD math as the drive, with ``stiffness = 0``,
``target_velocity = 0``, and ``max_force = μ``; the regularization
``gamma`` is sized from :data:`solver_config.PHOENX_FRICTION_SLIP_VELOCITY`
so the slip velocity at saturation equals the configured constant
regardless of joint impedance.

All analytical fixtures here use a **rotor** (a body anchored at the
joint origin so the body's COM-frame axial inertia equals the joint
effective inertia). PhoenX's friction iterate operates in maximal
coordinates and scales with the body's COM-frame inverse inertia
``n . I_com^-1 . n``; for a body offset from the joint the parallel-axis
term ``m * L^2`` is *not* folded into that scalar, so the impulse-domain
friction would not match the analytical Coulomb model on a point-mass
pendulum. Rotor fixtures sidestep this by collocating the COM with the
joint. The stiction test still uses a pendulum because "joint locked"
is parallel-axis-invariant (the joint doesn't rotate either way).

Four analytical fixtures, all CUDA + graph-captured:

* :class:`TestFrictionStiction` -- friction large enough to overcome
  gravity. A pendulum released at a non-equilibrium angle must stay
  near its initial angle when ``μ > m * g * L * sin(θ_0)``.

* :class:`TestFrictionFreeSpinDecay` -- the iconic Coulomb test. A
  rotor spinning at known ``ω_0`` in zero gravity, with friction ``μ``
  and inertia ``I``, must decelerate at ``alpha = -μ / I`` and reach zero
  velocity after ``t_stop = I * ω_0 / μ``. The post-stop state must
  remain stationary (saturated friction holds qd=0 once reached).

* :class:`TestFrictionConstantTorque` -- dual of free-spin decay. A
  rotor driven by a constant applied torque ``τ > μ`` in zero gravity
  must accelerate at ``alpha = (τ - μ) / I``.

* :class:`TestFrictionAndDriveCompose` -- drive and friction are
  independent saturations on the same axial row. With a PD drive
  pushing toward a target and a friction load opposing motion, the
  steady-state position lies wherever
  ``|τ_drive| <= μ`` (friction's deadband expands the drive's natural
  setpoint).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

_FPS = 240
_DT = 1.0 / _FPS


def _solver(model: newton.Model) -> newton.solvers.SolverPhoenX:
    """Substep / iteration counts tuned for clean analytical convergence at
    the cost of throughput. Friction is a saturated soft constraint;
    enough PGS iterations are required for the saturation to converge to
    the analytical ``μ * dt`` bound at the first frame."""
    return newton.solvers.SolverPhoenX(
        model,
        substeps=4,
        solver_iterations=20,
        velocity_iterations=2,
    )


def _capture_steps(
    model: newton.Model,
    solver: newton.solvers.SolverPhoenX,
    *,
    n_frames: int,
    control_setup: object = None,
    init_joint_qd: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture a single-step graph and replay ``n_frames`` times, returning
    ``(joint_q, joint_qd)`` histories per frame.

    ``control_setup`` is an optional ``(control, np.ndarray)`` tuple: the
    array is written into ``control.joint_f`` before the warm-up frame.
    The reference is preserved across graph replay so the captured launch
    keeps using the same backing memory each frame. ``None`` leaves
    forces zero.

    ``init_joint_qd`` is an optional initial joint velocity [rad/s] (first
    DOF only). Seeded via ``model.joint_qd`` so ``eval_fk`` derives a
    consistent ``body_qd`` -- writing into ``state.body_qd`` directly
    leaves the rigid-velocity manifold in an inconsistent state for the
    PhoenX import path."""
    device = wp.get_device()
    assert device.is_cuda, "PhoenX friction tests require CUDA"

    if init_joint_qd is not None and model.joint_dof_count > 0:
        model.joint_qd.assign(np.array([float(init_joint_qd)], dtype=np.float32))

    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    if control_setup is None:
        control = model.control()
    else:
        control, joint_f_vals = control_setup
        control.joint_f.assign(joint_f_vals)

    jq = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=device)
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    def _frame() -> None:
        s0.clear_forces()
        solver.step(s0, s1, control, None, _DT)
        newton.eval_ik(model, s1, jq, jqd)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)

    q_traj = np.empty(n_frames, dtype=np.float32)
    qd_traj = np.empty(n_frames, dtype=np.float32)

    if n_frames < 1:
        return q_traj, qd_traj

    # Warm-up: compile kernels, allocate scratch.
    _frame()
    q_traj[0] = float(jq.numpy()[0])
    qd_traj[0] = float(jqd.numpy()[0])
    if n_frames == 1:
        return q_traj, qd_traj

    with wp.ScopedCapture(device=device) as capture:
        _frame()
    graph = capture.graph

    for i in range(1, n_frames):
        wp.capture_launch(graph)
        q_traj[i] = float(jq.numpy()[0])
        qd_traj[i] = float(jqd.numpy()[0])
    return q_traj, qd_traj


def _build_rotor(
    *,
    inertia: float,
    friction: float,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    target_pos: float = 0.0,
) -> newton.Model:
    """Body anchored at the world origin via a +Y revolute joint. No
    gravity. The body's COM sits at the joint, so its COM-frame axial
    inertia equals the joint effective inertia -- the impulse-domain
    iterate (which scales with ``n . I_com^-1 . n``) reproduces the
    analytical Coulomb result exactly.

    Use this fixture for any test that compares simulated dynamics to
    the analytical model ``τ = clamp(-k * qd, -μ, +μ)``."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(friction=friction)
    body = mb.add_link(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, inertia)),
    )
    mb.add_shape_box(
        body,
        hx=0.01,
        hy=0.01,
        hz=0.01,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        target_pos=float(target_pos),
        target_ke=float(target_ke),
        target_kd=float(target_kd),
        actuator_mode=newton.JointTargetMode.POSITION if target_ke > 0.0 else newton.JointTargetMode.NONE,
    )
    mb.add_articulation([joint])
    mb.gravity = 0.0  # rotor tests are gravity-free
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_pendulum(
    *,
    mass: float,
    length: float,
    friction: float,
    init_angle: float = 0.0,
    gravity: float = -9.81,
) -> newton.Model:
    """Gravity-loaded pendulum. Bob offset from the joint by ``length``
    along ``-z``; joint axis ``+y``. Used only for the stiction test --
    "joint locked" is parallel-axis-invariant so the impulse-domain
    discrepancy doesn't matter there.

    COM-frame inertia is small but non-degenerate (PhoenX's maximal-
    coordinate solver requires at least ~1e-4 for stability)."""
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(friction=friction)
    bob = mb.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -float(length)), q=wp.quat_identity()),
        mass=float(mass),
        inertia=((1.0e-4, 0.0, 0.0), (0.0, 1.0e-4, 0.0), (0.0, 0.0, 1.0e-4)),
    )
    mb.add_shape_box(
        bob,
        hx=0.01,
        hy=0.01,
        hz=0.01,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    joint = mb.add_joint_revolute(
        parent=-1,
        child=bob,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, float(length)), q=wp.quat_identity()),
    )
    mb.add_articulation([joint])
    model = mb.finalize()
    model.set_gravity((0.0, 0.0, float(gravity)))
    if init_angle != 0.0:
        model.joint_q.assign(np.array([float(init_angle)], dtype=np.float32))
    return model


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX joint-friction tests run on CUDA only")
class TestFrictionStiction(unittest.TestCase):
    """Friction must hold the joint against an applied torque smaller than μ.
    The pendulum's gravity torque ``τ_g(θ) = m*g*L*sin(θ)`` is the loading;
    if ``μ > τ_g(θ_0)`` the joint must not slip."""

    def test_pendulum_locked_against_gravity(self) -> None:
        mass = 1.0
        length = 0.5
        g = 9.81
        init_angle = 0.3  # ~17 deg from vertical
        max_gravity_torque = mass * g * length * np.sin(init_angle)
        # μ at 2x the loading -- well inside the stiction zone.
        friction = 2.0 * max_gravity_torque

        model = _build_pendulum(mass=mass, length=length, friction=friction, init_angle=init_angle, gravity=-g)
        solver = _solver(model)
        # Half a second is well beyond any transient; if friction were
        # leaking, the pendulum would have swung visibly by then.
        q_traj, qd_traj = _capture_steps(model, solver, n_frames=120)

        q_final = float(q_traj[-1])
        qd_max = float(np.abs(qd_traj).max())
        # Position drift bounded by the friction regularization slip.
        self.assertAlmostEqual(
            q_final,
            init_angle,
            delta=0.02,
            msg=f"pendulum drifted from {init_angle} to {q_final} despite μ = 2 * τ_g_max",
        )
        self.assertLess(
            qd_max,
            0.5,
            msg=f"|qd| peaked at {qd_max} rad/s -- friction is leaking under sub-breakaway load",
        )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX joint-friction tests run on CUDA only")
class TestFrictionFreeSpinDecay(unittest.TestCase):
    """Rotor spinning at known ``ω_0`` with friction ``μ`` and inertia
    ``I`` must decelerate at ``alpha = μ / I`` and halt at
    ``t_stop = I * ω_0 / μ``. Post-stop state must remain stationary."""

    def test_constant_deceleration_to_halt(self) -> None:
        # Pure rotor at the joint origin; I_com == I_joint so the
        # impulse-domain iterate reproduces analytical Coulomb exactly.
        I = 0.25  # kg*m^2
        friction = 0.5  # N*m
        # ω_0 so that the analytical stop time is ~0.4 s.
        t_stop_target = 0.4
        omega_0 = (friction / I) * t_stop_target
        n_frames = int(np.ceil(t_stop_target * _FPS * 1.4))

        model = _build_rotor(inertia=I, friction=friction)
        solver = _solver(model)
        _q_traj, qd_traj = _capture_steps(model, solver, n_frames=n_frames, init_joint_qd=omega_0)

        # Fit deceleration in the strictly-positive-velocity regime.
        # Skip the first frame: the warm-up step takes the body through
        # one full solver step before recording, so qd_traj[0] is
        # already a hair below the seed value.
        slope_fit_idx = np.where(qd_traj > 0.2 * omega_0)[0]
        self.assertGreater(len(slope_fit_idx), 5, msg="not enough frames in the linear-decay regime")
        t_fit = slope_fit_idx * _DT
        qd_fit = qd_traj[slope_fit_idx]
        coeffs = np.polyfit(t_fit, qd_fit, 1)
        measured_alpha = -coeffs[0]
        expected_alpha = friction / I
        rel_err = abs(measured_alpha - expected_alpha) / expected_alpha
        self.assertLess(
            rel_err,
            0.10,
            msg=(
                f"deceleration {measured_alpha:.4f} rad/s² vs analytical μ/I = "
                f"{expected_alpha:.4f} (rel err {rel_err:.2%})"
            ),
        )

        # Post-stop quiescence: |qd| must stay near the slip velocity.
        stop_idx = int(np.ceil(t_stop_target * _FPS)) + 2
        if stop_idx < n_frames:
            tail_peak = float(np.abs(qd_traj[stop_idx:]).max())
            # The slip-velocity bound is :data:`PHOENX_FRICTION_SLIP_VELOCITY`
            # (1e-3 rad/s); allow a generous 10x margin for transient
            # oscillation around the stop.
            self.assertLess(
                tail_peak,
                0.10,
                msg=(
                    f"post-stop |qd| peaked at {tail_peak:.4f} rad/s; "
                    "friction is not holding qd=0 after the body should have halted"
                ),
            )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX joint-friction tests run on CUDA only")
class TestFrictionConstantTorque(unittest.TestCase):
    """Constant applied torque ``τ > μ`` with no gravity. Net torque is
    ``τ - μ * sign(qd)``, yielding constant ``alpha = (τ - μ) / I``."""

    def test_acceleration_under_constant_torque(self) -> None:
        I = 0.25  # kg*m^2
        friction = 0.3
        applied_torque = 1.0  # > friction -- joint slips
        expected_alpha = (applied_torque - friction) / I

        model = _build_rotor(inertia=I, friction=friction)
        solver = _solver(model)

        control = model.control()
        joint_f = np.array([applied_torque], dtype=np.float32)
        control.joint_f.assign(joint_f)

        n_frames = 240  # 1 s simulated
        _q_traj, qd_traj = _capture_steps(model, solver, n_frames=n_frames, control_setup=(control, joint_f))

        # Linear fit of qd vs t after the first ~50 ms transient.
        skip = int(0.05 * _FPS)
        t_fit = np.arange(skip, n_frames) * _DT
        qd_fit = qd_traj[skip:]
        coeffs = np.polyfit(t_fit, qd_fit, 1)
        measured_alpha = coeffs[0]
        rel_err = abs(measured_alpha - expected_alpha) / expected_alpha
        self.assertLess(
            rel_err,
            0.10,
            msg=(
                f"acceleration {measured_alpha:.4f} rad/s² vs analytical (τ-μ)/I = "
                f"{expected_alpha:.4f} (rel err {rel_err:.2%}); friction is "
                "not subtracting μ from the applied torque correctly"
            ),
        )

        # Sanity: final qd matches the analytical extrapolation.
        expected_omega_T = expected_alpha * (n_frames * _DT)
        actual_omega_T = float(qd_traj[-1])
        self.assertAlmostEqual(
            actual_omega_T,
            expected_omega_T,
            delta=0.1 * expected_omega_T,
            msg=f"ω(T) = {actual_omega_T:.4f} rad/s vs (τ-μ)/I * T = {expected_omega_T:.4f}",
        )


@unittest.skipUnless(wp.get_preferred_device().is_cuda, "PhoenX joint-friction tests run on CUDA only")
class TestFrictionAndDriveCompose(unittest.TestCase):
    """Drive and friction saturate independently on the same axial row.
    With a PD drive pushing toward a target and a friction load opposing
    motion, the steady-state position lies where the drive's restoring
    torque just reaches the friction deadband:
    ``|τ_drive| = |ke * (q_target - q_steady)| <= μ``."""

    def test_drive_target_undershoot_inside_friction_deadband(self) -> None:
        I = 0.25
        friction = 0.5  # N*m -- deadband size
        target_pos = 1.0  # rad
        target_ke = 5.0  # N*m/rad; deadband angle = friction/ke = 0.1 rad
        target_kd = 1.0

        model = _build_rotor(
            inertia=I,
            friction=friction,
            target_ke=target_ke,
            target_kd=target_kd,
            target_pos=target_pos,
        )
        solver = _solver(model)
        # 3 s to settle.
        q_traj, qd_traj = _capture_steps(model, solver, n_frames=3 * _FPS)

        # Steady-state in the tail (last 0.5 s).
        tail_q = q_traj[int(-0.5 * _FPS) :]
        tail_qd = qd_traj[int(-0.5 * _FPS) :]
        q_steady = float(np.mean(tail_q))
        qd_steady = float(np.mean(np.abs(tail_qd)))

        # At steady state: drive torque <= friction deadband.
        tau_drive = target_ke * (target_pos - q_steady)
        self.assertLessEqual(
            abs(tau_drive),
            friction * 1.1,  # 10 % tolerance for regularization slip
            msg=(
                f"|τ_drive| = {abs(tau_drive):.4f} > μ = {friction:.4f} "
                f"(q_steady={q_steady:.4f}, target={target_pos:.4f}); "
                "friction is not holding the drive's residual torque"
            ),
        )
        # Velocity quiescent at steady state.
        self.assertLess(
            qd_steady,
            0.05,
            msg=f"mean |qd| = {qd_steady:.4f} in the tail; should be settled inside the friction deadband",
        )
        # And the rotor must have moved (drive did push into the
        # deadband from below).
        self.assertGreater(
            q_steady,
            0.5 * target_pos,
            msg=f"q_steady={q_steady:.4f} -- rotor barely moved; drive may be stuck inside μ at θ=0",
        )


if __name__ == "__main__":
    unittest.main()
