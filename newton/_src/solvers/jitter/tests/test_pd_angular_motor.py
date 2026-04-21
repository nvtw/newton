# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for the PD angular-motor path.

These tests isolate the PD position-target branch of
:func:`~newton._src.solvers.jitter.world_builder.WorldBuilder.add_angular_motor`
from the actuated / limited revolute joint by composing *two* stand-alone
constraints:

  * :meth:`WorldBuilder.add_double_ball_socket` -- rigid 5-DoF hinge lock
    anchored at two world-space points on the axis.
  * :meth:`WorldBuilder.add_angular_motor` -- PD spring-damper driving
    the remaining axial DoF (``stiffness > 0`` / ``damping > 0``).

The motor measures ``target_angle`` *relative to the initial relative
pose* of the two bodies at finalize() time (see
:mod:`constraint_angular_motor`'s ``inv_initial_orientation`` derivation).
Two consequences the tests lean on:

  * **Gain calibration via gravity**. If the bob is placed at the
    configured equilibrium with ``target_angle = 0`` and gravity is
    then turned on, the bob deflects away from the initial pose until
    the PD spring torque ``k * delta`` balances the gravitational
    torque. For a lever orthogonal to gravity the torque at the
    (small) equilibrium is ``tau = m * g * L``, giving
    ``delta_ss = m * g * L / k`` -- a direct k-calibration check.
  * **Damped-harmonic oscillator identity**. With ``target_angle = 0``
    and gravity off the motor pulls the bob back toward its initial
    pose. Releasing the bob with a small initial angular velocity
    excites a clean DHO about that pose:
    ``I_eff * theta_ddot = -c * theta_dot - k * theta`` with
    ``omega_d^2 + gamma^2 = k / I_eff``,
    ``gamma = c / (2 * I_eff)``. We zero-cross the measured twist to
    recover ``omega_d``, log-linear fit the amplitude envelope for
    ``gamma``, and assert the identity to 5% relative error (the same
    budget every other analytical check in this package uses; see
    :mod:`test_actuated_double_ball_socket_physics`).
  * **Velocity fallback untouched**. ``stiffness == 0 and damping
    == 0`` must behave identically to the legacy ``AngularMotor``
    velocity path so that the PD extension is opt-in.

All scenes use CUDA graph capture via
:func:`newton._src.solvers.jitter.tests._test_helpers.run_settle_loop` and a
fixed substep / iteration config so assertions carry a predictable
numerical budget.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests._test_helpers import run_settle_loop
from newton._src.solvers.jitter.world_builder import WorldBuilder

# Solver config. Small substep and plenty of PGS iterations so the PD
# row converges well enough to hit 5% on frequency / decay.
FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32
HINGE_AXIS = (0.0, 0.0, 1.0)
GRAVITY_Y = -9.81


# ----------------------------------------------------------------------
# Scene builders
# ----------------------------------------------------------------------


def _build_revolute_pendulum(
    device,
    *,
    mass: float,
    lever: float,
    inertia_cm: float,
    stiffness: float,
    damping: float,
    initial_theta: float = 0.0,
    initial_omega: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Static pivot (world body) + one lever bob on a +Z hinge.

    The bob is placed along world +X at distance ``lever``, rotated
    about the pivot by ``initial_theta``. Its local inertia tensor is
    isotropic ``diag(inertia_cm)``; with the point mass at offset
    ``lever`` the effective inertia about the hinge is
    ``I_eff = inertia_cm + mass * lever^2`` (parallel axis).

    ``inertia_cm`` must be strictly positive: the kernel-side
    :func:`inertia_world_update_kernel` divides by it, so a "point
    mass" fudge like ``1e-6`` inverts to ``1e6`` and explodes the
    sim. Pick a physically sensible value, e.g. the inertia of a
    solid sphere (``(2/5) * m * r^2``) or cube
    (``(1/6) * m * a^2``).

    The motor's ``target_angle`` is ``0`` -- i.e. the motor holds the
    bob at its *initial* relative orientation at finalize() time
    (which is ``initial_theta`` in world space, because
    ``inv_initial_orientation`` snapshots that pose). This is the
    natural configuration for the two analytical checks:

      * Gravity pulls the bob *away* from the initial pose; the
        equilibrium offset ``delta_eq`` balances the PD spring
        torque ``k * delta_eq`` against the gravitational torque.
      * Releasing the bob with ``initial_omega != 0`` excites a DHO
        about the initial pose (which is also the target).

    Returns ``(world, bob_body_index)``.
    """
    b = WorldBuilder()

    # Body 0 is the built-in static world body; use it directly as
    # the pivot so we don't pay for a second static body.
    pivot_body = 0

    if inertia_cm <= 0.0:
        raise ValueError("inertia_cm must be > 0 (kernel divides by it)")
    x = lever * math.cos(initial_theta)
    y = lever * math.sin(initial_theta)
    q0 = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), float(initial_theta))
    inv_i = 1.0 / inertia_cm
    bob_body = b.add_dynamic_body(
        position=(x, y, 0.0),
        orientation=(q0[0], q0[1], q0[2], q0[3]),
        inverse_mass=1.0 / mass,
        inverse_inertia=(
            (inv_i, 0.0, 0.0),
            (0.0, inv_i, 0.0),
            (0.0, 0.0, inv_i),
        ),
        # NOTE: Jitter's linear / angular "damping" knobs are
        # *per-substep velocity multipliers* (1.0 == no damping;
        # 0.0 == instantaneous zero-out). Leaving them at 1.0 means
        # the PD damping term is the only source of velocity decay,
        # which is what the DHO identity needs.
        linear_damping=1.0,
        angular_damping=1.0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
        angular_velocity=(0.0, 0.0, float(initial_omega)),
    )

    # 5-DoF hinge lock: two anchors sit on the hinge axis at
    # z = +/- 0.25 m at the pivot's XY origin. Rigid (hertz=0) so the
    # only free DoF left is the hinge twist.
    b.add_double_ball_socket(
        body1=pivot_body,
        body2=bob_body,
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        hertz=0.0,
    )

    # PD angular motor, target = 0 (= current relative orientation at
    # finalize). ``max_force`` = 1e6 is effectively uncapped for our
    # 1 kg lever scenes.
    b.add_angular_motor(
        body1=pivot_body,
        body2=bob_body,
        axis=HINGE_AXIS,
        target_angle=0.0,
        stiffness=stiffness,
        damping=damping,
        max_force=1.0e6,
    )

    world = b.finalize(
        enable_all_constraints=True,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=gravity,
        device=device,
    )
    return world, bob_body


# ----------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------


def _read_twist_angle(world, bob_body: int) -> float:
    """Return the bob's current twist angle about +Z, in radians.

    For this test family the bob's orientation is always ``q =
    (0, 0, sin(theta/2), cos(theta/2))`` (rotation about +Z), so
    ``theta = 2 * atan2(q.z, q.w)``. Same identity as
    ``quat::ExtractRotationAngle(q, +Z)`` on the kernel side; kept on
    the host so we don't launch a kernel per sample.
    """
    q = world.bodies.orientation.numpy()[bob_body]
    return float(2.0 * math.atan2(q[2], q[3]))


def _sample_twist_trajectory(
    world, bob_body: int, *, frames: int, sample_stride: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Step ``frames`` times, sampling the bob's twist every ``sample_stride``.

    Records a single ``world.step(dt)`` into a CUDA graph on the first
    call and replays it via ``wp.capture_launch`` between samples. The
    per-stride overhead is then dominated by the one ``orientation.numpy()``
    sync + small host-side math instead of hundreds of small kernel
    launches, so a 1.5 s / 360 frame trajectory finishes in tens of ms
    of wall time instead of seconds.

    We deliberately *do not* route through :func:`run_settle_loop`
    here: that helper captures ``sample_stride`` frames per call, and
    since ``sample_stride`` is small for DHO sampling (we want fine
    temporal resolution) it always falls back to eager stepping.
    Capturing a single frame and replaying is strictly better for this
    access pattern.

    Falls back to eager stepping on CPU (``wp.ScopedCapture`` is a
    no-op there).
    """
    num_samples = frames // sample_stride + 1
    t = np.empty(num_samples, dtype=np.float64)
    theta = np.empty(num_samples, dtype=np.float64)
    t[0] = 0.0
    theta[0] = _read_twist_angle(world, bob_body)

    device = wp.get_device()
    if device.is_cuda:
        # Warm-up step outside the capture so all kernels are loaded
        # and any lazy scratch allocations settle before we record.
        world.step(dt)
        with wp.ScopedCapture(device=device) as capture:
            world.step(dt)
        graph = capture.graph

        # First sample already consumed one warm-up + one captured
        # step, so the first stride runs ``sample_stride - 2`` replays
        # (clamped at 0 for tiny strides).
        for i in range(1, num_samples):
            replays = sample_stride if i > 1 else max(0, sample_stride - 2)
            for _ in range(replays):
                wp.capture_launch(graph)
            t[i] = i * sample_stride * dt
            theta[i] = _read_twist_angle(world, bob_body)
    else:
        for i in range(1, num_samples):
            for _ in range(sample_stride):
                world.step(dt)
            t[i] = i * sample_stride * dt
            theta[i] = _read_twist_angle(world, bob_body)
    return t, theta


def _zero_cross_period(t: np.ndarray, x: np.ndarray) -> float:
    """Average period ``T`` (s) estimated from zero crossings of ``x``.

    Uses linear interpolation around each sign change. Requires at
    least 3 zero crossings (1.5 periods) for a stable average; raises
    :class:`RuntimeError` otherwise.
    """
    zeros = []
    for i in range(1, len(x)):
        a, b = x[i - 1], x[i]
        if a == 0.0:
            zeros.append(t[i - 1])
        elif a * b < 0.0:
            zeros.append(t[i - 1] - a * (t[i] - t[i - 1]) / (b - a))
    if len(zeros) < 3:
        raise RuntimeError(
            f"need >= 3 zero crossings to estimate period; found {len(zeros)}"
        )
    half_periods = np.diff(zeros)
    return float(2.0 * np.mean(half_periods))


def _log_decay_rate(t: np.ndarray, x: np.ndarray) -> float:
    """Exponential decay rate ``gamma`` (1/s) of the |x| envelope.

    For ``x(t) = A * exp(-gamma * t) * cos(omega_d * t + phi)`` the
    envelope at local extrema traces ``A * exp(-gamma * t)``. We pick
    local maxima of ``|x|`` and least-squares fit
    ``log(|x_max|) = log(A) - gamma * t``.
    """
    absx = np.abs(x)
    maxima_idx = []
    for i in range(1, len(absx) - 1):
        if absx[i] > absx[i - 1] and absx[i] >= absx[i + 1]:
            maxima_idx.append(i)
    if len(maxima_idx) < 2:
        raise RuntimeError(
            f"need >= 2 local maxima to fit decay rate; found {len(maxima_idx)}"
        )
    tm = t[maxima_idx]
    lm = np.log(absx[maxima_idx])
    slope, _intercept = np.polyfit(tm, lm, 1)
    return float(-slope)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PD-motor analytical tests run on CUDA only (graph capture is required to "
    "keep the 5-10 s simulated settle times tractable).",
)
class TestPDAngularMotor(unittest.TestCase):
    """Analytical behaviour of the PD angular-motor path."""

    def test_steady_state_spring_deflection_matches_tau_over_k(self):
        """Gravity on a lever gives ``delta_eq ~= m*g*L / k``.

        Bob mass 1 kg on a 1 m lever starting horizontal (``theta0 =
        0`` along +X, gravity along -Y), so the initial gravitational
        torque about the +Z hinge is ``tau0 = -m * g * L`` (negative
        because gravity pulls the bob toward -y, which about +z is a
        negative rotation). The PD target is the initial pose, so
        the bob settles at ``delta_eq`` such that
        ``k * delta_eq = tau0``, giving ``delta_eq = -m*g*L/k``.
        For small ``delta_eq`` the torque stays nearly ``-m*g*L`` so
        the small-angle form is accurate to better than 1%.

        Critical damping gets the bob to equilibrium inside a 2.5 s
        window. 5% relative error is the project-wide tolerance for
        PGS-settled steady states.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        # Solid sphere radius 0.1 m -> I_cm = (2/5) * m * r^2 = 4e-3.
        # Keeps inverse inertia bounded (1/I_cm = 250) so numerical
        # conditioning is fine.
        inertia_cm = 4.0e-3
        I_eff = inertia_cm + mass * lever * lever
        k = 1000.0  # N*m/rad; small-angle equilibrium is ~9.81 mrad
        c = 2.0 * math.sqrt(k * I_eff)

        world, bob = _build_revolute_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            initial_omega=0.0,
            gravity=(0.0, GRAVITY_Y, 0.0),
        )

        # 2.5 s settle; critically-damped PD converges well inside 1 s
        # at these gains.
        run_settle_loop(world, frames=int(2.5 * FPS), dt=1.0 / FPS)

        delta_ss = _read_twist_angle(world, bob)  # initial_theta was 0
        tau0 = mass * GRAVITY_Y * lever  # -9.81 N*m (negative about +z)
        expected = tau0 / k
        rel_err = abs(delta_ss - expected) / abs(expected)
        self.assertLess(
            rel_err,
            0.05,
            msg=(
                f"PD steady state mismatch: delta_ss={delta_ss:.6f} rad, "
                f"expected {expected:.6f} rad (tau0/k), rel_err={rel_err:.4f}"
            ),
        )

    def test_damped_harmonic_identity_omega_d_and_gamma(self):
        """Measured ``omega_d^2 + gamma^2`` matches ``k / I_eff``.

        Release the bob from the PD target with a small initial
        angular velocity, first undamped (to fit ``omega_0``), then
        lightly damped (to fit ``gamma`` and cross-check the DHO
        identity). Gravity is off so the PD spring alone supplies the
        restoring torque; equilibrium is at ``theta = 0`` (the initial
        / target pose), so zero-crossings of ``theta(t)`` directly
        give the period.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        # Solid sphere radius 0.1 m -> I_cm = (2/5) * m * r^2.
        inertia_cm = 4.0e-3
        I_eff = inertia_cm + mass * lever * lever
        k = 400.0  # omega_0 = sqrt(k / I_eff) ~ 20 rad/s ~ 3.18 Hz
        omega0_kick = 0.5  # rad/s initial kick; peak amplitude ~ 0.025 rad

        # --- undamped frequency read ---------------------------------
        world, bob = _build_revolute_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            stiffness=k,
            damping=0.0,
            initial_theta=0.0,
            initial_omega=omega0_kick,
            gravity=(0.0, 0.0, 0.0),
        )
        # Expected T = 2*pi/omega_0 ~ 0.314 s. 1.5 s captures ~5
        # periods -> ~10 zero crossings. sample_stride=2 at FPS=240
        # yields an 8.33 ms sample period, Nyquist ~ 60 Hz >> 3.18 Hz.
        t_u, theta_u = _sample_twist_trajectory(
            world, bob, frames=int(1.5 * FPS), sample_stride=2, dt=1.0 / FPS
        )
        T = _zero_cross_period(t_u, theta_u)
        omega_d_undamped = 2.0 * math.pi / T
        omega_0_expected = math.sqrt(k / I_eff)
        rel_err_omega = abs(omega_d_undamped - omega_0_expected) / omega_0_expected
        self.assertLess(
            rel_err_omega,
            0.05,
            msg=(
                f"undamped frequency mismatch: measured {omega_d_undamped:.4f} rad/s, "
                f"expected omega_0 = sqrt(k/I_eff) = {omega_0_expected:.4f}, "
                f"rel_err={rel_err_omega:.4f}"
            ),
        )

        # --- under-damped DHO identity -------------------------------
        zeta = 0.1  # lightly damped so we get several oscillations
        c = 2.0 * zeta * omega_0_expected * I_eff
        world, bob = _build_revolute_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            initial_omega=omega0_kick,
            gravity=(0.0, 0.0, 0.0),
        )
        t_d, theta_d = _sample_twist_trajectory(
            world, bob, frames=int(2.5 * FPS), sample_stride=2, dt=1.0 / FPS
        )
        T_d = _zero_cross_period(t_d, theta_d)
        omega_d = 2.0 * math.pi / T_d
        gamma = _log_decay_rate(t_d, theta_d)

        omega_id = omega_d * omega_d + gamma * gamma
        expected = k / I_eff
        rel_err = abs(omega_id - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            msg=(
                f"DHO identity mismatch: omega_d={omega_d:.4f}, gamma={gamma:.4f}, "
                f"omega_d^2+gamma^2={omega_id:.4f}, expected k/I_eff={expected:.4f}, "
                f"rel_err={rel_err:.4f}"
            ),
        )
        expected_gamma = c / (2.0 * I_eff)
        rel_err_gamma = abs(gamma - expected_gamma) / expected_gamma
        self.assertLess(
            rel_err_gamma,
            0.15,
            msg=(
                f"PD damping mismatch: measured gamma={gamma:.4f} 1/s, "
                f"expected {expected_gamma:.4f} 1/s, rel_err={rel_err_gamma:.4f}"
            ),
        )

    def test_velocity_fallback_preserved(self):
        """``stiffness=0, damping=0`` must still drive to ``target_velocity``.

        Narrow regression check: the PD branch is gated on
        ``stiffness > 0 || damping > 0``, so the legacy velocity-
        target path should be bit-for-bit unchanged. Re-uses the
        "two free cubes + lone motor" scene from
        :mod:`test_angular_motor` to keep the signal isolated to the
        motor code path.
        """
        device = wp.get_preferred_device()
        b = WorldBuilder()
        b.add_dynamic_body(
            position=(0.0, 0.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            affected_by_gravity=False,
        )
        b.add_dynamic_body(
            position=(2.0, 0.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            affected_by_gravity=False,
        )
        target = 2.0
        b.add_angular_motor(
            body1=1,
            body2=2,
            axis=HINGE_AXIS,
            target_velocity=target,
            max_force=50.0,
            # stiffness / damping default to 0 -> velocity path.
        )
        world = b.finalize(
        enable_all_constraints=True,
            substeps=SUBSTEPS,
            solver_iterations=SOLVER_ITERATIONS,
            device=device,
        )
        run_settle_loop(world, frames=int(2.0 * FPS), dt=1.0 / FPS)

        omegas = world.bodies.angular_velocity.numpy()
        rel_axial = omegas[2, 2] - omegas[1, 2]
        self.assertAlmostEqual(
            rel_axial,
            target,
            delta=0.1,
            msg=(
                "velocity fallback broken: rel axial velocity "
                f"{rel_axial:.4f} != target {target}"
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
