# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for the PD linear-motor path.

Translational twin of :mod:`test_pd_angular_motor`. These tests isolate
the PD position-target branch of
:func:`~newton._src.solvers.jitter.world_builder.WorldBuilder.add_linear_motor`
from any joint-specific logic by composing *two* stand-alone constraints:

  * :meth:`WorldBuilder.add_double_ball_socket_prismatic` -- rigid
    5-DoF slide lock anchored at two world-space points on the axis
    (rotation fully locked, one translation along the axis free).
  * :meth:`WorldBuilder.add_linear_motor` -- PD spring-damper driving
    the remaining slide DoF (``stiffness > 0`` / ``damping > 0``).

The motor measures ``target_position`` relative to the initial offset
``hat_n . (anchor2 - anchor1)`` at finalize() time (see
:mod:`constraint_linear_motor`'s ``rest_offset`` derivation). The three
identities the tests lean on:

  * **Gain calibration via gravity**. Align the slide axis with gravity
    and put a bob on it; with ``target_position = 0`` the bob settles
    at ``delta_ss`` where ``k * delta_ss = m * g`` -- directly checks
    the PD stiffness.
  * **Damped-harmonic oscillator identity**. With gravity off and a
    small initial velocity kick the slide oscillates about
    ``s = 0`` as ``m * s_ddot = -c * s_dot - k * s``, so
    ``omega_d^2 + gamma^2 = k / m`` and ``gamma = c / (2 * m)`` (the
    *effective* mass for the motor's scalar row is
    ``1 / (1/m1 + 1/m2)`` -- the twin of the body's point mass when
    body 1 is the static world body). We zero-cross ``s(t)`` to
    recover ``omega_d``, log-linear fit the amplitude envelope for
    ``gamma``, and assert the identity to 5% relative error.
  * **Velocity fallback untouched**. ``stiffness == 0 and damping
    == 0`` must behave identically to the legacy Jitter2
    ``LinearMotor`` velocity path; regression check so the PD
    extension is opt-in.

All scenes use CUDA graph capture via
:func:`newton._src.solvers.jitter._test_helpers.run_settle_loop` and a
fixed substep / iteration config so assertions carry a predictable
numerical budget.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.world_builder import WorldBuilder

# Solver config. Same defaults as the PD angular test family so both
# test files share a budget; see their docstring for rationale.
FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32

# Prismatic joint axis: +Y so the default scene's gravity (also -Y)
# lines up with the slide.
SLIDE_AXIS = (0.0, 1.0, 0.0)
GRAVITY_Y = -9.81


# ----------------------------------------------------------------------
# Scene builders
# ----------------------------------------------------------------------


def _build_prismatic_bob(
    device,
    *,
    mass: float,
    stiffness: float,
    damping: float,
    initial_slide: float = 0.0,
    initial_velocity: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Static pivot (world body) + one bob on a +Y prismatic slide.

    The bob is placed at ``y = initial_slide`` and attached to the
    world body by a rigid (``hertz = 0``) prismatic DBS lock and a
    stand-alone linear motor with the given PD gains. The motor's
    ``target_position`` is 0 -- i.e. the PD holds the bob at its
    *initial relative slide* at finalize() time, which for a static
    world body is just ``y = initial_slide``.

    ``initial_velocity`` is applied to the bob's world-frame linear
    velocity along ``+Y`` so the DHO test can release the bob with a
    small kick. The prismatic lock removes all transverse motion, so
    the kick drains entirely into the free axial DoF.

    The bob's orientation is the identity (no rotation about or off
    the slide axis); inverse_inertia is non-zero but physically
    irrelevant because the prismatic lock freezes all 3 rotations.
    We still pick a modest value so the inertia-world update stays
    well-conditioned (see ``_build_revolute_pendulum`` in the PD
    angular test for the same rationale).

    Returns ``(world, bob_body_index)``.
    """
    b = WorldBuilder()
    pivot_body = 0  # static world body

    bob_body = b.add_dynamic_body(
        position=(0.0, float(initial_slide), 0.0),
        inverse_mass=1.0 / mass,
        # Solid sphere radius 0.1 m -> I_cm = (2/5) * m * r^2 = 4e-3.
        # Keeps the inertia-world kernel's inverse bounded.
        inverse_inertia=(
            (250.0, 0.0, 0.0),
            (0.0, 250.0, 0.0),
            (0.0, 0.0, 250.0),
        ),
        # Per-substep velocity multipliers -- see _build_revolute_pendulum
        # for the 1.0 = "no damping" semantics.
        linear_damping=1.0,
        angular_damping=1.0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
        velocity=(0.0, float(initial_velocity), 0.0),
    )

    # Rigid prismatic lock: anchors at y = +-0.25 m on the slide axis.
    # Both anchors coincide in body1 (static world) and body2 at init
    # time (they share the world axis through the bob's COM), which
    # is exactly the "zero initial drift" configuration the DBS
    # prismatic expects.
    b.add_double_ball_socket_prismatic(
        body1=pivot_body,
        body2=bob_body,
        anchor1=(0.0, float(initial_slide) - 0.25, 0.0),
        anchor2=(0.0, float(initial_slide) + 0.25, 0.0),
        hertz=0.0,
    )

    # PD linear motor on the same slide axis. ``target_position = 0``
    # means "hold the initial offset" (same convention as the PD
    # angular motor's ``target_angle = 0``). ``max_force = 1e6`` is
    # effectively uncapped for the 1 kg bob scenes below.
    b.add_linear_motor(
        body1=pivot_body,
        body2=bob_body,
        axis=SLIDE_AXIS,
        anchor1=(0.0, float(initial_slide), 0.0),
        anchor2=(0.0, float(initial_slide), 0.0),
        target_position=0.0,
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


def _read_slide(world, bob_body: int, *, origin_y: float) -> float:
    """Return the bob's current slide relative to the initial pose [m].

    For this test family the bob only moves along +Y (prismatic +Y
    slide), so ``slide = y_now - y0`` directly measures displacement
    from the PD target.
    """
    y = float(world.bodies.position.numpy()[bob_body][1])
    return y - origin_y


def _sample_slide_trajectory(
    world,
    bob_body: int,
    *,
    origin_y: float,
    frames: int,
    sample_stride: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Step ``frames`` times, sampling the bob's slide every ``sample_stride``.

    Same CUDA-graph capture pattern as
    :func:`test_pd_angular_motor._sample_twist_trajectory`; see that
    function's docstring for the rationale behind the 1-step capture.
    Falls back to eager stepping on CPU.
    """
    num_samples = frames // sample_stride + 1
    t = np.empty(num_samples, dtype=np.float64)
    s = np.empty(num_samples, dtype=np.float64)
    t[0] = 0.0
    s[0] = _read_slide(world, bob_body, origin_y=origin_y)

    device = wp.get_device()
    if device.is_cuda:
        world.step(dt)
        with wp.ScopedCapture(device=device) as capture:
            world.step(dt)
        graph = capture.graph
        for i in range(1, num_samples):
            replays = sample_stride if i > 1 else max(0, sample_stride - 2)
            for _ in range(replays):
                wp.capture_launch(graph)
            t[i] = i * sample_stride * dt
            s[i] = _read_slide(world, bob_body, origin_y=origin_y)
    else:
        for i in range(1, num_samples):
            for _ in range(sample_stride):
                world.step(dt)
            t[i] = i * sample_stride * dt
            s[i] = _read_slide(world, bob_body, origin_y=origin_y)
    return t, s


def _zero_cross_period(t: np.ndarray, x: np.ndarray) -> float:
    """Average period ``T`` (s) from linear-interpolated zero crossings.

    See :func:`test_pd_angular_motor._zero_cross_period` -- same
    function, duplicated here so this test module is self-contained
    and doesn't depend on a sibling test file.
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
    """Exponential decay rate ``gamma`` (1/s) of ``|x|`` envelope.

    See :func:`test_pd_angular_motor._log_decay_rate` for the method.
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
class TestPDLinearMotor(unittest.TestCase):
    """Analytical behaviour of the PD linear-motor path."""

    def test_steady_state_spring_deflection_matches_mg_over_k(self):
        """Gravity along the slide axis gives ``delta_ss ~= m*g / k``.

        Bob of 1 kg on a +Y slide, gravity along -Y, PD target at the
        initial pose. The PD spring balances gravity at
        ``k * delta_ss = m * g`` giving ``delta_ss = m * g / k`` (with
        the correct sign: gravity pulls the bob to negative Y, so
        ``delta_ss`` is negative). Critical damping gets the bob to
        equilibrium inside a 1 s window; 5% relative error is the
        project-wide tolerance for PGS-settled steady states.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        k = 1000.0  # steady-state deflection ~= 9.81 mm
        c = 2.0 * math.sqrt(k * mass)  # critical damping

        origin_y = 0.5  # keep bob above ground, doesn't affect the math
        world, bob = _build_prismatic_bob(
            device,
            mass=mass,
            stiffness=k,
            damping=c,
            initial_slide=origin_y,
            initial_velocity=0.0,
            gravity=(0.0, GRAVITY_Y, 0.0),
        )

        # 2.5 s settle (same budget as the PD angular steady-state test).
        run_settle_loop(world, frames=int(2.5 * FPS), dt=1.0 / FPS)

        delta_ss = _read_slide(world, bob, origin_y=origin_y)
        expected = mass * GRAVITY_Y / k  # negative (slide = y_now - y0)
        rel_err = abs(delta_ss - expected) / abs(expected)
        self.assertLess(
            rel_err,
            0.05,
            msg=(
                f"PD steady state mismatch: delta_ss={delta_ss:.6f} m, "
                f"expected {expected:.6f} m (m*g/k), rel_err={rel_err:.4f}"
            ),
        )

    def test_damped_harmonic_identity_omega_d_and_gamma(self):
        """Measured ``omega_d^2 + gamma^2`` matches ``k / m``.

        Release the bob from the PD target with a small linear
        velocity kick, first undamped (to fit ``omega_0``), then
        lightly damped (to fit ``gamma`` and cross-check the DHO
        identity). Gravity is off so the PD spring alone supplies the
        restoring force; equilibrium is at ``s = 0``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        k = 400.0  # omega_0 = sqrt(k / m) = 20 rad/s ~ 3.18 Hz
        v0 = 0.5  # m/s initial kick; peak amplitude ~ 0.025 m

        # --- undamped frequency read ---------------------------------
        world, bob = _build_prismatic_bob(
            device,
            mass=mass,
            stiffness=k,
            damping=0.0,
            initial_slide=0.0,
            initial_velocity=v0,
            gravity=(0.0, 0.0, 0.0),
        )
        t_u, s_u = _sample_slide_trajectory(
            world,
            bob,
            origin_y=0.0,
            frames=int(1.5 * FPS),
            sample_stride=2,
            dt=1.0 / FPS,
        )
        T = _zero_cross_period(t_u, s_u)
        omega_d_undamped = 2.0 * math.pi / T
        omega_0_expected = math.sqrt(k / mass)
        rel_err_omega = abs(omega_d_undamped - omega_0_expected) / omega_0_expected
        self.assertLess(
            rel_err_omega,
            0.05,
            msg=(
                f"undamped frequency mismatch: measured {omega_d_undamped:.4f} rad/s, "
                f"expected omega_0 = sqrt(k/m) = {omega_0_expected:.4f}, "
                f"rel_err={rel_err_omega:.4f}"
            ),
        )

        # --- under-damped DHO identity -------------------------------
        zeta = 0.1
        c = 2.0 * zeta * omega_0_expected * mass
        world, bob = _build_prismatic_bob(
            device,
            mass=mass,
            stiffness=k,
            damping=c,
            initial_slide=0.0,
            initial_velocity=v0,
            gravity=(0.0, 0.0, 0.0),
        )
        t_d, s_d = _sample_slide_trajectory(
            world,
            bob,
            origin_y=0.0,
            frames=int(2.5 * FPS),
            sample_stride=2,
            dt=1.0 / FPS,
        )
        T_d = _zero_cross_period(t_d, s_d)
        omega_d = 2.0 * math.pi / T_d
        gamma = _log_decay_rate(t_d, s_d)

        omega_id = omega_d * omega_d + gamma * gamma
        expected = k / mass
        rel_err = abs(omega_id - expected) / expected
        self.assertLess(
            rel_err,
            0.05,
            msg=(
                f"DHO identity mismatch: omega_d={omega_d:.4f}, gamma={gamma:.4f}, "
                f"omega_d^2+gamma^2={omega_id:.4f}, expected k/m={expected:.4f}, "
                f"rel_err={rel_err:.4f}"
            ),
        )
        expected_gamma = c / (2.0 * mass)
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
        """``stiffness=0, damping=0`` still drives to ``target_velocity``.

        Two free bodies on a common slide axis with a stand-alone
        linear motor between them (no prismatic lock so the only
        forces are along ``SLIDE_AXIS``). The PD gains are both zero,
        so the motor must revert to the legacy Jitter2 velocity path
        and drive the rel. linear velocity along +Y to ``target``.
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
            position=(0.0, 2.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            affected_by_gravity=False,
        )
        target = 2.0
        b.add_linear_motor(
            body1=1,
            body2=2,
            axis=SLIDE_AXIS,
            anchor1=(0.0, 0.0, 0.0),
            anchor2=(0.0, 2.0, 0.0),
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

        velocities = world.bodies.velocity.numpy()
        rel_axial = velocities[2, 1] - velocities[1, 1]
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
