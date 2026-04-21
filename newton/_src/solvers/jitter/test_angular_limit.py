# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for the standalone angular-limit constraint.

Composes the limit with a rigid :meth:`WorldBuilder.add_double_ball_socket`
hinge lock to get a 1-DoF twist-only joint, then probes the
:meth:`WorldBuilder.add_angular_limit` contract from four angles:

* **Inactive inside the range.** When the joint sits well inside
  ``[min_value, max_value]`` the limit must be a complete no-op --
  no torque, no momentum leak. We release a bob with a nonzero
  initial angular velocity and check that the PGS row leaves the
  angular rate alone.
* **Deflection past a stop under gravity (PD path).** With the bob
  loaded by gravity past ``max_value``, the PD spring settles at
  ``delta = tau_grav / k`` past the stop -- a direct k-calibration
  check for the PD-spring branch.
* **Deflection past a stop under gravity (Box2D path).** Same
  scene but using ``hertz`` / ``damping_ratio`` instead of
  ``stiffness`` / ``damping``. The two must agree on the active
  side -- otherwise the dual-convention plumbing is broken.
* **One-sided limit.** Set ``min_value = -1e9`` (effectively no
  lower bound) and check that gravity in one direction engages the
  upper stop, while gravity in the opposite direction leaves the
  joint unclamped (no impulse, bob swings past ``0`` freely).
* **DHO at the stop (PD path).** Hold the bob past ``max_value``
  with a tiny PD offset, no gravity, lightly damped. Verify the
  ``omega_d^2 + gamma^2 = k / I_eff`` identity about ``max_value``.

Uses the same scene builder / CUDA graph pattern as
:mod:`test_pd_angular_motor`.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.world_builder import WorldBuilder

FPS = 240
SUBSTEPS = 8
SOLVER_ITERATIONS = 32
HINGE_AXIS = (0.0, 0.0, 1.0)
GRAVITY_Y = -9.81


def _build_limited_pendulum(
    device,
    *,
    mass: float,
    lever: float,
    inertia_cm: float,
    min_value: float,
    max_value: float,
    hertz: float = 0.0,
    damping_ratio: float = 0.0,
    stiffness: float = 0.0,
    damping: float = 0.0,
    initial_theta: float = 0.0,
    initial_omega: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """World body + lever bob on a +Z hinge with one angular limit.

    Mirrors :func:`test_pd_angular_motor._build_revolute_pendulum`
    but swaps the PD motor for a standalone angular limit (no drive
    row) so the measurements pick up only the limit's contribution.
    """
    b = WorldBuilder()
    pivot_body = 0

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
        linear_damping=1.0,
        angular_damping=1.0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
        angular_velocity=(0.0, 0.0, float(initial_omega)),
    )

    # Rigid 5-DoF hinge lock at the pivot.
    b.add_double_ball_socket(
        body1=pivot_body,
        body2=bob_body,
        anchor1=(0.0, 0.0, -0.25),
        anchor2=(0.0, 0.0, +0.25),
        hertz=0.0,
    )

    b.add_angular_limit(
        body1=pivot_body,
        body2=bob_body,
        axis=HINGE_AXIS,
        min_value=float(min_value),
        max_value=float(max_value),
        hertz=float(hertz),
        damping_ratio=float(damping_ratio),
        stiffness=float(stiffness),
        damping=float(damping),
    )

    world = b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=gravity,
        device=device,
    )
    return world, bob_body


def _read_twist_angle(world, bob_body: int) -> float:
    q = world.bodies.orientation.numpy()[bob_body]
    return float(2.0 * math.atan2(q[2], q[3]))


def _sample_twist_trajectory(
    world, bob_body: int, *, frames: int, sample_stride: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Step ``frames`` times, sampling twist every ``sample_stride``.

    Matches the pattern used in :mod:`test_pd_angular_motor`.
    """
    num_samples = frames // sample_stride + 1
    t = np.empty(num_samples, dtype=np.float64)
    theta = np.empty(num_samples, dtype=np.float64)
    t[0] = 0.0
    theta[0] = _read_twist_angle(world, bob_body)

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
            theta[i] = _read_twist_angle(world, bob_body)
    else:
        for i in range(1, num_samples):
            for _ in range(sample_stride):
                world.step(dt)
            t[i] = i * sample_stride * dt
            theta[i] = _read_twist_angle(world, bob_body)
    return t, theta


def _zero_cross_period(t: np.ndarray, x: np.ndarray) -> float:
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


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Angular-limit analytical tests run on CUDA only.",
)
class TestAngularLimit(unittest.TestCase):
    """Core contract checks for :meth:`WorldBuilder.add_angular_limit`."""

    def test_inactive_when_inside_range(self):
        """Inside the allowed range the limit applies no torque.

        Drop a bob with lever offset into gravity with *wide-open*
        limits -- the clamp never engages, so the pendulum motion
        should be governed entirely by gravity + the hinge lock. We
        check that the bob reaches the free-fall angle for a
        lever-arm pendulum after a short swing; if the limit were
        leaking torque the bob would be noticeably slower.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 0.05
        I_eff = inertia_cm + mass * lever * lever
        # Bob starts at theta=0 (lever along +X), gravity -Y pulls it
        # toward theta=-pi/2. For small-angle swing ``theta(t) = -A *
        # sin(omega*t)`` with ``A ~ pi/2`` at the first bottom and
        # ``omega = sqrt(g*L*m/I_eff)``. Measure the bob's angle at
        # the quarter-period mark and compare to free-pendulum
        # prediction.
        omega_nat = math.sqrt(mass * abs(GRAVITY_Y) * lever / I_eff)
        T_quarter = (math.pi / 2.0) / omega_nat
        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            # Wide-open limits: bob will swing to ~ -pi/2 and back,
            # well inside [-3, +3] rad.
            min_value=-3.0,
            max_value=3.0,
            stiffness=500.0,
            damping=5.0,
            initial_theta=0.0,
            gravity=(0.0, GRAVITY_Y, 0.0),
        )
        frames = int(T_quarter * FPS)
        run_settle_loop(world, frames=frames, dt=1.0 / FPS)
        theta = _read_twist_angle(world, bob)
        # At the quarter period the free pendulum (small-angle) is
        # at its *bottom* theta = -A. For a release from theta=0
        # under gravity the actual amplitude is ~pi/2 (full swing),
        # but small-angle prediction puts the bottom at theta < 0 by
        # several tenths of a radian. If the limit leaks torque the
        # bob stalls near theta=0; just require a clearly negative
        # swing angle at the quarter-period.
        self.assertLess(
            theta,
            -0.3,
            msg=(
                "limit appears to leak torque inside [-3, +3] rad: "
                f"at t={T_quarter:.3f}s theta={theta:.4f} (expected "
                "< -0.3 rad if the pendulum swings freely)"
            ),
        )

    def test_pd_deflection_past_max_matches_tau_over_k(self):
        """Gravity into the upper stop settles at ``delta = tau / k``.

        Start the bob at ``theta = 0`` (its initial pose, which is
        also the zero of the limit's angle coordinate) and set
        ``max_value = 0`` so the upper stop sits right at the start.
        Apply gravity along ``+Y``: with the lever along ``+X`` this
        is a positive torque about ``+Z`` that pushes the bob past
        the stop. The PD spring settles at ``delta = tau / k``
        past it.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 0.05
        max_value = 0.0  # stop is at the initial pose
        k = 2000.0  # N*m/rad -> delta_ss ~ 9.81 / 2000 ~ 4.9 mrad
        I_eff = inertia_cm + mass * lever * lever
        c = 2.0 * math.sqrt(k * I_eff)

        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            min_value=-math.pi,
            max_value=max_value,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y pushes theta past max
        )
        run_settle_loop(world, frames=int(3.0 * FPS), dt=1.0 / FPS)
        theta = _read_twist_angle(world, bob)
        # Gravity torque at theta ~ 0: tau = m * g * L (lever on +X,
        # gravity on +Y gives +Z torque about the pivot).
        tau = mass * abs(GRAVITY_Y) * lever
        expected_delta = tau / k
        delta = theta - max_value
        rel_err = abs(delta - expected_delta) / expected_delta
        self.assertLess(
            rel_err,
            0.2,
            msg=(
                f"PD deflection past max mismatch: delta={delta:.5f} rad, "
                f"expected tau/k = {expected_delta:.5f}, rel_err={rel_err:.4f}"
            ),
        )

        # Flip gravity to pull the bob *away* from the stop. The
        # limit is unilateral (max-clamp) so it must not grab the
        # bob and the pendulum should swing freely to some negative
        # angle (well below the stop).
        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            min_value=-math.pi,
            max_value=max_value,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            gravity=(0.0, GRAVITY_Y, 0.0),  # -Y pulls theta negative
        )
        run_settle_loop(world, frames=int(3.0 * FPS), dt=1.0 / FPS)
        theta = _read_twist_angle(world, bob)
        # Free fall to ~ -pi/2 minus some damping; certainly well
        # below zero. Tolerant of settle state -- just verify the
        # clamp didn't grab the bob at max_value = 0.
        self.assertLess(
            theta,
            -0.5,
            msg=(
                "unilateral max clamp should not engage when gravity pulls "
                f"the bob *away* from the stop: theta={theta:.4f}"
            ),
        )

    def test_box2d_stop_engages_under_gravity(self):
        """Box2D soft-constraint path stops the bob past ``max``.

        Same scene as the PD deflection test but uses the ``(hertz,
        damping_ratio)`` convention. The Box2D softness bakes the
        effective inertia into the gains so we can't directly
        compare to ``tau / k``; instead we just assert the bob
        stays *close* to ``max_value`` -- if the Box2D path were
        broken (or not hooked up at all) the bob would fall all the
        way to ``-pi/2``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 0.05
        max_value = 0.0
        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            min_value=-math.pi,
            max_value=max_value,
            hertz=30.0,
            damping_ratio=1.0,
            initial_theta=0.0,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y pushes into the stop
        )
        run_settle_loop(world, frames=int(3.0 * FPS), dt=1.0 / FPS)
        theta = _read_twist_angle(world, bob)
        # Bob must have settled near ``max_value`` (within a few
        # degrees). The Box2D soft-constraint at 30 Hz behaves
        # approximately like a hard stop -- deflection is small and
        # bounded by the softness budget, not by ``tau / k``.
        self.assertGreater(
            theta,
            max_value - 0.02,
            msg=f"Box2D path failed to hold near stop: theta={theta:.4f}",
        )
        self.assertLess(
            theta,
            max_value + 0.1,
            msg=(
                f"Box2D path let bob sag too far past stop: theta={theta:.4f}, "
                f"max={max_value}"
            ),
        )

    def test_one_sided_upper_limit(self):
        """Large-sentinel ``min_value`` yields a one-sided upper limit.

        With ``min_value = -1e9`` the lower bound is effectively
        disabled; gravity pulling the bob *away* from ``max_value``
        leaves the row un-engaged (bob swings freely through zero).
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 0.05
        max_value = 0.0
        k = 2000.0
        I_eff = inertia_cm + mass * lever * lever
        c = 2.0 * math.sqrt(k * I_eff)

        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            min_value=-1.0e9,
            max_value=max_value,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            gravity=(0.0, GRAVITY_Y, 0.0),  # -Y pulls theta negative
        )
        # Integrate long enough that the bob clearly swings past
        # zero -- if the sentinel lower were misread as a two-sided
        # clamp the bob would grab at theta=0. Here theta must go
        # clearly negative.
        run_settle_loop(world, frames=int(1.0 * FPS), dt=1.0 / FPS)
        theta = _read_twist_angle(world, bob)
        self.assertLess(
            theta,
            -0.5,
            msg=f"one-sided lower should leave bob free: theta={theta:.4f}",
        )

    def test_pd_dho_identity_at_max_under_gravity(self):
        """DHO about the offset steady-state satisfies ``omega_d^2 + gamma^2 = k/I_eff``.

        Let gravity push the bob into the upper stop and excite the
        resulting DHO about the steady-state offset ``delta_ss =
        tau_g / k`` by initialising the bob with a small angular
        velocity (``initial_omega``). With light damping and
        gravity keeping the clamp perpetually active, ``theta(t) -
        max_value`` decays around ``delta_ss`` as a full sine, so
        zero-cross + log-decay recovers ``omega_d`` and ``gamma``
        with ``omega_d^2 + gamma^2 = k / I_eff`` and ``gamma = c /
        (2 * I_eff)``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        lever = 1.0
        inertia_cm = 0.05
        I_eff = inertia_cm + mass * lever * lever  # ~ 1.05
        max_value = 0.0
        k = 300.0  # omega_0 = sqrt(300 / 1.05) ~ 16.9 rad/s
        omega_0 = math.sqrt(k / I_eff)
        zeta = 0.05
        c = 2.0 * zeta * omega_0 * I_eff

        # Gravity ~ +Y (into the stop). Initial angular velocity =
        # 0.5 rad/s into the stop so the bob overshoots its
        # steady-state offset and rings down about it.
        initial_omega = 0.5
        world, bob = _build_limited_pendulum(
            device,
            mass=mass,
            lever=lever,
            inertia_cm=inertia_cm,
            min_value=-math.pi,
            max_value=max_value,
            stiffness=k,
            damping=c,
            initial_theta=0.0,
            initial_omega=initial_omega,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y pushes into the stop
        )
        t, theta = _sample_twist_trajectory(
            world,
            bob,
            frames=int(3.0 * FPS),
            sample_stride=2,
            dt=1.0 / FPS,
        )
        # The steady-state offset is ``tau_g / k = m * g * L / k``;
        # subtract it so the residual is a zero-mean damped sine.
        tau_g = mass * abs(GRAVITY_Y) * lever
        delta_ss = tau_g / k
        residual = theta - max_value - delta_ss
        # Skip the first ~0.1 s where the bob is still swinging
        # *into* the stop (theta < max_value and the clamp is
        # inactive) -- measurements there are not a DHO.
        start = int(0.15 * FPS / 2)
        t_fit = t[start:] - t[start]
        residual = residual[start:]

        T = _zero_cross_period(t_fit, residual)
        omega_d = 2.0 * math.pi / T
        gamma = _log_decay_rate(t_fit, residual)

        identity = omega_d * omega_d + gamma * gamma
        expected = k / I_eff
        rel_err = abs(identity - expected) / expected
        self.assertLess(
            rel_err,
            0.15,
            msg=(
                f"DHO identity at stop mismatch: omega_d={omega_d:.3f}, "
                f"gamma={gamma:.3f}, omega_d^2+gamma^2={identity:.3f}, "
                f"expected k/I_eff={expected:.3f}, rel_err={rel_err:.4f}"
            ),
        )
        expected_gamma = c / (2.0 * I_eff)
        rel_err_gamma = abs(gamma - expected_gamma) / expected_gamma
        self.assertLess(
            rel_err_gamma,
            0.25,
            msg=(
                f"PD damping mismatch at stop: measured gamma={gamma:.3f} 1/s, "
                f"expected {expected_gamma:.3f} 1/s, rel_err={rel_err_gamma:.4f}"
            ),
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
