# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytical tests for the standalone linear-limit constraint.

Translational twin of :mod:`test_angular_limit`. Composes
:meth:`WorldBuilder.add_double_ball_socket_prismatic` (rigid 5-DoF
slide lock) with :meth:`WorldBuilder.add_linear_limit` to isolate
the limit's scalar row on a single-DoF prismatic slide. The tests
probe the same five contract points as the angular sibling:

* **Inactive inside the range** -- no force.
* **PD deflection past a stop under gravity.** Gravity along the
  slide into the upper stop settles at ``delta = m * g / k``.
* **Box2D stop engages.** ``hertz`` / ``damping_ratio`` hold the
  bob near the stop when gravity pushes into it.
* **One-sided limit.** ``min_value = -1e9`` makes the lower bound
  inert; gravity *away* from the upper stop leaves the bob free.
* **DHO at the stop under gravity.** Excite an under-damped ring
  about the steady-state offset; recover ``omega_d^2 + gamma^2``
  and ``gamma`` from zero-cross / log-decay.
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
SLIDE_AXIS = (0.0, 1.0, 0.0)
GRAVITY_Y = -9.81


def _build_limited_bob(
    device,
    *,
    mass: float,
    min_value: float,
    max_value: float,
    hertz: float = 0.0,
    damping_ratio: float = 0.0,
    stiffness: float = 0.0,
    damping: float = 0.0,
    initial_slide: float = 0.0,
    initial_velocity: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Static pivot + one bob on a +Y prismatic slide with one limit.

    Mirrors :func:`test_pd_linear_motor._build_prismatic_bob` but
    swaps the PD motor for a standalone linear limit (no drive row)
    so measurements pick up only the limit's contribution.
    """
    b = WorldBuilder()
    pivot_body = 0

    bob_body = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0 / mass,
        inverse_inertia=(
            (250.0, 0.0, 0.0),
            (0.0, 250.0, 0.0),
            (0.0, 0.0, 250.0),
        ),
        linear_damping=1.0,
        angular_damping=1.0,
        affected_by_gravity=any(v != 0.0 for v in gravity),
        velocity=(0.0, float(initial_velocity), 0.0),
    )

    # Rigid prismatic lock at the origin -- anchors coincide in the
    # world-frame at finalize (zero initial slide).
    b.add_double_ball_socket_prismatic(
        body1=pivot_body,
        body2=bob_body,
        anchor1=(0.0, -0.25, 0.0),
        anchor2=(0.0, +0.25, 0.0),
        hertz=0.0,
    )

    b.add_linear_limit(
        body1=pivot_body,
        body2=bob_body,
        axis=SLIDE_AXIS,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(0.0, 0.0, 0.0),
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


def _read_slide(world, bob_body: int) -> float:
    return float(world.bodies.position.numpy()[bob_body][1])


def _sample_slide_trajectory(
    world, bob_body: int, *, frames: int, sample_stride: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = frames // sample_stride + 1
    t = np.empty(num_samples, dtype=np.float64)
    s = np.empty(num_samples, dtype=np.float64)
    t[0] = 0.0
    s[0] = _read_slide(world, bob_body)

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
            s[i] = _read_slide(world, bob_body)
    else:
        for i in range(1, num_samples):
            for _ in range(sample_stride):
                world.step(dt)
            t[i] = i * sample_stride * dt
            s[i] = _read_slide(world, bob_body)
    return t, s


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
    "Linear-limit analytical tests run on CUDA only.",
)
class TestLinearLimit(unittest.TestCase):
    """Core contract checks for :meth:`WorldBuilder.add_linear_limit`."""

    def test_inactive_when_inside_range(self):
        """Inside the allowed range the limit applies no force.

        Release a bob with a +Y kick into wide-open limits, no
        gravity. Without gravitational / spring forces the bob
        should glide with constant velocity -- measured slope of
        ``y(t)`` should match the initial velocity.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        v0 = 0.5  # m/s
        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-10.0,
            max_value=10.0,
            stiffness=500.0,
            damping=5.0,
            initial_velocity=v0,
        )
        t, s = _sample_slide_trajectory(
            world,
            bob,
            frames=int(0.4 * FPS),
            sample_stride=2,
            dt=1.0 / FPS,
        )
        slope, _ = np.polyfit(t, s, 1)
        rel_err = abs(slope - v0) / v0
        self.assertLess(
            rel_err,
            0.05,
            msg=(
                f"limit leaked force inside the range: measured slope "
                f"{slope:.4f} m/s vs v0 {v0:.4f}"
            ),
        )

    def test_pd_deflection_past_max_matches_mg_over_k(self):
        """Gravity into the upper stop settles at ``delta = m * g / k``.

        Set ``max_value = 0`` so the upper stop sits right at the
        initial slide. Apply gravity along ``-Y``: pushes the bob
        past ``max`` (since the slide axis and gravity are both
        ``-Y``, ``slide = y_now - y0`` goes *negative*... oh wait,
        we want gravity to drive ``slide`` *positive* to engage a
        max-side clamp). Flip: gravity ``+Y`` pulls ``slide`` to
        positive values; with ``max_value = 0`` the stop engages
        and the spring settles at ``delta_ss = m * g / k``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        max_value = 0.0
        k = 1000.0  # N/m -> delta_ss = 9.81 / 1000 = 9.81 mm
        c = 2.0 * math.sqrt(k * mass)  # critical damping

        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-1.0e3,
            max_value=max_value,
            stiffness=k,
            damping=c,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y drives slide past max
        )
        run_settle_loop(world, frames=int(2.5 * FPS), dt=1.0 / FPS)
        s = _read_slide(world, bob)
        expected_delta = mass * abs(GRAVITY_Y) / k
        rel_err = abs(s - (max_value + expected_delta)) / expected_delta
        self.assertLess(
            rel_err,
            0.1,
            msg=(
                f"PD deflection past max mismatch: s={s:.5f} m, "
                f"expected max + m*g/k = {max_value + expected_delta:.5f}, "
                f"rel_err={rel_err:.4f}"
            ),
        )

        # Flip gravity -> pulls the bob *away* from the upper stop.
        # Unilateral clamp must not engage; bob free-falls below
        # ``max_value`` unopposed.
        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-1.0e3,
            max_value=max_value,
            stiffness=k,
            damping=c,
            gravity=(0.0, GRAVITY_Y, 0.0),
        )
        run_settle_loop(world, frames=int(1.0 * FPS), dt=1.0 / FPS)
        s = _read_slide(world, bob)
        self.assertLess(
            s,
            -1.0,
            msg=(
                "unilateral max clamp should not grab the bob when gravity "
                f"pulls it *away* from the stop: s={s:.4f} m"
            ),
        )

    def test_box2d_stop_engages_under_gravity(self):
        """Box2D soft-constraint path stops the bob past ``max``.

        Uses ``(hertz, damping_ratio)`` instead of ``(stiffness,
        damping)``. The softness at 30 Hz acts like a near-rigid
        stop; deflection is bounded and small, not ``m*g/k``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        max_value = 0.0
        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-1.0e3,
            max_value=max_value,
            hertz=30.0,
            damping_ratio=1.0,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y into the stop
        )
        run_settle_loop(world, frames=int(2.5 * FPS), dt=1.0 / FPS)
        s = _read_slide(world, bob)
        # Deflection past the stop must be small and non-negative.
        # If the Box2D path weren't wired up at all the bob would
        # accelerate past several meters in 2.5 s.
        self.assertGreater(
            s,
            max_value - 0.005,
            msg=f"Box2D path failed to hold near stop: s={s:.4f}",
        )
        self.assertLess(
            s,
            max_value + 0.05,
            msg=(
                f"Box2D path let bob sag too far past stop: s={s:.4f}, "
                f"max={max_value}"
            ),
        )

    def test_one_sided_upper_limit(self):
        """Large-sentinel ``min_value`` yields a one-sided upper limit.

        Gravity pulling the bob *away* from the upper stop must
        leave the row inert; with ``min_value = -1e9`` a misread
        of the sentinel as a two-sided clamp would pin the bob
        near zero. Here it must slide clearly negative.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        max_value = 0.0
        k = 1000.0
        c = 2.0 * math.sqrt(k * mass)
        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-1.0e9,
            max_value=max_value,
            stiffness=k,
            damping=c,
            gravity=(0.0, GRAVITY_Y, 0.0),  # -Y pulls slide negative
        )
        run_settle_loop(world, frames=int(1.0 * FPS), dt=1.0 / FPS)
        s = _read_slide(world, bob)
        self.assertLess(
            s,
            -1.0,
            msg=f"one-sided lower should leave bob free: s={s:.4f}",
        )

    def test_pd_dho_identity_at_max_under_gravity(self):
        """DHO about the offset steady-state: ``omega_d^2 + gamma^2 = k/m``.

        Apply gravity into the stop and kick the bob with a small
        +Y velocity so it overshoots the steady-state offset and
        rings down about it. Since gravity keeps the clamp
        perpetually active, ``s(t) - delta_ss`` is a zero-mean
        damped sine; zero-cross + log-decay recover ``omega_d``
        and ``gamma`` such that ``omega_d^2 + gamma^2 = k / m``
        and ``gamma = c / (2 * m)``.
        """
        device = wp.get_preferred_device()
        mass = 1.0
        max_value = 0.0
        k = 400.0  # omega_0 = sqrt(400/1) = 20 rad/s
        omega_0 = math.sqrt(k / mass)
        zeta = 0.1
        c = 2.0 * zeta * omega_0 * mass

        initial_velocity = 0.5  # m/s into the stop
        world, bob = _build_limited_bob(
            device,
            mass=mass,
            min_value=-1.0e3,
            max_value=max_value,
            stiffness=k,
            damping=c,
            initial_velocity=initial_velocity,
            gravity=(0.0, -GRAVITY_Y, 0.0),  # +Y pushes into stop
        )
        t, s = _sample_slide_trajectory(
            world,
            bob,
            frames=int(2.5 * FPS),
            sample_stride=2,
            dt=1.0 / FPS,
        )
        delta_ss = mass * abs(GRAVITY_Y) / k
        residual = s - max_value - delta_ss

        T = _zero_cross_period(t, residual)
        omega_d = 2.0 * math.pi / T
        gamma = _log_decay_rate(t, residual)

        identity = omega_d * omega_d + gamma * gamma
        expected = k / mass
        rel_err = abs(identity - expected) / expected
        self.assertLess(
            rel_err,
            0.15,
            msg=(
                f"DHO identity at stop mismatch: omega_d={omega_d:.3f}, "
                f"gamma={gamma:.3f}, omega_d^2+gamma^2={identity:.3f}, "
                f"expected k/m={expected:.3f}, rel_err={rel_err:.4f}"
            ),
        )
        expected_gamma = c / (2.0 * mass)
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
