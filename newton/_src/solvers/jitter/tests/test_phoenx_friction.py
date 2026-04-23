# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction tests for :mod:`solver_phoenx`.

Ports the jitter-solver friction checks in :mod:`test_friction` and
:mod:`test_friction_slide`:

* :class:`TestKineticFrictionStopDistance` -- kinetic friction must
  produce the analytic stop distance ``v0^2 / (2 mu g)`` within a
  20% tolerance across a sweep of mu and v0.
* :class:`TestStaticFrictionNoDrift` -- a cube at rest does not
  drift horizontally.
* :class:`TestStaticFrictionThreshold` -- above ``F_applied = mu *
  m * g`` the cube slides, below it stays put.
* :class:`TestKineticSlideDeceleration` -- sliding cube decelerates
  at ``a = mu_k * g`` and stops after ``t = v0 / (mu_k * g)``.

Drives :class:`PhoenXWorld` through the shared
:class:`~newton._src.solvers.jitter.tests.test_phoenx_stacking._PhoenXScene`
harness; CUDA-only same as the jitter originals.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests.test_phoenx_stacking import _PhoenXScene

_G = 9.81


def _analytic_stop_distance(v0: float, mu: float) -> float:
    """Rigid-box stopping distance under kinetic friction.
    ``d = v0^2 / (2 mu g)``, zero applied force.
    """
    if mu <= 0.0:
        return float("inf")
    return (v0 * v0) / (2.0 * mu * _G)


def _analytic_stop_time(v0: float, mu: float) -> float:
    """Time until kinetic friction brings ``v0`` down to zero."""
    if mu <= 0.0:
        return float("inf")
    return v0 / (mu * _G)


# ---------------------------------------------------------------------------
# Kinetic friction: stopping distance
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX friction tests require CUDA")
class TestKineticFrictionStopDistance(unittest.TestCase):
    """A cube with initial +X velocity must decelerate at
    ``a = mu * g`` and stop at ``d = v0^2 / (2 mu g)``.

    Same analytic check as jitter's
    :class:`test_friction.TestKineticFrictionStopDistance`; we pin
    four ``mu`` values and four ``v0`` values and accept 20 %
    relative error (the PGS + Baumgarte normal-bias solver has a
    small warm-up transient that inflates distance by a few %).
    """

    MU_VALUES = (0.1, 0.3, 0.5, 0.8)
    V0_VALUES = (1.0, 3.0, 5.0, 8.0)
    FPS = 120
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 12

    def _run_single_cube(self, mu: float, v0: float) -> tuple[float, float]:
        """Settle a single cube with initial +X velocity ``v0`` under
        friction ``mu`` until it stops. Returns ``(stop_distance,
        residual_speed)``.
        """
        scene = _PhoenXScene(
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            friction=mu,
        )
        scene.add_ground_plane()
        box = scene.add_box(
            position=(0.0, 0.0, 0.5 + 1.0e-3),
            half_extents=(0.5, 0.5, 0.5),
        )
        scene.finalize()
        scene.set_body_velocity(box, (v0, 0.0, 0.0))

        # Run long enough for the cube to fully stop under the
        # worst-case (largest v0 at lowest mu). Matches the jitter
        # test's ``2.5 * t_stop_max + 2 s`` margin.
        total_time = _analytic_stop_time(v0, mu) * 2.5 + 2.0
        num_frames = int(np.ceil(total_time * self.FPS))
        for _ in range(num_frames):
            scene.step()

        pos = scene.body_position(box)
        vel = scene.body_velocity(box)
        return float(pos[0]), float(np.linalg.norm(vel))

    def test_stop_distances_match_analytic(self) -> None:
        for mu in self.MU_VALUES:
            for v0 in self.V0_VALUES:
                with self.subTest(mu=mu, v0=v0):
                    stop_x, v_res = self._run_single_cube(mu, v0)
                    expected = _analytic_stop_distance(v0, mu)
                    rel_err = abs(stop_x - expected) / expected
                    self.assertLess(
                        rel_err,
                        0.20,
                        f"mu={mu} v0={v0}: stop={stop_x:.3f} m vs "
                        f"expected {expected:.3f} m (rel_err {rel_err:+.2%})",
                    )
                    self.assertLess(
                        v_res,
                        0.05,
                        f"mu={mu} v0={v0}: cube still moving at "
                        f"|v|={v_res:.4f} m/s",
                    )


# ---------------------------------------------------------------------------
# Static friction: zero-velocity drift
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX friction tests require CUDA")
class TestStaticFrictionNoDrift(unittest.TestCase):
    """A cube at rest on a flat floor must stay at rest.

    Pins the pyramid-demo symptom that motivated the jitter-side
    audit: resting cubes drifting sideways with no applied load.
    Catches regressions in the tangent-row clamp or a biased normal
    projection.
    """

    def test_cube_at_rest_does_not_drift(self) -> None:
        scene = _PhoenXScene(fps=60, substeps=4, solver_iterations=16, friction=0.5)
        scene.add_ground_plane()
        box = scene.add_box(
            position=(0.0, 0.0, 0.5 + 1.0e-3),
            half_extents=(0.5, 0.5, 0.5),
        )
        scene.finalize()
        for _ in range(180):  # 3 s
            scene.step()
        pos = scene.body_position(box)
        vel = scene.body_velocity(box)
        h_drift = float(np.hypot(pos[0], pos[1]))
        h_speed = float(np.hypot(vel[0], vel[1]))
        self.assertLess(h_drift, 1.0e-3, f"drift={h_drift:.6f} m")
        self.assertLess(h_speed, 1.0e-3, f"speed={h_speed:.6f} m/s")


# ---------------------------------------------------------------------------
# Static friction threshold
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX friction tests require CUDA")
class TestStaticFrictionThreshold(unittest.TestCase):
    """Above ``F = mu * m * g`` the cube slides; below, it stays put.

    Unit-density 1 m^3 cube weighs 1000 kg; applied push is half of
    ``m * g`` (``~4905 N``). mu < 0.5 -> slides; mu > 0.5 -> stays.
    """

    N_FRAMES = 120
    PUSH_RATIO = 0.5

    def _run_mu(self, mu: float) -> tuple[float, float]:
        scene = _PhoenXScene(fps=60, substeps=4, solver_iterations=16, friction=mu)
        scene.add_ground_plane()
        he = 0.5
        # Use density-derived mass so the body has consistent
        # inertia; cube mass = 1000 kg.
        box = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            density=1000.0,
        )
        scene.finalize()
        cube_mass = 1000.0 * (2 * he) ** 3
        push = self.PUSH_RATIO * cube_mass * _G
        # Let the cube settle vertically first so the normal
        # response is established, matching the jitter test.
        for _ in range(10):
            scene.step()
        # Then apply the push every frame.
        for _ in range(self.N_FRAMES):
            scene.apply_body_force(box, force=(push, 0.0, 0.0))
            scene.step()
        p = scene.body_position(box)
        v = scene.body_velocity(box)
        return float(p[0]), float(v[0])

    def test_high_friction_holds(self) -> None:
        """mu=0.7 > push_ratio=0.5 -> cube stays."""
        x, vx = self._run_mu(0.7)
        self.assertLess(abs(x), 0.1, f"mu=0.7 cube drifted to x={x:.4f} m")
        self.assertLess(abs(vx), 0.05, f"mu=0.7 cube has vx={vx:.4f} m/s")

    def test_low_friction_slides(self) -> None:
        """mu=0.1 < push_ratio=0.5 -> cube slides."""
        x, vx = self._run_mu(0.1)
        self.assertGreater(x, 1.0, f"mu=0.1 cube only reached x={x:.4f} m")


# ---------------------------------------------------------------------------
# Kinetic slide deceleration timing
# ---------------------------------------------------------------------------


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX friction tests require CUDA")
class TestKineticSlideDeceleration(unittest.TestCase):
    """Cube given initial +X velocity decelerates at ``a = mu * g``.

    Traces per-frame velocity and fits the early-phase decel rate;
    also measures the total time-to-stop against
    ``t = v0 / (mu * g)``.
    """

    V0 = 5.0
    MU = 0.3
    FPS = 120
    SUBSTEPS = 4
    SOLVER_ITERATIONS = 16

    def test_deceleration_matches_analytic(self) -> None:
        scene = _PhoenXScene(
            fps=self.FPS,
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERATIONS,
            friction=self.MU,
        )
        scene.add_ground_plane()
        he = 0.5
        box = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3), half_extents=(he, he, he)
        )
        scene.finalize()
        scene.set_body_velocity(box, (self.V0, 0.0, 0.0))

        # Seed normal contact (two frames should be enough).
        for _ in range(2):
            scene.step()

        # Measure deceleration across a window well below the full
        # stop time to avoid PGS end-of-slide transients.
        t_stop = _analytic_stop_time(self.V0, self.MU)
        measure_frames = max(3, int(0.3 * t_stop * self.FPS))
        vx_start = float(scene.body_velocity(box)[0])
        for _ in range(measure_frames):
            scene.step()
        vx_end = float(scene.body_velocity(box)[0])
        dt = measure_frames / self.FPS
        measured_decel = (vx_start - vx_end) / dt
        expected_decel = self.MU * _G
        rel_err = abs(measured_decel - expected_decel) / expected_decel
        self.assertLess(
            rel_err,
            0.15,
            f"decel={measured_decel:.3f} m/s^2 vs expected "
            f"{expected_decel:.3f} m/s^2 (rel_err {rel_err:+.2%})",
        )

        # Now run past the analytic stop time and assert the cube
        # has come to rest.
        remaining_frames = int(np.ceil((t_stop * 1.5 - dt) * self.FPS))
        for _ in range(remaining_frames):
            scene.step()
        vx_final = float(scene.body_velocity(box)[0])
        self.assertLess(
            abs(vx_final),
            0.05,
            f"cube did not stop: vx_final={vx_final:.4f} m/s after "
            f"{t_stop * 1.5:.3f} s",
        )


if __name__ == "__main__":
    unittest.main()
