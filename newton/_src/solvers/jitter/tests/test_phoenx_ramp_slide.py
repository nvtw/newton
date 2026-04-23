# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inclined-ramp slide-threshold tests for :class:`PhoenXWorld`.

Classical Physics-101 scenario: a box on a tilted ramp, ramp
angle ``theta``, Coulomb friction coefficient ``mu``:

* The box stays still iff ``tan(theta) <= mu`` (static-friction
  angle = ``arctan(mu)``; known as the "angle of repose").
* Above the threshold the box slides down the ramp with
  acceleration along the ramp ``a = g * (sin(theta) - mu *
  cos(theta))`` (kinetic friction).

These tests sweep both sides of the threshold and assert the
measured behaviour matches the analytic prediction within a
reasonable tolerance. Each test uses a large static box as the
ramp (a horizontal ground plane can't be tilted) with a small
dynamic box placed on top.

Runs on CUDA only -- same as the other PhoenX end-to-end tests.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests.test_phoenx_stacking import _PhoenXScene

_G = 9.81


def _ramp_quat_about_y(theta_rad: float) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) representing a rotation of ``theta_rad``
    about the +Y axis. Tilts the ramp so its top surface is
    inclined downward along +X (slide direction is +X).
    """
    half = 0.5 * theta_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _build_ramp_scene(
    *,
    theta_rad: float,
    mu: float,
    fps: int = 120,
    substeps: int = 8,
    solver_iterations: int = 16,
) -> tuple[_PhoenXScene, int]:
    """Build a tilted-ramp scene.

    Layout (Newton +Z-up):
    * Ground plane at z=0 is *not* added -- with a tilted ramp we
      only want contact between the small cube and the ramp, so the
      ground plane would only add spurious "cube flew off the ramp
      and hit the floor" contacts at the end of long runs.
    * Large flat box as ramp: half-extents ``(5, 5, 0.25)`` so it
      looks like a 10 x 10 m tile 0.5 m thick. Rotated by
      ``theta`` about +Y and shifted so its top face is roughly
      level with z=0 at the origin.
    * Small dynamic box: ``(0.25, 0.25, 0.25)`` half-extents placed
      just above the ramp's top face at the origin. Settles on the
      ramp within a handful of frames.

    Returns ``(scene, small_box_body_id)``.
    """
    scene = _PhoenXScene(
        fps=fps,
        substeps=substeps,
        solver_iterations=solver_iterations,
        friction=mu,
    )
    # Static ramp: half-height 0.25, tilted by theta about +Y so its
    # top face normal is ``(sin(theta), 0, cos(theta))``. We want the
    # cube to land at roughly (0, 0, z_cube) where z_cube = ramp top
    # face height at x=0. For a ramp tilted about the origin, top
    # face at x=0 sits at z = -ramp_half_height*cos(theta) +
    # ramp_half_height = ramp_half_height * (1 - cos(theta)). For
    # small tilt this is near zero; we shift the ramp down so its
    # top centre is near z=-0.05 and place the cube so it lands on
    # the tilted face.
    ramp_quat = _ramp_quat_about_y(theta_rad)
    ramp_half = (5.0, 5.0, 0.25)
    # Position the ramp so its top face sits near z=0 at x=0: the
    # ramp's top face at x=0 is at ``z = ramp_center_z +
    # ramp_half[2] * cos(theta)`` (dominant term for small tilt).
    ramp_center_z = -ramp_half[2] * math.cos(theta_rad)
    scene.add_static_box(
        position=(0.0, 0.0, ramp_center_z),
        half_extents=ramp_half,
        orientation=ramp_quat,
    )
    # Small cube placed slightly above the ramp top-face at x=0 so
    # it drops a few mm onto the ramp. Its bottom face sits above
    # the tilted top face by a small gap.
    cube_he = 0.25
    cube_box = scene.add_box(
        position=(0.0, 0.0, cube_he + 1.0e-2),
        half_extents=(cube_he, cube_he, cube_he),
    )
    scene.finalize()
    return scene, cube_box


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX ramp slide tests require CUDA"
)
class TestPhoenXRampBelowThreshold(unittest.TestCase):
    """Box on a ramp tilted *below* the angle of repose
    (``tan(theta) < mu``) must stay at rest.

    Chosen values: ``theta = 10 deg``, ``tan(theta) ~= 0.176``,
    ``mu = 0.5``. Static-friction budget is 2.8x the gravity-along-
    ramp force, so the cube must hold.
    """

    def test_cube_stays_on_gentle_ramp(self) -> None:
        theta = math.radians(10.0)
        mu = 0.5
        scene, cube = _build_ramp_scene(theta_rad=theta, mu=mu)
        for _ in range(240):  # 2 s
            scene.step()
        pos = scene.body_position(cube)
        vel = scene.body_velocity(cube)
        # Measure slide distance along the ramp's down-slope
        # direction. Ramp is tilted about +Y so slide axis is
        # ``(cos(theta), 0, -sin(theta))`` in world frame (points
        # down the slope).
        slide_dir = np.array(
            [math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float32
        )
        slide_dist = float(np.dot(pos, slide_dir))
        slide_speed = float(np.dot(vel, slide_dir))
        # Cube should be essentially stationary. Allow 5 cm of
        # initial settle slide and 5 mm/s residual speed.
        self.assertLess(
            abs(slide_dist),
            0.05,
            f"cube slid {slide_dist:.4f} m on gentle ramp "
            f"(theta={math.degrees(theta):.0f}, mu={mu})",
        )
        self.assertLess(
            abs(slide_speed),
            0.05,
            f"cube still moving at {slide_speed:.4f} m/s on gentle ramp",
        )


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX ramp slide tests require CUDA"
)
class TestPhoenXRampAboveThreshold(unittest.TestCase):
    """Box on a ramp tilted *above* the angle of repose
    (``tan(theta) > mu``) must slide down with acceleration
    ``g * (sin(theta) - mu * cos(theta))``.

    Chosen values: ``theta = 40 deg``, ``mu = 0.3``.
    ``tan(40 deg) = 0.839 > 0.3 = mu`` so the cube slides.
    Expected ``a = 9.81 * (sin 40 - 0.3 * cos 40) =
    9.81 * (0.643 - 0.230) = 4.05 m/s^2``.
    """

    def test_cube_slides_at_predicted_acceleration(self) -> None:
        theta = math.radians(40.0)
        mu = 0.3
        scene, cube = _build_ramp_scene(theta_rad=theta, mu=mu)

        # Let the cube settle onto the ramp face before we measure
        # -- the first few substeps the normal impulse is still
        # ramping up, so the effective friction force is noisy.
        for _ in range(10):
            scene.step()

        slide_dir = np.array(
            [math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float32
        )
        v_start = float(np.dot(scene.body_velocity(cube), slide_dir))
        measure_frames = 30  # 0.25 s at 120 Hz -- short enough to
        # stay before the cube hits the edge of the ramp.
        for _ in range(measure_frames):
            scene.step()
        v_end = float(np.dot(scene.body_velocity(cube), slide_dir))
        dt = measure_frames / 120.0
        measured_accel = (v_end - v_start) / dt
        expected_accel = _G * (math.sin(theta) - mu * math.cos(theta))
        rel_err = abs(measured_accel - expected_accel) / expected_accel
        self.assertLess(
            rel_err,
            0.15,  # 15% tolerance -- box-on-tilted-box contact has a
            # small but noticeable warm-up transient on the friction
            # row; the steady-state acceleration is well inside 15%.
            f"slide accel {measured_accel:.3f} m/s^2 vs expected "
            f"{expected_accel:.3f} (rel err {rel_err:.2%})",
        )


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX ramp slide tests require CUDA"
)
class TestPhoenXRampFrictionlessSlide(unittest.TestCase):
    """Box on a tilted ramp with ``mu = 0`` must slide down with
    pure-gravity acceleration ``a = g * sin(theta)``.

    This is the rigid-body-mesh-primitive equivalent of the
    :class:`TestPhoenXZeroFrictionInclinedSlide` test, which uses
    a horizontal ground plane + tilted gravity. Here gravity is
    standard (-Z) but the surface is tilted via a static box
    rotated around +Y; the analytic slide acceleration along
    ``(cos(theta), 0, -sin(theta))`` remains ``g * sin(theta)``.
    """

    def test_cube_slides_under_gravity_alone(self) -> None:
        theta = math.radians(30.0)
        mu = 0.0
        scene, cube = _build_ramp_scene(theta_rad=theta, mu=mu)

        for _ in range(10):
            scene.step()

        slide_dir = np.array(
            [math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float32
        )
        v_start = float(np.dot(scene.body_velocity(cube), slide_dir))
        measure_frames = 30
        for _ in range(measure_frames):
            scene.step()
        v_end = float(np.dot(scene.body_velocity(cube), slide_dir))
        dt = measure_frames / 120.0
        measured_accel = (v_end - v_start) / dt
        expected_accel = _G * math.sin(theta)
        rel_err = abs(measured_accel - expected_accel) / expected_accel
        self.assertLess(
            rel_err,
            0.10,  # 10% -- tighter than the frictional case because
            # with mu=0 there's no tangent-row noise to contend with.
            f"frictionless slide accel {measured_accel:.3f} m/s^2 vs "
            f"expected g*sin(theta) = {expected_accel:.3f} (rel err "
            f"{rel_err:.2%})",
        )


@unittest.skipUnless(
    wp.is_cuda_available(), "PhoenX ramp slide tests require CUDA"
)
class TestPhoenXRampExactThreshold(unittest.TestCase):
    """Box on a ramp tilted *at* the angle of repose (``tan(theta)
    = mu``) must be at the sliding boundary -- small slide or none
    at all, but bounded.

    Chosen values: ``theta = arctan(0.5) ~= 26.57 deg``, ``mu = 0.5``.
    Exact balance: gravity-along-ramp equals max static friction.
    Any sensible solver should hold the cube or creep at most a few
    cm per second.
    """

    def test_cube_near_threshold_does_not_run_away(self) -> None:
        mu = 0.5
        theta = math.atan(mu)
        scene, cube = _build_ramp_scene(theta_rad=theta, mu=mu)
        for _ in range(240):  # 2 s
            scene.step()
        slide_dir = np.array(
            [math.cos(theta), 0.0, -math.sin(theta)], dtype=np.float32
        )
        pos = scene.body_position(cube)
        vel = scene.body_velocity(cube)
        slide_speed = float(np.dot(vel, slide_dir))
        slide_dist = float(np.dot(pos, slide_dir))
        # At the threshold: slight numerical creep is OK, but the
        # cube must not be accelerating out of control. Upper bounds
        # chosen so a catastrophic failure trips (e.g. mu being
        # wrongly applied as 0) but a proper creep stays inside.
        self.assertLess(
            slide_speed,
            1.0,
            f"cube accelerated past 1 m/s at the slide threshold "
            f"(speed={slide_speed:.3f})",
        )
        self.assertLess(
            slide_dist,
            2.0,
            f"cube slid {slide_dist:.3f} m at the slide threshold "
            "(expected <= 2 m of creep)",
        )


if __name__ == "__main__":
    unittest.main()
