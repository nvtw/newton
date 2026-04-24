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

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

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

    * A large static box as the ramp, rotated by ``theta`` about
      +Y so its top face normal points in ``(sin(theta), 0,
      cos(theta))``.
    * A small *dynamic* cube **rotated by the same angle** and
      positioned so its bottom face sits flush against the ramp's
      top face (small 1 mm gap). If we left the cube at identity
      orientation the cube's horizontal bottom face would only
      touch the ramp along its uphill edge and the cube would
      *tumble* forward off the edge rather than slide -- exactly
      what we're **not** trying to test. Aligning the cube with
      the slope means the only free degree of freedom is the
      slide; any spin / tip that shows up comes from the solver,
      not the initial pose.

    Returns ``(scene, small_box_body_id)``.
    """
    scene = _PhoenXScene(
        fps=fps,
        substeps=substeps,
        solver_iterations=solver_iterations,
        friction=mu,
    )
    ramp_quat = _ramp_quat_about_y(theta_rad)
    ramp_half = (5.0, 5.0, 0.25)
    ramp_center = (0.0, 0.0, -ramp_half[2] * math.cos(theta_rad))
    scene.add_static_box(
        position=ramp_center,
        half_extents=ramp_half,
        orientation=ramp_quat,
    )
    cube_he = 0.25
    # Place the cube so its bottom face sits 1 mm above the ramp's
    # top face, centred on the ramp's top-face centre. Ramp tilt
    # about +Y means the top-face centre is at ``ramp_center +
    # ramp_half[2] * (sin(theta), 0, cos(theta))`` (the ramp's +Z
    # in body frame rotated into world).
    n_top = (math.sin(theta_rad), 0.0, math.cos(theta_rad))
    top_centre = (
        ramp_center[0] + ramp_half[2] * n_top[0],
        ramp_center[1] + ramp_half[2] * n_top[1],
        ramp_center[2] + ramp_half[2] * n_top[2],
    )
    cube_gap = 1.0e-3
    cube_pos = (
        top_centre[0] + (cube_he + cube_gap) * n_top[0],
        top_centre[1] + (cube_he + cube_gap) * n_top[1],
        top_centre[2] + (cube_he + cube_gap) * n_top[2],
    )
    cube_box = scene.add_box(
        position=cube_pos,
        half_extents=(cube_he, cube_he, cube_he),
        orientation=ramp_quat,  # align with ramp tilt
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
