# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Zero-friction validation tests for :class:`PhoenXWorld`.

Cases where the Coulomb friction row must contribute exactly zero
impulse -- any leak into the tangent plane shows up as
decelerating cubes, horizontal momentum drift, or the wrong
acceleration on an inclined slide. These are the tests that
confirm ``default_friction = 0.0`` / ``mu = 0`` scenes behave as
pure-normal-contact simulations, which is the configuration the
nut-bolt scene relies on (the nut's rotation comes from the
SDF's helical contact normals, not friction).

Runs on CUDA only, same as the other PhoenX end-to-end tests.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene

_G = 9.81


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX zero-friction tests require CUDA")
class TestPhoenXZeroFrictionCubeSlide(unittest.TestCase):
    """A cube launched along +X on a flat plane with friction=0
    must maintain its velocity for the entire run.

    Kinetic-friction prediction at ``mu = 0`` is ``a = 0`` (zero
    deceleration). Any tangent-row leak would appear as a slow
    deceleration -- the test bounds it tightly.
    """

    def test_cube_velocity_unchanged_without_friction(self) -> None:
        scene = _PhoenXScene(fps=120, substeps=4, solver_iterations=16, friction=0.0)
        scene.add_ground_plane()
        he = 0.5
        v0 = 3.0
        cube = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
        )
        scene.finalize()
        scene.set_body_velocity(cube, (v0, 0.0, 0.0))

        # Let the normal contact settle for a few frames so the
        # velocity we measure isn't corrupted by initial vertical
        # penetration transient.
        for _ in range(5):
            scene.step()
        v_initial = float(scene.body_velocity(cube)[0])

        # Run 2 s of simulated time; any friction leak > 0.01 * m * g
        # would bleed a measurable fraction of the initial velocity.
        for _ in range(240):
            scene.step()

        v_final = float(scene.body_velocity(cube)[0])
        decel = (v_initial - v_final) / 2.0  # m/s^2 average over 2 s
        # Tolerance chosen so a friction leak equivalent to
        # ``mu_effective > 0.01`` (i.e. a 1 % tangent-row leak)
        # would trip. Pure FP noise from CUDA graph replay measures
        # well below 1e-3 m/s^2 on current hardware.
        self.assertLess(
            abs(decel),
            0.01 * _G,
            f"cube decelerated by {decel:.4f} m/s^2 with mu=0 -- "
            f"tangent row leaked (v0={v0}, v_final={v_final:.4f})",
        )
        # Residual vertical/lateral motion must stay in the cube's
        # normal-row noise floor.
        v_final_full = scene.body_velocity(cube)
        v_lateral = float(
            math.hypot(float(v_final_full[1]), float(v_final_full[2]))
        )
        self.assertLess(
            v_lateral,
            0.05,
            f"cube picked up lateral/vertical velocity {v_lateral:.4f} "
            "m/s with mu=0",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX zero-friction tests require CUDA")
class TestPhoenXZeroFrictionMomentumPreserved(unittest.TestCase):
    """System momentum in the tangent plane must stay exactly
    constant when friction=0.

    One cube with initial +X velocity on a plane; gravity pulls it
    down but no tangent force is allowed. The X-component of the
    cube's momentum must stay at ``m * v0`` for the entire run.
    """

    def test_horizontal_momentum_preserved(self) -> None:
        scene = _PhoenXScene(fps=120, substeps=4, solver_iterations=16, friction=0.0)
        scene.add_ground_plane()
        he = 0.5
        # Explicit analytic mass so we can compute the expected
        # momentum to machine precision.
        mass = 1.0
        cube = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            mass=mass,
        )
        scene.finalize()
        v0 = 2.5
        scene.set_body_velocity(cube, (v0, 0.0, 0.0))
        # Let the vertical transient settle first so the initial
        # momentum snapshot isn't polluted by the penetration
        # pushout.
        for _ in range(5):
            scene.step()

        p_initial = mass * float(scene.body_velocity(cube)[0])

        for _ in range(120):  # 1 s
            scene.step()
        p_final = mass * float(scene.body_velocity(cube)[0])

        rel_err = abs(p_final - p_initial) / max(abs(p_initial), 1.0e-6)
        self.assertLess(
            rel_err,
            0.005,  # 0.5 %: well above FP noise, below any real leak.
            f"horizontal momentum changed from {p_initial:.4f} to "
            f"{p_final:.4f} kg m/s (rel err {rel_err:.2%}) with mu=0",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX zero-friction tests require CUDA")
class TestPhoenXZeroFrictionInclinedSlide(unittest.TestCase):
    """A cube released on a plane tilted about +Y by ``theta`` must
    accelerate along the plane at ``g sin(theta)`` when friction=0.

    The plane-primitive in Newton is always horizontal, so we
    rotate the *cube* -- the acceleration along +X matches
    ``g sin(theta)`` as long as gravity is decomposed against the
    contact normal the narrow phase reports. With mu=0 the
    tangent-row impulse must stay at zero, so any miscalculated
    normal projection (leaking into tangent) would reduce the
    apparent acceleration.

    Concrete setup: we tilt GRAVITY instead of the plane so the
    geometry stays Newton-primitive-friendly. Gravity vector
    ``(g sin(theta), 0, -g cos(theta))`` is physically identical
    to a plane tilted by ``theta`` about +Y under pure gravity.
    """

    def test_slide_acceleration_matches_gravity_component(self) -> None:
        theta = math.radians(20.0)
        g_x = _G * math.sin(theta)
        g_z = -_G * math.cos(theta)
        scene = _PhoenXScene(
            fps=120,
            substeps=4,
            solver_iterations=16,
            friction=0.0,
        )
        scene.add_ground_plane()
        he = 0.5
        cube = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
        )
        scene.finalize()
        # Override gravity in-place with the tilted vector.
        scene.world.gravity.fill_(wp.vec3f(g_x, 0.0, g_z))

        # Settle the vertical normal contact first.
        for _ in range(10):
            scene.step()
        v0 = float(scene.body_velocity(cube)[0])

        # Time the cube's acceleration over a short window to avoid
        # running into numerical drift from the over-sized run.
        measure_frames = 60  # 0.5 s
        for _ in range(measure_frames):
            scene.step()
        v1 = float(scene.body_velocity(cube)[0])
        dt = measure_frames / 120.0
        measured_accel = (v1 - v0) / dt

        rel_err = abs(measured_accel - g_x) / g_x
        self.assertLess(
            rel_err,
            0.05,  # 5 % -- allows a small normal-impulse warm-up transient.
            f"measured accel={measured_accel:.4f} m/s^2 vs expected "
            f"g*sin(theta)={g_x:.4f} m/s^2 (rel err {rel_err:.2%}), "
            "with mu=0 the only tangent force should be gravity itself.",
        )


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX zero-friction tests require CUDA")
class TestPhoenXZeroFrictionAngularMomentumPreserved(unittest.TestCase):
    """A spinning cube dropped on a plane with mu=0 must keep its
    spin (within the normal-row's very small tangent noise).

    Angular momentum about the cube's COM can only change through
    a torque. Contact normal force acts on the cube's base at
    ``r = (0, 0, -he)`` -- its torque about the COM is
    ``r x F_n = (0, 0, -he) x (0, 0, F_nz) = 0``. So in the
    frictionless case the angular velocity about +Z must stay at
    its initial value, regardless of how long we simulate.
    """

    def test_spinning_cube_retains_spin_without_friction(self) -> None:
        scene = _PhoenXScene(fps=120, substeps=4, solver_iterations=16, friction=0.0)
        scene.add_ground_plane()
        he = 0.5
        cube = scene.add_box(
            position=(0.0, 0.0, he + 1.0e-3),
            half_extents=(he, he, he),
            mass=1.0,
        )
        scene.finalize()
        # Seed a pure yaw spin.
        spin_z = 4.0
        body_qd_np = scene.state.body_qd.numpy()
        body_qd_np[cube, 5] = spin_z  # angular component along +Z
        scene.state.body_qd.assign(body_qd_np)

        # Settle vertical contact.
        for _ in range(5):
            scene.step()
        # Angular velocity comes from PhoenX body container slot = newton_idx + 1.
        w0 = float(scene.bodies.angular_velocity.numpy()[cube + 1][2])

        # Run 1 s and check the spin hasn't decayed significantly.
        for _ in range(120):
            scene.step()
        w1 = float(scene.bodies.angular_velocity.numpy()[cube + 1][2])
        rel_err = abs(w1 - w0) / abs(w0)
        self.assertLess(
            rel_err,
            0.02,
            f"spinning cube's omega_z drifted from {w0:.4f} to {w1:.4f} "
            f"rad/s (rel err {rel_err:.2%}) with mu=0 -- friction row leaked",
        )


if __name__ == "__main__":
    unittest.main()
