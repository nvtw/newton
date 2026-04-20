# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the Prismatic (slider) constraint.

A Prismatic joint locks 5 of the 6 relative DoF: 3 rotational (the
two bodies must keep their relative orientation) and 2 translational
(perpendicular to the slide axis). The remaining free DoF is
translation along the slide axis.

Tests:

* **Free axial slide.** With gravity *along* the axis the suspended
  body falls along the axis with no perpendicular drift. The fall
  distance after a known time matches the analytic free-fall
  ``y = ½ g t²`` (within slop).
* **Perpendicular lock.** With gravity *perpendicular* to the axis
  the joint must hold the body in place: no translation along the
  loaded axis; the lateral perpendicular-translation lock fights
  gravity directly.
* **Angular lock.** Spin the suspended body about an arbitrary axis;
  after many steps its angular velocity must decay to that of the
  static partner (i.e. ~0). The relative orientation must stay near
  the rest pose.

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter._test_helpers import run_settle_loop
from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 120  # 2 s @ 60 fps -- PGS warm-start converges well within this
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_prismatic_scene(
    device,
    *,
    axis: tuple[float, float, float],
    affected_by_gravity: bool = True,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Static body + one dynamic cube joined by a Prismatic joint.

    The cube starts at the origin; the static body also sits at the
    origin so the prismatic anchor coincides with both COMs at t=0.
    The slide axis is whatever the caller passes in -- choose +y for
    "vertical slider" tests, +x for "horizontal slider under
    perpendicular gravity" tests, etc.
    """
    b = WorldBuilder()
    anchor = b.add_static_body()
    cube = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_prismatic(
        body1=anchor,
        body2=cube,
        anchor=(0.0, 0.0, 0.0),
        axis=axis,
    )
    return b.finalize(
        substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
    ), handle


@scene(
    "Prismatic: vertical slider (gravity along axis)",
    description="Static body + cube on a +y prismatic; cube falls freely along +y.",
    tags=("prismatic",),
)
def build_prismatic_vertical_slider_scene(device) -> Scene:
    world, _ = _build_prismatic_scene(device, axis=(0.0, 1.0, 0.0))
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "Prismatic: horizontal slider (gravity perpendicular to axis)",
    description="Cube on a +x prismatic with gravity along -y; the lateral lock holds it.",
    tags=("prismatic",),
)
def build_prismatic_horizontal_slider_scene(device) -> Scene:
    world, _ = _build_prismatic_scene(device, axis=(1.0, 0.0, 0.0))
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


def _quat_relative_angle(q):
    """Return the rotation angle of unit quaternion ``q`` (xyzw) [rad].

    For a unit quaternion ``(x, y, z, w)`` the rotation angle is
    ``2 * acos(|w|)``; the absolute value collapses ``q`` and
    ``-q`` (the same physical rotation).
    """
    w = abs(float(q[3]))
    w = min(1.0, max(-1.0, w))
    return 2.0 * math.acos(w)


class TestPrismatic(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_prismatic`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        run_settle_loop(world, frames, dt=1.0 / FPS)

    def test_free_axial_slide_under_gravity(self):
        """Slide axis along gravity -> free fall along the axis only.

        Cube on a +y prismatic with default gravity ``-9.81 y``: the
        only unconstrained DoF is translation along +y, so the cube
        accelerates downward at ``g`` and at ``t`` seconds is at
        ``y = -½ g t²``. Lateral position must stay at zero
        (perpendicular lock); orientation must stay at identity
        (angular lock); the lateral velocity must stay at zero.
        """
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(device, axis=(0.0, 1.0, 0.0))

        frames = 120  # 2 s
        t = frames / FPS
        self._step(world, frames=frames)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        cube = 2

        expected_y = -0.5 * GRAVITY * t * t
        # 5% slop accounts for symplectic-Euler integrator drift over
        # 2 s; the *shape* of the motion (constant acceleration along
        # +y, zero in x/z) is the meaningful invariant.
        self.assertAlmostEqual(
            positions[cube, 1],
            expected_y,
            delta=0.05 * abs(expected_y),
            msg=f"expected y~{expected_y:.3f} m, got {positions[cube, 1]:.4f}",
        )
        self.assertLess(
            abs(positions[cube, 0]),
            0.02,
            msg=f"lateral x drift {positions[cube, 0]:.4f} -- perp lock failed",
        )
        self.assertLess(
            abs(positions[cube, 2]),
            0.02,
            msg=f"lateral z drift {positions[cube, 2]:.4f} -- perp lock failed",
        )
        self.assertLess(
            abs(velocities[cube, 0]) + abs(velocities[cube, 2]),
            0.05,
            msg=f"lateral velocity leaked: v={velocities[cube]}",
        )

    def test_perpendicular_gravity_held(self):
        """Slide axis perpendicular to gravity -> body holds in place.

        Cube on a +x prismatic with default gravity ``-9.81 y``: -y
        is fully orthogonal to the slide axis, so the lateral lock
        must absorb the full weight and the cube must not translate
        along *any* axis. After 4 s the position must still be ~0.
        """
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(device, axis=(1.0, 0.0, 0.0))
        self._step(world)

        positions = world.bodies.position.numpy()
        cube = 2
        self.assertLess(
            np.linalg.norm(positions[cube]),
            0.05,
            msg=f"slider drifted under perpendicular gravity: pos={positions[cube]}",
        )

    def test_angular_lock_holds(self):
        """Initial angular velocity must be killed by the angular lock.

        Cube on a +z prismatic, no gravity, spun about an arbitrary
        axis. The 3-row angular lock pins the relative orientation
        of body 2 to body 1 (which is static), so all angular
        velocity must decay to ~0 and the cube's orientation must
        end up close to the rest pose (identity).
        """
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(
            device,
            axis=(0.0, 0.0, 1.0),
            affected_by_gravity=False,
            initial_angular_velocity=(0.6, -0.4, 0.3),
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        orientations = world.bodies.orientation.numpy()
        cube = 2

        self.assertLess(
            np.linalg.norm(omegas[cube]),
            0.05,
            msg=f"angular lock failed to brake: omega={omegas[cube]}",
        )
        rot_angle = _quat_relative_angle(orientations[cube])
        self.assertLess(
            rot_angle,
            math.radians(2.0),
            msg=f"cube ended at {math.degrees(rot_angle):.2f} deg from rest pose",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
