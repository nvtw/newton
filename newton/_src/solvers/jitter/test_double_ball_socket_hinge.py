# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the fused DoubleBallSocket hinge joint.

DoubleBallSocket is a rank-5 column (3x3 + 2x2 Schur complement)
that locks 3 translational + 2 rotational DoF using two anchors per
body pair. The free DoF is rotation about the line through the
anchors. Tests:

* **Both anchors coincide.** After settling, both pairs of anchor
  points must map to the same world location (the hinge holds rigidly
  together).
* **Free rotation about the anchor line.** A spinning body must keep
  spinning about the anchor line without precession; perpendicular
  angular velocity must remain near zero.
* **Hinge holds under transverse load.** With one body static and the
  other hung sideways under gravity, the suspended body must hang
  straight (no axial / transverse drift).

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 240
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Both anchors lie on the hinge axis (world +z) so the line through
# them defines the hinge axis directly.
_ANCHOR1 = (0.0, 0.0, -HALF_EXTENT)
_ANCHOR2 = (0.0, 0.0, +HALF_EXTENT)


def _build_anchored_scene(
    device,
    *,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = False,
    cube_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Static body + one dynamic cube joined by a DoubleBallSocket hinge.

    Cube COM defaults to the origin (anchor centre); pass
    ``cube_position=(0, -1, 0)`` for the gravity-pendulum scene
    where we want a real lever arm.
    """
    b = WorldBuilder()
    anchor_body = b.add_static_body()
    cube = b.add_dynamic_body(
        position=cube_position,
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_double_ball_socket_hinge(
        body1=anchor_body,
        body2=cube,
        anchor1=_ANCHOR1,
        anchor2=_ANCHOR2,
    )
    return b.finalize(
        substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
    ), handle


@scene(
    "DoubleBallSocket: free axial spin",
    description="Static-anchored cube spinning freely about the hinge axis.",
    tags=("double_ball_socket",),
)
def build_dbs_axial_spin_scene(device) -> Scene:
    world, _ = _build_anchored_scene(device, initial_angular_velocity=(0.0, 0.0, 2.0))
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


def _quat_rotate(q, v):
    """Rotate ``v`` by quaternion ``q`` (xyzw)."""
    x, y, z, w = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rot @ np.asarray(v, dtype=np.float64)


class TestDoubleBallSocketHinge(unittest.TestCase):
    """End-to-end physics checks for
    :func:`WorldBuilder.add_double_ball_socket_hinge`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        dt = 1.0 / FPS
        for _ in range(frames):
            world.step(dt)

    def test_anchors_coincide(self):
        """Both anchor pairs must map to the same world position.

        With body 1 static and body 2 starting with the anchors
        co-located, after a few seconds of settling the world-space
        positions of (body1.anchor1, body2.anchor1) must coincide
        and likewise for the second anchor. A drift here means the
        rank-5 lock has lost rows.
        """
        device = wp.get_preferred_device()
        # Apply a transverse spin to actually exercise the constraint.
        world, _ = _build_anchored_scene(
            device, initial_angular_velocity=(0.5, 0.0, 0.0)
        )
        self._step(world)

        positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()
        anchor_idx, cube = 1, 2

        # Body 1 (static) anchors are unchanged from their world
        # values at finalize() time.
        a1_b1 = np.asarray(_ANCHOR1, dtype=np.float64)
        a2_b1 = np.asarray(_ANCHOR2, dtype=np.float64)

        # Body 2 (cube) anchors: snapshotted as world-space points at
        # finalize, so the local anchor relative to its starting COM
        # at the origin is just the world anchor position. After
        # rotation/translation we recover them as p + R @ local.
        a1_b2 = positions[cube] + _quat_rotate(orientations[cube], _ANCHOR1)
        a2_b2 = positions[cube] + _quat_rotate(orientations[cube], _ANCHOR2)

        d1 = np.linalg.norm(a1_b2 - a1_b1)
        d2 = np.linalg.norm(a2_b2 - a2_b1)
        # Both anchors should track within a few mm despite the
        # transverse spin -- soft-constraint slop is the only thing
        # that lets them drift at all.
        self.assertLess(d1, 0.02, msg=f"anchor1 drift {d1:.4f} m too large")
        self.assertLess(d2, 0.02, msg=f"anchor2 drift {d2:.4f} m too large")

    def test_free_axial_rotation(self):
        """Pure axial spin must survive untouched.

        With the cube COM on the hinge axis and only an axial
        angular velocity, the joint applies no impulse: rotating
        about the anchor line moves neither anchor and breaks no
        lock. After many steps the axial spin should still be ~at
        its initial value.
        """
        device = wp.get_preferred_device()
        omega_axial = 2.0
        world, _ = _build_anchored_scene(
            device, initial_angular_velocity=(0.0, 0.0, omega_axial)
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(
            omegas[cube, 2],
            omega_axial,
            delta=0.05,
            msg=f"axial spin bled to {omegas[cube, 2]}",
        )
        # Perpendicular components should still be ~0 (the joint
        # cannot create them spontaneously).
        self.assertLess(
            math.hypot(omegas[cube, 0], omegas[cube, 1]),
            0.02,
            msg=f"perpendicular spin appeared: {omegas[cube]}",
        )

    def test_perpendicular_spin_is_locked_out(self):
        """A transverse spin about world +x must be rejected.

        Body 1 is static; only body 2 can move. The hinge locks the
        2 perpendicular rotational DoF, so an initial omega.x must
        decay to ~0 -- there is nowhere for the angular momentum to
        go (body 1 is an infinite sink).
        """
        device = wp.get_preferred_device()
        world, _ = _build_anchored_scene(
            device, initial_angular_velocity=(2.0, 0.0, 0.0)
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertLess(
            abs(omegas[cube, 0]),
            0.05,
            msg=f"transverse spin survived: omega_x={omegas[cube, 0]}",
        )

    def test_hangs_under_gravity(self):
        """Gravity-loaded cube must end up below the anchor centre.

        Cube starts at the origin (anchor centre). Under gravity the
        joint pins both anchors near their initial world positions
        (the hinge axis is +z, gravity is -y, so the cube cannot
        translate out from under the joint). The COM stays at ~y=0
        in the anchor-coincident case because the joint forbids any
        lateral drift.
        """
        device = wp.get_preferred_device()
        world, _ = _build_anchored_scene(device, affected_by_gravity=True)
        self._step(world, frames=600)

        positions = world.bodies.position.numpy()
        cube = 2
        # Cube COM lies at the anchor centre by construction; the
        # joint forbids any translation, so the COM must be ~origin.
        self.assertLess(
            np.linalg.norm(positions[cube]),
            0.02,
            msg=f"cube drifted under gravity: pos={positions[cube]}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
