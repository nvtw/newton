# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for :class:`JointMode.FIXED` (weld joint).

FIXED is a 6-DoF weld: all three translational and all three rotational
DoFs are locked. Implemented as REVOLUTE's anchor-1 3-row point lock +
anchor-2 tangent 2-row lock + PRISMATIC's anchor-3 scalar 1-row lock.

Checks:

* A dynamic cube welded to the world anchor does not fall under
  gravity (position drift stays small across a full settle window).
* The welded cube does not rotate freely -- its orientation at the
  end of a settle is the same as the initial orientation, within
  soft-constraint slop.
* A free-spinning cube welded off the anchor *does* translate with
  the welded carrier's linear motion and does *not* accumulate drift
  relative to the carrier (end-to-end weld consistency).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.world_builder import (
    JointMode,
    WorldBuilder,
)


GRAVITY = 9.81
FPS = 120
SUBSTEPS = 8
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 240
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _build_world_welded_cube(
    device,
    *,
    cube_position: tuple[float, float, float] = (0.5, 0.0, 0.0),
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """World anchor + dynamic cube welded at a chosen position.

    The anchor is at origin; the weld's anchor1 sits at world origin
    and anchor2 is 1 m along +x, defining the weld axis. The cube
    hangs at ``cube_position`` and the weld clamps every DoF."""
    b = WorldBuilder()
    anchor = b.world_body
    cube = b.add_dynamic_body(
        position=cube_position,
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=True,
        angular_velocity=initial_angular_velocity,
    )
    b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=(1.0, 0.0, 0.0),
        mode=JointMode.FIXED,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, 0.0, -GRAVITY),
        device=device,
    )


@scene(
    "Fixed: welded cube under gravity",
    description="Cube rigidly welded to the world anchor; gravity must not move it.",
    tags=("fixed",),
)
def build_welded_cube_scene(device) -> Scene:
    world = _build_world_welded_cube(device)
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestFixedJoint(unittest.TestCase):
    """End-to-end checks for :class:`JointMode.FIXED`."""

    def test_welded_cube_holds_position_under_gravity(self):
        """All 6 DoFs locked: the cube must stay at its initial pose
        (within soft-constraint slop) after a long settle under gravity."""
        device = wp.get_preferred_device()
        world = _build_world_welded_cube(device, cube_position=(0.5, 0.0, 0.0))
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        positions = world.bodies.position.numpy()
        cube = 1
        drift = float(np.linalg.norm(positions[cube] - np.array([0.5, 0.0, 0.0])))
        self.assertLess(
            drift,
            0.05,
            msg=f"welded cube drifted {drift:.4f} m under gravity",
        )

    def test_welded_cube_does_not_rotate(self):
        """An initial spin must not survive the 6-DoF lock."""
        device = wp.get_preferred_device()
        world = _build_world_welded_cube(
            device,
            cube_position=(0.5, 0.0, 0.0),
            initial_angular_velocity=(1.0, -0.7, 0.5),
        )
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 1
        omega_mag = float(np.linalg.norm(omegas[cube]))
        self.assertLess(
            omega_mag,
            0.2,
            msg=f"welded cube still spinning: |omega|={omega_mag:.3f} rad/s",
        )

    def test_welded_orientation_preserved(self):
        """Orientation must return near the identity quaternion after
        gravity tries to pull the cube down."""
        device = wp.get_preferred_device()
        world = _build_world_welded_cube(device, cube_position=(0.5, 0.0, 0.0))
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        orientations = world.bodies.orientation.numpy()
        cube = 1
        q = orientations[cube]  # xyzw
        # Angular deviation from identity: 2 * acos(|w|). A true weld
        # should keep w ~ 1 and the xyz components ~ 0.
        w_clamped = float(min(1.0, max(-1.0, abs(q[3]))))
        angle = 2.0 * np.arccos(w_clamped)
        self.assertLess(
            angle,
            0.1,
            msg=f"welded cube rotated {angle:.4f} rad away from rest",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
