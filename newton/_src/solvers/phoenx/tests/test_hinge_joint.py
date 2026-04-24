# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for a :class:`JointMode.REVOLUTE` (hinge) joint.

Properties checked:

* Pendulum settles below the anchor with the lever arm preserved and
  no perpendicular spin leaks through the angular lock.
* A position-drive PD spring parks the joint at ``target`` with no
  steady-state perpendicular spin.
* A velocity-drive PD servo brakes an initial spin about the hinge
  axis.

Scenes are registered with :func:`scene` so the test visualizer can
replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.scene_registry import Scene, scene
from newton._src.solvers.phoenx.tests._test_helpers import run_settle_loop
from newton._src.solvers.phoenx.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

GRAVITY = 9.81
FPS = 120
SUBSTEPS = 8
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 360  # 3 s @ 120 fps
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_HINGE_AXIS = (0.0, 0.0, 1.0)  # +z


def _build_pendulum(
    device,
    *,
    drive_mode: DriveMode = DriveMode.OFF,
    target: float = 0.0,
    target_velocity: float = 0.0,
    stiffness_drive: float = 0.0,
    damping_drive: float = 0.0,
    max_force_drive: float = 0.0,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = True,
):
    """Static-anchored hinge with a 1 m lever arm under gravity.

    Cube COM at ``(0, -1, 0)``, hinge centre at origin, hinge axis +z.
    The cube can swing about +z but other angular DoFs are locked.
    """
    b = WorldBuilder()
    anchor = b.world_body
    cube = b.add_dynamic_body(
        position=(0.0, -1.0, 0.0) if affected_by_gravity else (0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    b.add_joint(
        body1=anchor,
        body2=cube,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=_HINGE_AXIS,  # anchor1 + axis yields rest_length = 1 m
        mode=JointMode.REVOLUTE,
        drive_mode=drive_mode,
        target=target,
        target_velocity=target_velocity,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        max_force_drive=max_force_drive,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=(0.0, -GRAVITY, 0.0),
        device=device,
    )


@scene(
    "Hinge: pendulum (no motor)",
    description="1 m hinge pendulum under gravity; +z hinge axis.",
    tags=("hinge",),
)
def build_hinge_pendulum_scene(device) -> Scene:
    world = _build_pendulum(device)
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "Hinge: PD brake",
    description="Hinge with PD velocity servo at 0 rad/s braking an initial spin.",
    tags=("hinge", "drive"),
)
def build_hinge_brake_scene(device) -> Scene:
    world = _build_pendulum(
        device,
        drive_mode=DriveMode.VELOCITY,
        target_velocity=0.0,
        damping_drive=5.0,
        max_force_drive=20.0,
        initial_angular_velocity=(0.0, 0.0, 3.0),
        affected_by_gravity=False,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestHingeJoint(unittest.TestCase):
    """End-to-end physics checks for a :class:`JointMode.REVOLUTE` joint."""

    def test_pendulum_settles(self):
        """Pendulum ends up below the anchor with no perpendicular spin."""
        device = wp.get_preferred_device()
        world = _build_pendulum(device)
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        positions = world.bodies.position.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        cube = 1

        self.assertAlmostEqual(positions[cube, 0], 0.0, delta=0.1, msg=f"x drift: {positions[cube]}")
        self.assertAlmostEqual(positions[cube, 1], -1.0, delta=0.1, msg=f"lever: {positions[cube]}")
        self.assertAlmostEqual(positions[cube, 2], 0.0, delta=0.1, msg=f"z drift: {positions[cube]}")

        omega_perp = math.hypot(omegas[cube, 0], omegas[cube, 1])
        self.assertLess(
            omega_perp,
            0.3,
            msg=f"angular lock leaked: omega={omegas[cube]}",
        )

    def test_pd_brake_kills_axial_spin(self):
        """PD velocity servo at 0 rad/s with generous torque brakes the spin."""
        device = wp.get_preferred_device()
        world = _build_pendulum(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=0.0,
            damping_drive=5.0,
            max_force_drive=20.0,
            initial_angular_velocity=(0.0, 0.0, 3.0),
            affected_by_gravity=False,
        )
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 1
        self.assertLess(
            abs(omegas[cube, 2]),
            0.2,
            msg=f"PD brake failed to stop axial spin: omega_z={omegas[cube, 2]:.3f}",
        )

    def test_position_drive_tracks_target(self):
        """PD position drive at target=pi/3 parks the cube there."""
        device = wp.get_preferred_device()
        target_angle = math.pi / 3.0
        world = _build_pendulum(
            device,
            drive_mode=DriveMode.POSITION,
            target=target_angle,
            stiffness_drive=40.0,
            damping_drive=5.0,
            max_force_drive=50.0,
            affected_by_gravity=False,
        )
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        # Read the drive row's position-level residual through gather_constraint_errors;
        # for a settled joint the angular residual about the hinge axis is
        # ``target_angle - current_angle``, so a near-zero residual means the
        # drive parked on target.
        out = wp.zeros(world.num_constraints, dtype=wp.spatial_vector, device=device)
        world.gather_constraint_errors(out)
        err = out.numpy()[0]
        angular_residual_z = err[5]  # spatial_bottom.z
        self.assertLess(
            abs(angular_residual_z),
            0.05,
            msg=f"position drive off-target: residual_z={angular_residual_z:.4f} rad",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
