# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for a :class:`JointMode.PRISMATIC` (slider) joint.

Properties checked:

* Under gravity aligned with the slide axis the body slides freely
  (no spurious damping from the slider's locked DoFs).
* A PD position drive parks the body at ``target`` displacement.
* A PD velocity drive tracks a constant-speed slide setpoint.

Scenes are registered with :func:`scene` so the test visualizer can
replay them interactively.
"""

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
SETTLE_FRAMES = 240  # 2 s @ 120 fps
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_SLIDE_AXIS = (0.0, 1.0, 0.0)  # +y


def _build_slider(
    device,
    *,
    drive_mode: DriveMode = DriveMode.OFF,
    target: float = 0.0,
    target_velocity: float = 0.0,
    stiffness_drive: float = 0.0,
    damping_drive: float = 0.0,
    max_force_drive: float = 0.0,
    affected_by_gravity: bool = False,
):
    """Static anchor at origin; puck on the +y slide axis.

    Puck starts at ``(0, 0, 0)`` and slides only along +y. All rotations
    and perpendicular translations are locked.
    """
    b = WorldBuilder()
    anchor = b.world_body
    puck = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
    )
    b.add_joint(
        body1=anchor,
        body2=puck,
        anchor1=(0.0, 0.0, 0.0),
        anchor2=_SLIDE_AXIS,  # rest_length = 1 m along +y
        mode=JointMode.PRISMATIC,
        drive_mode=drive_mode,
        target=target,
        target_velocity=target_velocity,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        max_force_drive=max_force_drive,
    )
    gravity = (0.0, -GRAVITY, 0.0) if affected_by_gravity else (0.0, 0.0, 0.0)
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        gravity=gravity,
        device=device,
    )


@scene(
    "Prismatic: gravity-slide",
    description="Slider along +y with gravity; free-fall acceleration.",
    tags=("prismatic",),
)
def build_prismatic_slide_scene(device) -> Scene:
    world = _build_slider(device, affected_by_gravity=True)
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "Prismatic: PD velocity servo",
    description="Slider with PD velocity target = 1 m/s along +y.",
    tags=("prismatic", "drive"),
)
def build_prismatic_velocity_scene(device) -> Scene:
    world = _build_slider(
        device,
        drive_mode=DriveMode.VELOCITY,
        target_velocity=1.0,
        damping_drive=10.0,
        max_force_drive=50.0,
    )
    he = np.zeros((world.num_bodies, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "PhoenX simulation tests run on CUDA only (graph capture required for reasonable run-time).",
)
class TestPrismatic(unittest.TestCase):
    """End-to-end physics checks for a :class:`JointMode.PRISMATIC` joint."""

    def test_gravity_slide_is_free(self):
        """With gravity along the slide axis, the puck accelerates at ``g``.

        After ``t`` seconds of free-sliding from rest, ``v_y ≈ -g t``
        and ``x`` / ``z`` stay zero.
        """
        device = wp.get_preferred_device()
        world = _build_slider(device, affected_by_gravity=True)
        dt = 1.0 / FPS
        run_settle_loop(world, SETTLE_FRAMES, dt=dt)

        vels = world.bodies.velocity.numpy()
        positions = world.bodies.position.numpy()
        puck = 1

        expected_v = -GRAVITY * (SETTLE_FRAMES * dt)
        self.assertAlmostEqual(
            vels[puck, 1],
            expected_v,
            delta=abs(expected_v) * 0.05,
            msg=f"free-slide velocity off: got {vels[puck, 1]:.3f}, expected {expected_v:.3f}",
        )
        self.assertLess(abs(vels[puck, 0]), 0.05, msg=f"vx leaked: {vels[puck, 0]}")
        self.assertLess(abs(vels[puck, 2]), 0.05, msg=f"vz leaked: {vels[puck, 2]}")
        self.assertLess(abs(positions[puck, 0]), 0.05, msg=f"x drift: {positions[puck, 0]}")
        self.assertLess(abs(positions[puck, 2]), 0.05, msg=f"z drift: {positions[puck, 2]}")

    def test_velocity_drive_tracks_target(self):
        """PD velocity servo at 1 m/s pulls the puck to that slide speed."""
        device = wp.get_preferred_device()
        world = _build_slider(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=1.0,
            damping_drive=10.0,
            max_force_drive=50.0,
        )
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        vels = world.bodies.velocity.numpy()
        puck = 1
        self.assertAlmostEqual(
            vels[puck, 1],
            1.0,
            delta=0.1,
            msg=f"PD velocity tracking off: v_y={vels[puck, 1]:.3f}",
        )

    def test_position_drive_tracks_target(self):
        """PD position drive at target=0.5 m parks the puck there."""
        device = wp.get_preferred_device()
        target_pos = 0.5
        world = _build_slider(
            device,
            drive_mode=DriveMode.POSITION,
            target=target_pos,
            stiffness_drive=80.0,
            damping_drive=10.0,
            max_force_drive=100.0,
        )
        run_settle_loop(world, SETTLE_FRAMES, dt=1.0 / FPS)

        positions = world.bodies.position.numpy()
        puck = 1
        self.assertAlmostEqual(
            positions[puck, 1],
            target_pos,
            delta=0.05,
            msg=f"PD position tracking off: y={positions[puck, 1]:.3f}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
