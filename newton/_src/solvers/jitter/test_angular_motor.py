# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the stand-alone AngularMotor constraint.

AngularMotor drives the *relative* angular velocity of body 2 vs.
body 1 along a hinge axis toward a target rad/s, capped by an axial
torque budget. There is no positional component -- it's pure
velocity-target. Tests:

* **Velocity tracking.** With ample torque budget the motor pulls the
  relative axial velocity onto the target and holds it there.
* **Disabled by max_force=0.** ``max_force=0`` is the documented
  "motor present but inactive" default; the relative axial velocity
  must not be touched.
* **Reaction symmetry / Newton 3.** With identical inverse inertias on
  both bodies and gravity off, total angular momentum about the hinge
  axis must be conserved -- whatever the motor adds to body 2 it must
  take from body 1 in equal measure.
* **Per-axis isolation.** The motor must only act along its hinge
  axis; perpendicular angular velocity components must be untouched.

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 240
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_HINGE_AXIS = (0.0, 0.0, 1.0)


def _build_two_body_motor(
    device,
    *,
    target_velocity: float = 0.0,
    max_force: float = 0.0,
    initial_angular_velocity_b1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_angular_velocity_b2: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Two free dynamic cubes joined only by an AngularMotor.

    No socket, no hinge angle: the only thing the constraint can do
    is bend the *axial* relative angular velocity. This keeps the
    test signal isolated to exactly the motor code path.
    """
    b = WorldBuilder()
    b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b1,
    )
    b.add_dynamic_body(
        position=(2.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity_b2,
    )
    b.add_angular_motor(
        body1=1,
        body2=2,
        axis=_HINGE_AXIS,
        target_velocity=target_velocity,
        max_force=max_force,
    )
    return b.finalize(
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )


@scene(
    "AngularMotor: target +2 rad/s",
    description="Two free bodies; motor drives relative spin to +2 rad/s.",
    tags=("angular_motor",),
)
def build_angular_motor_target_scene(device) -> Scene:
    world = _build_two_body_motor(device, target_velocity=2.0, max_force=20.0)
    he = np.zeros((3, 3), dtype=np.float32)
    he[1] = HALF_EXTENT
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


class TestAngularMotor(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_angular_motor`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        dt = 1.0 / FPS
        for _ in range(frames):
            world.step(dt)

    def test_tracks_target_velocity(self):
        """Motor with ample torque budget converges to the target.

        Both bodies start at rest; the motor must end up holding the
        relative axial velocity at the target. Per-body velocities
        also follow from angular-momentum conservation: with equal
        inverse inertias, body 1 spins at -target/2 and body 2 at
        +target/2.
        """
        device = wp.get_preferred_device()
        target = 2.0
        world = _build_two_body_motor(device, target_velocity=target, max_force=50.0)
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        rel_axial = omegas[2, 2] - omegas[1, 2]
        self.assertAlmostEqual(
            rel_axial,
            target,
            delta=0.1,
            msg=f"relative axial velocity {rel_axial:.4f} did not converge to {target}",
        )
        self.assertAlmostEqual(omegas[2, 2], target / 2.0, delta=0.1)
        self.assertAlmostEqual(omegas[1, 2], -target / 2.0, delta=0.1)

    def test_disabled_when_max_force_zero(self):
        """``max_force=0`` should make the motor a no-op.

        The user-facing API documents this as the "motor present but
        inactive" default; spin body 2 about the axis and verify the
        velocity is unchanged after a long simulation.
        """
        device = wp.get_preferred_device()
        omega0 = 1.5
        world = _build_two_body_motor(
            device,
            target_velocity=0.0,  # would otherwise brake body 2
            max_force=0.0,
            initial_angular_velocity_b2=(0.0, 0.0, omega0),
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        self.assertAlmostEqual(
            omegas[2, 2],
            omega0,
            delta=0.05,
            msg=f"disabled motor still applied torque: omega_z={omegas[2, 2]}",
        )
        self.assertAlmostEqual(omegas[1, 2], 0.0, delta=0.05)

    def test_axial_momentum_conserved(self):
        """Newton 3: motor torque on body 2 = -torque on body 1.

        With identical inverse inertias and zero gravity, the sum
        ``omega_b1.z + omega_b2.z`` is the system's axial angular
        momentum and must be conserved across any motor action,
        including motor brake / drive.
        """
        device = wp.get_preferred_device()
        omega1, omega2 = -0.4, 1.6  # net momentum 1.2 about +z
        world = _build_two_body_motor(
            device,
            target_velocity=5.0,  # impossible target -> motor saturates torque
            max_force=10.0,
            initial_angular_velocity_b1=(0.0, 0.0, omega1),
            initial_angular_velocity_b2=(0.0, 0.0, omega2),
        )
        net_initial = omega1 + omega2
        self._step(world, frames=60)

        omegas = world.bodies.angular_velocity.numpy()
        net_final = omegas[1, 2] + omegas[2, 2]
        self.assertAlmostEqual(
            net_final,
            net_initial,
            delta=0.1,
            msg=(
                f"axial angular momentum drifted: net_initial={net_initial:.4f}, "
                f"net_final={net_final:.4f}"
            ),
        )

    def test_perpendicular_velocity_untouched(self):
        """The motor's reaction must lie strictly along the hinge axis.

        Spin body 2 about world +x (perpendicular to the +z motor
        axis); the motor has no business modifying perpendicular
        velocity. After many steps body 2's perpendicular spin must
        be ~unchanged and body 1 must still be at rest along x.
        """
        device = wp.get_preferred_device()
        omega_perp = 1.0
        world = _build_two_body_motor(
            device,
            target_velocity=2.0,  # arbitrary, motor is along +z
            max_force=10.0,
            initial_angular_velocity_b2=(omega_perp, 0.0, 0.0),
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        self.assertAlmostEqual(
            omegas[2, 0],
            omega_perp,
            delta=0.05,
            msg=f"motor leaked into +x: body2.omega_x={omegas[2, 0]}",
        )
        self.assertAlmostEqual(
            omegas[1, 0],
            0.0,
            delta=0.05,
            msg=f"motor pushed body1 along +x: body1.omega_x={omegas[1, 0]}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
