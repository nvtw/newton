# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the *fused* HingeJoint constraint.

A fused HingeJoint packs three sub-blocks into one PGS column:

* a BallSocket at ``hinge_center`` (3-row positional lock),
* a HingeAngle about ``hinge_axis`` (2-row perpendicular angular
  lock + optional 1-row axial limit),
* an AngularMotor along the same axis (1-row velocity drive).

These tests cover the joint as a whole -- the same physical
behaviours that user-facing code relies on -- without depending on
the implementation details of each sub-block (those are exercised
individually in their own ``test_*.py`` files).

Three scenes:

* **Hanging pendulum, motor off.** A unit cube is hung from the
  static world body via a single hinge with axis along world +z. The
  cube must end up below the anchor (gravity pulls it straight down),
  the anchor points must coincide, and angular velocity perpendicular
  to +z must remain ~0 (the angular lock works under gravity).
* **Motor brake.** Same scene, but with the motor enabled at
  ``target_velocity=0`` and a generous torque budget. The cube starts
  with a +z spin; the motor must drive the spin to ~0.
* **Motor drive against the world anchor.** Static-anchored cube,
  motor at non-zero target velocity, no gravity. The cube must reach
  the target axial spin.

All scenes are also registered with :func:`scene` so the test
visualizer can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.tests._test_helpers import run_settle_loop
from newton._src.solvers.jitter.examples.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 120  # 2 s @ 60 fps -- PGS warm-start converges well within this
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_HINGE_AXIS = (0.0, 0.0, 1.0)


def _build_anchored_hinge(
    device,
    *,
    motor: bool = False,
    target_velocity: float = 0.0,
    max_force: float = 0.0,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = True,
):
    """One fused HingeJoint between an explicit static body and a unit cube.

    Cube COM at the hinge centre (origin) so there is no lever arm
    from the joint to the COM. With gravity off + COM-aligned hinge
    the cube only sees the joint's reactions, which makes it easy to
    isolate motor / lock effects.

    NOTE: this helper uses an explicit :meth:`WorldBuilder.add_static_body`
    anchor; an exact parallel test (``test_hanging_pendulum_uses_world_body``
    below) anchors directly to ``b.world_body`` so a regression in the
    world-body inertia seed (which must be zero, not the default
    identity) shows up as a different motor / lock effective mass. The
    auto-created world body's inverse inertia is set to zero in
    :meth:`WorldBuilder.__init__`; if a future change reverts that to
    identity, the per-body-pair tests below will keep passing while the
    ``world_body`` tests start failing.
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
    handle = b.add_hinge_joint(
        body1=anchor,
        body2=cube,
        hinge_center=(0.0, 0.0, 0.0),
        hinge_axis=_HINGE_AXIS,
        motor=motor,
        target_velocity=target_velocity,
        max_force=max_force,
    )
    world = b.finalize(
        enable_all_constraints=True,
        substeps=SUBSTEPS,
        solver_iterations=SOLVER_ITERATIONS,
        device=device,
    )
    return world, handle


def _build_hanging_pendulum(device, *, motor: bool = False, target_velocity: float = 0.0,
                            max_force: float = 0.0):
    """Static-anchored hinge with a 1 m lever arm under gravity.

    The cube COM is at ``(0, -1, 0)``, the hinge centre is the origin
    and the hinge axis is +z. The cube can swing about +z (pendulum
    motion) but must keep its other angular DoF locked. See
    :func:`_build_anchored_hinge` for the rationale behind also having
    a parallel ``world_body`` test.
    """
    b = WorldBuilder()
    anchor = b.add_static_body()
    cube = b.add_dynamic_body(
        position=(0.0, -1.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=True,
    )
    handle = b.add_hinge_joint(
        body1=anchor,
        body2=cube,
        hinge_center=(0.0, 0.0, 0.0),
        hinge_axis=_HINGE_AXIS,
        motor=motor,
        target_velocity=target_velocity,
        max_force=max_force,
    )
    return b.finalize(
        enable_all_constraints=True,
        substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
    ), handle


@scene(
    "HingeJoint: pendulum (no motor)",
    description="Single fused hinge holding a cube on a 1 m lever arm under gravity.",
    tags=("hinge_joint",),
)
def build_hinge_joint_pendulum_scene(device) -> Scene:
    # Bodies: 0 = auto world body (unused), 1 = explicit static anchor,
    # 2 = the swinging cube.
    world, _ = _build_hanging_pendulum(device)
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@scene(
    "HingeJoint: motor brake",
    description="Hinge with motor at target_velocity=0; cube spins, motor stops it.",
    tags=("hinge_joint", "motor"),
)
def build_hinge_joint_brake_scene(device) -> Scene:
    world, _ = _build_anchored_hinge(
        device,
        motor=True,
        target_velocity=0.0,
        max_force=20.0,
        initial_angular_velocity=(0.0, 0.0, 3.0),
        affected_by_gravity=False,
    )
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return Scene(world=world, body_half_extents=he, frame_dt=1.0 / FPS, substeps=SUBSTEPS)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestHingeJoint(unittest.TestCase):
    """End-to-end physics checks for :func:`WorldBuilder.add_hinge_joint`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        run_settle_loop(world, frames, dt=1.0 / FPS)

    def test_hanging_pendulum_settles(self):
        """Hung from a single hinge under gravity, the cube ends up
        directly below the anchor and only spins (if at all) about +z.

        * COM should sit at ~(0, -1, 0): pure -y suspension, no
          lateral drift, lever arm preserved.
        * Perpendicular angular velocity (omega.x, omega.y) must be
          near zero -- the 2-DoF angular lock has done its job.
        """
        device = wp.get_preferred_device()
        world, _ = _build_hanging_pendulum(device)
        # Long settle so the swing damps out via the joint's
        # soft-constraint slip (default Hz / damping_ratio).
        self._step(world, frames=600)

        positions = world.bodies.position.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        cube = 2  # body 0: auto world, body 1: explicit static anchor, body 2: cube

        self.assertAlmostEqual(positions[cube, 0], 0.0, delta=0.1, msg=f"COM drifted in x: {positions[cube]}")
        self.assertAlmostEqual(positions[cube, 1], -1.0, delta=0.05, msg=f"lever arm changed: {positions[cube]}")
        self.assertAlmostEqual(positions[cube, 2], 0.0, delta=0.1, msg=f"COM drifted in z: {positions[cube]}")

        omega_perp = math.hypot(omegas[cube, 0], omegas[cube, 1])
        self.assertLess(
            omega_perp,
            0.2,
            msg=f"perpendicular spin survived the angular lock: omega={omegas[cube]}",
        )

    def test_motor_brake_kills_axial_spin(self):
        """Motor at target=0 with a generous torque budget brakes the cube.

        A unit cube starts spinning at 3 rad/s about +z (no gravity,
        no lever arm). After 4 s the axial velocity must be ~0 and
        the perpendicular components must still be ~0 (motor is axis-
        only).
        """
        device = wp.get_preferred_device()
        world, _ = _build_anchored_hinge(
            device,
            motor=True,
            target_velocity=0.0,
            max_force=20.0,
            initial_angular_velocity=(0.0, 0.0, 3.0),
            affected_by_gravity=False,
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertLess(
            abs(omegas[cube, 2]),
            0.1,
            msg=f"motor failed to brake axial spin: omega_z={omegas[cube, 2]}",
        )
        for k in (0, 1):
            self.assertLess(
                abs(omegas[cube, k]),
                0.05,
                msg=f"motor leaked into perpendicular axis {k}: omega={omegas[cube]}",
            )

    def test_motor_drive_to_target_via_world_body(self):
        """Same as :meth:`test_motor_drive_to_target`, anchored to
        :attr:`WorldBuilder.world_body` rather than an explicit static.

        Locks in the fix that ``WorldBuilder.__init__`` seeds the
        auto-created world body's ``inverse_inertia`` to zero. Before
        the fix the world body had identity inertia and the motor
        impulse split 50/50 between the cube and the (non-integrated)
        world body, so the cube only reached half the requested
        target velocity.
        """
        device = wp.get_preferred_device()
        target = 1.5
        b = WorldBuilder()
        cube = b.add_dynamic_body(
            position=(0.0, 0.0, 0.0),
            inverse_mass=1.0,
            inverse_inertia=_INV_INERTIA,
            affected_by_gravity=False,
        )
        b.add_hinge_joint(
            body1=b.world_body,
            body2=cube,
            hinge_center=(0.0, 0.0, 0.0),
            hinge_axis=_HINGE_AXIS,
            motor=True,
            target_velocity=target,
            max_force=20.0,
        )
        world = b.finalize(
        enable_all_constraints=True,
            substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        self.assertAlmostEqual(
            omegas[cube, 2],
            target,
            delta=0.1,
            msg=(
                f"motor anchored to world_body failed to reach target "
                f"{target}: omega_z={omegas[cube, 2]} "
                f"(regression: world_body identity inertia bug)"
            ),
        )

    def test_motor_drive_to_target(self):
        """Motor at non-zero target reaches the target axial velocity.

        Static body 1 is the world anchor (infinite mass / inertia
        sink), so the motor only has to spin one body up. After
        settling, body 2's omega.z must equal ``target`` (no /2
        because the world anchor doesn't move).
        """
        device = wp.get_preferred_device()
        target = 1.5
        world, _ = _build_anchored_hinge(
            device,
            motor=True,
            target_velocity=target,
            max_force=20.0,
            affected_by_gravity=False,
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(
            omegas[cube, 2],
            target,
            delta=0.1,
            msg=f"motor failed to reach target {target}: omega_z={omegas[cube, 2]}",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
