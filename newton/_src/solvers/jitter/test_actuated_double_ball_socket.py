# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the *actuated* DoubleBallSocket hinge joint.

ActuatedDoubleBallSocket extends the parent DBS rank-5 column (3
translational + 2 rotational locked DoF) with one extra scalar PGS row
on the free axial twist. The extra row supports three independent
sub-features, each exercised here against a single-cube + static-anchor
scene:

* **Position drive** -- soft spring towards ``target_angle``. The cube
  must come to rest near the requested twist.
* **Velocity drive** -- soft tracker for ``target_velocity``. The cube
  must reach steady-state spin near the requested rate, and a small
  ``max_force_drive`` must visibly cap that rate (saturate before the
  setpoint).
* **Angular limits** -- one-sided spring-damper on the relative twist
  confined to ``[min_angle, max_angle]``. A cube spinning into the
  upper stop must clamp at ~``max_angle``; one spinning into the
  lower stop must clamp at ~``min_angle``.

Drive + limit can coexist; the limit always wins because it is
unilateral. We also verify that ``DriveMode.OFF`` with no limit
reproduces the parent DBS's free-axial-spin behaviour (back-to-back
sanity check that the actuator block does not leak impulses when both
features are disabled).

All scenes are registered with :func:`scene` so the test visualizer
can replay them interactively.
"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.jitter.scene_registry import Scene, scene
from newton._src.solvers.jitter.world_builder import DriveMode, WorldBuilder

GRAVITY = 9.81
FPS = 60
SUBSTEPS = 4
SOLVER_ITERATIONS = 16
SETTLE_FRAMES = 240
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

_ANCHOR1 = (0.0, 0.0, -HALF_EXTENT)
_ANCHOR2 = (0.0, 0.0, +HALF_EXTENT)


def _build_actuated_scene(
    device,
    *,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drive_mode: DriveMode = DriveMode.OFF,
    target_angle: float = 0.0,
    target_velocity: float = 0.0,
    max_force_drive: float = 0.0,
    hertz_drive: float = 8.0,
    damping_ratio_drive: float = 1.0,
    min_angle: float = 0.0,
    max_angle: float = 0.0,
    hertz_limit: float = 30.0,
    damping_ratio_limit: float = 1.0,
):
    """Static body + one dynamic cube joined by an actuated DBS hinge.

    Cube COM sits at the origin (anchor centre). The hinge axis is
    world ``+z`` because the two anchors lie at ``(0, 0, +/- HALF_EXTENT)``.
    """
    b = WorldBuilder()
    anchor_body = b.add_static_body()
    cube = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=False,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_actuated_double_ball_socket_hinge(
        body1=anchor_body,
        body2=cube,
        anchor1=_ANCHOR1,
        anchor2=_ANCHOR2,
        drive_mode=drive_mode,
        target_angle=target_angle,
        target_velocity=target_velocity,
        max_force_drive=max_force_drive,
        hertz_drive=hertz_drive,
        damping_ratio_drive=damping_ratio_drive,
        min_angle=min_angle,
        max_angle=max_angle,
        hertz_limit=hertz_limit,
        damping_ratio_limit=damping_ratio_limit,
    )
    world = b.finalize(
        substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device
    )
    return world, handle


def _scene_half_extents() -> np.ndarray:
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return he


@scene(
    "ActuatedDBS: position drive",
    description=(
        "Static-anchored cube driven to a +0.7 rad axial twist by a soft "
        "position drive."
    ),
    tags=("actuated_double_ball_socket",),
)
def build_adbs_position_drive_scene(device) -> Scene:
    world, _ = _build_actuated_scene(
        device,
        drive_mode=DriveMode.POSITION,
        target_angle=0.7,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "ActuatedDBS: velocity drive",
    description=(
        "Static-anchored cube driven to 1.5 rad/s axial spin by a soft "
        "velocity drive (max_force=20 N*m)."
    ),
    tags=("actuated_double_ball_socket",),
)
def build_adbs_velocity_drive_scene(device) -> Scene:
    world, _ = _build_actuated_scene(
        device,
        drive_mode=DriveMode.VELOCITY,
        target_velocity=1.5,
        max_force_drive=20.0,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "ActuatedDBS: limit clamp",
    description=(
        "Cube spinning into the upper +0.5 rad stop; the unilateral "
        "spring-damper must clamp the twist at ~+0.5 rad."
    ),
    tags=("actuated_double_ball_socket",),
)
def build_adbs_limit_scene(device) -> Scene:
    world, _ = _build_actuated_scene(
        device,
        initial_angular_velocity=(0.0, 0.0, 0.6),
        min_angle=-0.5,
        max_angle=0.5,
        hertz_limit=10.0,
        damping_ratio_limit=2.0,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


def _axial_twist(orientation: np.ndarray) -> float:
    """Return the signed axial twist of ``orientation`` about world +z.

    The cube starts with identity orientation. Body 1 is static at
    identity, so the relative twist about world +z reduces to the
    axial component of body 2's quaternion in xyzw form: ``2 *
    atan2(z, w)`` (standard z-axis swing-twist decomposition for the
    case where the swing is ~zero, which it must be because the
    parent DBS lock kills the off-axis rotational DoF).
    """
    x, y, z, w = orientation
    return 2.0 * math.atan2(z, w)


class TestActuatedDoubleBallSocket(unittest.TestCase):
    """End-to-end physics checks for
    :func:`WorldBuilder.add_actuated_double_ball_socket_hinge`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        dt = 1.0 / FPS
        for _ in range(frames):
            world.step(dt)

    def test_off_mode_preserves_axial_spin(self):
        """``DriveMode.OFF`` with no limit must not bleed axial spin.

        Drops back to the parent DBS behaviour: the cube spinning
        purely about +z keeps spinning at the same rate -- the
        actuator block is a no-op in this configuration.
        """
        device = wp.get_preferred_device()
        omega_axial = 2.0
        world, _ = _build_actuated_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, omega_axial),
            drive_mode=DriveMode.OFF,
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(
            omegas[cube, 2],
            omega_axial,
            delta=0.05,
            msg=f"axial spin bled with OFF drive: {omegas[cube, 2]}",
        )
        self.assertLess(
            math.hypot(omegas[cube, 0], omegas[cube, 1]),
            0.02,
            msg=f"perpendicular spin appeared: {omegas[cube]}",
        )

    def test_position_drive_reaches_target(self):
        """Position drive must seat the cube near ``target_angle``.

        Soft spring with ``hertz_drive=8`` at zero load should
        converge well within a quarter-second; we wait the full
        settle window to give the damping ratio room to kill the
        residual oscillation.
        """
        device = wp.get_preferred_device()
        target = 0.7
        world, _ = _build_actuated_scene(
            device,
            drive_mode=DriveMode.POSITION,
            target_angle=target,
            hertz_drive=8.0,
            damping_ratio_drive=1.0,
        )
        self._step(world)

        orientations = world.bodies.orientation.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertAlmostEqual(
            twist,
            target,
            delta=0.05,
            msg=f"position drive landed at {twist:.3f} rad, expected {target}",
        )
        # Body must be at rest (the spring/damper killed the
        # transient).
        omegas = world.bodies.angular_velocity.numpy()
        self.assertLess(
            np.linalg.norm(omegas[cube]),
            0.05,
            msg=f"position drive left residual spin: {omegas[cube]}",
        )

    def test_velocity_drive_reaches_target(self):
        """Velocity drive must spin the cube up to ``target_velocity``.

        With a generous ``max_force_drive`` the drive should saturate
        the cube to ~``target_velocity`` within a fraction of a
        second; check both that the steady-state rate is close and
        that the perpendicular components stay locked at zero.
        """
        device = wp.get_preferred_device()
        target = 1.5
        world, _ = _build_actuated_scene(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target,
            max_force_drive=50.0,
        )
        self._step(world, frames=120)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(
            omegas[cube, 2],
            target,
            delta=0.1,
            msg=f"velocity drive at {omegas[cube, 2]:.3f}, expected {target}",
        )
        self.assertLess(
            math.hypot(omegas[cube, 0], omegas[cube, 1]),
            0.02,
            msg=f"perpendicular spin appeared: {omegas[cube]}",
        )

    def test_velocity_drive_force_cap_saturates(self):
        """A tiny ``max_force_drive`` must cap the steady-state spin.

        With ``max_force_drive`` set so that the per-substep impulse
        cap is well below what's needed to overcome the implicit
        soft-spring damping at ``target_velocity``, the cube cannot
        reach the setpoint -- it converges to a much smaller axial
        rate. We just check that the resulting axial spin is
        strictly below the target by a wide margin.
        """
        device = wp.get_preferred_device()
        target = 5.0
        world, _ = _build_actuated_scene(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target,
            max_force_drive=0.1,
        )
        self._step(world, frames=120)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertLess(
            abs(omegas[cube, 2]),
            target * 0.5,
            msg=(
                f"force cap failed: omega_z={omegas[cube, 2]:.3f} "
                f"approached target {target}"
            ),
        )
        # Even saturated, the drive must still be pushing in the
        # right sign (positive target -> positive axial velocity).
        self.assertGreater(
            omegas[cube, 2],
            0.001,
            msg=f"capped drive went the wrong way: omega_z={omegas[cube, 2]}",
        )

    def test_upper_limit_clamps_twist(self):
        """A spin into the upper stop must clamp at ~``max_angle``.

        Cube starts with a gentle axial spin (no drive); the
        unilateral limit row engages once the twist exceeds
        ``max_angle`` and brings the cube to rest near the stop.
        We use a gentle initial velocity + a critically-damped soft
        limit so the body seats against the stop without bouncing
        back through ``min_angle``.
        """
        device = wp.get_preferred_device()
        max_a = 0.5
        world, _ = _build_actuated_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, 0.5),
            min_angle=-2.0,
            max_angle=max_a,
            hertz_limit=10.0,
            damping_ratio_limit=2.0,
        )
        self._step(world, frames=300)

        orientations = world.bodies.orientation.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        # Soft limit -> some over/under-shoot is OK; we just need the
        # steady-state to sit near the stop, not back through zero.
        self.assertLessEqual(
            twist,
            max_a + 0.1,
            msg=f"limit failed: twist={twist:.3f} > {max_a} + slop",
        )
        self.assertGreaterEqual(
            twist,
            max_a - 0.2,
            msg=f"limit overshot back: twist={twist:.3f} < {max_a} - slop",
        )
        # Critical damping kills the spin within the settle window.
        self.assertLess(
            abs(omegas[cube, 2]),
            0.2,
            msg=f"limit didn't damp spin: omega_z={omegas[cube, 2]}",
        )

    def test_lower_limit_clamps_twist(self):
        """Symmetric check: spin into the lower stop must clamp at ~``min_angle``."""
        device = wp.get_preferred_device()
        min_a = -0.5
        world, _ = _build_actuated_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, -0.5),
            min_angle=min_a,
            max_angle=2.0,
            hertz_limit=10.0,
            damping_ratio_limit=2.0,
        )
        self._step(world, frames=300)

        orientations = world.bodies.orientation.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertGreaterEqual(
            twist,
            min_a - 0.1,
            msg=f"lower limit failed: twist={twist:.3f} < {min_a} - slop",
        )
        self.assertLessEqual(
            twist,
            min_a + 0.2,
            msg=f"lower limit overshot back: twist={twist:.3f} > {min_a} + slop",
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
