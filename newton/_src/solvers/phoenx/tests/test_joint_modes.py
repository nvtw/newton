# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the unified ball-socket / revolute / prismatic joint.

A single :class:`JointDescriptor` with a ``mode`` field materialises
a 3-DoF ball-socket, a 5-DoF hinge (revolute), or a 5-DoF slider
(prismatic); revolute and prismatic additionally carry an optional
scalar actuator row on the free DoF (position or velocity drive +
one-sided spring-damper limits). All three modes share the same Warp
kernel and are exercised below back-to-back against the same single-
cube + static-anchor fixture:

Ball-socket mode
----------------
* **Anchor coincides.** After settling the single anchor on both
  bodies maps to the same world point -- the 3-row point lock holds.
* **Free rotation around any axis.** An initial spin about an
  arbitrary axis must be preserved (all 3 rotational DoF free).
* **Gravity pendulum.** With the cube offset from the anchor and
  gravity on, the cube swings as a pendulum -- the anchor holds, but
  the cube's centre of mass can move (under rotation about the anchor).
* **API validation.** Passing ``anchor2`` / ``drive_mode != OFF`` /
  ``min_value != 0`` etc. in ball-socket mode must raise.

Revolute mode
-------------
* **Anchors coincide.** With body 1 static, both anchor-pairs must map
  to the same world point after settling (the rank-5 positional lock
  holds).
* **Free axial spin survives.** Pure axial spin must persist: rotating
  about the hinge line moves neither anchor, so the lock applies no
  impulse.
* **Transverse spin is locked out.** An initial omega.x must decay to
  ~0 -- the locked 2 rotational DoF absorb it.
* **Position / velocity drive, drive-force cap, angular limits.**
  Same sub-features that the original actuated hinge exposed.

Prismatic mode
--------------
* **Free axial slide under gravity.** Gravity along the slide axis
  produces analytic free-fall along that axis only; lateral drift
  stays at zero.
* **Perpendicular gravity held.** Gravity orthogonal to the slide
  axis must not move the body -- the lateral lock absorbs the full
  weight.
* **Angular lock.** An initial spin about an arbitrary axis must
  decay to zero; the 3 angular DoF are locked.
* **Linear drive + limits.** Position / velocity drive and linear
  ``[min_value, max_value]`` limits on the slide DoF.

All scenes are registered with :func:`scene` so the test visualizer
can replay them interactively.
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
FPS = 60
SUBSTEPS = 2
SOLVER_ITERATIONS = 4
SETTLE_FRAMES = 60
HALF_EXTENT = 0.5
_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Both anchors lie on the joint axis (world +z) so the line through
# them defines the hinge / slide axis directly.
_ANCHOR1 = (0.0, 0.0, -HALF_EXTENT)
_ANCHOR2 = (0.0, 0.0, +HALF_EXTENT)


# ---------------------------------------------------------------------------
# Revolute scene builder
# ---------------------------------------------------------------------------


def _build_revolute_scene(
    device,
    *,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = False,
    cube_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drive_mode: DriveMode = DriveMode.OFF,
    target: float = 0.0,
    target_velocity: float = 0.0,
    max_force_drive: float = 0.0,
    # Default PD gains chosen for the unit-inertia cube: omega_0 ~= 14
    # rad/s (softer than the prior 8 Hz Box2D default but plenty
    # responsive at 60 Hz*2 substeps), mildly overdamped.
    stiffness_drive: float = 200.0,
    damping_drive: float = 20.0,
    # Default ``min > max`` disables the limit row (the sentinel the
    # unified joint uses; matches the standalone angular_limit one).
    # Individual tests that exercise the stop override both.
    min_value: float = 1.0,
    max_value: float = -1.0,
    hertz_limit: float = 30.0,
    damping_ratio_limit: float = 1.0,
):
    """Static body + one dynamic cube joined by a revolute joint.

    The hinge axis is world ``+z`` because the two anchors lie at
    ``(0, 0, +/- HALF_EXTENT)``. Cube COM defaults to the origin
    (anchor centre); pass ``cube_position=(0, -1, 0)`` for the
    gravity-pendulum scene.
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
    handle = b.add_joint(
        body1=anchor_body,
        body2=cube,
        anchor1=_ANCHOR1,
        anchor2=_ANCHOR2,
        mode=JointMode.REVOLUTE,
        drive_mode=drive_mode,
        target=target,
        target_velocity=target_velocity,
        max_force_drive=max_force_drive,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        min_value=min_value,
        max_value=max_value,
        hertz_limit=hertz_limit,
        damping_ratio_limit=damping_ratio_limit,
    )
    world = b.finalize(substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device)
    return world, handle


# ---------------------------------------------------------------------------
# Prismatic scene builder
# ---------------------------------------------------------------------------


def _build_prismatic_scene(
    device,
    *,
    axis: tuple[float, float, float],
    affected_by_gravity: bool = True,
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    drive_mode: DriveMode = DriveMode.OFF,
    target: float = 0.0,
    target_velocity: float = 0.0,
    max_force_drive: float = 0.0,
    # Unit-mass cube: omega_0 = sqrt(200) ~= 14 rad/s, ~70% damping.
    stiffness_drive: float = 200.0,
    damping_drive: float = 20.0,
    # ``min > max`` -> limit row disabled; see :func:`_build_revolute_scene`.
    min_value: float = 1.0,
    max_value: float = -1.0,
    hertz_limit: float = 30.0,
    damping_ratio_limit: float = 1.0,
):
    """Static body + one dynamic cube joined by a prismatic joint.

    The slide axis is whatever the caller passes in (must be unit
    length). We derive the two anchors as
    ``anchor1 = origin`` and ``anchor2 = anchor1 + axis`` so the
    rest length is ``1 m``, matching the convention the unified joint
    encourages (see :class:`JointDescriptor`).
    """
    anchor1 = (0.0, 0.0, 0.0)
    anchor2 = (
        anchor1[0] + axis[0],
        anchor1[1] + axis[1],
        anchor1[2] + axis[2],
    )
    b = WorldBuilder()
    anchor_body = b.add_static_body()
    cube = b.add_dynamic_body(
        position=(0.0, 0.0, 0.0),
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_joint(
        body1=anchor_body,
        body2=cube,
        anchor1=anchor1,
        anchor2=anchor2,
        mode=JointMode.PRISMATIC,
        drive_mode=drive_mode,
        target=target,
        target_velocity=target_velocity,
        max_force_drive=max_force_drive,
        stiffness_drive=stiffness_drive,
        damping_drive=damping_drive,
        min_value=min_value,
        max_value=max_value,
        hertz_limit=hertz_limit,
        damping_ratio_limit=damping_ratio_limit,
    )
    world = b.finalize(substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device)
    return world, handle


# ---------------------------------------------------------------------------
# Ball-socket scene builder
# ---------------------------------------------------------------------------


def _build_ball_socket_scene(
    device,
    *,
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cube_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    affected_by_gravity: bool = False,
):
    """Static body + one dynamic cube joined by a ball-socket.

    A single world-space anchor pins the two bodies together; all
    three rotational DoF are free. Pass ``cube_position != anchor``
    together with ``affected_by_gravity=True`` to get a pendulum.
    """
    b = WorldBuilder()
    anchor_body = b.add_static_body()
    cube = b.add_dynamic_body(
        position=cube_position,
        inverse_mass=1.0,
        inverse_inertia=_INV_INERTIA,
        affected_by_gravity=affected_by_gravity,
        velocity=initial_velocity,
        angular_velocity=initial_angular_velocity,
    )
    handle = b.add_joint(
        body1=anchor_body,
        body2=cube,
        anchor1=anchor,
        mode=JointMode.BALL_SOCKET,
    )
    world = b.finalize(substeps=SUBSTEPS, solver_iterations=SOLVER_ITERATIONS, device=device)
    return world, handle


def _scene_half_extents() -> np.ndarray:
    he = np.zeros((3, 3), dtype=np.float32)
    he[2] = HALF_EXTENT
    return he


# ---------------------------------------------------------------------------
# Scene registrations (visualizer fixtures)
# ---------------------------------------------------------------------------


@scene(
    "Joint (ball-socket): free 3D spin",
    description=(
        "Static-anchored cube pinned at one point, spinning freely about all "
        "three axes. Ball-socket has no preferred axis."
    ),
    tags=("joint", "ball_socket"),
)
def build_ball_socket_free_spin_scene(device) -> Scene:
    world, _ = _build_ball_socket_scene(device, initial_angular_velocity=(1.5, 0.7, 2.0))
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (ball-socket): gravity pendulum",
    description=(
        "Cube hanging below a static anchor, forming a pendulum that swings "
        "freely while the single anchor point stays fixed."
    ),
    tags=("joint", "ball_socket"),
)
def build_ball_socket_pendulum_scene(device) -> Scene:
    # Anchor at world origin on the cube's top face; cube COM one metre below.
    world, _ = _build_ball_socket_scene(
        device,
        anchor=(0.0, 0.0, 0.0),
        cube_position=(0.0, 0.0, -1.0),
        initial_velocity=(0.4, 0.0, 0.0),
        affected_by_gravity=True,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (revolute): free axial spin",
    description="Static-anchored cube spinning freely about the hinge axis.",
    tags=("joint", "revolute"),
)
def build_revolute_axial_spin_scene(device) -> Scene:
    world, _ = _build_revolute_scene(device, initial_angular_velocity=(0.0, 0.0, 2.0))
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (revolute): position drive",
    description=("Static-anchored cube driven to a +0.7 rad axial twist by a soft position drive."),
    tags=("joint", "revolute", "actuated"),
)
def build_revolute_position_drive_scene(device) -> Scene:
    world, _ = _build_revolute_scene(
        device,
        drive_mode=DriveMode.POSITION,
        target=0.7,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (revolute): velocity drive",
    description=("Static-anchored cube driven to 1.5 rad/s axial spin by a soft velocity drive (max_force=20 N*m)."),
    tags=("joint", "revolute", "actuated"),
)
def build_revolute_velocity_drive_scene(device) -> Scene:
    world, _ = _build_revolute_scene(
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
    "Joint (revolute): limit clamp",
    description=(
        "Cube spinning into the upper +0.5 rad stop; the unilateral spring-damper must clamp the twist at ~+0.5 rad."
    ),
    tags=("joint", "revolute", "actuated"),
)
def build_revolute_limit_scene(device) -> Scene:
    world, _ = _build_revolute_scene(
        device,
        initial_angular_velocity=(0.0, 0.0, 0.6),
        min_value=-0.5,
        max_value=0.5,
        hertz_limit=10.0,
        damping_ratio_limit=2.0,
    )
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (prismatic): vertical slider (gravity along axis)",
    description="Static body + cube on a +y prismatic; cube falls freely along +y.",
    tags=("joint", "prismatic"),
)
def build_prismatic_vertical_slider_scene(device) -> Scene:
    world, _ = _build_prismatic_scene(device, axis=(0.0, 1.0, 0.0))
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


@scene(
    "Joint (prismatic): horizontal slider (gravity perpendicular to axis)",
    description="Cube on a +x prismatic with gravity along -y; the lateral lock holds it.",
    tags=("joint", "prismatic"),
)
def build_prismatic_horizontal_slider_scene(device) -> Scene:
    world, _ = _build_prismatic_scene(device, axis=(1.0, 0.0, 0.0))
    return Scene(
        world=world,
        body_half_extents=_scene_half_extents(),
        frame_dt=1.0 / FPS,
        substeps=SUBSTEPS,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _axial_twist(orientation: np.ndarray) -> float:
    """Return the signed axial twist of ``orientation`` about world +z.

    The cube starts with identity orientation. Body 1 is static at
    identity, so the relative twist about world +z reduces to the
    axial component of body 2's quaternion in xyzw form: ``2 *
    atan2(z, w)`` (standard z-axis swing-twist decomposition for the
    case where the swing is ~zero, which it must be because the
    revolute lock kills the off-axis rotational DoF).
    """
    _x, _y, z, w = orientation
    return 2.0 * math.atan2(z, w)


def _run_settle_loop(world, frames: int) -> None:
    """Thin wrapper that fixes the timestep at ``1/FPS`` for this file.

    Delegates to :func:`newton._src.solvers.phoenx.tests._test_helpers.run_settle_loop`,
    which takes care of the CUDA graph-capture fast path.
    """
    run_settle_loop(world, frames, dt=1.0 / FPS)


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


def _quat_relative_angle(q):
    """Return the rotation angle of unit quaternion ``q`` (xyzw) [rad]."""
    w = abs(float(q[3]))
    w = min(1.0, max(-1.0, w))
    return 2.0 * math.acos(w)


# ---------------------------------------------------------------------------
# Revolute mode tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestJointRevolute(unittest.TestCase):
    """Revolute-mode physics checks for :meth:`WorldBuilder.add_joint`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        _run_settle_loop(world, frames)

    def test_anchors_coincide(self):
        """Both anchor pairs must map to the same world position.

        A drift here means the rank-5 lock has lost rows.
        """
        device = wp.get_preferred_device()
        world, _ = _build_revolute_scene(device, initial_angular_velocity=(0.5, 0.0, 0.0))
        self._step(world)

        positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()
        cube = 2

        a1_b1 = np.asarray(_ANCHOR1, dtype=np.float64)
        a2_b1 = np.asarray(_ANCHOR2, dtype=np.float64)
        a1_b2 = positions[cube] + _quat_rotate(orientations[cube], _ANCHOR1)
        a2_b2 = positions[cube] + _quat_rotate(orientations[cube], _ANCHOR2)

        self.assertLess(np.linalg.norm(a1_b2 - a1_b1), 0.02)
        self.assertLess(np.linalg.norm(a2_b2 - a2_b1), 0.02)

    def test_free_axial_rotation(self):
        """Pure axial spin must survive untouched."""
        device = wp.get_preferred_device()
        omega_axial = 2.0
        world, _ = _build_revolute_scene(device, initial_angular_velocity=(0.0, 0.0, omega_axial))
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(
            omegas[cube, 2],
            omega_axial,
            delta=0.05,
            msg=f"axial spin bled to {omegas[cube, 2]}",
        )
        self.assertLess(math.hypot(omegas[cube, 0], omegas[cube, 1]), 0.02)

    def test_perpendicular_spin_is_locked_out(self):
        """Transverse omega.x must decay to ~0 -- nowhere for it to go."""
        device = wp.get_preferred_device()
        world, _ = _build_revolute_scene(device, initial_angular_velocity=(2.0, 0.0, 0.0))
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertLess(abs(omegas[cube, 0]), 0.05)

    def test_hangs_under_gravity(self):
        """Gravity-loaded cube centred on the hinge axis must not drift."""
        device = wp.get_preferred_device()
        world, _ = _build_revolute_scene(device, affected_by_gravity=True)
        self._step(world, frames=150)

        positions = world.bodies.position.numpy()
        cube = 2
        self.assertLess(np.linalg.norm(positions[cube]), 0.02)

    def test_off_mode_preserves_axial_spin(self):
        """``DriveMode.OFF`` with no limit must not bleed axial spin."""
        device = wp.get_preferred_device()
        omega_axial = 2.0
        world, _ = _build_revolute_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, omega_axial),
            drive_mode=DriveMode.OFF,
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(omegas[cube, 2], omega_axial, delta=0.05)
        self.assertLess(math.hypot(omegas[cube, 0], omegas[cube, 1]), 0.02)

    def test_position_drive_reaches_target(self):
        """Position drive must seat the cube near ``target``."""
        device = wp.get_preferred_device()
        target = 0.7
        world, _ = _build_revolute_scene(
            device,
            drive_mode=DriveMode.POSITION,
            target=target,
        )
        self._step(world)

        orientations = world.bodies.orientation.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertAlmostEqual(twist, target, delta=0.05)

        omegas = world.bodies.angular_velocity.numpy()
        self.assertLess(np.linalg.norm(omegas[cube]), 0.05)

    def test_position_drive_holds_against_initial_spin(self):
        """Position drive must damp an initial spin back to ``target``.

        Gives the cube a sizeable initial axial angular velocity and
        verifies the position drive pulls it back to the setpoint and
        then holds there. This exercises the drive's role as an
        implicit PD controller under a kinetic perturbation -- exactly
        the scenario that motivated making ``max_force_drive`` clamp
        POSITION-mode impulses, because without the clamp an
        aggressive spin could push the PGS accumulator arbitrarily
        far before the spring ever catches up.
        """
        device = wp.get_preferred_device()
        target = 0.0
        world, _ = _build_revolute_scene(
            device,
            drive_mode=DriveMode.POSITION,
            target=target,
            initial_angular_velocity=(0.0, 0.0, 2.5),
        )
        self._step(world, frames=90)

        orientations = world.bodies.orientation.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertAlmostEqual(twist, target, delta=0.05)
        self.assertLess(abs(omegas[cube, 2]), 0.05)

    def test_position_drive_respects_max_force(self):
        """``max_force_drive`` must cap the POSITION-drive impulse.

        With a tiny torque cap a POSITION drive physically cannot
        exert enough torque to reach a far target -- the cap means
        the per-substep impulse clamps at ``max_force_drive * dt``
        regardless of the PD error, mirroring the well-known "stall
        at the torque limit" behaviour of a real servo.

        Previously ``max_force_drive`` was silently ignored in
        POSITION mode and the drive saturated the target regardless
        of the cap; this test guards against that regression.
        """
        device = wp.get_preferred_device()
        target = 1.5
        world, _ = _build_revolute_scene(
            device,
            drive_mode=DriveMode.POSITION,
            target=target,
            max_force_drive=0.05,
            stiffness_drive=200.0,
            damping_drive=20.0,
        )
        self._step(world, frames=60)

        orientations = world.bodies.orientation.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertLess(
            twist,
            target * 0.5,
            msg=f"twist={twist} should stall well below target={target} with tiny max_force_drive",
        )

    def test_velocity_drive_reaches_target(self):
        """Velocity drive must spin the cube up to ``target_velocity``."""
        device = wp.get_preferred_device()
        target = 1.5
        world, _ = _build_revolute_scene(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target,
            max_force_drive=50.0,
        )
        self._step(world, frames=30)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertAlmostEqual(omegas[cube, 2], target, delta=0.1)
        self.assertLess(math.hypot(omegas[cube, 0], omegas[cube, 1]), 0.02)

    def test_velocity_drive_force_cap_saturates(self):
        """A tiny ``max_force_drive`` must cap the steady-state spin."""
        device = wp.get_preferred_device()
        target = 5.0
        world, _ = _build_revolute_scene(
            device,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target,
            max_force_drive=0.1,
        )
        self._step(world, frames=30)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        self.assertLess(abs(omegas[cube, 2]), target * 0.5)
        self.assertGreater(omegas[cube, 2], 0.001)

    def test_upper_limit_clamps_twist(self):
        """Spin into the upper stop must clamp at ~``max_value``."""
        device = wp.get_preferred_device()
        max_v = 0.5
        world, _ = _build_revolute_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, 0.5),
            min_value=-2.0,
            max_value=max_v,
            hertz_limit=10.0,
            damping_ratio_limit=2.0,
        )
        self._step(world, frames=75)

        orientations = world.bodies.orientation.numpy()
        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertLessEqual(twist, max_v + 0.1)
        self.assertGreaterEqual(twist, max_v - 0.2)
        self.assertLess(abs(omegas[cube, 2]), 0.2)

    def test_lower_limit_clamps_twist(self):
        """Symmetric: spin into the lower stop must clamp at ~``min_value``."""
        device = wp.get_preferred_device()
        min_v = -0.5
        world, _ = _build_revolute_scene(
            device,
            initial_angular_velocity=(0.0, 0.0, -0.5),
            min_value=min_v,
            max_value=2.0,
            hertz_limit=10.0,
            damping_ratio_limit=2.0,
        )
        self._step(world, frames=75)

        orientations = world.bodies.orientation.numpy()
        cube = 2
        twist = _axial_twist(orientations[cube])
        self.assertGreaterEqual(twist, min_v - 0.1)
        self.assertLessEqual(twist, min_v + 0.2)


# ---------------------------------------------------------------------------
# Prismatic mode tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestJointPrismatic(unittest.TestCase):
    """Prismatic-mode physics checks for :meth:`WorldBuilder.add_joint`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        _run_settle_loop(world, frames)

    def test_free_axial_slide_under_gravity(self):
        """Slide axis along gravity -> free fall along the axis only."""
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(device, axis=(0.0, 1.0, 0.0))

        frames = 120  # 2 s
        t = frames / FPS
        self._step(world, frames=frames)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        cube = 2

        expected_y = -0.5 * GRAVITY * t * t
        self.assertAlmostEqual(
            positions[cube, 1],
            expected_y,
            delta=0.05 * abs(expected_y),
        )
        self.assertLess(abs(positions[cube, 0]), 0.02)
        self.assertLess(abs(positions[cube, 2]), 0.02)
        self.assertLess(abs(velocities[cube, 0]) + abs(velocities[cube, 2]), 0.05)

    def test_perpendicular_gravity_held(self):
        """Slide axis perpendicular to gravity -> body holds in place."""
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(device, axis=(1.0, 0.0, 0.0))
        self._step(world)

        positions = world.bodies.position.numpy()
        cube = 2
        self.assertLess(np.linalg.norm(positions[cube]), 0.05)

    def test_angular_lock_holds(self):
        """Initial angular velocity must be killed by the angular lock."""
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

        self.assertLess(np.linalg.norm(omegas[cube]), 0.05)
        rot_angle = _quat_relative_angle(orientations[cube])
        self.assertLess(rot_angle, math.radians(2.0))

    def test_position_drive_reaches_target(self):
        """Linear position drive must seat the cube near ``target`` [m].

        The slide axis is world +x; the drive pulls the cube towards
        ``target`` metres from the ``anchor1`` position along that
        axis. After settling, the cube's x coordinate must land at
        approximately ``target``.
        """
        device = wp.get_preferred_device()
        target = 0.4
        world, _ = _build_prismatic_scene(
            device,
            axis=(1.0, 0.0, 0.0),
            affected_by_gravity=False,
            drive_mode=DriveMode.POSITION,
            target=target,
        )
        self._step(world, frames=75)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        cube = 2
        self.assertAlmostEqual(positions[cube, 0], target, delta=0.05)
        self.assertLess(np.linalg.norm(velocities[cube]), 0.05)

    def test_velocity_drive_reaches_target(self):
        """Linear velocity drive must push cube to ``target_velocity`` [m/s]."""
        device = wp.get_preferred_device()
        target = 0.6
        world, _ = _build_prismatic_scene(
            device,
            axis=(1.0, 0.0, 0.0),
            affected_by_gravity=False,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=target,
            max_force_drive=50.0,
        )
        self._step(world, frames=30)

        velocities = world.bodies.velocity.numpy()
        cube = 2
        self.assertAlmostEqual(velocities[cube, 0], target, delta=0.1)
        self.assertLess(abs(velocities[cube, 1]) + abs(velocities[cube, 2]), 0.05)

    def test_position_drive_holds_against_gravity(self):
        """Position drive must hold the slide at ``target`` against gravity.

        The slide axis is vertical (``+y``) and gravity pulls the
        cube down along it. A stiff position drive at ``target=0``
        must keep the cube near the anchor -- no free-fall -- and
        stall at a small static offset where the spring torque
        balances gravity.
        """
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(
            device,
            axis=(0.0, 1.0, 0.0),
            affected_by_gravity=True,
            drive_mode=DriveMode.POSITION,
            target=0.0,
            max_force_drive=50.0,
            stiffness_drive=200.0,
            damping_drive=20.0,
        )
        self._step(world, frames=90)

        positions = world.bodies.position.numpy()
        velocities = world.bodies.velocity.numpy()
        cube = 2
        self.assertLess(abs(positions[cube, 1]), 0.1)
        self.assertLess(abs(positions[cube, 0]) + abs(positions[cube, 2]), 0.02)
        self.assertLess(np.linalg.norm(velocities[cube]), 0.1)

    def test_position_drive_respects_max_force(self):
        """``max_force_drive`` must cap the POSITION-drive impulse.

        The slide axis is vertical; with a tiny force cap the drive
        physically cannot fight gravity -- the cube must drop well
        below its ``target=0`` setpoint. Guards against the earlier
        defect where POSITION mode ignored ``max_force_drive``
        entirely.
        """
        device = wp.get_preferred_device()
        world, _ = _build_prismatic_scene(
            device,
            axis=(0.0, 1.0, 0.0),
            affected_by_gravity=True,
            drive_mode=DriveMode.POSITION,
            target=0.0,
            max_force_drive=0.5,
            stiffness_drive=500.0,
            damping_drive=20.0,
        )
        self._step(world, frames=30)

        positions = world.bodies.position.numpy()
        cube = 2
        self.assertLess(
            positions[cube, 1],
            -0.3,
            msg=f"cube.y={positions[cube, 1]} should fall far below 0 under tiny max_force_drive",
        )

    def test_upper_limit_clamps_slide(self):
        """Slide into the upper stop must clamp at ~``max_value`` [m]."""
        device = wp.get_preferred_device()
        max_v = 0.3
        # Push the cube in +x with an initial velocity; the one-sided
        # linear spring must seat it at ~max_v.
        world, _ = _build_prismatic_scene(
            device,
            axis=(1.0, 0.0, 0.0),
            affected_by_gravity=False,
            initial_angular_velocity=(0.0, 0.0, 0.0),
            min_value=-2.0,
            max_value=max_v,
            hertz_limit=10.0,
            damping_ratio_limit=2.0,
            drive_mode=DriveMode.VELOCITY,
            target_velocity=0.3,
            max_force_drive=50.0,
        )
        self._step(world, frames=100)

        positions = world.bodies.position.numpy()
        cube = 2
        self.assertLessEqual(positions[cube, 0], max_v + 0.1)
        self.assertGreaterEqual(positions[cube, 0], max_v - 0.2)


# ---------------------------------------------------------------------------
# Ball-socket mode tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Jitter simulation tests run on CUDA only (graph capture is required for reasonable run-time).",
)
class TestJointBallSocket(unittest.TestCase):
    """Ball-socket-mode physics checks for :meth:`WorldBuilder.add_joint`."""

    def _step(self, world, frames=SETTLE_FRAMES):
        _run_settle_loop(world, frames)

    def test_anchor_coincides(self):
        """The single anchor must map to the same world point on both bodies.

        A drift here means the 3-row point lock has lost rows.
        """
        device = wp.get_preferred_device()
        anchor = (0.0, 0.0, 0.0)
        # Start the cube off-anchor and with a spin to stress the lock.
        world, _ = _build_ball_socket_scene(
            device,
            anchor=anchor,
            cube_position=(0.0, 0.0, 0.0),
            initial_angular_velocity=(1.0, 0.5, 2.0),
            affected_by_gravity=True,
        )
        self._step(world)

        positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()
        cube = 2
        # Anchor in body-1 (static) frame is just its world position.
        a_b1 = np.asarray(anchor, dtype=np.float64)
        # Anchor in cube frame, rotated into world.
        a_local_b2 = np.asarray(anchor, dtype=np.float64) - np.zeros(3)
        a_b2 = positions[cube] + _quat_rotate(orientations[cube], a_local_b2)
        self.assertLess(
            np.linalg.norm(a_b2 - a_b1),
            0.02,
            msg=f"anchor drift {np.linalg.norm(a_b2 - a_b1)} exceeds 0.02 m",
        )

    def test_free_rotation_survives(self):
        """An initial 3D spin must be preserved (all rotational DoF free).

        Without gravity and with the cube COM at the anchor, no torques
        act on the cube, so all three components of the initial angular
        velocity must persist.
        """
        device = wp.get_preferred_device()
        omega0 = (1.2, 0.7, 2.0)
        world, _ = _build_ball_socket_scene(
            device,
            anchor=(0.0, 0.0, 0.0),
            cube_position=(0.0, 0.0, 0.0),
            initial_angular_velocity=omega0,
            affected_by_gravity=False,
        )
        self._step(world)

        omegas = world.bodies.angular_velocity.numpy()
        cube = 2
        for i, name in enumerate(("x", "y", "z")):
            self.assertAlmostEqual(
                omegas[cube, i],
                omega0[i],
                delta=0.05,
                msg=(f"angular velocity.{name} bled from {omega0[i]} to {omegas[cube, i]}"),
            )

    def test_pendulum_cm_swings_but_anchor_holds(self):
        """With gravity and an offset COM, the anchor must hold and the cube must move.

        Classic pendulum test: anchor at the origin, cube COM 1 m below.
        Gravity swings the cube around the anchor -- we assert (a) the
        anchor-on-cube stays near the world origin, and (b) the cube
        COM has a non-trivial horizontal velocity / displacement after
        settling (it is not frozen in place).
        """
        device = wp.get_preferred_device()
        anchor = (0.0, 0.0, 0.0)
        world, _ = _build_ball_socket_scene(
            device,
            anchor=anchor,
            cube_position=(0.0, 0.0, -1.0),
            initial_velocity=(0.5, 0.0, 0.0),
            affected_by_gravity=True,
        )
        self._step(world, frames=60)

        positions = world.bodies.position.numpy()
        orientations = world.bodies.orientation.numpy()
        cube = 2
        # Anchor in cube frame is the world anchor minus the cube's
        # initial (finalize-time) position rotated back into cube frame.
        # With the initial orientation being identity and the initial
        # cube position (0, 0, -1), the anchor in cube local is (0, 0, 1).
        anchor_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        a_b2 = positions[cube] + _quat_rotate(orientations[cube], anchor_local)
        self.assertLess(
            np.linalg.norm(a_b2 - np.asarray(anchor, dtype=np.float64)),
            0.05,
            msg=(f"pendulum anchor drifted to {a_b2} (expected near origin); ball-socket lock is leaking"),
        )
        # The cube should have swung off the vertical -- either x-coordinate
        # or x-velocity must be non-trivial.
        vel = world.bodies.velocity.numpy()[cube]
        moved = abs(positions[cube, 0]) > 0.05 or abs(vel[0]) > 0.05
        self.assertTrue(
            moved,
            msg=(
                f"pendulum did not swing: pos={positions[cube]}, vel={vel}. Ball-socket is over-constraining rotation."
            ),
        )

    def test_rejects_anchor2(self):
        """Passing ``anchor2`` in ball-socket mode must raise."""
        b = WorldBuilder()
        b.add_static_body()
        b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_joint(
                body1=0,
                body2=1,
                anchor1=(0.0, 0.0, 0.0),
                anchor2=(1.0, 0.0, 0.0),
                mode=JointMode.BALL_SOCKET,
            )

    def test_rejects_drive_mode(self):
        """Passing ``drive_mode != OFF`` in ball-socket mode must raise."""
        b = WorldBuilder()
        b.add_static_body()
        b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_joint(
                body1=0,
                body2=1,
                anchor1=(0.0, 0.0, 0.0),
                mode=JointMode.BALL_SOCKET,
                drive_mode=DriveMode.POSITION,
            )

    def test_rejects_limits(self):
        """Passing non-zero limits in ball-socket mode must raise."""
        b = WorldBuilder()
        b.add_static_body()
        b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_joint(
                body1=0,
                body2=1,
                anchor1=(0.0, 0.0, 0.0),
                mode=JointMode.BALL_SOCKET,
                min_value=-0.5,
                max_value=0.5,
            )

    def test_requires_anchor2_for_revolute(self):
        """Revolute mode without ``anchor2`` must raise (symmetric check)."""
        b = WorldBuilder()
        b.add_static_body()
        b.add_dynamic_body()
        with self.assertRaises(ValueError):
            b.add_joint(
                body1=0,
                body2=1,
                anchor1=(0.0, 0.0, 0.0),
                mode=JointMode.REVOLUTE,
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
