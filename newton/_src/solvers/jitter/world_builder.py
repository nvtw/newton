# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side builder that assembles a :class:`World` from plain Python.

Construction is split in two phases:

1. *Append* phase -- ``add_static_body`` / ``add_dynamic_body`` /
   ``add_ball_socket`` / ... record per-body / per-constraint
   *descriptors* in plain Python lists. Nothing touches the GPU.
2. *Finalize* phase -- :meth:`WorldBuilder.finalize` allocates a
   :class:`BodyContainer` and a :class:`ConstraintContainer` sized
   exactly to the appended counts, packs every descriptor into the SoA
   body arrays via NumPy, computes the per-constraint-type cid ranges,
   uploads per-type batch arrays, and launches one initialise kernel
   per type to fill the column-major constraint storage. Returns a
   ready-to-step :class:`World`.

This keeps the user-facing API trivial (regular Python objects, no
Warp types until the very end) while still hitting the GPU in batched
launches at finalize time.

Body 0 is always the static world body (created in ``__init__``); user
descriptor indices start at 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.jitter.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    actuated_double_ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_angular_motor import (
    AM_DWORDS,
    angular_motor_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_ball_socket import (
    BS_DWORDS,
    ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_ANGULAR,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
    DEFAULT_HERTZ_MOTOR,
    ConstraintContainer,
    constraint_container_zeros,
)
from newton._src.solvers.jitter.constraint_d6 import (
    D6_DWORDS,
    d6_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_double_ball_socket import (
    DBS_DWORDS,
    double_ball_socket_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_hinge_angle import (
    HA_DWORDS,
    hinge_angle_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_hinge_joint import (
    HJ_DWORDS,
    hinge_joint_initialize_kernel,
)
from newton._src.solvers.jitter.constraint_prismatic import (
    PR_DWORDS,
    prismatic_initialize_kernel,
)
from newton._src.solvers.jitter.solver_jitter import World

__all__ = [
    "ActuatedDoubleBallSocketHingeDescriptor",
    "ActuatedDoubleBallSocketHingeHandle",
    "AngularMotorDescriptor",
    "BallSocketDescriptor",
    "D6AxisDrive",
    "D6Descriptor",
    "D6Handle",
    "DoubleBallSocketHingeDescriptor",
    "DoubleBallSocketHingeHandle",
    "DriveMode",
    "HingeAngleDescriptor",
    "HingeJointDescriptor",
    "HingeJointHandle",
    "PrismaticDescriptor",
    "PrismaticHandle",
    "RigidBodyDescriptor",
    "WorldBuilder",
]


_IDENTITY_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass
class RigidBodyDescriptor:
    """Plain-Python description of one rigid body.

    Mirrors the writable subset of Jitter2's ``RigidBody`` constructor
    arguments. Defaults match Jitter (no damping, gravity on, identity
    inertia) so a bare ``RigidBodyDescriptor()`` produces a
    static-by-default body sitting at the origin.

    Units (where applicable):
        * ``position`` [m]
        * ``inverse_mass`` [1/kg]
        * ``inverse_inertia`` [1/(kg m^2)] in the *body* frame; the
          solver rotates this into world space every step.
        * ``velocity`` [m/s]
        * ``angular_velocity`` [rad/s]
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    motion_type: int = int(MOTION_STATIC)
    inverse_mass: float = 0.0
    inverse_inertia: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = _IDENTITY_INERTIA
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    affected_by_gravity: bool = True
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BallSocketDescriptor:
    """Plain-Python description of one ball-and-socket constraint.

    The ``anchor`` is a single point in *world* space; finalize() asks
    each body to express it in its own local frame (mirrors
    ``BallSocket.Initialize``). ``hertz`` and ``damping_ratio`` follow
    the Box2D v3 / Bepu soft-constraint formulation: ``hertz`` is the
    undamped natural frequency [Hz] of the joint as a virtual spring,
    ``damping_ratio`` is non-dimensional (1 = critically damped). See
    :func:`soft_constraint_coefficients`.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class HingeAngleDescriptor:
    """Plain-Python description of one hinge-angle (2-DoF) constraint.

    The ``axis`` is in *world* space at construction time; finalize()
    transforms it into body 2's local frame, mirroring
    ``HingeAngle.Initialize``. ``min_angle`` / ``max_angle`` are in
    radians.

    ``hertz_lock`` / ``damping_ratio_lock`` parametrise the perpendicular
    -to-hinge angular lock (rows 0 / 1 of the Jacobian) as a Box2D / Bepu
    soft-constraint pair; ``hertz_limit`` / ``damping_ratio_limit`` do
    the same for the axial min / max limit (row 2). See
    :func:`soft_constraint_coefficients` for the formulation.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    min_angle: float = -math.pi
    max_angle: float = math.pi
    hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO)
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class AngularMotorDescriptor:
    """Plain-Python description of one angular-velocity motor constraint.

    The motor drives the relative angular velocity along ``axis`` (world
    space at construction time) toward ``target_velocity`` [rad/s] using
    at most ``max_force`` [N·m] of torque. ``max_force = 0`` (Jitter
    default) yields a *disabled* motor that does nothing.

    ``hertz`` / ``damping_ratio`` follow the Box2D v3 / Bepu soft-
    constraint formulation (see :func:`soft_constraint_coefficients`).
    For a velocity motor ``hertz`` controls how aggressively the
    accumulated impulse decays toward the velocity setpoint -- higher
    values give a stiffer motor that tracks the target more tightly.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    target_velocity: float = 0.0
    max_force: float = 0.0
    hertz: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class HingeJointDescriptor:
    """Plain-Python description of one fused hinge-joint constraint.

    A Jitter-style hinge joint composed of a HingeAngle (2-DoF angular
    lock, optionally with limits), a BallSocket (3-DoF positional lock),
    and an AngularMotor (drives the remaining axial DoF). All three
    sub-constraints live in *one* column of the shared
    :class:`ConstraintContainer` and are owned by a single PGS thread
    -- see :mod:`newton._src.solvers.jitter.constraint_hinge_joint` for
    why this fuses better than three separate constraints.

    ``hinge_center`` and ``hinge_axis`` are in *world* space at
    finalize() time. Set ``max_force = 0.0`` for a passive (unmotorised)
    joint -- the AngularMotor sub then applies zero corrective impulse
    and acts as a no-op.
    """

    body1: int
    body2: int
    hinge_center: tuple[float, float, float]
    hinge_axis: tuple[float, float, float]
    min_angle: float = -math.pi
    max_angle: float = math.pi
    target_velocity: float = 0.0
    max_force: float = 0.0
    hertz_linear: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO)
    hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO)
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)
    hertz_motor: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio_motor: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class HingeJointHandle:
    """Global cid of the fused hinge-joint constraint created by
    :meth:`WorldBuilder.add_hinge_joint`.

    Returned with a sentinel cid (``-1``) at the time of the
    ``add_hinge_joint`` call and rewritten in place by
    :meth:`WorldBuilder.finalize` to the actual cid in the shared
    :class:`ConstraintContainer`. Pass it to
    :meth:`World.gather_constraint_wrenches` (indirectly via
    ``wrenches[handle.cid]``) to read the fused joint's combined
    reaction wrench -- the sum of its BallSocket / HingeAngle /
    AngularMotor sub-contributions on body 2.
    """

    cid: int = -1


@dataclass
class DoubleBallSocketHingeDescriptor:
    """Plain-Python description of one fused two-anchor (Schur-complement)
    hinge constraint.

    Locks 5 DoF (3 translational + 2 rotational) by anchoring two
    points -- ``anchor1`` and ``anchor2`` -- on the line of the hinge
    axis between the two bodies. The third rotational DoF (rotation
    about the line through both anchors) stays free. Solved as one
    rank-5 column via a 3x3 + 2x2 Schur complement, no quaternion math
    and no parameter tuning, see
    :mod:`newton._src.solvers.jitter.constraint_double_ball_socket`.

    ``anchor1`` and ``anchor2`` are in *world* space at finalize() time
    and define the hinge axis as ``anchor2 - anchor1``.

    ``hertz`` / ``damping_ratio`` follow the Box2D v3 / Bepu soft-
    constraint formulation (see :func:`soft_constraint_coefficients`).
    A single pair governs both anchor blocks of the Schur solve since
    they're modelling the same physical hinge.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class DoubleBallSocketHingeHandle:
    """Global cid of the fused double-ball-socket hinge constraint
    created by :meth:`WorldBuilder.add_double_ball_socket_hinge`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's reaction wrench on
    body 2.
    """

    cid: int = -1


class DriveMode(IntEnum):
    """Drive mode for an actuated joint's free DoF.

    See :mod:`newton._src.solvers.jitter.constraint_actuated_double_ball_socket`
    for the per-mode formulation. Values match the underlying
    ``DRIVE_MODE_*`` Warp constants exactly so they can be passed
    through ``np.int32`` arrays without translation.
    """

    OFF = int(DRIVE_MODE_OFF)
    POSITION = int(DRIVE_MODE_POSITION)
    VELOCITY = int(DRIVE_MODE_VELOCITY)


@dataclass
class ActuatedDoubleBallSocketHingeDescriptor:
    """Plain-Python description of one *actuated* fused two-anchor hinge.

    Identical 5-DoF positional lock as
    :class:`DoubleBallSocketHingeDescriptor` (3 translational +
    2 rotational, solved as a 3x3 + 2x2 Schur complement) plus a
    scalar PGS row that controls the free axial DoF with either a
    soft *position* or *velocity* drive and -- independently -- clamps
    the relative axial twist to ``[min_angle, max_angle]`` via a
    one-sided spring-damper.

    ``anchor1`` and ``anchor2`` are in *world* space at finalize() time
    and define the hinge axis as ``anchor2 - anchor1``.

    All ``hertz`` / ``damping_ratio`` parameters follow the Box2D v3 /
    Bepu / Nordby soft-constraint formulation; see
    :func:`soft_constraint_coefficients`.

    See :mod:`newton._src.solvers.jitter.constraint_actuated_double_ball_socket`
    for the per-row math.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    drive_mode: DriveMode = DriveMode.OFF
    target_angle: float = 0.0
    target_velocity: float = 0.0
    max_force_drive: float = 0.0
    hertz_drive: float = float(DEFAULT_HERTZ_MOTOR)
    damping_ratio_drive: float = float(DEFAULT_DAMPING_RATIO)
    min_angle: float = 0.0
    max_angle: float = 0.0
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class ActuatedDoubleBallSocketHingeHandle:
    """Global cid of the actuated fused-hinge constraint created by
    :meth:`WorldBuilder.add_actuated_double_ball_socket_hinge`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's reaction wrench on
    body 2.
    """

    cid: int = -1


@dataclass
class PrismaticDescriptor:
    """Plain-Python description of one prismatic (sliding) joint.

    Locks 5 DoF (3 rotational + 2 translational): the bodies must keep
    their initial relative orientation and may only translate along the
    user-supplied slide axis. Solved as one rank-5 column via a 3x3 +
    2x2 Schur complement -- see
    :mod:`newton._src.solvers.jitter.constraint_prismatic` for the
    derivation.

    ``anchor`` is a single point in *world* space at finalize() time
    (snapshotted into each body's local frame); both lever arms
    coincide at rest by construction. ``axis`` is a *world*-space
    direction at finalize() time, snapshotted into body 1's local
    frame so the slide direction "rides" body 1 rigidly (matches the
    Bullet/ODE convention). ``axis`` need not be unit length -- the
    init kernel normalises it.

    ``hertz_*`` / ``damping_ratio_*`` follow the Box2D v3 / Bepu soft-
    constraint formulation (see :func:`soft_constraint_coefficients`).
    Two independent knob pairs because the angular and linear blocks
    are physically distinct: tweak ``hertz_angular`` /
    ``damping_ratio_angular`` to soften the rotational lock and
    ``hertz_linear`` / ``damping_ratio_linear`` to soften the
    perpendicular-translation lock.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    hertz_angular: float = float(DEFAULT_HERTZ_ANGULAR)
    damping_ratio_angular: float = float(DEFAULT_DAMPING_RATIO)
    hertz_linear: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO)


@dataclass
class PrismaticHandle:
    """Global cid of the prismatic joint created by
    :meth:`WorldBuilder.add_prismatic`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's reaction wrench on
    body 2 (linear constraint force + angular reaction torque, both in
    world frame).
    """

    cid: int = -1


@dataclass
class D6AxisDrive:
    """Per-axis configuration for one axis of a :class:`D6Descriptor`.

    Each of the 6 axes (3 angular + 3 linear) is independently
    configurable as:

    * **Rigid lock**       -- ``hertz=0`` (default) and
      ``max_force=math.inf`` (default). The default
      ``D6AxisDrive()`` is a rigid lock; an unconfigured D6 is a
      6-DoF rigid weld.
    * **Soft lock**        -- ``hertz>0``, ``target_position=0``,
      ``target_velocity=0``, ``max_force=inf``. A spring-and-damper
      restoring the rest pose with the Box2D / Bepu / Nordby implicit
      formulation.
    * **Position drive**   -- ``hertz>0``, ``target_position!=0``,
      ``max_force`` finite. Implicit-PD drive that pulls the axis
      towards the target with the given soft-spring stiffness, capped
      at ``max_force``.
    * **Velocity drive**   -- ``hertz>0``, ``target_velocity!=0``,
      ``max_force`` finite. Implicit-PD velocity-tracking drive (the
      "motor" mode); ``hertz`` controls how aggressively the motor
      tracks the setpoint.
    * **Free**             -- ``max_force=0``. The accumulated impulse
      is pinned to zero every PGS iteration regardless of ``hertz``;
      this is the canonical way to mark an axis as un-constrained.

    Position + velocity targets compose: setting both gives an
    over-damped spring with a steady-state velocity (useful for
    e.g. "open the door at 1 rad/s while pulling it towards 90°").

    Units (per axis):
        * Angular axes: ``target_position`` [rad], ``target_velocity``
          [rad/s], ``max_force`` [N*m].
        * Linear axes:  ``target_position`` [m],   ``target_velocity``
          [m/s],   ``max_force`` [N].
        * ``hertz`` is always [Hz]; ``damping_ratio`` is dimensionless
          (1 = critically damped).
    """

    hertz: float = 0.0
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    target_position: float = 0.0
    target_velocity: float = 0.0
    max_force: float = math.inf


_D6_RIGID_LOCK = D6AxisDrive()


@dataclass
class D6Descriptor:
    """Plain-Python description of one 6-DoF generalised ("D6") joint.

    A single D6 owns *all* 6 relative DoF of body 2 with respect to
    body 1 and configures each axis independently via a
    :class:`D6AxisDrive`. The default ``D6Descriptor`` (no per-axis
    overrides) is a 6-DoF rigid weld at ``anchor``; pass non-default
    drives to soften axes, add PD setpoints, cap forces, or free
    individual axes.

    Solved as one column with a 6x6 Schur complement (3x3 + 3x3 + 3x3
    cross-block) -- see
    :mod:`newton._src.solvers.jitter.constraint_d6` for the math.

    The 3 angular axes and 3 linear axes are interpreted in *body 1's
    local frame*: the joint frame "rides" body 1 rigidly. Position
    targets are folded into the rest pose at finalize() time so the
    runtime kernel never branches on "is there a position target".

    ``anchor`` is in *world* space at finalize() time. The init
    kernel snapshots it into both bodies' local frames; the linear
    position targets are added on top of the body-1-local anchor so
    that "drive to rest pose" tracks the user-specified offset
    automatically.
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]
    angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] = (
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
    )
    linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] = (
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
        _D6_RIGID_LOCK,
    )


@dataclass
class D6Handle:
    """Global cid of the D6 joint created by :meth:`WorldBuilder.add_d6`.

    Returned with a sentinel cid (``-1``) and rewritten in place by
    :meth:`WorldBuilder.finalize`. Pass to
    :meth:`World.gather_constraint_wrenches` (via
    ``wrenches[handle.cid]``) to read the joint's combined reaction
    wrench on body 2 (linear constraint force + angular reaction
    torque, both in world frame). Per-axis force/torque limits show up
    as saturated (``+/- max_force``) components in the reported wrench.
    """

    cid: int = -1


class WorldBuilder:
    """Append bodies and constraints, then materialise a :class:`World`.

    Usage::

        b = WorldBuilder()
        anchor = b.add_static_body()                # body 0 (auto)
        body_a = b.add_dynamic_body(position=(0, 0, 0), inverse_mass=1.0)
        body_b = b.add_dynamic_body(position=(1, 0, 0), inverse_mass=1.0)
        b.add_ball_socket(anchor, body_a, anchor=(0, 0, 0))
        b.add_ball_socket(body_a, body_b, anchor=(0.5, 0, 0))
        world = b.finalize(substeps=1, solver_iterations=8)

    The builder is single-use; call :meth:`finalize` exactly once.
    """

    def __init__(self):
        # Body 0 is the static world body; user calls to add_*_body
        # start at index 1. We seed it with the canonical defaults
        # (origin, identity rotation, MOTION_STATIC, no inertia).
        self._bodies: list[RigidBodyDescriptor] = [RigidBodyDescriptor()]
        self._ball_sockets: list[BallSocketDescriptor] = []
        self._hinge_angles: list[HingeAngleDescriptor] = []
        self._angular_motors: list[AngularMotorDescriptor] = []
        # Fused hinge-joint descriptors and their handles. The handle
        # list is parallel to the descriptor list -- finalize() walks
        # both to patch each handle with its global cid in place so the
        # user-held reference becomes valid after finalize() returns.
        self._hinge_joint_descriptors: list[HingeJointDescriptor] = []
        self._hinge_joint_handles: list[HingeJointHandle] = []
        # Fused two-anchor (Schur-complement) hinge descriptors. Same
        # parallel-list pattern as the fused HingeJoint above.
        self._double_ball_socket_hinge_descriptors: list[DoubleBallSocketHingeDescriptor] = []
        self._double_ball_socket_hinge_handles: list[DoubleBallSocketHingeHandle] = []
        # Actuated fused two-anchor hinge descriptors (drive + limits on
        # the axial DoF). Same parallel-list pattern.
        self._actuated_dbs_hinge_descriptors: list[ActuatedDoubleBallSocketHingeDescriptor] = []
        self._actuated_dbs_hinge_handles: list[ActuatedDoubleBallSocketHingeHandle] = []
        # Prismatic (sliding) joint descriptors. Same parallel-list
        # pattern as the fused-hinge variants above; finalize() walks
        # both lists to patch each handle with its global cid.
        self._prismatic_descriptors: list[PrismaticDescriptor] = []
        self._prismatic_handles: list[PrismaticHandle] = []
        # 6-DoF generalised (D6) joint descriptors. Same parallel-list
        # pattern as the prismatic.
        self._d6_descriptors: list[D6Descriptor] = []
        self._d6_handles: list[D6Handle] = []

    # ------------------------------------------------------------------
    # Body API
    # ------------------------------------------------------------------

    @property
    def world_body(self) -> int:
        """Index of the auto-created static world body."""
        return 0

    def add_body(self, descriptor: RigidBodyDescriptor) -> int:
        """Append a fully-formed descriptor and return its body index."""
        self._bodies.append(descriptor)
        return len(self._bodies) - 1

    def add_static_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> int:
        """Append a static body (zero inverse mass / inertia, no integration)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
            )
        )

    def add_kinematic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a kinematic body: integrates its user-set velocities but
        ignores forces/contacts (matches Jitter ``MotionType.Kinematic``)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_KINEMATIC),
                inverse_mass=0.0,
                inverse_inertia=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                affected_by_gravity=False,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    def add_dynamic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        inverse_mass: float = 1.0,
        inverse_inertia: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = _IDENTITY_INERTIA,
        linear_damping: float = 1.0,
        angular_damping: float = 1.0,
        affected_by_gravity: bool = True,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Append a fully dynamic body. ``inverse_inertia`` is in the
        *body* frame; the solver rotates it to world space each step."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inverse_mass,
                inverse_inertia=inverse_inertia,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                affected_by_gravity=affected_by_gravity,
                velocity=velocity,
                angular_velocity=angular_velocity,
            )
        )

    # ------------------------------------------------------------------
    # Constraint API
    # ------------------------------------------------------------------

    def add_ball_socket(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a ball-and-socket constraint and return its (per-type)
        index. Both ``body1`` and ``body2`` must be valid body indices
        (i.e. returned from one of the ``add_*_body`` methods or 0 for
        the static world body).

        Args:
            body1: First body index.
            body2: Second body index.
            anchor: World-space anchor point [m].
            hertz: Soft-constraint stiffness target [Hz]; lower values
                make the joint more compliant. Defaults to
                :data:`DEFAULT_HERTZ_LINEAR`.
            damping_ratio: Non-dimensional damping ratio (1 is critical,
                <1 underdamped, >1 overdamped). Defaults to
                :data:`DEFAULT_DAMPING_RATIO`.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._ball_sockets.append(
            BallSocketDescriptor(body1, body2, anchor, hertz, damping_ratio)
        )
        return len(self._ball_sockets) - 1

    def add_hinge_angle(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
        hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO),
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a stand-alone hinge-angle (2-DoF) constraint and return
        its per-type index. ``axis`` is interpreted in *world* space at
        finalize() time and snapshotted into body 2's local frame, just
        like ``HingeAngle.Initialize``. Limits are in radians; the
        defaults (``+- pi``) effectively disable the limit.

        ``hertz_lock`` / ``damping_ratio_lock`` set the soft-constraint
        spring/damper for the two perpendicular angular DoFs (the lock
        itself); ``hertz_limit`` / ``damping_ratio_limit`` do the same
        for the axial min/max limit only.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        self._hinge_angles.append(
            HingeAngleDescriptor(
                body1,
                body2,
                axis,
                min_angle,
                max_angle,
                hertz_lock,
                damping_ratio_lock,
                hertz_limit,
                damping_ratio_limit,
            )
        )
        return len(self._hinge_angles) - 1

    def add_angular_motor(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        target_velocity: float = 0.0,
        max_force: float = 0.0,
        hertz: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> int:
        """Append a stand-alone angular-velocity motor and return its
        per-type index. ``axis`` is in world space; ``target_velocity``
        in rad/s; ``max_force`` in N·m. ``max_force=0`` (the default)
        gives a *disabled* motor that applies no torque.

        ``hertz`` / ``damping_ratio`` parametrise the velocity-tracking
        spring/damper using the Box2D v3 / Bepu soft-constraint
        formulation. See :class:`AngularMotorDescriptor`."""
        self._validate_body(body1)
        self._validate_body(body2)
        self._angular_motors.append(
            AngularMotorDescriptor(
                body1, body2, axis, target_velocity, max_force, hertz, damping_ratio
            )
        )
        return len(self._angular_motors) - 1

    def add_hinge_joint(
        self,
        body1: int,
        body2: int,
        hinge_center: tuple[float, float, float],
        hinge_axis: tuple[float, float, float],
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
        motor: bool = False,
        target_velocity: float = 0.0,
        max_force: float = 0.0,
        hertz_linear: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO),
        hertz_lock: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_lock: float = float(DEFAULT_DAMPING_RATIO),
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
        hertz_motor: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio_motor: float = float(DEFAULT_DAMPING_RATIO),
    ) -> HingeJointHandle:
        """Append a Jitter-style hinge joint and return its handle.

        Builds a *fused* :data:`CONSTRAINT_TYPE_HINGE_JOINT` constraint
        that packs the HingeAngle (2-DoF angular lock with optional
        limits), BallSocket (3-DoF positional lock at ``hinge_center``),
        and AngularMotor (drives the axial velocity) into one column.
        The single-thread fused dispatch typically converges noticeably
        better than the same triple as three separate constraints --
        see :mod:`newton._src.solvers.jitter.constraint_hinge_joint`
        for the rationale.

        Mirrors Jitter2's ``HingeJoint`` constructor
        (``Jitter2/Dynamics/Joints/HingeJoint.cs``); the ``hasMotor``
        flag is spelt ``motor`` here for Python ergonomics. When
        ``motor`` is false the AngularMotor sub stores ``max_force = 0``
        and applies no torque, matching the C# ``hasMotor=false`` path.
        Both ``hinge_center`` and ``hinge_axis`` are in *world* space at
        finalize() time.

        Returns a :class:`HingeJointHandle` whose ``cid`` field is
        rewritten in place by :meth:`finalize` to the joint's global
        cid in the shared :class:`ConstraintContainer`.
        """
        self._validate_body(body1)
        self._validate_body(body2)

        if motor:
            mf = max_force
            tv = target_velocity
        else:
            mf = 0.0
            tv = 0.0
        descriptor = HingeJointDescriptor(
            body1=body1,
            body2=body2,
            hinge_center=hinge_center,
            hinge_axis=hinge_axis,
            min_angle=min_angle,
            max_angle=max_angle,
            target_velocity=tv,
            max_force=mf,
            hertz_linear=hertz_linear,
            damping_ratio_linear=damping_ratio_linear,
            hertz_lock=hertz_lock,
            damping_ratio_lock=damping_ratio_lock,
            hertz_limit=hertz_limit,
            damping_ratio_limit=damping_ratio_limit,
            hertz_motor=hertz_motor,
            damping_ratio_motor=damping_ratio_motor,
        )
        handle = HingeJointHandle(cid=-1)
        self._hinge_joint_descriptors.append(descriptor)
        self._hinge_joint_handles.append(handle)
        return handle

    def add_double_ball_socket_hinge(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
    ) -> DoubleBallSocketHingeHandle:
        """Append a fused two-anchor "double ball-socket" hinge constraint
        and return its handle.

        Solves a single rank-5 column (3x3 + 2x2 Schur complement,
        no quaternion math) that locks 3 translational + 2 rotational
        DoF. The free rotational DoF is rotation about the line through
        ``anchor1`` and ``anchor2``. Mathematically equivalent to two
        independent ball-sockets at the same body pair but without the
        rank deficiency / compliance leak of stacking two 3-row
        Jacobians; see
        :mod:`newton._src.solvers.jitter.constraint_double_ball_socket`
        for the derivation.

        Both anchors are interpreted in *world* space at
        :meth:`finalize` time; the hinge axis is implicit in their
        relative direction.

        Returns a :class:`DoubleBallSocketHingeHandle` whose ``cid``
        field is rewritten in place by :meth:`finalize` to the joint's
        global cid in the shared :class:`ConstraintContainer`.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        descriptor = DoubleBallSocketHingeDescriptor(
            body1=body1,
            body2=body2,
            anchor1=anchor1,
            anchor2=anchor2,
            hertz=hertz,
            damping_ratio=damping_ratio,
        )
        handle = DoubleBallSocketHingeHandle(cid=-1)
        self._double_ball_socket_hinge_descriptors.append(descriptor)
        self._double_ball_socket_hinge_handles.append(handle)
        return handle

    def add_actuated_double_ball_socket_hinge(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float],
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        drive_mode: DriveMode = DriveMode.OFF,
        target_angle: float = 0.0,
        target_velocity: float = 0.0,
        max_force_drive: float = 0.0,
        hertz_drive: float = float(DEFAULT_HERTZ_MOTOR),
        damping_ratio_drive: float = float(DEFAULT_DAMPING_RATIO),
        min_angle: float = 0.0,
        max_angle: float = 0.0,
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
    ) -> ActuatedDoubleBallSocketHingeHandle:
        """Append an *actuated* fused two-anchor hinge and return its handle.

        Same 5-DoF positional lock as
        :meth:`add_double_ball_socket_hinge` plus a soft scalar PGS row
        on the free axial DoF that:

        * (optional) drives the relative axial twist towards
          ``target_angle`` [rad] (``drive_mode = DriveMode.POSITION``)
          or the relative axial rate towards ``target_velocity``
          [rad/s] (``drive_mode = DriveMode.VELOCITY``); a velocity
          drive is capped at ``\u00b1 max_force_drive * dt`` per substep
          [N*m], a position drive is bounded only by its soft-spring
          stiffness (``hertz_drive`` / ``damping_ratio_drive``).
        * (optional) clamps the relative axial twist to
          ``[min_angle, max_angle]`` [rad] via a one-sided spring-
          damper (``hertz_limit`` / ``damping_ratio_limit``).
          ``min_angle == max_angle == 0`` disables the limit.

        Drive and limit are independent and can be active
        simultaneously; the limit always wins because it's unilateral.
        See :mod:`newton._src.solvers.jitter.constraint_actuated_double_ball_socket`
        for the math.

        Args:
            body1: Index of the first body.
            body2: Index of the second body.
            anchor1: First hinge anchor in *world* space at finalize()
                time [m].
            anchor2: Second hinge anchor in *world* space at finalize()
                time [m]; the line ``anchor1 -> anchor2`` defines the
                hinge axis.
            hertz: Positional Schur block soft-constraint frequency [Hz].
            damping_ratio: Positional Schur block damping ratio.
            drive_mode: One of :class:`DriveMode`.
            target_angle: Position-drive setpoint [rad].
            target_velocity: Velocity-drive setpoint [rad/s].
            max_force_drive: Velocity-drive torque cap [N*m]; ignored
                in position-drive mode.
            hertz_drive: Drive soft-constraint frequency [Hz].
            damping_ratio_drive: Drive damping ratio.
            min_angle: Lower angular limit [rad].
            max_angle: Upper angular limit [rad];
                ``min_angle == max_angle == 0`` disables the limit.
            hertz_limit: Limit soft-constraint frequency [Hz].
            damping_ratio_limit: Limit damping ratio.

        Returns:
            :class:`ActuatedDoubleBallSocketHingeHandle` whose ``cid``
            is rewritten in place by :meth:`finalize` to the joint's
            global cid.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        descriptor = ActuatedDoubleBallSocketHingeDescriptor(
            body1=body1,
            body2=body2,
            anchor1=anchor1,
            anchor2=anchor2,
            hertz=hertz,
            damping_ratio=damping_ratio,
            drive_mode=DriveMode(int(drive_mode)),
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
        handle = ActuatedDoubleBallSocketHingeHandle(cid=-1)
        self._actuated_dbs_hinge_descriptors.append(descriptor)
        self._actuated_dbs_hinge_handles.append(handle)
        return handle

    def add_prismatic(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        axis: tuple[float, float, float],
        hertz_angular: float = float(DEFAULT_HERTZ_ANGULAR),
        damping_ratio_angular: float = float(DEFAULT_DAMPING_RATIO),
        hertz_linear: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio_linear: float = float(DEFAULT_DAMPING_RATIO),
    ) -> PrismaticHandle:
        """Append a prismatic (slider) joint and return its handle.

        Locks 5 of the 6 relative DoF between ``body1`` and ``body2``:

        * 3 rotational DoF -- the bodies must keep the same relative
          orientation as at finalize() time.
        * 2 translational DoF -- the bodies may only separate along
          ``axis``; lateral drift is forced to zero.

        The remaining (free) DoF is translation along the slide axis.
        Solved as one rank-5 column via a 3x3 + 2x2 Schur complement
        with separate Hertz/damping for the angular vs linear blocks
        (Box2D v3 / Bepu / Nordby soft-constraint formulation). See
        :mod:`newton._src.solvers.jitter.constraint_prismatic` for the
        derivation.

        Both ``anchor`` and ``axis`` are interpreted in *world* space
        at :meth:`finalize` time. The init kernel snapshots ``anchor``
        into both bodies' local frames (so they coincide at rest by
        construction) and snapshots ``axis`` into body 1's local frame
        (so the slide direction "rides" body 1 rigidly, matching the
        Bullet/ODE convention). ``axis`` need not be unit length.

        Args:
            body1: First body index. The slide axis is fixed in this
                body's local frame.
            body2: Second body index.
            anchor: World-space point through which the slide axis
                passes [m]. Both bodies measure their lever arms from
                this point; lateral drift of body 2's anchor relative
                to body 1's anchor is what the 2-row linear-lock
                constraint sees.
            axis: World-space slide direction at finalize() time;
                automatically normalised. The relative translation along
                this direction is the joint's free DoF.
            hertz_angular: Soft-constraint stiffness target [Hz] for
                the 3-row rotational lock. Defaults to
                :data:`DEFAULT_HERTZ_ANGULAR`. Set to 0 to make the
                rotational lock perfectly rigid (rigid plain-PGS update).
            damping_ratio_angular: Non-dimensional damping ratio for
                the rotational lock. Defaults to
                :data:`DEFAULT_DAMPING_RATIO` (critically damped).
            hertz_linear: Soft-constraint stiffness target [Hz] for the
                2-row perpendicular-translation lock. Defaults to
                :data:`DEFAULT_HERTZ_LINEAR`. Set to 0 for a rigid
                lateral lock.
            damping_ratio_linear: Non-dimensional damping ratio for
                the perpendicular-translation lock.

        Returns:
            A :class:`PrismaticHandle` whose ``cid`` field is rewritten
            in place by :meth:`finalize` to the joint's global cid in
            the shared :class:`ConstraintContainer`.
        """
        self._validate_body(body1)
        self._validate_body(body2)
        descriptor = PrismaticDescriptor(
            body1=body1,
            body2=body2,
            anchor=anchor,
            axis=axis,
            hertz_angular=hertz_angular,
            damping_ratio_angular=damping_ratio_angular,
            hertz_linear=hertz_linear,
            damping_ratio_linear=damping_ratio_linear,
        )
        handle = PrismaticHandle(cid=-1)
        self._prismatic_descriptors.append(descriptor)
        self._prismatic_handles.append(handle)
        return handle

    def add_d6(
        self,
        body1: int,
        body2: int,
        anchor: tuple[float, float, float],
        *,
        angular: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
        linear: tuple[D6AxisDrive, D6AxisDrive, D6AxisDrive] | None = None,
    ) -> D6Handle:
        """Append a 6-DoF generalised ("D6") joint and return its handle.

        Owns *all* six relative DoF between ``body1`` and ``body2`` and
        configures each axis independently via a :class:`D6AxisDrive`:
        rigid lock, soft lock, position-PD drive, velocity-PD drive,
        force-limited drive, or free axis. Solved as one column with a
        6x6 Schur complement -- see
        :mod:`newton._src.solvers.jitter.constraint_d6` for the math.

        The default call ``b.add_d6(b1, b2, anchor)`` (no per-axis
        overrides) is a 6-DoF *rigid weld*: all 6 axes default to
        ``D6AxisDrive(hertz=0, max_force=inf)`` which collapses to a
        plain-PGS rigid lock on every axis. Pass per-axis
        :class:`D6AxisDrive` triples to soften axes, set position /
        velocity targets, cap the force/torque per axis, or free
        individual axes (``max_force=0``).

        ``anchor`` is in *world* space at finalize() time. The 3
        angular and 3 linear axes are interpreted in body-1's local
        frame ("frame A" in PhysX D6 / Bullet 6Dof2 convention). All
        position targets and velocity targets are in this body-1-local
        frame.

        Args:
            body1: First body index. The joint frame rides this body
                rigidly.
            body2: Second body index. Driven against the joint frame
                by the per-axis spec.
            anchor: World-space anchor point [m]. Both bodies measure
                their lever arms from this point at finalize() time.
            angular: 3-tuple of :class:`D6AxisDrive` for the
                (body-1-local) ``e_x``, ``e_y``, ``e_z`` angular axes.
                ``None`` -> all-rigid (default).
            linear: 3-tuple of :class:`D6AxisDrive` for the
                (body-1-local) ``e_x``, ``e_y``, ``e_z`` linear axes.
                ``None`` -> all-rigid (default).

        Returns:
            A :class:`D6Handle` whose ``cid`` field is rewritten in
            place by :meth:`finalize` to the joint's global cid in
            the shared :class:`ConstraintContainer`.

        Examples:

        Rigid weld (no kwargs needed)::

            b.add_d6(b1, b2, anchor=(0.0, 0.0, 0.0))

        Position-PD drive on the body-1-local +y angular axis (90 deg
        target, 5 Hz spring, 10 N*m torque cap), other 5 axes rigid::

            b.add_d6(
                b1, b2, anchor=(0.0, 0.0, 0.0),
                angular=(
                    D6AxisDrive(),
                    D6AxisDrive(hertz=5.0, target_position=math.pi/2,
                                max_force=10.0),
                    D6AxisDrive(),
                ),
            )

        A "free hinge" about body-1-local +y (3 angular DoF: x and z
        rigid, y free), positionally locked::

            b.add_d6(
                b1, b2, anchor=(0.0, 0.0, 0.0),
                angular=(
                    D6AxisDrive(),                        # rigid lock
                    D6AxisDrive(max_force=0.0),           # FREE
                    D6AxisDrive(),                        # rigid lock
                ),
            )
        """
        self._validate_body(body1)
        self._validate_body(body2)

        if angular is None:
            angular = (_D6_RIGID_LOCK, _D6_RIGID_LOCK, _D6_RIGID_LOCK)
        if linear is None:
            linear = (_D6_RIGID_LOCK, _D6_RIGID_LOCK, _D6_RIGID_LOCK)
        if len(angular) != 3:
            raise ValueError(
                f"D6 'angular' must be a 3-tuple of D6AxisDrive (got len={len(angular)})"
            )
        if len(linear) != 3:
            raise ValueError(
                f"D6 'linear' must be a 3-tuple of D6AxisDrive (got len={len(linear)})"
            )

        descriptor = D6Descriptor(
            body1=body1,
            body2=body2,
            anchor=anchor,
            angular=tuple(angular),
            linear=tuple(linear),
        )
        handle = D6Handle(cid=-1)
        self._d6_descriptors.append(descriptor)
        self._d6_handles.append(handle)
        return handle

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(
        self,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_relaxations: int = 1,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        device: wp.context.Devicelike = None,
    ) -> World:
        """Allocate GPU storage and build a ready-to-step :class:`World`.

        The current registered descriptor lists are consumed in place
        (cleared after success). Calling :meth:`finalize` twice produces
        an empty world the second time -- clone the builder first if
        that's not what you want.

        Args:
            substeps: Forwarded to :class:`World`. Number of substeps
                per :meth:`World.step` call.
            solver_iterations: PGS iterations per substep.
            velocity_relaxations: Relaxation passes (currently a stub
                inside ``World``).
            gravity: World gravity vector [m/s^2].
            device: Warp device. ``None`` selects the current default
                device (typically ``cuda:0``).
        """
        device = wp.get_device(device)

        bodies = self._build_body_container(device)
        constraints, num_constraints = self._build_constraint_container(bodies, device)

        world = World(
            bodies=bodies,
            constraints=constraints,
            num_constraints=num_constraints,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_relaxations=velocity_relaxations,
            gravity=gravity,
            device=device,
        )

        # Reset internal state so the builder can't accidentally leak
        # references into a second finalize call.
        self._bodies = [RigidBodyDescriptor()]
        self._ball_sockets = []
        self._hinge_angles = []
        self._angular_motors = []
        self._hinge_joint_descriptors = []
        self._hinge_joint_handles = []
        self._double_ball_socket_hinge_descriptors = []
        self._double_ball_socket_hinge_handles = []
        self._actuated_dbs_hinge_descriptors = []
        self._actuated_dbs_hinge_handles = []
        self._prismatic_descriptors = []
        self._prismatic_handles = []
        self._d6_descriptors = []
        self._d6_handles = []
        return world

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_body(self, idx: int) -> None:
        if not (0 <= idx < len(self._bodies)):
            raise IndexError(
                f"body index {idx} out of range [0, {len(self._bodies)})"
            )

    def _build_body_container(self, device: wp.context.Device) -> BodyContainer:
        """Pack the descriptor list into a :class:`BodyContainer` via NumPy.

        Builds one NumPy array per field and uploads it. This is one
        host-to-device transfer per field, which dwarfs anything we
        could save by interleaving -- finalize() is a one-shot setup
        path, not a hot loop.
        """
        n = len(self._bodies)
        positions = np.zeros((n, 3), dtype=np.float32)
        orientations = np.zeros((n, 4), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        angular_velocities = np.zeros((n, 3), dtype=np.float32)
        inverse_inertia = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_inertia_world = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_mass = np.zeros(n, dtype=np.float32)
        linear_damping = np.ones(n, dtype=np.float32)
        angular_damping = np.ones(n, dtype=np.float32)
        affected_by_gravity = np.ones(n, dtype=np.int32)
        motion_type = np.full(n, int(MOTION_STATIC), dtype=np.int32)

        for i, b in enumerate(self._bodies):
            positions[i] = b.position
            orientations[i] = b.orientation
            velocities[i] = b.velocity
            angular_velocities[i] = b.angular_velocity
            inverse_inertia[i] = b.inverse_inertia
            # Seed inverse_inertia_world with the body-frame value; the
            # first _update_bodies launch will rotate it into world
            # space using the actual orientation. Without this seed the
            # very first prepare_for_iteration sees zero effective mass.
            inverse_inertia_world[i] = b.inverse_inertia
            inverse_mass[i] = b.inverse_mass
            linear_damping[i] = b.linear_damping
            angular_damping[i] = b.angular_damping
            affected_by_gravity[i] = 1 if b.affected_by_gravity else 0
            motion_type[i] = int(b.motion_type)

        c = BodyContainer()
        c.position = wp.array(positions, dtype=wp.vec3f, device=device)
        c.velocity = wp.array(velocities, dtype=wp.vec3f, device=device)
        c.angular_velocity = wp.array(angular_velocities, dtype=wp.vec3f, device=device)
        c.orientation = wp.array(orientations, dtype=wp.quatf, device=device)
        c.inverse_inertia_world = wp.array(inverse_inertia_world, dtype=wp.mat33f, device=device)
        c.inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
        c.inverse_mass = wp.array(inverse_mass, dtype=wp.float32, device=device)
        c.force = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.torque = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.delta_angular_velocity = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.linear_damping = wp.array(linear_damping, dtype=wp.float32, device=device)
        c.angular_damping = wp.array(angular_damping, dtype=wp.float32, device=device)
        c.affected_by_gravity = wp.array(affected_by_gravity, dtype=wp.int32, device=device)
        c.motion_type = wp.array(motion_type, dtype=wp.int32, device=device)
        return c

    def _build_constraint_container(
        self,
        bodies: BodyContainer,
        device: wp.context.Device,
    ) -> tuple[ConstraintContainer, int]:
        """Allocate the shared :class:`ConstraintContainer`, pack every
        constraint type into a contiguous cid range, and patch any
        outstanding :class:`HingeJointHandle` with their global cid.

        Layout:

            * cids ``[0, n_ball)``                              -> ball-sockets
            * cids ``[n_ball, +n_hinge)``                       -> hinge-angles
            * cids ``[..., +n_motor)``                          -> angular-motors
            * cids ``[..., +n_hinge_joint)``                    -> fused hinge-joints
            * cids ``[..., +n_dbs_hinge)``                      -> fused double-ball-socket hinges
            * cids ``[..., +n_prismatic)``                      -> prismatic (slider) joints
            * cids ``[..., +n_d6)``                             -> D6 (6-DoF generalised) joints

        The container's per-column dword count is ``max`` of all
        registered constraint schemas so any column can hold any type.
        The unused trailing rows of the smaller types are simply
        ignored by their kernels (per-type ``*_DWORDS`` constants only
        describe the populated prefix).
        """
        n_ball = len(self._ball_sockets)
        n_hinge = len(self._hinge_angles)
        n_motor = len(self._angular_motors)
        n_hinge_joint = len(self._hinge_joint_descriptors)
        n_dbs_hinge = len(self._double_ball_socket_hinge_descriptors)
        n_adbs_hinge = len(self._actuated_dbs_hinge_descriptors)
        n_prismatic = len(self._prismatic_descriptors)
        n_d6 = len(self._d6_descriptors)
        total = (
            n_ball
            + n_hinge
            + n_motor
            + n_hinge_joint
            + n_dbs_hinge
            + n_adbs_hinge
            + n_prismatic
            + n_d6
        )

        # All types share the column-major storage so the unified
        # dispatcher in solver_jitter_kernels can index any column the
        # same way; size to the widest schema.
        per_constraint_dwords = max(
            BS_DWORDS,
            HA_DWORDS,
            AM_DWORDS,
            HJ_DWORDS,
            DBS_DWORDS,
            ADBS_DWORDS,
            PR_DWORDS,
            D6_DWORDS,
        )

        # Always allocate at least 1 column so the wp.array2d shape is
        # non-degenerate; the kernels gate on cid bounds anyway.
        container = constraint_container_zeros(
            num_constraints=max(1, total),
            num_dwords=per_constraint_dwords,
            device=device,
        )

        ball_socket_offset = 0
        hinge_angle_offset = n_ball
        angular_motor_offset = n_ball + n_hinge
        hinge_joint_offset = n_ball + n_hinge + n_motor
        dbs_hinge_offset = n_ball + n_hinge + n_motor + n_hinge_joint
        adbs_hinge_offset = (
            n_ball + n_hinge + n_motor + n_hinge_joint + n_dbs_hinge
        )
        prismatic_offset = (
            n_ball + n_hinge + n_motor + n_hinge_joint + n_dbs_hinge + n_adbs_hinge
        )
        d6_offset = (
            n_ball
            + n_hinge
            + n_motor
            + n_hinge_joint
            + n_dbs_hinge
            + n_adbs_hinge
            + n_prismatic
        )

        if n_ball > 0:
            self._init_ball_sockets(container, bodies, ball_socket_offset, device)
        if n_hinge > 0:
            self._init_hinge_angles(container, bodies, hinge_angle_offset, device)
        if n_motor > 0:
            self._init_angular_motors(container, bodies, angular_motor_offset, device)
        if n_hinge_joint > 0:
            self._init_hinge_joints(container, bodies, hinge_joint_offset, device)
        if n_dbs_hinge > 0:
            self._init_double_ball_socket_hinges(container, bodies, dbs_hinge_offset, device)
        if n_adbs_hinge > 0:
            self._init_actuated_double_ball_socket_hinges(
                container, bodies, adbs_hinge_offset, device
            )
        if n_prismatic > 0:
            self._init_prismatics(container, bodies, prismatic_offset, device)
        if n_d6 > 0:
            self._init_d6s(container, bodies, d6_offset, device)

        # Resolve fused-joint handles in place. Each handle was created
        # with cid=-1 and is parallel to its descriptor list; rewrite
        # to the joint's global cid so user-held references become
        # valid after finalize() returns.
        for i, h in enumerate(self._hinge_joint_handles):
            h.cid = hinge_joint_offset + i
        for i, h in enumerate(self._double_ball_socket_hinge_handles):
            h.cid = dbs_hinge_offset + i
        for i, h in enumerate(self._actuated_dbs_hinge_handles):
            h.cid = adbs_hinge_offset + i
        for i, h in enumerate(self._prismatic_handles):
            h.cid = prismatic_offset + i
        for i, h in enumerate(self._d6_handles):
            h.cid = d6_offset + i

        return container, total

    def _init_ball_sockets(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._ball_sockets)

        body1 = np.asarray([d.body1 for d in self._ball_sockets], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._ball_sockets], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in self._ball_sockets], dtype=np.float32)
        hertz = np.asarray([d.hertz for d in self._ball_sockets], dtype=np.float32)
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._ball_sockets], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_hinge_angles(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._hinge_angles)

        body1 = np.asarray([d.body1 for d in self._hinge_angles], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._hinge_angles], dtype=np.int32)
        axis = np.asarray([d.axis for d in self._hinge_angles], dtype=np.float32)
        min_angle = np.asarray([d.min_angle for d in self._hinge_angles], dtype=np.float32)
        max_angle = np.asarray([d.max_angle for d in self._hinge_angles], dtype=np.float32)
        hertz_lock = np.asarray(
            [d.hertz_lock for d in self._hinge_angles], dtype=np.float32
        )
        damping_ratio_lock = np.asarray(
            [d.damping_ratio_lock for d in self._hinge_angles], dtype=np.float32
        )
        hertz_limit = np.asarray(
            [d.hertz_limit for d in self._hinge_angles], dtype=np.float32
        )
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in self._hinge_angles], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        hertz_lock_d = wp.array(hertz_lock, dtype=wp.float32, device=device)
        damping_ratio_lock_d = wp.array(damping_ratio_lock, dtype=wp.float32, device=device)
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(
            damping_ratio_limit, dtype=wp.float32, device=device
        )

        wp.launch(
            hinge_angle_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis_d,
                min_angle_d,
                max_angle_d,
                hertz_lock_d,
                damping_ratio_lock_d,
                hertz_limit_d,
                damping_ratio_limit_d,
            ],
            device=device,
        )

    def _init_angular_motors(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        n = len(self._angular_motors)

        body1 = np.asarray([d.body1 for d in self._angular_motors], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in self._angular_motors], dtype=np.int32)
        # Jitter's HingeJoint passes the same world-space axis to body1
        # and body2 (axis1 == axis2 == hinge_axis); mirror that here so
        # both bodies snapshot the same initial rotational reference.
        axis = np.asarray([d.axis for d in self._angular_motors], dtype=np.float32)
        target_velocity = np.asarray(
            [d.target_velocity for d in self._angular_motors], dtype=np.float32
        )
        max_force = np.asarray(
            [d.max_force for d in self._angular_motors], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._angular_motors], dtype=np.float32
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._angular_motors], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            angular_motor_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                axis1_d,
                axis2_d,
                target_velocity_d,
                max_force_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_hinge_joints(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the fused hinge-joint descriptors into ``constraints``.

        One launch of :func:`hinge_joint_initialize_kernel` writes all
        three sub-blocks (BallSocket / HingeAngle / AngularMotor) plus
        the shared header into each fused column.
        """
        n = len(self._hinge_joint_descriptors)

        body1 = np.asarray(
            [d.body1 for d in self._hinge_joint_descriptors], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._hinge_joint_descriptors], dtype=np.int32
        )
        anchor = np.asarray(
            [d.hinge_center for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        axis = np.asarray(
            [d.hinge_axis for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        min_angle = np.asarray(
            [d.min_angle for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        max_angle = np.asarray(
            [d.max_angle for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        target_velocity = np.asarray(
            [d.target_velocity for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        max_force = np.asarray(
            [d.max_force for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        hertz_linear = np.asarray(
            [d.hertz_linear for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_linear = np.asarray(
            [d.damping_ratio_linear for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_lock = np.asarray(
            [d.hertz_lock for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_lock = np.asarray(
            [d.damping_ratio_lock for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_limit = np.asarray(
            [d.hertz_limit for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )
        hertz_motor = np.asarray(
            [d.hertz_motor for d in self._hinge_joint_descriptors], dtype=np.float32
        )
        damping_ratio_motor = np.asarray(
            [d.damping_ratio_motor for d in self._hinge_joint_descriptors],
            dtype=np.float32,
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)
        hertz_linear_d = wp.array(hertz_linear, dtype=wp.float32, device=device)
        damping_ratio_linear_d = wp.array(
            damping_ratio_linear, dtype=wp.float32, device=device
        )
        hertz_lock_d = wp.array(hertz_lock, dtype=wp.float32, device=device)
        damping_ratio_lock_d = wp.array(
            damping_ratio_lock, dtype=wp.float32, device=device
        )
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(
            damping_ratio_limit, dtype=wp.float32, device=device
        )
        hertz_motor_d = wp.array(hertz_motor, dtype=wp.float32, device=device)
        damping_ratio_motor_d = wp.array(
            damping_ratio_motor, dtype=wp.float32, device=device
        )

        wp.launch(
            hinge_joint_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                axis_d,
                min_angle_d,
                max_angle_d,
                target_velocity_d,
                max_force_d,
                hertz_linear_d,
                damping_ratio_linear_d,
                hertz_lock_d,
                damping_ratio_lock_d,
                hertz_limit_d,
                damping_ratio_limit_d,
                hertz_motor_d,
                damping_ratio_motor_d,
            ],
            device=device,
        )

    def _init_double_ball_socket_hinges(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the fused two-anchor hinge descriptors into ``constraints``.

        One launch of :func:`double_ball_socket_initialize_kernel` writes
        each fused column. The init kernel snapshots both world-space
        anchors into each body's local frame; the hinge axis is implicit
        in the line ``anchor1 -> anchor2``.
        """
        n = len(self._double_ball_socket_hinge_descriptors)

        body1 = np.asarray(
            [d.body1 for d in self._double_ball_socket_hinge_descriptors], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._double_ball_socket_hinge_descriptors], dtype=np.int32
        )
        anchor1 = np.asarray(
            [d.anchor1 for d in self._double_ball_socket_hinge_descriptors], dtype=np.float32
        )
        anchor2 = np.asarray(
            [d.anchor2 for d in self._double_ball_socket_hinge_descriptors], dtype=np.float32
        )
        hertz = np.asarray(
            [d.hertz for d in self._double_ball_socket_hinge_descriptors],
            dtype=np.float32,
        )
        damping_ratio = np.asarray(
            [d.damping_ratio for d in self._double_ball_socket_hinge_descriptors],
            dtype=np.float32,
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)

        wp.launch(
            double_ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor1_d,
                anchor2_d,
                hertz_d,
                damping_ratio_d,
            ],
            device=device,
        )

    def _init_actuated_double_ball_socket_hinges(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the actuated fused two-anchor hinge descriptors.

        Layered on top of :meth:`_init_double_ball_socket_hinges`: the
        same per-anchor world->local snapshot, plus the per-actuator
        drive + limit parameters. One launch of
        :func:`actuated_double_ball_socket_initialize_kernel` writes
        each fused column.
        """
        descs = self._actuated_dbs_hinge_descriptors
        n = len(descs)

        body1 = np.asarray([d.body1 for d in descs], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in descs], dtype=np.int32)
        anchor1 = np.asarray([d.anchor1 for d in descs], dtype=np.float32)
        anchor2 = np.asarray([d.anchor2 for d in descs], dtype=np.float32)
        hertz = np.asarray([d.hertz for d in descs], dtype=np.float32)
        damping_ratio = np.asarray([d.damping_ratio for d in descs], dtype=np.float32)
        drive_mode = np.asarray([int(d.drive_mode) for d in descs], dtype=np.int32)
        target_angle = np.asarray([d.target_angle for d in descs], dtype=np.float32)
        target_velocity = np.asarray([d.target_velocity for d in descs], dtype=np.float32)
        max_force_drive = np.asarray([d.max_force_drive for d in descs], dtype=np.float32)
        hertz_drive = np.asarray([d.hertz_drive for d in descs], dtype=np.float32)
        damping_ratio_drive = np.asarray(
            [d.damping_ratio_drive for d in descs], dtype=np.float32
        )
        min_angle = np.asarray([d.min_angle for d in descs], dtype=np.float32)
        max_angle = np.asarray([d.max_angle for d in descs], dtype=np.float32)
        hertz_limit = np.asarray([d.hertz_limit for d in descs], dtype=np.float32)
        damping_ratio_limit = np.asarray(
            [d.damping_ratio_limit for d in descs], dtype=np.float32
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)
        hertz_d = wp.array(hertz, dtype=wp.float32, device=device)
        damping_ratio_d = wp.array(damping_ratio, dtype=wp.float32, device=device)
        drive_mode_d = wp.array(drive_mode, dtype=wp.int32, device=device)
        target_angle_d = wp.array(target_angle, dtype=wp.float32, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_drive_d = wp.array(max_force_drive, dtype=wp.float32, device=device)
        hertz_drive_d = wp.array(hertz_drive, dtype=wp.float32, device=device)
        damping_ratio_drive_d = wp.array(damping_ratio_drive, dtype=wp.float32, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        hertz_limit_d = wp.array(hertz_limit, dtype=wp.float32, device=device)
        damping_ratio_limit_d = wp.array(damping_ratio_limit, dtype=wp.float32, device=device)

        wp.launch(
            actuated_double_ball_socket_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor1_d,
                anchor2_d,
                hertz_d,
                damping_ratio_d,
                drive_mode_d,
                target_angle_d,
                target_velocity_d,
                max_force_drive_d,
                hertz_drive_d,
                damping_ratio_drive_d,
                min_angle_d,
                max_angle_d,
                hertz_limit_d,
                damping_ratio_limit_d,
            ],
            device=device,
        )

    def _init_prismatics(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the prismatic-joint descriptors into ``constraints``.

        One launch of :func:`prismatic_initialize_kernel` writes each
        prismatic column. The init kernel snapshots ``anchor`` into
        both bodies' local frames and ``axis`` into body 1's local
        frame (normalising it on the fly), then snapshots the rest-pose
        relative orientation ``q0 = q2^* q1`` so the runtime math
        operates entirely on quantities that move rigidly with the
        bodies.
        """
        n = len(self._prismatic_descriptors)

        body1 = np.asarray(
            [d.body1 for d in self._prismatic_descriptors], dtype=np.int32
        )
        body2 = np.asarray(
            [d.body2 for d in self._prismatic_descriptors], dtype=np.int32
        )
        anchor = np.asarray(
            [d.anchor for d in self._prismatic_descriptors], dtype=np.float32
        )
        axis = np.asarray(
            [d.axis for d in self._prismatic_descriptors], dtype=np.float32
        )
        hertz_angular = np.asarray(
            [d.hertz_angular for d in self._prismatic_descriptors], dtype=np.float32
        )
        damping_ratio_angular = np.asarray(
            [d.damping_ratio_angular for d in self._prismatic_descriptors],
            dtype=np.float32,
        )
        hertz_linear = np.asarray(
            [d.hertz_linear for d in self._prismatic_descriptors], dtype=np.float32
        )
        damping_ratio_linear = np.asarray(
            [d.damping_ratio_linear for d in self._prismatic_descriptors],
            dtype=np.float32,
        )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        hertz_angular_d = wp.array(hertz_angular, dtype=wp.float32, device=device)
        damping_ratio_angular_d = wp.array(
            damping_ratio_angular, dtype=wp.float32, device=device
        )
        hertz_linear_d = wp.array(hertz_linear, dtype=wp.float32, device=device)
        damping_ratio_linear_d = wp.array(
            damping_ratio_linear, dtype=wp.float32, device=device
        )

        wp.launch(
            prismatic_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                axis_d,
                hertz_angular_d,
                damping_ratio_angular_d,
                hertz_linear_d,
                damping_ratio_linear_d,
            ],
            device=device,
        )

    def _init_d6s(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        cid_offset: int,
        device: wp.context.Device,
    ) -> None:
        """Pack the D6 (6-DoF) descriptors into ``constraints``.

        One launch of :func:`d6_initialize_kernel` writes each D6
        column. Per-axis :class:`D6AxisDrive`s on the descriptor are
        un-tupled into 6 ``vec3f`` arrays per knob (hertz_ang,
        damping_ang, hertz_lin, damping_lin, max_force_ang,
        max_force_lin, target_position_*, target_velocity_*) so the
        kernel sees a flat per-cid layout. ``max_force = math.inf``
        becomes a finite "huge" cap (1e30) to keep the per-axis clamp
        well-defined inside the kernel without needing an explicit
        "no clamp" branch.
        """
        n = len(self._d6_descriptors)
        descriptors = self._d6_descriptors

        body1 = np.asarray([d.body1 for d in descriptors], dtype=np.int32)
        body2 = np.asarray([d.body2 for d in descriptors], dtype=np.int32)
        anchor = np.asarray([d.anchor for d in descriptors], dtype=np.float32)

        # Per-axis knob arrays, one row per cid, three components per
        # row (the three body-1-local axes).
        target_position_ang = np.zeros((n, 3), dtype=np.float32)
        target_position_lin = np.zeros((n, 3), dtype=np.float32)
        target_velocity_ang = np.zeros((n, 3), dtype=np.float32)
        target_velocity_lin = np.zeros((n, 3), dtype=np.float32)
        hertz_ang = np.zeros((n, 3), dtype=np.float32)
        damping_ang = np.zeros((n, 3), dtype=np.float32)
        hertz_lin = np.zeros((n, 3), dtype=np.float32)
        damping_lin = np.zeros((n, 3), dtype=np.float32)
        max_force_ang = np.zeros((n, 3), dtype=np.float32)
        max_force_lin = np.zeros((n, 3), dtype=np.float32)

        # Cap "infinite" max_force at a large finite number so the
        # iterate-path clamp stays a well-defined float operation.
        # 1e30 is far above any realistic per-substep impulse but well
        # under float32's ~3.4e38 ceiling so multiplications inside
        # the kernel can't overflow.
        _HUGE_FORCE = 1.0e30

        for i, d in enumerate(descriptors):
            for k in range(3):
                a = d.angular[k]
                lin = d.linear[k]

                target_position_ang[i, k] = a.target_position
                target_position_lin[i, k] = lin.target_position
                target_velocity_ang[i, k] = a.target_velocity
                target_velocity_lin[i, k] = lin.target_velocity
                hertz_ang[i, k] = a.hertz
                damping_ang[i, k] = a.damping_ratio
                hertz_lin[i, k] = lin.hertz
                damping_lin[i, k] = lin.damping_ratio
                max_force_ang[i, k] = (
                    _HUGE_FORCE if math.isinf(a.max_force) else a.max_force
                )
                max_force_lin[i, k] = (
                    _HUGE_FORCE if math.isinf(lin.max_force) else lin.max_force
                )

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        target_position_ang_d = wp.array(target_position_ang, dtype=wp.vec3f, device=device)
        target_position_lin_d = wp.array(target_position_lin, dtype=wp.vec3f, device=device)
        target_velocity_ang_d = wp.array(target_velocity_ang, dtype=wp.vec3f, device=device)
        target_velocity_lin_d = wp.array(target_velocity_lin, dtype=wp.vec3f, device=device)
        hertz_ang_d = wp.array(hertz_ang, dtype=wp.vec3f, device=device)
        damping_ang_d = wp.array(damping_ang, dtype=wp.vec3f, device=device)
        hertz_lin_d = wp.array(hertz_lin, dtype=wp.vec3f, device=device)
        damping_lin_d = wp.array(damping_lin, dtype=wp.vec3f, device=device)
        max_force_ang_d = wp.array(max_force_ang, dtype=wp.vec3f, device=device)
        max_force_lin_d = wp.array(max_force_lin, dtype=wp.vec3f, device=device)

        wp.launch(
            d6_initialize_kernel,
            dim=n,
            inputs=[
                constraints,
                bodies,
                int(cid_offset),
                body1_d,
                body2_d,
                anchor_d,
                target_position_ang_d,
                target_position_lin_d,
                target_velocity_ang_d,
                target_velocity_lin_d,
                hertz_ang_d,
                damping_ang_d,
                hertz_lin_d,
                damping_lin_d,
                max_force_ang_d,
                max_force_lin_d,
            ],
            device=device,
        )
