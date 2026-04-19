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

import numpy as np
import warp as wp

from newton._src.solvers.jitter.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
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
    ConstraintContainer,
    constraint_container_zeros,
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
from newton._src.solvers.jitter.solver_jitter import World

__all__ = [
    "AngularMotorDescriptor",
    "BallSocketDescriptor",
    "DoubleBallSocketHingeDescriptor",
    "DoubleBallSocketHingeHandle",
    "HingeAngleDescriptor",
    "HingeJointDescriptor",
    "HingeJointHandle",
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
    ``BallSocket.Initialize``).
    """

    body1: int
    body2: int
    anchor: tuple[float, float, float]


@dataclass
class HingeAngleDescriptor:
    """Plain-Python description of one hinge-angle (2-DoF) constraint.

    The ``axis`` is in *world* space at construction time; finalize()
    transforms it into body 2's local frame, mirroring
    ``HingeAngle.Initialize``. ``min_angle`` / ``max_angle`` are in
    radians.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    min_angle: float = -math.pi
    max_angle: float = math.pi


@dataclass
class AngularMotorDescriptor:
    """Plain-Python description of one angular-velocity motor constraint.

    The motor drives the relative angular velocity along ``axis`` (world
    space at construction time) toward ``target_velocity`` [rad/s] using
    at most ``max_force`` [N·m] of torque. ``max_force = 0`` (Jitter
    default) yields a *disabled* motor that does nothing.
    """

    body1: int
    body2: int
    axis: tuple[float, float, float]
    target_velocity: float = 0.0
    max_force: float = 0.0


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
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]


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
    ) -> int:
        """Append a ball-and-socket constraint and return its (per-type)
        index. Both ``body1`` and ``body2`` must be valid body indices
        (i.e. returned from one of the ``add_*_body`` methods or 0 for
        the static world body)."""
        self._validate_body(body1)
        self._validate_body(body2)
        self._ball_sockets.append(BallSocketDescriptor(body1, body2, anchor))
        return len(self._ball_sockets) - 1

    def add_hinge_angle(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        min_angle: float = -math.pi,
        max_angle: float = math.pi,
    ) -> int:
        """Append a stand-alone hinge-angle (2-DoF) constraint and return
        its per-type index. ``axis`` is interpreted in *world* space at
        finalize() time and snapshotted into body 2's local frame, just
        like ``HingeAngle.Initialize``. Limits are in radians; the
        defaults (``+- pi``) effectively disable the limit."""
        self._validate_body(body1)
        self._validate_body(body2)
        self._hinge_angles.append(
            HingeAngleDescriptor(body1, body2, axis, min_angle, max_angle)
        )
        return len(self._hinge_angles) - 1

    def add_angular_motor(
        self,
        body1: int,
        body2: int,
        axis: tuple[float, float, float],
        target_velocity: float = 0.0,
        max_force: float = 0.0,
    ) -> int:
        """Append a stand-alone angular-velocity motor and return its
        per-type index. ``axis`` is in world space; ``target_velocity``
        in rad/s; ``max_force`` in N·m. ``max_force=0`` (the default)
        gives a *disabled* motor that applies no torque."""
        self._validate_body(body1)
        self._validate_body(body2)
        self._angular_motors.append(
            AngularMotorDescriptor(body1, body2, axis, target_velocity, max_force)
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
        )
        handle = DoubleBallSocketHingeHandle(cid=-1)
        self._double_ball_socket_hinge_descriptors.append(descriptor)
        self._double_ball_socket_hinge_handles.append(handle)
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
        total = n_ball + n_hinge + n_motor + n_hinge_joint + n_dbs_hinge

        # All types share the column-major storage so the unified
        # dispatcher in solver_jitter_kernels can index any column the
        # same way; size to the widest schema.
        per_constraint_dwords = max(BS_DWORDS, HA_DWORDS, AM_DWORDS, HJ_DWORDS, DBS_DWORDS)

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

        # Resolve fused-joint handles in place. Each handle was created
        # with cid=-1 and is parallel to its descriptor list; rewrite
        # to the joint's global cid so user-held references become
        # valid after finalize() returns.
        for i, h in enumerate(self._hinge_joint_handles):
            h.cid = hinge_joint_offset + i
        for i, h in enumerate(self._double_ball_socket_hinge_handles):
            h.cid = dbs_hinge_offset + i

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

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)

        wp.launch(
            ball_socket_initialize_kernel,
            dim=n,
            inputs=[constraints, bodies, int(cid_offset), body1_d, body2_d, anchor_d],
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

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)

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

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        axis1_d = wp.array(axis, dtype=wp.vec3f, device=device)
        axis2_d = wp.array(axis, dtype=wp.vec3f, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)

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

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor_d = wp.array(anchor, dtype=wp.vec3f, device=device)
        axis_d = wp.array(axis, dtype=wp.vec3f, device=device)
        min_angle_d = wp.array(min_angle, dtype=wp.float32, device=device)
        max_angle_d = wp.array(max_angle, dtype=wp.float32, device=device)
        target_velocity_d = wp.array(target_velocity, dtype=wp.float32, device=device)
        max_force_d = wp.array(max_force, dtype=wp.float32, device=device)

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

        body1_d = wp.array(body1, dtype=wp.int32, device=device)
        body2_d = wp.array(body2, dtype=wp.int32, device=device)
        anchor1_d = wp.array(anchor1, dtype=wp.vec3f, device=device)
        anchor2_d = wp.array(anchor2, dtype=wp.vec3f, device=device)

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
            ],
            device=device,
        )
