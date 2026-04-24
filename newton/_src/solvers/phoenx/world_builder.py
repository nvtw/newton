# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side builder assembling a :class:`PhoenXWorld` from plain Python.

Two-phase construction:

1. *Append* -- ``add_static_body`` / ``add_dynamic_body`` / ``add_joint``
   record descriptors. Nothing touches the GPU.
2. *Finalize* -- :meth:`WorldBuilder.finalize` allocates containers,
   packs descriptors via NumPy, launches one init kernel for the
   joints, and returns a ready-to-step :class:`PhoenXWorld`.

Body 0 is the static world anchor for single-world scenes; for
``num_worlds > 1`` bodies ``[0, num_worlds)`` are each world's anchor.
User descriptors start after that.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    ADBS_DWORDS,
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
    ConstraintContainer,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

__all__ = [
    "DriveMode",
    "JointDescriptor",
    "JointHandle",
    "JointMode",
    "RigidBodyDescriptor",
    "WORLD_BODY",
    "WorldBuilder",
]


#: Sentinel for joint add helpers to refer to the static world anchor.
WORLD_BODY = 0


_IDENTITY_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ZERO_INERTIA = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_QUAT_NORM_TOL = 1e-3


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _all_finite(seq) -> bool:
    if isinstance(seq, (tuple, list)):
        return all(_all_finite(v) for v in seq)
    return _is_finite(seq)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DriveMode(IntEnum):
    """Drive mode for an actuated joint's free DoF. Values match the
    underlying Warp ``DRIVE_MODE_*`` constants exactly."""

    OFF = int(DRIVE_MODE_OFF)
    POSITION = int(DRIVE_MODE_POSITION)
    VELOCITY = int(DRIVE_MODE_VELOCITY)


class JointMode(IntEnum):
    """Which physical joint an ADBS descriptor materialises.

    * :attr:`REVOLUTE` -- 5-DoF hinge along ``anchor1 -> anchor2``.
    * :attr:`PRISMATIC` -- 5-DoF slider along ``anchor1 -> anchor2``.
    * :attr:`BALL_SOCKET` -- 3-DoF point lock at ``anchor1``
      (rotations free, no drive/limit).
    """

    REVOLUTE = int(JOINT_MODE_REVOLUTE)
    PRISMATIC = int(JOINT_MODE_PRISMATIC)
    BALL_SOCKET = int(JOINT_MODE_BALL_SOCKET)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass
class RigidBodyDescriptor:
    """Plain-Python description of one rigid body. Defaults produce a
    valid static body at the origin.

    Units: ``position`` [m], ``inverse_mass`` [1/kg], ``inverse_inertia``
    [1/(kg m^2)] in *body* frame, ``velocity`` [m/s], ``angular_velocity``
    [rad/s]."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    motion_type: int = int(MOTION_STATIC)
    inverse_mass: float = 0.0
    inverse_inertia: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = _ZERO_INERTIA
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    affected_by_gravity: bool = True
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Index of the world this body belongs to. Must be in
    #: ``[0, num_worlds)``.
    world_id: int = 0


@dataclass
class JointDescriptor:
    """Plain-Python description of one actuated double-ball-socket joint.

    The ``mode`` selects ball-socket / revolute / prismatic. For
    revolute and prismatic, ``anchor1`` and ``anchor2`` are two
    world-space points on the joint axis at finalize() time. For
    ball-socket, ``anchor2`` is internally set to ``anchor1``.

    Drive: ``stiffness_drive == damping_drive == 0`` disables the
    drive row. ``max_force_drive > 0`` caps the per-substep impulse.

    Limit: ``min_value > max_value`` disables the limit row. If either
    PD gain (``stiffness_limit`` / ``damping_limit``) is positive the
    PD formulation is used; otherwise the Box2D
    ``(hertz_limit, damping_ratio_limit)`` formulation applies.
    """

    body1: int
    body2: int
    anchor1: tuple[float, float, float]
    anchor2: tuple[float, float, float]
    mode: JointMode = JointMode.REVOLUTE
    hertz: float = float(DEFAULT_HERTZ_LINEAR)
    damping_ratio: float = float(DEFAULT_DAMPING_RATIO)
    drive_mode: DriveMode = DriveMode.OFF
    target: float = 0.0
    target_velocity: float = 0.0
    max_force_drive: float = 0.0
    stiffness_drive: float = 0.0
    damping_drive: float = 0.0
    min_value: float = 1.0
    max_value: float = -1.0
    hertz_limit: float = float(DEFAULT_HERTZ_LIMIT)
    damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO)
    stiffness_limit: float = 0.0
    damping_limit: float = 0.0


@dataclass
class JointHandle:
    """Global cid of a joint created by :meth:`WorldBuilder.add_joint`.

    Returned with sentinel ``cid = -1``; rewritten in place by
    :meth:`finalize` to the joint's cid in the shared
    :class:`ConstraintContainer`."""

    cid: int = -1


# ---------------------------------------------------------------------------
# Body descriptor validation
# ---------------------------------------------------------------------------


def _validate_body_descriptor(desc: RigidBodyDescriptor, body_index: int) -> None:
    """Reject obviously-broken descriptors: NaN fields, negative
    inverse mass, dynamic body with zero inverse mass, non-unit
    quaternion, static body with non-zero velocity / mass."""
    prefix = f"add_body(body_index={body_index}): "

    if not _all_finite(desc.position):
        raise ValueError(prefix + "position has non-finite component")
    if not _all_finite(desc.orientation):
        raise ValueError(prefix + "orientation has non-finite component")
    if not _all_finite(desc.velocity):
        raise ValueError(prefix + "velocity has non-finite component")
    if not _all_finite(desc.angular_velocity):
        raise ValueError(prefix + "angular_velocity has non-finite component")
    if not _is_finite(desc.inverse_mass):
        raise ValueError(prefix + "inverse_mass not finite")
    if desc.inverse_mass < 0.0:
        raise ValueError(prefix + f"inverse_mass must be >= 0 (got {desc.inverse_mass})")
    if not _all_finite(desc.inverse_inertia):
        raise ValueError(prefix + "inverse_inertia has non-finite component")

    q = desc.orientation
    qnorm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if abs(qnorm - 1.0) > _QUAT_NORM_TOL:
        raise ValueError(
            prefix + f"orientation quaternion must be unit-norm (got |q|={qnorm:.6f})"
        )

    mt = int(desc.motion_type)
    if mt == int(MOTION_DYNAMIC) and desc.inverse_mass == 0.0:
        raise ValueError(prefix + "DYNAMIC body must have inverse_mass > 0")
    if mt == int(MOTION_STATIC):
        if desc.inverse_mass != 0.0:
            raise ValueError(prefix + "STATIC body must have inverse_mass == 0")
        if desc.velocity != (0.0, 0.0, 0.0) or desc.angular_velocity != (0.0, 0.0, 0.0):
            raise ValueError(prefix + "STATIC body must have zero velocities")
    if mt not in (int(MOTION_STATIC), int(MOTION_KINEMATIC), int(MOTION_DYNAMIC)):
        raise ValueError(
            prefix + f"unknown motion_type {mt} (expected STATIC/KINEMATIC/DYNAMIC)"
        )


# ---------------------------------------------------------------------------
# WorldBuilder
# ---------------------------------------------------------------------------


class WorldBuilder:
    """Append bodies and joints, then materialise a :class:`PhoenXWorld`.

    Usage::

        b = WorldBuilder()
        anchor = b.add_static_body()
        body_a = b.add_dynamic_body(position=(0, 0, 0), inverse_mass=1.0)
        body_b = b.add_dynamic_body(position=(1, 0, 0), inverse_mass=1.0)
        b.add_joint(body_a, body_b, anchor1=(0.5, 0, 0), mode=JointMode.BALL_SOCKET)
        world = b.finalize(substeps=1, solver_iterations=8)

    Single-use; call :meth:`finalize` exactly once.
    """

    def __init__(self, num_worlds: int = 1):
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1 (got {num_worlds})")
        self._num_worlds: int = int(num_worlds)
        # One static world anchor per world. The ``world_id`` field
        # drives the per-world CSR bucketing.
        self._bodies: list[RigidBodyDescriptor] = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                world_id=w,
            )
            for w in range(self._num_worlds)
        ]
        self._joint_descriptors: list[JointDescriptor] = []
        self._joint_handles: list[JointHandle] = []
        self._collision_filter_pairs: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Body API
    # ------------------------------------------------------------------

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    @property
    def world_body(self) -> int:
        """Index of world 0's static anchor."""
        return 0

    def world_body_of(self, world_id: int) -> int:
        """Body index of world ``world_id``'s static anchor."""
        if not (0 <= world_id < self._num_worlds):
            raise IndexError(
                f"world_id {world_id} out of range [0, {self._num_worlds})"
            )
        return world_id

    def add_body(self, descriptor: RigidBodyDescriptor) -> int:
        """Append a validated descriptor and return its body index."""
        next_index = len(self._bodies)
        _validate_body_descriptor(descriptor, next_index)
        if not (0 <= descriptor.world_id < self._num_worlds):
            raise ValueError(
                f"body {next_index} has world_id {descriptor.world_id} out of "
                f"range [0, {self._num_worlds})"
            )
        self._bodies.append(descriptor)
        return next_index

    def add_static_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        world_id: int = 0,
    ) -> int:
        """Append a static body (zero inverse mass / inertia, no integration)."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_STATIC),
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                world_id=world_id,
            )
        )

    def add_kinematic_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        world_id: int = 0,
    ) -> int:
        """Append a kinematic body: integrates its user-set velocities
        but ignores forces and contacts."""
        return self.add_body(
            RigidBodyDescriptor(
                position=position,
                orientation=orientation,
                motion_type=int(MOTION_KINEMATIC),
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                velocity=velocity,
                angular_velocity=angular_velocity,
                world_id=world_id,
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
        world_id: int = 0,
    ) -> int:
        """Append a dynamic body. ``inverse_inertia`` is in the body
        frame; the solver rotates it to world space every step."""
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
                world_id=world_id,
            )
        )

    # ------------------------------------------------------------------
    # Joint API
    # ------------------------------------------------------------------

    def add_joint(
        self,
        body1: int,
        body2: int,
        anchor1: tuple[float, float, float],
        anchor2: tuple[float, float, float] | None = None,
        mode: JointMode = JointMode.REVOLUTE,
        hertz: float = float(DEFAULT_HERTZ_LINEAR),
        damping_ratio: float = float(DEFAULT_DAMPING_RATIO),
        drive_mode: DriveMode = DriveMode.OFF,
        target: float = 0.0,
        target_velocity: float = 0.0,
        max_force_drive: float = 0.0,
        stiffness_drive: float = 0.0,
        damping_drive: float = 0.0,
        min_value: float = 1.0,
        max_value: float = -1.0,
        hertz_limit: float = float(DEFAULT_HERTZ_LIMIT),
        damping_ratio_limit: float = float(DEFAULT_DAMPING_RATIO),
        stiffness_limit: float = 0.0,
        damping_limit: float = 0.0,
    ) -> JointHandle:
        """Append an actuated double-ball-socket joint and return its handle.

        Modes:

        * :attr:`JointMode.BALL_SOCKET` -- 3-DoF point lock at
          ``anchor1``. ``anchor2`` must be ``None``; drive and limit
          fields must be left at defaults.
        * :attr:`JointMode.REVOLUTE` -- 5-DoF hinge about the line
          from ``anchor1`` to ``anchor2``. Drive / limit interpret
          ``target`` / ``min_value`` / ``max_value`` as angles [rad]
          and ``max_force_drive`` as torque [N*m].
        * :attr:`JointMode.PRISMATIC` -- 5-DoF slider along
          ``anchor1 -> anchor2``. Drive / limit interpret values as
          displacements [m] along the axis and ``max_force_drive`` as
          force [N].

        ``stiffness_drive == damping_drive == 0`` disables the drive
        row; ``min_value > max_value`` disables the limit row. Setting
        either limit PD gain positive selects the PD formulation over
        the Box2D ``(hertz_limit, damping_ratio_limit)`` path.

        Returns:
            A :class:`JointHandle` whose ``cid`` field is rewritten in
            place by :meth:`finalize` to the joint's global cid.
        """
        self._validate_body(body1)
        self._validate_body(body2)

        mode_enum = JointMode(int(mode))
        drive_mode_enum = DriveMode(int(drive_mode))

        if mode_enum is JointMode.BALL_SOCKET:
            if anchor2 is not None:
                raise ValueError(
                    "add_joint(mode=BALL_SOCKET) must not receive an ``anchor2``"
                )
            if drive_mode_enum is not DriveMode.OFF:
                raise ValueError(
                    "add_joint(mode=BALL_SOCKET) has no drive row; "
                    "leave drive_mode=DriveMode.OFF"
                )
            if target != 0.0 or target_velocity != 0.0 or max_force_drive != 0.0:
                raise ValueError(
                    "add_joint(mode=BALL_SOCKET) has no drive row; leave "
                    "target/target_velocity/max_force_drive at defaults"
                )
            if min_value <= max_value:
                raise ValueError(
                    "add_joint(mode=BALL_SOCKET) has no limit row; leave "
                    "min_value/max_value at defaults (min > max disables)"
                )
            if stiffness_limit != 0.0 or damping_limit != 0.0:
                raise ValueError(
                    "add_joint(mode=BALL_SOCKET) has no limit row; leave "
                    "stiffness_limit/damping_limit at 0"
                )
            anchor2_effective: tuple[float, float, float] = tuple(anchor1)  # type: ignore[assignment]
        else:
            if anchor2 is None:
                raise ValueError(
                    f"add_joint(mode={mode_enum.name}) requires ``anchor2``; "
                    "the line anchor1 -> anchor2 defines the joint axis"
                )
            anchor2_effective = anchor2
            # VELOCITY drive requires damping_drive > 0. A zero-damping
            # VELOCITY drive used to fall through to a pure-velocity
            # motor; that path is gone in the unified joint.
            if drive_mode_enum is DriveMode.VELOCITY and damping_drive <= 0.0:
                raise ValueError(
                    f"add_joint(mode={mode_enum.name}, drive_mode=VELOCITY) "
                    "requires damping_drive > 0 (PD velocity servo)."
                )

        descriptor = JointDescriptor(
            body1=body1,
            body2=body2,
            anchor1=anchor1,
            anchor2=anchor2_effective,
            mode=mode_enum,
            hertz=hertz,
            damping_ratio=damping_ratio,
            drive_mode=drive_mode_enum,
            target=target,
            target_velocity=target_velocity,
            max_force_drive=max_force_drive,
            stiffness_drive=stiffness_drive,
            damping_drive=damping_drive,
            min_value=min_value,
            max_value=max_value,
            hertz_limit=hertz_limit,
            damping_ratio_limit=damping_ratio_limit,
            stiffness_limit=stiffness_limit,
            damping_limit=damping_limit,
        )
        handle = JointHandle(cid=-1)
        self._joint_descriptors.append(descriptor)
        self._joint_handles.append(handle)
        return handle

    def add_collision_filter_pair(self, body_a: int, body_b: int) -> None:
        """Ignore contacts between bodies ``body_a`` and ``body_b``.
        Pair is stored canonical ``(min, max)``; self-pair rejected."""
        self._validate_body(body_a)
        self._validate_body(body_b)
        if body_a == body_b:
            raise ValueError(
                f"add_collision_filter_pair: bodies must differ (got both = {body_a})"
            )
        self._collision_filter_pairs.add(
            (min(int(body_a), int(body_b)), max(int(body_a), int(body_b)))
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(
        self,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        position_iterations: int = 0,
        gravity: tuple[float, float, float]
        | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        max_contact_columns: int = 0,
        rigid_contact_max: int = 0,
        num_shapes: int = 0,
        default_friction: float = 0.5,
        device: wp.context.Devicelike = None,
    ) -> PhoenXWorld:
        """Allocate GPU storage and build a ready-to-step :class:`PhoenXWorld`.

        Descriptor lists are consumed in place (cleared after
        success). Calling :meth:`finalize` twice returns a new empty
        world the second time.
        """
        device = wp.get_device(device)

        num_joints = len(self._joint_descriptors)
        bodies = self._build_body_container(device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            max_contact_columns=int(max_contact_columns),
            device=device,
        )

        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
            position_iterations=position_iterations,
            gravity=gravity,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            num_joints=num_joints,
            collision_filter_pairs=self._collision_filter_pairs,
            default_friction=default_friction,
            num_worlds=self._num_worlds,
            device=device,
        )

        if num_joints > 0:
            joint_arrays = self._pack_joint_arrays(device)
            world.initialize_actuated_double_ball_socket_joints(**joint_arrays)
            # Patch handle cids (joints occupy cids [0, num_joints)).
            for i, handle in enumerate(self._joint_handles):
                handle.cid = i

        # Reset internal state so the builder can't leak references
        # into a second finalize call.
        self._bodies = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                world_id=w,
            )
            for w in range(self._num_worlds)
        ]
        self._joint_descriptors = []
        self._joint_handles = []
        self._collision_filter_pairs = set()
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
        """Pack descriptors into a :class:`BodyContainer`. One
        host-to-device transfer per field."""
        n = len(self._bodies)
        positions = np.zeros((n, 3), dtype=np.float32)
        orientations = np.zeros((n, 4), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        angular_velocities = np.zeros((n, 3), dtype=np.float32)
        inverse_inertia = np.zeros((n, 3, 3), dtype=np.float32)
        inverse_mass = np.zeros(n, dtype=np.float32)
        linear_damping = np.ones(n, dtype=np.float32)
        angular_damping = np.ones(n, dtype=np.float32)
        affected_by_gravity = np.ones(n, dtype=np.int32)
        motion_type = np.full(n, int(MOTION_STATIC), dtype=np.int32)
        world_id_arr = np.zeros(n, dtype=np.int32)

        for i, b in enumerate(self._bodies):
            positions[i] = b.position
            orientations[i] = b.orientation
            velocities[i] = b.velocity
            angular_velocities[i] = b.angular_velocity
            inverse_inertia[i] = b.inverse_inertia
            inverse_mass[i] = b.inverse_mass
            linear_damping[i] = b.linear_damping
            angular_damping[i] = b.angular_damping
            affected_by_gravity[i] = 1 if b.affected_by_gravity else 0
            motion_type[i] = int(b.motion_type)
            world_id_arr[i] = int(b.world_id)

        # Seed inverse_inertia_world with the body-frame value; the
        # first _update_inertia launch will rotate it into world space.
        c = BodyContainer()
        c.position = wp.array(positions, dtype=wp.vec3f, device=device)
        c.velocity = wp.array(velocities, dtype=wp.vec3f, device=device)
        c.angular_velocity = wp.array(angular_velocities, dtype=wp.vec3f, device=device)
        c.orientation = wp.array(orientations, dtype=wp.quatf, device=device)
        # Bodies added via the builder assume mesh origin == COM;
        # callers with meshed offsets set ``bodies.body_com`` directly.
        c.body_com = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.inverse_inertia_world = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
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
        c.world_id = wp.array(world_id_arr, dtype=wp.int32, device=device)
        return c

    def _pack_joint_arrays(self, device: wp.context.Device) -> dict:
        """Pack the joint descriptor list into ``wp.array`` kwargs for
        :meth:`PhoenXWorld.initialize_actuated_double_ball_socket_joints`."""
        n = len(self._joint_descriptors)
        body1 = np.empty(n, dtype=np.int32)
        body2 = np.empty(n, dtype=np.int32)
        anchor1 = np.empty((n, 3), dtype=np.float32)
        anchor2 = np.empty((n, 3), dtype=np.float32)
        hertz = np.empty(n, dtype=np.float32)
        damping_ratio = np.empty(n, dtype=np.float32)
        joint_mode = np.empty(n, dtype=np.int32)
        drive_mode = np.empty(n, dtype=np.int32)
        target = np.empty(n, dtype=np.float32)
        target_velocity = np.empty(n, dtype=np.float32)
        max_force_drive = np.empty(n, dtype=np.float32)
        stiffness_drive = np.empty(n, dtype=np.float32)
        damping_drive = np.empty(n, dtype=np.float32)
        min_value = np.empty(n, dtype=np.float32)
        max_value = np.empty(n, dtype=np.float32)
        hertz_limit = np.empty(n, dtype=np.float32)
        damping_ratio_limit = np.empty(n, dtype=np.float32)
        stiffness_limit = np.empty(n, dtype=np.float32)
        damping_limit = np.empty(n, dtype=np.float32)

        for i, d in enumerate(self._joint_descriptors):
            body1[i] = int(d.body1)
            body2[i] = int(d.body2)
            anchor1[i] = d.anchor1
            anchor2[i] = d.anchor2
            hertz[i] = float(d.hertz)
            damping_ratio[i] = float(d.damping_ratio)
            joint_mode[i] = int(d.mode)
            drive_mode[i] = int(d.drive_mode)
            target[i] = float(d.target)
            target_velocity[i] = float(d.target_velocity)
            max_force_drive[i] = float(d.max_force_drive)
            stiffness_drive[i] = float(d.stiffness_drive)
            damping_drive[i] = float(d.damping_drive)
            min_value[i] = float(d.min_value)
            max_value[i] = float(d.max_value)
            hertz_limit[i] = float(d.hertz_limit)
            damping_ratio_limit[i] = float(d.damping_ratio_limit)
            stiffness_limit[i] = float(d.stiffness_limit)
            damping_limit[i] = float(d.damping_limit)

        return dict(
            body1=wp.array(body1, dtype=wp.int32, device=device),
            body2=wp.array(body2, dtype=wp.int32, device=device),
            anchor1=wp.array(anchor1, dtype=wp.vec3f, device=device),
            anchor2=wp.array(anchor2, dtype=wp.vec3f, device=device),
            hertz=wp.array(hertz, dtype=wp.float32, device=device),
            damping_ratio=wp.array(damping_ratio, dtype=wp.float32, device=device),
            joint_mode=wp.array(joint_mode, dtype=wp.int32, device=device),
            drive_mode=wp.array(drive_mode, dtype=wp.int32, device=device),
            target=wp.array(target, dtype=wp.float32, device=device),
            target_velocity=wp.array(target_velocity, dtype=wp.float32, device=device),
            max_force_drive=wp.array(max_force_drive, dtype=wp.float32, device=device),
            stiffness_drive=wp.array(stiffness_drive, dtype=wp.float32, device=device),
            damping_drive=wp.array(damping_drive, dtype=wp.float32, device=device),
            min_value=wp.array(min_value, dtype=wp.float32, device=device),
            max_value=wp.array(max_value, dtype=wp.float32, device=device),
            hertz_limit=wp.array(hertz_limit, dtype=wp.float32, device=device),
            damping_ratio_limit=wp.array(damping_ratio_limit, dtype=wp.float32, device=device),
            stiffness_limit=wp.array(stiffness_limit, dtype=wp.float32, device=device),
            damping_limit=wp.array(damping_limit, dtype=wp.float32, device=device),
        )
