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
from dataclasses import replace as dataclass_replace
from enum import IntEnum

import numpy as np
import warp as wp

from newton._src.geometry.inertia import compute_inertia_shape, transform_inertia
from newton._src.geometry.types import GeoType, Heightfield, Mesh
from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

__all__ = [
    "WORLD_BODY",
    "DriveMode",
    "JointDescriptor",
    "JointHandle",
    "JointMode",
    "RigidBodyDescriptor",
    "ShapeDescriptor",
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
    * :attr:`FIXED` -- 6-DoF weld along ``anchor1 -> anchor2``
      (no drive/limit).
    * :attr:`CABLE` -- soft fixed joint built on the same 3+2+1 row
      layout as :attr:`FIXED` but with PD softness on the anchor-2
      tangent rows (bend) and the anchor-3 scalar row (twist). User
      supplies ``bend_stiffness`` / ``bend_damping`` [N*m/rad,
      N*m*s/rad] and ``twist_stiffness`` / ``twist_damping``
      [N*m/rad, N*m*s/rad]; gains are rescaled by ``1 / rest_length^2``
      to obtain the equivalent positional spring at the lever-armed
      anchors. Converges to revolute as ``k_bend -> infinity`` and to
      :attr:`FIXED` as both gains diverge. The user-facing kwargs
      reuse the ADBS column's drive / limit slots: ``bend_*`` ->
      drive slots, ``twist_*`` -> limit slots.
    """

    REVOLUTE = int(JOINT_MODE_REVOLUTE)
    PRISMATIC = int(JOINT_MODE_PRISMATIC)
    BALL_SOCKET = int(JOINT_MODE_BALL_SOCKET)
    FIXED = int(JOINT_MODE_FIXED)
    CABLE = int(JOINT_MODE_CABLE)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

#: Geometries that are always static and contribute no mass: infinite
#: half-space (``PLANE``) and terrain heightfields (``HFIELD``).
_MASSLESS_STATIC_TYPES = frozenset({GeoType.PLANE, GeoType.HFIELD})


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
    #: Initial substep-start position snapshot, mirroring Jitter2's
    #: ``body.Position`` viewed from ``TinyRigidState``
    #: (``MassSplitting/TinyRigidState.cs``). Overwritten every
    #: substep by the access-mode pre-pass; the descriptor value seeds
    #: only the first substep so a constraint flipping a body to
    #: ``POSITION_LEVEL`` before any substep ran sees a sensible
    #: snapshot. Defaults to ``position`` if left at ``None``.
    position_prev_substep: tuple[float, float, float] | None = None
    #: Initial per-body access mode tag. See
    #: :mod:`newton._src.solvers.phoenx.access_mode` for the integer
    #: values. Defaults to
    #: :data:`~newton._src.solvers.phoenx.access_mode.ACCESS_MODE_VELOCITY_LEVEL`,
    #: matching the per-substep reset.
    access_mode: int = int(ACCESS_MODE_VELOCITY_LEVEL)


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


@dataclass
class ShapeDescriptor:
    """Plain-Python description of one collider attached to a body.

    Geometry parametrisation lives in :attr:`scale` (and, for mesh /
    heightfield shapes, in :attr:`mesh` / :attr:`hfield`). The exact
    meaning of ``scale`` is chosen to match
    :func:`newton._src.geometry.inertia.compute_inertia_shape`, which
    the builder defers to for mass / inertia. By type:

    * :attr:`~newton.GeoType.SPHERE`: ``scale = (radius, _, _)`` [m].
    * :attr:`~newton.GeoType.BOX`: ``scale = (hx, hy, hz)`` half-extents [m].
    * :attr:`~newton.GeoType.CAPSULE` / :attr:`~newton.GeoType.CYLINDER`
      / :attr:`~newton.GeoType.CONE`:
      ``scale = (radius, half_height, _)`` [m], aligned along local +z.
    * :attr:`~newton.GeoType.ELLIPSOID`: ``scale = (rx, ry, rz)``
      semi-axes [m].
    * :attr:`~newton.GeoType.MESH` / :attr:`~newton.GeoType.CONVEX_MESH`:
      ``scale`` is a per-axis multiplier on :attr:`mesh` vertices.
    * :attr:`~newton.GeoType.PLANE`: ``geom_vec3 = normal`` (unit),
      ``geom_scalar_a = offset`` along the normal. Static-only,
      infinite half-space; contributes no mass.
    * :attr:`~newton.GeoType.HFIELD`: ``scale`` is the per-axis
      multiplier on :attr:`hfield`. Always static; contributes no mass.

    ``local_pos`` / ``local_rot`` give the shape's pose in the parent
    body's local frame; at finalize the builder uses them to translate
    / rotate the per-shape inertia tensor into body coordinates.

    Mass sourcing: exactly one of ``density`` (kg/m^3) or ``mass`` (kg)
    may be set. If neither is set, the shape is mass-less (collision-
    only); :meth:`WorldBuilder.finalize` does not fold it into the
    body's compound inertia. ``is_solid`` / ``thickness`` are forwarded
    to :func:`compute_inertia_shape` for hollow primitives.
    """

    body: int
    shape_type: GeoType
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Plane normal (unit) in the parent body's local frame. Only used
    #: when :attr:`shape_type` is :attr:`~newton.GeoType.PLANE`.
    plane_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: Plane offset along :attr:`plane_normal` [m]. Only used when
    #: :attr:`shape_type` is :attr:`~newton.GeoType.PLANE`.
    plane_offset: float = 0.0
    #: Mesh source for ``MESH`` / ``CONVEX_MESH`` shape types.
    mesh: Mesh | None = None
    #: Heightfield source for the ``HFIELD`` shape type.
    hfield: Heightfield | None = None
    is_solid: bool = True
    thickness: float = 0.001
    density: float | None = None
    mass: float | None = None
    material_id: int | None = None


# ---------------------------------------------------------------------------
# Shape mass / inertia
# ---------------------------------------------------------------------------


def _shape_mass_com_inertia(desc: ShapeDescriptor) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mass, com_local, inertia_about_com)`` for one shape,
    expressed in the *shape's* local frame.

    Defers to :func:`newton._src.geometry.inertia.compute_inertia_shape`
    for the per-type closed forms, then rescales the result if the user
    pinned ``desc.mass`` instead of ``desc.density``.
    """
    if desc.shape_type in _MASSLESS_STATIC_TYPES or (desc.density is None and desc.mass is None):
        return 0.0, np.zeros(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)

    src: Mesh | Heightfield | None
    if desc.shape_type in (GeoType.MESH, GeoType.CONVEX_MESH):
        src = desc.mesh
    elif desc.shape_type == GeoType.HFIELD:
        src = desc.hfield
    else:
        src = None

    # ``compute_inertia_shape`` takes density directly. When the user
    # pinned ``mass`` instead, run with unit density to discover the
    # shape's natural unit-density mass and then rescale.
    density = float(desc.density) if desc.density is not None else 1.0
    m, c, i = compute_inertia_shape(
        type=int(desc.shape_type),
        scale=wp.vec3(*(float(x) for x in desc.scale)),
        src=src,
        density=density,
        is_solid=desc.is_solid,
        thickness=float(desc.thickness),
    )
    m = float(m)
    com = np.array([c[0], c[1], c[2]], dtype=np.float64)
    inertia = np.array(i, dtype=np.float64).reshape(3, 3)

    if desc.mass is not None:
        if m <= 0.0:
            raise ValueError(
                f"shape of type {desc.shape_type.name} has zero unit-density "
                "mass; cannot rescale to user-supplied mass (check geometry)"
            )
        rescale = float(desc.mass) / m
        m = float(desc.mass)
        inertia = inertia * rescale
    return m, com, inertia


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
        raise ValueError(prefix + f"orientation quaternion must be unit-norm (got |q|={qnorm:.6f})")

    mt = int(desc.motion_type)
    if mt == int(MOTION_DYNAMIC) and desc.inverse_mass == 0.0:
        raise ValueError(prefix + "DYNAMIC body must have inverse_mass > 0")
    if mt == int(MOTION_STATIC):
        if desc.inverse_mass != 0.0:
            raise ValueError(prefix + "STATIC body must have inverse_mass == 0")
        if desc.velocity != (0.0, 0.0, 0.0) or desc.angular_velocity != (0.0, 0.0, 0.0):
            raise ValueError(prefix + "STATIC body must have zero velocities")
    if mt not in (int(MOTION_STATIC), int(MOTION_KINEMATIC), int(MOTION_DYNAMIC)):
        raise ValueError(prefix + f"unknown motion_type {mt} (expected STATIC/KINEMATIC/DYNAMIC)")


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
        self._shapes: list[ShapeDescriptor] = []

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
            raise IndexError(f"world_id {world_id} out of range [0, {self._num_worlds})")
        return world_id

    def add_body(self, descriptor: RigidBodyDescriptor) -> int:
        """Append a validated descriptor and return its body index."""
        next_index = len(self._bodies)
        _validate_body_descriptor(descriptor, next_index)
        if not (0 <= descriptor.world_id < self._num_worlds):
            raise ValueError(
                f"body {next_index} has world_id {descriptor.world_id} out of range [0, {self._num_worlds})"
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
    # Shape API
    # ------------------------------------------------------------------
    #
    # Optional: users can manage mass/inertia by hand via explicit
    # ``inverse_mass`` / ``inverse_inertia`` on :meth:`add_dynamic_body`,
    # or attach shapes with ``density`` (kg/m^3) or ``mass`` (kg) and
    # let :meth:`finalize` sum via the parallel-axis theorem (overwriting
    # the body's inverse mass/inertia).
    #
    # Mixing modes is an error: a shape with a mass source AND a body
    # with explicit non-default mass/inertia makes :meth:`finalize`
    # raise ``ValueError`` -- intent is declared once, on the shape.

    def _attach_shape(self, desc: ShapeDescriptor) -> int:
        """Validate + append a shape, return its index."""
        self._validate_body(desc.body)
        if desc.density is not None and desc.mass is not None:
            raise ValueError(f"add_shape(body={desc.body}): set exactly one of density (kg/m^3) or mass (kg), not both")
        if desc.density is not None and desc.density <= 0.0:
            raise ValueError(f"add_shape(body={desc.body}): density must be > 0 (got {desc.density})")
        if desc.mass is not None and desc.mass <= 0.0:
            raise ValueError(f"add_shape(body={desc.body}): mass must be > 0 (got {desc.mass})")
        if not _all_finite(desc.local_pos):
            raise ValueError(f"add_shape(body={desc.body}): local_pos has non-finite component")
        if not _all_finite(desc.local_rot):
            raise ValueError(f"add_shape(body={desc.body}): local_rot has non-finite component")
        # Planes / heightfields are static-only and cannot carry mass.
        if desc.shape_type in _MASSLESS_STATIC_TYPES and (desc.density is not None or desc.mass is not None):
            kind = desc.shape_type.name.lower()
            raise ValueError(
                f"add_shape_{kind}(body={desc.body}): {kind}s are infinite "
                "static surfaces and cannot contribute mass; omit density / mass"
            )
        shape_id = len(self._shapes)
        self._shapes.append(desc)
        return shape_id

    def add_shape_sphere(
        self,
        body: int,
        radius: float,
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a sphere collider to ``body``. Returns the shape index.

        Args:
            body: Parent body index.
            radius: Sphere radius [m], must be > 0.
            local_pos: Shape origin in the body's local frame [m].
            local_rot: Shape orientation relative to the body as a
                ``(x, y, z, w)`` quaternion. Meaningless for a sphere
                (rotation-invariant) but accepted for symmetry with
                the other shape types.
            density: Uniform density [kg / m^3]. Set this to have the
                builder compute the body's compound mass / inertia
                automatically.
            mass: Total mass [kg]. Alternative to ``density``; exactly
                one of the two may be set.
            is_solid: If ``False``, treat the sphere as a hollow shell
                of the given ``thickness`` for inertia purposes.
            thickness: Wall thickness [m] for hollow shells.
            material_id: Index into the material table; ``None`` uses
                the solver's ``default_friction`` at contact time.
        """
        if radius <= 0.0 or not _is_finite(radius):
            raise ValueError(f"add_shape_sphere(body={body}): radius must be > 0 (got {radius})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=GeoType.SPHERE,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=(float(radius), float(radius), float(radius)),
                density=density,
                mass=mass,
                is_solid=is_solid,
                thickness=float(thickness),
                material_id=material_id,
            )
        )

    def add_shape_box(
        self,
        body: int,
        half_extents: tuple[float, float, float],
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a box collider to ``body``.

        Args:
            body: Parent body index.
            half_extents: ``(hx, hy, hz)`` in the shape's local frame [m],
                each > 0.
            local_pos / local_rot: Shape pose in the body's local frame.
            density / mass: See :meth:`add_shape_sphere`.
            is_solid / thickness: See :meth:`add_shape_sphere`.
            material_id: See :meth:`add_shape_sphere`.
        """
        if any((not _is_finite(h)) or h <= 0.0 for h in half_extents):
            raise ValueError(f"add_shape_box(body={body}): half_extents must all be > 0 (got {half_extents})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=GeoType.BOX,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=tuple(float(h) for h in half_extents),
                density=density,
                mass=mass,
                is_solid=is_solid,
                thickness=float(thickness),
                material_id=material_id,
            )
        )

    def _add_radius_height_shape(
        self,
        *,
        kind: GeoType,
        body: int,
        radius: float,
        half_height: float,
        local_pos: tuple[float, float, float],
        local_rot: tuple[float, float, float, float],
        density: float | None,
        mass: float | None,
        is_solid: bool,
        thickness: float,
        material_id: int | None,
        allow_zero_height: bool,
    ) -> int:
        """Shared validation for the capsule / cylinder / cone helpers.
        All three are aligned along local +z and parametrised by
        ``(radius, half_height)``.
        """
        name = f"add_shape_{kind.name.lower()}"
        if radius <= 0.0 or not _is_finite(radius):
            raise ValueError(f"{name}(body={body}): radius must be > 0 (got {radius})")
        min_h = 0.0 if allow_zero_height else -math.inf
        if half_height < min_h or not _is_finite(half_height):
            cmp = ">= 0" if allow_zero_height else "finite"
            raise ValueError(f"{name}(body={body}): half_height must be {cmp} (got {half_height})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=kind,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=(float(radius), float(half_height), float(radius)),
                density=density,
                mass=mass,
                is_solid=is_solid,
                thickness=float(thickness),
                material_id=material_id,
            )
        )

    def add_shape_capsule(
        self,
        body: int,
        radius: float,
        half_height: float,
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a capsule collider (cylinder + two hemispheres) to
        ``body``. The capsule is aligned along its local +z axis.

        Args:
            body: Parent body index.
            radius: Hemisphere / cylinder radius [m], > 0.
            half_height: Half-length of the cylindrical mid-section [m],
                >= 0 (``0`` collapses to a sphere).
            local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_sphere`.
        """
        return self._add_radius_height_shape(
            kind=GeoType.CAPSULE,
            body=body,
            radius=radius,
            half_height=half_height,
            local_pos=local_pos,
            local_rot=local_rot,
            density=density,
            mass=mass,
            is_solid=is_solid,
            thickness=thickness,
            material_id=material_id,
            allow_zero_height=True,
        )

    def add_shape_cylinder(
        self,
        body: int,
        radius: float,
        half_height: float,
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a cylinder collider to ``body``. The cylinder is
        aligned along its local +z axis.

        Args:
            body: Parent body index.
            radius: Cylinder radius [m], > 0.
            half_height: Half-length [m], > 0.
            local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_sphere`.
        """
        if half_height <= 0.0:
            raise ValueError(f"add_shape_cylinder(body={body}): half_height must be > 0 (got {half_height})")
        return self._add_radius_height_shape(
            kind=GeoType.CYLINDER,
            body=body,
            radius=radius,
            half_height=half_height,
            local_pos=local_pos,
            local_rot=local_rot,
            density=density,
            mass=mass,
            is_solid=is_solid,
            thickness=thickness,
            material_id=material_id,
            allow_zero_height=False,
        )

    def add_shape_cone(
        self,
        body: int,
        radius: float,
        half_height: float,
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a cone collider to ``body``. The cone is aligned
        along its local +z axis with its base at ``-half_height`` and
        its apex at ``+half_height``.

        Args:
            body: Parent body index.
            radius: Base radius [m], > 0.
            half_height: Half-length along the axis [m], > 0.
            local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_sphere`.
        """
        if half_height <= 0.0:
            raise ValueError(f"add_shape_cone(body={body}): half_height must be > 0 (got {half_height})")
        return self._add_radius_height_shape(
            kind=GeoType.CONE,
            body=body,
            radius=radius,
            half_height=half_height,
            local_pos=local_pos,
            local_rot=local_rot,
            density=density,
            mass=mass,
            is_solid=is_solid,
            thickness=thickness,
            material_id=material_id,
            allow_zero_height=False,
        )

    def add_shape_ellipsoid(
        self,
        body: int,
        semi_axes: tuple[float, float, float],
        *,
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach an ellipsoid collider to ``body``.

        Args:
            body: Parent body index.
            semi_axes: ``(rx, ry, rz)`` semi-axes [m], each > 0.
            local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_sphere`.
        """
        if any((not _is_finite(r)) or r <= 0.0 for r in semi_axes):
            raise ValueError(f"add_shape_ellipsoid(body={body}): semi_axes must all be > 0 (got {semi_axes})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=GeoType.ELLIPSOID,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=tuple(float(r) for r in semi_axes),
                density=density,
                mass=mass,
                is_solid=is_solid,
                thickness=float(thickness),
                material_id=material_id,
            )
        )

    def _add_mesh_shape(
        self,
        *,
        kind: GeoType,
        body: int,
        mesh: Mesh,
        scale: tuple[float, float, float],
        local_pos: tuple[float, float, float],
        local_rot: tuple[float, float, float, float],
        density: float | None,
        mass: float | None,
        is_solid: bool,
        thickness: float,
        material_id: int | None,
    ) -> int:
        name = f"add_shape_{kind.name.lower()}"
        if mesh is None:
            raise ValueError(f"{name}(body={body}): mesh must be a Mesh instance, got None")
        if any((not _is_finite(s)) or s <= 0.0 for s in scale):
            raise ValueError(f"{name}(body={body}): scale must be > 0 in every axis (got {scale})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=kind,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=tuple(float(s) for s in scale),
                mesh=mesh,
                density=density,
                mass=mass,
                is_solid=is_solid,
                thickness=float(thickness),
                material_id=material_id,
            )
        )

    def add_shape_mesh(
        self,
        body: int,
        mesh: Mesh,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a triangle-mesh collider to ``body``. Mass / inertia
        are computed from the mesh geometry (or read from
        :attr:`Mesh.mass` / :attr:`Mesh.inertia` when available).

        Args:
            body: Parent body index.
            mesh: :class:`newton.Mesh` source.
            scale: Per-axis multiplier on the mesh vertices, all > 0.
            local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_sphere`.
        """
        return self._add_mesh_shape(
            kind=GeoType.MESH,
            body=body,
            mesh=mesh,
            scale=scale,
            local_pos=local_pos,
            local_rot=local_rot,
            density=density,
            mass=mass,
            is_solid=is_solid,
            thickness=thickness,
            material_id=material_id,
        )

    def add_shape_convex_mesh(
        self,
        body: int,
        mesh: Mesh,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        density: float | None = None,
        mass: float | None = None,
        is_solid: bool = True,
        thickness: float = 0.001,
        material_id: int | None = None,
    ) -> int:
        """Attach a convex-hull collider to ``body``. The hull is
        derived from ``mesh``; mass / inertia are computed identically
        to :meth:`add_shape_mesh`.

        Args:
            body: Parent body index.
            mesh: :class:`newton.Mesh` source for the hull's input
                vertices.
            scale / local_pos / local_rot / density / mass / is_solid /
                thickness / material_id: As for :meth:`add_shape_mesh`.
        """
        return self._add_mesh_shape(
            kind=GeoType.CONVEX_MESH,
            body=body,
            mesh=mesh,
            scale=scale,
            local_pos=local_pos,
            local_rot=local_rot,
            density=density,
            mass=mass,
            is_solid=is_solid,
            thickness=thickness,
            material_id=material_id,
        )

    def add_shape_plane(
        self,
        body: int,
        *,
        normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        offset: float = 0.0,
        material_id: int | None = None,
    ) -> int:
        """Attach an infinite-plane collider to ``body``. Planes
        carry no mass and must be attached to static bodies.

        Args:
            body: Parent body index (must be static).
            normal: World-space outward normal (unit vector).
            offset: Signed distance from the body's origin to the
                plane along ``normal`` [m].
            material_id: Material table index for contact resolution.
        """
        nlen = math.sqrt(sum(float(c) * float(c) for c in normal))
        if nlen <= 1e-12:
            raise ValueError(f"add_shape_plane(body={body}): normal must be non-zero")
        unit = tuple(float(c) / nlen for c in normal)
        if int(self._bodies[body].motion_type) != int(MOTION_STATIC):
            raise ValueError(f"add_shape_plane(body={body}): planes may only be attached to static bodies")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=GeoType.PLANE,
                plane_normal=unit,
                plane_offset=float(offset),
                material_id=material_id,
            )
        )

    def add_shape_hfield(
        self,
        body: int,
        hfield: Heightfield,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        material_id: int | None = None,
    ) -> int:
        """Attach a heightfield (terrain) collider to ``body``.
        Heightfields carry no mass and must be attached to static
        bodies.

        Args:
            body: Parent body index (must be static).
            hfield: :class:`newton.Heightfield` source.
            scale: Per-axis multiplier on the hfield extents.
            local_pos / local_rot: Shape pose in the body's local frame.
            material_id: Material table index for contact resolution.
        """
        if hfield is None:
            raise ValueError(f"add_shape_hfield(body={body}): hfield must be a Heightfield instance, got None")
        if int(self._bodies[body].motion_type) != int(MOTION_STATIC):
            raise ValueError(f"add_shape_hfield(body={body}): heightfields may only be attached to static bodies")
        if any((not _is_finite(s)) or s <= 0.0 for s in scale):
            raise ValueError(f"add_shape_hfield(body={body}): scale must be > 0 in every axis (got {scale})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=GeoType.HFIELD,
                local_pos=local_pos,
                local_rot=local_rot,
                scale=tuple(float(s) for s in scale),
                hfield=hfield,
                material_id=material_id,
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
        # CABLE mode only -- per-component bend / twist stiffness and damping.
        bend_stiffness: float = 0.0,
        twist_stiffness: float = 0.0,
        bend_damping: float = 0.0,
        twist_damping: float = 0.0,
    ) -> JointHandle:
        """Append an actuated double-ball-socket joint and return its handle.

        Modes:

        * :attr:`JointMode.BALL_SOCKET` -- 3-DoF point lock at
          ``anchor1``. ``anchor2`` must be ``None``; drive / limit
          must stay at defaults.
        * :attr:`JointMode.REVOLUTE` -- 5-DoF hinge about
          ``anchor1 -> anchor2``. Drive / limit values are angles
          [rad], ``max_force_drive`` is torque [N*m].
        * :attr:`JointMode.PRISMATIC` -- 5-DoF slider along
          ``anchor1 -> anchor2``. Drive / limit values are
          displacements [m], ``max_force_drive`` is force [N].
        * :attr:`JointMode.FIXED` -- 6-DoF weld along
          ``anchor1 -> anchor2``; no drive / limit.
        * :attr:`JointMode.CABLE` -- soft fixed joint with PD bend /
          twist springs (3+2+1 row layout). ``bend_{stiffness,damping}``
          [N*m/rad, N*m*s/rad] govern the two axes perp to
          ``anchor1 -> anchor2``, ``twist_{stiffness,damping}`` the
          axis along it. Gains are rescaled by ``1 / rest_length^2``
          to convert rotational SI units to the equivalent positional
          spring at the lever-armed anchors. No drive / limit rows.

        ``stiffness_drive == damping_drive == 0`` disables the drive
        (REVOLUTE / PRISMATIC); ``min_value > max_value`` disables
        the limit. Any positive ``{stiffness,damping}_limit`` selects
        the PD limit formulation over Box2D ``(hertz_limit,
        damping_ratio_limit)``.

        Returns:
            A :class:`JointHandle` whose ``cid`` is rewritten by
            :meth:`finalize` to the joint's global cid.
        """
        self._validate_body(body1)
        self._validate_body(body2)

        mode_enum = JointMode(int(mode))
        drive_mode_enum = DriveMode(int(drive_mode))

        if mode_enum is JointMode.BALL_SOCKET:
            if anchor2 is not None:
                raise ValueError("add_joint(mode=BALL_SOCKET) must not receive an ``anchor2``")
            if drive_mode_enum is not DriveMode.OFF:
                raise ValueError("add_joint(mode=BALL_SOCKET) has no drive row; leave drive_mode=DriveMode.OFF")
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
                    "add_joint(mode=BALL_SOCKET) has no limit row; leave stiffness_limit/damping_limit at 0"
                )
            anchor2_effective: tuple[float, float, float] = tuple(anchor1)  # type: ignore[assignment]
        elif mode_enum is JointMode.FIXED:
            if anchor2 is None:
                raise ValueError(
                    "add_joint(mode=FIXED) requires ``anchor2``; the line "
                    "anchor1 -> anchor2 defines the weld's body-frame axis"
                )
            if drive_mode_enum is not DriveMode.OFF:
                raise ValueError("add_joint(mode=FIXED) has no drive row; leave drive_mode=DriveMode.OFF")
            if target != 0.0 or target_velocity != 0.0 or max_force_drive != 0.0:
                raise ValueError(
                    "add_joint(mode=FIXED) has no drive row; leave target/target_velocity/max_force_drive at defaults"
                )
            if min_value <= max_value:
                raise ValueError(
                    "add_joint(mode=FIXED) has no limit row; leave min_value/max_value at defaults (min > max disables)"
                )
            if stiffness_limit != 0.0 or damping_limit != 0.0:
                raise ValueError("add_joint(mode=FIXED) has no limit row; leave stiffness_limit/damping_limit at 0")
            anchor2_effective = anchor2
        elif mode_enum is JointMode.CABLE:
            if anchor2 is None:
                raise ValueError(
                    "add_joint(mode=CABLE) requires ``anchor2``; the line "
                    "anchor1 -> anchor2 defines the cable's reference "
                    "axis. Bend stiffness acts on the two axes "
                    "perpendicular to it; twist stiffness on the axis "
                    "itself."
                )
            if drive_mode_enum is not DriveMode.OFF:
                raise ValueError(
                    "add_joint(mode=CABLE) has no drive row; leave "
                    "drive_mode=DriveMode.OFF (use bend_stiffness / "
                    "twist_stiffness instead of the drive)"
                )
            if target != 0.0 or target_velocity != 0.0 or max_force_drive != 0.0:
                raise ValueError(
                    "add_joint(mode=CABLE) has no drive row; leave target/target_velocity/max_force_drive at defaults"
                )
            if min_value <= max_value:
                raise ValueError(
                    "add_joint(mode=CABLE) has no limit row; leave min_value/max_value at defaults (min > max disables)"
                )
            if stiffness_drive != 0.0 or damping_drive != 0.0:
                raise ValueError(
                    "add_joint(mode=CABLE) does not use stiffness_drive / "
                    "damping_drive directly -- pass bend_stiffness / "
                    "bend_damping (they share the same column slots)"
                )
            if stiffness_limit != 0.0 or damping_limit != 0.0:
                raise ValueError(
                    "add_joint(mode=CABLE) does not use stiffness_limit / "
                    "damping_limit directly -- pass twist_stiffness / "
                    "twist_damping (they share the same column slots)"
                )
            if bend_stiffness < 0.0 or twist_stiffness < 0.0 or bend_damping < 0.0 or twist_damping < 0.0:
                raise ValueError("add_joint(mode=CABLE) requires non-negative bend / twist stiffness and damping")
            # The kernel's cable prepare reads bend_* / twist_* from the
            # drive / limit slots and rescales by 1 / rest_length^2 to
            # convert N*m/rad -> N/m at the lever-armed anchors.
            stiffness_drive = float(bend_stiffness)
            damping_drive = float(bend_damping)
            stiffness_limit = float(twist_stiffness)
            damping_limit = float(twist_damping)
            anchor2_effective = anchor2
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

        if mode_enum is not JointMode.CABLE and (
            bend_stiffness != 0.0 or twist_stiffness != 0.0 or bend_damping != 0.0 or twist_damping != 0.0
        ):
            raise ValueError(
                f"add_joint(mode={mode_enum.name}): bend/twist stiffness / "
                "damping are only meaningful for JointMode.CABLE"
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
            raise ValueError(f"add_collision_filter_pair: bodies must differ (got both = {body_a})")
        self._collision_filter_pairs.add((min(int(body_a), int(body_b)), max(int(body_a), int(body_b))))

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(
        self,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        gravity: tuple[float, float, float] | Iterable[tuple[float, float, float]] = (0.0, -9.81, 0.0),
        rigid_contact_max: int = 0,
        default_friction: float = 0.5,
        step_layout: str = "multi_world",
        device: wp.context.Devicelike = None,
    ) -> PhoenXWorld:
        """Allocate GPU storage and build a ready-to-step :class:`PhoenXWorld`.

        Descriptor lists are consumed in place (cleared after
        success). Calling :meth:`finalize` twice returns a new empty
        world the second time.

        If shapes were attached via :meth:`add_shape_*`, the builder:

        1. Folds shape-provided density / mass into each body's
           compound inverse mass and body-frame inverse inertia,
           transferring shape local-frame inertias to the body origin
           via the parallel-axis theorem.
        2. Emits ``shape_body`` and (optionally) ``shape_material``
           arrays, stores them on the :class:`PhoenXWorld` so
           :meth:`PhoenXWorld.step` can resolve contact shape ids
           without the caller threading them through manually.
        """
        device = wp.get_device(device)

        # Shape-driven mass / inertia must happen BEFORE the body
        # container is materialised, since the container reads the
        # final ``inverse_mass`` / ``inverse_inertia`` on each
        # descriptor.
        self._accumulate_mass_inertia_from_shapes()

        num_joints = len(self._joint_descriptors)
        bodies = self._build_body_container(device)
        # One constraint column per ``(shape_a, shape_b)`` pair covers
        # an arbitrary contact count per pair, so the column buffer
        # sizes 1:1 against ``rigid_contact_max``.
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            device=device,
        )

        # Detect compound bodies (any rigid body with > 1 shape) so the
        # contact ingest can opt into body-pair grouping. Single-shape
        # scenes pay nothing.
        has_compound_bodies = False
        if self._shapes:
            shape_bodies = np.asarray([int(s.body) for s in self._shapes], dtype=np.int32)
            shape_bodies = shape_bodies[shape_bodies >= 0]
            if shape_bodies.size > 0:
                counts = np.bincount(shape_bodies)
                has_compound_bodies = bool((counts > 1).any())

        world = PhoenXWorld(
            bodies=bodies,
            constraints=constraints,
            substeps=substeps,
            solver_iterations=solver_iterations,
            velocity_iterations=velocity_iterations,
            gravity=gravity,
            rigid_contact_max=rigid_contact_max,
            num_joints=num_joints,
            collision_filter_pairs=self._collision_filter_pairs,
            default_friction=default_friction,
            num_worlds=self._num_worlds,
            step_layout=step_layout,
            enable_body_pair_grouping=has_compound_bodies,
            device=device,
        )

        if num_joints > 0:
            joint_arrays = self._pack_joint_arrays(device)
            world.initialize_actuated_double_ball_socket_joints(**joint_arrays)
            # Patch handle cids (joints occupy cids [0, num_joints)).
            for i, handle in enumerate(self._joint_handles):
                handle.cid = i

        # Shape-driven column state: shape_body always, shape_material
        # only when any shape declared a material_id. Stored on the
        # world so ``world.step()`` doesn't require the caller to
        # re-supply them each frame.
        if self._shapes:
            shape_body_np = np.asarray([int(s.body) for s in self._shapes], dtype=np.int32)
            shape_body_wp = wp.array(shape_body_np, dtype=wp.int32, device=device)
            world.set_shape_body(shape_body_wp)
            if any(s.material_id is not None for s in self._shapes):
                shape_material_np = np.asarray(
                    [int(s.material_id) if s.material_id is not None else 0 for s in self._shapes],
                    dtype=np.int32,
                )
                shape_material_wp = wp.array(shape_material_np, dtype=wp.int32, device=device)
                # Preserve an externally-installed material table;
                # callers who want custom materials should register
                # them via :meth:`PhoenXWorld.set_materials` before
                # stepping, or augment this path with a builder-side
                # material table.
                world.set_materials(world._materials, shape_material_wp)

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
        self._shapes = []
        return world

    def _accumulate_mass_inertia_from_shapes(self) -> None:
        """Fold shape density / mass into compound body mass + body-
        frame inertia. Raises ``ValueError`` if a body has shapes
        with mass *and* the descriptor carries a non-default explicit
        mass or inertia (declare the intent once, on the shape).
        """
        if not self._shapes:
            return
        # Bucket shapes by body.
        per_body: dict[int, list[ShapeDescriptor]] = {}
        for s in self._shapes:
            per_body.setdefault(int(s.body), []).append(s)

        for body_idx, shapes in per_body.items():
            desc = self._bodies[body_idx]
            mass_shapes = [s for s in shapes if (s.density is not None or s.mass is not None)]
            if not mass_shapes:
                # Attached shapes are collision-only; the body's
                # descriptor-set mass / inertia are authoritative.
                continue

            # Mass-bearing shapes only make sense on dynamic bodies.
            if int(desc.motion_type) != int(MOTION_DYNAMIC):
                raise ValueError(
                    f"body {body_idx}: mass-providing shapes are only meaningful "
                    "for DYNAMIC bodies (static / kinematic bodies carry no mass)"
                )
            # Reject ambiguous mass sources: if the user attached
            # shapes with density / mass AND also set non-default
            # ``inverse_mass`` / ``inverse_inertia`` on the body,
            # we can't tell which one they meant. Demand they remove
            # one. ``inverse_mass == 1.0`` / ``inverse_inertia ==
            # identity`` is the ``add_dynamic_body`` default and is
            # treated as "no explicit override".
            if desc.inverse_mass != 1.0 or desc.inverse_inertia != _IDENTITY_INERTIA:
                raise ValueError(
                    f"body {body_idx}: mass is declared both on the body "
                    "(explicit inverse_mass / inverse_inertia on "
                    "add_dynamic_body) and on "
                    f"{len(mass_shapes)} attached shape(s) with density / mass. "
                    "Remove one source: either drop the explicit body-level "
                    "mass, or drop density / mass from the shapes."
                )

            # Per-shape: get (mass, com-in-shape-frame, I-about-shape-COM)
            # from ``compute_inertia_shape``, then transform each shape's
            # inertia into the parent body frame about the *body origin*
            # via :func:`transform_inertia` (rotation + Steiner shift),
            # and sum.
            total_mass = 0.0
            inertia = np.zeros((3, 3), dtype=np.float64)
            for s in mass_shapes:
                m_i, com_local, i_about_com = _shape_mass_com_inertia(s)
                if m_i <= 0.0:
                    raise ValueError(
                        f"body {body_idx}: shape of type {s.shape_type.name} produced "
                        f"zero mass (density={s.density}, mass={s.mass}). "
                        "Check geometry parameters."
                    )
                # Compose shape -> body offset: local_pos + R(local_rot) * com_local.
                q = wp.quat(*(float(c) for c in s.local_rot))
                R = np.asarray(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
                offset = np.asarray(s.local_pos, dtype=np.float64) + R @ com_local
                I_body = np.asarray(
                    transform_inertia(
                        m_i,
                        wp.mat33(*(float(v) for v in i_about_com.flatten())),
                        wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])),
                        q,
                    ),
                    dtype=np.float64,
                ).reshape(3, 3)
                total_mass += m_i
                inertia += I_body
            if total_mass <= 0.0:
                raise ValueError(f"body {body_idx}: compound mass is zero; check shape parameters")

            # Invert into the descriptor's inverse-mass / inverse-
            # inertia fields. We leave the body's position AT the user-
            # provided origin rather than shifting it to the COM -- the
            # solver consumes inverse_inertia in the *body frame*, so
            # any COM offset is baked into the inertia tensor via the
            # parallel-axis theorem above. (PhoenX currently assumes
            # the body origin coincides with its COM; if the shapes'
            # aggregate COM is non-zero, ``inertia`` already accounts
            # for that offset about the body origin.)
            inv_m = 1.0 / total_mass
            try:
                inv_i = np.linalg.inv(inertia)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"body {body_idx}: compound inertia tensor is singular "
                    "(degenerate shape arrangement). "
                    f"inertia matrix:\n{inertia}"
                ) from exc

            self._bodies[body_idx] = dataclass_replace(
                desc,
                inverse_mass=float(inv_m),
                inverse_inertia=(
                    (float(inv_i[0, 0]), float(inv_i[0, 1]), float(inv_i[0, 2])),
                    (float(inv_i[1, 0]), float(inv_i[1, 1]), float(inv_i[1, 2])),
                    (float(inv_i[2, 0]), float(inv_i[2, 1]), float(inv_i[2, 2])),
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_body(self, idx: int) -> None:
        if not (0 <= idx < len(self._bodies)):
            raise IndexError(f"body index {idx} out of range [0, {len(self._bodies)})")

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
        position_prev_substep = np.zeros((n, 3), dtype=np.float32)
        access_mode = np.full(n, int(ACCESS_MODE_VELOCITY_LEVEL), dtype=np.int32)

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
            position_prev_substep[i] = b.position_prev_substep if b.position_prev_substep is not None else b.position
            access_mode[i] = int(b.access_mode)

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
        c.linear_damping = wp.array(linear_damping, dtype=wp.float32, device=device)
        c.angular_damping = wp.array(angular_damping, dtype=wp.float32, device=device)
        c.affected_by_gravity = wp.array(affected_by_gravity, dtype=wp.int32, device=device)
        c.motion_type = wp.array(motion_type, dtype=wp.int32, device=device)
        c.world_id = wp.array(world_id_arr, dtype=wp.int32, device=device)
        # Kinematic pose-scripting scratch. ``position_prev`` must be
        # seeded with the initial pose so the first step's velocity
        # inference sees zero delta for bodies the user hasn't scripted
        # yet; the kinematic_target_* fields are likewise seeded so a
        # constant-velocity kinematic body's first step synthesises a
        # correct target from (initial pose + velocity * dt).
        c.position_prev = wp.array(positions, dtype=wp.vec3f, device=device)
        c.orientation_prev = wp.array(orientations, dtype=wp.quatf, device=device)
        c.kinematic_target_pos = wp.array(positions, dtype=wp.vec3f, device=device)
        c.kinematic_target_orient = wp.array(orientations, dtype=wp.quatf, device=device)
        c.kinematic_target_valid = wp.zeros(n, dtype=wp.int32, device=device)
        # Substep access-mode snapshots. Seeded from the descriptor;
        # the per-substep pre-pass overwrites them on every step.
        # Orientation snapshot mirrors the body's initial orientation
        # so a constraint flipping a body to POSITION_LEVEL before the
        # first substep ran sees a sensible reference frame.
        c.position_prev_substep = wp.array(position_prev_substep, dtype=wp.vec3f, device=device)
        c.orientation_prev_substep = wp.array(orientations, dtype=wp.quatf, device=device)
        c.access_mode = wp.array(access_mode, dtype=wp.int32, device=device)
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

        return {
            "body1": wp.array(body1, dtype=wp.int32, device=device),
            "body2": wp.array(body2, dtype=wp.int32, device=device),
            "anchor1": wp.array(anchor1, dtype=wp.vec3f, device=device),
            "anchor2": wp.array(anchor2, dtype=wp.vec3f, device=device),
            "hertz": wp.array(hertz, dtype=wp.float32, device=device),
            "damping_ratio": wp.array(damping_ratio, dtype=wp.float32, device=device),
            "joint_mode": wp.array(joint_mode, dtype=wp.int32, device=device),
            "drive_mode": wp.array(drive_mode, dtype=wp.int32, device=device),
            "target": wp.array(target, dtype=wp.float32, device=device),
            "target_velocity": wp.array(target_velocity, dtype=wp.float32, device=device),
            "max_force_drive": wp.array(max_force_drive, dtype=wp.float32, device=device),
            "stiffness_drive": wp.array(stiffness_drive, dtype=wp.float32, device=device),
            "damping_drive": wp.array(damping_drive, dtype=wp.float32, device=device),
            "min_value": wp.array(min_value, dtype=wp.float32, device=device),
            "max_value": wp.array(max_value, dtype=wp.float32, device=device),
            "hertz_limit": wp.array(hertz_limit, dtype=wp.float32, device=device),
            "damping_ratio_limit": wp.array(damping_ratio_limit, dtype=wp.float32, device=device),
            "stiffness_limit": wp.array(stiffness_limit, dtype=wp.float32, device=device),
            "damping_limit": wp.array(damping_limit, dtype=wp.float32, device=device),
        }
