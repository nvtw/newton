# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side builder assembling a contact-only :class:`PhoenXWorld`.

Two-phase: append body and shape descriptors, then
:meth:`WorldBuilder.finalize` packs them and returns the ready
:class:`PhoenXWorld`. Bodies ``[0, num_worlds)`` are static anchors; user
bodies start at index ``num_worlds``. Jointed scenes must use
:class:`ModelBuilder` so PhoenX can assemble direct equality systems.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import IntEnum

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
    body_alloc_velocity_storage,
    inertia_sym6,
    inertia_sym6_pack_np,
    reduced_articulation_data_zeros,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

__all__ = [
    "WORLD_BODY",
    "RigidBodyDescriptor",
    "ShapeDescriptor",
    "ShapeType",
    "WorldBuilder",
]


#: Body index of the world 0 static anchor.
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


class ShapeType(IntEnum):
    """Primitive collider types. Shapes carry a local transform + geometry; the
    builder uses them for mass/inertia accumulation and contact ingest."""

    SPHERE = 0
    BOX = 1
    CAPSULE = 2
    PLANE = 3  # static-only half-space, contributes no mass


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
class ShapeDescriptor:
    """One collider. Geometry depends on :attr:`shape_type`:

    * SPHERE: geom_scalar_a = radius [m].
    * BOX: geom_vec3 = (hx, hy, hz) half-extents [m].
    * CAPSULE: geom_scalar_a = radius, geom_scalar_b = half-height [m].
    * PLANE: geom_vec3 = normal (unit), geom_scalar_a = offset.

    Mass: at most one of ``density`` (kg/m^3) or ``mass`` (kg). If neither, the
    shape is collision-only (not folded into the body's inertia)."""

    body: int
    shape_type: ShapeType
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    geom_scalar_a: float = 0.0
    geom_scalar_b: float = 0.0
    geom_vec3: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float | None = None
    mass: float | None = None
    material_id: int | None = None


# ---------------------------------------------------------------------------
# Shape mass / inertia formulas
# ---------------------------------------------------------------------------


def _shape_volume_and_inertia(desc: ShapeDescriptor) -> tuple[float, np.ndarray]:
    """``(volume, body-frame inertia tensor)`` for a unit-density version of the shape."""
    t = desc.shape_type
    if t == ShapeType.SPHERE:
        r = float(desc.geom_scalar_a)
        v = 4.0 / 3.0 * math.pi * r * r * r
        # Solid sphere: I = (2/5) m r^2 on every diagonal.
        diag = 0.4 * v * r * r  # unit-density m = v, so I_i = 0.4 * v * r^2
        return v, np.diag([diag, diag, diag])
    if t == ShapeType.BOX:
        hx, hy, hz = (float(x) for x in desc.geom_vec3)
        v = 8.0 * hx * hy * hz
        # Solid box half-extents (hx, hy, hz): I_xx = m/3 * (hy^2 + hz^2), etc.
        ixx = v / 3.0 * (hy * hy + hz * hz)
        iyy = v / 3.0 * (hx * hx + hz * hz)
        izz = v / 3.0 * (hx * hx + hy * hy)
        return v, np.diag([ixx, iyy, izz])
    if t == ShapeType.CAPSULE:
        r = float(desc.geom_scalar_a)
        h = 2.0 * float(desc.geom_scalar_b)  # full cylindrical length
        # Solid capsule = cylinder(radius=r, length=h) + sphere(radius=r).
        # Closed-form, capsule aligned along local +z:
        v_cyl = math.pi * r * r * h
        v_sph = 4.0 / 3.0 * math.pi * r * r * r
        v = v_cyl + v_sph
        # Cylinder about its COM (aligned +z): Izz = (1/2) m r^2;
        # Ixx = Iyy = (1/12) m (3 r^2 + h^2).
        izz_cyl = 0.5 * v_cyl * r * r
        ixx_cyl = v_cyl / 12.0 * (3.0 * r * r + h * h)
        # Two hemispheres treated as a sphere displaced +/-h/2 (within a few percent).
        izz_sph = 0.4 * v_sph * r * r
        ixx_sph = 0.4 * v_sph * r * r + v_sph * (0.5 * h) * (0.5 * h)
        ixx = ixx_cyl + ixx_sph
        iyy = ixx
        izz = izz_cyl + izz_sph
        return v, np.diag([ixx, iyy, izz])
    if t == ShapeType.PLANE:
        return 0.0, np.zeros((3, 3))
    raise ValueError(f"unknown shape_type: {t}")


def _quat_to_mat33(q: tuple[float, float, float, float]) -> np.ndarray:
    """``[x, y, z, w]`` unit quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(c) for c in q)
    n = x * x + y * y + z * z + w * w
    if n > 0:
        s = 2.0 / n
    else:
        s = 0.0
    xs = x * s
    ys = y * s
    zs = z * s
    return np.asarray(
        [
            [1.0 - (y * ys + z * zs), x * ys - w * zs, x * zs + w * ys],
            [x * ys + w * zs, 1.0 - (x * xs + z * zs), y * zs - w * xs],
            [x * zs - w * ys, y * zs + w * xs, 1.0 - (x * xs + y * ys)],
        ],
        dtype=np.float64,
    )


def _translate_inertia(inertia: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    """Parallel-axis theorem: shift a COM-frame inertia tensor by
    ``offset`` (from shape COM to parent-body origin).

    :math:`I' = I + m (|r|^2 \\mathbf{1} - r r^T)`
    """
    r2 = float(np.dot(offset, offset))
    outer = np.outer(offset, offset)
    return inertia + mass * (r2 * np.eye(3) - outer)


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
    """Append bodies and shapes, then materialise a :class:`PhoenXWorld`.

    Usage::

        b = WorldBuilder()
        body = b.add_dynamic_body(position=(0, 0, 1), inverse_mass=1.0)
        b.add_shape_sphere(body, radius=0.5)
        world = b.finalize(substeps=5, solver_iterations=2)

    This low-level builder is contact-only. Use :class:`ModelBuilder` for
    jointed scenes so PhoenX can detect mechanisms and solve their equalities
    directly. Single-use; call :meth:`finalize` exactly once.
    """

    def __init__(self, num_worlds: int = 1):
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1 (got {num_worlds})")
        self._num_worlds: int = int(num_worlds)
        # One static anchor per world.
        self._bodies: list[RigidBodyDescriptor] = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                world_id=w,
            )
            for w in range(self._num_worlds)
        ]
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

    # Shape API. Optionally set density/mass on shapes for finalize() to
    # auto-compute the body's compound inertia (parallel-axis). Mixing this
    # with explicit body inverse_mass/inertia raises in finalize().

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
        # Plane is static-only and cannot carry mass (infinite volume).
        if desc.shape_type == ShapeType.PLANE and (desc.density is not None or desc.mass is not None):
            raise ValueError(
                f"add_shape_plane(body={desc.body}): planes are infinite "
                "half-spaces and cannot contribute mass; omit density / mass"
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
        material_id: int | None = None,
    ) -> int:
        """Attach a solid-sphere collider. Set ``density`` (kg/m^3) or ``mass``
        (kg) for auto-computed inertia (at most one)."""
        if radius <= 0.0 or not _is_finite(radius):
            raise ValueError(f"add_shape_sphere(body={body}): radius must be > 0 (got {radius})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=ShapeType.SPHERE,
                local_pos=local_pos,
                local_rot=local_rot,
                geom_scalar_a=float(radius),
                density=density,
                mass=mass,
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
        material_id: int | None = None,
    ) -> int:
        """Attach a solid-box collider. ``half_extents`` (hx, hy, hz) all > 0."""
        if any((not _is_finite(h)) or h <= 0.0 for h in half_extents):
            raise ValueError(f"add_shape_box(body={body}): half_extents must all be > 0 (got {half_extents})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=ShapeType.BOX,
                local_pos=local_pos,
                local_rot=local_rot,
                geom_vec3=tuple(float(h) for h in half_extents),
                density=density,
                mass=mass,
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
        material_id: int | None = None,
    ) -> int:
        """Attach a capsule collider (cylinder + two hemispheres) along local +z.
        ``half_height = 0`` collapses to a sphere."""
        if radius <= 0.0 or not _is_finite(radius):
            raise ValueError(f"add_shape_capsule(body={body}): radius must be > 0 (got {radius})")
        if half_height < 0.0 or not _is_finite(half_height):
            raise ValueError(f"add_shape_capsule(body={body}): half_height must be >= 0 (got {half_height})")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=ShapeType.CAPSULE,
                local_pos=local_pos,
                local_rot=local_rot,
                geom_scalar_a=float(radius),
                geom_scalar_b=float(half_height),
                density=density,
                mass=mass,
                material_id=material_id,
            )
        )

    def add_shape_plane(
        self,
        body: int,
        *,
        normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        offset: float = 0.0,
        material_id: int | None = None,
    ) -> int:
        """Attach an infinite plane (static body only, no mass)."""
        nlen = math.sqrt(sum(float(c) * float(c) for c in normal))
        if nlen <= 1e-12:
            raise ValueError(f"add_shape_plane(body={body}): normal must be non-zero")
        unit = tuple(float(c) / nlen for c in normal)
        if int(self._bodies[body].motion_type) != int(MOTION_STATIC):
            raise ValueError(f"add_shape_plane(body={body}): planes may only be attached to static bodies")
        return self._attach_shape(
            ShapeDescriptor(
                body=body,
                shape_type=ShapeType.PLANE,
                geom_vec3=unit,
                geom_scalar_a=float(offset),
                material_id=material_id,
            )
        )

    def add_collision_filter_pair(self, body_a: int, body_b: int) -> None:
        """Ignore contacts between ``body_a`` and ``body_b`` (canonical (min, max))."""
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
        mass_splitting: bool = False,
        max_colored_partitions: int = 12,
        mass_splitting_batch_size: int = 8,
        mass_splitting_unrolled: bool = False,
        enable_column_timers: bool = False,
        device: wp.context.Devicelike = None,
    ) -> PhoenXWorld:
        """Allocate GPU storage and build a ready-to-step :class:`PhoenXWorld`.

        Descriptor lists are consumed (cleared on success). Shapes attached via
        :meth:`add_shape_*` fold density/mass into per-body compound inertia
        (parallel-axis); shape_body / shape_material are stored on the world.

        ``mass_splitting`` (and the related ``max_colored_partitions`` /
        ``mass_splitting_batch_size`` / ``mass_splitting_unrolled`` knobs)
        are forwarded straight to :class:`PhoenXWorld`; see its docstring
        for the full description. Requires ``step_layout="single_world"``.
        ``enable_column_timers`` is the opt-in
        ``%globaltimer``-based per-column wall-clock profiler.
        """
        device = wp.get_device(device)
        # Shape mass/inertia must run before _build_body_container reads it.
        self._accumulate_mass_inertia_from_shapes()

        bodies = self._build_body_container(device)
        constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=device,
        )

        # Auto grouping for single-scene compound contact graphs.
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
            num_joints=0,
            collision_filter_pairs=self._collision_filter_pairs,
            default_friction=default_friction,
            num_worlds=self._num_worlds,
            step_layout=step_layout,
            enable_body_pair_grouping=has_compound_bodies and (step_layout == "single_world" or self._num_worlds == 1),
            mass_splitting=mass_splitting,
            max_colored_partitions=max_colored_partitions,
            mass_splitting_batch_size=mass_splitting_batch_size,
            mass_splitting_unrolled=mass_splitting_unrolled,
            enable_column_timers=enable_column_timers,
            device=device,
        )

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
                world.set_materials(world._materials, shape_material_wp)

        # Reset state so a second finalize starts clean.
        self._bodies = [
            RigidBodyDescriptor(
                inverse_mass=0.0,
                inverse_inertia=_ZERO_INERTIA,
                affected_by_gravity=False,
                world_id=w,
            )
            for w in range(self._num_worlds)
        ]
        self._collision_filter_pairs = set()
        self._shapes = []
        return world

    def _accumulate_mass_inertia_from_shapes(self) -> None:
        """Fold shape density/mass into per-body compound mass + body-frame inertia.
        Raises if a body declares mass twice (explicit body fields + shape mass)."""
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
            # Reject body-level mass override + shape-derived mass.
            # inverse_mass=1.0 / inverse_inertia=identity is the default = no override.
            if desc.inverse_mass != 1.0 or desc.inverse_inertia != _IDENTITY_INERTIA:
                raise ValueError(
                    f"body {body_idx}: mass is declared both on the body "
                    "(explicit inverse_mass / inverse_inertia on "
                    "add_dynamic_body) and on "
                    f"{len(mass_shapes)} attached shape(s) with density / mass. "
                    "Remove one source: either drop the explicit body-level "
                    "mass, or drop density / mass from the shapes."
                )

            # Parallel-axis accumulation:
            #   total_mass = Σ m_i
            #   com = (Σ m_i r_i) / total_mass
            #   I_body = Σ (R_i I_i^shape R_i^T + m_i (|r_i - com|^2 I - (r_i - com)(r_i - com)^T))
            total_mass = 0.0
            com = np.zeros(3, dtype=np.float64)
            for s in mass_shapes:
                v, _ = _shape_volume_and_inertia(s)
                m_i = float(s.mass) if s.mass is not None else float(s.density) * v
                if m_i <= 0.0:
                    raise ValueError(
                        f"body {body_idx}: shape of type {s.shape_type.name} produced "
                        f"zero mass (density={s.density}, mass={s.mass}, volume={v}). "
                        "Check geometry parameters."
                    )
                total_mass += m_i
                com += m_i * np.asarray(s.local_pos, dtype=np.float64)
            if total_mass <= 0.0:
                raise ValueError(f"body {body_idx}: compound mass is zero; check shape parameters")
            com /= total_mass

            inertia = np.zeros((3, 3), dtype=np.float64)
            for s in mass_shapes:
                v, i_local = _shape_volume_and_inertia(s)
                if s.mass is not None:
                    scale = float(s.mass) / v if v > 0.0 else 0.0
                else:
                    scale = float(s.density)
                i_shape = scale * i_local
                r_local = _quat_to_mat33(s.local_rot)
                i_shape_body = r_local @ i_shape @ r_local.T
                offset = np.asarray(s.local_pos, dtype=np.float64) - com
                m_i = float(s.mass) if s.mass is not None else float(s.density) * v
                inertia += _translate_inertia(i_shape_body, m_i, offset)

            # The solver consumes inverse_inertia in the body frame; any COM
            # offset is already baked in via parallel-axis above.
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

    def _validate_body(self, idx: int) -> None:
        if not (0 <= idx < len(self._bodies)):
            raise IndexError(f"body index {idx} out of range [0, {len(self._bodies)})")

    def _build_body_container(self, device: wp.context.Device) -> BodyContainer:
        """Pack descriptors into a :class:`BodyContainer`."""
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

        # First _update_inertia launch rotates inverse_inertia_world into world space.
        c = BodyContainer()
        c.position = wp.array(positions, dtype=wp.vec3f, device=device)
        c.velocity, c.angular_velocity = body_alloc_velocity_storage(n, device, velocities, angular_velocities)
        c.orientation = wp.array(orientations, dtype=wp.quatf, device=device)
        # Builder bodies assume mesh origin == COM; meshed offsets must set body_com directly.
        c.body_com = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.inverse_inertia_world = wp.array(inertia_sym6_pack_np(inverse_inertia), dtype=inertia_sym6, device=device)
        c.inverse_inertia = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
        c.inverse_mass = wp.array(inverse_mass, dtype=wp.float32, device=device)
        c.force = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.torque = wp.zeros(n, dtype=wp.vec3f, device=device)
        c.linear_damping = wp.array(linear_damping, dtype=wp.float32, device=device)
        c.angular_damping = wp.array(angular_damping, dtype=wp.float32, device=device)
        c.affected_by_gravity = wp.array(affected_by_gravity, dtype=wp.int32, device=device)
        c.motion_type = wp.array(motion_type, dtype=wp.int32, device=device)
        c.world_id = wp.array(world_id_arr, dtype=wp.int32, device=device)
        c.constraint_node = wp.array(np.arange(n, dtype=np.int32), device=device)
        c.reduced = reduced_articulation_data_zeros(device)
        # Seed prev=target=initial pose so the first step infers zero delta
        # for un-scripted kinematic bodies.
        c.position_prev = wp.array(positions, dtype=wp.vec3f, device=device)
        c.orientation_prev = wp.array(orientations, dtype=wp.quatf, device=device)
        c.kinematic_target_pos = wp.array(positions, dtype=wp.vec3f, device=device)
        c.kinematic_target_orient = wp.array(orientations, dtype=wp.quatf, device=device)
        c.kinematic_target_valid = wp.zeros(n, dtype=wp.int32, device=device)
        # Substep-entry pose snapshot + access-mode tag (see
        # :mod:`newton._src.solvers.phoenx.access_mode`). Seed
        # snapshots to the initial pose so the first synchronize call
        # finite-diffs against a sensible anchor; mode defaults to
        # VELOCITY_LEVEL (the apply-forces kernel re-stamps every
        # substep).
        c.position_prev_substep = wp.array(positions, dtype=wp.vec3f, device=device)
        c.orientation_prev_substep = wp.array(orientations, dtype=wp.quatf, device=device)
        c.access_mode = wp.full(n, value=int(ACCESS_MODE_VELOCITY_LEVEL), dtype=wp.int32, device=device)
        c.has_position_level_writers = wp.zeros(1, dtype=wp.int32, device=device)
        c.island_root = wp.full(n, value=-1, dtype=wp.int32, device=device)
        c.frames_below_threshold = wp.zeros(n, dtype=wp.int32, device=device)
        return c
