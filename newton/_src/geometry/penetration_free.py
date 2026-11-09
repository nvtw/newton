# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration-free rigid-rigid motion certification.

The collision pipeline turns every rigid shape pair that could meet during the
next update into *half-space constraints*. Each constraint states that the
support of one collision feature along a fixed world direction ``n`` must stay
at or below a limit ``L``:

.. math::

    h(t) = n \\cdot c(t) + h_F(R(t)^T n) + m \\le L,

where ``c`` and ``R`` are the owning body's center-of-mass position and
orientation, ``h_F`` is the feature support function in the body frame and
``m`` is the shape margin. The two constraints of a pair use opposite
directions and limits whose difference is at most the configured tolerance, so
the pair can never overlap by more than that tolerance along ``n``. The minimum
overlap over all directions is the penetration depth, so the same bound holds
for the penetration depth.

Pairs without constraints are covered by per-body *motion budgets*. Every body
receives a budget ``beta`` and the broad phase finds all pairs whose distance
is at most ``beta_a + beta_b``. As long as no point of a body travels more than
``beta`` from the detection reference, pairs that the broad phase rejected
cannot meet.

A solver certifies a candidate motion from ``(c0, q0)`` to ``(c1, q1)`` along
the screw path that translates the center of mass linearly and rotates about a
fixed axis through it. For a feature within radius ``rho`` of the center of
mass, the support function is ``rho``-Lipschitz in the direction, so ``h`` changes
at most at rate ``|n . dc| + rho |theta| |a x n|`` along the path. Exact samples
of ``h`` combined with that rate bound certify the whole path (see
:func:`halfspace_motion_fraction`). Solvers commit only the admissible fraction
of each motion, which also keeps the accumulated travel within the budget.

This module holds solver-independent pieces: the configuration, the buffers
attached to :class:`~newton.Contacts`, the generation kernels launched by
:class:`~newton.CollisionPipeline`, and the Warp functions solvers use to limit
and commit motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import get_triangle_shape_from_mesh
from .flags import ShapeFlags
from .mpr import create_solve_mpr
from .simplex_solver import create_solve_closest_distance
from .support_function import (
    TRIANGLE_PRISM_EXTRUSION,
    GenericShapeData,
    SupportMapDataProvider,
    extract_shape_data,
    support_map,
)
from .types import GeoType

__all__ = [
    "ConstraintFeature",
    "Constraints",
    "PenetrationFreeConfig",
    "PenetrationFreeData",
]

# Status bits stored in ``PenetrationFreeData.status``.
STATUS_CONSTRAINT_OVERFLOW = wp.constant(1 << 0)
STATUS_WORK_OVERFLOW = wp.constant(1 << 1)
STATUS_BROAD_PHASE_OVERFLOW = wp.constant(1 << 2)
STATUS_UNSUPPORTED_GEOMETRY = wp.constant(1 << 3)
STATUS_REFERENCE_MISMATCH = wp.constant(1 << 4)
STATUS_KINEMATIC_VIOLATION = wp.constant(1 << 5)
STATUS_BUDGET_EXHAUSTED = wp.constant(1 << 6)

STATUS_FATAL_MASK = wp.constant((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4))
"""Status bits after which no constraint set is complete; solvers must stop all motion."""

_STATUS_NAMES = {
    1 << 0: "constraint buffer overflow",
    1 << 1: "triangle work buffer overflow",
    1 << 2: "broad-phase pair buffer overflow",
    1 << 3: "unsupported geometry",
    1 << 4: "state does not match the detection reference",
    1 << 5: "kinematic body violated a constraint",
    1 << 6: "motion budget exhausted",
}

# Relative slack absorbing float32 roundoff in support evaluations.
_SUPPORT_RELATIVE_SLACK = wp.constant(4.0e-6)
_SUPPORT_ABSOLUTE_SLACK = wp.constant(1.0e-7)
_SMALL_ANGLE = wp.constant(1.0e-9)

MOTION_SAMPLES = wp.constant(8)
"""Number of intervals over which a substep motion's support height is certified."""

_SUPPORTED_GEO_TYPES = frozenset(
    {
        int(GeoType.PLANE),
        int(GeoType.HFIELD),
        int(GeoType.SPHERE),
        int(GeoType.CAPSULE),
        int(GeoType.ELLIPSOID),
        int(GeoType.CYLINDER),
        int(GeoType.BOX),
        int(GeoType.MESH),
        int(GeoType.CONE),
        int(GeoType.CONVEX_MESH),
        int(GeoType.TRIANGLE),
        int(GeoType.TETRAHEDRON),
    }
)


@dataclass
class PenetrationFreeConfig:
    """Configuration for penetration-free rigid-rigid collision handling.

    Passing an instance to :class:`~newton.CollisionPipeline` makes the
    pipeline produce certified half-space constraints and motion budgets in
    addition to the regular contacts. Solvers that support the mode limit every
    rigid-body position update so that no pair of colliding rigid shapes
    penetrates deeper than :attr:`tolerance`, measured along the certified
    separating direction of the pair. Solvers without support reject the
    resulting :class:`~newton.Contacts` buffer.

    The guarantee assumes the configuration at detection time is valid: pairs
    that already overlap by more than :attr:`tolerance` are only prevented from
    overlapping further. Motion that exceeds a constraint or a budget is
    truncated, which can remove kinetic energy and stall fast bodies until the
    next collision detection.

    .. experimental::

        The penetration-free mode is experimental and its configuration may
        change without prior notice.
    """

    tolerance: float = 1.0e-3
    """Maximum penetration depth [m] permitted between two shapes."""

    budget_velocity_scale: float = 2.0
    """Multiplier on the predicted travel ``(|v| + |omega| r) dt`` that sets each body's motion budget [unitless]."""

    min_motion_budget: float = 0.01
    """Lower bound on the motion budget of every body [m]."""

    max_motion_budget: float = 1.0
    """Upper bound on the motion budget of every body [m]."""

    max_constraints: int | None = None
    """Capacity of the half-space constraint buffer, or ``None`` to size it from the model."""

    max_triangle_work: int | None = None
    """Capacity of the mesh and heightfield triangle work buffer, or ``None`` to size it from the model."""

    def validate(self) -> None:
        """Raise :class:`ValueError` if any setting is out of range."""
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError(f"tolerance must be a non-negative finite number, got {self.tolerance!r}")
        if not np.isfinite(self.budget_velocity_scale) or self.budget_velocity_scale < 1.0:
            raise ValueError(f"budget_velocity_scale must be at least 1, got {self.budget_velocity_scale!r}")
        if not np.isfinite(self.min_motion_budget) or self.min_motion_budget <= 0.0:
            raise ValueError(f"min_motion_budget must be positive, got {self.min_motion_budget!r}")
        if not np.isfinite(self.max_motion_budget) or self.max_motion_budget < self.min_motion_budget:
            raise ValueError("max_motion_budget must be finite and at least min_motion_budget")
        for name in ("max_constraints", "max_triangle_work"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value!r}")


@wp.struct
class ConstraintFeature:
    """Collision feature of a half-space constraint in its body's center-of-mass frame."""

    local_pos: wp.vec3
    local_rot: wp.quat
    shape_type: int
    scale: wp.vec3
    aux: wp.vec3
    margin: float
    lipschitz: float
    """Radius [m] about the center of mass bounding the feature minus its spherical rounding."""


@wp.struct
class Constraints:
    """Half-space constraints ``support(feature, normal) <= limit`` of one detection."""

    count: wp.array[wp.int32]
    status: wp.array[wp.int32]
    recovery_count: wp.array[wp.int32]
    body: wp.array[wp.int32]
    normal: wp.array[wp.vec3]
    limit: wp.array[wp.float32]
    feature: wp.array[ConstraintFeature]


class PenetrationFreeData:
    """Device buffers describing the certified motion limits of one detection.

    Allocated by :meth:`~newton.CollisionPipeline.contacts` when the pipeline
    has a :class:`PenetrationFreeConfig`, and refreshed by every
    :meth:`~newton.CollisionPipeline.collide`. Constraint geometry is stored in
    each body's center-of-mass frame so solvers evaluate it from their native
    center-of-mass pose without access to model shape arrays.
    """

    def __init__(self, config: PenetrationFreeConfig, body_count: int, max_constraints: int, max_work: int, device):
        self.config = config
        self.body_count = int(body_count)
        self.max_constraints = int(max_constraints)
        self.max_work = int(max_work)
        self.device = device
        nb = max(self.body_count, 1)
        nw = max(self.max_work, 1)
        with wp.ScopedDevice(device):
            c = Constraints()
            c.count = wp.zeros(1, dtype=wp.int32)
            c.status = wp.zeros(1, dtype=wp.int32)
            c.recovery_count = wp.zeros(1, dtype=wp.int32)
            nc = max(self.max_constraints, 1)
            c.body = wp.zeros(nc, dtype=wp.int32)
            c.normal = wp.zeros(nc, dtype=wp.vec3)
            c.limit = wp.zeros(nc, dtype=wp.float32)
            c.feature = wp.zeros(nc, dtype=ConstraintFeature)
            self.constraints = c
            # Triangle work lists: (pair, source triangle, source is shape b, other triangle).
            self.work_count = wp.zeros(2, dtype=wp.int32)
            self.work_items = wp.zeros(nw, dtype=wp.vec4i)
            self.work_pairs = wp.zeros(nw, dtype=wp.vec4i)

            self.body_tracked_q = wp.zeros(nb, dtype=wp.transform)
            """Body origin poses last committed by the consuming solver; detection resets them."""
            self.body_budget = wp.zeros(nb, dtype=wp.float32)
            """Maximum travel [m] of any point of the body since detection."""
            self.body_radius = wp.zeros(nb, dtype=wp.float32)
            """Radius [m] about the center of mass enclosing every collision shape of the body."""
            self.body_used = wp.zeros(nb, dtype=wp.float32)
            """Accumulated travel bound [m] committed since detection."""

    @property
    def status(self) -> wp.array:
        """Bit mask of ``STATUS_*`` flags raised since the last detection."""
        return self.constraints.status

    @property
    def constraint_count(self) -> wp.array:
        """Number of emitted half-space constraints, possibly above capacity on overflow."""
        return self.constraints.count

    @property
    def recovery_count(self) -> wp.array:
        """Number of feature pairs that already overlapped by more than the tolerance."""
        return self.constraints.recovery_count

    def reset(self) -> None:
        """Clear the counters before a detection."""
        self.constraints.count.zero_()
        self.constraints.status.zero_()
        self.constraints.recovery_count.zero_()
        self.work_count.zero_()

    def status_messages(self) -> list[str]:
        """Return readable descriptions of the raised status bits. Synchronizes with the device."""
        status = int(self.status.numpy()[0])
        return [name for bit, name in _STATUS_NAMES.items() if status & bit]


def penetration_free_capacities(model, config: PenetrationFreeConfig) -> tuple[int, int]:
    """Return default ``(max_constraints, max_triangle_work)`` capacities for ``model``."""
    default = max(4096, 128 * int(model.shape_count))
    return int(config.max_constraints or default), int(config.max_triangle_work or default)


def validate_penetration_free_model(model) -> None:
    """Raise :class:`ValueError` if ``model`` contains collision geometry the mode cannot certify."""
    if model.shape_count == 0:
        return
    shape_type = model.shape_type.numpy()
    shape_body = model.shape_body.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_flags = model.shape_flags.numpy()
    for s in range(int(model.shape_count)):
        if not (int(shape_flags[s]) & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        t = int(shape_type[s])
        if t not in _SUPPORTED_GEO_TYPES:
            raise ValueError(f"penetration-free collisions do not support shape {s} of type {GeoType(t).name}")
        infinite_plane = t == int(GeoType.PLANE) and shape_scale[s][0] == 0.0 and shape_scale[s][1] == 0.0
        if infinite_plane and int(shape_body[s]) >= 0:
            raise ValueError(f"penetration-free collisions require infinite plane shape {s} to be static")


# ---------------------------------------------------------------------------
# Motion certification shared by solvers
# ---------------------------------------------------------------------------


@wp.func
def screw_motion_delta(x0: wp.vec3, q0: wp.quat, x1: wp.vec3, q1: wp.quat):
    """Return ``(dx, axis, angle)`` of the screw path from ``(x0, q0)`` to ``(x1, q1)``.

    The path translates the center of mass linearly and rotates by the
    shortest arc about a fixed world axis through it, so ``angle`` lies in
    ``[0, pi]``.
    """
    dq = q1 * wp.quat_inverse(q0)
    if dq[3] < 0.0:
        dq = -dq
    xyz = wp.vec3(dq[0], dq[1], dq[2])
    s = wp.length(xyz)
    axis = wp.vec3(1.0, 0.0, 0.0)
    angle = float(0.0)
    if s > _SMALL_ANGLE:
        axis = xyz / s
        angle = 2.0 * wp.atan2(s, wp.clamp(dq[3], -1.0, 1.0))
    return x1 - x0, axis, angle


@wp.func
def screw_motion_interpolate(x0: wp.vec3, q0: wp.quat, dx: wp.vec3, axis: wp.vec3, angle: float, alpha: float):
    """Return the pose at fraction ``alpha`` of a screw path from :func:`screw_motion_delta`."""
    q = wp.normalize(wp.quat_from_axis_angle(axis, alpha * angle) * q0)
    return x0 + alpha * dx, q


@wp.func
def screw_motion_travel(dx: wp.vec3, angle: float, radius: float) -> float:
    """Return an upper bound [m] on the travel of any point within ``radius`` of the center of mass."""
    return wp.length(dx) + radius * angle


@wp.func
def motion_budget_fraction(used: float, budget: float, travel: float) -> float:
    """Return the largest fraction of a motion with bound ``travel`` that keeps ``used`` within ``budget``."""
    if travel <= 0.0:
        return 1.0
    return wp.clamp((budget - used) / travel, 0.0, 1.0)


@wp.func
def _support(geom: GenericShapeData, rot: wp.quat, direction: wp.vec3) -> float:
    """Return ``max_x dot(direction, x)`` over a shape rotated by ``rot`` about its origin."""
    d_local = wp.quat_rotate_inv(rot, direction)
    return wp.dot(support_map(geom, d_local, SupportMapDataProvider()), d_local)


@wp.func
def _feature_geom(f: ConstraintFeature) -> GenericShapeData:
    geom = GenericShapeData()
    geom.shape_type = f.shape_type
    geom.scale = f.scale
    geom.auxiliary = f.aux
    return geom


@wp.func
def feature_support(f: ConstraintFeature, direction_body: wp.vec3) -> float:
    """Return the feature's support along a body-frame direction, relative to the center of mass.

    Includes the margin and a conservative float32 roundoff slack.
    """
    value = wp.dot(direction_body, f.local_pos) + _support(_feature_geom(f), f.local_rot, direction_body) + f.margin
    magnitude = wp.length(f.local_pos) + f.lipschitz + wp.length(f.scale) + wp.abs(f.margin)
    return value + _SUPPORT_RELATIVE_SLACK * magnitude + _SUPPORT_ABSOLUTE_SLACK


@wp.func
def _feature_height(
    f: ConstraintFeature, normal: wp.vec3, x0: wp.vec3, q0: wp.quat, dx: wp.vec3, axis: wp.vec3, angle: float, t: float
) -> float:
    """Return the feature's world support along ``normal`` at fraction ``t`` of a screw motion."""
    x, q = screw_motion_interpolate(x0, q0, dx, axis, angle, t)
    return wp.dot(normal, x) + feature_support(f, wp.quat_rotate_inv(q, normal))


@wp.func
def halfspace_motion_fraction(
    f: ConstraintFeature,
    normal: wp.vec3,
    limit: float,
    x0: wp.vec3,
    q0: wp.quat,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
) -> float:
    """Return the admissible fraction of a screw motion for one half-space constraint.

    The support height ``h(t)`` is evaluated exactly at
    :data:`MOTION_SAMPLES` + 1 equally spaced fractions. Its rate of change is
    bounded by ``S = |n . dx| + rho |theta| |a x n|``, so between two samples
    ``h(t) <= (h_i + h_{i+1}) / 2 + S delta / 2``. The returned fraction keeps
    this certified bound at or below ``max(limit, h(0))``: a feature that
    starts above its limit may move but never rise further.
    """
    rate = wp.abs(wp.dot(normal, dx)) + f.lipschitz * angle * wp.length(wp.cross(axis, normal))
    if rate <= 0.0:
        return 1.0
    h_prev = _feature_height(f, normal, x0, q0, dx, axis, angle, 0.0)
    bound = wp.max(limit, h_prev)
    delta = 1.0 / float(MOTION_SAMPLES)
    for i in range(MOTION_SAMPLES):
        t0 = float(i) * delta
        h_next = _feature_height(f, normal, x0, q0, dx, axis, angle, t0 + delta)
        if 0.5 * (h_prev + h_next) + 0.5 * rate * delta > bound:
            # Admissible prefix of this interval from the Lipschitz bound at its start.
            return wp.clamp(t0 + wp.max(bound - h_prev, 0.0) / rate, t0, t0 + delta)
        h_prev = h_next
    return 1.0


@wp.kernel(enable_backward=False)
def limit_motion_kernel(
    constraints: Constraints,
    position_start: wp.array[wp.vec3],
    orientation_start: wp.array[wp.quat],
    position_end: wp.array[wp.vec3],
    orientation_end: wp.array[wp.quat],
    slot_offset: int,
    # output
    fraction: wp.array[wp.float32],
):
    """Lower each body's admissible fraction by every half-space constraint it owns.

    Body poses are center-of-mass poses in the solver's slot layout, where
    Newton body ``b`` lives in slot ``b + slot_offset``. ``fraction`` must be
    initialized to ``1`` before the launch.
    """
    c = wp.tid()
    if c >= wp.min(constraints.count[0], constraints.body.shape[0]):
        return
    slot = constraints.body[c] + slot_offset
    x0 = position_start[slot]
    q0 = orientation_start[slot]
    dx, axis, angle = screw_motion_delta(x0, q0, position_end[slot], orientation_end[slot])
    t = halfspace_motion_fraction(
        constraints.feature[c], constraints.normal[c], constraints.limit[c], x0, q0, dx, axis, angle
    )
    if t < 1.0:
        wp.atomic_min(fraction, slot, t)


def launch_limit_motion(
    data: PenetrationFreeData,
    position_start: wp.array,
    orientation_start: wp.array,
    position_end: wp.array,
    orientation_end: wp.array,
    slot_offset: int,
    fraction: wp.array,
) -> None:
    """Lower ``fraction`` per body slot so the motion start -> end respects every constraint.

    ``fraction`` must be initialized to ``1`` by the caller, which applies
    budgets and commits the motion in its native representation.
    """
    wp.launch(
        limit_motion_kernel,
        dim=data.constraints.body.shape[0],
        inputs=[data.constraints, position_start, orientation_start, position_end, orientation_end, int(slot_offset)],
        outputs=[fraction],
        device=data.device,
    )


# ---------------------------------------------------------------------------
# Generation (collision pipeline)
# ---------------------------------------------------------------------------

_solve_gjk = create_solve_closest_distance(support_map)
_solve_mpr = create_solve_mpr(support_map)


@wp.struct
class GenerationScene:
    """Read-only inputs of the constraint generation kernels."""

    pairs: wp.array[wp.vec2i]
    pair_count: wp.array[wp.int32]
    shape_count: int
    tolerance: float
    geom_transform: wp.array[wp.transform]
    geom_data: wp.array[wp.vec4]
    shape_type: wp.array[int]
    shape_source_ptr: wp.array[wp.uint64]
    shape_body: wp.array[int]
    shape_aabb_lower_local: wp.array[wp.vec3]
    shape_aabb_upper_local: wp.array[wp.vec3]
    aabb_lower: wp.array[wp.vec3]
    aabb_upper: wp.array[wp.vec3]
    heightfield_index: wp.array[wp.int32]
    heightfield_data: wp.array[HeightfieldData]
    heightfield_elevations: wp.array[wp.float32]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    body_budget: wp.array[wp.float32]


@wp.struct
class _Feature:
    """A convex feature in world space plus its body-frame constraint descriptor."""

    geom: GenericShapeData
    pos: wp.vec3
    rot: wp.quat
    body: int
    budget: float
    infinite_plane: int
    local: ConstraintFeature


@wp.func
def _shape_core_radius(geom: GenericShapeData, local_aabb_lower: wp.vec3, local_aabb_upper: wp.vec3) -> float:
    """Return the radius about the shape origin of the shape minus its spherical rounding."""
    t = geom.shape_type
    s = geom.scale
    if t == GeoType.SPHERE:
        return 0.0
    if t == GeoType.CAPSULE:
        return s[1]
    if t == GeoType.ELLIPSOID:
        return wp.max(s[0], wp.max(s[1], s[2]))
    if t == GeoType.CYLINDER:
        radius = s[0]
        if s[2] >= s[1] and s[2] > 0.0:
            radius += (s[1] * s[1]) / (s[2] + wp.sqrt(s[2] * s[2] - s[1] * s[1]))
        return wp.sqrt(radius * radius + s[1] * s[1])
    if t == GeoType.CONE or t == GeoType.PLANE:
        return wp.sqrt(s[0] * s[0] + s[1] * s[1])
    if t == GeoType.BOX:
        return wp.length(s)
    if t == GeoType.TRIANGLE or t == GeoType.TETRAHEDRON:
        return wp.max(wp.max(s[0], wp.sqrt(s[1] * s[1] + s[2] * s[2])), wp.length(geom.auxiliary))
    return wp.length(wp.max(wp.abs(local_aabb_lower), wp.abs(local_aabb_upper)))


@wp.func
def _make_feature(
    scene: GenerationScene, shape: int, geom: GenericShapeData, pos: wp.vec3, rot: wp.quat, core_radius: float
) -> _Feature:
    """Attach the owning body's frame, budget and margin to a world-space feature."""
    f = _Feature()
    f.geom = geom
    f.pos = pos
    f.rot = rot
    b = scene.shape_body[shape]
    f.body = b
    f.budget = 0.0
    com = wp.vec3(0.0)
    rb = wp.quat_identity()
    if b >= 0:
        f.budget = scene.body_budget[b]
        X = scene.body_q[b]
        com = wp.transform_point(X, scene.body_com[b])
        rb = wp.transform_get_rotation(X)
    f.infinite_plane = 0
    if geom.shape_type == GeoType.PLANE and geom.scale[0] == 0.0 and geom.scale[1] == 0.0:
        f.infinite_plane = 1
    local = ConstraintFeature()
    local.local_pos = wp.quat_rotate_inv(rb, pos - com)
    local.local_rot = wp.quat_inverse(rb) * rot
    local.shape_type = geom.shape_type
    local.scale = geom.scale
    local.aux = geom.auxiliary
    local.margin = scene.geom_data[shape][3]
    local.lipschitz = wp.length(local.local_pos) + core_radius
    f.local = local
    return f


@wp.func
def _shape_feature(scene: GenerationScene, shape: int) -> _Feature:
    pos, rot, geom, _scale, _margin = extract_shape_data(
        shape, scene.geom_transform, scene.shape_type, scene.geom_data, scene.shape_source_ptr
    )
    core = _shape_core_radius(geom, scene.shape_aabb_lower_local[shape], scene.shape_aabb_upper_local[shape])
    return _make_feature(scene, shape, geom, pos, rot, core)


@wp.func
def _triangle_feature(scene: GenerationScene, shape: int, tri: int) -> _Feature:
    """Return a mesh triangle, or a heightfield cell's solid triangle prism, as a convex feature."""
    X_ws = scene.geom_transform[shape]
    geom = GenericShapeData()
    v0 = wp.vec3(0.0)
    rot = wp.quat_identity()
    extrusion = float(0.0)
    if scene.shape_type[shape] == GeoType.HFIELD:
        hfd = scene.heightfield_data[scene.heightfield_index[shape]]
        geom, v0 = get_triangle_shape_from_heightfield(hfd, scene.heightfield_elevations, X_ws, tri)
        rot = wp.transform_get_rotation(X_ws)
        extrusion = TRIANGLE_PRISM_EXTRUSION
    else:
        data = scene.geom_data[shape]
        geom, v0 = get_triangle_shape_from_mesh(
            scene.shape_source_ptr[shape], wp.vec3(data[0], data[1], data[2]), X_ws, tri
        )
    # Core radius about the feature origin v0, which _make_feature offsets by |v0 - com|.
    down = wp.vec3(0.0, 0.0, -extrusion)
    core = wp.max(
        wp.max(wp.length(geom.scale), wp.length(geom.auxiliary)),
        wp.max(wp.length(down), wp.max(wp.length(geom.scale + down), wp.length(geom.auxiliary + down))),
    )
    return _make_feature(scene, shape, geom, v0, rot, core)


@wp.func
def _world_support(f: _Feature, direction: wp.vec3) -> float:
    """Return ``max_x dot(direction, x)`` over the feature in its world pose, margin included."""
    return wp.dot(direction, f.pos) + _support(f.geom, f.rot, direction) + f.local.margin


@wp.func
def _emit_constraint(f: _Feature, normal: wp.vec3, limit: float, out: Constraints):
    if f.body < 0:
        return
    if f.infinite_plane != 0:
        wp.atomic_or(out.status, 0, STATUS_UNSUPPORTED_GEOMETRY)
        return
    idx = wp.atomic_add(out.count, 0, 1)
    if idx >= out.body.shape[0]:
        wp.atomic_or(out.status, 0, STATUS_CONSTRAINT_OVERFLOW)
        return
    out.body[idx] = f.body
    out.normal[idx] = normal
    out.limit[idx] = limit
    out.feature[idx] = f.local


@wp.func
def _process_feature_pair(a: _Feature, b: _Feature, tolerance: float, out: Constraints):
    """Certify a direction ``n`` separating two convex features and emit their half-space constraints.

    The limits split the available clearance along ``n`` in proportion to the
    bodies' motion budgets, so the features never overlap along ``n`` by more
    than ``tolerance``. Features that already overlap more only may not
    overlap further.
    """
    if (a.body < 0 and b.body < 0) or (a.infinite_plane != 0 and b.infinite_plane != 0):
        return

    normal = wp.vec3(0.0, 0.0, 1.0)
    if a.infinite_plane != 0:
        normal = wp.quat_rotate(a.rot, wp.vec3(0.0, 0.0, 1.0))
    elif b.infinite_plane != 0:
        normal = -wp.quat_rotate(b.rot, wp.vec3(0.0, 0.0, 1.0))
    else:
        provider = SupportMapDataProvider()
        overlap, _gjk_distance, _gjk_point, n = _solve_gjk(a.geom, b.geom, a.rot, b.rot, a.pos, b.pos, 0.0, provider)
        if overlap:
            _hit, _mpr_distance, _mpr_point, n_mpr = _solve_mpr(
                a.geom, b.geom, a.rot, b.rot, a.pos, b.pos, 0.0, provider
            )
            n = n_mpr
        if wp.length(n) > 0.5:
            normal = wp.normalize(n)
        elif wp.length(b.pos - a.pos) > 0.0:
            normal = wp.normalize(b.pos - a.pos)

    # Upper support of A and lower support of B along the normal. An infinite
    # plane is a half-space whose support along its own normal is its offset.
    s_a = _world_support(a, normal)
    if a.infinite_plane != 0:
        s_a = wp.dot(normal, a.pos) + a.local.margin
    s_b = -_world_support(b, -normal)
    if b.infinite_plane != 0:
        s_b = wp.dot(normal, b.pos) - b.local.margin

    # The solvers' direction may come out reversed; keep the larger separation.
    if a.infinite_plane == 0 and b.infinite_plane == 0:
        s_a_reversed = _world_support(a, -normal)
        s_b_reversed = -_world_support(b, normal)
        if s_b_reversed - s_a_reversed > s_b - s_a:
            normal = -normal
            s_a = s_a_reversed
            s_b = s_b_reversed

    gap = s_b - s_a
    if gap > a.budget + b.budget:
        return  # The motion budgets alone keep this pair apart.

    available = gap + tolerance
    if available < 0.0:
        wp.atomic_add(out.recovery_count, 0, 1)
        available = 0.0
    w_a = float(0.5)
    if a.budget + b.budget > 0.0:
        w_a = a.budget / (a.budget + b.budget)
    _emit_constraint(a, normal, s_a + w_a * available, out)
    _emit_constraint(b, -normal, -(s_b - (1.0 - w_a) * available), out)


@wp.func
def _is_triangle_source(t: int) -> bool:
    return t == GeoType.MESH or t == GeoType.HFIELD


@wp.func
def _push_work(items: wp.array[wp.vec4i], counts: wp.array[wp.int32], list_id: int, item: wp.vec4i, out: Constraints):
    idx = wp.atomic_add(counts, list_id, 1)
    if idx >= items.shape[0]:
        wp.atomic_or(out.status, 0, STATUS_WORK_OVERFLOW)
        return
    items[idx] = item


@wp.func
def _enumerate_triangles(
    scene: GenerationScene,
    shape: int,
    lower: wp.vec3,
    upper: wp.vec3,
    item: wp.vec4i,
    slot: int,
    items: wp.array[wp.vec4i],
    counts: wp.array[wp.int32],
    list_id: int,
    out: Constraints,
):
    """Push ``item`` with component ``slot`` set to every triangle of ``shape`` overlapping a world AABB."""
    X_inv = wp.transform_inverse(scene.geom_transform[shape])
    is_heightfield = scene.shape_type[shape] == GeoType.HFIELD
    data = scene.geom_data[shape]
    scale = wp.vec3(1.0)
    if not is_heightfield:
        scale = wp.vec3(data[0], data[1], data[2])
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    for i in range(8):
        corner = lower
        if (i & 1) != 0:
            corner[0] = upper[0]
        if (i & 2) != 0:
            corner[1] = upper[1]
        if (i & 4) != 0:
            corner[2] = upper[2]
        p = wp.cw_div(wp.transform_point(X_inv, corner), scale)
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)

    if is_heightfield:
        hfd = scene.heightfield_data[scene.heightfield_index[shape]]
        # Terrain is solid below its surface, so only its top bounds the query.
        if hfd.nrow <= 1 or hfd.ncol <= 1 or lo[2] > hfd.max_z:
            return
        if hi[0] < -hfd.hx or lo[0] > hfd.hx or hi[1] < -hfd.hy or lo[1] > hfd.hy:
            return
        dx = 2.0 * hfd.hx / float(hfd.ncol - 1)
        dy = 2.0 * hfd.hy / float(hfd.nrow - 1)
        c0 = wp.max(int(wp.floor((lo[0] + hfd.hx) / dx)), 0)
        c1 = wp.min(int(wp.floor((hi[0] + hfd.hx) / dx)), hfd.ncol - 2)
        r0 = wp.max(int(wp.floor((lo[1] + hfd.hy) / dy)), 0)
        r1 = wp.min(int(wp.floor((hi[1] + hfd.hy) / dy)), hfd.nrow - 2)
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                for sub in range(2):
                    item[slot] = (row * (hfd.ncol - 1) + col) * 2 + sub
                    _push_work(items, counts, list_id, item, out)
    else:
        query = wp.mesh_query_aabb(scene.shape_source_ptr[shape], lo, hi)
        tri = int(0)
        while wp.mesh_query_aabb_next(query, tri):
            item[slot] = tri
            _push_work(items, counts, list_id, item, out)


@wp.func
def _pair_shapes(scene: GenerationScene, item: wp.vec4i):
    """Return ``(source, other)`` shapes of a triangle work item."""
    pair = scene.pairs[item[0]]
    if item[2] != 0:
        return pair[1], pair[0]
    return pair[0], pair[1]


@wp.func
def _search_growth(scene: GenerationScene, shape: int) -> float:
    """Return how far a shape's features may reach beyond its geometry during the update."""
    growth = scene.geom_data[shape][3] + scene.tolerance
    b = scene.shape_body[shape]
    if b >= 0:
        growth += scene.body_budget[b]
    return growth


@wp.func
def _process_ordered(src: _Feature, other: _Feature, source_is_b: int, tolerance: float, out: Constraints):
    if source_is_b != 0:
        _process_feature_pair(other, src, tolerance, out)
    else:
        _process_feature_pair(src, other, tolerance, out)


@wp.kernel(enable_backward=False)
def generate_pair_constraints_kernel(
    scene: GenerationScene,
    out: Constraints,
    work_items: wp.array[wp.vec4i],
    work_count: wp.array[wp.int32],
):
    """Emit constraints for convex pairs and source-triangle work items for mesh and heightfield pairs."""
    tid = wp.tid()
    count = scene.pair_count[0]
    if tid == 0 and count > scene.pairs.shape[0]:
        wp.atomic_or(out.status, 0, STATUS_BROAD_PHASE_OVERFLOW)
    if tid >= wp.min(count, scene.pairs.shape[0]):
        return
    pair = scene.pairs[tid]
    sa = pair[0]
    sb = pair[1]
    if sa >= scene.shape_count or sb >= scene.shape_count:
        return
    if scene.shape_body[sa] < 0 and scene.shape_body[sb] < 0:
        return
    if _is_triangle_source(scene.shape_type[sa]) or _is_triangle_source(scene.shape_type[sb]):
        item = wp.vec4i(tid, 0, 0, -1)
        if not _is_triangle_source(scene.shape_type[sa]):
            item[2] = 1
        src, other = _pair_shapes(scene, item)
        grow = wp.vec3(_search_growth(scene, src) + _search_growth(scene, other))
        lower = scene.aabb_lower[other] - grow
        upper = scene.aabb_upper[other] + grow
        _enumerate_triangles(scene, src, lower, upper, item, 1, work_items, work_count, 0, out)
        return
    _process_feature_pair(_shape_feature(scene, sa), _shape_feature(scene, sb), scene.tolerance, out)


@wp.kernel(enable_backward=False)
def generate_triangle_constraints_kernel(
    scene: GenerationScene,
    out: Constraints,
    work_items: wp.array[wp.vec4i],
    work_pairs: wp.array[wp.vec4i],
    work_count: wp.array[wp.int32],
):
    """Emit constraints for source triangles against convex shapes, or enumerate triangle-triangle pairs."""
    tid = wp.tid()
    if tid >= wp.min(work_count[0], work_items.shape[0]):
        return
    item = work_items[tid]
    src, other = _pair_shapes(scene, item)
    ft = _triangle_feature(scene, src, item[1])
    if not _is_triangle_source(scene.shape_type[other]):
        _process_ordered(ft, _shape_feature(scene, other), item[2], scene.tolerance, out)
        return
    if scene.shape_type[src] == GeoType.HFIELD and scene.shape_type[other] == GeoType.HFIELD:
        wp.atomic_or(out.status, 0, STATUS_UNSUPPORTED_GEOMETRY)
        return
    grow = _search_growth(scene, src) + _search_growth(scene, other)
    lower = wp.vec3(0.0)
    upper = wp.vec3(0.0)
    for axis in range(3):
        d = wp.vec3(0.0)
        d[axis] = 1.0
        upper[axis] = _world_support(ft, d) + grow
        lower[axis] = -_world_support(ft, -d) - grow
    _enumerate_triangles(scene, other, lower, upper, item, 3, work_pairs, work_count, 1, out)


@wp.kernel(enable_backward=False)
def generate_triangle_pair_constraints_kernel(
    scene: GenerationScene,
    out: Constraints,
    work_pairs: wp.array[wp.vec4i],
    work_count: wp.array[wp.int32],
):
    """Emit constraints for triangle-triangle pairs."""
    tid = wp.tid()
    if tid >= wp.min(work_count[1], work_pairs.shape[0]):
        return
    item = work_pairs[tid]
    src, other = _pair_shapes(scene, item)
    _process_ordered(
        _triangle_feature(scene, src, item[1]), _triangle_feature(scene, other, item[3]), item[2], scene.tolerance, out
    )


@wp.kernel(enable_backward=False)
def body_radius_kernel(
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_flags: wp.array[wp.int32],
    shape_collision_radius: wp.array[float],
    shape_aabb_lower_local: wp.array[wp.vec3],
    shape_aabb_upper_local: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    body_com: wp.array[wp.vec3],
    # output
    body_radius: wp.array[wp.float32],
):
    """Accumulate the radius about each body's center of mass that encloses its collision shapes."""
    s = wp.tid()
    b = shape_body[s]
    if b < 0 or (shape_flags[s] & ShapeFlags.COLLIDE_SHAPES) == 0:
        return
    offset = wp.length(wp.transform_get_translation(shape_transform[s]) - body_com[b])
    furthest = wp.max(wp.abs(shape_aabb_lower_local[s]), wp.abs(shape_aabb_upper_local[s]))
    wp.atomic_max(body_radius, b, offset + wp.max(wp.length(furthest), shape_collision_radius[s]) + shape_margin[s])


@wp.kernel(enable_backward=False)
def body_budget_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_radius: wp.array[wp.float32],
    dt: float,
    velocity_scale: float,
    min_budget: float,
    max_budget: float,
    # outputs
    body_budget: wp.array[wp.float32],
    body_used: wp.array[wp.float32],
    body_tracked_q: wp.array[wp.transform],
):
    """Set each body's motion budget from its predicted travel and snapshot the detection reference."""
    b = wp.tid()
    twist = body_qd[b]
    speed = wp.length(wp.spatial_top(twist)) + wp.length(wp.spatial_bottom(twist)) * body_radius[b]
    body_budget[b] = wp.clamp(velocity_scale * speed * dt, min_budget, max_budget)
    body_used[b] = 0.0
    body_tracked_q[b] = body_q[b]


@wp.kernel(enable_backward=False)
def expand_aabbs_kernel(
    shape_body: wp.array[int],
    body_budget: wp.array[wp.float32],
    sign: float,
    # outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
):
    """Grow (``sign=1``) or shrink (``sign=-1``) every moving shape's AABB by its body's motion budget."""
    s = wp.tid()
    b = shape_body[s]
    if b < 0:
        return
    e = wp.vec3(sign * body_budget[b])
    aabb_lower[s] = aabb_lower[s] - e
    aabb_upper[s] = aabb_upper[s] + e
