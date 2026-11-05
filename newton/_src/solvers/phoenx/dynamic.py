# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Runtime body, shape, and joint creation for :class:`SolverPhoenX`.

:meth:`ModelBuilder.reserve_dynamic_capacity` reserves inactive model rows. This module hands
them out from device code: each kind of row has a slot pool with a per-slot state, a compact
list of active slots, and a list of free slots that :func:`phoenx_add_body` and friends pop.

Everything runs on the device, so a CUDA graph can capture creation and removal along with the
simulation. Changes are staged by the ``wp.func`` API and applied by
:meth:`PhoenXDynamics.commit` at the start of :meth:`SolverPhoenX.step`:

* Added rows become active. Their shapes collide from the next collision pass on, because that
  pass reads the pose the step exports.
* Removed rows are deactivated, and removing a body also removes its shapes and joints. A
  removed row is quarantined for one commit before it can be handed out again, so no stale
  contact or warm-start history can match its next occupant.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import MAXVAL
from ...geometry.inertia import _inverse_inertia_wp
from ...geometry.types import GeoType
from ...sim.enums import BodyFlags, JointTargetMode, JointType
from .access_mode import ACCESS_MODE_VELOCITY_LEVEL
from .adapter_kernels import write_body_properties
from .body import BodyContainer, sym6_from_mat33
from .constraints.constraint_container import (
    CONSTRAINT_BODY1_OFFSET,
    CONSTRAINT_BODY2_OFFSET,
    ConstraintContainer,
    write_int,
)
from .helpers.math_helpers import create_orthonormal, rotate_inertia
from .materials import MaterialData

__all__ = [
    "BodyDesc",
    "DynamicsData",
    "JointDesc",
    "PhoenXDynamics",
    "ShapeDesc",
    "joint_desc",
    "phoenx_add_body",
    "phoenx_add_joint",
    "phoenx_add_shape",
    "phoenx_body_is_active",
    "phoenx_remove_body",
    "phoenx_remove_joint",
    "phoenx_remove_shape",
]

# Slot states.
_FREE = wp.constant(0)
_ACTIVE = wp.constant(1)
_ADDED = wp.constant(2)
_REMOVED = wp.constant(3)
_QUARANTINE = wp.constant(4)

_KINEMATIC = wp.constant(int(BodyFlags.KINEMATIC))
_VELOCITY_LEVEL = wp.constant(ACCESS_MODE_VELOCITY_LEVEL)
_COLLIDE_FLAGS = wp.constant((1 << 1) | (1 << 2))  # ShapeFlags.COLLIDE_SHAPES | COLLIDE_PARTICLES

_PLANE = wp.constant(int(GeoType.PLANE))
_SPHERE = wp.constant(int(GeoType.SPHERE))
_CAPSULE = wp.constant(int(GeoType.CAPSULE))
_ELLIPSOID = wp.constant(int(GeoType.ELLIPSOID))
_CYLINDER = wp.constant(int(GeoType.CYLINDER))
_BOX = wp.constant(int(GeoType.BOX))
_MESH = wp.constant(int(GeoType.MESH))
_CONE = wp.constant(int(GeoType.CONE))
_CONVEX_MESH = wp.constant(int(GeoType.CONVEX_MESH))
_VOXEL_BUDGET = wp.constant(100)  # matches ModelBuilder.finalize
_D6 = wp.constant(int(JointType.D6))
_POSITION = wp.constant(int(JointTargetMode.POSITION))
_VELOCITY = wp.constant(int(JointTargetMode.VELOCITY))
_POSITION_VELOCITY = wp.constant(int(JointTargetMode.POSITION_VELOCITY))
_ROWS = wp.constant(6)  # rows and degrees of freedom per reserved joint
_NO_FILTER = wp.constant(-2)  # body_filter_parent value of bodies without a filtered parent

vec6f = wp.types.vector(length=6, dtype=wp.float32)
vec6i = wp.types.vector(length=6, dtype=wp.int32)


@wp.struct
class SlotPool:
    """Device view of one kind of reserved rows. Slots are local; model row = ``start + slot``."""

    start: wp.int32
    state: wp.array[wp.int32]
    active_list: wp.array[wp.int32]
    free_list: wp.array[wp.int32]
    #: ``[active count, free count, next free-list entry to pop]``.
    counts: wp.array[wp.int32]


@wp.func
def _pool_alloc(pool: SlotPool) -> wp.int32:
    i = wp.atomic_add(pool.counts, 2, 1)
    if i >= pool.counts[1]:
        return -1
    slot = pool.free_list[i]
    pool.state[slot] = _ADDED
    return slot


@wp.func
def _pool_release(pool: SlotPool, row: wp.int32) -> wp.bool:
    slot = row - pool.start
    if slot < 0 or slot >= pool.state.shape[0]:
        return False
    if wp.atomic_cas(pool.state, slot, _ACTIVE, _REMOVED) == _ACTIVE:
        return True
    return wp.atomic_cas(pool.state, slot, _ADDED, _REMOVED) == _ADDED


@wp.struct
class BodyDesc:
    """Rigid body for :func:`phoenx_add_body`. Mass properties are not derived from shapes."""

    #: Body origin in the world frame.
    pose: wp.transform
    #: Center-of-mass linear velocity and angular velocity [m/s, rad/s], as in ``State.body_qd``.
    velocity: wp.spatial_vector
    #: Mass [kg]; zero makes the body static.
    mass: wp.float32
    #: Center of mass in the body frame [m].
    com: wp.vec3
    #: Inertia about the center of mass in the body frame [kg·m²].
    inertia: wp.mat33
    #: :class:`~newton.BodyFlags`; ``KINEMATIC`` makes the body follow its imported pose.
    flags: wp.int32
    #: World index.
    world: wp.int32


@wp.struct
class ShapeDesc:
    """Collision shape for :func:`phoenx_add_shape`.

    Primitive types are complete as given. ``CONVEX_MESH`` and ``MESH`` take their geometry from
    ``prototype``, a shape listed in ``prototype_shapes`` of the reservation, rescaled to ``scale``;
    meshes with SDFs keep the prototype's scale.
    """

    type: wp.int32
    #: Shape frame relative to its body [m].
    transform: wp.transform
    scale: wp.vec3
    margin: wp.float32
    gap: wp.float32
    mu: wp.float32
    restitution: wp.float32
    collision_group: wp.int32
    #: :class:`~newton.ShapeFlags`, typically ``COLLIDE_SHAPES | VISIBLE``.
    flags: wp.int32
    #: Model shape whose geometry a mesh or convex hull uses, or ``-1``.
    prototype: wp.int32
    #: World of a static shape (``body == -1``); a body's shapes take the body's world.
    world: wp.int32


@wp.struct
class JointDesc:
    """Joint for :func:`phoenx_add_joint`, a D6 joint that also covers the common joint types.

    The listed axes are free; every other direction is locked. Per-axis arrays index the free
    axes, linear ones first. Start from :func:`joint_desc`, which leaves every listed axis
    unlimited and undriven: fixed is no axes, revolute one angular axis, prismatic one linear
    axis, and ball three angular axes.
    """

    #: Parent body, or ``-1`` for the world.
    parent: wp.int32
    child: wp.int32
    #: Joint frame in the parent and child body frames [m].
    parent_xform: wp.transform
    child_xform: wp.transform
    #: Free linear and angular axes in the joint frame, one per row.
    linear_axes: wp.mat33
    angular_axes: wp.mat33
    linear_count: wp.int32
    angular_count: wp.int32
    #: Position limits [m or rad]; ``lower > upper`` locks the axis.
    limit_lower: vec6f
    limit_upper: vec6f
    #: Speed caps [m/s or rad/s]; zero is uncapped.
    velocity_limit: vec6f
    friction: vec6f
    armature: vec6f
    damping: vec6f
    #: :class:`~newton.JointTargetMode` and gains; targets come from ``Control.joint_target_q/qd``.
    target_mode: vec6i
    target_ke: vec6f
    target_kd: vec6f
    #: Whether shapes of the parent and child bodies collide.
    collide_parent: wp.bool


@wp.func
def joint_desc(parent: wp.int32, child: wp.int32, parent_xform: wp.transform, child_xform: wp.transform) -> JointDesc:
    """A fixed joint; add free axes by filling ``linear_axes``/``angular_axes`` and their counts."""
    d = JointDesc()
    d.parent = parent
    d.child = child
    d.parent_xform = parent_xform
    d.child_xform = child_xform
    d.linear_axes = wp.identity(3, dtype=wp.float32)
    d.angular_axes = wp.identity(3, dtype=wp.float32)
    d.limit_lower = vec6f(-MAXVAL)
    d.limit_upper = vec6f(MAXVAL)
    return d


@wp.struct
class DynamicsData:
    """Everything the device API writes. Pass it to user kernels as ``solver.dynamics.data``."""

    body_pool: SlotPool
    shape_pool: SlotPool
    #: Failed requests: full pools, unknown rows, or invalid descriptors.
    error_count: wp.array[wp.int32]

    body_mass: wp.array[wp.float32]
    body_inv_mass: wp.array[wp.float32]
    body_inertia: wp.array[wp.mat33]
    body_inv_inertia: wp.array[wp.mat33]
    body_com: wp.array[wp.vec3]
    body_flags: wp.array[wp.int32]
    body_world: wp.array[wp.int32]
    bodies: BodyContainer
    #: PhoenX slots whose pose must not be imported from the stale state yet.
    body_fresh: wp.array[wp.int32]

    shape_type: wp.array[wp.int32]
    shape_transform: wp.array[wp.transform]
    shape_scale: wp.array[wp.vec3]
    shape_body: wp.array[wp.int32]
    shape_world: wp.array[wp.int32]
    shape_flags: wp.array[wp.int32]
    shape_collision_group: wp.array[wp.int32]
    shape_margin: wp.array[wp.float32]
    shape_gap: wp.array[wp.float32]
    shape_is_solid: wp.array[wp.bool]
    shape_mu: wp.array[wp.float32]
    shape_restitution: wp.array[wp.float32]
    shape_collision_radius: wp.array[wp.float32]
    shape_aabb_lower: wp.array[wp.vec3]
    shape_aabb_upper: wp.array[wp.vec3]
    shape_interior_point: wp.array[wp.vec3]
    shape_voxel_resolution: wp.array[wp.vec3i]
    shape_source_ptr: wp.array[wp.uint64]
    shape_support_data: wp.array[wp.vec4i]
    shape_sdf_index: wp.array[wp.int32]
    shape_edge_range: wp.array[wp.vec2i]
    shape_mesh_properties: wp.array[wp.int32]
    shape_mesh_boundary_start: wp.array[wp.int32]
    shape_heightfield_index: wp.array[wp.int32]
    #: Flags a shape gets once its addition is committed, per shape slot.
    shape_pending_flags: wp.array[wp.int32]
    #: PhoenX's own shape -> slot map and material table.
    phoenx_shape_body: wp.array[wp.int32]
    materials: wp.array[MaterialData]
    friction_combine_mode: wp.int32
    world_map_dirty: wp.array[wp.int32]
    world_count: wp.int32

    joint_pool: SlotPool
    #: Constraint column of each model joint.
    joint_cid: wp.array[wp.int32]
    joint_type: wp.array[wp.int32]
    joint_parent: wp.array[wp.int32]
    joint_child: wp.array[wp.int32]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    joint_dof_dim: wp.array2d[wp.int32]
    joint_enabled: wp.array[wp.bool]
    joint_qd_start: wp.array[wp.int32]
    joint_target_q_start: wp.array[wp.int32]
    joint_axis: wp.array[wp.vec3]
    joint_limit_lower: wp.array[wp.float32]
    joint_limit_upper: wp.array[wp.float32]
    joint_velocity_limit: wp.array[wp.float32]
    joint_friction: wp.array[wp.float32]
    joint_armature: wp.array[wp.float32]
    joint_damping: wp.array[wp.float32]
    joint_target_mode: wp.array[wp.int32]
    joint_target_ke: wp.array[wp.float32]
    joint_target_kd: wp.array[wp.float32]
    joint_effort_limit: wp.array[wp.float32]
    joint_gear: wp.array[wp.float32]
    #: Fixed six-row joint blocks of the block joint system.
    generic_linear_axes: wp.array[wp.vec3]
    generic_angular_axes: wp.array[wp.vec3]
    generic_linear_count: wp.array[wp.int32]
    generic_angular_count: wp.array[wp.int32]
    row_dynamic: wp.array[wp.bool]
    row_dof: wp.array[wp.int32]
    row_direct_drive: wp.array[wp.bool]
    row_bounded_drive: wp.array[wp.bool]
    row_target_q: wp.array[wp.int32]
    previous_coordinate: wp.array[wp.float32]
    coordinate_revolutions: wp.array[wp.int32]
    constraints: ConstraintContainer
    joint_pgs_enabled: wp.array[wp.int32]
    #: Per model body, the joint parent whose shapes it must not collide with.
    body_filter_parent: wp.array[wp.int32]


@wp.func
def phoenx_add_body(dyn: DynamicsData, desc: BodyDesc) -> wp.int32:
    """Create a body from a reserved row; returns its model body index, or ``-1`` when full.

    The body is simulated from the next :meth:`SolverPhoenX.step` on.
    """
    slot = _pool_alloc(dyn.body_pool)
    if slot < 0 or desc.world < 0 or desc.world >= dyn.world_count:
        wp.atomic_add(dyn.error_count, 0, 1)
        if slot >= 0:
            dyn.body_pool.state[slot] = _REMOVED
        return -1
    body = dyn.body_pool.start + slot
    inv_mass = wp.float32(0.0)
    inv_inertia = wp.mat33(0.0)
    if desc.mass > 0.0:
        inv_mass = 1.0 / desc.mass
        inv_inertia = _inverse_inertia_wp(desc.inertia)
    dyn.body_mass[body] = desc.mass
    dyn.body_inv_mass[body] = inv_mass
    dyn.body_inertia[body] = desc.inertia
    dyn.body_inv_inertia[body] = inv_inertia
    dyn.body_com[body] = desc.com
    dyn.body_flags[body] = desc.flags
    dyn.body_world[body] = desc.world

    bodies = dyn.bodies
    p = body + 1
    kinematic = (desc.flags & _KINEMATIC) != 0
    write_body_properties(bodies, p, inv_mass, inv_inertia, desc.com, kinematic, desc.world)
    rot = wp.transform_get_rotation(desc.pose)
    position = wp.transform_get_translation(desc.pose) + wp.quat_rotate(rot, desc.com)
    bodies.position[p] = position
    bodies.orientation[p] = rot
    bodies.inverse_inertia_world[p] = sym6_from_mat33(rotate_inertia(wp.quat_to_matrix(rot), bodies.inverse_inertia[p]))
    bodies.velocity[p] = wp.spatial_top(desc.velocity)
    bodies.angular_velocity[p] = wp.spatial_bottom(desc.velocity)
    bodies.force[p] = wp.vec3(0.0)
    bodies.torque[p] = wp.vec3(0.0)
    bodies.position_prev[p] = position
    bodies.orientation_prev[p] = rot
    bodies.kinematic_target_pos[p] = position
    bodies.kinematic_target_orient[p] = rot
    bodies.kinematic_target_valid[p] = 0
    if kinematic:
        bodies.kinematic_target_valid[p] = 1
    bodies.position_prev_substep[p] = position
    bodies.orientation_prev_substep[p] = rot
    bodies.access_mode[p] = _VELOCITY_LEVEL
    bodies.island_root[p] = -1
    bodies.frames_below_threshold[p] = 0
    dyn.body_fresh[p] = 1
    return body


@wp.func
def phoenx_body_is_active(dyn: DynamicsData, body: wp.int32) -> wp.bool:
    """Whether a reserved body row holds a body that is not being removed."""
    slot = body - dyn.body_pool.start
    if slot < 0 or slot >= dyn.body_pool.state.shape[0]:
        return False
    state = dyn.body_pool.state[slot]
    return state == _ACTIVE or state == _ADDED


@wp.func
def phoenx_remove_body(dyn: DynamicsData, body: wp.int32) -> wp.bool:
    """Remove a runtime body together with its shapes at the next commit; ``False`` if not removable."""
    if _pool_release(dyn.body_pool, body):
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


@wp.func
def _voxel_resolution(lower: wp.vec3, upper: wp.vec3) -> wp.vec3i:
    """Port of ``compute_voxel_resolution_from_aabb`` in :meth:`ModelBuilder.finalize`."""
    size = wp.max(upper - lower, wp.vec3(1.0e-6))
    v = wp.max(wp.pow(size[0] * size[1] * size[2] / float(_VOXEL_BUDGET), 1.0 / 3.0), 1.0e-6)
    nx = wp.max(1, int(wp.round(size[0] / v)))
    ny = wp.max(1, int(wp.round(size[1] / v)))
    nz = wp.max(1, int(wp.round(size[2] / v)))
    for _ in range(3 * _VOXEL_BUDGET):
        if nx * ny * nz <= _VOXEL_BUDGET:
            break
        if nx >= ny and nx >= nz and nx > 1:
            nx -= 1
        elif ny >= nz and ny > 1:
            ny -= 1
        elif nz > 1:
            nz -= 1
    return wp.vec3i(nx, ny, nz)


@wp.func
def _primitive_bounds(geo: wp.int32, s: wp.vec3):
    """Local AABB and bounding-sphere radius of a primitive, as :meth:`ModelBuilder.finalize` derives them."""
    a = wp.vec3(wp.abs(s[0]), wp.abs(s[1]), wp.abs(s[2]))
    half = wp.vec3(1.0)
    radius = wp.length(a)
    if geo == _SPHERE:
        half = wp.vec3(a[0])
        radius = a[0]
    elif geo == _BOX:
        half = a
    elif geo == _CAPSULE:
        half = wp.vec3(a[0], a[0], a[1] + a[0])
        radius = a[0] + a[1]
    elif geo == _CYLINDER:
        radial = a[0]
        if s[2] > 0.0:
            radial += a[1] * a[1] / (s[2] + wp.sqrt(wp.max(s[2] * s[2] - a[1] * a[1], 0.0)))
        half = wp.vec3(radial, radial, a[1])
        radius = a[0] + a[1]
    elif geo == _CONE:
        half = wp.vec3(a[0], a[0], a[1])
        radius = a[0] + a[1]
    elif geo == _ELLIPSOID:
        half = a
        radius = wp.max(a[0], wp.max(a[1], a[2]))
    elif geo == _PLANE:
        radius = 1.0e6
        if a[0] > 0.0 and a[1] > 0.0:
            radius = 0.5 * wp.length(a)
    return -half, half, radius


@wp.func
def phoenx_add_shape(dyn: DynamicsData, body: wp.int32, desc: ShapeDesc) -> wp.int32:
    """Create a collision shape on ``body`` (``-1`` for static); returns its model shape index or ``-1``.

    The shape collides from the first collision pass after the next commit.
    """
    geo = desc.type
    proto = desc.prototype
    is_reference = geo == _CONVEX_MESH or geo == _MESH
    valid = (geo >= _PLANE and geo <= _CONE) or (is_reference and proto >= 0)
    if is_reference and valid:
        valid = dyn.shape_type[proto] == geo
    world = desc.world
    if body >= 0:
        world = dyn.body_world[body]
    if not valid or world < 0 or world >= dyn.world_count:
        wp.atomic_add(dyn.error_count, 0, 1)
        return -1
    slot = _pool_alloc(dyn.shape_pool)
    if slot < 0:
        wp.atomic_add(dyn.error_count, 0, 1)
        return -1
    shape = dyn.shape_pool.start + slot
    s = desc.scale
    lower = wp.vec3(0.0)
    upper = wp.vec3(0.0)
    radius = wp.float32(0.0)
    interior = wp.vec3(0.0)
    if is_reference:
        # Undo the prototype's scale and apply the new one.
        ratio = wp.cw_div(s, dyn.shape_scale[proto])
        a = wp.cw_mul(dyn.shape_aabb_lower[proto], ratio)
        b = wp.cw_mul(dyn.shape_aabb_upper[proto], ratio)
        lower = wp.min(a, b)
        upper = wp.max(a, b)
        radius = 0.5 * wp.length(upper - lower)
        interior = wp.cw_mul(dyn.shape_interior_point[proto], ratio)
        dyn.shape_source_ptr[shape] = dyn.shape_source_ptr[proto]
        dyn.shape_support_data[shape] = dyn.shape_support_data[proto]
        dyn.shape_sdf_index[shape] = dyn.shape_sdf_index[proto]
        dyn.shape_edge_range[shape] = dyn.shape_edge_range[proto]
        dyn.shape_mesh_properties[shape] = dyn.shape_mesh_properties[proto]
        dyn.shape_mesh_boundary_start[shape] = dyn.shape_mesh_boundary_start[proto]
    else:
        lower, upper, radius = _primitive_bounds(geo, s)
        dyn.shape_source_ptr[shape] = wp.uint64(0)
        dyn.shape_support_data[shape] = wp.vec4i(-1, -1, -1, 0)
        dyn.shape_sdf_index[shape] = -1
        dyn.shape_edge_range[shape] = wp.vec2i(-1, 0)
        dyn.shape_mesh_properties[shape] = 0
        dyn.shape_mesh_boundary_start[shape] = -1
    dyn.shape_heightfield_index[shape] = -1
    dyn.shape_type[shape] = geo
    dyn.shape_transform[shape] = desc.transform
    dyn.shape_scale[shape] = s
    dyn.shape_body[shape] = body
    dyn.shape_world[shape] = world
    dyn.shape_collision_group[shape] = desc.collision_group
    dyn.shape_margin[shape] = desc.margin
    dyn.shape_gap[shape] = desc.gap
    dyn.shape_is_solid[shape] = True
    dyn.shape_mu[shape] = desc.mu
    dyn.shape_restitution[shape] = desc.restitution
    dyn.shape_collision_radius[shape] = radius
    dyn.shape_aabb_lower[shape] = lower
    dyn.shape_aabb_upper[shape] = upper
    dyn.shape_interior_point[shape] = interior
    dyn.shape_voxel_resolution[shape] = _voxel_resolution(lower, upper)
    dyn.shape_pending_flags[slot] = desc.flags
    material = MaterialData()
    material.static_friction = desc.mu
    material.dynamic_friction = desc.mu
    material.restitution = desc.restitution
    material.friction_combine_mode = dyn.friction_combine_mode
    material.restitution_combine_mode = 0
    dyn.materials[shape] = material
    dyn.phoenx_shape_body[shape] = body + 1
    return shape


@wp.func
def phoenx_remove_shape(dyn: DynamicsData, shape: wp.int32) -> wp.bool:
    """Remove a runtime shape at the next commit; ``False`` if not removable."""
    if _pool_release(dyn.shape_pool, shape):
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


@wp.func
def _store_locked_axes(axes: wp.array[wp.vec3], joint: wp.int32, free: wp.mat33, free_count: wp.int32) -> wp.int32:
    """Store an orthonormal basis of the directions ``free`` leaves locked; returns its size.

    With one free axis the third entry holds that axis, as the two-locked-axes rows expect.
    """
    base = joint * 3
    if free_count == 0:
        axes[base] = wp.vec3(1.0, 0.0, 0.0)
        axes[base + 1] = wp.vec3(0.0, 1.0, 0.0)
        axes[base + 2] = wp.vec3(0.0, 0.0, 1.0)
        return 3
    a = wp.normalize(free[0])
    if free_count == 1:
        t = wp.normalize(create_orthonormal(a))
        axes[base] = t
        axes[base + 1] = wp.cross(a, t)
        axes[base + 2] = a
        return 2
    if free_count == 2:
        axes[base] = wp.normalize(wp.cross(a, free[1]))
        return 1
    return 0


@wp.func
def _axes_valid(axes: wp.mat33, count: wp.int32) -> wp.bool:
    """Nonzero, independent free axes."""
    if count < 0 or count > 3:
        return False
    for i in range(count):
        if wp.length(axes[i]) < 1.0e-6:
            return False
    if count == 2:
        return wp.length(wp.cross(axes[0], axes[1])) > 1.0e-6
    if count == 3:
        return wp.abs(wp.determinant(axes)) > 1.0e-6
    return True


@wp.func
def phoenx_add_joint(dyn: DynamicsData, desc: JointDesc) -> wp.int32:
    """Create a joint from a reserved row; returns its model joint index or ``-1``.

    The joint is solved from the next :meth:`SolverPhoenX.step` on. Drives and armature become
    local joint rows; limits, speed caps, and friction become D6 inequality rows. Effort limits
    are not enforced.
    """
    n_lin = desc.linear_count
    n_free = n_lin + desc.angular_count
    body_count = dyn.body_mass.shape[0]
    valid = _axes_valid(desc.linear_axes, n_lin) and _axes_valid(desc.angular_axes, desc.angular_count)
    valid = valid and desc.child >= 0 and desc.child < body_count and desc.parent >= -1
    valid = valid and desc.parent < body_count and desc.parent != desc.child
    for k in range(_ROWS):
        if k < n_free and desc.limit_lower[k] > desc.limit_upper[k]:
            valid = False
    slot = wp.int32(-1)
    if valid:
        slot = _pool_alloc(dyn.joint_pool)
    if slot < 0:
        wp.atomic_add(dyn.error_count, 0, 1)
        return -1
    joint = dyn.joint_pool.start + slot
    cid = dyn.joint_cid[joint]
    qd = dyn.joint_qd_start[joint]
    target_q = dyn.joint_target_q_start[joint]
    dyn.joint_type[joint] = _D6
    dyn.joint_parent[joint] = desc.parent
    dyn.joint_child[joint] = desc.child
    dyn.joint_X_p[joint] = desc.parent_xform
    dyn.joint_X_c[joint] = desc.child_xform
    dyn.joint_dof_dim[joint, 0] = n_lin
    dyn.joint_dof_dim[joint, 1] = desc.angular_count
    dyn.joint_enabled[joint] = True

    lin = _store_locked_axes(dyn.generic_linear_axes, joint, desc.linear_axes, n_lin)
    ang = _store_locked_axes(dyn.generic_angular_axes, joint, desc.angular_axes, desc.angular_count)
    dyn.generic_linear_count[joint] = lin
    dyn.generic_angular_count[joint] = ang

    d6 = dyn.constraints.d6
    d6.linear_count[cid] = n_lin
    d6.angular_count[cid] = desc.angular_count
    d6.joint_x_p[cid] = desc.parent_xform
    d6.joint_x_c[cid] = desc.child_xform
    d6.reaction_wrench[cid] = wp.spatial_vector()
    row_base = joint * _ROWS
    used_rows = lin + ang  # structural rows come first
    d6_rows = wp.int32(0)
    for k in range(_ROWS):
        dof = qd + k
        axis = wp.vec3(0.0)
        if k < n_lin:
            axis = desc.linear_axes[k]
        elif k < n_free:
            axis = desc.angular_axes[k - n_lin]
        lower = wp.float32(-MAXVAL)
        upper = wp.float32(MAXVAL)
        mode = wp.int32(0)
        if k < n_free:
            lower = desc.limit_lower[k]
            upper = desc.limit_upper[k]
            mode = desc.target_mode[k]
        ke = wp.where(k < n_free, desc.target_ke[k], 0.0)
        kd = wp.where(k < n_free, desc.target_kd[k], 0.0)
        armature = wp.where(k < n_free, desc.armature[k], 0.0)
        damping = wp.where(k < n_free, desc.damping[k], 0.0)
        friction = wp.where(k < n_free, wp.max(desc.friction[k], 0.0), 0.0)
        speed = wp.where(k < n_free, desc.velocity_limit[k], 0.0)
        dyn.joint_axis[dof] = axis
        dyn.joint_limit_lower[dof] = lower
        dyn.joint_limit_upper[dof] = upper
        dyn.joint_velocity_limit[dof] = speed
        dyn.joint_friction[dof] = friction
        dyn.joint_armature[dof] = armature
        dyn.joint_damping[dof] = damping
        dyn.joint_target_mode[dof] = mode
        dyn.joint_target_ke[dof] = ke
        dyn.joint_target_kd[dof] = kd
        dyn.joint_effort_limit[dof] = MAXVAL
        dyn.joint_gear[dof] = 1.0

        row = row_base + k
        dyn.row_dynamic[row] = False
        dyn.row_dof[row] = -1
        dyn.row_direct_drive[row] = False
        dyn.row_bounded_drive[row] = False
        dyn.row_target_q[row] = 0
        dyn.previous_coordinate[row] = 0.0
        dyn.coordinate_revolutions[row] = 0

        d6.axis[cid, k] = axis
        d6.revolution_counter[cid, k] = 0
        d6.previous_angle[cid, k] = 0.0
        d6.coordinate[cid, k] = 0.0
        d6.lower_impulse[cid, k] = 0.0
        d6.upper_impulse[cid, k] = 0.0
        d6.friction_impulse[cid, k] = 0.0
        d6.row_axis[cid, k] = -1
        d6.row_kind[cid, k] = 0
    for k in range(_ROWS):
        if k < n_free:
            mode = desc.target_mode[k]
            position = mode == _POSITION or mode == _POSITION_VELOCITY
            drive = (position and desc.target_ke[k] > 0.0) or (mode == _VELOCITY and desc.target_kd[k] > 0.0)
            if drive or desc.armature[k] > 0.0 or desc.damping[k] > 0.0:
                # A dynamic row per driven or armature-carrying axis, after the structural rows.
                row = row_base + used_rows
                dyn.row_dynamic[row] = True
                dyn.row_dof[row] = qd + k
                dyn.row_direct_drive[row] = drive
                dyn.row_target_q[row] = target_q + k
                used_rows += 1
            speed = desc.velocity_limit[k]
            if speed <= 0.0 or speed >= 1.0e5:
                speed = 0.0
            friction = wp.max(desc.friction[k], 0.0)
            finite = desc.limit_lower[k] > -1.0e5 or desc.limit_upper[k] < 1.0e5
            if finite or speed > 0.0 or friction > 0.0:
                d6.row_axis[cid, d6_rows] = k
                d6.lower[cid, d6_rows] = desc.limit_lower[k]
                d6.upper[cid, d6_rows] = desc.limit_upper[k]
                d6.velocity_limit[cid, d6_rows] = speed
                d6.friction[cid, d6_rows] = friction
                d6.friction_slip_scale[cid, d6_rows] = -1.0
                d6.unwrap_angle[cid, d6_rows] = wp.where(desc.angular_count == 1 and k >= n_lin, 1, 0)
                d6.condense_translation[cid, d6_rows] = 0
                d6_rows += 1
    d6.row_count[cid] = d6_rows
    dyn.constraints.bilateral.row_count[cid] = used_rows
    dyn.constraints.bilateral.valid[cid] = 0
    write_int(dyn.constraints, CONSTRAINT_BODY1_OFFSET, cid, desc.parent + 1)
    write_int(dyn.constraints, CONSTRAINT_BODY2_OFFSET, cid, desc.child + 1)
    if not desc.collide_parent:
        dyn.body_filter_parent[desc.child] = desc.parent
    dyn.joint_pgs_enabled[cid] = 1
    return joint


@wp.func
def phoenx_remove_joint(dyn: DynamicsData, joint: wp.int32) -> wp.bool:
    """Remove a runtime joint at the next commit; ``False`` if not removable."""
    if _pool_release(dyn.joint_pool, joint):
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


@wp.func
def _resolve(state: wp.array[wp.int32], slot: wp.int32, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    """Advance one slot's state; returns the state it had."""
    old = state[slot]
    new = old
    if old == _ADDED:
        new = _ACTIVE
    elif old == _REMOVED:
        new = _QUARANTINE
    elif old == _QUARANTINE:
        new = _FREE
    state[slot] = new
    active[slot] = wp.where(new == _ACTIVE, 1, 0)
    free[slot] = wp.where(new == _FREE, 1, 0)
    return old


@wp.kernel(enable_backward=False)
def _cascade_shapes_kernel(dyn: DynamicsData):
    """Remove the shapes of removed bodies."""
    slot = wp.tid()
    body = dyn.shape_body[dyn.shape_pool.start + slot]
    body_slot = body - dyn.body_pool.start
    if body_slot >= 0 and body_slot < dyn.body_pool.state.shape[0]:
        if dyn.body_pool.state[body_slot] == _REMOVED:
            _pool_release(dyn.shape_pool, dyn.shape_pool.start + slot)


@wp.func
def _body_removed(dyn: DynamicsData, body: wp.int32) -> wp.bool:
    slot = body - dyn.body_pool.start
    if slot < 0 or slot >= dyn.body_pool.state.shape[0]:
        return False
    return dyn.body_pool.state[slot] == _REMOVED


@wp.kernel(enable_backward=False)
def _cascade_joints_kernel(dyn: DynamicsData):
    """Remove the joints of removed bodies."""
    joint = dyn.joint_pool.start + wp.tid()
    if _body_removed(dyn, dyn.joint_parent[joint]) or _body_removed(dyn, dyn.joint_child[joint]):
        _pool_release(dyn.joint_pool, joint)


@wp.kernel(enable_backward=False)
def _commit_joints_kernel(dyn: DynamicsData, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    slot = wp.tid()
    if _resolve(dyn.joint_pool.state, slot, active, free) != _REMOVED:
        return
    joint = dyn.joint_pool.start + slot
    cid = dyn.joint_cid[joint]
    dyn.joint_pgs_enabled[cid] = 0
    dyn.constraints.bilateral.row_count[cid] = 0
    dyn.constraints.d6.row_count[cid] = 0
    dyn.joint_enabled[joint] = False
    dyn.generic_linear_count[joint] = 0
    dyn.generic_angular_count[joint] = 0
    for k in range(_ROWS):
        dyn.row_dynamic[joint * _ROWS + k] = False
    child = dyn.joint_child[joint]
    if dyn.body_filter_parent[child] == dyn.joint_parent[joint]:
        dyn.body_filter_parent[child] = _NO_FILTER


@wp.kernel(enable_backward=False)
def _commit_bodies_kernel(dyn: DynamicsData, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    slot = wp.tid()
    if _resolve(dyn.body_pool.state, slot, active, free) != _REMOVED:
        return
    # A removed body becomes an inert static slot until it is reused.
    body = dyn.body_pool.start + slot
    dyn.body_mass[body] = 0.0
    dyn.body_inv_mass[body] = 0.0
    dyn.body_inertia[body] = wp.mat33(0.0)
    dyn.body_inv_inertia[body] = wp.mat33(0.0)
    dyn.body_flags[body] = 0
    p = body + 1
    write_body_properties(dyn.bodies, p, 0.0, wp.mat33(0.0), dyn.body_com[body], False, 0)
    dyn.bodies.velocity[p] = wp.vec3(0.0)
    dyn.bodies.angular_velocity[p] = wp.vec3(0.0)
    dyn.bodies.force[p] = wp.vec3(0.0)
    dyn.bodies.torque[p] = wp.vec3(0.0)
    dyn.bodies.island_root[p] = -1
    dyn.bodies.frames_below_threshold[p] = 0
    dyn.body_fresh[p] = 0


@wp.kernel(enable_backward=False)
def _wake_orphaned_islands_kernel(dyn: DynamicsData):
    """Wake sleeping bodies whose island root was just removed, so the slot cannot alias them."""
    p = wp.tid()
    root = dyn.bodies.island_root[p]
    root_slot = root - 1 - dyn.body_pool.start
    if root >= 0 and root_slot >= 0 and root_slot < dyn.body_pool.state.shape[0]:
        if dyn.body_pool.state[root_slot] == _QUARANTINE:
            dyn.bodies.island_root[p] = -1
            dyn.bodies.frames_below_threshold[p] = 0


@wp.kernel(enable_backward=False)
def _commit_shapes_kernel(dyn: DynamicsData, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    slot = wp.tid()
    old = _resolve(dyn.shape_pool.state, slot, active, free)
    shape = dyn.shape_pool.start + slot
    if old == _ADDED:
        dyn.shape_flags[shape] = dyn.shape_pending_flags[slot]
        dyn.world_map_dirty[0] = 1
    elif old == _REMOVED:
        dyn.shape_flags[shape] = dyn.shape_flags[shape] & ~_COLLIDE_FLAGS
        dyn.shape_body[shape] = -1
        dyn.phoenx_shape_body[shape] = 0
        dyn.world_map_dirty[0] = 1


@wp.kernel(enable_backward=False)
def _scatter_lists_kernel(
    pool: SlotPool,
    active: wp.array[wp.int32],
    active_prefix: wp.array[wp.int32],
    free: wp.array[wp.int32],
    free_prefix: wp.array[wp.int32],
):
    slot = wp.tid()
    if active[slot] != 0:
        pool.active_list[active_prefix[slot]] = slot
    if free[slot] != 0:
        pool.free_list[free_prefix[slot]] = slot
    last = pool.state.shape[0] - 1
    if slot == last:
        pool.counts[0] = active_prefix[last] + active[last]
        pool.counts[1] = free_prefix[last] + free[last]
        pool.counts[2] = 0


class _HostPool:
    """Host owner of one :class:`SlotPool` and its commit scratch."""

    def __init__(self, start: int, capacity: int, device):
        cap = max(1, capacity)
        self.capacity = capacity
        self.data = SlotPool()
        self.data.start = start
        self.data.state = wp.zeros(cap, dtype=wp.int32, device=device)
        self.data.active_list = wp.zeros(cap, dtype=wp.int32, device=device)
        self.data.free_list = wp.array(np.arange(cap, dtype=np.int32), device=device)
        self.data.counts = wp.array([0, capacity, 0], dtype=wp.int32, device=device)
        self.active = wp.zeros(cap, dtype=wp.int32, device=device)
        self.free = wp.zeros(cap, dtype=wp.int32, device=device)
        self.active_prefix = wp.zeros(cap, dtype=wp.int32, device=device)
        self.free_prefix = wp.zeros(cap, dtype=wp.int32, device=device)

    def rebuild_lists(self):
        if self.capacity == 0:
            return
        wp.utils.array_scan(self.active, self.active_prefix, inclusive=False)
        wp.utils.array_scan(self.free, self.free_prefix, inclusive=False)
        wp.launch(
            _scatter_lists_kernel,
            dim=self.capacity,
            inputs=[self.data, self.active, self.active_prefix, self.free, self.free_prefix],
            device=self.active.device,
        )


class PhoenXDynamics:
    """Runtime creation state of a :class:`SolverPhoenX`; exposed as ``solver.dynamics``.

    Pass :attr:`data` to kernels that call :func:`phoenx_add_body`, :func:`phoenx_add_shape`,
    and the removal functions.
    """

    def __init__(self, solver, collision_pipeline, friction_combine_mode: int):
        model = solver.model
        capacity = model._dynamic_capacity
        device = model.device
        self.capacity = capacity
        self.bodies = _HostPool(capacity.body_start, capacity.body_count, device)
        self.shapes = _HostPool(capacity.shape_start, capacity.shape_count, device)
        self.joints = _HostPool(capacity.joint_start, capacity.joint_count, device)
        self._sleeping = solver._sleeping_enabled
        world = solver.world

        d = DynamicsData()
        d.body_pool = self.bodies.data
        d.shape_pool = self.shapes.data
        d.error_count = wp.zeros(1, dtype=wp.int32, device=device)
        for name in ("body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia", "body_com", "body_flags"):
            setattr(d, name, getattr(model, name))
        d.body_world = model.body_world
        d.bodies = solver.bodies
        d.body_fresh = solver._body_fresh
        for name in (
            "shape_type",
            "shape_transform",
            "shape_scale",
            "shape_body",
            "shape_world",
            "shape_flags",
            "shape_collision_group",
            "shape_margin",
            "shape_gap",
            "shape_is_solid",
            "shape_collision_radius",
            "shape_source_ptr",
            "shape_heightfield_index",
        ):
            setattr(d, name, getattr(model, name))
        d.shape_mu = model.shape_material_mu
        d.shape_restitution = model.shape_material_restitution
        d.shape_aabb_lower = model.shape_collision_aabb_lower
        d.shape_aabb_upper = model.shape_collision_aabb_upper
        d.shape_interior_point = model._shape_interior_point
        d.shape_voxel_resolution = model._shape_voxel_resolution
        d.shape_support_data = model._shape_support_data
        d.shape_sdf_index = model._shape_sdf_index
        d.shape_edge_range = model._shape_edge_range
        d.shape_mesh_properties = model._shape_mesh_properties
        d.shape_mesh_boundary_start = model._shape_mesh_boundary_start
        d.shape_pending_flags = wp.zeros(max(1, capacity.shape_count), dtype=wp.int32, device=device)
        d.phoenx_shape_body = solver._shape_body
        d.materials = world._materials
        d.friction_combine_mode = wp.int32(friction_combine_mode)
        d.world_map_dirty = collision_pipeline._dynamic_world_map_dirty
        d.world_count = wp.int32(max(1, int(model.world_count)))

        d.joint_pool = self.joints.data
        d.constraints = world.constraints
        d.joint_pgs_enabled = world._joint_pgs_enabled
        d.body_filter_parent = solver._body_filter_parent
        if capacity.joint_count:
            d.joint_cid = solver._joint_constraints.joint_idx_to_cid
            for name in (
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_X_p",
                "joint_X_c",
                "joint_dof_dim",
                "joint_enabled",
                "joint_qd_start",
                "joint_target_q_start",
                "joint_axis",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_velocity_limit",
                "joint_friction",
                "joint_armature",
                "joint_damping",
                "joint_target_mode",
                "joint_target_ke",
                "joint_target_kd",
                "joint_effort_limit",
                "joint_gear",
            ):
                setattr(d, name, getattr(model, name))
            rows = solver._direct_equality_system
            for name in (
                "generic_linear_axes",
                "generic_angular_axes",
                "generic_linear_count",
                "generic_angular_count",
                "row_dynamic",
                "row_dof",
                "row_direct_drive",
                "row_bounded_drive",
                "row_target_q",
                "previous_coordinate",
                "coordinate_revolutions",
            ):
                setattr(d, name, getattr(rows, name))
        self.data = d

        # Prototypes only carry geometry; they never collide or render.
        prototypes = list(capacity.prototype_shapes)
        if prototypes:
            flags = model.shape_flags.numpy()
            flags[prototypes] &= ~(int(_COLLIDE_FLAGS) | (1 << 0))  # also clear VISIBLE
            model.shape_flags.assign(flags)
            d.world_map_dirty.fill_(1)

    @property
    def error_count(self) -> wp.array[wp.int32]:
        """Device counter of failed requests since construction."""
        return self.data.error_count

    def commit(self) -> None:
        """Apply staged additions and removals; captured by :meth:`SolverPhoenX.step`."""
        device = self.data.error_count.device
        # Cascades read the removed bodies before their commit quarantines them.
        if self.shapes.capacity:
            wp.launch(_cascade_shapes_kernel, dim=self.shapes.capacity, inputs=[self.data], device=device)
        if self.joints.capacity:
            wp.launch(_cascade_joints_kernel, dim=self.joints.capacity, inputs=[self.data], device=device)
            wp.launch(
                _commit_joints_kernel,
                dim=self.joints.capacity,
                inputs=[self.data, self.joints.active, self.joints.free],
                device=device,
            )
            self.joints.rebuild_lists()
        if self.bodies.capacity:
            wp.launch(
                _commit_bodies_kernel,
                dim=self.bodies.capacity,
                inputs=[self.data, self.bodies.active, self.bodies.free],
                device=device,
            )
            if self._sleeping:
                wp.launch(
                    _wake_orphaned_islands_kernel,
                    dim=self.data.bodies.island_root.shape[0],
                    inputs=[self.data],
                    device=device,
                )
            self.bodies.rebuild_lists()
        if self.shapes.capacity:
            wp.launch(
                _commit_shapes_kernel,
                dim=self.shapes.capacity,
                inputs=[self.data, self.shapes.active, self.shapes.free],
                device=device,
            )
            self.shapes.rebuild_lists()
