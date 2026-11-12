# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Runtime body, shape, and joint creation for :class:`SolverPhoenX`.

:meth:`ModelBuilder.reserve_dynamic_capacity` reserves inactive model rows. This module hands
them out from device code: each kind of row has a slot pool with a per-slot state, a compact
list of active slots, and a list of free slots that :func:`add_body` and friends pop.

Everything runs on the device, so a CUDA graph can capture creation and removal along with the
simulation. The ``wp.func`` API only claims slots and stages descriptors; :meth:`Dynamics._commit`
writes them into the model and solver rows in parallel at the start of :meth:`SolverPhoenX.step`:

* Added rows become active. Their shapes collide from the next collision pass on, because that
  pass reads the pose the step exports.
* Removed rows are deactivated, and removing a body also removes its shapes and joints. A
  removed row is quarantined for one commit before it can be handed out again, so no stale
  contact or warm-start history can match its next occupant.

The device handle :class:`DynamicsData` stays small on purpose: Warp passes structs to functions
by value, and a struct of more than about 8 KB forwarded through nested functions is copied
through local memory on every call. The commit kernels take the rows they write as separate
structs, each well below that size.
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
from .articulations.maximal_projector_general import GeneralMaximalTreeProjectorData
from .body import MOTION_ARTICULATED, BodyContainer, sym6_from_mat33
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
    "Dynamics",
    "DynamicsData",
    "JointDesc",
    "ShapeDesc",
    "add_articulation",
    "add_body",
    "add_joint",
    "add_shape",
    "articulation_body",
    "articulation_joint",
    "body_desc",
    "is_articulation_active",
    "is_body_active",
    "joint_desc",
    "joint_wrench",
    "remove_articulation",
    "remove_body",
    "remove_joint",
    "remove_shape",
    "shape_desc",
]

# Slot states. A row removed before its addition was committed is cancelled: nothing of it was
# written, so the commit only quarantines it.
_FREE = wp.constant(0)
_ACTIVE = wp.constant(1)
_ADDED = wp.constant(2)
_REMOVED = wp.constant(3)
_QUARANTINE = wp.constant(4)
_CANCELLED = wp.constant(5)

_KINEMATIC = wp.constant(int(BodyFlags.KINEMATIC))
_VELOCITY_LEVEL = wp.constant(ACCESS_MODE_VELOCITY_LEVEL)
_COLLIDE_FLAGS = wp.constant((1 << 1) | (1 << 2))  # ShapeFlags.COLLIDE_SHAPES | COLLIDE_PARTICLES
_COLLIDE_SHAPES = wp.constant(1 << 1)
_VISIBLE = wp.constant(1 << 0)

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
_FIXED_JOINT = wp.constant(int(JointType.FIXED))
_FREE_JOINT = wp.constant(int(JointType.FREE))
_POSITION = wp.constant(int(JointTargetMode.POSITION))
_VELOCITY = wp.constant(int(JointTargetMode.VELOCITY))
_POSITION_VELOCITY = wp.constant(int(JointTargetMode.POSITION_VELOCITY))
_ROWS = wp.constant(6)  # rows and degrees of freedom per reserved joint
_NO_FILTER = wp.constant(-2)  # body_filter_parent value of bodies without a filtered parent
_ARTICULATED = wp.constant(MOTION_ARTICULATED)
_ARTICULATION_GROUP = wp.constant(-(1 << 24))  # first per-instance negative collision group
_TREE_LANES = wp.constant(32)  # lanes of an exact tree solve

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
    return wp.atomic_cas(pool.state, slot, _ADDED, _CANCELLED) == _ADDED


@wp.func
def _pool_removed(pool: SlotPool, row: wp.int32) -> wp.bool:
    """Whether ``row`` is a slot of ``pool`` that is being removed or cancelled."""
    slot = row - pool.start
    if slot < 0 or slot >= pool.state.shape[0]:
        return False
    state = pool.state[slot]
    return state == _REMOVED or state == _CANCELLED


@wp.struct
class BodyDesc:
    """Rigid body for :func:`add_body`; start from :func:`body_desc`. Shapes do not add mass."""

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
    """Collision shape for :func:`add_shape`; start from :func:`shape_desc`.

    ``CONVEX_MESH`` and ``MESH`` shapes take their geometry from ``prototype``, one of the
    ``prototype_shapes`` given to :meth:`~newton.ModelBuilder.reserve_dynamic_capacity`, rescaled
    to ``scale``; meshes with SDFs keep the prototype's scale.
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
    #: Detect overlaps without a contact response, as ``SolverPhoenX(tracking_volumes=...)``.
    tracking_volume: wp.bool


@wp.struct
class JointDesc:
    """Joint for :func:`add_joint`, a D6 joint that also covers the common joint types.

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
def body_desc(pose: wp.transform, mass: wp.float32, inertia: wp.mat33) -> BodyDesc:
    """A body at rest in world 0 with its center of mass at the body origin."""
    d = BodyDesc()
    d.pose = pose
    d.mass = mass
    d.inertia = inertia
    return d


@wp.func
def shape_desc(geo_type: wp.int32, scale: wp.vec3) -> ShapeDesc:
    """A colliding, visible shape at its body's origin with :class:`~newton.ModelBuilder.ShapeConfig` defaults."""
    d = ShapeDesc()
    d.type = geo_type
    d.transform = wp.transform_identity()
    d.scale = scale
    d.gap = 0.1
    d.mu = 1.0
    d.collision_group = 1
    d.flags = _COLLIDE_SHAPES | _VISIBLE
    d.prototype = -1
    return d


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
    """Device handle of the runtime-creation functions; pass ``solver.dynamics.data`` to kernels."""

    body_pool: SlotPool
    shape_pool: SlotPool
    joint_pool: SlotPool
    #: Runtime articulation instances.
    articulation_pool: SlotPool
    #: Failed requests: full pools, unknown rows, or invalid descriptors.
    error_count: wp.array[wp.int32]
    world_count: wp.int32
    body_count: wp.int32

    #: Model body worlds, assigned when a body is added so shapes can follow in the same kernel.
    body_world: wp.array[wp.int32]
    #: Model shape types, to check mesh prototypes.
    shape_type: wp.array[wp.int32]
    #: Descriptors of added rows per slot, written into the rows at the next commit.
    staged_bodies: wp.array[BodyDesc]
    staged_shapes: wp.array[ShapeDesc]
    staged_shape_body: wp.array[wp.int32]
    staged_joints: wp.array[JointDesc]

    #: Constraint column of each model joint and the wrench it applied; see :func:`joint_wrench`.
    joint_cid: wp.array[wp.int32]
    constraint_wrench: wp.array[wp.spatial_vector]

    #: Templates, laid out per template by ``*_start``.
    template_body_start: wp.array[wp.int32]
    template_shape_start: wp.array[wp.int32]
    template_joint_start: wp.array[wp.int32]
    #: Template bodies posed relative to the template's first body.
    template_bodies: wp.array[BodyDesc]
    template_shapes: wp.array[ShapeDesc]
    #: Template-local body of each template shape.
    template_shape_body: wp.array[wp.int32]
    #: Template joints whose parent and child are template-local bodies (``-1`` is the world).
    template_joints: wp.array[JointDesc]
    #: Per template: tree lanes (zero when it is not solved as a tree), and per lane its
    #: template body and joint (``-1`` for a floating root).
    template_tree_count: wp.array[wp.int32]
    template_tree_body: wp.array2d[wp.int32]
    template_tree_joint: wp.array2d[wp.int32]

    #: Template, model bodies, and model joints of each instance; ``-1`` past the template's size.
    articulation_template: wp.array[wp.int32]
    articulation_bodies: wp.array2d[wp.int32]
    articulation_joints: wp.array2d[wp.int32]
    #: Spawn pose, velocity, and world of each instance, laid out by the commit.
    articulation_pose: wp.array[wp.transform]
    articulation_velocity: wp.array[wp.spatial_vector]
    articulation_world: wp.array[wp.int32]
    #: Bodies of an exact tree solve per instance, zero without one; the commit lays out the tree.
    tree_body_count: wp.array[wp.int32]
    #: Instance whose tree solve owns each reserved body and joint slot, or ``-1``.
    body_articulation: wp.array[wp.int32]
    joint_articulation: wp.array[wp.int32]

    #: Reduced-coordinate instance slots; ``slot_size`` is zero without them. Slot ``s`` owns
    #: ``slot_size`` consecutive bodies and joints.
    slot_size: wp.int32
    slot_body_start: wp.int32
    slot_joint_start: wp.int32
    #: Runtime shapes of each reduced instance, removed with it.
    instance_shapes: wp.array2d[wp.int32]
    #: Nonzero when every reduced instance is built with the joint tree of one template:
    #: template ``t`` owns the instances from ``template_instance_start[t]`` on, and the free
    #: ones among them are ``template_free_start[t]`` entries into the pool's free list.
    built_instances: wp.int32
    template_instance_start: wp.array[wp.int32]
    template_free_start: wp.array[wp.int32]
    template_free_next: wp.array[wp.int32]
    #: First model body and joint of each reduced instance.
    instance_body_start: wp.array[wp.int32]
    instance_joint_start: wp.array[wp.int32]


# ---------------------------------------------------------------------------
# Device API: claim slots and stage descriptors
# ---------------------------------------------------------------------------


@wp.func
def _fail(dyn: DynamicsData) -> wp.int32:
    wp.atomic_add(dyn.error_count, 0, 1)
    return -1


@wp.func
def _drop_tree(dyn: DynamicsData, owners: wp.array[wp.int32], slot: wp.int32):
    """Stop solving the tree of the instance owning ``slot`` exactly; its rows fall back to local blocks."""
    if owners.shape[0] > slot:
        instance = owners[slot]
        if instance >= 0:
            dyn.tree_body_count[instance] = 0
            owners[slot] = -1


@wp.func
def add_body(dyn: DynamicsData, desc: BodyDesc) -> wp.int32:
    """Add a body; returns its model body index, or ``-1`` when the reserved rows are used up.

    The body is simulated from the next :meth:`~newton.solvers.SolverPhoenX.step` on.
    """
    if desc.world < 0 or desc.world >= dyn.world_count:
        return _fail(dyn)
    slot = _pool_alloc(dyn.body_pool)
    if slot < 0:
        return _fail(dyn)
    body = dyn.body_pool.start + slot
    if dyn.body_articulation.shape[0] > slot:
        dyn.body_articulation[slot] = -1
    dyn.staged_bodies[slot] = desc
    dyn.body_world[body] = desc.world
    return body


@wp.func
def is_body_active(dyn: DynamicsData, body: wp.int32) -> wp.bool:
    """Whether a reserved body row holds a body that is not being removed."""
    slot = body - dyn.body_pool.start
    if slot < 0 or slot >= dyn.body_pool.state.shape[0]:
        return False
    state = dyn.body_pool.state[slot]
    return state == _ACTIVE or state == _ADDED


@wp.func
def remove_body(dyn: DynamicsData, body: wp.int32) -> wp.bool:
    """Remove a runtime body with its shapes and joints at the next :meth:`~newton.solvers.SolverPhoenX.step`; ``False`` if not removable."""
    if _pool_release(dyn.body_pool, body):
        _drop_tree(dyn, dyn.body_articulation, body - dyn.body_pool.start)
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


@wp.func
def add_shape(dyn: DynamicsData, body: wp.int32, desc: ShapeDesc) -> wp.int32:
    """Add a collision shape to ``body``, or a static one for ``-1``; returns its model shape index or ``-1``.

    The shape collides from the first collision pass after the next :meth:`~newton.solvers.SolverPhoenX.step`.
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
        return _fail(dyn)
    slot = _pool_alloc(dyn.shape_pool)
    if slot < 0:
        return _fail(dyn)
    dyn.staged_shapes[slot] = desc
    dyn.staged_shape_body[slot] = body
    return dyn.shape_pool.start + slot


@wp.func
def remove_shape(dyn: DynamicsData, shape: wp.int32) -> wp.bool:
    """Remove a runtime shape at the next :meth:`~newton.solvers.SolverPhoenX.step`; ``False`` if not removable."""
    if _pool_release(dyn.shape_pool, shape):
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


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
def add_joint(dyn: DynamicsData, desc: JointDesc) -> wp.int32:
    """Add a joint; returns its model joint index, or ``-1`` when rows are used up or ``desc`` is invalid.

    The joint is solved from the next :meth:`~newton.solvers.SolverPhoenX.step` on and requires ``joint_mode="maximal_pgs"``. Drive
    targets come from :class:`~newton.Control` at the joint's rows. Effort limits are not enforced.
    """
    n_free = desc.linear_count + desc.angular_count
    valid = _axes_valid(desc.linear_axes, desc.linear_count) and _axes_valid(desc.angular_axes, desc.angular_count)
    valid = valid and desc.child >= 0 and desc.child < dyn.body_count and desc.parent >= -1
    valid = valid and desc.parent < dyn.body_count and desc.parent != desc.child
    for k in range(_ROWS):
        if k < n_free and desc.limit_lower[k] > desc.limit_upper[k]:
            valid = False
    slot = wp.int32(-1)
    if valid:
        slot = _pool_alloc(dyn.joint_pool)
    if slot < 0:
        return _fail(dyn)
    if dyn.joint_articulation.shape[0] > slot:
        dyn.joint_articulation[slot] = -1
    dyn.staged_joints[slot] = desc
    return dyn.joint_pool.start + slot


@wp.func
def remove_joint(dyn: DynamicsData, joint: wp.int32) -> wp.bool:
    """Remove a runtime joint at the next :meth:`~newton.solvers.SolverPhoenX.step`; ``False`` if not removable."""
    if _pool_release(dyn.joint_pool, joint):
        _drop_tree(dyn, dyn.joint_articulation, joint - dyn.joint_pool.start)
        return True
    wp.atomic_add(dyn.error_count, 0, 1)
    return False


@wp.func
def joint_wrench(dyn: DynamicsData, joint: wp.int32) -> wp.spatial_vector:
    """World-frame force and torque [N, N·m] that ``joint`` applies to its child body.

    Valid for any model joint, as of the latest :meth:`Dynamics.update_joint_wrenches`; it averages
    the last substep of the step before.
    """
    return dyn.constraint_wrench[dyn.joint_cid[joint]]


@wp.func
def _remove_articulation_rows(dyn: DynamicsData, instance: wp.int32):
    if dyn.tree_body_count.shape[0] > instance:
        dyn.tree_body_count[instance] = 0
    for i in range(dyn.articulation_bodies.shape[1]):
        body = dyn.articulation_bodies[instance, i]
        if body >= 0:
            _pool_release(dyn.body_pool, body)  # its shapes and joints follow at the commit


@wp.func
def _release_instance_shapes(dyn: DynamicsData, instance: wp.int32):
    for k in range(dyn.instance_shapes.shape[1]):
        shape = dyn.instance_shapes[instance, k]
        if shape >= 0:
            _pool_release(dyn.shape_pool, shape)
            dyn.instance_shapes[instance, k] = -1


@wp.func
def _clear_instance_maps(dyn: DynamicsData, instance: wp.int32):
    for i in range(dyn.articulation_bodies.shape[1]):
        dyn.articulation_bodies[instance, i] = -1
    for i in range(dyn.articulation_joints.shape[1]):
        dyn.articulation_joints[instance, i] = -1


@wp.func
def _spawn_body(template_body: BodyDesc, pose: wp.transform, velocity: wp.spatial_vector, world: wp.int32) -> BodyDesc:
    """A template body moved by the spawn pose and carried by the instance's rigid motion."""
    desc = template_body
    desc.pose = pose * template_body.pose
    com = wp.transform_point(desc.pose, desc.com)
    angular = wp.spatial_bottom(velocity)
    desc.velocity = wp.spatial_vector(
        wp.spatial_top(velocity) + wp.cross(angular, com - wp.transform_get_translation(pose)), angular
    )
    desc.world = world
    return desc


@wp.func
def _has_reduced_instances(dyn: DynamicsData) -> wp.bool:
    return dyn.slot_size > 0 or dyn.built_instances != 0


@wp.func
def _alloc_reduced(dyn: DynamicsData, template: wp.int32) -> wp.int32:
    """Claim a free reduced instance that can hold ``template``, or ``-1``."""
    if dyn.built_instances == 0:
        return _pool_alloc(dyn.articulation_pool)
    first = dyn.template_free_start[template]
    i = wp.atomic_add(dyn.template_free_next, template, 1)
    if i >= dyn.template_free_start[template + 1] - first:
        return -1
    instance = dyn.articulation_pool.free_list[first + i]
    dyn.articulation_pool.state[instance] = _ADDED
    return instance


@wp.func
def _add_reduced(
    dyn: DynamicsData, template: wp.int32, pose: wp.transform, velocity: wp.spatial_vector, world: wp.int32
) -> wp.int32:
    """Claim a reduced instance; the commit lays the template tree into it."""
    lanes = dyn.template_tree_count[template]
    if lanes <= 0 or (dyn.built_instances == 0 and lanes > dyn.slot_size) or world < 0 or world >= dyn.world_count:
        return _fail(dyn)
    instance = _alloc_reduced(dyn, template)
    if instance < 0:
        return _fail(dyn)
    _clear_instance_maps(dyn, instance)
    b0 = dyn.instance_body_start[instance]
    j0 = dyn.instance_joint_start[instance]
    for lane in range(lanes):
        dyn.articulation_bodies[instance, dyn.template_tree_body[template, lane]] = b0 + lane
        dyn.body_world[b0 + lane] = world
        local_joint = dyn.template_tree_joint[template, lane]
        if local_joint >= 0:
            dyn.articulation_joints[instance, local_joint] = j0 + lane
    tree_group = _ARTICULATION_GROUP - instance  # no collisions within an instance
    ok = wp.bool(True)
    s0 = dyn.template_shape_start[template]
    for i in range(s0, dyn.template_shape_start[template + 1]):
        sdesc = dyn.template_shapes[i]
        sdesc.collision_group = tree_group
        shape = add_shape(dyn, dyn.articulation_bodies[instance, dyn.template_shape_body[i]], sdesc)
        dyn.instance_shapes[instance, i - s0] = shape
        ok = ok and shape >= 0
    if not ok:
        _release_instance_shapes(dyn, instance)
        _pool_release(dyn.articulation_pool, instance)
        return -1
    return instance


@wp.func
def add_articulation(
    dyn: DynamicsData, template: wp.int32, pose: wp.transform, velocity: wp.spatial_vector, world: wp.int32
) -> wp.int32:
    """Add an instance of an articulation template; returns its index, or ``-1`` when rows run out.

    The instance is simulated from the next :meth:`~newton.solvers.SolverPhoenX.step` on, in the coordinates chosen by
    ``articulation_coordinates`` in :meth:`~newton.ModelBuilder.reserve_dynamic_capacity`. Its
    joints start at their template coordinates; drive targets come from :class:`~newton.Control`
    at the rows of :func:`articulation_joint`. With instances reserved per template, the call
    fails once every instance of ``template`` is in the scene.

    Args:
        dyn: ``solver.dynamics.data``.
        template: Index into the reservation's ``articulation_templates``.
        pose: World pose of the template's first body; the other bodies keep their offsets.
        velocity: Rigid motion of the whole instance: linear velocity at the origin of ``pose``
            and angular velocity [m/s, rad/s], world frame.
        world: World index of the instance.
    """
    if template < 0 or template >= dyn.template_body_start.shape[0] - 1:
        return _fail(dyn)
    reduced = _has_reduced_instances(dyn)
    if not reduced and (world < 0 or world >= dyn.world_count):
        return _fail(dyn)
    instance = wp.int32(-1)
    if reduced:
        instance = _add_reduced(dyn, template, pose, velocity, world)
    else:
        instance = _pool_alloc(dyn.articulation_pool)
        if instance < 0:
            return _fail(dyn)
    if instance < 0:
        return -1
    dyn.articulation_template[instance] = template
    dyn.articulation_pose[instance] = pose
    dyn.articulation_velocity[instance] = velocity
    dyn.articulation_world[instance] = world
    if reduced:
        return instance
    _clear_instance_maps(dyn, instance)

    ok = wp.bool(True)
    b0 = dyn.template_body_start[template]
    for i in range(b0, dyn.template_body_start[template + 1]):
        body = add_body(dyn, _spawn_body(dyn.template_bodies[i], pose, velocity, world))
        dyn.articulation_bodies[instance, i - b0] = body
        ok = ok and body >= 0
    if ok:
        for i in range(dyn.template_shape_start[template], dyn.template_shape_start[template + 1]):
            sdesc = dyn.template_shapes[i]
            sdesc.collision_group = _ARTICULATION_GROUP - instance  # no collisions within an instance
            body = dyn.articulation_bodies[instance, dyn.template_shape_body[i]]
            ok = ok and add_shape(dyn, body, sdesc) >= 0
    if ok:
        j0 = dyn.template_joint_start[template]
        for i in range(j0, dyn.template_joint_start[template + 1]):
            jdesc = dyn.template_joints[i]
            if jdesc.parent >= 0:
                jdesc.parent = dyn.articulation_bodies[instance, jdesc.parent]
            else:
                jdesc.parent_xform = pose * jdesc.parent_xform
            jdesc.child = dyn.articulation_bodies[instance, jdesc.child]
            joint = add_joint(dyn, jdesc)
            dyn.articulation_joints[instance, i - j0] = joint
            ok = ok and joint >= 0
    if not ok:
        _remove_articulation_rows(dyn, instance)
        _pool_release(dyn.articulation_pool, instance)
        return -1
    lanes = dyn.template_tree_count[template]
    if lanes > 0 and dyn.tree_body_count.shape[0] > instance:
        # The tree solve owns these rows until one of them is removed.
        for lane in range(lanes):
            body = dyn.articulation_bodies[instance, dyn.template_tree_body[template, lane]]
            dyn.body_articulation[body - dyn.body_pool.start] = instance
            local_joint = dyn.template_tree_joint[template, lane]
            if local_joint >= 0:
                joint = dyn.articulation_joints[instance, local_joint]
                dyn.joint_articulation[joint - dyn.joint_pool.start] = instance
        dyn.tree_body_count[instance] = lanes
    return instance


@wp.func
def remove_articulation(dyn: DynamicsData, instance: wp.int32) -> wp.bool:
    """Remove an instance with all its bodies, shapes, and joints at the next :meth:`~newton.solvers.SolverPhoenX.step`.

    Instances in reduced coordinates can only be removed as a whole. Removing a body or joint of an
    instance in maximal coordinates keeps the rest, solved as independent joints.
    """
    if not _pool_release(dyn.articulation_pool, instance):
        wp.atomic_add(dyn.error_count, 0, 1)
        return False
    if _has_reduced_instances(dyn):
        _release_instance_shapes(dyn, instance)
    else:
        _remove_articulation_rows(dyn, instance)
    return True


@wp.func
def is_articulation_active(dyn: DynamicsData, instance: wp.int32) -> wp.bool:
    """Whether ``instance`` is an articulation that is not being removed."""
    if instance < 0 or instance >= dyn.articulation_pool.state.shape[0]:
        return False
    state = dyn.articulation_pool.state[instance]
    return state == _ACTIVE or state == _ADDED


@wp.func
def articulation_body(dyn: DynamicsData, instance: wp.int32, index: wp.int32) -> wp.int32:
    """Model body of ``instance`` that template body ``index`` became, or ``-1``."""
    return dyn.articulation_bodies[instance, index]


@wp.func
def articulation_joint(dyn: DynamicsData, instance: wp.int32, index: wp.int32) -> wp.int32:
    """Model joint of ``instance`` that template joint ``index`` became, or ``-1``; free root joints are not counted."""
    return dyn.articulation_joints[instance, index]


# ---------------------------------------------------------------------------
# Commit: rows written from staged descriptors, one thread per slot
# ---------------------------------------------------------------------------


@wp.struct
class _BodyRows:
    """Model and PhoenX rows of bodies."""

    mass: wp.array[wp.float32]
    inv_mass: wp.array[wp.float32]
    inertia: wp.array[wp.mat33]
    inv_inertia: wp.array[wp.mat33]
    com: wp.array[wp.vec3]
    flags: wp.array[wp.int32]
    bodies: BodyContainer
    #: PhoenX slots whose pose must not be imported from the stale state yet.
    fresh: wp.array[wp.int32]


@wp.struct
class _ShapeRows:
    """Model and PhoenX rows of shapes."""

    type: wp.array[wp.int32]
    transform: wp.array[wp.transform]
    scale: wp.array[wp.vec3]
    body: wp.array[wp.int32]
    body_world: wp.array[wp.int32]
    world: wp.array[wp.int32]
    flags: wp.array[wp.int32]
    collision_group: wp.array[wp.int32]
    margin: wp.array[wp.float32]
    gap: wp.array[wp.float32]
    is_solid: wp.array[wp.bool]
    mu: wp.array[wp.float32]
    restitution: wp.array[wp.float32]
    collision_radius: wp.array[wp.float32]
    aabb_lower: wp.array[wp.vec3]
    aabb_upper: wp.array[wp.vec3]
    interior_point: wp.array[wp.vec3]
    voxel_resolution: wp.array[wp.vec3i]
    source_ptr: wp.array[wp.uint64]
    support_data: wp.array[wp.vec4i]
    sdf_index: wp.array[wp.int32]
    edge_range: wp.array[wp.vec2i]
    mesh_properties: wp.array[wp.int32]
    mesh_boundary_start: wp.array[wp.int32]
    heightfield_index: wp.array[wp.int32]
    #: PhoenX's own shape -> slot map and material table.
    phoenx_body: wp.array[wp.int32]
    no_response: wp.array[wp.int32]
    materials: wp.array[MaterialData]
    friction_combine_mode: wp.int32
    world_map_dirty: wp.array[wp.int32]


@wp.struct
class _JointRows:
    """Model rows of joints and their degrees of freedom."""

    type: wp.array[wp.int32]
    parent: wp.array[wp.int32]
    child: wp.array[wp.int32]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    dof_dim: wp.array2d[wp.int32]
    enabled: wp.array[wp.bool]
    qd_start: wp.array[wp.int32]
    target_q_start: wp.array[wp.int32]
    axis: wp.array[wp.vec3]
    limit_lower: wp.array[wp.float32]
    limit_upper: wp.array[wp.float32]
    velocity_limit: wp.array[wp.float32]
    friction: wp.array[wp.float32]
    armature: wp.array[wp.float32]
    damping: wp.array[wp.float32]
    target_mode: wp.array[wp.int32]
    target_ke: wp.array[wp.float32]
    target_kd: wp.array[wp.float32]
    effort_limit: wp.array[wp.float32]
    gear: wp.array[wp.float32]


@wp.struct
class _BlockRows:
    """PhoenX rows of maximal joints: six-row blocks of the block joint system and D6 columns."""

    joint_cid: wp.array[wp.int32]
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
    pgs_enabled: wp.array[wp.int32]
    #: Per model body, the joint parent whose shapes it must not collide with.
    body_filter_parent: wp.array[wp.int32]


@wp.struct
class _TreeTemplates:
    """Per template and lane: depth, parent lane, and children of an exact tree solve."""

    floating: wp.array[wp.int32]
    max_depth: wp.array[wp.int32]
    depth: wp.array2d[wp.int32]
    parent: wp.array2d[wp.int32]
    child_start: wp.array2d[wp.int32]
    child_index: wp.array2d[wp.int32]


@wp.struct
class ReducedSlots:
    """Reduced solver tables that spawning rewrites in place for an instance slot.

    A slot's rows hold its joint range, depth-walk tables, child lists, and per-body joint
    paths, all sized for a full chain.
    """

    #: Reduced articulation index of slot 0; slots are consecutive.
    articulation: wp.int32
    articulation_start: wp.array[wp.int32]
    articulation_end: wp.array[wp.int32]
    depth_start: wp.array2d[wp.int32]
    depth_joint: wp.array[wp.int32]
    parent_lane: wp.array[wp.int32]
    child_start: wp.array[wp.int32]
    child_joint: wp.array[wp.int32]
    child_depth_index: wp.array[wp.int32]
    body_path_start: wp.array[wp.int32]
    body_path_joint: wp.array[wp.int32]
    body_x_com: wp.array[wp.transform]
    joint_s: wp.array[wp.spatial_vector]
    joint_s_publish: wp.array[wp.spatial_vector]
    #: Authoritative generalized and body state of the reduced solver.
    joint_q: wp.array[wp.float32]
    joint_qd: wp.array[wp.float32]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    joint_q_start: wp.array[wp.int32]
    joint_limit_ke: wp.array[wp.float32]
    joint_limit_kd: wp.array[wp.float32]
    #: Per template joint: Newton joint type, coordinates (up to 7), and limit stiffness and damping.
    template_joint_type: wp.array[wp.int32]
    template_joint_q: wp.array2d[wp.float32]
    template_joint_limit_ke: wp.array2d[wp.float32]
    template_joint_limit_kd: wp.array2d[wp.float32]
    #: Contact Jacobian rows the reduced contact solver keeps across steps; a slot whose old
    #: topology can no longer appear clears its columns and row owners before reuse.
    packed_jacobian: wp.array2d[wp.float32]
    packed_response: wp.array2d[wp.float32]
    previous_row_body: wp.array[wp.int32]
    previous_row_body_pair: wp.array[wp.int32]
    cache_rows: wp.int32
    dof_start: wp.int32
    clear: wp.array[wp.int32]


@wp.func
def _resolve(pool: SlotPool, slot: wp.int32, active: wp.array[wp.int32], free: wp.array[wp.int32]) -> wp.int32:
    """Advance one slot's state; returns the state it had."""
    old = pool.state[slot]
    new = old
    if old == _ADDED:
        new = _ACTIVE
    elif old == _REMOVED or old == _CANCELLED:
        new = _QUARANTINE
    elif old == _QUARANTINE:
        new = _FREE
    pool.state[slot] = new
    active[slot] = wp.where(new == _ACTIVE, 1, 0)
    free[slot] = wp.where(new == _FREE, 1, 0)
    return old


@wp.func
def _write_body(rows: _BodyRows, body: wp.int32, desc: BodyDesc):
    """Write a body's model rows and PhoenX state."""
    inv_mass = wp.float32(0.0)
    inv_inertia = wp.mat33(0.0)
    if desc.mass > 0.0:
        inv_mass = 1.0 / desc.mass
        inv_inertia = _inverse_inertia_wp(desc.inertia)
    rows.mass[body] = desc.mass
    rows.inv_mass[body] = inv_mass
    rows.inertia[body] = desc.inertia
    rows.inv_inertia[body] = inv_inertia
    rows.com[body] = desc.com
    rows.flags[body] = desc.flags

    bodies = rows.bodies
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
    rows.fresh[p] = 1


@wp.kernel(enable_backward=False)
def _commit_bodies_kernel(dyn: DynamicsData, rows: _BodyRows, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    slot = wp.tid()
    old = _resolve(dyn.body_pool, slot, active, free)
    body = dyn.body_pool.start + slot
    if old == _ADDED:
        _write_body(rows, body, dyn.staged_bodies[slot])
    elif old == _REMOVED:
        # A removed body becomes an inert static slot until it is reused.
        rows.mass[body] = 0.0
        rows.inv_mass[body] = 0.0
        rows.inertia[body] = wp.mat33(0.0)
        rows.inv_inertia[body] = wp.mat33(0.0)
        rows.flags[body] = 0
        p = body + 1
        write_body_properties(rows.bodies, p, 0.0, wp.mat33(0.0), rows.com[body], False, 0)
        rows.bodies.velocity[p] = wp.vec3(0.0)
        rows.bodies.angular_velocity[p] = wp.vec3(0.0)
        rows.bodies.force[p] = wp.vec3(0.0)
        rows.bodies.torque[p] = wp.vec3(0.0)
        rows.bodies.island_root[p] = -1
        rows.bodies.frames_below_threshold[p] = 0
        rows.fresh[p] = 0


@wp.kernel(enable_backward=False)
def _wake_orphaned_islands_kernel(pool: SlotPool, bodies: BodyContainer):
    """Wake sleeping bodies whose island root was just removed, so the slot cannot alias them."""
    p = wp.tid()
    root = bodies.island_root[p]
    root_slot = root - 1 - pool.start
    if root >= 0 and root_slot >= 0 and root_slot < pool.state.shape[0]:
        if pool.state[root_slot] == _QUARANTINE:
            bodies.island_root[p] = -1
            bodies.frames_below_threshold[p] = 0


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
def _write_shape(rows: _ShapeRows, shape: wp.int32, body: wp.int32, desc: ShapeDesc):
    """Write a shape's model rows, bounds, and PhoenX material."""
    geo = desc.type
    proto = desc.prototype
    s = desc.scale
    lower = wp.vec3(0.0)
    upper = wp.vec3(0.0)
    radius = wp.float32(0.0)
    interior = wp.vec3(0.0)
    if geo == _CONVEX_MESH or geo == _MESH:
        # Undo the prototype's scale and apply the new one.
        ratio = wp.cw_div(s, rows.scale[proto])
        a = wp.cw_mul(rows.aabb_lower[proto], ratio)
        b = wp.cw_mul(rows.aabb_upper[proto], ratio)
        lower = wp.min(a, b)
        upper = wp.max(a, b)
        radius = 0.5 * wp.length(upper - lower)
        interior = wp.cw_mul(rows.interior_point[proto], ratio)
        rows.source_ptr[shape] = rows.source_ptr[proto]
        rows.support_data[shape] = rows.support_data[proto]
        rows.sdf_index[shape] = rows.sdf_index[proto]
        rows.edge_range[shape] = rows.edge_range[proto]
        rows.mesh_properties[shape] = rows.mesh_properties[proto]
        rows.mesh_boundary_start[shape] = rows.mesh_boundary_start[proto]
    else:
        lower, upper, radius = _primitive_bounds(geo, s)
        rows.source_ptr[shape] = wp.uint64(0)
        rows.support_data[shape] = wp.vec4i(-1, -1, -1, 0)
        rows.sdf_index[shape] = -1
        rows.edge_range[shape] = wp.vec2i(-1, 0)
        rows.mesh_properties[shape] = 0
        rows.mesh_boundary_start[shape] = -1
    rows.heightfield_index[shape] = -1
    rows.type[shape] = geo
    rows.transform[shape] = desc.transform
    rows.scale[shape] = s
    rows.body[shape] = body
    rows.world[shape] = wp.where(body >= 0, rows.body_world[wp.max(body, 0)], desc.world)
    rows.flags[shape] = desc.flags
    rows.collision_group[shape] = desc.collision_group
    rows.margin[shape] = desc.margin
    rows.gap[shape] = desc.gap
    rows.is_solid[shape] = True
    rows.mu[shape] = desc.mu
    rows.restitution[shape] = desc.restitution
    rows.collision_radius[shape] = radius
    rows.aabb_lower[shape] = lower
    rows.aabb_upper[shape] = upper
    rows.interior_point[shape] = interior
    rows.voxel_resolution[shape] = _voxel_resolution(lower, upper)
    material = MaterialData()
    material.static_friction = desc.mu
    material.dynamic_friction = desc.mu
    material.restitution = desc.restitution
    material.friction_combine_mode = rows.friction_combine_mode
    material.restitution_combine_mode = 0
    rows.materials[shape] = material
    rows.phoenx_body[shape] = body + 1
    rows.no_response[shape] = wp.where(desc.tracking_volume, 1, 0)


@wp.kernel(enable_backward=False)
def _cascade_shapes_kernel(dyn: DynamicsData, shape_body: wp.array[wp.int32]):
    """Remove the shapes of removed bodies."""
    slot = wp.tid()
    shape = dyn.shape_pool.start + slot
    body = shape_body[shape]
    if dyn.shape_pool.state[slot] == _ADDED:
        body = dyn.staged_shape_body[slot]
    if _pool_removed(dyn.body_pool, body):
        _pool_release(dyn.shape_pool, shape)


@wp.kernel(enable_backward=False)
def _commit_shapes_kernel(dyn: DynamicsData, rows: _ShapeRows, active: wp.array[wp.int32], free: wp.array[wp.int32]):
    slot = wp.tid()
    old = _resolve(dyn.shape_pool, slot, active, free)
    shape = dyn.shape_pool.start + slot
    if old == _ADDED:
        _write_shape(rows, shape, dyn.staged_shape_body[slot], dyn.staged_shapes[slot])
        rows.world_map_dirty[0] = 1
    elif old == _REMOVED:
        rows.flags[shape] = rows.flags[shape] & ~_COLLIDE_FLAGS
        rows.body[shape] = -1
        rows.phoenx_body[shape] = 0
        rows.world_map_dirty[0] = 1


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
def _write_joint(rows: _JointRows, block: _BlockRows, joint: wp.int32, desc: JointDesc):
    """Write a maximal D6 joint's model rows, block rows, and constraint column."""
    n_lin = desc.linear_count
    n_free = n_lin + desc.angular_count
    cid = block.joint_cid[joint]
    qd = rows.qd_start[joint]
    target_q = rows.target_q_start[joint]
    rows.type[joint] = _D6
    rows.parent[joint] = desc.parent
    rows.child[joint] = desc.child
    rows.X_p[joint] = desc.parent_xform
    rows.X_c[joint] = desc.child_xform
    rows.dof_dim[joint, 0] = n_lin
    rows.dof_dim[joint, 1] = desc.angular_count
    rows.enabled[joint] = True

    lin = _store_locked_axes(block.generic_linear_axes, joint, desc.linear_axes, n_lin)
    ang = _store_locked_axes(block.generic_angular_axes, joint, desc.angular_axes, desc.angular_count)
    block.generic_linear_count[joint] = lin
    block.generic_angular_count[joint] = ang

    d6 = block.constraints.d6
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
        rows.axis[dof] = axis
        rows.limit_lower[dof] = lower
        rows.limit_upper[dof] = upper
        rows.velocity_limit[dof] = wp.where(k < n_free, desc.velocity_limit[k], 0.0)
        rows.friction[dof] = wp.where(k < n_free, wp.max(desc.friction[k], 0.0), 0.0)
        rows.armature[dof] = wp.where(k < n_free, desc.armature[k], 0.0)
        rows.damping[dof] = wp.where(k < n_free, desc.damping[k], 0.0)
        rows.target_mode[dof] = mode
        rows.target_ke[dof] = wp.where(k < n_free, desc.target_ke[k], 0.0)
        rows.target_kd[dof] = wp.where(k < n_free, desc.target_kd[k], 0.0)
        rows.effort_limit[dof] = MAXVAL
        rows.gear[dof] = 1.0

        row = row_base + k
        block.row_dynamic[row] = False
        block.row_dof[row] = -1
        block.row_direct_drive[row] = False
        block.row_bounded_drive[row] = False
        block.row_target_q[row] = 0
        block.previous_coordinate[row] = 0.0
        block.coordinate_revolutions[row] = 0

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
                block.row_dynamic[row] = True
                block.row_dof[row] = qd + k
                block.row_direct_drive[row] = drive
                block.row_target_q[row] = target_q + k
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
    block.constraints.bilateral.row_count[cid] = used_rows
    block.constraints.bilateral.valid[cid] = 0
    write_int(block.constraints, CONSTRAINT_BODY1_OFFSET, cid, desc.parent + 1)
    write_int(block.constraints, CONSTRAINT_BODY2_OFFSET, cid, desc.child + 1)
    if not desc.collide_parent:
        block.body_filter_parent[desc.child] = desc.parent
    block.pgs_enabled[cid] = 1


@wp.kernel(enable_backward=False)
def _cascade_joints_kernel(dyn: DynamicsData, joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32]):
    """Remove the joints of removed bodies."""
    slot = wp.tid()
    joint = dyn.joint_pool.start + slot
    parent = joint_parent[joint]
    child = joint_child[joint]
    if dyn.joint_pool.state[slot] == _ADDED:
        parent = dyn.staged_joints[slot].parent
        child = dyn.staged_joints[slot].child
    if _pool_removed(dyn.body_pool, parent) or _pool_removed(dyn.body_pool, child):
        _pool_release(dyn.joint_pool, joint)


@wp.kernel(enable_backward=False)
def _commit_joints_kernel(
    dyn: DynamicsData, rows: _JointRows, block: _BlockRows, active: wp.array[wp.int32], free: wp.array[wp.int32]
):
    slot = wp.tid()
    old = _resolve(dyn.joint_pool, slot, active, free)
    joint = dyn.joint_pool.start + slot
    if old == _ADDED:
        _write_joint(rows, block, joint, dyn.staged_joints[slot])
    elif old == _REMOVED:
        cid = block.joint_cid[joint]
        block.pgs_enabled[cid] = 0
        block.constraints.bilateral.row_count[cid] = 0
        block.constraints.d6.row_count[cid] = 0
        rows.enabled[joint] = False
        block.generic_linear_count[joint] = 0
        block.generic_angular_count[joint] = 0
        for k in range(_ROWS):
            block.row_dynamic[joint * _ROWS + k] = False
        child = rows.child[joint]
        if block.body_filter_parent[child] == rows.parent[joint]:
            block.body_filter_parent[child] = _NO_FILTER


@wp.kernel(enable_backward=False)
def _layout_trees_kernel(dyn: DynamicsData, trees: _TreeTemplates, tree: GeneralMaximalTreeProjectorData):
    """Lay out the exact tree solve of each added instance from its template's lane table."""
    tid = wp.tid()
    instance = tid // (_TREE_LANES + 1)
    lane = tid - instance * (_TREE_LANES + 1)
    lanes = dyn.tree_body_count[instance]
    if dyn.articulation_pool.state[instance] != _ADDED or lanes == 0:
        return
    template = dyn.articulation_template[instance]
    tree.child_start[instance, lane] = trees.child_start[template, lane]
    if lane == _TREE_LANES:
        return
    body = wp.int32(-1)
    joint = wp.int32(-1)
    if lane < lanes:
        body = dyn.articulation_bodies[instance, dyn.template_tree_body[template, lane]] + 1  # PhoenX slot
        local_joint = dyn.template_tree_joint[template, lane]
        if local_joint >= 0:
            joint = dyn.articulation_joints[instance, local_joint]
    tree.body_slot[instance, lane] = body
    tree.joint_index[instance, lane] = joint
    tree.depth[instance, lane] = trees.depth[template, lane]
    tree.parent[instance, lane] = trees.parent[template, lane]
    tree.child_index[instance, lane] = trees.child_index[template, lane]
    if lane == 0:
        tree.floating_root[instance] = trees.floating[template]
        tree.max_depth[instance] = trees.max_depth[template]


@wp.func
def _write_reduced_joint(
    rows: _JointRows,
    slots: ReducedSlots,
    joint: wp.int32,
    joint_type: wp.int32,
    parent: wp.int32,
    child: wp.int32,
    parent_xform: wp.transform,
    child_xform: wp.transform,
    desc: JointDesc,
    template_joint: wp.int32,
):
    """Write one slot joint's model rows; unused degrees of freedom stay inert."""
    rows.type[joint] = joint_type
    rows.parent[joint] = parent
    rows.child[joint] = child
    rows.X_p[joint] = parent_xform
    rows.X_c[joint] = child_xform
    rows.dof_dim[joint, 0] = desc.linear_count
    rows.dof_dim[joint, 1] = desc.angular_count
    rows.enabled[joint] = True
    n_lin = desc.linear_count
    n_free = n_lin + desc.angular_count
    qd = rows.qd_start[joint]
    for k in range(rows.qd_start[joint + 1] - qd):
        dof = qd + k
        used = k < n_free
        axis = wp.vec3(0.0)
        if k < n_lin:
            axis = desc.linear_axes[k]
        elif used:
            axis = desc.angular_axes[k - n_lin]
        rows.axis[dof] = axis
        rows.limit_lower[dof] = wp.where(used, desc.limit_lower[k], -MAXVAL)
        rows.limit_upper[dof] = wp.where(used, desc.limit_upper[k], MAXVAL)
        rows.velocity_limit[dof] = wp.where(used, desc.velocity_limit[k], MAXVAL)
        rows.friction[dof] = wp.where(used, desc.friction[k], 0.0)
        # Unused rows are padding: a unit armature keeps their factor pivot regular.
        rows.armature[dof] = wp.where(used, desc.armature[k], 1.0)
        rows.damping[dof] = wp.where(used, desc.damping[k], 0.0)
        rows.target_mode[dof] = wp.where(used, desc.target_mode[k], 0)
        rows.target_ke[dof] = wp.where(used, desc.target_ke[k], 0.0)
        rows.target_kd[dof] = wp.where(used, desc.target_kd[k], 0.0)
        rows.effort_limit[dof] = MAXVAL
        rows.gear[dof] = 1.0
        ke = wp.float32(0.0)
        kd = wp.float32(0.0)
        if used and template_joint >= 0:
            ke = slots.template_joint_limit_ke[template_joint, k]
            kd = slots.template_joint_limit_kd[template_joint, k]
        slots.joint_limit_ke[dof] = ke
        slots.joint_limit_kd[dof] = kd
        slots.joint_qd[dof] = 0.0
        slots.joint_s[dof] = wp.spatial_vector()
        slots.joint_s_publish[dof] = wp.spatial_vector()
    q = slots.joint_q_start[joint]
    for k in range(slots.joint_q_start[joint + 1] - q):
        value = wp.float32(0.0)
        if template_joint >= 0 and k < 7:
            value = slots.template_joint_q[template_joint, k]
        slots.joint_q[q + k] = value


@wp.func
def _write_slot_topology(
    trees: _TreeTemplates,
    slots: ReducedSlots,
    template: wp.int32,
    a: wp.int32,
    j0: wp.int32,
    b0: wp.int32,
    lanes: wp.int32,
    size: wp.int32,
):
    """Rewrite the slot's rows of the reduced solver's tables for the template tree.

    Lanes are parent first, so lanes sorted by depth are already in lane order, and each body's
    root path is no longer than the placeholder chain's at that lane.
    """
    slots.articulation_end[a] = slots.articulation_start[a] + lanes
    base = slots.depth_start[a, 0]
    for d in range(slots.depth_start.shape[1]):
        below = wp.int32(0)
        for lane in range(lanes):
            if trees.depth[template, lane] < d:
                below += 1
        slots.depth_start[a, d] = base + below
    for lane in range(lanes):
        slots.depth_joint[base + lane] = j0 + lane
        parent = trees.parent[template, lane]
        parent_lane = wp.int32(0)
        if parent >= 0:
            parent_first = parent
            while parent_first > 0 and trees.depth[template, parent_first - 1] == trees.depth[template, parent]:
                parent_first -= 1
            parent_lane = parent - parent_first
        slots.parent_lane[j0 + lane] = parent_lane
    child_base = slots.child_start[j0]
    for lane in range(size):
        slots.child_start[j0 + lane] = child_base + trees.child_start[template, wp.min(lane, lanes)]
    for c in range(trees.child_start[template, lanes]):
        child_lane = trees.child_index[template, c]
        slots.child_joint[child_base + c] = j0 + child_lane
        slots.child_depth_index[child_base + c] = base + child_lane
    # Exact root paths, packed into the slot's path region.
    cursor = slots.body_path_start[b0]
    for lane in range(size):
        slots.body_path_start[b0 + lane] = cursor
        if lane < lanes:
            depth = trees.depth[template, lane]
            node = lane
            for k in range(depth + 1):
                slots.body_path_joint[cursor + depth - k] = j0 + node
                node = trees.parent[template, node]
            cursor += depth + 1


@wp.kernel(enable_backward=False)
def _spawn_reduced_kernel(
    dyn: DynamicsData, trees: _TreeTemplates, bodies: _BodyRows, joints: _JointRows, slots: ReducedSlots
):
    """Lay each added reduced instance's template tree into its slot, one thread per lane."""
    tid = wp.tid()
    size = dyn.slot_size
    instance = tid // size
    lane = tid - instance * size
    if dyn.articulation_pool.state[instance] != _ADDED:
        return
    template = dyn.articulation_template[instance]
    lanes = dyn.template_tree_count[template]
    b0 = dyn.slot_body_start + instance * size
    j0 = dyn.slot_joint_start + instance * size
    body = b0 + lane
    joint = j0 + lane
    if lane == 0:
        _write_slot_topology(trees, slots, template, slots.articulation + instance, j0, b0, lanes, size)
    if lane >= lanes:
        # Spare lanes: fixed joints outside the joint range and the depth walk.
        _write_reduced_joint(
            joints,
            slots,
            joint,
            _FIXED_JOINT,
            wp.where(lane > 0, body - 1, -1),
            body,
            wp.transform_identity(),
            wp.transform_identity(),
            JointDesc(),
            -1,
        )
        return
    pose = dyn.articulation_pose[instance]
    velocity = dyn.articulation_velocity[instance]
    template_body = dyn.template_tree_body[template, lane]
    bdesc = _spawn_body(
        dyn.template_bodies[dyn.template_body_start[template] + template_body],
        pose,
        velocity,
        dyn.articulation_world[instance],
    )
    _write_body(bodies, body, bdesc)
    bodies.bodies.motion_type[body + 1] = _ARTICULATED
    slots.body_q[body] = bdesc.pose
    slots.body_qd[body] = bdesc.velocity
    slots.body_x_com[body] = wp.transform(bdesc.com, wp.quat_identity())

    local_joint = dyn.template_tree_joint[template, lane]
    if local_joint < 0:  # floating root: a free joint at the root body's pose
        free = JointDesc()
        free.linear_count = 3
        free.angular_count = 3
        free.linear_axes = wp.identity(3, dtype=wp.float32)
        free.angular_axes = wp.identity(3, dtype=wp.float32)
        free.limit_lower = vec6f(-MAXVAL)
        free.limit_upper = vec6f(MAXVAL)
        _write_reduced_joint(
            joints, slots, joint, _FREE_JOINT, -1, body, wp.transform_identity(), wp.transform_identity(), free, -1
        )
        q = slots.joint_q_start[joint]
        position = wp.transform_get_translation(bdesc.pose)
        rotation = wp.transform_get_rotation(bdesc.pose)
        for k in range(3):
            slots.joint_q[q + k] = position[k]
        for k in range(4):
            slots.joint_q[q + 3 + k] = rotation[k]
        qd = joints.qd_start[joint]
        v = wp.spatial_top(bdesc.velocity)
        w = wp.spatial_bottom(bdesc.velocity)
        for k in range(3):
            slots.joint_qd[qd + k] = v[k]
            slots.joint_qd[qd + 3 + k] = w[k]
    else:
        tj = dyn.template_joint_start[template] + local_joint
        jdesc = dyn.template_joints[tj]
        parent = wp.int32(-1)
        parent_xform = pose * jdesc.parent_xform
        if jdesc.parent >= 0:
            parent = dyn.articulation_bodies[instance, jdesc.parent]
            parent_xform = jdesc.parent_xform
        _write_reduced_joint(
            joints,
            slots,
            joint,
            slots.template_joint_type[tj],
            parent,
            body,
            parent_xform,
            jdesc.child_xform,
            jdesc,
            tj,
        )


@wp.kernel(enable_backward=False)
def _spawn_built_kernel(dyn: DynamicsData, bodies: _BodyRows, joints: _JointRows, slots: ReducedSlots):
    """Pose each added instance that is built with its template's tree, one thread per lane."""
    tid = wp.tid()
    size = dyn.articulation_bodies.shape[1]
    instance = tid // size
    lane = tid - instance * size
    if dyn.articulation_pool.state[instance] != _ADDED:
        return
    template = dyn.articulation_template[instance]
    lanes = dyn.template_tree_count[template]
    if lane >= lanes:
        return
    body = dyn.instance_body_start[instance] + lane
    joint = dyn.instance_joint_start[instance] + lane
    pose = dyn.articulation_pose[instance]
    bdesc = _spawn_body(
        dyn.template_bodies[dyn.template_body_start[template] + dyn.template_tree_body[template, lane]],
        pose,
        dyn.articulation_velocity[instance],
        dyn.articulation_world[instance],
    )
    _write_body(bodies, body, bdesc)
    bodies.bodies.motion_type[body + 1] = _ARTICULATED
    slots.body_q[body] = bdesc.pose
    slots.body_qd[body] = bdesc.velocity
    # No contact between trees outlives the instance that held it.
    reduced = bodies.bodies.reduced
    reduced.carried_wrench[body] = wp.spatial_vector()
    reduced.deferred_wrench[body] = wp.spatial_vector()
    reduced.biased_wrench[body] = wp.spatial_vector()

    q = slots.joint_q_start[joint]
    qd = joints.qd_start[joint]
    for k in range(joints.qd_start[joint + 1] - qd):
        slots.joint_qd[qd + k] = 0.0
        reduced.deferred_rate[qd + k] = 0.0
    local_joint = dyn.template_tree_joint[template, lane]
    if local_joint < 0:  # floating root: the free joint holds the root body's pose and motion
        position = wp.transform_get_translation(bdesc.pose)
        rotation = wp.transform_get_rotation(bdesc.pose)
        v = wp.spatial_top(bdesc.velocity)
        w = wp.spatial_bottom(bdesc.velocity)
        for k in range(3):
            slots.joint_q[q + k] = position[k]
            slots.joint_qd[qd + k] = v[k]
            slots.joint_qd[qd + 3 + k] = w[k]
        for k in range(4):
            slots.joint_q[q + 3 + k] = rotation[k]
    else:
        tj = dyn.template_joint_start[template] + local_joint
        for k in range(slots.joint_q_start[joint + 1] - q):
            slots.joint_q[q + k] = slots.template_joint_q[tj, k]
        if lane == 0:  # anchored to the world at the spawn pose
            joints.X_p[joint] = pose * dyn.template_joints[tj].parent_xform
    if lane == 0:
        a = slots.articulation + instance
        slots.articulation_end[a] = slots.articulation_start[a] + lanes
        reduced.deferred_pending[a] = 0


@wp.kernel(enable_backward=False)
def _template_free_lists_kernel(dyn: DynamicsData, free_prefix: wp.array[wp.int32]):
    """Find each template's free instances in the pool's free list, which is sorted by instance."""
    template = wp.tid()
    first = dyn.template_instance_start[template]
    start = dyn.articulation_pool.counts[1]
    if first < free_prefix.shape[0]:
        start = free_prefix[first]
    dyn.template_free_start[template] = start
    if template < dyn.template_free_next.shape[0]:
        dyn.template_free_next[template] = 0


@wp.func
def _empty_slot(slots: ReducedSlots, joint_type: wp.array[wp.int32], instance: wp.int32):
    """Detach a slot from the reduced solver: no joints, no depth lanes, a fixed root."""
    a = slots.articulation + instance
    start = slots.articulation_start[a]
    slots.articulation_end[a] = start
    joint_type[start] = _FIXED_JOINT  # keeps the linear momentum boost away from the empty range
    base = slots.depth_start[a, 0]
    for d in range(slots.depth_start.shape[1]):
        slots.depth_start[a, d] = base


@wp.kernel(enable_backward=False)
def _empty_slots_kernel(slots: ReducedSlots, joint_type: wp.array[wp.int32]):
    _empty_slot(slots, joint_type, wp.tid())


@wp.kernel(enable_backward=False)
def _commit_articulations_kernel(
    dyn: DynamicsData,
    slots: ReducedSlots,
    joint_type: wp.array[wp.int32],
    qd_start: wp.array[wp.int32],
    bodies: BodyContainer,
    active: wp.array[wp.int32],
    free: wp.array[wp.int32],
):
    slot = wp.tid()
    old = _resolve(dyn.articulation_pool, slot, active, free)
    if dyn.built_instances != 0:
        if old == _REMOVED:
            # An articulation without joints: the solver leaves its bodies where they are.
            a = slots.articulation + slot
            first = slots.articulation_start[a]
            for dof in range(qd_start[first], qd_start[slots.articulation_end[a]]):
                slots.joint_qd[dof] = 0.0
            for lane in range(slots.articulation_end[a] - first):
                body = dyn.instance_body_start[slot] + lane
                slots.body_qd[body] = wp.spatial_vector()
                bodies.velocity[body + 1] = wp.vec3(0.0)
                bodies.angular_velocity[body + 1] = wp.vec3(0.0)
            slots.articulation_end[a] = first
    elif dyn.slot_size > 0:
        if old == _REMOVED:
            _empty_slot(slots, joint_type, slot)
        # Leaving quarantine: no contact of the slot's last occupant can be solved anymore.
        slots.clear[slot] = wp.where(old == _QUARANTINE, 1, 0)


@wp.kernel(enable_backward=False)
def _clear_slot_contact_cache_kernel(slots: ReducedSlots, size: wp.int32):
    """Forget the cached contact rows of slots that left quarantine, before their reuse."""
    tid = wp.tid()
    slot = tid // slots.cache_rows
    row = tid - slot * slots.cache_rows
    if slots.clear[slot] == 0:
        return
    dofs = 6 * size
    first = slots.dof_start + slot * dofs
    for dof in range(first, first + dofs):
        slots.packed_jacobian[row, dof] = 0.0
        slots.packed_response[row, dof] = 0.0
    cache_row = (slots.articulation + slot) * slots.cache_rows + row
    slots.previous_row_body[cache_row] = -1
    slots.previous_row_body_pair[cache_row] = -1


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


def _bind(struct, source, names, prefix: str = ""):
    """Point the struct fields ``names`` at the arrays ``source.<prefix><name>``."""
    for name in names:
        setattr(struct, name, getattr(source, prefix + name))
    return struct


class Dynamics:
    """Runtime creation state of a :class:`~newton.solvers.SolverPhoenX`, exposed as ``solver.dynamics``.

    Pass :attr:`data` to kernels that call the ``add_*`` and ``remove_*`` functions.
    """

    def __init__(self, solver, collision_pipeline, friction_combine_mode: int):
        model = solver.model
        capacity = model._dynamic_capacity
        device = model.device
        self.capacity = capacity
        self.bodies = _HostPool(capacity.body_start, capacity.body_count, device)
        self.shapes = _HostPool(capacity.shape_start, capacity.shape_count, device)
        self.joints = _HostPool(capacity.joint_start, capacity.joint_count, device)
        self.articulations = _HostPool(0, capacity.articulation_count, device)
        self._sleeping = solver._sleeping_enabled
        self._solver = solver
        world = solver.world
        self._world = world

        d = DynamicsData()
        d.body_pool = self.bodies.data
        d.shape_pool = self.shapes.data
        d.joint_pool = self.joints.data
        d.articulation_pool = self.articulations.data
        d.error_count = wp.zeros(1, dtype=wp.int32, device=device)
        d.world_count = wp.int32(max(1, int(model.world_count)))
        d.body_count = wp.int32(model.body_count)
        d.body_world = model.body_world
        d.shape_type = model.shape_type
        d.staged_bodies = wp.empty(max(1, capacity.body_count), dtype=BodyDesc, device=device)
        d.staged_shapes = wp.empty(max(1, capacity.shape_count), dtype=ShapeDesc, device=device)
        d.staged_shape_body = wp.zeros(max(1, capacity.shape_count), dtype=wp.int32, device=device)
        d.staged_joints = wp.empty(max(1, capacity.joint_count), dtype=JointDesc, device=device)
        d.joint_cid = solver._joint_constraints.joint_idx_to_cid
        d.constraint_wrench = wp.zeros(max(1, world.num_constraints), dtype=wp.spatial_vector, device=device)

        body_rows = _bind(_BodyRows(), model, ("mass", "inv_mass", "inertia", "inv_inertia", "com", "flags"), "body_")
        body_rows.bodies = solver.bodies
        body_rows.fresh = solver._body_fresh
        self._body_rows = body_rows

        shape_rows = _bind(
            _ShapeRows(),
            model,
            (
                "type",
                "transform",
                "scale",
                "body",
                "world",
                "flags",
                "collision_group",
                "margin",
                "gap",
                "is_solid",
                "collision_radius",
                "source_ptr",
                "heightfield_index",
            ),
            "shape_",
        )
        _bind(
            shape_rows,
            model,
            (
                "interior_point",
                "voxel_resolution",
                "support_data",
                "sdf_index",
                "edge_range",
                "mesh_properties",
                "mesh_boundary_start",
            ),
            "_shape_",
        )
        shape_rows.body_world = model.body_world
        shape_rows.mu = model.shape_material_mu
        shape_rows.restitution = model.shape_material_restitution
        shape_rows.aabb_lower = model.shape_collision_aabb_lower
        shape_rows.aabb_upper = model.shape_collision_aabb_upper
        shape_rows.phoenx_body = solver._shape_body
        shape_rows.no_response = world.shape_no_response
        shape_rows.materials = world._materials
        shape_rows.friction_combine_mode = wp.int32(friction_combine_mode)
        shape_rows.world_map_dirty = collision_pipeline._dynamic_world_map_dirty
        self._shape_rows = shape_rows

        joint_rows = _JointRows()
        if capacity.joint_count or capacity.slot_size or capacity.instance_counts:
            _bind(
                joint_rows,
                model,
                (
                    "type",
                    "parent",
                    "child",
                    "X_p",
                    "X_c",
                    "dof_dim",
                    "enabled",
                    "qd_start",
                    "target_q_start",
                    "axis",
                    "limit_lower",
                    "limit_upper",
                    "velocity_limit",
                    "friction",
                    "armature",
                    "damping",
                    "target_mode",
                    "target_ke",
                    "target_kd",
                    "effort_limit",
                    "gear",
                ),
                "joint_",
            )
        self._joint_rows = joint_rows
        block_rows = _BlockRows()
        if capacity.joint_count:
            _bind(
                block_rows,
                solver._direct_equality_system,
                (
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
                ),
            )
            block_rows.joint_cid = d.joint_cid
            block_rows.constraints = world.constraints
            block_rows.pgs_enabled = world._joint_pgs_enabled
            block_rows.body_filter_parent = solver._body_filter_parent
        self._block_rows = block_rows

        self._init_articulations(d, capacity.articulation_templates, device)
        self.data = d

        # Prototypes only carry geometry; they never collide or render.
        prototypes = list(capacity.prototype_shapes)
        if prototypes:
            flags = model.shape_flags.numpy()
            flags[prototypes] &= ~(int(_COLLIDE_FLAGS) | (1 << 0))  # also clear VISIBLE
            model.shape_flags.assign(flags)
            shape_rows.world_map_dirty.fill_(1)

    def _init_articulations(self, d: DynamicsData, templates, device) -> None:
        bodies, shapes, shape_body, joints = [], [], [], []
        starts = ([0], [0], [0])
        for template in templates:
            for b in template["bodies"]:
                desc = BodyDesc()
                desc.pose = b["pose"]
                desc.mass = b["mass"]
                desc.com = b["com"]
                desc.inertia = b["inertia"]
                desc.flags = b["flags"]
                bodies.append(desc)
            for s in template["shapes"]:
                desc = ShapeDesc()
                for name in ("type", "transform", "scale", "margin", "gap", "mu", "restitution", "flags", "prototype"):
                    setattr(desc, name, s[name])
                shapes.append(desc)
                shape_body.append(s["body"])
            for j in template["joints"]:
                n_lin, n_ang = j["linear_count"], j["angular_count"]
                desc = JointDesc()
                for name in ("parent", "child", "parent_xform", "child_xform", "linear_count", "angular_count"):
                    setattr(desc, name, j[name])
                axes = [np.asarray(a, dtype=np.float32) for a in j["axes"]]
                desc.linear_axes = wp.mat33(
                    *np.vstack(axes[:n_lin] + [np.eye(3, dtype=np.float32)[k] for k in range(n_lin, 3)]).ravel()
                )
                desc.angular_axes = wp.mat33(
                    *np.vstack(axes[n_lin:] + [np.eye(3, dtype=np.float32)[k] for k in range(n_ang, 3)]).ravel()
                )
                pad = [0.0] * (6 - n_lin - n_ang)
                desc.limit_lower = vec6f(*j["limit_lower"], *pad)
                desc.limit_upper = vec6f(*j["limit_upper"], *pad)
                for name in ("velocity_limit", "friction", "armature", "damping", "target_ke", "target_kd"):
                    setattr(desc, name, vec6f(*j[name], *pad))
                desc.target_mode = vec6i(*j["target_mode"], *[0] * len(pad))
                joints.append(desc)
            for start, items in zip(starts, (bodies, shapes, joints), strict=True):
                start.append(len(items))
        d.template_body_start, d.template_shape_start, d.template_joint_start = (
            wp.array(start, dtype=wp.int32, device=device) for start in starts
        )
        d.template_bodies = wp.array(bodies or [BodyDesc()], dtype=BodyDesc, device=device)
        d.template_shapes = wp.array(shapes or [ShapeDesc()], dtype=ShapeDesc, device=device)
        d.template_shape_body = wp.array(shape_body or [0], dtype=wp.int32, device=device)
        d.template_joints = wp.array(joints or [JointDesc()], dtype=JointDesc, device=device)
        count = max(1, self.articulations.capacity)
        max_bodies = max([len(t["bodies"]) for t in templates] + [1])
        max_joints = max([len(t["joints"]) for t in templates] + [1])
        max_shapes = max([len(t["shapes"]) for t in templates] + [1])
        d.articulation_template = wp.full(count, -1, dtype=wp.int32, device=device)
        d.articulation_bodies = wp.full((count, max_bodies), -1, dtype=wp.int32, device=device)
        d.articulation_joints = wp.full((count, max_joints), -1, dtype=wp.int32, device=device)
        d.articulation_pose = wp.zeros(count, dtype=wp.transform, device=device)
        d.articulation_velocity = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        d.articulation_world = wp.zeros(count, dtype=wp.int32, device=device)
        d.instance_shapes = wp.full((count, max_shapes), -1, dtype=wp.int32, device=device)

        # Exact joint trees, when the solver brings a runtime projector.
        projector = getattr(self._solver, "_runtime_tree_projector", None)
        lanes = (len(templates), 32)
        tree_count, floating, max_depth = (np.zeros(len(templates), dtype=np.int32) for _ in range(3))
        tree_body, tree_joint, depth, parent, child_index = (np.full(lanes, -1, dtype=np.int32) for _ in range(5))
        child_start = np.zeros((len(templates), 33), dtype=np.int32)
        for t, template in enumerate(templates):
            tree = template.get("tree")
            reduced = self.capacity.slot_size or self.capacity.instance_counts
            if (projector is None and not reduced) or tree is None or not template["joints"]:
                continue
            n = len(tree["body"])
            tree_count[t] = n
            floating[t] = int(tree["floating"])
            max_depth[t] = max(tree["depth"])
            tree_body[t, :n] = tree["body"]
            tree_joint[t, :n] = tree["joint"]
            depth[t, :n] = tree["depth"]
            parent[t, :n] = tree["parent"]
            child_start[t, : n + 1] = tree["child_start"]
            child_start[t, n + 1 :] = tree["child_start"][-1]
            child_index[t, : len(tree["child_index"])] = tree["child_index"]
        rows = max(1, len(templates))
        pad = lambda a: np.concatenate([a, np.zeros((rows - len(a), *a.shape[1:]), a.dtype)])  # noqa: E731
        array = lambda a: wp.array(pad(a), dtype=wp.int32, device=device)  # noqa: E731
        d.template_tree_count, d.template_tree_body, d.template_tree_joint = (
            array(a) for a in (tree_count, tree_body, tree_joint)
        )
        trees = _TreeTemplates()
        trees.floating, trees.max_depth, trees.depth, trees.parent, trees.child_start, trees.child_index = (
            array(a) for a in (floating, max_depth, depth, parent, child_start, child_index)
        )
        self._trees = trees
        self._tree = projector.data if projector is not None else None
        d.tree_body_count = (
            projector.data.body_count if projector is not None else wp.zeros(0, dtype=wp.int32, device=device)
        )
        d.body_articulation = wp.full(max(1, self.bodies.capacity), -1, dtype=wp.int32, device=device)
        d.joint_articulation = wp.full(max(1, self.joints.capacity), -1, dtype=wp.int32, device=device)
        d.slot_size = wp.int32(self.capacity.slot_size)
        d.slot_body_start = wp.int32(self.capacity.slot_body_start if self.capacity.slot_size else 0)
        d.slot_joint_start = wp.int32(self.capacity.slot_joint_start if self.capacity.slot_size else 0)
        # First rows of each reduced instance: equal slots, or the trees built per template.
        counts = self.capacity.instance_counts
        first = np.arange(count, dtype=np.int32) * self.capacity.slot_size
        body_start, joint_start = self.capacity.slot_body_start + first, self.capacity.slot_joint_start + first
        if counts:
            sizes = np.repeat([len(t["bodies"]) for t in templates], counts)
            first = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
            body_start = self.capacity.instance_body_start + first
            joint_start = self.capacity.instance_joint_start + first
        d.built_instances = wp.int32(1 if counts else 0)
        d.instance_body_start = wp.array(body_start[:count], dtype=wp.int32, device=device)
        d.instance_joint_start = wp.array(joint_start[:count], dtype=wp.int32, device=device)
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        d.template_instance_start = wp.array(starts, dtype=wp.int32, device=device)
        d.template_free_start = wp.array(starts, dtype=wp.int32, device=device)  # every instance starts free
        d.template_free_next = wp.zeros(max(1, len(counts)), dtype=wp.int32, device=device)
        self._slots = self._init_slots(templates)

    def _init_slots(self, templates) -> ReducedSlots:
        """Bind reduced instance slots to the reduced solver's tables and empty every slot."""
        capacity = self.capacity
        device = self._solver.model.device
        slots = ReducedSlots()
        self._clear_slot_caches = False
        if not capacity.slot_size and not capacity.instance_counts:
            return slots
        reduced = self._solver._reduced_articulation
        system, data = reduced.system, reduced.bodies.reduced
        starts = reduced.model.articulation_start.numpy()[: reduced.model.articulation_count]
        first_joint = capacity.slot_joint_start if capacity.slot_size else capacity.instance_joint_start
        first = np.flatnonzero(starts == first_joint)
        if first.size != 1:
            raise RuntimeError("reduced articulation slots were not found in the reduced solver")
        slots.articulation = int(first[0])
        slots.articulation_start = data.articulation_start
        slots.articulation_end = data.articulation_end
        if capacity.instance_counts:
            # Built instances keep their tables; spawning only poses them.
            _bind(slots, reduced.state, ("joint_q", "joint_qd", "body_q", "body_qd"))
            slots.joint_q_start = reduced.model.joint_q_start
            joint_q = np.zeros((max(1, sum(len(t["joints"]) for t in templates)), 7), np.float32)
            for i, j in enumerate(j for t in templates for j in t["joints"]):
                joint_q[i, : len(j["q"])] = j["q"]
            slots.template_joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
            ends = data.articulation_end.numpy()
            ends[slots.articulation : slots.articulation + capacity.articulation_count] = starts[slots.articulation :][
                : capacity.articulation_count
            ]
            data.articulation_end.assign(ends)
            return slots
        slots.depth_start = system.advance_articulation_depth_start
        slots.depth_joint = system.advance_articulation_depth_joint
        slots.parent_lane = system.advance_joint_parent_lane
        slots.child_start = system.factor_child_start
        slots.child_joint = system.factor_child_joint
        slots.child_depth_index = system.advance_child_depth_index
        slots.body_path_start = data.body_path_start
        slots.body_path_joint = data.body_path_joint
        _bind(slots, system, ("body_x_com", "joint_s", "joint_s_publish"))
        _bind(slots, reduced.state, ("joint_q", "joint_qd", "body_q", "body_qd"))
        _bind(slots, reduced.model, ("joint_q_start", "joint_limit_ke", "joint_limit_kd"))
        joints = [j for t in templates for j in t["joints"]]
        rows = max(1, len(joints))
        joint_type, joint_q = np.zeros(rows, np.int32), np.zeros((rows, 7), np.float32)
        limit_ke, limit_kd = np.zeros((rows, 6), np.float32), np.zeros((rows, 6), np.float32)
        for i, j in enumerate(joints):
            joint_type[i] = j["type"]
            joint_q[i, : len(j["q"])] = j["q"]
            limit_ke[i, : len(j["limit_ke"])] = j["limit_ke"]
            limit_kd[i, : len(j["limit_kd"])] = j["limit_kd"]
        slots.template_joint_type = wp.array(joint_type, dtype=wp.int32, device=device)
        slots.template_joint_q = wp.array(joint_q, dtype=wp.float32, device=device)
        slots.template_joint_limit_ke = wp.array(limit_ke, dtype=wp.float32, device=device)
        slots.template_joint_limit_kd = wp.array(limit_kd, dtype=wp.float32, device=device)
        slots.clear = wp.zeros(capacity.articulation_count, dtype=wp.int32, device=device)
        slots.dof_start = int(reduced.model.joint_qd_start.numpy()[capacity.slot_joint_start])
        contacts = reduced.contact_block_system
        if contacts.packed_jacobian is not None:
            self._clear_slot_caches = True
            slots.packed_jacobian = contacts.packed_jacobian
            slots.packed_response = contacts.packed_response
            slots.previous_row_body = contacts.packed_previous_row_body
            slots.previous_row_body_pair = contacts.packed_previous_row_body_pair
            slots.cache_rows = contacts.packed_previous_row_body.shape[0] // contacts.articulation_count
        wp.launch(
            _empty_slots_kernel,
            dim=capacity.articulation_count,
            inputs=[slots, self._joint_rows.type],
            device=device,
        )
        return slots

    def update_joint_wrenches(self) -> None:
        """Refresh the joint reactions read by :func:`joint_wrench`; call after a step, capturable."""
        self._world.gather_constraint_wrenches(self.data.constraint_wrench)

    @property
    def error_count(self) -> wp.array[wp.int32]:
        """Device counter of failed ``add_*`` and ``remove_*`` requests since construction."""
        return self.data.error_count

    def _commit(self) -> None:
        """Write staged additions and apply removals at the start of a step, one thread per slot."""
        d = self.data
        device = d.error_count.device
        count = self.articulations.capacity
        if count:
            # Instances are laid out while still marked added; the state commit follows.
            if self.capacity.instance_counts:
                wp.launch(
                    _spawn_built_kernel,
                    dim=count * d.articulation_bodies.shape[1],
                    inputs=[d, self._body_rows, self._joint_rows, self._slots],
                    device=device,
                )
            elif self.capacity.slot_size:
                wp.launch(
                    _spawn_reduced_kernel,
                    dim=count * self.capacity.slot_size,
                    inputs=[d, self._trees, self._body_rows, self._joint_rows, self._slots],
                    device=device,
                )
            elif self._tree is not None:
                wp.launch(
                    _layout_trees_kernel,
                    dim=count * (_TREE_LANES + 1),
                    inputs=[d, self._trees, self._tree],
                    device=device,
                )
            wp.launch(
                _commit_articulations_kernel,
                dim=count,
                inputs=[
                    d,
                    self._slots,
                    self._joint_rows.type,
                    self._joint_rows.qd_start,
                    self._body_rows.bodies,
                    self.articulations.active,
                    self.articulations.free,
                ],
                device=device,
            )
            if self._clear_slot_caches:
                wp.launch(
                    _clear_slot_contact_cache_kernel,
                    dim=count * self._slots.cache_rows,
                    inputs=[self._slots, d.slot_size],
                    device=device,
                )
            self.articulations.rebuild_lists()
            if self.capacity.instance_counts:
                wp.launch(
                    _template_free_lists_kernel,
                    dim=d.template_free_start.shape[0],
                    inputs=[d, self.articulations.free_prefix],
                    device=device,
                )
        # Cascades read the removed bodies before their commit quarantines them.
        if self.shapes.capacity:
            wp.launch(
                _cascade_shapes_kernel, dim=self.shapes.capacity, inputs=[d, self._shape_rows.body], device=device
            )
        if self.joints.capacity:
            wp.launch(
                _cascade_joints_kernel,
                dim=self.joints.capacity,
                inputs=[d, self._joint_rows.parent, self._joint_rows.child],
                device=device,
            )
            wp.launch(
                _commit_joints_kernel,
                dim=self.joints.capacity,
                inputs=[d, self._joint_rows, self._block_rows, self.joints.active, self.joints.free],
                device=device,
            )
            self.joints.rebuild_lists()
        if self.bodies.capacity:
            wp.launch(
                _commit_bodies_kernel,
                dim=self.bodies.capacity,
                inputs=[d, self._body_rows, self.bodies.active, self.bodies.free],
                device=device,
            )
            if self._sleeping:
                bodies = self._body_rows.bodies
                wp.launch(
                    _wake_orphaned_islands_kernel,
                    dim=bodies.island_root.shape[0],
                    inputs=[d.body_pool, bodies],
                    device=device,
                )
            self.bodies.rebuild_lists()
        if self.shapes.capacity:
            wp.launch(
                _commit_shapes_kernel,
                dim=self.shapes.capacity,
                inputs=[d, self._shape_rows, self.shapes.active, self.shapes.free],
                device=device,
            )
            self.shapes.rebuild_lists()
