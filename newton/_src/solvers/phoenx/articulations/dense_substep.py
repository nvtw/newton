# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fused substep of dense reduced trees whose contacts are all private.

When every contact of a reduced articulation touches static or kinematic
geometry, a substep needs nothing from other articulations, so one warp per
articulation can run it end to end. :class:`DenseSubstep` launches two kernels
per substep and dense bucket:

* the biased kernel forms the kinematics, motion subspaces, spatial inertias,
  and implicit joint diagonal at the current coordinates, factors
  ``M = L^T L`` into shared memory, applies the joint, external, and gravity
  forces, solves the contacts with bias, applies the velocity-product step,
  and integrates and publishes the new pose;
* the relax kernel repeats the factorization at the integrated pose, solves
  the contacts without bias, and publishes the corrected velocities.

Each lane owns one joint and its child body. Tree recurrences run by depth
levels on shared memory, and subtree sums read the shared per-body values
directly, so no pass chains global loads level after level.

Contacts are solved in contact space. Each lane owns one row ``r`` with
Jacobian ``J_r``, and ``z_r = L^-T J_r^T`` is supported on the root path of the
row's body, so ``A = J M^-1 J^T`` has entries ``z_r . z_s`` over the common
prefix of two root paths. The projected Gauss-Seidel sweep updates every row
velocity with one column of ``A`` per impulse, and the generalized velocity
change ``M^-1 J^T lambda = L^-1 sum_r lambda_r z_r`` is one triangular solve.
Points beyond one page of rows are solved page after page, each page seeing
the velocity changes of the previous ones.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import warp as wp

from newton._src.sim import JointTargetMode, JointType
from newton._src.solvers.featherstone.kernels import (
    integrate_body_pose_from_com_twist,
    jcalc_integrate,
    jcalc_motion,
    jcalc_transform,
    spatial_cross,
    spatial_cross_dual,
)
from newton._src.solvers.phoenx.articulations.dense_tree import (
    _INERTIA_STRIDE,
    _LANES,
    _MOTION_STRIDE,
    _load,
    _load_motion,
    _load_u8,
    _load_u16,
    _make_shared_base,
    _shared_handle,
    _store,
    _store_u8,
    _unrolled_module,
    dense_unroll,
    make_dense_tree_ops,
)
from newton._src.solvers.phoenx.articulations.reduced import (
    _MIDPOINT_MAX_ITERATIONS,
    _boost_root_coordinates,
    _carry_floating_root_twist,
    _explicit_step_drift_terms,
    _explicit_step_drifts,
    _floating_root_frame,
    _joint_dissipation,
    _midpoint_momentum_step,
    _MidpointTree,
    _MidpointWorkspace,
    _pack_transformed_body_inertia,
    _restore_incoming_rates,
    _shuffle_reduced_spatial,
    _sum_reduced_group_float,
    _sum_reduced_group_vec4,
    _sym_mat66,
    _transport_root_twist,
    _unpack_symmetric_mat66,
    _vec6,
    _vec8,
    initialize_reduced_joint,
    initialize_reduced_rates,
    saturate_joint_drive,
)
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _CACHED_PAGE_COUNT,
    _POINTS_PER_PAGE,
    _combine_shared_external_kernel,
    _gather_packed_contact_points,
    _gather_packed_patch_points,
    _patch_normal_row_update,
    _patch_tangent_rows_update,
    _solve_reduced_contact_point,
    _store_point_row_effective_mass,
    _sync_contact_warp,
)
from newton._src.solvers.phoenx.body import EXTERNAL_SHARED, BodyContainer, DenseTreeData, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_reduced_soft_coefficients,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)

CHUNK_POINTS = 20
"""Contact points one chunk solves at once with point friction.

A point's lane projects it, and each lane forms the rows ``lane`` and ``lane + 32`` of the
chunk's three rows per point. Two flat feet with a carried object on them make up to
eighteen points: in smaller chunks such robots formed every chunk's rows on every round, and
set the kernels' time.
"""

PATCH_CHUNK_POINTS = 10
"""Contact points one chunk solves at once with patch friction, whose rows take one lane each."""
_FULL_MASK = wp.constant(wp.uint32(0xFFFFFFFF))
_BLOCKS_PER_SM = 11
"""Most blocks per SM the fused kernels are compiled for; more add little, see PERF_NOTES."""

_SM_SHARED_BYTES = 84 * 1024
"""Shared memory an SM gives the blocks of one kernel (measured on an RTX PRO 6000 Blackwell)."""


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
"""
)
def _lane_float(value: wp.float32, source: wp.int32) -> wp.float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
"""
)
def _lane_int(value: wp.int32, source: wp.int32) -> wp.int32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __reduce_add_sync(0xffffffffu, value);
#else
    return value;
#endif
"""
)
def _warp_sum_int(value: wp.int32) -> wp.int32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __any_sync(0xffffffffu, value) != 0;
#else
    return value;
#endif
"""
)
def _warp_any(value: wp.bool) -> wp.bool: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
#endif
    return value;
"""
)
def _warp_sum_float(value: wp.float32) -> wp.float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
#else
    return 0;
#endif
"""
)
def _block_index() -> wp.int32:
    """Return the block of a lean launch, which the compiler knows is uniform across the block."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return threadIdx.x;
#else
    return 0;
#endif
"""
)
def _lane_index() -> wp.int32: ...


@wp.func
def _store_spatial(shared: _shared_handle, index: wp.int32, value: wp.spatial_vector):
    for component in range(6):
        _store(shared, index + wp.int32(component), value[component])


@wp.func
def _store_transform(shared: _shared_handle, index: wp.int32, value: wp.transform):
    translation = wp.transform_get_translation(value)
    rotation = wp.transform_get_rotation(value)
    for component in range(3):
        _store(shared, index + wp.int32(component), translation[component])
    for component in range(4):
        _store(shared, index + wp.int32(3 + component), rotation[component])


@wp.func
def _load_transform(shared: _shared_handle, index: wp.int32) -> wp.transform:
    return wp.transform(
        wp.vec3(_load(shared, index), _load(shared, index + wp.int32(1)), _load(shared, index + wp.int32(2))),
        wp.quat(
            _load(shared, index + wp.int32(3)),
            _load(shared, index + wp.int32(4)),
            _load(shared, index + wp.int32(5)),
            _load(shared, index + wp.int32(6)),
        ),
    )


@wp.struct
class _TreeJoint:
    """One lane's joint, its child body, and their model data; ``joint < 0`` marks an idle lane."""

    joint: wp.int32
    kind: wp.int32
    parent: wp.int32
    depth: wp.int32
    child: wp.int32
    q_start: wp.int32
    qd_start: wp.int32
    dof_count: wp.int32
    lin: wp.int32
    ang: wp.int32
    x_p: wp.transform
    x_c: wp.transform
    com: wp.vec3
    mass: wp.float32


@wp.func
def _load_tree_joint(
    lane: wp.int32,
    joint_start: wp.int32,
    joint_count: wp.int32,
    joint_base: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    joint_parent_local: wp.array[wp.int32],
    joint_depth: wp.array[wp.int32],
) -> _TreeJoint:
    tree = _TreeJoint()
    tree.joint = wp.int32(-1)
    tree.parent = wp.int32(-1)
    tree.depth = wp.int32(-1)
    tree.child = wp.int32(0)
    if lane < joint_count:
        joint = joint_start + lane
        child = joint_child[joint]
        tree.joint = joint
        tree.kind = joint_type[joint]
        tree.parent = joint_parent_local[joint_base + lane]
        tree.depth = joint_depth[joint_base + lane]
        tree.child = child
        tree.q_start = joint_q_start[joint]
        tree.qd_start = joint_qd_start[joint]
        tree.dof_count = joint_qd_start[joint + wp.int32(1)] - tree.qd_start
        tree.lin = joint_dof_dim[joint, 0]
        tree.ang = joint_dof_dim[joint, 1]
        tree.x_p = joint_x_p[joint]
        tree.x_c = joint_x_c[joint]
        tree.com = body_com[child]
        tree.mass = body_mass[child]
    return tree


@wp.func
def warm_patch_impulses(
    cc: ContactContainer,
    packed_articulation: wp.int32,
    point_total: wp.int32,
    point_contact: wp.array2d[wp.int32],
    point_column: wp.array2d[wp.int32],
    patch_lambda: wp.array2d[wp.vec2],
):
    """Set each patch's tangent impulse to the sum of its points' accumulated impulses."""
    point = wp.int32(0)
    ordinal = wp.int32(0)
    while point < point_total:
        patch_column = point_column[packed_articulation, point]
        lambda_t = wp.vec2(0.0)
        while point < point_total and point_column[packed_articulation, point] == patch_column:
            contact = point_contact[packed_articulation, point]
            lambda_t += wp.vec2(cc_get_tangent1_lambda(cc, contact), cc_get_tangent2_lambda(cc, contact))
            point += wp.int32(1)
        patch_lambda[packed_articulation, ordinal] = lambda_t
        ordinal += wp.int32(1)


@wp.func
def _root_path_sum(value: wp.spatial_vector, parent: wp.int32, max_depth: wp.int32) -> wp.spatial_vector:
    """Return the sum of ``value`` over the lane's joint and its ancestors.

    Lanes are joints and ``parent`` is the parent joint's lane, negative at a
    root. Each round adds the sum held by the lane's current ancestor and then
    doubles the distance to that ancestor, so the sums along paths of up to
    ``max_depth`` edges complete in ``log2(max_depth)`` rounds of shuffles.
    """
    total = value
    ancestor = parent
    reach = wp.int32(1)
    while reach <= max_depth:
        source = wp.max(ancestor, wp.int32(0))
        above = _shuffle_reduced_spatial(total, source, wp.int32(_LANES), _FULL_MASK)
        next_ancestor = _lane_int(ancestor, source)
        total += wp.where(ancestor >= wp.int32(0), above, wp.spatial_vector())
        ancestor = wp.where(ancestor >= wp.int32(0), next_ancestor, ancestor)
        reach += reach
    return total


@wp.func
def _in_first_tree(parent: wp.int32, lane: wp.int32, max_depth: wp.int32) -> wp.bool:
    """Return whether a lane's joint hangs below the block's first root, lane 0."""
    root = lane
    ancestor = parent
    for _level in range(max_depth + wp.int32(1)):
        above = _lane_int(parent, wp.max(ancestor, wp.int32(0)))
        root = wp.where(ancestor >= wp.int32(0), ancestor, root)
        ancestor = wp.where(ancestor >= wp.int32(0), above, ancestor)
    return root == wp.int32(0)


#: Columns of the packed per-DOF joint parameters, see _pack_dof_parameters_kernel.
_DOF_PARAMETERS = 10


@wp.kernel(enable_backward=False)
def _pack_dof_parameters_kernel(
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_limit_ke: wp.array[wp.float32],
    joint_limit_kd: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    parameters: wp.array2d[wp.float32],
):
    """Copy the drive, limit, damping, armature, and effort limit parameters of each DOF into one row.

    The fused kernels read them for every DOF in every substep; from ten separate
    arrays that is ten cache lines per DOF, from one row it is one.
    """
    dof = wp.tid()
    parameters[dof, 0] = joint_target_ke[dof]
    parameters[dof, 1] = joint_target_kd[dof]
    parameters[dof, 2] = wp.float32(joint_target_mode[dof])
    parameters[dof, 3] = joint_limit_lower[dof]
    parameters[dof, 4] = joint_limit_upper[dof]
    parameters[dof, 5] = joint_limit_ke[dof]
    parameters[dof, 6] = joint_limit_kd[dof]
    parameters[dof, 7] = joint_damping[dof]
    parameters[dof, 8] = joint_armature[dof]
    parameters[dof, 9] = joint_effort_limit[dof]


@wp.func
def _configure_implicit_joint_dynamics_packed(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
    params: wp.array2d[wp.float32],
    dt: wp.float32,
    factor_diagonal: wp.array[wp.float32],
    implicit_force: wp.array[wp.float32],
):
    """Build one joint's implicit-Euler operator diagonal and right-hand side from packed parameters.

    The arithmetic of :func:`configure_implicit_joint_dynamics`, reading the rows of
    :func:`_pack_dof_parameters_kernel`.
    """
    dof_start = joint_qd_start[joint]
    dof_end = joint_qd_start[joint + wp.int32(1)]
    coord_start = joint_q_start[joint]
    target_start = joint_target_q_start[joint]
    kind = joint_type[joint]
    for dof in range(dof_start, dof_end):
        diagonal = params[dof, 8]
        force = wp.float32(0.0)
        if kind == JointType.REVOLUTE or kind == JointType.PRISMATIC or kind == JointType.D6:
            local = dof - dof_start
            q = joint_q[coord_start + local]
            qd = joint_qd[dof]
            stiffness = params[dof, 0]
            drive_damping = params[dof, 1]
            target_position = joint_target_q[target_start + local]
            target_velocity = joint_target_qd[dof]
            target_mode = wp.int32(params[dof, 2])
            if target_mode == JointTargetMode.NONE or target_mode == JointTargetMode.EFFORT:
                stiffness = wp.float32(0.0)
                drive_damping = wp.float32(0.0)
                target_velocity = wp.float32(0.0)
            elif target_mode == JointTargetMode.POSITION:
                target_velocity = wp.float32(0.0)
            elif target_mode == JointTargetMode.VELOCITY:
                stiffness = wp.float32(0.0)
            stiffness, drive_damping, drive_effort = saturate_joint_drive(
                stiffness, drive_damping, target_position, target_velocity, q, qd, dt, params[dof, 9]
            )
            predicted_q = q + dt * qd
            lower = params[dof, 3]
            upper = params[dof, 4]
            if q < lower or predicted_q < lower:
                stiffness = params[dof, 5]
                drive_damping = params[dof, 6]
                target_position = lower
                target_velocity = wp.float32(0.0)
                drive_effort = wp.float32(0.0)
            elif q > upper or predicted_q > upper:
                stiffness = params[dof, 5]
                drive_damping = params[dof, 6]
                target_position = upper
                target_velocity = wp.float32(0.0)
                drive_effort = wp.float32(0.0)
            passive_damping = params[dof, 7]
            damping = drive_damping + passive_damping
            diagonal += dt * damping + dt * dt * stiffness
            force = (
                drive_effort
                + stiffness * (target_position - q)
                + drive_damping * (target_velocity - qd)
                - passive_damping * qd
                - dt * stiffness * qd
            )
        factor_diagonal[dof] = diagonal
        implicit_force[dof] = force


@wp.func
def _is_floating(kind: wp.int32) -> wp.bool:
    return kind == JointType.FREE or kind == JointType.DISTANCE


@wp.func
def _cholesky3(matrix: wp.mat33) -> wp.mat33:
    """Return the lower factor ``C`` of a symmetric positive semidefinite ``C C^T``."""
    tiny = wp.float32(1.0e-30)
    c00 = wp.sqrt(wp.max(matrix[0, 0], wp.float32(0.0)))
    c10 = matrix[1, 0] / wp.max(c00, tiny)
    c20 = matrix[2, 0] / wp.max(c00, tiny)
    c11 = wp.sqrt(wp.max(matrix[1, 1] - c10 * c10, wp.float32(0.0)))
    c21 = (matrix[2, 1] - c20 * c10) / wp.max(c11, tiny)
    c22 = wp.sqrt(wp.max(matrix[2, 2] - c20 * c20 - c21 * c21, wp.float32(0.0)))
    return wp.mat33(c00, 0.0, 0.0, c10, c11, 0.0, c20, c21, c22)


@functools.cache
def make_dense_substep_kernel(
    max_dofs: int,
    joint_capacity: int,
    entry_capacity: int,
    path_capacity: int,
    subtree_capacity: int,
    patch: bool,
    relax: bool,
    external: bool = False,
    pairs: bool = False,
    forest: bool = False,
    coupled: bool = False,
):
    """Build the biased or the relax substep kernel of one dense bucket.

    Args:
        max_dofs: DOF capacity of the bucket.
        joint_capacity: Joint capacity of the bucket, at most one joint per lane.
        entry_capacity: Factor entries of the bucket's largest template.
        path_capacity: Longest root-to-DOF path of the bucket's templates.
        subtree_capacity: Longest subtree list of the bucket's templates.
        patch: Solve patch friction rows instead of point friction rows.
        relax: Build the relax kernel instead of the biased one.
        external: Solve contact rows against free rigid bodies, whose six velocity
            components join the rows' DOFs.
        pairs: Solve contact rows between two links of the tree, whose second root
            path takes a second block of path positions.
        forest: Run blocks that hold several trees: a floating-base tree first, if any,
            then trees on fixed roots. Frames stay about the first root's COM.
    """
    # Scratch after the factor's tables: the composite inertias while factoring, then the tree
    # recurrences and the contact pages share one region; the rates and a solve vector follow.
    tree_words = 3 * _MOTION_STRIDE * _LANES
    # A chunk's rows: their vectors z by path position, the DOFs of those positions as bytes,
    # their start velocities, z . z, and the points' normal loads. Lane l forms rows l and
    # l + 32. The solve vector ends with a spare DOF that the padding of short paths addresses.
    # A row against a free rigid body carries that body's six whitened components after its path.
    chunk_capacity = PATCH_CHUNK_POINTS if patch else CHUNK_POINTS
    page_rows = 3 * chunk_capacity
    row_slots = (page_rows + _LANES - 1) // _LANES
    ext_position = path_capacity * (2 if pairs else 1)
    z_positions = ext_position + (6 if external else 0)
    # Each free body of a chunk takes six components, which lane l keeps at l + 32 g.
    external_dofs = 6 * chunk_capacity if external else 0
    external_groups = (external_dofs + _LANES - 1) // _LANES
    lane_vector = wp.types.vector(length=2 + external_groups, dtype=wp.float32)
    lane_indices = wp.types.vector(length=2 + external_groups, dtype=wp.int32)
    # The DOF bytes of z: a point's three rows act on the same links and share them, while
    # patch rows each take their own.
    byte_rows = page_rows if patch else chunk_capacity
    row_words = page_rows * z_positions
    byte_words = (byte_rows * z_positions + 3) // 4
    contact_words = row_words + byte_words + 2 * page_rows + chunk_capacity
    qd_offset = max(tree_words, contact_words, _INERTIA_STRIDE * joint_capacity)
    scratch_words = qd_offset + 2 * max_dofs + 1 + external_dofs
    ops = make_dense_tree_ops(max_dofs, joint_capacity, entry_capacity, subtree_capacity, scratch_words)
    layout = ops.layout
    shared_base = _make_shared_base(layout.size, layout.tables)
    stage = ops.stage
    factor = ops.factor
    restore = ops.restore
    solve_mass = ops.solve
    solve_lower = ops.solve_lower
    # The link frames are formed before and after the recurrences on the three spatial arrays.
    pose_base = layout.scratch
    first_base = layout.scratch
    second_base = first_base + _MOTION_STRIDE * _LANES
    third_base = second_base + _MOTION_STRIDE * _LANES
    z_base = layout.scratch
    z_dof_byte = 4 * (z_base + row_words)
    start_velocity_base = z_base + row_words + byte_words
    row_mass_base = start_velocity_base + page_rows
    point_load_base = row_mass_base + page_rows
    gather = _gather_packed_patch_points if patch else _gather_packed_contact_points
    qd_base = layout.scratch + qd_offset
    rhs_base = qd_base + max_dofs
    motion_base = layout.motion
    dof_entry = layout.dof_entry
    path = layout.path
    subtree_start = layout.subtree_start
    subtree_joint = layout.subtree_joint
    joint_last_dof = layout.joint_last_dof
    module = _unrolled_module(
        f"dense_substep_{max_dofs}_{joint_capacity}_{entry_capacity}_{path_capacity}_{subtree_capacity}"
        f"_{int(patch)}_{int(relax)}_{int(external)}_{int(pairs)}_{int(forest)}_{int(coupled)}",
        dense_unroll(layout, joint_capacity),
    )

    @wp.func
    def local_kinematics(
        shared: _shared_handle,
        tree: _TreeJoint,
        lane: wp.int32,
        articulation: wp.int32,
        max_depth: wp.int32,
        joint_transform: wp.transform,
        floating_world: wp.transform,
        floating_child: wp.bool,
        joint_q: wp.array[wp.float32],
        body_x_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
    ):
        """Form the link frames about the root COM from the joint transforms and publish them.

        Matches the local kinematics of :func:`_finish_and_publish_reduced_warp_device`.
        ``floating_child`` marks a floating joint below the root whose child was
        integrated to the world pose ``floating_world``; its joint transform
        follows from its parent's frame and is stored in the coordinates.
        Returns the child's link frame, the world position of the root COM, and the
        joint's parent anchor.
        """
        world = wp.transform_identity()
        if tree.joint >= wp.int32(0) and tree.parent < wp.int32(0):
            world = tree.x_p * joint_transform * wp.transform_inverse(tree.x_c)
        origin = wp.transform_point(world, tree.com)
        origin = wp.vec3(_lane_float(origin[0], 0), _lane_float(origin[1], 0), _lane_float(origin[2], 0))
        # A root's frame and anchor follow from its own pose.
        rotation = wp.transform_get_rotation(world)
        root_local = wp.transform(-wp.quat_rotate(rotation, tree.com), rotation)
        if wp.static(forest):
            # A further root sits where its joint puts it, relative to the first root's COM.
            if lane > wp.int32(0):
                root_local = wp.transform(wp.transform_get_translation(world) - origin, rotation)
        root_transform = joint_transform
        if _is_floating(tree.kind):
            root_transform = wp.transform(wp.vec3(0.0), wp.transform_get_rotation(joint_transform))
        root_anchor = root_local * tree.x_c * wp.transform_inverse(root_transform)
        is_root = tree.parent < wp.int32(0)
        parent_slot = wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * wp.max(tree.parent, wp.int32(0))
        local = root_local
        anchor = root_anchor
        relative = joint_transform
        child_anchor_inverse = wp.transform_inverse(tree.x_c)
        # Every lane chains from its parent's stored frame at every level: a joint at depth d
        # is final after d levels and repeats that value, so no lane branches on its depth.
        for _level in range(max_depth + wp.int32(1)):
            chained_anchor = _load_transform(shared, parent_slot) * tree.x_p
            if floating_child:
                anchor_world = wp.transform(
                    origin + wp.transform_get_translation(chained_anchor), wp.transform_get_rotation(chained_anchor)
                )
                relative = wp.transform_inverse(anchor_world) * floating_world * tree.x_c
            anchor = wp.where(is_root, root_anchor, chained_anchor)
            local = wp.where(is_root, root_local, chained_anchor * relative * child_anchor_inverse)
            _store_transform(shared, wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * lane, local)
            _sync_contact_warp()
        if tree.joint >= wp.int32(0):
            if floating_child:
                position = wp.transform_get_translation(relative)
                orientation = wp.transform_get_rotation(relative)
                for component in range(3):
                    joint_q[tree.q_start + component] = position[component]
                for component in range(4):
                    joint_q[tree.q_start + 3 + component] = orientation[component]
            joint_anchor_local[tree.joint] = anchor
            body_q_local[tree.child] = local
            body_q_com[tree.child] = local * body_x_com[tree.child]
        if lane == wp.int32(0):
            articulation_origin[articulation] = origin
        return local, origin, anchor

    @wp.func
    def joint_twist(shared: _shared_handle, tree: _TreeJoint, dof_start: wp.int32) -> wp.spatial_vector:
        """Return ``sum S qd`` over the joint's DOFs from the shared motion subspaces and rates."""
        twist = wp.spatial_vector()
        for offset in range(tree.dof_count):
            local = tree.qd_start + offset - dof_start
            twist += _load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * local) * _load(
                shared, wp.int32(qd_base) + local
            )
        return twist

    @wp.func
    def solve_forces(
        shared: _shared_handle,
        tree: _TreeJoint,
        lane: wp.int32,
        dof_start: wp.int32,
        force_base: wp.int32,
        applied: wp.array[wp.float32],
        implicit_force: wp.array[wp.float32],
        n: wp.int32,
    ):
        """Solve ``M qdd = tau - S^T F`` into the shared right-hand side.

        ``F`` sums each subtree's shared body bias forces at ``force_base``; ``tau``
        holds ``applied`` for jointed DOFs and ``implicit_force`` for driven ones.
        """
        if tree.joint >= wp.int32(0):
            force = wp.spatial_vector()
            for index in range(
                _load_u16(shared, wp.int32(subtree_start) + lane),
                _load_u16(shared, wp.int32(subtree_start) + lane + wp.int32(1)),
            ):
                force += _load_motion(
                    shared, force_base + wp.int32(_MOTION_STRIDE) * _load_u8(shared, wp.int32(subtree_joint) + index)
                )
            driven = tree.kind == JointType.REVOLUTE or tree.kind == JointType.PRISMATIC or tree.kind == JointType.D6
            for offset in range(tree.dof_count):
                dof = tree.qd_start + offset
                local = dof - dof_start
                value = -wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * local), force)
                if not _is_floating(tree.kind):
                    value += applied[dof]
                if driven:
                    value += implicit_force[dof]
                _store(shared, wp.int32(rhs_base) + local, value)
        _sync_contact_warp()
        solve_mass(shared, wp.int32(rhs_base), n, lane)

    @wp.func
    def dof_byte(position: wp.int32, row: wp.int32) -> wp.int32:
        """Return the shared byte address of the DOF of row ``row``'s ``z`` at ``position``."""
        byte_row = row
        if wp.static(not patch):
            byte_row = row // wp.int32(3)
        return wp.int32(z_dof_byte) + wp.int32(byte_rows) * position + byte_row

    @wp.func
    def row_entries(shared: _shared_handle, row: wp.int32, positions: Any, dofs: Any):
        """Return ``z_row`` at a lane's DOFs ``dofs``, zero where a DOF is off the row's path.

        A DOF sits at the same position, its depth, of every row through it, so the
        row's DOF byte at that position tells whether the row passes it.
        """
        entries = lane_vector()
        index = wp.int32(page_rows) * positions[0] + row
        if _load_u8(shared, dof_byte(positions[0], row)) == dofs[0]:
            entries[0] = _load(shared, wp.int32(z_base) + index)
        if wp.static(pairs):
            # The second link's path may pass the DOF too; both parts add.
            index += wp.int32(page_rows * path_capacity)
            if _load_u8(shared, dof_byte(positions[0] + wp.int32(path_capacity), row)) == dofs[0]:
                entries[0] += _load(shared, wp.int32(z_base) + index)
        if wp.static(max_dofs > _LANES):
            index = wp.int32(page_rows) * positions[1] + row
            if _load_u8(shared, dof_byte(positions[1], row)) == dofs[1]:
                entries[1] = _load(shared, wp.int32(z_base) + index)
            if wp.static(pairs):
                index += wp.int32(page_rows * path_capacity)
                if _load_u8(shared, dof_byte(positions[1] + wp.int32(path_capacity), row)) == dofs[1]:
                    entries[1] += _load(shared, wp.int32(z_base) + index)
        for group in range(wp.static(external_groups)):
            index = wp.int32(page_rows) * positions[2 + group] + row
            if _load_u8(shared, dof_byte(positions[2 + group], row)) == dofs[2 + group]:
                entries[2 + group] = _load(shared, wp.int32(z_base) + index)
        return entries

    @wp.func
    def point_entries(shared: _shared_handle, point: wp.int32, positions: Any, dofs: Any):
        """Return :func:`row_entries` of a point's normal and tangent rows, which share DOF bytes."""
        row = wp.int32(3) * point
        normal_entries = lane_vector()
        tangent0_entries = lane_vector()
        tangent1_entries = lane_vector()
        for entry in range(wp.static(2 + external_groups)):
            if wp.static(entry == 1 and max_dofs <= _LANES):
                continue
            for part in range(wp.static(2 if pairs and entry < 2 else 1)):
                position = positions[entry] + wp.int32(part * path_capacity)
                if _load_u8(shared, dof_byte(position, row)) == dofs[entry]:
                    index = wp.int32(z_base) + wp.int32(page_rows) * position + row
                    normal_entries[entry] += _load(shared, index)
                    tangent0_entries[entry] += _load(shared, index + wp.int32(1))
                    tangent1_entries[entry] += _load(shared, index + wp.int32(2))
        return normal_entries, tangent0_entries, tangent1_entries

    @wp.func
    def row_product(shared: _shared_handle, lane: wp.int32, row: wp.int32, other: wp.int32) -> wp.float32:
        """Return ``z_row . z_other`` like :func:`row_dot`; root paths share DOFs on a common prefix only."""
        total = wp.float32(0.0)
        position = lane
        while position < wp.int32(z_positions):
            if _load_u8(shared, dof_byte(position, row)) == _load_u8(shared, dof_byte(position, other)):
                total += _load(shared, wp.int32(z_base) + wp.int32(page_rows) * position + row) * _load(
                    shared, wp.int32(z_base) + wp.int32(page_rows) * position + other
                )
            position += wp.int32(_LANES)
        if wp.static(pairs):
            # One row's first path against the other row's second path, and the reverse.
            position = lane
            while position < wp.int32(path_capacity):
                second = position + wp.int32(path_capacity)
                if _load_u8(shared, dof_byte(position, row)) == _load_u8(shared, dof_byte(second, other)):
                    total += _load(shared, wp.int32(z_base) + wp.int32(page_rows) * position + row) * _load(
                        shared, wp.int32(z_base) + wp.int32(page_rows) * second + other
                    )
                if _load_u8(shared, dof_byte(second, row)) == _load_u8(shared, dof_byte(position, other)):
                    total += _load(shared, wp.int32(z_base) + wp.int32(page_rows) * second + row) * _load(
                        shared, wp.int32(z_base) + wp.int32(page_rows) * position + other
                    )
                position += wp.int32(_LANES)
        return _warp_sum_float(total)

    @wp.func
    def row_path(
        shared: _shared_handle,
        row: wp.int32,
        joint_start: wp.int32,
        bodies: BodyContainer,
        body: wp.int32,
        wrench: wp.spatial_vector,
        offset: wp.int32,
    ) -> wp.vec2:
        """Form row ``row``'s ``z = L^-T J^T`` for a wrench on ``body`` at the path positions from ``offset``.

        Returns ``(J qd, z . z)``. ``J`` runs along the root path of the body; ``z``
        solves leaves first and pads the positions beyond the path with zeros on a DOF
        no lane holds.
        """
        velocity = wp.float32(0.0)
        inverse_mass = wp.float32(0.0)
        leaf = _load_u8(shared, wp.int32(joint_last_dof) + bodies.reduced.body_joint[body] - joint_start) - 1
        path_length = wp.int32(0)
        if leaf >= wp.int32(0):
            path_entry = _load_u16(shared, wp.int32(dof_entry) + leaf)
            path_length = _load_u16(shared, wp.int32(dof_entry) + leaf + wp.int32(1)) - path_entry
            for position in range(path_length):
                k = _load_u8(shared, wp.int32(path) + path_entry + position)
                value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * k), wrench)
                velocity += value * _load(shared, wp.int32(qd_base) + k)
                _store(shared, wp.int32(z_base) + wp.int32(page_rows) * (offset + position) + row, value)
                _store_u8(shared, dof_byte(offset + position, row), k)
            for reverse in range(path_length):
                position = path_length - wp.int32(1) - reverse
                value = _load(shared, wp.int32(z_base) + wp.int32(page_rows) * (offset + position) + row)
                for later in range(position + wp.int32(1), path_length):
                    i = _load_u8(shared, wp.int32(path) + path_entry + later)
                    value -= _load(shared, _load_u16(shared, wp.int32(dof_entry) + i) + position) * _load(
                        shared, wp.int32(z_base) + wp.int32(page_rows) * (offset + later) + row
                    )
                k = _load_u8(shared, wp.int32(path) + path_entry + position)
                value /= _load(shared, _load_u16(shared, wp.int32(dof_entry) + k + wp.int32(1)) - wp.int32(1))
                _store(shared, wp.int32(z_base) + wp.int32(page_rows) * (offset + position) + row, value)
                inverse_mass += value * value
        for position in range(path_length, wp.int32(path_capacity)):
            _store(shared, wp.int32(z_base) + wp.int32(page_rows) * (offset + position) + row, wp.float32(0.0))
            # The padding of the two paths must not match each other.
            _store_u8(
                shared,
                dof_byte(offset + position, row),
                wp.where(offset > wp.int32(0), wp.int32(max_dofs + 1), wp.int32(max_dofs)),
            )
        return wp.vec2(velocity, inverse_mass)

    @wp.func
    def packed_row(chunk: wp.vec4i, point_total: wp.int32, local: wp.int32) -> wp.int32:
        """Return the gathered row of a chunk's local row ``local``."""
        if wp.static(patch):
            chunk_points = chunk[1] - chunk[0]
            if local < chunk_points:
                return chunk[0] + local
            return point_total + wp.int32(2) * chunk[2] + local - chunk_points
        return wp.int32(3) * chunk[0] + local

    @wp.func
    def solve_chunk(
        shared: _shared_handle,
        articulation: wp.int32,
        lane: wp.int32,
        page: wp.int32,
        chunk: wp.vec4i,
        joint_start: wp.int32,
        dof_start: wp.int32,
        n: wp.int32,
        bodies: BodyContainer,
        columns: ContactColumnContainer,
        cc: ContactContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        first_sweep: wp.int32,
        sweeps: wp.int32,
        use_bias: wp.bool,
        warm_start: wp.bool,
        prepare: wp.bool,
        point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        point0: wp.array2d[wp.vec3],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_body_pair: wp.array2d[wp.int32],
        row_wrench_pair: wp.array2d[wp.spatial_vector],
        row_velocity: wp.array2d[wp.float32],
        patch_lambda: wp.array2d[wp.vec2],
        normal_load: wp.array2d[wp.float32],
        joint_qd: wp.array[wp.float32],
        point_coupling: wp.array2d[wp.vec3],
        first_visit: wp.bool,
    ):
        """Sweep one chunk of gathered contact rows and add its velocity change to the shared and global rates.

        A chunk holds the points ``chunk[0]`` to ``chunk[1]`` and, with patch friction,
        the columns ``chunk[2]`` to ``chunk[3]`` that start among them. Lane ``r`` forms
        local row ``r`` and lane ``q`` projects the chunk's point ``q``. The sweep keeps
        ``w = sum_r lambda_r z_r`` by DOF, so a row's velocity is its start velocity
        plus ``z_r . w`` and ``M^-1 J^T lambda = L^-1 w`` needs no matrix. Point friction
        rows are each point's normal and tangents; patch friction rows are one normal
        per point and two tangents per column, projected after the normals as in the
        articulated patch tile. ``warm_start`` applies the accumulated impulses first,
        which the following velocity change includes, and ``prepare`` marks the first
        sweep of the chunk's rows, which scales their biases once.
        """
        packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + wp.min(
            page, wp.int32(_CACHED_PAGE_COUNT - 1)
        )
        point_total = point_count[packed_articulation]
        chunk_points = chunk[1] - chunk[0]
        row_count = wp.int32(3) * chunk_points
        if wp.static(patch):
            row_count = chunk_points + wp.int32(2) * (chunk[3] - chunk[2])
        contact = wp.int32(0)
        column = wp.int32(0)
        point_normal = wp.vec3(0.0)
        point_tangent = wp.vec3(0.0)
        if lane < chunk_points:
            contact = point_contact[packed_articulation, chunk[0] + lane]
            column = point_column[packed_articulation, chunk[0] + lane]
            point_normal = normal[packed_articulation, chunk[0] + lane]
            point_tangent = tangent0[packed_articulation, chunk[0] + lane]
        # Patch friction sweeps without global memory: a point's lane keeps its normal row's
        # bias and impulse, and the lane of a column's first tangent row keeps the column's
        # points, mean tangent bias, friction, and impulse.
        bias_n = wp.float32(0.0)
        lambda_n = wp.float32(0.0)
        column_first = wp.int32(0)
        column_points = wp.int32(0)
        column_ordinal = wp.int32(0)
        column_owner = wp.bool(False)
        # The column and the application point of the lane's row, for rows against a free body.
        row_column = wp.int32(0)
        row_point = wp.vec3(0.0)
        bias_t = wp.vec2f(0.0, 0.0)
        friction = wp.float32(0.0)
        friction_dynamic = wp.float32(0.0)
        lambda_t = wp.vec2f(0.0, 0.0)
        if wp.static(patch):
            if lane < chunk_points:
                bias_n = cc_get_bias(cc, contact)
                lambda_n = cc_get_normal_lambda(cc, contact)
                row_column = column
                if wp.static(external):
                    row_point = point0[packed_articulation, chunk[0] + lane]
            elif lane < row_count and (wp.static(external) or ((lane - chunk_points) & wp.int32(1)) == wp.int32(0)):
                # With free bodies the second tangent lane walks its column too, for its point.
                column_owner = ((lane - chunk_points) & wp.int32(1)) == wp.int32(0)
                column_ordinal = chunk[2] + ((lane - chunk_points) >> wp.int32(1))
                # The chunk's columns start after the points of a column that began earlier.
                walk = chunk[0]
                while walk < chunk[1] and walk > wp.int32(0):
                    if point_column[packed_articulation, walk] != point_column[packed_articulation, walk - 1]:
                        break
                    walk += wp.int32(1)
                for _earlier in range(column_ordinal - chunk[2]):
                    earlier_column = point_column[packed_articulation, walk]
                    while walk < point_total and point_column[packed_articulation, walk] == earlier_column:
                        walk += wp.int32(1)
                column_first = walk
                patch_column = point_column[packed_articulation, walk]
                while walk < point_total and point_column[packed_articulation, walk] == patch_column:
                    if use_bias:
                        column_contact = point_contact[packed_articulation, walk]
                        bias_t[0] += cc_get_bias_t1(cc, column_contact)
                        bias_t[1] += cc_get_bias_t2(cc, column_contact)
                    if wp.static(external):
                        row_point += point0[packed_articulation, walk]
                    column_points += wp.int32(1)
                    walk += wp.int32(1)
                bias_t *= wp.float32(1.0) / wp.float32(column_points)
                row_point *= wp.float32(1.0) / wp.float32(column_points)
                row_column = patch_column
                friction = contact_get_friction(columns, patch_column)
                friction_dynamic = contact_get_friction_dynamic(columns, patch_column)
                lambda_t = patch_lambda[packed_articulation, column_ordinal]

        # The free rigid body that a lane's point touches, if this tree solves it, with its half
        # when two trees share it: with patch friction the body of the lane's row, which may be
        # a tangent row of a column that began in an earlier chunk. A chunk's free bodies take
        # the slot of the first such lane that touches them, and that lane applies the body's
        # velocity change.
        point_external = wp.int32(-1)
        point_half = wp.int32(-1)
        external_slot = wp.int32(-1)
        if wp.static(external):
            external_column = column
            external_lanes = chunk_points
            if wp.static(patch):
                external_column = row_column
                external_lanes = row_count
            if lane < external_lanes:
                other = contact_get_body1(columns, external_column)
                if bodies.reduced.body_articulation[other] >= wp.int32(0):
                    other = contact_get_body2(columns, external_column)
                owner = bodies.reduced.external_owner[other]
                if owner != wp.int32(-1):
                    point_external = other
                    if owner == EXTERNAL_SHARED:
                        # Two trees solve this body, each with half of its mass and its own
                        # copy of the velocity, which a later kernel averages.
                        tree = wp.max(
                            bodies.reduced.body_articulation[contact_get_body1(columns, external_column)],
                            bodies.reduced.body_articulation[contact_get_body2(columns, external_column)],
                        )
                        point_half = wp.int32(2) * other
                        if tree != bodies.reduced.external_lowest[other]:
                            point_half += wp.int32(1)
            for point_offset in range(chunk_capacity):
                candidate = _lane_int(point_external, wp.int32(point_offset))
                if external_slot < wp.int32(0) and point_external >= wp.int32(0) and candidate == point_external:
                    external_slot = wp.int32(point_offset)

        # Row r: J_r along the root path of its body, its velocity J_r qd plus the gathered
        # external part, then z_r = L^-T J_r^T leaves first, padded with zeros on the spare DOF.
        # Lane l forms the rows l and l + 32.
        velocity = wp.float32(0.0)
        inverse_mass = wp.float32(0.0)
        for row_slot in range(row_slots):
            row = lane + wp.int32(_LANES * row_slot)
            # The free body of the row's point and its slot.
            row_external = point_external
            row_half = point_half
            row_external_slot = external_slot
            if wp.static(external and not patch):
                source = wp.min(row // wp.int32(3), wp.int32(_LANES - 1))
                row_external = _lane_int(point_external, source)
                row_half = _lane_int(point_half, source)
                row_external_slot = _lane_int(external_slot, source)
            gathered_row = packed_row(chunk, point_total, row)
            if row < row_count:
                body = row_body[packed_articulation, gathered_row]
                wrench = row_wrench[packed_articulation, gathered_row]
                row_velocity_value = row_velocity[packed_articulation, gathered_row]
                first_path = row_path(shared, row, joint_start, bodies, body, wrench, wp.int32(0))
                row_velocity_value += first_path[0]
                row_inverse_mass = first_path[1]
                body_pair = wp.int32(-1)
                wrench_pair = wp.spatial_vector()
                if wp.static(pairs):
                    # A row between two links is the sum of one row per link; their z add on the
                    # DOFs both root paths share, which sit at equal positions.
                    body_pair = row_body_pair[packed_articulation, gathered_row]
                    wrench_pair = row_wrench_pair[packed_articulation, gathered_row]
                    if body_pair >= wp.int32(0):
                        second_path = row_path(
                            shared, row, joint_start, bodies, body_pair, wrench_pair, wp.int32(path_capacity)
                        )
                        row_velocity_value += second_path[0]
                        # z . z with the two parts added on shared DOFs before squaring: links that
                        # cannot move against each other then cancel to rounding squared, not to the
                        # rounding of the two parts' squares.
                        row_inverse_mass = wp.float32(0.0)
                        for position in range(path_capacity):
                            slot = wp.int32(page_rows * position) + row
                            first_value = _load(shared, wp.int32(z_base) + slot)
                            second_value = _load(shared, wp.int32(z_base) + slot + wp.int32(page_rows * path_capacity))
                            if _load_u8(shared, dof_byte(position, row)) == _load_u8(
                                shared, dof_byte(position + path_capacity, row)
                            ):
                                row_inverse_mass += (first_value + second_value) * (first_value + second_value)
                            else:
                                row_inverse_mass += first_value * first_value + second_value * second_value
                    else:
                        for position in range(path_capacity, 2 * path_capacity):
                            _store(shared, wp.int32(z_base) + wp.int32(page_rows * position) + row, wp.float32(0.0))
                            _store_u8(shared, dof_byte(position, row), wp.int32(max_dofs + 1))
                if wp.static(external):
                    # The other endpoint, if a free rigid body of this tree: the row's impulse acts
                    # on it reversed, and its mass is diagonal after whitening by the factor of its
                    # inverse inertia. Its components take the DOFs after the spare one.
                    if wp.static(not patch):
                        row_point = point0[packed_articulation, gathered_row // wp.int32(3)]
                    external_linear = wp.vec3(0.0)
                    external_angular = wp.vec3(0.0)
                    if row_external >= wp.int32(0):
                        direction = -wp.spatial_top(wrench)
                        moment = wp.cross(row_point - bodies.position[row_external], direction)
                        linear_velocity = bodies.velocity[row_external]
                        angular_velocity = bodies.angular_velocity[row_external]
                        whitening = wp.float32(1.0)
                        if row_half >= wp.int32(0):
                            linear_velocity += bodies.reduced.external_delta_linear[row_half]
                            angular_velocity += bodies.reduced.external_delta_angular[row_half]
                            whitening = wp.float32(1.4142135623730951)
                        row_velocity_value += wp.dot(direction, linear_velocity) + wp.dot(moment, angular_velocity)
                        external_linear = whitening * wp.sqrt(bodies.inverse_mass[row_external]) * direction
                        external_angular = whitening * (
                            wp.transpose(_cholesky3(mat33_from_sym6(bodies.inverse_inertia_world[row_external])))
                            * moment
                        )
                        row_inverse_mass += wp.dot(external_linear, external_linear) + wp.dot(
                            external_angular, external_angular
                        )
                    first_dof = wp.int32(max_dofs)
                    stride = wp.int32(0)
                    if row_external_slot >= wp.int32(0):
                        first_dof = wp.int32(max_dofs + 1) + wp.int32(6) * row_external_slot
                        stride = wp.int32(1)
                    for axis in range(3):
                        position = wp.int32(ext_position + axis)
                        _store(shared, wp.int32(z_base) + wp.int32(page_rows) * position + row, external_linear[axis])
                        _store_u8(shared, dof_byte(position, row), first_dof + stride * wp.int32(axis))
                        position += wp.int32(3)
                        _store(shared, wp.int32(z_base) + wp.int32(page_rows) * position + row, external_angular[axis])
                        _store_u8(shared, dof_byte(position, row), first_dof + stride * wp.int32(axis + 3))
                _store(shared, wp.int32(start_velocity_base) + row, row_velocity_value)
                _store(shared, wp.int32(row_mass_base) + row, row_inverse_mass)
                if row_slot == 0:
                    velocity = row_velocity_value
                    inverse_mass = row_inverse_mass
                if wp.static(not patch):
                    _store_point_row_effective_mass(
                        bodies,
                        cc,
                        point_contact,
                        packed_articulation,
                        gathered_row,
                        body,
                        wrench,
                        body_pair,
                        wrench_pair,
                        row_inverse_mass,
                        prepare,
                    )
        _sync_contact_warp()

        # Each lane keeps w = sum_r lambda_r z_r at its own DOFs in registers: lane l holds DOF l
        # and, in trees beyond a warp, DOF l + 32; with free bodies also their components
        # l + 32 g. A DOF never on a path keeps the impossible id -1.
        positions = lane_indices()
        dofs = lane_indices()
        for entry in range(2 + external_groups):
            dofs[entry] = wp.int32(-1)
        if lane < n:
            dofs[0] = lane
            positions[0] = (
                _load_u16(shared, wp.int32(dof_entry) + lane + wp.int32(1))
                - wp.int32(1)
                - _load_u16(shared, wp.int32(dof_entry) + lane)
            )
        if wp.static(max_dofs > _LANES):
            if lane + wp.int32(_LANES) < n:
                dofs[1] = lane + wp.int32(_LANES)
                positions[1] = (
                    _load_u16(shared, wp.int32(dof_entry) + lane + wp.int32(_LANES + 1))
                    - wp.int32(1)
                    - _load_u16(shared, wp.int32(dof_entry) + lane + wp.int32(_LANES))
                )
        for group in range(external_groups):
            component = lane + wp.int32(_LANES * group)
            if component < wp.int32(external_dofs):
                dofs[2 + group] = wp.int32(max_dofs + 1) + component
                positions[2 + group] = wp.int32(ext_position) + component - wp.int32(6) * (component // wp.int32(6))
        w = lane_vector()

        # Biased solves restart from the accumulated impulses, relax solves add to them.
        if warm_start:
            if wp.static(patch):
                lambda_row = wp.float32(0.0)
                if lane < chunk_points:
                    lambda_row = cc_get_normal_lambda(cc, contact)
                elif lane < row_count:
                    tangent = lane - chunk_points
                    lambda_row = patch_lambda[packed_articulation, chunk[2] + (tangent >> wp.int32(1))][
                        tangent & wp.int32(1)
                    ]
                for r in range(row_count):
                    w += _lane_float(lambda_row, r) * row_entries(shared, r, positions, dofs)
            else:
                # A point's lane holds its three impulses.
                point_lambda = wp.vec3(0.0)
                if lane < chunk_points:
                    point_lambda = wp.vec3(
                        cc_get_normal_lambda(cc, contact),
                        cc_get_tangent1_lambda(cc, contact),
                        cc_get_tangent2_lambda(cc, contact),
                    )
                for point in range(chunk_points):
                    entries_n, entries_t0, entries_t1 = point_entries(shared, point, positions, dofs)
                    w += _lane_float(point_lambda[0], point) * entries_n
                    w += _lane_float(point_lambda[1], point) * entries_t0
                    w += _lane_float(point_lambda[2], point) * entries_t1
        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if use_bias:
            _bias_rate, mass_coeff, impulse_coeff = cc_reduced_soft_coefficients(cc, idt)
        # The products between a point's rows, or a column's tangent rows, hold for every sweep;
        # the lane that projects them keeps them.
        coupling = wp.vec3(0.0)
        if wp.static(patch):
            if sweeps > wp.int32(0):
                for column_offset in range(chunk[3] - chunk[2]):
                    row = chunk_points + wp.int32(2) * wp.int32(column_offset)
                    product = row_product(shared, lane, row, row + wp.int32(1))
                    if lane == row:
                        coupling[0] = product
        elif first_visit:
            # A chunk's first visit may only apply its impulses; its later visits read these.
            # A point's rows act on the same links and share their DOF bytes, so its lane sums
            # their products position by position. A DOF on both paths of a row between two
            # links adds the products of the two parts too.
            if lane < chunk_points:
                row = wp.int32(3) * lane
                for position in range(z_positions):
                    index = wp.int32(z_base + page_rows * position) + row
                    z_n = _load(shared, index)
                    z_t0 = _load(shared, index + wp.int32(1))
                    z_t1 = _load(shared, index + wp.int32(2))
                    coupling += wp.vec3(z_n * z_t0, z_n * z_t1, z_t0 * z_t1)
                if wp.static(pairs):
                    for position in range(path_capacity):
                        if _load_u8(shared, dof_byte(wp.int32(position), row)) == _load_u8(
                            shared, dof_byte(wp.int32(position + path_capacity), row)
                        ):
                            first = wp.int32(z_base + page_rows * position) + row
                            second = first + wp.int32(page_rows * path_capacity)
                            n_first = _load(shared, first)
                            t0_first = _load(shared, first + wp.int32(1))
                            t1_first = _load(shared, first + wp.int32(2))
                            n_second = _load(shared, second)
                            t0_second = _load(shared, second + wp.int32(1))
                            t1_second = _load(shared, second + wp.int32(2))
                            coupling += wp.vec3(
                                n_first * t0_second + n_second * t0_first,
                                n_first * t1_second + n_second * t1_first,
                                t0_first * t1_second + t0_second * t1_first,
                            )
                point_coupling[packed_articulation, chunk[0] + lane] = coupling
        elif lane < chunk_points:
            # The rows are those of the chunk's first visit: so are their products.
            coupling = point_coupling[packed_articulation, chunk[0] + lane]
        for sweep in range(sweeps):
            iteration = first_sweep + sweep
            for point_offset in range(chunk_points):
                point = wp.int32(point_offset)
                if (iteration & wp.int32(1)) != wp.int32(0):
                    point = chunk_points - wp.int32(1) - wp.int32(point_offset)
                if wp.static(patch):
                    delta_n = wp.float32(0.0)
                    entries = row_entries(shared, point, positions, dofs)
                    change = _warp_sum_float(wp.dot(entries, w))
                    if lane == point:
                        update = _patch_normal_row_update(
                            bias_n,
                            lambda_n,
                            velocity + change,
                            inverse_mass,
                            use_bias,
                            idt,
                            mass_coeff,
                            impulse_coeff,
                            sor_boost,
                        )
                        delta_n = update[0]
                        lambda_n = update[1]
                        _store(shared, wp.int32(point_load_base) + point, update[2])
                        # Columns that continue in another chunk read the load from there.
                        normal_load[packed_articulation, chunk[0] + point] = update[2]
                    w += _lane_float(delta_n, point) * entries
                else:
                    row = wp.int32(3) * point
                    delta = wp.vec3(0.0)
                    entries_n, entries_t0, entries_t1 = point_entries(shared, point, positions, dofs)
                    change = wp.vec3(
                        _warp_sum_float(wp.dot(entries_n, w)),
                        _warp_sum_float(wp.dot(entries_t0, w)),
                        _warp_sum_float(wp.dot(entries_t1, w)),
                    )
                    if lane == point:
                        delta = _solve_reduced_contact_point(
                            columns,
                            cc,
                            contact,
                            column,
                            point_normal,
                            point_tangent,
                            wp.vec3(
                                _load(shared, wp.int32(start_velocity_base) + row) + change[0],
                                _load(shared, wp.int32(start_velocity_base) + row + wp.int32(1)) + change[1],
                                _load(shared, wp.int32(start_velocity_base) + row + wp.int32(2)) + change[2],
                            ),
                            coupling,
                            use_bias,
                            idt,
                            mass_coeff,
                            impulse_coeff,
                            sor_boost,
                        )
                    w += (
                        _lane_float(delta[0], point) * entries_n
                        + _lane_float(delta[1], point) * entries_t0
                        + _lane_float(delta[2], point) * entries_t1
                    )
            if wp.static(patch):
                # The chunk's columns' tangent rows, after the normals; they read the normal loads.
                for column_offset in range(chunk[3] - chunk[2]):
                    row = chunk_points + wp.int32(2) * wp.int32(column_offset)
                    delta_t = wp.vec2f(0.0, 0.0)
                    entries_t0 = row_entries(shared, row, positions, dofs)
                    entries_t1 = row_entries(shared, row + wp.int32(1), positions, dofs)
                    change_t = wp.vec2f(_warp_sum_float(wp.dot(entries_t0, w)), _warp_sum_float(wp.dot(entries_t1, w)))
                    if lane == row:
                        load_sum = wp.float32(0.0)
                        for offset in range(column_points):
                            column_point = column_first + wp.int32(offset)
                            if column_point < chunk[1]:
                                load_sum += _load(shared, wp.int32(point_load_base) + column_point - chunk[0])
                            else:
                                load_sum += normal_load[packed_articulation, column_point]
                        tangent_update = _patch_tangent_rows_update(
                            lambda_t,
                            wp.vec2f(
                                _load(shared, wp.int32(start_velocity_base) + row) + change_t[0],
                                _load(shared, wp.int32(start_velocity_base) + row + wp.int32(1)) + change_t[1],
                            ),
                            bias_t,
                            wp.vec3(
                                _load(shared, wp.int32(row_mass_base) + row),
                                coupling[0],
                                _load(shared, wp.int32(row_mass_base) + row + wp.int32(1)),
                            ),
                            load_sum,
                            friction,
                            friction_dynamic,
                            sor_boost,
                        )
                        delta_t = tangent_update.delta
                        lambda_t = tangent_update.lambda_new
                    w += _lane_float(delta_t[0], row) * entries_t0 + _lane_float(delta_t[1], row) * entries_t1
        if wp.static(patch):
            # Store the swept impulses; a patch shares its impulse evenly among its points.
            if sweeps > wp.int32(0):
                if lane < chunk_points:
                    cc_set_normal_lambda(cc, contact, lambda_n)
                elif column_owner:
                    patch_lambda[packed_articulation, column_ordinal] = lambda_t
                    point_lambda = lambda_t * (wp.float32(1.0) / wp.float32(column_points))
                    for offset in range(column_points):
                        column_contact = point_contact[packed_articulation, column_first + wp.int32(offset)]
                        cc_set_tangent1_lambda(cc, column_contact, point_lambda[0])
                        cc_set_tangent2_lambda(cc, column_contact, point_lambda[1])

        # M^-1 J^T lambda = L^-1 w.
        if dofs[0] >= wp.int32(0):
            _store(shared, wp.int32(rhs_base) + lane, w[0])
        if wp.static(max_dofs > _LANES):
            if dofs[1] >= wp.int32(0):
                _store(shared, wp.int32(rhs_base) + lane + wp.int32(_LANES), w[1])
        for group in range(external_groups):
            if dofs[2 + group] >= wp.int32(0):
                _store(shared, wp.int32(rhs_base + max_dofs + 1 + _LANES * group) + lane, w[2 + group])
        _sync_contact_warp()
        solve_lower(shared, wp.int32(rhs_base), n, lane)
        k = lane
        while k < n:
            rate = _load(shared, wp.int32(qd_base) + k) + _load(shared, wp.int32(rhs_base) + k)
            _store(shared, wp.int32(qd_base) + k, rate)
            joint_qd[dof_start + k] = rate
            k += wp.int32(_LANES)
        if wp.static(external):
            # The free bodies' velocity changes, unwhitened, by the lane that named each slot.
            if external_slot == lane:
                first = wp.int32(rhs_base + max_dofs + 1) + wp.int32(6) * external_slot
                delta_linear = wp.sqrt(bodies.inverse_mass[point_external]) * wp.vec3(
                    _load(shared, first), _load(shared, first + wp.int32(1)), _load(shared, first + wp.int32(2))
                )
                delta_angular = _cholesky3(mat33_from_sym6(bodies.inverse_inertia_world[point_external])) * wp.vec3(
                    _load(shared, first + wp.int32(3)),
                    _load(shared, first + wp.int32(4)),
                    _load(shared, first + wp.int32(5)),
                )
                if point_half >= wp.int32(0):
                    half_scale = wp.float32(1.4142135623730951)
                    bodies.reduced.external_delta_linear[point_half] += half_scale * delta_linear
                    bodies.reduced.external_delta_angular[point_half] += half_scale * delta_angular
                else:
                    bodies.velocity[point_external] += delta_linear
                    bodies.angular_velocity[point_external] += delta_angular
        _sync_contact_warp()

    @wp.func
    def publish_rates(
        shared: _shared_handle,
        tree: _TreeJoint,
        lane: wp.int32,
        dof_start: wp.int32,
        max_depth: wp.int32,
        x_com: wp.vec3,
        dt: wp.float32,
        update_configuration: wp.bool,
        integrated: wp.bool,
        root_twist: wp.spatial_vector,
        target_linear_momentum: wp.vec3,
        joint_axis: wp.array[wp.vec3],
        joint_q: wp.array[wp.float32],
        joint_qd: wp.array[wp.float32],
        joint_qd_public: wp.array[wp.float32],
        joint_x_p: wp.array[wp.transform],
        joint_qd_start: wp.array[wp.int32],
        motion_array: wp.array[wp.spatial_vector],
        bodies: BodyContainer,
    ):
        """Publish the public rates and link COM twists from the shared rates and motion subspaces.

        Matches the rate publication of :func:`_finish_and_publish_reduced_warp_device`;
        ``update_configuration`` re-references a floating root's twist, which the
        caller formed with the motion subspaces of the start pose, to the integrated
        pose and restores the tree's incoming linear momentum.
        """
        if tree.joint >= wp.int32(0):
            if not _is_floating(tree.kind):
                for offset in range(tree.dof_count):
                    dof = tree.qd_start + offset
                    joint_qd_public[dof] = _load(shared, wp.int32(qd_base) + dof - dof_start)
            elif tree.parent < wp.int32(0):
                if not integrated:
                    twist = joint_twist(shared, tree, dof_start)
                    if update_configuration:
                        twist = _carry_floating_root_twist(joint_qd, tree.qd_start, root_twist, dt)
                        for axis in range(3):
                            dof = tree.qd_start + axis
                            _store(shared, wp.int32(qd_base) + dof - dof_start, joint_qd[dof])
                    parent_rotation = wp.transform_get_rotation(tree.x_p)
                    velocity_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_top(twist))
                    omega_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_bottom(twist))
                    for axis in range(3):
                        joint_qd_public[tree.qd_start + axis] = velocity_parent[axis]
                        joint_qd_public[tree.qd_start + 3 + axis] = omega_parent[axis]
            else:
                joint_transform = jcalc_transform(
                    tree.kind, joint_axis, tree.qd_start, tree.lin, tree.ang, joint_q, tree.q_start
                )
                child_com_parent = wp.transform_point(joint_transform * wp.transform_inverse(tree.x_c), tree.com)
                local = tree.qd_start - dof_start
                velocity_origin = wp.vec3(
                    _load(shared, wp.int32(qd_base) + local),
                    _load(shared, wp.int32(qd_base) + local + wp.int32(1)),
                    _load(shared, wp.int32(qd_base) + local + wp.int32(2)),
                )
                omega_parent = wp.vec3(
                    _load(shared, wp.int32(qd_base) + local + wp.int32(3)),
                    _load(shared, wp.int32(qd_base) + local + wp.int32(4)),
                    _load(shared, wp.int32(qd_base) + local + wp.int32(5)),
                )
                velocity_com = velocity_origin + wp.cross(omega_parent, child_com_parent)
                for axis in range(3):
                    joint_qd_public[tree.qd_start + axis] = velocity_com[axis]
                    joint_qd_public[tree.qd_start + 3 + axis] = omega_parent[axis]
        _sync_contact_warp()
        twist = wp.spatial_vector()
        if tree.joint >= wp.int32(0):
            twist = joint_twist(shared, tree, dof_start)
        twist = _root_path_sum(twist, tree.parent, max_depth)
        omega = wp.spatial_bottom(twist)
        velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
        # A floating tree gets its incoming linear momentum back through a uniform boost,
        # see ReducedPhoenXArticulation.end_substep.
        if update_configuration and not integrated and _is_floating(_lane_int(tree.kind, 0)):
            # Only the floating tree, the block's first, carries momentum to restore.
            boosted = wp.bool(True)
            if wp.static(forest):
                boosted = _in_first_tree(tree.parent, lane, max_depth)
            local_momentum = wp.vec4(0.0)
            if tree.joint >= wp.int32(0) and boosted:
                local_momentum = wp.vec4(
                    tree.mass * velocity_com[0], tree.mass * velocity_com[1], tree.mass * velocity_com[2], tree.mass
                )
            total = _sum_reduced_group_vec4(local_momentum, wp.int32(_LANES), _FULL_MASK)
            if total[3] > wp.float32(0.0):
                boost = (target_linear_momentum - wp.vec3(total[0], total[1], total[2])) / total[3]
                if boosted:
                    velocity_com += boost
                if lane == wp.int32(0):
                    _boost_root_coordinates(
                        tree.joint, joint_qd_start, joint_x_p, motion_array, boost, joint_qd, joint_qd_public
                    )
        if tree.joint >= wp.int32(0):
            slot = tree.child + wp.int32(1)
            bodies.velocity[slot] = velocity_com
            bodies.angular_velocity[slot] = omega

    # The kernels' time is their rounds of blocks per SM: bound the registers so that as many
    # blocks fit an SM as its shared memory holds. The biased kernel's spills cost less than
    # the round they save.
    blocks_per_sm = max(1, min(_BLOCKS_PER_SM, _SM_SHARED_BYTES // (4 * layout.size)))

    @wp.kernel(enable_backward=False, module=module, grid_stride=False, launch_bounds=(_LANES, blocks_per_sm))
    def dense_substep_kernel(
        bucket_articulation: wp.array[wp.int32],
        dt: wp.float32,
        sor_boost: wp.float32,
        iterations: wp.int32,
        contacts_active: wp.bool,
        reuse_factor: wp.bool,
        reuse_rows: wp.bool,
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_target_q_start: wp.array[wp.int32],
        joint_x_p: wp.array[wp.transform],
        joint_x_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_dof_dim: wp.array2d[wp.int32],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[wp.float32],
        body_inertia: wp.array[wp.mat33],
        body_world: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        joint_f: wp.array[wp.float32],
        joint_target_q: wp.array[wp.float32],
        joint_target_qd: wp.array[wp.float32],
        joint_q: wp.array[wp.float32],
        joint_qd_public: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        body_f: wp.array[wp.spatial_vector],
        joint_qd: wp.array[wp.float32],
        factor_diagonal: wp.array[wp.float32],
        implicit_force: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        body_i_s: wp.array[_sym_mat66],
        motion: wp.array[wp.float32],
        body_inertia_flat: wp.array[wp.float32],
        body_x_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
        joint_s_publish: wp.array[wp.spatial_vector],
        zero_generalized_force: wp.array[wp.float32],
        body_velocity: wp.array[wp.spatial_vector],
        midpoint: _MidpointWorkspace,
        max_depth_all: wp.int32,
        depth_start: wp.array2d[wp.int32],
        depth_joint: wp.array[wp.int32],
        template_joint_start: wp.array[wp.int32],
        template_max_depth: wp.array[wp.int32],
        joint_parent_local: wp.array[wp.int32],
        joint_depth: wp.array[wp.int32],
        template_image: wp.array[wp.uint32],
        template_image_start: wp.array[wp.int32],
        dense: DenseTreeData,
        schedule_section_end: wp.array[wp.int32],
        schedule_columns: wp.array[wp.int32],
        columns: ContactColumnContainer,
        cc: ContactContainer,
        contacts: ContactViews,
        point_count: wp.array[wp.int32],
        row_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        point0: wp.array2d[wp.vec3],
        point1: wp.array2d[wp.vec3],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_body_pair: wp.array2d[wp.int32],
        row_wrench_pair: wp.array2d[wp.spatial_vector],
        row_velocity: wp.array2d[wp.float32],
        patch_lambda: wp.array2d[wp.vec2],
        normal_load: wp.array2d[wp.float32],
        point_coupling: wp.array2d[wp.vec3],
        dof_parameters: wp.array2d[wp.float32],
        bodies: BodyContainer,
    ):
        block = _block_index()
        lane = _lane_index()
        articulation = bucket_articulation[block]
        data = bodies.reduced
        joint_start = data.articulation_start[articulation]
        joint_end = data.articulation_end[articulation]
        joint_count = joint_end - joint_start
        if joint_count <= wp.int32(0):
            # A runtime instance that is not in the scene: its whole block leaves.
            return
        dof_start = joint_qd_start[joint_start]
        n = joint_qd_start[joint_end] - dof_start
        template = dense.articulation_template[articulation]
        joint_base = template_joint_start[template]
        max_depth = template_max_depth[template]
        shared = shared_base(template_image, template_image_start[template])
        idt = wp.float32(1.0) / dt
        heights = stage(shared, articulation, lane, dense, template_image, template_image_start)
        tree = _load_tree_joint(
            lane,
            joint_start,
            joint_count,
            joint_base,
            joint_type,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_x_p,
            joint_x_c,
            joint_dof_dim,
            body_com,
            body_mass,
            joint_parent_local,
            joint_depth,
        )

        # Frames, implicit joint operator, motion subspaces, and spatial inertias at the coordinates.
        # The biased step leaves the frames, motion subspaces, and spatial inertias of its
        # integrated pose, which the relax step and a following biased step at that pose reuse.
        frames_current = reuse_factor or wp.static(relax)
        local = wp.transform_identity()
        origin = wp.vec3(0.0)
        if frames_current:
            if tree.joint >= wp.int32(0):
                local = body_q_local[tree.child]
            origin = articulation_origin[articulation]
        else:
            start_transform = wp.transform_identity()
            if tree.joint >= wp.int32(0):
                start_transform = jcalc_transform(
                    tree.kind, joint_axis, tree.qd_start, tree.lin, tree.ang, joint_q, tree.q_start
                )
            local, origin, _anchor = local_kinematics(
                shared,
                tree,
                lane,
                articulation,
                max_depth,
                start_transform,
                wp.transform_identity(),
                wp.bool(False),
                joint_q,
                body_x_com,
                articulation_origin,
                body_q_local,
                body_q_com,
                joint_anchor_local,
            )
        diagonal_changed = wp.bool(False)
        if tree.joint >= wp.int32(0):
            previous_diagonal = _vec6()
            for offset in range(6):
                if wp.int32(offset) < tree.dof_count:
                    previous_diagonal[offset] = factor_diagonal[tree.qd_start + offset]
            _configure_implicit_joint_dynamics_packed(
                tree.joint,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_target_q_start,
                joint_q,
                joint_qd_public,
                joint_target_q,
                joint_target_qd,
                dof_parameters,
                dt,
                factor_diagonal,
                implicit_force,
            )
            for offset in range(6):
                if wp.int32(offset) < tree.dof_count:
                    if factor_diagonal[tree.qd_start + offset] != previous_diagonal[offset]:
                        diagonal_changed = wp.bool(True)
            if frames_current:
                initialize_reduced_rates(
                    tree.joint,
                    joint_type,
                    joint_child,
                    joint_qd_start,
                    joint_qd_public,
                    joint_anchor_local,
                    body_q_com,
                    joint_qd,
                )
            else:
                initialize_reduced_joint(
                    tree.joint,
                    joint_type,
                    joint_child,
                    joint_q,
                    joint_q_start,
                    joint_qd_start,
                    joint_axis,
                    joint_dof_dim,
                    joint_qd_public,
                    joint_anchor_local,
                    body_q_com,
                    body_mass,
                    body_inertia,
                    joint_qd,
                    body_i_s,
                    joint_s,
                )
        _sync_contact_warp()
        k = lane
        while k < n:
            _store(shared, wp.int32(qd_base) + k, joint_qd[dof_start + k])
            k += wp.int32(_LANES)
        if reuse_factor and not _warp_any(diagonal_changed):
            restore(shared, articulation, lane, dof_start, n, motion, dense)
        else:
            # The relax step stores its factor for the next biased step.
            factor(
                shared,
                articulation,
                lane,
                joint_start,
                joint_count,
                dof_start,
                n,
                heights,
                joint_child,
                motion,
                body_inertia_flat,
                factor_diagonal,
                dense,
                wp.bool(relax),
            )
        _sync_contact_warp()
        x_com = wp.transform_get_translation(local * body_x_com[tree.child])

        held = wp.spatial_vector()
        held_origin = origin
        if wp.static(not relax):
            # Free rates under joint, drive, external, and gravity forces.
            if tree.joint >= wp.int32(0):
                gravity_force = tree.mass * gravity[wp.max(body_world[tree.child], wp.int32(0))]
                force_com = body_f[tree.child]
                force = wp.spatial_top(force_com)
                bias = -wp.spatial_vector(gravity_force, wp.cross(x_com, gravity_force)) - wp.spatial_vector(
                    force, wp.spatial_bottom(force_com) + wp.cross(x_com, force)
                )
                if _is_floating(tree.kind):
                    control_force = wp.vec3(
                        joint_f[tree.qd_start], joint_f[tree.qd_start + 1], joint_f[tree.qd_start + 2]
                    )
                    control_torque = wp.vec3(
                        joint_f[tree.qd_start + 3], joint_f[tree.qd_start + 4], joint_f[tree.qd_start + 5]
                    )
                    bias -= wp.spatial_vector(control_force, control_torque + wp.cross(x_com, control_force))
                if wp.static(coupled):
                    # The contacts between trees act with the impulses of their last biased
                    # sweeps, push-out included, so that this tree's own contacts and its
                    # integration answer them. The sweeps after this kernel solve for the
                    # change: they start from the opposite of what is applied here and add
                    # the impulses the contacts still hold.
                    held = bodies.reduced.carried_wrench[tree.child]
                    if wp.length_sq(held) > wp.float32(0.0):
                        bias -= idt * held
                        bodies.reduced.deferred_pending[articulation] = wp.int32(1)
                _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, bias)
            _sync_contact_warp()
            solve_forces(shared, tree, lane, dof_start, wp.int32(first_base), joint_f, implicit_force, n)
            k = lane
            while k < n:
                rate = _load(shared, wp.int32(qd_base) + k) + dt * _load(shared, wp.int32(rhs_base) + k)
                _store(shared, wp.int32(qd_base) + k, rate)
                joint_qd[dof_start + k] = rate
                k += wp.int32(_LANES)
            _sync_contact_warp()

        if wp.static(relax and coupled):
            # Changes of the impulses between trees against what the biased kernel applied,
            # see ReducedPhoenXArticulation._solve_cross_contacts: this tree's rates change by
            # M^-1 J^T of the change after the relax sweeps, and the next biased kernel
            # applies the impulses after the biased sweeps.
            if bodies.reduced.deferred_pending[articulation] != wp.int32(0):
                if tree.joint >= wp.int32(0):
                    _store_spatial(
                        shared,
                        wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane,
                        -bodies.reduced.deferred_wrench[tree.child],
                    )
                    bodies.reduced.carried_wrench[tree.child] = (
                        bodies.reduced.carried_wrench[tree.child] + bodies.reduced.biased_wrench[tree.child]
                    )
                    bodies.reduced.deferred_wrench[tree.child] = wp.spatial_vector()
                    bodies.reduced.biased_wrench[tree.child] = wp.spatial_vector()
                _sync_contact_warp()
                solve_forces(
                    shared,
                    tree,
                    lane,
                    dof_start,
                    wp.int32(first_base),
                    zero_generalized_force,
                    zero_generalized_force,
                    n,
                )
                k = lane
                while k < n:
                    rate = _load(shared, wp.int32(qd_base) + k) + _load(shared, wp.int32(rhs_base) + k)
                    _store(shared, wp.int32(qd_base) + k, rate)
                    joint_qd[dof_start + k] = rate
                    k += wp.int32(_LANES)
                _sync_contact_warp()
                if lane == wp.int32(0):
                    bodies.reduced.deferred_pending[articulation] = wp.int32(0)

        if contacts_active:
            section_start = wp.int32(0)
            if articulation > wp.int32(0):
                section_start = schedule_section_end[articulation - wp.int32(1)]
            local_points = wp.int32(0)
            index = section_start + lane
            while index < schedule_section_end[articulation]:
                local_points += contact_get_contact_count(columns, schedule_columns[index])
                index += wp.int32(_LANES)
            total_points = _warp_sum_int(local_points)
            pages = (total_points + wp.int32(_POINTS_PER_PAGE - 1)) // wp.int32(_POINTS_PER_PAGE)
            # The relax step before left the rows of every page that has storage of its own.
            gathered = reuse_rows and pages <= wp.int32(_CACHED_PAGE_COUNT)
            for page in range(pages):
                if not gathered:
                    gather(
                        articulation,
                        lane,
                        wp.int32(_LANES),
                        page,
                        wp.int32(_POINTS_PER_PAGE),
                        wp.bool(True),
                        total_points,
                        wp.bool(False),
                        schedule_section_end,
                        schedule_columns,
                        columns,
                        bodies,
                        idt,
                        wp.bool(True),
                        cc,
                        contacts,
                        point_count,
                        row_count,
                        point_contact,
                        point_column,
                        point0,
                        point1,
                        normal,
                        tangent0,
                        row_body,
                        row_wrench,
                        row_body_pair,
                        row_wrench_pair,
                        row_velocity,
                    )
                _sync_contact_warp()
                packed_articulation = articulation * wp.int32(_CACHED_PAGE_COUNT) + wp.min(
                    page, wp.int32(_CACHED_PAGE_COUNT - 1)
                )
                page_points = point_count[packed_articulation]
                column_start = wp.bool(False)
                if wp.static(patch):
                    if lane < page_points:
                        column_start = lane == wp.int32(0) or (
                            point_column[packed_articulation, lane] != point_column[packed_articulation, lane - 1]
                        )
                    if wp.static(not relax) and lane == wp.int32(0):
                        warm_patch_impulses(
                            cc, packed_articulation, page_points, point_contact, point_column, patch_lambda
                        )
                    _sync_contact_warp()
                # Chunks of rows that fit the lanes; several chunks alternate within each sweep.
                chunks = (page_points + wp.int32(chunk_capacity - 1)) // wp.int32(chunk_capacity)
                # Several chunks first apply every chunk's accumulated impulses, then sweep in turn.
                rounds = wp.int32(1)
                sweeps = iterations
                warm_rounds = wp.int32(0)
                if chunks > wp.int32(1):
                    rounds = iterations
                    sweeps = wp.int32(1)
                    if wp.static(not relax):
                        warm_rounds = wp.int32(1)
                # Rounds alternate their direction, so the chunk that ends a round starts the
                # next one: it stays for that sweep too instead of being formed again.
                for round_offset in range(rounds + warm_rounds):
                    round_index = round_offset - warm_rounds
                    for chunk_offset in range(chunks):
                        if chunks > wp.int32(1) and round_offset > wp.int32(0) and chunk_offset == wp.int32(0):
                            continue
                        stays = (
                            chunks > wp.int32(1)
                            and chunk_offset == chunks - wp.int32(1)
                            and round_offset < rounds + warm_rounds - wp.int32(1)
                        )
                        chunk_index = wp.int32(chunk_offset)
                        if (round_index & wp.int32(1)) != wp.int32(0):
                            chunk_index = chunks - wp.int32(1) - wp.int32(chunk_offset)
                        first = wp.int32(chunk_capacity) * chunk_index
                        last = wp.min(first + wp.int32(chunk_capacity), page_points)
                        chunk = wp.vec4i(
                            first,
                            last,
                            _warp_sum_int(wp.where(column_start and lane < first, 1, 0)),
                            _warp_sum_int(wp.where(column_start and lane < last, 1, 0)),
                        )
                        solve_chunk(
                            shared,
                            articulation,
                            lane,
                            page,
                            chunk,
                            joint_start,
                            dof_start,
                            n,
                            bodies,
                            columns,
                            cc,
                            idt,
                            sor_boost,
                            wp.max(round_index, wp.int32(0)),
                            wp.where(round_index < wp.int32(0), wp.int32(0), sweeps) + wp.where(stays, 1, 0),
                            wp.bool(not relax),
                            wp.bool(not relax) and round_offset == wp.int32(0),
                            # Reused rows keep the biases the relax step already projected.
                            round_offset == wp.int32(0) and not gathered,
                            point_count,
                            point_contact,
                            point_column,
                            point0,
                            normal,
                            tangent0,
                            row_body,
                            row_wrench,
                            row_body_pair,
                            row_wrench_pair,
                            row_velocity,
                            patch_lambda,
                            normal_load,
                            joint_qd,
                            point_coupling,
                            round_offset == wp.int32(0),
                        )

        if wp.static(relax):
            publish_rates(
                shared,
                tree,
                lane,
                dof_start,
                max_depth,
                x_com,
                wp.float32(0.0),
                wp.bool(False),
                wp.bool(False),
                wp.spatial_vector(),
                wp.vec3(0.0),
                joint_axis,
                joint_q,
                joint_qd,
                joint_qd_public,
                joint_x_p,
                joint_qd_start,
                joint_s,
                bodies,
            )
            return

        # Velocity-product step: twists, Coriolis accelerations, and bias forces root to leaf.
        twist = wp.spatial_vector()
        coriolis = wp.spatial_vector()
        root_velocity = wp.spatial_vector()
        rates = wp.spatial_vector()
        if tree.joint >= wp.int32(0):
            rates = joint_twist(shared, tree, dof_start)
        # Every lane adds its parent's stored values at every level, see local_kinematics.
        has_parent = tree.parent >= wp.int32(0)
        parent_offset = wp.int32(_MOTION_STRIDE) * wp.max(tree.parent, wp.int32(0))
        for _level in range(max_depth + wp.int32(1)):
            twist = rates + wp.where(
                has_parent, _load_motion(shared, wp.int32(first_base) + parent_offset), wp.spatial_vector()
            )
            coriolis = wp.where(
                has_parent, _load_motion(shared, wp.int32(second_base) + parent_offset), wp.spatial_vector()
            ) + spatial_cross(twist, rates)
            _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, twist)
            _store_spatial(shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * lane, coriolis)
            _sync_contact_warp()
        if tree.joint >= wp.int32(0) and not has_parent and _is_floating(tree.kind):
            root_velocity = twist
        momentum = wp.spatial_vector()
        if tree.joint >= wp.int32(0):
            inertia = _unpack_symmetric_mat66(
                _pack_transformed_body_inertia(local * body_x_com[tree.child], tree.mass, body_inertia[tree.child])
            )
            momentum = inertia * twist
            midpoint.body_momentum[tree.child] = momentum
            _store_spatial(
                shared,
                wp.int32(third_base) + wp.int32(_MOTION_STRIDE) * lane,
                inertia * coriolis + spatial_cross_dual(twist, momentum),
            )
        _sync_contact_warp()
        solve_forces(
            shared, tree, lane, dof_start, wp.int32(third_base), zero_generalized_force, zero_generalized_force, n
        )
        # The explicit step and its energy drift in the frame of a floating root, see
        # _explicit_step_drift_terms.
        frame = _shuffle_reduced_spatial(_floating_root_frame(root_velocity), wp.int32(0), wp.int32(_LANES), _FULL_MASK)
        first_tree = wp.bool(True)
        if wp.static(forest):
            # Trees on fixed roots step in the frame at rest and hold no momentum to restore.
            first_tree = _in_first_tree(tree.parent, lane, max_depth)
            if not first_tree:
                frame = wp.spatial_vector()
        drift = wp.float32(0.0)
        energy = wp.float32(0.0)
        dissipation = wp.float32(0.0)
        velocity_com = wp.vec3(0.0)
        omega = wp.vec3(0.0)
        # A joint's own share of the step needs no parent: kick its rates and sum its twists.
        joint_velocity = wp.spatial_vector()
        joint_acceleration = wp.spatial_vector()
        joint_new_twist = wp.spatial_vector()
        if tree.joint >= wp.int32(0):
            for offset in range(tree.dof_count):
                dof = tree.qd_start + offset
                local_dof = dof - dof_start
                subspace = _load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * local_dof)
                rate = _load(shared, wp.int32(qd_base) + local_dof)
                acceleration = _load(shared, wp.int32(rhs_base) + local_dof)
                midpoint.joint_qd_mid[dof] = rate
                dissipation += _joint_dissipation(midpoint, dof, rate)
                joint_velocity += subspace * rate
                joint_acceleration += subspace * acceleration
                rate += acceleration * dt
                _store(shared, wp.int32(qd_base) + local_dof, rate)
                joint_qd[dof] = rate
                joint_new_twist += subspace * rate
        # The twists, accelerations, and Coriolis rates chain from the parents' stored values
        # at every level, see local_kinematics.
        new_twist = wp.spatial_vector()
        terms = _vec8()
        for _level in range(max_depth + wp.int32(1)):
            new_twist = (
                wp.where(has_parent, _load_motion(shared, wp.int32(first_base) + parent_offset), wp.spatial_vector())
                + joint_new_twist
            )
            child_acceleration = (
                wp.where(has_parent, _load_motion(shared, wp.int32(second_base) + parent_offset), wp.spatial_vector())
                + joint_acceleration
            )
            terms = _explicit_step_drift_terms(
                new_twist - dt * child_acceleration,
                child_acceleration,
                joint_velocity,
                joint_acceleration,
                tree.depth == wp.int32(0),
                frame,
                momentum,
                tree.mass,
                x_com,
                wp.where(has_parent, _load_motion(shared, wp.int32(third_base) + parent_offset), wp.spatial_vector()),
            )
            _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, new_twist)
            _store_spatial(shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * lane, child_acceleration)
            _store_spatial(
                shared,
                wp.int32(third_base) + wp.int32(_MOTION_STRIDE) * lane,
                wp.spatial_vector(terms[0], terms[1], terms[2], terms[3], terms[4], terms[5]),
            )
            _sync_contact_warp()
        if tree.joint >= wp.int32(0):
            drift = terms[6]
            energy = terms[7]
            body_velocity[tree.child] = new_twist
            omega = wp.spatial_bottom(new_twist)
            velocity_com = wp.spatial_top(new_twist) + wp.cross(omega, x_com)
        linear = wp.spatial_top(momentum)
        if not first_tree:
            linear = wp.vec3(0.0)
        totals = _sum_reduced_group_vec4(wp.vec4(linear[0], linear[1], linear[2], drift), wp.int32(_LANES), _FULL_MASK)
        target_linear_momentum = wp.vec3(totals[0], totals[1], totals[2])
        tree_energy = _sum_reduced_group_float(energy, wp.int32(_LANES), _FULL_MASK)
        tree_dissipation = _sum_reduced_group_float(dissipation, wp.int32(_LANES), _FULL_MASK)
        integrated = _explicit_step_drifts(totals[3], tree_energy, tree_dissipation, dt)
        if integrated:
            # Fast trees take the momentum-conserving midpoint step on the global workspace,
            # which also advances their coordinates.
            midpoint_tree = _MidpointTree()
            midpoint_tree.articulation = articulation
            midpoint_tree.lane = lane
            midpoint_tree.width = wp.int32(_LANES)
            midpoint_tree.mask = _FULL_MASK
            midpoint_tree.max_depth = max_depth_all
            midpoint_tree.depth_start = depth_start
            midpoint_tree.order = depth_joint
            _restore_incoming_rates(midpoint_tree, joint_qd_start, midpoint, joint_qd)
            _midpoint_momentum_step(
                midpoint_tree,
                joint_type,
                joint_parent,
                joint_child,
                joint_qd_start,
                joint_s,
                body_i_s,
                midpoint,
                dt,
                wp.int32(_MIDPOINT_MAX_ITERATIONS),
                joint_qd,
                body_velocity,
            )
            k = lane
            while k < n:
                _store(shared, wp.int32(qd_base) + k, joint_qd[dof_start + k])
                k += wp.int32(_LANES)
        elif lane == wp.int32(0):
            midpoint.flag[articulation] = wp.int32(0)

        # Integrate the coordinates. Joints integrate on their own; a floating joint integrates
        # its child's pose, and a floating root's coordinates follow from it at once.
        chained = tree.joint >= wp.int32(0) and (integrated or not _is_floating(tree.kind))
        floating_child = tree.joint >= wp.int32(0) and not chained and tree.parent >= wp.int32(0)
        joint_transform = wp.transform_identity()
        world = wp.transform_identity()
        if chained:
            if not integrated:
                jcalc_integrate(
                    tree.parent,
                    tree.x_c,
                    tree.com,
                    tree.kind,
                    joint_q,
                    joint_qd,
                    zero_generalized_force,
                    tree.q_start,
                    tree.qd_start,
                    tree.lin,
                    tree.ang,
                    dt,
                    joint_q,
                    joint_qd,
                )
            joint_transform = jcalc_transform(
                tree.kind, joint_axis, tree.qd_start, tree.lin, tree.ang, joint_q, tree.q_start
            )
        elif tree.joint >= wp.int32(0):
            com_twist = wp.spatial_vector(velocity_com, omega)
            if tree.parent < wp.int32(0):
                # Move the root COM with its re-referenced velocity, see _carry_floating_root_twist.
                com_twist = _transport_root_twist(com_twist, dt)
            old = wp.transform(origin + wp.transform_get_translation(local), wp.transform_get_rotation(local))
            world = integrate_body_pose_from_com_twist(old, tree.com, com_twist, dt)
            if tree.parent < wp.int32(0):
                joint_transform = wp.transform_inverse(tree.x_p) * world * tree.x_c
                position = wp.transform_get_translation(joint_transform)
                rotation = wp.transform_get_rotation(joint_transform)
                for component in range(3):
                    joint_q[tree.q_start + component] = position[component]
                for component in range(4):
                    joint_q[tree.q_start + 3 + component] = rotation[component]

        # Frames about the new root COM in one pass root to leaf, which also poses the links,
        # then the motion subspaces and the rates. A floating root's rates still read the
        # motion subspaces of the start pose.
        root_twist = joint_twist(shared, tree, dof_start)
        local, origin, anchor = local_kinematics(
            shared,
            tree,
            lane,
            articulation,
            max_depth,
            joint_transform,
            world,
            floating_child,
            joint_q,
            body_x_com,
            articulation_origin,
            body_q_local,
            body_q_com,
            joint_anchor_local,
        )
        if wp.static(coupled):
            if tree.joint >= wp.int32(0) and wp.length_sq(held) > wp.float32(0.0):
                # The same impulses, about the origin this step moved.
                force = wp.spatial_top(held)
                held = wp.spatial_vector(force, wp.spatial_bottom(held) - wp.cross(origin - held_origin, force))
                bodies.reduced.carried_wrench[tree.child] = held
                bodies.reduced.deferred_wrench[tree.child] = -held
                bodies.reduced.biased_wrench[tree.child] = -held
        if tree.joint >= wp.int32(0):
            if chained:
                world = wp.transform(origin + wp.transform_get_translation(local), wp.transform_get_rotation(local))
            body_q[tree.child] = world
            slot = tree.child + wp.int32(1)
            rotation = wp.transform_get_rotation(world)
            bodies.position[slot] = wp.transform_get_translation(world) + wp.quat_rotate(rotation, tree.com)
            bodies.orientation[slot] = rotation
        if tree.joint >= wp.int32(0):
            jcalc_motion(
                tree.kind,
                joint_axis,
                joint_q,
                tree.lin,
                tree.ang,
                anchor,
                joint_qd,
                tree.q_start,
                tree.qd_start,
                joint_s_publish,
            )
            for offset in range(tree.dof_count):
                local_dof = tree.qd_start + offset - dof_start
                subspace = joint_s_publish[tree.qd_start + offset]
                joint_s[tree.qd_start + offset] = subspace
                _store_spatial(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * local_dof, subspace)
            body_i_s[tree.child] = _pack_transformed_body_inertia(
                local * body_x_com[tree.child], tree.mass, body_inertia[tree.child]
            )
        x_com = wp.transform_get_translation(local * body_x_com[tree.child])
        _sync_contact_warp()
        publish_rates(
            shared,
            tree,
            lane,
            dof_start,
            max_depth,
            x_com,
            dt,
            wp.bool(True),
            integrated,
            root_twist,
            target_linear_momentum,
            joint_axis,
            joint_q,
            joint_qd,
            joint_qd_public,
            joint_x_p,
            joint_qd_start,
            joint_s_publish,
            bodies,
        )

    return dense_substep_kernel


def dense_substep_blockers(articulation) -> list[str]:
    """Return why :class:`DenseSubstep` cannot run a reduced articulation system; empty when it can."""
    reasons = []
    dense = articulation.system.dense
    if dense is None:
        reasons.append("the articulated-body backend is selected")
    elif dense.has_fallback:
        reasons.extend(dense.fallback_reasons)
    # Contacts that couple two trees are solved between the fused kernels, see
    # ReducedPhoenXArticulation._solve_cross_contacts. Contacts within one tree that its
    # block cannot own still need the articulated-body pair response.
    contact = articulation.contact_block_system
    if contact.requires_impulse_response and not contact.packed_internal_contacts:
        reasons.append("contacts within a tree that its block does not own")
    if articulation.loop_system.count:
        reasons.append("loop joints")
    if articulation.system.runtime_slots:
        reasons.append("runtime articulation slots")
    model = articulation.model
    if int(getattr(model, "particle_count", 0) or 0):
        reasons.append("deformable particles")
    count = int(model.articulation_count)
    start = model.articulation_start.numpy()[:count]
    end = model.articulation_end.numpy()[:count]
    if int((end - start).max(initial=0)) > _LANES:
        reasons.append(f"articulations with more than {_LANES} joints")
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    for first, last in zip(start, end, strict=True):
        children = set(joint_child[int(first) : int(last)].tolist())
        roots = [joint for joint in range(int(first), int(last)) if int(joint_parent[joint]) not in children]
        if any(int(joint_parent[joint]) >= 0 for joint in roots):
            reasons.append("roots attached to a body")
            break
        # Frames follow the first root; further roots must be fixed to the world.
        if any(int(joint_type[joint]) in (int(JointType.FREE), int(JointType.DISTANCE)) for joint in roots[1:]):
            reasons.append("a floating root that is not its block's first joint")
            break
    return reasons


class DenseSubstep:
    """Fused substep kernels for dense reduced trees whose contacts are all private.

    Args:
        articulation: The :class:`ReducedPhoenXArticulation` whose dense
            operator holds every tree.
    """

    def __init__(self, articulation):
        self.articulation = articulation
        self.system = articulation.system
        contact = articulation.contact_block_system
        if contact.requires_impulse_response:
            # Contacts between trees are solved between the kernels with the links' responses
            # from the dense factor, and the relax kernel applies the collected link impulses.
            contact.flush_fallback = False
            reduced = articulation.bodies.reduced
            model = articulation.model
            path = max(bucket.path_capacity for bucket in self.system.dense.buckets)
            reduced.link_response = 1
            reduced.link_path = wp.zeros((int(model.body_count), path), dtype=wp.spatial_vector, device=model.device)
            reduced.link_path_dof = wp.full((int(model.body_count), path), -1, dtype=wp.int32, device=model.device)
        self.dense = self.system.dense
        self.contact = articulation.contact_block_system
        # The relax step gathers the contact rows at the pose the next biased step starts
        # from: (substep dt, contact ingest count) of that gather, or None.
        self._gathered_rows = None
        # Packed per-DOF joint parameters; SolverPhoenX.notify_model_changed refreshes them.
        self._dof_parameters = wp.zeros(
            (max(1, int(articulation.model.joint_dof_count)), _DOF_PARAMETERS),
            dtype=wp.float32,
            device=articulation.model.device,
        )
        self.refresh_joint_parameters()
        # Blocks that hold several trees, see _get_reduced_model.
        model = articulation.model
        count = int(model.articulation_count)
        starts = model.articulation_start.numpy()[:count]
        ends = model.articulation_end.numpy()[:count]
        parents = model.joint_parent.numpy()
        self.forest = any(
            int(np.count_nonzero(parents[start:end] < 0)) > 1 for start, end in zip(starts, ends, strict=True)
        )

    def refresh_joint_parameters(self) -> None:
        """Repack the joint parameters after the model's joint properties changed."""
        model = self.articulation.model
        wp.launch(
            _pack_dof_parameters_kernel,
            dim=self._dof_parameters.shape[0],
            inputs=[
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_target_mode,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                model.joint_damping,
                model.joint_armature,
                model.joint_effort_limit,
            ],
            outputs=[self._dof_parameters],
            device=model.device,
        )

    def _launch(self, world, dt: float, *, relax: bool, reuse_factor: bool = False, reuse_rows: bool = False) -> None:
        articulation = self.articulation
        model = articulation.model
        system = self.system
        dense = self.dense
        contact = self.contact
        state = articulation.state
        control = articulation.control
        contacts_active = bool(world._reduced_contacts_active_this_step and world.max_contact_columns > 0)
        iterations = world._active_velocity_iterations if relax else world.solver_iterations
        for bucket in dense.buckets:
            kernel = make_dense_substep_kernel(
                bucket.max_dofs,
                bucket.joint_capacity,
                bucket.entry_capacity,
                bucket.path_capacity,
                bucket.subtree_capacity,
                contact.patch_rows,
                relax,
                articulation.external_rigid_rows,
                contact.internal_rows,
                self.forest,
                contact.requires_impulse_response,
            )
            wp.launch(
                kernel,
                dim=bucket.articulations.shape[0] * _LANES,
                block_dim=_LANES,
                inputs=[
                    bucket.articulations,
                    wp.float32(dt),
                    wp.float32(world.sor_boost),
                    wp.int32(iterations),
                    wp.bool(contacts_active),
                    wp.bool(reuse_factor),
                    wp.bool(reuse_rows),
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_target_q_start,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_world,
                    model.gravity,
                    control.joint_f,
                    control.joint_target_q,
                    control.joint_target_qd,
                    state.joint_q,
                    state.joint_qd,
                    state.body_q,
                    state.body_f,
                    system.joint_qd_internal,
                    system.joint_factor_diagonal,
                    system.joint_implicit_force,
                    system.joint_s,
                    system.body_i_s,
                    dense.motion,
                    dense.body_inertia,
                    system.body_x_com,
                    system.articulation_origin,
                    system.body_q_local,
                    system.body_q_com,
                    system.joint_anchor_local,
                    system.joint_s_publish,
                    system.zero_generalized_force,
                    system.body_velocity,
                    system.bind_midpoint_state(state),
                    wp.int32(system.advance_max_depth),
                    system.advance_articulation_depth_start,
                    system.advance_articulation_depth_joint,
                    dense.template_joint_start,
                    dense.template_max_depth,
                    dense.joint_parent_local,
                    dense.joint_depth,
                    dense.template_image,
                    dense.template_image_start,
                    dense.data,
                    contact.schedule_section_end,
                    contact.schedule_columns,
                    world._contact_cols,
                    world._contact_container,
                    world._active_contact_views(),
                    contact.point_count,
                    contact.row_count,
                    contact.point_contact,
                    contact.point_column,
                    contact.point0,
                    contact.point1,
                    contact.normal,
                    contact.tangent0,
                    contact.row_body,
                    contact.row_wrench,
                    contact.row_body_pair,
                    contact.row_wrench_pair,
                    contact.row_velocity,
                    contact.patch_lambda,
                    contact.normal_load,
                    contact.point_coupling,
                    self._dof_parameters,
                    articulation.bodies,
                ],
                device=model.device,
            )
        if contact.dynamic_external_owner:
            wp.launch(
                _combine_shared_external_kernel,
                dim=int(model.body_count) + 1,
                inputs=[articulation.bodies],
                device=model.device,
            )

    def biased(self, world, dt: float, *, reuse_factor: bool) -> None:
        """Advance every tree through the biased solve, the velocity-product step, and publication.

        ``reuse_factor`` starts from the frames and factor of a relax step at the same pose;
        trees whose implicit joint diagonal changed refactor. The contact rows that relax
        step gathered are reused too when the contacts and the substep ``dt``, which
        scales their biases, are unchanged since.
        """
        reuse_rows = reuse_factor and self._gathered_rows == (float(dt), world._contact_ingest_count)
        self._gathered_rows = None
        self._launch(world, dt, relax=False, reuse_factor=reuse_factor, reuse_rows=reuse_rows)

    def relax(self, world, dt: float) -> None:
        """Refactor at the integrated pose, apply the bias-free contact correction, and store the factor."""
        self._launch(world, dt, relax=True)
        if world._reduced_contacts_active_this_step and world.max_contact_columns > 0:
            self._gathered_rows = (float(dt), world._contact_ingest_count)


__all__ = ["CHUNK_POINTS", "PATCH_CHUNK_POINTS", "DenseSubstep", "dense_substep_blockers", "make_dense_substep_kernel"]
