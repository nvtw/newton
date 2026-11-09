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

import warp as wp

from newton._src.sim import JointType
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
    configure_implicit_joint_dynamics,
    initialize_reduced_joint,
)
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _CACHED_PAGE_COUNT,
    _POINTS_PER_PAGE,
    _gather_packed_contact_points,
    _gather_packed_patch_points,
    _solve_patch_normal_row,
    _solve_patch_tangent_rows,
    _solve_reduced_contact_point,
    _store_point_row_effective_mass,
    _sync_contact_warp,
)
from newton._src.solvers.phoenx.body import BodyContainer, DenseTreeData
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_contact_count,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)

PAGE_ROWS = 24
"""Contact rows one chunk solves at once; chunks hold ``PAGE_ROWS // 3`` points."""

_CHUNK_POINTS = PAGE_ROWS // 3
_FULL_MASK = wp.constant(wp.uint32(0xFFFFFFFF))


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
def _store_spatial(shared: wp.uint64, index: wp.int32, value: wp.spatial_vector):
    for component in range(6):
        _store(shared, index + wp.int32(component), value[component])


@wp.func
def _store_transform(shared: wp.uint64, index: wp.int32, value: wp.transform):
    translation = wp.transform_get_translation(value)
    rotation = wp.transform_get_rotation(value)
    for component in range(3):
        _store(shared, index + wp.int32(component), translation[component])
    for component in range(4):
        _store(shared, index + wp.int32(3 + component), rotation[component])


@wp.func
def _load_transform(shared: wp.uint64, index: wp.int32) -> wp.transform:
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
    inertia: wp.mat33


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
    body_inertia: wp.array[wp.mat33],
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
        tree.inertia = body_inertia[child]
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
def _is_floating(kind: wp.int32) -> wp.bool:
    return kind == JointType.FREE or kind == JointType.DISTANCE


@functools.cache
def make_dense_substep_kernel(
    max_dofs: int,
    joint_capacity: int,
    entry_capacity: int,
    path_capacity: int,
    subtree_capacity: int,
    patch: bool,
    relax: bool,
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
    """
    # Scratch after the factor's tables: the composite inertias while factoring, then the tree
    # recurrences and the contact pages share one region; the rates and a solve vector follow.
    tree_words = 4 * _MOTION_STRIDE * _LANES
    contact_words = _LANES * (path_capacity + PAGE_ROWS + 2)
    qd_offset = max(tree_words, contact_words, _INERTIA_STRIDE * joint_capacity)
    scratch_words = qd_offset + 2 * max_dofs
    ops = make_dense_tree_ops(max_dofs, joint_capacity, entry_capacity, subtree_capacity, scratch_words)
    layout = ops.layout
    shared_base = _make_shared_base(layout.size)
    stage = ops.stage
    factor = ops.factor
    restore = ops.restore
    solve_mass = ops.solve
    solve_lower = ops.solve_lower
    pose_base = layout.scratch
    first_base = pose_base + _MOTION_STRIDE * _LANES
    second_base = first_base + _MOTION_STRIDE * _LANES
    third_base = second_base + _MOTION_STRIDE * _LANES
    z_base = layout.scratch
    a_base = z_base + _LANES * path_capacity
    lambda_base = a_base + _LANES * PAGE_ROWS
    leaf_byte = 4 * (lambda_base + _LANES)
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
        f"_{int(patch)}_{int(relax)}",
        dense_unroll(layout, joint_capacity),
    )

    @wp.func
    def local_kinematics(
        shared: wp.uint64,
        tree: _TreeJoint,
        lane: wp.int32,
        articulation: wp.int32,
        max_depth: wp.int32,
        joint_axis: wp.array[wp.vec3],
        joint_q: wp.array[wp.float32],
        body_x_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
    ):
        """Form the link frames about the root COM from the coordinates and publish them.

        Matches the local kinematics of :func:`_finish_and_publish_reduced_warp_device`.
        Returns the child's link frame and the world position of the root COM.
        """
        joint_transform = wp.transform_identity()
        world = wp.transform_identity()
        if tree.joint >= wp.int32(0):
            joint_transform = jcalc_transform(
                tree.kind, joint_axis, tree.qd_start, tree.lin, tree.ang, joint_q, tree.q_start
            )
            if tree.parent < wp.int32(0):
                world = tree.x_p * joint_transform * wp.transform_inverse(tree.x_c)
        origin = wp.transform_point(world, tree.com)
        origin = wp.vec3(_lane_float(origin[0], 0), _lane_float(origin[1], 0), _lane_float(origin[2], 0))
        local = wp.transform_identity()
        for depth in range(max_depth + wp.int32(1)):
            if tree.depth == depth:
                anchor = wp.transform_identity()
                if tree.parent < wp.int32(0):
                    rotation = wp.transform_get_rotation(world)
                    local = wp.transform(-wp.quat_rotate(rotation, tree.com), rotation)
                    root_transform = joint_transform
                    if _is_floating(tree.kind):
                        root_transform = wp.transform(wp.vec3(0.0), wp.transform_get_rotation(joint_transform))
                    anchor = local * tree.x_c * wp.transform_inverse(root_transform)
                else:
                    anchor = (
                        _load_transform(shared, wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * tree.parent) * tree.x_p
                    )
                    local = anchor * joint_transform * wp.transform_inverse(tree.x_c)
                _store_transform(shared, wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * lane, local)
                joint_anchor_local[tree.joint] = anchor
                body_q_local[tree.child] = local
                body_q_com[tree.child] = local * body_x_com[tree.child]
            _sync_contact_warp()
        if lane == wp.int32(0):
            articulation_origin[articulation] = origin
        return local, origin

    @wp.func
    def joint_twist(shared: wp.uint64, tree: _TreeJoint, dof_start: wp.int32) -> wp.spatial_vector:
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
        shared: wp.uint64,
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
    def common_prefix(
        shared: wp.uint64, entry: wp.int32, length: wp.int32, other_entry: wp.int32, other_length: wp.int32
    ) -> wp.int32:
        """Return how many leading entries two root paths share; tree paths agree on a prefix only."""
        low = wp.int32(0)
        high = wp.min(length, other_length)
        while low < high:
            middle = (low + high + wp.int32(1)) >> wp.int32(1)
            if _load_u8(shared, wp.int32(path) + entry + middle - wp.int32(1)) == _load_u8(
                shared, wp.int32(path) + other_entry + middle - wp.int32(1)
            ):
                low = middle
            else:
                high = middle - wp.int32(1)
        return low

    @wp.func
    def run_end(shared: wp.uint64, row: wp.int32, row_count: wp.int32) -> wp.int32:
        """Return the end of the run of rows from ``row`` whose bodies share one root path."""
        leaf = _load_u8(shared, wp.int32(leaf_byte) + row)
        end = row + wp.int32(1)
        while end < row_count and _load_u8(shared, wp.int32(leaf_byte) + end) == leaf:
            end += wp.int32(1)
        return end

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
        shared: wp.uint64,
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
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_body: wp.array2d[wp.int32],
        row_wrench: wp.array2d[wp.spatial_vector],
        row_velocity: wp.array2d[wp.float32],
        patch_lambda: wp.array2d[wp.vec2],
        normal_load: wp.array2d[wp.float32],
        joint_qd: wp.array[wp.float32],
    ):
        """Sweep one chunk of gathered contact rows and add its velocity change to the shared and global rates.

        A chunk holds the points ``chunk[0]`` to ``chunk[1]`` and, with patch friction,
        the columns ``chunk[2]`` to ``chunk[3]`` that start among them. Lane ``r`` owns
        local row ``r`` and lane ``q`` projects the chunk's point ``q``. Point friction
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

        # Row r: J_r along the root path of its body, its velocity J_r qd plus the gathered
        # external part, then z_r = L^-T J_r^T leaves first.
        velocity = wp.float32(0.0)
        leaf = wp.int32(-1)
        lambda_row = wp.float32(0.0)
        path_entry = wp.int32(0)
        path_length = wp.int32(0)
        gathered_row = packed_row(chunk, point_total, lane)
        if lane < row_count:
            body = row_body[packed_articulation, gathered_row]
            wrench = row_wrench[packed_articulation, gathered_row]
            leaf = _load_u8(shared, wp.int32(joint_last_dof) + bodies.reduced.body_joint[body] - joint_start) - 1
            velocity = row_velocity[packed_articulation, gathered_row]
            inverse_mass = wp.float32(0.0)
            if leaf >= wp.int32(0):
                path_entry = _load_u16(shared, wp.int32(dof_entry) + leaf)
                path_length = _load_u16(shared, wp.int32(dof_entry) + leaf + wp.int32(1)) - path_entry
                for position in range(path_length):
                    k = _load_u8(shared, wp.int32(path) + path_entry + position)
                    value = wp.dot(_load_motion(shared, wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * k), wrench)
                    velocity += value * _load(shared, wp.int32(qd_base) + k)
                    _store(shared, wp.int32(z_base) + wp.int32(_LANES) * position + lane, value)
                for reverse in range(path_length):
                    position = path_length - wp.int32(1) - reverse
                    value = _load(shared, wp.int32(z_base) + wp.int32(_LANES) * position + lane)
                    for later in range(position + wp.int32(1), path_length):
                        i = _load_u8(shared, wp.int32(path) + path_entry + later)
                        value -= _load(shared, _load_u16(shared, wp.int32(dof_entry) + i) + position) * _load(
                            shared, wp.int32(z_base) + wp.int32(_LANES) * later + lane
                        )
                    k = _load_u8(shared, wp.int32(path) + path_entry + position)
                    value /= _load(shared, _load_u16(shared, wp.int32(dof_entry) + k + wp.int32(1)) - wp.int32(1))
                    _store(shared, wp.int32(z_base) + wp.int32(_LANES) * position + lane, value)
                    inverse_mass += value * value
            if wp.static(not patch):
                _store_point_row_effective_mass(
                    bodies,
                    cc,
                    point_contact,
                    packed_articulation,
                    gathered_row,
                    body,
                    wrench,
                    wp.int32(-1),
                    wp.spatial_vector(),
                    inverse_mass,
                    prepare,
                )
        _store_u8(shared, wp.int32(leaf_byte) + lane, leaf + wp.int32(1))
        _sync_contact_warp()

        # A_rs = z_r . z_s over the common prefix of both root paths; lane s keeps column s.
        row = wp.int32(0)
        while row < row_count:
            end = run_end(shared, row, row_count)
            other = _load_u8(shared, wp.int32(leaf_byte) + row) - wp.int32(1)
            common = wp.int32(0)
            if other >= wp.int32(0) and leaf >= wp.int32(0):
                other_entry = _load_u16(shared, wp.int32(dof_entry) + other)
                common = common_prefix(
                    shared,
                    path_entry,
                    path_length,
                    other_entry,
                    _load_u16(shared, wp.int32(dof_entry) + other + wp.int32(1)) - other_entry,
                )
            for run_row in range(row, end):
                product = wp.float32(0.0)
                for position in range(common):
                    product += _load(shared, wp.int32(z_base) + wp.int32(_LANES) * position + run_row) * _load(
                        shared, wp.int32(z_base) + wp.int32(_LANES) * position + lane
                    )
                _store(shared, wp.int32(a_base) + wp.int32(_LANES) * run_row + lane, product)
            row = end
        _sync_contact_warp()

        # Biased solves restart from the accumulated impulses, relax solves add to them.
        if warm_start:
            if wp.static(patch):
                if lane < chunk_points:
                    lambda_row = cc_get_normal_lambda(cc, contact)
                elif lane < row_count:
                    tangent = lane - chunk_points
                    lambda_row = patch_lambda[packed_articulation, chunk[2] + (tangent >> wp.int32(1))][
                        tangent & wp.int32(1)
                    ]
            else:
                if lane < row_count:
                    row_contact = point_contact[packed_articulation, gathered_row // wp.int32(3)]
                    axis = gathered_row - wp.int32(3) * (gathered_row // wp.int32(3))
                    if axis == wp.int32(0):
                        lambda_row = cc_get_normal_lambda(cc, row_contact)
                    elif axis == wp.int32(1):
                        lambda_row = cc_get_tangent1_lambda(cc, row_contact)
                    else:
                        lambda_row = cc_get_tangent2_lambda(cc, row_contact)
            for r in range(row_count):
                velocity += _load(shared, wp.int32(a_base) + wp.int32(_LANES) * r + lane) * _lane_float(lambda_row, r)
        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if use_bias:
            _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
                DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, wp.float32(1.0) / idt
            )
        for sweep in range(sweeps):
            iteration = first_sweep + sweep
            for point_offset in range(chunk_points):
                point = wp.int32(point_offset)
                if (iteration & wp.int32(1)) != wp.int32(0):
                    point = chunk_points - wp.int32(1) - wp.int32(point_offset)
                if wp.static(patch):
                    row_velocity_n = _lane_float(velocity, point)
                    delta_n = wp.float32(0.0)
                    if lane == point:
                        update = _solve_patch_normal_row(
                            cc,
                            contact,
                            row_velocity_n,
                            _load(shared, wp.int32(a_base) + wp.int32(_LANES + 1) * point),
                            use_bias,
                            idt,
                            mass_coeff,
                            impulse_coeff,
                            sor_boost,
                        )
                        delta_n = update[0]
                        normal_load[packed_articulation, chunk[0] + point] = update[1]
                    delta_n = _lane_float(delta_n, point)
                    velocity += _load(shared, wp.int32(a_base) + wp.int32(_LANES) * point + lane) * delta_n
                    if lane == point:
                        lambda_row += delta_n
                else:
                    row = wp.int32(3) * point
                    row_velocities = wp.vec3(
                        _lane_float(velocity, row),
                        _lane_float(velocity, row + wp.int32(1)),
                        _lane_float(velocity, row + wp.int32(2)),
                    )
                    delta = wp.vec3(0.0)
                    if lane == point:
                        delta = _solve_reduced_contact_point(
                            columns,
                            cc,
                            contact,
                            column,
                            point_normal,
                            point_tangent,
                            row_velocities,
                            wp.vec3(
                                _load(shared, wp.int32(a_base) + wp.int32(_LANES) * row + row + wp.int32(1)),
                                _load(shared, wp.int32(a_base) + wp.int32(_LANES) * row + row + wp.int32(2)),
                                _load(
                                    shared,
                                    wp.int32(a_base) + wp.int32(_LANES) * (row + wp.int32(1)) + row + wp.int32(2),
                                ),
                            ),
                            use_bias,
                            idt,
                            mass_coeff,
                            impulse_coeff,
                            sor_boost,
                        )
                    delta = wp.vec3(
                        _lane_float(delta[0], point), _lane_float(delta[1], point), _lane_float(delta[2], point)
                    )
                    velocity += (
                        _load(shared, wp.int32(a_base) + wp.int32(_LANES) * row + lane) * delta[0]
                        + _load(shared, wp.int32(a_base) + wp.int32(_LANES) * (row + wp.int32(1)) + lane) * delta[1]
                        + _load(shared, wp.int32(a_base) + wp.int32(_LANES) * (row + wp.int32(2)) + lane) * delta[2]
                    )
                    if lane >= row and lane < row + wp.int32(3):
                        lambda_row += delta[lane - row]
            if wp.static(patch):
                # The chunk's columns' tangent rows, after the normals; they read the normal loads.
                _sync_contact_warp()
                point = chunk[0]
                while point < chunk[1] and point > wp.int32(0):
                    if point_column[packed_articulation, point] != point_column[packed_articulation, point - 1]:
                        break
                    point += wp.int32(1)
                ordinal = chunk[2]
                while point < chunk[1]:
                    patch_column = point_column[packed_articulation, point]
                    row = chunk_points + wp.int32(2) * (ordinal - chunk[2])
                    tangent_velocity = wp.vec2f(_lane_float(velocity, row), _lane_float(velocity, row + wp.int32(1)))
                    delta_t = wp.vec2f(0.0, 0.0)
                    if lane == wp.int32(0):
                        delta_t = _solve_patch_tangent_rows(
                            columns,
                            cc,
                            patch_column,
                            point,
                            point_total,
                            packed_articulation,
                            ordinal,
                            point_contact,
                            point_column,
                            normal_load,
                            patch_lambda,
                            tangent_velocity,
                            wp.vec3(
                                _load(shared, wp.int32(a_base) + wp.int32(_LANES + 1) * row),
                                _load(shared, wp.int32(a_base) + wp.int32(_LANES) * row + row + wp.int32(1)),
                                _load(shared, wp.int32(a_base) + wp.int32(_LANES + 1) * (row + wp.int32(1))),
                            ),
                            use_bias,
                            sor_boost,
                        )
                    delta_t = wp.vec2f(_lane_float(delta_t[0], 0), _lane_float(delta_t[1], 0))
                    velocity += (
                        _load(shared, wp.int32(a_base) + wp.int32(_LANES) * row + lane) * delta_t[0]
                        + _load(shared, wp.int32(a_base) + wp.int32(_LANES) * (row + wp.int32(1)) + lane) * delta_t[1]
                    )
                    if lane == row:
                        lambda_row += delta_t[0]
                    if lane == row + wp.int32(1):
                        lambda_row += delta_t[1]
                    ordinal += wp.int32(1)
                    while point < point_total and point_column[packed_articulation, point] == patch_column:
                        point += wp.int32(1)
        _store(shared, wp.int32(lambda_base) + lane, lambda_row)
        _sync_contact_warp()

        # M^-1 J^T lambda = L^-1 sum_r lambda_r z_r, gathered per run of rows.
        k = lane
        while k < n:
            entry = _load_u16(shared, wp.int32(dof_entry) + k)
            depth = _load_u16(shared, wp.int32(dof_entry) + k + wp.int32(1)) - wp.int32(1) - entry
            value = wp.float32(0.0)
            row = wp.int32(0)
            while row < row_count:
                end = run_end(shared, row, row_count)
                other = _load_u8(shared, wp.int32(leaf_byte) + row) - wp.int32(1)
                if other >= wp.int32(0):
                    other_entry = _load_u16(shared, wp.int32(dof_entry) + other)
                    if depth < _load_u16(shared, wp.int32(dof_entry) + other + wp.int32(1)) - other_entry:
                        if _load_u8(shared, wp.int32(path) + other_entry + depth) == k:
                            for run_row in range(row, end):
                                value += _load(shared, wp.int32(lambda_base) + run_row) * _load(
                                    shared, wp.int32(z_base) + wp.int32(_LANES) * depth + run_row
                                )
                row = end
            _store(shared, wp.int32(rhs_base) + k, value)
            k += wp.int32(_LANES)
        _sync_contact_warp()
        solve_lower(shared, wp.int32(rhs_base), n, lane)
        k = lane
        while k < n:
            rate = _load(shared, wp.int32(qd_base) + k) + _load(shared, wp.int32(rhs_base) + k)
            _store(shared, wp.int32(qd_base) + k, rate)
            joint_qd[dof_start + k] = rate
            k += wp.int32(_LANES)
        _sync_contact_warp()

    @wp.func
    def publish_rates(
        shared: wp.uint64,
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
        for depth in range(max_depth + wp.int32(1)):
            if tree.depth == depth:
                twist = joint_twist(shared, tree, dof_start)
                if tree.parent >= wp.int32(0):
                    twist += _load_motion(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * tree.parent)
                _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, twist)
            _sync_contact_warp()
        omega = wp.spatial_bottom(twist)
        velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
        # A floating tree gets its incoming linear momentum back through a uniform boost,
        # see ReducedPhoenXArticulation.end_substep.
        if update_configuration and not integrated and _is_floating(_lane_int(tree.kind, 0)):
            local_momentum = wp.vec4(0.0)
            if tree.joint >= wp.int32(0):
                local_momentum = wp.vec4(
                    tree.mass * velocity_com[0], tree.mass * velocity_com[1], tree.mass * velocity_com[2], tree.mass
                )
            total = _sum_reduced_group_vec4(local_momentum, wp.int32(_LANES), _FULL_MASK)
            if total[3] > wp.float32(0.0):
                boost = (target_linear_momentum - wp.vec3(total[0], total[1], total[2])) / total[3]
                velocity_com += boost
                if lane == wp.int32(0):
                    _boost_root_coordinates(
                        tree.joint, joint_qd_start, joint_x_p, motion_array, boost, joint_qd, joint_qd_public
                    )
        if tree.joint >= wp.int32(0):
            slot = tree.child + wp.int32(1)
            bodies.velocity[slot] = velocity_com
            bodies.angular_velocity[slot] = omega

    @wp.kernel(enable_backward=False, module=module, grid_stride=False)
    def dense_substep_kernel(
        bucket_articulation: wp.array[wp.int32],
        dt: wp.float32,
        sor_boost: wp.float32,
        iterations: wp.int32,
        contacts_active: wp.bool,
        reuse_factor: wp.bool,
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
        joint_target_ke: wp.array[wp.float32],
        joint_target_kd: wp.array[wp.float32],
        joint_target_mode: wp.array[wp.int32],
        joint_limit_lower: wp.array[wp.float32],
        joint_limit_upper: wp.array[wp.float32],
        joint_limit_ke: wp.array[wp.float32],
        joint_limit_kd: wp.array[wp.float32],
        joint_damping: wp.array[wp.float32],
        joint_armature: wp.array[wp.float32],
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
        bodies: BodyContainer,
    ):
        block = _block_index()
        lane = _lane_index()
        articulation = bucket_articulation[block]
        data = bodies.reduced
        joint_start = data.articulation_start[articulation]
        joint_end = data.articulation_end[articulation]
        joint_count = joint_end - joint_start
        dof_start = joint_qd_start[joint_start]
        n = joint_qd_start[joint_end] - dof_start
        template = dense.articulation_template[articulation]
        joint_base = template_joint_start[template]
        max_depth = template_max_depth[template]
        shared = shared_base()
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
            body_inertia,
            joint_parent_local,
            joint_depth,
        )

        # Frames, implicit joint operator, motion subspaces, and spatial inertias at the coordinates.
        # A biased step that follows a relax step at the same pose starts from its frames and factor.
        local = wp.transform_identity()
        origin = wp.vec3(0.0)
        if reuse_factor:
            if tree.joint >= wp.int32(0):
                local = body_q_local[tree.child]
            origin = articulation_origin[articulation]
        else:
            local, origin = local_kinematics(
                shared,
                tree,
                lane,
                articulation,
                max_depth,
                joint_axis,
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
            configure_implicit_joint_dynamics(
                tree.joint,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_target_q_start,
                joint_q,
                joint_qd_public,
                joint_target_q,
                joint_target_qd,
                joint_target_ke,
                joint_target_kd,
                joint_target_mode,
                joint_limit_lower,
                joint_limit_upper,
                joint_limit_ke,
                joint_limit_kd,
                joint_damping,
                joint_armature,
                dt,
                factor_diagonal,
                implicit_force,
            )
            for offset in range(6):
                if wp.int32(offset) < tree.dof_count:
                    if factor_diagonal[tree.qd_start + offset] != previous_diagonal[offset]:
                        diagonal_changed = wp.bool(True)
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
            for page in range((total_points + wp.int32(_POINTS_PER_PAGE - 1)) // wp.int32(_POINTS_PER_PAGE)):
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
                chunks = (page_points + wp.int32(_CHUNK_POINTS - 1)) // wp.int32(_CHUNK_POINTS)
                # Several chunks first apply every chunk's accumulated impulses, then sweep in turn.
                rounds = wp.int32(1)
                sweeps = iterations
                warm_rounds = wp.int32(0)
                if chunks > wp.int32(1):
                    rounds = iterations
                    sweeps = wp.int32(1)
                    if wp.static(not relax):
                        warm_rounds = wp.int32(1)
                for round_offset in range(rounds + warm_rounds):
                    round_index = round_offset - warm_rounds
                    for chunk_offset in range(chunks):
                        chunk_index = wp.int32(chunk_offset)
                        if (round_index & wp.int32(1)) != wp.int32(0):
                            chunk_index = chunks - wp.int32(1) - wp.int32(chunk_offset)
                        first = wp.int32(_CHUNK_POINTS) * chunk_index
                        last = wp.min(first + wp.int32(_CHUNK_POINTS), page_points)
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
                            wp.where(round_index < wp.int32(0), wp.int32(0), sweeps),
                            wp.bool(not relax),
                            wp.bool(not relax) and round_offset == wp.int32(0),
                            round_offset == wp.int32(0),
                            point_count,
                            point_contact,
                            point_column,
                            normal,
                            tangent0,
                            row_body,
                            row_wrench,
                            row_velocity,
                            patch_lambda,
                            normal_load,
                            joint_qd,
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
        for depth in range(max_depth + wp.int32(1)):
            if tree.depth == depth:
                rates = joint_twist(shared, tree, dof_start)
                twist = rates
                if tree.parent >= wp.int32(0):
                    twist += _load_motion(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * tree.parent)
                    coriolis = _load_motion(shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * tree.parent)
                elif _is_floating(tree.kind):
                    root_velocity = twist
                coriolis += spatial_cross(twist, rates)
                _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, twist)
                _store_spatial(shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * lane, coriolis)
            _sync_contact_warp()
        momentum = wp.spatial_vector()
        if tree.joint >= wp.int32(0):
            inertia = _unpack_symmetric_mat66(
                _pack_transformed_body_inertia(local * body_x_com[tree.child], tree.mass, tree.inertia)
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
        drift = wp.float32(0.0)
        energy = wp.float32(0.0)
        dissipation = wp.float32(0.0)
        velocity_com = wp.vec3(0.0)
        omega = wp.vec3(0.0)
        for depth in range(max_depth + wp.int32(1)):
            if tree.depth == depth:
                new_twist = wp.spatial_vector()
                child_acceleration = wp.spatial_vector()
                parent_rate = wp.spatial_vector()
                if tree.parent >= wp.int32(0):
                    new_twist = _load_motion(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * tree.parent)
                    child_acceleration = _load_motion(
                        shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * tree.parent
                    )
                    parent_rate = _load_motion(shared, wp.int32(third_base) + wp.int32(_MOTION_STRIDE) * tree.parent)
                joint_velocity = wp.spatial_vector()
                joint_acceleration = wp.spatial_vector()
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
                    new_twist += subspace * rate
                child_acceleration += joint_acceleration
                terms = _explicit_step_drift_terms(
                    new_twist - dt * child_acceleration,
                    child_acceleration,
                    joint_velocity,
                    joint_acceleration,
                    depth == wp.int32(0),
                    frame,
                    momentum,
                    tree.mass,
                    x_com,
                    parent_rate,
                )
                drift += terms[6]
                energy += terms[7]
                _store_spatial(shared, wp.int32(first_base) + wp.int32(_MOTION_STRIDE) * lane, new_twist)
                _store_spatial(shared, wp.int32(second_base) + wp.int32(_MOTION_STRIDE) * lane, child_acceleration)
                _store_spatial(
                    shared,
                    wp.int32(third_base) + wp.int32(_MOTION_STRIDE) * lane,
                    wp.spatial_vector(terms[0], terms[1], terms[2], terms[3], terms[4], terms[5]),
                )
                body_velocity[tree.child] = new_twist
                omega = wp.spatial_bottom(new_twist)
                velocity_com = wp.spatial_top(new_twist) + wp.cross(omega, x_com)
            _sync_contact_warp()
        linear = wp.spatial_top(momentum)
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

        # Integrate the coordinates and pose the links root to leaf.
        for depth in range(max_depth + wp.int32(1)):
            if tree.depth == depth:
                parent_anchor = tree.x_p
                if tree.parent >= wp.int32(0):
                    parent_anchor = (
                        _load_transform(shared, wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * tree.parent) * tree.x_p
                    )
                world = wp.transform_identity()
                if _is_floating(tree.kind) and not integrated:
                    com_twist = wp.spatial_vector(velocity_com, omega)
                    if tree.parent < wp.int32(0):
                        # Move the root COM with its re-referenced velocity, see _carry_floating_root_twist.
                        com_twist = _transport_root_twist(com_twist, dt)
                    old = wp.transform(origin + wp.transform_get_translation(local), wp.transform_get_rotation(local))
                    world = integrate_body_pose_from_com_twist(old, tree.com, com_twist, dt)
                    joint_transform = wp.transform_inverse(parent_anchor) * world * tree.x_c
                    position = wp.transform_get_translation(joint_transform)
                    rotation = wp.transform_get_rotation(joint_transform)
                    for component in range(3):
                        joint_q[tree.q_start + component] = position[component]
                    for component in range(4):
                        joint_q[tree.q_start + 3 + component] = rotation[component]
                else:
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
                    world = (
                        parent_anchor
                        * jcalc_transform(
                            tree.kind, joint_axis, tree.qd_start, tree.lin, tree.ang, joint_q, tree.q_start
                        )
                        * wp.transform_inverse(tree.x_c)
                    )
                _store_transform(shared, wp.int32(pose_base) + wp.int32(_MOTION_STRIDE) * lane, world)
                body_q[tree.child] = world
                slot = tree.child + wp.int32(1)
                rotation = wp.transform_get_rotation(world)
                bodies.position[slot] = wp.transform_get_translation(world) + wp.quat_rotate(rotation, tree.com)
                bodies.orientation[slot] = rotation
            _sync_contact_warp()

        # Frames and motion subspaces at the new pose, then the rates. A floating root's
        # rates still read the motion subspaces of the start pose.
        root_twist = joint_twist(shared, tree, dof_start)
        local, origin = local_kinematics(
            shared,
            tree,
            lane,
            articulation,
            max_depth,
            joint_axis,
            joint_q,
            body_x_com,
            articulation_origin,
            body_q_local,
            body_q_com,
            joint_anchor_local,
        )
        if tree.joint >= wp.int32(0):
            jcalc_motion(
                tree.kind,
                joint_axis,
                joint_q,
                tree.lin,
                tree.ang,
                joint_anchor_local[tree.joint],
                joint_qd,
                tree.q_start,
                tree.qd_start,
                joint_s_publish,
            )
            for offset in range(tree.dof_count):
                local_dof = tree.qd_start + offset - dof_start
                _store_spatial(
                    shared,
                    wp.int32(motion_base) + wp.int32(_MOTION_STRIDE) * local_dof,
                    joint_s_publish[tree.qd_start + offset],
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
    contact = articulation.contact_block_system
    if contact.requires_impulse_response:
        reasons.append("contacts may couple articulations to dynamic bodies or to themselves")
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
    if any(int(joint_parent[int(first)]) >= 0 for first in start):
        reasons.append("roots attached to a body")
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
        self.dense = self.system.dense
        self.contact = articulation.contact_block_system

    def _launch(self, world, dt: float, *, relax: bool, reuse_factor: bool = False) -> None:
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
                    model.joint_target_ke,
                    model.joint_target_kd,
                    model.joint_target_mode,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    model.joint_damping,
                    model.joint_armature,
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
                    articulation.bodies,
                ],
                device=model.device,
            )

    def biased(self, world, dt: float, *, reuse_factor: bool) -> None:
        """Advance every tree through the biased solve, the velocity-product step, and publication.

        ``reuse_factor`` starts from the frames and factor of a relax step at the same pose;
        trees whose implicit joint diagonal changed refactor.
        """
        self._launch(world, dt, relax=False, reuse_factor=reuse_factor)

    def relax(self, world, dt: float) -> None:
        """Refactor at the integrated pose, apply the bias-free contact correction, and store the factor."""
        self._launch(world, dt, relax=True)


__all__ = ["PAGE_ROWS", "DenseSubstep", "dense_substep_blockers", "make_dense_substep_kernel"]
