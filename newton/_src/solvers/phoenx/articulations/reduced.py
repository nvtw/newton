# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear-time articulated-body factorization for PhoenX."""

from __future__ import annotations

import copy
import functools

import numpy as np
import warp as wp

from newton._src.sim import BodyFlags, Control, JointTargetMode, JointType, Model, State
from newton._src.sim.articulation import eval_fk
from newton._src.solvers.featherstone.kernels import (
    compute_com_transforms,
    integrate_body_pose_from_com_twist,
    jcalc_integrate,
    jcalc_motion,
    jcalc_transform,
    spatial_cross,
    spatial_cross_dual,
)
from newton._src.solvers.phoenx.articulations.dense_tree import DenseTreeOperator, dense_tree_solve
from newton._src.solvers.phoenx.articulations.reduced_contact import (
    flush_deferred_articulation,
    reduced_contact_iterate,
    reduced_contact_prepare,
)
from newton._src.solvers.phoenx.articulations.reduced_contact_block import (
    _POINTS_PER_PAGE,
    _SOLVE_GENERALIZED_CONTACT_TILE_DEVICES,
    ReducedContactBlockSystem,
    _finalize_reduced_contact_rows_device,
    _model_pair_endpoints,
    _sync_contact_warp,
    shape_pairs_are_dense,
)
from newton._src.solvers.phoenx.articulations.reduced_loop import ReducedLoopSystem
from newton._src.solvers.phoenx.body import (
    MOTION_ARTICULATED,
    BodyContainer,
    ReducedArticulationData,
    dense_tree_data_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.semi_implicit.kernels_body import joint_force

_MAX_JOINT_DOF = 6
# A tree takes the variational midpoint step once the explicit step's secular
# energy drift, dt^2 v^T C(v, a) with the velocity products' bilinear form C, see
# _explicit_step_drift_terms, exceeds this fraction of its kinetic energy per
# second in either direction. The drift is exact for a free rigid body and
# includes the coupling of chains, whose links can each look slow while the
# chain gains energy. A 1%/s budget, 5e-5 per step at 200 Hz, keeps free-flying
# ropes within about 1% over 30 steps. The budget also grants the energy the
# tree's joint damping dissipates over the step, see _explicit_step_drifts;
# contact dissipation is not counted.
_MIDPOINT_ENERGY_RATE = wp.constant(0.01)
# The midpoint step's fixed point contracts by roughly the fastest tumbling
# rate times dt per iteration (about 0.3 at 0.6 rad per step, 0.5 at 1.2), so it
# stops once the momentum residual's energy norm falls below this fraction of
# twice the kinetic energy (a 1e-4 relative residual) or the iteration count
# reaches the cap. A 6-link chain tumbling at 0.6 rad per step keeps its
# energy to 0.1% over 1.5 s at that residual, against 1.3% at 1e-3.
_MIDPOINT_TOLERANCE = wp.constant(1.0e-8)
_MIDPOINT_MAX_ITERATIONS = 16
# A substep whose fixed point does not reach the tolerance is retried at half
# length, at most this many times (down to dt / 16). The step equations of long
# chains of thin links lose their nearby solution at moderate rates: the lever
# arms turn within the step, which couples the links' axial spins (tiny inertia)
# to the chain's bending. A free 14-link capsule rope fails at 0.15 to 0.3 rad per
# step, where even an exact Newton iteration diverges, and accepting the stalled
# iterate gains tens of percent of energy per step. Its half step converges.
_MIDPOINT_MAX_HALVINGS = wp.constant(4)
# Float32 COM positions, with a few roundings each, bound how closely the secant
# momenta can meet the start momenta; see _midpoint_momentum_step.
_MIDPOINT_ROUNDING = wp.constant(8.0 * 2.0**-24)
_WARP_FACTOR_MIN_ARTICULATIONS = 32
_WARP_FACTOR_MIN_DEPTH_LEVELS = 8
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_vec8 = wp.types.vector(length=8, dtype=wp.float32)
_sym_mat66 = wp.types.vector(length=21, dtype=wp.float32)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def _pack_symmetric_mat66(matrix: wp.spatial_matrix) -> _sym_mat66:
    packed = _sym_mat66(0.0)
    index = wp.int32(0)
    for row in range(6):
        for column in range(row, 6):
            packed[index] = matrix[row, column]
            index += wp.int32(1)
    return packed


@wp.func
def _pack_transformed_body_inertia(
    transform: wp.transform,
    mass: wp.float32,
    inertia: wp.mat33,
) -> _sym_mat66:
    """Transform and pack a COM-centered rigid-body inertia."""
    position = wp.transform_get_translation(transform)
    rotation = wp.transform_get_rotation(transform)
    local_x = wp.quat_rotate_inv(rotation, wp.vec3(1.0, 0.0, 0.0))
    local_y = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 1.0, 0.0))
    local_z = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, 1.0))
    inertia_x = inertia * local_x
    inertia_y = inertia * local_y
    inertia_z = inertia * local_z
    i_xx = wp.dot(local_x, inertia_x)
    i_xy = wp.dot(local_x, inertia_y)
    i_xz = wp.dot(local_x, inertia_z)
    i_yy = wp.dot(local_y, inertia_y)
    i_yz = wp.dot(local_y, inertia_z)
    i_zz = wp.dot(local_z, inertia_z)
    x = position[0]
    y = position[1]
    z = position[2]

    packed = _sym_mat66(0.0)
    packed[0] = mass
    packed[3] = wp.float32(0.0)
    packed[4] = mass * z
    packed[5] = -mass * y
    packed[6] = mass
    packed[8] = -mass * z
    packed[9] = wp.float32(0.0)
    packed[10] = mass * x
    packed[11] = mass
    packed[12] = mass * y
    packed[13] = -mass * x
    packed[14] = wp.float32(0.0)
    packed[15] = i_xx + mass * (y * y + z * z)
    packed[16] = i_xy - mass * x * y
    packed[17] = i_xz - mass * x * z
    packed[18] = i_yy + mass * (x * x + z * z)
    packed[19] = i_yz - mass * y * z
    packed[20] = i_zz + mass * (x * x + y * y)
    return packed


@wp.func
def _unpack_symmetric_mat66(packed: _sym_mat66) -> wp.spatial_matrix:
    matrix = wp.spatial_matrix()
    index = wp.int32(0)
    for row in range(6):
        for column in range(row, 6):
            value = packed[index]
            matrix[row, column] = value
            matrix[column, row] = value
            index += wp.int32(1)
    return matrix


@wp.func
def _invert_spd(matrix: _mat66, size: wp.int32) -> _mat66:
    factor = _mat66(0.0)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < size:
            for col in range(_MAX_JOINT_DOF):
                if wp.int32(col) <= wp.int32(row) and wp.int32(col) < size:
                    value = matrix[row, col]
                    for inner in range(_MAX_JOINT_DOF):
                        if wp.int32(inner) < wp.int32(col):
                            value -= factor[row, inner] * factor[col, inner]
                    if row == col:
                        factor[row, col] = wp.sqrt(wp.max(value, wp.float32(1.0e-20)))
                    else:
                        factor[row, col] = value / factor[col, col]
        else:
            factor[row, row] = wp.float32(1.0)

    inverse = _mat66(0.0)
    for column in range(_MAX_JOINT_DOF):
        if wp.int32(column) < size:
            forward = _vec6(0.0)
            solution = _vec6(0.0)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < size:
                    value = wp.float32(1.0) if row == column else wp.float32(0.0)
                    for col in range(_MAX_JOINT_DOF):
                        if wp.int32(col) < wp.int32(row):
                            value -= factor[row, col] * forward[col]
                    forward[row] = value / factor[row, row]
            for reverse in range(_MAX_JOINT_DOF):
                row = _MAX_JOINT_DOF - 1 - reverse
                if wp.int32(row) < size:
                    value = forward[row]
                    for col in range(_MAX_JOINT_DOF):
                        if wp.int32(col) > wp.int32(row) and wp.int32(col) < size:
                            value -= factor[col, row] * solution[col]
                    solution[row] = value / factor[row, row]
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < size:
                    inverse[row, column] = solution[row]
    return inverse


@wp.struct
class _MidpointWorkspace:
    """Model data and scratch for the variational midpoint step of fast trees.

    ``flag`` records per articulation whether the last velocity-product
    advance already integrated the tree, so publication must only rebuild its
    poses from ``joint_q``.
    """

    flag: wp.array[wp.int32]
    joint_q: wp.array[wp.float32]
    joint_qd_public: wp.array[wp.float32]
    joint_q_start: wp.array[wp.int32]
    joint_x_p: wp.array[wp.transform]
    joint_x_c: wp.array[wp.transform]
    joint_axis: wp.array[wp.vec3]
    joint_dof_dim: wp.array2d[wp.int32]
    joint_armature: wp.array[wp.float32]
    joint_factor_diagonal: wp.array[wp.float32]
    body_com: wp.array[wp.vec3]
    body_x_com: wp.array[wp.transform]
    body_mass: wp.array[wp.float32]
    body_inertia: wp.array[wp.mat33]
    body_inv_inertia: wp.array[wp.mat33]
    articulation_origin: wp.array[wp.vec3]
    body_q_local: wp.array[wp.transform]
    joint_anchor_local: wp.array[wp.transform]
    child_start: wp.array[wp.int32]
    child_joint: wp.array[wp.int32]
    zero_dof: wp.array[wp.float32]
    body_pose: wp.array[wp.transform]
    body_articulated: wp.array[wp.spatial_matrix]
    body_reduced: wp.array[wp.spatial_matrix]
    body_momentum: wp.array[wp.spatial_vector]
    body_force: wp.array[wp.spatial_vector]
    joint_q_mid: wp.array[wp.float32]
    joint_qd_mid: wp.array[wp.float32]
    joint_s_mid: wp.array[wp.spatial_vector]
    joint_u: wp.array[wp.spatial_vector]
    joint_d_inv: wp.array2d[wp.float32]
    joint_force: wp.array[wp.float32]
    joint_reduced: wp.array[wp.float32]
    joint_step: wp.array[wp.float32]
    joint_momentum: wp.array[wp.float32]


@wp.struct
class _MidpointTree:
    """One tree's lane group: its depth walk and this thread's lane in it."""

    articulation: wp.int32
    lane: wp.int32
    width: wp.int32
    mask: wp.uint32
    max_depth: wp.int32
    depth_start: wp.array2d[wp.int32]
    order: wp.array[wp.int32]


@wp.func
def _floating_root_frame(root_velocity: wp.spatial_vector) -> wp.spatial_vector:
    """Return the Galilean frame of a floating root's step: its velocity ``v`` and ``omega x v``.

    ABA steps twists about the fixed articulation origin, where the root's
    spatial acceleration includes ``-omega x v`` even for torque-free uniform
    motion; :func:`_carry_floating_root_twist` and the momentum boost undo that
    part, so the explicit step's energy drift is measured in the frame moving
    with ``v``: there, every twist loses ``v`` and every acceleration gains
    ``omega x v``. A zero twist (fixed roots) leaves the frame at rest.
    """
    linear = wp.spatial_top(root_velocity)
    return wp.spatial_vector(linear, wp.cross(wp.spatial_bottom(root_velocity), linear))


@wp.func
def _explicit_step_drift_terms(
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    joint_velocity: wp.spatial_vector,
    joint_acceleration: wp.spatial_vector,
    is_root: wp.bool,
    frame: wp.spatial_vector,
    momentum: wp.spatial_vector,
    mass: wp.float32,
    com: wp.vec3,
    parent_coriolis_rate: wp.spatial_vector,
) -> _vec8:
    """Return one body's terms of the explicit step's secular energy drift.

    The explicit step kicks the rates by ``dt a`` with ``a = -M^-1 c(v)`` at the
    start pose, then moves the pose at the new rates. Expanding the energy at
    the end pose, the first-order terms cancel because the exact flow conserves
    energy, and the second-order terms are ``dt^2 v^T C(v, a)`` plus a total
    time derivative that only oscillates. ``C`` is the symmetric bilinear form
    of the velocity products, ``c(v) = C(v, v)``. Over the bodies,
    ``2 v^T C(v, a) = sum h . k - A . (V x* h)`` with twist ``V``, momentum
    ``h = I V``, acceleration ``A = J a`` and ``k``, the Coriolis acceleration's
    derivative along ``a``, accumulated root to leaf by
    ``k = k_parent + A x (S v) + V x (S a)``. For a free rigid body the drift is
    exactly the explicit step's energy gain ``dt^2 / 2 alpha^T I alpha``. Unlike
    an estimate from each link's gyroscopic frequency it includes the chain's
    coupling, and its sign tells a gaining step from a dissipating one.

    ``momentum`` is the incoming ``I V`` about the origin, ``mass`` and ``com``
    shift it into the moving frame.

    Returns ``k`` (components 0-5), this body's share of ``2 v^T C(v, a)``
    (component 6) and of the kinetic energy (component 7), all in the frame of
    :func:`_floating_root_frame`.
    """
    shift_velocity = wp.spatial_vector(wp.spatial_top(frame), wp.vec3(0.0))
    shift_acceleration = wp.spatial_vector(wp.spatial_bottom(frame), wp.vec3(0.0))
    velocity = velocity - shift_velocity
    acceleration = acceleration + shift_acceleration
    if is_root:
        joint_velocity = joint_velocity - shift_velocity
        joint_acceleration = joint_acceleration + shift_acceleration
    coriolis_rate = (
        parent_coriolis_rate + spatial_cross(acceleration, joint_velocity) + spatial_cross(velocity, joint_acceleration)
    )
    # The body's momentum I V in the moving frame: the frame velocity (u, 0) carries m u
    # and, about the origin, m com x u.
    frame_linear = wp.spatial_top(frame)
    momentum = momentum - wp.spatial_vector(mass * frame_linear, mass * wp.cross(com, frame_linear))
    drift = wp.dot(momentum, coriolis_rate) - wp.dot(acceleration, spatial_cross_dual(velocity, momentum))
    return _vec8(
        coriolis_rate[0],
        coriolis_rate[1],
        coriolis_rate[2],
        coriolis_rate[3],
        coriolis_rate[4],
        coriolis_rate[5],
        drift,
        wp.float32(0.5) * wp.dot(velocity, momentum),
    )


@wp.func
def _joint_dissipation(ws: _MidpointWorkspace, dof: wp.int32, rate: wp.float32) -> wp.float32:
    """Return the energy one rate's implicit joint damping removes over the step, ``dt c qd^2``.

    The factorization adds ``dt c + dt^2 k`` to the armature for drive, limit
    and passive damping ``c`` and stiffness ``k``; implicit Euler damps the
    spring by ``dt k`` on top of ``c``, so the step's effective damping is
    ``c + dt k`` and its dissipation over ``dt`` is that diagonal times
    ``qd^2``.
    """
    return (ws.joint_factor_diagonal[dof] - ws.joint_armature[dof]) * rate * rate


@wp.func
def _explicit_step_drifts(
    drift: wp.float32, kinetic_energy: wp.float32, dissipation: wp.float32, dt: wp.float32
) -> wp.bool:
    """Return whether the explicit step's energy drift ``dt^2 / 2 * drift`` exceeds its budget.

    The budget is ``_MIDPOINT_ENERGY_RATE`` of the kinetic energy per second plus
    the energy the tree's joint damping dissipates over the step: while that
    physical dissipation dominates, the tree's energy still decreases and the
    numerical drift stays bounded by the physics. Contacts, which also
    dissipate, are left out. Both signs count: mixing explicit steps into the
    energy-conserving midpoint steps rectifies the explicit step's oscillating
    energy error, so a switch that only bounds gains still let tumbling ropes
    lose 57% or gain 23x.
    """
    return wp.float32(0.5) * dt * dt * wp.abs(drift) > _MIDPOINT_ENERGY_RATE * dt * kinetic_energy + dissipation


@wp.func
def _restore_incoming_rates(
    tree: _MidpointTree,
    joint_qd_start: wp.array[wp.int32],
    ws: _MidpointWorkspace,
    joint_qd: wp.array[wp.float32],
):
    """Undo a tree's explicit step from the incoming rates it kept in ``ws.joint_qd_mid``."""
    index = tree.depth_start[tree.articulation, 0] + tree.lane
    end = tree.depth_start[tree.articulation, tree.max_depth + wp.int32(1)]
    while index < end:
        joint = tree.order[index]
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
            joint_qd[dof] = ws.joint_qd_mid[dof]
        index += tree.width
    _sync_reduced_group(tree.mask)


@wp.func
def _rotate_by(rotation: wp.quat, omega: wp.vec3, dt: wp.float32) -> wp.quat:
    """Turn ``rotation`` by the world rotation vector ``omega dt`` (exponential map)."""
    rate = wp.length(omega)
    if rate * dt < wp.float32(1.0e-8):
        return rotation
    return wp.normalize(wp.quat_from_axis_angle(omega / rate, rate * dt) * rotation)


@wp.func
def _midpoint_rotational_momentum(rotation0: wp.quat, rotation1: wp.quat, inertia: wp.mat33, dt: wp.float32) -> wp.vec3:
    """Return ``I(R_half) omega`` for the geodesic from ``rotation0`` to ``rotation1``.

    ``omega`` is the constant world rate of that geodesic over ``dt`` and
    ``R_half`` its midpoint, the momentum of the midpoint discrete Lagrangian.
    It is odd under swapping the endpoints, which makes the step symmetric.
    """
    delta = rotation1 * wp.quat_inverse(rotation0)
    if delta[3] < wp.float32(0.0):
        delta = -delta
    vector = wp.vec3(delta[0], delta[1], delta[2])
    sine = wp.length(vector)
    omega = vector * (wp.float32(2.0) / dt)
    if sine > wp.float32(1.0e-8):
        omega = vector * (wp.float32(2.0) * wp.atan2(sine, delta[3]) / (sine * dt))
    half = wp.normalize(delta + wp.quat_identity()) * rotation0
    return wp.quat_rotate(half, inertia * wp.quat_rotate_inv(half, omega))


@wp.func
def _midpoint_gather_momentum(tree: _MidpointTree, joint_child: wp.array[wp.int32], ws: _MidpointWorkspace):
    """Turn each body's ``ws.body_momentum`` into its subtree sum."""
    for reverse_depth in range(tree.max_depth + wp.int32(1)):
        depth = tree.max_depth - reverse_depth
        index = tree.depth_start[tree.articulation, depth] + tree.lane
        depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = tree.order[index]
            child = joint_child[joint]
            momentum = ws.body_momentum[child]
            for child_index in range(ws.child_start[joint], ws.child_start[joint + wp.int32(1)]):
                momentum += ws.body_momentum[joint_child[ws.child_joint[child_index]]]
            ws.body_momentum[child] = momentum
            index += tree.width
        _sync_reduced_group(tree.mask)


@wp.func
def _midpoint_pose_pass(
    tree: _MidpointTree,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    ws: _MidpointWorkspace,
    dt: wp.float32,
    out_q: wp.array[wp.float32],
    final: wp.bool,
    body_velocity: wp.array[wp.spatial_vector],
):
    """Move a tree from its start pose by ``dt`` at the rates ``ws.joint_qd_mid``.

    Floating roots and BALL children turn by their own world angular velocity,
    the way a free rigid body does; composing per-joint relative rates instead
    adds ``dt^2 omega_parent x omega_rel`` errors once neighbours spin at
    different rates. Other joints advance their own coordinates. Each body
    then stores the subtree sum of its step momentum in ``ws.body_momentum``:
    the secant linear momentum and the midpoint rotational momentum, as a
    spatial momentum about the articulation origin with the moment arm at the
    start (``final`` false) or end (``final`` true) pose. ``final`` also writes
    the end pose's motion subspaces and rigid inertias for its factorization.
    """
    for depth in range(tree.max_depth + wp.int32(1)):
        index = tree.depth_start[tree.articulation, depth] + tree.lane
        depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = tree.order[index]
            parent = joint_parent[joint]
            child = joint_child[joint]
            kind = joint_type[joint]
            dof_start = joint_qd_start[joint]
            coord_start = ws.joint_q_start[joint]
            linear_count = ws.joint_dof_dim[joint, 0]
            angular_count = ws.joint_dof_dim[joint, 1]
            joint_x_c = ws.joint_x_c[joint]
            twist = wp.spatial_vector()
            if parent >= wp.int32(0):
                twist = body_velocity[parent]
            for dof in range(dof_start, joint_qd_start[joint + wp.int32(1)]):
                twist += joint_s[dof] * ws.joint_qd_mid[dof]
            body_velocity[child] = twist
            omega = wp.spatial_bottom(twist)
            base = ws.body_q_local[child]
            com = ws.body_com[child]
            com0 = wp.transform_point(base, com)
            anchor = ws.joint_anchor_local[joint]
            if parent >= wp.int32(0):
                anchor = ws.body_pose[parent] * ws.joint_x_p[joint]
            pose = base
            if parent < wp.int32(0) and (kind == JointType.FREE or kind == JointType.DISTANCE):
                rotation = _rotate_by(wp.transform_get_rotation(base), omega, dt)
                moved_com = com0 + dt * (wp.spatial_top(twist) + wp.cross(omega, com0))
                pose = wp.transform(moved_com - wp.quat_rotate(rotation, com), rotation)
            elif kind == JointType.BALL:
                rotation = _rotate_by(wp.transform_get_rotation(base), omega, dt)
                child_anchor_rotation = rotation * wp.transform_get_rotation(joint_x_c)
                pose = wp.transform(wp.transform_get_translation(anchor), child_anchor_rotation) * wp.transform_inverse(
                    joint_x_c
                )
                relative = wp.quat_inverse(wp.transform_get_rotation(anchor)) * child_anchor_rotation
                out_q[coord_start] = relative[0]
                out_q[coord_start + wp.int32(1)] = relative[1]
                out_q[coord_start + wp.int32(2)] = relative[2]
                out_q[coord_start + wp.int32(3)] = relative[3]
            else:
                jcalc_integrate(
                    parent,
                    joint_x_c,
                    com,
                    kind,
                    ws.joint_q,
                    ws.joint_qd_mid,
                    ws.zero_dof,
                    coord_start,
                    dof_start,
                    linear_count,
                    angular_count,
                    dt,
                    out_q,
                    ws.joint_qd_mid,
                )
                joint_transform = jcalc_transform(
                    kind, ws.joint_axis, dof_start, linear_count, angular_count, out_q, coord_start
                )
                pose = anchor * joint_transform * wp.transform_inverse(joint_x_c)
            ws.body_pose[child] = pose

            com1 = wp.transform_point(pose, com)
            mass = ws.body_mass[child]
            linear = (mass / dt) * (com1 - com0)
            angular = _midpoint_rotational_momentum(
                wp.transform_get_rotation(base), wp.transform_get_rotation(pose), ws.body_inertia[child], dt
            )
            arm = com0
            if final:
                arm = com1
                # Runtime slots pad each joint's rates to a fixed width. The padding has no
                # motion, whatever template used the slot before.
                for dof in range(dof_start + linear_count + angular_count, joint_qd_start[joint + wp.int32(1)]):
                    ws.joint_s_mid[dof] = wp.spatial_vector()
                jcalc_motion(
                    kind,
                    ws.joint_axis,
                    out_q,
                    linear_count,
                    angular_count,
                    anchor,
                    ws.joint_qd_mid,
                    coord_start,
                    dof_start,
                    ws.joint_s_mid,
                )
                ws.body_articulated[child] = _unpack_symmetric_mat66(
                    _pack_transformed_body_inertia(pose * ws.body_x_com[child], mass, ws.body_inertia[child])
                )
            ws.body_momentum[child] = wp.spatial_vector(linear, angular + wp.cross(arm, linear))
            index += tree.width
        _sync_reduced_group(tree.mask)
    _midpoint_gather_momentum(tree, joint_child, ws)


@wp.func
def _midpoint_factor(
    tree: _MidpointTree,
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    ws: _MidpointWorkspace,
):
    """Factor ``M + armature`` by ABA from the rigid inertias in ``ws.body_articulated``."""
    for reverse_depth in range(tree.max_depth + wp.int32(1)):
        depth = tree.max_depth - reverse_depth
        index = tree.depth_start[tree.articulation, depth] + tree.lane
        depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = tree.order[index]
            child = joint_child[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_qd_start[joint + wp.int32(1)] - dof_start
            articulated = ws.body_articulated[child]
            for child_index in range(ws.child_start[joint], ws.child_start[joint + wp.int32(1)]):
                articulated += ws.body_reduced[joint_child[ws.child_joint[child_index]]]
            u_columns = _mat66(0.0)
            d = _mat66(0.0)
            for column in range(_MAX_JOINT_DOF):
                if wp.int32(column) < dof_count:
                    dof = dof_start + wp.int32(column)
                    u = articulated * joint_s[dof]
                    ws.joint_u[dof] = u
                    for row in range(6):
                        u_columns[row, column] = u[row]
                    d[column, column] = ws.joint_armature[dof]
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    s = joint_s[dof_start + wp.int32(row)]
                    for column in range(_MAX_JOINT_DOF):
                        if wp.int32(column) < dof_count:
                            value = wp.float32(0.0)
                            for entry in range(6):
                                value += s[entry] * u_columns[entry, column]
                            d[row, column] = d[row, column] + value
            d_inv = _invert_spd(d, dof_count)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    for column in range(_MAX_JOINT_DOF):
                        ws.joint_d_inv[dof_start + wp.int32(row), column] = d_inv[row, column]
            # What the parent sees of this subtree: A - U D^-1 U^T.
            gain = u_columns * d_inv
            for row in range(6):
                for column in range(6):
                    value = wp.float32(0.0)
                    for dof in range(_MAX_JOINT_DOF):
                        if wp.int32(dof) < dof_count:
                            value += gain[row, dof] * u_columns[column, dof]
                    articulated[row, column] = articulated[row, column] - value
            ws.body_reduced[child] = articulated
            index += tree.width
        _sync_reduced_group(tree.mask)


@wp.func
def _midpoint_solve(
    tree: _MidpointTree,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    ws: _MidpointWorkspace,
    out: wp.array[wp.float32],
) -> wp.float32:
    """Solve the factored system for the generalized force in ``ws.joint_force``.

    Writes the solution ``x`` to ``out``, leaves its body twists in
    ``ws.body_force``, and returns ``x . f`` to every lane of the tree.
    """
    for reverse_depth in range(tree.max_depth + wp.int32(1)):
        depth = tree.max_depth - reverse_depth
        index = tree.depth_start[tree.articulation, depth] + tree.lane
        depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = tree.order[index]
            child = joint_child[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_qd_start[joint + wp.int32(1)] - dof_start
            bias = wp.spatial_vector()
            for child_index in range(ws.child_start[joint], ws.child_start[joint + wp.int32(1)]):
                bias += ws.body_force[joint_child[ws.child_joint[child_index]]]
            propagated = bias
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    dof = dof_start + wp.int32(row)
                    ws.joint_reduced[dof] = ws.joint_force[dof] - wp.dot(joint_s[dof], bias)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    gain = wp.float32(0.0)
                    for column in range(_MAX_JOINT_DOF):
                        if wp.int32(column) < dof_count:
                            gain += (
                                ws.joint_d_inv[dof_start + wp.int32(row), column]
                                * ws.joint_reduced[dof_start + wp.int32(column)]
                            )
                    propagated += ws.joint_u[dof_start + wp.int32(row)] * gain
            ws.body_force[child] = propagated
            index += tree.width
        _sync_reduced_group(tree.mask)

    work = wp.float32(0.0)
    for depth in range(tree.max_depth + wp.int32(1)):
        index = tree.depth_start[tree.articulation, depth] + tree.lane
        depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = tree.order[index]
            parent = joint_parent[joint]
            child = joint_child[joint]
            dof_start = joint_qd_start[joint]
            dof_count = joint_qd_start[joint + wp.int32(1)] - dof_start
            twist = wp.spatial_vector()
            if parent >= wp.int32(0):
                twist = ws.body_force[parent]
            rhs = _vec6(0.0)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    dof = dof_start + wp.int32(row)
                    rhs[row] = ws.joint_reduced[dof] - wp.dot(ws.joint_u[dof], twist)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    dof = dof_start + wp.int32(row)
                    value = wp.float32(0.0)
                    for column in range(_MAX_JOINT_DOF):
                        if wp.int32(column) < dof_count:
                            value += ws.joint_d_inv[dof, column] * rhs[column]
                    work += value * ws.joint_force[dof]
                    out[dof] = value
                    twist += joint_s[dof] * value
            ws.body_force[child] = twist
            index += tree.width
        _sync_reduced_group(tree.mask)
    return _sum_reduced_group_float(work, tree.width, tree.mask)


@wp.func
def _midpoint_momentum_step(
    tree: _MidpointTree,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    body_inertia: wp.array[_sym_mat66],
    ws: _MidpointWorkspace,
    dt: wp.float32,
    max_iterations: wp.int32,
    joint_qd: wp.array[wp.float32],
    body_velocity: wp.array[wp.spatial_vector],
):
    """Integrate one fast tree's torque-free motion over ``dt`` and set its new pose.

    A discrete variational step with the midpoint rule: over the step each body
    carries the secant linear momentum ``m (x_1 - x_0) / dt`` and the rotational
    momentum ``I(R_half) omega`` of the geodesic from ``R_0`` to ``R_1``. The end
    pose ``q_1`` solves ``J(q_0)^T h(q_0, q_1) = p_0``, the start momentum, by a
    fixed point preconditioned with the start factorization, and the end
    momentum is ``p_1 = J(q_1)^T h(q_0, q_1)``. Joint reactions then exchange
    momentum only through joint impulses at a consistent pose, so a floating
    tree conserves linear and angular momentum, a link keeps the spin its joints
    cannot transmit, and the symmetric step keeps the energy without drift.

    A step whose fixed point stalls or diverges above its rounding floor has no
    solution near the start rates; it is retried at half length, see
    ``_MIDPOINT_MAX_HALVINGS``. Each converged substep restarts from its end pose,
    overwriting ``joint_s``, ``body_inertia`` and ``ws.body_q_local`` with that
    pose's tables; the next factorization rebuilds them from the coordinates.

    The tree's lanes split each depth level like the explicit advance; every
    branch is uniform over them. Writes the end coordinates to ``ws.joint_q``,
    internal rates to ``joint_qd`` (root anchor velocity at the new pose), and
    the public floating-root rates to ``ws.joint_qd_public``.
    """
    begin = tree.depth_start[tree.articulation, 0]
    end = tree.depth_start[tree.articulation, tree.max_depth + wp.int32(1)]
    root_joint = tree.order[begin]
    root_kind = joint_type[root_joint]
    floating = root_kind == JointType.FREE or root_kind == JointType.DISTANCE
    # Progress in units of the shortest substep; integers keep the substeps summing to dt.
    units_left = wp.int32(1) << wp.int32(_MIDPOINT_MAX_HALVINGS)
    units = units_left
    while units_left > wp.int32(0):
        # Start momenta and the start factorization, which preconditions the fixed point.
        for depth in range(tree.max_depth + wp.int32(1)):
            index = tree.depth_start[tree.articulation, depth] + tree.lane
            depth_end = tree.depth_start[tree.articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = tree.order[index]
                parent = joint_parent[joint]
                child = joint_child[joint]
                twist = wp.spatial_vector()
                if parent >= wp.int32(0):
                    twist = body_velocity[parent]
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                    twist += joint_s[dof] * joint_qd[dof]
                body_velocity[child] = twist
                inertia = _unpack_symmetric_mat66(body_inertia[child])
                ws.body_articulated[child] = inertia
                ws.body_momentum[child] = inertia * twist
                index += tree.width
            _sync_reduced_group(tree.mask)
        _midpoint_gather_momentum(tree, joint_child, ws)
        scale = wp.float32(0.0)
        moment = wp.float32(0.0)
        index = begin + tree.lane
        while index < end:
            joint = tree.order[index]
            child = joint_child[joint]
            for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                momentum = wp.dot(joint_s[dof], ws.body_momentum[child]) + ws.joint_armature[dof] * joint_qd[dof]
                ws.joint_momentum[dof] = momentum
                scale += momentum * joint_qd[dof]
            com = wp.transform_point(ws.body_q_local[child], ws.body_com[child])
            moment += ws.body_mass[child] * wp.dot(com, com)
            index += tree.width
        scale = _sum_reduced_group_float(scale, tree.width, tree.mask)
        # Rounding the COM positions to float32 limits the secant momenta m (x1 - x0) / h
        # to about m |x| eps / h, so the residual cannot fall below (eps / h)^2 sum m |x|^2.
        moment = _sum_reduced_group_float(moment, tree.width, tree.mask)
        _midpoint_factor(tree, joint_child, joint_qd_start, joint_s, ws)

        units = wp.min(units, units_left)
        step = dt * wp.float32(units) / wp.float32(wp.int32(1) << wp.int32(_MIDPOINT_MAX_HALVINGS))
        rounding = _MIDPOINT_ROUNDING / step
        solved = wp.bool(False)
        while not solved:
            index = begin + tree.lane
            while index < end:
                joint = tree.order[index]
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                    ws.joint_qd_mid[dof] = joint_qd[dof]
                index += tree.width
            _sync_reduced_group(tree.mask)
            converged = wp.bool(False)
            previous = wp.float32(3.4e38)
            iteration = wp.int32(0)
            while iteration < max_iterations:
                _midpoint_pose_pass(
                    tree,
                    joint_type,
                    joint_parent,
                    joint_child,
                    joint_qd_start,
                    joint_s,
                    ws,
                    step,
                    ws.joint_q_mid,
                    wp.bool(False),
                    body_velocity,
                )
                index = begin + tree.lane
                while index < end:
                    joint = tree.order[index]
                    child = joint_child[joint]
                    for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                        ws.joint_force[dof] = (
                            ws.joint_momentum[dof]
                            - wp.dot(joint_s[dof], ws.body_momentum[child])
                            - ws.joint_armature[dof] * ws.joint_qd_mid[dof]
                        )
                    index += tree.width
                _sync_reduced_group(tree.mask)
                # The residual's energy norm r M^-1 r measures convergence.
                residual = _midpoint_solve(tree, joint_parent, joint_child, joint_qd_start, joint_s, ws, ws.joint_step)
                iteration += wp.int32(1)
                if residual > previous:
                    # Diverging: keep the current iterate.
                    iteration = max_iterations
                else:
                    index = begin + tree.lane
                    while index < end:
                        joint = tree.order[index]
                        for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                            ws.joint_qd_mid[dof] = ws.joint_qd_mid[dof] + ws.joint_step[dof]
                        index += tree.width
                    _sync_reduced_group(tree.mask)
                    if residual <= _MIDPOINT_TOLERANCE * scale:
                        converged = wp.bool(True)
                        iteration = max_iterations
                previous = residual
            # A fixed point that stalls above the rounding floor has no solution near the
            # start rates: the step is too long for the tree's tumbling, so halve it.
            floor = _MIDPOINT_TOLERANCE * scale + rounding * rounding * moment
            if converged or previous <= floor or units == wp.int32(1):
                solved = wp.bool(True)
            else:
                units = units >> wp.int32(1)
                step = wp.float32(0.5) * step

        # End pose, its momentum and velocity.
        _midpoint_pose_pass(
            tree,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_s,
            ws,
            step,
            ws.joint_q,
            wp.bool(True),
            body_velocity,
        )
        index = begin + tree.lane
        while index < end:
            joint = tree.order[index]
            child = joint_child[joint]
            for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                ws.joint_force[dof] = (
                    wp.dot(ws.joint_s_mid[dof], ws.body_momentum[child]) + ws.joint_armature[dof] * ws.joint_qd_mid[dof]
                )
                if floating and joint == root_joint:
                    # A floating root's fixed axes project the whole tree's momentum about the
                    # articulation origin, which the converged step conserves exactly. Keep it
                    # exact for any fixed-point residual.
                    ws.joint_force[dof] = ws.joint_momentum[dof]
            index += tree.width
        _sync_reduced_group(tree.mask)
        _midpoint_factor(tree, joint_child, joint_qd_start, ws.joint_s_mid, ws)
        _midpoint_solve(tree, joint_parent, joint_child, joint_qd_start, ws.joint_s_mid, ws, joint_qd)
        units_left -= units
        if units_left > wp.int32(0):
            # The next substep starts from this end pose. The factorization rebuilds
            # these start-pose tables from the coordinates before anything else reads them.
            index = begin + tree.lane
            while index < end:
                joint = tree.order[index]
                child = joint_child[joint]
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                    joint_s[dof] = ws.joint_s_mid[dof]
                pose = ws.body_pose[child]
                ws.body_q_local[child] = pose
                body_inertia[child] = _pack_transformed_body_inertia(
                    pose * ws.body_x_com[child], ws.body_mass[child], ws.body_inertia[child]
                )
                index += tree.width
            _sync_reduced_group(tree.mask)

    if tree.lane == wp.int32(0):
        if floating:
            root = joint_child[root_joint]
            dof_start = joint_qd_start[root_joint]
            coord_start = ws.joint_q_start[root_joint]
            joint_x_p = ws.joint_x_p[root_joint]
            joint_x_c = ws.joint_x_c[root_joint]
            twist = ws.body_force[root]
            omega = wp.spatial_bottom(twist)
            pose = ws.body_pose[root]
            # The internal root rate is the velocity of the child anchor, which moved.
            anchor_velocity = wp.spatial_top(twist) + wp.cross(omega, wp.transform_get_translation(pose * joint_x_c))
            anchor_velocity = wp.quat_rotate_inv(
                wp.transform_get_rotation(ws.joint_anchor_local[root_joint]), anchor_velocity
            )
            joint_qd[dof_start] = anchor_velocity[0]
            joint_qd[dof_start + wp.int32(1)] = anchor_velocity[1]
            joint_qd[dof_start + wp.int32(2)] = anchor_velocity[2]
            parent_rotation = wp.transform_get_rotation(joint_x_p)
            com_velocity = wp.quat_rotate_inv(
                parent_rotation,
                wp.spatial_top(twist) + wp.cross(omega, wp.transform_point(pose, ws.body_com[root])),
            )
            omega_parent = wp.quat_rotate_inv(parent_rotation, omega)
            for axis in range(3):
                ws.joint_qd_public[dof_start + wp.int32(axis)] = com_velocity[axis]
                ws.joint_qd_public[dof_start + wp.int32(3 + axis)] = omega_parent[axis]
            world_pose = wp.transform(
                ws.articulation_origin[tree.articulation] + wp.transform_get_translation(pose),
                wp.transform_get_rotation(pose),
            )
            joint_transform = wp.transform_inverse(joint_x_p) * world_pose * joint_x_c
            position = wp.transform_get_translation(joint_transform)
            rotation = wp.transform_get_rotation(joint_transform)
            for axis in range(3):
                ws.joint_q[coord_start + wp.int32(axis)] = position[axis]
            for axis in range(4):
                ws.joint_q[coord_start + wp.int32(3 + axis)] = rotation[axis]
        ws.flag[tree.articulation] = wp.int32(1)
    _sync_reduced_group(tree.mask)


@wp.kernel(enable_backward=False)
def _mark_articulated_bodies_kernel(
    body_is_reduced: wp.array[wp.int32],
    bodies: BodyContainer,
):
    body = wp.tid()
    if body_is_reduced[body] != wp.int32(0):
        bodies.motion_type[body + wp.int32(1)] = MOTION_ARTICULATED


@wp.kernel(enable_backward=False)
def _sync_reduced_bodies_kernel(
    body_is_reduced: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    bodies: BodyContainer,
):
    body = wp.tid()
    if body_is_reduced[body] == wp.int32(0):
        return
    slot = body + wp.int32(1)
    transform = body_q[body]
    rotation = wp.transform_get_rotation(transform)
    origin = wp.transform_get_translation(transform)
    bodies.position[slot] = origin + wp.quat_rotate(rotation, body_com[body])
    bodies.orientation[slot] = rotation
    twist = body_qd[body]
    bodies.velocity[slot] = wp.spatial_top(twist)
    bodies.angular_velocity[slot] = wp.spatial_bottom(twist)


@wp.func
def configure_implicit_joint_dynamics(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_limit_ke: wp.array[wp.float32],
    joint_limit_kd: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    dt: wp.float32,
    factor_diagonal: wp.array[wp.float32],
    implicit_force: wp.array[wp.float32],
) -> None:
    """Build one joint's linearized implicit-Euler operator diagonal and right-hand side."""
    dof_start = joint_qd_start[joint]
    dof_end = joint_qd_start[joint + wp.int32(1)]
    coord_start = joint_q_start[joint]
    target_start = joint_target_q_start[joint]
    kind = joint_type[joint]
    for dof in range(dof_start, dof_end):
        factor_diagonal[dof] = joint_armature[dof]
        implicit_force[dof] = wp.float32(0.0)
        if kind == JointType.REVOLUTE or kind == JointType.PRISMATIC or kind == JointType.D6:
            local = dof - dof_start
            q = joint_q[coord_start + local]
            qd = joint_qd[dof]
            stiffness = joint_target_ke[dof]
            drive_damping = joint_target_kd[dof]
            target_position = joint_target_q[target_start + local]
            target_velocity = joint_target_qd[dof]
            target_mode = joint_target_mode[dof]
            if target_mode == JointTargetMode.NONE or target_mode == JointTargetMode.EFFORT:
                stiffness = wp.float32(0.0)
                drive_damping = wp.float32(0.0)
                target_velocity = wp.float32(0.0)
            elif target_mode == JointTargetMode.POSITION:
                target_velocity = wp.float32(0.0)
            elif target_mode == JointTargetMode.VELOCITY:
                stiffness = wp.float32(0.0)
            predicted_q = q + dt * qd
            if q < joint_limit_lower[dof] or predicted_q < joint_limit_lower[dof]:
                stiffness = joint_limit_ke[dof]
                drive_damping = joint_limit_kd[dof]
                target_position = joint_limit_lower[dof]
                target_velocity = wp.float32(0.0)
            elif q > joint_limit_upper[dof] or predicted_q > joint_limit_upper[dof]:
                stiffness = joint_limit_ke[dof]
                drive_damping = joint_limit_kd[dof]
                target_position = joint_limit_upper[dof]
                target_velocity = wp.float32(0.0)

            passive_damping = joint_damping[dof]
            damping = drive_damping + passive_damping
            factor_diagonal[dof] += dt * damping + dt * dt * stiffness
            implicit_force[dof] = (
                stiffness * (target_position - q)
                + drive_damping * (target_velocity - qd)
                - passive_damping * qd
                - dt * stiffness * qd
            )


@wp.kernel(enable_backward=False)
def _configure_implicit_joint_dynamics_kernel(
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_target_mode: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_limit_ke: wp.array[wp.float32],
    joint_limit_kd: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    dt: wp.float32,
    factor_diagonal: wp.array[wp.float32],
    implicit_force: wp.array[wp.float32],
):
    configure_implicit_joint_dynamics(
        wp.tid(),
        joint_type,
        joint_q_start,
        joint_qd_start,
        joint_target_q_start,
        joint_q,
        joint_qd,
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


@wp.kernel(enable_backward=False)
def _compute_forward_dynamics_rhs_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    joint_f: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    joint_limit_ke: wp.array[wp.float32],
    joint_limit_kd: wp.array[wp.float32],
    joint_damping: wp.array[wp.float32],
    joint_dof_dim: wp.array2d[wp.int32],
    body_mass: wp.array[wp.float32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    body_q_com: wp.array[wp.transform],
    body_inertia: wp.array[_sym_mat66],
    body_force: wp.array[wp.spatial_vector],
    body_velocity: wp.array[wp.spatial_vector],
    body_coriolis: wp.array[wp.spatial_vector],
    body_bias: wp.array[wp.spatial_vector],
    generalized_rhs: wp.array[wp.float32],
):
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        parent_velocity = wp.spatial_vector()
        parent_coriolis = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_velocity = body_velocity[parent]
            parent_coriolis = body_coriolis[parent]
        joint_velocity = wp.spatial_vector()
        for dof in range(dof_start, dof_end):
            joint_velocity += joint_s[dof] * joint_qd[dof]
        velocity = parent_velocity + joint_velocity
        coriolis = parent_coriolis + spatial_cross(velocity, joint_velocity)
        inertia = _unpack_symmetric_mat66(body_inertia[child])
        gravity_force = body_mass[child] * gravity[wp.max(body_world[child], wp.int32(0))]
        x_com = wp.transform_get_translation(body_q_com[child])
        gravity_wrench = wp.spatial_vector(gravity_force, wp.cross(x_com, gravity_force))
        force_com = body_force[child]
        force = wp.spatial_top(force_com)
        external_wrench = wp.spatial_vector(force, wp.spatial_bottom(force_com) + wp.cross(x_com, force))
        joint_kind = joint_type[joint]
        if joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE:
            control_force = wp.vec3(
                joint_f[dof_start],
                joint_f[dof_start + wp.int32(1)],
                joint_f[dof_start + wp.int32(2)],
            )
            control_torque = wp.vec3(
                joint_f[dof_start + wp.int32(3)],
                joint_f[dof_start + wp.int32(4)],
                joint_f[dof_start + wp.int32(5)],
            )
            external_wrench += wp.spatial_vector(
                control_force,
                control_torque + wp.cross(x_com, control_force),
            )
        body_velocity[child] = velocity
        body_coriolis[child] = coriolis
        body_bias[child] = (
            inertia * coriolis + spatial_cross_dual(velocity, inertia * velocity) - gravity_wrench - external_wrench
        )

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        coord_start = joint_q_start[joint]
        target_start = joint_target_q_start[joint]
        joint_kind = joint_type[joint]
        subtree_bias = body_bias[child]
        for dof in range(dof_start, dof_end):
            local = dof - dof_start
            applied = wp.float32(0.0)
            if joint_kind != JointType.FREE and joint_kind != JointType.DISTANCE:
                applied = joint_f[dof]
            if joint_kind == JointType.REVOLUTE or joint_kind == JointType.PRISMATIC or joint_kind == JointType.D6:
                applied += joint_force(
                    joint_q[coord_start + local],
                    joint_qd[dof],
                    joint_target_q[target_start + local],
                    joint_target_qd[dof],
                    joint_target_ke[dof],
                    joint_target_kd[dof],
                    joint_limit_lower[dof],
                    joint_limit_upper[dof],
                    joint_limit_ke[dof],
                    joint_limit_kd[dof],
                    joint_damping[dof],
                )
            generalized_rhs[dof] = applied - wp.dot(joint_s[dof], subtree_bias)
        if parent >= wp.int32(0):
            body_bias[parent] = body_bias[parent] + subtree_bias


@wp.kernel(enable_backward=False)
def _compute_reduced_local_kinematics_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    body_q_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_anchor_local: wp.array[wp.transform],
):
    """Reconstruct one tree without subtracting large world positions."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    root = joint_child[start]
    articulation_origin[articulation] = wp.transform_point(body_q[root], body_com[root])

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        joint_transform = jcalc_transform(
            joint_type[joint],
            joint_axis,
            joint_qd_start[joint],
            joint_dof_dim[joint, 0],
            joint_dof_dim[joint, 1],
            joint_q,
            joint_q_start[joint],
        )
        if parent < wp.int32(0):
            rotation = wp.transform_get_rotation(body_q[child])
            child_transform = wp.transform(-wp.quat_rotate(rotation, body_com[child]), rotation)
            child_anchor = child_transform * joint_x_c[joint]
            root_joint_transform = joint_transform
            if joint_type[joint] == JointType.FREE or joint_type[joint] == JointType.DISTANCE:
                root_joint_transform = wp.transform(wp.vec3(0.0), wp.transform_get_rotation(joint_transform))
            parent_anchor = child_anchor * wp.transform_inverse(root_joint_transform)
        else:
            parent_anchor = body_q_local[parent] * joint_x_p[joint]
            child_anchor = parent_anchor * joint_transform
            child_transform = child_anchor * wp.transform_inverse(joint_x_c[joint])
        joint_anchor_local[joint] = parent_anchor
        body_q_local[child] = child_transform
        body_q_com[child] = child_transform * body_x_com[child]


@wp.kernel(enable_backward=False)
def _integrate_rate_change_kernel(
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_c: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rates: wp.array[wp.float32],
    rates_before: wp.array[wp.float32],
    dt: wp.float32,
    joint_q: wp.array[wp.float32],
):
    """Advance joint coordinates by the change of their public rates over ``dt``.

    A free joint's public rates are its child's COM velocity and angular velocity in the
    parent anchor frame, so the COM translates and the child turns about it.
    """
    joint = wp.tid()
    kind = joint_type[joint]
    q = joint_q_start[joint]
    qd = joint_qd_start[joint]
    count = joint_qd_start[joint + 1] - qd
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        linear = wp.vec3(
            rates[qd] - rates_before[qd],
            rates[qd + 1] - rates_before[qd + 1],
            rates[qd + 2] - rates_before[qd + 2],
        )
        angular = wp.vec3(
            rates[qd + 3] - rates_before[qd + 3],
            rates[qd + 4] - rates_before[qd + 4],
            rates[qd + 5] - rates_before[qd + 5],
        )
        position = wp.vec3(joint_q[q], joint_q[q + 1], joint_q[q + 2])
        rotation = wp.quat(joint_q[q + 3], joint_q[q + 4], joint_q[q + 5], joint_q[q + 6])
        com_joint = wp.transform_point(wp.transform_inverse(joint_x_c[joint]), body_com[joint_child[joint]])
        com = position + wp.quat_rotate(rotation, com_joint)
        rotation = wp.normalize(rotation + wp.quat(angular, 0.0) * rotation * (0.5 * dt))
        position = com + dt * linear - wp.quat_rotate(rotation, com_joint)
        for axis in range(3):
            joint_q[q + axis] = position[axis]
        for axis in range(4):
            joint_q[q + 3 + axis] = rotation[axis]
    elif kind == JointType.BALL:
        angular = wp.vec3(
            rates[qd] - rates_before[qd],
            rates[qd + 1] - rates_before[qd + 1],
            rates[qd + 2] - rates_before[qd + 2],
        )
        rotation = wp.quat(joint_q[q], joint_q[q + 1], joint_q[q + 2], joint_q[q + 3])
        rotation = wp.normalize(rotation + wp.quat(angular, 0.0) * rotation * (0.5 * dt))
        for axis in range(4):
            joint_q[q + axis] = rotation[axis]
    else:
        # One coordinate per rate: revolute, prismatic, universal, and D6 axes.
        for axis in range(count):
            joint_q[q + axis] += dt * (rates[qd + axis] - rates_before[qd + axis])


@wp.func
def initialize_reduced_rates(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_qd_public: wp.array[wp.float32],
    joint_anchor_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_qd_internal: wp.array[wp.float32],
) -> None:
    """Form one joint's internal rates from its public rates at the current pose."""
    qd_start = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + wp.int32(1)]
    for dof in range(qd_start, qd_end):
        joint_qd_internal[dof] = joint_qd_public[dof]

    kind = joint_type[joint]
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        # Newton exposes the relative child-COM velocity in the parent-anchor
        # frame. ABA uses the velocity at that frame origin. Convert in the
        # articulation-local frame so large world translations never enter.
        child = joint_child[joint]
        x_wpj = joint_anchor_local[joint]
        anchor_rotation = wp.transform_get_rotation(x_wpj)
        child_com_offset = wp.quat_rotate_inv(
            anchor_rotation,
            wp.transform_get_translation(body_q_com[child]) - wp.transform_get_translation(x_wpj),
        )
        velocity_com = wp.vec3(
            joint_qd_public[qd_start],
            joint_qd_public[qd_start + wp.int32(1)],
            joint_qd_public[qd_start + wp.int32(2)],
        )
        omega = wp.vec3(
            joint_qd_public[qd_start + wp.int32(3)],
            joint_qd_public[qd_start + wp.int32(4)],
            joint_qd_public[qd_start + wp.int32(5)],
        )
        velocity_origin = velocity_com - wp.cross(omega, child_com_offset)
        joint_qd_internal[qd_start] = velocity_origin[0]
        joint_qd_internal[qd_start + wp.int32(1)] = velocity_origin[1]
        joint_qd_internal[qd_start + wp.int32(2)] = velocity_origin[2]


@wp.func
def initialize_reduced_joint(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_qd_public: wp.array[wp.float32],
    joint_anchor_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    joint_qd_internal: wp.array[wp.float32],
    body_i_s: wp.array[_sym_mat66],
    joint_s: wp.array[wp.spatial_vector],
) -> None:
    """Form one joint's motion subspace, its child's spatial inertia, and its internal rates at the current pose."""
    initialize_reduced_rates(
        joint,
        joint_type,
        joint_child,
        joint_qd_start,
        joint_qd_public,
        joint_anchor_local,
        body_q_com,
        joint_qd_internal,
    )
    child = joint_child[joint]
    body_i_s[child] = _pack_transformed_body_inertia(body_q_com[child], body_mass[child], body_inertia[child])
    jcalc_motion(
        joint_type[joint],
        joint_axis,
        joint_q,
        joint_dof_dim[joint, 0],
        joint_dof_dim[joint, 1],
        joint_anchor_local[joint],
        joint_qd_internal,
        joint_q_start[joint],
        joint_qd_start[joint],
        joint_s,
    )


@wp.kernel(enable_backward=False)
def _initialize_reduced_factor_kernel(
    factor_joint: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_qd_public: wp.array[wp.float32],
    joint_anchor_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    joint_qd_internal: wp.array[wp.float32],
    body_i_s: wp.array[_sym_mat66],
    joint_s: wp.array[wp.spatial_vector],
):
    """Initialize independent per-joint quantities for a depth factorization."""
    initialize_reduced_joint(
        factor_joint[wp.tid()],
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
        joint_qd_internal,
        body_i_s,
        joint_s,
    )


@wp.kernel(enable_backward=False)
def _factor_reduced_depth_kernel(
    depth_joint: wp.array[wp.int32],
    depth_offset: wp.int32,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    factor_diagonal: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    child_start: wp.array[wp.int32],
    child_joint: wp.array[wp.int32],
    body_i_s: wp.array[_sym_mat66],
    reduced_inertia: wp.array[_sym_mat66],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
):
    """Factor one tree depth with stable per-parent child reductions."""
    joint = depth_joint[depth_offset + wp.tid()]
    child = joint_child[joint]
    inertia = _unpack_symmetric_mat66(body_i_s[child])
    for child_index in range(child_start[joint], child_start[joint + wp.int32(1)]):
        descendant_joint = child_joint[child_index]
        inertia += _unpack_symmetric_mat66(reduced_inertia[joint_child[descendant_joint]])

    dof_start = joint_qd_start[joint]
    dof_end = joint_qd_start[joint + wp.int32(1)]
    dof_count = dof_end - dof_start
    reduced = inertia
    if dof_count == wp.int32(1):
        dof = dof_start
        u = inertia * joint_s[dof]
        joint_u[dof] = u
        d_inv_scalar = wp.float32(1.0) / wp.max(wp.dot(joint_s[dof], u) + factor_diagonal[dof], wp.float32(1.0e-20))
        joint_d_inv[dof, 0] = d_inv_scalar
        for row in range(6):
            for column in range(6):
                reduced[row, column] -= u[row] * d_inv_scalar * u[column]
    elif dof_count > wp.int32(1):
        d = _mat66(0.0)
        for column in range(_MAX_JOINT_DOF):
            if wp.int32(column) < dof_count:
                dof = dof_start + wp.int32(column)
                u = inertia * joint_s[dof]
                joint_u[dof] = u
                for row in range(_MAX_JOINT_DOF):
                    if wp.int32(row) < dof_count:
                        d[row, column] = wp.dot(joint_s[dof_start + wp.int32(row)], u)
                d[column, column] += factor_diagonal[dof]

        d_inv = _invert_spd(d, dof_count)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    joint_d_inv[dof_start + wp.int32(row), column] = d_inv[row, column]

        for row in range(6):
            for column in range(6):
                correction = wp.float32(0.0)
                for a in range(_MAX_JOINT_DOF):
                    if wp.int32(a) < dof_count:
                        u_a = joint_u[dof_start + wp.int32(a)]
                        for b in range(_MAX_JOINT_DOF):
                            if wp.int32(b) < dof_count:
                                u_b = joint_u[dof_start + wp.int32(b)]
                                correction += u_a[row] * d_inv[a, b] * u_b[column]
                reduced[row, column] -= correction
    reduced_inertia[child] = _pack_symmetric_mat66(reduced)


@wp.func
def _multiply_impulse_response(
    impulse_response: wp.array2d[wp.spatial_vector],
    body: wp.int32,
    wrench: wp.spatial_vector,
) -> wp.spatial_vector:
    result = wp.spatial_vector()
    for column in range(6):
        result += impulse_response[body, column] * wrench[column]
    return result


@wp.func
def _compute_reduced_impulse_response_basis(
    bodies: BodyContainer,
    joint: wp.int32,
    parent: wp.int32,
    child: wp.int32,
    basis: wp.int32,
):
    data = bodies.reduced
    dof_start = data.joint_qd_start[joint]
    dof_end = data.joint_qd_start[joint + wp.int32(1)]
    dof_count = dof_end - dof_start
    wrench = wp.spatial_vector()
    wrench[basis] = wp.float32(1.0)
    negative_force = -wrench
    reduced_force = _vec6(0.0)
    d_inv_u = _vec6(0.0)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            dof = dof_start + wp.int32(row)
            reduced_force[row] = -wp.dot(data.joint_s[dof], negative_force)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            for column in range(_MAX_JOINT_DOF):
                if wp.int32(column) < dof_count:
                    d_inv_u[row] += data.joint_d_inv[dof_start + wp.int32(row), column] * reduced_force[column]

    propagated_negative_force = negative_force
    for column in range(_MAX_JOINT_DOF):
        if wp.int32(column) < dof_count:
            propagated_negative_force += data.joint_u[dof_start + wp.int32(column)] * d_inv_u[column]

    parent_response = wp.spatial_vector()
    if parent >= wp.int32(0):
        parent_response = _multiply_impulse_response(
            data.impulse_response,
            parent,
            -propagated_negative_force,
        )

    rhs = _vec6(0.0)
    response = _vec6(0.0)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            dof = dof_start + wp.int32(row)
            rhs[row] = reduced_force[row] - wp.dot(data.joint_u[dof], parent_response)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            for column in range(_MAX_JOINT_DOF):
                if wp.int32(column) < dof_count:
                    response[row] += data.joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]

    child_response = parent_response
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            child_response += data.joint_s[dof_start + wp.int32(row)] * response[row]
    data.impulse_response[child, basis] = child_response


@wp.kernel(enable_backward=False)
def _compute_reduced_impulse_response_kernel(
    bodies: BodyContainer,
):
    articulation = wp.tid()
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    for joint in range(start, end):
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        data.deferred_wrench[child] = wp.spatial_vector()
        for basis in range(6):
            _compute_reduced_impulse_response_basis(bodies, joint, parent, child, wp.int32(basis))


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
"""
)
def _sync_reduced_group(mask: wp.uint32): ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    float result = value;
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        result += __shfl_down_sync(mask, result, offset, width);
    }
    return __shfl_sync(mask, result, 0, width);
#else
    return value;
#endif
"""
)
def _sum_reduced_group_float(
    value: wp.float32,
    width: wp.int32,
    mask: wp.uint32,
) -> wp.float32:
    """Sum a value over one articulation's lane group and broadcast the total."""
    ...


@wp.func
def _sum_reduced_group_vec4(value: wp.vec4, width: wp.int32, mask: wp.uint32) -> wp.vec4:
    return wp.vec4(
        _sum_reduced_group_float(value[0], width, mask),
        _sum_reduced_group_float(value[1], width, mask),
        _sum_reduced_group_float(value[2], width, mask),
        _sum_reduced_group_float(value[3], width, mask),
    )


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    wp::spatial_vector result;
    for (int component = 0; component < 6; ++component) {
        result[component] = __shfl_sync(mask, value[component], source_lane, width);
    }
    return result;
#else
    return value;
#endif
"""
)
def _shuffle_reduced_spatial(
    value: wp.spatial_vector,
    source_lane: wp.int32,
    width: wp.int32,
    mask: wp.uint32,
) -> wp.spatial_vector: ...


@wp.kernel(enable_backward=False, module="reduced_impulse_response")
def _compute_reduced_impulse_response_warp_kernel(
    bodies: BodyContainer,
    articulation_count: wp.int32,
):
    thread = wp.tid()
    articulation = thread // wp.int32(8)
    lane = thread - articulation * wp.int32(8)
    if articulation >= articulation_count:
        return
    warp_lane = thread & wp.int32(31)
    group = warp_lane // wp.int32(8)
    group_mask = wp.uint32(255) << wp.uint32(group * wp.int32(8))
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    for joint in range(start, end):
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        if lane == wp.int32(0):
            data.deferred_wrench[child] = wp.spatial_vector()
        if lane < wp.int32(6):
            _compute_reduced_impulse_response_basis(bodies, joint, parent, child, lane)
        _sync_reduced_group(group_mask)


@wp.kernel(enable_backward=False, module="reduced_kinematics")
def _compute_reduced_local_kinematics_warp_kernel(
    articulation_count: wp.int32,
    tile_width: wp.int32,
    max_depth: wp.int32,
    articulation_depth_start: wp.array2d[wp.int32],
    articulation_depth_joint: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    body_q_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_anchor_local: wp.array[wp.transform],
):
    thread = wp.tid()
    articulation = thread // tile_width
    lane = thread - articulation * tile_width
    if articulation >= articulation_count:
        return
    warp_lane = thread & wp.int32(31)
    group = warp_lane // tile_width
    group_mask = wp.uint32(4294967295)
    if tile_width < wp.int32(32):
        group_mask = ((wp.uint32(1) << wp.uint32(tile_width)) - wp.uint32(1)) << wp.uint32(group * tile_width)
    if lane == wp.int32(0):
        root_joint = articulation_depth_joint[articulation_depth_start[articulation, 0]]
        root = joint_child[root_joint]
        articulation_origin[articulation] = wp.transform_point(body_q[root], body_com[root])

    for depth in range(max_depth + wp.int32(1)):
        index = articulation_depth_start[articulation, depth] + lane
        depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = articulation_depth_joint[index]
            parent = joint_parent[joint]
            child = joint_child[joint]
            joint_transform = jcalc_transform(
                joint_type[joint],
                joint_axis,
                joint_qd_start[joint],
                joint_dof_dim[joint, 0],
                joint_dof_dim[joint, 1],
                joint_q,
                joint_q_start[joint],
            )
            if parent < wp.int32(0):
                rotation = wp.transform_get_rotation(body_q[child])
                child_transform = wp.transform(-wp.quat_rotate(rotation, body_com[child]), rotation)
                child_anchor = child_transform * joint_x_c[joint]
                root_joint_transform = joint_transform
                if joint_type[joint] == JointType.FREE or joint_type[joint] == JointType.DISTANCE:
                    root_joint_transform = wp.transform(wp.vec3(0.0), wp.transform_get_rotation(joint_transform))
                parent_anchor = child_anchor * wp.transform_inverse(root_joint_transform)
            else:
                parent_anchor = body_q_local[parent] * joint_x_p[joint]
                child_anchor = parent_anchor * joint_transform
                child_transform = child_anchor * wp.transform_inverse(joint_x_c[joint])
            joint_anchor_local[joint] = parent_anchor
            body_q_local[child] = child_transform
            body_q_com[child] = child_transform * body_x_com[child]
            index += tile_width
        _sync_reduced_group(group_mask)


@wp.func
def _factor_reduced_multidof_at(
    joint: wp.int32,
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    factor_diagonal: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    child_start: wp.array[wp.int32],
    child_joint: wp.array[wp.int32],
    body_i_s: wp.array[_sym_mat66],
    reduced_inertia: wp.array[_sym_mat66],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
):
    """Articulated-body reduction for one multi-DOF joint (block 6x6 inverse)."""
    child = joint_child[joint]
    inertia = _unpack_symmetric_mat66(body_i_s[child])
    for child_index in range(child_start[joint], child_start[joint + wp.int32(1)]):
        descendant_joint = child_joint[child_index]
        inertia += _unpack_symmetric_mat66(reduced_inertia[joint_child[descendant_joint]])

    dof_start = joint_qd_start[joint]
    dof_end = joint_qd_start[joint + wp.int32(1)]
    dof_count = dof_end - dof_start
    d = _mat66(0.0)
    for column in range(_MAX_JOINT_DOF):
        if wp.int32(column) < dof_count:
            dof = dof_start + wp.int32(column)
            u = inertia * joint_s[dof]
            joint_u[dof] = u
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    d[row, column] = wp.dot(joint_s[dof_start + wp.int32(row)], u)
            d[column, column] += factor_diagonal[dof]

    d_inv = _invert_spd(d, dof_count)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            for column in range(_MAX_JOINT_DOF):
                joint_d_inv[dof_start + wp.int32(row), column] = d_inv[row, column]

    for row in range(6):
        for column in range(6):
            correction = wp.float32(0.0)
            for a in range(_MAX_JOINT_DOF):
                if wp.int32(a) < dof_count:
                    u_a = joint_u[dof_start + wp.int32(a)]
                    for b in range(_MAX_JOINT_DOF):
                        if wp.int32(b) < dof_count:
                            u_b = joint_u[dof_start + wp.int32(b)]
                            correction += u_a[row] * d_inv[a, b] * u_b[column]
            inertia[row, column] -= correction
    reduced_inertia[child] = _pack_symmetric_mat66(inertia)


@wp.kernel(enable_backward=False, module="reduced_factor_multidof")
def _factor_reduced_multidof_kernel(
    multidof_joint: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    factor_diagonal: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    child_start: wp.array[wp.int32],
    child_joint: wp.array[wp.int32],
    body_i_s: wp.array[_sym_mat66],
    reduced_inertia: wp.array[_sym_mat66],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
):
    """Factor the deferred multi-DOF (root) joints, one thread per joint.

    Kept out of the depth-walk kernel so its 6x6 SPD inverse does not inflate
    that kernel's register footprint; see :func:`_make_factor_reduced_warp_kernel`.
    """
    _factor_reduced_multidof_at(
        multidof_joint[wp.tid()],
        joint_child,
        joint_qd_start,
        factor_diagonal,
        joint_s,
        child_start,
        child_joint,
        body_i_s,
        reduced_inertia,
        joint_u,
        joint_d_inv,
    )


def _make_factor_reduced_warp_kernel(single_dof_only: bool, tile_width: int = 32):
    """Build the reverse-depth articulated-body factorization kernel.

    The per-joint articulated-inertia reduction reduces one warp lane per joint
    across a depth-serial tree walk. Single-DOF joints need only a rank-1
    downdate, but multi-DOF joints (e.g. a floating base) carry a 6x6 SPD
    inverse whose live registers roughly double the kernel's footprint and halve
    achieved occupancy. When ``single_dof_only`` that branch is compile-time
    removed, so the kernel keeps only the single-DOF working set; the deferred
    multi-DOF joints are then factored by :func:`_factor_reduced_multidof_kernel`
    in a companion launch. The split is only correct when every multi-DOF joint
    is an articulation root (depth 0), whose reduced inertia the depth walk never
    consumes.
    """
    module_name = "reduced_factor_singledof" if single_dof_only else "reduced_factor"
    if tile_width != 32:
        module_name += f"_{tile_width}"
    module = wp.get_module(module_name)

    @wp.func
    def _factor_reduced_warp_device(
        thread: wp.int32,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        factor_diagonal: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        body_i_s: wp.array[_sym_mat66],
        reduced_inertia: wp.array[_sym_mat66],
        joint_u: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
    ):
        articulation = thread // wp.int32(tile_width)
        lane = thread - articulation * wp.int32(tile_width)
        warp_lane = thread % wp.int32(32)
        group_start = warp_lane - lane
        group_mask = wp.uint32(wp.static((1 << tile_width) - 1)) << wp.uint32(group_start)

        for reverse_depth in range(max_depth + wp.int32(1)):
            depth = max_depth - reverse_depth
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                child = joint_child[joint]
                inertia = _unpack_symmetric_mat66(body_i_s[child])
                for child_index in range(child_start[joint], child_start[joint + wp.int32(1)]):
                    descendant_joint = child_joint[child_index]
                    inertia += _unpack_symmetric_mat66(reduced_inertia[joint_child[descendant_joint]])

                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + wp.int32(1)]
                dof_count = dof_end - dof_start
                if dof_count == wp.int32(1):
                    dof = dof_start
                    u = inertia * joint_s[dof]
                    joint_u[dof] = u
                    d_inv_scalar = wp.float32(1.0) / wp.max(
                        wp.dot(joint_s[dof], u) + factor_diagonal[dof], wp.float32(1.0e-20)
                    )
                    joint_d_inv[dof, 0] = d_inv_scalar
                    # In-place rank-1 downdate: u is already stored, so no live
                    # copy of the articulated inertia is needed (bit-identical).
                    for row in range(6):
                        for column in range(6):
                            inertia[row, column] -= u[row] * d_inv_scalar * u[column]
                else:
                    if wp.static(not single_dof_only):
                        if dof_count > wp.int32(1):
                            d = _mat66(0.0)
                            for column in range(_MAX_JOINT_DOF):
                                if wp.int32(column) < dof_count:
                                    dof = dof_start + wp.int32(column)
                                    u = inertia * joint_s[dof]
                                    joint_u[dof] = u
                                    for row in range(_MAX_JOINT_DOF):
                                        if wp.int32(row) < dof_count:
                                            d[row, column] = wp.dot(joint_s[dof_start + wp.int32(row)], u)
                                    d[column, column] += factor_diagonal[dof]

                            d_inv = _invert_spd(d, dof_count)
                            for row in range(_MAX_JOINT_DOF):
                                if wp.int32(row) < dof_count:
                                    for column in range(_MAX_JOINT_DOF):
                                        joint_d_inv[dof_start + wp.int32(row), column] = d_inv[row, column]

                            for row in range(6):
                                for column in range(6):
                                    correction = wp.float32(0.0)
                                    for a in range(_MAX_JOINT_DOF):
                                        if wp.int32(a) < dof_count:
                                            u_a = joint_u[dof_start + wp.int32(a)]
                                            for b in range(_MAX_JOINT_DOF):
                                                if wp.int32(b) < dof_count:
                                                    u_b = joint_u[dof_start + wp.int32(b)]
                                                    correction += u_a[row] * d_inv[a, b] * u_b[column]
                                    inertia[row, column] -= correction
                    # In single-DOF-only mode a multi-DOF joint leaves a defined
                    # (pre-reduction) entry here; its companion launch overwrites it.
                reduced_inertia[child] = _pack_symmetric_mat66(inertia)
                index += wp.int32(tile_width)
            _sync_reduced_group(group_mask)

    @wp.kernel(enable_backward=False, module=module, grid_stride=False)
    def _factor_reduced_warp_kernel(
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        factor_diagonal: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        body_i_s: wp.array[_sym_mat66],
        reduced_inertia: wp.array[_sym_mat66],
        joint_u: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
    ):
        _factor_reduced_warp_device(
            wp.tid(),
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_child,
            joint_qd_start,
            factor_diagonal,
            joint_s,
            child_start,
            child_joint,
            body_i_s,
            reduced_inertia,
            joint_u,
            joint_d_inv,
        )

    return _factor_reduced_warp_kernel


_FACTOR_REDUCED_WARP_KERNELS = {
    (single_dof_only, tile_width): _make_factor_reduced_warp_kernel(single_dof_only, tile_width)
    for single_dof_only in (False, True)
    for tile_width in (8, 16, 32)
}
_factor_reduced_warp_kernel = _FACTOR_REDUCED_WARP_KERNELS[False, 32]
_factor_reduced_singledof_warp_kernel = _FACTOR_REDUCED_WARP_KERNELS[True, 32]


@wp.func
def _boost_root_coordinates(
    root_joint: wp.int32,
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_s: wp.array[wp.spatial_vector],
    boost: wp.vec3,
    joint_qd_internal: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
):
    """Add a uniform translation to a floating root's internal and public rates.

    FREE and DISTANCE roots map their first three rates to orthonormal pure
    translations, so the boost projects onto those columns exactly.
    """
    dof_start = joint_qd_start[root_joint]
    for axis in range(3):
        dof = dof_start + wp.int32(axis)
        joint_qd_internal[dof] += wp.dot(wp.spatial_top(joint_s[dof]), boost)
    boost_parent = wp.quat_rotate_inv(wp.transform_get_rotation(joint_x_p[root_joint]), boost)
    joint_qd_public[dof_start] += boost_parent[0]
    joint_qd_public[dof_start + wp.int32(1)] += boost_parent[1]
    joint_qd_public[dof_start + wp.int32(2)] += boost_parent[2]


@wp.func
def _transport_root_twist(root_twist: wp.spatial_vector, dt: wp.float32) -> wp.spatial_vector:
    """Return a root COM twist re-referenced to the COM it advected to over ``dt``."""
    omega = wp.spatial_bottom(root_twist)
    linear = wp.spatial_top(root_twist)
    return wp.spatial_vector(linear + dt * wp.cross(omega, linear), omega)


@wp.func
def _carry_floating_root_twist(
    joint_qd: wp.array[wp.float32],
    dof_start: wp.int32,
    root_twist: wp.spatial_vector,
    dt: wp.float32,
) -> wp.spatial_vector:
    """Re-reference an integrated floating root's twist to its new pose.

    ABA advances spatial twists about the fixed articulation origin, the root
    COM at the factored pose. Integration moves that point and the joint
    anchor by their own velocity times ``dt`` and the next pose re-centers on
    them, so the unchanged spatial twist field reads ``dt * omega x v`` more
    there. Dropping that term rotates a spinning body's linear momentum.

    Updates the internal anchor-point velocity in place and returns the root
    twist about the new origin.
    """
    anchor_linear = wp.vec3(joint_qd[dof_start], joint_qd[dof_start + wp.int32(1)], joint_qd[dof_start + wp.int32(2)])
    anchor_omega = wp.vec3(
        joint_qd[dof_start + wp.int32(3)], joint_qd[dof_start + wp.int32(4)], joint_qd[dof_start + wp.int32(5)]
    )
    anchor_linear += dt * wp.cross(anchor_omega, anchor_linear)
    joint_qd[dof_start] = anchor_linear[0]
    joint_qd[dof_start + wp.int32(1)] = anchor_linear[1]
    joint_qd[dof_start + wp.int32(2)] = anchor_linear[2]
    return _transport_root_twist(root_twist, dt)


@wp.func
def _articulation_linear_momentum(
    start: wp.int32,
    end: wp.int32,
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    bodies: BodyContainer,
) -> wp.vec4:
    """Return the tree's linear momentum and total mass from published COM velocities."""
    momentum = wp.vec3(0.0)
    total_mass = wp.float32(0.0)
    for joint in range(start, end):
        body = joint_child[joint]
        mass = body_mass[body]
        momentum += mass * bodies.velocity[body + wp.int32(1)]
        total_mass += mass
    return wp.vec4(momentum[0], momentum[1], momentum[2], total_mass)


@wp.kernel(enable_backward=False)
def _capture_reduced_linear_momentum_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    bodies: BodyContainer,
    target_momentum: wp.array[wp.vec3],
):
    articulation = wp.tid()
    momentum = _articulation_linear_momentum(
        articulation_start[articulation], articulation_end[articulation], joint_child, body_mass, bodies
    )
    target_momentum[articulation] = wp.vec3(momentum[0], momentum[1], momentum[2])


@wp.kernel(enable_backward=False)
def _boost_reduced_linear_momentum_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    body_mass: wp.array[wp.float32],
    target_momentum: wp.array[wp.vec3],
    joint_s_publish: wp.array[wp.spatial_vector],
    bodies: BodyContainer,
    joint_qd_internal: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
    body_qd_public: wp.array[wp.spatial_vector],
    midpoint_flag: wp.array[wp.int32],
):
    """Restore a floating tree's linear momentum with a uniform Galilean boost.

    Trees integrated by :func:`_midpoint_momentum_step` conserve it already.
    """
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    root_type = joint_type[start]
    if root_type != JointType.FREE and root_type != JointType.DISTANCE:
        return
    if midpoint_flag[articulation] != wp.int32(0):
        return
    current = _articulation_linear_momentum(start, end, joint_child, body_mass, bodies)
    if current[3] <= wp.float32(0.0):
        return
    boost = (target_momentum[articulation] - wp.vec3(current[0], current[1], current[2])) / current[3]
    for joint in range(start, end):
        body = joint_child[joint]
        slot = body + wp.int32(1)
        bodies.velocity[slot] = bodies.velocity[slot] + boost
        body_qd_public[body] = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
    _boost_root_coordinates(
        start, joint_qd_start, joint_x_p, joint_s_publish, boost, joint_qd_internal, joint_qd_public
    )


@wp.func
def _pose_from_coordinates(
    joint: wp.int32,
    parent: wp.int32,
    kind: wp.int32,
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
) -> wp.transform:
    """Return a child's world pose from its joint coordinates and its parent's pose."""
    parent_anchor = joint_x_p[joint]
    if parent >= wp.int32(0):
        parent_anchor = body_q[parent] * parent_anchor
    joint_transform = jcalc_transform(
        kind,
        joint_axis,
        joint_qd_start[joint],
        joint_dof_dim[joint, 0],
        joint_dof_dim[joint, 1],
        joint_q,
        joint_q_start[joint],
    )
    return parent_anchor * joint_transform * wp.transform_inverse(joint_x_c[joint])


@wp.kernel(enable_backward=False)
def _finish_reduced_articulations_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd_local: wp.array[wp.float32],
    joint_qd_integrator: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    zero_generalized_acceleration: wp.array[wp.float32],
    dt: wp.float32,
    update_configuration: wp.bool,
    midpoint_flag: wp.array[wp.int32],
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    bodies: BodyContainer,
):
    """Integrate and publish one articulation without world-origin cancellation.

    Trees flagged in ``midpoint_flag`` already hold their end-of-step
    coordinates and floating-root rates, see :func:`_midpoint_momentum_step`.
    """
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    data = bodies.reduced
    integrated = update_configuration and midpoint_flag[articulation] != wp.int32(0)

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        kind = joint_type[joint]
        if integrated:
            body_q[child] = _pose_from_coordinates(
                joint,
                parent,
                kind,
                joint_x_p,
                joint_x_c,
                joint_axis,
                joint_dof_dim,
                joint_q,
                joint_q_start,
                joint_qd_start,
                body_q,
            )
            continue
        if kind == JointType.FREE or kind == JointType.DISTANCE:
            slot = child + wp.int32(1)
            child_twist = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
            if parent < wp.int32(0):
                # Move the root COM with its re-referenced velocity, see _carry_floating_root_twist.
                child_twist = _transport_root_twist(child_twist, dt)
            new_body_q = integrate_body_pose_from_com_twist(body_q[child], body_com[child], child_twist, dt)
            parent_anchor = joint_x_p[joint]
            if parent >= wp.int32(0):
                parent_anchor = body_q[parent] * parent_anchor
            joint_transform = wp.transform_inverse(parent_anchor) * new_body_q * joint_x_c[joint]
            coord_start = joint_q_start[joint]
            position = wp.transform_get_translation(joint_transform)
            rotation = wp.transform_get_rotation(joint_transform)
            joint_q[coord_start] = position[0]
            joint_q[coord_start + wp.int32(1)] = position[1]
            joint_q[coord_start + wp.int32(2)] = position[2]
            joint_q[coord_start + wp.int32(3)] = rotation[0]
            joint_q[coord_start + wp.int32(4)] = rotation[1]
            joint_q[coord_start + wp.int32(5)] = rotation[2]
            joint_q[coord_start + wp.int32(6)] = rotation[3]
        else:
            jcalc_integrate(
                parent,
                joint_x_c[joint],
                body_com[child],
                kind,
                joint_q,
                joint_qd_integrator,
                zero_generalized_acceleration,
                joint_q_start[joint],
                joint_qd_start[joint],
                joint_dof_dim[joint, 0],
                joint_dof_dim[joint, 1],
                dt,
                joint_q,
                joint_qd_integrator,
            )
            joint_transform = jcalc_transform(
                kind,
                joint_axis,
                joint_qd_start[joint],
                joint_dof_dim[joint, 0],
                joint_dof_dim[joint, 1],
                joint_q,
                joint_q_start[joint],
            )
            parent_anchor = joint_x_p[joint]
            if parent >= wp.int32(0):
                parent_anchor = body_q[parent] * parent_anchor
            new_body_q = parent_anchor * joint_transform * wp.transform_inverse(joint_x_c[joint])
        body_q[child] = new_body_q

    for joint in range(start, end):
        kind = joint_type[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        if kind != JointType.FREE and kind != JointType.DISTANCE:
            for dof in range(dof_start, dof_end):
                joint_qd_public[dof] = joint_qd_local[dof]
        elif joint_parent[joint] < wp.int32(0):
            if not integrated:
                root_twist = wp.spatial_vector()
                for dof in range(dof_start, dof_end):
                    root_twist += data.joint_s[dof] * joint_qd_local[dof]
                root_twist = _carry_floating_root_twist(joint_qd_local, dof_start, root_twist, dt)
                parent_rotation = wp.transform_get_rotation(joint_x_p[joint])
                velocity_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_top(root_twist))
                omega_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_bottom(root_twist))
                joint_qd_public[dof_start] = velocity_parent[0]
                joint_qd_public[dof_start + wp.int32(1)] = velocity_parent[1]
                joint_qd_public[dof_start + wp.int32(2)] = velocity_parent[2]
                joint_qd_public[dof_start + wp.int32(3)] = omega_parent[0]
                joint_qd_public[dof_start + wp.int32(4)] = omega_parent[1]
                joint_qd_public[dof_start + wp.int32(5)] = omega_parent[2]
        else:
            joint_transform = jcalc_transform(
                kind,
                joint_axis,
                dof_start,
                joint_dof_dim[joint, 0],
                joint_dof_dim[joint, 1],
                joint_q,
                joint_q_start[joint],
            )
            child_from_parent_anchor = joint_transform * wp.transform_inverse(joint_x_c[joint])
            child_com_parent = wp.transform_point(child_from_parent_anchor, body_com[joint_child[joint]])
            velocity_origin = wp.vec3(
                joint_qd_integrator[dof_start],
                joint_qd_integrator[dof_start + wp.int32(1)],
                joint_qd_integrator[dof_start + wp.int32(2)],
            )
            omega_parent = wp.vec3(
                joint_qd_integrator[dof_start + wp.int32(3)],
                joint_qd_integrator[dof_start + wp.int32(4)],
                joint_qd_integrator[dof_start + wp.int32(5)],
            )
            velocity_com = velocity_origin + wp.cross(omega_parent, child_com_parent)
            joint_qd_public[dof_start] = velocity_com[0]
            joint_qd_public[dof_start + wp.int32(1)] = velocity_com[1]
            joint_qd_public[dof_start + wp.int32(2)] = velocity_com[2]
            joint_qd_public[dof_start + wp.int32(3)] = omega_parent[0]
            joint_qd_public[dof_start + wp.int32(4)] = omega_parent[1]
            joint_qd_public[dof_start + wp.int32(5)] = omega_parent[2]

    for joint in range(start, end):
        child = joint_child[joint]
        transform = body_q[child]
        rotation = wp.transform_get_rotation(transform)
        origin = wp.transform_get_translation(transform)
        slot = child + wp.int32(1)
        bodies.position[slot] = origin + wp.quat_rotate(rotation, body_com[child])
        bodies.orientation[slot] = rotation


@wp.kernel(enable_backward=False)
def _publish_reduced_local_velocities_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_anchor_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_qd_local: wp.array[wp.float32],
    bodies: BodyContainer,
    joint_s_publish: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Publish COM twists entirely in articulation-local coordinates."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]

    for joint in range(start, end):
        jcalc_motion(
            joint_type[joint],
            joint_axis,
            joint_q,
            joint_dof_dim[joint, 0],
            joint_dof_dim[joint, 1],
            joint_anchor_local[joint],
            joint_qd_local,
            joint_q_start[joint],
            joint_qd_start[joint],
            joint_s_publish,
        )

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        twist = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_com_twist = body_qd[parent]
            parent_omega = wp.spatial_bottom(parent_com_twist)
            parent_com_position = wp.transform_get_translation(body_q_com[parent])
            twist = wp.spatial_vector(
                wp.spatial_top(parent_com_twist) - wp.cross(parent_omega, parent_com_position),
                parent_omega,
            )
        for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
            twist += joint_s_publish[dof] * joint_qd_local[dof]
        omega = wp.spatial_bottom(twist)
        com_position = wp.transform_get_translation(body_q_com[child])
        com_twist = wp.spatial_vector(wp.spatial_top(twist) + wp.cross(omega, com_position), omega)
        body_qd[child] = com_twist
        slot = child + wp.int32(1)
        bodies.velocity[slot] = wp.spatial_top(com_twist)
        bodies.angular_velocity[slot] = omega


@wp.func
def _finish_and_publish_reduced_warp_device(
    thread: wp.int32,
    articulation_count: wp.int32,
    tile_width: wp.int32,
    max_depth: wp.int32,
    articulation_depth_start: wp.array2d[wp.int32],
    articulation_depth_joint: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd_local: wp.array[wp.float32],
    joint_qd_integrator: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    zero_generalized_acceleration: wp.array[wp.float32],
    dt: wp.float32,
    update_configuration: wp.bool,
    body_mass: wp.array[wp.float32],
    target_linear_momentum: wp.vec3,
    restore_linear_momentum: wp.bool,
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    bodies: BodyContainer,
    articulation_origin: wp.array[wp.vec3],
    body_q_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_anchor_local: wp.array[wp.transform],
    joint_s_publish: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
    midpoint_flag: wp.array[wp.int32],
):
    articulation = thread // tile_width
    lane = thread - articulation * tile_width
    if articulation >= articulation_count:
        return
    warp_lane = thread & wp.int32(31)
    group = warp_lane // tile_width
    group_mask = wp.uint32(4294967295)
    if tile_width < wp.int32(32):
        group_mask = ((wp.uint32(1) << wp.uint32(tile_width)) - wp.uint32(1)) << wp.uint32(group * tile_width)
    data = bodies.reduced
    # A tree the midpoint step integrated already holds its end-of-step coordinates
    # and floating-root rates, see _midpoint_momentum_step.
    integrated = wp.bool(False)
    if update_configuration:
        integrated = midpoint_flag[articulation] != wp.int32(0)

    if update_configuration:
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                parent = joint_parent[joint]
                child = joint_child[joint]
                kind = joint_type[joint]
                if integrated:
                    body_q[child] = _pose_from_coordinates(
                        joint,
                        parent,
                        kind,
                        joint_x_p,
                        joint_x_c,
                        joint_axis,
                        joint_dof_dim,
                        joint_q,
                        joint_q_start,
                        joint_qd_start,
                        body_q,
                    )
                    index += tile_width
                    continue
                if kind == JointType.FREE or kind == JointType.DISTANCE:
                    slot = child + wp.int32(1)
                    child_twist = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
                    if parent < wp.int32(0):
                        # Move the root COM with its re-referenced velocity, see _carry_floating_root_twist.
                        child_twist = _transport_root_twist(child_twist, dt)
                    new_body_q = integrate_body_pose_from_com_twist(body_q[child], body_com[child], child_twist, dt)
                    parent_anchor = joint_x_p[joint]
                    if parent >= wp.int32(0):
                        parent_anchor = body_q[parent] * parent_anchor
                    joint_transform = wp.transform_inverse(parent_anchor) * new_body_q * joint_x_c[joint]
                    coord_start = joint_q_start[joint]
                    position = wp.transform_get_translation(joint_transform)
                    rotation = wp.transform_get_rotation(joint_transform)
                    joint_q[coord_start] = position[0]
                    joint_q[coord_start + wp.int32(1)] = position[1]
                    joint_q[coord_start + wp.int32(2)] = position[2]
                    joint_q[coord_start + wp.int32(3)] = rotation[0]
                    joint_q[coord_start + wp.int32(4)] = rotation[1]
                    joint_q[coord_start + wp.int32(5)] = rotation[2]
                    joint_q[coord_start + wp.int32(6)] = rotation[3]
                else:
                    jcalc_integrate(
                        parent,
                        joint_x_c[joint],
                        body_com[child],
                        kind,
                        joint_q,
                        joint_qd_integrator,
                        zero_generalized_acceleration,
                        joint_q_start[joint],
                        joint_qd_start[joint],
                        joint_dof_dim[joint, 0],
                        joint_dof_dim[joint, 1],
                        dt,
                        joint_q,
                        joint_qd_integrator,
                    )
                    joint_transform = jcalc_transform(
                        kind,
                        joint_axis,
                        joint_qd_start[joint],
                        joint_dof_dim[joint, 0],
                        joint_dof_dim[joint, 1],
                        joint_q,
                        joint_q_start[joint],
                    )
                    parent_anchor = joint_x_p[joint]
                    if parent >= wp.int32(0):
                        parent_anchor = body_q[parent] * parent_anchor
                    new_body_q = parent_anchor * joint_transform * wp.transform_inverse(joint_x_c[joint])
                body_q[child] = new_body_q
                index += tile_width
            _sync_reduced_group(group_mask)

    for depth in range(max_depth + wp.int32(1)):
        index = articulation_depth_start[articulation, depth] + lane
        depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = articulation_depth_joint[index]
            kind = joint_type[joint]
            dof_start = joint_qd_start[joint]
            dof_end = joint_qd_start[joint + wp.int32(1)]
            if kind != JointType.FREE and kind != JointType.DISTANCE:
                for dof in range(dof_start, dof_end):
                    joint_qd_public[dof] = joint_qd_local[dof]
            elif joint_parent[joint] < wp.int32(0):
                if not integrated:
                    root_twist = wp.spatial_vector()
                    for dof in range(dof_start, dof_end):
                        root_twist += data.joint_s[dof] * joint_qd_local[dof]
                    if update_configuration:
                        root_twist = _carry_floating_root_twist(joint_qd_local, dof_start, root_twist, dt)
                    parent_rotation = wp.transform_get_rotation(joint_x_p[joint])
                    velocity_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_top(root_twist))
                    omega_parent = wp.quat_rotate_inv(parent_rotation, wp.spatial_bottom(root_twist))
                    joint_qd_public[dof_start] = velocity_parent[0]
                    joint_qd_public[dof_start + wp.int32(1)] = velocity_parent[1]
                    joint_qd_public[dof_start + wp.int32(2)] = velocity_parent[2]
                    joint_qd_public[dof_start + wp.int32(3)] = omega_parent[0]
                    joint_qd_public[dof_start + wp.int32(4)] = omega_parent[1]
                    joint_qd_public[dof_start + wp.int32(5)] = omega_parent[2]
            else:
                joint_transform = jcalc_transform(
                    kind,
                    joint_axis,
                    dof_start,
                    joint_dof_dim[joint, 0],
                    joint_dof_dim[joint, 1],
                    joint_q,
                    joint_q_start[joint],
                )
                child_from_parent_anchor = joint_transform * wp.transform_inverse(joint_x_c[joint])
                child_com_parent = wp.transform_point(child_from_parent_anchor, body_com[joint_child[joint]])
                velocity_origin = wp.vec3(
                    joint_qd_integrator[dof_start],
                    joint_qd_integrator[dof_start + wp.int32(1)],
                    joint_qd_integrator[dof_start + wp.int32(2)],
                )
                omega_parent = wp.vec3(
                    joint_qd_integrator[dof_start + wp.int32(3)],
                    joint_qd_integrator[dof_start + wp.int32(4)],
                    joint_qd_integrator[dof_start + wp.int32(5)],
                )
                if not update_configuration:
                    velocity_origin = wp.vec3(
                        joint_qd_local[dof_start],
                        joint_qd_local[dof_start + wp.int32(1)],
                        joint_qd_local[dof_start + wp.int32(2)],
                    )
                    omega_parent = wp.vec3(
                        joint_qd_local[dof_start + wp.int32(3)],
                        joint_qd_local[dof_start + wp.int32(4)],
                        joint_qd_local[dof_start + wp.int32(5)],
                    )
                velocity_com = velocity_origin + wp.cross(omega_parent, child_com_parent)
                joint_qd_public[dof_start] = velocity_com[0]
                joint_qd_public[dof_start + wp.int32(1)] = velocity_com[1]
                joint_qd_public[dof_start + wp.int32(2)] = velocity_com[2]
                joint_qd_public[dof_start + wp.int32(3)] = omega_parent[0]
                joint_qd_public[dof_start + wp.int32(4)] = omega_parent[1]
                joint_qd_public[dof_start + wp.int32(5)] = omega_parent[2]

            child = joint_child[joint]
            if update_configuration:
                transform = body_q[child]
                rotation = wp.transform_get_rotation(transform)
                origin = wp.transform_get_translation(transform)
                slot = child + wp.int32(1)
                bodies.position[slot] = origin + wp.quat_rotate(rotation, body_com[child])
                bodies.orientation[slot] = rotation
            index += tile_width
        _sync_reduced_group(group_mask)

    if update_configuration:
        if lane == wp.int32(0):
            root_joint = articulation_depth_joint[articulation_depth_start[articulation, 0]]
            root = joint_child[root_joint]
            articulation_origin[articulation] = wp.transform_point(body_q[root], body_com[root])

        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                parent = joint_parent[joint]
                child = joint_child[joint]
                joint_transform = jcalc_transform(
                    joint_type[joint],
                    joint_axis,
                    joint_qd_start[joint],
                    joint_dof_dim[joint, 0],
                    joint_dof_dim[joint, 1],
                    joint_q,
                    joint_q_start[joint],
                )
                if parent < wp.int32(0):
                    rotation = wp.transform_get_rotation(body_q[child])
                    child_transform = wp.transform(-wp.quat_rotate(rotation, body_com[child]), rotation)
                    child_anchor = child_transform * joint_x_c[joint]
                    root_joint_transform = joint_transform
                    if joint_type[joint] == JointType.FREE or joint_type[joint] == JointType.DISTANCE:
                        root_joint_transform = wp.transform(wp.vec3(0.0), wp.transform_get_rotation(joint_transform))
                    parent_anchor = child_anchor * wp.transform_inverse(root_joint_transform)
                else:
                    parent_anchor = body_q_local[parent] * joint_x_p[joint]
                    child_anchor = parent_anchor * joint_transform
                    child_transform = child_anchor * wp.transform_inverse(joint_x_c[joint])
                joint_anchor_local[joint] = parent_anchor
                body_q_local[child] = child_transform
                body_q_com[child] = child_transform * body_x_com[child]
                index += tile_width
            _sync_reduced_group(group_mask)

    for depth in range(max_depth + wp.int32(1)):
        index = articulation_depth_start[articulation, depth] + lane
        depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = articulation_depth_joint[index]
            parent = joint_parent[joint]
            child = joint_child[joint]
            if update_configuration:
                jcalc_motion(
                    joint_type[joint],
                    joint_axis,
                    joint_q,
                    joint_dof_dim[joint, 0],
                    joint_dof_dim[joint, 1],
                    joint_anchor_local[joint],
                    joint_qd_local,
                    joint_q_start[joint],
                    joint_qd_start[joint],
                    joint_s_publish,
                )
            twist = wp.spatial_vector()
            if parent >= wp.int32(0):
                parent_slot = parent + wp.int32(1)
                parent_com_twist = wp.spatial_vector(
                    bodies.velocity[parent_slot],
                    bodies.angular_velocity[parent_slot],
                )
                parent_omega = wp.spatial_bottom(parent_com_twist)
                parent_com_position = wp.transform_get_translation(body_q_com[parent])
                twist = wp.spatial_vector(
                    wp.spatial_top(parent_com_twist) - wp.cross(parent_omega, parent_com_position),
                    parent_omega,
                )
            for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                twist += joint_s_publish[dof] * joint_qd_local[dof]
            omega = wp.spatial_bottom(twist)
            com_position = wp.transform_get_translation(body_q_com[child])
            com_twist = wp.spatial_vector(wp.spatial_top(twist) + wp.cross(omega, com_position), omega)
            slot = child + wp.int32(1)
            bodies.velocity[slot] = wp.spatial_top(com_twist)
            bodies.angular_velocity[slot] = omega
            index += tile_width
        _sync_reduced_group(group_mask)

    tree_start = articulation_depth_start[articulation, 0]
    tree_end = articulation_depth_start[articulation, max_depth + wp.int32(1)]
    root_kind = wp.int32(JointType.FIXED)
    if tree_end > tree_start:
        root_kind = joint_type[articulation_depth_joint[tree_start]]
    if (
        update_configuration
        and restore_linear_momentum
        and not integrated
        and (root_kind == JointType.FREE or root_kind == JointType.DISTANCE)
    ):
        root_joint = articulation_depth_joint[tree_start]
        local = wp.vec4(0.0)
        index = tree_start + lane
        while index < tree_end:
            body = joint_child[articulation_depth_joint[index]]
            mass = body_mass[body]
            momentum = mass * bodies.velocity[body + wp.int32(1)]
            local += wp.vec4(momentum[0], momentum[1], momentum[2], mass)
            index += tile_width
        total = _sum_reduced_group_vec4(local, tile_width, group_mask)
        if total[3] > wp.float32(0.0):
            boost = (target_linear_momentum - wp.vec3(total[0], total[1], total[2])) / total[3]
            index = tree_start + lane
            while index < tree_end:
                slot = joint_child[articulation_depth_joint[index]] + wp.int32(1)
                bodies.velocity[slot] = bodies.velocity[slot] + boost
                index += tile_width
            if lane == wp.int32(0):
                _boost_root_coordinates(
                    root_joint, joint_qd_start, joint_x_p, joint_s_publish, boost, joint_qd_local, joint_qd_public
                )
        _sync_reduced_group(group_mask)


@wp.kernel(enable_backward=False, module="reduced_publish", grid_stride=False)
def _finish_and_publish_reduced_warp_kernel(
    articulation_count: wp.int32,
    tile_width: wp.int32,
    max_depth: wp.int32,
    articulation_depth_start: wp.array2d[wp.int32],
    articulation_depth_joint: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd_local: wp.array[wp.float32],
    joint_qd_integrator: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    zero_generalized_acceleration: wp.array[wp.float32],
    dt: wp.float32,
    update_configuration: wp.bool,
    body_mass: wp.array[wp.float32],
    target_linear_momentum: wp.array[wp.vec3],
    restore_linear_momentum: wp.bool,
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    bodies: BodyContainer,
    articulation_origin: wp.array[wp.vec3],
    body_q_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_anchor_local: wp.array[wp.transform],
    joint_s_publish: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
    midpoint_flag: wp.array[wp.int32],
):
    articulation = wp.tid() // tile_width
    target = wp.vec3(0.0)
    if restore_linear_momentum and articulation < articulation_count:
        target = target_linear_momentum[articulation]
    _finish_and_publish_reduced_warp_device(
        wp.tid(),
        articulation_count,
        tile_width,
        max_depth,
        articulation_depth_start,
        articulation_depth_joint,
        joint_q,
        joint_qd_local,
        joint_qd_integrator,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_x_p,
        joint_x_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        body_x_com,
        zero_generalized_acceleration,
        dt,
        update_configuration,
        body_mass,
        target,
        restore_linear_momentum,
        joint_qd_public,
        body_q,
        bodies,
        articulation_origin,
        body_q_local,
        body_q_com,
        joint_anchor_local,
        joint_s_publish,
        body_qd,
        midpoint_flag,
    )


@wp.kernel(enable_backward=False)
def _solve_articulated_system_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    joint_u_matrix: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
    generalized_force: wp.array[wp.float32],
    body_force: wp.array[wp.spatial_vector],
    body_work: wp.array[wp.spatial_vector],
    joint_work: wp.array[wp.float32],
    body_acceleration: wp.array[wp.spatial_vector],
    generalized_acceleration: wp.array[wp.float32],
):
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]

    for joint in range(start, end):
        child = joint_child[joint]
        body_work[child] = -body_force[child]
        body_acceleration[child] = wp.spatial_vector()

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        p = body_work[child]
        reduced_force = _vec6(0.0)
        d_inv_u = _vec6(0.0)

        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                reduced_force[row] = generalized_force[dof] - wp.dot(joint_s[dof], p)
                joint_work[dof] = reduced_force[row]
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        d_inv_u[row] += joint_d_inv[dof_start + wp.int32(row), column] * reduced_force[column]

        propagated = p
        for column in range(_MAX_JOINT_DOF):
            if wp.int32(column) < dof_count:
                propagated += joint_u_matrix[dof_start + wp.int32(column)] * d_inv_u[column]
        if parent >= wp.int32(0):
            body_work[parent] = body_work[parent] + propagated

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        parent_acceleration = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_acceleration = body_acceleration[parent]

        rhs = _vec6(0.0)
        qdd = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = joint_work[dof] - wp.dot(joint_u_matrix[dof], parent_acceleration)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        qdd[row] += joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]

        child_acceleration = parent_acceleration
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                generalized_acceleration[dof] = qdd[row]
                child_acceleration += joint_s[dof] * qdd[row]
        body_acceleration[child] = child_acceleration


@wp.kernel(enable_backward=False)
def _advance_reduced_articulations_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    max_depth: wp.int32,
    articulation_depth_start: wp.array2d[wp.int32],
    articulation_depth_joint: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_qd: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    joint_u_matrix: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array2d[wp.float32],
    joint_f: wp.array[wp.float32],
    joint_implicit_force: wp.array[wp.float32],
    body_mass: wp.array[wp.float32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    body_q_com: wp.array[wp.transform],
    body_inertia: wp.array[_sym_mat66],
    external_force_com: wp.array[wp.spatial_vector],
    dt: wp.float32,
    include_external: wp.bool,
    include_coriolis: wp.bool,
    body_velocity: wp.array[wp.spatial_vector],
    body_coriolis: wp.array[wp.spatial_vector],
    body_bias: wp.array[wp.spatial_vector],
    generalized_rhs: wp.array[wp.float32],
    body_work: wp.array[wp.spatial_vector],
    joint_work: wp.array[wp.float32],
    body_acceleration: wp.array[wp.spatial_vector],
    generalized_acceleration: wp.array[wp.float32],
    public_body_qd: wp.array[wp.spatial_vector],
    bodies: BodyContainer,
    midpoint: _MidpointWorkspace,
):
    """Advance unconstrained ABA dynamics and publish all link velocities.

    A velocity-product-only pass (``include_coriolis`` without
    ``include_external``) hands a tree whose explicit step would gain more
    energy than ``_MIDPOINT_ENERGY_RATE`` per second, see
    :func:`_explicit_step_drift_terms`, to :func:`_midpoint_momentum_step`,
    which also advances its pose. Other trees keep the explicit step.
    """
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    measure_drift = include_coriolis and not include_external

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        parent_velocity = wp.spatial_vector()
        parent_coriolis = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_velocity = body_velocity[parent]
            parent_coriolis = body_coriolis[parent]
        joint_velocity = wp.spatial_vector()
        for dof in range(dof_start, dof_end):
            joint_velocity += joint_s[dof] * joint_qd[dof]
        velocity = parent_velocity + joint_velocity
        coriolis = parent_coriolis + spatial_cross(velocity, joint_velocity)
        inertia = _unpack_symmetric_mat66(body_inertia[child])
        x_com = wp.transform_get_translation(body_q_com[child])
        bias = wp.spatial_vector()
        if include_coriolis:
            bias += inertia * coriolis + spatial_cross_dual(velocity, inertia * velocity)
        if include_external:
            gravity_force = body_mass[child] * gravity[wp.max(body_world[child], wp.int32(0))]
            gravity_wrench = wp.spatial_vector(gravity_force, wp.cross(x_com, gravity_force))
            force_com = external_force_com[child]
            force = wp.spatial_top(force_com)
            torque = wp.spatial_bottom(force_com) + wp.cross(x_com, force)
            bias -= gravity_wrench + wp.spatial_vector(force, torque)

            joint_kind = joint_type[joint]
            if joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE:
                control_force = wp.vec3(
                    joint_f[dof_start],
                    joint_f[dof_start + wp.int32(1)],
                    joint_f[dof_start + wp.int32(2)],
                )
                control_torque = wp.vec3(
                    joint_f[dof_start + wp.int32(3)],
                    joint_f[dof_start + wp.int32(4)],
                    joint_f[dof_start + wp.int32(5)],
                )
                bias -= wp.spatial_vector(
                    control_force,
                    control_torque + wp.cross(x_com, control_force),
                )
        body_velocity[child] = velocity
        body_coriolis[child] = coriolis
        body_bias[child] = bias

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        joint_kind = joint_type[joint]
        subtree_bias = body_bias[child]
        for dof in range(dof_start, dof_end):
            applied = wp.float32(0.0)
            if include_external and joint_kind != JointType.FREE and joint_kind != JointType.DISTANCE:
                applied = joint_f[dof]
            if include_external and (
                joint_kind == JointType.REVOLUTE or joint_kind == JointType.PRISMATIC or joint_kind == JointType.D6
            ):
                applied += joint_implicit_force[dof]
            generalized_rhs[dof] = applied - wp.dot(joint_s[dof], subtree_bias)
        if parent >= wp.int32(0):
            body_bias[parent] = body_bias[parent] + subtree_bias

    for joint in range(start, end):
        child = joint_child[joint]
        body_work[child] = wp.spatial_vector()
        body_acceleration[child] = wp.spatial_vector()

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        p = body_work[child]
        reduced_force = _vec6(0.0)
        d_inv_u = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                reduced_force[row] = generalized_rhs[dof] - wp.dot(joint_s[dof], p)
                joint_work[dof] = reduced_force[row]
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        d_inv_u[row] += joint_d_inv[dof_start + wp.int32(row), column] * reduced_force[column]
        propagated = p
        for column in range(_MAX_JOINT_DOF):
            if wp.int32(column) < dof_count:
                propagated += joint_u_matrix[dof_start + wp.int32(column)] * d_inv_u[column]
        if parent >= wp.int32(0):
            body_work[parent] = body_work[parent] + propagated

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        parent_acceleration = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_acceleration = body_acceleration[parent]
        rhs = _vec6(0.0)
        qdd = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = joint_work[dof] - wp.dot(joint_u_matrix[dof], parent_acceleration)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        qdd[row] += joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]
        child_acceleration = parent_acceleration
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                generalized_acceleration[dof] = qdd[row]
                if measure_drift:
                    # The midpoint step restarts from the incoming rates.
                    midpoint.joint_qd_mid[dof] = joint_qd[dof]
                joint_qd[dof] += qdd[row] * dt
                child_acceleration += joint_s[dof] * qdd[row]
        body_acceleration[child] = child_acceleration

    # Same per-tree integrator choice as the warp advance.
    frame = wp.spatial_vector()
    drift = wp.float32(0.0)
    kinetic_energy = wp.float32(0.0)
    dissipation = wp.float32(0.0)
    root_kind = joint_type[start]
    if measure_drift and (root_kind == JointType.FREE or root_kind == JointType.DISTANCE):
        frame = _floating_root_frame(body_velocity[joint_child[start]])

    # The configuration is unchanged, so the factored motion subspaces
    # directly reconstruct twists about each articulation origin.
    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        twist = wp.spatial_vector()
        if parent >= wp.int32(0):
            twist = body_velocity[parent]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        for dof in range(dof_start, dof_end):
            twist += joint_s[dof] * joint_qd[dof]
        if measure_drift:
            joint_velocity = wp.spatial_vector()
            joint_acceleration = wp.spatial_vector()
            for dof in range(dof_start, dof_end):
                joint_velocity += joint_s[dof] * midpoint.joint_qd_mid[dof]
                dissipation += _joint_dissipation(midpoint, dof, midpoint.joint_qd_mid[dof])
                joint_acceleration += joint_s[dof] * generalized_acceleration[dof]
            parent_coriolis_rate = wp.spatial_vector()
            if parent >= wp.int32(0):
                parent_coriolis_rate = body_coriolis[parent]
            terms = _explicit_step_drift_terms(
                body_velocity[child],
                body_acceleration[child],
                joint_velocity,
                joint_acceleration,
                joint == start,
                frame,
                _unpack_symmetric_mat66(body_inertia[child]) * body_velocity[child],
                body_mass[child],
                wp.transform_get_translation(body_q_com[child]),
                parent_coriolis_rate,
            )
            body_coriolis[child] = wp.spatial_vector(terms[0], terms[1], terms[2], terms[3], terms[4], terms[5])
            drift += terms[6]
            kinetic_energy += terms[7]
        body_velocity[child] = twist
        omega = wp.spatial_bottom(twist)
        x_com = wp.transform_get_translation(body_q_com[child])
        velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
        public_body_qd[child] = wp.spatial_vector(velocity_com, omega)
        slot = child + wp.int32(1)
        bodies.velocity[slot] = velocity_com
        bodies.angular_velocity[slot] = omega

    if measure_drift:
        if _explicit_step_drifts(drift, kinetic_energy, dissipation, dt):
            tree = _MidpointTree()
            tree.articulation = articulation
            tree.lane = wp.int32(0)
            tree.width = wp.int32(1)
            tree.mask = wp.uint32(1) << wp.uint32(articulation & wp.int32(31))
            tree.max_depth = max_depth
            tree.depth_start = articulation_depth_start
            tree.order = articulation_depth_joint
            _restore_incoming_rates(tree, joint_qd_start, midpoint, joint_qd)
            _midpoint_momentum_step(
                tree,
                joint_type,
                joint_parent,
                joint_child,
                joint_qd_start,
                joint_s,
                body_inertia,
                midpoint,
                dt,
                wp.int32(_MIDPOINT_MAX_ITERATIONS),
                joint_qd,
                body_velocity,
            )
        else:
            midpoint.flag[articulation] = wp.int32(0)


@functools.cache
def _make_advance_reduced_articulations_warp_ops(
    *,
    include_external: bool,
    include_coriolis: bool,
    fuse_publish: bool = False,
    static_tile_width: int = 0,
    dense: bool = False,
):
    """Build the advance device function and kernel.

    ``dense`` replaces the articulated-body solve by the dense factor of
    :class:`~newton._src.solvers.phoenx.articulations.dense_tree.DenseTreeOperator`:
    the reverse pass only accumulates the generalized right-hand side, which
    the lane group then solves with the stored tree-sparse factor.
    """
    suffix = "_publish" if fuse_publish else ""
    if dense:
        suffix += "_dense"
    module = wp.get_module(
        f"reduced_advance_{int(include_external)}_{int(include_coriolis)}_{static_tile_width}{suffix}"
    )

    @wp.func
    def _advance_reduced_articulations_warp_device(
        thread: wp.int32,
        articulation_count: wp.int32,
        tile_width: wp.int32,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        joint_parent_lane: wp.array[wp.int32],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_qd: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        joint_u_matrix: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
        joint_f: wp.array[wp.float32],
        joint_implicit_force: wp.array[wp.float32],
        body_mass: wp.array[wp.float32],
        body_world: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        body_q_com: wp.array[wp.transform],
        body_inertia: wp.array[_sym_mat66],
        external_force_com: wp.array[wp.spatial_vector],
        dt: wp.float32,
        body_velocity: wp.array[wp.spatial_vector],
        body_coriolis: wp.array[wp.spatial_vector],
        body_bias: wp.array[wp.spatial_vector],
        generalized_rhs: wp.array[wp.float32],
        body_work: wp.array[wp.spatial_vector],
        joint_work: wp.array[wp.float32],
        body_acceleration: wp.array[wp.spatial_vector],
        generalized_acceleration: wp.array[wp.float32],
        public_body_qd: wp.array[wp.spatial_vector],
        bodies: BodyContainer,
        midpoint: _MidpointWorkspace,
    ) -> wp.vec3:
        """Advance one tree; with ``include_coriolis`` also return its incoming linear momentum."""
        if wp.static(static_tile_width > 0):
            tile_width = wp.int32(static_tile_width)
        articulation = thread // tile_width
        lane = thread - articulation * tile_width
        if articulation >= articulation_count:
            return wp.vec3(0.0)
        warp_lane = thread & wp.int32(31)
        group = warp_lane // tile_width
        group_mask = wp.uint32(4294967295)
        if tile_width < wp.int32(32):
            group_mask = ((wp.uint32(1) << wp.uint32(tile_width)) - wp.uint32(1)) << wp.uint32(group * tile_width)
        # Trees the dense backend cannot hold keep the articulated-body recurrences.
        use_dense = wp.bool(False)
        if wp.static(dense):
            use_dense = bodies.reduced.dense.factor_start[articulation] >= wp.int32(0)

        local_momentum = wp.vec3(0.0)
        # The root lane keeps a floating root's twist for the step's energy estimate.
        root_velocity = wp.spatial_vector()
        previous_velocity = wp.spatial_vector()
        previous_coriolis = wp.spatial_vector()
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            active = index < depth_end
            joint = wp.int32(-1)
            parent_lane = wp.int32(0)
            if active:
                joint = articulation_depth_joint[index]
                parent_lane = joint_parent_lane[joint]
            parent_velocity = _shuffle_reduced_spatial(previous_velocity, parent_lane, tile_width, group_mask)
            parent_coriolis = _shuffle_reduced_spatial(previous_coriolis, parent_lane, tile_width, group_mask)
            velocity = wp.spatial_vector()
            coriolis = wp.spatial_vector()
            if active:
                child = joint_child[joint]
                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + wp.int32(1)]
                joint_velocity = wp.spatial_vector()
                for dof in range(dof_start, dof_end):
                    joint_velocity += joint_s[dof] * joint_qd[dof]
                velocity = parent_velocity + joint_velocity
                coriolis = parent_coriolis + spatial_cross(velocity, joint_velocity)
                inertia = _unpack_symmetric_mat66(body_inertia[child])
                x_com = wp.transform_get_translation(body_q_com[child])
                bias = wp.spatial_vector()
                bias_slot = child
                if wp.static(fuse_publish and static_tile_width > 0):
                    bias_slot = index
                if wp.static(include_coriolis):
                    spatial_momentum = inertia * velocity
                    local_momentum += wp.spatial_top(spatial_momentum)
                    bias += inertia * coriolis + spatial_cross_dual(velocity, spatial_momentum)
                    if wp.static(not include_external):
                        root_kind = joint_type[joint]
                        if depth == wp.int32(0) and (root_kind == JointType.FREE or root_kind == JointType.DISTANCE):
                            root_velocity = velocity
                if wp.static(include_external):
                    gravity_force = body_mass[child] * gravity[wp.max(body_world[child], wp.int32(0))]
                    gravity_wrench = wp.spatial_vector(gravity_force, wp.cross(x_com, gravity_force))
                    force_com = external_force_com[child]
                    force = wp.spatial_top(force_com)
                    torque = wp.spatial_bottom(force_com) + wp.cross(x_com, force)
                    bias -= gravity_wrench + wp.spatial_vector(force, torque)

                    joint_kind = joint_type[joint]
                    if joint_kind == JointType.FREE or joint_kind == JointType.DISTANCE:
                        control_force = wp.vec3(
                            joint_f[dof_start],
                            joint_f[dof_start + wp.int32(1)],
                            joint_f[dof_start + wp.int32(2)],
                        )
                        control_torque = wp.vec3(
                            joint_f[dof_start + wp.int32(3)],
                            joint_f[dof_start + wp.int32(4)],
                            joint_f[dof_start + wp.int32(5)],
                        )
                        bias -= wp.spatial_vector(
                            control_force,
                            control_torque + wp.cross(x_com, control_force),
                        )
                # The final forward pass publishes the authoritative twist before any consumer.
                # External-only passes also omit Coriolis scratch so CUDA can remove both recurrences.
                if wp.static(include_coriolis and include_external):
                    body_coriolis[child] = coriolis
                if wp.static(include_coriolis and not include_external):
                    # The velocity-product pass reuses each body's incoming momentum to measure
                    # the explicit step's energy drift; nothing reads the Coriolis scratch.
                    midpoint.body_momentum[child] = spatial_momentum
                body_bias[bias_slot] = bias
            previous_velocity = velocity
            previous_coriolis = coriolis
            _sync_reduced_group(group_mask)

        incoming_momentum = wp.vec3(0.0)
        if wp.static(include_coriolis and include_external):
            total = _sum_reduced_group_vec4(
                wp.vec4(local_momentum[0], local_momentum[1], local_momentum[2], wp.float32(0.0)),
                tile_width,
                group_mask,
            )
            incoming_momentum = wp.vec3(total[0], total[1], total[2])

        # Bias and articulated-force propagation only depend on the completed child
        # depth, so one reverse traversal can advance both recurrences.
        for reverse_depth in range(max_depth + wp.int32(1)):
            depth = max_depth - reverse_depth
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                child = joint_child[joint]
                body_slot = child
                if wp.static(fuse_publish and static_tile_width > 0):
                    body_slot = index
                subtree_bias = body_bias[body_slot]
                p = wp.spatial_vector()
                for child_index in range(child_start[joint], child_start[joint + wp.int32(1)]):
                    descendant_slot = child_joint[child_index]
                    if wp.static(not (fuse_publish and static_tile_width > 0)):
                        descendant_slot = joint_child[descendant_slot]
                    subtree_bias += body_bias[descendant_slot]
                    if not use_dense:
                        p += body_work[descendant_slot]
                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + wp.int32(1)]
                dof_count = dof_end - dof_start
                joint_kind = joint_type[joint]
                reduced_force = _vec6(0.0)
                d_inv_u = _vec6(0.0)
                for row in range(_MAX_JOINT_DOF):
                    if wp.int32(row) < dof_count:
                        dof = dof_start + wp.int32(row)
                        applied = wp.float32(0.0)
                        if (
                            wp.static(include_external)
                            and joint_kind != JointType.FREE
                            and joint_kind != JointType.DISTANCE
                        ):
                            applied = joint_f[dof]
                        if wp.static(include_external) and (
                            joint_kind == JointType.REVOLUTE
                            or joint_kind == JointType.PRISMATIC
                            or joint_kind == JointType.D6
                        ):
                            applied += joint_implicit_force[dof]
                        dof_rhs = applied - wp.dot(joint_s[dof], subtree_bias)
                        if use_dense:
                            joint_work[dof] = dof_rhs
                        else:
                            # Integrated advances consume both values locally; publishing them only
                            # dirties response workspaces before the next solve.
                            reduced_force[row] = dof_rhs - wp.dot(joint_s[dof], p)
                            joint_work[dof] = reduced_force[row]
                body_bias[body_slot] = subtree_bias
                if use_dense:
                    index += tile_width
                    continue
                for row in range(_MAX_JOINT_DOF):
                    if wp.int32(row) < dof_count:
                        for column in range(_MAX_JOINT_DOF):
                            if wp.int32(column) < dof_count:
                                d_inv_u[row] += joint_d_inv[dof_start + wp.int32(row), column] * reduced_force[column]
                propagated = p
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        propagated += joint_u_matrix[dof_start + wp.int32(column)] * d_inv_u[column]
                body_work[body_slot] = propagated
                index += tile_width
            _sync_reduced_group(group_mask)

        if use_dense:
            data = bodies.reduced
            dense_dof_start = joint_qd_start[data.articulation_start[articulation]]
            dense_tree_solve(
                data.dense,
                articulation,
                dense_dof_start,
                lane,
                tile_width,
                group_mask,
                joint_work,
                generalized_acceleration,
            )

        # Each depth assigns at most one joint per lane, so parent recurrences can
        # stay in the warp instead of round-tripping through response workspaces.
        previous_acceleration = wp.spatial_vector()
        previous_twist = wp.spatial_vector()
        # A velocity-product pass also measures the explicit step's energy drift in the
        # frame that moves with a floating root, see _explicit_step_drift_terms.
        previous_drift_rate = wp.spatial_vector()
        local_drift = wp.float32(0.0)
        local_energy = wp.float32(0.0)
        local_dissipation = wp.float32(0.0)
        frame = wp.spatial_vector()
        if wp.static(include_coriolis and not include_external):
            frame = _shuffle_reduced_spatial(_floating_root_frame(root_velocity), wp.int32(0), tile_width, group_mask)
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            active = index < depth_end
            joint = wp.int32(-1)
            parent_lane = wp.int32(0)
            if active:
                joint = articulation_depth_joint[index]
                parent_lane = joint_parent_lane[joint]
            parent_acceleration = _shuffle_reduced_spatial(previous_acceleration, parent_lane, tile_width, group_mask)
            twist = _shuffle_reduced_spatial(previous_twist, parent_lane, tile_width, group_mask)
            drift_rate = wp.spatial_vector()
            if wp.static(include_coriolis and not include_external):
                drift_rate = _shuffle_reduced_spatial(previous_drift_rate, parent_lane, tile_width, group_mask)
            child_acceleration = wp.spatial_vector()
            if active:
                child = joint_child[joint]
                dof_start = joint_qd_start[joint]
                dof_end = joint_qd_start[joint + wp.int32(1)]
                dof_count = dof_end - dof_start
                rhs = _vec6(0.0)
                qdd = _vec6(0.0)
                if use_dense:
                    for row in range(_MAX_JOINT_DOF):
                        if wp.int32(row) < dof_count:
                            qdd[row] = generalized_acceleration[dof_start + wp.int32(row)]
                else:
                    for row in range(_MAX_JOINT_DOF):
                        if wp.int32(row) < dof_count:
                            dof = dof_start + wp.int32(row)
                            rhs[row] = joint_work[dof] - wp.dot(joint_u_matrix[dof], parent_acceleration)
                    for row in range(_MAX_JOINT_DOF):
                        if wp.int32(row) < dof_count:
                            for column in range(_MAX_JOINT_DOF):
                                if wp.int32(column) < dof_count:
                                    qdd[row] += joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]
                child_acceleration = parent_acceleration
                joint_velocity = wp.spatial_vector()
                joint_acceleration = wp.spatial_vector()
                for row in range(_MAX_JOINT_DOF):
                    if wp.int32(row) < dof_count:
                        dof = dof_start + wp.int32(row)
                        rate = joint_qd[dof]
                        if wp.static(include_coriolis and not include_external):
                            # The midpoint step restarts from the incoming rates.
                            midpoint.joint_qd_mid[dof] = rate
                            local_dissipation += _joint_dissipation(midpoint, dof, rate)
                            joint_velocity += joint_s[dof] * rate
                            joint_acceleration += joint_s[dof] * qdd[row]
                        joint_qd[dof] = rate + qdd[row] * dt
                        child_acceleration += joint_s[dof] * qdd[row]
                        twist += joint_s[dof] * joint_qd[dof]
                x_com = wp.transform_get_translation(body_q_com[child])
                if wp.static(include_coriolis and not include_external):
                    terms = _explicit_step_drift_terms(
                        twist - dt * child_acceleration,
                        child_acceleration,
                        joint_velocity,
                        joint_acceleration,
                        depth == wp.int32(0),
                        frame,
                        midpoint.body_momentum[child],
                        body_mass[child],
                        x_com,
                        drift_rate,
                    )
                    drift_rate = wp.spatial_vector(terms[0], terms[1], terms[2], terms[3], terms[4], terms[5])
                    local_drift += terms[6]
                    local_energy += terms[7]
                body_velocity[child] = twist
                omega = wp.spatial_bottom(twist)
                velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
                slot = child + wp.int32(1)
                bodies.velocity[slot] = velocity_com
                bodies.angular_velocity[slot] = omega
            previous_acceleration = child_acceleration
            previous_twist = twist
            previous_drift_rate = drift_rate
            _sync_reduced_group(group_mask)

        if wp.static(include_coriolis and not include_external):
            # Every lane sees the same group sums, so the branch is uniform per tree,
            # and only trees whose explicit step would gain energy pay for the midpoint step.
            totals = _sum_reduced_group_vec4(
                wp.vec4(local_momentum[0], local_momentum[1], local_momentum[2], local_drift),
                tile_width,
                group_mask,
            )
            incoming_momentum = wp.vec3(totals[0], totals[1], totals[2])
            tree_energy = _sum_reduced_group_float(local_energy, tile_width, group_mask)
            tree_dissipation = _sum_reduced_group_float(local_dissipation, tile_width, group_mask)
            if _explicit_step_drifts(totals[3], tree_energy, tree_dissipation, dt):
                tree = _MidpointTree()
                tree.articulation = articulation
                tree.lane = lane
                tree.width = tile_width
                tree.mask = group_mask
                tree.max_depth = max_depth
                tree.depth_start = articulation_depth_start
                tree.order = articulation_depth_joint
                _restore_incoming_rates(tree, joint_qd_start, midpoint, joint_qd)
                _midpoint_momentum_step(
                    tree,
                    joint_type,
                    joint_parent,
                    joint_child,
                    joint_qd_start,
                    joint_s,
                    body_inertia,
                    midpoint,
                    dt,
                    wp.int32(_MIDPOINT_MAX_ITERATIONS),
                    joint_qd,
                    body_velocity,
                )
            elif lane == wp.int32(0):
                midpoint.flag[articulation] = wp.int32(0)
        return incoming_momentum

    @wp.kernel(enable_backward=False, module=module)
    def _advance_reduced_articulations_warp_kernel(
        articulation_count: wp.int32,
        tile_width: wp.int32,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        joint_parent_lane: wp.array[wp.int32],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_qd: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        joint_u_matrix: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
        joint_f: wp.array[wp.float32],
        joint_implicit_force: wp.array[wp.float32],
        body_mass: wp.array[wp.float32],
        body_world: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        body_q_com: wp.array[wp.transform],
        body_inertia: wp.array[_sym_mat66],
        external_force_com: wp.array[wp.spatial_vector],
        dt: wp.float32,
        body_velocity: wp.array[wp.spatial_vector],
        body_coriolis: wp.array[wp.spatial_vector],
        body_bias: wp.array[wp.spatial_vector],
        generalized_rhs: wp.array[wp.float32],
        body_work: wp.array[wp.spatial_vector],
        joint_work: wp.array[wp.float32],
        body_acceleration: wp.array[wp.spatial_vector],
        generalized_acceleration: wp.array[wp.float32],
        public_body_qd: wp.array[wp.spatial_vector],
        bodies: BodyContainer,
        midpoint: _MidpointWorkspace,
    ):
        _advance_reduced_articulations_warp_device(
            wp.tid(),
            articulation_count,
            tile_width,
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_parent_lane,
            child_start,
            child_joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_s,
            joint_u_matrix,
            joint_d_inv,
            joint_f,
            joint_implicit_force,
            body_mass,
            body_world,
            gravity,
            body_q_com,
            body_inertia,
            external_force_com,
            dt,
            body_velocity,
            body_coriolis,
            body_bias,
            generalized_rhs,
            body_work,
            joint_work,
            body_acceleration,
            generalized_acceleration,
            public_body_qd,
            bodies,
            midpoint,
        )

    if not fuse_publish:
        return _advance_reduced_articulations_warp_device, _advance_reduced_articulations_warp_kernel

    @wp.kernel(enable_backward=False, module=module)
    def _advance_and_publish_reduced_articulations_warp_kernel(
        articulation_count: wp.int32,
        tile_width: wp.int32,
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        joint_parent_lane: wp.array[wp.int32],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_x_p: wp.array[wp.transform],
        joint_x_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_dof_dim: wp.array2d[wp.int32],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[wp.float32],
        body_world: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        body_x_com: wp.array[wp.transform],
        zero_generalized_acceleration: wp.array[wp.float32],
        joint_f: wp.array[wp.float32],
        joint_implicit_force: wp.array[wp.float32],
        external_force_com: wp.array[wp.spatial_vector],
        dt: wp.float32,
        joint_q: wp.array[wp.float32],
        joint_qd: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        joint_u_matrix: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
        body_inertia_spatial: wp.array[_sym_mat66],
        body_velocity: wp.array[wp.spatial_vector],
        body_coriolis: wp.array[wp.spatial_vector],
        body_bias: wp.array[wp.spatial_vector],
        generalized_rhs: wp.array[wp.float32],
        body_work: wp.array[wp.spatial_vector],
        joint_work: wp.array[wp.float32],
        body_acceleration: wp.array[wp.spatial_vector],
        generalized_acceleration: wp.array[wp.float32],
        joint_qd_public: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        public_body_qd: wp.array[wp.spatial_vector],
        bodies: BodyContainer,
        midpoint: _MidpointWorkspace,
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
        joint_s_publish: wp.array[wp.spatial_vector],
    ):
        thread = wp.tid()
        if wp.static(static_tile_width > 0):
            tile_width = wp.int32(static_tile_width)
        articulation = thread // tile_width
        if articulation >= articulation_count:
            return

        target_linear_momentum = _advance_reduced_articulations_warp_device(
            thread,
            articulation_count,
            tile_width,
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_parent_lane,
            child_start,
            child_joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_s,
            joint_u_matrix,
            joint_d_inv,
            joint_f,
            joint_implicit_force,
            body_mass,
            body_world,
            gravity,
            body_q_com,
            body_inertia_spatial,
            external_force_com,
            dt,
            body_velocity,
            body_coriolis,
            body_bias,
            generalized_rhs,
            body_work,
            joint_work,
            body_acceleration,
            generalized_acceleration,
            public_body_qd,
            bodies,
            midpoint,
        )

        _finish_and_publish_reduced_warp_device(
            thread,
            articulation_count,
            tile_width,
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_q,
            joint_qd,
            joint_qd,
            joint_q_start,
            joint_qd_start,
            joint_type,
            joint_parent,
            joint_child,
            joint_x_p,
            joint_x_c,
            joint_axis,
            joint_dof_dim,
            body_com,
            body_x_com,
            zero_generalized_acceleration,
            dt,
            wp.bool(True),
            body_mass,
            target_linear_momentum,
            wp.bool(True),
            joint_qd_public,
            body_q,
            bodies,
            articulation_origin,
            body_q_local,
            body_q_com,
            joint_anchor_local,
            joint_s_publish,
            public_body_qd,
            midpoint.flag,
        )

    return _advance_reduced_articulations_warp_device, _advance_and_publish_reduced_articulations_warp_kernel


def _make_advance_reduced_articulations_warp_kernel(
    *,
    include_external: bool,
    include_coriolis: bool,
    fuse_publish: bool = False,
    static_tile_width: int = 0,
    dense: bool = False,
):
    return _make_advance_reduced_articulations_warp_ops(
        include_external=include_external,
        include_coriolis=include_coriolis,
        fuse_publish=fuse_publish,
        static_tile_width=static_tile_width,
        dense=dense,
    )[1]


@functools.cache
def _make_biased_contact_advance_publish_kernel(max_dofs: int):
    solve_device = _SOLVE_GENERALIZED_CONTACT_TILE_DEVICES[max_dofs]
    advance_device = _make_advance_reduced_articulations_warp_ops(
        include_external=False, include_coriolis=True, fuse_publish=True
    )[0]
    module = wp.get_module(f"reduced_contact_bias_advance_publish_{max_dofs}")

    @wp.kernel(enable_backward=False, module=module)
    def _biased_contact_advance_publish_kernel(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        cc: ContactContainer,
        iterations: wp.int32,
        enabled: wp.array[wp.int32],
        point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_velocity: wp.array2d[wp.float32],
        page_index: wp.array[wp.int32],
        max_page_count: wp.array[wp.int32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        generalized_delta: wp.array2d[wp.float32],
        body_delta: wp.array2d[wp.spatial_vector],
        articulation_count: wp.int32,
        joint_parent_lane: wp.array[wp.int32],
        child_start: wp.array[wp.int32],
        child_joint: wp.array[wp.int32],
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_x_p: wp.array[wp.transform],
        joint_x_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_dof_dim: wp.array2d[wp.int32],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[wp.float32],
        body_world: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        body_x_com: wp.array[wp.transform],
        zero_generalized_acceleration: wp.array[wp.float32],
        joint_f: wp.array[wp.float32],
        joint_implicit_force: wp.array[wp.float32],
        external_force_com: wp.array[wp.spatial_vector],
        dt: wp.float32,
        joint_q: wp.array[wp.float32],
        joint_qd: wp.array[wp.float32],
        joint_qd_integrator: wp.array[wp.float32],
        joint_s: wp.array[wp.spatial_vector],
        joint_u_matrix: wp.array[wp.spatial_vector],
        joint_d_inv: wp.array2d[wp.float32],
        body_inertia_spatial: wp.array[_sym_mat66],
        body_velocity: wp.array[wp.spatial_vector],
        body_coriolis: wp.array[wp.spatial_vector],
        body_bias: wp.array[wp.spatial_vector],
        generalized_rhs: wp.array[wp.float32],
        body_work: wp.array[wp.spatial_vector],
        joint_work: wp.array[wp.float32],
        body_acceleration: wp.array[wp.spatial_vector],
        generalized_acceleration: wp.array[wp.float32],
        joint_qd_public: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        public_body_qd: wp.array[wp.spatial_vector],
        midpoint: _MidpointWorkspace,
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
        joint_s_publish: wp.array[wp.spatial_vector],
        advanced: wp.array[wp.int32],
    ):
        articulation, lane = wp.tid()
        solve_device(
            articulation,
            lane,
            columns,
            idt,
            sor_boost,
            cc,
            iterations,
            wp.bool(True),
            wp.bool(True),
            enabled,
            point_count,
            point_contact,
            point_column,
            normal,
            tangent0,
            row_velocity,
            page_index,
            packed_jacobian,
            packed_response,
            bodies,
            wp.bool(True),
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            generalized_delta,
            body_delta,
        )
        _sync_contact_warp()
        if page_index[0] + wp.int32(1) < max_page_count[0]:
            return

        thread = articulation * wp.int32(32) + lane
        target_linear_momentum = advance_device(
            thread,
            articulation_count,
            wp.int32(32),
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_parent_lane,
            child_start,
            child_joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_s,
            joint_u_matrix,
            joint_d_inv,
            joint_f,
            joint_implicit_force,
            body_mass,
            body_world,
            gravity,
            body_q_com,
            body_inertia_spatial,
            external_force_com,
            dt,
            body_velocity,
            body_coriolis,
            body_bias,
            generalized_rhs,
            body_work,
            joint_work,
            body_acceleration,
            generalized_acceleration,
            public_body_qd,
            bodies,
            midpoint,
        )
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + wp.int32(1)]):
                    joint_qd_integrator[dof] = joint_qd[dof]
                index += wp.int32(32)
            _sync_contact_warp()
        _finish_and_publish_reduced_warp_device(
            thread,
            articulation_count,
            wp.int32(32),
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            joint_q,
            joint_qd,
            joint_qd_integrator,
            joint_q_start,
            joint_qd_start,
            joint_type,
            joint_parent,
            joint_child,
            joint_x_p,
            joint_x_c,
            joint_axis,
            joint_dof_dim,
            body_com,
            body_x_com,
            zero_generalized_acceleration,
            dt,
            wp.bool(True),
            body_mass,
            target_linear_momentum,
            wp.bool(True),
            joint_qd_public,
            body_q,
            bodies,
            articulation_origin,
            body_q_local,
            body_q_com,
            joint_anchor_local,
            joint_s_publish,
            public_body_qd,
            midpoint.flag,
        )
        if articulation == wp.int32(0) and lane == wp.int32(0):
            advanced[0] = wp.int32(1)

    return _biased_contact_advance_publish_kernel


@functools.cache
def _make_relax_and_publish_reduced_contacts_kernel(max_dofs: int):
    solve_device = _SOLVE_GENERALIZED_CONTACT_TILE_DEVICES[max_dofs]
    module = wp.get_module(f"reduced_contact_relax_publish_{max_dofs}")

    @wp.kernel(enable_backward=False, module=module)
    def _relax_and_publish_reduced_contacts_kernel(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: wp.float32,
        cc: ContactContainer,
        iterations: wp.int32,
        enabled: wp.array[wp.int32],
        point_count: wp.array[wp.int32],
        point_contact: wp.array2d[wp.int32],
        point_column: wp.array2d[wp.int32],
        point0: wp.array2d[wp.vec3],
        point1: wp.array2d[wp.vec3],
        normal: wp.array2d[wp.vec3],
        tangent0: wp.array2d[wp.vec3],
        row_velocity: wp.array2d[wp.float32],
        page_index: wp.array[wp.int32],
        max_page_count: wp.array[wp.int32],
        packed_jacobian: wp.array2d[wp.float32],
        packed_response: wp.array2d[wp.float32],
        max_depth: wp.int32,
        articulation_depth_start: wp.array2d[wp.int32],
        articulation_depth_joint: wp.array[wp.int32],
        generalized_delta: wp.array2d[wp.float32],
        body_delta: wp.array2d[wp.spatial_vector],
        articulation_count: wp.int32,
        joint_q: wp.array[wp.float32],
        joint_qd_integrator: wp.array[wp.float32],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_type: wp.array[wp.int32],
        joint_parent: wp.array[wp.int32],
        joint_child: wp.array[wp.int32],
        joint_x_p: wp.array[wp.transform],
        joint_x_c: wp.array[wp.transform],
        joint_axis: wp.array[wp.vec3],
        joint_dof_dim: wp.array2d[wp.int32],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[wp.float32],
        body_x_com: wp.array[wp.transform],
        zero_generalized_acceleration: wp.array[wp.float32],
        joint_qd_public: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        body_q_local: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        joint_anchor_local: wp.array[wp.transform],
        joint_s_publish: wp.array[wp.spatial_vector],
        body_qd: wp.array[wp.spatial_vector],
        midpoint_flag: wp.array[wp.int32],
    ):
        articulation, lane = wp.tid()
        if lane < wp.int32(_POINTS_PER_PAGE):
            _finalize_reduced_contact_rows_device(
                articulation,
                lane,
                columns,
                bodies,
                enabled,
                point_count,
                page_index,
                point_column,
                point0,
                point1,
                normal,
                tangent0,
                row_velocity,
            )
        _sync_contact_warp()
        solve_device(
            articulation,
            lane,
            columns,
            idt,
            sor_boost,
            cc,
            iterations,
            wp.bool(False),
            wp.bool(False),
            enabled,
            point_count,
            point_contact,
            point_column,
            normal,
            tangent0,
            row_velocity,
            page_index,
            packed_jacobian,
            packed_response,
            bodies,
            wp.bool(True),
            max_depth,
            articulation_depth_start,
            articulation_depth_joint,
            generalized_delta,
            body_delta,
        )
        _sync_contact_warp()
        if page_index[0] + wp.int32(1) >= max_page_count[0]:
            _finish_and_publish_reduced_warp_device(
                articulation * wp.int32(32) + lane,
                articulation_count,
                wp.int32(32),
                max_depth,
                articulation_depth_start,
                articulation_depth_joint,
                joint_q,
                bodies.reduced.joint_qd,
                joint_qd_integrator,
                joint_q_start,
                joint_qd_start,
                joint_type,
                joint_parent,
                joint_child,
                joint_x_p,
                joint_x_c,
                joint_axis,
                joint_dof_dim,
                body_com,
                body_x_com,
                zero_generalized_acceleration,
                wp.float32(0.0),
                wp.bool(False),
                body_mass,
                wp.vec3(0.0),
                wp.bool(False),
                joint_qd_public,
                body_q,
                bodies,
                articulation_origin,
                body_q_local,
                body_q_com,
                joint_anchor_local,
                joint_s_publish,
                body_qd,
                midpoint_flag,
            )

    return _relax_and_publish_reduced_contacts_kernel


@wp.kernel(enable_backward=False)
def _solve_reduced_deferred_contacts_kernel(
    schedule_section_end: wp.array[wp.int32],
    scheduled_column: wp.array[wp.int32],
    block_enabled: wp.array[wp.int32],
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    cross_mobility: wp.array[wp.vec3],
    iterations: wp.int32,
    use_bias: wp.bool,
    prepare: wp.bool,
):
    articulation = wp.tid()
    start = wp.int32(0)
    if articulation > wp.int32(0):
        start = schedule_section_end[articulation - wp.int32(1)]
    end = schedule_section_end[articulation]

    if block_enabled[articulation] != wp.int32(0):
        return

    if prepare:
        for index in range(start, end):
            column = scheduled_column[index]
            reduced_contact_prepare(
                contact_cols, column, bodies, idt, cc, contacts, wp.bool(True), use_bias, wp.bool(True), cross_mobility
            )

    for iteration in range(iterations):
        for offset in range(end - start):
            index = start + offset
            if (iteration & wp.int32(1)) != wp.int32(0):
                index = end - wp.int32(1) - offset
            column = scheduled_column[index]
            reduced_contact_iterate(
                contact_cols,
                column,
                bodies,
                idt,
                sor_boost,
                cc,
                contacts,
                use_bias,
                wp.bool(True),
                cross_mobility,
            )

    flush_deferred_articulation(bodies, articulation)


class ReducedArticulationSystem:
    """Cached linear-time reduced-coordinate factorization for a Newton model."""

    def __init__(self, model: Model, *, dense: bool = False):
        if model.articulation_count <= 0:
            raise ValueError("reduced articulation system requires at least one articulation")
        self.model = model
        self.device = model.device
        self.use_warp_kinematics = bool(self.device.is_cuda)
        self.use_warp_publish = bool(self.device.is_cuda)
        body_count = int(model.body_count)
        joint_count = int(model.joint_count)
        dof_count = int(model.joint_dof_count)
        articulation_count = int(model.articulation_count)

        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        articulation_start_np = model.articulation_start.numpy()
        articulation_end_np = model.articulation_end.numpy()
        tree_joint_np = np.concatenate(
            [
                np.arange(articulation_start_np[articulation], articulation_end_np[articulation], dtype=np.int32)
                for articulation in range(int(model.articulation_count))
            ]
        )
        body_joint_np = np.full(body_count, -1, dtype=np.int32)
        for joint in tree_joint_np:
            body_joint_np[int(joint_child_np[joint])] = joint
        joint_depth_np = np.zeros(joint_count, dtype=np.int32)
        child_lists: list[list[int]] = [[] for _ in range(joint_count)]
        for joint in tree_joint_np:
            parent = int(joint_parent_np[joint])
            if parent >= 0:
                parent_joint = int(body_joint_np[parent])
                joint_depth_np[joint] = joint_depth_np[parent_joint] + 1
                child_lists[parent_joint].append(int(joint))
        tree_depth_np = joint_depth_np[tree_joint_np]
        depth_order = np.argsort(tree_depth_np, kind="stable")
        depth_joint_np = tree_joint_np[depth_order]
        depth_counts = np.bincount(tree_depth_np, minlength=int(tree_depth_np.max()) + 1)
        self.factor_depth_ranges: list[tuple[int, int]] = []
        depth_offset = 0
        for depth_count in depth_counts:
            count = int(depth_count)
            self.factor_depth_ranges.append((depth_offset, count))
            depth_offset += count

        # Deep small trees also benefit from replacing many tiny depth launches
        # with a cooperative walk. Keep the established path for shallow small
        # batches; this changes scheduling only, never the current-pose operator.
        # Runtime instance slots rewrite their topology on the device, which only the
        # cooperative walk reads; the per-depth launches use host-built joint lists.
        capacity = getattr(model, "_dynamic_capacity", None)
        self.runtime_slots = bool(capacity is not None and capacity.slot_size)
        self.use_warp_factor = bool(
            self.device.is_cuda
            and (
                self.runtime_slots
                or model.articulation_count >= _WARP_FACTOR_MIN_ARTICULATIONS
                or len(self.factor_depth_ranges) >= _WARP_FACTOR_MIN_DEPTH_LEVELS
            )
        )
        self.advance_max_depth = len(self.factor_depth_ranges) - 1
        articulation_depth_start_np = np.empty(
            (articulation_count, self.advance_max_depth + 2),
            dtype=np.int32,
        )
        joint_depth_lane_np = np.zeros(joint_count, dtype=np.int32)
        articulation_depth_joint_list: list[int] = []
        for articulation in range(articulation_count):
            joint_start = int(articulation_start_np[articulation])
            joint_end = int(articulation_end_np[articulation])
            for depth in range(self.advance_max_depth + 1):
                articulation_depth_start_np[articulation, depth] = len(articulation_depth_joint_list)
                depth_lane = 0
                for joint in range(joint_start, joint_end):
                    if int(joint_depth_np[joint]) == depth:
                        joint_depth_lane_np[joint] = depth_lane
                        articulation_depth_joint_list.append(joint)
                        depth_lane += 1
            articulation_depth_start_np[articulation, self.advance_max_depth + 1] = len(articulation_depth_joint_list)

        child_start_np = np.zeros(joint_count + 1, dtype=np.int32)
        child_joint_list: list[int] = []
        for joint, children in enumerate(child_lists):
            child_joint_list.extend(children)
            child_start_np[joint + 1] = len(child_joint_list)
        self.factor_joint_count = int(tree_joint_np.shape[0])
        self.factor_joint = wp.array(tree_joint_np, device=self.device)
        self.factor_depth_joint = wp.array(depth_joint_np, device=self.device)
        self.advance_articulation_depth_start = wp.array(articulation_depth_start_np, device=self.device)
        self.advance_articulation_depth_joint = wp.array(
            np.asarray(articulation_depth_joint_list, dtype=np.int32),
            device=self.device,
        )
        # The static fused recurrence uses the depth walk itself as its compact scratch address space.
        joint_depth_index_np = np.zeros(joint_count, dtype=np.int32)
        for depth_index, joint in enumerate(articulation_depth_joint_list):
            joint_depth_index_np[joint] = depth_index
        self.advance_child_depth_index = wp.array(
            np.asarray([joint_depth_index_np[joint] for joint in child_joint_list], dtype=np.int32),
            device=self.device,
        )
        joint_parent_lane_np = np.zeros(joint_count, dtype=np.int32)
        for joint in tree_joint_np:
            parent = int(joint_parent_np[joint])
            if parent >= 0:
                joint_parent_lane_np[joint] = joint_depth_lane_np[body_joint_np[parent]]
        self.advance_joint_parent_lane = wp.array(joint_parent_lane_np, device=self.device)
        max_articulation_breadth = int(np.max(np.diff(articulation_depth_start_np, axis=1)))
        if self.runtime_slots:
            # Slots are placeholder chains; size the depth lanes for the widest template level.
            for template in capacity.articulation_templates:
                max_articulation_breadth = max(
                    max_articulation_breadth, int(np.bincount(template["tree"]["depth"]).max())
                )
        self.use_warp_advance = bool(self.device.is_cuda and max_articulation_breadth <= 32)
        if max_articulation_breadth <= 8:
            self.advance_tile_width = 8
        elif max_articulation_breadth <= 16:
            self.advance_tile_width = 16
        else:
            self.advance_tile_width = 32
        self.factor_child_start = wp.array(child_start_np, device=self.device)
        self.factor_child_joint = wp.array(np.asarray(child_joint_list, dtype=np.int32), device=self.device)

        # DOF-type factor split: single-DOF joints reduce via a rank-1 downdate,
        # but multi-DOF joints (e.g. a floating base) carry a 6x6 SPD inverse that
        # roughly doubles the depth-walk kernel's live registers and halves its
        # occupancy. Defer those joints to a companion launch so the depth walk
        # keeps only the lean single-DOF register set. Correct only when every
        # multi-DOF joint is an articulation root (depth 0), whose reduced inertia
        # the depth walk never consumes; otherwise fall back to the unified kernel.
        joint_qd_start_np = model.joint_qd_start.numpy()
        multidof_joints = [
            int(joint)
            for joint in tree_joint_np
            if int(joint_qd_start_np[joint + 1]) - int(joint_qd_start_np[joint]) > 1
        ]
        self._factor_multidof_count = len(multidof_joints)
        multidof_all_roots = all(int(joint_depth_np[joint]) == 0 for joint in multidof_joints)
        self._use_factor_dof_split = bool(
            self.use_warp_factor and self._factor_multidof_count > 0 and multidof_all_roots
        )
        self.factor_multidof_joint = (
            wp.array(np.asarray(multidof_joints, dtype=np.int32), device=self.device)
            if self._factor_multidof_count > 0
            else None
        )

        # Zeroed rather than left uninitialized: the unfused dense path reads some of this
        # workspace before its first write, which reused memory then makes nondeterministic.
        self.body_x_com = wp.zeros(body_count, dtype=wp.transform, device=self.device)
        self.body_q_com = wp.zeros(body_count, dtype=wp.transform, device=self.device)
        self.body_q_local = wp.zeros(body_count, dtype=wp.transform, device=self.device)
        self.joint_anchor_local = wp.zeros(joint_count, dtype=wp.transform, device=self.device)
        self.articulation_origin = wp.zeros(int(model.articulation_count), dtype=wp.vec3, device=self.device)
        self.body_i_s = wp.zeros(body_count, dtype=_sym_mat66, device=self.device)
        self.joint_s = wp.zeros(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_s_publish = wp.zeros(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_qd_internal = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.joint_qd_integrator = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.joint_factor_diagonal = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.joint_implicit_force = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.reduced_inertia = wp.zeros(body_count, dtype=_sym_mat66, device=self.device)
        self.joint_u_matrix = wp.zeros(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_d_inv = wp.zeros((max(1, dof_count), 6), dtype=wp.float32, device=self.device)
        self.body_force = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_velocity = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_coriolis = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_bias = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_work = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_work = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.body_acceleration = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.generalized_acceleration = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.zero_generalized_force = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.generalized_rhs = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.impulse_response = wp.zeros((body_count, 6), dtype=wp.spatial_vector, device=self.device)
        self.deferred_wrench = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.deferred_pending = wp.zeros(max(1, articulation_count), dtype=wp.int32, device=self.device)
        self.target_linear_momentum = wp.zeros(articulation_count, dtype=wp.vec3, device=self.device)
        # Model data and scratch for the midpoint step of fast trees, see _midpoint_momentum_step.
        midpoint = _MidpointWorkspace()
        midpoint.flag = wp.zeros(articulation_count, dtype=wp.int32, device=self.device)
        midpoint.joint_q_start = model.joint_q_start
        midpoint.joint_x_p = model.joint_X_p
        midpoint.joint_x_c = model.joint_X_c
        midpoint.joint_axis = model.joint_axis
        midpoint.joint_dof_dim = model.joint_dof_dim
        midpoint.joint_armature = model.joint_armature
        midpoint.joint_factor_diagonal = self.joint_factor_diagonal
        midpoint.body_com = model.body_com
        midpoint.body_x_com = self.body_x_com
        midpoint.body_mass = model.body_mass
        midpoint.body_inertia = model.body_inertia
        midpoint.body_inv_inertia = model.body_inv_inertia
        midpoint.articulation_origin = self.articulation_origin
        midpoint.body_q_local = self.body_q_local
        midpoint.joint_anchor_local = self.joint_anchor_local
        midpoint.child_start = self.factor_child_start
        midpoint.child_joint = self.factor_child_joint
        midpoint.zero_dof = self.zero_generalized_force
        midpoint.body_pose = wp.zeros(body_count, dtype=wp.transform, device=self.device)
        midpoint.body_articulated = wp.zeros(body_count, dtype=wp.spatial_matrix, device=self.device)
        midpoint.body_reduced = wp.zeros(body_count, dtype=wp.spatial_matrix, device=self.device)
        midpoint.body_momentum = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        midpoint.body_force = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        midpoint.joint_q_mid = wp.zeros(max(1, int(model.joint_coord_count)), dtype=wp.float32, device=self.device)
        midpoint.joint_qd_mid = wp.zeros(max(1, dof_count), dtype=wp.float32, device=self.device)
        midpoint.joint_s_mid = wp.zeros(max(1, dof_count), dtype=wp.spatial_vector, device=self.device)
        midpoint.joint_u = wp.zeros(max(1, dof_count), dtype=wp.spatial_vector, device=self.device)
        midpoint.joint_d_inv = wp.zeros((max(1, dof_count), 6), dtype=wp.float32, device=self.device)
        midpoint.joint_force = wp.zeros(max(1, dof_count), dtype=wp.float32, device=self.device)
        midpoint.joint_reduced = wp.zeros(max(1, dof_count), dtype=wp.float32, device=self.device)
        midpoint.joint_step = wp.zeros(max(1, dof_count), dtype=wp.float32, device=self.device)
        midpoint.joint_momentum = wp.zeros(max(1, dof_count), dtype=wp.float32, device=self.device)
        self.midpoint = midpoint

        self.body_joint = wp.array(body_joint_np, device=self.device)
        self.dense: DenseTreeOperator | None = None
        if dense:
            if not self.use_warp_advance:
                raise ValueError("the dense reduced backend requires CUDA and at most 32 joints per tree depth")
            self.dense = DenseTreeOperator(model, self)
            if not self.dense.buckets:
                raise ValueError(
                    f"the dense reduced backend holds no articulation: {'; '.join(self.dense.fallback_reasons)}"
                )

        self.refresh_inertial_properties()

    def refresh_inertial_properties(self) -> None:
        """Refresh cached COM transforms; factorization reads body inertia directly."""
        wp.launch(
            compute_com_transforms,
            dim=int(self.model.body_count),
            inputs=[self.model.body_com],
            outputs=[self.body_x_com],
            device=self.device,
        )

    def update_local_kinematics(self, state: State) -> None:
        """Reconstruct translation-invariant link and COM frames."""
        kinematics_inputs = [
            self.model.joint_type,
            self.model.joint_parent,
            self.model.joint_child,
            self.model.joint_q_start,
            self.model.joint_qd_start,
            state.joint_q,
            self.model.joint_X_p,
            self.model.joint_X_c,
            self.model.joint_axis,
            self.model.joint_dof_dim,
            state.body_q,
            self.model.body_com,
            self.body_x_com,
        ]
        kinematics_outputs = [
            self.articulation_origin,
            self.body_q_local,
            self.body_q_com,
            self.joint_anchor_local,
        ]
        if self.use_warp_kinematics:
            articulation_count = int(self.model.articulation_count)
            tile_width = self.advance_tile_width
            thread_count = ((articulation_count * tile_width + 31) // 32) * 32
            wp.launch(
                _compute_reduced_local_kinematics_warp_kernel,
                dim=thread_count,
                block_dim=128 if tile_width < 32 else 32,
                inputs=[
                    wp.int32(articulation_count),
                    wp.int32(tile_width),
                    wp.int32(self.advance_max_depth),
                    self.advance_articulation_depth_start,
                    self.advance_articulation_depth_joint,
                    *kinematics_inputs,
                ],
                outputs=kinematics_outputs,
                device=self.device,
            )
        else:
            wp.launch(
                _compute_reduced_local_kinematics_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.model.articulation_start,
                    self.model.articulation_end,
                    *kinematics_inputs,
                ],
                outputs=kinematics_outputs,
                device=self.device,
            )

    def factor(
        self,
        state: State,
        control: Control | None = None,
        dt: float = 0.0,
        *,
        factorize: bool = True,
        update_kinematics: bool = True,
        articulated_body: bool = False,
    ) -> None:
        """Factor the generalized mass and optional implicit joint operator.

        With the dense backend, ``articulated_body`` also forms the articulated-body
        factor that :meth:`solve_forces` applies.
        """
        if control is None:
            wp.copy(self.joint_factor_diagonal, self.model.joint_armature)
            self.joint_implicit_force.zero_()
        else:
            wp.launch(
                _configure_implicit_joint_dynamics_kernel,
                dim=int(self.model.joint_count),
                inputs=[
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_target_q_start,
                    state.joint_q,
                    state.joint_qd,
                    control.joint_target_q,
                    control.joint_target_qd,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_target_mode,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.model.joint_limit_ke,
                    self.model.joint_limit_kd,
                    self.model.joint_damping,
                    self.model.joint_armature,
                    wp.float32(dt),
                ],
                outputs=[self.joint_factor_diagonal, self.joint_implicit_force],
                device=self.device,
            )
        if update_kinematics:
            self.update_local_kinematics(state)
        wp.launch(
            _initialize_reduced_factor_kernel,
            dim=self.factor_joint_count,
            inputs=[
                self.factor_joint,
                self.model.joint_type,
                self.model.joint_child,
                state.joint_q,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                state.joint_qd,
                self.joint_anchor_local,
                self.body_q_com,
                self.model.body_mass,
                self.model.body_inertia,
            ],
            outputs=[
                self.joint_qd_internal,
                self.body_i_s,
                self.joint_s,
            ],
            device=self.device,
        )
        if not factorize:
            return
        if self.dense is not None:
            self.dense.factor_mass()
            if not (self.dense.requires_aba or articulated_body):
                return
        if self.use_warp_factor:
            depth_walk_kernel = _FACTOR_REDUCED_WARP_KERNELS[
                self._use_factor_dof_split,
                self.advance_tile_width,
            ]
            wp.launch(
                depth_walk_kernel,
                dim=int(self.model.articulation_count) * self.advance_tile_width,
                block_dim=32,
                inputs=[
                    wp.int32(self.advance_max_depth),
                    self.advance_articulation_depth_start,
                    self.advance_articulation_depth_joint,
                    self.model.joint_child,
                    self.model.joint_qd_start,
                    self.joint_factor_diagonal,
                    self.joint_s,
                    self.factor_child_start,
                    self.factor_child_joint,
                    self.body_i_s,
                    self.reduced_inertia,
                ],
                outputs=[
                    self.joint_u_matrix,
                    self.joint_d_inv,
                ],
                device=self.device,
            )
            if self._use_factor_dof_split:
                # Companion launch factors the deferred multi-DOF root joints,
                # whose descendants the depth walk already reduced. block_dim=32
                # spreads the few heavy roots across many SMs; the default 256
                # packs them onto a handful of blocks and stalls (measured ~2.5x).
                wp.launch(
                    _factor_reduced_multidof_kernel,
                    dim=self._factor_multidof_count,
                    block_dim=32,
                    inputs=[
                        self.factor_multidof_joint,
                        self.model.joint_child,
                        self.model.joint_qd_start,
                        self.joint_factor_diagonal,
                        self.joint_s,
                        self.factor_child_start,
                        self.factor_child_joint,
                        self.body_i_s,
                        self.reduced_inertia,
                    ],
                    outputs=[
                        self.joint_u_matrix,
                        self.joint_d_inv,
                    ],
                    device=self.device,
                )
        else:
            for depth_offset, depth_count in reversed(self.factor_depth_ranges):
                wp.launch(
                    _factor_reduced_depth_kernel,
                    dim=depth_count,
                    inputs=[
                        self.factor_depth_joint,
                        wp.int32(depth_offset),
                        self.model.joint_parent,
                        self.model.joint_child,
                        self.model.joint_qd_start,
                        self.joint_factor_diagonal,
                        self.joint_s,
                        self.factor_child_start,
                        self.factor_child_joint,
                        self.body_i_s,
                        self.reduced_inertia,
                    ],
                    outputs=[
                        self.joint_u_matrix,
                        self.joint_d_inv,
                    ],
                    device=self.device,
                )

    def forward_dynamics(self, state: State, control: Control) -> wp.array[wp.float32]:
        """Compute generalized acceleration from common Newton state and control."""
        self.factor(state, articulated_body=True)
        wp.launch(
            _compute_forward_dynamics_rhs_kernel,
            dim=int(self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_target_q_start,
                state.joint_q,
                self.joint_qd_internal,
                self.joint_s,
                control.joint_f,
                control.joint_target_q,
                control.joint_target_qd,
                self.model.joint_target_ke,
                self.model.joint_target_kd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.model.joint_damping,
                self.model.joint_dof_dim,
                self.model.body_mass,
                self.model.body_world,
                self.model.gravity,
                self.body_q_com,
                self.body_i_s,
                state.body_f,
            ],
            outputs=[
                self.body_velocity,
                self.body_coriolis,
                self.body_bias,
                self.generalized_rhs,
            ],
            device=self.device,
        )
        return self.solve_generalized(self.generalized_rhs)

    def bind_midpoint_state(self, state: State) -> _MidpointWorkspace:
        """Point the midpoint workspace at the state whose coordinates the advance integrates."""
        self.midpoint.joint_q = state.joint_q
        self.midpoint.joint_qd_public = state.joint_qd
        return self.midpoint

    def advance(
        self,
        state: State,
        control: Control,
        bodies: BodyContainer,
        dt: float,
        *,
        include_external: bool,
        include_coriolis: bool,
    ) -> None:
        """Apply one split ABA velocity update and publish link twists.

        A velocity-product-only pass (``include_coriolis`` alone) integrates
        trees whose explicit step would gain energy (``_MIDPOINT_ENERGY_RATE``) with
        :func:`_midpoint_momentum_step`, which also advances their coordinates
        in ``state.joint_q`` and flags them for publication.
        """
        advance_inputs = [
            self.model.joint_type,
            self.model.joint_parent,
            self.model.joint_child,
            self.model.joint_qd_start,
            self.joint_qd_internal,
            self.joint_s,
            self.joint_u_matrix,
            self.joint_d_inv,
            control.joint_f,
            self.joint_implicit_force,
            self.model.body_mass,
            self.model.body_world,
            self.model.gravity,
            self.body_q_com,
            self.body_i_s,
            state.body_f,
            wp.float32(dt),
        ]
        advance_outputs = [
            self.body_velocity,
            self.body_coriolis,
            self.body_bias,
            self.generalized_rhs,
            self.body_work,
            self.joint_work,
            self.body_acceleration,
            self.generalized_acceleration,
            state.body_qd,
            bodies,
            self.bind_midpoint_state(state),
        ]
        if self.use_warp_advance:
            articulation_count = int(self.model.articulation_count)
            tile_width = self.advance_tile_width
            thread_count = ((articulation_count * tile_width + 31) // 32) * 32
            advance_kernel = _make_advance_reduced_articulations_warp_kernel(
                include_external=include_external,
                include_coriolis=include_coriolis,
                dense=self.dense is not None,
            )
            wp.launch(
                advance_kernel,
                dim=thread_count,
                block_dim=128 if tile_width < 32 else 32,
                inputs=[
                    wp.int32(articulation_count),
                    wp.int32(tile_width),
                    wp.int32(self.advance_max_depth),
                    self.advance_articulation_depth_start,
                    self.advance_articulation_depth_joint,
                    self.advance_joint_parent_lane,
                    self.factor_child_start,
                    self.factor_child_joint,
                    *advance_inputs,
                ],
                outputs=advance_outputs,
                device=self.device,
            )
        else:
            wp.launch(
                _advance_reduced_articulations_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.model.articulation_start,
                    self.model.articulation_end,
                    wp.int32(self.advance_max_depth),
                    self.advance_articulation_depth_start,
                    self.advance_articulation_depth_joint,
                    *advance_inputs,
                    wp.bool(include_external),
                    wp.bool(include_coriolis),
                ],
                outputs=advance_outputs,
                device=self.device,
            )

    def solve_forces(
        self,
        generalized_force: wp.array[wp.float32],
        body_force: wp.array[wp.spatial_vector],
    ) -> wp.array[wp.float32]:
        """Apply the factored inverse mass to generalized and body forces."""
        wp.launch(
            _solve_articulated_system_kernel,
            dim=int(self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_qd_start,
                self.joint_s,
                self.joint_u_matrix,
                self.joint_d_inv,
                generalized_force,
                body_force,
            ],
            outputs=[
                self.body_work,
                self.joint_work,
                self.body_acceleration,
                self.generalized_acceleration,
            ],
            device=self.device,
        )
        return self.generalized_acceleration

    def solve_generalized(self, generalized_force: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Apply the factored inverse mass to generalized forces."""
        self.body_force.zero_()
        return self.solve_forces(generalized_force, self.body_force)

    def solve_body_forces(self, body_force: wp.array[wp.spatial_vector]) -> wp.array[wp.float32]:
        """Apply the factored inverse mass to world-origin body wrenches."""
        return self.solve_forces(self.zero_generalized_force, body_force)


_PERSISTENT_ELIGIBLE = "eligible"


def _build_persistent_articulation_eligibility(
    model: Model,
    loop_joint_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Prove which reduced trees have no dynamic coupling outside the tree."""
    articulation_count = int(model.articulation_count)
    eligible = np.ones(articulation_count, dtype=bool)
    reasons = [_PERSISTENT_ELIGIBLE] * articulation_count
    if articulation_count == 0:
        return eligible, tuple(reasons)

    def reject(articulation: int, reason: str) -> None:
        if articulation >= 0 and eligible[articulation]:
            eligible[articulation] = False
            reasons[articulation] = reason

    articulation_start = model.articulation_start.numpy()
    articulation_end = model.articulation_end.numpy()
    joint_child = model.joint_child.numpy()
    body_articulation = np.full(int(model.body_count), -1, dtype=np.int32)
    for articulation in range(articulation_count):
        start = int(articulation_start[articulation])
        end = int(articulation_end[articulation])
        body_articulation[joint_child[start:end]] = articulation

    if int(getattr(model, "particle_count", 0) or 0) > 0:
        for articulation in range(articulation_count):
            reject(articulation, "deformable scene")

    joint_parent = model.joint_parent.numpy()
    for joint in np.asarray(loop_joint_indices, dtype=np.int32):
        parent = int(joint_parent[joint])
        child = int(joint_child[joint])
        parent_articulation = int(body_articulation[parent]) if parent >= 0 else -1
        child_articulation = int(body_articulation[child]) if child >= 0 else -1
        reject(parent_articulation, "loop or external joint")
        reject(child_articulation, "loop or external joint")

    # An all-against-all list of one large world proves nothing and is costly to build.
    pairs_array = None if shape_pairs_are_dense(model) else getattr(model, "shape_contact_pairs", None)
    if pairs_array is None:
        for articulation in range(articulation_count):
            reject(articulation, "unproven collision topology")
        return eligible, tuple(reasons)

    if model.shape_body is None:
        if int(model.shape_count) > 0:
            for articulation in range(articulation_count):
                reject(articulation, "missing shape ownership")
        return eligible, tuple(reasons)

    shape_body = model.shape_body.numpy()
    for shape_a, shape_b in pairs_array.numpy():
        body_a = int(shape_body[int(shape_a)])
        body_b = int(shape_body[int(shape_b)])
        articulation_a = int(body_articulation[body_a]) if body_a >= 0 else -1
        articulation_b = int(body_articulation[body_b]) if body_b >= 0 else -1
        if articulation_a >= 0 and body_b >= 0:
            reject(articulation_a, "dynamic contact")
        if articulation_b >= 0 and body_a >= 0:
            reject(articulation_b, "dynamic contact")

    return eligible, tuple(reasons)


def _merge_touching_trees(model: Model, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Join consecutive trees of one world whose shapes may touch into one articulation range.

    One block of the fused dense substep then holds them, and their contacts are rows
    between two links. A group starts with any tree and continues with trees on fixed
    roots, while it fits a block: 32 joints and 64 DOFs.
    """
    endpoints = _model_pair_endpoints(model)
    if endpoints is None or len(starts) < 2:
        return starts, ends
    _pairs, _body0, _body1, tree0, tree1, _static0, _static1 = endpoints
    touching = {(int(a), int(b)) for a, b in zip(tree0, tree1, strict=True) if a >= 0 and b >= 0 and a != b}
    touching |= {(b, a) for a, b in touching}
    joint_world = model.joint_world.numpy() if model.joint_world is not None else np.zeros(model.joint_count, np.int32)
    joint_type = model.joint_type.numpy()
    dof_start = model.joint_qd_start.numpy()
    group_of = np.arange(len(starts))
    for tree in range(1, len(starts)):
        group = int(group_of[tree - 1])
        first = int(starts[group])
        members = range(group, tree)
        if (
            int(starts[tree]) == int(ends[tree - 1])
            and int(joint_world[starts[tree]]) == int(joint_world[first])
            and int(joint_type[starts[tree]]) not in (int(JointType.FREE), int(JointType.DISTANCE))
            and any((member, tree) in touching for member in members)
            and int(ends[tree]) - first <= 32
            and int(dof_start[ends[tree]]) - int(dof_start[first]) <= 64
        ):
            group_of[tree] = group
    keep = np.flatnonzero(group_of == np.arange(len(starts)))
    last = np.array([np.flatnonzero(group_of == group)[-1] for group in keep], dtype=np.int64)
    return starts[keep], ends[last]


def _get_reduced_model(model: Model, merge_trees: bool = False, selected: np.ndarray | None = None) -> Model:
    """Keep trees with prescribed body motion and free rigid bodies outside reduced dynamics.

    ``merge_trees`` also joins trees that may touch, see :func:`_merge_touching_trees`.
    ``selected`` keeps only the marked articulations reduced; the others become rigid
    bodies whose joints the regular solver holds.

    A lone body on a free joint beside jointed trees is a rigid body: the
    maximal-coordinate solver integrates it and solves its contacts with other
    such bodies and static geometry directly, without an articulated response.
    A model of free bodies only keeps them reduced.
    """
    starts = model.articulation_start.numpy()[: model.articulation_count]
    ends = model.articulation_end.numpy()
    children = model.joint_child.numpy()
    parents = model.joint_parent.numpy()
    kinds = model.joint_type.numpy()
    flags = model.body_flags.numpy()
    free_body = np.array(
        [
            end - start == 1 and parents[start] < 0 and kinds[start] == int(JointType.FREE)
            for start, end in zip(starts, ends, strict=False)
        ],
        dtype=bool,
    )
    eligible = np.array(
        [
            not np.any(flags[children[start:end]] & int(BodyFlags.KINEMATIC))
            for start, end in zip(starts, ends, strict=False)
        ],
        dtype=bool,
    )
    if selected is not None:
        eligible &= selected[: len(eligible)]
    if np.any(eligible & ~free_body):
        eligible &= ~free_body
    if np.all(eligible) and not merge_trees:
        return model
    reduced_model = copy.copy(model)
    selected_starts = starts[eligible]
    selected_ends = ends[eligible]
    reduced_model.articulation_start = wp.array(
        np.append(selected_starts, selected_ends[-1] if selected_ends.size else 0).astype(np.int32),
        dtype=wp.int32,
        device=model.device,
    )
    reduced_model.articulation_end = wp.array(selected_ends.astype(np.int32), dtype=wp.int32, device=model.device)
    reduced_model.articulation_count = int(selected_ends.size)
    if merge_trees and selected_ends.size:
        # The pair census numbers the trees as the model above lists them.
        merged_starts, merged_ends = _merge_touching_trees(reduced_model, selected_starts, selected_ends)
        if merged_ends.size == selected_ends.size:
            return model if np.all(eligible) else reduced_model
        reduced_model.articulation_start = wp.array(
            np.append(merged_starts, merged_ends[-1]).astype(np.int32), dtype=wp.int32, device=model.device
        )
        reduced_model.articulation_end = wp.array(merged_ends.astype(np.int32), dtype=wp.int32, device=model.device)
        reduced_model.articulation_count = int(merged_ends.size)
    return reduced_model


@wp.kernel(enable_backward=False)
def _copy_indexed_kernel(index: wp.array[wp.int32], source: wp.array[wp.float32], destination: wp.array[wp.float32]):
    destination[index[wp.tid()]] = source[index[wp.tid()]]


@wp.kernel(enable_backward=False)
def _phoenx_body_poses_kernel(bodies: BodyContainer, body_com: wp.array[wp.vec3], body_q: wp.array[wp.transform]):
    """Body origin poses from the PhoenX body centers of mass."""
    body = wp.tid()
    rotation = bodies.orientation[body + wp.int32(1)]
    position = bodies.position[body + wp.int32(1)] - wp.quat_rotate(rotation, body_com[body])
    body_q[body] = wp.transform(position, rotation)


class ReducedPhoenXArticulation:
    """Graph-stable bridge between common Newton articulation state and PhoenX bodies."""

    def __init__(
        self,
        model: Model,
        bodies: BodyContainer,
        *,
        execution_path: str = "reference",
        contact_friction_model: str = "point",
        backend: str = "aba",
        external_rigid_rows: bool = False,
    ):
        if backend not in ("aba", "dense"):
            raise ValueError(f"backend must be 'aba' or 'dense', got {backend!r}")
        if backend == "dense" and execution_path != "reference":
            raise ValueError("the dense reduced backend requires the reference execution path")
        if execution_path not in ("reference", "persistent"):
            raise ValueError(f"execution_path must be 'reference' or 'persistent', got {execution_path!r}")
        if contact_friction_model not in ("point", "patch"):
            raise ValueError(f"contact_friction_model must be 'point' or 'patch', got {contact_friction_model!r}")
        if execution_path == "persistent" and contact_friction_model == "patch":
            raise ValueError("persistent reduced articulations do not support patch friction")
        self.execution_path = execution_path
        self.model = model
        self.bodies = bodies
        self.system = ReducedArticulationSystem(model, dense=backend == "dense")
        if self.system.runtime_slots and (execution_path != "reference" or not model.device.is_cuda):
            raise ValueError("runtime reduced articulations require CUDA and the reference execution path")
        self.state = model.state()
        self.control = model.control(clone_variables=True)
        self._kinematics_current = False
        # Rows imported from the caller's state; runtime instance slots, if any, follow them.
        capacity = getattr(model, "_dynamic_capacity", None)
        if capacity is not None and capacity.slot_size:
            self._imported_rows = (
                int(model.joint_q_start.numpy()[capacity.slot_joint_start]),
                int(model.joint_qd_start.numpy()[capacity.slot_joint_start]),
                capacity.slot_body_start,
            )
        else:
            self._imported_rows = (
                int(model.joint_coord_count),
                int(model.joint_dof_count),
                int(model.body_count),
            )

        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        joint_child = model.joint_child.numpy()
        body_is_reduced = wp.zeros(int(model.body_count), dtype=wp.int32, device=model.device)
        body_is_reduced_np = body_is_reduced.numpy()
        tree_joint_mask = wp.zeros(int(model.joint_count), dtype=wp.int32, device=model.device)
        tree_joint_mask_np = tree_joint_mask.numpy()
        constraint_node_np = bodies.constraint_node.numpy()
        body_articulation_np = np.full(int(model.body_count) + 1, -1, dtype=np.int32)
        articulation_constraint_node_np = np.empty(int(model.articulation_count), dtype=np.int32)
        body_joint_np = np.full(int(model.body_count), -1, dtype=np.int32)
        for joint, child in enumerate(joint_child):
            body_joint_np[int(child)] = joint
        body_path_start_np = np.zeros(int(model.body_count) + 1, dtype=np.int32)
        body_path_joint_list: list[int] = []
        joint_parent_np = model.joint_parent.numpy()
        for body in range(int(model.body_count)):
            path: list[int] = []
            joint = int(body_joint_np[body])
            while joint >= 0:
                path.append(joint)
                parent = int(joint_parent_np[joint])
                joint = int(body_joint_np[parent]) if parent >= 0 else -1
            body_path_joint_list.extend(reversed(path))
            body_path_start_np[body + 1] = len(body_path_joint_list)
        for articulation in range(int(model.articulation_count)):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            children = joint_child[start:end]
            tree_joint_mask_np[start:end] = 1
            body_is_reduced_np[children] = 1
            constraint_node = int(children[0]) + 1
            constraint_node_np[children + 1] = constraint_node
            articulation_constraint_node_np[articulation] = constraint_node
            body_articulation_np[children + 1] = articulation
        bodies.constraint_node.assign(constraint_node_np)
        self.articulation_constraint_node = wp.array(articulation_constraint_node_np, device=model.device)

        reduced = ReducedArticulationData()
        reduced.enabled = wp.ones(1, dtype=wp.int32, device=model.device)
        reduced.body_articulation = wp.array(body_articulation_np, device=model.device)
        reduced.articulation_origin = self.system.articulation_origin
        reduced.body_q_com = self.system.body_q_com
        reduced.articulation_start = model.articulation_start
        reduced.articulation_end = model.articulation_end
        reduced.joint_parent = model.joint_parent
        reduced.joint_child = model.joint_child
        reduced.joint_qd_start = model.joint_qd_start
        reduced.joint_s = self.system.joint_s
        reduced.joint_u = self.system.joint_u_matrix
        reduced.joint_d_inv = self.system.joint_d_inv
        reduced.joint_qd = self.system.joint_qd_internal
        reduced.body_work = self.system.body_work
        reduced.joint_work = self.system.joint_work
        reduced.body_acceleration = self.system.body_acceleration
        reduced.generalized_response = self.system.generalized_acceleration
        reduced.impulse_response = self.system.impulse_response
        reduced.deferred_wrench = self.system.deferred_wrench
        reduced.deferred_pending = self.system.deferred_pending
        reduced.external_owner = wp.full(int(model.body_count) + 1, value=-1, dtype=wp.int32, device=model.device)
        reduced.external_lowest = wp.full_like(reduced.external_owner, value=-1)
        reduced.external_highest = wp.full_like(reduced.external_owner, value=-1)
        reduced.external_delta_linear = wp.zeros(2 * (int(model.body_count) + 1), dtype=wp.vec3, device=model.device)
        reduced.external_delta_angular = wp.zeros_like(reduced.external_delta_linear)
        reduced.body_joint = wp.array(body_joint_np, device=model.device)
        reduced.body_path_start = wp.array(body_path_start_np, device=model.device)
        reduced.body_path_joint = wp.array(
            np.asarray(body_path_joint_list, dtype=np.int32),
            device=model.device,
        )
        dense = self.system.dense
        reduced.dense = dense.data if dense is not None else dense_tree_data_zeros(model.device)
        bodies.reduced = reduced
        body_is_reduced.assign(body_is_reduced_np)
        tree_joint_mask.assign(tree_joint_mask_np)
        self.body_is_reduced = body_is_reduced
        self.tree_joint_mask = tree_joint_mask
        self.body_is_reduced_np = body_is_reduced_np.astype(bool)
        self.tree_joint_mask_np = tree_joint_mask_np.astype(bool)
        self.loop_system = ReducedLoopSystem(model, self.body_is_reduced_np)
        self.persistent_eligible_np, self.persistent_ineligible_reasons = _build_persistent_articulation_eligibility(
            model, self.loop_system.joint_indices_np
        )
        self.persistent_eligible = wp.array(
            self.persistent_eligible_np.astype(np.int32), dtype=wp.int32, device=model.device
        )
        if execution_path == "persistent":
            if not model.device.is_cuda:
                raise ValueError("execution_path='persistent' currently requires CUDA")
            if not self.system.use_warp_advance:
                raise ValueError("persistent reduced articulations require at most 32 joints at each tree depth")
            if not bool(np.all(self.persistent_eligible_np)):
                rejected = sorted(
                    {reason for reason in self.persistent_ineligible_reasons if reason != _PERSISTENT_ELIGIBLE}
                )
                raise ValueError(f"persistent articulation path is not proven safe: {', '.join(rejected)}")
        self.contact_block_system = ReducedContactBlockSystem(
            model,
            articulation_depth_start=self.system.advance_articulation_depth_start,
            articulation_depth_joint=self.system.advance_articulation_depth_joint,
            joint_parent_lane=self.system.advance_joint_parent_lane,
            row_tile_width=self.system.advance_tile_width,
            max_depth=self.system.advance_max_depth,
            use_depth_warp=self.system.use_warp_advance,
            allow_patch_rows=execution_path != "persistent",
            use_patch_rows=contact_friction_model == "patch",
            external_rigid_rows=external_rigid_rows,
        )
        #: Free rigid bodies whose contacts with a tree that tree's fused substep solves.
        self.external_rigid_rows = False
        external_owner = self.contact_block_system.external_owner
        if external_owner is not None and np.any(external_owner >= 0):
            self.external_rigid_rows = True
            owner_np = np.full(int(model.body_count) + 1, -1, dtype=np.int32)
            owner_np[1:] = external_owner
            reduced.external_owner.assign(owner_np)
        elif self.contact_block_system.dynamic_external_owner:
            self.external_rigid_rows = True
        if dense is not None:
            # Coupled contacts and loop closures still read the articulated-body factor.
            dense.requires_aba = (
                dense.has_fallback
                or self.contact_block_system.requires_impulse_response
                or bool(self.loop_system.joint_indices_np.size)
            )
            self.contact_block_system.dense_rows = dense
        if execution_path == "persistent":
            self.contact_block_system.biased_page_launcher = self._launch_persistent_biased_page
            self.contact_block_system.relax_page_launcher = self._launch_persistent_relax_page
        # Fused substep of dense trees with private contacts, installed by the solver. A relax
        # step leaves its factor for the next biased step at the same pose.
        self.fused_substep = None
        #: Public rates before the cross-contact solve of a fused substep.
        self._cross_rates_before = wp.zeros(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=model.device)
        self._cross_boost_all = wp.zeros(max(1, int(model.articulation_count)), dtype=wp.int32, device=model.device)
        #: Linear momentum of every tree after the cross-contact impulses.
        self._cross_momentum = wp.zeros(max(1, int(model.articulation_count)), dtype=wp.vec3, device=model.device)
        self._relaxed_pose_factored = False
        self.owned_joint_mask_np = self.tree_joint_mask_np.copy()
        self.owned_joint_mask_np[self.loop_system.joint_indices_np] = True
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        owned_boundaries = np.flatnonzero(np.diff(np.r_[False, self.owned_joint_mask_np, False]))
        self._owned_coordinate_ranges = [
            (int(joint_q_start[start]), int(joint_q_start[end]), int(joint_qd_start[start]), int(joint_qd_start[end]))
            for start, end in owned_boundaries.reshape(-1, 2)
        ]
        # Many ranges, as with free bodies between the trees of many worlds, export by index.
        self._owned_coordinates = None
        if len(self._owned_coordinate_ranges) > 4:
            self._owned_coordinates = tuple(
                wp.array(
                    np.concatenate([np.arange(r[first], r[first + 1]) for r in self._owned_coordinate_ranges]),
                    dtype=wp.int32,
                    device=model.device,
                )
                for first in (0, 2)
            )

        wp.launch(
            _mark_articulated_bodies_kernel,
            dim=int(model.body_count),
            inputs=[self.body_is_reduced],
            outputs=[bodies],
            device=model.device,
        )

    def invalidate_kinematics(self) -> None:
        """Force reconstruction after a model-property change."""
        self._kinematics_current = False
        self._relaxed_pose_factored = False

    def import_step(
        self,
        state: State,
        control: Control,
        *,
        state_is_continuation: bool = False,
        state_kinematics_valid: bool = False,
    ) -> None:
        """Import common generalized state and controls into stable graph buffers."""
        # A continuation keeps the pose-derived buffers published by the preceding step. Runtime
        # instance slots, the last model rows, change topology between steps and own their state:
        # their scenes rebuild the buffers every step from the PhoenX body poses, which spawns
        # write, and import only the rows before the slots.
        self._kinematics_current = state_is_continuation and not self.system.runtime_slots
        # A continuation also keeps the factor the last relax step left at this pose.
        self._relaxed_pose_factored = self._relaxed_pose_factored and self._kinematics_current
        if self.system.runtime_slots:
            wp.launch(
                _phoenx_body_poses_kernel,
                dim=int(self.model.body_count),
                inputs=[self.bodies, self.model.body_com],
                outputs=[self.state.body_q],
                device=self.model.device,
            )
        if not state_is_continuation:
            q_end, qd_end, body_end = self._imported_rows
            for dest, src, count in (
                (self.state.joint_q, state.joint_q, q_end),
                (self.state.joint_qd, state.joint_qd, qd_end),
                *(
                    ((self.state.body_q, state.body_q, body_end), (self.state.body_qd, state.body_qd, body_end))
                    if state_kinematics_valid
                    else ()
                ),
            ):
                if count > 0:  # a zero count would copy everything
                    wp.copy(dest, src, count=count)
            if not state_kinematics_valid:
                eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
                self.sync_bodies()
        wp.copy(self.state.body_f, state.body_f)
        wp.copy(self.control.joint_f, control.joint_f)
        wp.copy(self.control.joint_target_q, control.joint_target_q)
        wp.copy(self.control.joint_target_qd, control.joint_target_qd)

    def begin_substep(
        self,
        dt: float,
        *,
        split_dynamics: bool,
        compute_impulse_response: bool = True,
        world=None,
    ) -> None:
        """Apply pre-PGS reduced dynamics at the current configuration.

        With a fused substep, see :attr:`fused_substep`, this runs the whole biased
        substep through publication and needs the owning ``world``.
        """
        if self.fused_substep is not None:
            self.fused_substep.biased(world, dt, reuse_factor=self._relaxed_pose_factored)
            self._relaxed_pose_factored = False
            self._kinematics_current = True
            if compute_impulse_response:
                self._solve_cross_contacts(world, dt)
            return
        self.system.factor(
            self.state,
            self.control,
            dt,
            update_kinematics=not self._kinematics_current,
        )
        self.system.advance(
            self.state,
            self.control,
            self.bodies,
            dt,
            include_external=True,
            include_coriolis=not split_dynamics,
        )
        if compute_impulse_response:
            self._refresh_impulse_response()

    def _solve_cross_contacts(self, world, dt: float, *, relax: bool = False) -> None:
        """Solve the contacts between articulations that the fused substep leaves out.

        The fused kernels own the contacts of one tree. Contacts that cross between two trees,
        or reach a free body no tree owns, are scheduled as fallback columns; they are solved
        here, between the biased and the relax kernel, with the articulated-body impulse
        response at the integrated pose. The impulses on both sides are equal and opposite and
        enter the rates the relax kernel continues from. The poses follow in the next substep.
        """
        contacts = self.contact_block_system
        if world.max_contact_columns <= 0:
            return
        idt = wp.float32(1.0 / float(dt))

        def solve_fallback(iterations: int, *, relax: bool, prepare: bool) -> None:
            if iterations > 0:
                contacts.solve_fallback(
                    world._contact_cols,
                    world.bodies,
                    idt,
                    world.sor_boost,
                    world._contact_container,
                    world._active_contact_views(),
                    iterations,
                    use_bias=not relax,
                    prepare=prepare,
                )

        def solve() -> None:
            model = self.model
            # The fused kernels keep the public rates; the articulated-body factor forms its
            # internal rates from them at the published pose.
            wp.copy(self._cross_rates_before, self.state.joint_qd)
            self.system.factor(self.state, self.control, dt, update_kinematics=True)
            self._refresh_impulse_response()
            solve_fallback(world.solver_iterations, relax=False, prepare=True)
            self._publish_state(0.0, update_configuration=False)
            momentum_inputs = [model.articulation_start, model.articulation_end]
            wp.launch(
                _capture_reduced_linear_momentum_kernel,
                dim=int(model.articulation_count),
                inputs=[*momentum_inputs, model.joint_child, model.body_mass, self.bodies],
                outputs=[self._cross_momentum],
                device=model.device,
            )
            # The fused kernel integrated the coordinates with the rates from before these
            # impulses: integrate the change of the rates as well, as one solve of all rows
            # would have, and have the relax kernel form its frames at the corrected pose.
            wp.launch(
                _integrate_rate_change_kernel,
                dim=int(model.joint_count),
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_X_c,
                    model.body_com,
                    self.state.joint_qd,
                    self._cross_rates_before,
                    wp.float32(dt),
                ],
                outputs=[self.state.joint_q],
                device=model.device,
            )
            eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
            self.sync_bodies()
            # The same rates at the corrected pose carry a slightly different linear momentum.
            wp.launch(
                _boost_reduced_linear_momentum_kernel,
                dim=int(model.articulation_count),
                inputs=[
                    *momentum_inputs,
                    model.joint_type,
                    model.joint_child,
                    model.joint_qd_start,
                    model.joint_X_p,
                    model.body_mass,
                    self._cross_momentum,
                    self.system.joint_s_publish,
                    self.bodies,
                ],
                outputs=[
                    self.system.joint_qd_internal,
                    self.state.joint_qd,
                    self.state.body_qd,
                    # Also for trees on the midpoint step, which only conserves its own integration.
                    self._cross_boost_all,
                ],
                device=model.device,
            )
            self.fused_substep.pose_changed.fill_(1)

        def relax_rows() -> None:
            # After the free bodies integrated too, relax sweeps remove the velocity the bias left.
            self.system.factor(self.state, self.control, dt, update_kinematics=True)
            self._refresh_impulse_response()
            solve_fallback(world._active_velocity_iterations, relax=True, prepare=True)
            self._publish_state(0.0, update_configuration=False)

        if relax:
            solve = relax_rows

        # Most steps of a large scene have no such contact: skip the stage on the device.
        wp.capture_if(contacts.cross_column_count, on_true=solve)

    def _refresh_impulse_response(self) -> None:
        if self.model.device.is_cuda:
            articulation_count = int(self.model.articulation_count)
            impulse_response_threads = ((articulation_count + 3) // 4) * 32
            wp.launch(
                _compute_reduced_impulse_response_warp_kernel,
                dim=impulse_response_threads,
                block_dim=128,
                inputs=[self.bodies, wp.int32(articulation_count)],
                device=self.model.device,
            )
        else:
            wp.launch(
                _compute_reduced_impulse_response_kernel,
                dim=int(self.model.articulation_count),
                inputs=[self.bodies],
                device=self.model.device,
            )

    def _publish_state(
        self,
        dt: float,
        *,
        update_configuration: bool = True,
        restore_linear_momentum: bool = False,
    ) -> None:
        if update_configuration or not self.system.use_warp_publish:
            wp.copy(self.system.joint_qd_integrator, self.system.joint_qd_internal)
        if self.system.use_warp_publish:
            articulation_count = int(self.model.articulation_count)
            tile_width = self.system.advance_tile_width
            thread_count = ((articulation_count * tile_width + 31) // 32) * 32
            wp.launch(
                _finish_and_publish_reduced_warp_kernel,
                dim=thread_count,
                block_dim=128 if tile_width < 32 else 32,
                inputs=[
                    wp.int32(articulation_count),
                    wp.int32(tile_width),
                    wp.int32(self.system.advance_max_depth),
                    self.system.advance_articulation_depth_start,
                    self.system.advance_articulation_depth_joint,
                    self.state.joint_q,
                    self.system.joint_qd_internal,
                    self.system.joint_qd_integrator,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.model.joint_axis,
                    self.model.joint_dof_dim,
                    self.model.body_com,
                    self.system.body_x_com,
                    self.system.zero_generalized_force,
                    wp.float32(dt),
                    wp.bool(update_configuration),
                    self.model.body_mass,
                    self.system.target_linear_momentum,
                    wp.bool(restore_linear_momentum),
                ],
                outputs=[
                    self.state.joint_qd,
                    self.state.body_q,
                    self.bodies,
                    self.system.articulation_origin,
                    self.system.body_q_local,
                    self.system.body_q_com,
                    self.system.joint_anchor_local,
                    self.system.joint_s_publish,
                    self.state.body_qd,
                    self.system.midpoint.flag,
                ],
                device=self.model.device,
            )
        else:
            wp.launch(
                _finish_reduced_articulations_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.model.articulation_start,
                    self.model.articulation_end,
                    self.state.joint_q,
                    self.system.joint_qd_internal,
                    self.system.joint_qd_integrator,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.model.joint_axis,
                    self.model.joint_dof_dim,
                    self.model.body_com,
                    self.system.zero_generalized_force,
                    wp.float32(dt),
                    wp.bool(update_configuration),
                    self.system.midpoint.flag,
                ],
                outputs=[
                    self.state.joint_qd,
                    self.state.body_q,
                    self.bodies,
                ],
                device=self.model.device,
            )
            self.system.update_local_kinematics(self.state)
            wp.launch(
                _publish_reduced_local_velocities_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.model.articulation_start,
                    self.model.articulation_end,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.state.joint_q,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_axis,
                    self.model.joint_dof_dim,
                    self.system.joint_anchor_local,
                    self.system.body_q_com,
                    self.system.joint_qd_internal,
                    self.bodies,
                ],
                outputs=[self.system.joint_s_publish, self.state.body_qd],
                device=self.model.device,
            )
            if update_configuration and restore_linear_momentum:
                wp.launch(
                    _boost_reduced_linear_momentum_kernel,
                    dim=int(self.model.articulation_count),
                    inputs=[
                        self.model.articulation_start,
                        self.model.articulation_end,
                        self.model.joint_type,
                        self.model.joint_child,
                        self.model.joint_qd_start,
                        self.model.joint_X_p,
                        self.model.body_mass,
                        self.system.target_linear_momentum,
                        self.system.joint_s_publish,
                        self.bodies,
                    ],
                    outputs=[
                        self.system.joint_qd_internal,
                        self.state.joint_qd,
                        self.state.body_qd,
                        self.system.midpoint.flag,
                    ],
                    device=self.model.device,
                )

    def _advance_and_publish_fused(self, dt: float) -> None:
        articulation_count = int(self.model.articulation_count)
        tile_width = self.system.advance_tile_width
        thread_count = ((articulation_count * tile_width + 31) // 32) * 32
        kernel = _make_advance_reduced_articulations_warp_kernel(
            include_external=False,
            include_coriolis=True,
            fuse_publish=True,
            static_tile_width=tile_width,
            dense=self.system.dense is not None,
        )
        wp.launch(
            kernel,
            dim=thread_count,
            block_dim=128 if tile_width < 32 else 32,
            inputs=[
                wp.int32(articulation_count),
                wp.int32(tile_width),
                wp.int32(self.system.advance_max_depth),
                self.system.advance_articulation_depth_start,
                self.system.advance_articulation_depth_joint,
                self.system.advance_joint_parent_lane,
                self.system.factor_child_start,
                self.system.advance_child_depth_index,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_world,
                self.model.gravity,
                self.system.body_x_com,
                self.system.zero_generalized_force,
                self.control.joint_f,
                self.system.joint_implicit_force,
                self.state.body_f,
                wp.float32(dt),
                self.state.joint_q,
                self.system.joint_qd_internal,
                self.system.joint_s,
                self.system.joint_u_matrix,
                self.system.joint_d_inv,
                self.system.body_i_s,
                self.system.body_velocity,
                self.system.body_coriolis,
                self.system.body_bias,
                self.system.generalized_rhs,
                self.system.body_work,
                self.system.joint_work,
                self.system.body_acceleration,
                self.system.generalized_acceleration,
                self.state.joint_qd,
                self.state.body_q,
                self.state.body_qd,
                self.bodies,
                self.system.bind_midpoint_state(self.state),
                self.system.articulation_origin,
                self.system.body_q_local,
                self.system.body_q_com,
                self.system.joint_anchor_local,
                self.system.joint_s_publish,
            ],
            device=self.model.device,
        )

    def _launch_persistent_biased_page(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        cc: ContactContainer,
        iterations: int,
    ) -> None:
        contact = self.contact_block_system
        assert contact.packed_jacobian is not None
        assert contact.packed_response is not None
        kernel = _make_biased_contact_advance_publish_kernel(contact.contact_dof_width)
        wp.launch_tiled(
            kernel,
            dim=[int(self.model.articulation_count)],
            block_dim=32,
            inputs=[
                columns,
                bodies,
                idt,
                wp.float32(sor_boost),
                cc,
                wp.int32(iterations),
                contact.enabled,
                contact.point_count,
                contact.point_contact,
                contact.point_column,
                contact.normal,
                contact.tangent0,
                contact.row_velocity,
                contact.page_index,
                contact.max_page_count,
                contact.packed_jacobian,
                contact.packed_response,
                wp.int32(self.system.advance_max_depth),
                self.system.advance_articulation_depth_start,
                self.system.advance_articulation_depth_joint,
                contact.generalized_delta,
                contact.generalized_body_delta,
                wp.int32(self.model.articulation_count),
                self.system.advance_joint_parent_lane,
                self.system.factor_child_start,
                self.system.factor_child_joint,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_world,
                self.model.gravity,
                self.system.body_x_com,
                self.system.zero_generalized_force,
                self.control.joint_f,
                self.system.joint_implicit_force,
                self.state.body_f,
                wp.float32(1.0 / float(idt)),
                self.state.joint_q,
                self.system.joint_qd_internal,
                self.system.joint_qd_integrator,
                self.system.joint_s,
                self.system.joint_u_matrix,
                self.system.joint_d_inv,
                self.system.body_i_s,
                self.system.body_velocity,
                self.system.body_coriolis,
                self.system.body_bias,
                self.system.generalized_rhs,
                self.system.body_work,
                self.system.joint_work,
                self.system.body_acceleration,
                self.system.generalized_acceleration,
                self.state.joint_qd,
                self.state.body_q,
                self.state.body_qd,
                self.system.bind_midpoint_state(self.state),
                self.system.articulation_origin,
                self.system.body_q_local,
                self.system.body_q_com,
                self.system.joint_anchor_local,
                self.system.joint_s_publish,
                contact.biased_advanced,
            ],
            device=self.model.device,
        )

    def _launch_persistent_relax_page(
        self,
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        idt: wp.float32,
        sor_boost: float,
        cc: ContactContainer,
        iterations: int,
    ) -> None:
        contact = self.contact_block_system
        assert contact.packed_jacobian is not None
        assert contact.packed_response is not None
        kernel = _make_relax_and_publish_reduced_contacts_kernel(contact.contact_dof_width)
        wp.launch_tiled(
            kernel,
            dim=[int(self.model.articulation_count)],
            block_dim=32,
            inputs=[
                columns,
                bodies,
                idt,
                wp.float32(sor_boost),
                cc,
                wp.int32(iterations),
                contact.enabled,
                contact.point_count,
                contact.point_contact,
                contact.point_column,
                contact.point0,
                contact.point1,
                contact.normal,
                contact.tangent0,
                contact.row_velocity,
                contact.page_index,
                contact.max_page_count,
                contact.packed_jacobian,
                contact.packed_response,
                wp.int32(self.system.advance_max_depth),
                self.system.advance_articulation_depth_start,
                self.system.advance_articulation_depth_joint,
                contact.generalized_delta,
                contact.generalized_body_delta,
                wp.int32(self.model.articulation_count),
                self.state.joint_q,
                self.system.joint_qd_integrator,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                self.model.body_com,
                self.model.body_mass,
                self.system.body_x_com,
                self.system.zero_generalized_force,
                self.state.joint_qd,
                self.state.body_q,
                self.system.articulation_origin,
                self.system.body_q_local,
                self.system.body_q_com,
                self.system.joint_anchor_local,
                self.system.joint_s_publish,
                self.state.body_qd,
                self.system.midpoint.flag,
            ],
            device=self.model.device,
        )

    def finish_relax(self) -> None:
        """Publish bias-free velocity corrections without advancing configuration."""
        if self.execution_path == "persistent" or self.fused_substep is not None:
            return
        # Relax changes only velocities, so the pose-derived buffers from the
        # preceding full publication remain current.
        self._publish_state(0.0, update_configuration=False)

    def end_substep(self, dt: float, *, split_dynamics: bool) -> None:
        """Apply the post-impulse velocity-product step and integrate.

        Each tree picks the explicit step or the momentum-conserving midpoint
        step from its own fastest link rate, see
        :func:`_advance_reduced_articulations_kernel`.

        Floating trees on the explicit step also get their incoming linear
        momentum back through a uniform boost. That update conserves it only to
        first order in ``dt``, and internal spin otherwise turns the drift into
        spurious center-of-mass motion. A boost leaves every angular velocity,
        joint rate, and the angular momentum about the center of mass unchanged.
        """
        self._kinematics_current = True
        if self.fused_substep is not None:
            return
        if self.execution_path == "persistent" and split_dynamics:
            wp.capture_if(
                self.contact_block_system.biased_advanced,
                on_true=lambda: None,
                on_false=lambda: self._advance_and_publish_fused(dt),
            )
            return
        if (
            split_dynamics
            and self.model.device.is_cuda
            and self.system.use_warp_advance
            and self.system.use_warp_publish
        ):
            self._advance_and_publish_fused(dt)
            return
        wp.launch(
            _capture_reduced_linear_momentum_kernel,
            dim=int(self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_child,
                self.model.body_mass,
                self.bodies,
            ],
            outputs=[self.system.target_linear_momentum],
            device=self.model.device,
        )
        if split_dynamics:
            self.system.advance(
                self.state,
                self.control,
                self.bodies,
                dt,
                include_external=False,
                include_coriolis=True,
            )
        self._publish_state(dt, restore_linear_momentum=True)

    def sync_bodies(self) -> None:
        """Write reduced link pose and COM twist into PhoenX body storage."""
        wp.launch(
            _sync_reduced_bodies_kernel,
            dim=int(self.model.body_count),
            inputs=[
                self.body_is_reduced,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[self.bodies],
            device=self.model.device,
        )

    def solve_constraints(self, world, idt: wp.float32, *, relax: bool) -> None:
        """Solve loop and contact blocks through the reduced inverse-mass operator."""
        iterations = world._active_velocity_iterations if relax else world.solver_iterations
        if iterations <= 0:
            return
        if self.fused_substep is not None:
            # The biased solve already ran in begin_substep.
            if relax and world._reduced_contacts_active_this_step:
                if self.contact_block_system.requires_impulse_response:
                    self._solve_cross_contacts(world, world.substep_dt, relax=True)
                self.fused_substep.relax(world, world.substep_dt)
                self._relaxed_pose_factored = True
            return
        if relax:
            # Integration changes the mass operator and contact Jacobians.
            # Reusing their old values applies impulses at the wrong pose.
            self.system.factor(
                self.state, self.control, 1.0 / float(idt), update_kinematics=not self._kinematics_current
            )
            if world._reduced_contacts_active_this_step and self.contact_block_system.requires_impulse_response:
                self._refresh_impulse_response()
        self.loop_system.solve(
            self.model,
            self.bodies,
            idt,
            world.sor_boost,
            iterations,
            use_bias=not relax,
            warmstart=not relax,
        )
        if world.max_contact_columns <= 0 or not world._reduced_contacts_active_this_step:
            return
        forest = getattr(self, "forest_contact_system", None)
        if forest is not None:
            forest.solve(
                world, idt, iterations, use_bias=not relax, prepare=relax or world._refresh_prepare_this_substep()
            )
            return
        contact_views = world._active_contact_views()
        common_inputs = [
            world._contact_cols,
            world.bodies,
            idt,
            wp.float32(world.sor_boost),
        ]
        prepare = relax or world._refresh_prepare_this_substep()
        common_tail = [
            world._contact_container,
            contact_views,
            self.contact_block_system.cross_mobility,
            wp.int32(iterations),
            wp.bool(not relax),
            wp.bool(prepare),
        ]
        self.contact_block_system.solve(
            world._contact_cols,
            world.bodies,
            idt,
            world.sor_boost,
            world._contact_container,
            contact_views,
            iterations,
            use_bias=not relax,
            prepare=prepare,
        )

        def solve_deferred_contacts() -> None:
            wp.launch(
                _solve_reduced_deferred_contacts_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.contact_block_system.schedule_section_end,
                    self.contact_block_system.schedule_columns,
                    self.contact_block_system.enabled,
                    *common_inputs,
                    *common_tail,
                ],
                device=self.model.device,
            )

        wp.capture_if(self.contact_block_system.deferred_active, on_true=solve_deferred_contacts)
        self.contact_block_system.solve_fallback(
            world._contact_cols,
            world.bodies,
            idt,
            world.sor_boost,
            world._contact_container,
            contact_views,
            iterations,
            use_bias=not relax,
            prepare=prepare,
        )

    def export_step(self, state: State) -> None:
        """Export authoritative generalized coordinates to common Newton state."""
        if self._owned_coordinates is not None:
            for index, source, destination in zip(
                self._owned_coordinates,
                (self.state.joint_q, self.state.joint_qd),
                (state.joint_q, state.joint_qd),
                strict=True,
            ):
                wp.launch(
                    _copy_indexed_kernel,
                    dim=index.shape[0],
                    inputs=[index, source],
                    outputs=[destination],
                    device=self.model.device,
                )
            return
        for q_start, q_end, qd_start, qd_end in self._owned_coordinate_ranges:
            if q_end > q_start:
                wp.copy(state.joint_q, self.state.joint_q, q_start, q_start, q_end - q_start)
            if qd_end > qd_start:
                wp.copy(state.joint_qd, self.state.joint_qd, qd_start, qd_start, qd_end - qd_start)


__all__ = ["ReducedArticulationSystem", "ReducedPhoenXArticulation"]
