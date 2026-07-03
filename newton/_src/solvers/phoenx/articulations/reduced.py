# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear-time articulated-body factorization for PhoenX."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Control, JointType, Model, State
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
from newton._src.solvers.phoenx.articulations.reduced_contact import (
    reduced_contact_iterate,
    reduced_contact_prepare,
)
from newton._src.solvers.phoenx.articulations.reduced_contact_block import ReducedContactBlockSystem
from newton._src.solvers.phoenx.articulations.reduced_loop import ReducedLoopSystem
from newton._src.solvers.phoenx.body import (
    MOTION_ARTICULATED,
    BodyContainer,
    ReducedArticulationData,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactColumnContainer, ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.semi_implicit.kernels_body import joint_force

_MAX_JOINT_DOF = 6
_WARP_FACTOR_MIN_ARTICULATIONS = 32
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_sym_mat66 = wp.types.vector(length=21, dtype=wp.float32)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def _spatial_to_vec6(value: wp.spatial_vector) -> _vec6:
    return _vec6(value[0], value[1], value[2], value[3], value[4], value[5])


@wp.func
def _vec6_to_spatial(value: _vec6) -> wp.spatial_vector:
    return wp.spatial_vector(value[0], value[1], value[2], value[3], value[4], value[5])


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


@wp.kernel(enable_backward=False)
def _advance_generalized_velocity_kernel(
    acceleration: wp.array[wp.float32],
    dt: wp.float32,
    velocity: wp.array[wp.float32],
):
    dof = wp.tid()
    velocity[dof] += acceleration[dof] * dt


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
    """Build the linearized implicit-Euler joint operator and right-hand side."""
    joint = wp.tid()
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
    factor_index = wp.tid()
    joint = factor_joint[factor_index]
    qd_start = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + wp.int32(1)]
    for dof in range(qd_start, qd_end):
        joint_qd_internal[dof] = joint_qd_public[dof]

    child = joint_child[joint]
    body_i_s[child] = _pack_transformed_body_inertia(body_q_com[child], body_mass[child], body_inertia[child])

    x_wpj = joint_anchor_local[joint]
    kind = joint_type[joint]
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        # Newton exposes the relative child-COM velocity in the parent-anchor
        # frame. ABA uses the velocity at that frame origin. Convert in the
        # articulation-local frame so large world translations never enter.
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

    jcalc_motion(
        kind,
        joint_axis,
        joint_q,
        joint_dof_dim[joint, 0],
        joint_dof_dim[joint, 1],
        x_wpj,
        joint_qd_internal,
        joint_q_start[joint],
        joint_qd_start[joint],
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
    __syncwarp();
#endif
"""
)
def _sync_reduced_warp(): ...


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
    return result;
#else
    return value;
#endif
"""
)
def _sum_reduced_group_float(
    value: wp.float32,
    width: wp.int32,
    mask: wp.uint32,
) -> wp.float32: ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(mask, value, source_lane, width);
#else
    return value;
#endif
"""
)
def _shuffle_reduced_float(
    value: wp.float32,
    source_lane: wp.int32,
    width: wp.int32,
    mask: wp.uint32,
) -> wp.float32: ...


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


@wp.kernel(enable_backward=False, module="reduced_factor")
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
    thread = wp.tid()
    articulation = thread // wp.int32(32)
    lane = thread - articulation * wp.int32(32)

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
            reduced = inertia
            if dof_count == wp.int32(1):
                dof = dof_start
                u = inertia * joint_s[dof]
                joint_u[dof] = u
                d_inv_scalar = wp.float32(1.0) / wp.max(
                    wp.dot(joint_s[dof], u) + factor_diagonal[dof], wp.float32(1.0e-20)
                )
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
            index += wp.int32(32)
        _sync_reduced_warp()


@wp.func
def _articulation_system_com_local(
    start: wp.int32,
    end: wp.int32,
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_q_com: wp.array[wp.transform],
) -> wp.vec3:
    weighted_position = wp.vec3(0.0)
    total_mass = wp.float32(0.0)
    for joint in range(start, end):
        body = joint_child[joint]
        mass = body_mass[body]
        weighted_position += mass * wp.transform_get_translation(body_q_com[body])
        total_mass += mass
    if total_mass > wp.float32(0.0):
        return weighted_position / total_mass
    return wp.vec3(0.0)


@wp.func
def _articulation_momentum_about_local_origin(
    start: wp.int32,
    end: wp.int32,
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    body_q_com: wp.array[wp.transform],
    origin: wp.vec3,
    bodies: BodyContainer,
) -> wp.spatial_vector:
    linear_momentum = wp.vec3(0.0)
    angular_momentum = wp.vec3(0.0)
    for joint in range(start, end):
        body = joint_child[joint]
        slot = body + wp.int32(1)
        mass = body_mass[body]
        velocity = bodies.velocity[slot]
        omega = bodies.angular_velocity[slot]
        rotation = bodies.orientation[slot]
        body_omega = wp.quat_rotate_inv(rotation, omega)
        spin = wp.quat_rotate(rotation, body_inertia[body] * body_omega)
        momentum = mass * velocity
        position = wp.transform_get_translation(body_q_com[body])
        linear_momentum += momentum
        angular_momentum += spin + wp.cross(position - origin, momentum)
    return wp.spatial_vector(linear_momentum, angular_momentum)


@wp.kernel(enable_backward=False)
def _capture_reduced_momentum_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    bodies: BodyContainer,
    momentum: wp.array[wp.spatial_vector],
):
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    origin = _articulation_system_com_local(start, end, joint_child, body_mass, bodies.reduced.body_q_com)
    momentum[articulation] = _articulation_momentum_about_local_origin(
        start,
        end,
        joint_child,
        body_mass,
        body_inertia,
        bodies.reduced.body_q_com,
        origin,
        bodies,
    )


@wp.kernel(enable_backward=False, module="reduced_momentum_capture")
def _capture_reduced_momentum_warp_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    bodies: BodyContainer,
    momentum_out: wp.array[wp.spatial_vector],
):
    articulation, lane = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    body_q_com = bodies.reduced.body_q_com

    local_mass = wp.float32(0.0)
    local_weighted_x = wp.float32(0.0)
    local_weighted_y = wp.float32(0.0)
    local_weighted_z = wp.float32(0.0)
    joint = start + lane
    while joint < end:
        body = joint_child[joint]
        mass = body_mass[body]
        position = wp.transform_get_translation(body_q_com[body])
        local_mass += mass
        local_weighted_x += mass * position[0]
        local_weighted_y += mass * position[1]
        local_weighted_z += mass * position[2]
        joint += wp.int32(32)

    total_mass = wp.tile_sum(wp.tile(local_mass))[0]
    weighted_position = wp.vec3(
        wp.tile_sum(wp.tile(local_weighted_x))[0],
        wp.tile_sum(wp.tile(local_weighted_y))[0],
        wp.tile_sum(wp.tile(local_weighted_z))[0],
    )
    origin = wp.vec3(0.0)
    if total_mass > wp.float32(0.0):
        origin = weighted_position / total_mass

    local_linear_x = wp.float32(0.0)
    local_linear_y = wp.float32(0.0)
    local_linear_z = wp.float32(0.0)
    local_angular_x = wp.float32(0.0)
    local_angular_y = wp.float32(0.0)
    local_angular_z = wp.float32(0.0)
    joint = start + lane
    while joint < end:
        body = joint_child[joint]
        slot = body + wp.int32(1)
        mass = body_mass[body]
        velocity = bodies.velocity[slot]
        omega = bodies.angular_velocity[slot]
        q_com = body_q_com[body]
        position = wp.transform_get_translation(q_com)
        rotation = wp.transform_get_rotation(q_com)
        body_omega = wp.quat_rotate_inv(rotation, omega)
        spin = wp.quat_rotate(rotation, body_inertia[body] * body_omega)
        linear = mass * velocity
        angular = spin + wp.cross(position - origin, linear)
        local_linear_x += linear[0]
        local_linear_y += linear[1]
        local_linear_z += linear[2]
        local_angular_x += angular[0]
        local_angular_y += angular[1]
        local_angular_z += angular[2]
        joint += wp.int32(32)

    momentum = wp.spatial_vector(
        wp.tile_sum(wp.tile(local_linear_x))[0],
        wp.tile_sum(wp.tile(local_linear_y))[0],
        wp.tile_sum(wp.tile(local_linear_z))[0],
        wp.tile_sum(wp.tile(local_angular_x))[0],
        wp.tile_sum(wp.tile(local_angular_y))[0],
        wp.tile_sum(wp.tile(local_angular_z))[0],
    )
    if lane == wp.int32(0):
        momentum_out[articulation] = momentum


@wp.kernel(enable_backward=False)
def _restore_reduced_momentum_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    target_momentum: wp.array[wp.spatial_vector],
    joint_s_projection: wp.array[wp.spatial_vector],
    bodies: BodyContainer,
    joint_qd_internal: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
    body_qd_public: wp.array[wp.spatial_vector],
):
    """Restore momentum through the floating-base rigid-motion subspace."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    root_type = joint_type[start]
    if root_type != JointType.FREE and root_type != JointType.DISTANCE:
        return

    body_q_com = bodies.reduced.body_q_com
    origin = _articulation_system_com_local(start, end, joint_child, body_mass, body_q_com)
    current = _articulation_momentum_about_local_origin(
        start,
        end,
        joint_child,
        body_mass,
        body_inertia,
        body_q_com,
        origin,
        bodies,
    )
    residual = target_momentum[articulation] - current

    # At the system COM, rigid translation and rotation decouple. Assemble
    # the ordinary physical inertia directly so spatial-coordinate translation
    # conventions cannot leak into the correction.
    total_mass = wp.float32(0.0)
    inertia_com = wp.mat33(0.0)
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for joint in range(start, end):
        body = joint_child[joint]
        mass = body_mass[body]
        q_com = body_q_com[body]
        rotation = wp.quat_to_matrix(wp.transform_get_rotation(q_com))
        offset = wp.transform_get_translation(q_com) - origin
        inertia_com += rotation * body_inertia[body] * wp.transpose(rotation)
        inertia_com += mass * (wp.dot(offset, offset) * identity - wp.outer(offset, offset))
        total_mass += mass
    delta_v_origin = wp.spatial_top(residual) / total_mass
    delta_omega = wp.inverse(inertia_com) * wp.spatial_bottom(residual)

    root = joint_child[start]
    for joint in range(start, end):
        body = joint_child[joint]
        slot = body + wp.int32(1)
        position = wp.transform_get_translation(body_q_com[body])
        delta_v = delta_v_origin + wp.cross(delta_omega, position - origin)
        bodies.velocity[slot] += delta_v
        bodies.angular_velocity[slot] += delta_omega
        body_qd_public[body] = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])

    # Keep the authoritative reduced velocity consistent with the physical
    # body correction. The floating-root subspace is six-dimensional but may
    # be rotated/shifted, so solve its small Gram system instead of assuming
    # canonical component ordering.
    dof_start = joint_qd_start[start]
    dof_end = joint_qd_start[start + wp.int32(1)]
    dof_count = dof_end - dof_start
    current_root_twist = wp.spatial_vector()
    for dof in range(dof_start, dof_end):
        current_root_twist += joint_s_projection[dof] * joint_qd_internal[dof]
    root_slot = root + wp.int32(1)
    corrected_root_twist = wp.spatial_vector(bodies.velocity[root_slot], bodies.angular_velocity[root_slot])
    root_twist_delta = corrected_root_twist - current_root_twist
    gram = _mat66(0.0)
    projected = _vec6(0.0)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            s_row = joint_s_projection[dof_start + wp.int32(row)]
            projected[row] = wp.dot(s_row, root_twist_delta)
            for column in range(_MAX_JOINT_DOF):
                if wp.int32(column) < dof_count:
                    gram[row, column] = wp.dot(s_row, joint_s_projection[dof_start + wp.int32(column)])
    inverse_gram = _invert_spd(gram, dof_count)
    for row in range(_MAX_JOINT_DOF):
        if wp.int32(row) < dof_count:
            delta_qd = wp.float32(0.0)
            for column in range(_MAX_JOINT_DOF):
                if wp.int32(column) < dof_count:
                    delta_qd += inverse_gram[row, column] * projected[column]
            joint_qd_internal[dof_start + wp.int32(row)] += delta_qd

    # Newton exposes FREE/DISTANCE linear speeds at the child COM in the
    # parent-joint frame, not in the world-origin integrator basis.
    q_parent = wp.transform_get_rotation(joint_x_p[start])
    delta_v_joint = wp.quat_rotate_inv(q_parent, wp.spatial_top(root_twist_delta))
    delta_omega_joint = wp.quat_rotate_inv(q_parent, wp.spatial_bottom(root_twist_delta))
    joint_qd_public[dof_start] += delta_v_joint[0]
    joint_qd_public[dof_start + wp.int32(1)] += delta_v_joint[1]
    joint_qd_public[dof_start + wp.int32(2)] += delta_v_joint[2]
    joint_qd_public[dof_start + wp.int32(3)] += delta_omega_joint[0]
    joint_qd_public[dof_start + wp.int32(4)] += delta_omega_joint[1]
    joint_qd_public[dof_start + wp.int32(5)] += delta_omega_joint[2]


@wp.kernel(enable_backward=False, module="reduced_momentum")
def _restore_reduced_momentum_warp_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    target_momentum: wp.array[wp.spatial_vector],
    joint_s_projection: wp.array[wp.spatial_vector],
    bodies: BodyContainer,
    joint_qd_internal: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
    body_qd_public: wp.array[wp.spatial_vector],
):
    # Restore floating-base momentum with one deterministic warp per tree.
    articulation, lane = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    root_type = joint_type[start]
    if root_type != JointType.FREE and root_type != JointType.DISTANCE:
        return

    body_q_com = bodies.reduced.body_q_com
    local_mass = wp.float32(0.0)
    local_weighted_x = wp.float32(0.0)
    local_weighted_y = wp.float32(0.0)
    local_weighted_z = wp.float32(0.0)
    joint = start + lane
    while joint < end:
        body = joint_child[joint]
        mass = body_mass[body]
        position = wp.transform_get_translation(body_q_com[body])
        local_mass += mass
        local_weighted_x += mass * position[0]
        local_weighted_y += mass * position[1]
        local_weighted_z += mass * position[2]
        joint += wp.int32(32)

    total_mass = wp.tile_sum(wp.tile(local_mass))[0]
    origin = wp.vec3(
        wp.tile_sum(wp.tile(local_weighted_x))[0] / total_mass,
        wp.tile_sum(wp.tile(local_weighted_y))[0] / total_mass,
        wp.tile_sum(wp.tile(local_weighted_z))[0] / total_mass,
    )

    local_linear_x = wp.float32(0.0)
    local_linear_y = wp.float32(0.0)
    local_linear_z = wp.float32(0.0)
    local_angular_x = wp.float32(0.0)
    local_angular_y = wp.float32(0.0)
    local_angular_z = wp.float32(0.0)
    local_inertia_xx = wp.float32(0.0)
    local_inertia_xy = wp.float32(0.0)
    local_inertia_xz = wp.float32(0.0)
    local_inertia_yy = wp.float32(0.0)
    local_inertia_yz = wp.float32(0.0)
    local_inertia_zz = wp.float32(0.0)
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    joint = start + lane
    while joint < end:
        body = joint_child[joint]
        slot = body + wp.int32(1)
        mass = body_mass[body]
        velocity = bodies.velocity[slot]
        omega = bodies.angular_velocity[slot]
        q_com = body_q_com[body]
        position = wp.transform_get_translation(q_com)
        rotation_quat = wp.transform_get_rotation(q_com)
        body_omega = wp.quat_rotate_inv(rotation_quat, omega)
        spin = wp.quat_rotate(rotation_quat, body_inertia[body] * body_omega)
        momentum = mass * velocity
        angular = spin + wp.cross(position - origin, momentum)
        local_linear_x += momentum[0]
        local_linear_y += momentum[1]
        local_linear_z += momentum[2]
        local_angular_x += angular[0]
        local_angular_y += angular[1]
        local_angular_z += angular[2]

        rotation = wp.quat_to_matrix(rotation_quat)
        offset = position - origin
        inertia = rotation * body_inertia[body] * wp.transpose(rotation)
        inertia += mass * (wp.dot(offset, offset) * identity - wp.outer(offset, offset))
        local_inertia_xx += inertia[0, 0]
        local_inertia_xy += inertia[0, 1]
        local_inertia_xz += inertia[0, 2]
        local_inertia_yy += inertia[1, 1]
        local_inertia_yz += inertia[1, 2]
        local_inertia_zz += inertia[2, 2]
        joint += wp.int32(32)

    current = wp.spatial_vector(
        wp.tile_sum(wp.tile(local_linear_x))[0],
        wp.tile_sum(wp.tile(local_linear_y))[0],
        wp.tile_sum(wp.tile(local_linear_z))[0],
        wp.tile_sum(wp.tile(local_angular_x))[0],
        wp.tile_sum(wp.tile(local_angular_y))[0],
        wp.tile_sum(wp.tile(local_angular_z))[0],
    )
    residual = target_momentum[articulation] - current
    inertia_xx = wp.tile_sum(wp.tile(local_inertia_xx))[0]
    inertia_xy = wp.tile_sum(wp.tile(local_inertia_xy))[0]
    inertia_xz = wp.tile_sum(wp.tile(local_inertia_xz))[0]
    inertia_yy = wp.tile_sum(wp.tile(local_inertia_yy))[0]
    inertia_yz = wp.tile_sum(wp.tile(local_inertia_yz))[0]
    inertia_zz = wp.tile_sum(wp.tile(local_inertia_zz))[0]
    inertia_com = wp.mat33(
        inertia_xx,
        inertia_xy,
        inertia_xz,
        inertia_xy,
        inertia_yy,
        inertia_yz,
        inertia_xz,
        inertia_yz,
        inertia_zz,
    )
    delta_v_origin = wp.spatial_top(residual) / total_mass
    delta_omega = wp.inverse(inertia_com) * wp.spatial_bottom(residual)

    joint = start + lane
    while joint < end:
        body = joint_child[joint]
        slot = body + wp.int32(1)
        position = wp.transform_get_translation(body_q_com[body])
        delta_v = delta_v_origin + wp.cross(delta_omega, position - origin)
        bodies.velocity[slot] += delta_v
        bodies.angular_velocity[slot] += delta_omega
        body_qd_public[body] = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
        joint += wp.int32(32)
    _sync_reduced_warp()

    if lane == wp.int32(0):
        root = joint_child[start]
        dof_start = joint_qd_start[start]
        dof_end = joint_qd_start[start + wp.int32(1)]
        dof_count = dof_end - dof_start
        current_root_twist = wp.spatial_vector()
        for dof in range(dof_start, dof_end):
            current_root_twist += joint_s_projection[dof] * joint_qd_internal[dof]
        root_slot = root + wp.int32(1)
        corrected_root_twist = wp.spatial_vector(bodies.velocity[root_slot], bodies.angular_velocity[root_slot])
        root_twist_delta = corrected_root_twist - current_root_twist
        gram = _mat66(0.0)
        projected = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                s_row = joint_s_projection[dof_start + wp.int32(row)]
                projected[row] = wp.dot(s_row, root_twist_delta)
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        gram[row, column] = wp.dot(s_row, joint_s_projection[dof_start + wp.int32(column)])
        inverse_gram = _invert_spd(gram, dof_count)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                delta_qd = wp.float32(0.0)
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        delta_qd += inverse_gram[row, column] * projected[column]
                joint_qd_internal[dof_start + wp.int32(row)] += delta_qd

        q_parent = wp.transform_get_rotation(joint_x_p[start])
        delta_v_joint = wp.quat_rotate_inv(q_parent, wp.spatial_top(root_twist_delta))
        delta_omega_joint = wp.quat_rotate_inv(q_parent, wp.spatial_bottom(root_twist_delta))
        joint_qd_public[dof_start] += delta_v_joint[0]
        joint_qd_public[dof_start + wp.int32(1)] += delta_v_joint[1]
        joint_qd_public[dof_start + wp.int32(2)] += delta_v_joint[2]
        joint_qd_public[dof_start + wp.int32(3)] += delta_omega_joint[0]
        joint_qd_public[dof_start + wp.int32(4)] += delta_omega_joint[1]
        joint_qd_public[dof_start + wp.int32(5)] += delta_omega_joint[2]


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
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    bodies: BodyContainer,
):
    """Integrate and publish one articulation without world-origin cancellation."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    data = bodies.reduced

    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        kind = joint_type[joint]
        if kind == JointType.FREE or kind == JointType.DISTANCE:
            slot = child + wp.int32(1)
            child_twist = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
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
            root_twist = wp.spatial_vector()
            for dof in range(dof_start, dof_end):
                root_twist += data.joint_s[dof] * joint_qd_local[dof]
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


@wp.kernel(enable_backward=False, module="reduced_publish")
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
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    target_momentum: wp.array[wp.spatial_vector],
    body_x_com: wp.array[wp.transform],
    zero_generalized_acceleration: wp.array[wp.float32],
    dt: wp.float32,
    update_configuration: wp.bool,
    restore_momentum: wp.bool,
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    bodies: BodyContainer,
    articulation_origin: wp.array[wp.vec3],
    body_q_local: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_anchor_local: wp.array[wp.transform],
    joint_s_publish: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
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
    data = bodies.reduced

    if update_configuration:
        for depth in range(max_depth + wp.int32(1)):
            index = articulation_depth_start[articulation, depth] + lane
            depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
            while index < depth_end:
                joint = articulation_depth_joint[index]
                parent = joint_parent[joint]
                child = joint_child[joint]
                kind = joint_type[joint]
                if kind == JointType.FREE or kind == JointType.DISTANCE:
                    slot = child + wp.int32(1)
                    child_twist = wp.spatial_vector(bodies.velocity[slot], bodies.angular_velocity[slot])
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
                root_twist = wp.spatial_vector()
                for dof in range(dof_start, dof_end):
                    root_twist += data.joint_s[dof] * joint_qd_local[dof]
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
            index += tile_width
        _sync_reduced_group(group_mask)

    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    if restore_momentum and (joint_type[start] == JointType.FREE or joint_type[start] == JointType.DISTANCE):
        local_mass = wp.float32(0.0)
        local_weighted_x = wp.float32(0.0)
        local_weighted_y = wp.float32(0.0)
        local_weighted_z = wp.float32(0.0)
        joint = start + lane
        while joint < end:
            body = joint_child[joint]
            mass = body_mass[body]
            position = wp.transform_get_translation(body_q_com[body])
            local_mass += mass
            local_weighted_x += mass * position[0]
            local_weighted_y += mass * position[1]
            local_weighted_z += mass * position[2]
            joint += tile_width
        total_mass = _sum_reduced_group_float(local_mass, tile_width, group_mask)
        weighted_x = _sum_reduced_group_float(local_weighted_x, tile_width, group_mask)
        weighted_y = _sum_reduced_group_float(local_weighted_y, tile_width, group_mask)
        weighted_z = _sum_reduced_group_float(local_weighted_z, tile_width, group_mask)
        total_mass = _shuffle_reduced_float(total_mass, wp.int32(0), tile_width, group_mask)
        weighted_x = _shuffle_reduced_float(weighted_x, wp.int32(0), tile_width, group_mask)
        weighted_y = _shuffle_reduced_float(weighted_y, wp.int32(0), tile_width, group_mask)
        weighted_z = _shuffle_reduced_float(weighted_z, wp.int32(0), tile_width, group_mask)
        origin = wp.vec3(weighted_x, weighted_y, weighted_z) / total_mass

        local_linear_x = wp.float32(0.0)
        local_linear_y = wp.float32(0.0)
        local_linear_z = wp.float32(0.0)
        local_angular_x = wp.float32(0.0)
        local_angular_y = wp.float32(0.0)
        local_angular_z = wp.float32(0.0)
        local_inertia_xx = wp.float32(0.0)
        local_inertia_xy = wp.float32(0.0)
        local_inertia_xz = wp.float32(0.0)
        local_inertia_yy = wp.float32(0.0)
        local_inertia_yz = wp.float32(0.0)
        local_inertia_zz = wp.float32(0.0)
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        joint = start + lane
        while joint < end:
            body = joint_child[joint]
            mass = body_mass[body]
            q_com = body_q_com[body]
            position = wp.transform_get_translation(q_com)
            rotation_quat = wp.transform_get_rotation(q_com)
            twist = body_qd[body]
            velocity = wp.spatial_top(twist)
            omega = wp.spatial_bottom(twist)
            body_omega = wp.quat_rotate_inv(rotation_quat, omega)
            spin = wp.quat_rotate(rotation_quat, body_inertia[body] * body_omega)
            linear = mass * velocity
            angular = spin + wp.cross(position - origin, linear)
            local_linear_x += linear[0]
            local_linear_y += linear[1]
            local_linear_z += linear[2]
            local_angular_x += angular[0]
            local_angular_y += angular[1]
            local_angular_z += angular[2]
            rotation_matrix = wp.quat_to_matrix(rotation_quat)
            offset = position - origin
            inertia = rotation_matrix * body_inertia[body] * wp.transpose(rotation_matrix)
            inertia += mass * (wp.dot(offset, offset) * identity - wp.outer(offset, offset))
            local_inertia_xx += inertia[0, 0]
            local_inertia_xy += inertia[0, 1]
            local_inertia_xz += inertia[0, 2]
            local_inertia_yy += inertia[1, 1]
            local_inertia_yz += inertia[1, 2]
            local_inertia_zz += inertia[2, 2]
            joint += tile_width

        linear_x = _sum_reduced_group_float(local_linear_x, tile_width, group_mask)
        linear_y = _sum_reduced_group_float(local_linear_y, tile_width, group_mask)
        linear_z = _sum_reduced_group_float(local_linear_z, tile_width, group_mask)
        angular_x = _sum_reduced_group_float(local_angular_x, tile_width, group_mask)
        angular_y = _sum_reduced_group_float(local_angular_y, tile_width, group_mask)
        angular_z = _sum_reduced_group_float(local_angular_z, tile_width, group_mask)
        inertia_xx = _sum_reduced_group_float(local_inertia_xx, tile_width, group_mask)
        inertia_xy = _sum_reduced_group_float(local_inertia_xy, tile_width, group_mask)
        inertia_xz = _sum_reduced_group_float(local_inertia_xz, tile_width, group_mask)
        inertia_yy = _sum_reduced_group_float(local_inertia_yy, tile_width, group_mask)
        inertia_yz = _sum_reduced_group_float(local_inertia_yz, tile_width, group_mask)
        inertia_zz = _sum_reduced_group_float(local_inertia_zz, tile_width, group_mask)
        linear_x = _shuffle_reduced_float(linear_x, wp.int32(0), tile_width, group_mask)
        linear_y = _shuffle_reduced_float(linear_y, wp.int32(0), tile_width, group_mask)
        linear_z = _shuffle_reduced_float(linear_z, wp.int32(0), tile_width, group_mask)
        angular_x = _shuffle_reduced_float(angular_x, wp.int32(0), tile_width, group_mask)
        angular_y = _shuffle_reduced_float(angular_y, wp.int32(0), tile_width, group_mask)
        angular_z = _shuffle_reduced_float(angular_z, wp.int32(0), tile_width, group_mask)
        inertia_xx = _shuffle_reduced_float(inertia_xx, wp.int32(0), tile_width, group_mask)
        inertia_xy = _shuffle_reduced_float(inertia_xy, wp.int32(0), tile_width, group_mask)
        inertia_xz = _shuffle_reduced_float(inertia_xz, wp.int32(0), tile_width, group_mask)
        inertia_yy = _shuffle_reduced_float(inertia_yy, wp.int32(0), tile_width, group_mask)
        inertia_yz = _shuffle_reduced_float(inertia_yz, wp.int32(0), tile_width, group_mask)
        inertia_zz = _shuffle_reduced_float(inertia_zz, wp.int32(0), tile_width, group_mask)
        current = wp.spatial_vector(linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
        residual = target_momentum[articulation] - current
        inertia_com = wp.mat33(
            inertia_xx,
            inertia_xy,
            inertia_xz,
            inertia_xy,
            inertia_yy,
            inertia_yz,
            inertia_xz,
            inertia_yz,
            inertia_zz,
        )
        delta_v_origin = wp.spatial_top(residual) / total_mass
        delta_omega = wp.inverse(inertia_com) * wp.spatial_bottom(residual)

        joint = start + lane
        while joint < end:
            body = joint_child[joint]
            position = wp.transform_get_translation(body_q_com[body])
            delta_v = delta_v_origin + wp.cross(delta_omega, position - origin)
            corrected = body_qd[body] + wp.spatial_vector(delta_v, delta_omega)
            body_qd[body] = corrected
            slot = body + wp.int32(1)
            bodies.velocity[slot] = wp.spatial_top(corrected)
            bodies.angular_velocity[slot] = wp.spatial_bottom(corrected)
            joint += tile_width
        _sync_reduced_group(group_mask)

        if lane == wp.int32(0):
            root = joint_child[start]
            dof_start = joint_qd_start[start]
            dof_end = joint_qd_start[start + wp.int32(1)]
            dof_count = dof_end - dof_start
            current_root_twist = wp.spatial_vector()
            for dof in range(dof_start, dof_end):
                current_root_twist += joint_s_publish[dof] * joint_qd_local[dof]
            corrected_root_twist = body_qd[root]
            root_twist_delta = corrected_root_twist - current_root_twist
            gram = _mat66(0.0)
            projected = _vec6(0.0)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    s_row = joint_s_publish[dof_start + wp.int32(row)]
                    projected[row] = wp.dot(s_row, root_twist_delta)
                    for column in range(_MAX_JOINT_DOF):
                        if wp.int32(column) < dof_count:
                            gram[row, column] = wp.dot(s_row, joint_s_publish[dof_start + wp.int32(column)])
            inverse_gram = _invert_spd(gram, dof_count)
            for row in range(_MAX_JOINT_DOF):
                if wp.int32(row) < dof_count:
                    delta_qd = wp.float32(0.0)
                    for column in range(_MAX_JOINT_DOF):
                        if wp.int32(column) < dof_count:
                            delta_qd += inverse_gram[row, column] * projected[column]
                    joint_qd_local[dof_start + wp.int32(row)] += delta_qd
            q_parent = wp.transform_get_rotation(joint_x_p[start])
            delta_v_joint = wp.quat_rotate_inv(q_parent, wp.spatial_top(root_twist_delta))
            delta_omega_joint = wp.quat_rotate_inv(q_parent, wp.spatial_bottom(root_twist_delta))
            joint_qd_public[dof_start] += delta_v_joint[0]
            joint_qd_public[dof_start + wp.int32(1)] += delta_v_joint[1]
            joint_qd_public[dof_start + wp.int32(2)] += delta_v_joint[2]
            joint_qd_public[dof_start + wp.int32(3)] += delta_omega_joint[0]
            joint_qd_public[dof_start + wp.int32(4)] += delta_omega_joint[1]
            joint_qd_public[dof_start + wp.int32(5)] += delta_omega_joint[2]


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
):
    """Advance unconstrained ABA dynamics and publish all link velocities."""
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
                joint_qd[dof] += qdd[row] * dt
                child_acceleration += joint_s[dof] * qdd[row]
        body_acceleration[child] = child_acceleration

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
        body_velocity[child] = twist
        omega = wp.spatial_bottom(twist)
        x_com = wp.transform_get_translation(body_q_com[child])
        velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
        public_body_qd[child] = wp.spatial_vector(velocity_com, omega)
        slot = child + wp.int32(1)
        bodies.velocity[slot] = velocity_com
        bodies.angular_velocity[slot] = omega


@wp.kernel(enable_backward=False, module="reduced_advance")
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
    include_external: wp.bool,
    include_coriolis: wp.bool,
    capture_momentum: wp.bool,
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
    captured_momentum: wp.array[wp.spatial_vector],
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

    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    momentum_origin = wp.vec3(0.0)
    if capture_momentum:
        local_mass = wp.float32(0.0)
        local_weighted_x = wp.float32(0.0)
        local_weighted_y = wp.float32(0.0)
        local_weighted_z = wp.float32(0.0)
        joint_index = start + lane
        while joint_index < end:
            body = joint_child[joint_index]
            mass = body_mass[body]
            position = wp.transform_get_translation(body_q_com[body])
            local_mass += mass
            local_weighted_x += mass * position[0]
            local_weighted_y += mass * position[1]
            local_weighted_z += mass * position[2]
            joint_index += tile_width
        total_mass = _sum_reduced_group_float(local_mass, tile_width, group_mask)
        weighted_x = _sum_reduced_group_float(local_weighted_x, tile_width, group_mask)
        weighted_y = _sum_reduced_group_float(local_weighted_y, tile_width, group_mask)
        weighted_z = _sum_reduced_group_float(local_weighted_z, tile_width, group_mask)
        total_mass = _shuffle_reduced_float(total_mass, wp.int32(0), tile_width, group_mask)
        weighted_x = _shuffle_reduced_float(weighted_x, wp.int32(0), tile_width, group_mask)
        weighted_y = _shuffle_reduced_float(weighted_y, wp.int32(0), tile_width, group_mask)
        weighted_z = _shuffle_reduced_float(weighted_z, wp.int32(0), tile_width, group_mask)
        if total_mass > wp.float32(0.0):
            momentum_origin = wp.vec3(weighted_x, weighted_y, weighted_z) / total_mass

    local_momentum_x = wp.float32(0.0)
    local_momentum_y = wp.float32(0.0)
    local_momentum_z = wp.float32(0.0)
    local_angular_x = wp.float32(0.0)
    local_angular_y = wp.float32(0.0)
    local_angular_z = wp.float32(0.0)
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
            spatial_momentum = wp.spatial_vector()
            if include_coriolis or capture_momentum:
                spatial_momentum = inertia * velocity
            if include_coriolis:
                bias += inertia * coriolis + spatial_cross_dual(velocity, spatial_momentum)
            if capture_momentum:
                linear = wp.spatial_top(spatial_momentum)
                angular = wp.spatial_bottom(spatial_momentum) - wp.cross(momentum_origin, linear)
                local_momentum_x += linear[0]
                local_momentum_y += linear[1]
                local_momentum_z += linear[2]
                local_angular_x += angular[0]
                local_angular_y += angular[1]
                local_angular_z += angular[2]
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
        previous_velocity = velocity
        previous_coriolis = coriolis
        _sync_reduced_group(group_mask)

    if capture_momentum:
        linear_x = _sum_reduced_group_float(local_momentum_x, tile_width, group_mask)
        linear_y = _sum_reduced_group_float(local_momentum_y, tile_width, group_mask)
        linear_z = _sum_reduced_group_float(local_momentum_z, tile_width, group_mask)
        angular_x = _sum_reduced_group_float(local_angular_x, tile_width, group_mask)
        angular_y = _sum_reduced_group_float(local_angular_y, tile_width, group_mask)
        angular_z = _sum_reduced_group_float(local_angular_z, tile_width, group_mask)
        if lane == wp.int32(0):
            captured_momentum[articulation] = wp.spatial_vector(
                linear_x, linear_y, linear_z, angular_x, angular_y, angular_z
            )

    # Bias and articulated-force propagation only depend on the completed child
    # depth, so one reverse traversal can advance both recurrences.
    for reverse_depth in range(max_depth + wp.int32(1)):
        depth = max_depth - reverse_depth
        index = articulation_depth_start[articulation, depth] + lane
        depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = articulation_depth_joint[index]
            child = joint_child[joint]
            subtree_bias = body_bias[child]
            p = wp.spatial_vector()
            for child_index in range(child_start[joint], child_start[joint + wp.int32(1)]):
                descendant_joint = child_joint[child_index]
                descendant = joint_child[descendant_joint]
                subtree_bias += body_bias[descendant]
                p += body_work[descendant]
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
                    if include_external and joint_kind != JointType.FREE and joint_kind != JointType.DISTANCE:
                        applied = joint_f[dof]
                    if include_external and (
                        joint_kind == JointType.REVOLUTE
                        or joint_kind == JointType.PRISMATIC
                        or joint_kind == JointType.D6
                    ):
                        applied += joint_implicit_force[dof]
                    dof_rhs = applied - wp.dot(joint_s[dof], subtree_bias)
                    generalized_rhs[dof] = dof_rhs
                    reduced_force[row] = dof_rhs - wp.dot(joint_s[dof], p)
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
            body_bias[child] = subtree_bias
            body_work[child] = propagated
            index += tile_width
        _sync_reduced_group(group_mask)

    for depth in range(max_depth + wp.int32(1)):
        index = articulation_depth_start[articulation, depth] + lane
        depth_end = articulation_depth_start[articulation, depth + wp.int32(1)]
        while index < depth_end:
            joint = articulation_depth_joint[index]
            parent = joint_parent[joint]
            child = joint_child[joint]
            dof_start = joint_qd_start[joint]
            dof_end = joint_qd_start[joint + wp.int32(1)]
            dof_count = dof_end - dof_start
            parent_acceleration = wp.spatial_vector()
            twist = wp.spatial_vector()
            if parent >= wp.int32(0):
                parent_acceleration = body_acceleration[parent]
                twist = body_velocity[parent]
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
                    joint_qd[dof] += qdd[row] * dt
                    child_acceleration += joint_s[dof] * qdd[row]
                    twist += joint_s[dof] * joint_qd[dof]
            body_acceleration[child] = child_acceleration
            body_velocity[child] = twist
            omega = wp.spatial_bottom(twist)
            x_com = wp.transform_get_translation(body_q_com[child])
            velocity_com = wp.spatial_top(twist) + wp.cross(omega, x_com)
            public_body_qd[child] = wp.spatial_vector(velocity_com, omega)
            slot = child + wp.int32(1)
            bodies.velocity[slot] = velocity_com
            bodies.angular_velocity[slot] = omega
            index += tile_width
        _sync_reduced_group(group_mask)


@wp.func
def _flush_deferred_articulation(bodies: BodyContainer, articulation: wp.int32):
    data = bodies.reduced
    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    for joint in range(start, end):
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        parent_delta = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_delta = data.body_acceleration[parent]
        wrench = data.deferred_wrench[child]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        rhs = _vec6(0.0)
        response = _vec6(0.0)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = wp.dot(data.joint_s[dof], wrench) - wp.dot(data.joint_u[dof], parent_delta)
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                for column in range(_MAX_JOINT_DOF):
                    if wp.int32(column) < dof_count:
                        response[row] += data.joint_d_inv[dof_start + wp.int32(row), column] * rhs[column]
        child_delta = parent_delta
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                data.joint_qd[dof] += response[row]
                child_delta += data.joint_s[dof] * response[row]
        data.body_acceleration[child] = child_delta
        slot = child + wp.int32(1)
        delta_omega = wp.spatial_bottom(child_delta)
        bodies.angular_velocity[slot] = bodies.angular_velocity[slot] + delta_omega
        local_com_position = wp.transform_get_translation(data.body_q_com[child])
        bodies.velocity[slot] = (
            bodies.velocity[slot] + wp.spatial_top(child_delta) + wp.cross(delta_omega, local_com_position)
        )
        data.deferred_wrench[child] = wp.spatial_vector()


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
                contact_cols, column, bodies, idt, cc, contacts, wp.bool(True), wp.bool(True), wp.bool(True)
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
            )

    _flush_deferred_articulation(bodies, articulation)


class ReducedArticulationSystem:
    """Cached linear-time reduced-coordinate factorization for a Newton model."""

    def __init__(self, model: Model):
        if model.articulation_count <= 0:
            raise ValueError("reduced articulation system requires at least one articulation")
        self.model = model
        self.device = model.device
        self.use_warp_advance = bool(self.device.is_cuda)
        self.use_warp_factor = bool(self.device.is_cuda and model.articulation_count >= _WARP_FACTOR_MIN_ARTICULATIONS)
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
        joint_parent_lane_np = np.zeros(joint_count, dtype=np.int32)
        for joint in tree_joint_np:
            parent = int(joint_parent_np[joint])
            if parent >= 0:
                joint_parent_lane_np[joint] = joint_depth_lane_np[body_joint_np[parent]]
        self.advance_joint_parent_lane = wp.array(joint_parent_lane_np, device=self.device)
        max_articulation_breadth = int(np.max(np.diff(articulation_depth_start_np, axis=1)))
        if max_articulation_breadth <= 8:
            self.advance_tile_width = 8
        elif max_articulation_breadth <= 16:
            self.advance_tile_width = 16
        else:
            self.advance_tile_width = 32
        self.factor_child_start = wp.array(child_start_np, device=self.device)
        self.factor_child_joint = wp.array(np.asarray(child_joint_list, dtype=np.int32), device=self.device)

        self.body_x_com = wp.empty(body_count, dtype=wp.transform, device=self.device)
        self.body_q_com = wp.empty(body_count, dtype=wp.transform, device=self.device)
        self.body_q_local = wp.empty(body_count, dtype=wp.transform, device=self.device)
        self.joint_anchor_local = wp.empty(joint_count, dtype=wp.transform, device=self.device)
        self.articulation_origin = wp.empty(int(model.articulation_count), dtype=wp.vec3, device=self.device)
        self.body_i_s = wp.empty(body_count, dtype=_sym_mat66, device=self.device)
        self.joint_s = wp.empty(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_s_publish = wp.empty(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_qd_internal = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.joint_qd_integrator = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.joint_factor_diagonal = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.joint_implicit_force = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.reduced_inertia = wp.empty(body_count, dtype=_sym_mat66, device=self.device)
        self.joint_u_matrix = wp.empty(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_d_inv = wp.zeros((max(1, dof_count), 6), dtype=wp.float32, device=self.device)
        self.body_force = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_velocity = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_coriolis = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_bias = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_work = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_work = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.body_acceleration = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.generalized_acceleration = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.zero_generalized_force = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.generalized_rhs = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.impulse_response = wp.zeros((body_count, 6), dtype=wp.spatial_vector, device=self.device)
        self.deferred_wrench = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.target_world_momentum = wp.zeros(
            int(model.articulation_count),
            dtype=wp.spatial_vector,
            device=self.device,
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

    def factor(self, state: State, control: Control | None = None, dt: float = 0.0) -> None:
        """Factor the generalized mass and optional implicit joint operator."""
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
        if self.use_warp_factor:
            wp.launch(
                _factor_reduced_warp_kernel,
                dim=int(self.model.articulation_count) * 32,
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
        self.factor(state)
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

    def advance(
        self,
        state: State,
        control: Control,
        bodies: BodyContainer,
        dt: float,
        *,
        include_external: bool,
        include_coriolis: bool,
        capture_momentum: bool = False,
    ) -> None:
        """Apply one split ABA velocity update and publish link twists."""
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
            wp.bool(include_external),
            wp.bool(include_coriolis),
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
        ]
        if self.use_warp_advance:
            articulation_count = int(self.model.articulation_count)
            tile_width = self.advance_tile_width
            thread_count = ((articulation_count * tile_width + 31) // 32) * 32
            wp.launch(
                _advance_reduced_articulations_warp_kernel,
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
                    wp.bool(capture_momentum),
                ],
                outputs=[*advance_outputs, self.target_world_momentum],
                device=self.device,
            )
        else:
            wp.launch(
                _advance_reduced_articulations_kernel,
                dim=int(self.model.articulation_count),
                inputs=[
                    self.model.articulation_start,
                    self.model.articulation_end,
                    *advance_inputs,
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


class ReducedPhoenXArticulation:
    """Graph-stable bridge between common Newton articulation state and PhoenX bodies."""

    def __init__(self, model: Model, bodies: BodyContainer):
        self.model = model
        self.bodies = bodies
        self.system = ReducedArticulationSystem(model)
        self.state = model.state()
        self.control = model.control(clone_variables=True)

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
        reduced.body_joint = wp.array(body_joint_np, device=model.device)
        reduced.body_path_start = wp.array(body_path_start_np, device=model.device)
        reduced.body_path_joint = wp.array(
            np.asarray(body_path_joint_list, dtype=np.int32),
            device=model.device,
        )
        bodies.reduced = reduced
        body_is_reduced.assign(body_is_reduced_np)
        tree_joint_mask.assign(tree_joint_mask_np)
        self.body_is_reduced = body_is_reduced
        self.tree_joint_mask = tree_joint_mask
        self.body_is_reduced_np = body_is_reduced_np.astype(bool)
        self.tree_joint_mask_np = tree_joint_mask_np.astype(bool)
        self.loop_system = ReducedLoopSystem(model, self.body_is_reduced_np)
        self.contact_block_system = ReducedContactBlockSystem(
            model,
            articulation_depth_start=self.system.advance_articulation_depth_start,
            articulation_depth_joint=self.system.advance_articulation_depth_joint,
            max_depth=self.system.advance_max_depth,
        )
        self.owned_joint_mask_np = self.tree_joint_mask_np.copy()
        self.owned_joint_mask_np[self.loop_system.joint_indices_np] = True

        wp.launch(
            _mark_articulated_bodies_kernel,
            dim=int(model.body_count),
            inputs=[self.body_is_reduced],
            outputs=[bodies],
            device=model.device,
        )

    def import_step(
        self,
        state: State,
        control: Control,
        *,
        state_is_continuation: bool = False,
    ) -> None:
        """Import common generalized state and controls into stable graph buffers."""
        if not state_is_continuation:
            wp.copy(self.state.joint_q, state.joint_q)
            wp.copy(self.state.joint_qd, state.joint_qd)
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
    ) -> None:
        """Apply pre-PGS reduced dynamics at the current configuration."""
        self.system.factor(self.state, self.control, dt)
        self.system.advance(
            self.state,
            self.control,
            self.bodies,
            dt,
            include_external=True,
            include_coriolis=not split_dynamics,
        )
        if compute_impulse_response:
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
        restore_momentum: bool = False,
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
                    self.model.body_mass,
                    self.model.body_inertia,
                    self.system.target_world_momentum,
                    self.system.body_x_com,
                    self.system.zero_generalized_force,
                    wp.float32(dt),
                    wp.bool(update_configuration),
                    wp.bool(restore_momentum),
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

    def finish_relax(self) -> None:
        """Publish bias-free velocity corrections without advancing configuration."""
        # Relax changes only velocities, so the pose-derived buffers from the
        # preceding full publication remain current.
        self._publish_state(0.0, update_configuration=False)

    def end_substep(self, dt: float, *, split_dynamics: bool) -> None:
        """Apply post-impulse Coriolis dynamics, integrate, and conserve momentum."""
        capture_inputs = [
            self.model.articulation_start,
            self.model.articulation_end,
            self.model.joint_child,
            self.model.body_mass,
            self.model.body_inertia,
            self.bodies,
        ]
        fused_capture = self.model.device.is_cuda and self.system.use_warp_advance and split_dynamics
        if not fused_capture:
            if self.model.device.is_cuda:
                wp.launch_tiled(
                    _capture_reduced_momentum_warp_kernel,
                    dim=[int(self.model.articulation_count)],
                    inputs=capture_inputs,
                    outputs=[self.system.target_world_momentum],
                    block_dim=32,
                    device=self.model.device,
                )
            else:
                wp.launch(
                    _capture_reduced_momentum_kernel,
                    dim=int(self.model.articulation_count),
                    inputs=capture_inputs,
                    outputs=[self.system.target_world_momentum],
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
                capture_momentum=fused_capture,
            )

        fused_restore = self.model.device.is_cuda and self.system.use_warp_publish
        self._publish_state(dt, restore_momentum=fused_restore)
        if fused_restore:
            return
        restore_inputs = [
            self.model.articulation_start,
            self.model.articulation_end,
            self.model.joint_type,
            self.model.joint_child,
            self.model.joint_qd_start,
            self.model.joint_X_p,
            self.model.body_mass,
            self.model.body_inertia,
            self.system.target_world_momentum,
            self.system.joint_s_publish,
            self.bodies,
        ]
        restore_outputs = [self.system.joint_qd_internal, self.state.joint_qd, self.state.body_qd]
        if self.model.device.is_cuda:
            wp.launch_tiled(
                _restore_reduced_momentum_warp_kernel,
                dim=[int(self.model.articulation_count)],
                inputs=restore_inputs,
                outputs=restore_outputs,
                block_dim=32,
                device=self.model.device,
            )
        else:
            wp.launch(
                _restore_reduced_momentum_kernel,
                dim=int(self.model.articulation_count),
                inputs=restore_inputs,
                outputs=restore_outputs,
                device=self.model.device,
            )

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
        iterations = world.velocity_iterations if relax else world.solver_iterations
        if iterations <= 0:
            return
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
        contact_views = world._active_contact_views()
        common_inputs = [
            world._contact_cols,
            world.bodies,
            idt,
            wp.float32(world.sor_boost),
        ]
        prepare = not relax and world._refresh_prepare_this_substep()
        common_tail = [
            world._contact_container,
            contact_views,
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
        wp.copy(state.joint_q, self.state.joint_q)
        wp.copy(state.joint_qd, self.state.joint_qd)


__all__ = ["ReducedArticulationSystem", "ReducedPhoenXArticulation"]
