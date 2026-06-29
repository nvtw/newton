# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear-time articulated-body factorization for PhoenX."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Control, JointType, Model, State
from newton._src.sim.articulation import eval_fk
from newton._src.solvers.featherstone.kernels import (
    accumulate_free_distance_joint_f_to_body_force,
    compute_com_transforms,
    compute_spatial_inertia,
    convert_body_force_com_to_origin,
    eval_single_articulation_fk_with_velocity_conversion,
    jcalc_integrate,
    jcalc_motion,
    spatial_cross,
    spatial_cross_dual,
    transform_spatial_inertia,
)
from newton._src.solvers.phoenx.articulations.reduced_contact import reduced_contact_iterate, reduced_contact_prepare
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
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)


@wp.func
def _spatial_to_vec6(value: wp.spatial_vector) -> _vec6:
    return _vec6(value[0], value[1], value[2], value[3], value[4], value[5])


@wp.func
def _vec6_to_spatial(value: _vec6) -> wp.spatial_vector:
    return wp.spatial_vector(value[0], value[1], value[2], value[3], value[4], value[5])


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
    body_inertia: wp.array[wp.spatial_matrix],
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
        inertia = body_inertia[child]
        gravity_force = body_mass[child] * gravity[wp.max(body_world[child], wp.int32(0))]
        x_com = wp.transform_get_translation(body_q_com[child])
        gravity_wrench = wp.spatial_vector(gravity_force, wp.cross(x_com, gravity_force))
        body_velocity[child] = velocity
        body_coriolis[child] = coriolis
        body_bias[child] = (
            inertia * coriolis + spatial_cross_dual(velocity, inertia * velocity) - gravity_wrench + body_force[child]
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


@wp.func
def _factor_single_reduced_articulation(
    start: wp.int32,
    end: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_armature: wp.array[wp.float32],
    joint_qd_internal: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    body_i_m: wp.array[wp.spatial_matrix],
    body_q_com: wp.array[wp.transform],
    body_i_s: wp.array[wp.spatial_matrix],
    joint_s: wp.array[wp.spatial_vector],
    articulated_inertia: wp.array[wp.spatial_matrix],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array3d[wp.float32],
):
    for joint in range(start, end):
        parent = joint_parent[joint]
        child = joint_child[joint]
        q_com = body_q[child] * body_x_com[child]
        body_q_com[child] = q_com
        inertia = transform_spatial_inertia(q_com, body_i_m[child])
        body_i_s[child] = inertia
        articulated_inertia[child] = inertia

        x_wpj = joint_x_p[joint]
        if parent >= wp.int32(0):
            x_wpj = body_q[parent] * x_wpj
        jcalc_motion(
            joint_type[joint],
            joint_axis,
            joint_dof_dim[joint, 0],
            joint_dof_dim[joint, 1],
            x_wpj,
            joint_qd_internal,
            joint_qd_start[joint],
            joint_s,
        )

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        dof_end = joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        inertia = articulated_inertia[child]
        d = _mat66(0.0)

        for column in range(_MAX_JOINT_DOF):
            if wp.int32(column) < dof_count:
                dof = dof_start + wp.int32(column)
                u = inertia * joint_s[dof]
                joint_u[dof] = u
                for row in range(_MAX_JOINT_DOF):
                    if wp.int32(row) < dof_count:
                        d[row, column] = wp.dot(joint_s[dof_start + wp.int32(row)], u)
                d[column, column] += joint_armature[dof]

        d_inv = _invert_spd(d, dof_count)
        for row in range(_MAX_JOINT_DOF):
            for column in range(_MAX_JOINT_DOF):
                joint_d_inv[joint, row, column] = d_inv[row, column]

        reduced = inertia
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

        if parent >= wp.int32(0):
            articulated_inertia[parent] = articulated_inertia[parent] + reduced


@wp.kernel(enable_backward=False)
def _factor_reduced_articulations_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_armature: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    body_i_m: wp.array[wp.spatial_matrix],
    joint_qd_internal: wp.array[wp.float32],
    body_q_com: wp.array[wp.transform],
    body_i_s: wp.array[wp.spatial_matrix],
    joint_s: wp.array[wp.spatial_vector],
    articulated_inertia: wp.array[wp.spatial_matrix],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array3d[wp.float32],
):
    """Convert public speeds and build world-origin ABA factorizations."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]
    for joint in range(start, end):
        _convert_joint_velocity_to_integrator_basis(
            joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_x_p,
            body_q,
            body_com,
            joint_qd_public,
            joint_qd_internal,
        )
    _factor_single_reduced_articulation(
        start,
        end,
        joint_type,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_x_p,
        joint_axis,
        joint_dof_dim,
        joint_armature,
        joint_qd_internal,
        body_q,
        body_com,
        body_x_com,
        body_i_m,
        body_q_com,
        body_i_s,
        joint_s,
        articulated_inertia,
        joint_u,
        joint_d_inv,
    )


@wp.func
def _convert_joint_velocity_to_integrator_basis(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_qd_public: wp.array[wp.float32],
    joint_qd_internal: wp.array[wp.float32],
):
    qd_start = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + wp.int32(1)]
    kind = joint_type[joint]
    if kind != JointType.FREE and kind != JointType.DISTANCE:
        for dof in range(qd_start, qd_end):
            joint_qd_internal[dof] = joint_qd_public[dof]
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    x_wpj = joint_x_p[joint]
    if parent >= wp.int32(0):
        x_wpj = body_q[parent] * x_wpj

    q_parent = wp.transform_get_rotation(x_wpj)
    x_anchor = wp.transform_get_translation(x_wpj)
    x_child_com = wp.transform_point(body_q[child], body_com[child])
    r_child_com_parent = wp.quat_rotate_inv(q_parent, x_child_com - x_anchor)
    v_com_parent = wp.vec3(
        joint_qd_public[qd_start],
        joint_qd_public[qd_start + wp.int32(1)],
        joint_qd_public[qd_start + wp.int32(2)],
    )
    omega_parent = wp.vec3(
        joint_qd_public[qd_start + wp.int32(3)],
        joint_qd_public[qd_start + wp.int32(4)],
        joint_qd_public[qd_start + wp.int32(5)],
    )
    v_internal_parent = v_com_parent - wp.cross(omega_parent, r_child_com_parent)
    joint_qd_internal[qd_start] = v_internal_parent[0]
    joint_qd_internal[qd_start + wp.int32(1)] = v_internal_parent[1]
    joint_qd_internal[qd_start + wp.int32(2)] = v_internal_parent[2]
    joint_qd_internal[qd_start + wp.int32(3)] = omega_parent[0]
    joint_qd_internal[qd_start + wp.int32(4)] = omega_parent[1]
    joint_qd_internal[qd_start + wp.int32(5)] = omega_parent[2]


@wp.func
def _convert_joint_velocity_to_public_basis(
    joint: wp.int32,
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_qd_internal: wp.array[wp.float32],
    joint_qd_public: wp.array[wp.float32],
):
    qd_start = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + wp.int32(1)]
    kind = joint_type[joint]
    if kind != JointType.FREE and kind != JointType.DISTANCE:
        for dof in range(qd_start, qd_end):
            joint_qd_public[dof] = joint_qd_internal[dof]
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    x_wpj = joint_x_p[joint]
    if parent >= wp.int32(0):
        x_wpj = body_q[parent] * x_wpj

    q_parent = wp.transform_get_rotation(x_wpj)
    x_anchor = wp.transform_get_translation(x_wpj)
    x_child_com = wp.transform_point(body_q[child], body_com[child])
    r_child_com_parent = wp.quat_rotate_inv(q_parent, x_child_com - x_anchor)
    v_internal_parent = wp.vec3(
        joint_qd_internal[qd_start],
        joint_qd_internal[qd_start + wp.int32(1)],
        joint_qd_internal[qd_start + wp.int32(2)],
    )
    omega_parent = wp.vec3(
        joint_qd_internal[qd_start + wp.int32(3)],
        joint_qd_internal[qd_start + wp.int32(4)],
        joint_qd_internal[qd_start + wp.int32(5)],
    )
    v_com_parent = v_internal_parent + wp.cross(omega_parent, r_child_com_parent)
    joint_qd_public[qd_start] = v_com_parent[0]
    joint_qd_public[qd_start + wp.int32(1)] = v_com_parent[1]
    joint_qd_public[qd_start + wp.int32(2)] = v_com_parent[2]
    joint_qd_public[qd_start + wp.int32(3)] = omega_parent[0]
    joint_qd_public[qd_start + wp.int32(4)] = omega_parent[1]
    joint_qd_public[qd_start + wp.int32(5)] = omega_parent[2]


@wp.kernel(enable_backward=False)
def _finish_reduced_articulations_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd_internal: wp.array[wp.float32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_armature: wp.array[wp.float32],
    body_com: wp.array[wp.vec3],
    body_x_com: wp.array[wp.transform],
    body_i_m: wp.array[wp.spatial_matrix],
    zero_generalized_acceleration: wp.array[wp.float32],
    dt: wp.float32,
    joint_qd_public: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_com: wp.array[wp.transform],
    body_i_s: wp.array[wp.spatial_matrix],
    joint_s: wp.array[wp.spatial_vector],
    articulated_inertia: wp.array[wp.spatial_matrix],
    joint_u: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array3d[wp.float32],
    bodies: BodyContainer,
):
    """Integrate, evaluate FK, refactor, and publish one articulation."""
    articulation = wp.tid()
    start = articulation_start[articulation]
    end = articulation_end[articulation]

    for joint in range(start, end):
        child = joint_child[joint]
        jcalc_integrate(
            joint_parent[joint],
            joint_x_c[joint],
            body_com[child],
            joint_type[joint],
            joint_q,
            joint_qd_internal,
            zero_generalized_acceleration,
            joint_q_start[joint],
            joint_qd_start[joint],
            joint_dof_dim[joint, 0],
            joint_dof_dim[joint, 1],
            dt,
            joint_q,
            joint_qd_internal,
        )

    eval_single_articulation_fk_with_velocity_conversion(
        start,
        end,
        joint_q,
        joint_qd_internal,
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
        body_q,
        body_qd,
    )
    for joint in range(start, end):
        _convert_joint_velocity_to_public_basis(
            joint,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_x_p,
            body_q,
            body_com,
            joint_qd_internal,
            joint_qd_public,
        )

    _factor_single_reduced_articulation(
        start,
        end,
        joint_type,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_x_p,
        joint_axis,
        joint_dof_dim,
        joint_armature,
        joint_qd_internal,
        body_q,
        body_com,
        body_x_com,
        body_i_m,
        body_q_com,
        body_i_s,
        joint_s,
        articulated_inertia,
        joint_u,
        joint_d_inv,
    )

    for joint in range(start, end):
        child = joint_child[joint]
        transform = body_q[child]
        rotation = wp.transform_get_rotation(transform)
        origin = wp.transform_get_translation(transform)
        slot = child + wp.int32(1)
        bodies.position[slot] = origin + wp.quat_rotate(rotation, body_com[child])
        bodies.orientation[slot] = rotation
        twist = body_qd[child]
        bodies.velocity[slot] = wp.spatial_top(twist)
        bodies.angular_velocity[slot] = wp.spatial_bottom(twist)


@wp.kernel(enable_backward=False)
def _solve_articulated_system_kernel(
    articulation_start: wp.array[wp.int32],
    articulation_end: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_s: wp.array[wp.spatial_vector],
    joint_u_matrix: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array3d[wp.float32],
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
                        d_inv_u[row] += joint_d_inv[joint, row, column] * reduced_force[column]

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
                        qdd[row] += joint_d_inv[joint, row, column] * rhs[column]

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
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_s: wp.array[wp.spatial_vector],
    joint_u_matrix: wp.array[wp.spatial_vector],
    joint_d_inv: wp.array3d[wp.float32],
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
    body_mass: wp.array[wp.float32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    body_q_com: wp.array[wp.transform],
    body_inertia: wp.array[wp.spatial_matrix],
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
        inertia = body_inertia[child]
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
        coord_start = joint_q_start[joint]
        target_start = joint_target_q_start[joint]
        joint_kind = joint_type[joint]
        subtree_bias = body_bias[child]
        for dof in range(dof_start, dof_end):
            local = dof - dof_start
            applied = wp.float32(0.0)
            if include_external and joint_kind != JointType.FREE and joint_kind != JointType.DISTANCE:
                applied = joint_f[dof]
            if include_external and (
                joint_kind == JointType.REVOLUTE or joint_kind == JointType.PRISMATIC or joint_kind == JointType.D6
            ):
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
                        d_inv_u[row] += joint_d_inv[joint, row, column] * reduced_force[column]
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
                        qdd[row] += joint_d_inv[joint, row, column] * rhs[column]
        child_acceleration = parent_acceleration
        for row in range(_MAX_JOINT_DOF):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                generalized_acceleration[dof] = qdd[row]
                joint_qd[dof] += qdd[row] * dt
                child_acceleration += joint_s[dof] * qdd[row]
        body_acceleration[child] = child_acceleration

    # The configuration is unchanged in this phase, so the factored motion
    # subspaces directly reconstruct the updated world-origin twists.
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


@wp.kernel(enable_backward=False)
def _solve_reduced_contacts_multi_world_kernel(
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array2d[wp.int32],
    world_csr_offsets: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    contact_start: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    iterations: wp.int32,
    use_bias: wp.bool,
    prepare: wp.bool,
):
    world = wp.tid()
    world_base = world_csr_offsets[world]
    num_colors = world_num_colors[world]

    if prepare:
        for color in range(num_colors):
            start = world_base + color_starts[world, color]
            end = world_base + color_starts[world, color + wp.int32(1)]
            for index in range(start, end):
                cid = element_ids_by_color[index]
                if cid >= contact_start:
                    reduced_contact_prepare(
                        contact_cols,
                        cid - contact_start,
                        bodies,
                        idt,
                        cc,
                        contacts,
                    )

    for iteration in range(iterations):
        for color_offset in range(num_colors):
            color = color_offset
            if (iteration & wp.int32(1)) != wp.int32(0):
                color = num_colors - wp.int32(1) - color_offset
            start = world_base + color_starts[world, color]
            end = world_base + color_starts[world, color + wp.int32(1)]
            for index in range(start, end):
                cid = element_ids_by_color[index]
                if cid >= contact_start:
                    reduced_contact_iterate(
                        contact_cols,
                        cid - contact_start,
                        bodies,
                        idt,
                        sor_boost,
                        cc,
                        contacts,
                        use_bias,
                    )


@wp.kernel(enable_backward=False)
def _solve_reduced_contacts_single_world_kernel(
    contact_cols: ContactColumnContainer,
    bodies: BodyContainer,
    idt: wp.float32,
    sor_boost: wp.float32,
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_colors_buffer: wp.array[wp.int32],
    contact_start: wp.int32,
    cc: ContactContainer,
    contacts: ContactViews,
    iterations: wp.int32,
    use_bias: wp.bool,
    prepare: wp.bool,
):
    num_colors = num_colors_buffer[0]

    if prepare:
        for color in range(num_colors):
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            for index in range(start, end):
                cid = element_ids_by_color[index]
                if cid >= contact_start:
                    reduced_contact_prepare(
                        contact_cols,
                        cid - contact_start,
                        bodies,
                        idt,
                        cc,
                        contacts,
                    )

    for iteration in range(iterations):
        for color_offset in range(num_colors):
            color = color_offset
            if (iteration & wp.int32(1)) != wp.int32(0):
                color = num_colors - wp.int32(1) - color_offset
            start = color_starts[color]
            end = color_starts[color + wp.int32(1)]
            for index in range(start, end):
                cid = element_ids_by_color[index]
                if cid >= contact_start:
                    reduced_contact_iterate(
                        contact_cols,
                        cid - contact_start,
                        bodies,
                        idt,
                        sor_boost,
                        cc,
                        contacts,
                        use_bias,
                    )


class ReducedArticulationSystem:
    """Cached linear-time reduced-coordinate factorization for a Newton model."""

    def __init__(self, model: Model):
        if model.articulation_count <= 0:
            raise ValueError("reduced articulation system requires at least one articulation")
        self.model = model
        self.device = model.device
        body_count = int(model.body_count)
        joint_count = int(model.joint_count)
        dof_count = int(model.joint_dof_count)

        self.body_i_m = wp.empty(body_count, dtype=wp.spatial_matrix, device=self.device)
        self.body_x_com = wp.empty(body_count, dtype=wp.transform, device=self.device)
        self.body_q_com = wp.empty(body_count, dtype=wp.transform, device=self.device)
        self.body_i_s = wp.empty(body_count, dtype=wp.spatial_matrix, device=self.device)
        self.joint_s = wp.empty(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_qd_internal = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.articulated_inertia = wp.empty(body_count, dtype=wp.spatial_matrix, device=self.device)
        self.joint_u_matrix = wp.empty(dof_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_d_inv = wp.zeros((joint_count, 6, 6), dtype=wp.float32, device=self.device)
        self.body_force = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.external_body_force = wp.zeros(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_velocity = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_coriolis = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_bias = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.body_work = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.joint_work = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.body_acceleration = wp.empty(body_count, dtype=wp.spatial_vector, device=self.device)
        self.generalized_acceleration = wp.empty(dof_count, dtype=wp.float32, device=self.device)
        self.zero_generalized_force = wp.zeros(dof_count, dtype=wp.float32, device=self.device)
        self.generalized_rhs = wp.empty(dof_count, dtype=wp.float32, device=self.device)

        wp.launch(
            compute_spatial_inertia,
            dim=body_count,
            inputs=[model.body_inertia, model.body_mass],
            outputs=[self.body_i_m],
            device=self.device,
        )
        wp.launch(
            compute_com_transforms,
            dim=body_count,
            inputs=[model.body_com],
            outputs=[self.body_x_com],
            device=self.device,
        )

    def factor(self, state: State) -> None:
        """Factor the generalized mass operator at the current joint configuration."""
        wp.launch(
            _factor_reduced_articulations_kernel,
            dim=int(self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_qd_start,
                self.model.joint_X_p,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                self.model.joint_armature,
                state.joint_qd,
                state.body_q,
                self.model.body_com,
                self.body_x_com,
                self.body_i_m,
            ],
            outputs=[
                self.joint_qd_internal,
                self.body_q_com,
                self.body_i_s,
                self.joint_s,
                self.articulated_inertia,
                self.joint_u_matrix,
                self.joint_d_inv,
            ],
            device=self.device,
        )

    def forward_dynamics(self, state: State, control: Control) -> wp.array[wp.float32]:
        """Compute generalized acceleration from common Newton state and control."""
        self.factor(state)
        wp.copy(self.external_body_force, state.body_f)
        wp.launch(
            convert_body_force_com_to_origin,
            dim=int(self.model.body_count),
            inputs=[state.body_q, self.body_x_com],
            outputs=[self.external_body_force],
            device=self.device,
        )
        wp.launch(
            accumulate_free_distance_joint_f_to_body_force,
            dim=int(self.model.joint_count),
            inputs=[
                self.model.joint_type,
                self.model.joint_child,
                self.model.joint_qd_start,
                state.body_q,
                self.body_x_com,
                control.joint_f,
            ],
            outputs=[self.external_body_force],
            device=self.device,
        )
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
                self.external_body_force,
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
    ) -> None:
        """Apply one split ABA velocity update and publish link twists."""
        wp.launch(
            _advance_reduced_articulations_kernel,
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
                self.joint_u_matrix,
                self.joint_d_inv,
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
                self.model.body_mass,
                self.model.body_world,
                self.model.gravity,
                self.body_q_com,
                self.body_i_s,
                state.body_f,
                wp.float32(dt),
                wp.bool(include_external),
                wp.bool(include_coriolis),
            ],
            outputs=[
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
            ],
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
        for articulation in range(int(model.articulation_count)):
            start = int(articulation_start[articulation])
            end = int(articulation_end[articulation])
            children = joint_child[start:end]
            tree_joint_mask_np[start:end] = 1
            body_is_reduced_np[children] = 1
            constraint_node_np[children + 1] = int(children[0]) + 1
            body_articulation_np[children + 1] = articulation
        bodies.constraint_node.assign(constraint_node_np)

        reduced = ReducedArticulationData()
        reduced.enabled = wp.ones(1, dtype=wp.int32, device=model.device)
        reduced.body_articulation = wp.array(body_articulation_np, device=model.device)
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
        bodies.reduced = reduced
        body_is_reduced.assign(body_is_reduced_np)
        tree_joint_mask.assign(tree_joint_mask_np)
        self.body_is_reduced = body_is_reduced
        self.tree_joint_mask = tree_joint_mask
        self.body_is_reduced_np = body_is_reduced_np.astype(bool)
        self.tree_joint_mask_np = tree_joint_mask_np.astype(bool)
        self.loop_system = ReducedLoopSystem(model, self.body_is_reduced_np)
        self.owned_joint_mask_np = self.tree_joint_mask_np.copy()
        self.owned_joint_mask_np[self.loop_system.joint_indices_np] = True

        wp.launch(
            _mark_articulated_bodies_kernel,
            dim=int(model.body_count),
            inputs=[self.body_is_reduced],
            outputs=[bodies],
            device=model.device,
        )

    def import_step(self, state: State, control: Control) -> None:
        """Import common generalized state and controls into stable graph buffers."""
        wp.copy(self.state.joint_q, state.joint_q)
        wp.copy(self.state.joint_qd, state.joint_qd)
        wp.copy(self.state.body_f, state.body_f)
        wp.copy(self.control.joint_f, control.joint_f)
        wp.copy(self.control.joint_target_q, control.joint_target_q)
        wp.copy(self.control.joint_target_qd, control.joint_target_qd)
        eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.sync_bodies()

    def begin_substep(self, dt: float, *, split_dynamics: bool) -> None:
        """Apply pre-PGS reduced dynamics at the current configuration."""
        self.system.factor(self.state)
        self.system.advance(
            self.state,
            self.control,
            self.bodies,
            dt,
            include_external=True,
            include_coriolis=not split_dynamics,
        )

    def end_substep(self, dt: float, *, split_dynamics: bool) -> None:
        """Apply post-impulse Coriolis dynamics, integrate, and refactor."""
        if split_dynamics:
            self.system.advance(
                self.state,
                self.control,
                self.bodies,
                dt,
                include_external=False,
                include_coriolis=True,
            )
        wp.launch(
            _finish_reduced_articulations_kernel,
            dim=int(self.model.articulation_count),
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.state.joint_q,
                self.system.joint_qd_internal,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_type,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_dof_dim,
                self.model.joint_armature,
                self.model.body_com,
                self.system.body_x_com,
                self.system.body_i_m,
                self.system.zero_generalized_force,
                wp.float32(dt),
            ],
            outputs=[
                self.state.joint_qd,
                self.state.body_q,
                self.state.body_qd,
                self.system.body_q_com,
                self.system.body_i_s,
                self.system.joint_s,
                self.system.articulated_inertia,
                self.system.joint_u_matrix,
                self.system.joint_d_inv,
                self.bodies,
            ],
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
        common_tail = [
            wp.int32(world._contact_offset),
            world._contact_container,
            contact_views,
            wp.int32(iterations),
            wp.bool(not relax),
            wp.bool(not relax),
        ]
        if world.step_layout == "single_world":
            wp.launch(
                _solve_reduced_contacts_single_world_kernel,
                dim=1,
                inputs=[
                    *common_inputs,
                    world._partitioner.element_ids_by_color,
                    world._partitioner.color_starts,
                    world._partitioner.num_colors,
                    *common_tail,
                ],
                device=self.model.device,
            )
        else:
            wp.launch(
                _solve_reduced_contacts_multi_world_kernel,
                dim=world.num_worlds,
                inputs=[
                    *common_inputs,
                    world._world_element_ids_by_color,
                    world._world_color_starts,
                    world._world_csr_offsets,
                    world._world_num_colors,
                    *common_tail,
                ],
                device=self.model.device,
            )

    def export_step(self, state: State) -> None:
        """Export authoritative generalized coordinates to common Newton state."""
        wp.copy(state.joint_q, self.state.joint_q)
        wp.copy(state.joint_qd, self.state.joint_qd)


__all__ = ["ReducedArticulationSystem", "ReducedPhoenXArticulation"]
