# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free kernels for the PhoenX mini rigid-body experiment."""

import warp as wp

from newton._src.sim import BodyFlags, JointType


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
"""
)
def _block_sync(): ...


@wp.func
def _zero_mat33() -> wp.mat33:
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def _inv_inertia_world(q: wp.quat, inv_inertia: wp.mat33, torque: wp.vec3) -> wp.vec3:
    return wp.quat_rotate(q, inv_inertia * wp.quat_rotate_inv(q, torque))


@wp.func
def _body_velocity(body: wp.int32, qd: wp.array[wp.spatial_vector]) -> wp.spatial_vector:
    value = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    if body >= wp.int32(0):
        value = qd[body]
    return value


@wp.func
def _body_pose(body: wp.int32, q: wp.array[wp.transform]) -> wp.transform:
    value = wp.transform_identity()
    if body >= wp.int32(0):
        value = q[body]
    return value


@wp.func
def _body_com_world(body: wp.int32, q: wp.array[wp.transform], com: wp.array[wp.vec3]) -> wp.vec3:
    value = wp.vec3(0.0)
    if body >= wp.int32(0):
        value = wp.transform_point(q[body], com[body])
    return value


@wp.func
def _apply_impulse(
    body: wp.int32,
    linear_impulse: wp.vec3,
    angular_impulse: wp.vec3,
    q: wp.array[wp.transform],
    inv_mass: wp.array[wp.float32],
    inv_inertia: wp.array[wp.mat33],
    qd: wp.array[wp.spatial_vector],
):
    if body >= wp.int32(0):
        pose = q[body]
        velocity = qd[body]
        linear = wp.spatial_top(velocity) + inv_mass[body] * linear_impulse
        angular = wp.spatial_bottom(velocity) + _inv_inertia_world(
            wp.transform_get_rotation(pose), inv_inertia[body], angular_impulse
        )
        qd[body] = wp.spatial_vector(linear, angular)


@wp.func
def _effective_mass(
    body_a: wp.int32,
    body_b: wp.int32,
    arm_a: wp.vec3,
    arm_b: wp.vec3,
    direction: wp.vec3,
    q: wp.array[wp.transform],
    inv_mass: wp.array[wp.float32],
    inv_inertia: wp.array[wp.mat33],
) -> wp.float32:
    denominator = wp.float32(0.0)
    if body_a >= wp.int32(0):
        angular_a = wp.cross(arm_a, direction)
        denominator += inv_mass[body_a] + wp.dot(
            angular_a,
            _inv_inertia_world(wp.transform_get_rotation(q[body_a]), inv_inertia[body_a], angular_a),
        )
    if body_b >= wp.int32(0):
        angular_b = wp.cross(arm_b, direction)
        denominator += inv_mass[body_b] + wp.dot(
            angular_b,
            _inv_inertia_world(wp.transform_get_rotation(q[body_b]), inv_inertia[body_b], angular_b),
        )
    result = wp.float32(0.0)
    if denominator > wp.float32(1.0e-12):
        result = wp.float32(1.0) / denominator
    return result


@wp.func
def _relative_velocity(
    body_a: wp.int32,
    body_b: wp.int32,
    arm_a: wp.vec3,
    arm_b: wp.vec3,
    qd: wp.array[wp.spatial_vector],
) -> wp.vec3:
    velocity_a = _body_velocity(body_a, qd)
    velocity_b = _body_velocity(body_b, qd)
    point_a = wp.spatial_top(velocity_a) + wp.cross(wp.spatial_bottom(velocity_a), arm_a)
    point_b = wp.spatial_top(velocity_b) + wp.cross(wp.spatial_bottom(velocity_b), arm_b)
    return point_b - point_a


@wp.func
def _contact_tangent(normal: wp.vec3) -> wp.vec3:
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.float32(0.57735):
        reference = wp.vec3(0.0, 1.0, 0.0)
    return wp.normalize(wp.cross(normal, reference))


@wp.func
def _solve_contact(
    contact: wp.int32,
    dt: wp.float32,
    beta: wp.float32,
    slop: wp.float32,
    shape_body: wp.array[wp.int32],
    shape_mu: wp.array[wp.float32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    lambda_n: wp.array[wp.float32],
    lambda_t1: wp.array[wp.float32],
    lambda_t2: wp.array[wp.float32],
):
    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    body_a = wp.int32(-1)
    body_b = wp.int32(-1)
    if shape_a >= wp.int32(0):
        body_a = shape_body[shape_a]
    if shape_b >= wp.int32(0):
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return

    pose_a = _body_pose(body_a, body_q)
    pose_b = _body_pose(body_b, body_q)
    normal = contact_normal[contact]
    point_a = wp.transform_point(pose_a, contact_point0[contact]) + contact_margin0[contact] * normal
    point_b = wp.transform_point(pose_b, contact_point1[contact]) - contact_margin1[contact] * normal
    arm_a = point_a - _body_com_world(body_a, body_q, body_com)
    arm_b = point_b - _body_com_world(body_b, body_q, body_com)
    separation = wp.dot(point_b - point_a, normal)

    relative = _relative_velocity(body_a, body_b, arm_a, arm_b, body_qd)
    bias = beta * wp.min(separation + slop, wp.float32(0.0)) / dt
    old_n = lambda_n[contact]
    new_n = wp.max(
        old_n
        - _effective_mass(body_a, body_b, arm_a, arm_b, normal, body_q, body_inv_mass, body_inv_inertia)
        * (wp.dot(relative, normal) + bias),
        wp.float32(0.0),
    )
    delta_n = new_n - old_n
    lambda_n[contact] = new_n
    _apply_impulse(
        body_a, -normal * delta_n, -wp.cross(arm_a, normal) * delta_n, body_q, body_inv_mass, body_inv_inertia, body_qd
    )
    _apply_impulse(
        body_b, normal * delta_n, wp.cross(arm_b, normal) * delta_n, body_q, body_inv_mass, body_inv_inertia, body_qd
    )

    tangent1 = _contact_tangent(normal)
    tangent2 = wp.cross(normal, tangent1)
    mu = wp.float32(0.0)
    material_count = wp.int32(0)
    if shape_a >= wp.int32(0):
        mu += shape_mu[shape_a]
        material_count += wp.int32(1)
    if shape_b >= wp.int32(0):
        mu += shape_mu[shape_b]
        material_count += wp.int32(1)
    if material_count > wp.int32(0):
        mu /= wp.float32(material_count)
    friction_limit = mu * new_n

    relative = _relative_velocity(body_a, body_b, arm_a, arm_b, body_qd)
    old_t1 = lambda_t1[contact]
    new_t1 = wp.clamp(
        old_t1
        - _effective_mass(body_a, body_b, arm_a, arm_b, tangent1, body_q, body_inv_mass, body_inv_inertia)
        * wp.dot(relative, tangent1),
        -friction_limit,
        friction_limit,
    )
    delta_t1 = new_t1 - old_t1
    lambda_t1[contact] = new_t1
    _apply_impulse(
        body_a,
        -tangent1 * delta_t1,
        -wp.cross(arm_a, tangent1) * delta_t1,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        body_qd,
    )
    _apply_impulse(
        body_b,
        tangent1 * delta_t1,
        wp.cross(arm_b, tangent1) * delta_t1,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        body_qd,
    )

    relative = _relative_velocity(body_a, body_b, arm_a, arm_b, body_qd)
    old_t2 = lambda_t2[contact]
    new_t2 = wp.clamp(
        old_t2
        - _effective_mass(body_a, body_b, arm_a, arm_b, tangent2, body_q, body_inv_mass, body_inv_inertia)
        * wp.dot(relative, tangent2),
        -friction_limit,
        friction_limit,
    )
    delta_t2 = new_t2 - old_t2
    lambda_t2[contact] = new_t2
    _apply_impulse(
        body_a,
        -tangent2 * delta_t2,
        -wp.cross(arm_a, tangent2) * delta_t2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        body_qd,
    )
    _apply_impulse(
        body_b,
        tangent2 * delta_t2,
        wp.cross(arm_b, tangent2) * delta_t2,
        body_q,
        body_inv_mass,
        body_inv_inertia,
        body_qd,
    )


@wp.func
def _solve_revolute_row(
    body_a: wp.int32,
    body_b: wp.int32,
    arm_a: wp.vec3,
    arm_b: wp.vec3,
    direction: wp.vec3,
    bias: wp.float32,
    angular_only: bool,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
):
    relative_speed = wp.float32(0.0)
    denominator = wp.float32(0.0)
    angular_a = direction
    angular_b = direction
    if not angular_only:
        relative_speed = wp.dot(_relative_velocity(body_a, body_b, arm_a, arm_b, body_qd), direction)
        angular_a = wp.cross(arm_a, direction)
        angular_b = wp.cross(arm_b, direction)
        denominator = _effective_mass(body_a, body_b, arm_a, arm_b, direction, body_q, body_inv_mass, body_inv_inertia)
    else:
        velocity_a = _body_velocity(body_a, body_qd)
        velocity_b = _body_velocity(body_b, body_qd)
        relative_speed = wp.dot(wp.spatial_bottom(velocity_b) - wp.spatial_bottom(velocity_a), direction)
        inverse = wp.float32(0.0)
        if body_a >= wp.int32(0):
            inverse += wp.dot(
                direction,
                _inv_inertia_world(wp.transform_get_rotation(body_q[body_a]), body_inv_inertia[body_a], direction),
            )
        if body_b >= wp.int32(0):
            inverse += wp.dot(
                direction,
                _inv_inertia_world(wp.transform_get_rotation(body_q[body_b]), body_inv_inertia[body_b], direction),
            )
        if inverse > wp.float32(1.0e-12):
            denominator = wp.float32(1.0) / inverse
    impulse = -denominator * (relative_speed + bias)
    linear = direction * impulse
    if angular_only:
        linear = wp.vec3(0.0)
    _apply_impulse(body_a, -linear, -angular_a * impulse, body_q, body_inv_mass, body_inv_inertia, body_qd)
    _apply_impulse(body_b, linear, angular_b * impulse, body_q, body_inv_mass, body_inv_inertia, body_qd)


@wp.func
def _solve_revolute(
    joint: wp.int32,
    dt: wp.float32,
    beta: wp.float32,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_act: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
):
    body_a = joint_parent[joint]
    body_b = joint_child[joint]
    pose_a = _body_pose(body_a, body_q)
    pose_b = body_q[body_b]
    anchor_a = pose_a * joint_x_p[joint]
    anchor_b = pose_b * joint_x_c[joint]
    point_a = wp.transform_get_translation(anchor_a)
    point_b = wp.transform_get_translation(anchor_b)
    arm_a = point_a - _body_com_world(body_a, body_q, body_com)
    arm_b = point_b - _body_com_world(body_b, body_q, body_com)
    error = point_b - point_a
    _solve_revolute_row(
        body_a,
        body_b,
        arm_a,
        arm_b,
        wp.vec3(1.0, 0.0, 0.0),
        beta * error[0] / dt,
        False,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
    )
    _solve_revolute_row(
        body_a,
        body_b,
        arm_a,
        arm_b,
        wp.vec3(0.0, 1.0, 0.0),
        beta * error[1] / dt,
        False,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
    )
    _solve_revolute_row(
        body_a,
        body_b,
        arm_a,
        arm_b,
        wp.vec3(0.0, 0.0, 1.0),
        beta * error[2] / dt,
        False,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
    )

    dof = joint_qd_start[joint]
    axis_local = joint_axis[dof]
    axis_a = wp.normalize(wp.transform_vector(anchor_a, axis_local))
    axis_b = wp.normalize(wp.transform_vector(anchor_b, axis_local))
    tangent1 = _contact_tangent(axis_a)
    tangent2 = wp.cross(axis_a, tangent1)
    angular_error = wp.cross(axis_a, axis_b)
    _solve_revolute_row(
        body_a,
        body_b,
        arm_a,
        arm_b,
        tangent1,
        beta * wp.dot(angular_error, tangent1) / dt,
        True,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
    )
    _solve_revolute_row(
        body_a,
        body_b,
        arm_a,
        arm_b,
        tangent2,
        beta * wp.dot(angular_error, tangent2) / dt,
        True,
        body_q,
        body_qd,
        body_inv_mass,
        body_inv_inertia,
    )

    effort = wp.clamp(joint_act[dof], -joint_effort_limit[dof], joint_effort_limit[dof])
    torque_impulse = axis_a * effort * dt
    _apply_impulse(body_a, wp.vec3(0.0), -torque_impulse, body_q, body_inv_mass, body_inv_inertia, body_qd)
    _apply_impulse(body_b, wp.vec3(0.0), torque_impulse, body_q, body_inv_mass, body_inv_inertia, body_qd)


@wp.kernel(enable_backward=False)
def integrate_velocities_kernel(
    q_in: wp.array[wp.transform],
    qd_in: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    angular_damping: wp.float32,
    dt: wp.float32,
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    pose = q_in[body]
    velocity = qd_in[body]
    q_out[body] = pose
    if (body_flags[body] & BodyFlags.KINEMATIC) != wp.int32(0):
        qd_out[body] = velocity
        return
    force = body_f[body]
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    world = wp.max(body_world[body], wp.int32(0))
    linear += (wp.spatial_top(force) * body_inv_mass[body] + gravity[world] * wp.nonzero(body_inv_mass[body])) * dt
    rotation = wp.transform_get_rotation(pose)
    angular_body = wp.quat_rotate_inv(rotation, angular)
    torque_body = wp.quat_rotate_inv(rotation, wp.spatial_bottom(force)) - wp.cross(
        angular_body, body_inertia[body] * angular_body
    )
    angular = wp.quat_rotate(rotation, angular_body + body_inv_inertia[body] * torque_body * dt)
    angular *= wp.max(wp.float32(0.0), wp.float32(1.0) - angular_damping * dt)
    qd_out[body] = wp.spatial_vector(linear, angular)


@wp.kernel(enable_backward=False)
def integrate_poses_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    dt: wp.float32,
):
    body = wp.tid()
    if (body_flags[body] & BodyFlags.KINEMATIC) != wp.int32(0):
        return
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    com_world = wp.transform_point(pose, body_com[body])
    velocity = body_qd[body]
    com_world += wp.spatial_top(velocity) * dt
    angular = wp.spatial_bottom(velocity)
    rotation = wp.normalize(rotation + wp.quat(angular, wp.float32(0.0)) * rotation * wp.float32(0.5) * dt)
    translation = com_world - wp.quat_rotate(rotation, body_com[body])
    body_q[body] = wp.transform(translation, rotation)


@wp.func
def _store_colored_constraint(
    world: wp.int32,
    color: wp.int32,
    encoded: wp.int32,
    max_colors: wp.int32,
    color_capacity: wp.int32,
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    color_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    color_index = world * max_colors + color
    slot = wp.atomic_add(world_color_count, color_index, wp.int32(1))
    wp.atomic_max(world_num_colors, world, color + wp.int32(1))
    if slot < color_capacity:
        color_constraints[color_index * color_capacity + slot] = encoded
    else:
        wp.atomic_add(overflow, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def append_revolute_constraints_kernel(
    joint_type: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_world: wp.array[wp.int32],
    max_colors: wp.int32,
    capacity: wp.int32,
    color_capacity: wp.int32,
    body_color_owner: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    color_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    joint = wp.tid()
    if not joint_enabled[joint] or joint_type[joint] != JointType.REVOLUTE:
        return
    # Newton uses world -1 for the ordinary single-world builder.
    world = wp.max(joint_world[joint], wp.int32(0))
    slot = wp.atomic_add(world_constraint_count, world, wp.int32(1))
    if slot >= capacity:
        wp.atomic_add(overflow, 0, wp.int32(1))
        return
    body_a = joint_parent[joint]
    body_b = joint_child[joint]
    token = world * capacity + slot + wp.int32(1)
    color = wp.int32(0)
    while color < max_colors:
        claimed_a = body_a < wp.int32(0)
        if body_a >= wp.int32(0):
            claimed_a = wp.atomic_cas(body_color_owner, body_a * max_colors + color, wp.int32(-1), token) == wp.int32(
                -1
            )
        if claimed_a:
            claimed_b = body_b < wp.int32(0) or body_b == body_a
            if body_b >= wp.int32(0) and body_b != body_a:
                claimed_b = wp.atomic_cas(
                    body_color_owner, body_b * max_colors + color, wp.int32(-1), token
                ) == wp.int32(-1)
            if claimed_b:
                _store_colored_constraint(
                    world,
                    color,
                    -joint - wp.int32(1),
                    max_colors,
                    color_capacity,
                    world_num_colors,
                    world_color_count,
                    color_constraints,
                    overflow,
                )
                return
            if body_a >= wp.int32(0):
                wp.atomic_cas(body_color_owner, body_a * max_colors + color, token, wp.int32(-1))
        color += wp.int32(1)
    wp.atomic_add(overflow, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def append_contact_constraints_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    max_colors: wp.int32,
    capacity: wp.int32,
    color_capacity: wp.int32,
    body_color_owner: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    color_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    body_a = wp.int32(-1)
    body_b = wp.int32(-1)
    if shape_a >= wp.int32(0):
        body_a = shape_body[shape_a]
    if shape_b >= wp.int32(0):
        body_b = shape_body[shape_b]
    if body_a == body_b:
        return
    world = wp.int32(-1)
    if body_a >= wp.int32(0):
        world = body_world[body_a]
    elif body_b >= wp.int32(0):
        world = body_world[body_b]
    # Newton uses world -1 for the ordinary single-world builder.
    world = wp.max(world, wp.int32(0))
    slot = wp.atomic_add(world_constraint_count, world, wp.int32(1))
    if slot >= capacity:
        wp.atomic_add(overflow, 0, wp.int32(1))
        return
    token = world * capacity + slot + wp.int32(1)
    color = wp.int32(0)
    while color < max_colors:
        claimed_a = body_a < wp.int32(0)
        if body_a >= wp.int32(0):
            claimed_a = wp.atomic_cas(body_color_owner, body_a * max_colors + color, wp.int32(-1), token) == wp.int32(
                -1
            )
        if claimed_a:
            claimed_b = body_b < wp.int32(0) or body_b == body_a
            if body_b >= wp.int32(0) and body_b != body_a:
                claimed_b = wp.atomic_cas(
                    body_color_owner, body_b * max_colors + color, wp.int32(-1), token
                ) == wp.int32(-1)
            if claimed_b:
                _store_colored_constraint(
                    world,
                    color,
                    contact + wp.int32(1),
                    max_colors,
                    color_capacity,
                    world_num_colors,
                    world_color_count,
                    color_constraints,
                    overflow,
                )
                return
            if body_a >= wp.int32(0):
                wp.atomic_cas(body_color_owner, body_a * max_colors + color, token, wp.int32(-1))
        color += wp.int32(1)
    wp.atomic_add(overflow, 0, wp.int32(1))


@wp.func
def _contact_world(
    contact: wp.int32,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
) -> wp.int32:
    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    body = wp.int32(-1)
    if shape_a >= wp.int32(0):
        body = shape_body[shape_a]
    if body < wp.int32(0) and shape_b >= wp.int32(0):
        body = shape_body[shape_b]
    if body >= wp.int32(0):
        return wp.max(body_world[body], wp.int32(0))
    return wp.int32(0)


@wp.kernel(enable_backward=False)
def initialize_contact_topology_stable_kernel(
    contact_count: wp.array[wp.int32],
    match_capacity: wp.int32,
    previous_count: wp.array[wp.int32],
    topology_stable: wp.array[wp.int32],
):
    count = wp.min(contact_count[0], match_capacity)
    topology_stable[0] = wp.int32(count == previous_count[0])
    previous_count[0] = count


@wp.kernel(enable_backward=False)
def validate_contact_topology_stable_kernel(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    previous_shape0: wp.array[wp.int32],
    previous_shape1: wp.array[wp.int32],
    topology_stable: wp.array[wp.int32],
):
    contact = wp.tid()
    count = wp.min(contact_count[0], wp.int32(shape0.shape[0]))
    if contact >= count:
        return
    current_shape0 = shape0[contact]
    current_shape1 = shape1[contact]
    if previous_shape0[contact] != current_shape0 or previous_shape1[contact] != current_shape1:
        wp.atomic_min(topology_stable, 0, wp.int32(0))
    previous_shape0[contact] = current_shape0
    previous_shape1[contact] = current_shape1


@wp.kernel(enable_backward=False)
def mark_sorted_contact_world_runs_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    world_contact_start: wp.array[wp.int32],
    world_contact_end: wp.array[wp.int32],
    world_run_count: wp.array[wp.int32],
):
    contact = wp.tid()
    count = contact_count[0]
    if contact >= count:
        return
    world = _contact_world(contact, contact_shape0, contact_shape1, shape_body, body_world)
    previous_world = wp.int32(-1)
    if contact > wp.int32(0):
        previous_world = _contact_world(contact - wp.int32(1), contact_shape0, contact_shape1, shape_body, body_world)
    if previous_world != world:
        world_contact_start[world] = contact
        wp.atomic_add(world_run_count, world, wp.int32(1))

    next_world = wp.int32(-1)
    if contact + wp.int32(1) < count:
        next_world = _contact_world(contact + wp.int32(1), contact_shape0, contact_shape1, shape_body, body_world)
    if next_world != world:
        world_contact_end[world] = contact + wp.int32(1)


@wp.kernel(enable_backward=False)
def gather_sorted_mixed_world_runs_kernel(
    capacity: wp.int32,
    world_joint_offset: wp.array[wp.int32],
    world_joint: wp.array[wp.int32],
    world_contact_start: wp.array[wp.int32],
    world_contact_end: wp.array[wp.int32],
    world_run_count: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    world_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    world = wp.tid()
    joint_begin = world_joint_offset[world]
    joint_end = world_joint_offset[world + wp.int32(1)]
    joint_count = joint_end - joint_begin

    contact_count = wp.int32(0)
    runs = world_run_count[world]
    if runs == wp.int32(1):
        contact_count = world_contact_end[world] - world_contact_start[world]
    elif runs != wp.int32(0):
        wp.atomic_add(overflow, 0, runs)

    total = joint_count + contact_count
    world_constraint_count[world] = total
    if total > capacity:
        wp.atomic_add(overflow, 0, total - capacity)
        total = capacity

    slot = wp.int32(0)
    while slot < joint_count and slot < total:
        world_constraints[world * capacity + slot] = -world_joint[joint_begin + slot] - wp.int32(1)
        slot += wp.int32(1)
    contact_start = world_contact_start[world]
    while slot < total:
        world_constraints[world * capacity + slot] = contact_start + slot - joint_count + wp.int32(1)
        slot += wp.int32(1)


@wp.kernel(enable_backward=False)
def gather_sorted_contact_world_runs_kernel(
    capacity: wp.int32,
    world_contact_start: wp.array[wp.int32],
    world_contact_end: wp.array[wp.int32],
    world_run_count: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    world_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    world = wp.tid()
    runs = world_run_count[world]
    if runs == wp.int32(0):
        world_constraint_count[world] = wp.int32(0)
        return
    if runs != wp.int32(1):
        wp.atomic_add(overflow, 0, runs)
        world_constraint_count[world] = wp.int32(0)
        return

    start = world_contact_start[world]
    count = world_contact_end[world] - start
    world_constraint_count[world] = count
    if count > capacity:
        wp.atomic_add(overflow, 0, count - capacity)
        count = capacity
    slot = wp.int32(0)
    while slot < count:
        world_constraints[world * capacity + slot] = start + slot + wp.int32(1)
        slot += wp.int32(1)


@wp.kernel(enable_backward=False)
def color_world_constraints_kernel(
    capacity: wp.int32,
    max_colors: wp.int32,
    color_capacity: wp.int32,
    world_constraint_count: wp.array[wp.int32],
    world_constraints: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_color_mask: wp.array[wp.uint64],
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    color_constraints: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    world = wp.tid()
    count = wp.min(world_constraint_count[world], capacity)
    slot = wp.int32(0)
    while slot < count:
        encoded = world_constraints[world * capacity + slot]
        body_a = wp.int32(-1)
        body_b = wp.int32(-1)
        if encoded > wp.int32(0):
            contact = encoded - wp.int32(1)
            shape_a = contact_shape0[contact]
            shape_b = contact_shape1[contact]
            if shape_a >= wp.int32(0):
                body_a = shape_body[shape_a]
            if shape_b >= wp.int32(0):
                body_b = shape_body[shape_b]
        else:
            joint = -encoded - wp.int32(1)
            body_a = joint_parent[joint]
            body_b = joint_child[joint]

        used_colors = wp.uint64(0)
        if body_a >= wp.int32(0):
            used_colors = used_colors | body_color_mask[body_a]
        if body_b >= wp.int32(0) and body_b != body_a:
            used_colors = used_colors | body_color_mask[body_b]

        color = wp.int32(0)
        while color < max_colors:
            color_bit = wp.uint64(1) << wp.uint64(color)
            if (used_colors & color_bit) == wp.uint64(0):
                if body_a >= wp.int32(0):
                    body_color_mask[body_a] = body_color_mask[body_a] | color_bit
                if body_b >= wp.int32(0) and body_b != body_a:
                    body_color_mask[body_b] = body_color_mask[body_b] | color_bit
                _store_colored_constraint(
                    world,
                    color,
                    encoded,
                    max_colors,
                    color_capacity,
                    world_num_colors,
                    world_color_count,
                    color_constraints,
                    overflow,
                )
                break
            color += wp.int32(1)
        if color == max_colors:
            wp.atomic_add(overflow, 0, wp.int32(1))
        slot += wp.int32(1)


@wp.kernel(enable_backward=False)
def solve_worlds_kernel(
    block_dim: wp.int32,
    color_capacity: wp.int32,
    max_colors: wp.int32,
    iterations: wp.int32,
    dt: wp.float32,
    contact_beta: wp.float32,
    contact_slop: wp.float32,
    joint_beta: wp.float32,
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    color_constraints: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_mu: wp.array[wp.float32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_x_p: wp.array[wp.transform],
    joint_x_c: wp.array[wp.transform],
    joint_qd_start: wp.array[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_act: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    lambda_n: wp.array[wp.float32],
    lambda_t1: wp.array[wp.float32],
    lambda_t2: wp.array[wp.float32],
    body_qd: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    world = tid // block_dim
    lane = tid - world * block_dim
    num_colors = wp.min(world_num_colors[world], max_colors)
    iteration = wp.int32(0)
    while iteration < iterations:
        color = wp.int32(0)
        while color < num_colors:
            color_index = world * max_colors + color
            count = wp.min(world_color_count[color_index], color_capacity)
            slot = lane
            while slot < count:
                encoded = color_constraints[color_index * color_capacity + slot]
                if encoded > wp.int32(0):
                    _solve_contact(
                        encoded - wp.int32(1),
                        dt,
                        contact_beta,
                        contact_slop,
                        shape_body,
                        shape_mu,
                        contact_shape0,
                        contact_shape1,
                        contact_point0,
                        contact_point1,
                        contact_normal,
                        contact_margin0,
                        contact_margin1,
                        body_q,
                        body_qd,
                        body_com,
                        body_inv_mass,
                        body_inv_inertia,
                        lambda_n,
                        lambda_t1,
                        lambda_t2,
                    )
                else:
                    _solve_revolute(
                        -encoded - wp.int32(1),
                        dt,
                        joint_beta,
                        joint_parent,
                        joint_child,
                        joint_x_p,
                        joint_x_c,
                        joint_qd_start,
                        joint_axis,
                        joint_act,
                        joint_effort_limit,
                        body_q,
                        body_qd,
                        body_com,
                        body_inv_mass,
                        body_inv_inertia,
                    )
                slot += block_dim
            _block_sync()
            color += wp.int32(1)
        iteration += wp.int32(1)
