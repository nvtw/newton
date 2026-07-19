# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed mixed contact and revolute kernels for PhoenX mini."""

import warp as wp

from .packed_kernels import (
    _block_sync,
    _contact_tangent,
    _effective_mass,
    _mul_world_inertia,
    _prepare_packed_contact,
    _solve_packed_contact,
    _xyz,
)


@wp.func
def _angular_mass(
    body_a: wp.int32,
    body_b: wp.int32,
    direction: wp.vec3,
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
) -> wp.float32:
    denominator = wp.float32(0.0)
    if body_a >= wp.int32(0):
        denominator += wp.dot(direction, _mul_world_inertia(body_a, direction, inertia0, inertia1, inertia2))
    if body_b >= wp.int32(0):
        denominator += wp.dot(direction, _mul_world_inertia(body_b, direction, inertia0, inertia1, inertia2))
    result = wp.float32(0.0)
    if denominator > wp.float32(1.0e-12):
        result = wp.float32(1.0) / denominator
    return result


@wp.func
def _prepare_packed_revolute(
    joint: wp.int32,
    packed: wp.int32,
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
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    body_pair: wp.array[wp.vec4i],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    row0: wp.array[wp.vec4],
    row1: wp.array[wp.vec4],
    row2: wp.array[wp.vec4],
    row3: wp.array[wp.vec4],
    row4: wp.array[wp.vec4],
    row_mass: wp.array[wp.vec4],
    row_mass4: wp.array[wp.vec4],
    motor: wp.array[wp.vec4],
):
    body_a = joint_parent[joint]
    body_b = joint_child[joint]
    pose_a = wp.transform_identity()
    if body_a >= wp.int32(0):
        pose_a = body_q[body_a]
    pose_b = body_q[body_b]
    anchor_a = pose_a * joint_x_p[joint]
    anchor_b = pose_b * joint_x_c[joint]
    point_a = wp.transform_get_translation(anchor_a)
    point_b = wp.transform_get_translation(anchor_b)
    com_a = wp.vec3(0.0)
    inv_mass_a = wp.float32(0.0)
    if body_a >= wp.int32(0):
        com_a = wp.transform_point(pose_a, body_com[body_a])
        inv_mass_a = body_inv_mass[body_a]
    com_b = wp.transform_point(pose_b, body_com[body_b])
    inv_mass_b = body_inv_mass[body_b]
    arm_a = point_a - com_a
    arm_b = point_b - com_b
    error = point_b - point_a

    direction0 = wp.vec3(1.0, 0.0, 0.0)
    direction1 = wp.vec3(0.0, 1.0, 0.0)
    direction2 = wp.vec3(0.0, 0.0, 1.0)
    dof = joint_qd_start[joint]
    axis_a = wp.normalize(wp.transform_vector(anchor_a, joint_axis[dof]))
    axis_b = wp.normalize(wp.transform_vector(anchor_b, joint_axis[dof]))
    direction3 = _contact_tangent(axis_a)
    direction4 = wp.cross(axis_a, direction3)
    angular_error = wp.cross(axis_a, axis_b)

    body_pair[packed] = wp.vec4i(body_a, body_b, 1, 0)
    arm_mass_a[packed] = wp.vec4(arm_a[0], arm_a[1], arm_a[2], inv_mass_a)
    arm_mass_b[packed] = wp.vec4(arm_b[0], arm_b[1], arm_b[2], inv_mass_b)
    row0[packed] = wp.vec4(direction0[0], direction0[1], direction0[2], beta * error[0] / dt)
    row1[packed] = wp.vec4(direction1[0], direction1[1], direction1[2], beta * error[1] / dt)
    row2[packed] = wp.vec4(direction2[0], direction2[1], direction2[2], beta * error[2] / dt)
    row3[packed] = wp.vec4(direction3[0], direction3[1], direction3[2], beta * wp.dot(angular_error, direction3) / dt)
    row4[packed] = wp.vec4(direction4[0], direction4[1], direction4[2], beta * wp.dot(angular_error, direction4) / dt)
    row_mass[packed] = wp.vec4(
        _effective_mass(body_a, body_b, arm_a, arm_b, direction0, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        _effective_mass(body_a, body_b, arm_a, arm_b, direction1, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        _effective_mass(body_a, body_b, arm_a, arm_b, direction2, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        _angular_mass(body_a, body_b, direction3, inertia0, inertia1, inertia2),
    )
    row_mass4[packed] = wp.vec4(_angular_mass(body_a, body_b, direction4, inertia0, inertia1, inertia2), 0.0, 0.0, 0.0)
    effort = wp.clamp(joint_act[dof], -joint_effort_limit[dof], joint_effort_limit[dof])
    motor_impulse = axis_a * effort * dt
    motor[packed] = wp.vec4(motor_impulse[0], motor_impulse[1], motor_impulse[2], 0.0)


@wp.func
def _solve_packed_revolute(
    packed: wp.int32,
    body_pair: wp.array[wp.vec4i],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    row0: wp.array[wp.vec4],
    row1: wp.array[wp.vec4],
    row2: wp.array[wp.vec4],
    row3: wp.array[wp.vec4],
    row4: wp.array[wp.vec4],
    row_mass: wp.array[wp.vec4],
    row_mass4: wp.array[wp.vec4],
    motor: wp.array[wp.vec4],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
):
    bodies = body_pair[packed]
    body_a = bodies[0]
    body_b = bodies[1]
    ama = arm_mass_a[packed]
    amb = arm_mass_b[packed]
    arm_a = _xyz(ama)
    arm_b = _xyz(amb)
    masses = row_mass[packed]
    mass4 = row_mass4[packed][0]
    linear_a = wp.vec3(0.0)
    angular_a = wp.vec3(0.0)
    inverse_mass_a = wp.float32(0.0)
    packed_linear_b = linear_velocity[body_b]
    linear_b = _xyz(packed_linear_b)
    inverse_mass_b = packed_linear_b[3]
    angular_b = _xyz(angular_velocity[body_b])
    if body_a >= wp.int32(0):
        packed_linear_a = linear_velocity[body_a]
        linear_a = _xyz(packed_linear_a)
        inverse_mass_a = packed_linear_a[3]
        angular_a = _xyz(angular_velocity[body_a])

    row = wp.int32(0)
    while row < wp.int32(5):
        data = row0[packed]
        mass = masses[0]
        if row == wp.int32(1):
            data = row1[packed]
            mass = masses[1]
        elif row == wp.int32(2):
            data = row2[packed]
            mass = masses[2]
        elif row == wp.int32(3):
            data = row3[packed]
            mass = masses[3]
        elif row == wp.int32(4):
            data = row4[packed]
            mass = mass4
        direction = _xyz(data)
        impulse_value = wp.float32(0.0)
        if row < wp.int32(3):
            relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
            impulse_value = -mass * (wp.dot(relative, direction) + data[3])
            if body_a >= wp.int32(0):
                linear_a -= direction * (impulse_value * inverse_mass_a)
                angular_a -= _mul_world_inertia(
                    body_a, wp.cross(arm_a, direction) * impulse_value, inertia0, inertia1, inertia2
                )
            linear_b += direction * (impulse_value * inverse_mass_b)
            angular_b += _mul_world_inertia(
                body_b, wp.cross(arm_b, direction) * impulse_value, inertia0, inertia1, inertia2
            )
        else:
            impulse_value = -mass * (wp.dot(angular_b - angular_a, direction) + data[3])
            if body_a >= wp.int32(0):
                angular_a -= _mul_world_inertia(body_a, direction * impulse_value, inertia0, inertia1, inertia2)
            angular_b += _mul_world_inertia(body_b, direction * impulse_value, inertia0, inertia1, inertia2)
        row += wp.int32(1)

    motor_impulse = _xyz(motor[packed])
    if body_a >= wp.int32(0):
        angular_a -= _mul_world_inertia(body_a, motor_impulse, inertia0, inertia1, inertia2)
        linear_velocity[body_a] = wp.vec4(linear_a[0], linear_a[1], linear_a[2], inverse_mass_a)
        angular_velocity[body_a] = wp.vec4(angular_a[0], angular_a[1], angular_a[2], 0.0)
    angular_b += _mul_world_inertia(body_b, motor_impulse, inertia0, inertia1, inertia2)
    linear_velocity[body_b] = wp.vec4(linear_b[0], linear_b[1], linear_b[2], inverse_mass_b)
    angular_velocity[body_b] = wp.vec4(angular_b[0], angular_b[1], angular_b[2], 0.0)


@wp.kernel(enable_backward=False)
def prepare_mixed_constraints_kernel(
    block_dim: wp.int32,
    capacity: wp.int32,
    max_colors: wp.int32,
    color_capacity: wp.int32,
    dt: wp.float32,
    contact_beta: wp.float32,
    contact_slop: wp.float32,
    joint_beta: wp.float32,
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    world_color_offset: wp.array[wp.int32],
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
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    body_pair: wp.array[wp.vec4i],
    normal_bias: wp.array[wp.vec4],
    tangent_mu: wp.array[wp.vec4],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    effective_mass: wp.array[wp.vec4],
    impulse: wp.array[wp.vec4],
    row0: wp.array[wp.vec4],
    row1: wp.array[wp.vec4],
    row2: wp.array[wp.vec4],
    row3: wp.array[wp.vec4],
    row4: wp.array[wp.vec4],
    row_mass: wp.array[wp.vec4],
    row_mass4: wp.array[wp.vec4],
    motor: wp.array[wp.vec4],
):
    tid = wp.tid()
    world = tid // block_dim
    lane = tid - world * block_dim
    color = wp.int32(0)
    num_colors = wp.min(world_num_colors[world], max_colors)
    while color < num_colors:
        color_index = world * max_colors + color
        count = wp.min(world_color_count[color_index], color_capacity)
        slot = lane
        while slot < count:
            encoded = color_constraints[color_index * color_capacity + slot]
            packed = world * capacity + world_color_offset[color_index] + slot
            if encoded > wp.int32(0):
                _prepare_packed_contact(
                    encoded - wp.int32(1),
                    packed,
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
                    body_com,
                    body_inv_mass,
                    inertia0,
                    inertia1,
                    inertia2,
                    body_pair,
                    normal_bias,
                    tangent_mu,
                    arm_mass_a,
                    arm_mass_b,
                    effective_mass,
                    impulse,
                )
            else:
                _prepare_packed_revolute(
                    -encoded - wp.int32(1),
                    packed,
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
                    body_com,
                    body_inv_mass,
                    inertia0,
                    inertia1,
                    inertia2,
                    body_pair,
                    arm_mass_a,
                    arm_mass_b,
                    row0,
                    row1,
                    row2,
                    row3,
                    row4,
                    row_mass,
                    row_mass4,
                    motor,
                )
            slot += block_dim
        color += wp.int32(1)


@wp.kernel(enable_backward=False)
def solve_mixed_constraints_kernel(
    block_dim: wp.int32,
    capacity: wp.int32,
    max_colors: wp.int32,
    color_capacity: wp.int32,
    iterations: wp.int32,
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    world_color_offset: wp.array[wp.int32],
    body_pair: wp.array[wp.vec4i],
    normal_bias: wp.array[wp.vec4],
    tangent_mu: wp.array[wp.vec4],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    effective_mass: wp.array[wp.vec4],
    impulse: wp.array[wp.vec4],
    row0: wp.array[wp.vec4],
    row1: wp.array[wp.vec4],
    row2: wp.array[wp.vec4],
    row3: wp.array[wp.vec4],
    row4: wp.array[wp.vec4],
    row_mass: wp.array[wp.vec4],
    row_mass4: wp.array[wp.vec4],
    motor: wp.array[wp.vec4],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
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
                packed = world * capacity + world_color_offset[color_index] + slot
                if body_pair[packed][2] == wp.int32(0):
                    _solve_packed_contact(
                        packed,
                        body_pair,
                        normal_bias,
                        tangent_mu,
                        arm_mass_a,
                        arm_mass_b,
                        effective_mass,
                        inertia0,
                        inertia1,
                        inertia2,
                        impulse,
                        linear_velocity,
                        angular_velocity,
                    )
                else:
                    _solve_packed_revolute(
                        packed,
                        body_pair,
                        arm_mass_a,
                        arm_mass_b,
                        row0,
                        row1,
                        row2,
                        row3,
                        row4,
                        row_mass,
                        row_mass4,
                        motor,
                        inertia0,
                        inertia1,
                        inertia2,
                        linear_velocity,
                        angular_velocity,
                    )
                slot += block_dim
            _block_sync()
            color += wp.int32(1)
        iteration += wp.int32(1)
