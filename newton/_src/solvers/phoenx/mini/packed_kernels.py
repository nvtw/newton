# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed contact-only kernels for the PhoenX mini experiment."""

import warp as wp

from newton._src.sim import BodyFlags


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
"""
)
def _block_sync(): ...


@wp.func
def _xyz(value: wp.vec4) -> wp.vec3:
    return wp.vec3(value[0], value[1], value[2])


@wp.func
def _inv_inertia_world(q: wp.quat, inv_inertia: wp.mat33, torque: wp.vec3) -> wp.vec3:
    return wp.quat_rotate(q, inv_inertia * wp.quat_rotate_inv(q, torque))


@wp.func
def _mul_world_inertia(
    body: wp.int32,
    value: wp.vec3,
    row0: wp.array[wp.vec4],
    row1: wp.array[wp.vec4],
    row2: wp.array[wp.vec4],
) -> wp.vec3:
    return wp.vec3(wp.dot(_xyz(row0[body]), value), wp.dot(_xyz(row1[body]), value), wp.dot(_xyz(row2[body]), value))


@wp.func
def _mul_inertia_rows(row0: wp.vec4, row1: wp.vec4, row2: wp.vec4, value: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.dot(_xyz(row0), value), wp.dot(_xyz(row1), value), wp.dot(_xyz(row2), value))


@wp.func
def _effective_mass(
    body_a: wp.int32,
    body_b: wp.int32,
    arm_a: wp.vec3,
    arm_b: wp.vec3,
    direction: wp.vec3,
    inv_mass_a: wp.float32,
    inv_mass_b: wp.float32,
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
) -> wp.float32:
    denominator = inv_mass_a + inv_mass_b
    if body_a >= wp.int32(0):
        angular = wp.cross(arm_a, direction)
        denominator += wp.dot(angular, _mul_world_inertia(body_a, angular, inertia0, inertia1, inertia2))
    if body_b >= wp.int32(0):
        angular = wp.cross(arm_b, direction)
        denominator += wp.dot(angular, _mul_world_inertia(body_b, angular, inertia0, inertia1, inertia2))
    result = wp.float32(0.0)
    if denominator > wp.float32(1.0e-12):
        result = wp.float32(1.0) / denominator
    return result


@wp.func
def _contact_tangent(normal: wp.vec3) -> wp.vec3:
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > wp.float32(0.57735):
        reference = wp.vec3(0.0, 1.0, 0.0)
    return wp.normalize(wp.cross(normal, reference))


@wp.kernel(enable_backward=False)
def integrate_velocities_packed_kernel(
    q_in: wp.array[wp.transform],
    qd_in: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    angular_damping: wp.float32,
    dt: wp.float32,
    q_out: wp.array[wp.transform],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
):
    body = wp.tid()
    pose = q_in[body]
    rotation = wp.transform_get_rotation(pose)
    velocity = qd_in[body]
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    if (body_flags[body] & BodyFlags.KINEMATIC) == wp.int32(0):
        force = body_f[body]
        world = wp.max(body_world[body], wp.int32(0))
        linear += (wp.spatial_top(force) * body_inv_mass[body] + gravity[world] * wp.nonzero(body_inv_mass[body])) * dt
        angular_body = wp.quat_rotate_inv(rotation, angular)
        torque_body = wp.quat_rotate_inv(rotation, wp.spatial_bottom(force)) - wp.cross(
            angular_body, body_inertia[body] * angular_body
        )
        angular = wp.quat_rotate(rotation, angular_body + body_inv_inertia[body] * torque_body * dt)
        angular *= wp.max(wp.float32(0.0), wp.float32(1.0) - angular_damping * dt)

    column0 = _inv_inertia_world(rotation, body_inv_inertia[body], wp.vec3(1.0, 0.0, 0.0))
    column1 = _inv_inertia_world(rotation, body_inv_inertia[body], wp.vec3(0.0, 1.0, 0.0))
    column2 = _inv_inertia_world(rotation, body_inv_inertia[body], wp.vec3(0.0, 0.0, 1.0))
    q_out[body] = pose
    linear_velocity[body] = wp.vec4(linear[0], linear[1], linear[2], body_inv_mass[body])
    angular_velocity[body] = wp.vec4(angular[0], angular[1], angular[2], 0.0)
    inertia0[body] = wp.vec4(column0[0], column1[0], column2[0], 0.0)
    inertia1[body] = wp.vec4(column0[1], column1[1], column2[1], 0.0)
    inertia2[body] = wp.vec4(column0[2], column1[2], column2[2], 0.0)


@wp.kernel(enable_backward=False)
def compute_color_offsets_kernel(
    max_colors: wp.int32,
    color_capacity: wp.int32,
    world_num_colors: wp.array[wp.int32],
    world_color_count: wp.array[wp.int32],
    world_color_offset: wp.array[wp.int32],
):
    world = wp.tid()
    count = wp.int32(0)
    color = wp.int32(0)
    num_colors = wp.min(world_num_colors[world], max_colors)
    while color < num_colors:
        color_index = world * max_colors + color
        world_color_offset[color_index] = count
        count += wp.min(world_color_count[color_index], color_capacity)
        color += wp.int32(1)


@wp.kernel(enable_backward=False)
def prepare_packed_contacts_kernel(
    block_dim: wp.int32,
    capacity: wp.int32,
    max_colors: wp.int32,
    color_capacity: wp.int32,
    dt: wp.float32,
    beta: wp.float32,
    slop: wp.float32,
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
            contact = encoded - wp.int32(1)
            shape_a = contact_shape0[contact]
            shape_b = contact_shape1[contact]
            body_a = wp.int32(-1)
            body_b = wp.int32(-1)
            if shape_a >= wp.int32(0):
                body_a = shape_body[shape_a]
            if shape_b >= wp.int32(0):
                body_b = shape_body[shape_b]

            pose_a = wp.transform_identity()
            pose_b = wp.transform_identity()
            com_a = wp.vec3(0.0)
            com_b = wp.vec3(0.0)
            inv_mass_a = wp.float32(0.0)
            inv_mass_b = wp.float32(0.0)
            if body_a >= wp.int32(0):
                pose_a = body_q[body_a]
                com_a = wp.transform_point(pose_a, body_com[body_a])
                inv_mass_a = body_inv_mass[body_a]
            if body_b >= wp.int32(0):
                pose_b = body_q[body_b]
                com_b = wp.transform_point(pose_b, body_com[body_b])
                inv_mass_b = body_inv_mass[body_b]

            normal = contact_normal[contact]
            point_a = wp.transform_point(pose_a, contact_point0[contact]) + contact_margin0[contact] * normal
            point_b = wp.transform_point(pose_b, contact_point1[contact]) - contact_margin1[contact] * normal
            arm_a = point_a - com_a
            arm_b = point_b - com_b
            tangent1 = _contact_tangent(normal)
            tangent2 = wp.cross(normal, tangent1)
            separation = wp.dot(point_b - point_a, normal)
            bias = beta * wp.min(separation + slop, wp.float32(0.0)) / dt
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

            packed = world * capacity + world_color_offset[color_index] + slot
            body_pair[packed] = wp.vec4i(body_a, body_b, 0, 0)
            normal_bias[packed] = wp.vec4(normal[0], normal[1], normal[2], bias)
            tangent_mu[packed] = wp.vec4(tangent1[0], tangent1[1], tangent1[2], mu)
            arm_mass_a[packed] = wp.vec4(arm_a[0], arm_a[1], arm_a[2], inv_mass_a)
            arm_mass_b[packed] = wp.vec4(arm_b[0], arm_b[1], arm_b[2], inv_mass_b)
            effective_mass[packed] = wp.vec4(
                _effective_mass(
                    body_a, body_b, arm_a, arm_b, normal, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2
                ),
                _effective_mass(
                    body_a, body_b, arm_a, arm_b, tangent1, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2
                ),
                _effective_mass(
                    body_a, body_b, arm_a, arm_b, tangent2, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2
                ),
                0.0,
            )
            impulse[packed] = wp.vec4(0.0)
            slot += block_dim
        color += wp.int32(1)


@wp.kernel(enable_backward=False)
def solve_packed_contacts_kernel(
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
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    impulse: wp.array[wp.vec4],
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
                bodies = body_pair[packed]
                body_a = bodies[0]
                body_b = bodies[1]
                nb = normal_bias[packed]
                tm = tangent_mu[packed]
                ama = arm_mass_a[packed]
                amb = arm_mass_b[packed]
                mass = effective_mass[packed]
                lambdas = impulse[packed]
                normal = _xyz(nb)
                tangent1 = _xyz(tm)
                tangent2 = wp.cross(normal, tangent1)
                arm_a = _xyz(ama)
                arm_b = _xyz(amb)
                linear_a = wp.vec3(0.0)
                angular_a = wp.vec3(0.0)
                linear_b = wp.vec3(0.0)
                angular_b = wp.vec3(0.0)
                if body_a >= wp.int32(0):
                    linear_a = _xyz(linear_velocity[body_a])
                    angular_a = _xyz(angular_velocity[body_a])
                if body_b >= wp.int32(0):
                    linear_b = _xyz(linear_velocity[body_b])
                    angular_b = _xyz(angular_velocity[body_b])

                relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                new_n = wp.max(lambdas[0] - mass[0] * (wp.dot(relative, normal) + nb[3]), wp.float32(0.0))
                delta = new_n - lambdas[0]
                if body_a >= wp.int32(0):
                    linear_a -= normal * (delta * ama[3])
                    angular_a -= _mul_world_inertia(
                        body_a, wp.cross(arm_a, normal) * delta, inertia0, inertia1, inertia2
                    )
                if body_b >= wp.int32(0):
                    linear_b += normal * (delta * amb[3])
                    angular_b += _mul_world_inertia(
                        body_b, wp.cross(arm_b, normal) * delta, inertia0, inertia1, inertia2
                    )

                limit = tm[3] * new_n
                relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                new_t1 = wp.clamp(lambdas[1] - mass[1] * wp.dot(relative, tangent1), -limit, limit)
                delta = new_t1 - lambdas[1]
                if body_a >= wp.int32(0):
                    linear_a -= tangent1 * (delta * ama[3])
                    angular_a -= _mul_world_inertia(
                        body_a, wp.cross(arm_a, tangent1) * delta, inertia0, inertia1, inertia2
                    )
                if body_b >= wp.int32(0):
                    linear_b += tangent1 * (delta * amb[3])
                    angular_b += _mul_world_inertia(
                        body_b, wp.cross(arm_b, tangent1) * delta, inertia0, inertia1, inertia2
                    )

                relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                new_t2 = wp.clamp(lambdas[2] - mass[2] * wp.dot(relative, tangent2), -limit, limit)
                delta = new_t2 - lambdas[2]
                if body_a >= wp.int32(0):
                    linear_a -= tangent2 * (delta * ama[3])
                    angular_a -= _mul_world_inertia(
                        body_a, wp.cross(arm_a, tangent2) * delta, inertia0, inertia1, inertia2
                    )
                    linear_velocity[body_a] = wp.vec4(linear_a[0], linear_a[1], linear_a[2], ama[3])
                    angular_velocity[body_a] = wp.vec4(angular_a[0], angular_a[1], angular_a[2], 0.0)
                if body_b >= wp.int32(0):
                    linear_b += tangent2 * (delta * amb[3])
                    angular_b += _mul_world_inertia(
                        body_b, wp.cross(arm_b, tangent2) * delta, inertia0, inertia1, inertia2
                    )
                    linear_velocity[body_b] = wp.vec4(linear_b[0], linear_b[1], linear_b[2], amb[3])
                    angular_velocity[body_b] = wp.vec4(angular_b[0], angular_b[1], angular_b[2], 0.0)
                impulse[packed] = wp.vec4(new_n, new_t1, new_t2, 0.0)
                slot += block_dim
            _block_sync()
            color += wp.int32(1)
        iteration += wp.int32(1)


@wp.kernel(enable_backward=False)
def integrate_poses_packed_kernel(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
    dt: wp.float32,
    body_qd: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    linear = _xyz(linear_velocity[body])
    angular = _xyz(angular_velocity[body])
    body_qd[body] = wp.spatial_vector(linear, angular)
    if (body_flags[body] & BodyFlags.KINEMATIC) != wp.int32(0):
        return
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    com_world = wp.transform_point(pose, body_com[body]) + linear * dt
    rotation = wp.normalize(rotation + wp.quat(angular, wp.float32(0.0)) * rotation * wp.float32(0.5) * dt)
    translation = com_world - wp.quat_rotate(rotation, body_com[body])
    body_q[body] = wp.transform(translation, rotation)


def make_solve_packed_contacts_shared_kernel(bodies_per_world: int):
    """Create a contact solver that retains one world's body state in shared memory."""
    module = wp.get_module(f"mini_packed_shared_{bodies_per_world}")

    @wp.kernel(module=module, enable_backward=False)
    def solve_packed_contacts_shared_kernel(
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
        linear_velocity: wp.array[wp.vec4],
        angular_velocity: wp.array[wp.vec4],
        inertia0: wp.array[wp.vec4],
        inertia1: wp.array[wp.vec4],
        inertia2: wp.array[wp.vec4],
    ):
        world, lane = wp.tid()
        body_base = world * bodies_per_world
        linear_tile = wp.tile_load(linear_velocity, shape=bodies_per_world, offset=body_base, storage="shared")
        angular_tile = wp.tile_load(angular_velocity, shape=bodies_per_world, offset=body_base, storage="shared")
        inertia0_tile = wp.tile_load(inertia0, shape=bodies_per_world, offset=body_base, storage="shared")
        inertia1_tile = wp.tile_load(inertia1, shape=bodies_per_world, offset=body_base, storage="shared")
        inertia2_tile = wp.tile_load(inertia2, shape=bodies_per_world, offset=body_base, storage="shared")

        num_colors = wp.min(world_num_colors[world], max_colors)
        iteration = wp.int32(0)
        while iteration < iterations:
            color = wp.int32(0)
            while color < num_colors:
                color_index = world * max_colors + color
                count = wp.min(world_color_count[color_index], color_capacity)
                if lane < count:
                    packed = world * capacity + world_color_offset[color_index] + lane
                    bodies = body_pair[packed]
                    body_a = bodies[0]
                    body_b = bodies[1]
                    local_a = body_a - body_base
                    local_b = body_b - body_base
                    nb = normal_bias[packed]
                    tm = tangent_mu[packed]
                    ama = arm_mass_a[packed]
                    amb = arm_mass_b[packed]
                    mass = effective_mass[packed]
                    lambdas = impulse[packed]
                    normal = _xyz(nb)
                    tangent1 = _xyz(tm)
                    tangent2 = wp.cross(normal, tangent1)
                    arm_a = _xyz(ama)
                    arm_b = _xyz(amb)
                    linear_a = wp.vec3(0.0)
                    angular_a = wp.vec3(0.0)
                    linear_b = wp.vec3(0.0)
                    angular_b = wp.vec3(0.0)
                    row0_a = wp.vec4(0.0)
                    row1_a = wp.vec4(0.0)
                    row2_a = wp.vec4(0.0)
                    row0_b = wp.vec4(0.0)
                    row1_b = wp.vec4(0.0)
                    row2_b = wp.vec4(0.0)
                    if body_a >= wp.int32(0):
                        linear_a = _xyz(wp.tile_extract(linear_tile, local_a))
                        angular_a = _xyz(wp.tile_extract(angular_tile, local_a))
                        row0_a = wp.tile_extract(inertia0_tile, local_a)
                        row1_a = wp.tile_extract(inertia1_tile, local_a)
                        row2_a = wp.tile_extract(inertia2_tile, local_a)
                    if body_b >= wp.int32(0):
                        linear_b = _xyz(wp.tile_extract(linear_tile, local_b))
                        angular_b = _xyz(wp.tile_extract(angular_tile, local_b))
                        row0_b = wp.tile_extract(inertia0_tile, local_b)
                        row1_b = wp.tile_extract(inertia1_tile, local_b)
                        row2_b = wp.tile_extract(inertia2_tile, local_b)

                    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                    new_n = wp.max(
                        lambdas[0] - mass[0] * (wp.dot(relative, normal) + nb[3]),
                        wp.float32(0.0),
                    )
                    delta = new_n - lambdas[0]
                    if body_a >= wp.int32(0):
                        linear_a -= normal * (delta * ama[3])
                        angular_a -= _mul_inertia_rows(row0_a, row1_a, row2_a, wp.cross(arm_a, normal) * delta)
                    if body_b >= wp.int32(0):
                        linear_b += normal * (delta * amb[3])
                        angular_b += _mul_inertia_rows(row0_b, row1_b, row2_b, wp.cross(arm_b, normal) * delta)

                    limit = tm[3] * new_n
                    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                    new_t1 = wp.clamp(lambdas[1] - mass[1] * wp.dot(relative, tangent1), -limit, limit)
                    delta = new_t1 - lambdas[1]
                    if body_a >= wp.int32(0):
                        linear_a -= tangent1 * (delta * ama[3])
                        angular_a -= _mul_inertia_rows(row0_a, row1_a, row2_a, wp.cross(arm_a, tangent1) * delta)
                    if body_b >= wp.int32(0):
                        linear_b += tangent1 * (delta * amb[3])
                        angular_b += _mul_inertia_rows(row0_b, row1_b, row2_b, wp.cross(arm_b, tangent1) * delta)

                    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
                    new_t2 = wp.clamp(lambdas[2] - mass[2] * wp.dot(relative, tangent2), -limit, limit)
                    delta = new_t2 - lambdas[2]
                    if body_a >= wp.int32(0):
                        linear_a -= tangent2 * (delta * ama[3])
                        angular_a -= _mul_inertia_rows(row0_a, row1_a, row2_a, wp.cross(arm_a, tangent2) * delta)
                    if body_b >= wp.int32(0):
                        linear_b += tangent2 * (delta * amb[3])
                        angular_b += _mul_inertia_rows(row0_b, row1_b, row2_b, wp.cross(arm_b, tangent2) * delta)

                    wp.tile_scatter_masked(
                        linear_tile,
                        local_a,
                        wp.vec4(linear_a[0], linear_a[1], linear_a[2], ama[3]),
                        body_a >= wp.int32(0),
                    )
                    wp.tile_scatter_masked(
                        angular_tile,
                        local_a,
                        wp.vec4(angular_a[0], angular_a[1], angular_a[2], 0.0),
                        body_a >= wp.int32(0),
                    )
                    wp.tile_scatter_masked(
                        linear_tile,
                        local_b,
                        wp.vec4(linear_b[0], linear_b[1], linear_b[2], amb[3]),
                        body_b >= wp.int32(0),
                    )
                    wp.tile_scatter_masked(
                        angular_tile,
                        local_b,
                        wp.vec4(angular_b[0], angular_b[1], angular_b[2], 0.0),
                        body_b >= wp.int32(0),
                    )
                    impulse[packed] = wp.vec4(new_n, new_t1, new_t2, 0.0)
                _block_sync()
                color += wp.int32(1)
            iteration += wp.int32(1)

        wp.tile_store(linear_velocity, linear_tile, offset=body_base)
        wp.tile_store(angular_velocity, angular_tile, offset=body_base)

    return solve_packed_contacts_shared_kernel


@wp.func
def _prepare_packed_contact(
    contact: wp.int32,
    packed: wp.int32,
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
):
    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    body_a = wp.int32(-1)
    body_b = wp.int32(-1)
    if shape_a >= wp.int32(0):
        body_a = shape_body[shape_a]
    if shape_b >= wp.int32(0):
        body_b = shape_body[shape_b]

    pose_a = wp.transform_identity()
    pose_b = wp.transform_identity()
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    inv_mass_a = wp.float32(0.0)
    inv_mass_b = wp.float32(0.0)
    if body_a >= wp.int32(0):
        pose_a = body_q[body_a]
        com_a = wp.transform_point(pose_a, body_com[body_a])
        inv_mass_a = body_inv_mass[body_a]
    if body_b >= wp.int32(0):
        pose_b = body_q[body_b]
        com_b = wp.transform_point(pose_b, body_com[body_b])
        inv_mass_b = body_inv_mass[body_b]

    normal = contact_normal[contact]
    point_a = wp.transform_point(pose_a, contact_point0[contact]) + contact_margin0[contact] * normal
    point_b = wp.transform_point(pose_b, contact_point1[contact]) - contact_margin1[contact] * normal
    arm_a = point_a - com_a
    arm_b = point_b - com_b
    tangent1 = _contact_tangent(normal)
    tangent2 = wp.cross(normal, tangent1)
    separation = wp.dot(point_b - point_a, normal)
    bias = beta * wp.min(separation + slop, wp.float32(0.0)) / dt
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

    body_pair[packed] = wp.vec4i(body_a, body_b, 0, 0)
    normal_bias[packed] = wp.vec4(normal[0], normal[1], normal[2], bias)
    tangent_mu[packed] = wp.vec4(tangent1[0], tangent1[1], tangent1[2], mu)
    arm_mass_a[packed] = wp.vec4(arm_a[0], arm_a[1], arm_a[2], inv_mass_a)
    arm_mass_b[packed] = wp.vec4(arm_b[0], arm_b[1], arm_b[2], inv_mass_b)
    effective_mass[packed] = wp.vec4(
        _effective_mass(body_a, body_b, arm_a, arm_b, normal, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        _effective_mass(body_a, body_b, arm_a, arm_b, tangent1, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        _effective_mass(body_a, body_b, arm_a, arm_b, tangent2, inv_mass_a, inv_mass_b, inertia0, inertia1, inertia2),
        0.0,
    )
    impulse[packed] = wp.vec4(0.0)


@wp.kernel(enable_backward=False)
def prepare_world_contacts_kernel(
    block_dim: wp.int32,
    capacity: wp.int32,
    dt: wp.float32,
    beta: wp.float32,
    slop: wp.float32,
    world_constraint_count: wp.array[wp.int32],
    world_constraints: wp.array[wp.int32],
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
):
    tid = wp.tid()
    world = tid // block_dim
    slot = tid - world * block_dim
    count = wp.min(world_constraint_count[world], capacity)
    while slot < count:
        packed = world * capacity + slot
        contact = world_constraints[packed] - wp.int32(1)
        _prepare_packed_contact(
            contact,
            packed,
            dt,
            beta,
            slop,
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
        slot += block_dim


@wp.func
def _solve_packed_contact(
    packed: wp.int32,
    body_pair: wp.array[wp.vec4i],
    normal_bias: wp.array[wp.vec4],
    tangent_mu: wp.array[wp.vec4],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    effective_mass: wp.array[wp.vec4],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    impulse: wp.array[wp.vec4],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
):
    bodies = body_pair[packed]
    body_a = bodies[0]
    body_b = bodies[1]
    nb = normal_bias[packed]
    tm = tangent_mu[packed]
    ama = arm_mass_a[packed]
    amb = arm_mass_b[packed]
    mass = effective_mass[packed]
    lambdas = impulse[packed]
    normal = _xyz(nb)
    tangent1 = _xyz(tm)
    tangent2 = wp.cross(normal, tangent1)
    arm_a = _xyz(ama)
    arm_b = _xyz(amb)
    linear_a = wp.vec3(0.0)
    angular_a = wp.vec3(0.0)
    linear_b = wp.vec3(0.0)
    angular_b = wp.vec3(0.0)
    if body_a >= wp.int32(0):
        linear_a = _xyz(linear_velocity[body_a])
        angular_a = _xyz(angular_velocity[body_a])
    if body_b >= wp.int32(0):
        linear_b = _xyz(linear_velocity[body_b])
        angular_b = _xyz(angular_velocity[body_b])

    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
    new_n = wp.max(lambdas[0] - mass[0] * (wp.dot(relative, normal) + nb[3]), wp.float32(0.0))
    delta = new_n - lambdas[0]
    if body_a >= wp.int32(0):
        linear_a -= normal * (delta * ama[3])
        angular_a -= _mul_world_inertia(body_a, wp.cross(arm_a, normal) * delta, inertia0, inertia1, inertia2)
    if body_b >= wp.int32(0):
        linear_b += normal * (delta * amb[3])
        angular_b += _mul_world_inertia(body_b, wp.cross(arm_b, normal) * delta, inertia0, inertia1, inertia2)

    limit = tm[3] * new_n
    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
    new_t1 = wp.clamp(lambdas[1] - mass[1] * wp.dot(relative, tangent1), -limit, limit)
    delta = new_t1 - lambdas[1]
    if body_a >= wp.int32(0):
        linear_a -= tangent1 * (delta * ama[3])
        angular_a -= _mul_world_inertia(body_a, wp.cross(arm_a, tangent1) * delta, inertia0, inertia1, inertia2)
    if body_b >= wp.int32(0):
        linear_b += tangent1 * (delta * amb[3])
        angular_b += _mul_world_inertia(body_b, wp.cross(arm_b, tangent1) * delta, inertia0, inertia1, inertia2)

    relative = linear_b + wp.cross(angular_b, arm_b) - linear_a - wp.cross(angular_a, arm_a)
    new_t2 = wp.clamp(lambdas[2] - mass[2] * wp.dot(relative, tangent2), -limit, limit)
    delta = new_t2 - lambdas[2]
    if body_a >= wp.int32(0):
        linear_a -= tangent2 * (delta * ama[3])
        angular_a -= _mul_world_inertia(body_a, wp.cross(arm_a, tangent2) * delta, inertia0, inertia1, inertia2)
        linear_velocity[body_a] = wp.vec4(linear_a[0], linear_a[1], linear_a[2], ama[3])
        angular_velocity[body_a] = wp.vec4(angular_a[0], angular_a[1], angular_a[2], 0.0)
    if body_b >= wp.int32(0):
        linear_b += tangent2 * (delta * amb[3])
        angular_b += _mul_world_inertia(body_b, wp.cross(arm_b, tangent2) * delta, inertia0, inertia1, inertia2)
        linear_velocity[body_b] = wp.vec4(linear_b[0], linear_b[1], linear_b[2], amb[3])
        angular_velocity[body_b] = wp.vec4(angular_b[0], angular_b[1], angular_b[2], 0.0)
    impulse[packed] = wp.vec4(new_n, new_t1, new_t2, 0.0)


@wp.kernel(enable_backward=False)
def solve_serial_worlds_kernel(
    capacity: wp.int32,
    iterations: wp.int32,
    world_constraint_count: wp.array[wp.int32],
    body_pair: wp.array[wp.vec4i],
    normal_bias: wp.array[wp.vec4],
    tangent_mu: wp.array[wp.vec4],
    arm_mass_a: wp.array[wp.vec4],
    arm_mass_b: wp.array[wp.vec4],
    effective_mass: wp.array[wp.vec4],
    inertia0: wp.array[wp.vec4],
    inertia1: wp.array[wp.vec4],
    inertia2: wp.array[wp.vec4],
    impulse: wp.array[wp.vec4],
    linear_velocity: wp.array[wp.vec4],
    angular_velocity: wp.array[wp.vec4],
):
    world = wp.tid()
    count = wp.min(world_constraint_count[world], capacity)
    iteration = wp.int32(0)
    while iteration < iterations:
        slot = wp.int32(0)
        while slot < count:
            _solve_packed_contact(
                world * capacity + slot,
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
            slot += wp.int32(1)
        iteration += wp.int32(1)
