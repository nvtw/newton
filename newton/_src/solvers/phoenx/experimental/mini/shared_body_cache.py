# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native shared-body-cache experiment for the packed contact mini solver."""

import warp as wp

_SNIPPET = r"""
#if defined(__CUDA_ARCH__)
    using vec3 = wp::vec_t<3, wp::float32>;
    using vec4 = wp::vec_t<4, wp::float32>;
    constexpr int physical_block = 128;
    __shared__ vec4 shared_linear[physical_block];
    __shared__ vec4 shared_angular[physical_block];
    __shared__ vec4 shared_i0[physical_block];
    __shared__ vec4 shared_i1[physical_block];
    __shared__ vec4 shared_i2[physical_block];

    const int lane = tid & (logical_width - 1);
    const int world = tid / logical_width;
    const int local_world = threadIdx.x / logical_width;
    const int shared_base = local_world * bodies_per_world;
    const int body_base = world * bodies_per_world;
    if (lane < bodies_per_world) {
        const int shared_body = shared_base + lane;
        const int global_body = body_base + lane;
        shared_linear[shared_body] = linear_velocity[global_body];
        shared_angular[shared_body] = angular_velocity[global_body];
        shared_i0[shared_body] = inertia0[global_body];
        shared_i1[shared_body] = inertia1[global_body];
        shared_i2[shared_body] = inertia2[global_body];
    }

    const int warp_lane = threadIdx.x & 31;
    const int first_lane = warp_lane - lane;
    const unsigned lane_mask = logical_width == 32 ? 0xffffffffu : ((1u << logical_width) - 1u);
    const unsigned mask = lane_mask << first_lane;
    __syncwarp(mask);

    const int num_colors = wp::min(world_num_colors[world], max_colors);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int color = 0; color < num_colors; ++color) {
            const int color_index = world * max_colors + color;
            const int count = wp::min(world_color_count[color_index], color_capacity);
            if (lane < count) {
                const int offset = world_color_offset[color_index] + lane;
                const int tile = world / worlds_per_tile;
                const int world_in_tile = world - tile * worlds_per_tile;
                const int packed = tile * worlds_per_tile * capacity + offset * worlds_per_tile + world_in_tile;
                const auto bodies = body_pair[packed];
                const int body_a = bodies[0];
                const int body_b = bodies[1];
                const auto nb = normal_bias[packed];
                const auto tm = tangent_mu[packed];
                const auto ama = arm_mass_a[packed];
                const auto amb = arm_mass_b[packed];
                const auto mass = effective_mass[packed];
                const auto lambdas = impulse[packed];
                const vec3 normal(nb[0], nb[1], nb[2]);
                const vec3 tangent1(tm[0], tm[1], tm[2]);
                const vec3 tangent2 = wp::cross(normal, tangent1);
                const vec3 arm_a(ama[0], ama[1], ama[2]);
                const vec3 arm_b(amb[0], amb[1], amb[2]);

                vec3 linear_a(0.0f), angular_a(0.0f), linear_b(0.0f), angular_b(0.0f);
                float inv_mass_a = 0.0f, inv_mass_b = 0.0f;
                vec4 row0_a(0.0f), row1_a(0.0f), row2_a(0.0f);
                vec4 row0_b(0.0f), row1_b(0.0f), row2_b(0.0f);
                int cache_a = -1, cache_b = -1;
                if (body_a >= 0) {
                    cache_a = shared_base + body_a - body_base;
                    const auto value = shared_linear[cache_a];
                    linear_a = vec3(value[0], value[1], value[2]);
                    inv_mass_a = value[3];
                    const auto omega = shared_angular[cache_a];
                    angular_a = vec3(omega[0], omega[1], omega[2]);
                    row0_a = shared_i0[cache_a];
                    row1_a = shared_i1[cache_a];
                    row2_a = shared_i2[cache_a];
                }
                if (body_b >= 0) {
                    cache_b = shared_base + body_b - body_base;
                    const auto value = shared_linear[cache_b];
                    linear_b = vec3(value[0], value[1], value[2]);
                    inv_mass_b = value[3];
                    const auto omega = shared_angular[cache_b];
                    angular_b = vec3(omega[0], omega[1], omega[2]);
                    row0_b = shared_i0[cache_b];
                    row1_b = shared_i1[cache_b];
                    row2_b = shared_i2[cache_b];
                }

#define MUL_I(out, r0, r1, r2, value)                 out = vec3(                     r0[0] * value[0] + r0[1] * value[1] + r0[2] * value[2],                     r1[0] * value[0] + r1[1] * value[1] + r1[2] * value[2],                     r2[0] * value[0] + r2[1] * value[1] + r2[2] * value[2])

                vec3 relative = linear_b + wp::cross(angular_b, arm_b) - linear_a - wp::cross(angular_a, arm_a);
                const float new_n = wp::max(lambdas[0] - mass[0] * (wp::dot(relative, normal) + nb[3]), 0.0f);
                float delta = new_n - lambdas[0];
                if (body_a >= 0) {
                    linear_a -= normal * (delta * inv_mass_a);
                    const vec3 torque = wp::cross(arm_a, normal) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_a, row1_a, row2_a, torque);
                    angular_a -= angular_delta;
                }
                if (body_b >= 0) {
                    linear_b += normal * (delta * inv_mass_b);
                    const vec3 torque = wp::cross(arm_b, normal) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_b, row1_b, row2_b, torque);
                    angular_b += angular_delta;
                }

                const float limit = tm[3] * new_n;
                relative = linear_b + wp::cross(angular_b, arm_b) - linear_a - wp::cross(angular_a, arm_a);
                const float new_t1 = wp::clamp(lambdas[1] - mass[1] * wp::dot(relative, tangent1), -limit, limit);
                delta = new_t1 - lambdas[1];
                if (body_a >= 0) {
                    linear_a -= tangent1 * (delta * inv_mass_a);
                    const vec3 torque = wp::cross(arm_a, tangent1) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_a, row1_a, row2_a, torque);
                    angular_a -= angular_delta;
                }
                if (body_b >= 0) {
                    linear_b += tangent1 * (delta * inv_mass_b);
                    const vec3 torque = wp::cross(arm_b, tangent1) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_b, row1_b, row2_b, torque);
                    angular_b += angular_delta;
                }

                relative = linear_b + wp::cross(angular_b, arm_b) - linear_a - wp::cross(angular_a, arm_a);
                const float new_t2 = wp::clamp(lambdas[2] - mass[2] * wp::dot(relative, tangent2), -limit, limit);
                delta = new_t2 - lambdas[2];
                if (body_a >= 0) {
                    linear_a -= tangent2 * (delta * inv_mass_a);
                    const vec3 torque = wp::cross(arm_a, tangent2) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_a, row1_a, row2_a, torque);
                    angular_a -= angular_delta;
                }
                if (body_b >= 0) {
                    linear_b += tangent2 * (delta * inv_mass_b);
                    const vec3 torque = wp::cross(arm_b, tangent2) * delta;
                    vec3 angular_delta;
                    MUL_I(angular_delta, row0_b, row1_b, row2_b, torque);
                    angular_b += angular_delta;
                }
#undef MUL_I

                if (body_a >= 0) {
                    shared_linear[cache_a] = vec4(linear_a[0], linear_a[1], linear_a[2], inv_mass_a);
                    shared_angular[cache_a] = vec4(angular_a[0], angular_a[1], angular_a[2], 0.0f);
                }
                if (body_b >= 0) {
                    shared_linear[cache_b] = vec4(linear_b[0], linear_b[1], linear_b[2], inv_mass_b);
                    shared_angular[cache_b] = vec4(angular_b[0], angular_b[1], angular_b[2], 0.0f);
                }
                impulse[packed] = vec4(new_n, new_t1, new_t2, 0.0f);
            }
            __syncwarp(mask);
        }
    }
    if (lane < bodies_per_world) {
        const int shared_body = shared_base + lane;
        const int global_body = body_base + lane;
        linear_velocity[global_body] = shared_linear[shared_body];
        angular_velocity[global_body] = shared_angular[shared_body];
    }
#endif
"""


@wp.func_native(_SNIPPET)
def _solve_native(
    logical_width: wp.int32,
    bodies_per_world: wp.int32,
    worlds_per_tile: wp.int32,
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
    tid: wp.int32,
): ...


@wp.kernel(enable_backward=False, grid_stride=False)
def solve_packed_contacts_shared_kernel(
    logical_width: wp.int32,
    bodies_per_world: wp.int32,
    worlds_per_tile: wp.int32,
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
    _solve_native(
        logical_width,
        bodies_per_world,
        worlds_per_tile,
        capacity,
        max_colors,
        color_capacity,
        iterations,
        world_num_colors,
        world_color_count,
        world_color_offset,
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
        wp.tid(),
    )
