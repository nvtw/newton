# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Small CUDA spatial-row reference; serial within each independent slab.

This intentionally tests copy algebra, not throughput. No production solver
dispatch, physical mass, row, or default is modified.
"""

import warp as wp

from newton._src.solvers.phoenx.solver_phoenx_kernels import _sync_threads


@wp.kernel
def solve_slabs(
    row_order: wp.array[wp.int32],
    row_slab: wp.array[wp.int32],
    endpoints: wp.array[wp.vec2i],
    linear0: wp.array[wp.vec3],
    angular0: wp.array[wp.vec3],
    linear1: wp.array[wp.vec3],
    angular1: wp.array[wp.vec3],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia: wp.array[wp.mat33],
    count: wp.array[wp.int32],
    input_linear: wp.array[wp.vec3],
    input_angular: wp.array[wp.vec3],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    target: wp.array[wp.float32],
    velocity: wp.array2d[wp.vec3],
    omega: wp.array2d[wp.vec3],
    impulses: wp.array[wp.float32],
):
    slab = wp.tid()
    for body in range(input_linear.shape[0]):
        velocity[slab, body] = input_linear[body]
        omega[slab, body] = input_angular[body]
    for index in range(row_order.shape[0]):
        row = row_order[index]
        if row_slab[row] == slab:
            a = endpoints[row][0]
            b = endpoints[row][1]
            wa = inverse_mass[a] * wp.float32(count[a])
            wb = inverse_mass[b] * wp.float32(count[b])
            ia = inverse_inertia[a] * wp.float32(count[a])
            ib = inverse_inertia[b] * wp.float32(count[b])
            ja = angular0[row]
            jb = angular1[row]
            na = linear0[row]
            nb = linear1[row]
            response = wa * wp.dot(na, na) + wb * wp.dot(nb, nb) + wp.dot(ja, ia * ja) + wp.dot(jb, ib * jb)
            residual = (
                wp.dot(na, velocity[slab, a])
                + wp.dot(nb, velocity[slab, b])
                + wp.dot(ja, omega[slab, a])
                + wp.dot(jb, omega[slab, b])
                - target[row]
            )
            impulse = wp.clamp(-residual / response, lower[row], upper[row])
            impulses[row] = impulse
            velocity[slab, a] = velocity[slab, a] + wa * na * impulse
            velocity[slab, b] = velocity[slab, b] + wb * nb * impulse
            omega[slab, a] = omega[slab, a] + ia * ja * impulse
            omega[slab, b] = omega[slab, b] + ib * jb * impulse


@wp.kernel
def average_slabs(
    membership: wp.array2d[wp.int32],
    count: wp.array[wp.int32],
    velocity: wp.array2d[wp.vec3],
    omega: wp.array2d[wp.vec3],
    result_velocity: wp.array[wp.vec3],
    result_omega: wp.array[wp.vec3],
):
    body = wp.tid()
    if count[body] > 0:
        v = wp.vec3(0.0)
        w = wp.vec3(0.0)
        for slab in range(membership.shape[0]):
            if membership[slab, body] != 0:
                v += velocity[slab, body]
                w += omega[slab, body]
        result_velocity[body] = v / wp.float32(count[body])
        result_omega[body] = w / wp.float32(count[body])


@wp.kernel
def solve_slabs_colored(
    row_color: wp.array[wp.int32],
    colors_per_slab: wp.int32,
    row_order: wp.array[wp.int32],
    row_slab: wp.array[wp.int32],
    endpoints: wp.array[wp.vec2i],
    linear0: wp.array[wp.vec3],
    angular0: wp.array[wp.vec3],
    linear1: wp.array[wp.vec3],
    angular1: wp.array[wp.vec3],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia: wp.array[wp.mat33],
    count: wp.array[wp.int32],
    input_linear: wp.array[wp.vec3],
    input_angular: wp.array[wp.vec3],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    target: wp.array[wp.float32],
    velocity: wp.array2d[wp.vec3],
    omega: wp.array2d[wp.vec3],
    impulses: wp.array[wp.float32],
):
    slab, lane = wp.tid()
    for body in range(lane, input_linear.shape[0], 32):
        velocity[slab, body] = input_linear[body]
        omega[slab, body] = input_angular[body]
    _sync_threads()
    for color in range(colors_per_slab):
        for index in range(lane, row_order.shape[0], 32):
            row = row_order[index]
            if row_slab[row] == slab and row_color[row] == slab * colors_per_slab + color:
                a = endpoints[row][0]
                b = endpoints[row][1]
                wa = inverse_mass[a] * wp.float32(count[a])
                wb = inverse_mass[b] * wp.float32(count[b])
                ia = inverse_inertia[a] * wp.float32(count[a])
                ib = inverse_inertia[b] * wp.float32(count[b])
                ja = angular0[row]
                jb = angular1[row]
                na = linear0[row]
                nb = linear1[row]
                response = wa * wp.dot(na, na) + wb * wp.dot(nb, nb) + wp.dot(ja, ia * ja) + wp.dot(jb, ib * jb)
                residual = (
                    wp.dot(na, velocity[slab, a])
                    + wp.dot(nb, velocity[slab, b])
                    + wp.dot(ja, omega[slab, a])
                    + wp.dot(jb, omega[slab, b])
                    - target[row]
                )
                impulse = wp.clamp(-residual / response, lower[row], upper[row])
                impulses[row] = impulse
                velocity[slab, a] = velocity[slab, a] + wa * na * impulse
                velocity[slab, b] = velocity[slab, b] + wb * nb * impulse
                omega[slab, a] = omega[slab, a] + ia * ja * impulse
                omega[slab, b] = omega[slab, b] + ib * jb * impulse
        _sync_threads()
