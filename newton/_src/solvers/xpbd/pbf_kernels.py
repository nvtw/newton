# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def _is_active_fluid(flags: int) -> bool:
    return (flags & ParticleFlags.ACTIVE) != 0 and (flags & ParticleFlags.FLUID) != 0


@wp.func
def _kernel_w(distance: float, spiky1: float, inv_radius: float) -> float:
    q = 1.0 - distance * inv_radius
    return spiky1 * q * q


@wp.func
def _kernel_dw(distance: float, spiky2: float, inv_radius: float) -> float:
    return -spiky2 * (1.0 - distance * inv_radius)


@wp.func
def _cohesion_w(distance: float, cohesion1: float, cohesion2: float, inv_radius: float) -> float:
    q = distance * inv_radius
    return cohesion1 * q * q * q + cohesion2 * q * q - 1.0


@wp.kernel
def calculate_density(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    contact_distance_sq: float,
    inv_radius: float,
    spiky1: float,
    spiky2: float,
    rest_density: float,
    lambda_scale: float,
    surface_tension: float,
    densities: wp.array[float],
    surface_normals: wp.array[wp.vec3],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    radius = 1.0 / inv_radius
    query = wp.hash_grid_query(grid, xi, radius)
    j = int(0)
    density = float(0.0)
    normal = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid(particle_flags[j]):
            continue

        xij = xi - particle_q[j]
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue

        distance = wp.sqrt(distance_sq)
        density += _kernel_w(distance, spiky1, inv_radius)
        if surface_tension > 0.0:
            normal += _kernel_dw(distance, spiky2, inv_radius) * xij / distance

    constraint = wp.max(density - rest_density, -0.005 * rest_density)
    densities[i] = constraint * lambda_scale
    surface_normals[i] = normal * surface_tension


@wp.kernel
def solve_density(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_inv_mass: wp.array[float],
    densities: wp.array[float],
    surface_normals: wp.array[wp.vec3],
    accumulated_delta: wp.array[wp.vec3],
    contact_distance_sq: float,
    inv_radius: float,
    spiky1: float,
    spiky2: float,
    viscosity: float,
    inv_rest_density: float,
    cohesion: float,
    cohesion1: float,
    cohesion2: float,
    surface_tension: float,
    cfl_coefficient: float,
    coefficient: float,
    dt: float,
    deltas: wp.array[wp.vec3],
    weights: wp.array[float],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or not _is_active_fluid(particle_flags[i]) or particle_inv_mass[i] == 0.0:
        return

    xi = particle_q[i]
    delta_i = accumulated_delta[i]
    density_i = densities[i]
    normal_i = surface_normals[i]
    radius = 1.0 / inv_radius
    cfl_radius = radius * cfl_coefficient
    query = wp.hash_grid_query(grid, xi, radius)
    j = int(0)
    delta = wp.vec3(0.0)
    weight = float(0.0)

    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid(particle_flags[j]):
            continue

        xij = xi - particle_q[j]
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue

        distance = wp.sqrt(distance_sq)
        normal = xij / distance
        density_correction = 0.5 * (density_i + densities[j]) * _kernel_dw(distance, spiky2, inv_radius)
        cohesion_correction = cohesion * dt * _cohesion_w(distance, cohesion1, cohesion2, inv_radius)
        delta -= normal * (density_correction + cohesion_correction)

        relative_delta = delta_i - accumulated_delta[j]
        viscosity_amount = viscosity * dt * inv_rest_density * _kernel_w(distance, spiky1, inv_radius)
        viscosity_scale = 1.0 - 1.0 / (1.0 + viscosity_amount)
        delta -= viscosity_scale * relative_delta

        relative_normal_delta = wp.dot(normal, relative_delta)
        if relative_normal_delta < -cfl_radius:
            delta -= 0.5 * normal * (relative_normal_delta + cfl_radius)

        if surface_tension > 0.0:
            delta -= (normal_i - surface_normals[j]) * dt

        weight += 1.0

    deltas[i] = delta * coefficient
    weights[i] = weight


@wp.kernel
def apply_pbf_deltas(
    particle_flags: wp.array[wp.int32],
    deltas: wp.array[wp.vec3],
    weights: wp.array[float],
    relaxation: float,
    particle_q: wp.array[wp.vec3],
    accumulated_delta: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active_fluid(particle_flags[i]):
        return

    scale = 1.0 / wp.max(weights[i] * relaxation, 1.0)
    correction = deltas[i] * scale
    particle_q[i] += correction
    accumulated_delta[i] += correction


@wp.kernel
def finalize_pbf_velocities(
    particle_q: wp.array[wp.vec3],
    particle_q_initial: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    dt: float,
    max_velocity: float,
    particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if not _is_active_fluid(particle_flags[i]):
        return

    velocity = (particle_q[i] - particle_q_initial[i]) / dt
    speed = wp.length(velocity)
    if speed > max_velocity:
        velocity *= max_velocity / speed
    particle_qd[i] = velocity


@wp.kernel
def vorticity_confinement(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    contact_distance_sq: float,
    inv_radius: float,
    spiky2: float,
    curl: wp.array[wp.vec3],
    curl_magnitude: wp.array[float],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    query = wp.hash_grid_query(grid, xi, 1.0 / inv_radius)
    j = int(0)
    value = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid(particle_flags[j]):
            continue
        xij = xi - particle_q[j]
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        distance = wp.sqrt(distance_sq)
        gradient = _kernel_dw(distance, spiky2, inv_radius) * xij / distance
        value += wp.cross(particle_qd[j] - vi, gradient)

    curl[i] = value
    curl_magnitude[i] = wp.length(value)


@wp.kernel
def apply_vorticity(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    curl: wp.array[wp.vec3],
    curl_magnitude: wp.array[float],
    contact_distance_sq: float,
    inv_radius: float,
    spiky2: float,
    confinement: float,
    inv_rest_density: float,
    dt: float,
    particle_qd: wp.array[wp.vec3],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    query = wp.hash_grid_query(grid, xi, 1.0 / inv_radius)
    j = int(0)
    gradient = wp.vec3(0.0)
    weight = float(0.0)

    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid(particle_flags[j]):
            continue
        xij = xi - particle_q[j]
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        distance = wp.sqrt(distance_sq)
        gradient += curl_magnitude[j] * _kernel_dw(distance, spiky2, inv_radius) * xij / distance
        weight += 1.0

    direction = wp.normalize(gradient)
    impulse = dt * inv_rest_density * confinement * wp.cross(direction, curl[i])
    particle_qd[i] += impulse / wp.max(weight, 1.0)


@wp.kernel
def apply_damping(
    particle_flags: wp.array[wp.int32],
    damping: float,
    dt: float,
    particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if _is_active_fluid(particle_flags[i]):
        particle_qd[i] *= wp.max(1.0 - damping * dt, 0.0)
