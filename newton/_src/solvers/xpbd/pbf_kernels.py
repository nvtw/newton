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


@wp.func
def _psi(q: float) -> float:
    """Fraction of the kernel's weight lying inside a solid half-space.

    Closed form of ``int_solid W dV / int_ball W dV`` for the PBF spiky kernel
    ``W(r) = spiky1 (1 - r/h)^2``, with ``q = d/h`` the distance from the
    particle centre to the surface. Derived by integrating the projected slice
    moment ``A0(z) = 2 pi int_|z|^h t W(t) dt`` over the solid side:

        Psi(q) = 1/2 - (5/4) q + (5/2) q^3 - (5/2) q^4 + (3/4) q^5

    ``Psi(0) = 1/2`` exactly (half-space symmetry) and ``Psi(1) = Psi'(1) = 0``,
    so the boundary density vanishes smoothly at the support radius.
    """
    c = wp.clamp(q, 0.0, 1.0)
    return 0.5 + c * (-1.25 + c * c * (2.5 + c * (-2.5 + 0.75 * c)))


@wp.func
def _dpsi(q: float) -> float:
    """``d Psi / d q``; multiply by ``rest_density / h`` for ``d rho_B / d d``."""
    c = wp.clamp(q, 0.0, 1.0)
    return -1.25 + c * c * (7.5 + c * (-10.0 + 3.75 * c))


@wp.kernel
def accumulate_boundary_density(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    inv_radius: float,
    rest_density: float,
    boundary_log: wp.array[float],
    boundary_grad: wp.array[wp.vec3],
):
    """Accumulate the solid boundary's contribution to each fluid particle's density.

    A particle near a solid has part of its kernel support occupied by the
    solid, where there are no fluid neighbours to sum, so the plain neighbour
    sum under-estimates its density and the pressure solver fails to push back.
    ``rho_B = rest_density * Psi(d/h)`` supplies exactly the missing weight.

    Multiple boundaries are combined as a union, ``1 - prod(1 - Psi_i)``,
    accumulated here in log space so it composes with an atomic add. This is
    exact for a single boundary and for mutually orthogonal ones (the box and
    corner cases); it over-estimates for acute dihedrals. ``Psi <= 1/2`` always,
    so ``1 - Psi >= 1/2`` and the logarithm is well conditioned.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    if not _is_active_fluid(particle_flags[particle_index]):
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    n = contact_normal[tid]

    # Distance from the particle centre (not its surface) to the solid, since
    # the density integral is evaluated at the particle's sample point.
    q = wp.clamp(wp.dot(n, particle_q[particle_index] - bx) * inv_radius, 0.0, 1.0)
    if q >= 1.0:
        return

    wp.atomic_add(boundary_log, particle_index, -wp.log(1.0 - _psi(q)))
    wp.atomic_add(boundary_grad, particle_index, n * (rest_density * inv_radius * _dpsi(q)))


@wp.kernel
def build_neighbor_list(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    contact_distance_sq: float,
    radius: float,
    max_neighbors: int,
    num_particles: int,
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """Cache each fluid particle's neighbors once per substep.

    The density, pressure and vorticity kernels all traverse the neighborhood,
    several times per substep each. Walking the hash grid means visiting 27
    cells and rejecting most candidates every single time; doing that traversal
    once and replaying the surviving indices removes the dominant cost of the
    fluid step.

    Indices are stored strided (``k * num_particles + i``) so consecutive
    threads read consecutive addresses. ``overflow`` counts neighbors dropped
    because a particle exceeded ``max_neighbors``, so truncation is observable
    rather than silently altering the simulation.
    """
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    if not _is_active_fluid(particle_flags[i]):
        neighbor_counts[i] = 0
        return

    xi = particle_q[i]
    query = wp.hash_grid_query(grid, xi, radius)
    j = int(0)
    count = int(0)

    while wp.hash_grid_query_next(query, j):
        if j == i or not _is_active_fluid(particle_flags[j]):
            continue

        distance_sq = wp.length_sq(xi - particle_q[j])
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue

        if count < max_neighbors:
            neighbors[count * num_particles + i] = j
            count += 1
        else:
            wp.atomic_add(overflow, 0, 1)

    neighbor_counts[i] = count


@wp.kernel
def calculate_density(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
    contact_distance_sq: float,
    inv_radius: float,
    spiky1: float,
    spiky2: float,
    rest_density: float,
    lambda_scale: float,
    surface_tension: float,
    boundary_log: wp.array[float],
    densities: wp.array[float],
    surface_normals: wp.array[wp.vec3],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    # Solid boundaries occupy part of the kernel support and contribute no
    # neighbours; seed the sum with the density they stand in for. The union
    # form collapses to 0 when no boundary is in range (exp(0) - 1 == 0).
    density = rest_density * (1.0 - wp.exp(-boundary_log[i]))
    normal = wp.vec3(0.0)

    count = neighbor_counts[i]
    for k in range(count):
        j = neighbors[k * num_particles + i]

        xij = xi - particle_q[j]
        distance_sq = wp.length_sq(xij)
        # The list is built once per substep but positions move across the
        # iterations, so a cached neighbour may have drifted out of range.
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
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
    densities: wp.array[float],
    surface_normals: wp.array[wp.vec3],
    boundary_grad: wp.array[wp.vec3],
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
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return
    if particle_inv_mass[i] == 0.0:
        deltas[i] = wp.vec3(0.0)
        weights[i] = 0.0
        return

    xi = particle_q[i]
    delta_i = accumulated_delta[i]
    density_i = densities[i]
    normal_i = surface_normals[i]
    radius = 1.0 / inv_radius
    cfl_radius = radius * cfl_coefficient
    # Gradient of the boundary density term, the counterpart of the fluid
    # neighbour gradients below. Without it the constraint and its derivative
    # disagree and the solver overshoots against walls.
    delta = -boundary_grad[i] * density_i
    weight = float(0.0)

    count = neighbor_counts[i]
    for k in range(count):
        j = neighbors[k * num_particles + i]

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
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
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
    value = wp.vec3(0.0)

    count = neighbor_counts[i]
    for k in range(count):
        j = neighbors[k * num_particles + i]
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
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
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
    gradient = wp.vec3(0.0)
    weight = float(0.0)

    count = neighbor_counts[i]
    for k in range(count):
        j = neighbors[k * num_particles + i]
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
