# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def _is_active_fluid(flags: int) -> bool:
    return (flags & ParticleFlags.ACTIVE) != 0 and (flags & ParticleFlags.FLUID) != 0


# The PBF spiky kernel is not normalised: int W dV == 2, not 1. Dividing the
# mass-weighted sum by this turns it into a true density in kg/m^3.
KERNEL_VOLUME_INTEGRAL = wp.constant(2.0)


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
def _half_space_fraction(signed_distance: float, inv_radius: float) -> float:
    """Fraction of the kernel ball lying inside a half-space.

    ``signed_distance`` is from the particle to the bounding plane, positive when
    the particle is outside the solid. Inside, the solid covers all but the
    complementary cap, hence ``1 - Psi``.
    """
    q = signed_distance * inv_radius
    if q >= 0.0:
        return _psi(q)
    return 1.0 - _psi(-q)


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
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    box_type: int,
    contact_max: int,
    num_threads: int,
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
    # Grid-stride over the contacts actually produced this substep. The
    # soft-contact buffer is sized for every particle-shape pair, which for a
    # scene with many shapes is orders of magnitude larger than the real
    # contact count; launching one thread per slot would dominate the step.
    count = wp.min(contact_max, contact_count[0])

    for tid in range(wp.tid(), count, num_threads):
        particle_index = contact_particle[tid]
        if not _is_active_fluid(particle_flags[particle_index]):
            continue

        shape_index = contact_shape[tid]
        body_index = shape_body[shape_index]

        X_wb = wp.transform_identity()
        if body_index >= 0:
            X_wb = body_q[body_index]

        bx = wp.transform_point(X_wb, contact_body_pos[tid])
        n = contact_normal[tid]
        px = particle_q[particle_index]

        if shape_type[shape_index] == box_type:
            # A box is the intersection of three axis slabs, so the fraction of
            # the kernel ball inside it is the product of three half-space
            # fractions -- exactly 1/2 on a face, 1/4 on an edge, 1/8 at a
            # corner. Treating the nearest surface as a single half-space, as a
            # closest-point normal implies, over-estimates by 2x on an edge and
            # 8x at a corner, which shows up as fluid repelled from box edges.
            X_ws = wp.transform_multiply(X_wb, shape_transform[shape_index])
            local = wp.transform_point(wp.transform_inverse(X_ws), px)
            extent = shape_scale[shape_index]

            frac = float(1.0)
            for axis in range(3):
                frac *= _half_space_fraction(wp.abs(local[axis]) - extent[axis], inv_radius)
            if frac <= 0.0:
                continue

            # Gradient of the product, mapped back to world. Each axis
            # contributes its own derivative times the other two factors.
            grad_local = wp.vec3(0.0)
            for axis in range(3):
                other = float(1.0)
                for k in range(3):
                    if k != axis:
                        other *= _half_space_fraction(wp.abs(local[k]) - extent[k], inv_radius)
                d = wp.abs(local[axis]) - extent[axis]
                slope = _dpsi(wp.abs(d) * inv_radius) * inv_radius
                sign = 1.0
                if local[axis] < 0.0:
                    sign = -1.0
                grad_local[axis] = other * slope * sign
            grad_world = wp.transform_vector(X_ws, grad_local)

            wp.atomic_add(boundary_log, particle_index, -wp.log(1.0 - wp.min(frac, 0.999)))
            wp.atomic_add(boundary_grad, particle_index, grad_world * rest_density)
            continue

        # Distance from the particle centre (not its surface) to the solid,
        # since the density integral is evaluated at the particle's sample point.
        q = wp.clamp(wp.dot(n, px - bx) * inv_radius, 0.0, 1.0)
        if q >= 1.0:
            continue

        wp.atomic_add(boundary_log, particle_index, -wp.log(1.0 - _psi(q)))
        wp.atomic_add(boundary_grad, particle_index, n * (rest_density * inv_radius * _dpsi(q)))


@wp.kernel
def build_sorted_order(
    grid: wp.uint64,
    sorted_to_orig: wp.array[wp.int32],
    orig_to_sorted: wp.array[wp.int32],
):
    """Record the hash grid's own cell ordering as an explicit permutation.

    Spatially close particles land in close slots, so gathering neighbour data
    through slots instead of particle indices turns scattered reads into mostly
    local ones. The ordering comes free with the grid, which is already rebuilt
    once per substep -- nothing extra is sorted.
    """
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0:
        sorted_to_orig[slot] = -1
        return
    sorted_to_orig[slot] = i
    orig_to_sorted[i] = slot


@wp.kernel
def gather_sorted_positions(
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    sorted_to_orig: wp.array[wp.int32],
    pos_sorted: wp.array[wp.vec4],
):
    """Refresh slot-ordered position and mass.

    Positions move every iteration while the ordering only changes per substep,
    so this is the amortisation point: O(N) coalesced writes to avoid
    O(N * neighbours) scattered reads several times over.

    Mass rides along in ``w``. The SI density sum needs a neighbour's mass, and
    fetching it as ``particle_mass[sorted_to_orig[sj]]`` costs two scattered
    reads per neighbour; packed here it costs none, since the position read
    already brings it in.
    """
    slot = wp.tid()
    i = sorted_to_orig[slot]
    if i < 0:
        return
    q = particle_q[i]
    pos_sorted[slot] = wp.vec4(q[0], q[1], q[2], particle_mass[i])


@wp.kernel
def build_neighbor_list(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    orig_to_sorted: wp.array[wp.int32],
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
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0:
        return
    if not _is_active_fluid(particle_flags[i]):
        neighbor_counts[slot] = 0
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
            # Store the neighbour's slot, not its particle index, so downstream
            # gathers hit nearby addresses.
            neighbors[count * num_particles + slot] = orig_to_sorted[j]
            count += 1
        else:
            wp.atomic_add(overflow, 0, 1)

    neighbor_counts[slot] = count


@wp.kernel
def calculate_density(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    pos_sorted: wp.array[wp.vec4],
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
    pos_lambda: wp.array[wp.vec4],
    surface_normals: wp.array[wp.vec3],
):
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    # Solid boundaries occupy part of the kernel support and contribute no
    # neighbours; seed the sum with the density they stand in for. The union
    # form collapses to 0 when no boundary is in range (exp(0) - 1 == 0).
    # rest_density is in kg/m^3, so this seed already is too.
    density = rest_density * (1.0 - wp.exp(-boundary_log[i]))
    normal = wp.vec3(0.0)

    # Mass weighted and kernel normalised, so `density` is kg/m^3 and the
    # constraint is against a real material density rather than a kernel sum.
    inv_norm = 1.0 / KERNEL_VOLUME_INTEGRAL

    count = neighbor_counts[slot]
    for k in range(count):
        sj = neighbors[k * num_particles + slot]

        pm = pos_sorted[sj]
        xij = xi - wp.vec3(pm[0], pm[1], pm[2])
        distance_sq = wp.length_sq(xij)
        # The list is built once per substep but positions move across the
        # iterations, so a cached neighbour may have drifted out of range.
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue

        distance = wp.sqrt(distance_sq)
        density += pm[3] * _kernel_w(distance, spiky1, inv_radius) * inv_norm
        if surface_tension > 0.0:
            normal += _kernel_dw(distance, spiky2, inv_radius) * xij / distance

    # Include this particle's own kernel contribution, as an SPH density sum does.
    density += particle_mass[i] * _kernel_w(0.0, spiky1, inv_radius) * inv_norm
    constraint = wp.max(density - rest_density, -0.005 * rest_density)
    scaled = constraint * lambda_scale
    densities[i] = scaled
    # Neighbour position and lambda are always read together in the pressure
    # solve; packing them halves the scattered gathers in its inner loop.
    pos_lambda[slot] = wp.vec4(xi[0], xi[1], xi[2], scaled)
    surface_normals[slot] = normal * surface_tension


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
    pos_lambda: wp.array[wp.vec4],
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
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return
    if particle_inv_mass[i] == 0.0:
        deltas[slot] = wp.vec3(0.0)
        weights[slot] = 0.0
        return

    xi = particle_q[i]
    delta_i = accumulated_delta[slot]
    density_i = densities[i]
    normal_i = surface_normals[slot]
    radius = 1.0 / inv_radius
    cfl_radius = radius * cfl_coefficient
    # Gradient of the boundary density term, the counterpart of the fluid
    # neighbour gradients below. Without it the constraint and its derivative
    # disagree and the solver overshoots against walls.
    delta = -boundary_grad[i] * density_i
    weight = float(0.0)

    count = neighbor_counts[slot]
    for k in range(count):
        sj = neighbors[k * num_particles + slot]

        pl = pos_lambda[sj]
        xij = xi - wp.vec3(pl[0], pl[1], pl[2])
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue

        distance = wp.sqrt(distance_sq)
        normal = xij / distance
        density_correction = 0.5 * (density_i + pl[3]) * _kernel_dw(distance, spiky2, inv_radius)
        cohesion_correction = cohesion * dt * _cohesion_w(distance, cohesion1, cohesion2, inv_radius)
        delta -= normal * (density_correction + cohesion_correction)

        relative_delta = delta_i - accumulated_delta[sj]
        viscosity_amount = viscosity * dt * inv_rest_density * _kernel_w(distance, spiky1, inv_radius)
        viscosity_scale = 1.0 - 1.0 / (1.0 + viscosity_amount)
        delta -= viscosity_scale * relative_delta

        relative_normal_delta = wp.dot(normal, relative_delta)
        if relative_normal_delta < -cfl_radius:
            delta -= 0.5 * normal * (relative_normal_delta + cfl_radius)

        if surface_tension > 0.0:
            delta -= (normal_i - surface_normals[sj]) * dt

        weight += 1.0

    deltas[slot] = delta * coefficient
    weights[slot] = weight


@wp.kernel
def apply_pbf_deltas(
    particle_flags: wp.array[wp.int32],
    sorted_to_orig: wp.array[wp.int32],
    deltas: wp.array[wp.vec3],
    weights: wp.array[float],
    relaxation: float,
    particle_q: wp.array[wp.vec3],
    accumulated_delta: wp.array[wp.vec3],
):
    slot = wp.tid()
    i = sorted_to_orig[slot]
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    # Jacobi averaging: divide the accumulated correction by the number of
    # neighbours that contributed, then apply the relaxation factor. Relaxation
    # scales the correction, so values below 1 under-relax as the name implies.
    # It used to multiply the divisor instead, which inverted its sense --
    # pbf_relaxation=0.5 doubled the correction rather than halving it. At the
    # default of 1.0 both forms agree, and match PhysX's 1/max(weight, 1).
    scale = relaxation / wp.max(weights[slot], 1.0)
    correction = deltas[slot] * scale
    particle_q[i] += correction
    accumulated_delta[slot] += correction


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
    pos_sorted: wp.array[wp.vec4],
    sorted_to_orig: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
    contact_distance_sq: float,
    inv_radius: float,
    spiky2: float,
    curl: wp.array[wp.vec3],
    curl_magnitude: wp.array[float],
):
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    value = wp.vec3(0.0)

    count = neighbor_counts[slot]
    for k in range(count):
        sj = neighbors[k * num_particles + slot]
        j = sorted_to_orig[sj]
        pm = pos_sorted[sj]
        xij = xi - wp.vec3(pm[0], pm[1], pm[2])
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        distance = wp.sqrt(distance_sq)
        gradient = _kernel_dw(distance, spiky2, inv_radius) * xij / distance
        value += wp.cross(particle_qd[j] - vi, gradient)

    curl[slot] = value
    curl_magnitude[slot] = wp.length(value)


@wp.kernel
def apply_vorticity(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    curl: wp.array[wp.vec3],
    curl_magnitude: wp.array[float],
    pos_sorted: wp.array[wp.vec4],
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
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    gradient = wp.vec3(0.0)
    weight = float(0.0)

    count = neighbor_counts[slot]
    for k in range(count):
        sj = neighbors[k * num_particles + slot]
        pm = pos_sorted[sj]
        xij = xi - wp.vec3(pm[0], pm[1], pm[2])
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        distance = wp.sqrt(distance_sq)
        gradient += curl_magnitude[sj] * _kernel_dw(distance, spiky2, inv_radius) * xij / distance
        weight += 1.0

    direction = wp.normalize(gradient)
    impulse = dt * inv_rest_density * confinement * wp.cross(direction, curl[slot])
    particle_qd[i] += impulse / wp.max(weight, 1.0)


@wp.kernel
def apply_viscosity(
    grid: wp.uint64,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    pos_sorted: wp.array[wp.vec4],
    sorted_to_orig: wp.array[wp.int32],
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    num_particles: int,
    contact_distance_sq: float,
    inv_radius: float,
    spiky1: float,
    spiky2: float,
    dynamic_viscosity: float,
    dt: float,
    delta_qd: wp.array[wp.vec3],
):
    """Physical viscous acceleration, with ``dynamic_viscosity`` in Pa s.

    Morris et al. (1997) SPH viscosity:

        a_i = sum_j m_j * 2 mu / (rho_i rho_j) * (r_ij . grad W_ij) / |r_ij|^2 * v_ij

    with ``r_ij = x_i - x_j`` and ``v_ij = v_i - v_j``. Since
    ``r_ij . grad W_ij = r W'(r)`` and ``W' < 0``, the term pulls each particle's
    velocity toward its neighbours', which is what momentum diffusion does. It
    reproduces the closed-form Stokes decay of a sinusoidal shear,
    ``exp(-(mu/rho) k^2 t)``, so the coefficient is a real material property and
    not a solver knob.

    Densities are the true SPH mass-weighted sums in kg/m^3, so a scene set up in
    SI units gets SI behaviour: water is 1e-3 Pa s, olive oil 0.08, honey ~10.
    """
    slot = wp.tid()
    i = wp.hash_grid_point_id(grid, slot)
    if i < 0 or not _is_active_fluid(particle_flags[i]):
        return

    xi = particle_q[i]
    vi = particle_qd[i]
    count = neighbor_counts[slot]

    # Local density, mass weighted, including this particle's own contribution.
    # The PBF spiky kernel is not normalised -- it integrates to 2, not 1 -- so
    # the raw sum overstates density by that factor. Morris goes as 1/rho^2, so
    # leaving it in makes the viscosity four times too weak.
    rho_i = particle_mass[i] * _kernel_w(0.0, spiky1, inv_radius)
    for k in range(count):
        sj = neighbors[k * num_particles + slot]
        pm = pos_sorted[sj]
        distance_sq = wp.length_sq(xi - wp.vec3(pm[0], pm[1], pm[2]))
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        rho_i += pm[3] * _kernel_w(wp.sqrt(distance_sq), spiky1, inv_radius)

    rho_i *= 0.5  # normalise: int W dV == 2 for this kernel
    if rho_i <= 0.0:
        return

    accel = wp.vec3(0.0)
    for k in range(count):
        sj = neighbors[k * num_particles + slot]
        j = sorted_to_orig[sj]
        pm = pos_sorted[sj]
        xij = xi - wp.vec3(pm[0], pm[1], pm[2])
        distance_sq = wp.length_sq(xij)
        if distance_sq >= contact_distance_sq or distance_sq <= 1.0e-12:
            continue
        distance = wp.sqrt(distance_sq)
        # Neighbour density is approximated by the local one; the fluid is
        # near-incompressible by construction, so rho_j ~ rho_i holds well.
        coeff = (
            pm[3]
            * 2.0
            * dynamic_viscosity
            / (rho_i * rho_i)
            * _kernel_dw(distance, spiky2, inv_radius)
            / distance
        )
        accel += coeff * (vi - particle_qd[j])

    delta_qd[i] = accel * dt


@wp.kernel
def apply_velocity_delta(
    particle_flags: wp.array[wp.int32],
    delta_qd: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if _is_active_fluid(particle_flags[i]):
        particle_qd[i] += delta_qd[i]


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
