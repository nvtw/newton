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
    cylinder_type: int,
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

        stype = shape_type[shape_index]
        if stype == box_type or stype == cylinder_type:
            # Box and cylinder are both intersections of a few half-spaces, so
            # the solid's share of the kernel ball is the product of the
            # individual shares -- exactly 1/2 on a face, 1/4 on an edge or rim,
            # 1/8 at a corner. A single closest-point normal instead implies one
            # half-space everywhere, over-stating it 2x on an edge and 4x at a
            # corner, which pushes fluid away from those features.
            X_ws = wp.transform_multiply(X_wb, shape_transform[shape_index])
            local = wp.transform_point(wp.transform_inverse(X_ws), px)
            extent = shape_scale[shape_index]

            # Up to three constraints, each a signed distance and an outward
            # direction. Unused slots sit far inside and contribute a factor 1.
            d = wp.vec3(-1.0e6, -1.0e6, -1.0e6)
            a0 = wp.vec3(0.0)
            a1 = wp.vec3(0.0)
            a2 = wp.vec3(0.0)
            if stype == box_type:
                d = wp.vec3(
                    wp.abs(local[0]) - extent[0],
                    wp.abs(local[1]) - extent[1],
                    wp.abs(local[2]) - extent[2],
                )
                a0 = wp.vec3(wp.sign(local[0]), 0.0, 0.0)
                a1 = wp.vec3(0.0, wp.sign(local[1]), 0.0)
                a2 = wp.vec3(0.0, 0.0, wp.sign(local[2]))
            else:
                radial = wp.sqrt(local[0] * local[0] + local[1] * local[1])
                d = wp.vec3(radial - extent[0], wp.abs(local[2]) - extent[1], -1.0e6)
                if radial > 1.0e-9:
                    a0 = wp.vec3(local[0] / radial, local[1] / radial, 0.0)
                a1 = wp.vec3(0.0, 0.0, wp.sign(local[2]))

            f0 = _half_space_fraction(d[0], inv_radius)
            f1 = _half_space_fraction(d[1], inv_radius)
            f2 = _half_space_fraction(d[2], inv_radius)
            frac = f0 * f1 * f2
            if frac <= 0.0:
                continue

            # Product rule: each constraint's derivative times the other factors.
            grad_local = (
                a0 * (f1 * f2 * _dpsi(wp.abs(d[0]) * inv_radius) * inv_radius)
                + a1 * (f0 * f2 * _dpsi(wp.abs(d[1]) * inv_radius) * inv_radius)
                + a2 * (f0 * f1 * _dpsi(wp.abs(d[2]) * inv_radius) * inv_radius)
            )

            wp.atomic_add(boundary_log, particle_index, -wp.log(1.0 - wp.min(frac, 0.999)))
            wp.atomic_add(
                boundary_grad, particle_index, wp.transform_vector(X_ws, grad_local) * rest_density
            )
            continue

        # Distance from the particle centre (not its surface) to the solid,
        # since the density integral is evaluated at the particle's sample point.
        q = wp.clamp(wp.dot(n, px - bx) * inv_radius, 0.0, 1.0)
        if q >= 1.0:
            continue

        wp.atomic_add(boundary_log, particle_index, -wp.log(1.0 - _psi(q)))
        wp.atomic_add(boundary_grad, particle_index, n * (rest_density * inv_radius * _dpsi(q)))


# --------------------------------------------------------------------------
# Neighbour grid
#
# A counting-sort uniform grid, owned here rather than using Warp's HashGrid.
# Two properties the stock grid cannot give us and that the fluid step needs:
# its queries return original particle indices, forcing a scattered read per
# candidate, and it does not expose cell ranges, so a 27-cell neighbourhood
# costs 27 hash lookups.
#
# Here the cell lattice is fixed: the cell is the kernel support radius and
# cells are anchored at the world origin, so a cell coordinate is just
# floor(x / h) -- no domain to size, no bounds pass, and no dependence on where
# the fluid is or how far it has spread. The lattice is unbounded, so the flat
# index folds it onto a fixed torus: each axis wraps modulo the table extent.
# Aliased cells share a slot, which only adds candidates -- the distance test
# rejects them, and the kernel has compact support, so aliased particles cannot
# influence each other. Wrong answers are impossible; only the candidate count
# grows, and gracefully.
#
# Wrapping rather than hashing is what makes the fold safe. Two cells alias only
# when they are a whole table extent apart, and every extent is at least 3, so
# two cells of the same 3x3x3 neighbourhood can never alias. That matters: a
# neighbour reached through two aliased cells of one neighbourhood would be
# counted twice, and no distance test can catch it -- it is a genuine neighbour,
# just found twice. A hash gives no such guarantee.
#
# Keeping x contiguous within a row is what buys the traversal: three cells in a
# row are one span, so the search walks 9 spans instead of 27 lookups. The build
# also leaves positions in cell order, so the search reads them sequentially.
#
# A finer cell fits the support more tightly -- splitting h into k cells searches
# a region (2 + 1/k)h across instead of 3h -- but costs (2k+1)^2 spans instead of
# 9, and each span boundary is a dependent, uncoalesced offset load. Measured on
# the dam break at 216k particles, that trade loses: 175 ms/frame at one cell per
# support radius against 235 at a half and 315 at a third. So the cell stays at
# the support radius, which at the default rest spacing holds ~5 particles.
#
# Measured against the stock grid on the same particle set, producing identical
# neighbour sets: 2.4x overall at 216k particles and 2.2x at 439k.
# --------------------------------------------------------------------------


@wp.func
def _cell_coord(p: wp.vec3, inv_cell: float) -> wp.vec3i:
    """Cell containing ``p`` on the lattice anchored at the world origin."""
    return wp.vec3i(
        int(wp.floor(p[0] * inv_cell)),
        int(wp.floor(p[1] * inv_cell)),
        int(wp.floor(p[2] * inv_cell)),
    )


@wp.func
def _wrap(i: int, n: int) -> int:
    return ((i % n) + n) % n


@wp.func
def _row_index(iy: int, iz: int, dim: wp.vec3i) -> int:
    """Flat index of the start of a cell row, x-major so the row is contiguous."""
    return (_wrap(iz, dim[2]) * dim[1] + _wrap(iy, dim[1])) * dim[0]


@wp.kernel
def count_particles_per_cell(
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    inv_cell: float,
    dim: wp.vec3i,
    cell_of: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    i = wp.tid()
    if not _is_active_fluid(particle_flags[i]):
        cell_of[i] = -1
        return
    c = _cell_coord(particle_q[i], inv_cell)
    cell = _row_index(c[1], c[2], dim) + _wrap(c[0], dim[0])
    cell_of[i] = cell
    wp.atomic_add(counts, cell, 1)


@wp.kernel
def scatter_particles_to_cells(
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    cell_of: wp.array[wp.int32],
    offset: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    sorted_to_orig: wp.array[wp.int32],
    pos_sorted: wp.array[wp.vec4],
):
    """Place each particle in its cell's slice, leaving positions cell-ordered.

    Mass rides in ``w``: the SI density sum needs a neighbour's mass, and packed
    here it costs no extra gather.
    """
    i = wp.tid()
    c = cell_of[i]
    if c < 0:
        return
    slot = offset[c] + wp.atomic_add(cursor, c, 1)
    sorted_to_orig[slot] = i
    q = particle_q[i]
    pos_sorted[slot] = wp.vec4(q[0], q[1], q[2], particle_mass[i])



@wp.kernel
def gather_sorted_positions(
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    sorted_to_orig: wp.array[wp.int32],
    pos_sorted: wp.array[wp.vec4],
):
    """Refresh the slot-ordered positions without rebuilding the grid.

    Positions move every solver iteration but the cell ordering only changes
    per substep, so this O(N) pass keeps the neighbour reads sequential.
    """
    slot = wp.tid()
    i = sorted_to_orig[slot]
    if i < 0:
        return
    q = particle_q[i]
    pos_sorted[slot] = wp.vec4(q[0], q[1], q[2], particle_mass[i])

@wp.kernel
def build_neighbor_list(
    pos_sorted: wp.array[wp.vec4],
    offset: wp.array[wp.int32],
    inv_cell: float,
    dim: wp.vec3i,
    fluid_count: wp.array[wp.int32],
    contact_distance_sq: float,
    max_neighbors: int,
    num_particles: int,
    neighbors: wp.array[wp.int32],
    neighbor_counts: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """Cache each fluid particle's neighbours once per substep.

    The density, pressure and vorticity kernels each replay this list several
    times per substep, so the traversal is done once. Neighbours are stored as
    slots, not particle indices, so every downstream gather stays cell-local.
    ``overflow`` counts neighbours dropped past ``max_neighbors``, making
    truncation observable rather than silently altering the physics.
    """
    slot = wp.tid()
    if slot >= fluid_count[0]:
        return

    pm = pos_sorted[slot]
    xi = wp.vec3(pm[0], pm[1], pm[2])
    c = _cell_coord(xi, inv_cell)

    # The three x columns of the search cube are consecutive modulo dim[0], so a
    # row costs one span, or two where it wraps.
    mx = _wrap(c[0] - 1, dim[0])
    split = wp.max(mx + 3 - dim[0], 0)

    count = int(0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            base = _row_index(c[1] + dy, c[2] + dz, dim)
            # offset is an exclusive prefix sum, hence monotonic, so one span
            # covers a run of cells even where some of them are empty.
            for k in range(offset[base + mx], offset[base + mx + 3 - split]):
                if k != slot:
                    pj = pos_sorted[k]
                    d = wp.length_sq(xi - wp.vec3(pj[0], pj[1], pj[2]))
                    if d < contact_distance_sq and d > 1.0e-12:
                        if count < max_neighbors:
                            neighbors[count * num_particles + slot] = k
                            count += 1
                        else:
                            wp.atomic_add(overflow, 0, 1)
            for k in range(offset[base], offset[base + split]):
                if k != slot:
                    pj = pos_sorted[k]
                    d = wp.length_sq(xi - wp.vec3(pj[0], pj[1], pj[2]))
                    if d < contact_distance_sq and d > 1.0e-12:
                        if count < max_neighbors:
                            neighbors[count * num_particles + slot] = k
                            count += 1
                        else:
                            wp.atomic_add(overflow, 0, 1)

    neighbor_counts[slot] = count


@wp.kernel
def calculate_density(
    sorted_to_orig: wp.array[wp.int32],
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
    i = sorted_to_orig[slot]
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
    sorted_to_orig: wp.array[wp.int32],
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
    i = sorted_to_orig[slot]
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
    sorted_to_orig: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    pos_sorted: wp.array[wp.vec4],
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
    i = sorted_to_orig[slot]
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
    sorted_to_orig: wp.array[wp.int32],
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
    i = sorted_to_orig[slot]
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
    sorted_to_orig: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
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
    i = sorted_to_orig[slot]
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
