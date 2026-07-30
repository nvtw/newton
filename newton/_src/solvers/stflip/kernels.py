# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for sparse ST-FLIP particle/grid transfers."""

import warp as wp

from ...geometry import ParticleFlags
from .sparse_grid import (
    SparseGridData,
    sparse_grid_cell_index,
    sparse_grid_index_from_cell,
)

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.kernel(enable_backward=False)
def build_particle_active_mask(flags: wp.array[int], active: wp.array[int]):
    """Convert particle flags to the sparse builder's binary active mask."""
    particle = wp.tid()
    active[particle] = int((flags[particle] & _ACTIVE) != 0)


@wp.kernel(enable_backward=False)
def initialize_temporal_offsets(seed: int, offsets: wp.array[float]):
    """Initialize deterministic particle phase offsets in [-0.5, 0.5)."""
    particle = wp.tid()
    random = wp.rand_init(seed, particle)
    offsets[particle] = wp.randf(random) - 0.5


@wp.kernel(enable_backward=False)
def update_particle_clocks(
    flags: wp.array[int],
    offsets: wp.array[float],
    dt: float,
    age_in: wp.array[float],
    residual_out: wp.array[float],
    age_out: wp.array[float],
):
    """Store temporal residual and advance particle age."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        residual_out[particle] = 0.0
        age_out[particle] = age_in[particle]
        return
    residual_out[particle] = offsets[particle] * dt
    age_out[particle] = age_in[particle] + dt


@wp.func
def _grid_index(position: wp.vec3, offset: wp.vec3, inv_cell_size: float) -> wp.vec3i:
    p = position * inv_cell_size - offset
    return wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))


@wp.func
def _stencil_coordinates(grid: SparseGridData, base_index: int) -> tuple[int, int, int, int]:
    """Return the owner tile and local coordinates of a packed cell."""
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    tile = base_index // tile_volume
    local = base_index - tile * tile_volume
    local_x = local % grid.tile_size
    local_y = (local // grid.tile_size) % grid.tile_size
    local_z = local // (grid.tile_size * grid.tile_size)
    return tile, local_x, local_y, local_z


@wp.func
def _stencil_index(
    grid: SparseGridData,
    tile: int,
    local_x: int,
    local_y: int,
    local_z: int,
    x: int,
    y: int,
    z: int,
) -> int:
    return sparse_grid_cell_index(grid, tile, local_x + x, local_y + y, local_z + z)


@wp.func
def _scatter_component(
    grid: SparseGridData,
    position: wp.vec3,
    velocity: wp.vec3,
    affine: wp.mat33,
    mass: float,
    inv_cell_size: float,
    offset: wp.vec3,
    component: int,
    face_mass: wp.array[float],
    face_momentum: wp.array[float],
):
    base = _grid_index(position, offset, inv_cell_size)
    base_index = sparse_grid_index_from_cell(grid, base)
    if base_index < 0:
        return
    tile, local_x, local_y, local_z = _stencil_coordinates(grid, base_index)
    p_grid = position * inv_cell_size - offset
    fraction = p_grid - wp.vec3(base)
    weight_x = wp.vec2(1.0 - fraction[0], fraction[0])
    weight_y = wp.vec2(1.0 - fraction[1], fraction[1])
    weight_z = wp.vec2(1.0 - fraction[2], fraction[2])
    cell_size = 1.0 / inv_cell_size
    base_displacement = (wp.vec3(base) + offset) * cell_size - position
    for z in range(2):
        for y in range(2):
            for x in range(2):
                index = _stencil_index(grid, tile, local_x, local_y, local_z, x, y, z)
                if index >= 0:
                    displacement = base_displacement + wp.vec3(float(x), float(y), float(z)) * cell_size
                    w = weight_x[x] * weight_y[y] * weight_z[z]
                    affine_velocity = (
                        affine[component, 0] * displacement[0]
                        + affine[component, 1] * displacement[1]
                        + affine[component, 2] * displacement[2]
                    )
                    momentum = mass * w * (velocity[component] + affine_velocity)
                    wp.atomic_add(face_mass, 3 * index + component, mass * w)
                    wp.atomic_add(face_momentum, 3 * index + component, momentum)


@wp.kernel(enable_backward=False)
def particles_to_grid(
    grid: SparseGridData,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    masses: wp.array[float],
    inverse_masses: wp.array[float],
    flags: wp.array[int],
    affine: wp.array[wp.mat33],
    temporal_offsets: wp.array[float],
    inv_cell_size: float,
    dt: float,
    cell_mass: wp.array[float],
):
    """Transfer active particle mass and APIC momentum to a sparse MAC grid."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        return

    velocity = velocities[particle] + forces[particle] * inverse_masses[particle] * dt
    position = positions[particle] + velocity * (temporal_offsets[particle] * dt)
    mass = masses[particle]
    base = _grid_index(position, wp.vec3(0.5), inv_cell_size)
    base_index = sparse_grid_index_from_cell(grid, base)
    if base_index < 0:
        return
    tile, local_x, local_y, local_z = _stencil_coordinates(grid, base_index)
    p_grid = position * inv_cell_size - wp.vec3(0.5)
    fraction = p_grid - wp.vec3(base)
    weight_x = wp.vec2(1.0 - fraction[0], fraction[0])
    weight_y = wp.vec2(1.0 - fraction[1], fraction[1])
    weight_z = wp.vec2(1.0 - fraction[2], fraction[2])
    for z in range(2):
        for y in range(2):
            for x in range(2):
                index = _stencil_index(grid, tile, local_x, local_y, local_z, x, y, z)
                if index >= 0:
                    w = weight_x[x] * weight_y[y] * weight_z[z]
                    wp.atomic_add(cell_mass, index, mass * w)


@wp.kernel(enable_backward=False)
def particle_faces_to_grid(
    grid: SparseGridData,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    masses: wp.array[float],
    inverse_masses: wp.array[float],
    flags: wp.array[int],
    affine: wp.array[wp.mat33],
    temporal_offsets: wp.array[float],
    inv_cell_size: float,
    dt: float,
    face_mass: wp.array[float],
    face_momentum: wp.array[float],
):
    """Transfer one particle velocity component per thread to MAC faces."""
    thread = wp.tid()
    particle = thread // 3
    component = thread - 3 * particle
    if (flags[particle] & _ACTIVE) == 0:
        return

    offset = wp.vec3(0.5)
    if component == 0:
        offset = wp.vec3(0.0, 0.5, 0.5)
    elif component == 1:
        offset = wp.vec3(0.5, 0.0, 0.5)
    else:
        offset = wp.vec3(0.5, 0.5, 0.0)
    velocity = velocities[particle] + forces[particle] * inverse_masses[particle] * dt
    position = positions[particle] + velocity * (temporal_offsets[particle] * dt)
    _scatter_component(
        grid,
        position,
        velocity,
        affine[particle],
        masses[particle],
        inv_cell_size,
        offset,
        component,
        face_mass,
        face_momentum,
    )


@wp.kernel(enable_backward=False)
def normalize_grid(
    face_mass: wp.array[float],
    face_momentum: wp.array[float],
    gravity: wp.array[wp.vec3],
    dt: float,
    face_velocity: wp.array[float],
    face_velocity_old: wp.array[float],
):
    """Normalize face momentum and apply gravity."""
    index = wp.tid()
    for axis in range(3):
        face = 3 * index + axis
        mass = face_mass[face]
        velocity = 0.0
        if mass > 0.0:
            velocity = face_momentum[face] / mass
        face_velocity_old[face] = velocity
        if mass > 0.0:
            velocity += gravity[0][axis] * dt
        face_velocity[face] = velocity


@wp.func
def _is_liquid(cell_mass: wp.array[float], index: int, min_mass: float) -> bool:
    return index >= 0 and cell_mass[index] > min_mass


@wp.func
def _axis_aligned_neighbors(
    grid: SparseGridData,
    index: int,
    tile: int,
    x: int,
    y: int,
    z: int,
) -> tuple[int, int, int, int, int, int]:
    """Resolve six cell neighbors with contiguous interior fast paths."""
    tile_size = grid.tile_size
    tile_area = tile_size * tile_size
    x_lo = index - 1
    x_hi = index + 1
    y_lo = index - tile_size
    y_hi = index + tile_size
    z_lo = index - tile_area
    z_hi = index + tile_area
    if x == 0:
        x_lo = sparse_grid_cell_index(grid, tile, -1, y, z)
    if x == tile_size - 1:
        x_hi = sparse_grid_cell_index(grid, tile, tile_size, y, z)
    if y == 0:
        y_lo = sparse_grid_cell_index(grid, tile, x, -1, z)
    if y == tile_size - 1:
        y_hi = sparse_grid_cell_index(grid, tile, x, tile_size, z)
    if z == 0:
        z_lo = sparse_grid_cell_index(grid, tile, x, y, -1)
    if z == tile_size - 1:
        z_hi = sparse_grid_cell_index(grid, tile, x, y, tile_size)
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


@wp.kernel(enable_backward=False)
def initialize_face_validity(face_mass: wp.array[float], face_valid: wp.array[int]):
    """Mark MAC faces that received particle weight."""
    face = wp.tid()
    face_valid[face] = int(face_mass[face] > 0.0)


@wp.kernel(enable_backward=False)
def extrapolate_face_velocities(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    min_mass: float,
    velocity_in: wp.array[float],
    valid_in: wp.array[int],
    velocity_out: wp.array[float],
    valid_out: wp.array[int],
):
    """Extrapolate sampled velocities onto adjacent liquid-air faces."""
    face = wp.tid()
    index = face // 3
    component = face - 3 * index
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    if index >= grid.tile_count[0] * tile_volume:
        velocity_out[face] = 0.0
        valid_out[face] = 0
        return
    if valid_in[face] != 0:
        velocity_out[face] = velocity_in[face]
        valid_out[face] = 1
        return

    tile = index // tile_volume
    local = index - tile * tile_volume
    local_x = local % grid.tile_size
    local_y = (local // grid.tile_size) % grid.tile_size
    local_z = local // (grid.tile_size * grid.tile_size)
    left_x = local_x
    left_y = local_y
    left_z = local_z
    if component == 0:
        left_x -= 1
    elif component == 1:
        left_y -= 1
    else:
        left_z -= 1
    left = sparse_grid_cell_index(grid, tile, left_x, left_y, left_z)
    right = index
    if not (_is_liquid(cell_mass, left, min_mass) or _is_liquid(cell_mass, right, min_mass)):
        velocity_out[face] = velocity_in[face]
        valid_out[face] = 0
        return

    velocity_sum = 0.0
    count = 0
    for z in range(-1, 2):
        for y in range(-1, 2):
            for x in range(-1, 2):
                neighbor = sparse_grid_cell_index(grid, tile, local_x + x, local_y + y, local_z + z)
                if neighbor >= 0:
                    neighbor_face = 3 * neighbor + component
                    if valid_in[neighbor_face] != 0:
                        velocity_sum += velocity_in[neighbor_face]
                        count += 1
    if count > 0:
        velocity_out[face] = velocity_sum / float(count)
        valid_out[face] = 1
    else:
        velocity_out[face] = velocity_in[face]
        valid_out[face] = 0


@wp.kernel(enable_backward=False)
def build_pressure_system(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    face_velocity: wp.array[float],
    min_mass: float,
    rhs_scale: float,
    pressure_rhs: wp.array[float],
    pressure_diag: wp.array[float],
):
    """Build the free-surface Poisson right-hand side and diagonal."""
    index = wp.tid()
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    tile = index // tile_volume
    if tile >= grid.tile_count[0]:
        pressure_rhs[index] = 0.0
        pressure_diag[index] = 0.0
        return
    if not _is_liquid(cell_mass, index, min_mass):
        pressure_rhs[index] = 0.0
        pressure_diag[index] = 0.0
        return

    local = index - tile * tile_volume
    x = local % grid.tile_size
    y = (local // grid.tile_size) % grid.tile_size
    z = local // (grid.tile_size * grid.tile_size)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _axis_aligned_neighbors(grid, index, tile, x, y, z)
    divergence = 0.0
    if x_hi >= 0:
        divergence += face_velocity[3 * x_hi]
    divergence -= face_velocity[3 * index]
    if y_hi >= 0:
        divergence += face_velocity[3 * y_hi + 1]
    divergence -= face_velocity[3 * index + 1]
    if z_hi >= 0:
        divergence += face_velocity[3 * z_hi + 2]
    divergence -= face_velocity[3 * index + 2]

    diagonal = 0.0
    if x_lo >= 0:
        diagonal += 1.0
    if x_hi >= 0:
        diagonal += 1.0
    if y_lo >= 0:
        diagonal += 1.0
    if y_hi >= 0:
        diagonal += 1.0
    if z_lo >= 0:
        diagonal += 1.0
    if z_hi >= 0:
        diagonal += 1.0
    pressure_rhs[index] = divergence * rhs_scale
    pressure_diag[index] = diagonal


@wp.kernel(enable_backward=False)
def pressure_jacobi(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    min_mass: float,
    pressure_rhs: wp.array[float],
    pressure_diag: wp.array[float],
    pressure_in: wp.array[float],
    pressure_out: wp.array[float],
):
    """Perform one sparse free-surface Jacobi pressure iteration."""
    index = wp.tid()
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    tile = index // tile_volume
    if tile >= grid.tile_count[0]:
        return
    diagonal = pressure_diag[index]
    if diagonal == 0.0:
        pressure_out[index] = 0.0
        return

    local = index - tile * tile_volume
    x = local % grid.tile_size
    y = (local // grid.tile_size) % grid.tile_size
    z = local // (grid.tile_size * grid.tile_size)
    neighbor_sum = 0.0
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _axis_aligned_neighbors(grid, index, tile, x, y, z)
    if _is_liquid(cell_mass, x_lo, min_mass):
        neighbor_sum += pressure_in[x_lo]
    if _is_liquid(cell_mass, x_hi, min_mass):
        neighbor_sum += pressure_in[x_hi]
    if _is_liquid(cell_mass, y_lo, min_mass):
        neighbor_sum += pressure_in[y_lo]
    if _is_liquid(cell_mass, y_hi, min_mass):
        neighbor_sum += pressure_in[y_hi]
    if _is_liquid(cell_mass, z_lo, min_mass):
        neighbor_sum += pressure_in[z_lo]
    if _is_liquid(cell_mass, z_hi, min_mass):
        neighbor_sum += pressure_in[z_hi]
    pressure_out[index] = (neighbor_sum - pressure_rhs[index]) / diagonal


@wp.kernel(enable_backward=False)
def pressure_chebyshev(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    min_mass: float,
    pressure_rhs: wp.array[float],
    pressure_diag: wp.array[float],
    alpha: float,
    beta: float,
    pressure_in: wp.array[float],
    direction_in: wp.array[float],
    pressure_out: wp.array[float],
    direction_out: wp.array[float],
):
    """Perform one Chebyshev-accelerated sparse pressure iteration."""
    index = wp.tid()
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    tile = index // tile_volume
    if tile >= grid.tile_count[0]:
        pressure_out[index] = 0.0
        direction_out[index] = 0.0
        return
    diagonal = pressure_diag[index]
    if diagonal == 0.0:
        pressure_out[index] = 0.0
        direction_out[index] = 0.0
        return

    local = index - tile * tile_volume
    x = local % grid.tile_size
    y = (local // grid.tile_size) % grid.tile_size
    z = local // (grid.tile_size * grid.tile_size)
    neighbor_sum = 0.0
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _axis_aligned_neighbors(grid, index, tile, x, y, z)
    if _is_liquid(cell_mass, x_lo, min_mass):
        neighbor_sum += pressure_in[x_lo]
    if _is_liquid(cell_mass, x_hi, min_mass):
        neighbor_sum += pressure_in[x_hi]
    if _is_liquid(cell_mass, y_lo, min_mass):
        neighbor_sum += pressure_in[y_lo]
    if _is_liquid(cell_mass, y_hi, min_mass):
        neighbor_sum += pressure_in[y_hi]
    if _is_liquid(cell_mass, z_lo, min_mass):
        neighbor_sum += pressure_in[z_lo]
    if _is_liquid(cell_mass, z_hi, min_mass):
        neighbor_sum += pressure_in[z_hi]
    residual = (-pressure_rhs[index] - diagonal * pressure_in[index] + neighbor_sum) / diagonal
    direction = alpha * residual + beta * direction_in[index]
    direction_out[index] = direction
    pressure_out[index] = pressure_in[index] + direction


@wp.kernel(enable_backward=False)
def apply_pressure(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    min_mass: float,
    pressure: wp.array[float],
    pressure_scale: float,
    face_valid: wp.array[int],
    face_velocity: wp.array[float],
):
    """Apply pressure gradients to active MAC faces."""
    index = wp.tid()
    tile_volume = grid.tile_size * grid.tile_size * grid.tile_size
    tile = index // tile_volume
    if tile >= grid.tile_count[0]:
        return
    local = index - tile * tile_volume
    x = local % grid.tile_size
    y = (local // grid.tile_size) % grid.tile_size
    z = local // (grid.tile_size * grid.tile_size)
    for axis in range(3):
        face = 3 * index + axis
        if face_valid[face] != 0:
            left = -1
            if axis == 0:
                left = sparse_grid_cell_index(grid, tile, x - 1, y, z)
            elif axis == 1:
                left = sparse_grid_cell_index(grid, tile, x, y - 1, z)
            else:
                left = sparse_grid_cell_index(grid, tile, x, y, z - 1)
            right = index
            left_liquid = _is_liquid(cell_mass, left, min_mass)
            right_liquid = _is_liquid(cell_mass, right, min_mass)
            if left_liquid or right_liquid:
                p_left = 0.0
                p_right = 0.0
                if left_liquid:
                    p_left = pressure[left]
                if right_liquid:
                    p_right = pressure[right]
                face_velocity[face] -= pressure_scale * (p_right - p_left)


@wp.func
def _sample_component(
    grid: SparseGridData,
    position: wp.vec3,
    inv_cell_size: float,
    offset: wp.vec3,
    component: int,
    values: wp.array[float],
) -> float:
    base = _grid_index(position, offset, inv_cell_size)
    base_index = sparse_grid_index_from_cell(grid, base)
    if base_index < 0:
        return 0.0
    tile, local_x, local_y, local_z = _stencil_coordinates(grid, base_index)
    p_grid = position * inv_cell_size - offset
    fraction = p_grid - wp.vec3(base)
    weight_x = wp.vec2(1.0 - fraction[0], fraction[0])
    weight_y = wp.vec2(1.0 - fraction[1], fraction[1])
    weight_z = wp.vec2(1.0 - fraction[2], fraction[2])
    result = 0.0
    weight_sum = 0.0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                index = _stencil_index(grid, tile, local_x, local_y, local_z, x, y, z)
                if index >= 0:
                    w = weight_x[x] * weight_y[y] * weight_z[z]
                    result += w * values[3 * index + component]
                    weight_sum += w
    if weight_sum > 0.0:
        return result / weight_sum
    return 0.0


@wp.func
def _sample_velocity(
    grid: SparseGridData,
    position: wp.vec3,
    inv_cell_size: float,
    values: wp.array[float],
) -> wp.vec3:
    return wp.vec3(
        _sample_component(grid, position, inv_cell_size, wp.vec3(0.0, 0.5, 0.5), 0, values),
        _sample_component(grid, position, inv_cell_size, wp.vec3(0.5, 0.0, 0.5), 1, values),
        _sample_component(grid, position, inv_cell_size, wp.vec3(0.5, 0.5, 0.0), 2, values),
    )


@wp.func
def _sample_component_pair_gradient(
    grid: SparseGridData,
    position: wp.vec3,
    inv_cell_size: float,
    offset: wp.vec3,
    component: int,
    current_values: wp.array[float],
    previous_values: wp.array[float],
    compute_gradient: bool,
) -> tuple[float, float, wp.vec3]:
    p_grid = position * inv_cell_size - offset
    base = _grid_index(position, offset, inv_cell_size)
    for axis in range(3):
        if wp.abs(p_grid[axis] - float(base[axis])) < 1.0e-6:
            base[axis] -= 1
    base_index = sparse_grid_index_from_cell(grid, base)
    if base_index < 0:
        return 0.0, 0.0, wp.vec3(0.0)
    tile, local_x, local_y, local_z = _stencil_coordinates(grid, base_index)

    fraction = p_grid - wp.vec3(base)
    weight_x = wp.vec2(1.0 - fraction[0], fraction[0])
    weight_y = wp.vec2(1.0 - fraction[1], fraction[1])
    weight_z = wp.vec2(1.0 - fraction[2], fraction[2])
    current = 0.0
    previous = 0.0
    weight_sum = 0.0
    gradient = wp.vec3(0.0)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                index = _stencil_index(grid, tile, local_x, local_y, local_z, x, y, z)
                if index >= 0:
                    wx = weight_x[x]
                    wy = weight_y[y]
                    wz = weight_z[z]
                    weight = wx * wy * wz
                    value = current_values[3 * index + component]
                    current += weight * value
                    previous += weight * previous_values[3 * index + component]
                    weight_sum += weight
                    if compute_gradient:
                        dwx = float(2 * x - 1) * inv_cell_size
                        dwy = float(2 * y - 1) * inv_cell_size
                        dwz = float(2 * z - 1) * inv_cell_size
                        gradient += value * wp.vec3(dwx * wy * wz, wx * dwy * wz, wx * wy * dwz)
    if weight_sum > 0.0:
        current /= weight_sum
        previous /= weight_sum
    return current, previous, gradient


@wp.kernel(enable_backward=False)
def sample_transfer_components(
    grid: SparseGridData,
    positions: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    current_values: wp.array[float],
    previous_values: wp.array[float],
    compute_gradient: bool,
    samples_gradient_xy: wp.array[wp.vec4],
    gradient_z: wp.array[float],
):
    """Sample current/previous velocity and APIC gradients per MAC component."""
    thread = wp.tid()
    particle = thread // 3
    component = thread - 3 * particle
    current = 0.0
    previous = 0.0
    gradient = wp.vec3(0.0)
    if (flags[particle] & _ACTIVE) != 0:
        offset = wp.vec3(0.5)
        if component == 0:
            offset = wp.vec3(0.0, 0.5, 0.5)
        elif component == 1:
            offset = wp.vec3(0.5, 0.0, 0.5)
        else:
            offset = wp.vec3(0.5, 0.5, 0.0)
        current, previous, gradient = _sample_component_pair_gradient(
            grid,
            positions[particle],
            inv_cell_size,
            offset,
            component,
            current_values,
            previous_values,
            compute_gradient,
        )
    samples_gradient_xy[thread] = wp.vec4(current, previous, gradient[0], gradient[1])
    gradient_z[thread] = gradient[2]


@wp.kernel(enable_backward=False)
def finalize_grid_to_particles(
    grid: SparseGridData,
    positions_in: wp.array[wp.vec3],
    velocities_in: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    dt: float,
    face_velocity: wp.array[float],
    flip_blend: float,
    compute_affine: bool,
    samples_gradient_xy: wp.array[wp.vec4],
    gradient_z: wp.array[float],
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
    affine_out: wp.array[wp.mat33],
):
    """Finalize the grid transfer and advect particles with a midpoint sample."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        positions_out[particle] = positions_in[particle]
        velocities_out[particle] = velocities_in[particle]
        affine_out[particle] = wp.mat33(0.0)
        return

    component = 3 * particle
    sample_x = samples_gradient_xy[component]
    sample_y = samples_gradient_xy[component + 1]
    sample_z = samples_gradient_xy[component + 2]
    pic = wp.vec3(sample_x[0], sample_y[0], sample_z[0])
    old = wp.vec3(sample_x[1], sample_y[1], sample_z[1])
    flip = velocities_in[particle] + pic - old
    velocity = wp.lerp(pic, flip, flip_blend)
    midpoint = positions_in[particle] + velocity * (0.5 * dt)
    positions_out[particle] = (
        positions_in[particle] + _sample_velocity(grid, midpoint, inv_cell_size, face_velocity) * dt
    )
    velocities_out[particle] = velocity
    if compute_affine:
        affine_out[particle] = wp.matrix_from_rows(
            wp.vec3(sample_x[2], sample_x[3], gradient_z[component]),
            wp.vec3(sample_y[2], sample_y[3], gradient_z[component + 1]),
            wp.vec3(sample_z[2], sample_z[3], gradient_z[component + 2]),
        )
    else:
        affine_out[particle] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def constrain_particles(
    lower: wp.vec3,
    upper: wp.vec3,
    max_velocity: float,
    flags: wp.array[int],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    """Enforce a closed axis-aligned fluid domain and velocity limit."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        return
    position = positions[particle]
    velocity = velocities[particle]
    for axis in range(3):
        if position[axis] < lower[axis]:
            position[axis] = lower[axis]
            velocity[axis] = wp.max(velocity[axis], 0.0)
        elif position[axis] > upper[axis]:
            position[axis] = upper[axis]
            velocity[axis] = wp.min(velocity[axis], 0.0)
    speed = wp.length(velocity)
    if speed > max_velocity:
        velocity *= max_velocity / speed
    positions[particle] = position
    velocities[particle] = velocity
