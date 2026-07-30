# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for sparse ST-FLIP particle/grid transfers."""

import warp as wp

from ...geometry import ParticleFlags
from .sparse_grid import SparseGridData, sparse_grid_cell_coord, sparse_grid_index_from_cell

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
def _weight(x: float) -> float:
    x = wp.abs(x)
    return wp.max(1.0 - x, 0.0)


@wp.func
def _grid_index(position: wp.vec3, offset: wp.vec3, inv_cell_size: float) -> wp.vec3i:
    p = position * inv_cell_size - offset
    return wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))


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
    p_grid = position * inv_cell_size - offset
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cell = base + wp.vec3i(x, y, z)
                index = sparse_grid_index_from_cell(grid, cell)
                if index >= 0 and 3 * index + component < face_mass.shape[0]:
                    node = (wp.vec3(cell) + offset) / inv_cell_size
                    displacement = node - position
                    w = (
                        _weight(p_grid[0] - float(cell[0]))
                        * _weight(p_grid[1] - float(cell[1]))
                        * _weight(p_grid[2] - float(cell[2]))
                    )
                    momentum = mass * w * (velocity[component] + (affine * displacement)[component])
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
    face_mass: wp.array[float],
    face_momentum: wp.array[float],
):
    """Transfer active particle mass and APIC momentum to a sparse MAC grid."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        return

    velocity = velocities[particle] + forces[particle] * inverse_masses[particle] * dt
    position = positions[particle] + velocity * (temporal_offsets[particle] * dt)
    mass = masses[particle]
    base = _grid_index(position, wp.vec3(0.5), inv_cell_size)
    p_grid = position * inv_cell_size - wp.vec3(0.5)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cell = base + wp.vec3i(x, y, z)
                index = sparse_grid_index_from_cell(grid, cell)
                if index >= 0 and index < cell_mass.shape[0]:
                    w = (
                        _weight(p_grid[0] - float(cell[0]))
                        * _weight(p_grid[1] - float(cell[1]))
                        * _weight(p_grid[2] - float(cell[2]))
                    )
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
        face_velocity[face] = velocity + gravity[0][axis] * dt


@wp.func
def _is_liquid(cell_mass: wp.array[float], index: int, min_mass: float) -> bool:
    return index >= 0 and cell_mass[index] > min_mass


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
    if not _is_liquid(cell_mass, index, min_mass):
        pressure_rhs[index] = 0.0
        pressure_diag[index] = 0.0
        return

    cell = sparse_grid_cell_coord(grid, index)
    u0 = sparse_grid_index_from_cell(grid, cell)
    ux = sparse_grid_index_from_cell(grid, cell + wp.vec3i(1, 0, 0))
    vy = sparse_grid_index_from_cell(grid, cell + wp.vec3i(0, 1, 0))
    wz = sparse_grid_index_from_cell(grid, cell + wp.vec3i(0, 0, 1))
    divergence = 0.0
    if ux >= 0:
        divergence += face_velocity[3 * ux]
    if u0 >= 0:
        divergence -= face_velocity[3 * u0]
    if vy >= 0:
        divergence += face_velocity[3 * vy + 1]
    if u0 >= 0:
        divergence -= face_velocity[3 * u0 + 1]
    if wz >= 0:
        divergence += face_velocity[3 * wz + 2]
    if u0 >= 0:
        divergence -= face_velocity[3 * u0 + 2]

    diagonal = 0.0
    directions = wp.vec3i(0)
    for axis in range(3):
        directions = wp.vec3i(0)
        directions[axis] = 1
        if sparse_grid_index_from_cell(grid, cell - directions) >= 0:
            diagonal += 1.0
        if sparse_grid_index_from_cell(grid, cell + directions) >= 0:
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
    diagonal = pressure_diag[index]
    if diagonal == 0.0:
        pressure_out[index] = 0.0
        return

    cell = sparse_grid_cell_coord(grid, index)
    neighbor_sum = 0.0
    direction = wp.vec3i(0)
    for axis in range(3):
        direction = wp.vec3i(0)
        direction[axis] = 1
        lo = sparse_grid_index_from_cell(grid, cell - direction)
        hi = sparse_grid_index_from_cell(grid, cell + direction)
        if _is_liquid(cell_mass, lo, min_mass):
            neighbor_sum += pressure_in[lo]
        if _is_liquid(cell_mass, hi, min_mass):
            neighbor_sum += pressure_in[hi]
    pressure_out[index] = (neighbor_sum - pressure_rhs[index]) / diagonal


@wp.kernel(enable_backward=False)
def apply_pressure(
    grid: SparseGridData,
    cell_mass: wp.array[float],
    min_mass: float,
    pressure: wp.array[float],
    pressure_scale: float,
    face_mass: wp.array[float],
    face_velocity: wp.array[float],
):
    """Apply pressure gradients to active MAC faces."""
    index = wp.tid()
    cell = sparse_grid_cell_coord(grid, index)
    for axis in range(3):
        face = 3 * index + axis
        if face_mass[face] > 0.0:
            direction = wp.vec3i(0)
            direction[axis] = 1
            left = sparse_grid_index_from_cell(grid, cell - direction)
            right = sparse_grid_index_from_cell(grid, cell)
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
    p_grid = position * inv_cell_size - offset
    result = 0.0
    weight_sum = 0.0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cell = base + wp.vec3i(x, y, z)
                index = sparse_grid_index_from_cell(grid, cell)
                if index >= 0 and 3 * index + component < values.shape[0]:
                    w = (
                        _weight(p_grid[0] - float(cell[0]))
                        * _weight(p_grid[1] - float(cell[1]))
                        * _weight(p_grid[2] - float(cell[2]))
                    )
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
def _sample_component_gradient(
    grid: SparseGridData,
    position: wp.vec3,
    inv_cell_size: float,
    offset: wp.vec3,
    component: int,
    values: wp.array[float],
) -> wp.vec3:
    base = _grid_index(position, offset, inv_cell_size)
    p_grid = position * inv_cell_size - offset
    gradient = wp.vec3(0.0)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cell = base + wp.vec3i(x, y, z)
                index = sparse_grid_index_from_cell(grid, cell)
                if index >= 0 and 3 * index + component < values.shape[0]:
                    delta = p_grid - wp.vec3(cell)
                    wx = _weight(delta[0])
                    wy = _weight(delta[1])
                    wz = _weight(delta[2])
                    dwx = -wp.sign(delta[0]) * inv_cell_size
                    dwy = -wp.sign(delta[1]) * inv_cell_size
                    dwz = -wp.sign(delta[2]) * inv_cell_size
                    value = values[3 * index + component]
                    gradient += value * wp.vec3(dwx * wy * wz, wx * dwy * wz, wx * wy * dwz)
    return gradient


@wp.kernel(enable_backward=False)
def reconstruct_affine_rows(
    grid: SparseGridData,
    positions: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    face_velocity: wp.array[float],
    rows: wp.array[float],
):
    """Reconstruct one APIC velocity-gradient row per thread."""
    thread = wp.tid()
    particle = thread // 3
    component = thread - 3 * particle
    gradient = wp.vec3(0.0)
    if (flags[particle] & _ACTIVE) != 0:
        offset = wp.vec3(0.5)
        if component == 0:
            offset = wp.vec3(0.0, 0.5, 0.5)
        elif component == 1:
            offset = wp.vec3(0.5, 0.0, 0.5)
        else:
            offset = wp.vec3(0.5, 0.5, 0.0)
        gradient = _sample_component_gradient(
            grid, positions[particle], inv_cell_size, offset, component, face_velocity
        )
    row = 9 * particle + 3 * component
    rows[row] = gradient[0]
    rows[row + 1] = gradient[1]
    rows[row + 2] = gradient[2]


@wp.kernel(enable_backward=False)
def store_affine(rows: wp.array[float], affine: wp.array[wp.mat33]):
    """Store reconstructed APIC rows as particle matrices."""
    particle = wp.tid()
    row = 9 * particle
    affine[particle] = wp.matrix_from_rows(
        wp.vec3(rows[row], rows[row + 1], rows[row + 2]),
        wp.vec3(rows[row + 3], rows[row + 4], rows[row + 5]),
        wp.vec3(rows[row + 6], rows[row + 7], rows[row + 8]),
    )


@wp.kernel(enable_backward=False)
def sample_grid_velocity(
    grid: SparseGridData,
    positions: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    face_velocity: wp.array[float],
    result: wp.array[wp.vec3],
):
    """Sample one sparse MAC velocity field at particle positions."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        result[particle] = wp.vec3(0.0)
        return
    result[particle] = _sample_velocity(grid, positions[particle], inv_cell_size, face_velocity)


@wp.kernel(enable_backward=False)
def grid_to_particles(
    grid: SparseGridData,
    positions_in: wp.array[wp.vec3],
    velocities_in: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    flip_blend: float,
    face_velocity: wp.array[float],
    particle_velocity_old: wp.array[wp.vec3],
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
    affine_out: wp.array[wp.mat33],
):
    """Transfer projected velocity to particles and advect with midpoint RK2."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        positions_out[particle] = positions_in[particle]
        velocities_out[particle] = velocities_in[particle]
        affine_out[particle] = wp.mat33(0.0)
        return

    position = positions_in[particle]
    pic = _sample_velocity(grid, position, inv_cell_size, face_velocity)
    old = particle_velocity_old[particle]
    flip = velocities_in[particle] + pic - old
    velocity = wp.lerp(pic, flip, flip_blend)
    positions_out[particle] = position
    velocities_out[particle] = velocity
    affine_out[particle] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def advect_particles(
    grid: SparseGridData,
    positions_in: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    flags: wp.array[int],
    inv_cell_size: float,
    dt: float,
    face_velocity: wp.array[float],
    positions_out: wp.array[wp.vec3],
):
    """Advect particles with one midpoint sparse-grid sample."""
    particle = wp.tid()
    if (flags[particle] & _ACTIVE) == 0:
        positions_out[particle] = positions_in[particle]
        return
    midpoint = positions_in[particle] + velocities[particle] * (0.5 * dt)
    velocity = _sample_velocity(grid, midpoint, inv_cell_size, face_velocity)
    positions_out[particle] = positions_in[particle] + velocity * dt


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
