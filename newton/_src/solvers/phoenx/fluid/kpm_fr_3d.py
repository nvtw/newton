# SPDX-FileCopyrightText: Copyright (c) 2026 Zike Xu
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .operators import fr_operators

RT = wp.constant(1.0 / 3.0)
MAX_K3 = wp.constant(216)
vec7 = wp.types.vector(7, wp.float32)


@wp.func
def _moment(u: wp.vec4, a: int, b: int, c: int):
    rho = u[0]
    x = u[1] / rho
    y = u[2] / rho
    z = u[3] / rho
    value = 0.0
    if a == 0 and b == 0 and c == 0:
        value = rho
    elif a == 1 and b == 0 and c == 0:
        value = rho * x
    elif a == 0 and b == 1 and c == 0:
        value = rho * y
    elif a == 0 and b == 0 and c == 1:
        value = rho * z
    elif a == 2 and b == 0 and c == 0:
        value = rho * (x * x + RT)
    elif a == 0 and b == 2 and c == 0:
        value = rho * (y * y + RT)
    elif a == 0 and b == 0 and c == 2:
        value = rho * (z * z + RT)
    elif a == 1 and b == 1 and c == 0:
        value = rho * x * y
    elif a == 1 and b == 0 and c == 1:
        value = rho * x * z
    elif a == 0 and b == 1 and c == 1:
        value = rho * y * z
    elif a == 3 and b == 0 and c == 0:
        value = 3.0 * rho * RT * x
    elif a == 0 and b == 3 and c == 0:
        value = 3.0 * rho * RT * y
    elif a == 0 and b == 0 and c == 3:
        value = 3.0 * rho * RT * z
    elif a == 2 and b == 1 and c == 0:
        value = rho * RT * y
    elif a == 1 and b == 2 and c == 0:
        value = rho * RT * x
    elif a == 2 and b == 0 and c == 1:
        value = rho * RT * z
    elif a == 0 and b == 2 and c == 1:
        value = rho * RT * z
    elif a == 1 and b == 0 and c == 2:
        value = rho * RT * x
    elif a == 0 and b == 1 and c == 2:
        value = rho * RT * y
    return value


@wp.func
def _index(i: int, j: int, k: int, sp: int, ny: int, nz: int, k3: int):
    return ((i * ny + j) * nz + k) * k3 + sp


@wp.func
def _read_u(state: wp.array2d[wp.float32], index: int):
    return wp.vec4(state[0, index], state[1, index], state[2, index], state[3, index])


@wp.func
def _pi_values(xx: float, xy: float, xz: float, yy: float, yz: float):
    return wp.mat33(xx, xy, xz, xy, yy, yz, xz, yz, -xx - yy)


@wp.func
def _read_pi(stress: wp.array2d[wp.float16], index: int):
    xx = wp.float32(stress[0, index])
    xy = wp.float32(stress[1, index])
    xz = wp.float32(stress[2, index])
    yy = wp.float32(stress[3, index])
    yz = wp.float32(stress[4, index])
    return wp.mat33(xx, xy, xz, xy, yy, yz, xz, yz, -xx - yy)


@wp.kernel(enable_backward=False)
def _predictor(
    state: wp.array2d[wp.float32],
    half_state: wp.array2d[wp.float32],
    stress: wp.array2d[wp.float16],
    d: wp.array2d[wp.float32],
    order: int,
    ny: int,
    nz: int,
    scale: wp.vec3,
    half_dt: float,
    uc: float,
    coeff_a: float,
    coeff_b: float,
):
    cell, lane = wp.tid()
    i = cell / (ny * nz)
    j = (cell / nz) % ny
    k = cell % nz
    k2 = order * order
    k3 = k2 * order
    cell_offset = cell * k3
    local = wp.tile_transpose(wp.tile_load(state, shape=(4, MAX_K3), offset=(0, cell_offset), storage="shared"))
    for logical in range(lane, k2, wp.block_dim()):
        sx = logical % order
        sy = logical // order
        for sz in range(order):
            sp = logical + sz * k2
            p = _index(i, j, k, sp, ny, nz, k3)
            u0 = wp.vec4(local[sp, 0], local[sp, 1], local[sp, 2], local[sp, 3])
            grad_x = wp.vec4(0.0)
            grad_y = wp.vec4(0.0)
            grad_z = wp.vec4(0.0)
            for q in range(order):
                px = q + sy * order + sz * k2
                py = sx + q * order + sz * k2
                pz = sx + sy * order + q * k2
                grad_x += d[sx, q] * wp.vec4(local[px, 0], local[px, 1], local[px, 2], local[px, 3]) * scale[0]
                grad_y += d[sy, q] * wp.vec4(local[py, 0], local[py, 1], local[py, 2], local[py, 3]) * scale[1]
                grad_z += d[sz, q] * wp.vec4(local[pz, 0], local[pz, 1], local[pz, 2], local[pz, 3]) * scale[2]
            shifted_x = u0 - uc * half_dt * grad_x
            shifted_y = u0 - uc * half_dt * grad_y
            shifted_z = u0 - uc * half_dt * grad_z
            inv_uc = 1.0 / uc

            uh = wp.vec4(
                _moment(u0, 0, 0, 0)
                + (_moment(shifted_x, 1, 0, 0) - _moment(u0, 1, 0, 0)) * inv_uc
                + (_moment(shifted_y, 0, 1, 0) - _moment(u0, 0, 1, 0)) * inv_uc
                + (_moment(shifted_z, 0, 0, 1) - _moment(u0, 0, 0, 1)) * inv_uc,
                _moment(u0, 1, 0, 0)
                + (_moment(shifted_x, 2, 0, 0) - _moment(u0, 2, 0, 0)) * inv_uc
                + (_moment(shifted_y, 1, 1, 0) - _moment(u0, 1, 1, 0)) * inv_uc
                + (_moment(shifted_z, 1, 0, 1) - _moment(u0, 1, 0, 1)) * inv_uc,
                _moment(u0, 0, 1, 0)
                + (_moment(shifted_x, 1, 1, 0) - _moment(u0, 1, 1, 0)) * inv_uc
                + (_moment(shifted_y, 0, 2, 0) - _moment(u0, 0, 2, 0)) * inv_uc
                + (_moment(shifted_z, 0, 1, 1) - _moment(u0, 0, 1, 1)) * inv_uc,
                _moment(u0, 0, 0, 1)
                + (_moment(shifted_x, 1, 0, 1) - _moment(u0, 1, 0, 1)) * inv_uc
                + (_moment(shifted_y, 0, 1, 1) - _moment(u0, 0, 1, 1)) * inv_uc
                + (_moment(shifted_z, 0, 0, 2) - _moment(u0, 0, 0, 2)) * inv_uc,
            )
            pi = wp.mat33(0.0)
            pi[0, 0] = coeff_a * (
                _moment(u0, 2, 0, 0)
                + (_moment(shifted_x, 3, 0, 0) - _moment(u0, 3, 0, 0)) * inv_uc
                + (_moment(shifted_y, 2, 1, 0) - _moment(u0, 2, 1, 0)) * inv_uc
                + (_moment(shifted_z, 2, 0, 1) - _moment(u0, 2, 0, 1)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 2, 0, 0)
            pi[0, 1] = coeff_a * (
                _moment(u0, 1, 1, 0)
                + (_moment(shifted_x, 2, 1, 0) - _moment(u0, 2, 1, 0)) * inv_uc
                + (_moment(shifted_y, 1, 2, 0) - _moment(u0, 1, 2, 0)) * inv_uc
                + (_moment(shifted_z, 1, 1, 1) - _moment(u0, 1, 1, 1)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 1, 1, 0)
            pi[0, 2] = coeff_a * (
                _moment(u0, 1, 0, 1)
                + (_moment(shifted_x, 2, 0, 1) - _moment(u0, 2, 0, 1)) * inv_uc
                + (_moment(shifted_y, 1, 1, 1) - _moment(u0, 1, 1, 1)) * inv_uc
                + (_moment(shifted_z, 1, 0, 2) - _moment(u0, 1, 0, 2)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 1, 0, 1)
            pi[1, 1] = coeff_a * (
                _moment(u0, 0, 2, 0)
                + (_moment(shifted_x, 1, 2, 0) - _moment(u0, 1, 2, 0)) * inv_uc
                + (_moment(shifted_y, 0, 3, 0) - _moment(u0, 0, 3, 0)) * inv_uc
                + (_moment(shifted_z, 0, 2, 1) - _moment(u0, 0, 2, 1)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 0, 2, 0)
            pi[1, 2] = coeff_a * (
                _moment(u0, 0, 1, 1)
                + (_moment(shifted_x, 1, 1, 1) - _moment(u0, 1, 1, 1)) * inv_uc
                + (_moment(shifted_y, 0, 2, 1) - _moment(u0, 0, 2, 1)) * inv_uc
                + (_moment(shifted_z, 0, 1, 2) - _moment(u0, 0, 1, 2)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 0, 1, 1)
            pi[2, 2] = coeff_a * (
                _moment(u0, 0, 0, 2)
                + (_moment(shifted_x, 1, 0, 2) - _moment(u0, 1, 0, 2)) * inv_uc
                + (_moment(shifted_y, 0, 1, 2) - _moment(u0, 0, 1, 2)) * inv_uc
                + (_moment(shifted_z, 0, 0, 3) - _moment(u0, 0, 0, 3)) * inv_uc
            ) + (coeff_b - 1.0) * _moment(uh, 0, 0, 2)
            trace = (pi[0, 0] + pi[1, 1] + pi[2, 2]) / 3.0
            for c in range(4):
                half_state[c, p] = uh[c]
            stress[0, p] = wp.float16(pi[0, 0] - trace)
            stress[1, p] = wp.float16(pi[0, 1])
            stress[2, p] = wp.float16(pi[0, 2])
            stress[3, p] = wp.float16(pi[1, 1] - trace)
            stress[4, p] = wp.float16(pi[1, 2])


@wp.func
def _face_state_values(u: wp.vec4, pi: wp.mat33, axis: int):
    rho = u[0]
    velocity = wp.vec3(u[1], u[2], u[3]) / rho
    t0 = (axis + 1) % 3
    t1 = (axis + 2) % 3
    return vec7(rho, velocity[axis], velocity[t0], velocity[t1], pi[axis, axis], pi[axis, t0], pi[axis, t1])


@wp.func
def _face_state(state: wp.array2d[wp.float32], stress: wp.array2d[wp.float16], p: int, axis: int):
    u = _read_u(state, p)
    pi = _read_pi(stress, p)
    rho = u[0]
    velocity = wp.vec3(u[1], u[2], u[3]) / rho
    t0 = (axis + 1) % 3
    t1 = (axis + 2) % 3
    return vec7(rho, velocity[axis], velocity[t0], velocity[t1], pi[axis, axis], pi[axis, t0], pi[axis, t1])


@wp.func
def _common_flux(left: vec7, right: vec7, coeff_a: float, coeff_b: float, blend: float):
    c1 = wp.sqrt(2.0 * wp.pi * RT)
    c2 = 2.0 * wp.sqrt(2.0 * RT / wp.pi)
    c3 = wp.sqrt(RT / (2.0 * wp.pi))
    inv_c1 = 1.0 / c1
    a = (1.0 - blend) * coeff_a + blend
    b = (1.0 - blend) * coeff_b
    c_left = left[0] * (0.5 + left[1] * inv_c1)
    c_right = right[0] * (0.5 - right[1] * inv_c1)
    rho_star = c_left + c_right
    rhoun = (
        0.5
        * inv_c1
        * (
            left[0] * (2.0 * RT + c1 * left[1] + left[1] * left[1])
            - right[0] * (2.0 * RT - c1 * right[1] + right[1] * right[1])
            + left[4]
            - right[4]
        )
    )
    un = rhoun / rho_star
    flux = wp.vec4(0.0)
    flux[0] = (
        b * rhoun
        + 0.5
        * a
        * inv_c1
        * (
            left[0] * (2.0 * RT + c1 * left[1] + left[1] * left[1])
            - right[0] * (2.0 * RT - c1 * right[1] + right[1] * right[1])
        )
        + 0.5 * inv_c1 * (left[4] - right[4])
    )
    flux[1] = (
        b * rho_star * (un * un + RT)
        + 0.5
        * a
        * (left[0] * (RT + c2 * left[1] + left[1] * left[1]) + right[0] * (RT - c2 * right[1] + right[1] * right[1]))
        + 0.5 * (left[4] + right[4])
    )
    for t in range(2):
        velocity = t + 2
        stress = t + 5
        fbar = coeff_a * (
            left[0] * (c3 * left[velocity] + 0.5 * left[1] * left[velocity])
            + right[0] * (-c3 * right[velocity] + 0.5 * right[1] * right[velocity])
        ) + 0.5 * (left[stress] + right[stress])
        flux[velocity] = fbar + 0.5 * coeff_b * (
            rhoun * (left[velocity] + right[velocity]) + wp.abs(rhoun) * (left[velocity] - right[velocity])
        )
    return flux


@wp.func
def _rotate_flux(flux: wp.vec4, axis: int):
    result = wp.vec4(flux[0], flux[1], flux[2], flux[3])
    if axis == 1:
        result = wp.vec4(flux[0], flux[3], flux[1], flux[2])
    elif axis == 2:
        result = wp.vec4(flux[0], flux[2], flux[3], flux[1])
    return result


@wp.func
def _physical_flux(u: wp.vec4, pi: wp.mat33, axis: int):
    rho = u[0]
    velocity = wp.vec3(u[1], u[2], u[3]) / rho
    result = wp.vec4(rho * velocity[axis], 0.0, 0.0, 0.0)
    for c in range(3):
        result[c + 1] = rho * velocity[axis] * velocity[c] + pi[axis, c]
    result[axis + 1] += rho * RT
    return result


@wp.func
def _local_derivative(
    state: wp.array2d[wp.float32],
    stress: wp.array2d[wp.float16],
    d: wp.array2d[wp.float32],
    q_matrix: wp.array2d[wp.float32],
    cr: wp.array2d[wp.float32],
    i: int,
    j: int,
    k: int,
    point: wp.vec3i,
    order: int,
    ny: int,
    nz: int,
    axis: int,
):
    k2 = order * order
    k3 = k2 * order
    s = point[axis]
    center_sp = point[0] + point[1] * order + point[2] * k2
    center = _read_u(state, _index(i, j, k, center_sp, ny, nz, k3))
    rho = center[0]
    phi = center / rho
    phi[0] = 1.0
    rho_velocity = center[axis + 1]
    d_rv = wp.float32(0.0)
    d_phi = wp.vec4(0.0)
    d_rv_phi = wp.vec4(0.0)
    d_pressure = wp.float32(0.0)
    d_stress = wp.vec4(0.0)
    boundary_eq = wp.vec4(0.0)
    for line in range(order):
        q = point
        q[axis] = line
        sp = q[0] + q[1] * order + q[2] * k2
        p = _index(i, j, k, sp, ny, nz, k3)
        u = _read_u(state, p)
        pi = _read_pi(stress, p)
        line_rho = u[0]
        line_phi = u / line_rho
        line_phi[0] = 1.0
        rv = u[axis + 1]
        ne = wp.vec4(0.0, pi[axis, 0], pi[axis, 1], pi[axis, 2])
        weight = d[s, line]
        d_rv += weight * rv
        d_phi += weight * line_phi
        d_rv_phi += weight * rv * line_phi
        d_pressure += weight * line_rho * RT
        d_stress += q_matrix[s, line] * ne
        boundary_eq += cr[s, line] * _physical_flux(u, wp.mat33(0.0), axis)
    result = d_stress + 0.5 * (d_rv_phi + phi * d_rv + rho_velocity * d_phi) - boundary_eq
    result[axis + 1] += d_pressure
    return result


@wp.kernel(enable_backward=False)
def _corrector(
    state: wp.array2d[wp.float32],
    half_state: wp.array2d[wp.float32],
    stress: wp.array2d[wp.float16],
    d: wp.array2d[wp.float32],
    correction: wp.array2d[wp.float32],
    cr: wp.array2d[wp.float32],
    q_matrix: wp.array2d[wp.float32],
    order: int,
    resolution: wp.vec3i,
    scale: wp.vec3,
    dt: float,
    coeff_a: float,
    coeff_b: float,
    blend: float,
):
    cell_id, lane = wp.tid()
    ny = resolution[1]
    nz = resolution[2]
    i = cell_id // (ny * nz)
    j = (cell_id // nz) % ny
    k = cell_id % nz
    k2 = order * order
    k3 = k2 * order
    cell = wp.vec3i(i, j, k)
    local_half = wp.tile_transpose(
        wp.tile_load(half_state, shape=(4, MAX_K3), offset=(0, cell_id * k3), storage="shared")
    )
    local_stress = wp.tile_transpose(
        wp.tile_load(stress, shape=(5, MAX_K3), offset=(0, cell_id * k3), storage="shared")
    )
    for logical in range(lane, k2, wp.block_dim()):
        for sz in range(order):
            sp = logical + sz * k2
            point = wp.vec3i(logical % order, logical // order, sz)
            residual = wp.vec4(0.0)
            for axis in range(3):
                minus = cell
                plus = cell
                minus[axis] = (cell[axis] + resolution[axis] - 1) % resolution[axis]
                plus[axis] = (cell[axis] + 1) % resolution[axis]
                stride = 1
                if axis == 1:
                    stride = order
                elif axis == 2:
                    stride = k2
                low_sp = sp - point[axis] * stride
                high_sp = low_sp + (order - 1) * stride
                left = _index(minus[0], minus[1], minus[2], high_sp, resolution[1], resolution[2], k3)
                right = _index(plus[0], plus[1], plus[2], low_sp, resolution[1], resolution[2], k3)
                flux_left = _rotate_flux(
                    _common_flux(
                        _face_state(half_state, stress, left, axis),
                        _face_state_values(
                            wp.vec4(
                                local_half[low_sp, 0],
                                local_half[low_sp, 1],
                                local_half[low_sp, 2],
                                local_half[low_sp, 3],
                            ),
                            _pi_values(
                                wp.float32(local_stress[low_sp, 0]),
                                wp.float32(local_stress[low_sp, 1]),
                                wp.float32(local_stress[low_sp, 2]),
                                wp.float32(local_stress[low_sp, 3]),
                                wp.float32(local_stress[low_sp, 4]),
                            ),
                            axis,
                        ),
                        coeff_a,
                        coeff_b,
                        blend,
                    ),
                    axis,
                )
                flux_right = _rotate_flux(
                    _common_flux(
                        _face_state_values(
                            wp.vec4(
                                local_half[high_sp, 0],
                                local_half[high_sp, 1],
                                local_half[high_sp, 2],
                                local_half[high_sp, 3],
                            ),
                            _pi_values(
                                wp.float32(local_stress[high_sp, 0]),
                                wp.float32(local_stress[high_sp, 1]),
                                wp.float32(local_stress[high_sp, 2]),
                                wp.float32(local_stress[high_sp, 3]),
                                wp.float32(local_stress[high_sp, 4]),
                            ),
                            axis,
                        ),
                        _face_state(half_state, stress, right, axis),
                        coeff_a,
                        coeff_b,
                        blend,
                    ),
                    axis,
                )
                residual += scale[axis] * (
                    _local_derivative(
                        half_state, stress, d, q_matrix, cr, i, j, k, point, order, resolution[1], resolution[2], axis
                    )
                    + correction[point[axis], 0] * flux_left
                    + correction[point[axis], 1] * flux_right
                )
            p = _index(i, j, k, sp, resolution[1], resolution[2], k3)
            old = _read_u(state, p)
            updated = old - dt * residual
            for c in range(4):
                state[c, p] = updated[c]


@wp.kernel(enable_backward=False)
def _penalize(state: wp.array2d[wp.float32], volume_fraction: wp.array[wp.float16]):
    p = wp.tid()
    fluid = 1.0 - wp.float32(volume_fraction[p])
    state[1, p] *= fluid
    state[2, p] *= fluid
    state[3, p] *= fluid


@dataclass
class KPMFR3DConfig:
    """Configure a three-dimensional KPM-FR solve.

    Args:
        resolution: Element count per axis.
        size: Domain extents [m].
        order: Solution points per element axis.
        reynolds: Reynolds number.
        reference_velocity: Reference flow speed [m/s].
        cfl: Courant number.
        correction: Flux-reconstruction correction family.
        blend: Common-flux blend factor.
        epsilon: Kinetic transport-speed factor.
    """

    resolution: tuple[int, int, int]
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    order: int = 4
    reynolds: float = 100_000.0
    reference_velocity: float = 0.1
    cfl: float = 0.5
    correction: str = "g2"
    blend: float = 0.5
    epsilon: float = 0.1


class KPMFR3D:
    """Advance a three-dimensional KPM-FR fluid state.

    Args:
        config: Solver configuration.
        device: Warp device used for storage and launches.
    """

    def __init__(self, config: KPMFR3DConfig, device: wp.DeviceLike = None):
        self.config = config
        self.device = wp.get_device(device)
        nx, ny, nz = config.resolution
        order = config.order
        self.points, d, correction, cr, q_matrix = fr_operators(order, config.correction)
        self.derivative = wp.array(d, device=self.device)
        self.correction = wp.array(correction, device=self.device)
        self.cr = wp.array(cr, device=self.device)
        self.q_matrix = wp.array(q_matrix, device=self.device)
        point_count = nx * ny * nz * order**3
        self.state = wp.zeros((4, point_count), dtype=wp.float32, device=self.device)
        self.half_state = wp.zeros_like(self.state)
        self.stress = wp.zeros((5, point_count), dtype=wp.float16, device=self.device)
        self.volume_fraction = wp.zeros(point_count, dtype=wp.float16, device=self.device)
        spacing = tuple(config.size[i] / config.resolution[i] for i in range(3))
        viscosity = config.reference_velocity * min(config.size) / config.reynolds
        self.tau = viscosity / (1.0 / 3.0)
        cap = {3: 0.80, 4: 0.70, 5: 0.60, 6: 0.50}[order] * 0.8
        cfl = min(config.cfl, cap) * (0.5 if config.correction == "gdg" else 1.0)
        self.dt = cfl * min(spacing) / ((2 * order - 1) * (config.reference_velocity + math.sqrt(1.0 / 3.0)))
        self._half_dt = 0.5 * self.dt
        self._a = self.tau / self._half_dt
        self._b = 1.0 - self._a
        self._scale = wp.vec3(*(2.0 / value for value in spacing))
        self._uc = config.epsilon * math.sqrt(1.0 / 3.0)

    def initialize(self, density: float = 1.0, velocity: tuple[float, float, float] = (0.1, 0.0, 0.0)):
        """Set a constant initial state.

        Args:
            density: Initial density.
            velocity: Initial velocity [m/s].
        """
        host = np.empty(self.state.shape, dtype=np.float32)
        host[0].fill(density)
        for axis in range(3):
            host[axis + 1].fill(density * velocity[axis])
        self.state.assign(host)

    def step(self, count: int = 1):
        """Advance the fluid state.

        Args:
            count: Number of time steps.
        """
        nx, ny, nz = self.config.resolution
        order = self.config.order
        resolution = wp.vec3i(nx, ny, nz)
        for _ in range(count):
            wp.launch_tiled(
                _predictor,
                dim=nx * ny * nz,
                inputs=[
                    self.state,
                    self.half_state,
                    self.stress,
                    self.derivative,
                    order,
                    ny,
                    nz,
                    self._scale,
                    self._half_dt,
                    self._uc,
                    self._a,
                    self._b,
                ],
                device=self.device,
                block_dim=order**2 if self.device.is_cuda else 1,
            )
            wp.launch_tiled(
                _corrector,
                dim=nx * ny * nz,
                inputs=[
                    self.state,
                    self.half_state,
                    self.stress,
                    self.derivative,
                    self.correction,
                    self.cr,
                    self.q_matrix,
                    order,
                    resolution,
                    self._scale,
                    self.dt,
                    self._a,
                    self._b,
                    self.config.blend,
                ],
                device=self.device,
                block_dim=order**2 if self.device.is_cuda else 1,
            )
            wp.launch(
                _penalize,
                dim=self.volume_fraction.shape[0],
                inputs=[self.state, self.volume_fraction],
                device=self.device,
            )
