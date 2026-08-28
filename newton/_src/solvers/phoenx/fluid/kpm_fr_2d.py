# SPDX-FileCopyrightText: Copyright (c) 2026 Zike Xu
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .operators import fr_operators

RT = wp.constant(1.0 / 3.0)
vec5 = wp.types.vector(5, wp.float32)
RHO = wp.constant(0)
UN = wp.constant(1)
UT = wp.constant(2)
PI_NN = wp.constant(3)
PI_NT = wp.constant(4)


@wp.func
def _moment(u_state: wp.vec3, a: int, b: int):
    rho = u_state[0]
    u = u_state[1] / rho
    v = u_state[2] / rho
    value = 0.0
    if a == 0 and b == 0:
        value = rho
    elif a == 1 and b == 0:
        value = rho * u
    elif a == 0 and b == 1:
        value = rho * v
    elif a == 2 and b == 0:
        value = rho * (u * u + RT)
    elif a == 1 and b == 1:
        value = rho * u * v
    elif a == 0 and b == 2:
        value = rho * (v * v + RT)
    elif a == 3 and b == 0:
        value = 3.0 * rho * RT * u
    elif a == 2 and b == 1:
        value = rho * RT * v
    elif a == 1 and b == 2:
        value = rho * RT * u
    elif a == 0 and b == 3:
        value = 3.0 * rho * RT * v
    return value


@wp.func
def _read_u(state: wp.array4d[wp.float32], i: int, j: int, sx: int, sy: int, order: int):
    sp = sx + sy * order
    return wp.vec3(state[0, i, j, sp], state[1, i, j, sp], state[2, i, j, sp])


@wp.func
def _read_pi(stress: wp.array4d[wp.float16], i: int, j: int, sx: int, sy: int, order: int):
    sp = sx + sy * order
    return wp.vec3(
        wp.float32(stress[0, i, j, sp]),
        wp.float32(stress[1, i, j, sp]),
        wp.float32(stress[2, i, j, sp]),
    )


@wp.kernel(enable_backward=False)
def _predictor(
    state: wp.array4d[wp.float32],
    half_state: wp.array4d[wp.float32],
    half_stress: wp.array4d[wp.float16],
    derivative: wp.array2d[wp.float32],
    order: int,
    inv_dx: float,
    inv_dy: float,
    half_dt: float,
    characteristic_speed: float,
    a_coeff: float,
    b_coeff: float,
):
    i, j, sx, sy = wp.tid()
    u0 = _read_u(state, i, j, sx, sy, order)
    du_dx = wp.vec3(0.0)
    du_dy = wp.vec3(0.0)
    for q in range(order):
        du_dx += derivative[sx, q] * _read_u(state, i, j, q, sy, order) * inv_dx
        du_dy += derivative[sy, q] * _read_u(state, i, j, sx, q, order) * inv_dy

    ux = u0 - characteristic_speed * half_dt * du_dx
    uy = u0 - characteristic_speed * half_dt * du_dy
    inv_uc = 1.0 / characteristic_speed

    d_x_m10 = (_moment(ux, 1, 0) - _moment(u0, 1, 0)) * inv_uc
    d_x_m20 = (_moment(ux, 2, 0) - _moment(u0, 2, 0)) * inv_uc
    d_x_m11 = (_moment(ux, 1, 1) - _moment(u0, 1, 1)) * inv_uc
    d_y_m01 = (_moment(uy, 0, 1) - _moment(u0, 0, 1)) * inv_uc
    d_y_m11 = (_moment(uy, 1, 1) - _moment(u0, 1, 1)) * inv_uc
    d_y_m02 = (_moment(uy, 0, 2) - _moment(u0, 0, 2)) * inv_uc

    uh = wp.vec3(
        _moment(u0, 0, 0) + d_x_m10 + d_y_m01,
        _moment(u0, 1, 0) + d_x_m20 + d_y_m11,
        _moment(u0, 0, 1) + d_x_m11 + d_y_m02,
    )

    pi_xx = a_coeff * (
        _moment(u0, 2, 0)
        + (_moment(ux, 3, 0) - _moment(u0, 3, 0)) * inv_uc
        + (_moment(uy, 2, 1) - _moment(u0, 2, 1)) * inv_uc
    ) + (b_coeff - 1.0) * _moment(uh, 2, 0)
    pi_xy = a_coeff * (
        _moment(u0, 1, 1)
        + (_moment(ux, 2, 1) - _moment(u0, 2, 1)) * inv_uc
        + (_moment(uy, 1, 2) - _moment(u0, 1, 2)) * inv_uc
    ) + (b_coeff - 1.0) * _moment(uh, 1, 1)
    pi_yy = a_coeff * (
        _moment(u0, 0, 2)
        + (_moment(ux, 1, 2) - _moment(u0, 1, 2)) * inv_uc
        + (_moment(uy, 0, 3) - _moment(u0, 0, 3)) * inv_uc
    ) + (b_coeff - 1.0) * _moment(uh, 0, 2)

    sp = sx + sy * order
    for c in range(3):
        half_state[c, i, j, sp] = uh[c]
    half_stress[0, i, j, sp] = wp.float16(pi_xx)
    half_stress[1, i, j, sp] = wp.float16(pi_xy)
    half_stress[2, i, j, sp] = wp.float16(pi_yy)


@wp.func
def _face_state(
    state: wp.array4d[wp.float32],
    stress: wp.array4d[wp.float16],
    i: int,
    j: int,
    sx: int,
    sy: int,
    order: int,
    axis: int,
):
    u_state = _read_u(state, i, j, sx, sy, order)
    pi = _read_pi(stress, i, j, sx, sy, order)
    rho = u_state[0]
    u = u_state[1] / rho
    v = u_state[2] / rho
    result = vec5(rho, u, v, pi[0], pi[1])
    if axis == 1:
        result = vec5(rho, v, u, pi[2], pi[1])
    return result


@wp.func
def _common_flux(lhs: vec5, rhs: vec5, a_coeff: float, b_coeff: float, blend: float):
    c1 = wp.sqrt(2.0 * wp.pi * RT)
    c2 = 2.0 * wp.sqrt(2.0 * RT / wp.pi)
    c3 = wp.sqrt(RT / (2.0 * wp.pi))
    inv_c1 = 1.0 / c1
    inv_2c1 = 0.5 * inv_c1
    a_blend = (1.0 - blend) * a_coeff + blend
    b_blend = (1.0 - blend) * b_coeff

    c_l = lhs[RHO] * (0.5 + lhs[UN] * inv_c1)
    c_r = rhs[RHO] * (0.5 - rhs[UN] * inv_c1)
    rho_s = c_l + c_r
    rhoun_s = inv_2c1 * (
        lhs[RHO] * (2.0 * RT + c1 * lhs[UN] + lhs[UN] * lhs[UN])
        - rhs[RHO] * (2.0 * RT - c1 * rhs[UN] + rhs[UN] * rhs[UN])
        + lhs[PI_NN]
        - rhs[PI_NN]
    )
    un_s = rhoun_s / rho_s
    fbar_0 = a_blend * inv_2c1 * (
        lhs[RHO] * (2.0 * RT + c1 * lhs[UN] + lhs[UN] * lhs[UN])
        - rhs[RHO] * (2.0 * RT - c1 * rhs[UN] + rhs[UN] * rhs[UN])
    ) + inv_2c1 * (lhs[PI_NN] - rhs[PI_NN])
    fbar_1 = 0.5 * a_blend * (
        lhs[RHO] * (RT + c2 * lhs[UN] + lhs[UN] * lhs[UN]) + rhs[RHO] * (RT - c2 * rhs[UN] + rhs[UN] * rhs[UN])
    ) + 0.5 * (lhs[PI_NN] + rhs[PI_NN])
    fbar_2 = a_coeff * (
        lhs[RHO] * (c3 * lhs[UT] + 0.5 * lhs[UN] * lhs[UT]) + rhs[RHO] * (-c3 * rhs[UT] + 0.5 * rhs[UN] * rhs[UT])
    ) + 0.5 * (lhs[PI_NT] + rhs[PI_NT])
    mass = b_blend * rhoun_s + fbar_0
    normal = b_blend * (rho_s * un_s * un_s + rho_s * RT) + fbar_1
    tangent = fbar_2 + 0.5 * b_coeff * (rhoun_s * (lhs[UT] + rhs[UT]) + wp.abs(rhoun_s) * (lhs[UT] - rhs[UT]))
    return wp.vec3(mass, normal, tangent)


@wp.func
def _physical_flux(u_state: wp.vec3, pi: wp.vec3, axis: int):
    rho = u_state[0]
    u = u_state[1] / rho
    v = u_state[2] / rho
    result = wp.vec3(rho * u, rho * u * u + rho * RT + pi[0], rho * u * v + pi[1])
    if axis == 1:
        result = wp.vec3(rho * v, rho * u * v + pi[1], rho * v * v + rho * RT + pi[2])
    return result


@wp.func
def _local_derivative(
    state: wp.array4d[wp.float32],
    stress: wp.array4d[wp.float16],
    derivative: wp.array2d[wp.float32],
    q_matrix: wp.array2d[wp.float32],
    cr_matrix: wp.array2d[wp.float32],
    i: int,
    j: int,
    sx: int,
    sy: int,
    order: int,
    axis: int,
):
    s = sx
    if axis == 1:
        s = sy
    center = _read_u(state, i, j, sx, sy, order)
    rho = center[0]
    phi = wp.vec3(1.0, center[1] / rho, center[2] / rho)
    velocity = phi[1]
    if axis == 1:
        velocity = phi[2]
    rho_velocity = rho * velocity

    d_rv = wp.float32(0.0)
    d_phi = wp.vec3(0.0)
    d_rv_phi = wp.vec3(0.0)
    d_pressure = wp.float32(0.0)
    d_ne = wp.vec3(0.0)
    boundary_eq = wp.vec3(0.0)
    for t in range(order):
        tx = t
        ty = sy
        if axis == 1:
            tx = sx
            ty = t
        value = _read_u(state, i, j, tx, ty, order)
        pi = _read_pi(stress, i, j, tx, ty, order)
        rho_t = value[0]
        phi_t = wp.vec3(1.0, value[1] / rho_t, value[2] / rho_t)
        vel_t = phi_t[1]
        ne_t = wp.vec3(0.0, pi[0], pi[1])
        if axis == 1:
            vel_t = phi_t[2]
            ne_t = wp.vec3(0.0, pi[1], pi[2])
        rv_t = rho_t * vel_t
        d = derivative[s, t]
        d_rv += d * rv_t
        d_phi += d * phi_t
        d_rv_phi += d * rv_t * phi_t
        d_pressure += d * rho_t * RT
        d_ne += q_matrix[s, t] * ne_t
        boundary_eq += cr_matrix[s, t] * (_physical_flux(value, wp.vec3(0.0), axis))

    result = d_ne + 0.5 * (d_rv_phi + phi * d_rv + rho_velocity * d_phi) - boundary_eq
    if axis == 0:
        result[1] += d_pressure
    else:
        result[2] += d_pressure
    return result


@wp.kernel(enable_backward=False)
def _corrector(
    state: wp.array4d[wp.float32],
    half_state: wp.array4d[wp.float32],
    half_stress: wp.array4d[wp.float16],
    volume_fraction: wp.array3d[wp.float16],
    derivative: wp.array2d[wp.float32],
    correction: wp.array2d[wp.float32],
    cr_matrix: wp.array2d[wp.float32],
    q_matrix: wp.array2d[wp.float32],
    order: int,
    nx: int,
    ny: int,
    inv_dx: float,
    inv_dy: float,
    dt: float,
    a_coeff: float,
    b_coeff: float,
    blend: float,
):
    i, j, sx, sy = wp.tid()
    i_w = (i + nx - 1) % nx
    i_e = (i + 1) % nx
    j_s = (j + ny - 1) % ny
    j_n = (j + 1) % ny

    x_l = _common_flux(
        _face_state(half_state, half_stress, i_w, j, order - 1, sy, order, 0),
        _face_state(half_state, half_stress, i, j, 0, sy, order, 0),
        a_coeff,
        b_coeff,
        blend,
    )
    x_r = _common_flux(
        _face_state(half_state, half_stress, i, j, order - 1, sy, order, 0),
        _face_state(half_state, half_stress, i_e, j, 0, sy, order, 0),
        a_coeff,
        b_coeff,
        blend,
    )
    y_l_local = _common_flux(
        _face_state(half_state, half_stress, i, j_s, sx, order - 1, order, 1),
        _face_state(half_state, half_stress, i, j, sx, 0, order, 1),
        a_coeff,
        b_coeff,
        blend,
    )
    y_r_local = _common_flux(
        _face_state(half_state, half_stress, i, j, sx, order - 1, order, 1),
        _face_state(half_state, half_stress, i, j_n, sx, 0, order, 1),
        a_coeff,
        b_coeff,
        blend,
    )
    y_l = wp.vec3(y_l_local[0], y_l_local[2], y_l_local[1])
    y_r = wp.vec3(y_r_local[0], y_r_local[2], y_r_local[1])

    residual = (
        _local_derivative(half_state, half_stress, derivative, q_matrix, cr_matrix, i, j, sx, sy, order, 0)
        + correction[sx, 0] * x_l
        + correction[sx, 1] * x_r
    ) * inv_dx
    residual += (
        _local_derivative(half_state, half_stress, derivative, q_matrix, cr_matrix, i, j, sx, sy, order, 1)
        + correction[sy, 0] * y_l
        + correction[sy, 1] * y_r
    ) * inv_dy

    sp = sx + sy * order
    old = _read_u(state, i, j, sx, sy, order)
    fluid = 1.0 - wp.float32(volume_fraction[i, j, sp])
    old[1] *= fluid
    old[2] *= fluid
    updated = old - dt * residual
    for c in range(3):
        state[c, i, j, sp] = updated[c]


@dataclass
class KPMFR2DConfig:
    """KPM-FR 2D reference configuration."""

    resolution: tuple[int, int]
    size: tuple[float, float] = (2.0, 1.0)
    order: int = 4
    reynolds: float = 400_000.0
    reference_velocity: float = 0.1
    cfl: float = 0.5
    correction: str = "g2"
    blend: float = 0.0
    epsilon: float = 0.1


class KPMFR2D:
    """Direct Warp port of the KPM-FR 2D reference."""

    def __init__(self, config: KPMFR2DConfig, device: wp.DeviceLike = None):
        self.config = config
        self.device = wp.get_device(device)
        nx, ny = config.resolution
        k = config.order
        points, derivative, correction, cr_matrix, q_matrix = fr_operators(k, config.correction)
        self.points = points
        self.derivative = wp.array(derivative, device=self.device)
        self.correction = wp.array(correction, device=self.device)
        self.cr_matrix = wp.array(cr_matrix, device=self.device)
        self.q_matrix = wp.array(q_matrix, device=self.device)

        shape = (3, nx, ny, k * k)
        self.state = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.half_state = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.half_stress = wp.zeros(shape, dtype=wp.float16, device=self.device)
        self.volume_fraction = wp.zeros((nx, ny, k * k), dtype=wp.float16, device=self.device)

        dx = config.size[0] / nx
        dy = config.size[1] / ny
        viscosity = config.reference_velocity * min(config.size) / config.reynolds
        self.tau = viscosity / (1.0 / 3.0)
        cap = {3: 0.80, 4: 0.70, 5: 0.60, 6: 0.50}[k] * 0.8
        cfl = min(config.cfl, cap) * (0.5 if config.correction == "gdg" else 1.0)
        self.dt = cfl * min(dx, dy) / ((2 * k - 1) * (config.reference_velocity + math.sqrt(1.0 / 3.0)))
        self._half_dt = 0.5 * self.dt
        self._a = self.tau / self._half_dt
        self._b = 1.0 - self._a
        self._inv_dx = 2.0 / dx
        self._inv_dy = 2.0 / dy
        self._characteristic_speed = config.epsilon * math.sqrt(1.0 / 3.0)

    def initialize(self, density: float = 1.0, velocity: tuple[float, float] = (0.1, 0.0)):
        """Set a constant initial state."""
        host = np.empty(self.state.shape, dtype=np.float32)
        host[0].fill(density)
        host[1].fill(density * velocity[0])
        host[2].fill(density * velocity[1])
        self.state.assign(host)

    def step(self, count: int = 1):
        """Advance the fluid state."""
        nx, ny = self.config.resolution
        k = self.config.order
        for _ in range(count):
            wp.launch(
                _predictor,
                dim=(nx, ny, k, k),
                inputs=[
                    self.state,
                    self.half_state,
                    self.half_stress,
                    self.derivative,
                    k,
                    self._inv_dx,
                    self._inv_dy,
                    self._half_dt,
                    self._characteristic_speed,
                    self._a,
                    self._b,
                ],
                device=self.device,
            )
            wp.launch(
                _corrector,
                dim=(nx, ny, k, k),
                inputs=[
                    self.state,
                    self.half_state,
                    self.half_stress,
                    self.volume_fraction,
                    self.derivative,
                    self.correction,
                    self.cr_matrix,
                    self.q_matrix,
                    k,
                    nx,
                    ny,
                    self._inv_dx,
                    self._inv_dy,
                    self.dt,
                    self._a,
                    self._b,
                    self.config.blend,
                ],
                device=self.device,
            )
