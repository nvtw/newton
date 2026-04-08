# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drone hover environment -- Warp port of PufferLib ocean/drone (hover task).

A Crazyflie-style quadrotor must navigate to and hover near a randomly placed
target.  Physics are integrated with RK4 at 500 Hz (5 substeps per action at
100 Hz), matching the C++ reference exactly.

4 continuous actions (motor commands in [-1, 1]).
23 observations (body-frame velocity, angular velocity, quaternion, target
direction at two scales, target normal in body frame, normalised RPMs).
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

# -- Physical constants (Crazyflie 2.1) --
_MASS = wp.constant(0.027)
_IXX = wp.constant(3.85e-6)
_IYY = wp.constant(3.85e-6)
_IZZ = wp.constant(5.9675e-6)
_ARM_LEN = wp.constant(0.0396)
_K_THRUST = wp.constant(3.16e-10)
_K_DRAG = wp.constant(0.005964552)
_GRAVITY = wp.constant(9.81)
_MAX_RPM = wp.constant(21702.0)
_K_MOT = wp.constant(0.15)
_K_ANG_DAMP = wp.constant(0.0)
_B_DRAG = wp.constant(0.0)
_MAX_VEL = wp.constant(20.0)
_MAX_OMEGA = wp.constant(20.0)

_DT = wp.constant(0.002)
_ACTION_SUBSTEPS = wp.constant(5)
_SQRT2_INV = wp.constant(0.7071067811865476)  # 1/sqrt(2)

_MARGIN_X = wp.constant(29.0)
_MARGIN_Y = wp.constant(29.0)
_MARGIN_Z = wp.constant(9.0)

_HORIZON = wp.constant(1024)

# State layout per drone (flat float array):
#  0-2:  pos (x,y,z)
#  3-5:  vel (x,y,z)
#  6-9:  quat (w,x,y,z)
# 10-12: omega (x,y,z)
# 13-16: rpms[4]
# 17-19: prev_pos (x,y,z)
# 20:    prev_potential
# 21:    episode_return
# 22:    hover_ema
# 23:    ema_dist
# 24:    ema_vel
# 25:    ema_omega
_STATE_DIM = wp.constant(26)

# Params layout per drone (flat float array):
#  0: mass, 1: ixx, 2: iyy, 3: izz, 4: arm_len, 5: k_thrust
#  6: k_ang_damp, 7: k_drag, 8: b_drag, 9: gravity
# 10: max_rpm, 11: max_vel, 12: max_omega, 13: k_mot
_PARAMS_DIM = wp.constant(14)

# Target layout per drone: 0-2: pos, 3-5: vel, 6-8: normal
_TARGET_DIM = wp.constant(9)


@wp.func
def _qmul(aw: float, ax: float, ay: float, az: float,
           bw: float, bx: float, by: float, bz: float):
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return rw, rx, ry, rz


@wp.func
def _qrot(qw: float, qx: float, qy: float, qz: float,
           vx: float, vy: float, vz: float):
    tw, tx, ty, tz = _qmul(qw, qx, qy, qz, 0.0, vx, vy, vz)
    _, rx, ry, rz = _qmul(tw, tx, ty, tz, qw, -qx, -qy, -qz)
    return rx, ry, rz


@wp.func
def _qnorm(qw: float, qx: float, qy: float, qz: float):
    n = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    inv = 1.0 / wp.max(n, 1.0e-12)
    return qw * inv, qx * inv, qy * inv, qz * inv


@wp.func
def _norm3(x: float, y: float, z: float):
    return wp.sqrt(x * x + y * y + z * z)


@wp.func
def _rpm_hover(mass: float, gravity: float, k_thrust: float):
    return wp.sqrt((mass * gravity) / (4.0 * k_thrust))


@wp.func
def _rpm_min(mass: float, gravity: float, k_thrust: float, max_rpm: float):
    min_rpm = 2.0 * _rpm_hover(mass, gravity, k_thrust) - max_rpm
    return wp.clamp(min_rpm, 0.0, max_rpm)


@wp.func
def _hover_potential(
    tgt_x: float, tgt_y: float, tgt_z: float,
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    ox: float, oy: float, oz: float,
    hover_dist: float, hover_omega: float, hover_vel: float,
):
    dist = _norm3(tgt_x - px, tgt_y - py, tgt_z - pz)
    vel = _norm3(vx, vy, vz)
    omega = _norm3(ox, oy, oz)
    d = 1.0 / (1.0 + dist / hover_dist)
    v = 1.0 / (1.0 + vel / hover_vel)
    w = 1.0 / (1.0 + omega / hover_omega)
    return d * (0.7 + 0.15 * v + 0.15 * w)


@wp.func
def _check_hover(
    tgt_x: float, tgt_y: float, tgt_z: float,
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    ox: float, oy: float, oz: float,
    hover_dist: float, hover_omega: float, hover_vel: float,
):
    dist = _norm3(tgt_x - px, tgt_y - py, tgt_z - pz)
    vel = _norm3(vx, vy, vz)
    omega = _norm3(ox, oy, oz)
    d = dist / (hover_dist * 10.0)
    v = vel / (hover_vel * 10.0)
    w = omega / (hover_omega * 10.0)
    score = 1.0 - 0.7 * d - 0.15 * v - 0.15 * w
    return wp.max(score, 0.0)


# ---- RK4 integration (inlined for Warp kernel) ----
# Returns updated state components after one DT step.
# We pass actions as 4 floats, params as array slice.

@wp.func
def _compute_derivs(
    # state
    px: float, py: float, pz: float,
    vx: float, vy: float, vz: float,
    qw: float, qx: float, qy: float, qz: float,
    ox: float, oy: float, oz: float,
    rpm0: float, rpm1: float, rpm2: float, rpm3: float,
    # actions
    a0: float, a1: float, a2: float, a3: float,
    # params
    mass: float, ixx: float, iyy: float, izz: float,
    arm_len: float, k_thrust: float, k_ang_damp: float,
    k_drag: float, b_drag: float, gravity: float,
    max_rpm: float, k_mot: float,
):
    min_rpm = _rpm_min(mass, gravity, k_thrust, max_rpm)
    rng = max_rpm - min_rpm

    t0 = min_rpm + (a0 + 1.0) * 0.5 * rng
    t1 = min_rpm + (a1 + 1.0) * 0.5 * rng
    t2 = min_rpm + (a2 + 1.0) * 0.5 * rng
    t3 = min_rpm + (a3 + 1.0) * 0.5 * rng

    inv_k = 1.0 / k_mot
    rd0 = inv_k * (t0 - rpm0)
    rd1 = inv_k * (t1 - rpm1)
    rd2 = inv_k * (t2 - rpm2)
    rd3 = inv_k * (t3 - rpm3)

    r0 = wp.max(rpm0, 0.0)
    r1 = wp.max(rpm1, 0.0)
    r2 = wp.max(rpm2, 0.0)
    r3 = wp.max(rpm3, 0.0)
    T0 = k_thrust * r0 * r0
    T1 = k_thrust * r1 * r1
    T2 = k_thrust * r2 * r2
    T3 = k_thrust * r3 * r3

    Fz_body = T0 + T1 + T2 + T3
    Fpx, Fpy, Fpz = _qrot(qw, qx, qy, qz, 0.0, 0.0, Fz_body)

    vdx = (Fpx - b_drag * vx) / mass
    vdy = (Fpy - b_drag * vy) / mass
    vdz = (Fpz - b_drag * vz) / mass - gravity

    ow = 0.0
    owx = ox
    owy = oy
    owz = oz
    qdw, qdx, qdy, qdz = _qmul(qw, qx, qy, qz, ow, owx, owy, owz)
    qdw = qdw * 0.5
    qdx = qdx * 0.5
    qdy = qdy * 0.5
    qdz = qdz * 0.5

    af = arm_len * _SQRT2_INV
    tau_x = af * ((T2 + T3) - (T0 + T1))
    tau_y = af * ((T1 + T2) - (T0 + T3))
    tau_z = k_drag * (-T0 + T1 - T2 + T3)

    tau_x = tau_x - k_ang_damp * ox
    tau_y = tau_y - k_ang_damp * oy
    tau_z = tau_z - k_ang_damp * oz

    tau_x = tau_x + (iyy - izz) * oy * oz
    tau_y = tau_y + (izz - ixx) * oz * ox
    tau_z = tau_z + (ixx - iyy) * ox * oy

    wdx = tau_x / ixx
    wdy = tau_y / iyy
    wdz = tau_z / izz

    # returns: d_pos(3), d_vel(3), d_quat(4), d_omega(3), d_rpm(4) = 17 values
    return (vx, vy, vz,
            vdx, vdy, vdz,
            qdw, qdx, qdy, qdz,
            wdx, wdy, wdz,
            rd0, rd1, rd2, rd3)


@wp.kernel
def _drone_reset_kernel(
    state: wp.array2d(dtype=float),
    params: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    ep_len: wp.array(dtype=int, ndim=1),
    obs: wp.array2d(dtype=float),
    seed: int,
    hover_dist: float,
    hover_omega: float,
    hover_vel: float,
    hover_target_dist: float,
):
    i = wp.tid()
    rng = wp.rand_init(seed, i * 100)
    dr = 0.05

    mass = _MASS * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    ixx = _IXX * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    iyy = _IYY * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    izz = _IZZ * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    arm_len = _ARM_LEN * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    k_thrust = _K_THRUST * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    k_ang_damp = _K_ANG_DAMP * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    k_drag = _K_DRAG * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    b_drag = _B_DRAG * wp.randf(rng, 1.0 - dr, 1.0 + dr)
    gravity = _GRAVITY * wp.randf(rng, 0.99, 1.01)
    k_mot = _K_MOT * wp.randf(rng, 1.0 - dr, 1.0 + dr)

    params[i, 0] = mass
    params[i, 1] = ixx
    params[i, 2] = iyy
    params[i, 3] = izz
    params[i, 4] = arm_len
    params[i, 5] = k_thrust
    params[i, 6] = k_ang_damp
    params[i, 7] = k_drag
    params[i, 8] = b_drag
    params[i, 9] = gravity
    params[i, 10] = _MAX_RPM
    params[i, 11] = _MAX_VEL
    params[i, 12] = _MAX_OMEGA
    params[i, 13] = k_mot

    hover = _rpm_hover(mass, gravity, k_thrust)

    px = wp.randf(rng, -_MARGIN_X, _MARGIN_X)
    py = wp.randf(rng, -_MARGIN_Y, _MARGIN_Y)
    pz = wp.randf(rng, -_MARGIN_Z, _MARGIN_Z)

    state[i, 0] = px
    state[i, 1] = py
    state[i, 2] = pz
    state[i, 3] = 0.0  # vel
    state[i, 4] = 0.0
    state[i, 5] = 0.0
    state[i, 6] = 1.0  # quat w
    state[i, 7] = 0.0
    state[i, 8] = 0.0
    state[i, 9] = 0.0
    state[i, 10] = 0.0  # omega
    state[i, 11] = 0.0
    state[i, 12] = 0.0
    state[i, 13] = hover  # rpms
    state[i, 14] = hover
    state[i, 15] = hover
    state[i, 16] = hover
    state[i, 17] = px  # prev_pos
    state[i, 18] = py
    state[i, 19] = pz
    # prev_potential, episode_return, hover_ema, ema_dist/vel/omega set below
    state[i, 21] = 0.0
    state[i, 22] = 0.0
    state[i, 23] = 0.0
    state[i, 24] = 0.0
    state[i, 25] = 0.0

    # set hover target
    u = wp.randf(rng)
    v = wp.randf(rng)
    z_dir = 2.0 * v - 1.0
    a = 2.0 * 3.14159265358979 * u
    r_xy = wp.sqrt(wp.max(0.0, 1.0 - z_dir * z_dir))
    dx = r_xy * wp.cos(a)
    dy = r_xy * wp.sin(a)
    dz = z_dir
    rad = hover_target_dist * wp.pow(wp.randf(rng), 1.0 / 3.0)
    tx = wp.clamp(px + dx * rad, -_MARGIN_X, _MARGIN_X)
    ty = wp.clamp(py + dy * rad, -_MARGIN_Y, _MARGIN_Y)
    tz = wp.clamp(pz + dz * rad, -_MARGIN_Z, _MARGIN_Z)
    target[i, 0] = tx
    target[i, 1] = ty
    target[i, 2] = tz
    target[i, 3] = 0.0  # vel
    target[i, 4] = 0.0
    target[i, 5] = 0.0
    target[i, 6] = 0.0  # normal (not used for hover but needed for obs)
    target[i, 7] = 0.0
    target[i, 8] = 1.0

    pot = _hover_potential(tx, ty, tz, px, py, pz,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           hover_dist, hover_omega, hover_vel)
    state[i, 20] = pot

    ep_len[i] = 0

    # compute obs
    qw = 1.0
    qx_v = 0.0
    qy_v = 0.0
    qz_v = 0.0
    qi_w = qw
    qi_x = -qx_v
    qi_y = -qy_v
    qi_z = -qz_v
    max_vel = _MAX_VEL
    max_omega = _MAX_OMEGA
    denom = max_vel * 1.7320508
    bvx, bvy, bvz = _qrot(qi_w, qi_x, qi_y, qi_z, 0.0, 0.0, 0.0)
    obs[i, 0] = bvx / denom
    obs[i, 1] = bvy / denom
    obs[i, 2] = bvz / denom
    obs[i, 3] = 0.0
    obs[i, 4] = 0.0
    obs[i, 5] = 0.0
    obs[i, 6] = qw
    obs[i, 7] = qx_v
    obs[i, 8] = qy_v
    obs[i, 9] = qz_v
    ttx = tx - px
    tty = ty - py
    ttz = tz - pz
    btx, bty, btz = _qrot(qi_w, qi_x, qi_y, qi_z, ttx, tty, ttz)
    obs[i, 10] = wp.tanh(btx * 0.1)
    obs[i, 11] = wp.tanh(bty * 0.1)
    obs[i, 12] = wp.tanh(btz * 0.1)
    obs[i, 13] = wp.tanh(btx * 10.0)
    obs[i, 14] = wp.tanh(bty * 10.0)
    obs[i, 15] = wp.tanh(btz * 10.0)
    bnx, bny, bnz = _qrot(qi_w, qi_x, qi_y, qi_z, 0.0, 0.0, 1.0)
    obs[i, 16] = bnx
    obs[i, 17] = bny
    obs[i, 18] = bnz
    obs[i, 19] = hover / _MAX_RPM
    obs[i, 20] = hover / _MAX_RPM
    obs[i, 21] = hover / _MAX_RPM
    obs[i, 22] = hover / _MAX_RPM


@wp.kernel
def _drone_step_kernel(
    state: wp.array2d(dtype=float),
    params: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    ep_len: wp.array(dtype=int, ndim=1),
    actions: wp.array2d(dtype=float),
    obs: wp.array2d(dtype=float),
    rewards_out: wp.array(dtype=float, ndim=1),
    dones_out: wp.array(dtype=float, ndim=1),
    episode_returns_out: wp.array(dtype=float, ndim=1),
    episode_lengths_out: wp.array(dtype=float, ndim=1),
    seed_arr: wp.array(dtype=int, ndim=1),
    hover_dist: float,
    hover_omega: float,
    hover_vel: float,
    hover_target_dist: float,
    alpha_dist: float,
    alpha_hover: float,
    alpha_shaping: float,
    alpha_omega: float,
):
    i = wp.tid()

    # load params
    mass = params[i, 0]
    ixx = params[i, 1]
    iyy = params[i, 2]
    izz = params[i, 3]
    arm_len = params[i, 4]
    k_thrust = params[i, 5]
    k_ang_damp = params[i, 6]
    k_drag = params[i, 7]
    b_drag = params[i, 8]
    gravity = params[i, 9]
    max_rpm = params[i, 10]
    max_vel = params[i, 11]
    max_omega = params[i, 12]
    k_mot = params[i, 13]

    # load state
    px = state[i, 0]
    py = state[i, 1]
    pz = state[i, 2]
    vx = state[i, 3]
    vy = state[i, 4]
    vz = state[i, 5]
    qw = state[i, 6]
    qx_v = state[i, 7]
    qy_v = state[i, 8]
    qz_v = state[i, 9]
    ox = state[i, 10]
    oy = state[i, 11]
    oz = state[i, 12]
    rpm0 = state[i, 13]
    rpm1 = state[i, 14]
    rpm2 = state[i, 15]
    rpm3 = state[i, 16]

    prev_px = px
    prev_py = py
    prev_pz = pz
    state[i, 17] = px
    state[i, 18] = py
    state[i, 19] = pz

    a0 = wp.clamp(actions[i, 0], -1.0, 1.0)
    a1 = wp.clamp(actions[i, 1], -1.0, 1.0)
    a2 = wp.clamp(actions[i, 2], -1.0, 1.0)
    a3 = wp.clamp(actions[i, 3], -1.0, 1.0)

    ep = ep_len[i] + 1

    # RK4 x ACTION_SUBSTEPS
    for _s in range(5):
        # k1
        (dv1x, dv1y, dv1z, da1x, da1y, da1z,
         dq1w, dq1x, dq1y, dq1z,
         dw1x, dw1y, dw1z,
         dr10, dr11, dr12, dr13) = _compute_derivs(
            px, py, pz, vx, vy, vz, qw, qx_v, qy_v, qz_v,
            ox, oy, oz, rpm0, rpm1, rpm2, rpm3,
            a0, a1, a2, a3,
            mass, ixx, iyy, izz, arm_len, k_thrust, k_ang_damp,
            k_drag, b_drag, gravity, max_rpm, k_mot)

        dt = _DT
        hdt = dt * 0.5

        # k2 state
        t_px = px + dv1x * hdt
        t_py = py + dv1y * hdt
        t_pz = pz + dv1z * hdt
        t_vx = vx + da1x * hdt
        t_vy = vy + da1y * hdt
        t_vz = vz + da1z * hdt
        t_qw = qw + dq1w * hdt
        t_qx = qx_v + dq1x * hdt
        t_qy = qy_v + dq1y * hdt
        t_qz = qz_v + dq1z * hdt
        t_qw, t_qx, t_qy, t_qz = _qnorm(t_qw, t_qx, t_qy, t_qz)
        t_ox = ox + dw1x * hdt
        t_oy = oy + dw1y * hdt
        t_oz = oz + dw1z * hdt
        t_r0 = rpm0 + dr10 * hdt
        t_r1 = rpm1 + dr11 * hdt
        t_r2 = rpm2 + dr12 * hdt
        t_r3 = rpm3 + dr13 * hdt

        (dv2x, dv2y, dv2z, da2x, da2y, da2z,
         dq2w, dq2x, dq2y, dq2z,
         dw2x, dw2y, dw2z,
         dr20, dr21, dr22, dr23) = _compute_derivs(
            t_px, t_py, t_pz, t_vx, t_vy, t_vz, t_qw, t_qx, t_qy, t_qz,
            t_ox, t_oy, t_oz, t_r0, t_r1, t_r2, t_r3,
            a0, a1, a2, a3,
            mass, ixx, iyy, izz, arm_len, k_thrust, k_ang_damp,
            k_drag, b_drag, gravity, max_rpm, k_mot)

        # k3 state
        t_px = px + dv2x * hdt
        t_py = py + dv2y * hdt
        t_pz = pz + dv2z * hdt
        t_vx = vx + da2x * hdt
        t_vy = vy + da2y * hdt
        t_vz = vz + da2z * hdt
        t_qw = qw + dq2w * hdt
        t_qx = qx_v + dq2x * hdt
        t_qy = qy_v + dq2y * hdt
        t_qz = qz_v + dq2z * hdt
        t_qw, t_qx, t_qy, t_qz = _qnorm(t_qw, t_qx, t_qy, t_qz)
        t_ox = ox + dw2x * hdt
        t_oy = oy + dw2y * hdt
        t_oz = oz + dw2z * hdt
        t_r0 = rpm0 + dr20 * hdt
        t_r1 = rpm1 + dr21 * hdt
        t_r2 = rpm2 + dr22 * hdt
        t_r3 = rpm3 + dr23 * hdt

        (dv3x, dv3y, dv3z, da3x, da3y, da3z,
         dq3w, dq3x, dq3y, dq3z,
         dw3x, dw3y, dw3z,
         dr30, dr31, dr32, dr33) = _compute_derivs(
            t_px, t_py, t_pz, t_vx, t_vy, t_vz, t_qw, t_qx, t_qy, t_qz,
            t_ox, t_oy, t_oz, t_r0, t_r1, t_r2, t_r3,
            a0, a1, a2, a3,
            mass, ixx, iyy, izz, arm_len, k_thrust, k_ang_damp,
            k_drag, b_drag, gravity, max_rpm, k_mot)

        # k4 state
        t_px = px + dv3x * dt
        t_py = py + dv3y * dt
        t_pz = pz + dv3z * dt
        t_vx = vx + da3x * dt
        t_vy = vy + da3y * dt
        t_vz = vz + da3z * dt
        t_qw = qw + dq3w * dt
        t_qx = qx_v + dq3x * dt
        t_qy = qy_v + dq3y * dt
        t_qz = qz_v + dq3z * dt
        t_qw, t_qx, t_qy, t_qz = _qnorm(t_qw, t_qx, t_qy, t_qz)
        t_ox = ox + dw3x * dt
        t_oy = oy + dw3y * dt
        t_oz = oz + dw3z * dt
        t_r0 = rpm0 + dr30 * dt
        t_r1 = rpm1 + dr31 * dt
        t_r2 = rpm2 + dr32 * dt
        t_r3 = rpm3 + dr33 * dt

        (dv4x, dv4y, dv4z, da4x, da4y, da4z,
         dq4w, dq4x, dq4y, dq4z,
         dw4x, dw4y, dw4z,
         dr40, dr41, dr42, dr43) = _compute_derivs(
            t_px, t_py, t_pz, t_vx, t_vy, t_vz, t_qw, t_qx, t_qy, t_qz,
            t_ox, t_oy, t_oz, t_r0, t_r1, t_r2, t_r3,
            a0, a1, a2, a3,
            mass, ixx, iyy, izz, arm_len, k_thrust, k_ang_damp,
            k_drag, b_drag, gravity, max_rpm, k_mot)

        dt6 = dt / 6.0
        px = px + (dv1x + 2.0 * dv2x + 2.0 * dv3x + dv4x) * dt6
        py = py + (dv1y + 2.0 * dv2y + 2.0 * dv3y + dv4y) * dt6
        pz = pz + (dv1z + 2.0 * dv2z + 2.0 * dv3z + dv4z) * dt6
        vx = vx + (da1x + 2.0 * da2x + 2.0 * da3x + da4x) * dt6
        vy = vy + (da1y + 2.0 * da2y + 2.0 * da3y + da4y) * dt6
        vz = vz + (da1z + 2.0 * da2z + 2.0 * da3z + da4z) * dt6
        qw = qw + (dq1w + 2.0 * dq2w + 2.0 * dq3w + dq4w) * dt6
        qx_v = qx_v + (dq1x + 2.0 * dq2x + 2.0 * dq3x + dq4x) * dt6
        qy_v = qy_v + (dq1y + 2.0 * dq2y + 2.0 * dq3y + dq4y) * dt6
        qz_v = qz_v + (dq1z + 2.0 * dq2z + 2.0 * dq3z + dq4z) * dt6
        ox = ox + (dw1x + 2.0 * dw2x + 2.0 * dw3x + dw4x) * dt6
        oy = oy + (dw1y + 2.0 * dw2y + 2.0 * dw3y + dw4y) * dt6
        oz = oz + (dw1z + 2.0 * dw2z + 2.0 * dw3z + dw4z) * dt6
        rpm0 = rpm0 + (dr10 + 2.0 * dr20 + 2.0 * dr30 + dr40) * dt6
        rpm1 = rpm1 + (dr11 + 2.0 * dr21 + 2.0 * dr31 + dr41) * dt6
        rpm2 = rpm2 + (dr12 + 2.0 * dr22 + 2.0 * dr32 + dr42) * dt6
        rpm3 = rpm3 + (dr13 + 2.0 * dr23 + 2.0 * dr33 + dr43) * dt6

        qw, qx_v, qy_v, qz_v = _qnorm(qw, qx_v, qy_v, qz_v)

        vx = wp.clamp(vx, -max_vel, max_vel)
        vy = wp.clamp(vy, -max_vel, max_vel)
        vz = wp.clamp(vz, -max_vel, max_vel)
        ox = wp.clamp(ox, -max_omega, max_omega)
        oy = wp.clamp(oy, -max_omega, max_omega)
        oz = wp.clamp(oz, -max_omega, max_omega)
        rpm0 = wp.clamp(rpm0, 0.0, max_rpm)
        rpm1 = wp.clamp(rpm1, 0.0, max_rpm)
        rpm2 = wp.clamp(rpm2, 0.0, max_rpm)
        rpm3 = wp.clamp(rpm3, 0.0, max_rpm)

    # store updated state
    state[i, 0] = px
    state[i, 1] = py
    state[i, 2] = pz
    state[i, 3] = vx
    state[i, 4] = vy
    state[i, 5] = vz
    state[i, 6] = qw
    state[i, 7] = qx_v
    state[i, 8] = qy_v
    state[i, 9] = qz_v
    state[i, 10] = ox
    state[i, 11] = oy
    state[i, 12] = oz
    state[i, 13] = rpm0
    state[i, 14] = rpm1
    state[i, 15] = rpm2
    state[i, 16] = rpm3

    # reward
    tx = target[i, 0]
    ty = target[i, 1]
    tz = target[i, 2]

    curr_pot = _hover_potential(tx, ty, tz, px, py, pz, vx, vy, vz, ox, oy, oz,
                                hover_dist, hover_omega, hover_vel)
    prev_dist = _norm3(tx - prev_px, ty - prev_py, tz - prev_pz)
    curr_dist = _norm3(tx - px, ty - py, tz - pz)
    omega_mag = _norm3(ox, oy, oz)

    prev_pot = state[i, 20]
    reward = (alpha_dist * (prev_dist - curr_dist)
              + alpha_hover * curr_pot
              + alpha_shaping * (curr_pot - prev_pot)
              - alpha_omega * omega_mag)
    state[i, 20] = curr_pot

    h = _check_hover(tx, ty, tz, px, py, pz, vx, vy, vz, ox, oy, oz,
                     hover_dist, hover_omega, hover_vel)
    state[i, 22] = 0.98 * state[i, 22] + 0.02 * h
    state[i, 23] = 0.99 * state[i, 23] + 0.01 * curr_dist
    state[i, 24] = 0.99 * state[i, 24] + 0.01 * _norm3(vx, vy, vz)
    state[i, 25] = 0.99 * state[i, 25] + 0.01 * omega_mag

    ep_ret = state[i, 21] + reward
    state[i, 21] = ep_ret

    oob = 0
    if curr_dist > hover_target_dist + 1.0:
        oob = 1
    timeout = 0
    if ep >= _HORIZON:
        timeout = 1
    done = 0
    if oob == 1 or timeout == 1:
        done = 1

    rewards_out[i] = reward
    dones_out[i] = float(done)
    episode_returns_out[i] = ep_ret
    episode_lengths_out[i] = float(ep)

    if done == 1:
        rng = wp.rand_init(seed_arr[0], i + ep)
        dr2 = 0.05

        n_mass = _MASS * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_ixx = _IXX * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_iyy = _IYY * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_izz = _IZZ * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_arm = _ARM_LEN * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_kt = _K_THRUST * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_kad = _K_ANG_DAMP * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_kd = _K_DRAG * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_bd = _B_DRAG * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)
        n_grav = _GRAVITY * wp.randf(rng, 0.99, 1.01)
        n_km = _K_MOT * wp.randf(rng, 1.0 - dr2, 1.0 + dr2)

        params[i, 0] = n_mass
        params[i, 1] = n_ixx
        params[i, 2] = n_iyy
        params[i, 3] = n_izz
        params[i, 4] = n_arm
        params[i, 5] = n_kt
        params[i, 6] = n_kad
        params[i, 7] = n_kd
        params[i, 8] = n_bd
        params[i, 9] = n_grav
        params[i, 13] = n_km

        n_hover = _rpm_hover(n_mass, n_grav, n_kt)

        n_px = wp.randf(rng, -_MARGIN_X, _MARGIN_X)
        n_py = wp.randf(rng, -_MARGIN_Y, _MARGIN_Y)
        n_pz = wp.randf(rng, -_MARGIN_Z, _MARGIN_Z)

        state[i, 0] = n_px
        state[i, 1] = n_py
        state[i, 2] = n_pz
        state[i, 3] = 0.0
        state[i, 4] = 0.0
        state[i, 5] = 0.0
        state[i, 6] = 1.0
        state[i, 7] = 0.0
        state[i, 8] = 0.0
        state[i, 9] = 0.0
        state[i, 10] = 0.0
        state[i, 11] = 0.0
        state[i, 12] = 0.0
        state[i, 13] = n_hover
        state[i, 14] = n_hover
        state[i, 15] = n_hover
        state[i, 16] = n_hover
        state[i, 17] = n_px
        state[i, 18] = n_py
        state[i, 19] = n_pz
        state[i, 21] = 0.0
        state[i, 22] = 0.0
        state[i, 23] = 0.0
        state[i, 24] = 0.0
        state[i, 25] = 0.0

        u2 = wp.randf(rng)
        v2 = wp.randf(rng)
        z2 = 2.0 * v2 - 1.0
        a2 = 2.0 * 3.14159265358979 * u2
        r2 = wp.sqrt(wp.max(0.0, 1.0 - z2 * z2))
        d2x = r2 * wp.cos(a2)
        d2y = r2 * wp.sin(a2)
        d2z = z2
        rad2 = hover_target_dist * wp.pow(wp.randf(rng), 1.0 / 3.0)
        n_tx = wp.clamp(n_px + d2x * rad2, -_MARGIN_X, _MARGIN_X)
        n_ty = wp.clamp(n_py + d2y * rad2, -_MARGIN_Y, _MARGIN_Y)
        n_tz = wp.clamp(n_pz + d2z * rad2, -_MARGIN_Z, _MARGIN_Z)
        target[i, 0] = n_tx
        target[i, 1] = n_ty
        target[i, 2] = n_tz
        target[i, 3] = 0.0
        target[i, 4] = 0.0
        target[i, 5] = 0.0
        target[i, 6] = 0.0
        target[i, 7] = 0.0
        target[i, 8] = 1.0

        pot2 = _hover_potential(n_tx, n_ty, n_tz, n_px, n_py, n_pz,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                hover_dist, hover_omega, hover_vel)
        state[i, 20] = pot2
        ep_len[i] = 0

        px = n_px
        py = n_py
        pz = n_pz
        vx = 0.0
        vy = 0.0
        vz = 0.0
        qw = 1.0
        qx_v = 0.0
        qy_v = 0.0
        qz_v = 0.0
        ox = 0.0
        oy = 0.0
        oz = 0.0
        rpm0 = n_hover
        rpm1 = n_hover
        rpm2 = n_hover
        rpm3 = n_hover
        tx = n_tx
        ty = n_ty
        tz = n_tz
    else:
        ep_len[i] = ep

    # compute observations
    qi_w = qw
    qi_x = -qx_v
    qi_y = -qy_v
    qi_z = -qz_v
    denom_v = max_vel * 1.7320508
    bvx2, bvy2, bvz2 = _qrot(qi_w, qi_x, qi_y, qi_z, vx, vy, vz)
    obs[i, 0] = bvx2 / denom_v
    obs[i, 1] = bvy2 / denom_v
    obs[i, 2] = bvz2 / denom_v
    obs[i, 3] = ox / max_omega
    obs[i, 4] = oy / max_omega
    obs[i, 5] = oz / max_omega
    obs[i, 6] = qw
    obs[i, 7] = qx_v
    obs[i, 8] = qy_v
    obs[i, 9] = qz_v
    ttx2 = tx - px
    tty2 = ty - py
    ttz2 = tz - pz
    btx2, bty2, btz2 = _qrot(qi_w, qi_x, qi_y, qi_z, ttx2, tty2, ttz2)
    obs[i, 10] = wp.tanh(btx2 * 0.1)
    obs[i, 11] = wp.tanh(bty2 * 0.1)
    obs[i, 12] = wp.tanh(btz2 * 0.1)
    obs[i, 13] = wp.tanh(btx2 * 10.0)
    obs[i, 14] = wp.tanh(bty2 * 10.0)
    obs[i, 15] = wp.tanh(btz2 * 10.0)
    tnx = target[i, 6]
    tny = target[i, 7]
    tnz = target[i, 8]
    bnx2, bny2, bnz2 = _qrot(qi_w, qi_x, qi_y, qi_z, tnx, tny, tnz)
    obs[i, 16] = bnx2
    obs[i, 17] = bny2
    obs[i, 18] = bnz2
    obs[i, 19] = rpm0 / max_rpm
    obs[i, 20] = rpm1 / max_rpm
    obs[i, 21] = rpm2 / max_rpm
    obs[i, 22] = rpm3 / max_rpm


class DroneEnv:
    """Vectorized Crazyflie drone hover environment on GPU.

    Each environment instance is an independent drone that must hover near a
    randomly placed target.  Physics match the C++ PufferLib reference exactly
    (RK4 integration, 5 substeps at 500 Hz).

    Args:
        num_envs: Number of parallel drones.
        device: Warp device string.
        seed: Random seed.
        hover_target_dist: Max distance for target placement.
        hover_dist: Distance scale for hover potential.
        hover_omega: Angular velocity scale for hover potential.
        hover_vel: Velocity scale for hover potential.
        alpha_dist: Reward weight for distance reduction.
        alpha_hover: Reward weight for hover potential.
        alpha_shaping: Reward weight for potential shaping.
        alpha_omega: Reward penalty weight for angular velocity.
    """

    NUM_ACTIONS = 4
    OBS_SIZE = 23

    def __init__(
        self,
        num_envs: int = 2048,
        device: str = "cuda:0",
        seed: int = 42,
        hover_target_dist: float = 5.0,
        hover_dist: float = 0.1,
        hover_omega: float = 0.1,
        hover_vel: float = 0.1,
        alpha_dist: float = 0.782192,
        alpha_hover: float = 0.071445,
        alpha_shaping: float = 3.9754,
        alpha_omega: float = 0.00135588,
    ):
        self.num_envs = num_envs
        self.obs_size = 23
        self.device = device
        self.seed = seed

        self.hover_target_dist = hover_target_dist
        self.hover_dist = hover_dist
        self.hover_omega = hover_omega
        self.hover_vel = hover_vel
        self.alpha_dist = alpha_dist
        self.alpha_hover = alpha_hover
        self.alpha_shaping = alpha_shaping
        self.alpha_omega = alpha_omega

        self._state = wp.zeros((num_envs, 26), dtype=float, device=device)
        self._params = wp.zeros((num_envs, 14), dtype=float, device=device)
        self._target = wp.zeros((num_envs, 9), dtype=float, device=device)
        self._ep_len = wp.zeros(num_envs, dtype=int, device=device)

        self.obs = wp.zeros((num_envs, 23), dtype=float, device=device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=device)
        self.dones = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_returns = wp.zeros(num_envs, dtype=float, device=device)
        self.episode_lengths = wp.zeros(num_envs, dtype=float, device=device)

        self.reset()

    def reset(self):
        wp.launch(
            _drone_reset_kernel,
            dim=self.num_envs,
            inputs=[
                self._state, self._params, self._target, self._ep_len,
                self.obs, self.seed,
                self.hover_dist, self.hover_omega, self.hover_vel,
                self.hover_target_dist,
            ],
            device=self.device,
        )
        return self.obs

    def step_graphed(self, actions: wp.array, seed_arr: wp.array):
        """Graph-capture-compatible step.

        Args:
            actions: ``(N, 4)`` continuous motor commands.
            seed_arr: Device-side 1-element seed array.
        """
        wp.launch(
            _drone_step_kernel,
            dim=self.num_envs,
            inputs=[
                self._state, self._params, self._target, self._ep_len,
                actions, self.obs, self.rewards, self.dones,
                self.episode_returns, self.episode_lengths,
                seed_arr,
                self.hover_dist, self.hover_omega, self.hover_vel,
                self.hover_target_dist,
                self.alpha_dist, self.alpha_hover, self.alpha_shaping,
                self.alpha_omega,
            ],
            device=self.device,
        )

    def get_episode_stats(self) -> dict:
        returns_np = self.episode_returns.numpy()
        lengths_np = self.episode_lengths.numpy()
        dones_np = self.dones.numpy()
        done_mask = dones_np > 0.5
        if np.any(done_mask):
            return {
                "mean_return": float(np.mean(returns_np[done_mask])),
                "mean_length": float(np.mean(lengths_np[done_mask])),
                "num_episodes": int(np.sum(done_mask)),
            }
        return {"mean_return": 0.0, "mean_length": 0.0, "num_episodes": 0}
