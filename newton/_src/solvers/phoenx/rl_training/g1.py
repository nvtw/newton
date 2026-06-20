# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils

from . import g1_recipe
from .env import advance_seed_counter, collect_ppo_rollout, collect_ppo_rollout_seed_counter
from .ppo import BufferRollout, MirrorMapPPO, TrainerPPO

ACTION_DIM_G1 = 29
OBS_DIM_G1 = 98
NANOG1_PHASE_PERIOD = g1_recipe.PHASE_PERIOD

_G1_OBS_MIRROR_SRC = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    15,
    16,
    17,
    18,
    19,
    20,
    9,
    10,
    11,
    12,
    13,
    14,
    21,
    22,
    23,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    44,
    45,
    46,
    47,
    48,
    49,
    38,
    39,
    40,
    41,
    42,
    43,
    50,
    51,
    52,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    73,
    74,
    75,
    76,
    77,
    78,
    67,
    68,
    69,
    70,
    71,
    72,
    79,
    80,
    81,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    96,
    97,
)
_G1_OBS_MIRROR_SIGN = (
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
)
_G1_ACTION_MIRROR_SRC = (
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    12,
    13,
    14,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
)
_G1_ACTION_MIRROR_SIGN = (
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
)


def g1_mirror_map_ppo() -> MirrorMapPPO:
    """Return nanoG1's validated left/right G1 PPO mirror map."""

    return MirrorMapPPO(
        obs_src=_G1_OBS_MIRROR_SRC,
        obs_sign=_G1_OBS_MIRROR_SIGN,
        action_src=_G1_ACTION_MIRROR_SRC,
        action_sign=_G1_ACTION_MIRROR_SIGN,
    )


_DEFAULT_JOINT_POS_G1 = (
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20,
    0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    0.20,
    -0.20,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)

_UNITREE_KP_G1 = (
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
    75.0,
    75.0,
    75.0,
    75.0,
    2.0,
    2.0,
    2.0,
)

_UNITREE_KD_G1 = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
    2.0,
    2.0,
    2.0,
    2.0,
    0.2,
    0.2,
    0.2,
)

_CTRL_LO_G1 = (
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.97222,
    -1.61443,
    -1.61443,
)

_CTRL_HI_G1 = (
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.97222,
    1.61443,
    1.61443,
)


@wp.func
def _clip_float(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _finite_or_zero(x: wp.float32) -> wp.float32:
    if wp.isfinite(x):
        return x
    return wp.float32(0.0)


@wp.func
def _clip_finite(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return _clip_float(_finite_or_zero(x), lo, hi)


@wp.func
def _quat_rotate_inverse_wxyz(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    q = wp.vec3(qx, qy, qz)
    a = v * (wp.float32(2.0) * qw * qw - wp.float32(1.0))
    b = wp.cross(q, v) * qw * wp.float32(2.0)
    c = q * (wp.dot(q, v) * wp.float32(2.0))
    return a - b + c


@wp.kernel
def g1_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    ctrl_lower: wp.array[wp.float32],
    ctrl_upper: wp.array[wp.float32],
    action_scale: wp.float32,
    controlled_action_count: wp.int32,
    dof_stride: wp.int32,
    coord_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    world, action = wp.tid()
    value = _clip_float(actions[world, action], wp.float32(-1.0), wp.float32(1.0))
    if action >= controlled_action_count:
        value = wp.float32(0.0)
    current_actions[world, action] = value

    target = default_joint_pos[action] + action_scale * value
    target = _clip_float(target, ctrl_lower[action], ctrl_upper[action])
    if target_uses_coord_layout != 0:
        joint_target_q[world * coord_stride + wp.int32(7) + action] = target
    else:
        joint_target_q[world * dof_stride + wp.int32(6) + action] = target


@wp.kernel
def g1_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    phase_period: wp.int32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    action_scale: wp.float32,
    w_track_lin: wp.float32,
    w_track_ang: wp.float32,
    w_lin_vel_z: wp.float32,
    w_ang_vel_xy: wp.float32,
    w_orientation: wp.float32,
    w_torque: wp.float32,
    w_action_rate: wp.float32,
    w_alive: wp.float32,
    w_termination: wp.float32,
    actuator_ke: wp.array[wp.float32],
    actuator_kd: wp.array[wp.float32],
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride

    qw = joint_q[q_base + wp.int32(3)]
    qx = joint_q[q_base + wp.int32(4)]
    qy = joint_q[q_base + wp.int32(5)]
    qz = joint_q[q_base + wp.int32(6)]
    lin_w = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + wp.int32(1)], joint_qd[qd_base + wp.int32(2)])
    ang = wp.vec3(
        joint_qd[qd_base + wp.int32(3)],
        joint_qd[qd_base + wp.int32(4)],
        joint_qd[qd_base + wp.int32(5)],
    )
    lin_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, lin_w)
    gravity_b = _quat_rotate_inverse_wxyz(qw, qx, qy, qz, wp.vec3(0.0, 0.0, -1.0))

    state_bad = wp.int32(0)
    if not wp.isfinite(qw) or not wp.isfinite(qx) or not wp.isfinite(qy) or not wp.isfinite(qz):
        state_bad = wp.int32(1)
    if not wp.isfinite(joint_q[q_base + wp.int32(2)]):
        state_bad = wp.int32(1)

    value = wp.float32(0.0)
    if col < wp.int32(3):
        value = wp.float32(0.25) * _clip_finite(ang[col], wp.float32(-40.0), wp.float32(40.0))
    elif col < wp.int32(6):
        value = _clip_finite(gravity_b[col - wp.int32(3)], wp.float32(-1.0), wp.float32(1.0))
    elif col < wp.int32(9):
        value = command[world, col - wp.int32(6)]
    elif col < wp.int32(38):
        j = col - wp.int32(9)
        value = _clip_finite(
            joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j], wp.float32(-10.0), wp.float32(10.0)
        )
    elif col < wp.int32(67):
        j = col - wp.int32(38)
        value = wp.float32(0.05) * _clip_finite(
            joint_qd[qd_base + wp.int32(6) + j], wp.float32(-200.0), wp.float32(200.0)
        )
    elif col < wp.int32(96):
        value = current_actions[world, col - wp.int32(67)]
    elif col == wp.int32(96):
        phase_step = episode_steps[world] % phase_period
        phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
        value = wp.sin(phase)
    else:
        phase_step = episode_steps[world] % phase_period
        phase = wp.float32(6.283185307179586) * wp.float32(phase_step) / wp.float32(phase_period)
        value = wp.cos(phase)
    obs[world, col] = _clip_finite(value, wp.float32(-100.0), wp.float32(100.0))

    if col == 0:
        lin_b_x = _clip_finite(lin_b[0], wp.float32(-50.0), wp.float32(50.0))
        lin_b_y = _clip_finite(lin_b[1], wp.float32(-50.0), wp.float32(50.0))
        lin_b_z = _clip_finite(lin_b[2], wp.float32(-50.0), wp.float32(50.0))
        ang_x = _clip_finite(ang[0], wp.float32(-50.0), wp.float32(50.0))
        ang_y = _clip_finite(ang[1], wp.float32(-50.0), wp.float32(50.0))
        ang_z = _clip_finite(ang[2], wp.float32(-50.0), wp.float32(50.0))
        gravity_x = _clip_finite(gravity_b[0], wp.float32(-1.0), wp.float32(1.0))
        gravity_y = _clip_finite(gravity_b[1], wp.float32(-1.0), wp.float32(1.0))
        gravity_z = _clip_finite(gravity_b[2], wp.float32(-1.0), wp.float32(1.0))
        vx_err = _clip_finite(command[world, 0] - lin_b_x, wp.float32(-20.0), wp.float32(20.0))
        vy_err = _clip_finite(command[world, 1] - lin_b_y, wp.float32(-20.0), wp.float32(20.0))
        yaw_err = _clip_finite(command[world, 2] - ang_z, wp.float32(-20.0), wp.float32(20.0))
        track_lin = wp.exp(-(vx_err * vx_err + vy_err * vy_err) / wp.float32(0.25))
        track_ang = wp.exp(-(yaw_err * yaw_err) / wp.float32(0.25))
        lin_vel_z_penalty = lin_b_z * lin_b_z
        ang_vel_xy_penalty = ang_x * ang_x + ang_y * ang_y
        orientation_penalty = gravity_x * gravity_x + gravity_y * gravity_y
        upright = _clip_float(-gravity_z, wp.float32(0.0), wp.float32(1.0))

        action_rate_penalty = wp.float32(0.0)
        power_proxy = wp.float32(0.0)
        for j in range(ACTION_DIM_G1):
            da = current_actions[world, j] - previous_actions[world, j]
            action_rate_penalty = action_rate_penalty + da * da
            q_idx = q_base + wp.int32(7) + j
            qd_idx = qd_base + wp.int32(6) + j
            qd = _clip_finite(joint_qd[qd_idx], wp.float32(-100.0), wp.float32(100.0))
            q = _clip_finite(joint_q[q_idx], wp.float32(-20.0), wp.float32(20.0))
            target = default_joint_pos[j] + current_actions[world, j] * action_scale
            tau_proxy = actuator_ke[j] * (target - q) - actuator_kd[j] * qd
            tau_proxy = _clip_finite(tau_proxy, wp.float32(-10000.0), wp.float32(10000.0))
            power_proxy = power_proxy + wp.abs(tau_proxy * qd)

        fall = wp.float32(0.0)
        if joint_q[q_base + wp.int32(2)] < min_base_height or upright < min_upright_cos:
            fall = wp.float32(1.0)
        if state_bad != wp.int32(0):
            fall = wp.float32(1.0)

        reward = (
            w_track_lin * track_lin
            + w_track_ang * track_ang
            + w_lin_vel_z * lin_vel_z_penalty
            + w_ang_vel_xy * ang_vel_xy_penalty
            + w_orientation * orientation_penalty
            + w_torque * power_proxy
            + w_action_rate * action_rate_penalty
            + w_alive
            + w_termination * fall
        )
        if state_bad != wp.int32(0) or not wp.isfinite(reward):
            reward = w_termination
            track_lin = wp.float32(0.0)
        rewards[world] = reward
        successes[world] = track_lin

        done = wp.float32(0.0)
        if fall > wp.float32(0.5):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0):
            if episode_steps[world] >= max_episode_steps:
                done = wp.float32(1.0)
        dones[world] = done


@wp.kernel
def g1_increment_episode_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel
def g1_reset_done_worlds_kernel(
    seed: wp.int32,
    reset_noise: wp.float32,
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    action_dim: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    if col < coord_stride:
        idx = world * coord_stride + col
        value = default_joint_q[idx]
        if col >= wp.int32(7) and col < wp.int32(7) + action_dim and reset_noise > wp.float32(0.0):
            rng = wp.rand_init(seed, world * coord_stride + col)
            value = value + reset_noise * (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0))
        joint_q[idx] = value
    if col < dof_stride:
        idx = world * dof_stride + col
        joint_qd[idx] = default_joint_qd[idx]
    if col < action_dim:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
    if col == 0:
        episode_steps[world] = wp.int32(0)


@wp.kernel
def g1_sample_commands_kernel(
    seed: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    rng = wp.rand_init(seed, world * wp.int32(3) + col)
    u = wp.randf(rng)
    if col == 0:
        command[world, col] = x_min + (x_max - x_min) * u
    elif col == 1:
        command[world, col] = y_min + (y_max - y_min) * u
    else:
        command[world, col] = yaw_min + (yaw_max - yaw_min) * u


@wp.kernel
def g1_sample_commands_seed_counter_kernel(
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    x_min: wp.float32,
    x_max: wp.float32,
    y_min: wp.float32,
    y_max: wp.float32,
    yaw_min: wp.float32,
    yaw_max: wp.float32,
    command: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    rng = wp.rand_init(seed, world * wp.int32(3) + col)
    u = wp.randf(rng)
    if col == 0:
        command[world, col] = x_min + (x_max - x_min) * u
    elif col == 1:
        command[world, col] = y_min + (y_max - y_min) * u
    else:
        command[world, col] = yaw_min + (yaw_max - yaw_min) * u


@dataclass
class ConfigEnvG1PhoenX:
    """Configuration for :class:`EnvG1PhoenX`.

    Args:
        world_count: Number of vectorized G1 worlds.
        frame_dt: Policy step duration [s].
        sim_substeps: PhoenX physics steps per policy step. The G1
            environment owns this decimation loop and keeps the internal
            SolverPhoenX substep count at one.
        solver_iterations: PhoenX position iterations per substep.
        velocity_iterations: PhoenX velocity iterations per substep.
        action_scale: Position target scale [rad].
        controlled_action_count: Number of leading policy actions applied to joints.
        command: Target body-frame command ``(vx, vy, yaw_rate)`` [m/s, m/s, rad/s].
        max_episode_steps: Episode timeout in policy steps. Use ``0`` to disable.
        reset_noise: Uniform joint-position reset noise half-width [rad].
        min_base_height: Episode ends below this base height [m].
        min_upright_cos: Episode ends below this base-upright cosine threshold.
        phase_period: Gait clock period in policy steps.
        w_track_lin: Linear XY velocity tracking reward scale.
        w_track_ang: Yaw-rate tracking reward scale.
        w_lin_vel_z: Vertical body-velocity penalty scale.
        w_ang_vel_xy: Base roll/pitch angular-velocity penalty scale.
        w_orientation: Projected-gravity tilt penalty scale.
        w_torque: Mechanical power proxy penalty scale.
        w_action_rate: Action-rate penalty scale.
        w_alive: Alive reward per policy step.
        w_termination: Termination reward applied on fall.
        parse_meshes: Import MJCF mesh collision geoms. Disable for fast
            RL runs that use the primitive foot and arm geoms only.
        auto_reset: Reset worlds whose done flag is set after each step.
        rigid_contact_max_per_world: Rigid-contact capacity per vectorized world.
            ``0`` keeps the solver's automatic sizing. The default is
            intended for primitive-collision G1 RL runs and is ignored when
            ``parse_meshes`` is enabled.
        threads_per_world: PhoenX multi-world lane count, or ``"auto"``.
        multi_world_scheduler: PhoenX multi-world scheduler selection.
        prepare_refresh_stride: PhoenX cached-prepare refresh stride, or
            ``"auto"``.
    """

    world_count: int = g1_recipe.WORLD_COUNT
    frame_dt: float = g1_recipe.FRAME_DT
    sim_substeps: int = g1_recipe.SIM_SUBSTEPS
    solver_iterations: int = g1_recipe.SOLVER_ITERATIONS
    velocity_iterations: int = g1_recipe.VELOCITY_ITERATIONS
    action_scale: float = g1_recipe.ACTION_SCALE
    controlled_action_count: int = g1_recipe.CONTROLLED_ACTION_COUNT
    command: tuple[float, float, float] = g1_recipe.COMMAND
    max_episode_steps: int = g1_recipe.MAX_EPISODE_STEPS
    reset_noise: float = g1_recipe.RESET_NOISE
    min_base_height: float = g1_recipe.MIN_BASE_HEIGHT
    min_upright_cos: float = g1_recipe.MIN_UPRIGHT_COS
    phase_period: int = g1_recipe.PHASE_PERIOD
    w_track_lin: float = g1_recipe.W_TRACK_LIN
    w_track_ang: float = g1_recipe.W_TRACK_ANG
    w_lin_vel_z: float = g1_recipe.W_LIN_VEL_Z
    w_ang_vel_xy: float = g1_recipe.W_ANG_VEL_XY
    w_orientation: float = g1_recipe.W_ORIENTATION
    w_torque: float = g1_recipe.W_TORQUE
    w_action_rate: float = g1_recipe.W_ACTION_RATE
    w_alive: float = g1_recipe.W_ALIVE
    w_termination: float = g1_recipe.W_TERMINATION
    parse_meshes: bool = g1_recipe.PARSE_MESHES
    auto_reset: bool = g1_recipe.AUTO_RESET
    rigid_contact_max_per_world: int = g1_recipe.RIGID_CONTACT_MAX_PER_WORLD
    threads_per_world: int | str = g1_recipe.THREADS_PER_WORLD
    multi_world_scheduler: str = g1_recipe.MULTI_WORLD_SCHEDULER
    prepare_refresh_stride: int | str = g1_recipe.PREPARE_REFRESH_STRIDE


class EnvG1PhoenX:
    """Warp-only Unitree G1 locomotion environment backed by SolverPhoenX.

    The observation layout follows nanoG1 v3: scaled base angular velocity,
    projected gravity, command, joint position deltas, scaled joint velocities,
    previous actions, and a sin/cos gait phase clock.

    Args:
        config: Environment and reward configuration.
        device: Warp device.
    """

    obs_dim = OBS_DIM_G1
    action_dim = ACTION_DIM_G1

    def __init__(self, config: ConfigEnvG1PhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvG1PhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if not 0 < int(self.config.controlled_action_count) <= ACTION_DIM_G1:
            raise ValueError("controlled_action_count must be in [1, ACTION_DIM_G1]")
        if int(self.config.phase_period) <= 0:
            raise ValueError("phase_period must be positive")
        if int(self.config.rigid_contact_max_per_world) < 0:
            raise ValueError("rigid_contact_max_per_world must be non-negative")

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        if self.coord_stride != 7 + ACTION_DIM_G1 or self.dof_stride != 6 + ACTION_DIM_G1:
            raise RuntimeError(
                f"Expected nanoG1 dimensions coord={7 + ACTION_DIM_G1}, dof={6 + ACTION_DIM_G1}; "
                f"got coord={self.coord_stride}, dof={self.dof_stride}"
            )

        self.solver = self._make_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.default_joint_pos = wp.array(
            np.asarray(_DEFAULT_JOINT_POS_G1, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.ctrl_lower = wp.array(np.asarray(_CTRL_LO_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.ctrl_upper = wp.array(np.asarray(_CTRL_HI_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_ke = wp.array(np.asarray(_UNITREE_KP_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.actuator_kd = wp.array(np.asarray(_UNITREE_KD_G1, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_seed = 0
        self.sim_time = 0.0

        self.reset()

    def _build_model(self):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75
        articulation_builder.default_shape_cfg.gap = 0.0

        asset_path = newton.utils.download_asset("unitree_g1")
        articulation_builder.add_mjcf(
            str(asset_path / "mjcf" / "g1_29dof.xml"),
            floating=None,
            enable_self_collisions=False,
            ignore_names=("floor", "ground"),
            parse_visuals=False,
            parse_meshes=bool(self.config.parse_meshes),
            ignore_inertial_definitions=False,
        )
        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.785]
        articulation_builder.joint_q[3:7] = [1.0, 0.0, 0.0, 0.0]
        articulation_builder.joint_q[7 : 7 + ACTION_DIM_G1] = list(_DEFAULT_JOINT_POS_G1)
        for i in range(ACTION_DIM_G1):
            dof = i + 6
            articulation_builder.joint_target_ke[dof] = _UNITREE_KP_G1[i]
            articulation_builder.joint_target_kd[dof] = _UNITREE_KD_G1[i]
            articulation_builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.default_shape_cfg.gap = 0.0
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        contact_cap_per_world = int(self.config.rigid_contact_max_per_world)
        if contact_cap_per_world > 0 and not bool(self.config.parse_meshes):
            from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING  # noqa: PLC0415

            model._collision_pipeline = newton.CollisionPipeline(
                model,
                contact_matching=PHOENX_CONTACT_MATCHING,
                rigid_contact_max=max(1, self.world_count * contact_cap_per_world),
            )
        return model

    def _make_solver(self):
        # The environment loop already runs sim_substeps solver calls per policy
        # step; nesting SolverPhoenX substeps would square the recipe value.
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
            threads_per_world=self.config.threads_per_world,
            multi_world_scheduler=self.config.multi_world_scheduler,
            prepare_refresh_stride=self.config.prepare_refresh_stride,
            articulation_dvi=False,
            articulation_dvi_replaces_joint_pgs=False,
        )

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Set the same body-frame command for every world [m/s, m/s, rad/s]."""

        cmd = (float(command[0]), float(command[1]), float(command[2]))
        self.config.command = cmd
        command_np = np.tile(np.asarray(cmd, dtype=np.float32), (self.world_count, 1))
        self.command.assign(command_np)

    def set_commands(self, commands: np.ndarray) -> None:
        """Set per-world body-frame commands [m/s, m/s, rad/s]."""

        cmds = np.asarray(commands, dtype=np.float32)
        if cmds.shape != (self.world_count, 3):
            raise ValueError(f"Expected commands with shape {(self.world_count, 3)}, got {cmds.shape}")
        self.config.command = (float(cmds[0, 0]), float(cmds[0, 1]), float(cmds[0, 2]))
        self.command.assign(cmds)

    def randomize_commands(
        self,
        *,
        seed: int,
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
    ) -> None:
        """Sample per-world body-frame commands on the device [m/s, m/s, rad/s]."""

        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._validate_command_ranges(
            command_x_range, command_y_range, command_yaw_range
        )
        wp.launch(
            g1_sample_commands_kernel,
            dim=(self.world_count, 3),
            inputs=[int(seed), x_min, x_max, y_min, y_max, yaw_min, yaw_max],
            outputs=[self.command],
            device=self.device,
        )

    def randomize_commands_seed_counter(
        self,
        *,
        seed_counter: wp.array[wp.int32],
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
        seed_offset: int = 0,
        advance: int = 1,
    ) -> None:
        """Sample per-world commands using a graph-replay-safe device seed counter."""

        x_min, x_max, y_min, y_max, yaw_min, yaw_max = self._validate_command_ranges(
            command_x_range, command_y_range, command_yaw_range
        )
        wp.launch(
            g1_sample_commands_seed_counter_kernel,
            dim=(self.world_count, 3),
            inputs=[seed_counter, int(seed_offset), x_min, x_max, y_min, y_max, yaw_min, yaw_max],
            outputs=[self.command],
            device=self.device,
        )
        if int(advance) != 0:
            advance_seed_counter(seed_counter, int(advance), device=self.device)

    def _validate_command_ranges(
        self,
        command_x_range: tuple[float, float],
        command_y_range: tuple[float, float],
        command_yaw_range: tuple[float, float],
    ) -> tuple[float, float, float, float, float, float]:
        x_min, x_max = float(command_x_range[0]), float(command_x_range[1])
        y_min, y_max = float(command_y_range[0]), float(command_y_range[1])
        yaw_min, yaw_max = float(command_yaw_range[0]), float(command_yaw_range[1])
        if x_max < x_min or y_max < y_min or yaw_max < yaw_min:
            raise ValueError("command ranges must be ordered")
        self.config.command = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (yaw_min + yaw_max) * 0.5)
        return x_min, x_max, y_min, y_max, yaw_min, yaw_max

    def observe(self) -> wp.array:
        """Update and return the current observation array."""

        wp.launch(
            g1_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.default_joint_pos,
                self.current_actions,
                self.previous_actions,
                self.command,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                self.config.max_episode_steps,
                self.config.phase_period,
                self.config.min_base_height,
                self.config.min_upright_cos,
                self.config.action_scale,
                self.config.w_track_lin,
                self.config.w_track_ang,
                self.config.w_lin_vel_z,
                self.config.w_ang_vel_xy,
                self.config.w_orientation,
                self.config.w_torque,
                self.config.w_action_rate,
                self.config.w_alive,
                self.config.w_termination,
                self.actuator_ke,
                self.actuator_kd,
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        """Reset all worlds and return observations."""

        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_noisy(self, seed: int | None = None) -> wp.array:
        """Reset all worlds with configured joint-position reset noise."""

        self.reset()
        if seed is not None:
            self._reset_seed = int(seed)
        self.dones.assign(np.ones(self.world_count, dtype=np.float32))
        self.reset_done()
        self.dones.zero_()
        return self.observe()

    def reset_done(self) -> None:
        """Reset worlds whose done flag is set."""

        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        wp.launch(
            g1_reset_done_worlds_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
                int(self._reset_seed),
                self.config.reset_noise,
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
                self.action_dim,
            ],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.current_actions,
            ],
            device=self.device,
        )
        self._reset_seed += 1
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Apply actions, advance PhoenX, and return ``(obs, rewards, dones)``."""

        wp.launch(
            g1_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                actions,
                self.default_joint_pos,
                self.ctrl_lower,
                self.ctrl_upper,
                self.config.action_scale,
                int(self.config.controlled_action_count),
                self.dof_stride,
                self.coord_stride,
                int(bool(self.model.use_coord_layout_targets)),
            ],
            outputs=[self.current_actions, self.control.joint_target_q],
            device=self.device,
        )

        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        for _ in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            g1_increment_episode_steps_kernel,
            dim=self.world_count,
            outputs=[self.episode_steps],
            device=self.device,
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.copy(self.previous_actions, self.current_actions)
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones

    def collect_ppo_rollout(self, trainer: TrainerPPO, buffer: BufferRollout, *, seed: int) -> None:
        """Collect one rollout and compute GAE returns for PPO."""

        collect_ppo_rollout(self, trainer, buffer, seed=seed)

    def collect_ppo_rollout_seed_counter(
        self, trainer: TrainerPPO, buffer: BufferRollout, *, seed_counter: wp.array[wp.int32]
    ) -> None:
        """Collect one rollout using a graph-replay-safe device seed counter."""

        collect_ppo_rollout_seed_counter(self, trainer, buffer, seed_counter=seed_counter)
