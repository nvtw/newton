# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils

from .env import collect_ppo_rollout
from .ppo import BufferRollout, MirrorMapPPO, TrainerPPO
from .reward_functions import (
    abs_mechanical_power,
    action_rate_penalty,
    command_progress_2d,
    command_progress_quality_2d,
    fall_indicator,
    gaussian_reward,
    height_tracking_reward,
    joint_position_penalty,
    pd_torque,
    progress_delta,
    projected_gravity_flat_penalty,
    projected_gravity_upright_reward,
    radius_success_2d,
    square,
    tracking_reward_2d,
    vec2_length_sq,
)
from .sac import BufferReplaySAC, TrainerSAC

ACTION_DIM_ANYMAL = 12
COMMAND_DIM_ANYMAL = 4
COMMAND_OBS_OFFSET_ANYMAL = 9
JOINT_POS_OBS_OFFSET_ANYMAL = COMMAND_OBS_OFFSET_ANYMAL + COMMAND_DIM_ANYMAL
JOINT_VEL_OBS_OFFSET_ANYMAL = JOINT_POS_OBS_OFFSET_ANYMAL + ACTION_DIM_ANYMAL
ACTION_OBS_OFFSET_ANYMAL = JOINT_VEL_OBS_OFFSET_ANYMAL + ACTION_DIM_ANYMAL
OBS_DIM_ANYMAL = ACTION_OBS_OFFSET_ANYMAL + ACTION_DIM_ANYMAL
REWARD_MODE_DENSE_COMMAND = 0
REWARD_MODE_SPARSE_TARGET = 1
_REWARD_MODES = {
    "dense_command": REWARD_MODE_DENSE_COMMAND,
    "sparse_target": REWARD_MODE_SPARSE_TARGET,
}


def _normalize_command(command: tuple[float, ...]) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in command)
    if len(values) == 3:
        return (values[0], values[1], values[2], 0.0)
    if len(values) == COMMAND_DIM_ANYMAL:
        return (values[0], values[1], values[2], values[3])
    raise ValueError(f"Expected Anymal command with 3 or {COMMAND_DIM_ANYMAL} values, got {len(values)}")


_LAB_TO_MUJOCO = (0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11)
_INITIAL_JOINT_Q = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}


_ANYMAL_JOINT_MIRROR_SRC = (2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9)
_ANYMAL_JOINT_MIRROR_SIGN = (-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def anymal_mirror_map_ppo() -> MirrorMapPPO:
    """Return the left/right Anymal C PPO mirror map."""

    obs_src = list(range(OBS_DIM_ANYMAL))
    obs_sign = [1.0] * OBS_DIM_ANYMAL

    obs_sign[1] = -1.0
    obs_sign[3] = -1.0
    obs_sign[5] = -1.0
    obs_sign[7] = -1.0
    obs_sign[COMMAND_OBS_OFFSET_ANYMAL + 1] = -1.0
    obs_sign[COMMAND_OBS_OFFSET_ANYMAL + 2] = -1.0

    for base in (JOINT_POS_OBS_OFFSET_ANYMAL, JOINT_VEL_OBS_OFFSET_ANYMAL, ACTION_OBS_OFFSET_ANYMAL):
        for joint in range(ACTION_DIM_ANYMAL):
            obs_src[base + joint] = base + _ANYMAL_JOINT_MIRROR_SRC[joint]
            obs_sign[base + joint] = _ANYMAL_JOINT_MIRROR_SIGN[joint]

    return MirrorMapPPO(
        obs_src=tuple(obs_src),
        obs_sign=tuple(obs_sign),
        action_src=_ANYMAL_JOINT_MIRROR_SRC,
        action_sign=_ANYMAL_JOINT_MIRROR_SIGN,
    )


def _reward_mode_code(reward_mode: str) -> int:
    try:
        return _REWARD_MODES[str(reward_mode)]
    except KeyError as exc:
        modes = ", ".join(sorted(_REWARD_MODES))
        raise ValueError(f"Unknown Anymal reward mode {reward_mode!r}; expected one of: {modes}") from exc


@wp.func
def _clip_float(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _quat_rotate_inverse_xyzw(qx: wp.float32, qy: wp.float32, qz: wp.float32, qw: wp.float32, v: wp.vec3) -> wp.vec3:
    q = wp.vec3(qx, qy, qz)
    a = v * (wp.float32(2.0) * qw * qw - wp.float32(1.0))
    b = wp.cross(q, v) * qw * wp.float32(2.0)
    c = q * (wp.dot(q, v) * wp.float32(2.0))
    return a - b + c


@wp.kernel
def anymal_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    lab_to_mujoco: wp.array[wp.int32],
    action_scale: wp.float32,
    dof_stride: wp.int32,
    coord_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    joint_target_q: wp.array[wp.float32],
):
    world, lab_joint = wp.tid()
    model_joint = lab_to_mujoco[lab_joint]
    target = default_joint_pos[model_joint] + action_scale * actions[world, lab_joint]
    if target_uses_coord_layout != 0:
        joint_target_q[world * coord_stride + wp.int32(7) + model_joint] = target
    else:
        joint_target_q[world * dof_stride + wp.int32(6) + model_joint] = target


@wp.kernel
def anymal_apply_random_velocity_disturbance_kernel(
    seed: wp.int32,
    episode_steps: wp.array[wp.int32],
    warmup_steps: wp.int32,
    noise_velocity_xy: wp.float32,
    noise_yaw_velocity: wp.float32,
    kick_probability: wp.float32,
    kick_velocity_xy: wp.float32,
    kick_yaw_velocity: wp.float32,
    dof_stride: wp.int32,
    body_stride: wp.int32,
    joint_qd: wp.array[wp.float32],
    body_qd: wp.array[wp.spatial_vector],
):
    world = wp.tid()
    step = episode_steps[world]
    if step < warmup_steps:
        return
    if (
        noise_velocity_xy <= wp.float32(0.0)
        and noise_yaw_velocity <= wp.float32(0.0)
        and (
            kick_probability <= wp.float32(0.0)
            or (kick_velocity_xy <= wp.float32(0.0) and kick_yaw_velocity <= wp.float32(0.0))
        )
    ):
        return

    rng = wp.rand_init(seed, world * wp.int32(747796405) + step * wp.int32(289133645))
    dv = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    dyaw = wp.float32(0.0)

    if noise_velocity_xy > wp.float32(0.0):
        noise_angle = wp.float32(2.0) * wp.pi * wp.randf(rng)
        noise_radius = noise_velocity_xy * wp.sqrt(wp.randf(rng))
        dv = dv + wp.vec3(noise_radius * wp.cos(noise_angle), noise_radius * wp.sin(noise_angle), wp.float32(0.0))
    if noise_yaw_velocity > wp.float32(0.0):
        dyaw = dyaw + (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0)) * noise_yaw_velocity

    kick_p = _clip_float(kick_probability, wp.float32(0.0), wp.float32(1.0))
    if wp.randf(rng) < kick_p:
        if kick_velocity_xy > wp.float32(0.0):
            kick_angle = wp.float32(2.0) * wp.pi * wp.randf(rng)
            kick_radius = kick_velocity_xy * wp.sqrt(wp.randf(rng))
            dv = dv + wp.vec3(kick_radius * wp.cos(kick_angle), kick_radius * wp.sin(kick_angle), wp.float32(0.0))
        if kick_yaw_velocity > wp.float32(0.0):
            dyaw = dyaw + (wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0)) * kick_yaw_velocity

    qd_base = world * dof_stride
    joint_qd[qd_base] = joint_qd[qd_base] + dv[0]
    joint_qd[qd_base + wp.int32(1)] = joint_qd[qd_base + wp.int32(1)] + dv[1]
    joint_qd[qd_base + wp.int32(5)] = joint_qd[qd_base + wp.int32(5)] + dyaw

    body = world * body_stride
    qd = body_qd[body]
    lin = wp.spatial_top(qd) + dv
    ang = wp.spatial_bottom(qd) + wp.vec3(wp.float32(0.0), wp.float32(0.0), dyaw)
    body_qd[body] = wp.spatial_vector(lin, ang)


@wp.kernel
def anymal_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    target_position: wp.array2d[wp.float32],
    previous_target_distance: wp.array[wp.float32],
    lab_to_mujoco: wp.array[wp.int32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    reward_mode: wp.int32,
    target_base_height: wp.float32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    lin_vel_reward_scale: wp.float32,
    yaw_rate_reward_scale: wp.float32,
    lin_vel_tracking_sigma: wp.float32,
    yaw_rate_tracking_sigma: wp.float32,
    base_height_reward_scale: wp.float32,
    base_height_tracking_sigma: wp.float32,
    z_vel_reward_scale: wp.float32,
    ang_vel_reward_scale: wp.float32,
    action_rate_reward_scale: wp.float32,
    joint_speed_reward_scale: wp.float32,
    flat_orientation_reward_scale: wp.float32,
    forward_progress_reward_scale: wp.float32,
    sparse_success_reward_scale: wp.float32,
    target_progress_reward_scale: wp.float32,
    fall_reward_scale: wp.float32,
    energy_reward_scale: wp.float32,
    hip_abduction_reward_scale: wp.float32,
    target_radius: wp.float32,
    action_scale: wp.float32,
    actuator_ke: wp.float32,
    actuator_kd: wp.float32,
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride

    quat_x = joint_q[q_base + wp.int32(3)]
    quat_y = joint_q[q_base + wp.int32(4)]
    quat_z = joint_q[q_base + wp.int32(5)]
    quat_w = joint_q[q_base + wp.int32(6)]
    lin_w = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + wp.int32(1)], joint_qd[qd_base + wp.int32(2)])
    ang_w = wp.vec3(
        joint_qd[qd_base + wp.int32(3)],
        joint_qd[qd_base + wp.int32(4)],
        joint_qd[qd_base + wp.int32(5)],
    )
    lin_b = _quat_rotate_inverse_xyzw(quat_x, quat_y, quat_z, quat_w, lin_w)
    ang_b = _quat_rotate_inverse_xyzw(quat_x, quat_y, quat_z, quat_w, ang_w)
    gravity_b = _quat_rotate_inverse_xyzw(quat_x, quat_y, quat_z, quat_w, wp.vec3(0.0, 0.0, -1.0))
    target_delta_w = wp.vec3(
        target_position[world, 0] - joint_q[q_base],
        target_position[world, 1] - joint_q[q_base + wp.int32(1)],
        0.0,
    )
    target_delta_b = _quat_rotate_inverse_xyzw(quat_x, quat_y, quat_z, quat_w, target_delta_w)

    value = wp.float32(0.0)
    if col < wp.int32(3):
        value = lin_b[col]
    elif col < wp.int32(6):
        value = ang_b[col - wp.int32(3)]
    elif col < wp.int32(9):
        value = gravity_b[col - wp.int32(6)]
    elif col < wp.int32(13):
        command_col = col - wp.int32(9)
        value = command[world, command_col]
        if reward_mode == REWARD_MODE_SPARSE_TARGET and command_col < wp.int32(3):
            value = target_delta_b[command_col]
            if command_col == wp.int32(2):
                value = command[world, 2]
    elif col < wp.int32(25):
        j = col - wp.int32(13)
        model_joint = lab_to_mujoco[j]
        value = joint_q[q_base + wp.int32(7) + model_joint] - default_joint_pos[model_joint]
    elif col < wp.int32(37):
        j = col - wp.int32(25)
        model_joint = lab_to_mujoco[j]
        value = joint_qd[qd_base + wp.int32(6) + model_joint]
    else:
        value = current_actions[world, col - wp.int32(37)]
    obs[world, col] = value

    if col == 0:
        yaw_err = ang_b[2] - command[world, 2]
        vel_reward = tracking_reward_2d(
            lin_b[0], lin_b[1], command[world, 0], command[world, 1], lin_vel_tracking_sigma
        )
        yaw_reward = gaussian_reward(yaw_err, yaw_rate_tracking_sigma)
        height_reward = height_tracking_reward(
            joint_q[q_base + wp.int32(2)], target_base_height + command[world, 3], base_height_tracking_sigma
        )
        z_vel_penalty = square(lin_b[2])
        ang_xy_penalty = vec2_length_sq(ang_b[0], ang_b[1])
        flat_orientation_penalty = projected_gravity_flat_penalty(gravity_b)
        upright = projected_gravity_upright_reward(gravity_b)

        action_rate_cost = wp.float32(0.0)
        joint_speed_penalty = wp.float32(0.0)
        hip_abduction_penalty = wp.float32(0.0)
        power_proxy = wp.float32(0.0)
        for j in range(ACTION_DIM_ANYMAL):
            action_rate_cost = action_rate_cost + action_rate_penalty(
                current_actions[world, j], previous_actions[world, j]
            )
            model_joint = lab_to_mujoco[j]
            q_idx = q_base + wp.int32(7) + model_joint
            qd_idx = qd_base + wp.int32(6) + model_joint
            q = joint_q[q_idx]
            qd = joint_qd[qd_idx]
            joint_speed_penalty = joint_speed_penalty + square(qd)
            if j < 4:
                hip_abduction_penalty = hip_abduction_penalty + joint_position_penalty(q, wp.float32(0.0))
            target = default_joint_pos[model_joint] + current_actions[world, j] * action_scale
            tau_proxy = pd_torque(target, q, qd, actuator_ke, actuator_kd)
            power_proxy = power_proxy + abs_mechanical_power(tau_proxy, qd)

        forward_progress = command_progress_2d(lin_b[0], lin_b[1], command[world, 0], command[world, 1])
        speed_quality = command_progress_quality_2d(forward_progress, command[world, 0], command[world, 1])
        target_dist_sq = vec2_length_sq(target_delta_w[0], target_delta_w[1])
        target_dist = wp.sqrt(target_dist_sq)
        target_progress = progress_delta(previous_target_distance[world], target_dist)
        previous_target_distance[world] = target_dist
        success = radius_success_2d(target_delta_w[0], target_delta_w[1], target_radius)
        fall = fall_indicator(joint_q[q_base + wp.int32(2)], min_base_height, upright, min_upright_cos)

        dense_reward = (
            lin_vel_reward_scale * vel_reward
            + yaw_rate_reward_scale * yaw_reward
            + base_height_reward_scale * height_reward
            + forward_progress_reward_scale * forward_progress
            + z_vel_reward_scale * z_vel_penalty
            + ang_vel_reward_scale * ang_xy_penalty
            + action_rate_reward_scale * action_rate_cost
            + joint_speed_reward_scale * joint_speed_penalty
            + flat_orientation_reward_scale * flat_orientation_penalty
            + hip_abduction_reward_scale * hip_abduction_penalty
            + fall_reward_scale * fall
            + energy_reward_scale * power_proxy
        )
        sparse_reward = (
            sparse_success_reward_scale * success
            + base_height_reward_scale * height_reward
            + target_progress_reward_scale * target_progress
            + hip_abduction_reward_scale * hip_abduction_penalty
            + fall_reward_scale * fall
            + energy_reward_scale * power_proxy
        )
        success_metric = success
        reward = dense_reward
        if reward_mode == REWARD_MODE_SPARSE_TARGET:
            reward = sparse_reward
        else:
            success_metric = vel_reward * yaw_reward * height_reward * upright * speed_quality
        rewards[world] = reward
        successes[world] = success_metric

        done = wp.float32(0.0)
        if fall > wp.float32(0.5):
            done = wp.float32(1.0)
        if reward_mode == REWARD_MODE_SPARSE_TARGET and success > wp.float32(0.5):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0):
            if episode_steps[world] >= max_episode_steps:
                done = wp.float32(1.0)
        dones[world] = done


@wp.kernel
def anymal_increment_episode_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel
def anymal_reset_done_worlds_kernel(
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
        joint_q[idx] = default_joint_q[idx]
    if col < dof_stride:
        idx = world * dof_stride + col
        joint_qd[idx] = default_joint_qd[idx]
    if col < action_dim:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
    if col == 0:
        episode_steps[world] = wp.int32(0)


@wp.kernel
def anymal_uniform_actions_kernel(seed: wp.int32, actions: wp.array2d[wp.float32]):
    world, action = wp.tid()
    rng = wp.rand_init(seed, world * actions.shape[1] + action)
    actions[world, action] = wp.float32(2.0) * wp.randf(rng) - wp.float32(1.0)


@dataclass
class ConfigEnvAnymalPhoenX:
    """Configuration for :class:`EnvAnymalPhoenX`.

    Args:
        world_count: Number of vectorized Anymal worlds.
        frame_dt: Policy step duration [s].
        sim_substeps: PhoenX substeps per policy step.
        solver_iterations: PhoenX position iterations per substep.
        velocity_iterations: PhoenX velocity iterations per substep.
        action_scale: Position target scale [rad].
        reward_mode: Reward mode, either ``"sparse_target"`` or ``"dense_command"``.
        command: Target body-frame command ``(vx, vy, yaw_rate, base_height_offset)`` [m/s, m/s, rad/s, m].
            Three-value velocity commands are accepted and use a zero height offset.
        target_position: Sparse target XY position in each world [m].
        target_radius: Sparse target success radius [m].
        target_base_height: Nominal base height [m].
        min_base_height: Episode ends below this base height [m].
        min_upright_cos: Episode ends below this base-upright cosine threshold.
        max_episode_steps: Episode timeout in policy steps. Use ``0`` to disable.
        lin_vel_reward_scale: Linear XY velocity tracking reward scale.
        yaw_rate_reward_scale: Yaw-rate tracking reward scale.
        lin_vel_tracking_sigma: Linear velocity tracking Gaussian sigma [m/s].
        yaw_rate_tracking_sigma: Yaw-rate tracking Gaussian sigma [rad/s].
        base_height_reward_scale: Base-height tracking reward scale.
        base_height_tracking_sigma: Base-height tracking Gaussian sigma [m].
        z_vel_reward_scale: Vertical velocity penalty scale.
        ang_vel_reward_scale: Roll/pitch angular velocity penalty scale.
        action_rate_reward_scale: Action-rate penalty scale.
        joint_speed_reward_scale: Joint velocity penalty scale.
        flat_orientation_reward_scale: Flat-orientation penalty scale.
        forward_progress_reward_scale: Command-aligned horizontal velocity projection shaping scale.
        sparse_success_reward_scale: Sparse target success reward scale.
        target_progress_reward_scale: Target-distance reduction reward scale.
        fall_reward_scale: Fall penalty scale.
        energy_reward_scale: Mechanical power proxy penalty scale.
        hip_abduction_reward_scale: HAA joint position penalty scale that keeps lateral hip joints near zero.
        actuator_ke: Position actuator stiffness used by the model and power proxy.
        actuator_kd: Position actuator damping used by the model and power proxy.
        disturbance_warmup_steps: Policy steps before disturbances may start.
        disturbance_noise_velocity_xy: Maximum per-step horizontal root velocity jitter [m/s].
        disturbance_noise_yaw_velocity: Maximum per-step yaw-rate jitter [rad/s].
        disturbance_kick_probability: Per-policy-step probability of a larger root-velocity kick.
        disturbance_kick_velocity_xy: Maximum horizontal root velocity kick [m/s].
        disturbance_kick_yaw_velocity: Maximum yaw-rate kick [rad/s].
        disturbance_seed: Base RNG seed for stochastic disturbances.
        auto_reset: Reset worlds whose done flag is set after each step.
    """

    world_count: int = 1024
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 4
    solver_iterations: int = 8
    velocity_iterations: int = 1
    action_scale: float = 0.5
    reward_mode: str = "sparse_target"
    command: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    target_position: tuple[float, float] = (0.0, 0.45)
    target_radius: float = 0.4
    target_base_height: float = 0.62
    min_base_height: float = 0.25
    min_upright_cos: float = 0.3
    max_episode_steps: int = 96
    lin_vel_reward_scale: float = 1.0
    yaw_rate_reward_scale: float = 0.5
    lin_vel_tracking_sigma: float = 0.5
    yaw_rate_tracking_sigma: float = 0.5
    base_height_reward_scale: float = 0.0
    base_height_tracking_sigma: float = 0.06
    z_vel_reward_scale: float = -2.0
    ang_vel_reward_scale: float = -0.05
    action_rate_reward_scale: float = -0.01
    joint_speed_reward_scale: float = 0.0
    flat_orientation_reward_scale: float = -5.0
    forward_progress_reward_scale: float = 0.0
    sparse_success_reward_scale: float = 1.0
    target_progress_reward_scale: float = 0.8
    fall_reward_scale: float = -0.25
    energy_reward_scale: float = -1.0e-5
    hip_abduction_reward_scale: float = 0.0
    actuator_ke: float = 150.0
    actuator_kd: float = 5.0
    disturbance_warmup_steps: int = 0
    disturbance_noise_velocity_xy: float = 0.0
    disturbance_noise_yaw_velocity: float = 0.0
    disturbance_kick_probability: float = 0.0
    disturbance_kick_velocity_xy: float = 0.0
    disturbance_kick_yaw_velocity: float = 0.0
    disturbance_seed: int = 0
    auto_reset: bool = True


class EnvAnymalPhoenX:
    """Warp-only Anymal C locomotion environment backed by SolverPhoenX.

    The observation layout matches the Isaac/PhysX policy playback example:
    body linear velocity, body angular velocity, projected gravity, command
    or target direction, joint position deltas, joint velocities, and actions.

    Args:
        config: Environment and reward configuration.
        device: Warp device.
    """

    obs_dim = OBS_DIM_ANYMAL
    action_dim = ACTION_DIM_ANYMAL

    def __init__(self, config: ConfigEnvAnymalPhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvAnymalPhoenX()
        self.reward_mode = _reward_mode_code(self.config.reward_mode)
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        self.body_stride = int(self.model.body_count) // self.world_count
        self.solver = self._make_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.lab_to_mujoco = wp.array(np.array(_LAB_TO_MUJOCO, dtype=np.int32), dtype=wp.int32, device=self.device)
        default_q_np = self.model.joint_q.numpy()[: self.coord_stride]
        self.default_joint_pos = wp.array(default_q_np[7 : 7 + ACTION_DIM_ANYMAL], dtype=wp.float32, device=self.device)
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.config.command = _normalize_command(self.config.command)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        target_np = np.tile(np.asarray(self.config.target_position, dtype=np.float32), (self.world_count, 1))
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.target_position = wp.array(target_np, dtype=wp.float32, device=self.device)
        self.previous_target_distance = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.sim_time = 0.0

        self.reset()

    def _build_model(self):
        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        articulation_builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.config.target_base_height),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        for name, value in _INITIAL_JOINT_Q.items():
            idx = next(
                (i for i, label in enumerate(articulation_builder.joint_label) if label.endswith(f"/{name}")), None
            )
            if idx is None:
                raise ValueError(f"Joint {name!r} not found in Anymal URDF")
            articulation_builder.joint_q[idx + 6] = value
        for i in range(len(articulation_builder.joint_target_ke)):
            articulation_builder.joint_target_ke[i] = float(self.config.actuator_ke)
            articulation_builder.joint_target_kd[i] = float(self.config.actuator_kd)
            articulation_builder.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def _make_solver(self):
        return newton.solvers.SolverPhoenX(
            self.model,
            substeps=int(self.config.sim_substeps),
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
        )

    def set_command(self, command: tuple[float, ...]) -> None:
        """Set the same body-frame command for every world [m/s, m/s, rad/s, m]."""

        cmd = _normalize_command(command)
        self.config.command = cmd
        command_np = np.tile(np.asarray(cmd, dtype=np.float32), (self.world_count, 1))
        self.command.assign(command_np)

    def set_commands(self, commands: np.ndarray) -> None:
        """Set per-world body-frame commands [m/s, m/s, rad/s, m]."""

        cmds = np.asarray(commands, dtype=np.float32)
        if cmds.shape == (self.world_count, 3):
            cmds = np.concatenate((cmds, np.zeros((self.world_count, 1), dtype=np.float32)), axis=1)
        if cmds.shape != (self.world_count, COMMAND_DIM_ANYMAL):
            raise ValueError(f"Expected commands with shape {(self.world_count, COMMAND_DIM_ANYMAL)}, got {cmds.shape}")
        self.config.command = tuple(float(v) for v in cmds[0])
        self.command.assign(cmds)

    def set_target_position(self, target_position: tuple[float, float]) -> None:
        """Set the same sparse target XY position for every world [m]."""

        target = (float(target_position[0]), float(target_position[1]))
        self.config.target_position = target
        target_np = np.tile(np.asarray(target, dtype=np.float32), (self.world_count, 1))
        self.target_position.assign(target_np)

    def set_target_positions(self, target_positions: np.ndarray) -> None:
        """Set per-world sparse target XY positions [m]."""

        targets = np.asarray(target_positions, dtype=np.float32)
        if targets.shape != (self.world_count, 2):
            raise ValueError(f"Expected target positions with shape {(self.world_count, 2)}, got {targets.shape}")
        self.config.target_position = (float(targets[0, 0]), float(targets[0, 1]))
        self.target_position.assign(targets)

    def observe(self) -> wp.array:
        """Update and return the current observation array."""

        wp.launch(
            anymal_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.default_joint_pos,
                self.current_actions,
                self.previous_actions,
                self.command,
                self.target_position,
                self.previous_target_distance,
                self.lab_to_mujoco,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                self.config.max_episode_steps,
                self.reward_mode,
                self.config.target_base_height,
                self.config.min_base_height,
                self.config.min_upright_cos,
                self.config.lin_vel_reward_scale,
                self.config.yaw_rate_reward_scale,
                self.config.lin_vel_tracking_sigma,
                self.config.yaw_rate_tracking_sigma,
                self.config.base_height_reward_scale,
                self.config.base_height_tracking_sigma,
                self.config.z_vel_reward_scale,
                self.config.ang_vel_reward_scale,
                self.config.action_rate_reward_scale,
                self.config.joint_speed_reward_scale,
                self.config.flat_orientation_reward_scale,
                self.config.forward_progress_reward_scale,
                self.config.sparse_success_reward_scale,
                self.config.target_progress_reward_scale,
                self.config.fall_reward_scale,
                self.config.energy_reward_scale,
                self.config.hip_abduction_reward_scale,
                self.config.target_radius,
                self.config.action_scale,
                self.config.actuator_ke,
                self.config.actuator_kd,
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
        self.previous_target_distance.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        """Reset worlds whose done flag is set."""

        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        wp.launch(
            anymal_reset_done_worlds_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
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
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Apply actions, advance PhoenX, and return ``(obs, rewards, dones)``."""

        wp.copy(self.current_actions, actions)
        wp.launch(
            anymal_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                actions,
                self.default_joint_pos,
                self.lab_to_mujoco,
                self.config.action_scale,
                self.dof_stride,
                self.coord_stride,
                int(bool(self.model.use_coord_layout_targets)),
            ],
            outputs=[self.control.joint_target_q],
            device=self.device,
        )

        disturbances_enabled = (
            self.config.disturbance_noise_velocity_xy > 0.0
            or self.config.disturbance_noise_yaw_velocity > 0.0
            or (
                self.config.disturbance_kick_probability > 0.0
                and (self.config.disturbance_kick_velocity_xy > 0.0 or self.config.disturbance_kick_yaw_velocity > 0.0)
            )
        )
        if disturbances_enabled:
            wp.launch(
                anymal_apply_random_velocity_disturbance_kernel,
                dim=self.world_count,
                inputs=[
                    int(self.config.disturbance_seed),
                    self.episode_steps,
                    int(self.config.disturbance_warmup_steps),
                    float(self.config.disturbance_noise_velocity_xy),
                    float(self.config.disturbance_noise_yaw_velocity),
                    float(self.config.disturbance_kick_probability),
                    float(self.config.disturbance_kick_velocity_xy),
                    float(self.config.disturbance_kick_yaw_velocity),
                    self.dof_stride,
                    self.body_stride,
                ],
                outputs=[self.state_0.joint_qd, self.state_0.body_qd],
                device=self.device,
            )

        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        for _ in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            anymal_increment_episode_steps_kernel,
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

    def collect_sac_transitions(
        self,
        trainer: TrainerSAC,
        replay: BufferReplaySAC,
        *,
        num_steps: int,
        seed: int,
        random_steps: int = 0,
    ) -> None:
        """Collect transitions into a SAC replay buffer."""

        if replay.obs_dim != self.obs_dim or replay.action_dim != self.action_dim:
            raise ValueError("SAC replay dimensions do not match environment")

        obs = self.observe()
        for step in range(int(num_steps)):
            obs_before = wp.empty(self.obs.shape, dtype=wp.float32, device=self.device)
            wp.copy(obs_before, obs)
            if step < int(random_steps):
                actions = wp.empty((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
                wp.launch(
                    anymal_uniform_actions_kernel,
                    dim=actions.shape,
                    inputs=[int(seed) + step],
                    outputs=[actions],
                    device=self.device,
                )
            else:
                actions, _log_probs = trainer.act(obs, seed=int(seed) + step)
            next_obs, rewards, dones = self.step(actions)
            replay.add_batch(obs_before, actions, rewards, dones, next_obs)
            obs = next_obs
