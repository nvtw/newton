# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils

from .reward_functions import gaussian_reward, projected_gravity_flat_penalty, tracking_reward_2d

ACTION_DIM_H1 = 19
OBS_DIM_H1 = 12 + 3 * ACTION_DIM_H1

_H1_DEFAULT_Q = (
    0.0,
    0.0,
    -0.28,
    0.79,
    -0.52,
    0.0,
    0.0,
    -0.28,
    0.79,
    -0.52,
    0.0,
    0.28,
    0.0,
    0.0,
    0.52,
    0.28,
    0.0,
    0.0,
    0.52,
)
_H1_KP = (
    150.0,
    150.0,
    200.0,
    200.0,
    20.0,
    150.0,
    150.0,
    200.0,
    200.0,
    20.0,
    200.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
)
_H1_KD = (
    5.0,
    5.0,
    5.0,
    5.0,
    4.0,
    5.0,
    5.0,
    5.0,
    5.0,
    4.0,
    5.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)
_H1_EFFORT_LIMIT = (
    300.0,
    300.0,
    300.0,
    300.0,
    100.0,
    300.0,
    300.0,
    300.0,
    300.0,
    100.0,
    300.0,
    300.0,
    300.0,
    300.0,
    300.0,
    300.0,
    300.0,
    300.0,
    300.0,
)


@wp.func
def _clip_h1(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _quat_rotate_inverse_h1(q: wp.quat, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(q[0], q[1], q[2])
    return (
        v * (wp.float32(2.0) * q[3] * q[3] - wp.float32(1.0))
        - wp.cross(qv, v) * (wp.float32(2.0) * q[3])
        + qv * (wp.float32(2.0) * wp.dot(qv, v))
    )


@wp.kernel(enable_backward=False)
def h1_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    action_scale: wp.float32,
    dof_stride: wp.int32,
    coord_stride: wp.int32,
    target_uses_coord_layout: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    world, action = wp.tid()
    value = _clip_h1(actions[world, action], wp.float32(-1.0), wp.float32(1.0))
    current_actions[world, action] = value
    target = default_joint_pos[action] + action_scale * value
    if target_uses_coord_layout != wp.int32(0):
        joint_target_q[world * coord_stride + wp.int32(7) + action] = target
    else:
        joint_target_q[world * dof_stride + wp.int32(6) + action] = target


@wp.kernel(enable_backward=False)
def h1_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    default_joint_pos: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    previous_joint_velocity: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    frame_dt: wp.float32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    lin_vel_weight: wp.float32,
    yaw_rate_weight: wp.float32,
    alive_weight: wp.float32,
    lin_vel_sigma: wp.float32,
    yaw_rate_sigma: wp.float32,
    lin_vel_z_weight: wp.float32,
    ang_vel_xy_weight: wp.float32,
    flat_orientation_weight: wp.float32,
    action_rate_weight: wp.float32,
    joint_acc_weight: wp.float32,
    hip_deviation_weight: wp.float32,
    arm_deviation_weight: wp.float32,
    torso_deviation_weight: wp.float32,
    termination_weight: wp.float32,
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride
    rotation = wp.quat(
        joint_q[q_base + wp.int32(3)],
        joint_q[q_base + wp.int32(4)],
        joint_q[q_base + wp.int32(5)],
        joint_q[q_base + wp.int32(6)],
    )
    linear_world = wp.vec3(
        joint_qd[qd_base],
        joint_qd[qd_base + wp.int32(1)],
        joint_qd[qd_base + wp.int32(2)],
    )
    angular_world = wp.vec3(
        joint_qd[qd_base + wp.int32(3)],
        joint_qd[qd_base + wp.int32(4)],
        joint_qd[qd_base + wp.int32(5)],
    )
    linear_body = _quat_rotate_inverse_h1(rotation, linear_world)
    angular_body = _quat_rotate_inverse_h1(rotation, angular_world)
    gravity_body = _quat_rotate_inverse_h1(rotation, wp.vec3(0.0, 0.0, -1.0))

    value = wp.float32(0.0)
    if col < wp.int32(3):
        value = linear_body[col]
    elif col < wp.int32(6):
        value = angular_body[col - wp.int32(3)]
    elif col < wp.int32(9):
        value = gravity_body[col - wp.int32(6)]
    elif col < wp.int32(12):
        value = command[world, col - wp.int32(9)]
    elif col < wp.int32(12 + ACTION_DIM_H1):
        j = col - wp.int32(12)
        value = joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j]
    elif col < wp.int32(12 + 2 * ACTION_DIM_H1):
        j = col - wp.int32(12 + ACTION_DIM_H1)
        value = joint_qd[qd_base + wp.int32(6) + j]
    else:
        value = current_actions[world, col - wp.int32(12 + 2 * ACTION_DIM_H1)]
    obs[world, col] = _clip_h1(value, wp.float32(-100.0), wp.float32(100.0))

    if col == wp.int32(0):
        action_rate_cost = wp.float32(0.0)
        joint_acc_cost = wp.float32(0.0)
        hip_deviation = wp.float32(0.0)
        arm_deviation = wp.float32(0.0)
        torso_deviation = wp.float32(0.0)
        inv_dt = wp.float32(1.0) / wp.max(frame_dt, wp.float32(1.0e-6))
        for j in range(ACTION_DIM_H1):
            action_delta = current_actions[world, j] - previous_actions[world, j]
            action_rate_cost = action_rate_cost + action_delta * action_delta
            qd = joint_qd[qd_base + wp.int32(6) + j]
            qdd = (qd - previous_joint_velocity[world, j]) * inv_dt
            joint_acc_cost = joint_acc_cost + qdd * qdd
            q_delta = joint_q[q_base + wp.int32(7) + j] - default_joint_pos[j]
            if j == 0 or j == 1 or j == 5 or j == 6:
                hip_deviation = hip_deviation + wp.abs(q_delta)
            if j >= 11:
                arm_deviation = arm_deviation + wp.abs(q_delta)
            if j == 10:
                torso_deviation = torso_deviation + wp.abs(q_delta)

        linear_tracking = tracking_reward_2d(
            linear_body[0], linear_body[1], command[world, 0], command[world, 1], lin_vel_sigma
        )
        yaw_tracking = gaussian_reward(angular_world[2] - command[world, 2], yaw_rate_sigma)
        upright = -gravity_body[2]
        fall = joint_q[q_base + wp.int32(2)] < min_base_height or upright < min_upright_cos
        reward = (
            lin_vel_weight * linear_tracking
            + yaw_rate_weight * yaw_tracking
            + alive_weight
            + lin_vel_z_weight * linear_body[2] * linear_body[2]
            + ang_vel_xy_weight * (angular_body[0] * angular_body[0] + angular_body[1] * angular_body[1])
            + flat_orientation_weight * projected_gravity_flat_penalty(gravity_body)
            + action_rate_weight * action_rate_cost
            + joint_acc_weight * joint_acc_cost
            + hip_deviation_weight * hip_deviation
            + arm_deviation_weight * arm_deviation
            + torso_deviation_weight * torso_deviation
        )
        done = wp.float32(0.0)
        if fall or not wp.isfinite(reward):
            reward = reward + termination_weight
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0) and episode_steps[world] >= max_episode_steps:
            done = wp.float32(1.0)
        rewards[world] = reward
        dones[world] = done
        successes[world] = linear_tracking * yaw_tracking * _clip_h1(upright, wp.float32(0.0), wp.float32(1.0))


@wp.kernel(enable_backward=False)
def h1_finish_step_kernel(
    joint_qd: wp.array[wp.float32],
    dof_stride: wp.int32,
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    previous_joint_velocity: wp.array2d[wp.float32],
    episode_steps: wp.array[wp.int32],
):
    world, action = wp.tid()
    previous_actions[world, action] = current_actions[world, action]
    previous_joint_velocity[world, action] = joint_qd[world * dof_stride + wp.int32(6) + action]
    if action == wp.int32(0):
        episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel(enable_backward=False)
def h1_reset_done_kernel(
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_joint_velocity: wp.array2d[wp.float32],
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
    if col < ACTION_DIM_H1:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
        previous_joint_velocity[world, col] = wp.float32(0.0)
    if col == wp.int32(0):
        episode_steps[world] = wp.int32(0)


@dataclass
class ConfigEnvH1PhoenX:
    """Configuration for :class:`EnvH1PhoenX`."""

    world_count: int = 4096
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 4
    solver_iterations: int = 4
    velocity_iterations: int = 1
    action_scale: float = 0.5
    command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    max_episode_steps: int = 1000
    min_base_height: float = 0.55
    min_upright_cos: float = 0.54
    lin_vel_weight: float = 1.0
    yaw_rate_weight: float = 1.0
    alive_weight: float = 0.25
    lin_vel_sigma: float = 0.5
    yaw_rate_sigma: float = 0.5
    lin_vel_z_weight: float = -2.0
    ang_vel_xy_weight: float = -0.05
    flat_orientation_weight: float = -1.0
    action_rate_weight: float = -0.005
    joint_acc_weight: float = -1.25e-7
    hip_deviation_weight: float = -0.2
    arm_deviation_weight: float = -0.2
    torso_deviation_weight: float = -0.1
    termination_weight: float = -200.0
    ground_friction: float = 1.0
    auto_reset: bool = True
    articulation_mode: str = "reduced"


class EnvH1PhoenX:
    """Warp-only Unitree H1 flat-terrain velocity-tracking environment."""

    obs_dim = OBS_DIM_H1
    action_dim = ACTION_DIM_H1

    def __init__(self, config: ConfigEnvH1PhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvH1PhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if int(self.config.sim_substeps) <= 0:
            raise ValueError("sim_substeps must be positive")
        if self.config.articulation_mode not in ("maximal", "reduced"):
            raise ValueError("articulation_mode must be 'maximal' or 'reduced'")

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        if self.coord_stride != 26 or self.dof_stride != 25:
            raise RuntimeError(f"Expected H1 strides (26, 25), got ({self.coord_stride}, {self.dof_stride})")
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
            articulation_mode=self.config.articulation_mode,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.default_joint_pos = wp.array(_H1_DEFAULT_Q, dtype=wp.float32, device=self.device)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_joint_velocity = wp.zeros(
            (self.world_count, self.action_dim), dtype=wp.float32, device=self.device
        )
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
        robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1)
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = float(self.config.ground_friction)
        asset_path = newton.utils.download_asset("unitree_h1")
        robot.add_mjcf(
            str(asset_path / "mjcf" / "h1.xml"),
            up_axis="Z",
            parse_meshes=False,
            parse_visuals=False,
            ignore_names=("floor", "ground"),
            enable_self_collisions=False,
            parse_mujoco_options=False,
        )
        if len(robot.joint_q) != 26 or len(robot.joint_qd) != 25:
            raise RuntimeError(
                f"Expected H1 coordinate/dof counts (26, 25), got ({len(robot.joint_q)}, {len(robot.joint_qd)})"
            )
        robot.joint_q[:7] = [0.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.0]
        robot.joint_q[7:26] = _H1_DEFAULT_Q
        for action in range(ACTION_DIM_H1):
            dof = 6 + action
            robot.joint_target_ke[dof] = _H1_KP[action]
            robot.joint_target_kd[dof] = _H1_KD[action]
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            robot.joint_effort_limit[dof] = _H1_EFFORT_LIMIT[action]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = float(self.config.ground_friction)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Set the same body-frame velocity command in every world [m/s, m/s, rad/s]."""

        values = tuple(float(value) for value in command)
        self.config.command = values
        self.command.assign(np.tile(np.asarray(values, dtype=np.float32), (self.world_count, 1)))

    def observe(self) -> wp.array:
        wp.launch(
            h1_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.default_joint_pos,
                self.current_actions,
                self.previous_actions,
                self.previous_joint_velocity,
                self.command,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                int(self.config.max_episode_steps),
                float(self.config.frame_dt),
                float(self.config.min_base_height),
                float(self.config.min_upright_cos),
                float(self.config.lin_vel_weight),
                float(self.config.yaw_rate_weight),
                float(self.config.alive_weight),
                float(self.config.lin_vel_sigma),
                float(self.config.yaw_rate_sigma),
                float(self.config.lin_vel_z_weight),
                float(self.config.ang_vel_xy_weight),
                float(self.config.flat_orientation_weight),
                float(self.config.action_rate_weight),
                float(self.config.joint_acc_weight),
                float(self.config.hip_deviation_weight),
                float(self.config.arm_deviation_weight),
                float(self.config.torso_deviation_weight),
                float(self.config.termination_weight),
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.previous_joint_velocity.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        wp.launch(
            h1_reset_done_kernel,
            dim=(self.world_count, max(self.coord_stride, self.dof_stride, self.action_dim)),
            inputs=[self.dones, self.model.joint_q, self.model.joint_qd, self.coord_stride, self.dof_stride],
            outputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.current_actions,
                self.previous_joint_velocity,
            ],
            device=self.device,
        )
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        wp.launch(
            h1_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[
                actions,
                self.default_joint_pos,
                float(self.config.action_scale),
                self.dof_stride,
                self.coord_stride,
                int(bool(self.model.use_coord_layout_targets)),
            ],
            outputs=[self.current_actions, self.control.joint_target_q],
            device=self.device,
        )
        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        self.model.collide(self.state_0, self.contacts)
        for substep in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            self.solver.reuse_partition = substep > 0
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.launch(
            h1_finish_step_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[self.state_0.joint_qd, self.dof_stride, self.current_actions],
            outputs=[self.previous_actions, self.previous_joint_velocity, self.episode_steps],
            device=self.device,
        )
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones
