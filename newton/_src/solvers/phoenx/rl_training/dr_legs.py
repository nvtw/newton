# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils

from .reward_functions import gaussian_reward, projected_gravity_flat_penalty, tracking_reward_2d

ACTION_DIM_DR_LEGS = 12
OBS_DIM_DR_LEGS_HOLD = 42
OBS_DIM_DR_LEGS_WALK = 47
TASK_DR_LEGS_HOLD = 0
TASK_DR_LEGS_WALK = 1

_DR_LEGS_ACTUATED_JOINT = (0, 1, 5, 6, 10, 15, 18, 19, 23, 24, 28, 33)


@wp.func
def _clip_dr_legs(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _quat_rotate_inverse_dr_legs(q: wp.quat, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(q[0], q[1], q[2])
    return (
        v * (wp.float32(2.0) * q[3] * q[3] - wp.float32(1.0))
        - wp.cross(qv, v) * (wp.float32(2.0) * q[3])
        + qv * (wp.float32(2.0) * wp.dot(qv, v))
    )


@wp.kernel(enable_backward=False)
def dr_legs_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    actuated_joint: wp.array[wp.int32],
    action_scale: wp.float32,
    joint_stride: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    world, action = wp.tid()
    value = _clip_dr_legs(actions[world, action], wp.float32(-1.0), wp.float32(1.0))
    current_actions[world, action] = value
    joint_target_q[world * joint_stride + actuated_joint[action]] = action_scale * value


@wp.kernel(enable_backward=False)
def dr_legs_publish_joint_state_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_xform_parent: wp.array[wp.transform],
    joint_xform_child: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
):
    joint = wp.tid()
    parent = joint_parent[joint]
    child = joint_child[joint]
    parent_joint = body_q[parent] * joint_xform_parent[joint]
    child_joint = body_q[child] * joint_xform_child[joint]
    parent_rotation = wp.transform_get_rotation(parent_joint)
    child_rotation = wp.transform_get_rotation(child_joint)
    relative_rotation = wp.quat_inverse(parent_rotation) * child_rotation
    dof = joint_qd_start[joint]
    axis = joint_axis[dof]
    twist = wp.quat_twist(axis, relative_rotation)
    angle = wp.acos(_clip_dr_legs(twist[3], wp.float32(-1.0), wp.float32(1.0))) * wp.float32(2.0)
    angle = angle * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    parent_angular = wp.spatial_bottom(body_qd[parent])
    child_angular = wp.spatial_bottom(body_qd[child])
    axis_world = wp.transform_vector(parent_joint, axis)
    joint_q[joint_q_start[joint]] = angle
    joint_qd[dof] = wp.dot(child_angular - parent_angular, axis_world)


@wp.kernel(enable_backward=False)
def dr_legs_observe_reward_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    actuated_joint: wp.array[wp.int32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    previous_previous_actions: wp.array2d[wp.float32],
    command: wp.array2d[wp.float32],
    episode_steps: wp.array[wp.int32],
    body_stride: wp.int32,
    joint_stride: wp.int32,
    obs_dim: wp.int32,
    task: wp.int32,
    max_episode_steps: wp.int32,
    frame_dt: wp.float32,
    action_scale: wp.float32,
    gait_period: wp.float32,
    min_base_height: wp.float32,
    min_upright_cos: wp.float32,
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
):
    world, col = wp.tid()
    pelvis = world * body_stride
    pelvis_transform = body_q[pelvis]
    rotation = wp.transform_get_rotation(pelvis_transform)
    position = wp.transform_get_translation(pelvis_transform)
    velocity = body_qd[pelvis]
    linear_world = wp.spatial_top(velocity)
    angular_world = wp.spatial_bottom(velocity)
    linear_body = _quat_rotate_inverse_dr_legs(rotation, linear_world)
    angular_body = _quat_rotate_inverse_dr_legs(rotation, angular_world)
    gravity_body = _quat_rotate_inverse_dr_legs(rotation, wp.vec3(0.0, 0.0, -1.0))

    value = wp.float32(0.0)
    if col < wp.int32(3):
        value = gravity_body[col]
    elif col < wp.int32(6):
        value = angular_body[col - wp.int32(3)]
    elif col < wp.int32(18):
        action = col - wp.int32(6)
        value = joint_q[world * joint_stride + actuated_joint[action]]
    elif col < wp.int32(30):
        action = col - wp.int32(18)
        value = action_scale * previous_actions[world, action]
    elif col < wp.int32(42):
        action = col - wp.int32(30)
        value = action_scale * current_actions[world, action]
    elif task == TASK_DR_LEGS_WALK and col < wp.int32(45):
        value = command[world, col - wp.int32(42)]
    elif task == TASK_DR_LEGS_WALK and col < obs_dim:
        phase = wp.float32(2.0) * wp.pi * wp.float32(episode_steps[world]) * frame_dt / gait_period
        if col == wp.int32(45):
            value = wp.sin(phase)
        else:
            value = wp.cos(phase)
    obs[world, col] = _clip_dr_legs(value, wp.float32(-100.0), wp.float32(100.0))

    if col == wp.int32(0):
        action_rate = wp.float32(0.0)
        action_acceleration = wp.float32(0.0)
        joint_deviation = wp.float32(0.0)
        torque_proxy = wp.float32(0.0)
        for action in range(ACTION_DIM_DR_LEGS):
            q = joint_q[world * joint_stride + actuated_joint[action]]
            qd = joint_qd[world * joint_stride + actuated_joint[action]]
            target = action_scale * current_actions[world, action]
            torque = wp.float32(5.0) * (target - q) - wp.float32(0.2) * qd
            delta = current_actions[world, action] - previous_actions[world, action]
            delta2 = (
                current_actions[world, action]
                - wp.float32(2.0) * previous_actions[world, action]
                + previous_previous_actions[world, action]
            )
            action_rate = action_rate + delta * delta
            action_acceleration = action_acceleration + delta2 * delta2
            joint_deviation = joint_deviation + wp.abs(q)
            torque_proxy = torque_proxy + torque * torque

        upright = -gravity_body[2]
        flat_penalty = projected_gravity_flat_penalty(gravity_body)
        base_linear_cost = wp.dot(linear_body, linear_body)
        base_angular_cost = wp.dot(angular_body, angular_body)
        reward = (
            wp.float32(5.0)
            - flat_penalty
            - (position[2] - wp.float32(0.265)) * (position[2] - wp.float32(0.265))
            - base_linear_cost
            - wp.float32(0.5) * base_angular_cost
            - wp.float32(1.0e-5) * torque_proxy
            - wp.float32(0.1) * action_rate
            - wp.float32(0.01) * action_acceleration
            - wp.float32(2.0) * joint_deviation
        )
        success = gaussian_reward(position[2] - wp.float32(0.265), wp.float32(0.05)) * _clip_dr_legs(
            upright, wp.float32(0.0), wp.float32(1.0)
        )
        if task == TASK_DR_LEGS_WALK:
            linear_tracking = tracking_reward_2d(
                linear_body[0], linear_body[1], command[world, 0], command[world, 1], wp.sqrt(wp.float32(0.125))
            )
            yaw_tracking = gaussian_reward(angular_body[2] - command[world, 2], wp.sqrt(wp.float32(0.5)))
            orientation_reward = gaussian_reward(flat_penalty, wp.float32(0.2))
            reward = (
                wp.float32(10.0)
                + wp.float32(5.0) * linear_tracking
                + wp.float32(5.0) * orientation_reward
                + wp.float32(2.0) * yaw_tracking
                - wp.float32(0.5) * linear_body[2] * linear_body[2]
                - wp.float32(0.5) * (angular_body[0] * angular_body[0] + angular_body[1] * angular_body[1])
                - wp.float32(0.001) * torque_proxy
                - wp.float32(0.001) * action_rate
                - wp.float32(5.0e-5) * action_acceleration
            )
            success = linear_tracking * yaw_tracking * _clip_dr_legs(upright, wp.float32(0.0), wp.float32(1.0))

        done = wp.float32(0.0)
        if position[2] < min_base_height or upright < min_upright_cos or not wp.isfinite(reward):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0) and episode_steps[world] >= max_episode_steps:
            done = wp.float32(1.0)
        rewards[world] = reward
        dones[world] = done
        successes[world] = success


@wp.kernel(enable_backward=False)
def dr_legs_finish_step_kernel(
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    previous_previous_actions: wp.array2d[wp.float32],
    episode_steps: wp.array[wp.int32],
):
    world, action = wp.tid()
    previous_previous_actions[world, action] = previous_actions[world, action]
    previous_actions[world, action] = current_actions[world, action]
    if action == wp.int32(0):
        episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel(enable_backward=False)
def dr_legs_reset_done_kernel(
    dones: wp.array[wp.float32],
    default_body_q: wp.array[wp.transform],
    default_body_qd: wp.array[wp.spatial_vector],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    body_stride: wp.int32,
    joint_stride: wp.int32,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
    previous_actions: wp.array2d[wp.float32],
    previous_previous_actions: wp.array2d[wp.float32],
    current_actions: wp.array2d[wp.float32],
):
    world, col = wp.tid()
    if dones[world] <= wp.float32(0.5):
        return
    if col < body_stride:
        body = world * body_stride + col
        body_q[body] = default_body_q[body]
        body_qd[body] = default_body_qd[body]
    if col < joint_stride:
        joint = world * joint_stride + col
        joint_q[joint] = default_joint_q[joint]
        joint_qd[joint] = default_joint_qd[joint]
    if col < ACTION_DIM_DR_LEGS:
        previous_actions[world, col] = wp.float32(0.0)
        previous_previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
    if col == wp.int32(0):
        episode_steps[world] = wp.int32(0)


def _filter_nearby_dr_legs_bodies(builder: newton.ModelBuilder, max_hops: int = 2) -> None:
    adjacency = [set() for _ in builder.body_q]
    for parent, child in zip(builder.joint_parent, builder.joint_child, strict=True):
        if parent >= 0 and child >= 0:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
    shapes_by_body: list[list[int]] = [[] for _ in builder.body_q]
    for shape, body in enumerate(builder.shape_body):
        if body >= 0:
            shapes_by_body[body].append(shape)
    for body in range(len(adjacency)):
        visited = {body}
        frontier = {body}
        for _ in range(max_hops):
            frontier = {neighbor for current in frontier for neighbor in adjacency[current]} - visited
            visited.update(frontier)
        for other in visited:
            if other < body:
                continue
            for shape_a in shapes_by_body[body]:
                for shape_b in shapes_by_body[other]:
                    if shape_a != shape_b:
                        builder.add_shape_collision_filter_pair(shape_a, shape_b)


@dataclass
class ConfigEnvDrLegsPhoenX:
    """Configuration for :class:`EnvDrLegsPhoenX`."""

    task: str = "hold"
    world_count: int = 4096
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 20
    collision_refresh_interval: int = 4
    solver_iterations: int = 8
    velocity_iterations: int = 1
    action_scale: float = 0.3
    command: tuple[float, float, float] = (0.3, 0.0, 0.0)
    gait_period: float = 0.8
    max_episode_steps: int = 500
    min_base_height: float = 0.12
    min_upright_cos: float = math.cos(1.0)
    ground_friction: float = 0.8
    auto_reset: bool = True


class EnvDrLegsPhoenX:
    """Warp-only DR Legs closed-loop hold-pose or walking environment."""

    action_dim = ACTION_DIM_DR_LEGS

    def __init__(self, config: ConfigEnvDrLegsPhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvDrLegsPhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.config.task not in ("hold", "walk"):
            raise ValueError("task must be 'hold' or 'walk'")
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if int(self.config.sim_substeps) <= 0:
            raise ValueError("sim_substeps must be positive")
        if int(self.config.collision_refresh_interval) <= 0:
            raise ValueError("collision_refresh_interval must be positive")
        self.task = TASK_DR_LEGS_WALK if self.config.task == "walk" else TASK_DR_LEGS_HOLD
        self.obs_dim = OBS_DIM_DR_LEGS_WALK if self.task == TASK_DR_LEGS_WALK else OBS_DIM_DR_LEGS_HOLD

        self.model = self._build_model()
        self.body_stride = int(self.model.body_count) // self.world_count
        self.joint_stride = int(self.model.joint_coord_count) // self.world_count
        if self.body_stride != 31 or self.joint_stride != 36:
            raise RuntimeError(f"Expected DR Legs strides (31, 36), got ({self.body_stride}, {self.joint_stride})")
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=1,
            solver_iterations=int(self.config.solver_iterations),
            velocity_iterations=int(self.config.velocity_iterations),
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.actuated_joint = wp.array(_DR_LEGS_ACTUATED_JOINT, dtype=wp.int32, device=self.device)
        command_np = np.tile(np.asarray(self.config.command, dtype=np.float32), (self.world_count, 1))
        self.command = wp.array(command_np, dtype=wp.float32, device=self.device)
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_previous_actions = wp.zeros(
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
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = float(self.config.ground_friction)
        asset_path = newton.utils.download_asset("disneyresearch")
        robot.add_usd(
            str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda"),
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        if len(robot.joint_q) != 36 or len(robot.body_q) != 31:
            raise RuntimeError(
                f"Expected DR Legs joint/body counts (36, 31), got ({len(robot.joint_q)}, {len(robot.body_q)})"
            )
        pelvis_z = float(wp.transform_get_translation(robot.body_q[0])[2])
        translation_z = 0.28 - pelvis_z
        for body, transform in enumerate(robot.body_q):
            position = wp.transform_get_translation(transform)
            rotation = wp.transform_get_rotation(transform)
            robot.body_q[body] = wp.transform(position + wp.vec3(0.0, 0.0, translation_z), rotation)
        for dof in range(36):
            robot.joint_target_ke[dof] = 0.0
            robot.joint_target_kd[dof] = 0.0
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
            robot.joint_effort_limit[dof] = 400.0
        for dof in _DR_LEGS_ACTUATED_JOINT:
            robot.joint_target_ke[dof] = 5.0
            robot.joint_target_kd[dof] = 0.2
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            robot.joint_effort_limit[dof] = 3.1
        _filter_nearby_dr_legs_bodies(robot, max_hops=2)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = float(self.config.ground_friction)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device, skip_validation_joints=True)
        model.set_gravity((0.0, 0.0, -9.81))
        return model

    def set_command(self, command: tuple[float, float, float]) -> None:
        """Set the same body-frame walking command in every world [m/s, m/s, rad/s]."""

        values = tuple(float(value) for value in command)
        self.config.command = values
        self.command.assign(np.tile(np.asarray(values, dtype=np.float32), (self.world_count, 1)))

    def observe(self) -> wp.array:
        wp.launch(
            dr_legs_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.actuated_joint,
                self.current_actions,
                self.previous_actions,
                self.previous_previous_actions,
                self.command,
                self.episode_steps,
                self.body_stride,
                self.joint_stride,
                self.obs_dim,
                self.task,
                int(self.config.max_episode_steps),
                float(self.config.frame_dt),
                float(self.config.action_scale),
                float(self.config.gait_period),
                float(self.config.min_base_height),
                float(self.config.min_upright_cos),
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        wp.copy(self.state_0.body_q, self.model.body_q)
        wp.copy(self.state_0.body_qd, self.model.body_qd)
        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.previous_previous_actions.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        wp.launch(
            dr_legs_reset_done_kernel,
            dim=(self.world_count, max(self.body_stride, self.joint_stride, self.action_dim)),
            inputs=[
                self.dones,
                self.model.body_q,
                self.model.body_qd,
                self.model.joint_q,
                self.model.joint_qd,
                self.body_stride,
                self.joint_stride,
            ],
            outputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.episode_steps,
                self.previous_actions,
                self.previous_previous_actions,
                self.current_actions,
            ],
            device=self.device,
        )

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        wp.launch(
            dr_legs_apply_actions_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[actions, self.actuated_joint, float(self.config.action_scale), self.joint_stride],
            outputs=[self.current_actions, self.control.joint_target_q],
            device=self.device,
        )
        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        refresh = int(self.config.collision_refresh_interval)
        for substep in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            if substep % refresh == 0:
                self.model.collide(self.state_0, self.contacts)
            self.solver.reuse_partition = substep % refresh != 0
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            dr_legs_publish_joint_state_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_axis,
                self.model.joint_q_start,
                self.model.joint_qd_start,
            ],
            outputs=[self.state_0.joint_q, self.state_0.joint_qd],
            device=self.device,
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.launch(
            dr_legs_finish_step_kernel,
            dim=(self.world_count, self.action_dim),
            inputs=[self.current_actions],
            outputs=[self.previous_actions, self.previous_previous_actions, self.episode_steps],
            device=self.device,
        )
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones
