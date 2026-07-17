# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples

ACTION_DIM_HUMANOID = 21
OBS_DIM_HUMANOID = 75

_HUMANOID_JOINT_GEARS = (
    67.5,
    67.5,
    45.0,
    67.5,
    67.5,
    67.5,
    45.0,
    135.0,
    45.0,
    90.0,
    22.5,
    22.5,
    45.0,
    135.0,
    45.0,
    90.0,
    22.5,
    22.5,
    67.5,
    67.5,
    45.0,
)
_HUMANOID_JOINT_KP = (
    10.0,
    10.0,
    2.0,
    20.0,
    20.0,
    10.0,
    10.0,
    20.0,
    10.0,
    5.0,
    2.0,
    2.0,
    10.0,
    20.0,
    10.0,
    5.0,
    2.0,
    2.0,
    10.0,
    10.0,
    2.0,
)
_HUMANOID_JOINT_KD = (
    5.0,
    5.0,
    1.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    0.1,
    1.0,
    1.0,
    5.0,
    5.0,
    5.0,
    0.1,
    1.0,
    1.0,
    5.0,
    5.0,
    1.0,
)


@wp.func
def _clip_humanoid(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _quat_rotate_inverse_humanoid(q: wp.quat, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(q[0], q[1], q[2])
    return (
        v * (wp.float32(2.0) * q[3] * q[3] - wp.float32(1.0))
        - wp.cross(qv, v) * (wp.float32(2.0) * q[3])
        + qv * (wp.float32(2.0) * wp.dot(qv, v))
    )


@wp.kernel(enable_backward=False)
def humanoid_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    joint_gears: wp.array[wp.float32],
    action_scale: wp.float32,
    dof_stride: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_f: wp.array[wp.float32],
):
    world, col = wp.tid()
    if col < dof_stride:
        joint_f[world * dof_stride + col] = wp.float32(0.0)
    if col < ACTION_DIM_HUMANOID:
        action = _clip_humanoid(actions[world, col], wp.float32(-1.0), wp.float32(1.0))
        current_actions[world, col] = action
        joint_f[world * dof_stride + wp.int32(6) + col] = action_scale * joint_gears[col] * action


@wp.kernel(enable_backward=False)
def humanoid_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_lower: wp.array[wp.float32],
    joint_upper: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    angular_velocity_scale: wp.float32,
    dof_velocity_scale: wp.float32,
    heading_weight: wp.float32,
    up_weight: wp.float32,
    energy_cost_scale: wp.float32,
    actions_cost_scale: wp.float32,
    alive_reward_scale: wp.float32,
    death_cost: wp.float32,
    termination_height: wp.float32,
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
    linear_body = _quat_rotate_inverse_humanoid(rotation, linear_world)
    angular_body = _quat_rotate_inverse_humanoid(rotation, angular_world)
    euler = wp.quat_to_euler(rotation, 2, 1, 0)
    heading = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))[0]
    up = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))[2]

    value = wp.float32(0.0)
    if col == wp.int32(0):
        value = joint_q[q_base + wp.int32(2)]
    elif col < wp.int32(4):
        value = linear_body[col - wp.int32(1)]
    elif col < wp.int32(7):
        value = angular_velocity_scale * angular_body[col - wp.int32(4)]
    elif col == wp.int32(7):
        value = euler[2]
    elif col == wp.int32(8):
        value = euler[0]
    elif col == wp.int32(9):
        value = -euler[2]
    elif col == wp.int32(10):
        value = up
    elif col == wp.int32(11):
        value = heading
    elif col < wp.int32(33):
        j = col - wp.int32(12)
        q = joint_q[q_base + wp.int32(7) + j]
        lo = joint_lower[j]
        hi = joint_upper[j]
        value = wp.float32(2.0) * (q - lo) / wp.max(hi - lo, wp.float32(1.0e-6)) - wp.float32(1.0)
    elif col < wp.int32(54):
        j = col - wp.int32(33)
        value = dof_velocity_scale * joint_qd[qd_base + wp.int32(6) + j]
    else:
        value = current_actions[world, col - wp.int32(54)]
    obs[world, col] = _clip_humanoid(value, wp.float32(-100.0), wp.float32(100.0))

    if col == wp.int32(0):
        actions_cost = wp.float32(0.0)
        electricity_cost = wp.float32(0.0)
        limit_cost = wp.float32(0.0)
        for j in range(ACTION_DIM_HUMANOID):
            action = current_actions[world, j]
            q = joint_q[q_base + wp.int32(7) + j]
            qd = joint_qd[qd_base + wp.int32(6) + j]
            lo = joint_lower[j]
            hi = joint_upper[j]
            q_scaled = wp.float32(2.0) * (q - lo) / wp.max(hi - lo, wp.float32(1.0e-6)) - wp.float32(1.0)
            actions_cost = actions_cost + action * action
            electricity_cost = electricity_cost + wp.abs(action * qd * dof_velocity_scale)
            if q_scaled > wp.float32(0.98):
                limit_cost = limit_cost + wp.float32(1.0)

        heading_reward = heading_weight
        if heading <= wp.float32(0.8):
            heading_reward = heading_weight * heading / wp.float32(0.8)
        up_reward = wp.float32(0.0)
        if up > wp.float32(0.93):
            up_reward = up_weight
        reward = (
            linear_world[0]
            + alive_reward_scale
            + up_reward
            + heading_reward
            - actions_cost_scale * actions_cost
            - energy_cost_scale * electricity_cost
            - limit_cost
        )
        bad_state = joint_q[q_base + wp.int32(2)] < termination_height or not wp.isfinite(reward)
        done = wp.float32(0.0)
        if bad_state:
            reward = death_cost
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0) and episode_steps[world] >= max_episode_steps:
            done = wp.float32(1.0)
        rewards[world] = reward
        dones[world] = done
        successes[world] = _clip_humanoid(linear_world[0], wp.float32(0.0), wp.float32(1.0))


@wp.kernel(enable_backward=False)
def humanoid_increment_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel(enable_backward=False)
def humanoid_reset_done_kernel(
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    episode_steps: wp.array[wp.int32],
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
    if col < ACTION_DIM_HUMANOID:
        current_actions[world, col] = wp.float32(0.0)
    if col == wp.int32(0):
        episode_steps[world] = wp.int32(0)


@dataclass
class ConfigEnvHumanoidPhoenX:
    """Configuration for :class:`EnvHumanoidPhoenX`.

    Args:
        world_count: Number of vectorized Humanoid worlds.
        frame_dt: Policy step duration [s].
        sim_substeps: Physics steps per policy step.
        solver_iterations: PhoenX position iterations per physics step.
        velocity_iterations: PhoenX velocity iterations per physics step.
        action_scale: Scale applied to normalized geared torque actions.
        max_episode_steps: Episode timeout in policy steps. Use ``0`` to disable.
        heading_weight: Forward-heading reward weight.
        up_weight: Upright reward weight.
        energy_cost_scale: Joint electricity-proxy penalty scale.
        actions_cost_scale: Squared-action penalty scale.
        alive_reward_scale: Per-step survival reward.
        dof_velocity_scale: Joint velocity observation and energy scale [s/rad].
        death_cost: Reward assigned to a fallen world.
        termination_height: Minimum root height [m].
        angular_velocity_scale: Base angular velocity observation scale [s/rad].
        ground_friction: Ground-plane friction coefficient.
        auto_reset: Reset terminal worlds after each policy step.
        articulation_mode: PhoenX articulation mode, ``"reduced"`` or ``"maximal"``.
            Reduced mode represents armature on every multi-axis motor. Maximal
            mode uses zero armature because static body inertia cannot represent
            intermediate D6 rotor axes without virtual rotor bodies.
    """

    world_count: int = 4096
    frame_dt: float = 1.0 / 60.0
    sim_substeps: int = 2
    solver_iterations: int = 8
    velocity_iterations: int = 1
    action_scale: float = 1.0
    max_episode_steps: int = 900
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_velocity_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.8
    angular_velocity_scale: float = 0.25
    ground_friction: float = 1.0
    auto_reset: bool = True
    articulation_mode: str = "reduced"


class EnvHumanoidPhoenX:
    """Vectorized classic Humanoid locomotion environment backed by PhoenX.

    Args:
        config: Humanoid environment configuration.
        device: Warp device. CUDA is required for graph capture.
    """

    obs_dim = OBS_DIM_HUMANOID
    action_dim = ACTION_DIM_HUMANOID

    def __init__(self, config: ConfigEnvHumanoidPhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvHumanoidPhoenX()
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
        if self.coord_stride != 28 or self.dof_stride != 27:
            raise RuntimeError(
                f"Expected Humanoid coordinate/dof strides (28, 27), got ({self.coord_stride}, {self.dof_stride})"
            )
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
        self.joint_gears = wp.array(_HUMANOID_JOINT_GEARS, dtype=wp.float32, device=self.device)
        self.joint_lower = wp.array(
            np.asarray(self.model.joint_limit_lower.numpy()[6:27], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.joint_upper = wp.array(
            np.asarray(self.model.joint_limit_upper.numpy()[6:27], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
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
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.01 if self.config.articulation_mode == "reduced" else 0.0,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = float(self.config.ground_friction)
        robot.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            up_axis="Z",
            parse_meshes=False,
            parse_visuals=False,
            ignore_names=("floor", "ground"),
            enable_self_collisions=True,
            parse_mujoco_options=False,
        )
        if len(robot.joint_q) != 28 or len(robot.joint_qd) != 27:
            raise RuntimeError(
                f"Expected Humanoid coordinate/dof counts (28, 27), got ({len(robot.joint_q)}, {len(robot.joint_qd)})"
            )
        robot.joint_q[:7] = [0.0, 0.0, 1.34, 0.0, 0.0, 0.0, 1.0]
        for action in range(ACTION_DIM_HUMANOID):
            dof = 6 + action
            robot.joint_target_ke[dof] = _HUMANOID_JOINT_KP[action]
            robot.joint_target_kd[dof] = _HUMANOID_JOINT_KD[action]
            robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
            robot.joint_effort_limit[dof] = 150.0

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

    def observe(self) -> wp.array:
        wp.launch(
            humanoid_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.joint_lower,
                self.joint_upper,
                self.current_actions,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                int(self.config.max_episode_steps),
                float(self.config.angular_velocity_scale),
                float(self.config.dof_velocity_scale),
                float(self.config.heading_weight),
                float(self.config.up_weight),
                float(self.config.energy_cost_scale),
                float(self.config.actions_cost_scale),
                float(self.config.alive_reward_scale),
                float(self.config.death_cost),
                float(self.config.termination_height),
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.control.joint_f.zero_()
        self.current_actions.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        wp.launch(
            humanoid_reset_done_kernel,
            dim=(self.world_count, max(self.coord_stride, self.dof_stride, self.action_dim)),
            inputs=[
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
            ],
            outputs=[self.state_0.joint_q, self.state_0.joint_qd, self.episode_steps, self.current_actions],
            device=self.device,
        )
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        wp.launch(
            humanoid_apply_actions_kernel,
            dim=(self.world_count, max(self.dof_stride, self.action_dim)),
            inputs=[actions, self.joint_gears, float(self.config.action_scale), self.dof_stride],
            outputs=[self.current_actions, self.control.joint_f],
            device=self.device,
        )
        sub_dt = float(self.config.frame_dt) / float(self.config.sim_substeps)
        for substep in range(int(self.config.sim_substeps)):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                sub_dt,
                state_is_continuation=substep > 0,
                state_kinematics_valid=True,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            humanoid_increment_steps_kernel, dim=self.world_count, outputs=[self.episode_steps], device=self.device
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones
