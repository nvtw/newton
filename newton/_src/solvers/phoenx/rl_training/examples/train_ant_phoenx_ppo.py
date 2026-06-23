# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train the classic Ant locomotion task with PhoenX and Warp-only PPO.

This is an experimental validation target for the PhoenX RL stack. Ant is much
simpler than G1 but still exercises vectorized articulated contacts, direct
joint torques, PPO rollout/update code, and no-reset evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.rl as rl

ACTION_DIM_ANT = 8
OBS_DIM_ANT = 34
_DEFAULT_ANT_Q = (0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0)


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


@wp.kernel(enable_backward=False)
def ant_apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    torque_limit: wp.float32,
    dof_stride: wp.int32,
    current_actions: wp.array2d[wp.float32],
    joint_f: wp.array[wp.float32],
):
    world, col = wp.tid()
    if col < dof_stride:
        joint_f[world * dof_stride + col] = wp.float32(0.0)
    if col < ACTION_DIM_ANT:
        action = _clip_float(actions[world, col], wp.float32(-1.0), wp.float32(1.0))
        current_actions[world, col] = action
        joint_f[world * dof_stride + wp.int32(6) + col] = action * torque_limit


@wp.kernel(enable_backward=False)
def ant_observe_reward_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    current_actions: wp.array2d[wp.float32],
    previous_actions: wp.array2d[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    episode_steps: wp.array[wp.int32],
    max_episode_steps: wp.int32,
    min_height: wp.float32,
    max_height: wp.float32,
    min_upright_cos: wp.float32,
    forward_reward_weight: wp.float32,
    healthy_reward: wp.float32,
    ctrl_cost_weight: wp.float32,
    action_rate_cost_weight: wp.float32,
    angular_cost_weight: wp.float32,
    vertical_cost_weight: wp.float32,
    termination_reward: wp.float32,
    target_velocity: wp.float32,
    obs: wp.array2d[wp.float32],
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    forward_velocities: wp.array[wp.float32],
):
    world, col = wp.tid()
    q_base = world * coord_stride
    qd_base = world * dof_stride

    qx = joint_q[q_base + wp.int32(3)]
    qy = joint_q[q_base + wp.int32(4)]
    qz = joint_q[q_base + wp.int32(5)]
    qw = joint_q[q_base + wp.int32(6)]
    lin_w = wp.vec3(joint_qd[qd_base], joint_qd[qd_base + wp.int32(1)], joint_qd[qd_base + wp.int32(2)])
    ang_w = wp.vec3(joint_qd[qd_base + wp.int32(3)], joint_qd[qd_base + wp.int32(4)], joint_qd[qd_base + wp.int32(5)])
    lin_b = _quat_rotate_inverse_xyzw(qx, qy, qz, qw, lin_w)
    ang_b = _quat_rotate_inverse_xyzw(qx, qy, qz, qw, ang_w)
    gravity_b = _quat_rotate_inverse_xyzw(qx, qy, qz, qw, wp.vec3(0.0, -1.0, 0.0))

    value = wp.float32(0.0)
    if col == wp.int32(0):
        value = joint_q[q_base + wp.int32(1)]
    elif col < wp.int32(4):
        value = lin_b[col - wp.int32(1)]
    elif col < wp.int32(7):
        value = ang_b[col - wp.int32(4)]
    elif col < wp.int32(10):
        value = gravity_b[col - wp.int32(7)]
    elif col < wp.int32(18):
        value = joint_q[q_base + wp.int32(7) + col - wp.int32(10)]
    elif col < wp.int32(26):
        value = joint_qd[qd_base + wp.int32(6) + col - wp.int32(18)]
    elif col < wp.int32(34):
        value = previous_actions[world, col - wp.int32(26)]
    obs[world, col] = _clip_float(value, wp.float32(-100.0), wp.float32(100.0))

    if col == wp.int32(0):
        height = joint_q[q_base + wp.int32(1)]
        upright = _clip_float(-gravity_b[1], wp.float32(-1.0), wp.float32(1.0))
        bad_state = wp.int32(0)
        if not wp.isfinite(height):
            bad_state = wp.int32(1)
        if height < min_height or height > max_height or upright < min_upright_cos:
            bad_state = wp.int32(1)

        ctrl_cost = wp.float32(0.0)
        action_rate_cost = wp.float32(0.0)
        for j in range(ACTION_DIM_ANT):
            action = current_actions[world, j]
            delta = action - previous_actions[world, j]
            ctrl_cost = ctrl_cost + action * action
            action_rate_cost = action_rate_cost + delta * delta

        vertical_cost = lin_w[1] * lin_w[1]
        angular_cost = ang_b[0] * ang_b[0] + ang_b[2] * ang_b[2]
        forward_vel = lin_w[0]
        reward = (
            forward_reward_weight * forward_vel
            + healthy_reward
            - ctrl_cost_weight * ctrl_cost
            - action_rate_cost_weight * action_rate_cost
            - angular_cost_weight * angular_cost
            - vertical_cost_weight * vertical_cost
        )
        if bad_state != wp.int32(0) or not wp.isfinite(reward):
            reward = termination_reward

        done = wp.float32(0.0)
        if bad_state != wp.int32(0):
            done = wp.float32(1.0)
        if max_episode_steps > wp.int32(0) and episode_steps[world] >= max_episode_steps:
            done = wp.float32(1.0)
        success = wp.float32(0.0)
        if bad_state == wp.int32(0) and forward_vel >= target_velocity:
            success = wp.float32(1.0)

        rewards[world] = reward
        dones[world] = done
        successes[world] = success
        forward_velocities[world] = forward_vel


@wp.kernel(enable_backward=False)
def ant_increment_episode_steps_kernel(episode_steps: wp.array[wp.int32]):
    world = wp.tid()
    episode_steps[world] = episode_steps[world] + wp.int32(1)


@wp.kernel(enable_backward=False)
def ant_reset_done_worlds_kernel(
    dones: wp.array[wp.float32],
    default_joint_q: wp.array[wp.float32],
    default_joint_qd: wp.array[wp.float32],
    coord_stride: wp.int32,
    dof_stride: wp.int32,
    reset_seed: wp.int32,
    joint_noise: wp.float32,
    velocity_noise: wp.float32,
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
        if col >= wp.int32(7) and col < wp.int32(15):
            rng = wp.rand_init(reset_seed, world * wp.int32(97) + col)
            value = value + wp.randf(rng, -joint_noise, joint_noise)
        joint_q[idx] = value
    if col < dof_stride:
        idx = world * dof_stride + col
        value = default_joint_qd[idx]
        if col >= wp.int32(6):
            rng = wp.rand_init(reset_seed + wp.int32(13), world * wp.int32(101) + col)
            value = value + wp.randf(rng, -velocity_noise, velocity_noise)
        joint_qd[idx] = value
    if col < ACTION_DIM_ANT:
        previous_actions[world, col] = wp.float32(0.0)
        current_actions[world, col] = wp.float32(0.0)
    if col == wp.int32(0):
        episode_steps[world] = wp.int32(0)


@dataclass
class ConfigEnvAntPhoenX:
    """Experimental Ant locomotion environment backed by SolverPhoenX."""

    world_count: int = 2048
    frame_dt: float = 1.0 / 50.0
    sim_substeps: int = 4
    solver_iterations: int = 8
    velocity_iterations: int = 1
    torque_limit: float = 15.0
    max_episode_steps: int = 500
    min_height: float = 0.25
    max_height: float = 1.4
    min_upright_cos: float = 0.0
    forward_reward_weight: float = 1.0
    healthy_reward: float = 1.0
    ctrl_cost_weight: float = 0.05
    action_rate_cost_weight: float = 0.002
    angular_cost_weight: float = 0.02
    vertical_cost_weight: float = 0.02
    termination_reward: float = -1.0
    target_velocity: float = 0.7
    joint_noise: float = 0.03
    velocity_noise: float = 0.05
    ground_friction: float = 1.5
    auto_reset: bool = True


class EnvAntPhoenX:
    """Vectorized Ant torque-control environment for PPO validation."""

    obs_dim = OBS_DIM_ANT
    action_dim = ACTION_DIM_ANT

    def __init__(self, config: ConfigEnvAntPhoenX | None = None, *, device: wp.context.Devicelike = None):
        self.config = config or ConfigEnvAntPhoenX()
        self.device = wp.get_device(device)
        self.world_count = int(self.config.world_count)
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if int(self.config.sim_substeps) <= 0:
            raise ValueError("sim_substeps must be positive")

        self.model = self._build_model()
        self.coord_stride = int(self.model.joint_coord_count) // self.world_count
        self.dof_stride = int(self.model.joint_dof_count) // self.world_count
        if self.coord_stride != 15 or self.dof_stride != 14:
            raise ValueError(f"Expected Ant strides (15, 14), got ({self.coord_stride}, {self.dof_stride})")
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
        self.current_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.previous_actions = wp.zeros((self.world_count, self.action_dim), dtype=wp.float32, device=self.device)
        self.episode_steps = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((self.world_count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.forward_velocities = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_rewards = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_successes = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_forward_velocities = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._reset_seed = 0
        self.sim_time = 0.0
        self.reset()

    def _build_model(self):
        ant_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ant_builder.default_shape_cfg.ke = 5.0e4
        ant_builder.default_shape_cfg.kd = 5.0e2
        ant_builder.default_shape_cfg.kf = 1.0e3
        ant_builder.default_shape_cfg.mu = float(self.config.ground_friction)
        ant_builder.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            up_axis="Y",
            parse_meshes=False,
            parse_visuals=False,
            ignore_names=("floor",),
            enable_self_collisions=False,
            parse_mujoco_options=False,
        )
        ant_builder.joint_q[:3] = [0.0, 0.70, 0.0]
        ant_builder.joint_q[3:7] = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)
        ant_builder.joint_q[7:15] = list(_DEFAULT_ANT_Q)
        for dof in range(6, 14):
            ant_builder.joint_target_mode[dof] = int(newton.JointTargetMode.EFFORT)
            ant_builder.joint_effort_limit[dof] = float(self.config.torque_limit)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        for _ in range(self.world_count):
            builder.add_world(ant_builder)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = float(self.config.ground_friction)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)
        model.set_gravity((0.0, -9.81, 0.0))
        return model

    def observe(self) -> wp.array:
        wp.launch(
            ant_observe_reward_kernel,
            dim=(self.world_count, self.obs_dim),
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.current_actions,
                self.previous_actions,
                self.coord_stride,
                self.dof_stride,
                self.episode_steps,
                int(self.config.max_episode_steps),
                float(self.config.min_height),
                float(self.config.max_height),
                float(self.config.min_upright_cos),
                float(self.config.forward_reward_weight),
                float(self.config.healthy_reward),
                float(self.config.ctrl_cost_weight),
                float(self.config.action_rate_cost_weight),
                float(self.config.angular_cost_weight),
                float(self.config.vertical_cost_weight),
                float(self.config.termination_reward),
                float(self.config.target_velocity),
            ],
            outputs=[self.obs, self.rewards, self.dones, self.successes, self.forward_velocities],
            device=self.device,
        )
        return self.obs

    def reset(self) -> wp.array:
        wp.copy(self.state_0.joint_q, self.model.joint_q)
        wp.copy(self.state_0.joint_qd, self.model.joint_qd)
        self.control.joint_f.zero_()
        self.current_actions.zero_()
        self.previous_actions.zero_()
        self.episode_steps.zero_()
        self.dones.zero_()
        self.successes.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time = 0.0
        return self.observe()

    def reset_done(self) -> None:
        max_cols = max(self.coord_stride, self.dof_stride, self.action_dim)
        self._reset_seed = (self._reset_seed + 1_000_003) & 0x7FFFFFFF
        wp.launch(
            ant_reset_done_worlds_kernel,
            dim=(self.world_count, max_cols),
            inputs=[
                self.dones,
                self.model.joint_q,
                self.model.joint_qd,
                self.coord_stride,
                self.dof_stride,
                int(self._reset_seed),
                float(self.config.joint_noise),
                float(self.config.velocity_noise),
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
        max_cols = max(self.dof_stride, self.action_dim)
        wp.launch(
            ant_apply_actions_kernel,
            dim=(self.world_count, max_cols),
            inputs=[actions, float(self.config.torque_limit), self.dof_stride],
            outputs=[self.current_actions, self.control.joint_f],
            device=self.device,
        )

        substeps = int(self.config.sim_substeps)
        sub_dt = float(self.config.frame_dt) / float(substeps)
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            ant_increment_episode_steps_kernel, dim=self.world_count, outputs=[self.episode_steps], device=self.device
        )
        self.observe()
        wp.copy(self.step_rewards, self.rewards)
        wp.copy(self.step_dones, self.dones)
        wp.copy(self.step_successes, self.successes)
        wp.copy(self.step_forward_velocities, self.forward_velocities)
        wp.copy(self.previous_actions, self.current_actions)
        if self.config.auto_reset:
            self.reset_done()
            self.observe()
        self.sim_time += float(self.config.frame_dt)
        return self.obs, self.step_rewards, self.step_dones


@dataclass(frozen=True)
class StatsEvaluateAntPPO:
    steps: int
    mean_reward: float
    mean_done: float
    fall_fraction: float
    mean_survival_steps: float
    mean_forward_velocity: float
    mean_displacement_x: float
    mean_success: float
    samples_per_second: float


def _format_checkpoint_path(path: str | None, iteration: int) -> str | None:
    if path is None:
        return None
    return str(path).format(iteration=int(iteration))


def _make_trainer(args: argparse.Namespace, env: EnvAntPhoenX) -> rl.TrainerPPO:
    ppo_config = rl.ConfigPPO(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_ratio=float(args.clip_ratio),
        entropy_coeff=float(args.entropy_coeff),
        value_loss_coeff=float(args.value_loss_coeff),
        actor_lr=float(args.learning_rate),
        critic_lr=float(args.learning_rate),
        train_epochs=int(args.train_epochs),
        minibatch_size=int(args.minibatch_size),
        replay_ratio=float(args.replay_ratio),
        normalize_advantages=True,
        reward_clip=float(args.reward_clip),
        max_grad_norm=float(args.max_grad_norm),
        manual_actor_backward=not bool(args.disable_manual_backward),
        manual_critic_backward=not bool(args.disable_manual_backward),
    )
    if args.resume_checkpoint is not None:
        trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=env.device)
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match Ant")
        return trainer
    return rl.TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=tuple(int(v) for v in args.hidden_layers),
        config=ppo_config,
        device=env.device,
        seed=int(args.seed),
        squash_actions=True,
        activation=str(args.activation),
        log_std_init=float(args.log_std_init),
    )


def evaluate_ant_ppo(
    trainer: rl.TrainerPPO, env_config: ConfigEnvAntPhoenX, args: argparse.Namespace
) -> StatsEvaluateAntPPO:
    eval_config = ConfigEnvAntPhoenX(
        **{**asdict(env_config), "world_count": int(args.eval_world_count), "auto_reset": False, "max_episode_steps": 0}
    )
    env = EnvAntPhoenX(eval_config, device=args.device)
    obs = env.reset()
    trainer.reset_rollout_state()
    q0 = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    start_x = q0[:, 0].copy()
    last_alive_x = start_x.copy()
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    reward_sum = 0.0
    done_sum = 0.0
    success_sum = 0.0
    forward_sum = 0.0
    alive_sample_count = 0
    t0 = time.perf_counter()
    for step in range(int(args.eval_steps)):
        alive_before = first_done_step < 0
        actions, _log_probs, _values = trainer.act(obs, seed=int(args.seed) + 10_000 + step, deterministic=True)
        obs, rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        reward_sum += float(np.mean(rewards.numpy()))
        done_sum += float(np.mean(done_np))
        step_successes = env.step_successes.numpy()
        step_forward = env.step_forward_velocities.numpy()
        if np.any(alive_before):
            success_sum += float(np.sum(step_successes[alive_before]))
            forward_sum += float(np.sum(step_forward[alive_before]))
            alive_sample_count += int(np.sum(alive_before))
            q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
            last_alive_x[alive_before] = q[alive_before, 0]
        first_done_step[(first_done_step < 0) & done_np] = step + 1
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    survival_steps = np.where(first_done_step >= 0, first_done_step, int(args.eval_steps))
    alive_den = float(max(alive_sample_count, 1))
    return StatsEvaluateAntPPO(
        steps=int(args.eval_steps),
        mean_reward=reward_sum / float(args.eval_steps),
        mean_done=done_sum / float(args.eval_steps),
        fall_fraction=float(np.mean(first_done_step >= 0)),
        mean_survival_steps=float(np.mean(survival_steps)),
        mean_forward_velocity=forward_sum / alive_den,
        mean_displacement_x=float(np.mean(last_alive_x - start_x)),
        mean_success=success_sum / alive_den,
        samples_per_second=float(env.world_count * int(args.eval_steps)) / elapsed,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    env_config = ConfigEnvAntPhoenX(
        world_count=int(args.world_count),
        frame_dt=float(args.frame_dt),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        torque_limit=float(args.torque_limit),
        max_episode_steps=int(args.max_episode_steps),
        min_height=float(args.min_height),
        max_height=float(args.max_height),
        min_upright_cos=float(args.min_upright_cos),
        forward_reward_weight=float(args.forward_reward_weight),
        healthy_reward=float(args.healthy_reward),
        ctrl_cost_weight=float(args.ctrl_cost_weight),
        action_rate_cost_weight=float(args.action_rate_cost_weight),
        angular_cost_weight=float(args.angular_cost_weight),
        vertical_cost_weight=float(args.vertical_cost_weight),
        termination_reward=float(args.termination_reward),
        target_velocity=float(args.target_velocity),
        joint_noise=float(args.joint_noise),
        velocity_noise=float(args.velocity_noise),
        ground_friction=float(args.ground_friction),
    )
    env = EnvAntPhoenX(env_config, device=args.device)
    trainer = _make_trainer(args, env)
    if bool(args.eval_only):
        if args.resume_checkpoint is None:
            raise ValueError("--eval-only requires --resume-checkpoint")
        eval_stats = evaluate_ant_ppo(trainer, env_config, args)
        result = {
            "elapsed_seconds": 0.0,
            "final_checkpoint": args.resume_checkpoint,
            "final_train_stats": {},
            "eval_stats": asdict(eval_stats),
        }
        if args.summary_path is not None:
            Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return result
    buffer = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=env.device,
    )
    trainer.reserve_buffers(buffer.num_samples)

    history: list[dict[str, float]] = []
    start_iteration = int(getattr(trainer, "iteration", 0))
    t0 = time.perf_counter()
    for local_iteration in range(int(args.iterations)):
        iteration = start_iteration + local_iteration
        iter_t0 = time.perf_counter()
        rl.collect_ppo_rollout(env, trainer, buffer, seed=int(args.seed) + iteration * int(args.rollout_steps))
        rollout_seconds = time.perf_counter() - iter_t0
        mean_reward, mean_done, mean_success = buffer.reward_done_success_means()
        update_t0 = time.perf_counter()
        update_stats = trainer.update(buffer)
        update_seconds = time.perf_counter() - update_t0
        mean_forward_velocity = float(np.mean(env.step_forward_velocities.numpy()))
        samples = int(args.rollout_steps) * env.world_count
        stats = {
            "iteration": float(iteration),
            "mean_reward": float(mean_reward),
            "mean_done": float(mean_done),
            "mean_success": float(mean_success),
            "mean_forward_velocity": mean_forward_velocity,
            "samples_per_second": float(samples) / max(rollout_seconds + update_seconds, 1.0e-12),
            "rollout_seconds": float(rollout_seconds),
            "update_seconds": float(update_seconds),
            "policy_loss": float(update_stats.policy_loss),
            "value_loss": float(update_stats.value_loss),
            "approx_kl": float(update_stats.approx_kl),
            "clip_fraction": float(update_stats.clip_fraction),
        }
        history.append(stats)
        if int(args.log_interval) > 0 and (
            iteration % int(args.log_interval) == 0 or local_iteration == int(args.iterations) - 1
        ):
            print(
                f"iter={iteration:04d} reward={mean_reward:.3f} done={mean_done:.4f} "
                f"success={mean_success:.3f} vx={mean_forward_velocity:.3f} "
                f"sps={stats['samples_per_second']:.0f} rollout={rollout_seconds:.3f}s update={update_seconds:.3f}s "
                f"pi_loss={update_stats.policy_loss:.4f} v_loss={update_stats.value_loss:.4f}"
            )
        trainer.iteration = iteration + 1
        if (
            args.checkpoint_path is not None
            and int(args.checkpoint_interval) > 0
            and trainer.iteration % int(args.checkpoint_interval) == 0
        ):
            path = _format_checkpoint_path(args.checkpoint_path, trainer.iteration)
            assert path is not None
            trainer.save_checkpoint(path, iteration=trainer.iteration)

    final_checkpoint = _format_checkpoint_path(args.checkpoint_path, start_iteration + int(args.iterations))
    if final_checkpoint is not None:
        Path(final_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(final_checkpoint, iteration=start_iteration + int(args.iterations))
    eval_stats = evaluate_ant_ppo(trainer, env_config, args) if not bool(args.no_eval) else None
    result = {
        "elapsed_seconds": time.perf_counter() - t0,
        "final_checkpoint": final_checkpoint,
        "final_train_stats": history[-1] if history else {},
        "eval_stats": asdict(eval_stats) if eval_stats is not None else None,
    }
    if args.summary_path is not None:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--world-count", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--eval-only", action="store_true", help="Load --resume-checkpoint and run deterministic no-reset eval only."
    )

    parser.add_argument("--frame-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--torque-limit", type=float, default=15.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--min-height", type=float, default=0.25)
    parser.add_argument("--max-height", type=float, default=1.4)
    parser.add_argument("--min-upright-cos", type=float, default=0.0)
    parser.add_argument("--ground-friction", type=float, default=1.5)
    parser.add_argument("--joint-noise", type=float, default=0.03)
    parser.add_argument("--velocity-noise", type=float, default=0.05)

    parser.add_argument("--forward-reward-weight", type=float, default=1.0)
    parser.add_argument("--healthy-reward", type=float, default=1.0)
    parser.add_argument("--ctrl-cost-weight", type=float, default=0.05)
    parser.add_argument("--action-rate-cost-weight", type=float, default=0.002)
    parser.add_argument("--angular-cost-weight", type=float, default=0.02)
    parser.add_argument("--vertical-cost-weight", type=float, default=0.02)
    parser.add_argument("--termination-reward", type=float, default=-1.0)
    parser.add_argument("--target-velocity", type=float, default=0.7)

    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--activation", choices=("relu", "elu", "tanh"), default="elu")
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=1.0e-3)
    parser.add_argument("--value-loss-coeff", type=float, default=1.0)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--disable-manual-backward", action="store_true")

    parser.add_argument("--eval-world-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    train(_make_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
