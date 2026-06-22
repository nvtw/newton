# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot G1 RL Training (PhoenX)
#
# Runs the pure-Warp PPO G1 training loop with PhoenX physics while rendering
# a small subset of the vectorized training worlds. Use ``--mode train_replay``
# to save a final checkpoint and immediately replay it, ``--mode replay`` to
# load a saved checkpoint on one robot, or ``--mode sim`` for zero-action
# position hold with the same G1 environment settings.
#
# Command: python -m newton.examples robot_g1_rl_phoenx
#
###########################################################################

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
import newton.rl as rl


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError("hidden layer widths must be positive")
    return layers


def _parse_auto_int(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or 'auto'")
    return parsed


def _checkpoint_path(path: str, iteration: int) -> Path:
    if "{" in path:
        return Path(path.format(iteration=int(iteration)))
    return Path(path)


_DEFAULT_POLICY_PATH = "/tmp/phoenx_g1_policy_{iteration}.npz"


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device(args.device)
        if not self.device.is_cuda:
            raise RuntimeError(
                "G1 PhoenX example requires CUDA; pass --device cuda:0 or set Warp's default CUDA device"
            )

        self.mode = str(args.mode)
        self._training_mode = self.mode in ("train", "train_replay")
        if self.mode == "replay" and args.resume_checkpoint is None:
            raise ValueError("--mode replay requires --resume-checkpoint")
        if self.mode == "sim" and args.resume_checkpoint is not None:
            raise ValueError("--resume-checkpoint is only valid for train/replay modes")

        self.iterations = int(args.iterations)
        self.iterations_per_frame = max(int(args.iterations_per_frame), 1)
        self.replay_steps_per_frame = max(int(args.replay_steps_per_frame), 1)
        self.sim_steps_per_frame = max(int(args.sim_steps_per_frame), 1)
        self.deterministic_replay = bool(args.deterministic_replay)
        self.replay_step_count = 0
        self.sim_step_count = 0
        self.rollout_steps = int(args.rollout_steps)
        self.seed = int(args.seed)
        self.randomize_commands = bool(args.randomize_commands)
        self.command_sampling = str(args.command_sampling)
        self.reset_recurrent_state_on_rollout_start = bool(args.reset_recurrent_state_on_rollout_start)
        self.log_interval = int(args.log_interval)
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_interval = int(args.checkpoint_interval)
        self.save_final_policy = bool(args.save_final_policy)
        self.render_contacts = bool(args.render_contacts)
        self._last_stats: tuple[float, ...] | None = None
        self._samples_seen = 0
        self._training_complete = False
        self._train_replay_active = False
        self._last_checkpoint: tuple[int, Path] | None = None
        self.final_checkpoint_path: Path | None = None
        if self._training_mode and self.save_final_policy and self.checkpoint_path is None:
            self.checkpoint_path = _DEFAULT_POLICY_PATH

        self.replay_command = (float(args.command_x), float(args.command_y), float(args.command_yaw))
        self.interactive_command = bool(args.interactive_command)
        self.steer_forward_speed = float(args.steer_forward_speed)
        self.steer_lateral_speed = float(args.steer_lateral_speed)
        self.steer_yaw_rate = float(args.steer_yaw_rate)
        if (args.target_x is None) != (args.target_y is None):
            raise ValueError("--target-x and --target-y must be provided together")
        self.target_xy = None if args.target_x is None else (float(args.target_x), float(args.target_y))
        self.target_radius = float(args.target_radius)
        self.target_gain = float(args.target_gain)
        self.target_max_speed = float(args.target_max_speed)
        self.target_yaw_gain = float(args.target_yaw_gain)
        self._target_points: wp.array[wp.vec3] | None = None
        if self.target_xy is not None:
            target_point = np.asarray([[self.target_xy[0], self.target_xy[1], 0.04]], dtype=np.float32)
            self._target_points = wp.array(target_point, dtype=wp.vec3, device=self.device)
        self._last_applied_replay_command: tuple[float, float, float] | None = None
        self.command_x_range = (float(args.command_x_min), float(args.command_x_max))
        self.command_y_range = (float(args.command_y_min), float(args.command_y_max))
        self.command_yaw_range = (float(args.command_yaw_min), float(args.command_yaw_max))
        self.command_zero_probability = float(args.command_zero_probability)
        self.command_curriculum_start = float(args.command_curriculum_start)
        self.command_curriculum_samples = int(args.command_curriculum_samples)
        self.command_curriculum_counter: wp.array[wp.int32] | None = None

        if args.world_count is not None:
            world_count = int(args.world_count)
        elif self._training_mode:
            world_count = rl.g1_recipe.WORLD_COUNT
        elif self.mode == "replay":
            world_count = 1
        else:
            world_count = 4

        env_config = rl.g1_recipe.default_g1_env_config(
            world_count=world_count,
            frame_dt=float(args.frame_dt),
            sim_substeps=int(args.sim_substeps),
            solver_iterations=int(args.solver_iterations),
            velocity_iterations=int(args.velocity_iterations),
            joint_friction_model=str(args.joint_friction_model),
            joint_friction_scale=float(args.joint_friction_scale),
            actuation_model=str(args.actuation_model),
            action_scale=float(args.action_scale),
            controlled_action_count=int(args.controlled_action_count),
            command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
            command_x_range=self.command_x_range,
            command_y_range=self.command_y_range,
            command_yaw_range=self.command_yaw_range,
            randomize_commands_on_reset=self.randomize_commands and self.command_sampling == "episode",
            command_zero_probability=self.command_zero_probability,
            command_resample_steps=int(args.command_resample_steps)
            if self.randomize_commands and self.command_sampling == "episode"
            else 0,
            reward_mode=str(args.reward_mode),
            w_track_lin=float(args.w_track_lin),
            w_track_ang=float(args.w_track_ang),
            w_lin_vel_z=float(args.w_lin_vel_z),
            w_ang_vel_xy=float(args.w_ang_vel_xy),
            w_orientation=float(args.w_orientation),
            w_torque=float(args.w_torque),
            w_action_rate=float(args.w_action_rate),
            w_alive=float(args.w_alive),
            w_termination=float(args.w_termination),
            w_sparse_command_success=float(args.w_sparse_command_success),
            sparse_command_velocity_tolerance=float(args.sparse_command_velocity_tolerance),
            sparse_command_yaw_tolerance=float(args.sparse_command_yaw_tolerance),
            w_mechanical_power=float(args.w_mechanical_power),
            w_gait_contact=float(args.w_gait_contact),
            w_gait_swing=float(args.w_gait_swing),
            w_gait_swing_contact=float(args.w_gait_swing_contact),
            w_gait_hip=float(args.w_gait_hip),
            w_base_height=float(args.w_base_height),
            parse_visuals=bool(args.parse_visuals),
            parse_meshes=bool(args.parse_meshes),
            contact_geometry=str(args.contact_geometry),
            auto_reset=self.mode != "sim",
            rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
            threads_per_world=args.threads_per_world,
            multi_world_scheduler=str(args.multi_world_scheduler),
            prepare_refresh_stride=args.prepare_refresh_stride,
        )

        self.env = rl.EnvG1PhoenX(env_config, device=self.device)
        self.trainer: rl.TrainerPPO | None = None
        self.zero_actions: wp.array2d[float] | None = None
        if self.mode == "sim":
            self.zero_actions = wp.zeros(
                (self.env.world_count, self.env.action_dim), dtype=wp.float32, device=self.device
            )
        else:
            ppo_config = rl.g1_recipe.default_g1_ppo_config(
                mirror_loss_coeff=float(args.mirror_loss_coeff),
                policy_network=str(args.policy_network),
                minibatch_size=int(args.minibatch_size),
                replay_ratio=float(args.replay_ratio),
                priority_alpha=float(args.priority_alpha),
                priority_beta=float(args.priority_beta),
                manual_actor_backward=not bool(args.no_manual_actor_backward),
                manual_critic_backward=not bool(args.no_manual_critic_backward),
                manual_mlp_weight_grad_dtype=str(args.manual_mlp_weight_grad_dtype),
                manual_mlp_forward_dtype=str(args.manual_mlp_forward_dtype),
                vtrace_rho_clip=float(args.vtrace_rho_clip),
                vtrace_c_clip=float(args.vtrace_c_clip),
                reward_clip=float(args.reward_clip),
                max_grad_norm=float(args.max_grad_norm),
                value_loss_coeff=float(args.value_loss_coeff),
                value_clip_range=float(args.value_clip_range),
                optimizer=str(args.optimizer),
                optimizer_eps=float(args.optimizer_eps),
                optimizer_weight_decay=float(args.optimizer_weight_decay),
                muon_momentum=float(args.muon_momentum),
            )
        if self.mode != "sim" and args.resume_checkpoint is not None:
            checkpoint_config = None if self.mode == "replay" else ppo_config
            self.trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=checkpoint_config, device=self.device)
            ppo_config = self.trainer.config
            if self.trainer.obs_dim != self.env.obs_dim or self.trainer.action_dim != self.env.action_dim:
                raise ValueError("Checkpoint dimensions do not match the G1 environment")
            if self.trainer.config.mirror_loss_coeff > 0.0:
                self.trainer.set_mirror_map(rl.g1_mirror_map_ppo())
        elif self.mode != "sim":
            self.trainer = rl.TrainerPPO(
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                hidden_layers=args.hidden_layers,
                config=ppo_config,
                device=self.device,
                seed=self.seed,
                squash_actions=bool(args.squash_actions),
                activation=str(args.activation),
                log_std_init=float(args.log_std_init),
                mirror_map=rl.g1_mirror_map_ppo() if ppo_config.mirror_loss_coeff > 0.0 else None,
            )

        self._configure_command_curriculum()
        if self.mode == "replay":
            self._apply_replay_command(self.replay_command)

        self.buffer: rl.BufferRollout | None = None
        if self._training_mode:
            self.buffer = rl.BufferRollout(
                num_steps=self.rollout_steps,
                num_envs=self.env.world_count,
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
                device=self.device,
            )
            self.trainer.reserve_update_buffers(self.buffer)
        elif self.trainer is not None:
            self.trainer.reset_rollout_state()

        self.viewer.set_model(self.env.model)
        render_world_arg = 1 if args.render_worlds is None and self.mode == "replay" else args.render_worlds
        render_worlds = min(max(int(render_world_arg or 4), 1), self.env.world_count)
        self.viewer.set_visible_worlds(range(render_worlds))
        self.viewer.set_world_offsets((float(args.world_spacing), float(args.world_spacing), 0.0))
        self.viewer.set_camera(
            wp.vec3(float(args.camera_x), float(args.camera_y), float(args.camera_z)),
            args.camera_pitch,
            args.camera_yaw,
        )

    def _configure_command_curriculum(self) -> None:
        if not self.randomize_commands:
            return
        start_iteration = int(getattr(self.trainer, "iteration", 0)) if self.trainer is not None else 0
        start_samples = start_iteration * self.rollout_steps * self.env.world_count
        start_samples = min(start_samples, np.iinfo(np.int32).max)
        self.command_curriculum_counter = wp.array(
            np.asarray([start_samples], dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._advance_command_curriculum(0)
        if self.command_sampling == "episode":
            self.env.randomize_commands(
                seed=self.seed + 53_321 + start_iteration,
                command_x_range=self.command_x_range,
                command_y_range=self.command_y_range,
                command_yaw_range=self.command_yaw_range,
                zero_probability=self.command_zero_probability,
            )

    def _advance_command_curriculum(self, sample_delta: int) -> None:
        if self.command_curriculum_counter is None:
            return
        self.env.update_command_curriculum(
            self.command_curriculum_counter,
            sample_delta=int(sample_delta),
            start_scale=self.command_curriculum_start,
            ramp_samples=float(self.command_curriculum_samples),
        )

    def _maybe_randomize_rollout_commands(self, iteration: int) -> None:
        if not self.randomize_commands or self.command_sampling != "rollout":
            return
        self.env.randomize_commands(
            seed=self.seed + 53_321 + int(iteration),
            command_x_range=self.command_x_range,
            command_y_range=self.command_y_range,
            command_yaw_range=self.command_yaw_range,
            zero_probability=self.command_zero_probability,
        )

    def _command_mean_estimate(self) -> tuple[float, float, float]:
        if not self.randomize_commands:
            return tuple(float(v) for v in self.env.config.command)
        return (
            0.5 * (self.command_x_range[0] + self.command_x_range[1]),
            0.5 * (self.command_y_range[0] + self.command_y_range[1]),
            0.5 * (self.command_yaw_range[0] + self.command_yaw_range[1]),
        )

    def _train_iteration(self) -> None:
        if self.buffer is None or self.trainer is None:
            raise RuntimeError("Training mode requires a rollout buffer and PPO trainer")
        iteration = int(self.trainer.iteration)
        self._advance_command_curriculum(self.buffer.num_samples)
        self._maybe_randomize_rollout_commands(iteration)

        t0 = time.perf_counter()
        self.env.collect_ppo_rollout(
            self.trainer,
            self.buffer,
            seed=self.seed + iteration * self.rollout_steps,
            reset_state_at_start=self.reset_recurrent_state_on_rollout_start,
        )
        t1 = time.perf_counter()
        update_stats = self.trainer.update(self.buffer, read_stats=True)
        t2 = time.perf_counter()

        self.trainer.iteration = iteration + 1
        self._samples_seen += self.buffer.num_samples

        should_log = self.log_interval > 0 and (
            self.trainer.iteration % self.log_interval == 0 or self.trainer.iteration == self.iterations
        )
        if should_log:
            reward, done, tracking_perf = self.buffer.reward_done_success_means()
            score = reward * float(self.rollout_steps)
            command_x, command_y, command_yaw = self._command_mean_estimate()
            elapsed = max(t2 - t0, 1.0e-12)
            samples_per_second = float(self.buffer.num_samples) / elapsed
            self._last_stats = (
                score,
                reward,
                done,
                tracking_perf,
                samples_per_second,
                float(update_stats.policy_loss),
                float(update_stats.value_loss),
                float(update_stats.approx_kl),
            )
            print(
                f"iter={self.trainer.iteration:04d} "
                f"score={score:.3f} "
                f"reward_step={reward:.4f} "
                f"perf={tracking_perf:.3f} "
                f"done={done:.4f} "
                f"cmd=({command_x:.2f},{command_y:.2f},{command_yaw:.2f}) "
                f"sps={samples_per_second:.1f} "
                f"rollout={t1 - t0:.3f}s "
                f"update={t2 - t1:.3f}s "
                f"pi_loss={update_stats.policy_loss:.4f} "
                f"v_loss={update_stats.value_loss:.4f}"
            )

        if self.checkpoint_path is not None and self.checkpoint_interval > 0:
            if self.trainer.iteration % self.checkpoint_interval == 0:
                self._save_policy_checkpoint(self.trainer.iteration)

    def _save_policy_checkpoint(self, iteration: int) -> Path | None:
        if self.trainer is None or self.checkpoint_path is None:
            return None
        path = _checkpoint_path(self.checkpoint_path, iteration)
        checkpoint_key = (int(iteration), path)
        if self._last_checkpoint == checkpoint_key:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(path, iteration=iteration)
        self._last_checkpoint = checkpoint_key
        return path

    def _finish_training(self) -> None:
        if self._training_complete:
            return
        if self.trainer is None:
            raise RuntimeError("Training mode requires a PPO trainer")
        if self.save_final_policy:
            self.final_checkpoint_path = self._save_policy_checkpoint(int(self.trainer.iteration))
            if self.final_checkpoint_path is not None:
                print(f"saved_policy={self.final_checkpoint_path}")
        self._training_complete = True

    def _apply_replay_command(self, command: tuple[float, float, float]) -> None:
        command = (float(command[0]), float(command[1]), float(command[2]))
        if self._last_applied_replay_command == command:
            return
        self.env.set_command(command)
        self._last_applied_replay_command = command

    def _target_replay_command(self) -> tuple[float, float, float] | None:
        if self.target_xy is None:
            return None
        joint_q = self.env.state_0.joint_q.numpy()
        px = float(joint_q[0])
        py = float(joint_q[1])
        qx = float(joint_q[3])
        qy = float(joint_q[4])
        qz = float(joint_q[5])
        qw = float(joint_q[6])
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        dx = self.target_xy[0] - px
        dy = self.target_xy[1] - py
        distance = math.hypot(dx, dy)
        if distance <= self.target_radius:
            return (0.0, 0.0, 0.0)
        speed = min(self.target_max_speed, self.target_gain * max(distance - self.target_radius, 0.0))
        inv_distance = 1.0 / max(distance, 1.0e-6)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        body_dx = cos_yaw * dx + sin_yaw * dy
        body_dy = -sin_yaw * dx + cos_yaw * dy
        heading = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        yaw_rate = max(-self.steer_yaw_rate, min(self.steer_yaw_rate, self.target_yaw_gain * yaw_error))
        return (speed * body_dx * inv_distance, speed * body_dy * inv_distance, yaw_rate)

    def _update_interactive_replay_command(self) -> None:
        forward = 0.0
        lateral = 0.0
        yaw = 0.0
        key_used = False
        if self.interactive_command and hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):
                forward += self.steer_forward_speed
                key_used = True
            if self.viewer.is_key_down("k"):
                forward -= self.steer_forward_speed
                key_used = True
            if self.viewer.is_key_down("j"):
                lateral += self.steer_lateral_speed
                key_used = True
            if self.viewer.is_key_down("l"):
                lateral -= self.steer_lateral_speed
                key_used = True
            if self.viewer.is_key_down("u"):
                yaw += self.steer_yaw_rate
                key_used = True
            if self.viewer.is_key_down("o"):
                yaw -= self.steer_yaw_rate
                key_used = True
        target_command = self._target_replay_command()
        command = (forward, lateral, yaw) if key_used else target_command or self.replay_command
        self._apply_replay_command(command)

    def _start_train_replay(self) -> None:
        if self._train_replay_active:
            return
        if self.trainer is None:
            raise RuntimeError("Replay after training requires a PPO trainer")
        self.env.reset()
        self._apply_replay_command(self.replay_command)
        self.trainer.reset_rollout_state()
        self.replay_step_count = 0
        self.viewer.set_visible_worlds(range(1))
        self._train_replay_active = True
        print("mode=train_replay replay_started=true visible_worlds=1")

    def _replay_policy_step(self) -> None:
        if self.trainer is None:
            raise RuntimeError("Replay mode requires a PPO trainer")
        self._update_interactive_replay_command()
        actions, _log_probs, _values = self.trainer.act_reuse(
            self.env.obs,
            seed=self.seed + 1_000_003 + self.replay_step_count,
            deterministic=self.deterministic_replay,
        )
        _obs, _rewards, dones = self.env.step(actions)
        self.trainer.reset_rollout_state(dones)
        self.replay_step_count += 1

        if self.log_interval > 0 and self.replay_step_count % self.log_interval == 0:
            reward = float(np.mean(self.env.step_rewards.numpy()))
            done = float(np.mean(self.env.step_dones.numpy()))
            tracking_perf = float(np.mean(self.env.step_successes.numpy()))
            print(
                f"replay_step={self.replay_step_count:06d} reward={reward:.4f} perf={tracking_perf:.3f} done={done:.4f}"
            )

    def _sim_step(self) -> None:
        if self.zero_actions is None:
            raise RuntimeError("Simulation mode requires zero-action storage")
        _obs, _rewards, _dones = self.env.step(self.zero_actions)
        self.sim_step_count += 1

        if self.log_interval > 0 and self.sim_step_count % self.log_interval == 0:
            reward = float(np.mean(self.env.step_rewards.numpy()))
            done = float(np.mean(self.env.step_dones.numpy()))
            tracking_perf = float(np.mean(self.env.step_successes.numpy()))
            self._last_stats = (reward, done, tracking_perf)
            print(
                f"sim_step={self.sim_step_count:06d} reward_step={reward:.4f} perf={tracking_perf:.3f} done={done:.4f}"
            )

    def step(self):
        if self.mode == "sim":
            for _ in range(self.sim_steps_per_frame):
                self._sim_step()
            return

        if self.mode == "replay":
            for _ in range(self.replay_steps_per_frame):
                self._replay_policy_step()
            return

        if self.trainer is None:
            raise RuntimeError("Training mode requires a PPO trainer")
        remaining = self.iterations - int(self.trainer.iteration)
        if remaining <= 0:
            self._finish_training()
            if self.mode == "train_replay":
                self._start_train_replay()
                for _ in range(self.replay_steps_per_frame):
                    self._replay_policy_step()
            return
        for _ in range(min(self.iterations_per_frame, remaining)):
            self._train_iteration()
        if int(self.trainer.iteration) >= self.iterations:
            self._finish_training()
            if self.mode == "train_replay":
                self._start_train_replay()

    def render(self):
        self.viewer.begin_frame(self.env.sim_time)
        self.viewer.log_state(self.env.state_0)
        if self._target_points is not None:
            self.viewer.log_points("/g1_target", self._target_points, radii=0.08, colors=(1.0, 0.2, 0.1))
        if self.render_contacts:
            self.viewer.log_contacts(self.env.contacts, self.env.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.mode == "sim":
            if self.sim_step_count <= 0:
                raise RuntimeError("Expected at least one G1 simulation step")
        elif self.mode == "replay" or (self.mode == "train_replay" and self._train_replay_active):
            if self.replay_step_count <= 0:
                raise RuntimeError("Expected at least one G1 PPO replay step")
        elif self.trainer is None or int(self.trainer.iteration) <= 0:
            raise RuntimeError("Expected at least one G1 PPO training iteration")
        obs = self.env.obs.numpy()
        if not np.isfinite(obs).all():
            raise RuntimeError("G1 PPO observations contain non-finite values")
        joint_q = self.env.state_0.joint_q.numpy()
        joint_qd = self.env.state_0.joint_qd.numpy()
        if not np.isfinite(joint_q).all() or not np.isfinite(joint_qd).all():
            raise RuntimeError("G1 PPO state contains non-finite values")
        if self._last_stats is not None and not np.isfinite(np.asarray(self._last_stats)).all():
            raise RuntimeError("G1 PPO diagnostics contain non-finite values")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100000)
        parser.add_argument("--mode", choices=("train", "train_replay", "replay", "sim"), default="train")
        parser.add_argument("--world-count", type=int, default=None)
        parser.add_argument("--render-worlds", type=int, default=None)
        parser.add_argument("--world-spacing", type=float, default=2.0)
        parser.add_argument("--iterations", type=int, default=rl.g1_recipe.TRAIN_ITERATIONS)
        parser.add_argument("--iterations-per-frame", type=int, default=1)
        parser.add_argument("--replay-steps-per-frame", type=int, default=1)
        parser.add_argument("--sim-steps-per-frame", type=int, default=1)
        parser.add_argument("--deterministic-replay", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--rollout-steps", type=int, default=rl.g1_recipe.ROLLOUT_STEPS)
        parser.add_argument("--seed", type=int, default=rl.g1_recipe.SEED)
        parser.add_argument("--frame-dt", type=float, default=rl.g1_recipe.FRAME_DT)
        parser.add_argument("--sim-substeps", type=int, default=rl.g1_recipe.SIM_SUBSTEPS)
        parser.add_argument("--solver-iterations", type=int, default=rl.g1_recipe.SOLVER_ITERATIONS)
        parser.add_argument("--velocity-iterations", type=int, default=rl.g1_recipe.VELOCITY_ITERATIONS)
        parser.add_argument(
            "--joint-friction-model", choices=("hard", "mujoco"), default=rl.g1_recipe.JOINT_FRICTION_MODEL
        )
        parser.add_argument("--joint-friction-scale", type=float, default=rl.g1_recipe.JOINT_FRICTION_SCALE)
        parser.add_argument(
            "--actuation-model",
            choices=("explicit_torque", "constraint_drive"),
            default=rl.g1_recipe.ACTUATION_MODEL,
            help="G1 actuator path used by train, replay, and sim modes.",
        )
        parser.add_argument("--action-scale", type=float, default=rl.g1_recipe.ACTION_SCALE)
        parser.add_argument("--controlled-action-count", type=int, default=rl.g1_recipe.CONTROLLED_ACTION_COUNT)
        parser.add_argument("--command-x", type=float, default=rl.g1_recipe.COMMAND[0])
        parser.add_argument("--command-y", type=float, default=rl.g1_recipe.COMMAND[1])
        parser.add_argument("--command-yaw", type=float, default=rl.g1_recipe.COMMAND[2])
        parser.add_argument("--command-x-min", type=float, default=rl.g1_recipe.COMMAND_X_RANGE[0])
        parser.add_argument("--command-x-max", type=float, default=rl.g1_recipe.COMMAND_X_RANGE[1])
        parser.add_argument("--command-y-min", type=float, default=rl.g1_recipe.COMMAND_Y_RANGE[0])
        parser.add_argument("--command-y-max", type=float, default=rl.g1_recipe.COMMAND_Y_RANGE[1])
        parser.add_argument("--command-yaw-min", type=float, default=rl.g1_recipe.COMMAND_YAW_RANGE[0])
        parser.add_argument("--command-yaw-max", type=float, default=rl.g1_recipe.COMMAND_YAW_RANGE[1])
        parser.add_argument(
            "--randomize-commands", action=argparse.BooleanOptionalAction, default=rl.g1_recipe.RANDOMIZE_COMMANDS
        )
        parser.add_argument("--command-sampling", choices=("episode", "rollout"), default=rl.g1_recipe.COMMAND_SAMPLING)
        parser.add_argument("--command-zero-probability", type=float, default=rl.g1_recipe.COMMAND_ZERO_PROBABILITY)
        parser.add_argument("--command-resample-steps", type=int, default=rl.g1_recipe.COMMAND_RESAMPLE_STEPS)
        parser.add_argument("--command-curriculum-start", type=float, default=rl.g1_recipe.COMMAND_CURRICULUM_START)
        parser.add_argument("--command-curriculum-samples", type=int, default=rl.g1_recipe.COMMAND_CURRICULUM_SAMPLES)
        parser.add_argument(
            "--reset-recurrent-state-on-rollout-start",
            action=argparse.BooleanOptionalAction,
            default=rl.g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
        )
        parser.add_argument("--parse-meshes", action=argparse.BooleanOptionalAction, default=rl.g1_recipe.PARSE_MESHES)
        parser.add_argument("--parse-visuals", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=rl.g1_recipe.CONTACT_GEOMETRY
        )
        parser.add_argument(
            "--reward-mode", choices=("nanog1_dense", "sparse_command"), default=rl.g1_recipe.REWARD_MODE
        )
        parser.add_argument("--w-track-lin", type=float, default=rl.g1_recipe.W_TRACK_LIN)
        parser.add_argument("--w-track-ang", type=float, default=rl.g1_recipe.W_TRACK_ANG)
        parser.add_argument("--w-lin-vel-z", type=float, default=rl.g1_recipe.W_LIN_VEL_Z)
        parser.add_argument("--w-ang-vel-xy", type=float, default=rl.g1_recipe.W_ANG_VEL_XY)
        parser.add_argument("--w-orientation", type=float, default=rl.g1_recipe.W_ORIENTATION)
        parser.add_argument("--w-torque", type=float, default=rl.g1_recipe.W_TORQUE)
        parser.add_argument("--w-action-rate", type=float, default=rl.g1_recipe.W_ACTION_RATE)
        parser.add_argument("--w-alive", type=float, default=rl.g1_recipe.W_ALIVE)
        parser.add_argument("--w-termination", type=float, default=rl.g1_recipe.W_TERMINATION)
        parser.add_argument("--w-sparse-command-success", type=float, default=rl.g1_recipe.W_SPARSE_COMMAND_SUCCESS)
        parser.add_argument(
            "--sparse-command-velocity-tolerance",
            type=float,
            default=rl.g1_recipe.SPARSE_COMMAND_VELOCITY_TOLERANCE,
        )
        parser.add_argument(
            "--sparse-command-yaw-tolerance", type=float, default=rl.g1_recipe.SPARSE_COMMAND_YAW_TOLERANCE
        )
        parser.add_argument("--w-mechanical-power", type=float, default=rl.g1_recipe.W_MECHANICAL_POWER)
        parser.add_argument("--w-gait-contact", type=float, default=rl.g1_recipe.W_GAIT_CONTACT)
        parser.add_argument("--w-gait-swing", type=float, default=rl.g1_recipe.W_GAIT_SWING)
        parser.add_argument("--w-gait-swing-contact", type=float, default=rl.g1_recipe.W_GAIT_SWING_CONTACT)
        parser.add_argument("--w-gait-hip", type=float, default=rl.g1_recipe.W_GAIT_HIP)
        parser.add_argument("--w-base-height", type=float, default=rl.g1_recipe.W_BASE_HEIGHT)
        parser.add_argument("--rigid-contact-max-per-world", type=int, default=rl.g1_recipe.RIGID_CONTACT_MAX_PER_WORLD)
        parser.add_argument("--threads-per-world", type=_parse_auto_int, default=rl.g1_recipe.THREADS_PER_WORLD)
        parser.add_argument(
            "--multi-world-scheduler",
            choices=("auto", "persistent", "flat"),
            default=rl.g1_recipe.MULTI_WORLD_SCHEDULER,
        )
        parser.add_argument(
            "--prepare-refresh-stride", type=_parse_auto_int, default=rl.g1_recipe.PREPARE_REFRESH_STRIDE
        )
        parser.add_argument("--hidden-layers", type=_parse_hidden_layers, default=rl.g1_recipe.HIDDEN_LAYERS)
        parser.add_argument("--activation", default=rl.g1_recipe.ACTIVATION)
        parser.add_argument("--policy-network", choices=("mlp", "puffer_mingru"), default=rl.g1_recipe.POLICY_NETWORK)
        parser.add_argument("--log-std-init", type=float, default=rl.g1_recipe.LOG_STD_INIT)
        parser.add_argument(
            "--squash-actions", action=argparse.BooleanOptionalAction, default=rl.g1_recipe.SQUASH_ACTIONS
        )
        parser.add_argument("--mirror-loss-coeff", type=float, default=rl.g1_recipe.MIRROR_LOSS_COEFF)
        parser.add_argument("--minibatch-size", type=int, default=rl.g1_recipe.MINIBATCH_SIZE)
        parser.add_argument("--replay-ratio", type=float, default=rl.g1_recipe.REPLAY_RATIO)
        parser.add_argument("--priority-alpha", type=float, default=rl.g1_recipe.PRIORITY_ALPHA)
        parser.add_argument("--priority-beta", type=float, default=rl.g1_recipe.PRIORITY_BETA)
        parser.add_argument("--no-manual-actor-backward", action="store_true")
        parser.add_argument("--no-manual-critic-backward", action="store_true")
        parser.add_argument(
            "--manual-mlp-weight-grad-dtype",
            choices=("float32", "bfloat16"),
            default=rl.g1_recipe.MANUAL_MLP_WEIGHT_GRAD_DTYPE,
        )
        parser.add_argument(
            "--manual-mlp-forward-dtype", choices=("float32", "bfloat16"), default=rl.g1_recipe.MANUAL_MLP_FORWARD_DTYPE
        )
        parser.add_argument("--vtrace-rho-clip", type=float, default=rl.g1_recipe.VTRACE_RHO_CLIP)
        parser.add_argument("--vtrace-c-clip", type=float, default=rl.g1_recipe.VTRACE_C_CLIP)
        parser.add_argument("--reward-clip", type=float, default=rl.g1_recipe.REWARD_CLIP)
        parser.add_argument("--max-grad-norm", type=float, default=rl.g1_recipe.MAX_GRAD_NORM)
        parser.add_argument("--value-loss-coeff", type=float, default=rl.g1_recipe.VALUE_LOSS_COEFF)
        parser.add_argument("--value-clip-range", type=float, default=rl.g1_recipe.VALUE_CLIP_RANGE)
        parser.add_argument("--optimizer", choices=("adam", "muon"), default=rl.g1_recipe.OPTIMIZER)
        parser.add_argument("--optimizer-eps", type=float, default=rl.g1_recipe.OPTIMIZER_EPS)
        parser.add_argument("--optimizer-weight-decay", type=float, default=rl.g1_recipe.OPTIMIZER_WEIGHT_DECAY)
        parser.add_argument("--muon-momentum", type=float, default=rl.g1_recipe.MUON_MOMENTUM)
        parser.add_argument("--resume-checkpoint", default=None)
        parser.add_argument("--checkpoint-path", default=None)
        parser.add_argument("--checkpoint-interval", type=int, default=0)
        parser.add_argument("--save-final-policy", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--interactive-command", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--steer-forward-speed", type=float, default=rl.g1_recipe.COMMAND[0])
        parser.add_argument("--steer-lateral-speed", type=float, default=0.5)
        parser.add_argument("--steer-yaw-rate", type=float, default=1.0)
        parser.add_argument("--target-x", type=float, default=None)
        parser.add_argument("--target-y", type=float, default=None)
        parser.add_argument("--target-radius", type=float, default=0.4)
        parser.add_argument("--target-gain", type=float, default=1.5)
        parser.add_argument("--target-max-speed", type=float, default=rl.g1_recipe.COMMAND[0])
        parser.add_argument("--target-yaw-gain", type=float, default=2.0)
        parser.add_argument("--log-interval", type=int, default=1)
        parser.add_argument("--render-contacts", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--camera-x", type=float, default=2.0)
        parser.add_argument("--camera-y", type=float, default=-5.0)
        parser.add_argument("--camera-z", type=float, default=2.0)
        parser.add_argument("--camera-pitch", type=float, default=-25.0)
        parser.add_argument("--camera-yaw", type=float, default=25.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
