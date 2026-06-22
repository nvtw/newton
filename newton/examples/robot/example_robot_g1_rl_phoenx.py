# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot G1 RL Training (PhoenX)
#
# Runs the pure-Warp PPO G1 training loop with PhoenX physics while rendering
# a small subset of the vectorized training worlds.
#
# Command: python -m newton.examples robot_g1_rl_phoenx
#
###########################################################################

from __future__ import annotations

import argparse
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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device(args.device)
        if not self.device.is_cuda:
            raise RuntimeError("G1 RL training requires CUDA; pass --device cuda:0 or set Warp's default CUDA device")

        self.iterations = int(args.iterations)
        self.iterations_per_frame = max(int(args.iterations_per_frame), 1)
        self.rollout_steps = int(args.rollout_steps)
        self.seed = int(args.seed)
        self.randomize_commands = bool(args.randomize_commands)
        self.command_sampling = str(args.command_sampling)
        self.reset_recurrent_state_on_rollout_start = bool(args.reset_recurrent_state_on_rollout_start)
        self.log_interval = int(args.log_interval)
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_interval = int(args.checkpoint_interval)
        self.render_contacts = bool(args.render_contacts)
        self._last_stats: tuple[float, float, float, float, float, float, float] | None = None
        self._samples_seen = 0

        self.command_x_range = (float(args.command_x_min), float(args.command_x_max))
        self.command_y_range = (float(args.command_y_min), float(args.command_y_max))
        self.command_yaw_range = (float(args.command_yaw_min), float(args.command_yaw_max))
        self.command_zero_probability = float(args.command_zero_probability)

        env_config = rl.g1_recipe.default_g1_env_config(
            world_count=int(args.world_count),
            frame_dt=float(args.frame_dt),
            sim_substeps=int(args.sim_substeps),
            solver_iterations=int(args.solver_iterations),
            velocity_iterations=int(args.velocity_iterations),
            joint_friction_model=str(args.joint_friction_model),
            joint_friction_scale=float(args.joint_friction_scale),
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
            parse_meshes=bool(args.parse_meshes),
            contact_geometry=str(args.contact_geometry),
            rigid_contact_max_per_world=int(args.rigid_contact_max_per_world),
            threads_per_world=args.threads_per_world,
            multi_world_scheduler=str(args.multi_world_scheduler),
            prepare_refresh_stride=args.prepare_refresh_stride,
        )

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

        self.env = rl.EnvG1PhoenX(env_config, device=self.device)
        if args.resume_checkpoint is not None:
            self.trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=self.device)
            if self.trainer.obs_dim != self.env.obs_dim or self.trainer.action_dim != self.env.action_dim:
                raise ValueError("Checkpoint dimensions do not match the G1 environment")
            if self.trainer.config.mirror_loss_coeff > 0.0:
                self.trainer.set_mirror_map(rl.g1_mirror_map_ppo())
        else:
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

        self.buffer = rl.BufferRollout(
            num_steps=self.rollout_steps,
            num_envs=self.env.world_count,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            device=self.device,
        )
        self.trainer.reserve_update_buffers(self.buffer)

        self.viewer.set_model(self.env.model)
        render_worlds = min(max(int(args.render_worlds), 1), self.env.world_count)
        self.viewer.set_visible_worlds(range(render_worlds))
        self.viewer.set_world_offsets((float(args.world_spacing), float(args.world_spacing), 0.0))
        self.viewer.set_camera(
            wp.vec3(float(args.camera_x), float(args.camera_y), float(args.camera_z)),
            args.camera_pitch,
            args.camera_yaw,
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
        iteration = int(self.trainer.iteration)
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
            command_x, command_y, command_yaw = self._command_mean_estimate()
            elapsed = max(t2 - t0, 1.0e-12)
            samples_per_second = float(self.buffer.num_samples) / elapsed
            self._last_stats = (
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
                f"reward={reward:.4f} "
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
                self.trainer.save_checkpoint(
                    _checkpoint_path(self.checkpoint_path, self.trainer.iteration),
                    iteration=self.trainer.iteration,
                )

    def step(self):
        remaining = self.iterations - int(self.trainer.iteration)
        if remaining <= 0:
            return
        for _ in range(min(self.iterations_per_frame, remaining)):
            self._train_iteration()

    def render(self):
        self.viewer.begin_frame(self.env.sim_time)
        self.viewer.log_state(self.env.state_0)
        if self.render_contacts:
            self.viewer.log_contacts(self.env.contacts, self.env.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if int(self.trainer.iteration) <= 0:
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
        parser.add_argument("--world-count", type=int, default=rl.g1_recipe.WORLD_COUNT)
        parser.add_argument("--render-worlds", type=int, default=4)
        parser.add_argument("--world-spacing", type=float, default=2.0)
        parser.add_argument("--iterations", type=int, default=rl.g1_recipe.TRAIN_ITERATIONS)
        parser.add_argument("--iterations-per-frame", type=int, default=1)
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
        parser.add_argument(
            "--reset-recurrent-state-on-rollout-start",
            action=argparse.BooleanOptionalAction,
            default=rl.g1_recipe.RESET_RECURRENT_STATE_ON_ROLLOUT_START,
        )
        parser.add_argument("--parse-meshes", action=argparse.BooleanOptionalAction, default=rl.g1_recipe.PARSE_MESHES)
        parser.add_argument(
            "--contact-geometry", choices=("mjcf", "nanog1_foot_boxes"), default=rl.g1_recipe.CONTACT_GEOMETRY
        )
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
        parser.add_argument("--log-interval", type=int, default=1)
        parser.add_argument("--render-contacts", action=argparse.BooleanOptionalAction, default=True)
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
