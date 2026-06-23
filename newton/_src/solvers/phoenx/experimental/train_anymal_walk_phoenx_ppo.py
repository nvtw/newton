# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train Anymal C command walking with PhoenX and Warp-only PPO.

This is a focused experimental runner for validating PhoenX RL on a quadruped
that is simpler than G1 but uses the production Anymal PhoenX environment. It
uses dense velocity-command rewards and evaluates checkpoints without auto-reset
so short rollout reward cannot hide falls.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import newton.rl as rl

_BASE_ENV = {
    "reward_mode": "dense_command",
    "command": (0.6, 0.0, 0.0),
    "max_episode_steps": 500,
    "lin_vel_reward_scale": 2.0,
    "yaw_rate_reward_scale": 0.5,
    "z_vel_reward_scale": -1.0,
    "ang_vel_reward_scale": -0.2,
    "action_rate_reward_scale": -0.01,
    "joint_speed_reward_scale": -1.0e-4,
    "flat_orientation_reward_scale": -3.0,
    "forward_progress_reward_scale": 0.5,
    "fall_reward_scale": -2.0,
    "energy_reward_scale": -2.0e-5,
    "min_base_height": 0.30,
    "min_upright_cos": 0.35,
}


@dataclass(frozen=True)
class StatsEvaluateAnymalWalk:
    """No-reset command-walking evaluation metrics."""

    steps: int
    command: tuple[float, float, float]
    mean_reward: float
    mean_done: float
    fall_fraction: float
    mean_survival_steps: float
    survival_fraction: float
    mean_tracking_perf: float
    mean_forward_velocity: float
    mean_abs_forward_velocity_error: float
    mean_displacement_x: float
    mean_path_length: float
    samples_per_second: float


def build_env_config(
    args: argparse.Namespace, *, world_count: int | None = None, auto_reset: bool = True
) -> rl.ConfigEnvAnymalPhoenX:
    values = dict(_BASE_ENV)
    values.update(
        world_count=int(args.world_count if world_count is None else world_count),
        frame_dt=float(args.frame_dt),
        sim_substeps=int(args.sim_substeps),
        solver_iterations=int(args.solver_iterations),
        velocity_iterations=int(args.velocity_iterations),
        action_scale=float(args.action_scale),
        command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
        max_episode_steps=int(args.max_episode_steps),
        target_base_height=float(args.target_base_height),
        actuator_ke=float(args.actuator_ke),
        actuator_kd=float(args.actuator_kd),
        auto_reset=bool(auto_reset),
    )
    return rl.ConfigEnvAnymalPhoenX(**values)


def build_ppo_config(args: argparse.Namespace) -> rl.ConfigPPO:
    return rl.ConfigPPO(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_ratio=float(args.clip_ratio),
        entropy_coeff=float(args.entropy_coeff),
        value_loss_coeff=float(args.value_loss_coeff),
        value_clip_range=float(args.value_clip_range),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        train_epochs=int(args.train_epochs),
        minibatch_size=int(args.minibatch_size),
        replay_ratio=float(args.replay_ratio),
        normalize_advantages=True,
        reward_clip=float(args.reward_clip),
        max_grad_norm=float(args.max_grad_norm),
        manual_actor_backward=not bool(args.disable_manual_backward),
        manual_critic_backward=not bool(args.disable_manual_backward),
    )


def _format_checkpoint_path(pattern: str | None, iteration: int) -> str | None:
    if pattern is None:
        return None
    return str(pattern).format(iteration=int(iteration))


def evaluate_checkpoint(trainer: rl.TrainerPPO, args: argparse.Namespace) -> StatsEvaluateAnymalWalk:
    env = rl.EnvAnymalPhoenX(
        build_env_config(args, world_count=int(args.eval_world_count), auto_reset=False), device=args.device
    )
    obs = env.reset()
    trainer.reset_rollout_state()
    q0 = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
    start_xy = q0[:, 0:2].copy()
    previous_xy = start_xy.copy()
    last_alive_xy = start_xy.copy()
    first_done_step = np.full(env.world_count, -1, dtype=np.int32)
    reward_sum = 0.0
    done_sum = 0.0
    tracking_sum = 0.0
    forward_error_sum = 0.0
    forward_sum = 0.0
    alive_count = 0
    path_length = np.zeros(env.world_count, dtype=np.float64)
    t0 = time.perf_counter()
    command_x = float(args.command_x)
    for step in range(int(args.eval_steps)):
        alive_before = first_done_step < 0
        actions, _log_probs, _values = trainer.act(obs, seed=int(args.seed) + 90_000 + step, deterministic=True)
        obs, rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        reward_sum += float(np.mean(rewards.numpy()))
        done_sum += float(np.mean(done_np))
        step_successes = env.step_successes.numpy()
        obs_np = obs.numpy()
        q = env.state_0.joint_q.numpy().reshape(env.world_count, env.coord_stride)
        xy = q[:, 0:2].copy()
        if np.any(alive_before):
            alive_idx = alive_before
            path_length[alive_idx] += np.linalg.norm(xy[alive_idx] - previous_xy[alive_idx], axis=1)
            last_alive_xy[alive_idx] = xy[alive_idx]
            vx_alive = obs_np[alive_idx, 0]
            forward_sum += float(np.sum(vx_alive))
            forward_error_sum += float(np.sum(np.abs(vx_alive - command_x)))
            tracking_sum += float(np.sum(step_successes[alive_idx]))
            alive_count += int(np.sum(alive_idx))
        previous_xy = xy
        first_done_step[(first_done_step < 0) & done_np] = step + 1
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    survival_steps = np.where(first_done_step >= 0, first_done_step, int(args.eval_steps))
    alive_den = float(max(alive_count, 1))
    displacement = last_alive_xy - start_xy
    return StatsEvaluateAnymalWalk(
        steps=int(args.eval_steps),
        command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
        mean_reward=reward_sum / float(args.eval_steps),
        mean_done=done_sum / float(args.eval_steps),
        fall_fraction=float(np.mean(first_done_step >= 0)),
        mean_survival_steps=float(np.mean(survival_steps)),
        survival_fraction=float(np.mean(survival_steps)) / float(max(int(args.eval_steps), 1)),
        mean_tracking_perf=tracking_sum / alive_den,
        mean_forward_velocity=forward_sum / alive_den,
        mean_abs_forward_velocity_error=forward_error_sum / alive_den,
        mean_displacement_x=float(np.mean(displacement[:, 0])),
        mean_path_length=float(np.mean(path_length)),
        samples_per_second=float(env.world_count * int(args.eval_steps)) / elapsed,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    env_config = build_env_config(args)
    ppo_config = build_ppo_config(args)
    if bool(args.eval_only):
        if args.resume_checkpoint is None:
            raise ValueError("--eval-only requires --resume-checkpoint")
        trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=args.device)
        eval_stats = evaluate_checkpoint(trainer, args)
        result = {"final_checkpoint": args.resume_checkpoint, "final_train_stats": {}, "eval_stats": asdict(eval_stats)}
        if args.summary_path is not None:
            Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return result

    checkpoint_path = args.checkpoint_path
    result = rl.train_anymal_ppo(
        rl.ConfigTrainAnymalPPO(
            iterations=int(args.iterations),
            rollout_steps=int(args.rollout_steps),
            hidden_layers=tuple(int(v) for v in args.hidden_layers),
            activation=str(args.activation),
            log_std_init=float(args.log_std_init),
            env_config=env_config,
            ppo_config=ppo_config,
            device=args.device,
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            use_target_curriculum=False,
            randomize_target_positions=False,
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=int(args.checkpoint_interval),
        )
    )
    final_iteration = int(result.trainer.iteration)
    final_checkpoint = _format_checkpoint_path(checkpoint_path, final_iteration)
    eval_stats = None if bool(args.no_eval) else evaluate_checkpoint(result.trainer, args)
    final_stats = asdict(result.history[-1]) if result.history else {}
    payload = {
        "final_checkpoint": final_checkpoint,
        "final_train_stats": final_stats,
        "eval_stats": asdict(eval_stats) if eval_stats is not None else None,
    }
    if args.summary_path is not None:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return payload


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=12_321)
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--command-x", type=float, default=0.6)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--frame-dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--sim-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--target-base-height", type=float, default=0.62)
    parser.add_argument("--actuator-ke", type=float, default=150.0)
    parser.add_argument("--actuator-kd", type=float, default=5.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--activation", choices=("relu", "elu", "tanh"), default="elu")
    parser.add_argument("--log-std-init", type=float, default=-0.6)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=1.0e-4)
    parser.add_argument("--value-loss-coeff", type=float, default=1.0)
    parser.add_argument("--value-clip-range", type=float, default=0.0)
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
