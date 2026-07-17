# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train the classic 21-action Humanoid with PhoenX and Warp-only PPO.

The complete rollout and PPO update are captured into one CUDA graph. The
device seed counter changes stochastic actions on every replay without a host
round trip. No IsaacLab runtime dependency is used.

Example::

    python newton/_src/solvers/phoenx/rl_training/examples/train_humanoid_phoenx_ppo.py \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import warp as wp

import newton.rl as rl


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--sim-substeps", type=int, default=2)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(400, 200, 100))
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser


def _checkpoint_path(pattern: str, iteration: int) -> Path:
    return Path(pattern.format(iteration=int(iteration)))


def make_env(
    args: argparse.Namespace, *, world_count: int | None = None, auto_reset: bool = True
) -> rl.EnvHumanoidPhoenX:
    return rl.EnvHumanoidPhoenX(
        rl.ConfigEnvHumanoidPhoenX(
            world_count=int(args.world_count if world_count is None else world_count),
            sim_substeps=int(args.sim_substeps),
            solver_iterations=int(args.solver_iterations),
            velocity_iterations=int(args.velocity_iterations),
            max_episode_steps=900 if auto_reset else 0,
            auto_reset=auto_reset,
        ),
        device=args.device,
    )


def build_ppo_config(args: argparse.Namespace, sample_count: int) -> rl.ConfigPPO:
    return rl.ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.0,
        value_loss_coeff=1.0,
        value_clip_range=0.2,
        adaptive_kl_target=0.008,
        actor_lr=float(args.learning_rate),
        critic_lr=float(args.learning_rate),
        train_epochs=int(args.train_epochs),
        minibatch_size=max(sample_count // 4, 1),
        normalize_advantages=True,
        normalize_observations=True,
        observation_clip=10.0,
        max_grad_norm=1.0,
        manual_actor_backward=True,
        manual_critic_backward=True,
    )


def make_trainer(args: argparse.Namespace, env: rl.EnvHumanoidPhoenX, ppo_config: rl.ConfigPPO) -> rl.TrainerPPO:
    if args.resume_checkpoint is not None:
        trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=env.device)
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match Humanoid")
        return trainer
    return rl.TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=tuple(int(width) for width in args.hidden_layers),
        config=ppo_config,
        device=env.device,
        seed=int(args.seed),
        squash_actions=True,
        activation="elu",
        log_std_init=0.0,
    )


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("Humanoid training requires a CUDA device with Warp memory pooling enabled")

    env = make_env(args)
    sample_count = int(args.world_count) * int(args.rollout_steps)
    ppo_config = build_ppo_config(args, sample_count)
    trainer = make_trainer(args, env, ppo_config)

    buffer = rl.BufferRollout(
        num_steps=int(args.rollout_steps),
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    trainer.reserve_update_buffers(buffer)
    seed_counter = rl.make_seed_counter(
        int(args.seed) + int(trainer.iteration) * int(args.rollout_steps), device=device
    )

    with wp.ScopedCapture(device=device) as capture:
        rl.collect_ppo_rollout_seed_counter(env, trainer, buffer, seed_counter=seed_counter)
        trainer.update(buffer, read_stats=False)
    graph = capture.graph

    start_iteration = int(trainer.iteration)
    start = time.perf_counter()
    for local_iteration in range(int(args.iterations)):
        wp.capture_launch(graph)
        iteration = start_iteration + local_iteration + 1
        trainer.iteration = iteration
        if int(args.log_interval) > 0 and (
            iteration % int(args.log_interval) == 0 or local_iteration == int(args.iterations) - 1
        ):
            reward, done, success = buffer.reward_done_success_means()
            elapsed = max(time.perf_counter() - start, 1.0e-12)
            completed_samples = (local_iteration + 1) * sample_count
            print(
                f"iter={iteration:04d} reward={reward:.4f} done={done:.4f} "
                f"success={success:.3f} sps={completed_samples / elapsed:,.0f}"
            )
        if (
            args.checkpoint_path is not None
            and int(args.checkpoint_interval) > 0
            and iteration % int(args.checkpoint_interval) == 0
        ):
            path = _checkpoint_path(args.checkpoint_path, iteration)
            path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(path, iteration=iteration)

    if args.checkpoint_path is not None:
        path = _checkpoint_path(args.checkpoint_path, trainer.iteration)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(path, iteration=trainer.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
