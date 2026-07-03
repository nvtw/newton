# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Train DR Legs hold-pose or walking with PhoenX and Warp-only PPO.

The closed linkage runs in maximal coordinates and the complete rollout and
PPO update run in one CUDA graph. No IsaacLab runtime dependency is used.

Examples::

    python newton/_src/solvers/phoenx/rl_training/examples/train_dr_legs_phoenx_ppo.py --task hold
    python newton/_src/solvers/phoenx/rl_training/examples/train_dr_legs_phoenx_ppo.py --task walk
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import warp as wp

import newton.rl as rl


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("hold", "walk"), default="hold")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-count", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--sim-substeps", type=int, default=20)
    parser.add_argument("--collision-refresh-interval", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=8)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--command-x", type=float, default=0.3)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(512, 512, 512))
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser


def _checkpoint_path(pattern: str, iteration: int) -> Path:
    return Path(pattern.format(iteration=int(iteration)))


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    device = wp.get_device(args.device)
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        raise RuntimeError("DR Legs training requires a CUDA device with Warp memory pooling enabled")

    env = rl.EnvDrLegsPhoenX(
        rl.ConfigEnvDrLegsPhoenX(
            task=str(args.task),
            world_count=int(args.world_count),
            sim_substeps=int(args.sim_substeps),
            collision_refresh_interval=int(args.collision_refresh_interval),
            solver_iterations=int(args.solver_iterations),
            velocity_iterations=int(args.velocity_iterations),
            command=(float(args.command_x), float(args.command_y), float(args.command_yaw)),
        ),
        device=device,
    )
    sample_count = int(args.world_count) * int(args.rollout_steps)
    ppo_config = rl.ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        value_loss_coeff=1.0,
        actor_lr=float(args.learning_rate),
        critic_lr=float(args.learning_rate),
        train_epochs=int(args.train_epochs),
        minibatch_size=max(sample_count // 4, 1),
        normalize_advantages=True,
        max_grad_norm=1.0,
        manual_actor_backward=True,
        manual_critic_backward=True,
    )
    if args.resume_checkpoint is None:
        trainer = rl.TrainerPPO(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=tuple(int(width) for width in args.hidden_layers),
            config=ppo_config,
            device=device,
            seed=int(args.seed),
            squash_actions=True,
            activation="elu",
            log_std_init=0.0,
        )
    else:
        trainer = rl.load_ppo_checkpoint(args.resume_checkpoint, config=ppo_config, device=device)
        if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
            raise ValueError("Checkpoint dimensions do not match DR Legs")

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
        should_log = int(args.log_interval) > 0 and (
            iteration % int(args.log_interval) == 0 or local_iteration == int(args.iterations) - 1
        )
        if should_log:
            reward, done, success = buffer.reward_done_success_means()
            elapsed = max(time.perf_counter() - start, 1.0e-12)
            completed_samples = (local_iteration + 1) * sample_count
            print(
                f"task={args.task} iter={iteration:04d} reward={reward:.4f} done={done:.4f} "
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
