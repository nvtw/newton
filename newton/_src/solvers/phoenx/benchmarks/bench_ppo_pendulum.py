# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training.env import collect_ppo_rollout
from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout, ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.rl_training.standard_envs import (
    ConfigEnvPendulumV1Warp,
    EnvPendulumV1Warp,
    pendulum_v1_numpy_step,
)


def _mean_return(env: EnvPendulumV1Warp, trainer: TrainerPPO | None, *, steps: int, seed: int) -> float:
    env.reset(seed=seed)
    total = 0.0
    zero_actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=env.device)
    if trainer is not None:
        trainer.reset_rollout_state()
    for step in range(int(steps)):
        if trainer is None:
            actions = zero_actions
        else:
            actions, _log_probs, _values = trainer.act_reuse(env.obs, seed=seed + step, deterministic=True)
        _obs, rewards, dones = env.step(actions)
        if trainer is not None:
            trainer.reset_rollout_state(dones)
        total += float(np.mean(rewards.numpy()))
    return total / float(max(int(steps), 1))


def _reference_error(config: ConfigEnvPendulumV1Warp, device: wp.context.Device) -> float:
    env = EnvPendulumV1Warp(config, device=device)
    theta = np.linspace(-2.7, 2.2, env.world_count, dtype=np.float32)
    theta_dot = np.linspace(-3.0, 3.5, env.world_count, dtype=np.float32)
    actions = np.linspace(-1.3, 1.4, env.world_count, dtype=np.float32).reshape(env.world_count, 1)
    env.theta.assign(theta)
    env.theta_dot.assign(theta_dot)
    action_wp = wp.array(actions, dtype=wp.float32, device=device)
    env.step(action_wp)
    expected_theta, expected_theta_dot, expected_rewards, expected_obs = pendulum_v1_numpy_step(
        theta, theta_dot, actions, config
    )
    err = max(
        float(np.max(np.abs(env.theta.numpy() - expected_theta))),
        float(np.max(np.abs(env.theta_dot.numpy() - expected_theta_dot))),
        float(np.max(np.abs(env.rewards.numpy() - expected_rewards))),
        float(np.max(np.abs(env.obs.numpy() - expected_obs))),
    )
    return err


def run(args: argparse.Namespace) -> dict[str, float | int | dict]:
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("Pendulum PPO benchmark expects a CUDA device")
    env_config = ConfigEnvPendulumV1Warp(
        world_count=args.world_count,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    ppo_config = ConfigPPO(
        gamma=0.95,
        gae_lambda=0.9,
        clip_ratio=0.2,
        entropy_coeff=1.0e-3,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        train_epochs=args.train_epochs,
        normalize_advantages=True,
        value_loss_coeff=0.5,
        value_clip_range=0.0,
        max_grad_norm=0.5,
    )
    env = EnvPendulumV1Warp(env_config, device=device)
    trainer = TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=tuple(args.hidden_layers),
        config=ppo_config,
        device=device,
        seed=args.seed,
        squash_actions=True,
        activation="tanh",
        log_std_init=-0.5,
    )
    buffer = BufferRollout(
        num_steps=args.rollout_steps,
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    trainer.reserve_update_buffers(buffer)

    reference_error = _reference_error(env_config, device)
    random_return = _mean_return(env, None, steps=args.eval_steps, seed=args.seed + 10_000)
    t0 = time.perf_counter()
    for iteration in range(args.iterations):
        collect_ppo_rollout(env, trainer, buffer, seed=args.seed + iteration * args.rollout_steps)
        trainer.update(buffer, read_stats=False)
        trainer.iteration = iteration + 1
    elapsed = max(time.perf_counter() - t0, 1.0e-12)
    trained_return = _mean_return(env, trainer, steps=args.eval_steps, seed=args.seed + 20_000)
    return {
        "task": "Pendulum-v1 Warp",
        "reference": "Gymnasium Pendulum-v1 equations, reimplemented in NumPy",
        "device": str(device),
        "world_count": int(args.world_count),
        "iterations": int(args.iterations),
        "rollout_steps": int(args.rollout_steps),
        "samples": int(args.world_count * args.rollout_steps * args.iterations),
        "seconds": float(elapsed),
        "samples_per_second": float(args.world_count * args.rollout_steps * args.iterations) / elapsed,
        "numpy_reference_max_abs_error": float(reference_error),
        "zero_policy_mean_reward": float(random_return),
        "trained_policy_mean_reward": float(trained_return),
        "reward_delta": float(trained_return - random_return),
        "env_config": asdict(env_config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pure-Warp PPO sanity benchmark on Pendulum-v1 equations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-count", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=(64, 64))
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
