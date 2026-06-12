# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .anymal import ConfigEnvAnymalPhoenX, EnvAnymalPhoenX
from .ppo import BufferRollout, ConfigPPO, StatsPPOUpdate, TrainerPPO


@dataclass
class ConfigTrainAnymalPPO:
    """Configuration for :func:`train_anymal_ppo`.

    Args:
        iterations: Number of collect-update iterations.
        rollout_steps: Policy steps per environment and iteration.
        hidden_layers: Actor and critic hidden-layer widths.
        activation: Actor and critic hidden-layer activation.
        log_std_init: Initial actor log-standard-deviation.
        env_config: PhoenX Anymal environment configuration.
        ppo_config: PPO optimizer configuration.
        device: Warp device.
        seed: Base RNG seed.
        log_interval: Print every ``log_interval`` iterations when positive.
        use_target_curriculum: Increase sparse target distance after enough successes.
        target_distance_start: Initial sparse target distance [m].
        target_distance_end: Maximum sparse target distance [m].
        target_distance_step: Curriculum distance increment [m].
        target_success_threshold: Success rate required to advance the curriculum.
    """

    iterations: int = 100
    rollout_steps: int = 64
    hidden_layers: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    log_std_init: float = 0.0
    env_config: ConfigEnvAnymalPhoenX | None = None
    ppo_config: ConfigPPO | None = None
    device: wp.context.Devicelike = None
    seed: int = 0
    log_interval: int = 1
    use_target_curriculum: bool = True
    target_distance_start: float = 0.45
    target_distance_end: float = 1.0
    target_distance_step: float = 0.1
    target_success_threshold: float = 0.045


@dataclass
class StatsTrainAnymalPPO:
    """Per-iteration Anymal PPO training diagnostics."""

    iteration: int
    mean_reward: float
    mean_done: float
    mean_success: float
    target_distance: float
    mean_forward_velocity: float
    mean_abs_forward_velocity_error: float
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float


@dataclass
class ResultTrainAnymalPPO:
    """Result returned by :func:`train_anymal_ppo`."""

    trainer: TrainerPPO
    env: EnvAnymalPhoenX
    buffer: BufferRollout
    history: list[StatsTrainAnymalPPO]


def _default_ppo_config() -> ConfigPPO:
    return ConfigPPO(
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=5.0e-3,
        actor_lr=1.0e-3,
        critic_lr=1.0e-3,
        train_epochs=5,
        normalize_advantages=True,
    )


def train_anymal_ppo(config: ConfigTrainAnymalPPO | None = None) -> ResultTrainAnymalPPO:
    """Train an Anymal C walking policy from scratch with Warp-only PPO.

    Args:
        config: Training configuration.

    Returns:
        The trained PPO trainer, environment, rollout buffer, and metric history.
    """

    cfg = config or ConfigTrainAnymalPPO()
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    if cfg.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or ConfigEnvAnymalPhoenX()
    ppo_config = cfg.ppo_config or _default_ppo_config()
    env = EnvAnymalPhoenX(env_config, device=device)
    trainer = TrainerPPO(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=cfg.hidden_layers,
        config=ppo_config,
        device=device,
        seed=cfg.seed,
        squash_actions=True,
        activation=cfg.activation,
        log_std_init=cfg.log_std_init,
    )
    buffer = BufferRollout(
        num_steps=cfg.rollout_steps,
        num_envs=env.world_count,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )

    history: list[StatsTrainAnymalPPO] = []
    command_x = float(env.config.command[0])
    target_xy = np.asarray(env.config.target_position, dtype=np.float32)
    target_distance = float(np.linalg.norm(target_xy))
    if target_distance > 1.0e-6:
        target_direction = target_xy / target_distance
    else:
        target_direction = np.asarray((0.0, 1.0), dtype=np.float32)
    if cfg.use_target_curriculum and env.config.reward_mode == "sparse_target":
        target_distance = float(cfg.target_distance_start)
        target_xy = target_direction * target_distance
        env.set_target_position((float(target_xy[0]), float(target_xy[1])))

    for iteration in range(cfg.iterations):
        env.collect_ppo_rollout(trainer, buffer, seed=cfg.seed + iteration * cfg.rollout_steps)
        rollout_metrics = _rollout_metrics(buffer, command_x, target_distance)
        update_stats = trainer.update(buffer)
        stats = _merge_stats(iteration, rollout_metrics, update_stats)
        history.append(stats)
        if cfg.log_interval > 0 and (iteration % cfg.log_interval == 0 or iteration == cfg.iterations - 1):
            print(
                f"iter={iteration:04d} "
                f"reward={stats.mean_reward:.4f} "
                f"success={stats.mean_success:.3f} "
                f"target={stats.target_distance:.2f} "
                f"vx={stats.mean_forward_velocity:.4f} "
                f"|vx-cmd|={stats.mean_abs_forward_velocity_error:.4f} "
                f"done={stats.mean_done:.4f} "
                f"pi_loss={stats.policy_loss:.4f} "
                f"v_loss={stats.value_loss:.4f}"
            )
        if (
            cfg.use_target_curriculum
            and env.config.reward_mode == "sparse_target"
            and stats.mean_success >= cfg.target_success_threshold
            and target_distance < cfg.target_distance_end
        ):
            target_distance = min(float(cfg.target_distance_end), target_distance + float(cfg.target_distance_step))
            target_xy = target_direction * target_distance
            env.set_target_position((float(target_xy[0]), float(target_xy[1])))

    return ResultTrainAnymalPPO(trainer=trainer, env=env, buffer=buffer, history=history)


def _rollout_metrics(
    buffer: BufferRollout, command_x: float, target_distance: float
) -> tuple[float, float, float, float, float, float]:
    rewards = buffer.rewards.numpy()
    dones = buffer.dones.numpy()
    successes = buffer.successes.numpy()
    obs = buffer.obs.numpy()
    forward_velocity = obs[:, 0]
    return (
        float(np.mean(rewards)),
        float(np.mean(dones)),
        float(np.mean(successes)),
        float(target_distance),
        float(np.mean(forward_velocity)),
        float(np.mean(np.abs(forward_velocity - command_x))),
    )


def _merge_stats(
    iteration: int,
    rollout_metrics: tuple[float, float, float, float, float, float],
    update_stats: StatsPPOUpdate,
) -> StatsTrainAnymalPPO:
    (
        mean_reward,
        mean_done,
        mean_success,
        target_distance,
        mean_forward_velocity,
        mean_abs_forward_velocity_error,
    ) = rollout_metrics
    return StatsTrainAnymalPPO(
        iteration=iteration,
        mean_reward=mean_reward,
        mean_done=mean_done,
        mean_success=mean_success,
        target_distance=target_distance,
        mean_forward_velocity=mean_forward_velocity,
        mean_abs_forward_velocity_error=mean_abs_forward_velocity_error,
        policy_loss=update_stats.policy_loss,
        value_loss=update_stats.value_loss,
        approx_kl=update_stats.approx_kl,
        clip_fraction=update_stats.clip_fraction,
    )
