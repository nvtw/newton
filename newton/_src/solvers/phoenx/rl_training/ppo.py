# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import (
    compute_gae_kernel,
    gaussian_entropy_kernel,
    normalize_kernel,
    ppo_actor_loss_kernel,
    sum_and_sumsq_kernel,
    value_loss_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, WarpMLP
from .optim import Adam


@dataclass
class ConfigPPO:
    """Configuration for :class:`TrainerPPO`.

    Args:
        gamma: Discount factor.
        gae_lambda: Generalized advantage estimation trace decay.
        clip_ratio: PPO policy-ratio clipping threshold.
        entropy_coeff: Entropy bonus coefficient.
        actor_lr: Actor Adam learning rate.
        critic_lr: Critic Adam learning rate.
        train_epochs: Full-buffer optimization epochs per rollout.
        normalize_advantages: Whether to normalize advantages in-place before
            updating.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 1.0e-3
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    train_epochs: int = 4
    normalize_advantages: bool = True


@dataclass
class StatsPPOUpdate:
    """Scalar diagnostics from a PPO update."""

    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float


class BufferRollout:
    """On-policy storage for vectorized PPO rollouts.

    Args:
        num_steps: Number of policy steps per environment.
        num_envs: Number of parallel worlds/environments.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: wp.context.Devicelike = None,
    ):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        if self.num_steps <= 0 or self.num_envs <= 0:
            raise ValueError("num_steps and num_envs must be positive")

        count = self.num_steps * self.num_envs
        self.obs = wp.zeros((count, self.obs_dim), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((count, self.action_dim), dtype=wp.float32, device=self.device)
        self.old_log_probs = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.successes = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.values = wp.zeros((self.num_steps + 1) * self.num_envs, dtype=wp.float32, device=self.device)
        self.advantages = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)

    @property
    def num_samples(self) -> int:
        """Number of transition rows in the buffer."""

        return self.num_steps * self.num_envs

    def compute_returns(self, *, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns in place."""

        wp.launch(
            compute_gae_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.dones,
                self.values,
                self.num_steps,
                self.num_envs,
                float(gamma),
                float(gae_lambda),
            ],
            outputs=[self.advantages, self.returns],
            device=self.device,
        )

    def normalize_advantages(self, eps: float = 1.0e-8) -> None:
        """Normalize advantages in place."""

        stats = wp.zeros(2, dtype=wp.float32, device=self.device)
        wp.launch(
            sum_and_sumsq_kernel,
            dim=self.num_samples,
            inputs=[self.advantages, self.num_samples],
            outputs=[stats],
            device=self.device,
        )
        stats_np = stats.numpy()
        mean = float(stats_np[0]) / float(self.num_samples)
        var = max(float(stats_np[1]) / float(self.num_samples) - mean * mean, 0.0)
        inv_std = 1.0 / np.sqrt(var + float(eps))
        wp.launch(
            normalize_kernel,
            dim=self.num_samples,
            inputs=[self.advantages, mean, float(inv_std), self.num_samples],
            device=self.device,
        )


class TrainerPPO:
    """Pure-Warp PPO trainer for continuous actions.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Hidden layer widths for actor and critic.
        config: PPO hyperparameters.
        device: Warp device.
        seed: Initializer seed.
        squash_actions: Whether the actor uses tanh-squashed actions.
        activation: Hidden-layer activation.
        log_std_init: Initial actor log-standard-deviation.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (64, 64),
        config: ConfigPPO | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
        squash_actions: bool = True,
        activation: str = "tanh",
        log_std_init: float = 0.0,
    ):
        self.config = config or ConfigPPO()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        self.actor = GaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            state_dependent_std=False,
            log_std_init=log_std_init,
            squash=squash_actions,
            device=self.device,
            seed=seed,
        )
        self.critic = WarpMLP(
            (self.obs_dim, *hidden_layers, 1),
            activation=activation,
            output_activation="linear",
            device=self.device,
            seed=seed + 1,
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.critic_lr)

        self._policy_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._value_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._approx_kl = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._clip_fraction = wp.zeros(1, dtype=wp.float32, device=self.device)

    def act(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Sample actions and evaluate values for rollout collection.

        Args:
            obs: Observation batch.
            seed: Sampling seed.
            deterministic: Use deterministic policy means.

        Returns:
            Tuple ``(actions, log_probs, values)``.
        """

        actions, log_probs, _policy_out = self.actor.sample(
            obs, seed=seed, deterministic=deterministic, requires_grad=False
        )
        values = self.critic.forward(obs, requires_grad=False)
        return actions, log_probs, values

    def update(self, buffer: BufferRollout) -> StatsPPOUpdate:
        """Update actor and critic from a finished rollout buffer."""

        if buffer.obs_dim != self.obs_dim or buffer.action_dim != self.action_dim:
            raise ValueError("BufferRollout dimensions do not match trainer dimensions")
        if self.config.normalize_advantages:
            buffer.normalize_advantages()

        policy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clip_fraction = 0.0
        for _ in range(int(self.config.train_epochs)):
            policy_loss, approx_kl, clip_fraction = self._update_actor(buffer)
            value_loss = self._update_critic(buffer)
        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _update_actor(self, buffer: BufferRollout) -> tuple[float, float, float]:
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._policy_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._approx_kl], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._clip_fraction], device=self.device)
        with wp.Tape() as tape:
            policy_out, new_log_probs = self.actor.log_prob(buffer.obs, buffer.actions, requires_grad=True)
            entropy = wp.empty(buffer.num_samples, dtype=wp.float32, device=self.device, requires_grad=True)
            wp.launch(
                gaussian_entropy_kernel,
                dim=buffer.num_samples,
                inputs=[
                    policy_out,
                    self.actor.log_std,
                    self.action_dim,
                    int(self.actor.state_dependent_std),
                    self.actor.log_std_min,
                    self.actor.log_std_max,
                ],
                outputs=[entropy],
                device=self.device,
            )
            wp.launch(
                ppo_actor_loss_kernel,
                dim=buffer.num_samples,
                inputs=[
                    new_log_probs,
                    buffer.old_log_probs,
                    buffer.advantages,
                    entropy,
                    self.config.clip_ratio,
                    self.config.entropy_coeff,
                    buffer.num_samples,
                ],
                outputs=[self._policy_loss, self._approx_kl, self._clip_fraction],
                device=self.device,
            )
        tape.backward(self._policy_loss)
        loss = float(self._policy_loss.numpy()[0])
        kl = float(self._approx_kl.numpy()[0])
        clip_fraction = float(self._clip_fraction.numpy()[0])
        self.actor_optimizer.step()
        tape.zero()
        return loss, kl, clip_fraction

    def _update_critic(self, buffer: BufferRollout) -> float:
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._value_loss], device=self.device)
        with wp.Tape() as tape:
            values = self.critic.forward(buffer.obs, requires_grad=True)
            wp.launch(
                value_loss_kernel,
                dim=buffer.num_samples,
                inputs=[values, buffer.returns, buffer.num_samples],
                outputs=[self._value_loss],
                device=self.device,
            )
        tape.backward(self._value_loss)
        loss = float(self._value_loss.numpy()[0])
        self.critic_optimizer.step()
        tape.zero()
        return loss
