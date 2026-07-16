# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import (
    concat_obs_action_kernel,
    fill_eps_kernel,
    min_q_target_kernel,
    replay_sample_kernel,
    replay_store_kernel,
    sac_actor_policy_backward_kernel,
    sac_actor_q_backward_kernel,
    sac_alpha_loss_kernel,
    sac_critic_loss_backward_kernel,
    sac_critic_loss_kernel,
    sample_gaussian_actions_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, WarpMLP
from .optim import Adam


@dataclass
class ConfigSAC:
    """Configuration for :class:`TrainerSAC`.

    Args:
        gamma: Discount factor.
        tau: Target critic Polyak update factor.
        actor_lr: Actor Adam learning rate.
        critic_lr: Critic Adam learning rate.
        alpha_lr: Entropy-temperature Adam learning rate.
        initial_alpha: Initial entropy-temperature value.
        auto_alpha: Whether to tune entropy temperature automatically.
        target_entropy: Target action entropy. ``None`` uses ``-action_dim``.
        update_steps: Gradient updates per call to :meth:`TrainerSAC.update`.
    """

    gamma: float = 0.99
    tau: float = 5.0e-3
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    update_steps: int = 1


@dataclass
class StatsSACUpdate:
    """Scalar diagnostics from a SAC update."""

    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float


@dataclass
class BatchSAC:
    """Transition batch used by :class:`TrainerSAC`.

    Args:
        obs: Observation batch.
        actions: Action batch.
        rewards: Reward batch.
        dones: Terminal flags where ``1`` means terminal.
        next_obs: Next-observation batch.
    """

    obs: wp.array
    actions: wp.array
    rewards: wp.array
    dones: wp.array
    next_obs: wp.array

    @property
    def batch_size(self) -> int:
        """Number of transitions in the batch."""

        return int(self.rewards.shape[0])


class BufferReplaySAC:
    """Fixed-size Warp replay buffer for SAC.

    Args:
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        batch_size: Default sampled batch size.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        batch_size: int,
        device: wp.context.Devicelike = None,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.device = wp.get_device(device)
        if self.capacity <= 0 or self.batch_size <= 0:
            raise ValueError("capacity and batch_size must be positive")
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.obs = wp.zeros((self.capacity, self.obs_dim), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((self.capacity, self.action_dim), dtype=wp.float32, device=self.device)
        self.rewards = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.dones = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.next_obs = wp.zeros((self.capacity, self.obs_dim), dtype=wp.float32, device=self.device)
        self.size = 0
        self.position = 0

    def add_batch(
        self,
        obs: wp.array,
        actions: wp.array,
        rewards: wp.array,
        dones: wp.array,
        next_obs: wp.array,
    ) -> None:
        """Append a batch of transitions, overwriting old rows if needed."""

        row_count = int(rewards.shape[0])
        if row_count <= 0:
            return
        if row_count > self.capacity:
            raise ValueError("Cannot add more transitions than replay capacity in one call")
        if int(obs.shape[1]) != self.obs_dim or int(next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Observation dimensions do not match replay buffer")
        if int(actions.shape[1]) != self.action_dim:
            raise ValueError("Action dimensions do not match replay buffer")

        max_cols = max(self.obs_dim, self.action_dim, 1)
        wp.launch(
            replay_store_kernel,
            dim=(row_count, max_cols),
            inputs=[
                obs,
                actions,
                rewards,
                dones,
                next_obs,
                self.position,
                self.capacity,
                row_count,
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[self.obs, self.actions, self.rewards, self.dones, self.next_obs],
            device=self.device,
        )
        self.position = (self.position + row_count) % self.capacity
        self.size = min(self.capacity, self.size + row_count)

    def sample(self, *, seed: int, batch_size: int | None = None) -> BatchSAC:
        """Sample a random transition batch on the Warp device."""

        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        n = int(batch_size) if batch_size is not None else self.batch_size
        if n <= 0:
            raise ValueError("batch_size must be positive")

        obs = wp.empty((n, self.obs_dim), dtype=wp.float32, device=self.device)
        actions = wp.empty((n, self.action_dim), dtype=wp.float32, device=self.device)
        rewards = wp.empty(n, dtype=wp.float32, device=self.device)
        dones = wp.empty(n, dtype=wp.float32, device=self.device)
        next_obs = wp.empty((n, self.obs_dim), dtype=wp.float32, device=self.device)
        max_cols = max(self.obs_dim, self.action_dim, 1)
        wp.launch(
            replay_sample_kernel,
            dim=(n, max_cols),
            inputs=[
                self.obs,
                self.actions,
                self.rewards,
                self.dones,
                self.next_obs,
                self.size,
                int(seed),
                self.obs_dim,
                self.action_dim,
            ],
            outputs=[obs, actions, rewards, dones, next_obs],
            device=self.device,
        )
        return BatchSAC(obs=obs, actions=actions, rewards=rewards, dones=dones, next_obs=next_obs)


class TrainerSAC:
    """Pure-Warp soft actor-critic trainer for continuous actions.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Hidden layer widths for actor and critics.
        config: SAC hyperparameters.
        device: Warp device.
        seed: Initializer and update seed.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (256, 256),
        config: ConfigSAC | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        self.config = config or ConfigSAC()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = wp.get_device(device)
        self.seed = int(seed)
        self._update_count = 0
        if self.config.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        self.target_entropy = (
            -float(self.action_dim) if self.config.target_entropy is None else float(self.config.target_entropy)
        )

        self.actor = GaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=hidden_layers,
            activation="relu",
            state_dependent_std=True,
            squash=True,
            device=self.device,
            seed=self.seed,
        )
        q_input_dim = self.obs_dim + self.action_dim
        self.critic1 = WarpMLP((q_input_dim, *hidden_layers, 1), activation="relu", device=self.device, seed=seed + 1)
        self.critic2 = WarpMLP((q_input_dim, *hidden_layers, 1), activation="relu", device=self.device, seed=seed + 2)
        self.target_critic1 = WarpMLP(
            (q_input_dim, *hidden_layers, 1), activation="relu", device=self.device, seed=seed + 3
        )
        self.target_critic2 = WarpMLP(
            (q_input_dim, *hidden_layers, 1), activation="relu", device=self.device, seed=seed + 4
        )
        self.target_critic1.copy_from(self.critic1)
        self.target_critic2.copy_from(self.critic2)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.config.critic_lr)
        self.log_alpha = wp.array(
            np.array([np.log(float(self.config.initial_alpha))], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)

        self._actor_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._critic_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._alpha_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

    @property
    def alpha(self) -> float:
        """Current SAC entropy-temperature value."""

        return float(np.exp(float(self.log_alpha.numpy()[0])))

    def act(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array, wp.array]:
        """Sample actions for environment interaction."""

        actions, log_probs, _policy_out = self.actor.sample(
            obs, seed=seed, deterministic=deterministic, requires_grad=False
        )
        return actions, log_probs

    def update(self, batch: BatchSAC, *, seed: int | None = None) -> StatsSACUpdate:
        """Update actor, critics, targets, and entropy temperature."""

        if int(batch.obs.shape[1]) != self.obs_dim or int(batch.next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Batch observation dimensions do not match trainer")
        if int(batch.actions.shape[1]) != self.action_dim:
            raise ValueError("Batch action dimensions do not match trainer")

        base_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
        actor_loss = 0.0
        critic_loss = 0.0
        alpha_loss = 0.0
        for i in range(int(self.config.update_steps)):
            critic_loss = self._update_critics(batch, seed=base_seed + 3 * i)
            actor_loss = self._update_actor(batch, seed=base_seed + 3 * i + 1)
            if self.config.auto_alpha:
                alpha_loss = self._update_alpha(batch, seed=base_seed + 3 * i + 2)
            self.target_critic1.soft_update_from(self.critic1, self.config.tau)
            self.target_critic2.soft_update_from(self.critic2, self.config.tau)
        self._update_count += 1
        return StatsSACUpdate(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            alpha=self.alpha,
        )

    def _concat(self, obs: wp.array, actions: wp.array, *, requires_grad: bool) -> wp.array:
        batch_size = int(obs.shape[0])
        out = wp.empty(
            (batch_size, self.obs_dim + self.action_dim),
            dtype=wp.float32,
            device=self.device,
            requires_grad=requires_grad,
        )
        wp.launch(
            concat_obs_action_kernel,
            dim=(batch_size, self.obs_dim + self.action_dim),
            inputs=[obs, actions, self.obs_dim, self.action_dim],
            outputs=[out],
            device=self.device,
        )
        return out

    def _update_critics(self, batch: BatchSAC, *, seed: int) -> float:
        next_actions, next_log_probs, _policy_out = self.actor.sample(
            batch.next_obs, seed=seed, deterministic=False, requires_grad=False
        )
        next_q_input = self._concat(batch.next_obs, next_actions, requires_grad=False)
        target_q1 = self.target_critic1.forward(next_q_input, requires_grad=False)
        target_q2 = self.target_critic2.forward(next_q_input, requires_grad=False)
        targets = wp.empty(batch.batch_size, dtype=wp.float32, device=self.device)
        wp.launch(
            min_q_target_kernel,
            dim=batch.batch_size,
            inputs=[
                batch.rewards,
                batch.dones,
                target_q1,
                target_q2,
                next_log_probs,
                self.config.gamma,
                self.alpha,
            ],
            outputs=[targets],
            device=self.device,
        )

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._critic_loss], device=self.device)
        q_input = self._concat(batch.obs, batch.actions, requires_grad=False)
        q1 = self.critic1.forward_manual(q_input)
        q2 = self.critic2.forward_manual(q_input)
        q1_grad = wp.empty_like(q1)
        q2_grad = wp.empty_like(q2)
        wp.launch(
            sac_critic_loss_kernel,
            dim=batch.batch_size,
            inputs=[q1, q2, targets, batch.batch_size],
            outputs=[self._critic_loss],
            device=self.device,
        )
        wp.launch(
            sac_critic_loss_backward_kernel,
            dim=batch.batch_size,
            inputs=[q1, q2, targets, batch.batch_size],
            outputs=[q1_grad, q2_grad],
            device=self.device,
        )
        self.critic1.backward_manual(q1_grad)
        self.critic2.backward_manual(q2_grad)
        loss = float(self._critic_loss.numpy()[0])
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        return loss

    def _update_actor(self, batch: BatchSAC, *, seed: int) -> float:
        policy_out = self.actor.net.forward_manual(batch.obs)
        actions = wp.empty((batch.batch_size, self.action_dim), dtype=wp.float32, device=self.device)
        log_probs = wp.empty(batch.batch_size, dtype=wp.float32, device=self.device)
        eps = wp.empty((batch.batch_size, self.action_dim), dtype=wp.float32, device=self.device)
        wp.launch(fill_eps_kernel, dim=eps.shape, inputs=[int(seed)], outputs=[eps], device=self.device)
        wp.launch(
            sample_gaussian_actions_kernel,
            dim=batch.batch_size,
            inputs=[
                policy_out,
                self.actor.log_std,
                eps,
                self.action_dim,
                int(self.actor.state_dependent_std),
                int(self.actor.squash),
                0,
                self.actor.log_std_min,
                self.actor.log_std_max,
            ],
            outputs=[actions, log_probs],
            device=self.device,
        )

        q_input = self._concat(batch.obs, actions, requires_grad=False)
        q1 = self.critic1.forward_manual(q_input)
        q2 = self.critic2.forward_manual(q_input)
        q1_grad = wp.empty_like(q1)
        q2_grad = wp.empty_like(q2)
        wp.launch(
            sac_actor_q_backward_kernel,
            dim=batch.batch_size,
            inputs=[q1, q2, batch.batch_size],
            outputs=[q1_grad, q2_grad],
            device=self.device,
        )
        q_input_grad1 = wp.empty_like(q_input)
        q_input_grad2 = wp.empty_like(q_input)
        self.critic1.backward_manual(q1_grad, input_grad=q_input_grad1)
        self.critic2.backward_manual(q2_grad, input_grad=q_input_grad2)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        policy_out_grad = wp.empty_like(policy_out)
        wp.launch(
            sac_actor_policy_backward_kernel,
            dim=batch.batch_size,
            inputs=[
                policy_out,
                eps,
                q_input_grad1,
                q_input_grad2,
                q1,
                q2,
                log_probs,
                self.obs_dim,
                self.action_dim,
                batch.batch_size,
                self.alpha,
                self.actor.log_std_min,
                self.actor.log_std_max,
            ],
            outputs=[self._actor_loss, policy_out_grad],
            device=self.device,
        )
        self.actor.net.backward_manual(policy_out_grad)
        loss = float(self._actor_loss.numpy()[0])
        self.actor_optimizer.step()
        return loss

    def _update_alpha(self, batch: BatchSAC, *, seed: int) -> float:
        _actions, log_probs, _policy_out = self.actor.sample(
            batch.obs, seed=seed, deterministic=False, requires_grad=False
        )
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        with wp.Tape() as tape:
            wp.launch(
                sac_alpha_loss_kernel,
                dim=batch.batch_size,
                inputs=[log_probs, self.log_alpha, batch.batch_size, self.target_entropy],
                outputs=[self._alpha_loss],
                device=self.device,
            )
        tape.backward(self._alpha_loss)
        loss = float(self._alpha_loss.numpy()[0])
        self.alpha_optimizer.step()
        tape.zero()
        return loss
