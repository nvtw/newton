# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import (
    flash_sac_n_step_accumulate_kernel,
    flash_sac_normalize_rewards_kernel,
    flash_sac_return_stats_kernel,
    flash_sac_update_return_normalizer_kernel,
    zero_scalar_kernel,
)
from .sac import BatchSAC, BufferReplaySAC, ConfigSAC, StatsSACUpdate, TrainerSAC


@dataclass
class ConfigFlashSAC(ConfigSAC):
    """Configuration for :class:`TrainerFlashSAC`.

    These defaults follow the upstream FlashSAC configuration. The Warp port
    uses the existing PhoenX MLP implementation in place of upstream's
    batch-normalized residual blocks.

    Args:
        target_sigma: Standard deviation used to derive the target entropy.
        noise_zeta_mu: Exponent of the repeated-noise zeta distribution.
        noise_zeta_max: Maximum number of steps for which exploration noise is reused.
        learning_rate_end: Final cosine-decay learning rate.
        learning_rate_warmup_steps: Number of linear learning-rate warmup updates.
        learning_rate_decay_steps: Number of updates over which to decay the learning rate.
        normalize_weights: Whether to apply FlashSAC unit incoming-weight constraints.
        n_step: Number of transitions in replay returns.
        normalize_rewards: Whether replay samples use discounted-return normalization.
        normalized_return_max: Maximum normalized discounted-return magnitude.
        actor_num_blocks: Number of expanded actor blocks.
        actor_hidden_dim: Actor feature width.
        critic_num_blocks: Number of expanded critic blocks.
        critic_hidden_dim: Critic feature width.
    """

    tau: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    initial_alpha: float = 0.01
    policy_frequency: int = 2
    distributional_critic: bool = True
    distributional_atoms: int = 101
    distributional_v_min: float = -5.0
    distributional_v_max: float = 5.0
    distributional_min_target: bool = True
    target_sigma: float = 0.15
    noise_zeta_mu: float = 2.0
    noise_zeta_max: int = 16
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_steps: int = 0
    learning_rate_decay_steps: int = 1_000_000
    normalize_weights: bool = True
    n_step: int = 1
    normalize_rewards: bool = True
    normalized_return_max: float = 5.0
    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256


class RewardNormalizerFlashSAC:
    """Normalize rewards by running discounted-return scale using Warp arrays."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        normalized_return_max: float = 5.0,
        device: wp.context.Devicelike = None,
    ):
        if normalized_return_max <= 0.0:
            raise ValueError("normalized_return_max must be positive")
        self.gamma = float(gamma)
        self.normalized_return_max = float(normalized_return_max)
        self.device = wp.get_device(device)
        self.returns: wp.array[wp.float32] | None = None
        self.running_mean = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.running_var = wp.ones(1, dtype=wp.float32, device=self.device)
        self.running_count = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.max_abs_return = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._metrics = wp.zeros(3, dtype=wp.float32, device=self.device)

    def update(
        self,
        rewards: wp.array[wp.float32],
        terminateds: wp.array[wp.float32],
        truncateds: wp.array[wp.float32],
    ) -> None:
        """Update discounted returns and their running scale."""

        count = int(rewards.shape[0])
        if self.returns is None:
            self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        elif int(self.returns.shape[0]) != count:
            raise ValueError("Reward normalizer environment count cannot change")
        self._metrics.zero_()
        wp.launch(
            flash_sac_return_stats_kernel,
            dim=count,
            inputs=[rewards, terminateds, truncateds, self.gamma, self.returns],
            outputs=[self._metrics],
            device=self.device,
        )
        wp.launch(
            flash_sac_update_return_normalizer_kernel,
            dim=1,
            inputs=[self._metrics, count],
            outputs=[self.running_mean, self.running_var, self.running_count, self.max_abs_return],
            device=self.device,
        )

    def normalize(self, rewards: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Scale rewards using running return variance and range bounds."""

        normalized = wp.empty_like(rewards)
        wp.launch(
            flash_sac_normalize_rewards_kernel,
            dim=rewards.shape[0],
            inputs=[rewards, self.running_var, self.max_abs_return, self.normalized_return_max, 1.0e-8],
            outputs=[normalized],
            device=self.device,
        )
        return normalized


class BufferReplayFlashSAC(BufferReplaySAC):
    """Uniform Warp replay with n-step returns and reward normalization.

    Args:
        minimum_size: Number of stored transitions required before updates begin.
        capacity: Maximum number of transitions.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        batch_size: Default sampled batch size.
        n_step: Number of transitions accumulated into each replay row.
        gamma: Discount factor used by n-step returns and reward normalization.
        normalize_rewards: Whether sampled rewards are normalized.
        normalized_return_max: Maximum normalized discounted-return magnitude.
        device: Warp device.
    """

    def __init__(
        self,
        *,
        minimum_size: int = 10_000,
        capacity: int = 1_000_000,
        obs_dim: int,
        action_dim: int,
        batch_size: int = 2048,
        n_step: int = 1,
        gamma: float = 0.99,
        normalize_rewards: bool = True,
        normalized_return_max: float = 5.0,
        device: wp.context.Devicelike = None,
    ):
        super().__init__(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            batch_size=batch_size,
            device=device,
        )
        self.minimum_size = int(minimum_size)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.normalize_rewards = bool(normalize_rewards)
        if self.minimum_size < 0 or self.minimum_size > self.capacity:
            raise ValueError("minimum_size must be between zero and capacity")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        self.reward_normalizer = RewardNormalizerFlashSAC(
            gamma=self.gamma, normalized_return_max=normalized_return_max, device=self.device
        )
        self._n_step_transitions: deque[
            tuple[
                wp.array2d[wp.float32],
                wp.array2d[wp.float32],
                wp.array[wp.float32],
                wp.array[wp.float32],
                wp.array[wp.float32],
                wp.array2d[wp.float32],
            ]
        ] = deque(maxlen=self.n_step)

    def add_batch(
        self,
        obs: wp.array2d[wp.float32],
        actions: wp.array2d[wp.float32],
        rewards: wp.array[wp.float32],
        dones: wp.array[wp.float32],
        next_obs: wp.array2d[wp.float32],
        truncateds: wp.array[wp.float32] | None = None,
    ) -> None:
        """Append transitions using upstream terminated and truncated semantics."""

        if truncateds is None:
            truncateds = wp.zeros_like(dones)
        if self.normalize_rewards:
            self.reward_normalizer.update(rewards, dones, truncateds)
        transition = tuple(wp.clone(value) for value in (obs, actions, rewards, dones, truncateds, next_obs))
        self._n_step_transitions.append(transition)
        if len(self._n_step_transitions) < self.n_step:
            return

        first_obs, first_actions, _rewards, _dones, _truncateds, _next_obs = self._n_step_transitions[0]
        latest = self._n_step_transitions[-1]
        aggregate_rewards = wp.clone(latest[2])
        aggregate_dones = wp.clone(latest[3])
        aggregate_truncateds = wp.clone(latest[4])
        aggregate_next_obs = wp.clone(latest[5])
        for transition_row in reversed(tuple(self._n_step_transitions)[:-1]):
            wp.launch(
                flash_sac_n_step_accumulate_kernel,
                dim=(rewards.shape[0], max(self.obs_dim, 1)),
                inputs=[
                    transition_row[2],
                    transition_row[3],
                    transition_row[4],
                    transition_row[5],
                    self.gamma,
                    self.obs_dim,
                    aggregate_rewards,
                    aggregate_dones,
                    aggregate_truncateds,
                    aggregate_next_obs,
                ],
                device=self.device,
            )
        super().add_batch(first_obs, first_actions, aggregate_rewards, aggregate_dones, aggregate_next_obs)

    def sample(self, *, seed: int, batch_size: int | None = None) -> BatchSAC:
        """Sample replay rows and normalize their rewards at current scale."""

        batch = super().sample(seed=seed, batch_size=batch_size)
        if not self.normalize_rewards:
            return batch
        return BatchSAC(
            obs=batch.obs,
            actions=batch.actions,
            rewards=self.reward_normalizer.normalize(batch.rewards),
            dones=batch.dones,
            next_obs=batch.next_obs,
        )

    def can_sample(self) -> bool:
        """Return whether the upstream-style replay warmup is complete."""

        return self.size >= self.minimum_size


class TrainerFlashSAC(TrainerSAC):
    """FlashSAC training preset implemented entirely with Warp.

    The trainer preserves FlashSAC's categorical twin critic, delayed actor,
    learned temperature, unit weight constraints, target entropy rule,
    cosine learning-rate schedule, and zeta-distributed repeated exploration
    noise while reusing PhoenX's device-native SAC update kernels.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_layers: Optional shared hidden layers overriding upstream capacities.
        config: FlashSAC hyperparameters.
        device: Warp device.
        seed: Initializer and exploration seed.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] | None = None,
        config: ConfigFlashSAC | None = None,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        flash_config = config or ConfigFlashSAC()
        if flash_config.target_sigma <= 0.0:
            raise ValueError("target_sigma must be positive")
        if flash_config.noise_zeta_max < 1:
            raise ValueError("noise_zeta_max must be positive")
        if flash_config.noise_zeta_mu <= 0.0:
            raise ValueError("noise_zeta_mu must be positive")
        if flash_config.learning_rate_warmup_steps < 0 or flash_config.learning_rate_decay_steps < 1:
            raise ValueError("learning-rate schedule steps are invalid")
        if (
            min(
                flash_config.actor_num_blocks,
                flash_config.actor_hidden_dim,
                flash_config.critic_num_blocks,
                flash_config.critic_hidden_dim,
            )
            < 1
        ):
            raise ValueError("FlashSAC network dimensions and block counts must be positive")
        if hidden_layers is None:
            actor_layers = self._expanded_block_layers(flash_config.actor_hidden_dim, flash_config.actor_num_blocks)
            critic_layers = self._expanded_block_layers(flash_config.critic_hidden_dim, flash_config.critic_num_blocks)
        else:
            actor_layers = hidden_layers
            critic_layers = hidden_layers
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=actor_layers,
            critic_hidden_layers=critic_layers,
            config=flash_config,
            device=device,
            seed=seed,
        )
        self.config = flash_config
        if flash_config.target_entropy is None:
            sigma_sq = float(flash_config.target_sigma) ** 2
            self.target_entropy = 0.5 * self.action_dim * math.log(2.0 * math.pi * math.e * sigma_sq)
        self.actor.log_std_min = -10.0
        self._noise_rng = np.random.default_rng(seed)
        ranks = np.arange(1, flash_config.noise_zeta_max + 1, dtype=np.float64)
        probabilities = ranks ** (-float(flash_config.noise_zeta_mu))
        self._noise_cdf = np.cumsum(probabilities / probabilities.sum())
        self._noise_repeat_count = 0
        self._noise_repeat_steps = 0
        self._exploration_seed = int(seed)
        if self.config.normalize_weights:
            self._normalize_online_weights()
            self.target_critic1.copy_from(self.critic1)
            self.target_critic2.copy_from(self.critic2)

    def act(
        self,
        obs: wp.array,
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array, wp.array]:
        """Sample an action, reusing exploration noise for zeta-distributed durations."""

        if deterministic:
            return super().act(obs, seed=seed, deterministic=True)
        if self._noise_repeat_count >= self._noise_repeat_steps:
            self._exploration_seed = int(seed)
            uniform = float(self._noise_rng.random())
            self._noise_repeat_steps = int(np.searchsorted(self._noise_cdf, uniform, side="right")) + 1
            self._noise_repeat_count = 0
        self._noise_repeat_count += 1
        return super().act(obs, seed=self._exploration_seed, deterministic=False)

    def update(self, batch: BatchSAC, *, seed: int | None = None, read_stats: bool = True) -> StatsSACUpdate:
        """Update actor, temperature, critic, and target in upstream order."""

        if int(batch.obs.shape[1]) != self.obs_dim or int(batch.next_obs.shape[1]) != self.obs_dim:
            raise ValueError("Batch observation dimensions do not match trainer")
        if int(batch.actions.shape[1]) != self.action_dim:
            raise ValueError("Batch action dimensions do not match trainer")

        batch = self._normalize_batch(batch)
        base_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._actor_loss], device=self.device)
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        for i in range(int(self.config.update_steps)):
            learning_rate = self._scheduled_learning_rate(self._gradient_update_count)
            self.actor_optimizer.lr = learning_rate
            self.critic1_optimizer.lr = learning_rate
            self.critic2_optimizer.lr = learning_rate
            self.alpha_optimizer.lr = learning_rate
            update_seed = base_seed + 3 * i
            if self._gradient_update_count % int(self.config.policy_frequency) == 0:
                self._update_actor(batch, seed=update_seed)
                if self.config.auto_alpha:
                    self._update_alpha(batch, seed=update_seed + 1)
            self._update_critics(batch, seed=update_seed + 2)
            self.target_critic1.soft_update_from(self.critic1, self.config.tau)
            self.target_critic2.soft_update_from(self.critic2, self.config.tau)
            self._gradient_update_count += 1
        self._update_count += 1
        if read_stats:
            return self._read_update_stats()
        return StatsSACUpdate(actor_loss=0.0, critic_loss=0.0, alpha_loss=0.0, alpha=0.0)

    @staticmethod
    def _expanded_block_layers(hidden_dim: int, num_blocks: int) -> tuple[int, ...]:
        """Return widths matching upstream embedder and block expansions."""

        layers = [int(hidden_dim)]
        for _ in range(int(num_blocks)):
            layers.extend((int(hidden_dim) * 4, int(hidden_dim)))
        return tuple(layers)

    def _scheduled_learning_rate(self, step: int) -> float:
        warmup_steps = int(self.config.learning_rate_warmup_steps)
        peak = float(self.config.actor_lr)
        if warmup_steps > 0 and step < warmup_steps:
            return peak * float(step + 1) / float(warmup_steps)
        decay_step = min(max(step - warmup_steps, 0), int(self.config.learning_rate_decay_steps))
        progress = float(decay_step) / float(self.config.learning_rate_decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        end = float(self.config.learning_rate_end)
        return end + (peak - end) * cosine

    def _normalize_online_weights(self) -> None:
        self.actor.normalize_weights()
        self.critic1.normalize_weights()
        self.critic2.normalize_weights()

    def _update_actor(self, batch: BatchSAC, *, seed: int) -> None:
        super()._update_actor(batch, seed=seed)
        if self.config.normalize_weights:
            self.actor.normalize_weights()

    def _update_critics(self, batch: BatchSAC, *, seed: int) -> None:
        super()._update_critics(batch, seed=seed)
        if self.config.normalize_weights:
            self.critic1.normalize_weights()
            self.critic2.normalize_weights()
