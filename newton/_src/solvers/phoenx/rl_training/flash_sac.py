# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol

import numpy as np
import warp as wp

from .flash_sac_networks import NetworkFlashSAC
from .kernels import (
    flash_sac_n_step_accumulate_kernel,
    flash_sac_normalize_rewards_kernel,
    flash_sac_return_stats_kernel,
    flash_sac_update_return_normalizer_kernel,
    sac_refresh_alpha_kernel,
    zero_scalar_kernel,
)
from .optim import Adam
from .ppo import (
    _pack_optimizer,
    _pack_policy_network,
    _unpack_optimizer,
    _unpack_policy_network,
)
from .sac import BatchSAC, BufferReplaySAC, ConfigSAC, StatsSACUpdate, TrainerSAC


@wp.kernel
def _flash_sac_alpha_loss_kernel(
    log_probs: wp.array[wp.float32],
    log_alpha: wp.array[wp.float32],
    batch_size: wp.int32,
    target_entropy: wp.float32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    alpha = wp.exp(log_alpha[0])
    entropy = -log_probs[i]
    wp.atomic_add(loss, 0, alpha * (entropy - target_entropy) / wp.float32(batch_size))


class EnvFlashSAC(Protocol):
    """Vectorized environment interface consumed by :func:`train_flash_sac`."""

    world_count: int
    obs_dim: int
    action_dim: int
    device: wp.context.Device

    def reset(self) -> wp.array2d[wp.float32]: ...

    def observe(self) -> wp.array2d[wp.float32]: ...

    def step(self, actions: wp.array2d[wp.float32]) -> tuple[wp.array, wp.array, wp.array]: ...


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
        buffer_max_length: Maximum number of replay transitions.
        buffer_min_length: Number of replay transitions required before training.
        sample_batch_size: Number of replay transitions sampled per update.
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
    buffer_max_length: int = 1_000_000
    buffer_min_length: int = 10_000
    sample_batch_size: int = 2048
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

    def save(self, path: str | Path) -> None:
        """Save replay storage, normalization state, and pending n-step rows."""

        data: dict[str, np.ndarray] = {
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "minimum_size": np.asarray(self.minimum_size, dtype=np.int64),
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "batch_size": np.asarray(self.batch_size, dtype=np.int64),
            "n_step": np.asarray(self.n_step, dtype=np.int64),
            "gamma": np.asarray(self.gamma, dtype=np.float32),
            "normalize_rewards": np.asarray(self.normalize_rewards, dtype=np.bool_),
            "normalized_return_max": np.asarray(self.reward_normalizer.normalized_return_max, dtype=np.float32),
            "size": np.asarray(self.size, dtype=np.int64),
            "position": np.asarray(self.position, dtype=np.int64),
            "obs": self.obs.numpy(),
            "actions": self.actions.numpy(),
            "rewards": self.rewards.numpy(),
            "dones": self.dones.numpy(),
            "next_obs": self.next_obs.numpy(),
            "return_running_mean": self.reward_normalizer.running_mean.numpy(),
            "return_running_var": self.reward_normalizer.running_var.numpy(),
            "return_running_count": self.reward_normalizer.running_count.numpy(),
            "return_max_abs": self.reward_normalizer.max_abs_return.numpy(),
            "return_values_present": np.asarray(self.reward_normalizer.returns is not None, dtype=np.bool_),
            "pending_count": np.asarray(len(self._n_step_transitions), dtype=np.int64),
        }
        if self.reward_normalizer.returns is not None:
            data["return_values"] = self.reward_normalizer.returns.numpy()
        names = ("obs", "actions", "rewards", "dones", "truncateds", "next_obs")
        for index, transition in enumerate(self._n_step_transitions):
            for name, value in zip(names, transition, strict=True):
                data[f"pending_{index}_{name}"] = value.numpy()
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: wp.context.Devicelike = None,
    ) -> BufferReplayFlashSAC:
        """Restore replay storage, normalization state, and pending n-step rows."""

        with np.load(Path(path), allow_pickle=False) as data:
            replay = cls(
                minimum_size=int(data["minimum_size"]),
                capacity=int(data["capacity"]),
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                batch_size=int(data["batch_size"]),
                n_step=int(data["n_step"]),
                gamma=float(data["gamma"]),
                normalize_rewards=bool(data["normalize_rewards"]),
                normalized_return_max=float(data["normalized_return_max"]),
                device=device,
            )
            replay.obs.assign(data["obs"])
            replay.actions.assign(data["actions"])
            replay.rewards.assign(data["rewards"])
            replay.dones.assign(data["dones"])
            replay.next_obs.assign(data["next_obs"])
            replay.size = int(data["size"])
            replay.position = int(data["position"])
            replay.reward_normalizer.running_mean.assign(data["return_running_mean"])
            replay.reward_normalizer.running_var.assign(data["return_running_var"])
            replay.reward_normalizer.running_count.assign(data["return_running_count"])
            replay.reward_normalizer.max_abs_return.assign(data["return_max_abs"])
            if bool(data["return_values_present"]):
                replay.reward_normalizer.returns = wp.array(
                    data["return_values"], dtype=wp.float32, device=replay.device
                )
            names = ("obs", "actions", "rewards", "dones", "truncateds", "next_obs")
            for index in range(int(data["pending_count"])):
                transition = tuple(
                    wp.array(data[f"pending_{index}_{name}"], dtype=wp.float32, device=replay.device) for name in names
                )
                replay._n_step_transitions.append(transition)
            return replay


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
        reference_backbone = hidden_layers is None
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
        if reference_backbone:
            flash_config.normalize_observations = False
            self.actor.net = NetworkFlashSAC(
                input_dim=obs_dim,
                hidden_dim=flash_config.actor_hidden_dim,
                num_blocks=flash_config.actor_num_blocks,
                output_dim=action_dim * 2,
                actor_heads=True,
                device=self.device,
                seed=seed,
            )
            critic_kwargs = {
                "input_dim": obs_dim + action_dim,
                "hidden_dim": flash_config.critic_hidden_dim,
                "num_blocks": flash_config.critic_num_blocks,
                "output_dim": flash_config.distributional_atoms if flash_config.distributional_critic else 1,
                "actor_heads": False,
                "device": self.device,
            }
            self.critic1 = NetworkFlashSAC(**critic_kwargs, seed=seed + 1)
            self.critic2 = NetworkFlashSAC(**critic_kwargs, seed=seed + 2)
            self.target_critic1 = NetworkFlashSAC(**critic_kwargs, seed=seed + 3)
            self.target_critic2 = NetworkFlashSAC(**critic_kwargs, seed=seed + 4)
            self.target_critic1.default_training = True
            self.target_critic2.default_training = True
            self.target_critic1.copy_from(self.critic1)
            self.target_critic2.copy_from(self.critic2)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=flash_config.actor_lr)
            self.critic1_optimizer = Adam(self.critic1.parameters(), lr=flash_config.critic_lr)
            self.critic2_optimizer = Adam(self.critic2.parameters(), lr=flash_config.critic_lr)
        self.config = flash_config
        self._replay_buffer: BufferReplayFlashSAC | None = None
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

    @property
    def replay_buffer(self) -> BufferReplayFlashSAC | None:
        """Replay buffer owned by the trainer, if transition collection has started."""

        return self._replay_buffer

    def initialize_replay_buffer(self) -> BufferReplayFlashSAC:
        """Allocate the configured replay buffer and return it.

        Allocation is lazy because the upstream one-million-row default is
        substantial for G1-sized observations. Calling :meth:`process_transition`
        also initializes the buffer on first use.
        """

        if self._replay_buffer is None:
            self._replay_buffer = BufferReplayFlashSAC(
                minimum_size=self.config.buffer_min_length,
                capacity=self.config.buffer_max_length,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                batch_size=self.config.sample_batch_size,
                n_step=self.config.n_step,
                gamma=self.config.gamma,
                normalize_rewards=self.config.normalize_rewards,
                normalized_return_max=self.config.normalized_return_max,
                device=self.device,
            )
        return self._replay_buffer

    def process_transition(
        self,
        obs: wp.array2d[wp.float32],
        actions: wp.array2d[wp.float32],
        rewards: wp.array[wp.float32],
        terminateds: wp.array[wp.float32],
        next_obs: wp.array2d[wp.float32],
        truncateds: wp.array[wp.float32] | None = None,
    ) -> None:
        """Add one vectorized environment transition to the owned replay buffer."""

        self.initialize_replay_buffer().add_batch(
            obs,
            actions,
            rewards,
            terminateds,
            next_obs,
            truncateds=truncateds,
        )

    def can_start_training(self) -> bool:
        """Return whether the owned replay buffer has completed warmup."""

        return self._replay_buffer is not None and self._replay_buffer.can_sample()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete trainer state except the potentially large replay buffer.

        Replay serialization is intentionally separate from trainer checkpoints.
        """

        data: dict[str, np.ndarray] = {
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "reference_backbone": np.asarray(isinstance(self.actor.net, NetworkFlashSAC), dtype=np.bool_),
            "actor_hidden_layers": np.asarray(self.actor.net.layer_sizes[1:-1], dtype=np.int64),
            "critic_hidden_layers": np.asarray(self.critic1.layer_sizes[1:-1], dtype=np.int64),
            "actor_log_std": self.actor.log_std.numpy(),
            "log_alpha": self.log_alpha.numpy(),
            "alpha": self._alpha.numpy(),
            "update_count": np.asarray(self._update_count, dtype=np.int64),
            "gradient_update_count": np.asarray(self._gradient_update_count, dtype=np.int64),
            "noise_repeat_count": np.asarray(self._noise_repeat_count, dtype=np.int64),
            "noise_repeat_steps": np.asarray(self._noise_repeat_steps, dtype=np.int64),
            "exploration_seed": np.asarray(self._exploration_seed, dtype=np.int64),
            "noise_rng_state": np.asarray(json.dumps(self._noise_rng.bit_generator.state)),
            "obs_mean": self._obs_mean.numpy(),
            "obs_m2": self._obs_m2.numpy(),
            "obs_count": self._obs_count.numpy(),
        }
        for key, value in asdict(self.config).items():
            none_key = f"config_{key}_is_none"
            if value is None:
                data[none_key] = np.asarray(True, dtype=np.bool_)
            else:
                data[none_key] = np.asarray(False, dtype=np.bool_)
                data[f"config_{key}"] = np.asarray(value)
        for prefix, network in (
            ("actor", self.actor.net),
            ("critic1", self.critic1),
            ("critic2", self.critic2),
            ("target_critic1", self.target_critic1),
            ("target_critic2", self.target_critic2),
        ):
            _pack_flash_sac_network(data, prefix, network)
        for prefix, optimizer in (
            ("actor_optimizer", self.actor_optimizer),
            ("critic1_optimizer", self.critic1_optimizer),
            ("critic2_optimizer", self.critic2_optimizer),
            ("alpha_optimizer", self.alpha_optimizer),
        ):
            _pack_optimizer(data, prefix, optimizer)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        config: ConfigFlashSAC | None = None,
        device: wp.context.Devicelike = None,
    ) -> TrainerFlashSAC:
        """Restore networks, targets, optimizers, temperature, and counters.

        Args:
            path: Input ``.npz`` checkpoint path.
            config: Optional optimizer configuration override.
            device: Warp device for restored arrays.

        Returns:
            Restored FlashSAC trainer without replay contents.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            saved_config = config or _config_from_flash_sac_checkpoint(data)
            actor_hidden = tuple(int(width) for width in data["actor_hidden_layers"])
            critic_hidden = tuple(int(width) for width in data["critic_hidden_layers"])
            reference_backbone = (
                bool(data["reference_backbone"])
                if "reference_backbone" in data
                else str(data["actor_network_type"].item()) == "flash_sac_reference"
            )
            hidden_layers = None if reference_backbone else actor_hidden
            trainer = cls(
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                hidden_layers=hidden_layers,
                config=saved_config,
                device=device,
                seed=int(data["seed"]),
            )
            if (
                tuple(trainer.actor.net.layer_sizes[1:-1]) != actor_hidden
                or tuple(trainer.critic1.layer_sizes[1:-1]) != critic_hidden
            ):
                raise ValueError("Checkpoint network architecture does not match configuration")
            for prefix, network in (
                ("actor", trainer.actor.net),
                ("critic1", trainer.critic1),
                ("critic2", trainer.critic2),
                ("target_critic1", trainer.target_critic1),
                ("target_critic2", trainer.target_critic2),
            ):
                _unpack_flash_sac_network(data, prefix, network)
            trainer.actor.log_std.assign(data["actor_log_std"])
            trainer.log_alpha.assign(data["log_alpha"])
            trainer._alpha.assign(data["alpha"])
            trainer._obs_mean.assign(data["obs_mean"])
            trainer._obs_m2.assign(data["obs_m2"])
            trainer._obs_count.assign(data["obs_count"])
            trainer._update_count = int(data["update_count"])
            trainer._gradient_update_count = int(data["gradient_update_count"])
            if "noise_rng_state" in data:
                trainer._noise_repeat_count = int(data["noise_repeat_count"])
                trainer._noise_repeat_steps = int(data["noise_repeat_steps"])
                trainer._exploration_seed = int(data["exploration_seed"])
                trainer._noise_rng.bit_generator.state = json.loads(str(data["noise_rng_state"].item()))
            for prefix, optimizer in (
                ("actor_optimizer", trainer.actor_optimizer),
                ("critic1_optimizer", trainer.critic1_optimizer),
                ("critic2_optimizer", trainer.critic2_optimizer),
                ("alpha_optimizer", trainer.alpha_optimizer),
            ):
                _unpack_optimizer(data, prefix, optimizer)
            return trainer

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

    def _update_alpha(self, batch: BatchSAC, *, seed: int) -> None:
        """Update temperature from the actor update's detached entropy sample."""

        del seed
        log_probs = getattr(self, "_actor_update_log_probs", None)
        if log_probs is None or int(log_probs.shape[0]) != batch.batch_size:
            raise RuntimeError("FlashSAC temperature update requires a preceding actor update")
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._alpha_loss], device=self.device)
        with wp.Tape() as tape:
            wp.launch(
                _flash_sac_alpha_loss_kernel,
                dim=batch.batch_size,
                inputs=[log_probs, self.log_alpha, batch.batch_size, self.target_entropy],
                outputs=[self._alpha_loss],
                device=self.device,
            )
        tape.backward(self._alpha_loss)
        self.alpha_optimizer.step()
        wp.launch(
            sac_refresh_alpha_kernel,
            dim=1,
            inputs=[self.log_alpha],
            outputs=[self._alpha],
            device=self.device,
        )

    def update(
        self,
        batch: BatchSAC | None = None,
        *,
        seed: int | None = None,
        read_stats: bool = True,
    ) -> StatsSACUpdate:
        """Sample owned replay when needed and update networks in upstream order."""

        if batch is None:
            if not self.can_start_training() or self._replay_buffer is None:
                raise RuntimeError("FlashSAC replay buffer has not completed warmup")
            sample_seed = self.seed + self._update_count * 9973 if seed is None else int(seed)
            batch = self._replay_buffer.sample(seed=sample_seed)
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


def train_flash_sac(
    env: EnvFlashSAC,
    trainer: TrainerFlashSAC,
    *,
    interaction_steps: int,
    updates_per_step: int = 1,
    seed: int = 0,
    reset_at_start: bool = True,
) -> list[StatsSACUpdate]:
    """Collect vectorized transitions and perform replay-driven FlashSAC updates.

    Args:
        env: Vectorized PhoenX environment.
        trainer: FlashSAC trainer matching the environment dimensions.
        interaction_steps: Number of environment steps to collect.
        updates_per_step: Maximum replay updates after each environment step.
        seed: Base stochastic action and update seed.
        reset_at_start: Whether to reset all environments before collection.

    Returns:
        Update statistics produced after replay warmup.
    """

    if interaction_steps <= 0:
        raise ValueError("interaction_steps must be positive")
    if updates_per_step < 0:
        raise ValueError("updates_per_step must be non-negative")
    if trainer.obs_dim != env.obs_dim or trainer.action_dim != env.action_dim:
        raise ValueError("FlashSAC trainer dimensions do not match environment")

    obs = env.reset() if reset_at_start else env.observe()
    stats: list[StatsSACUpdate] = []
    zero_truncateds = wp.zeros(env.world_count, dtype=wp.float32, device=env.device)
    for step in range(int(interaction_steps)):
        actions, _log_probs = trainer.act(obs, seed=int(seed) + step)
        next_obs, rewards, dones = env.step(actions)
        replay_next_obs = getattr(env, "step_next_obs", next_obs)
        truncateds = getattr(env, "step_truncateds", zero_truncateds)
        terminateds = getattr(env, "step_terminateds", dones)
        trainer.process_transition(
            obs,
            actions,
            rewards,
            terminateds,
            replay_next_obs,
            truncateds=truncateds,
        )
        for update_index in range(int(updates_per_step)):
            if not trainer.can_start_training():
                break
            stats.append(trainer.update(seed=int(seed) + step * max(updates_per_step, 1) + update_index))
        obs = next_obs
    return stats


def _config_from_flash_sac_checkpoint(data: np.lib.npyio.NpzFile) -> ConfigFlashSAC:
    """Restore scalar FlashSAC configuration fields from an archive."""

    kwargs: dict[str, object] = {}
    for config_field in fields(ConfigFlashSAC):
        none_key = f"config_{config_field.name}_is_none"
        if none_key in data and bool(data[none_key]):
            kwargs[config_field.name] = None
        elif f"config_{config_field.name}" in data:
            kwargs[config_field.name] = data[f"config_{config_field.name}"].item()
    return ConfigFlashSAC(**kwargs)


def _pack_flash_sac_network(
    data: dict[str, np.ndarray],
    prefix: str,
    network: object,
) -> None:
    if not isinstance(network, NetworkFlashSAC):
        _pack_policy_network(data, prefix, network)
        return
    arrays = network.state_arrays()
    data[f"{prefix}_network_type"] = np.asarray("flash_sac_reference")
    data[f"{prefix}_state_count"] = np.asarray(len(arrays), dtype=np.int64)
    for index, array in enumerate(arrays):
        data[f"{prefix}_state_{index}"] = array.numpy()


def _unpack_flash_sac_network(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    network: object,
) -> None:
    if not isinstance(network, NetworkFlashSAC):
        _unpack_policy_network(data, prefix, network)
        return
    if str(data[f"{prefix}_network_type"].item()) != "flash_sac_reference":
        raise ValueError(f"Checkpoint {prefix} network type does not match FlashSAC reference backbone")
    arrays = network.state_arrays()
    if int(data[f"{prefix}_state_count"]) != len(arrays):
        raise ValueError(f"Checkpoint {prefix} reference state count does not match trainer")
    for index, array in enumerate(arrays):
        array.assign(data[f"{prefix}_state_{index}"])


def save_flash_sac_checkpoint(trainer: TrainerFlashSAC, path: str | Path) -> None:
    """Save a FlashSAC trainer checkpoint without replay contents."""

    trainer.save_checkpoint(path)


def load_flash_sac_checkpoint(
    path: str | Path,
    *,
    config: ConfigFlashSAC | None = None,
    device: wp.context.Devicelike = None,
) -> TrainerFlashSAC:
    """Load a FlashSAC trainer checkpoint without replay contents."""

    return TrainerFlashSAC.load_checkpoint(
        path,
        config=config,
        device=device,
    )
