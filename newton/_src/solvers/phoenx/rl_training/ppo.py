# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .kernels import (
    PPO_LOG_STD_PARTIAL_BATCH,
    compute_gae_kernel,
    compute_puffer_vtrace_returns_kernel,
    compute_vtrace_returns_kernel,
    gather_trajectory_minibatch_kernel,
    gaussian_entropy_kernel,
    mirror_2d_kernel,
    mirrored_action_mse_grad_kernel,
    mirrored_action_mse_loss_kernel,
    normalize_from_stats_kernel,
    pack_ppo_update_stats_kernel,
    ppo_actor_loss_backward_kernel,
    ppo_actor_loss_kernel,
    ppo_lr_scale_kernel,
    reduce_ppo_log_std_grad_kernel,
    rollout_reward_done_success_sums_kernel,
    sample_trajectory_env_ids_kernel,
    sample_trajectory_env_ids_seed_counter_kernel,
    scatter_trajectory_ratios_kernel,
    scatter_trajectory_values_kernel,
    seed_counter_increment_kernel,
    sum_and_sumsq_kernel,
    trajectory_priority_kernel,
    trajectory_priority_weight_kernel,
    value_column_loss_grad_kernel,
    value_column_symmetry_loss_grad_kernel,
    value_loss_grad_kernel,
    value_loss_kernel,
    value_symmetry_loss_grad_kernel,
    value_symmetry_loss_kernel,
    weight_trajectory_advantages_kernel,
    zero_ppo_actor_stats_kernel,
    zero_ppo_loss_stats_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, PufferMinGRUNet, WarpMLP
from .optim import Adam, Muon


@dataclass(frozen=True)
class MirrorMapPPO:
    """Observation/action reflection map for PPO symmetry regularization.

    Args:
        obs_src: Source observation index for each mirrored observation column.
        obs_sign: Sign multiplier for each mirrored observation column.
        action_src: Source action index for each mirrored action column.
        action_sign: Sign multiplier for each mirrored action column.
    """

    obs_src: tuple[int, ...]
    obs_sign: tuple[float, ...]
    action_src: tuple[int, ...]
    action_sign: tuple[float, ...]


@dataclass
class ConfigPPO:
    """Configuration for :class:`TrainerPPO`.

    Args:
        gamma: Discount factor.
        gae_lambda: Generalized advantage estimation trace decay.
        clip_ratio: PPO policy-ratio clipping threshold.
        entropy_coeff: Entropy bonus coefficient.
        value_loss_coeff: Value loss coefficient.
        value_clip_range: PPO value-function clip range against rollout
            value predictions. A value less than or equal to zero disables
            clipping.
        actor_lr: Actor optimizer learning rate.
        critic_lr: Critic optimizer learning rate.
        anneal_lr: Whether to cosine-anneal optimizer learning rates.
        lr_anneal_timesteps: Environment samples over which the learning rate
            anneals toward ``min_lr_ratio``. A value less than or equal to zero
            disables annealing.
        min_lr_ratio: Final learning-rate ratio for cosine annealing.
        optimizer: Optimizer implementation, either ``"adam"`` or ``"muon"``.
        optimizer_eps: Numerical stabilizer for the selected optimizer.
        optimizer_weight_decay: Decoupled weight decay for the selected optimizer.
        muon_momentum: Momentum used when ``optimizer`` is ``"muon"``.
        train_epochs: Full-buffer optimization epochs per rollout when replay
            minibatches are disabled.
        minibatch_size: Optional trajectory-minibatch size. A value less than or
            equal to zero disables minibatch replay.
        replay_ratio: Number of sampled transition updates per collected
            transition when trajectory minibatches are enabled.
        priority_alpha: Trajectory priority exponent. A value less than or
            equal to zero uses uniform trajectory sampling.
        priority_beta: Importance-correction exponent for priority replay. A
            value less than or equal to zero disables the correction.
        manual_actor_backward: Use hand-written kernels for the actor MLP and
            Gaussian PPO loss backward pass. This avoids Warp Tape for the
            actor update path.
        manual_critic_backward: Use hand-written kernels for the critic MLP and
            value loss backward pass. This avoids Warp Tape for the critic
            update path.
        manual_mlp_weight_grad_dtype: Accumulator input dtype for manual CUDA
            MLP backward tile matmuls. Supports ``"float32"`` and
            ``"bfloat16"``.
        manual_mlp_forward_dtype: Input dtype for manual CUDA hidden-layer
            forward tile matmul. Supports ``"float32"`` and ``"bfloat16"``.
        vtrace_rho_clip: V-trace policy-ratio clip for replayed trajectories.
            A value less than or equal to zero disables V-trace recomputation.
        vtrace_c_clip: V-trace trace-ratio clip for replayed trajectories.
            A value less than or equal to zero disables V-trace recomputation.
        normalize_advantages: Whether to normalize advantages in-place before
            updating. With minibatch replay this normalizes each sampled
            minibatch.
        reward_clip: Absolute reward clamp used before advantage/return
            computation. A value less than or equal to zero disables clipping.
        puffer_vtrace_advantage: Use PufferLib's shifted V-trace scan for
            replayed trajectories. This leaves the final horizon advantage zero
            and matches nanoG1's CUDA trainer.
        max_grad_norm: Global gradient-norm clipping threshold for actor and
            critic optimizers. A value less than or equal to zero disables clipping.
        mirror_loss_coeff: Coefficient for optional mirror-symmetry MSE on
            policy means and value predictions. Requires a mirror map.
        shared_value_network: Use one actor/value policy network with the
            final output column as the value estimate. This matches PufferLib's
            fused decoder layout and uses actor_lr for the shared network.
        policy_network: Policy backbone, either ``"mlp"`` or
            ``"puffer_mingru"``. The recurrent option follows PufferLib's
            bias-free encoder, MinGRU stack, and fused decoder.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 1.0e-3
    value_loss_coeff: float = 1.0
    value_clip_range: float = 0.0
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    anneal_lr: bool = False
    lr_anneal_timesteps: int = 0
    min_lr_ratio: float = 0.0
    optimizer: str = "adam"
    optimizer_eps: float = 1.0e-8
    optimizer_weight_decay: float = 0.0
    muon_momentum: float = 0.9
    train_epochs: int = 4
    minibatch_size: int = 0
    replay_ratio: float = 0.0
    priority_alpha: float = 0.0
    priority_beta: float = 0.0
    manual_actor_backward: bool = False
    manual_critic_backward: bool = False
    manual_mlp_weight_grad_dtype: str = "float32"
    manual_mlp_forward_dtype: str = "float32"
    vtrace_rho_clip: float = 0.0
    vtrace_c_clip: float = 0.0
    normalize_advantages: bool = True
    reward_clip: float = 0.0
    puffer_vtrace_advantage: bool = False
    max_grad_norm: float = 0.0
    mirror_loss_coeff: float = 0.0
    shared_value_network: bool = False
    policy_network: str = "mlp"


@dataclass
class StatsPPOUpdate:
    """Scalar diagnostics from a PPO update."""

    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float


class BatchPPO:
    """Sampled transition batch used by :class:`TrainerPPO`."""

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
        self.advantages = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.old_values = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.ratios = wp.ones(count, dtype=wp.float32, device=self.device)
        self.priority_weights = wp.ones(self.num_envs, dtype=wp.float32, device=self.device)
        self._advantage_stats = wp.zeros(2, dtype=wp.float32, device=self.device)

    @property
    def num_samples(self) -> int:
        """Number of transition rows in the batch."""

        return self.num_steps * self.num_envs

    def normalize_advantages(self, eps: float = 1.0e-8) -> None:
        """Normalize advantages in place."""

        _normalize_advantages(self.advantages, self.num_samples, self.device, eps=eps, stats=self._advantage_stats)


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
        self.old_values = self.values
        self.advantages = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.returns = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.ratios = wp.ones(count, dtype=wp.float32, device=self.device)
        self._advantage_stats = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._metric_sums = wp.zeros(3, dtype=wp.float32, device=self.device)
        self._metric_sums_host = wp.empty(3, dtype=wp.float32, device="cpu", pinned=self.device.is_cuda)

    @property
    def num_samples(self) -> int:
        """Number of transition rows in the buffer."""

        return self.num_steps * self.num_envs

    def compute_returns(self, *, gamma: float, gae_lambda: float, reward_clip: float = 0.0) -> None:
        """Compute GAE advantages and returns in place."""

        self.ratios.fill_(1.0)
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
                float(reward_clip),
            ],
            outputs=[self.advantages, self.returns],
            device=self.device,
        )

    def normalize_advantages(self, eps: float = 1.0e-8) -> None:
        """Normalize advantages in place."""

        _normalize_advantages(self.advantages, self.num_samples, self.device, eps=eps, stats=self._advantage_stats)

    def compute_reward_done_success_sums(self) -> wp.array[wp.float32]:
        """Compute rollout reward, done, and success sums in preallocated storage."""

        self._metric_sums.zero_()
        wp.launch(
            rollout_reward_done_success_sums_kernel,
            dim=self.num_samples,
            inputs=[self.rewards, self.dones, self.successes, self.num_samples],
            outputs=[self._metric_sums],
            device=self.device,
        )
        return self._metric_sums

    def copy_reward_done_success_sums_to_host(self) -> wp.array[wp.float32]:
        """Copy compact rollout metric sums into pinned host storage."""

        sums = self.compute_reward_done_success_sums()
        wp.copy(self._metric_sums_host, sums, count=3)
        return self._metric_sums_host

    def reward_done_success_means(self) -> tuple[float, float, float]:
        """Return rollout reward, done, and success means."""

        sums_host = self.copy_reward_done_success_sums_to_host()
        if self.device.is_cuda:
            # Pinned host arrays expose memory directly; wait for the async graph copy.
            wp.synchronize_device(self.device)
        sums = sums_host.numpy()
        inv_count = 1.0 / float(self.num_samples)
        return float(sums[0]) * inv_count, float(sums[1]) * inv_count, float(sums[2]) * inv_count

    def compute_vtrace_returns(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        rho_clip: float,
        c_clip: float,
        reward_clip: float = 0.0,
    ) -> None:
        """Compute V-trace advantages and returns in place."""

        wp.launch(
            compute_vtrace_returns_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.dones,
                self.values,
                self.ratios,
                self.num_steps,
                self.num_envs,
                float(gamma),
                float(gae_lambda),
                float(rho_clip),
                float(c_clip),
                float(reward_clip),
            ],
            outputs=[self.advantages, self.returns],
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
        mirror_map: Optional observation/action reflection map used when
            ``config.mirror_loss_coeff`` is positive.
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
        mirror_map: MirrorMapPPO | None = None,
    ):
        self.config = config or ConfigPPO()
        self.seed = int(seed)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.activation = activation
        self.squash_actions = bool(squash_actions)
        self.log_std_init = float(log_std_init)
        self.device = wp.get_device(device)
        self.shared_value_network = bool(self.config.shared_value_network)
        self.policy_network = _normalize_policy_network(self.config.policy_network)
        if self.shared_value_network and not (self.config.manual_actor_backward and self.config.manual_critic_backward):
            raise ValueError("shared_value_network currently requires manual actor and critic backward")
        if self.policy_network != "mlp" and not self.shared_value_network:
            raise ValueError("recurrent PPO policy backbones currently require shared_value_network")
        self.actor = GaussianActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.hidden_layers,
            activation=activation,
            state_dependent_std=False,
            log_std_init=log_std_init,
            squash=squash_actions,
            device=self.device,
            seed=seed,
            manual_weight_grad_dtype=self.config.manual_mlp_weight_grad_dtype,
            manual_forward_dtype=self.config.manual_mlp_forward_dtype,
        )
        if self.shared_value_network:
            if self.policy_network == "puffer_mingru":
                hidden_size, num_layers = _mingru_shape_from_hidden_layers(self.hidden_layers)
                self.actor.net = PufferMinGRUNet(
                    input_dim=self.obs_dim,
                    hidden_size=hidden_size,
                    output_dim=self.action_dim + 1,
                    num_layers=num_layers,
                    device=self.device,
                    seed=seed,
                )
            else:
                self.actor.net = WarpMLP(
                    (self.obs_dim, *self.hidden_layers, self.action_dim + 1),
                    activation=activation,
                    output_activation="linear",
                    device=self.device,
                    seed=seed,
                    manual_weight_grad_dtype=self.config.manual_mlp_weight_grad_dtype,
                    manual_forward_dtype=self.config.manual_mlp_forward_dtype,
                )
            self.critic: WarpMLP | None = None
            self.critic_optimizer: Adam | Muon | None = None
        else:
            self.critic = WarpMLP(
                (self.obs_dim, *self.hidden_layers, 1),
                activation=activation,
                output_activation="linear",
                device=self.device,
                seed=seed + 1,
                manual_weight_grad_dtype=self.config.manual_mlp_weight_grad_dtype,
                manual_forward_dtype=self.config.manual_mlp_forward_dtype,
            )
            self.critic_optimizer = _make_optimizer(self.critic.parameters(), self.config, lr=self.config.critic_lr)
        self.actor_optimizer = _make_optimizer(self.actor.parameters(), self.config, lr=self.config.actor_lr)
        self._iteration = 0
        self._iteration_counter = wp.array([0], dtype=wp.int32, device=self.device)
        self.mirror_map: MirrorMapPPO | None = None
        self._mirror_obs_src: wp.array[wp.int32] | None = None
        self._mirror_obs_sign: wp.array[wp.float32] | None = None
        self._mirror_action_src: wp.array[wp.int32] | None = None
        self._mirror_action_sign: wp.array[wp.float32] | None = None
        self._mirror_obs: wp.array2d[wp.float32] | None = None
        self._minibatch: BatchPPO | None = None
        self._minibatch_env_ids: wp.array[wp.int32] | None = None
        self._trajectory_priorities: wp.array[wp.float32] | None = None
        self._trajectory_priority_weights: wp.array[wp.float32] | None = None
        self._trajectory_priority_total: wp.array[wp.float32] | None = None
        self._actor_policy_out_grad: wp.array2d[wp.float32] | None = None
        self._critic_value_grad: wp.array2d[wp.float32] | None = None
        self._actor_log_std_grad = wp.zeros_like(self.actor.log_std, requires_grad=False)
        self._actor_log_std_grad_partials: wp.array2d[wp.float32] | None = None
        self._actor_log_std_grad_partial_count = 0
        if mirror_map is not None:
            self.set_mirror_map(mirror_map)

        self._policy_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._value_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._approx_kl = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._clip_fraction = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._update_stats = wp.zeros(4, dtype=wp.float32, device=self.device)
        self._update_stats_host = wp.empty(4, dtype=wp.float32, device="cpu", pinned=self.device.is_cuda)

    @property
    def iteration(self) -> int:
        """Number of PPO rollout-update iterations already applied."""

        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self._iteration = max(int(value), 0)
        if hasattr(self, "_iteration_counter"):
            self._iteration_counter.assign(np.asarray([self._iteration], dtype=np.int32))

    def set_mirror_map(self, mirror_map: MirrorMapPPO | None) -> None:
        """Set or clear the optional PPO symmetry map."""

        if mirror_map is None:
            self.mirror_map = None
            self._mirror_obs_src = None
            self._mirror_obs_sign = None
            self._mirror_action_src = None
            self._mirror_action_sign = None
            self._mirror_obs = None
            return
        _validate_mirror_map(mirror_map, self.obs_dim, self.action_dim)
        self.mirror_map = mirror_map
        self._mirror_obs_src = wp.array(
            np.asarray(mirror_map.obs_src, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._mirror_obs_sign = wp.array(
            np.asarray(mirror_map.obs_sign, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._mirror_action_src = wp.array(
            np.asarray(mirror_map.action_src, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._mirror_action_sign = wp.array(
            np.asarray(mirror_map.action_sign, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self._mirror_obs = None

    def save_checkpoint(self, path: str | Path, *, iteration: int | None = None) -> None:
        """Save network parameters and optimizer state to a NumPy archive.

        Args:
            path: Output ``.npz`` path.
            iteration: Optional training iteration stored as metadata.
        """

        data: dict[str, np.ndarray] = {
            "obs_dim": np.asarray(self.obs_dim, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "hidden_layers": np.asarray(self.hidden_layers, dtype=np.int64),
            "activation": np.asarray(self.activation),
            "policy_network": np.asarray(self.policy_network),
            "squash_actions": np.asarray(int(self.squash_actions), dtype=np.int64),
            "log_std_init": np.asarray(self.log_std_init, dtype=np.float32),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "iteration": np.asarray(self.iteration if iteration is None else int(iteration), dtype=np.int64),
        }
        for key, value in asdict(self.config).items():
            data[f"config_{key}"] = np.asarray(value)
        _pack_policy_network(data, "actor", self.actor.net)
        data["actor_log_std"] = self.actor.log_std.numpy()
        _pack_optimizer(data, "actor_optimizer", self.actor_optimizer)
        if not self.shared_value_network:
            if self.critic is None or self.critic_optimizer is None:
                raise RuntimeError("separate critic state was not initialized")
            _pack_policy_network(data, "critic", self.critic)
            _pack_optimizer(data, "critic_optimizer", self.critic_optimizer)
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        config: ConfigPPO | None = None,
        device: wp.context.Devicelike = None,
    ) -> TrainerPPO:
        """Load a PPO trainer from a checkpoint archive.

        Args:
            path: Input ``.npz`` path.
            config: Optional PPO config overriding the saved optimizer settings.
            device: Warp device for restored arrays.

        Returns:
            Restored trainer.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            saved_config = config or _config_from_checkpoint(data)
            trainer = cls(
                obs_dim=int(data["obs_dim"]),
                action_dim=int(data["action_dim"]),
                hidden_layers=tuple(int(width) for width in data["hidden_layers"]),
                config=saved_config,
                device=device,
                seed=int(data["seed"]) if "seed" in data else 0,
                squash_actions=bool(int(data["squash_actions"])),
                activation=str(data["activation"].item()),
                log_std_init=float(data["log_std_init"]),
            )
            _unpack_policy_network(data, "actor", trainer.actor.net)
            trainer.actor.log_std.assign(data["actor_log_std"])
            _unpack_optimizer(data, "actor_optimizer", trainer.actor_optimizer)
            if not trainer.shared_value_network:
                if trainer.critic is None or trainer.critic_optimizer is None:
                    raise RuntimeError("separate critic state was not initialized")
                _unpack_policy_network(data, "critic", trainer.critic)
                _unpack_optimizer(data, "critic_optimizer", trainer.critic_optimizer)
            if "iteration" in data:
                trainer.iteration = max(int(data["iteration"]), 0)
        return trainer

    @property
    def value_column(self) -> int:
        """Column in the returned value tensor containing value predictions."""

        return self.action_dim if self.shared_value_network else 0

    def value_reuse(self, obs: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        """Evaluate values into persistent no-grad buffers."""

        if self.shared_value_network:
            return self.actor.net.forward_reuse(obs)
        if self.critic is None:
            raise RuntimeError("separate critic state was not initialized")
        return self.critic.forward_reuse(obs)

    def reset_rollout_state(self, dones: wp.array[wp.float32] | None = None) -> None:
        """Reset recurrent rollout state when the selected policy has one."""

        reset_state = getattr(self.actor.net, "reset_state", None)
        if reset_state is not None:
            reset_state(dones)

    def _set_update_sequence_shape(self, buffer: BufferRollout | BatchPPO) -> None:
        set_sequence_shape = getattr(self.actor.net, "set_sequence_shape", None)
        if set_sequence_shape is not None:
            set_sequence_shape(buffer.num_steps, buffer.num_envs)

    def _policy_update_reuse(
        self, obs: wp.array2d[wp.float32], buffer: BufferRollout | BatchPPO
    ) -> wp.array2d[wp.float32]:
        forward_sequence_reuse = getattr(self.actor.net, "forward_sequence_reuse", None)
        if forward_sequence_reuse is not None:
            return forward_sequence_reuse(obs, num_steps=buffer.num_steps, num_envs=buffer.num_envs)
        return self.actor.net.forward_reuse(obs)

    def _value_reuse_for_update(self, buffer: BufferRollout | BatchPPO) -> wp.array2d[wp.float32]:
        if self.shared_value_network:
            return self._policy_update_reuse(buffer.obs, buffer)
        return self.value_reuse(buffer.obs)

    def act(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions and evaluate values for rollout collection.

        Args:
            obs: Observation batch.
            seed: Sampling seed.
            deterministic: Use deterministic policy means.

        Returns:
            Tuple ``(actions, log_probs, values)``.
        """

        actions, log_probs, policy_out = self.actor.sample(
            obs, seed=seed, deterministic=deterministic, requires_grad=False
        )
        if self.shared_value_network:
            return actions, log_probs, policy_out
        if self.critic is None:
            raise RuntimeError("separate critic state was not initialized")
        values = self.critic.forward(obs, requires_grad=False)
        return actions, log_probs, values

    def act_reuse(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed: int,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions and values into persistent no-grad buffers."""

        actions, log_probs, policy_out = self.actor.sample_reuse(obs, seed=seed, deterministic=deterministic)
        if self.shared_value_network:
            return actions, log_probs, policy_out
        if self.critic is None:
            raise RuntimeError("separate critic state was not initialized")
        values = self.critic.forward_reuse(obs)
        return actions, log_probs, values

    def act_reuse_seed_counter(
        self,
        obs: wp.array2d[wp.float32],
        *,
        seed_counter: wp.array[wp.int32],
        seed_offset: int = 0,
        deterministic: bool = False,
    ) -> tuple[wp.array2d[wp.float32], wp.array[wp.float32], wp.array2d[wp.float32]]:
        """Sample actions and values using a graph-replay-safe device seed counter."""

        actions, log_probs, policy_out = self.actor.sample_reuse_seed_counter(
            obs, seed_counter=seed_counter, seed_offset=int(seed_offset), deterministic=deterministic
        )
        if self.shared_value_network:
            return actions, log_probs, policy_out
        if self.critic is None:
            raise RuntimeError("separate critic state was not initialized")
        values = self.critic.forward_reuse(obs)
        return actions, log_probs, values

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve reusable trainer buffers for at least ``batch_size`` rows."""

        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        self.actor.reserve_reuse_buffers(rows)
        self._ensure_actor_backward_buffers(rows, self.actor.net.output_dim)
        self._ensure_actor_log_std_grad_partials(rows)
        if self.config.mirror_loss_coeff > 0.0:
            self._ensure_mirror_obs(rows)
        if self.critic is not None:
            self.critic.reserve_buffers(rows)
            self._ensure_critic_backward_buffers(rows)

    def reserve_update_buffers(self, buffer: BufferRollout) -> None:
        """Reserve all reusable buffers needed by :meth:`update` for ``buffer``."""

        if buffer.obs_dim != self.obs_dim or buffer.action_dim != self.action_dim:
            raise ValueError("BufferRollout dimensions do not match trainer dimensions")
        update_rows = buffer.num_samples
        if self._uses_minibatch_replay(buffer):
            minibatch_size = int(self.config.minibatch_size)
            if minibatch_size % buffer.num_steps != 0:
                raise ValueError("PPO trajectory minibatch_size must be a multiple of rollout steps")
            segment_count = minibatch_size // buffer.num_steps
            if segment_count <= 0:
                raise ValueError("PPO trajectory minibatch_size is smaller than one rollout trajectory")
            batch = self._ensure_minibatch(buffer, segment_count)
            self._ensure_trajectory_sampling_buffers(buffer.num_envs)
            update_rows = batch.num_samples
        self.reserve_buffers(max(buffer.num_envs, update_rows))

    def copy_update_stats_to_host(self) -> wp.array[wp.float32]:
        """Copy compact PPO update stats into pinned host storage."""

        wp.launch(
            pack_ppo_update_stats_kernel,
            dim=1,
            inputs=[self._policy_loss, self._value_loss, self._approx_kl, self._clip_fraction],
            outputs=[self._update_stats],
            device=self.device,
        )
        wp.copy(self._update_stats_host, self._update_stats, count=4)
        return self._update_stats_host

    def _read_update_stat_values(self) -> tuple[float, float, float, float]:
        stats_host = self.copy_update_stats_to_host()
        if self.device.is_cuda:
            # Pinned host arrays expose memory directly; wait for the async graph copy.
            wp.synchronize_device(self.device)
        stats = stats_host.numpy()
        return float(stats[0]), float(stats[1]), float(stats[2]), float(stats[3])

    def _read_update_stats(self) -> StatsPPOUpdate:
        policy_loss, value_loss, approx_kl, clip_fraction = self._read_update_stat_values()
        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def update(self, buffer: BufferRollout, *, read_stats: bool = True) -> StatsPPOUpdate:
        """Update actor and critic from a finished rollout buffer."""

        return self._update_impl(buffer, read_stats=read_stats, seed_counter=None)

    def update_seed_counter(
        self, buffer: BufferRollout, *, seed_counter: wp.array[wp.int32], read_stats: bool = True
    ) -> StatsPPOUpdate:
        """Update from a rollout buffer using device-side replay seeds."""

        return self._update_impl(buffer, read_stats=read_stats, seed_counter=seed_counter)

    def _update_impl(
        self, buffer: BufferRollout, *, read_stats: bool = True, seed_counter: wp.array[wp.int32] | None
    ) -> StatsPPOUpdate:
        if buffer.obs_dim != self.obs_dim or buffer.action_dim != self.action_dim:
            raise ValueError("BufferRollout dimensions do not match trainer dimensions")
        self._apply_lr_schedule(buffer.num_samples)
        if self._uses_minibatch_replay(buffer):
            return self._update_minibatches(buffer, read_stats=read_stats, seed_counter=seed_counter)

        if self.config.normalize_advantages:
            buffer.normalize_advantages()

        policy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clip_fraction = 0.0
        train_epochs = int(self.config.train_epochs)
        for epoch in range(train_epochs):
            epoch_read_stats = read_stats and epoch == train_epochs - 1
            if self.shared_value_network:
                stats = self._update_shared_manual(buffer, read_stats=epoch_read_stats)
                policy_loss = stats.policy_loss
                value_loss = stats.value_loss
                approx_kl = stats.approx_kl
                clip_fraction = stats.clip_fraction
            else:
                policy_loss, approx_kl, clip_fraction = self._update_actor(buffer, read_stats=epoch_read_stats)
                value_loss = self._update_critic(buffer, read_stats=epoch_read_stats)
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._iteration_counter, 1],
            device=self.device,
        )
        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _apply_lr_schedule(self, num_samples: int) -> None:
        critic_lr_scale = (
            self.critic_optimizer.lr_scale if self.critic_optimizer is not None else self.actor_optimizer.lr_scale
        )
        wp.launch(
            ppo_lr_scale_kernel,
            dim=1,
            inputs=[
                self._iteration_counter,
                int(num_samples),
                int(bool(self.config.anneal_lr)),
                int(self.config.lr_anneal_timesteps),
                float(self.config.min_lr_ratio),
            ],
            outputs=[self.actor_optimizer.lr_scale, critic_lr_scale],
            device=self.device,
        )

    def _uses_minibatch_replay(self, buffer: BufferRollout) -> bool:
        minibatch_size = int(self.config.minibatch_size)
        return minibatch_size > 0 and float(self.config.replay_ratio) > 0.0 and minibatch_size < buffer.num_samples

    def _update_minibatches(
        self, buffer: BufferRollout, *, read_stats: bool = True, seed_counter: wp.array[wp.int32] | None = None
    ) -> StatsPPOUpdate:
        minibatch_size = int(self.config.minibatch_size)
        if minibatch_size % buffer.num_steps != 0:
            raise ValueError("PPO trajectory minibatch_size must be a multiple of rollout steps")
        segment_count = minibatch_size // buffer.num_steps
        if segment_count <= 0:
            raise ValueError("PPO trajectory minibatch_size is smaller than one rollout trajectory")
        batch = self._ensure_minibatch(buffer, segment_count)
        num_minibatches = max(
            1, int(float(self.config.replay_ratio) * float(buffer.num_samples) / float(minibatch_size))
        )

        use_vtrace = self._uses_vtrace_replay()
        use_priority = False if use_vtrace else self._prepare_trajectory_priority_weights(buffer)
        max_cols = max(self.obs_dim, self.action_dim, 1)
        policy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clip_fraction = 0.0
        for minibatch_id in range(num_minibatches):
            minibatch_read_stats = read_stats and minibatch_id == num_minibatches - 1
            if use_vtrace:
                self._compute_vtrace_returns(buffer)
                use_priority = self._prepare_trajectory_priority_weights(buffer)
            if seed_counter is None:
                self._sample_minibatch_env_ids(
                    buffer,
                    batch,
                    seed=self.seed + 1000003 * self.iteration + minibatch_id,
                    use_priority=use_priority,
                )
            else:
                self._sample_minibatch_env_ids_seed_counter(
                    buffer,
                    batch,
                    seed_counter=seed_counter,
                    seed_offset=minibatch_id,
                    use_priority=use_priority,
                )
            wp.launch(
                gather_trajectory_minibatch_kernel,
                dim=(batch.num_samples, max_cols),
                inputs=[
                    self._minibatch_env_ids,
                    buffer.num_envs,
                    segment_count,
                    self.obs_dim,
                    self.action_dim,
                    buffer.obs,
                    buffer.actions,
                    buffer.old_log_probs,
                    buffer.advantages,
                    buffer.returns,
                    buffer.old_values,
                ],
                outputs=[
                    batch.obs,
                    batch.actions,
                    batch.old_log_probs,
                    batch.advantages,
                    batch.returns,
                    batch.old_values,
                ],
                device=self.device,
            )
            if self.config.normalize_advantages:
                batch.normalize_advantages()
            self._weight_minibatch_advantages(batch, use_priority)
            if self.shared_value_network:
                stats = self._update_shared_manual(batch, read_stats=minibatch_read_stats)
                policy_loss = stats.policy_loss
                value_loss = stats.value_loss
                approx_kl = stats.approx_kl
                clip_fraction = stats.clip_fraction
            else:
                policy_loss, approx_kl, clip_fraction = self._update_actor(batch, read_stats=minibatch_read_stats)
            if use_vtrace:
                self._scatter_minibatch_ratios(buffer, batch, segment_count)
            if not self.shared_value_network:
                value_loss = self._update_critic(batch, read_stats=minibatch_read_stats)
            if use_vtrace:
                self._scatter_minibatch_values(buffer, batch, segment_count)

        if seed_counter is not None:
            wp.launch(
                seed_counter_increment_kernel,
                dim=1,
                inputs=[seed_counter, 1000003],
                device=self.device,
            )
        wp.launch(
            seed_counter_increment_kernel,
            dim=1,
            inputs=[self._iteration_counter, 1],
            device=self.device,
        )

        return StatsPPOUpdate(
            policy_loss=policy_loss,
            value_loss=value_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _uses_vtrace_replay(self) -> bool:
        return float(self.config.vtrace_rho_clip) > 0.0 and float(self.config.vtrace_c_clip) > 0.0

    def _compute_vtrace_returns(self, buffer: BufferRollout) -> None:
        if self.config.puffer_vtrace_advantage:
            wp.launch(
                compute_puffer_vtrace_returns_kernel,
                dim=buffer.num_envs,
                inputs=[
                    buffer.rewards,
                    buffer.dones,
                    buffer.values,
                    buffer.ratios,
                    buffer.num_steps,
                    buffer.num_envs,
                    float(self.config.gamma),
                    float(self.config.gae_lambda),
                    float(self.config.vtrace_rho_clip),
                    float(self.config.vtrace_c_clip),
                    float(self.config.reward_clip),
                ],
                outputs=[buffer.advantages, buffer.returns],
                device=self.device,
            )
            return
        buffer.compute_vtrace_returns(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            rho_clip=self.config.vtrace_rho_clip,
            c_clip=self.config.vtrace_c_clip,
            reward_clip=self.config.reward_clip,
        )

    def _weight_minibatch_advantages(self, batch: BatchPPO, use_priority: bool) -> None:
        if float(self.config.priority_beta) <= 0.0 or not use_priority:
            return
        wp.launch(
            weight_trajectory_advantages_kernel,
            dim=batch.num_samples,
            inputs=[batch.priority_weights, batch.num_envs],
            outputs=[batch.advantages],
            device=self.device,
        )

    def _ensure_trajectory_sampling_buffers(self, num_envs: int) -> None:
        env_count = int(num_envs)
        if env_count <= 0:
            raise ValueError("num_envs must be positive")
        if self._trajectory_priorities is None or int(self._trajectory_priorities.shape[0]) != env_count:
            self._trajectory_priorities = wp.zeros(env_count, dtype=wp.float32, device=self.device)
            self._trajectory_priority_weights = wp.zeros(env_count, dtype=wp.float32, device=self.device)
            self._trajectory_priority_total = wp.zeros(1, dtype=wp.float32, device=self.device)

    def _prepare_trajectory_priority_weights(self, buffer: BufferRollout) -> bool:
        priority_alpha = float(self.config.priority_alpha)
        self._ensure_trajectory_sampling_buffers(buffer.num_envs)
        if priority_alpha <= 0.0:
            return False
        if self._trajectory_priorities is None or self._trajectory_priority_weights is None:
            raise RuntimeError("trajectory priority buffers were not initialized")
        if self._trajectory_priority_total is None:
            raise RuntimeError("trajectory priority total buffer was not initialized")
        self._trajectory_priority_total.zero_()
        wp.launch(
            trajectory_priority_kernel,
            dim=buffer.num_envs,
            inputs=[buffer.advantages, buffer.num_steps, buffer.num_envs],
            outputs=[self._trajectory_priorities],
            device=self.device,
        )
        wp.launch(
            trajectory_priority_weight_kernel,
            dim=buffer.num_envs,
            inputs=[self._trajectory_priorities, priority_alpha],
            outputs=[self._trajectory_priority_weights, self._trajectory_priority_total],
            device=self.device,
        )
        return True

    def _sample_minibatch_env_ids(
        self, buffer: BufferRollout, batch: BatchPPO, *, seed: int, use_priority: bool
    ) -> None:
        self._ensure_trajectory_sampling_buffers(buffer.num_envs)
        if self._minibatch_env_ids is None:
            raise RuntimeError("minibatch env id buffer was not initialized")
        if self._trajectory_priority_weights is None or self._trajectory_priority_total is None:
            raise RuntimeError("trajectory sampling buffers were not initialized")
        wp.launch(
            sample_trajectory_env_ids_kernel,
            dim=batch.num_envs,
            inputs=[
                self._trajectory_priority_weights,
                self._trajectory_priority_total,
                buffer.num_envs,
                int(seed) & 0x7FFFFFFF,
                float(self.config.priority_beta),
                int(use_priority),
            ],
            outputs=[self._minibatch_env_ids, batch.priority_weights],
            device=self.device,
        )

    def _sample_minibatch_env_ids_seed_counter(
        self,
        buffer: BufferRollout,
        batch: BatchPPO,
        *,
        seed_counter: wp.array[wp.int32],
        seed_offset: int,
        use_priority: bool,
    ) -> None:
        self._ensure_trajectory_sampling_buffers(buffer.num_envs)
        if self._minibatch_env_ids is None:
            raise RuntimeError("minibatch env id buffer was not initialized")
        if self._trajectory_priority_weights is None or self._trajectory_priority_total is None:
            raise RuntimeError("trajectory sampling buffers were not initialized")
        wp.launch(
            sample_trajectory_env_ids_seed_counter_kernel,
            dim=batch.num_envs,
            inputs=[
                self._trajectory_priority_weights,
                self._trajectory_priority_total,
                buffer.num_envs,
                seed_counter,
                int(seed_offset),
                float(self.config.priority_beta),
                int(use_priority),
            ],
            outputs=[self._minibatch_env_ids, batch.priority_weights],
            device=self.device,
        )

    def _scatter_minibatch_ratios(self, buffer: BufferRollout, batch: BatchPPO, segment_count: int) -> None:
        wp.launch(
            scatter_trajectory_ratios_kernel,
            dim=batch.num_samples,
            inputs=[self._minibatch_env_ids, buffer.num_envs, segment_count, batch.ratios],
            outputs=[buffer.ratios],
            device=self.device,
        )

    def _scatter_minibatch_values(self, buffer: BufferRollout, batch: BatchPPO, segment_count: int) -> None:
        values = self._value_reuse_for_update(batch)
        wp.launch(
            scatter_trajectory_values_kernel,
            dim=batch.num_samples,
            inputs=[self._minibatch_env_ids, buffer.num_envs, segment_count, values, self.value_column],
            outputs=[buffer.values],
            device=self.device,
        )

    def _ensure_minibatch(self, buffer: BufferRollout, segment_count: int) -> BatchPPO:
        if (
            self._minibatch is None
            or self._minibatch.num_steps != buffer.num_steps
            or self._minibatch.num_envs != int(segment_count)
        ):
            self._minibatch = BatchPPO(
                num_steps=buffer.num_steps,
                num_envs=int(segment_count),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            self._minibatch_env_ids = wp.zeros(int(segment_count), dtype=wp.int32, device=self.device)
        return self._minibatch

    def _update_shared_manual(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> StatsPPOUpdate:
        if not (self.config.manual_actor_backward and self.config.manual_critic_backward):
            raise RuntimeError("shared_value_network requires manual actor and critic backward")

        self._set_update_sequence_shape(buffer)
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self._policy_update_reuse(mirror_obs, buffer)

        log_std_grad_partials, log_std_partial_count = self._ensure_actor_log_std_grad_partials(buffer.num_samples)
        wp.launch(
            zero_ppo_actor_stats_kernel,
            dim=max(log_std_partial_count * self.action_dim, 1),
            inputs=[log_std_partial_count, self.action_dim],
            outputs=[self._policy_loss, self._approx_kl, self._clip_fraction, log_std_grad_partials],
            device=self.device,
        )
        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._value_loss], device=self.device)

        policy_out = self.actor.net.forward_manual(buffer.obs)
        policy_out_grad = self._ensure_actor_backward_buffers(buffer.num_samples, int(policy_out.shape[1]))
        wp.launch(
            ppo_actor_loss_backward_kernel,
            dim=buffer.num_samples,
            inputs=[
                policy_out,
                self.actor.log_std,
                buffer.actions,
                buffer.old_log_probs,
                buffer.advantages,
                self.config.clip_ratio,
                self.config.entropy_coeff,
                self.action_dim,
                int(self.actor.state_dependent_std),
                int(self.actor.squash),
                self.actor.log_std_min,
                self.actor.log_std_max,
                buffer.num_samples,
            ],
            outputs=[
                self._policy_loss,
                self._approx_kl,
                self._clip_fraction,
                buffer.ratios,
                policy_out_grad,
                log_std_grad_partials,
            ],
            device=self.device,
        )
        if mirror_policy_out is not None:
            wp.launch(
                mirrored_action_mse_grad_kernel,
                dim=buffer.num_samples,
                inputs=[
                    policy_out,
                    mirror_policy_out,
                    self._mirror_action_src,
                    self._mirror_action_sign,
                    self.action_dim,
                    self.config.mirror_loss_coeff,
                    buffer.num_samples,
                ],
                outputs=[policy_out_grad, self._policy_loss],
                device=self.device,
            )
        wp.launch(
            value_column_loss_grad_kernel,
            dim=buffer.num_samples,
            inputs=[
                policy_out,
                self.value_column,
                buffer.old_values,
                buffer.returns,
                self.config.value_loss_coeff,
                self.config.value_clip_range,
                buffer.num_samples,
            ],
            outputs=[self._value_loss, policy_out_grad],
            device=self.device,
        )
        if mirror_policy_out is not None:
            wp.launch(
                value_column_symmetry_loss_grad_kernel,
                dim=buffer.num_samples,
                inputs=[
                    policy_out,
                    self.value_column,
                    mirror_policy_out,
                    self.config.mirror_loss_coeff,
                    buffer.num_samples,
                ],
                outputs=[self._value_loss, policy_out_grad],
                device=self.device,
            )

        self.actor.net.backward_manual(policy_out_grad)
        if not self.actor.state_dependent_std:
            wp.launch(
                reduce_ppo_log_std_grad_kernel,
                dim=self.action_dim,
                inputs=[log_std_grad_partials, log_std_partial_count],
                outputs=[self._actor_log_std_grad],
                device=self.device,
            )
            self.actor.log_std.grad.assign(self._actor_log_std_grad)
        if read_stats:
            stats = self._read_update_stats()
        else:
            stats = StatsPPOUpdate(policy_loss=0.0, value_loss=0.0, approx_kl=0.0, clip_fraction=0.0)
        self.actor_optimizer.step()
        return stats

    def _update_actor(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> tuple[float, float, float]:
        if self.config.manual_actor_backward:
            return self._update_actor_manual(buffer, read_stats=read_stats)
        return self._update_actor_tape(buffer, read_stats=read_stats)

    def _ensure_actor_backward_buffers(self, rows: int, cols: int) -> wp.array2d[wp.float32]:
        requested_rows = int(rows)
        requested_cols = int(cols)
        if (
            self._actor_policy_out_grad is None
            or int(self._actor_policy_out_grad.shape[0]) < requested_rows
            or int(self._actor_policy_out_grad.shape[1]) != requested_cols
        ):
            self._actor_policy_out_grad = wp.zeros(
                (requested_rows, requested_cols), dtype=wp.float32, device=self.device, requires_grad=False
            )
        return self._actor_policy_out_grad

    def _ensure_actor_log_std_grad_partials(self, rows: int) -> tuple[wp.array2d[wp.float32], int]:
        partial_count = (int(rows) + PPO_LOG_STD_PARTIAL_BATCH - 1) // PPO_LOG_STD_PARTIAL_BATCH
        partial_count = max(partial_count, 1)
        if (
            self._actor_log_std_grad_partials is None
            or int(self._actor_log_std_grad_partials.shape[0]) < partial_count
            or int(self._actor_log_std_grad_partials.shape[1]) != self.action_dim
        ):
            self._actor_log_std_grad_partials = wp.zeros(
                (partial_count, self.action_dim), dtype=wp.float32, device=self.device, requires_grad=False
            )
        self._actor_log_std_grad_partial_count = partial_count
        return self._actor_log_std_grad_partials, partial_count

    def _update_actor_manual(
        self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True
    ) -> tuple[float, float, float]:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self.actor.net.forward_reuse(mirror_obs)

        log_std_grad_partials, log_std_partial_count = self._ensure_actor_log_std_grad_partials(buffer.num_samples)
        wp.launch(
            zero_ppo_actor_stats_kernel,
            dim=max(log_std_partial_count * self.action_dim, 1),
            inputs=[log_std_partial_count, self.action_dim],
            outputs=[self._policy_loss, self._approx_kl, self._clip_fraction, log_std_grad_partials],
            device=self.device,
        )
        policy_out = self.actor.net.forward_manual(buffer.obs)
        policy_out_grad = self._ensure_actor_backward_buffers(buffer.num_samples, int(policy_out.shape[1]))
        wp.launch(
            ppo_actor_loss_backward_kernel,
            dim=buffer.num_samples,
            inputs=[
                policy_out,
                self.actor.log_std,
                buffer.actions,
                buffer.old_log_probs,
                buffer.advantages,
                self.config.clip_ratio,
                self.config.entropy_coeff,
                self.action_dim,
                int(self.actor.state_dependent_std),
                int(self.actor.squash),
                self.actor.log_std_min,
                self.actor.log_std_max,
                buffer.num_samples,
            ],
            outputs=[
                self._policy_loss,
                self._approx_kl,
                self._clip_fraction,
                buffer.ratios,
                policy_out_grad,
                log_std_grad_partials,
            ],
            device=self.device,
        )
        if mirror_policy_out is not None:
            wp.launch(
                mirrored_action_mse_grad_kernel,
                dim=buffer.num_samples,
                inputs=[
                    policy_out,
                    mirror_policy_out,
                    self._mirror_action_src,
                    self._mirror_action_sign,
                    self.action_dim,
                    self.config.mirror_loss_coeff,
                    buffer.num_samples,
                ],
                outputs=[policy_out_grad, self._policy_loss],
                device=self.device,
            )
        self.actor.net.backward_manual(policy_out_grad)
        if not self.actor.state_dependent_std:
            wp.launch(
                reduce_ppo_log_std_grad_kernel,
                dim=self.action_dim,
                inputs=[log_std_grad_partials, log_std_partial_count],
                outputs=[self._actor_log_std_grad],
                device=self.device,
            )
            self.actor.log_std.grad.assign(self._actor_log_std_grad)
        if read_stats:
            loss, _value_loss, kl, clip_fraction = self._read_update_stat_values()
        else:
            loss = 0.0
            kl = 0.0
            clip_fraction = 0.0
        self.actor_optimizer.step()
        return loss, kl, clip_fraction

    def _update_actor_tape(
        self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True
    ) -> tuple[float, float, float]:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self.actor.net.forward_reuse(mirror_obs)

        wp.launch(
            zero_ppo_loss_stats_kernel,
            dim=1,
            outputs=[self._policy_loss, self._approx_kl, self._clip_fraction],
            device=self.device,
        )
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
                outputs=[self._policy_loss, self._approx_kl, self._clip_fraction, buffer.ratios],
                device=self.device,
            )
            if mirror_policy_out is not None:
                wp.launch(
                    mirrored_action_mse_loss_kernel,
                    dim=buffer.num_samples,
                    inputs=[
                        policy_out,
                        mirror_policy_out,
                        self._mirror_action_src,
                        self._mirror_action_sign,
                        self.action_dim,
                        self.config.mirror_loss_coeff,
                        buffer.num_samples,
                    ],
                    outputs=[self._policy_loss],
                    device=self.device,
                )
        tape.backward(self._policy_loss)
        if read_stats:
            loss, _value_loss, kl, clip_fraction = self._read_update_stat_values()
        else:
            loss = 0.0
            kl = 0.0
            clip_fraction = 0.0
        self.actor_optimizer.step()
        tape.zero()
        return loss, kl, clip_fraction

    def _ensure_critic_backward_buffers(self, rows: int) -> wp.array2d[wp.float32]:
        requested_rows = int(rows)
        if self._critic_value_grad is None or int(self._critic_value_grad.shape[0]) < requested_rows:
            self._critic_value_grad = wp.zeros(
                (requested_rows, 1), dtype=wp.float32, device=self.device, requires_grad=False
            )
        return self._critic_value_grad

    def _update_critic(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> float:
        if self.critic is None or self.critic_optimizer is None:
            raise RuntimeError("shared_value_network has no separate critic update")
        if self.config.manual_critic_backward:
            return self._update_critic_manual(buffer, read_stats=read_stats)
        return self._update_critic_tape(buffer, read_stats=read_stats)

    def _update_critic_manual(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> float:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_values = None
        if mirror_obs is not None:
            mirror_values = self.critic.forward_reuse(mirror_obs)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._value_loss], device=self.device)
        values = self.critic.forward_manual(buffer.obs)
        value_grad = self._ensure_critic_backward_buffers(buffer.num_samples)
        wp.launch(
            value_loss_grad_kernel,
            dim=buffer.num_samples,
            inputs=[
                values,
                buffer.old_values,
                buffer.returns,
                self.config.value_loss_coeff,
                self.config.value_clip_range,
                buffer.num_samples,
            ],
            outputs=[self._value_loss, value_grad],
            device=self.device,
        )
        if mirror_values is not None:
            wp.launch(
                value_symmetry_loss_grad_kernel,
                dim=buffer.num_samples,
                inputs=[values, mirror_values, self.config.mirror_loss_coeff, buffer.num_samples],
                outputs=[self._value_loss, value_grad],
                device=self.device,
            )
        self.critic.backward_manual(value_grad)
        loss = self._read_update_stat_values()[1] if read_stats else 0.0
        self.critic_optimizer.step()
        return loss

    def _update_critic_tape(self, buffer: BufferRollout | BatchPPO, *, read_stats: bool = True) -> float:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_values = None
        if mirror_obs is not None:
            mirror_values = self.critic.forward_reuse(mirror_obs)

        wp.launch(zero_scalar_kernel, dim=1, outputs=[self._value_loss], device=self.device)
        with wp.Tape() as tape:
            values = self.critic.forward(buffer.obs, requires_grad=True)
            wp.launch(
                value_loss_kernel,
                dim=buffer.num_samples,
                inputs=[
                    values,
                    buffer.old_values,
                    buffer.returns,
                    self.config.value_loss_coeff,
                    self.config.value_clip_range,
                    buffer.num_samples,
                ],
                outputs=[self._value_loss],
                device=self.device,
            )
            if mirror_values is not None:
                wp.launch(
                    value_symmetry_loss_kernel,
                    dim=buffer.num_samples,
                    inputs=[values, mirror_values, self.config.mirror_loss_coeff, buffer.num_samples],
                    outputs=[self._value_loss],
                    device=self.device,
                )
        tape.backward(self._value_loss)
        loss = self._read_update_stat_values()[1] if read_stats else 0.0
        self.critic_optimizer.step()
        tape.zero()
        return loss

    def _ensure_mirror_obs(self, rows: int) -> wp.array2d[wp.float32]:
        requested_rows = int(rows)
        if requested_rows <= 0:
            raise ValueError("mirror observation row count must be positive")
        if self._mirror_obs is None or int(self._mirror_obs.shape[0]) != requested_rows:
            self._mirror_obs = wp.empty(
                (requested_rows, self.obs_dim), dtype=wp.float32, device=self.device, requires_grad=False
            )
        return self._mirror_obs

    def _mirrored_obs(self, buffer: BufferRollout | BatchPPO) -> wp.array2d[wp.float32] | None:
        if self.config.mirror_loss_coeff <= 0.0:
            return None
        if self.mirror_map is None:
            raise ValueError("PPO mirror_loss_coeff requires a mirror map")
        self._ensure_mirror_obs(buffer.num_samples)
        wp.launch(
            mirror_2d_kernel,
            dim=(buffer.num_samples, self.obs_dim),
            inputs=[buffer.obs, self._mirror_obs_src, self._mirror_obs_sign],
            outputs=[self._mirror_obs],
            device=self.device,
        )
        return self._mirror_obs


def _normalize_advantages(
    advantages: wp.array[wp.float32],
    count: int,
    device: wp.context.Device,
    *,
    eps: float,
    stats: wp.array[wp.float32] | None = None,
) -> None:
    if count <= 0:
        return
    if stats is None:
        stats = wp.zeros(2, dtype=wp.float32, device=device)
    elif int(stats.shape[0]) != 2:
        raise ValueError("advantage normalization stats buffer must have length 2")
    stats.zero_()
    wp.launch(
        sum_and_sumsq_kernel,
        dim=count,
        inputs=[advantages, count],
        outputs=[stats],
        device=device,
    )
    wp.launch(
        normalize_from_stats_kernel,
        dim=count,
        inputs=[advantages, stats, count, float(eps)],
        device=device,
    )


def save_ppo_checkpoint(trainer: TrainerPPO, path: str | Path, *, iteration: int | None = None) -> None:
    """Save a PPO trainer checkpoint.

    Args:
        trainer: Trainer to save.
        path: Output ``.npz`` path.
        iteration: Optional training iteration stored as metadata.
    """

    trainer.save_checkpoint(path, iteration=iteration)


def load_ppo_checkpoint(
    path: str | Path,
    *,
    config: ConfigPPO | None = None,
    device: wp.context.Devicelike = None,
) -> TrainerPPO:
    """Load a PPO trainer checkpoint.

    Args:
        path: Input ``.npz`` path.
        config: Optional PPO config overriding saved settings.
        device: Warp device for restored arrays.

    Returns:
        Restored trainer.
    """

    return TrainerPPO.load_checkpoint(path, config=config, device=device)


def _validate_mirror_map(mirror_map: MirrorMapPPO, obs_dim: int, action_dim: int) -> None:
    obs_src = np.asarray(mirror_map.obs_src, dtype=np.int64)
    obs_sign = np.asarray(mirror_map.obs_sign, dtype=np.float32)
    action_src = np.asarray(mirror_map.action_src, dtype=np.int64)
    action_sign = np.asarray(mirror_map.action_sign, dtype=np.float32)
    if obs_src.shape != (int(obs_dim),) or obs_sign.shape != (int(obs_dim),):
        raise ValueError(f"Observation mirror map must have length {obs_dim}")
    if action_src.shape != (int(action_dim),) or action_sign.shape != (int(action_dim),):
        raise ValueError(f"Action mirror map must have length {action_dim}")
    if np.any(obs_src < 0) or np.any(obs_src >= int(obs_dim)):
        raise ValueError("Observation mirror source indices are out of range")
    if np.any(action_src < 0) or np.any(action_src >= int(action_dim)):
        raise ValueError("Action mirror source indices are out of range")
    if not np.isfinite(obs_sign).all() or not np.isfinite(action_sign).all():
        raise ValueError("Mirror signs must be finite")
    if not _is_signed_involution(obs_src, obs_sign):
        raise ValueError("Observation mirror map must be a signed involution")
    if not _is_signed_involution(action_src, action_sign):
        raise ValueError("Action mirror map must be a signed involution")


def _is_signed_involution(src: np.ndarray, sign: np.ndarray) -> bool:
    for i, j in enumerate(src):
        if int(src[int(j)]) != int(i):
            return False
        if not np.isclose(float(sign[i]) * float(sign[int(j)]), 1.0, rtol=0.0, atol=1.0e-6):
            return False
    return True


def _normalize_policy_network(policy_network: str) -> str:
    key = str(policy_network).lower().replace("-", "_")
    if key in ("mlp", "feedforward", "feed_forward"):
        return "mlp"
    if key in ("puffer_mingru", "mingru", "min_gru", "puffernet"):
        return "puffer_mingru"
    raise ValueError(f"Unsupported PPO policy_network {policy_network!r}")


def _mingru_shape_from_hidden_layers(hidden_layers: tuple[int, ...]) -> tuple[int, int]:
    if not hidden_layers:
        raise ValueError("puffer_mingru policy_network requires at least one hidden layer width")
    hidden_size = int(hidden_layers[0])
    if any(int(width) != hidden_size for width in hidden_layers):
        raise ValueError("puffer_mingru policy_network requires equal hidden layer widths")
    return hidden_size, len(hidden_layers)


def _make_optimizer(params: list[wp.array], config: ConfigPPO, *, lr: float) -> Adam | Muon:
    optimizer = str(config.optimizer).lower()
    if optimizer == "adam":
        return Adam(
            params,
            lr=lr,
            eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay,
            max_grad_norm=config.max_grad_norm,
        )
    if optimizer == "muon":
        return Muon(
            params,
            lr=lr,
            momentum=config.muon_momentum,
            eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay,
            max_grad_norm=config.max_grad_norm,
            matrix_transpose=True,
        )
    raise ValueError(f"Unsupported PPO optimizer {config.optimizer!r}")


def _optimizer_type(optimizer: Adam | Muon) -> str:
    if isinstance(optimizer, Adam):
        return "adam"
    if isinstance(optimizer, Muon):
        return "muon"
    raise TypeError(f"Unsupported optimizer type {type(optimizer)!r}")


def _pack_policy_network(data: dict[str, np.ndarray], prefix: str, network: WarpMLP | PufferMinGRUNet) -> None:
    if isinstance(network, PufferMinGRUNet):
        data[f"{prefix}_network_type"] = np.asarray("puffer_mingru")
        data[f"{prefix}_hidden_size"] = np.asarray(network.hidden_size, dtype=np.int64)
        data[f"{prefix}_recurrent_layer_count"] = np.asarray(network.num_layers, dtype=np.int64)
        data[f"{prefix}_encoder_weight"] = network.encoder_weight.numpy()
        data[f"{prefix}_decoder_weight"] = network.decoder_weight.numpy()
        for index, weight in enumerate(network.recurrent_weights):
            data[f"{prefix}_recurrent_weight_{index}"] = weight.numpy()
        return
    data[f"{prefix}_network_type"] = np.asarray("mlp")
    data[f"{prefix}_layer_count"] = np.asarray(len(network.weights), dtype=np.int64)
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases, strict=True)):
        data[f"{prefix}_weight_{index}"] = weight.numpy()
        data[f"{prefix}_bias_{index}"] = bias.numpy()


def _unpack_policy_network(data: np.lib.npyio.NpzFile, prefix: str, network: WarpMLP | PufferMinGRUNet) -> None:
    saved_type = str(data[f"{prefix}_network_type"].item()) if f"{prefix}_network_type" in data else "mlp"
    if isinstance(network, PufferMinGRUNet):
        if saved_type != "puffer_mingru":
            raise ValueError(f"Checkpoint {prefix} network type {saved_type!r} does not match puffer_mingru")
        layer_count = int(data[f"{prefix}_recurrent_layer_count"])
        if layer_count != network.num_layers:
            raise ValueError(f"Checkpoint {prefix} recurrent layer count does not match trainer")
        network.encoder_weight.assign(data[f"{prefix}_encoder_weight"])
        network.decoder_weight.assign(data[f"{prefix}_decoder_weight"])
        for index, weight in enumerate(network.recurrent_weights):
            weight.assign(data[f"{prefix}_recurrent_weight_{index}"])
        return
    if saved_type != "mlp":
        raise ValueError(f"Checkpoint {prefix} network type {saved_type!r} does not match mlp")
    layer_count = int(data[f"{prefix}_layer_count"])
    if layer_count != len(network.weights):
        raise ValueError(f"Checkpoint {prefix} layer count does not match trainer")
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases, strict=True)):
        weight.assign(data[f"{prefix}_weight_{index}"])
        bias.assign(data[f"{prefix}_bias_{index}"])


def _pack_adam(data: dict[str, np.ndarray], prefix: str, optimizer: Adam) -> None:
    data[f"{prefix}_step_count"] = np.asarray(optimizer.step_count, dtype=np.int64)
    data[f"{prefix}_state_count"] = np.asarray(len(optimizer.m), dtype=np.int64)
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        data[f"{prefix}_m_{index}"] = m.numpy()
        data[f"{prefix}_v_{index}"] = v.numpy()


def _pack_muon(data: dict[str, np.ndarray], prefix: str, optimizer: Muon) -> None:
    data[f"{prefix}_step_count"] = np.asarray(optimizer.step_count, dtype=np.int64)
    data[f"{prefix}_state_count"] = np.asarray(len(optimizer.m), dtype=np.int64)
    data[f"{prefix}_matrix_transpose"] = np.asarray(optimizer.matrix_transpose, dtype=np.bool_)
    for index, momentum in enumerate(optimizer.m):
        data[f"{prefix}_m_{index}"] = momentum.numpy()


def _pack_optimizer(data: dict[str, np.ndarray], prefix: str, optimizer: Adam | Muon) -> None:
    opt_type = _optimizer_type(optimizer)
    data[f"{prefix}_type"] = np.asarray(opt_type)
    if opt_type == "adam":
        _pack_adam(data, prefix, optimizer)
    else:
        _pack_muon(data, prefix, optimizer)


def _unpack_adam(data: np.lib.npyio.NpzFile, prefix: str, optimizer: Adam) -> None:
    state_count = int(data[f"{prefix}_state_count"])
    if state_count != len(optimizer.m):
        raise ValueError(f"Checkpoint {prefix} state count does not match optimizer")
    optimizer.step_count = int(data[f"{prefix}_step_count"])
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        m.assign(data[f"{prefix}_m_{index}"])
        v.assign(data[f"{prefix}_v_{index}"])


def _unpack_muon(data: np.lib.npyio.NpzFile, prefix: str, optimizer: Muon) -> None:
    state_count = int(data[f"{prefix}_state_count"])
    if state_count != len(optimizer.m):
        raise ValueError(f"Checkpoint {prefix} state count does not match optimizer")
    if f"{prefix}_matrix_transpose" in data:
        saved_matrix_transpose = bool(data[f"{prefix}_matrix_transpose"].item())
        if saved_matrix_transpose != optimizer.matrix_transpose:
            raise ValueError(f"Checkpoint {prefix} Muon matrix layout does not match optimizer")
    optimizer.step_count = int(data[f"{prefix}_step_count"])
    for index, momentum in enumerate(optimizer.m):
        momentum.assign(data[f"{prefix}_m_{index}"])


def _unpack_optimizer(data: np.lib.npyio.NpzFile, prefix: str, optimizer: Adam | Muon) -> None:
    saved_type = str(data[f"{prefix}_type"].item()) if f"{prefix}_type" in data else "adam"
    current_type = _optimizer_type(optimizer)
    if saved_type != current_type:
        raise ValueError(f"Checkpoint {prefix} optimizer type {saved_type!r} does not match {current_type!r}")
    if saved_type == "adam":
        if not isinstance(optimizer, Adam):
            raise TypeError("Adam checkpoint requires an Adam optimizer")
        _unpack_adam(data, prefix, optimizer)
    else:
        if not isinstance(optimizer, Muon):
            raise TypeError("Muon checkpoint requires a Muon optimizer")
        _unpack_muon(data, prefix, optimizer)


def _config_from_checkpoint(data: np.lib.npyio.NpzFile) -> ConfigPPO:
    values = {}
    for field in ConfigPPO.__dataclass_fields__:
        key = f"config_{field}"
        if key in data:
            value = data[key].item()
            if field in (
                "anneal_lr",
                "manual_actor_backward",
                "manual_critic_backward",
                "normalize_advantages",
                "puffer_vtrace_advantage",
                "shared_value_network",
            ):
                value = bool(value)
            values[field] = value
    return ConfigPPO(**values)
