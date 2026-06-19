# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .kernels import (
    compute_gae_kernel,
    gaussian_entropy_kernel,
    mirror_2d_kernel,
    mirrored_action_mse_loss_kernel,
    normalize_kernel,
    ppo_actor_loss_kernel,
    sum_and_sumsq_kernel,
    value_loss_kernel,
    value_symmetry_loss_kernel,
    zero_scalar_kernel,
)
from .networks import GaussianActor, WarpMLP
from .optim import Adam


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
        actor_lr: Actor Adam learning rate.
        critic_lr: Critic Adam learning rate.
        train_epochs: Full-buffer optimization epochs per rollout.
        normalize_advantages: Whether to normalize advantages in-place before
            updating.
        mirror_loss_coeff: Coefficient for optional mirror-symmetry MSE on
            policy means and value predictions. Requires a mirror map.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 1.0e-3
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    train_epochs: int = 4
    normalize_advantages: bool = True
    mirror_loss_coeff: float = 0.0


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
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.activation = activation
        self.squash_actions = bool(squash_actions)
        self.log_std_init = float(log_std_init)
        self.device = wp.get_device(device)
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
        )
        self.critic = WarpMLP(
            (self.obs_dim, *self.hidden_layers, 1),
            activation=activation,
            output_activation="linear",
            device=self.device,
            seed=seed + 1,
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.iteration = 0
        self.mirror_map: MirrorMapPPO | None = None
        self._mirror_obs_src: wp.array | None = None
        self._mirror_obs_sign: wp.array | None = None
        self._mirror_action_src: wp.array | None = None
        self._mirror_action_sign: wp.array | None = None
        self._mirror_obs: wp.array | None = None
        if mirror_map is not None:
            self.set_mirror_map(mirror_map)

        self._policy_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._value_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._approx_kl = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._clip_fraction = wp.zeros(1, dtype=wp.float32, device=self.device)

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
            "squash_actions": np.asarray(int(self.squash_actions), dtype=np.int64),
            "log_std_init": np.asarray(self.log_std_init, dtype=np.float32),
            "iteration": np.asarray(self.iteration if iteration is None else int(iteration), dtype=np.int64),
        }
        for key, value in asdict(self.config).items():
            data[f"config_{key}"] = np.asarray(value)
        _pack_mlp(data, "actor", self.actor.net)
        data["actor_log_std"] = self.actor.log_std.numpy()
        _pack_mlp(data, "critic", self.critic)
        _pack_adam(data, "actor_optimizer", self.actor_optimizer)
        _pack_adam(data, "critic_optimizer", self.critic_optimizer)
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
                seed=0,
                squash_actions=bool(int(data["squash_actions"])),
                activation=str(data["activation"].item()),
                log_std_init=float(data["log_std_init"]),
            )
            _unpack_mlp(data, "actor", trainer.actor.net)
            trainer.actor.log_std.assign(data["actor_log_std"])
            _unpack_mlp(data, "critic", trainer.critic)
            _unpack_adam(data, "actor_optimizer", trainer.actor_optimizer)
            _unpack_adam(data, "critic_optimizer", trainer.critic_optimizer)
            if "iteration" in data:
                trainer.iteration = max(int(data["iteration"]), 0)
        return trainer

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
        mirror_obs = self._mirrored_obs(buffer)
        mirror_policy_out = None
        if mirror_obs is not None:
            mirror_policy_out = self.actor.forward(mirror_obs, requires_grad=False)

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
            if mirror_policy_out is not None:
                wp.launch(
                    mirrored_action_mse_loss_kernel,
                    dim=(buffer.num_samples, self.action_dim),
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
        loss = float(self._policy_loss.numpy()[0])
        kl = float(self._approx_kl.numpy()[0])
        clip_fraction = float(self._clip_fraction.numpy()[0])
        self.actor_optimizer.step()
        tape.zero()
        return loss, kl, clip_fraction

    def _update_critic(self, buffer: BufferRollout) -> float:
        mirror_obs = self._mirrored_obs(buffer)
        mirror_values = None
        if mirror_obs is not None:
            mirror_values = self.critic.forward(mirror_obs, requires_grad=False)

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
            if mirror_values is not None:
                wp.launch(
                    value_symmetry_loss_kernel,
                    dim=buffer.num_samples,
                    inputs=[values, mirror_values, self.config.mirror_loss_coeff, buffer.num_samples],
                    outputs=[self._value_loss],
                    device=self.device,
                )
        tape.backward(self._value_loss)
        loss = float(self._value_loss.numpy()[0])
        self.critic_optimizer.step()
        tape.zero()
        return loss

    def _mirrored_obs(self, buffer: BufferRollout) -> wp.array | None:
        if self.config.mirror_loss_coeff <= 0.0:
            return None
        if self.mirror_map is None:
            raise ValueError("PPO mirror_loss_coeff requires a mirror map")
        if self._mirror_obs is None or int(self._mirror_obs.shape[0]) != buffer.num_samples:
            self._mirror_obs = wp.empty(
                (buffer.num_samples, self.obs_dim), dtype=wp.float32, device=self.device, requires_grad=False
            )
        wp.launch(
            mirror_2d_kernel,
            dim=(buffer.num_samples, self.obs_dim),
            inputs=[buffer.obs, self._mirror_obs_src, self._mirror_obs_sign],
            outputs=[self._mirror_obs],
            device=self.device,
        )
        return self._mirror_obs


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


def _pack_mlp(data: dict[str, np.ndarray], prefix: str, mlp: WarpMLP) -> None:
    data[f"{prefix}_layer_count"] = np.asarray(len(mlp.weights), dtype=np.int64)
    for index, (weight, bias) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
        data[f"{prefix}_weight_{index}"] = weight.numpy()
        data[f"{prefix}_bias_{index}"] = bias.numpy()


def _unpack_mlp(data: np.lib.npyio.NpzFile, prefix: str, mlp: WarpMLP) -> None:
    layer_count = int(data[f"{prefix}_layer_count"])
    if layer_count != len(mlp.weights):
        raise ValueError(f"Checkpoint {prefix} layer count does not match trainer")
    for index, (weight, bias) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
        weight.assign(data[f"{prefix}_weight_{index}"])
        bias.assign(data[f"{prefix}_bias_{index}"])


def _pack_adam(data: dict[str, np.ndarray], prefix: str, optimizer: Adam) -> None:
    data[f"{prefix}_step_count"] = np.asarray(optimizer.step_count, dtype=np.int64)
    data[f"{prefix}_state_count"] = np.asarray(len(optimizer.m), dtype=np.int64)
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        data[f"{prefix}_m_{index}"] = m.numpy()
        data[f"{prefix}_v_{index}"] = v.numpy()


def _unpack_adam(data: np.lib.npyio.NpzFile, prefix: str, optimizer: Adam) -> None:
    state_count = int(data[f"{prefix}_state_count"])
    if state_count != len(optimizer.m):
        raise ValueError(f"Checkpoint {prefix} state count does not match optimizer")
    optimizer.step_count = int(data[f"{prefix}_step_count"])
    for index, (m, v) in enumerate(zip(optimizer.m, optimizer.v, strict=True)):
        m.assign(data[f"{prefix}_m_{index}"])
        v.assign(data[f"{prefix}_v_{index}"])


def _config_from_checkpoint(data: np.lib.npyio.NpzFile) -> ConfigPPO:
    values = {}
    for field in ConfigPPO.__dataclass_fields__:
        key = f"config_{field}"
        if key in data:
            value = data[key].item()
            if field == "normalize_advantages":
                value = bool(value)
            values[field] = value
    return ConfigPPO(**values)
