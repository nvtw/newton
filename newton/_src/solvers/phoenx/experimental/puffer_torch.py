# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PufferLib-style Torch learner for PhoenX G1 isolation runs.

This module is intentionally separate from the pure-Warp PPO trainer. It keeps
the PhoenX G1 environment fixed while swapping the RL update to the same native
PufferNet/PPO conventions used by nanoG1: bias-free encoder/decoder, MinGRU,
Gaussian logstd, shifted Puffer V-trace advantages, prioritized trajectory
replay, Muon, and the G1 mirror regularizer.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.g1 import ConfigEnvG1PhoenX, EnvG1PhoenX, g1_mirror_map_ppo
from newton._src.solvers.phoenx.rl_training.training import (
    ConfigEvaluateG1GatePPO,
    ResultEvaluateG1GatePPO,
    StatsEvaluateG1GateCommandPPO,
    StatsEvaluateG1GatePPO,
    _g1_gate_commands_array,
    _g1_gate_passes,
    _joint_q_matrix_g1,
    _joint_qd_matrix_g1,
    _quat_rotate_inverse_xyzw_np,
    _reset_g1_done_worlds,
    _validate_g1_gate_config,
)

_LOG_2PI = 1.8378770664093453


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Puffer-style PhoenX training requires PyTorch, matching nanoG1/PufferLib") from exc
    return torch, nn, F


def _load_puffer_muon(pufferlib_root: str | None):
    if pufferlib_root:
        root = str(Path(pufferlib_root).expanduser())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        module = importlib.import_module("pufferlib.muon")
    except ImportError as exc:
        raise RuntimeError(
            "Puffer-style PhoenX training requires PufferLib's Muon optimizer. "
            "Pass --pufferlib-root or set PUFFERLIB_ROOT to a PufferLib checkout."
        ) from exc
    return module.Muon


@dataclass
class ConfigTrainG1PufferTorch:
    """Configuration for the PufferLib-style Torch learner on PhoenX G1."""

    iterations: int = g1_recipe.TRAIN_ITERATIONS
    rollout_steps: int = g1_recipe.ROLLOUT_STEPS
    hidden_layers: tuple[int, ...] = g1_recipe.HIDDEN_LAYERS
    env_config: ConfigEnvG1PhoenX | None = None
    device: wp.context.Devicelike = None
    seed: int = g1_recipe.SEED
    total_timesteps: int = g1_recipe.LR_ANNEAL_TIMESTEPS
    learning_rate: float = g1_recipe.ACTOR_LR
    anneal_lr: bool = g1_recipe.ANNEAL_LR
    min_lr_ratio: float = g1_recipe.MIN_LR_RATIO
    gamma: float = g1_recipe.GAMMA
    gae_lambda: float = g1_recipe.GAE_LAMBDA
    replay_ratio: float = g1_recipe.REPLAY_RATIO
    clip_coef: float = g1_recipe.CLIP_RATIO
    vf_coef: float = g1_recipe.VALUE_LOSS_COEFF
    vf_clip_coef: float = g1_recipe.VALUE_CLIP_RANGE
    max_grad_norm: float = g1_recipe.MAX_GRAD_NORM
    ent_coef: float = g1_recipe.ENTROPY_COEFF
    beta1: float = g1_recipe.MUON_MOMENTUM
    eps: float = g1_recipe.OPTIMIZER_EPS
    minibatch_size: int = g1_recipe.MINIBATCH_SIZE
    vtrace_rho_clip: float = g1_recipe.VTRACE_RHO_CLIP
    vtrace_c_clip: float = g1_recipe.VTRACE_C_CLIP
    prio_alpha: float = g1_recipe.PRIORITY_ALPHA
    prio_beta0: float = g1_recipe.PRIORITY_BETA
    reward_clip: float = g1_recipe.REWARD_CLIP
    mirror_loss_coeff: float = g1_recipe.MIRROR_LOSS_COEFF
    mirror_value_loss: bool = True
    randomize_commands: bool = g1_recipe.RANDOMIZE_COMMANDS
    command_sampling: str = g1_recipe.COMMAND_SAMPLING
    command_x_range: tuple[float, float] = g1_recipe.COMMAND_X_RANGE
    command_y_range: tuple[float, float] = g1_recipe.COMMAND_Y_RANGE
    command_yaw_range: tuple[float, float] = g1_recipe.COMMAND_YAW_RANGE
    command_zero_probability: float = g1_recipe.COMMAND_ZERO_PROBABILITY
    command_resample_steps: int = g1_recipe.COMMAND_RESAMPLE_STEPS
    command_curriculum_start: float = g1_recipe.COMMAND_CURRICULUM_START
    command_curriculum_samples: int = g1_recipe.COMMAND_CURRICULUM_SAMPLES
    reset_recurrent_state_on_rollout_start: bool = True
    pufferlib_root: str | None = None
    checkpoint_path: str | None = None
    checkpoint_interval: int = 0
    log_interval: int = 1
    readback_diagnostics: bool = True


@dataclass
class StatsTrainG1PufferTorch:
    """Per-iteration diagnostics for the PufferLib-style PhoenX run."""

    iteration: int
    mean_reward: float
    mean_done: float
    mean_tracking_perf: float
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float
    rollout_seconds: float
    update_seconds: float
    samples_per_second: float


@dataclass
class ResultTrainG1PufferTorch:
    """Result returned by :func:`train_g1_puffer_torch`."""

    trainer: PufferTorchTrainer
    env: EnvG1PhoenX
    history: list[StatsTrainG1PufferTorch]


class PufferNativePolicy:
    """Torch implementation of PufferLib's native bias-free PufferNet."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...],
        seed: int,
        torch_module: Any,
        nn_module: Any,
        functional: Any,
        device: Any,
    ):
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        hidden_size = int(hidden_layers[0])
        if any(int(width) != hidden_size for width in hidden_layers):
            raise ValueError("PufferNet MinGRU requires equal hidden widths")
        self.torch = torch_module
        self.F = functional
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = hidden_size
        self.num_layers = len(tuple(hidden_layers))
        self.device = device

        class _Module(nn_module.Module):
            pass

        self.module = _Module().to(device)
        generator = torch_module.Generator(device=device)
        generator.manual_seed(int(seed))
        self.module.encoder_weight = nn_module.Parameter(
            self._uniform_weight((hidden_size, self.obs_dim), np.sqrt(2.0), generator)
        )
        self.module.decoder_weight = nn_module.Parameter(
            self._uniform_weight((self.action_dim + 1, hidden_size), 1.0, generator)
        )
        self.module.log_std = nn_module.Parameter(torch_module.zeros(1, self.action_dim, device=device))
        weights = []
        for _ in range(self.num_layers):
            weights.append(nn_module.Parameter(self._uniform_weight((3 * hidden_size, hidden_size), 1.0, generator)))
        self.module.recurrent_weights = nn_module.ParameterList(weights)

    def _uniform_weight(self, shape: tuple[int, int], gain: float, generator) -> Any:
        rows, cols = shape
        bound = float(gain) / float(np.sqrt(cols))
        return (self.torch.rand((rows, cols), generator=generator, device=self.device) * (2.0 * bound) - bound).float()

    def parameters(self):
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)

    def initial_state(self, batch_size: int) -> Any:
        return self.torch.zeros(self.num_layers, int(batch_size), self.hidden_size, device=self.device)

    def forward_eval(self, observations: Any, state: Any) -> tuple[Any, Any, Any]:
        h = self.F.linear(observations.float(), self.module.encoder_weight)
        next_states = []
        for layer, weight in enumerate(self.module.recurrent_weights):
            combined = self.F.linear(h, weight)
            hidden, gate, proj = combined.chunk(3, dim=-1)
            prev = state[layer]
            recurrent = prev + gate.sigmoid() * (self._g(hidden) - prev)
            h = proj.sigmoid() * recurrent + (1.0 - proj.sigmoid()) * h
            next_states.append(recurrent)
        out = self.F.linear(h, self.module.decoder_weight)
        return out[:, : self.action_dim], out[:, self.action_dim], self.torch.stack(next_states, dim=0)

    def forward_train(self, observations: Any) -> tuple[Any, Any]:
        batch, horizon = observations.shape[:2]
        h = self.F.linear(observations.reshape(batch * horizon, -1).float(), self.module.encoder_weight)
        h = h.reshape(batch, horizon, self.hidden_size)
        for weight in self.module.recurrent_weights:
            combined = self.F.linear(h.reshape(batch * horizon, self.hidden_size), weight)
            combined = combined.reshape(batch, horizon, 3 * self.hidden_size)
            hidden, gate, proj = combined.chunk(3, dim=-1)
            log_coeffs = -self.F.softplus(gate)
            log_values = -self.F.softplus(-gate) + self._log_g(hidden)
            recurrent = self._heinsen_scan(log_coeffs, log_values)[:, -horizon:]
            h = proj.sigmoid() * recurrent + (1.0 - proj.sigmoid()) * h
        out = self.F.linear(h.reshape(batch * horizon, self.hidden_size), self.module.decoder_weight)
        out = out.reshape(batch, horizon, self.action_dim + 1)
        return out[..., : self.action_dim], out[..., self.action_dim]

    def _g(self, x: Any) -> Any:
        return self.torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x: Any) -> Any:
        return self.torch.where(x >= 0, (self.F.relu(x) + 0.5).log(), -self.F.softplus(-x))

    def _heinsen_scan(self, log_coeffs: Any, log_values: Any) -> Any:
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()


class PufferTorchTrainer:
    """PufferLib-style PPO/V-trace learner backed by Torch tensors."""

    def __init__(self, env: EnvG1PhoenX, cfg: ConfigTrainG1PufferTorch):
        torch, nn, F = _require_torch()
        Muon = _load_puffer_muon(cfg.pufferlib_root)
        if not env.device.is_cuda:
            raise RuntimeError("Puffer-style PhoenX G1 training requires CUDA")
        torch_device = torch.device(wp.device_to_torch(env.device))
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.torch = torch
        self.F = F
        self.env = env
        self.cfg = cfg
        self.device = torch_device
        self.world_count = int(env.world_count)
        self.rollout_steps = int(cfg.rollout_steps)
        self.batch_size = self.world_count * self.rollout_steps
        self.minibatch_segments = max(1, int(cfg.minibatch_size) // self.rollout_steps)
        self.total_epochs = max(1, int(cfg.total_timesteps) // self.batch_size)
        self.iteration = 0

        self.policy = PufferNativePolicy(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_layers=tuple(cfg.hidden_layers),
            seed=int(cfg.seed),
            torch_module=torch,
            nn_module=nn,
            functional=F,
            device=torch_device,
        )
        self.optimizer = Muon(
            self.policy.parameters(),
            lr=float(cfg.learning_rate),
            momentum=float(cfg.beta1),
            eps=float(cfg.eps),
        )
        self.state = self.policy.initial_state(self.world_count)
        self._rollout_generator = torch.Generator(device=torch_device)
        self._rollout_generator.manual_seed(int(cfg.seed) + 17)
        self._update_generator = torch.Generator(device=torch_device)
        self._update_generator.manual_seed(int(cfg.seed) + 1_000_003)
        self._last_action_torch = None

        h, b, obs_dim, action_dim = self.rollout_steps, self.world_count, env.obs_dim, env.action_dim
        self.observations = torch.zeros(h, b, obs_dim, dtype=torch.float32, device=torch_device)
        self.actions = torch.zeros(h, b, action_dim, dtype=torch.float32, device=torch_device)
        self.values = torch.zeros(h, b, dtype=torch.float32, device=torch_device)
        self.logprobs = torch.zeros(h, b, dtype=torch.float32, device=torch_device)
        self.rewards = torch.zeros(h, b, dtype=torch.float32, device=torch_device)
        self.terminals = torch.zeros(h, b, dtype=torch.float32, device=torch_device)
        self.successes = torch.zeros(h, b, dtype=torch.float32, device=torch_device)
        self.ratio = torch.ones(b, h, dtype=torch.float32, device=torch_device)
        self._reward_prev = torch.zeros(b, dtype=torch.float32, device=torch_device)
        self._done_prev = torch.zeros(b, dtype=torch.float32, device=torch_device)
        self._success_prev = torch.zeros(b, dtype=torch.float32, device=torch_device)
        mirror_map = g1_mirror_map_ppo()
        self._obs_mirror_src = torch.as_tensor(mirror_map.obs_src, dtype=torch.long, device=torch_device)
        self._obs_mirror_sign = torch.as_tensor(mirror_map.obs_sign, dtype=torch.float32, device=torch_device)
        self._action_mirror_src = torch.as_tensor(mirror_map.action_src, dtype=torch.long, device=torch_device)
        self._action_mirror_sign = torch.as_tensor(mirror_map.action_sign, dtype=torch.float32, device=torch_device)

    def reset_rollout_state(self, dones: wp.array | None = None) -> None:
        if dones is None:
            self.state.zero_()
            return
        done = wp.to_torch(dones).to(device=self.device).bool()
        if bool(done.any()):
            self.state[:, done, :] = 0.0

    def act(self, obs: wp.array, *, seed: int = 0, deterministic: bool = False) -> tuple[wp.array, None, None]:
        del seed
        obs_t = wp.to_torch(obs).to(device=self.device)
        with self.torch.no_grad():
            mean, _value, self.state = self.policy.forward_eval(obs_t, self.state)
            if deterministic:
                action = mean
            else:
                action = mean + self.torch.exp(self.policy.module.log_std) * self.torch.randn_like(mean)
        action = action.contiguous()
        self._last_action_torch = action
        return wp.from_torch(action, dtype=wp.float32), None, None

    def collect_rollout(self) -> tuple[float, float, float]:
        torch = self.torch
        env = self.env
        if bool(self.cfg.reset_recurrent_state_on_rollout_start):
            self.state.zero_()
        obs_wp = env.observe()
        self._reward_prev.zero_()
        self._done_prev.zero_()
        self._success_prev.zero_()

        for step in range(self.rollout_steps):
            obs_t = wp.to_torch(obs_wp).to(device=self.device)
            with torch.no_grad():
                mean, value, self.state = self.policy.forward_eval(obs_t, self.state)
                action = mean + torch.exp(self.policy.module.log_std) * torch.randn(
                    mean.shape, generator=self._rollout_generator, device=self.device
                )
                logprob, _entropy = self._log_prob_entropy(mean, action)
                self.observations[step].copy_(obs_t)
                self.actions[step].copy_(action)
                self.values[step].copy_(value)
                self.logprobs[step].copy_(logprob)
                self.rewards[step].copy_(self._reward_prev)
                self.terminals[step].copy_(self._done_prev)
                self.successes[step].copy_(self._success_prev)
            action = action.contiguous()
            self._last_action_torch = action
            torch.cuda.synchronize(self.device)
            obs_wp, reward_wp, done_wp = env.step(wp.from_torch(action, dtype=wp.float32))
            wp.synchronize_device(env.device)
            self._reward_prev.copy_(wp.to_torch(reward_wp).to(device=self.device))
            self._done_prev.copy_(wp.to_torch(done_wp).to(device=self.device))
            self._success_prev.copy_(wp.to_torch(env.step_successes).to(device=self.device))

        return (
            float(self.rewards.mean().item()),
            float(self.terminals.mean().item()),
            float(self.successes.mean().item()),
        )

    def update(self) -> dict[str, float]:
        torch = self.torch
        cfg = self.cfg
        if cfg.anneal_lr and self.iteration > 0:
            lr_ratio = float(self.iteration) / float(max(self.total_epochs, 1))
            lr_min = float(cfg.learning_rate) * float(cfg.min_lr_ratio)
            lr = lr_min + 0.5 * (float(cfg.learning_rate) - lr_min) * (1.0 + np.cos(np.pi * lr_ratio))
            self.optimizer.param_groups[0]["lr"] = float(lr)

        obs = self.observations.transpose(0, 1).contiguous()
        act = self.actions.transpose(0, 1).contiguous()
        val = self.values.T.contiguous()
        lp = self.logprobs.T.contiguous()
        rew = self.rewards.T.contiguous()
        if float(cfg.reward_clip) > 0.0:
            rew = rew.clamp(-float(cfg.reward_clip), float(cfg.reward_clip))
        ter = self.terminals.T.contiguous()
        self.ratio.fill_(1.0)

        losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "old_approx_kl": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "importance": 0.0,
            "mirror_loss": 0.0,
        }
        num_minibatches = max(1, int(float(cfg.replay_ratio) * self.batch_size / float(cfg.minibatch_size)))
        anneal_beta = float(cfg.prio_beta0) + (1.0 - float(cfg.prio_beta0)) * float(cfg.prio_alpha) * (
            float(self.iteration) / float(max(self.total_epochs, 1))
        )

        for _mb in range(num_minibatches):
            advantages = self._puffer_advantages(val, rew, ter, self.ratio)
            prio_weights = torch.nan_to_num(advantages.abs().sum(axis=1).pow(float(cfg.prio_alpha)), 0.0, 0.0, 0.0)
            prio_probs = (prio_weights + 1.0e-6) / (prio_weights.sum() + 1.0e-6)
            idx = torch.multinomial(
                prio_probs,
                self.minibatch_segments,
                replacement=True,
                generator=self._update_generator,
            )
            mb_prio = (self.world_count * prio_probs[idx, None]).pow(-anneal_beta)

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            mean, newvalue = self.policy.forward_train(mb_obs)
            newlogprob, entropy = self._log_prob_entropy(mean, mb_actions)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > float(cfg.clip_coef)).float().mean()

            adv = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1.0e-8)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - float(cfg.clip_coef), 1.0 + float(cfg.clip_coef))
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -float(cfg.vf_clip_coef), float(cfg.vf_clip_coef))
            v_loss_unclipped = (newvalue - mb_returns).square()
            v_loss_clipped = (v_clipped - mb_returns).square()
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            entropy_loss = entropy.mean()
            mirror_loss = self._mirror_loss(mb_obs, mean, newvalue)
            loss = pg_loss + float(cfg.vf_coef) * v_loss - float(cfg.ent_coef) * entropy_loss + mirror_loss
            val[idx] = newvalue.detach().float()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(cfg.max_grad_norm))
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses["policy_loss"] += float(pg_loss.detach().item())
            losses["value_loss"] += float(v_loss.detach().item())
            losses["entropy"] += float(entropy_loss.detach().item())
            losses["old_approx_kl"] += float(old_approx_kl.detach().item())
            losses["approx_kl"] += float(approx_kl.detach().item())
            losses["clipfrac"] += float(clipfrac.detach().item())
            losses["importance"] += float(ratio.detach().mean().item())
            losses["mirror_loss"] += float(mirror_loss.detach().item())

        inv = 1.0 / float(num_minibatches)
        self.iteration += 1
        return {key: value * inv for key, value in losses.items()}

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": int(self.iteration),
            "config": _checkpoint_config(self.cfg),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(checkpoint, path)

    def _puffer_advantages(self, values: Any, rewards: Any, dones: Any, ratios: Any) -> Any:
        torch = self.torch
        cfg = self.cfg
        advantages = torch.zeros_like(values)
        trace = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
        for step in range(values.shape[1] - 2, -1, -1):
            next_nonterminal = 1.0 - dones[:, step + 1]
            rho = ratios[:, step]
            c = ratios[:, step]
            if float(cfg.vtrace_rho_clip) > 0.0:
                rho = torch.minimum(rho, torch.as_tensor(float(cfg.vtrace_rho_clip), device=values.device))
            if float(cfg.vtrace_c_clip) > 0.0:
                c = torch.minimum(c, torch.as_tensor(float(cfg.vtrace_c_clip), device=values.device))
            delta = rho * (
                rewards[:, step + 1] + float(cfg.gamma) * values[:, step + 1] * next_nonterminal - values[:, step]
            )
            trace = delta + float(cfg.gamma) * float(cfg.gae_lambda) * c * trace * next_nonterminal
            advantages[:, step] = trace
        return advantages

    def _log_prob_entropy(self, mean: Any, action: Any) -> tuple[Any, Any]:
        log_std = self.policy.module.log_std
        var = torch_exp_square(log_std, self.torch)
        log_prob = -0.5 * (((action - mean).square() / var) + 2.0 * log_std + _LOG_2PI)
        entropy = 0.5 + 0.5 * _LOG_2PI + log_std
        return log_prob.sum(dim=-1), entropy.expand_as(mean).sum(dim=-1)

    def _mirror_loss(self, obs: Any, mean: Any, value: Any) -> Any:
        coeff = float(self.cfg.mirror_loss_coeff)
        if coeff <= 0.0:
            return mean.sum() * 0.0
        mirror_obs = obs[..., self._obs_mirror_src] * self._obs_mirror_sign
        mirror_mean, mirror_value = self.policy.forward_train(mirror_obs.contiguous())
        action_target = mirror_mean[..., self._action_mirror_src] * self._action_mirror_sign
        action_delta = mean - action_target
        loss = 0.5 * coeff * action_delta.square().sum(dim=-1).mean()
        if bool(self.cfg.mirror_value_loss):
            loss = loss + 0.5 * coeff * (value - mirror_value).square().mean()
        return loss


def _shallow_dataclass_dict(obj: Any) -> dict[str, Any]:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _checkpoint_config(cfg: ConfigTrainG1PufferTorch) -> dict[str, Any]:
    data = _shallow_dataclass_dict(cfg)
    data["device"] = None if cfg.device is None else str(cfg.device)
    if cfg.env_config is not None:
        data["env_config"] = _shallow_dataclass_dict(cfg.env_config)
    return data


def torch_exp_square(x: Any, torch_module: Any) -> Any:
    std = torch_module.exp(x)
    return std * std


class _TorchGateAdapter:
    def __init__(self, trainer: PufferTorchTrainer, world_count: int):
        self.trainer = trainer
        self.state = trainer.policy.initial_state(world_count)
        self._last_action_torch = None

    def reset_rollout_state(self, dones: wp.array | None = None) -> None:
        torch = self.trainer.torch
        if dones is None:
            self.state.zero_()
            return
        done = wp.to_torch(dones).to(device=self.trainer.device).bool()
        if bool(done.any()):
            self.state[:, done, :] = 0.0
        torch.cuda.synchronize(self.trainer.device)

    def act(self, obs: wp.array, *, seed: int = 0, deterministic: bool = True) -> tuple[wp.array, None, None]:
        del seed
        torch = self.trainer.torch
        with torch.no_grad():
            mean, _value, self.state = self.trainer.policy.forward_eval(
                wp.to_torch(obs).to(device=self.trainer.device), self.state
            )
            if deterministic:
                action = mean
            else:
                action = mean + torch.exp(self.trainer.policy.module.log_std) * torch.randn_like(mean)
        action = action.contiguous()
        self._last_action_torch = action
        torch.cuda.synchronize(self.trainer.device)
        return wp.from_torch(action, dtype=wp.float32), None, None


def train_g1_puffer_torch(config: ConfigTrainG1PufferTorch | None = None) -> ResultTrainG1PufferTorch:
    """Train G1 in PhoenX with a PufferLib-style Torch PPO learner."""

    cfg = config or ConfigTrainG1PufferTorch()
    _validate_train_config(cfg)
    device = wp.get_device(cfg.device)
    env_config = cfg.env_config or g1_recipe.default_g1_env_config()
    if cfg.randomize_commands and cfg.command_sampling == "episode":
        env_config = replace(
            env_config,
            command_x_range=cfg.command_x_range,
            command_y_range=cfg.command_y_range,
            command_yaw_range=cfg.command_yaw_range,
            randomize_commands_on_reset=True,
            command_zero_probability=cfg.command_zero_probability,
            command_resample_steps=cfg.command_resample_steps,
        )
    elif not cfg.randomize_commands or cfg.command_sampling == "rollout":
        env_config = replace(env_config, randomize_commands_on_reset=False, command_resample_steps=0)
    env = EnvG1PhoenX(env_config, device=device)
    trainer = PufferTorchTrainer(env, cfg)
    history: list[StatsTrainG1PufferTorch] = []
    command_curriculum_counter = wp.array(np.zeros(1, dtype=np.int32), dtype=wp.int32, device=device)
    samples = int(cfg.rollout_steps) * int(env.world_count)
    if cfg.randomize_commands:
        env.update_command_curriculum(
            command_curriculum_counter,
            sample_delta=0,
            start_scale=float(cfg.command_curriculum_start),
            ramp_samples=float(cfg.command_curriculum_samples),
        )
    if cfg.randomize_commands and cfg.command_sampling == "episode":
        env.randomize_commands(
            seed=int(cfg.seed) + 53_321,
            command_x_range=cfg.command_x_range,
            command_y_range=cfg.command_y_range,
            command_yaw_range=cfg.command_yaw_range,
            zero_probability=cfg.command_zero_probability,
        )

    for iteration in range(int(cfg.iterations)):
        if cfg.randomize_commands:
            env.update_command_curriculum(
                command_curriculum_counter,
                sample_delta=samples,
                start_scale=float(cfg.command_curriculum_start),
                ramp_samples=float(cfg.command_curriculum_samples),
            )
        if cfg.randomize_commands and cfg.command_sampling == "rollout":
            env.randomize_commands(
                seed=int(cfg.seed) + 53_321 + iteration,
                command_x_range=cfg.command_x_range,
                command_y_range=cfg.command_y_range,
                command_yaw_range=cfg.command_yaw_range,
                zero_probability=cfg.command_zero_probability,
            )
        t0 = time.perf_counter()
        reward, done, perf = trainer.collect_rollout()
        t1 = time.perf_counter()
        losses = trainer.update()
        trainer.torch.cuda.synchronize(trainer.device)
        t2 = time.perf_counter()
        stats = StatsTrainG1PufferTorch(
            iteration=iteration,
            mean_reward=reward,
            mean_done=done,
            mean_tracking_perf=perf,
            policy_loss=float(losses["policy_loss"]),
            value_loss=float(losses["value_loss"]),
            approx_kl=float(losses["approx_kl"]),
            clip_fraction=float(losses["clipfrac"]),
            rollout_seconds=t1 - t0,
            update_seconds=t2 - t1,
            samples_per_second=float(samples) / max(t2 - t0, 1.0e-12),
        )
        history.append(stats)
        if cfg.log_interval > 0 and (iteration % int(cfg.log_interval) == 0 or iteration == int(cfg.iterations) - 1):
            print(
                f"iter={iteration:04d} reward={stats.mean_reward:.4f} perf={stats.mean_tracking_perf:.3f} "
                f"done={stats.mean_done:.4f} sps={stats.samples_per_second:.1f} "
                f"pi_loss={stats.policy_loss:.4f} v_loss={stats.value_loss:.4f}"
            )
        if cfg.checkpoint_path is not None and int(cfg.checkpoint_interval) > 0:
            if (iteration + 1) % int(cfg.checkpoint_interval) == 0:
                trainer.save_checkpoint(_format_checkpoint_path(cfg.checkpoint_path, iteration + 1))

    if cfg.checkpoint_path is not None:
        trainer.save_checkpoint(_format_checkpoint_path(cfg.checkpoint_path, int(cfg.iterations)))
    return ResultTrainG1PufferTorch(trainer=trainer, env=env, history=history)


def evaluate_g1_gate_puffer_torch(
    trainer: PufferTorchTrainer, config: ConfigEvaluateG1GatePPO | None = None
) -> ResultEvaluateG1GatePPO:
    """Evaluate a PufferLib-style Torch policy with the existing G1 quality gate."""

    cfg = config or ConfigEvaluateG1GatePPO()
    _validate_g1_gate_config(cfg)
    commands = _g1_gate_commands_array(cfg.battery_commands)
    device = wp.get_device(cfg.device or trainer.env.device)
    env_config = cfg.env_config or replace(
        trainer.env.config, world_count=commands.shape[0] * int(cfg.seeds_per_command)
    )
    battery_env = EnvG1PhoenX(env_config, device=device)
    battery_adapter = _TorchGateAdapter(trainer, battery_env.world_count)
    t0 = time.perf_counter()
    per_command, battery_falls, battery_perf, battery_samples = _evaluate_g1_gate_battery_torch(
        battery_adapter, battery_env, cfg, commands
    )
    diagnostic_env = EnvG1PhoenX(
        replace(env_config, world_count=int(cfg.diagnostic_world_count), command=cfg.diagnostic_command),
        device=device,
    )
    diagnostic_adapter = _TorchGateAdapter(trainer, diagnostic_env.world_count)
    diagnostic = _evaluate_g1_gate_diagnostics_torch(diagnostic_adapter, diagnostic_env, cfg)
    elapsed = time.perf_counter() - t0
    diagnostic_falls, diagnostic_samples, action_jerk, ang_vel_xy, yaw_rate, leg_qvel = diagnostic
    pass_gate = _g1_gate_passes(
        cfg,
        battery_falls=battery_falls,
        battery_perf=battery_perf,
        action_jerk_rms=action_jerk,
        ang_vel_xy_rms=ang_vel_xy,
        yaw_rate_rms=yaw_rate,
        leg_qvel_rms=leg_qvel,
    )
    samples = int(battery_samples) + int(diagnostic_samples)
    return ResultEvaluateG1GatePPO(
        stats=StatsEvaluateG1GatePPO(
            battery_falls=int(battery_falls),
            battery_perf=float(battery_perf),
            action_jerk_rms=float(action_jerk),
            ang_vel_xy_rms=float(ang_vel_xy),
            yaw_rate_rms=float(yaw_rate),
            leg_qvel_rms=float(leg_qvel),
            diagnostic_falls=int(diagnostic_falls),
            battery_samples=int(battery_samples),
            diagnostic_samples=int(diagnostic_samples),
            samples_per_second=float(samples) / max(elapsed, 1.0e-12),
            pass_gate=bool(pass_gate),
            per_command=tuple(per_command),
        )
    )


def _evaluate_g1_gate_battery_torch(
    trainer: _TorchGateAdapter,
    env: EnvG1PhoenX,
    cfg: ConfigEvaluateG1GatePPO,
    commands: np.ndarray,
) -> tuple[tuple[StatsEvaluateG1GateCommandPPO, ...], int, float, int]:
    command_ids = np.repeat(np.arange(commands.shape[0], dtype=np.int32), int(cfg.seeds_per_command))
    command_np = commands[command_ids].astype(np.float32, copy=False)
    samples_per_step = np.bincount(command_ids, minlength=commands.shape[0]).astype(np.int64)
    falls = np.zeros(commands.shape[0], dtype=np.int64)
    perf_sum = np.zeros(commands.shape[0], dtype=np.float64)
    lin_err_sum = np.zeros(commands.shape[0], dtype=np.float64)
    yaw_err_sum = np.zeros(commands.shape[0], dtype=np.float64)
    sample_count = np.zeros(commands.shape[0], dtype=np.int64)

    env.set_commands(command_np)
    obs = env.reset_noisy(seed=int(cfg.seed))
    trainer.reset_rollout_state()
    for step in range(int(cfg.battery_steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        perf_np = env.step_successes.numpy().astype(np.float64)
        q = _joint_q_matrix_g1(env)
        qd = _joint_qd_matrix_g1(env)
        lin_b = _quat_rotate_inverse_xyzw_np(q[:, 3:7], qd[:, 0:3])
        lin_err = np.linalg.norm(command_np[:, 0:2] - lin_b[:, 0:2], axis=1)
        yaw_err = np.abs(command_np[:, 2] - qd[:, 5])

        falls += np.bincount(command_ids, weights=done_np.astype(np.float64), minlength=commands.shape[0]).astype(
            np.int64
        )
        perf_sum += np.bincount(command_ids, weights=perf_np, minlength=commands.shape[0])
        lin_err_sum += np.bincount(command_ids, weights=lin_err, minlength=commands.shape[0])
        yaw_err_sum += np.bincount(command_ids, weights=yaw_err, minlength=commands.shape[0])
        sample_count += samples_per_step
        if np.any(done_np):
            trainer.reset_rollout_state(dones)
            obs = _reset_g1_done_worlds(env)

    stats = []
    for command_index, command in enumerate(commands):
        denom = float(max(int(sample_count[command_index]), 1))
        stats.append(
            StatsEvaluateG1GateCommandPPO(
                command=(float(command[0]), float(command[1]), float(command[2])),
                falls=int(falls[command_index]),
                mean_tracking_perf=float(perf_sum[command_index] / denom),
                mean_linear_velocity_error=float(lin_err_sum[command_index] / denom),
                mean_yaw_rate_error=float(yaw_err_sum[command_index] / denom),
                samples=int(sample_count[command_index]),
            )
        )
    total_samples = int(np.sum(sample_count))
    return tuple(stats), int(np.sum(falls)), float(np.sum(perf_sum) / float(max(total_samples, 1))), total_samples


def _evaluate_g1_gate_diagnostics_torch(
    trainer: _TorchGateAdapter, env: EnvG1PhoenX, cfg: ConfigEvaluateG1GatePPO
) -> tuple[int, int, float, float, float, float]:
    obs = env.reset()
    trainer.reset_rollout_state()
    falls = 0
    valid_samples = 0
    jerk_sum = 0.0
    jerk_count = 0
    ang_vel_xy_sum = 0.0
    yaw_rate_sum = 0.0
    leg_qvel_sum = 0.0
    leg_qvel_count = 0
    previous_actions = np.zeros((env.world_count, 12), dtype=np.float32)
    has_previous = np.zeros(env.world_count, dtype=bool)

    for step in range(int(cfg.diagnostic_steps)):
        actions, _log_probs, _values = trainer.act(
            obs, seed=int(cfg.seed) + 1_000_003 + step, deterministic=bool(cfg.deterministic)
        )
        obs, _rewards, dones = env.step(actions)
        done_np = dones.numpy() > 0.5
        valid = ~done_np
        falls += int(np.sum(done_np))
        actions_np = env.current_actions.numpy()[:, :12]
        jerk_worlds = valid & has_previous
        if np.any(jerk_worlds):
            action_delta = actions_np[jerk_worlds] - previous_actions[jerk_worlds]
            jerk_sum += float(np.sum(action_delta * action_delta))
            jerk_count += int(action_delta.size)
        if np.any(valid):
            previous_actions[valid] = actions_np[valid]
            has_previous[valid] = True
        has_previous[done_np] = False
        qd = _joint_qd_matrix_g1(env)
        if np.any(valid):
            qd_valid = qd[valid]
            ang_vel_xy_sum += float(np.sum(qd_valid[:, 3] * qd_valid[:, 3] + qd_valid[:, 4] * qd_valid[:, 4]))
            yaw_rate_sum += float(np.sum(qd_valid[:, 5] * qd_valid[:, 5]))
            leg_qvel = qd_valid[:, 6:18]
            leg_qvel_sum += float(np.sum(leg_qvel * leg_qvel))
            leg_qvel_count += int(leg_qvel.size)
            valid_samples += int(qd_valid.shape[0])
        if np.any(done_np):
            trainer.reset_rollout_state(dones)
            obs = _reset_g1_done_worlds(env)

    action_jerk = float(np.sqrt(jerk_sum / float(max(jerk_count, 1))))
    ang_vel_xy = float(np.sqrt(ang_vel_xy_sum / float(max(valid_samples, 1))))
    yaw_rate = float(np.sqrt(yaw_rate_sum / float(max(valid_samples, 1))))
    leg_qvel = float(np.sqrt(leg_qvel_sum / float(max(leg_qvel_count, 1))))
    diagnostic_samples = int(env.world_count) * int(cfg.diagnostic_steps)
    return falls, diagnostic_samples, action_jerk, ang_vel_xy, yaw_rate, leg_qvel


def _validate_train_config(cfg: ConfigTrainG1PufferTorch) -> None:
    if int(cfg.iterations) <= 0:
        raise ValueError("iterations must be positive")
    if int(cfg.rollout_steps) <= 1:
        raise ValueError("rollout_steps must be greater than one for Puffer shifted advantages")
    if int(cfg.minibatch_size) <= 0:
        raise ValueError("minibatch_size must be positive")
    if float(cfg.replay_ratio) <= 0.0:
        raise ValueError("replay_ratio must be positive")
    if str(cfg.command_sampling) not in ("episode", "rollout"):
        raise ValueError("command_sampling must be 'episode' or 'rollout'")
    if not 0.0 <= float(cfg.command_curriculum_start) <= 1.0:
        raise ValueError("command_curriculum_start must be in [0, 1]")
    if int(cfg.command_curriculum_samples) < 0:
        raise ValueError("command_curriculum_samples must be non-negative")
    if int(cfg.checkpoint_interval) < 0:
        raise ValueError("checkpoint_interval must be non-negative")


def _format_checkpoint_path(path: str, iteration: int) -> Path:
    if "{" in path:
        return Path(path.format(iteration=int(iteration)))
    return Path(path)
