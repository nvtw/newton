# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PPO trainer for PufferLib on Warp.

Matches C++ PufferLib's architecture: one shared trainer handles all
environments.  Per-environment scripts only supply a :class:`PPOConfig`
and an environment factory.

Four CUDA graphs cover the entire hot path:

1. **Rollout step** — forward + sample + env.step + store (replayed HORIZON times)
2. **Flatten** — rollout flattening (replayed 1× per epoch)
3. **GAE** — advantage estimation with V-Trace (replayed each minibatch)
4. **Minibatch update** — gather + normalize + fwd + loss + bwd + Muon step
   + value/importance writeback (replayed ``num_minibatches`` times)

Scalar arguments that change between replays (seed, timestep, LR) are stored
in device-side 1-element arrays and updated via tiny kernels *inside* the
captured graphs, so the graph topology stays fixed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import warp as wp

from newton._src.pufferlib.network import SimpleMLP
from newton._src.pufferlib.obs_normalizer import ObsNormalizer
from newton._src.pufferlib.optimizer import AdamW, Muon
from newton._src.pufferlib.ppo import (
    _REDUCE_BLOCK_DIM,
    PPOLossBuffers,
    _reduce_logstd_grad_kernel,
    _reduce_loss_columns_kernel,
    gae_vtrace_kernel,
    ppo_loss_continuous_kernel,
    ppo_loss_discrete_kernel,
)
from newton._src.pufferlib.reduce import ArraySum

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# Helper kernels — all device-side scalars read from arrays for graph capture
# ---------------------------------------------------------------------------


@wp.kernel
def _inc_kernel(arr: wp.array(dtype=int, ndim=1)):
    arr[0] = arr[0] + 1


@wp.kernel
def _lookup_lr_kernel(
    schedule: wp.array(dtype=float, ndim=1),
    iteration: wp.array(dtype=int, ndim=1),
    d_lr: wp.array(dtype=float, ndim=1),
    max_idx: int,
):
    """Set d_lr[0] = schedule[min(iteration[0], max_idx)]."""
    idx = wp.min(iteration[0], max_idx)
    d_lr[0] = schedule[idx]


@wp.kernel
def _clamp_1d_kernel(arr: wp.array(dtype=float, ndim=1), lo: float, hi: float):
    i = wp.tid()
    arr[i] = wp.clamp(arr[i], lo, hi)


@wp.kernel
def _reset_epoch_state_kernel(
    d_t: wp.array(dtype=int, ndim=1),
    flat_importance: wp.array(dtype=float, ndim=1),
):
    """Zero timestep counter and fill importance buffer with 1.0."""
    i = wp.tid()
    if i == 0:
        d_t[0] = 0
    flat_importance[i] = 1.0


# ---------------------------------------------------------------------------
# Priority replay kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _prio_adv_reduction_kernel(
    advantages: wp.array2d(dtype=float),
    prio_weights: wp.array(dtype=float, ndim=1),
    prio_alpha: float,
):
    """Per-agent priority = (sum |advantages|)^alpha."""
    n = wp.tid()
    T = advantages.shape[1]
    s = float(0.0)
    for t in range(T):
        s = s + wp.abs(advantages[n, t])
    pw = wp.pow(s, prio_alpha)
    if wp.isnan(pw) or wp.isinf(pw):
        pw = 0.0
    prio_weights[n] = pw


@wp.kernel
def _prio_normalize_kernel(
    prio_weights: wp.array(dtype=float, ndim=1),
    total: wp.array(dtype=float, ndim=1),
):
    """Normalize priorities to probabilities: p[i] = (p[i] + eps) / (sum + eps*N)."""
    i = wp.tid()
    eps = 1.0e-6
    prio_weights[i] = (prio_weights[i] + eps) / (total[0] + eps * float(prio_weights.shape[0]))


@wp.kernel
def _prio_sample_kernel(
    cdf: wp.array(dtype=float, ndim=1),
    out_idx: wp.array(dtype=int, ndim=1),
    B: int,
    seed_arr: wp.array(dtype=int, ndim=1),
):
    """Multinomial sampling with replacement via binary search on CDF."""
    i = wp.tid()
    state = wp.rand_init(seed_arr[0], i + 77777)
    u = wp.randf(state)

    lo = int(0)
    hi = B - 1
    while lo < hi:
        mid = (lo + hi) / 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    out_idx[i] = lo


@wp.kernel
def _prio_expand_segments_kernel(
    seg_indices: wp.array(dtype=int, ndim=1),
    mb_indices: wp.array(dtype=int, ndim=1),
    T: int,
):
    """Expand segment indices to flat timestep indices: seg*T + t."""
    s, t = wp.tid()
    mb_indices[s * T + t] = seg_indices[s] * T + t


@wp.kernel
def _prio_imp_weights_kernel(
    seg_indices: wp.array(dtype=int, ndim=1),
    prio_probs: wp.array(dtype=float, ndim=1),
    prio_weight: wp.array(dtype=float, ndim=1),
    total_agents: int,
    anneal_beta_arr: wp.array(dtype=float, ndim=1),
    T: int,
):
    """Importance weights: (N * prob[seg])^(-beta), broadcast to all T timesteps."""
    s, t = wp.tid()
    anneal_beta = anneal_beta_arr[0]
    val = prio_probs[seg_indices[s]] * float(total_agents)
    w = wp.pow(val, -anneal_beta)
    if wp.isnan(w) or wp.isinf(w):
        w = 1.0
    prio_weight[s * T + t] = w


@wp.kernel
def _extract_value_kernel(
    dec_out: wp.array2d(dtype=float),
    value_col: int,
    values: wp.array(dtype=float, ndim=1),
):
    i = wp.tid()
    values[i] = dec_out[i, value_col]


@wp.kernel
def _store_obs_kernel(
    obs: wp.array2d(dtype=float),
    buf: wp.array(dtype=float, ndim=3),
    t_arr: wp.array(dtype=int, ndim=1),
):
    i, d = wp.tid()
    buf[i, t_arr[0], d] = obs[i, d]


@wp.kernel
def _store_1d_kernel(
    src: wp.array(dtype=float, ndim=1),
    buf: wp.array2d(dtype=float),
    t_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    buf[i, t_arr[0]] = src[i]


@wp.kernel
def _store_actions_kernel(
    src: wp.array2d(dtype=float),
    buf: wp.array(dtype=float, ndim=3),
    t_arr: wp.array(dtype=int, ndim=1),
):
    i, d = wp.tid()
    buf[i, t_arr[0], d] = src[i, d]


@wp.kernel
def _sample_discrete_dev_kernel(
    logits: wp.array2d(dtype=float),
    act_sizes: wp.array(dtype=int, ndim=1),
    num_heads: int,
    actions: wp.array2d(dtype=float),
    logprobs: wp.array(dtype=float, ndim=1),
    seed_arr: wp.array(dtype=int, ndim=1),
):
    agent = wp.tid()
    state = wp.rand_init(seed_arr[0], agent)

    total_logp = float(0.0)
    logit_offset = int(0)

    for h in range(num_heads):
        A = act_sizes[h]

        max_logit = float(-1.0e30)
        for a in range(A):
            val = logits[agent, logit_offset + a]
            if wp.isnan(val) or wp.isinf(val):
                val = 0.0
            if val > max_logit:
                max_logit = val

        exp_sum = float(0.0)
        for a in range(A):
            val = logits[agent, logit_offset + a]
            if wp.isnan(val) or wp.isinf(val):
                val = 0.0
            exp_sum = exp_sum + wp.exp(val - max_logit)
        logsumexp = max_logit + wp.log(exp_sum)

        rand_val = wp.randf(state)
        cumsum = float(0.0)
        sampled = int(0)
        found = int(0)
        for a in range(A):
            val = logits[agent, logit_offset + a]
            if wp.isnan(val) or wp.isinf(val):
                val = 0.0
            prob = wp.exp(val - logsumexp)
            cumsum = cumsum + prob
            if cumsum > rand_val and found == 0:
                sampled = a
                found = 1

        actions[agent, h] = float(sampled)
        sampled_logit = logits[agent, logit_offset + sampled]
        if wp.isnan(sampled_logit) or wp.isinf(sampled_logit):
            sampled_logit = 0.0
        log_prob = sampled_logit - logsumexp
        total_logp = total_logp + log_prob

        logit_offset = logit_offset + A

    logprobs[agent] = total_logp


@wp.kernel
def _sample_continuous_dev_kernel(
    logits: wp.array2d(dtype=float),
    logstd: wp.array(dtype=float, ndim=1),
    num_actions: int,
    actions: wp.array2d(dtype=float),
    logprobs: wp.array(dtype=float, ndim=1),
    seed_arr: wp.array(dtype=int, ndim=1),
):
    """Sample continuous actions from N(mean, exp(logstd)), reading seed from device array."""
    agent = wp.tid()
    state = wp.rand_init(seed_arr[0], agent)

    LOG_2PI = 1.8378770664093453
    total_logp = float(0.0)

    for h in range(num_actions):
        mean = logits[agent, h]
        log_std = wp.clamp(logstd[h], -5.0, 0.5)
        std = wp.exp(log_std)

        noise = wp.randn(state)
        action = mean + std * noise

        normalized = (action - mean) / std
        log_prob = -0.5 * normalized * normalized - 0.5 * LOG_2PI - log_std

        actions[agent, h] = action
        total_logp = total_logp + log_prob

    logprobs[agent] = total_logp


@wp.kernel
def _flatten_obs_kernel(
    buf: wp.array(dtype=float, ndim=3),
    flat: wp.array2d(dtype=float),
    T: int,
    obs_dim: int,
):
    n, t = wp.tid()
    idx = n * T + t
    for d in range(obs_dim):
        flat[idx, d] = buf[n, t, d]


@wp.kernel
def _flatten_2d_kernel(
    buf: wp.array2d(dtype=float),
    flat: wp.array(dtype=float, ndim=1),
    T: int,
):
    n, t = wp.tid()
    flat[n * T + t] = buf[n, t]


@wp.kernel
def _flatten_actions_kernel(
    buf: wp.array(dtype=float, ndim=3),
    flat: wp.array2d(dtype=float),
    T: int,
    act_dim: int,
):
    n, t = wp.tid()
    idx = n * T + t
    for d in range(act_dim):
        flat[idx, d] = buf[n, t, d]


@wp.kernel
def _fill_ones_kernel(arr: wp.array2d(dtype=float)):
    i, j = wp.tid()
    arr[i, j] = 1.0


@wp.kernel
def _clamp_2d_kernel(arr: wp.array2d(dtype=float), lo: float, hi: float):
    i, j = wp.tid()
    arr[i, j] = wp.clamp(arr[i, j], lo, hi)


@wp.kernel
def _iota_kernel(arr: wp.array(dtype=int, ndim=1)):
    i = wp.tid()
    arr[i] = i


@wp.kernel
def _rand_keys_dev_kernel(
    keys: wp.array(dtype=int, ndim=1),
    seed_arr: wp.array(dtype=int, ndim=1),
):
    i = wp.tid()
    state = wp.rand_init(seed_arr[0], i)
    keys[i] = wp.randi(state)


@wp.kernel
def _scatter_values_kernel(
    new_values: wp.array(dtype=float, ndim=1),
    indices: wp.array(dtype=int, ndim=1),
    flat_values: wp.array(dtype=float, ndim=1),
):
    """Write back updated value predictions to the flat rollout buffer."""
    i = wp.tid()
    flat_values[indices[i]] = new_values[i]


@wp.kernel
def _scatter_logprobs_to_importance_kernel(
    new_logprobs: wp.array(dtype=float, ndim=1),
    old_logprobs: wp.array(dtype=float, ndim=1),
    indices: wp.array(dtype=int, ndim=1),
    flat_importance: wp.array(dtype=float, ndim=1),
):
    """Compute importance ratio exp(new - old) and scatter to flat buffer."""
    i = wp.tid()
    idx = indices[i]
    ratio = wp.exp(new_logprobs[i] - old_logprobs[i])
    flat_importance[idx] = ratio


@wp.kernel
def _unflatten_to_2d_kernel(
    flat: wp.array(dtype=float, ndim=1),
    buf: wp.array2d(dtype=float),
    T: int,
):
    """Unflatten a 1D array back to (N, T)."""
    n, t = wp.tid()
    buf[n, t] = flat[n * T + t]


@wp.kernel
def _compute_cont_logprobs_kernel(
    logits: wp.array2d(dtype=float),
    logstd: wp.array(dtype=float, ndim=1),
    actions: wp.array2d(dtype=float),
    num_actions: int,
    logprobs: wp.array(dtype=float, ndim=1),
):
    """Compute log probabilities for continuous actions from current policy output."""
    i = wp.tid()
    LOG_2PI = 1.8378770664093453
    total = float(0.0)
    for h in range(num_actions):
        mean = logits[i, h]
        log_std = wp.clamp(logstd[h], -5.0, 0.5)
        std = wp.exp(log_std)
        diff = actions[i, h] - mean
        normalized = diff / std
        total = total + (-0.5 * normalized * normalized - 0.5 * LOG_2PI - log_std)
    logprobs[i] = total


@wp.kernel
def _compute_disc_logprobs_kernel(
    logits: wp.array2d(dtype=float),
    actions: wp.array2d(dtype=float),
    act_sizes: wp.array(dtype=int, ndim=1),
    num_heads: int,
    logprobs: wp.array(dtype=float, ndim=1),
):
    """Compute log probabilities for discrete actions from current policy output."""
    i = wp.tid()
    total = float(0.0)
    offset = int(0)
    for h in range(num_heads):
        A = act_sizes[h]
        max_val = float(-1.0e30)
        for a in range(A):
            v = logits[i, offset + a]
            if v > max_val:
                max_val = v
        exp_sum = float(0.0)
        for a in range(A):
            exp_sum = exp_sum + wp.exp(logits[i, offset + a] - max_val)
        lse = max_val + wp.log(exp_sum)
        sampled = int(actions[i, h])
        total = total + logits[i, offset + sampled] - lse
        offset = offset + A
    logprobs[i] = total


@wp.kernel
def _merge_decoder_grads_kernel(
    grad_logits: wp.array2d(dtype=float),
    grad_values: wp.array(dtype=float, ndim=1),
    grad_dec: wp.array2d(dtype=float),
    num_actions: int,
):
    i, j = wp.tid()
    if j < num_actions:
        grad_dec[i, j] = grad_logits[i, j]
    elif j == num_actions:
        grad_dec[i, j] = grad_values[i]


@wp.kernel
def _gather_2d_kernel(
    src: wp.array2d(dtype=float),
    indices: wp.array(dtype=int, ndim=1),
    dst: wp.array2d(dtype=float),
):
    s, j = wp.tid()
    dst[s, j] = src[indices[s], j]


@wp.kernel
def _gather_1d_kernel(
    src: wp.array(dtype=float, ndim=1),
    indices: wp.array(dtype=int, ndim=1),
    dst: wp.array(dtype=float, ndim=1),
):
    s = wp.tid()
    dst[s] = src[indices[s]]


@wp.kernel
def _normalize_adv_kernel(
    adv: wp.array(dtype=float, ndim=1),
    sum_val: wp.array(dtype=float, ndim=1),
    sq_sum_val: wp.array(dtype=float, ndim=1),
    count: float,
):
    i = wp.tid()
    m = sum_val[0] / count
    v = (sq_sum_val[0] / count - m * m) * count / (count - 1.0)
    std = wp.sqrt(v)
    adv[i] = (adv[i] - m) / (std + 1.0e-8)


@wp.kernel
def _sq_kernel(src: wp.array(dtype=float, ndim=1), dst: wp.array(dtype=float, ndim=1)):
    i = wp.tid()
    dst[i] = src[i] * src[i]


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """All hyperparameters for a PPO training run.

    Environment-specific values (obs_dim, num_actions, etc.) must be provided.
    PPO/optimizer defaults match common PufferLib settings.
    """

    # Environment geometry
    num_envs: int = 4096
    horizon: int = 64
    obs_dim: int = 4
    num_actions: int = 2
    hidden: int = 128

    # Training budget
    total_timesteps: int = 10_000_000
    seed: int = 42

    # Learning rate
    lr: float = 0.005
    anneal_lr: bool = True
    min_lr_ratio: float = 0.0

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 2.0
    vf_clip_coef: float = 0.2
    ent_coef: float = 0.01
    rho_clip: float = 1.0
    c_clip: float = 1.0

    # Optimizer
    optimizer: str = "muon"  # "muon" or "adamw"
    max_grad_norm: float = 1.5
    momentum: float = 0.95
    replay_ratio: float = 1.0
    minibatch_size: int = 32768

    # Priority replay
    prio_alpha: float = 0.0
    prio_beta0: float = 1.0

    # Action space
    continuous: bool = False
    init_logstd: float = 0.0

    # Observation normalization
    normalize_obs: bool = False

    # Reward clamping (applied during flatten)
    reward_clamp: float = 1.0

    # Display
    env_name: str = "Environment"
    best_return_init: float = -100.0
    banner_extras: dict[str, Any] = field(default_factory=dict)
    return_format: str = "7.1f"
    step_width: int = 9
    log_interval: int = 5

    # Optional callbacks
    format_return_fn: Callable | None = None
    log_condition_fn: Callable | None = None
    format_summary_fn: Callable | None = None
    checkpoint_fn: Callable | None = None  # checkpoint_fn(policy, iteration, total_steps, best_return, obs_normalizer)

    # Device
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Shared PPO trainer matching C++ PufferLib architecture.

    Args:
        config: Hyperparameters.
        make_env: Factory ``make_env(device) -> env``.  The returned env must
            expose ``obs``, ``rewards``, ``dones`` (Warp arrays), plus
            ``step_graphed(actions, d_seed)``, ``get_episode_stats()``, and
            ``reset()``.
    """

    def __init__(self, config: PPOConfig, make_env: Callable):
        self.cfg = config
        self.make_env = make_env

    def train(self):
        cfg = self.cfg
        device = cfg.device

        N = cfg.num_envs
        T = cfg.horizon
        OBS = cfg.obs_dim
        ACT = cfg.num_actions
        H = cfg.hidden
        MB = cfg.minibatch_size

        batch_size = N * T
        total_minibatches = max(1, int(cfg.replay_ratio * batch_size / MB))
        num_minibatches = total_minibatches
        flat_size = batch_size
        total_epochs = cfg.total_timesteps // batch_size
        is_cont = cfg.continuous

        # --- Banner ---
        print("=" * 60)
        print(f"PufferLib {cfg.env_name} on Warp — PPO + Muon (4 CUDA Graphs)")
        print("=" * 60)
        for k, v in cfg.banner_extras.items():
            print(f"  {k + ':':15s}{v}")
        print(f"  {'Environments:':15s}{N}")
        print(f"  {'Horizon:':15s}{T}")
        print(f"  {'Hidden size:':15s}{H}")
        print(f"  {'Total steps:':15s}{cfg.total_timesteps:,}")
        print(f"  {'Minibatches/ep:':15s}{num_minibatches} x {MB}")
        print(f"  {'Action space:':15s}{'continuous' if is_cont else 'discrete'} ({ACT})")
        print(f"  {'Optimizer:':15s}Muon (momentum={cfg.momentum}, lr={cfg.lr})")
        print(f"  {'Device:':15s}{device}")
        print()

        env = self.make_env(device)
        policy = SimpleMLP(OBS, H, ACT + 1,
                           max_batch=max(N, MB), device=device, seed=cfg.seed,
                           continuous=is_cont, num_actions=ACT,
                           init_logstd=cfg.init_logstd)
        params = policy.parameters()

        d_lr = wp.array([cfg.lr], dtype=float, device=device)
        if cfg.optimizer == "adamw":
            optimizer = AdamW(
                params, lr=cfg.lr, weight_decay=0.0,
                max_grad_norm=cfg.max_grad_norm,
                device_lr=d_lr,
            )
        else:
            optimizer = Muon(
                params, lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=0.0, max_grad_norm=cfg.max_grad_norm,
                device_lr=d_lr,
            )

        obs_normalizer = ObsNormalizer(OBS, device) if cfg.normalize_obs else None
        norm_obs = wp.zeros((max(N, MB), OBS), dtype=float, device=device) if cfg.normalize_obs else None

        act_sizes = wp.array([ACT], dtype=int, device=device)
        act_dim = ACT if is_cont else 1

        d_seed = wp.array([cfg.seed], dtype=int, device=device)
        d_t = wp.array([0], dtype=int, device=device)

        # --- Pre-allocate ALL buffers ---
        obs_buf = wp.zeros((N, T, OBS), dtype=float, device=device)
        act_buf = wp.zeros((N, T, act_dim), dtype=float, device=device)
        logp_buf = wp.zeros((N, T), dtype=float, device=device)
        rew_buf = wp.zeros((N, T), dtype=float, device=device)
        done_buf = wp.zeros((N, T), dtype=float, device=device)
        val_buf = wp.zeros((N, T), dtype=float, device=device)
        importance_buf = wp.zeros((N, T), dtype=float, device=device)
        advantages = wp.zeros((N, T), dtype=float, device=device)
        returns = wp.zeros((N, T), dtype=float, device=device)

        values_scratch = wp.zeros(N, dtype=float, device=device)
        actions_scratch = wp.zeros((N, act_dim), dtype=float, device=device)
        logprobs_scratch = wp.zeros(N, dtype=float, device=device)
        action_flat = wp.zeros(N, dtype=float, device=device) if not is_cont else None

        flat_obs = wp.zeros((flat_size, OBS), dtype=float, device=device)
        flat_actions = wp.zeros((flat_size, act_dim), dtype=float, device=device)
        flat_logprobs = wp.zeros(flat_size, dtype=float, device=device)
        flat_advantages = wp.zeros(flat_size, dtype=float, device=device)
        flat_values = wp.zeros(flat_size, dtype=float, device=device)
        flat_returns = wp.zeros(flat_size, dtype=float, device=device)

        sort_keys = wp.zeros(2 * flat_size, dtype=int, device=device)
        sort_vals = wp.zeros(2 * flat_size, dtype=int, device=device)

        mb_obs = wp.zeros((MB, OBS), dtype=float, device=device)
        mb_actions = wp.zeros((MB, act_dim), dtype=float, device=device)
        mb_logprobs = wp.zeros(MB, dtype=float, device=device)
        mb_advantages_buf = wp.zeros(MB, dtype=float, device=device)
        mb_old_values = wp.zeros(MB, dtype=float, device=device)
        mb_returns_buf = wp.zeros(MB, dtype=float, device=device)
        mb_values_pred = wp.zeros(MB, dtype=float, device=device)
        mb_new_logprobs = wp.zeros(MB, dtype=float, device=device)
        grad_dec = wp.zeros((MB, ACT + 1), dtype=float, device=device)

        loss_bufs = PPOLossBuffers(MB, ACT + 1, device=device,
                                   continuous=is_cont, num_actions=ACT)
        adv_summer = ArraySum(MB, device=device)
        adv_sq_buf = wp.zeros(MB, dtype=float, device=device)
        adv_sq_summer = ArraySum(MB, device=device)

        # Priority replay buffers
        use_prio = cfg.prio_alpha > 0.0
        mb_segments = MB // T
        prio_weights = wp.zeros(N, dtype=float, device=device)
        prio_cdf = wp.zeros(N, dtype=float, device=device)
        prio_seg_idx = wp.zeros(mb_segments, dtype=int, device=device)
        prio_mb_indices = wp.zeros(MB, dtype=int, device=device)
        prio_sum = ArraySum(N, device=device)
        d_anneal_beta = wp.array([cfg.prio_beta0], dtype=float, device=device)

        # --- Inner functions (closures over local state) ---

        def _run_rollout_step():
            wp.launch(_store_obs_kernel, dim=(N, OBS),
                      inputs=[env.obs, obs_buf, d_t], device=device)
            if obs_normalizer is not None:
                obs_normalizer.update(env.obs, N)
                obs_normalizer.normalize(env.obs, norm_obs, N)
                policy.forward(norm_obs, N)
            else:
                policy.forward(env.obs, N)
            wp.launch(_extract_value_kernel, dim=N,
                      inputs=[policy._out, ACT, values_scratch], device=device)
            if is_cont:
                wp.launch(_sample_continuous_dev_kernel, dim=N,
                          inputs=[policy._out, policy.logstd, ACT,
                                  actions_scratch, logprobs_scratch, d_seed],
                          device=device)
            else:
                wp.launch(_sample_discrete_dev_kernel, dim=N,
                          inputs=[policy._out, act_sizes, 1,
                                  actions_scratch, logprobs_scratch, d_seed],
                          device=device)
            wp.launch(_store_actions_kernel, dim=(N, act_dim),
                      inputs=[actions_scratch, act_buf, d_t], device=device)
            wp.launch(_store_1d_kernel, dim=N,
                      inputs=[logprobs_scratch, logp_buf, d_t], device=device)
            wp.launch(_store_1d_kernel, dim=N,
                      inputs=[values_scratch, val_buf, d_t], device=device)
            if is_cont:
                env.step_graphed(actions_scratch, d_seed)
            else:
                wp.copy(action_flat, actions_scratch.flatten(), count=N)
                env.step_graphed(action_flat, d_seed)
            wp.launch(_store_1d_kernel, dim=N,
                      inputs=[env.rewards, rew_buf, d_t], device=device)
            wp.launch(_store_1d_kernel, dim=N,
                      inputs=[env.dones, done_buf, d_t], device=device)
            wp.launch(_inc_kernel, dim=1, inputs=[d_t], device=device)
            wp.launch(_inc_kernel, dim=1, inputs=[d_seed], device=device)

        flat_importance = wp.ones(flat_size, dtype=float, device=device)

        def _run_flatten():
            """Flatten obs, actions, logprobs from (N,T) rollout buffers.  Run once per epoch."""
            wp.launch(_clamp_2d_kernel, dim=(N, T),
                      inputs=[rew_buf, -cfg.reward_clamp, cfg.reward_clamp], device=device)
            wp.launch(_flatten_obs_kernel, dim=(N, T),
                      inputs=[obs_buf, flat_obs, T, OBS], device=device)
            wp.launch(_flatten_actions_kernel, dim=(N, T),
                      inputs=[act_buf, flat_actions, T, act_dim], device=device)
            wp.launch(_flatten_2d_kernel, dim=(N, T),
                      inputs=[logp_buf, flat_logprobs, T], device=device)
            wp.launch(_flatten_2d_kernel, dim=(N, T),
                      inputs=[val_buf, flat_values, T], device=device)

        def _run_gae():
            """Recompute GAE from (N,T) buffers using current values + importance.  Run each minibatch."""
            wp.launch(_unflatten_to_2d_kernel, dim=(N, T),
                      inputs=[flat_values, val_buf, T], device=device)
            wp.launch(_unflatten_to_2d_kernel, dim=(N, T),
                      inputs=[flat_importance, importance_buf, T], device=device)
            wp.launch(gae_vtrace_kernel, dim=N,
                      inputs=[val_buf, rew_buf, done_buf, importance_buf,
                              advantages, returns,
                              cfg.gamma, cfg.gae_lambda, cfg.rho_clip, cfg.c_clip],
                      device=device)
            wp.launch(_flatten_2d_kernel, dim=(N, T),
                      inputs=[advantages, flat_advantages, T], device=device)
            wp.launch(_flatten_2d_kernel, dim=(N, T),
                      inputs=[returns, flat_returns, T], device=device)

        def _run_minibatch_update():
            wp.launch(_inc_kernel, dim=1, inputs=[d_seed], device=device)

            if use_prio:
                # Priority replay: sample segments proportional to |advantage|^alpha
                wp.launch(_prio_adv_reduction_kernel, dim=N,
                          inputs=[advantages, prio_weights, cfg.prio_alpha],
                          device=device)
                prio_total = prio_sum.compute(prio_weights, N)
                wp.launch(_prio_normalize_kernel, dim=N,
                          inputs=[prio_weights, prio_total], device=device)
                wp.utils.array_scan(prio_weights, prio_cdf, inclusive=True)
                wp.launch(_prio_sample_kernel, dim=mb_segments,
                          inputs=[prio_cdf, prio_seg_idx, N, d_seed],
                          device=device)
                wp.launch(_prio_expand_segments_kernel, dim=(mb_segments, T),
                          inputs=[prio_seg_idx, prio_mb_indices, T],
                          device=device)
                wp.launch(_prio_imp_weights_kernel, dim=(mb_segments, T),
                          inputs=[prio_seg_idx, prio_weights,
                                  loss_bufs.prio_weight, N, d_anneal_beta, T],
                          device=device)
                mb_indices = prio_mb_indices
            else:
                # Uniform random shuffle via radix sort
                wp.launch(_rand_keys_dev_kernel, dim=flat_size,
                          inputs=[sort_keys, d_seed], device=device)
                wp.launch(_iota_kernel, dim=flat_size, inputs=[sort_vals], device=device)
                wp.utils.radix_sort_pairs(sort_keys, sort_vals, flat_size)
                mb_indices = sort_vals

            wp.launch(_gather_2d_kernel, dim=(MB, OBS),
                      inputs=[flat_obs, mb_indices, mb_obs], device=device)
            wp.launch(_gather_2d_kernel, dim=(MB, act_dim),
                      inputs=[flat_actions, mb_indices, mb_actions], device=device)
            wp.launch(_gather_1d_kernel, dim=MB,
                      inputs=[flat_logprobs, mb_indices, mb_logprobs], device=device)
            wp.launch(_gather_1d_kernel, dim=MB,
                      inputs=[flat_advantages, mb_indices, mb_advantages_buf], device=device)
            wp.launch(_gather_1d_kernel, dim=MB,
                      inputs=[flat_values, mb_indices, mb_old_values], device=device)
            wp.launch(_gather_1d_kernel, dim=MB,
                      inputs=[flat_returns, mb_indices, mb_returns_buf], device=device)

            adv_sum = adv_summer.compute(mb_advantages_buf, MB)
            wp.launch(_sq_kernel, dim=MB,
                      inputs=[mb_advantages_buf, adv_sq_buf], device=device)
            adv_sq_sum = adv_sq_summer.compute(adv_sq_buf, MB)
            wp.launch(_normalize_adv_kernel, dim=MB,
                      inputs=[mb_advantages_buf, adv_sum, adv_sq_sum, float(MB)],
                      device=device)

            if obs_normalizer is not None:
                obs_normalizer.normalize(mb_obs, norm_obs, MB)
                dec_out = policy.forward(norm_obs, MB)
            else:
                dec_out = policy.forward(mb_obs, MB)
            wp.launch(_extract_value_kernel, dim=MB,
                      inputs=[dec_out, ACT, mb_values_pred], device=device)

            if is_cont:
                wp.launch(ppo_loss_continuous_kernel, dim=MB,
                          inputs=[dec_out, policy.logstd, mb_actions, mb_logprobs,
                                  mb_advantages_buf,
                                  mb_values_pred, mb_old_values, mb_returns_buf,
                                  ACT,
                                  cfg.clip_coef, cfg.vf_clip_coef, cfg.vf_coef, cfg.ent_coef,
                                  1.0 / float(MB),
                                  loss_bufs.prio_weight,
                                  loss_bufs.grad_logits, loss_bufs.grad_logstd_scratch,
                                  loss_bufs.grad_values,
                                  loss_bufs.loss_per_thread],
                          device=device)
                wp.launch_tiled(_reduce_logstd_grad_kernel, dim=[ACT],
                                inputs=[loss_bufs.grad_logstd_scratch,
                                        loss_bufs.grad_logstd, MB],
                                block_dim=_REDUCE_BLOCK_DIM, device=device)
            else:
                num_heads = act_sizes.shape[0]
                wp.launch(ppo_loss_discrete_kernel, dim=MB,
                          inputs=[dec_out, mb_actions, mb_logprobs, mb_advantages_buf,
                                  mb_values_pred, mb_old_values, mb_returns_buf,
                                  act_sizes, num_heads,
                                  cfg.clip_coef, cfg.vf_clip_coef, cfg.vf_coef, cfg.ent_coef,
                                  1.0 / float(MB),
                                  loss_bufs.prio_weight,
                                  loss_bufs.grad_logits, loss_bufs.grad_values,
                                  loss_bufs.loss_per_thread],
                          device=device)
            wp.launch_tiled(_reduce_loss_columns_kernel, dim=[7],
                            inputs=[loss_bufs.loss_per_thread, loss_bufs.loss_reduced, MB],
                            block_dim=_REDUCE_BLOCK_DIM, device=device)

            wp.launch(_merge_decoder_grads_kernel, dim=(MB, ACT + 1),
                      inputs=[loss_bufs.grad_logits, loss_bufs.grad_values, grad_dec, ACT],
                      device=device)

            # Compute new logprobs BEFORE optimizer step (matches C++ which
            # writes back ratio from the current forward pass, not post-update)
            if is_cont:
                wp.launch(_compute_cont_logprobs_kernel, dim=MB,
                          inputs=[dec_out, policy.logstd, mb_actions, ACT,
                                  mb_new_logprobs],
                          device=device)
            else:
                wp.launch(_compute_disc_logprobs_kernel, dim=MB,
                          inputs=[dec_out, mb_actions, act_sizes, 1,
                                  mb_new_logprobs],
                          device=device)

            grads = policy.backward(grad_dec, MB)
            if is_cont:
                grads.append(loss_bufs.grad_logstd)
            optimizer.step(grads)

            # Clamp logstd parameter to prevent entropy explosion
            if is_cont and policy.logstd is not None:
                wp.launch(_clamp_1d_kernel, dim=ACT,
                          inputs=[policy.logstd, -5.0, 0.5], device=device)

            # Write back updated values and importance ratios to flat buffers
            # so the next minibatch's GAE uses fresh estimates (matches C++)
            wp.launch(_scatter_values_kernel, dim=MB,
                      inputs=[mb_values_pred, mb_indices, flat_values],
                      device=device)
            wp.launch(_scatter_logprobs_to_importance_kernel, dim=MB,
                      inputs=[mb_new_logprobs, mb_logprobs, mb_indices,
                              flat_importance],
                      device=device)

        # --- Warmup + CUDA graph capture ---
        print("Warming up kernels...", flush=True)

        saved_weights = [wp.empty_like(p) for p in params]
        opt_state = getattr(optimizer, "momentum_buffers", None) or (optimizer.m + optimizer.v)
        saved_opt = [wp.empty_like(m) for m in opt_state]
        for dst, src in zip(saved_weights, params):
            wp.copy(dst, src)
        for dst, src in zip(saved_opt, opt_state):
            wp.copy(dst, src)

        _run_rollout_step()
        _run_flatten()
        _run_gae()
        _run_minibatch_update()
        wp.synchronize_device(device)

        print("Capturing CUDA graphs...", flush=True)

        with wp.ScopedCapture(device=device) as rollout_cap:
            _run_rollout_step()
        rollout_graph = rollout_cap.graph

        with wp.ScopedCapture(device=device) as flatten_cap:
            _run_flatten()
        flatten_graph = flatten_cap.graph

        with wp.ScopedCapture(device=device) as gae_cap:
            _run_gae()
        gae_graph = gae_cap.graph

        with wp.ScopedCapture(device=device) as mb_cap:
            _run_minibatch_update()
        mb_graph = mb_cap.graph

        for src, dst in zip(saved_weights, params):
            wp.copy(dst, src)
        for src, dst in zip(saved_opt, opt_state):
            wp.copy(dst, src)
        del saved_weights, saved_opt

        # --- Pre-compute LR schedule as a DEVICE array (zero host work in hot loop) ---
        lr_schedule_np = np.empty(max(total_epochs + 1, 1), dtype=np.float32)
        for i in range(len(lr_schedule_np)):
            ratio = min(max(i / max(total_epochs, 1), 0.0), 1.0)
            lr_min = cfg.min_lr_ratio * cfg.lr
            lr_schedule_np[i] = lr_min + 0.5 * (cfg.lr - lr_min) * (1.0 + math.cos(math.pi * ratio))
        d_lr_schedule = wp.array(lr_schedule_np, dtype=float, device=device)
        d_iteration = wp.array([0], dtype=int, device=device)
        lr_max_idx = len(lr_schedule_np) - 1

        # --- Capture per-epoch-start graph: reset d_t, fill importance, advance LR ---
        def _run_epoch_start():
            wp.launch(_reset_epoch_state_kernel, dim=flat_size,
                      inputs=[d_t, flat_importance], device=device)
            if cfg.anneal_lr:
                wp.launch(_lookup_lr_kernel, dim=1,
                          inputs=[d_lr_schedule, d_iteration, d_lr, lr_max_idx],
                          device=device)
            wp.launch(_inc_kernel, dim=1, inputs=[d_iteration], device=device)

        _run_epoch_start()
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as epoch_start_cap:
            _run_epoch_start()
        epoch_start_graph = epoch_start_cap.graph

        env.reset()
        d_t.zero_()
        d_seed.fill_(cfg.seed)
        d_iteration.zero_()
        d_lr.fill_(cfg.lr)
        wp.synchronize_device(device)

        print("Graphs captured. Training...\n", flush=True)

        # --- Main training loop ---
        # ZERO host allocations or host↔device transfers in the hot path.
        # Only sync to host at log intervals for stats/loss readback.
        total_steps = 0
        iteration = 0
        start_time = time.time()
        best_return = cfg.best_return_init
        recent_returns: list[float] = []

        ret_fmt = cfg.return_format
        step_w = cfg.step_width
        log_interval = cfg.log_interval
        log_cond = cfg.log_condition_fn
        fmt_ret = cfg.format_return_fn
        fmt_summary = cfg.format_summary_fn

        while total_steps < cfg.total_timesteps:
            iteration += 1

            # All GPU — epoch start: reset d_t, importance, advance LR
            wp.capture_launch(epoch_start_graph)

            # All GPU — rollout T steps
            for _ in range(T):
                wp.capture_launch(rollout_graph)

            total_steps += batch_size

            # All GPU — flatten + minibatch updates
            wp.capture_launch(flatten_graph)
            for _mb in range(num_minibatches):
                wp.capture_launch(gae_graph)
                wp.capture_launch(mb_graph)

            # --- Only sync with host at log intervals ---
            should_log = total_steps >= cfg.total_timesteps
            if not should_log:
                if log_cond is not None:
                    should_log = log_cond(iteration, total_steps)
                else:
                    should_log = (iteration % log_interval == 0)

            if should_log:
                stats = env.get_episode_stats()
                if stats["num_episodes"] > 0:
                    recent_returns.append(stats["mean_return"])
                    if len(recent_returns) > 20:
                        recent_returns = recent_returns[-20:]
                    best_return = max(best_return, np.mean(recent_returns))

                elapsed = time.time() - start_time
                sps = total_steps / elapsed
                loss_np = loss_bufs.loss_reduced.numpy()
                avg_ret = np.mean(recent_returns) if recent_returns else 0.0

                if fmt_ret is not None:
                    ret_str = fmt_ret(avg_ret, best_return, loss_np, sps)
                else:
                    ret_str = (
                        f"Step {total_steps:>{step_w},} | "
                        f"Return {avg_ret:{ret_fmt}} | "
                        f"Best {best_return:{ret_fmt}} | "
                        f"PG {loss_np[1]:8.4f} | "
                        f"VF {loss_np[2]:8.4f} | "
                        f"Ent {loss_np[3]:7.4f} | "
                        f"SPS {sps:,.0f}"
                    )
                print(ret_str)

                if cfg.checkpoint_fn is not None:
                    cfg.checkpoint_fn(policy, iteration, total_steps, best_return, obs_normalizer)

        elapsed = time.time() - start_time
        avg_ret = np.mean(recent_returns) if recent_returns else 0.0

        if fmt_summary is not None:
            summary = fmt_summary(total_steps, best_return, avg_ret, elapsed)
        else:
            print()
            print("=" * 60)
            print(f"Training complete in {elapsed:.1f}s")
            print(f"  Total steps:  {total_steps:,}")
            print(f"  Best return:  {best_return:{ret_fmt}}")
            print(f"  Final return: {avg_ret:{ret_fmt}}")
            print(f"  Throughput:   {total_steps / elapsed:,.0f} steps/sec")
            print("=" * 60)
            summary = None

        if summary is not None:
            print(summary)

        return {"policy": policy, "obs_normalizer": obs_normalizer}
