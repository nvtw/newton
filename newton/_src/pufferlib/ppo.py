# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO loss, action sampling, and GAE+V-Trace advantage estimation.

All kernels use ``enable_backward=False`` with hand-written backward passes.
The fused PPO loss kernel computes both the loss and gradients for logits and
values in a single pass, matching PufferLib's design.
"""

from __future__ import annotations

import warp as wp


wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


@wp.kernel
def sample_actions_discrete_kernel(
    logits: wp.array2d(dtype=float),
    act_sizes: wp.array(dtype=int, ndim=1),
    num_heads: int,
    actions: wp.array2d(dtype=float),
    logprobs: wp.array(dtype=float, ndim=1),
    seed: int,
):
    """Sample discrete actions via inverse-CDF with numerically stable softmax.

    One thread per agent.  Handles multi-discrete action spaces by iterating
    over action heads defined by ``act_sizes``.

    Args:
        logits: (B, total_action_dim) raw logits from the decoder.
        act_sizes: (num_heads,) number of actions per head.
        num_heads: number of action heads.
        actions: (B, num_heads) output sampled actions.
        logprobs: (B,) output total log-probability.
        seed: RNG seed.
    """
    agent = wp.tid()
    state = wp.rand_init(seed, agent)

    total_logp = float(0.0)
    logit_offset = int(0)

    for h in range(num_heads):
        A = act_sizes[h]

        max_logit = float(-1.0e30)
        for a in range(A):
            val = logits[agent, logit_offset + a]
            if val > max_logit:
                max_logit = val

        exp_sum = float(0.0)
        for a in range(A):
            exp_sum = exp_sum + wp.exp(logits[agent, logit_offset + a] - max_logit)
        logsumexp = max_logit + wp.log(exp_sum)

        rand_val = wp.randf(state)
        cumsum = float(0.0)
        sampled = int(0)
        found = int(0)
        for a in range(A):
            prob = wp.exp(logits[agent, logit_offset + a] - logsumexp)
            cumsum = cumsum + prob
            if cumsum > rand_val and found == 0:
                sampled = a
                found = 1

        actions[agent, h] = float(sampled)
        log_prob = logits[agent, logit_offset + sampled] - logsumexp
        total_logp = total_logp + log_prob

        logit_offset = logit_offset + A

    logprobs[agent] = total_logp


@wp.kernel
def sample_actions_continuous_kernel(
    logits: wp.array2d(dtype=float),
    logstd: wp.array(dtype=float, ndim=1),
    num_actions: int,
    actions: wp.array2d(dtype=float),
    logprobs: wp.array(dtype=float, ndim=1),
    seed: int,
):
    """Sample continuous actions from N(mean, exp(logstd)).

    One thread per agent.

    Args:
        logits: (B, num_actions) means from the decoder.
        logstd: (num_actions,) shared log standard deviations.
        num_actions: action dimensionality.
        actions: (B, num_actions) output sampled actions.
        logprobs: (B,) output total log-probability.
        seed: RNG seed.
    """
    agent = wp.tid()
    state = wp.rand_init(seed, agent)

    LOG_2PI = 1.8378770664093453
    total_logp = float(0.0)

    for h in range(num_actions):
        mean = logits[agent, h]
        log_std = logstd[h]
        std = wp.exp(log_std)

        noise = wp.randn(state)
        action = mean + std * noise

        normalized = (action - mean) / std
        log_prob = -0.5 * normalized * normalized - 0.5 * LOG_2PI - log_std

        actions[agent, h] = action
        total_logp = total_logp + log_prob

    logprobs[agent] = total_logp


def sample_actions_discrete(logits: wp.array, act_sizes: wp.array, num_heads: int,
                            seed: int = 0, *,
                            actions_buf: wp.array | None = None,
                            logprobs_buf: wp.array | None = None) -> tuple[wp.array, wp.array]:
    """Sample discrete actions. Returns (actions, logprobs).

    Pass pre-allocated *actions_buf* and *logprobs_buf* for graph-capture safety.
    """
    B = logits.shape[0]
    device = logits.device
    actions = actions_buf if actions_buf is not None else wp.zeros((B, num_heads), dtype=float, device=device)
    logprobs = logprobs_buf if logprobs_buf is not None else wp.zeros(B, dtype=float, device=device)
    wp.launch(sample_actions_discrete_kernel, dim=B,
              inputs=[logits, act_sizes, num_heads, actions, logprobs, seed],
              device=device)
    return actions, logprobs


def sample_actions_continuous(logits: wp.array, logstd: wp.array,
                              seed: int = 0, *,
                              actions_buf: wp.array | None = None,
                              logprobs_buf: wp.array | None = None) -> tuple[wp.array, wp.array]:
    """Sample continuous actions. Returns (actions, logprobs).

    Pass pre-allocated *actions_buf* and *logprobs_buf* for graph-capture safety.
    """
    B = logits.shape[0]
    num_actions = logits.shape[1]
    device = logits.device
    actions = actions_buf if actions_buf is not None else wp.zeros((B, num_actions), dtype=float, device=device)
    logprobs = logprobs_buf if logprobs_buf is not None else wp.zeros(B, dtype=float, device=device)
    wp.launch(sample_actions_continuous_kernel, dim=B,
              inputs=[logits, logstd, num_actions, actions, logprobs, seed],
              device=device)
    return actions, logprobs


# ---------------------------------------------------------------------------
# GAE + V-Trace advantage estimation
# ---------------------------------------------------------------------------


@wp.kernel
def gae_vtrace_kernel(
    values: wp.array2d(dtype=float),
    rewards: wp.array2d(dtype=float),
    dones: wp.array2d(dtype=float),
    importance: wp.array2d(dtype=float),
    advantages: wp.array2d(dtype=float),
    returns: wp.array2d(dtype=float),
    gamma: float,
    lam: float,
    rho_clip: float,
    c_clip: float,
):
    """Compute GAE + V-Trace advantages with importance sampling correction.

    One thread per agent (row).  Scans backward over the time dimension.

    Layout: all arrays are (num_agents, horizon).

    Args:
        values: (N, T) value predictions.
        rewards: (N, T) rewards.
        dones: (N, T) done flags (1.0 = terminal).
        importance: (N, T) importance sampling ratios.
        advantages: (N, T) output advantages.
        returns: (N, T) output returns (advantages + values).
        gamma: discount factor.
        lam: GAE lambda.
        rho_clip: V-Trace rho clipping threshold.
        c_clip: V-Trace c clipping threshold.
    """
    n = wp.tid()
    T = values.shape[1]

    last_gae = float(0.0)

    for t in range(T - 2, -1, -1):
        t_next = t + 1
        nonterminal = 1.0 - dones[n, t_next]

        imp = importance[n, t]
        rho_t = wp.min(imp, rho_clip)
        c_t = wp.min(imp, c_clip)

        r_next = rewards[n, t_next]
        v = values[n, t]
        v_next = values[n, t_next]

        delta = rho_t * (r_next + gamma * v_next * nonterminal - v)
        last_gae = delta + gamma * lam * c_t * last_gae * nonterminal

        advantages[n, t] = last_gae
        returns[n, t] = last_gae + v

    advantages[n, T - 1] = 0.0
    returns[n, T - 1] = values[n, T - 1]


def compute_gae_vtrace(values: wp.array, rewards: wp.array, dones: wp.array,
                       importance: wp.array, gamma: float = 0.99, lam: float = 0.95,
                       rho_clip: float = 1.0, c_clip: float = 1.0, *,
                       advantages_buf: wp.array | None = None,
                       returns_buf: wp.array | None = None) -> tuple[wp.array, wp.array]:
    """Compute GAE+V-Trace advantages. Returns (advantages, returns).

    Pass pre-allocated *advantages_buf* and *returns_buf* for graph-capture safety.
    """
    N = values.shape[0]
    T = values.shape[1]
    device = values.device
    advantages = advantages_buf if advantages_buf is not None else wp.zeros((N, T), dtype=float, device=device)
    returns = returns_buf if returns_buf is not None else wp.zeros((N, T), dtype=float, device=device)
    wp.launch(gae_vtrace_kernel, dim=N,
              inputs=[values, rewards, dones, importance, advantages, returns,
                      gamma, lam, rho_clip, c_clip],
              device=device)
    return advantages, returns


# ---------------------------------------------------------------------------
# Fused PPO loss + gradient kernel
# ---------------------------------------------------------------------------


@wp.kernel
def ppo_loss_discrete_kernel(
    logits: wp.array2d(dtype=float),
    actions: wp.array2d(dtype=float),
    old_logprobs: wp.array(dtype=float, ndim=1),
    advantages: wp.array(dtype=float, ndim=1),
    values_pred: wp.array(dtype=float, ndim=1),
    old_values: wp.array(dtype=float, ndim=1),
    returns_arr: wp.array(dtype=float, ndim=1),
    act_sizes: wp.array(dtype=int, ndim=1),
    num_heads: int,
    clip_coef: float,
    vf_clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    inv_batch: float,
    prio_weight: wp.array(dtype=float, ndim=1),
    # outputs
    grad_logits: wp.array2d(dtype=float),
    grad_values: wp.array(dtype=float, ndim=1),
    loss_out: wp.array2d(dtype=float),
):
    """Fused PPO loss + gradient for discrete actions.

    Matches C++ PufferLib ``ppo_loss_fwd_bwd``:
    - All gradients are scaled by ``inv_batch`` (= 1/B) so the optimizer
      sees mean gradients, not summed gradients.
    - Advantages should be pre-normalized by the caller.

    One thread per sample in the minibatch.
    loss_out: (B, 7) per-thread losses -- reduced externally via tile_sum.
    """
    idx = wp.tid()

    adv = advantages[idx]
    old_logp = old_logprobs[idx]
    val_pred = values_pred[idx]
    old_val = old_values[idx]
    ret = returns_arr[idx]

    dL = inv_batch

    # --- Value loss (clipped) ---
    v_error = val_pred - old_val
    v_clipped = old_val + wp.clamp(v_error, -vf_clip_coef, vf_clip_coef)
    v_loss_unclipped = (val_pred - ret) * (val_pred - ret)
    v_loss_clipped = (v_clipped - ret) * (v_clipped - ret)
    v_loss = 0.5 * wp.max(v_loss_unclipped, v_loss_clipped)

    # Value gradient
    use_clipped = 0
    if v_loss_clipped > v_loss_unclipped:
        use_clipped = 1

    d_val_pred = 0.0
    if use_clipped == 1:
        if v_error >= -vf_clip_coef and v_error <= vf_clip_coef:
            d_val_pred = v_clipped - ret
    else:
        d_val_pred = val_pred - ret

    grad_values[idx] = dL * vf_coef * d_val_pred

    # --- Policy loss (discrete) ---
    total_new_logp = float(0.0)
    total_entropy = float(0.0)

    logit_offset = int(0)
    for h in range(num_heads):
        A = act_sizes[h]
        act = int(actions[idx, h])

        max_logit = float(-1.0e30)
        for a in range(A):
            val2 = logits[idx, logit_offset + a]
            if val2 > max_logit:
                max_logit = val2

        exp_sum = float(0.0)
        for a in range(A):
            exp_sum = exp_sum + wp.exp(logits[idx, logit_offset + a] - max_logit)
        logsumexp = max_logit + wp.log(exp_sum)

        new_logp = logits[idx, logit_offset + act] - logsumexp
        total_new_logp = total_new_logp + new_logp

        ent = float(0.0)
        for a in range(A):
            logp_a = logits[idx, logit_offset + a] - logsumexp
            p_a = wp.exp(logp_a)
            ent = ent - p_a * logp_a
        total_entropy = total_entropy + ent

        logit_offset = logit_offset + A

    # PPO clipped surrogate
    logratio = total_new_logp - old_logp
    ratio = wp.exp(logratio)
    ratio_clipped = wp.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

    wa = -adv * prio_weight[idx]
    pg_loss1 = wa * ratio
    pg_loss2 = wa * ratio_clipped
    pg_loss = wp.max(pg_loss1, pg_loss2)

    # Policy gradient (scaled by dL = 1/B)
    d_ratio = wa * dL
    if pg_loss2 > pg_loss1:
        if ratio <= 1.0 - clip_coef or ratio >= 1.0 + clip_coef:
            d_ratio = 0.0

    d_new_logp = d_ratio * ratio
    d_entropy_term = dL * (-ent_coef)

    # Per-logit gradients
    logit_offset2 = int(0)
    for h2 in range(num_heads):
        A2 = act_sizes[h2]
        act2 = int(actions[idx, h2])

        max_logit2 = float(-1.0e30)
        for a2 in range(A2):
            val3 = logits[idx, logit_offset2 + a2]
            if val3 > max_logit2:
                max_logit2 = val3

        exp_sum2 = float(0.0)
        for a3 in range(A2):
            exp_sum2 = exp_sum2 + wp.exp(logits[idx, logit_offset2 + a3] - max_logit2)
        logsumexp2 = max_logit2 + wp.log(exp_sum2)

        ent2 = float(0.0)
        for a4 in range(A2):
            logp_a2 = logits[idx, logit_offset2 + a4] - logsumexp2
            ent2 = ent2 - wp.exp(logp_a2) * logp_a2

        for j in range(A2):
            logp_j = logits[idx, logit_offset2 + j] - logsumexp2
            p_j = wp.exp(logp_j)

            d_logit = 0.0
            if j == act2:
                d_logit = d_new_logp
            d_logit = d_logit - p_j * d_new_logp
            d_logit = d_logit + d_entropy_term * p_j * (-ent2 - logp_j)

            grad_logits[idx, logit_offset2 + j] = d_logit

        logit_offset2 = logit_offset2 + A2

    # Write per-thread loss components (reduced externally via tile_sum)
    total_loss = (pg_loss + vf_coef * v_loss - ent_coef * total_entropy) * dL
    loss_out[idx, 0] = total_loss
    loss_out[idx, 1] = pg_loss * dL
    loss_out[idx, 2] = v_loss * dL
    loss_out[idx, 3] = total_entropy * dL

    clipped_flag = 0.0
    if wp.abs(ratio - 1.0) > clip_coef:
        clipped_flag = 1.0
    loss_out[idx, 4] = clipped_flag * dL

    approx_kl = (ratio - 1.0) - logratio
    loss_out[idx, 5] = approx_kl * dL
    loss_out[idx, 6] = ratio * dL


@wp.kernel
def ppo_loss_continuous_kernel(
    logits: wp.array2d(dtype=float),
    logstd: wp.array(dtype=float, ndim=1),
    actions: wp.array2d(dtype=float),
    old_logprobs: wp.array(dtype=float, ndim=1),
    advantages: wp.array(dtype=float, ndim=1),
    values_pred: wp.array(dtype=float, ndim=1),
    old_values: wp.array(dtype=float, ndim=1),
    returns_arr: wp.array(dtype=float, ndim=1),
    num_actions: int,
    clip_coef: float,
    vf_clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    inv_batch: float,
    prio_weight: wp.array(dtype=float, ndim=1),
    # outputs
    grad_logits: wp.array2d(dtype=float),
    grad_logstd_scratch: wp.array2d(dtype=float),
    grad_values: wp.array(dtype=float, ndim=1),
    loss_out: wp.array2d(dtype=float),
):
    """Fused PPO loss + gradient for continuous actions.

    All gradients are scaled by ``inv_batch`` (= 1/B) so the optimizer
    sees mean gradients, matching C++ PufferLib ``ppo_loss_compute``.

    ``grad_logstd_scratch`` is ``(MB, num_actions)``; the caller must
    reduce columns to produce the final ``(num_actions,)`` gradient.
    """
    idx = wp.tid()

    HALF_LOG_2PI = 0.9189385332046727
    HALF_1_PLUS_LOG_2PI = 1.4189385332046727

    dL = inv_batch

    adv = advantages[idx]
    old_logp = old_logprobs[idx]
    val_pred = values_pred[idx]
    old_val = old_values[idx]
    ret = returns_arr[idx]

    # --- Value loss (clipped) ---
    v_error = val_pred - old_val
    v_clipped = old_val + wp.clamp(v_error, -vf_clip_coef, vf_clip_coef)
    v_loss_unclipped = (val_pred - ret) * (val_pred - ret)
    v_loss_clipped = (v_clipped - ret) * (v_clipped - ret)
    v_loss = 0.5 * wp.max(v_loss_unclipped, v_loss_clipped)

    use_clipped_vf = 0
    if v_loss_clipped > v_loss_unclipped:
        use_clipped_vf = 1

    d_val_pred = 0.0
    if use_clipped_vf == 1:
        if v_error >= -vf_clip_coef and v_error <= vf_clip_coef:
            d_val_pred = v_clipped - ret
    else:
        d_val_pred = val_pred - ret

    grad_values[idx] = dL * vf_coef * d_val_pred

    # --- Policy loss (continuous) ---
    total_new_logp = float(0.0)
    total_entropy = float(0.0)

    for h in range(num_actions):
        mean = logits[idx, h]
        log_std = logstd[h]
        std = wp.exp(log_std)
        action = actions[idx, h]

        normalized = (action - mean) / std
        new_logp = -0.5 * normalized * normalized - HALF_LOG_2PI - log_std
        entropy = HALF_1_PLUS_LOG_2PI + log_std

        total_new_logp = total_new_logp + new_logp
        total_entropy = total_entropy + entropy

    # PPO clipped surrogate
    logratio = total_new_logp - old_logp
    ratio = wp.exp(logratio)
    ratio_clipped = wp.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

    wa = -adv * prio_weight[idx]
    pg_loss1 = wa * ratio
    pg_loss2 = wa * ratio_clipped
    pg_loss = wp.max(pg_loss1, pg_loss2)

    d_pg = dL
    if pg_loss2 > pg_loss1:
        if ratio <= 1.0 - clip_coef or ratio >= 1.0 + clip_coef:
            d_pg = 0.0

    d_new_logp = wa * d_pg * ratio
    d_entropy = dL * (-ent_coef)

    # Per-action gradients
    for h2 in range(num_actions):
        mean2 = logits[idx, h2]
        log_std2 = logstd[h2]
        std2 = wp.exp(log_std2)
        action2 = actions[idx, h2]
        diff = action2 - mean2
        var = std2 * std2

        grad_logits[idx, h2] = d_new_logp * diff / var
        grad_logstd_scratch[idx, h2] = d_new_logp * (diff * diff / var - 1.0) + d_entropy

    # Write per-thread loss components (reduced separately via tile_sum)
    total_loss = (pg_loss + vf_coef * v_loss - ent_coef * total_entropy) * dL
    loss_out[idx, 0] = total_loss
    loss_out[idx, 1] = pg_loss * dL
    loss_out[idx, 2] = v_loss * dL
    loss_out[idx, 3] = total_entropy * dL

    clipped_flag = 0.0
    if wp.abs(ratio - 1.0) > clip_coef:
        clipped_flag = 1.0
    loss_out[idx, 4] = clipped_flag * dL

    approx_kl = (ratio - 1.0) - logratio
    loss_out[idx, 5] = approx_kl * dL
    loss_out[idx, 6] = ratio * dL


_REDUCE_BLOCK_DIM = 1024


@wp.kernel
def _reduce_loss_columns_kernel(
    per_thread: wp.array2d(dtype=float),
    reduced: wp.array(dtype=float, ndim=1),
    B: int,
):
    """Tile-sum column reduction of *per_thread* (B, C) into *reduced* (C,).

    Launched with ``wp.launch_tiled(dim=[C], block_dim=_REDUCE_BLOCK_DIM)``.
    Each of the C thread blocks cooperatively reduces one column using
    ``wp.tile`` + ``wp.tile_sum``, matching the pattern in ``reduce.py``.
    """
    col, lane = wp.tid()
    num_threads = wp.block_dim()

    partial = float(0.0)
    upper = ((B + num_threads - 1) // num_threads) * num_threads
    for row in range(lane, upper, num_threads):
        val = float(0.0)
        if row < B:
            val = per_thread[row, col]
        t = wp.tile(val)
        s = wp.tile_sum(t)
        partial += s[0]

    if lane == 0:
        reduced[col] = partial


@wp.kernel
def _reduce_logstd_grad_kernel(
    scratch: wp.array2d(dtype=float),
    reduced: wp.array(dtype=float, ndim=1),
    B: int,
):
    """Column-sum ``scratch`` (B, num_actions) into ``reduced`` (num_actions,).

    Same pattern as ``_reduce_loss_columns_kernel`` — launched with
    ``wp.launch_tiled(dim=[num_actions], block_dim=_REDUCE_BLOCK_DIM)``.
    """
    col, lane = wp.tid()
    num_threads = wp.block_dim()

    partial = float(0.0)
    upper = ((B + num_threads - 1) // num_threads) * num_threads
    for row in range(lane, upper, num_threads):
        val = float(0.0)
        if row < B:
            val = scratch[row, col]
        t = wp.tile(val)
        s = wp.tile_sum(t)
        partial += s[0]

    if lane == 0:
        reduced[col] = partial


class PPOLossBuffers:
    """Pre-allocated buffers for :func:`ppo_loss_and_grad`.  Graph-capture safe.

    Args:
        max_batch: Maximum minibatch size.
        action_dim: Total action dimension (logit width).
        device: Warp device string.
        continuous: Whether the action space is continuous.
        num_actions: Number of continuous actions (only used when *continuous* is True).
    """

    def __init__(self, max_batch: int, action_dim: int, device: str = "cuda:0",
                 continuous: bool = False, num_actions: int = 0):
        self.grad_logits = wp.zeros((max_batch, action_dim), dtype=float, device=device)
        self.grad_values = wp.zeros(max_batch, dtype=float, device=device)
        self.loss_per_thread = wp.zeros((max_batch, 7), dtype=float, device=device)
        self.loss_reduced = wp.zeros(7, dtype=float, device=device)
        self.num_actions = num_actions
        if continuous:
            self.grad_logstd_scratch = wp.zeros((max_batch, num_actions), dtype=float, device=device)
            self.grad_logstd = wp.zeros(num_actions, dtype=float, device=device)
        else:
            self.grad_logstd_scratch = None
            self.grad_logstd = None
        self.prio_weight = wp.ones(max_batch, dtype=float, device=device)


def ppo_loss_and_grad(
    logits: wp.array,
    actions: wp.array,
    old_logprobs: wp.array,
    advantages: wp.array,
    values_pred: wp.array,
    old_values: wp.array,
    returns: wp.array,
    *,
    act_sizes: wp.array | None = None,
    logstd: wp.array | None = None,
    prio_weights: wp.array | None = None,
    clip_coef: float = 0.2,
    vf_clip_coef: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    continuous: bool = False,
    buffers: PPOLossBuffers | None = None,
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
    """Compute fused PPO loss and gradients.

    Pass pre-allocated *buffers* (:class:`PPOLossBuffers`) for graph-capture
    compatibility.  If *buffers* is ``None``, temporary arrays are allocated
    (not graph-capture safe).

    Args:
        prio_weights: Per-sample priority weights (B,).  When ``None``,
            defaults to all-ones (uniform weighting).  Matches C++ PufferLib
            ``ppo_loss_compute`` which scales the advantage by ``w = prio[n]``.

    Returns:
        (loss_components, grad_logits, grad_values, grad_logstd_or_None)

        loss_components: (7,) array with [total, pg, vf, entropy, clipfrac, approx_kl, ratio_mean].
        grad_logits: (B, action_dim) gradient w.r.t. logits.
        grad_values: (B,) gradient w.r.t. value predictions.
        grad_logstd: (num_actions,) gradient w.r.t. logstd (continuous only).
    """
    B = logits.shape[0]
    device = logits.device

    if buffers is not None:
        grad_logits = buffers.grad_logits
        grad_values = buffers.grad_values
        loss_per_thread = buffers.loss_per_thread
        loss_reduced = buffers.loss_reduced
        pw = buffers.prio_weight
    else:
        grad_logits = wp.zeros_like(logits)
        grad_values = wp.zeros(B, dtype=float, device=device)
        loss_per_thread = wp.zeros((B, 7), dtype=float, device=device)
        loss_reduced = wp.zeros(7, dtype=float, device=device)
        pw = wp.ones(B, dtype=float, device=device)

    if prio_weights is not None:
        wp.copy(pw, prio_weights, count=B)
    elif buffers is not None:
        pw.fill_(1.0)

    if continuous:
        assert logstd is not None
        num_actions = logits.shape[1]
        grad_logstd = buffers.grad_logstd if buffers is not None else wp.zeros_like(logstd)
        wp.launch(ppo_loss_continuous_kernel, dim=B,
                  inputs=[logits, logstd, actions, old_logprobs, advantages,
                          values_pred, old_values, returns, num_actions,
                          clip_coef, vf_clip_coef, vf_coef, ent_coef,
                          1.0 / float(B), pw,
                          grad_logits, grad_logstd, grad_values, loss_per_thread],
                  device=device)
    else:
        assert act_sizes is not None
        num_heads = act_sizes.shape[0]
        grad_logstd = None
        wp.launch(ppo_loss_discrete_kernel, dim=B,
                  inputs=[logits, actions, old_logprobs, advantages,
                          values_pred, old_values, returns, act_sizes, num_heads,
                          clip_coef, vf_clip_coef, vf_coef, ent_coef,
                          1.0 / float(B), pw,
                          grad_logits, grad_values, loss_per_thread],
                  device=device)

    wp.launch_tiled(_reduce_loss_columns_kernel, dim=[7],
                    inputs=[loss_per_thread, loss_reduced, B],
                    block_dim=_REDUCE_BLOCK_DIM, device=device)

    return loss_reduced, grad_logits, grad_values, grad_logstd


# ---------------------------------------------------------------------------
# Priority replay sampling
# ---------------------------------------------------------------------------


@wp.kernel
def compute_priority_weights_kernel(
    advantages: wp.array2d(dtype=float),
    priority: wp.array(dtype=float, ndim=1),
    alpha: float,
):
    """Compute priority weights as sum(|advantage|)^alpha per agent."""
    n = wp.tid()
    T = advantages.shape[1]
    total = float(0.0)
    for t in range(T):
        total = total + wp.abs(advantages[n, t])
    priority[n] = wp.pow(total, alpha)


@wp.kernel
def normalize_kernel(arr: wp.array(dtype=float, ndim=1), total: wp.array(dtype=float, ndim=1)):
    """Normalize array to sum to 1 (in-place). *total* is a (1,) array with the sum."""
    i = wp.tid()
    arr[i] = arr[i] / total[0]


@wp.kernel
def multinomial_sample_kernel(
    cdf: wp.array(dtype=float, ndim=1),
    indices: wp.array(dtype=int, ndim=1),
    num_samples: int,
    seed: int,
):
    """Sample indices from a CDF via binary search."""
    s = wp.tid()
    state = wp.rand_init(seed, s)
    rand_val = wp.randf(state)

    N = cdf.shape[0]
    lo = int(0)
    hi = N - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < rand_val:
            lo = mid + 1
        else:
            hi = mid
    indices[s] = lo


class PrioritySampler:
    """Pre-allocated priority replay sampler.  Graph-capture safe.

    All scratch buffers are allocated once at construction.
    """

    def __init__(self, max_agents: int, max_samples: int, device: str = "cuda:0"):
        from newton._src.pufferlib.reduce import ArraySum  # noqa: PLC0415

        self.device = device
        self.priority = wp.zeros(max_agents, dtype=float, device=device)
        self.cdf = wp.zeros(max_agents, dtype=float, device=device)
        self.indices = wp.zeros(max_samples, dtype=int, device=device)
        self._summer = ArraySum(max_agents, device=device)

    def sample(self, advantages: wp.array, num_agents: int, num_samples: int,
               alpha: float, seed: int) -> wp.array:
        """Sample *num_samples* agent indices weighted by |advantage|^alpha."""
        from newton._src.pufferlib.reduce import array_prefix_sum  # noqa: PLC0415

        device = self.device

        wp.launch(compute_priority_weights_kernel, dim=num_agents,
                  inputs=[advantages, self.priority, alpha], device=device)

        total = self._summer.compute(self.priority, num_agents)
        wp.launch(normalize_kernel, dim=num_agents,
                  inputs=[self.priority, total], device=device)

        array_prefix_sum(self.priority, self.cdf)

        wp.launch(multinomial_sample_kernel, dim=num_samples,
                  inputs=[self.cdf, self.indices, num_samples, seed], device=device)
        return self.indices


def priority_sample(advantages: wp.array, num_samples: int, alpha: float = 1.0,
                    seed: int = 0) -> wp.array:
    """Sample agent indices weighted by |advantage|^alpha.

    .. note:: For graph-capture compatibility, prefer :class:`PrioritySampler`
       which pre-allocates all buffers.

    Returns:
        indices: (num_samples,) sampled agent indices.
    """
    from newton._src.pufferlib.reduce import ArraySum, array_prefix_sum  # noqa: PLC0415

    N = advantages.shape[0]
    device = advantages.device

    priority = wp.zeros(N, dtype=float, device=device)
    wp.launch(compute_priority_weights_kernel, dim=N,
              inputs=[advantages, priority, alpha], device=device)

    summer = ArraySum(N, device=device)
    total = summer.compute(priority)
    wp.launch(normalize_kernel, dim=N, inputs=[priority, total], device=device)

    cdf = wp.zeros(N, dtype=float, device=device)
    array_prefix_sum(priority, cdf)

    indices = wp.zeros(num_samples, dtype=int, device=device)
    wp.launch(multinomial_sample_kernel, dim=num_samples,
              inputs=[cdf, indices, num_samples, seed], device=device)
    return indices
