# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

LOG_SQRT_2PI = 0.9189385332046727
LOG_2PI_E = 2.8378770664093453
TANH_EPS = 1.0e-6

ACTIVATION_LINEAR = 0
ACTIVATION_TANH = 1
ACTIVATION_RELU = 2
ACTIVATION_ELU = 3


@wp.func
def _activation(x: wp.float32, activation: wp.int32) -> wp.float32:
    if activation == ACTIVATION_TANH:
        return wp.tanh(x)
    if activation == ACTIVATION_RELU:
        return wp.max(x, wp.float32(0.0))
    if activation == ACTIVATION_ELU:
        if x > wp.float32(0.0):
            return x
        return wp.exp(x) - wp.float32(1.0)
    return x


@wp.func
def _clip(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.min(wp.max(x, lo), hi)


@wp.func
def _normal_log_prob(x: wp.float32, mean: wp.float32, log_std: wp.float32) -> wp.float32:
    std = wp.exp(log_std)
    z = (x - mean) / std
    return -wp.float32(0.5) * z * z - log_std - wp.float32(LOG_SQRT_2PI)


@wp.func
def _atanh_clamped(x: wp.float32) -> wp.float32:
    a = _clip(x, -wp.float32(1.0) + wp.float32(TANH_EPS), wp.float32(1.0) - wp.float32(TANH_EPS))
    return wp.float32(0.5) * wp.log((wp.float32(1.0) + a) / (wp.float32(1.0) - a))


@wp.kernel
def dense_layer_kernel(
    x: wp.array2d[wp.float32],
    weight: wp.array2d[wp.float32],
    bias: wp.array[wp.float32],
    in_dim: wp.int32,
    activation: wp.int32,
    y: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    total = bias[col]
    for i in range(in_dim):
        total = total + x[row, i] * weight[i, col]
    y[row, col] = _activation(total, activation)


@wp.kernel
def zero_scalar_kernel(x: wp.array[wp.float32]):
    x[0] = wp.float32(0.0)


@wp.kernel
def fill_eps_kernel(seed: wp.int32, eps: wp.array2d[wp.float32]):
    row, col = wp.tid()
    flat = row * eps.shape[1] + col
    rng = wp.rand_init(seed, flat)
    eps[row, col] = wp.randn(rng)


@wp.kernel
def gaussian_log_prob_kernel(
    policy_out: wp.array2d[wp.float32],
    log_std_param: wp.array[wp.float32],
    actions: wp.array2d[wp.float32],
    action_dim: wp.int32,
    state_dependent_std: wp.int32,
    squash: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    log_probs: wp.array[wp.float32],
):
    row = wp.tid()
    total = wp.float32(0.0)
    for j in range(action_dim):
        mean = policy_out[row, j]
        log_std = log_std_param[j]
        if state_dependent_std != 0:
            log_std = policy_out[row, action_dim + j]
        log_std = _clip(log_std, log_std_min, log_std_max)
        a = actions[row, j]
        if squash != 0:
            pre_tanh = _atanh_clamped(a)
            total = total + _normal_log_prob(pre_tanh, mean, log_std)
            total = total - wp.log(wp.float32(1.0) - a * a + wp.float32(TANH_EPS))
        else:
            total = total + _normal_log_prob(a, mean, log_std)
    log_probs[row] = total


@wp.kernel
def gaussian_entropy_kernel(
    policy_out: wp.array2d[wp.float32],
    log_std_param: wp.array[wp.float32],
    action_dim: wp.int32,
    state_dependent_std: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    entropy: wp.array[wp.float32],
):
    row = wp.tid()
    total = wp.float32(0.0)
    for j in range(action_dim):
        log_std = log_std_param[j]
        if state_dependent_std != 0:
            log_std = policy_out[row, action_dim + j]
        log_std = _clip(log_std, log_std_min, log_std_max)
        total = total + wp.float32(0.5 * LOG_2PI_E) + log_std
    entropy[row] = total


@wp.kernel
def sample_gaussian_actions_kernel(
    policy_out: wp.array2d[wp.float32],
    log_std_param: wp.array[wp.float32],
    eps: wp.array2d[wp.float32],
    action_dim: wp.int32,
    state_dependent_std: wp.int32,
    squash: wp.int32,
    deterministic: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    actions: wp.array2d[wp.float32],
    log_probs: wp.array[wp.float32],
):
    row = wp.tid()
    total = wp.float32(0.0)
    for j in range(action_dim):
        mean = policy_out[row, j]
        log_std = log_std_param[j]
        if state_dependent_std != 0:
            log_std = policy_out[row, action_dim + j]
        log_std = _clip(log_std, log_std_min, log_std_max)
        pre_tanh = mean
        if deterministic == 0:
            pre_tanh = mean + wp.exp(log_std) * eps[row, j]
        a = pre_tanh
        if squash != 0:
            a = wp.tanh(pre_tanh)
            total = total + _normal_log_prob(pre_tanh, mean, log_std)
            total = total - wp.log(wp.float32(1.0) - a * a + wp.float32(TANH_EPS))
        else:
            total = total + _normal_log_prob(a, mean, log_std)
        actions[row, j] = a
    log_probs[row] = total


@wp.kernel
def ppo_actor_loss_kernel(
    new_log_probs: wp.array[wp.float32],
    old_log_probs: wp.array[wp.float32],
    advantages: wp.array[wp.float32],
    entropy: wp.array[wp.float32],
    clip_ratio: wp.float32,
    entropy_coeff: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
):
    i = wp.tid()
    log_ratio = new_log_probs[i] - old_log_probs[i]
    ratio = wp.exp(log_ratio)
    clipped = _clip(ratio, wp.float32(1.0) - clip_ratio, wp.float32(1.0) + clip_ratio)
    unclipped_obj = ratio * advantages[i]
    clipped_obj = clipped * advantages[i]
    obj = wp.min(unclipped_obj, clipped_obj)
    wp.atomic_add(loss, 0, (-obj - entropy_coeff * entropy[i]) / wp.float32(batch_size))
    wp.atomic_add(approx_kl, 0, ((ratio - wp.float32(1.0)) - log_ratio) / wp.float32(batch_size))
    if wp.abs(ratio - wp.float32(1.0)) > clip_ratio:
        wp.atomic_add(clip_fraction, 0, wp.float32(1.0) / wp.float32(batch_size))


@wp.kernel
def value_loss_kernel(
    values: wp.array2d[wp.float32],
    returns: wp.array[wp.float32],
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    delta = values[i, 0] - returns[i]
    wp.atomic_add(loss, 0, wp.float32(0.5) * delta * delta / wp.float32(batch_size))


@wp.kernel
def mirror_2d_kernel(
    src: wp.array2d[wp.float32],
    mirror_src: wp.array[wp.int32],
    mirror_sign: wp.array[wp.float32],
    dst: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    dst[row, col] = mirror_sign[col] * src[row, mirror_src[col]]


@wp.kernel
def mirrored_action_mse_loss_kernel(
    policy_out: wp.array2d[wp.float32],
    mirrored_policy_out: wp.array2d[wp.float32],
    action_mirror_src: wp.array[wp.int32],
    action_mirror_sign: wp.array[wp.float32],
    action_dim: wp.int32,
    coeff: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
):
    row, action = wp.tid()
    target = action_mirror_sign[action] * mirrored_policy_out[row, action_mirror_src[action]]
    delta = policy_out[row, action] - target
    wp.atomic_add(loss, 0, wp.float32(0.5) * coeff * delta * delta / wp.float32(batch_size))


@wp.kernel
def value_symmetry_loss_kernel(
    values: wp.array2d[wp.float32],
    mirrored_values: wp.array2d[wp.float32],
    coeff: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
):
    row = wp.tid()
    delta = values[row, 0] - mirrored_values[row, 0]
    wp.atomic_add(loss, 0, wp.float32(0.5) * coeff * delta * delta / wp.float32(batch_size))


@wp.kernel
def mse_loss_2d_kernel(
    predictions: wp.array2d[wp.float32],
    targets: wp.array2d[wp.float32],
    rows: wp.int32,
    cols: wp.int32,
    loss: wp.array[wp.float32],
):
    row, col = wp.tid()
    delta = predictions[row, col] - targets[row, col]
    wp.atomic_add(loss, 0, wp.float32(0.5) * delta * delta / wp.float32(rows * cols))


@wp.kernel
def compute_gae_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    values: wp.array[wp.float32],
    num_steps: wp.int32,
    num_envs: wp.int32,
    gamma: wp.float32,
    gae_lambda: wp.float32,
    advantages: wp.array[wp.float32],
    returns: wp.array[wp.float32],
):
    env = wp.tid()
    gae = wp.float32(0.0)
    for t_rev in range(num_steps):
        t = num_steps - wp.int32(1) - t_rev
        idx = t * num_envs + env
        next_idx = (t + wp.int32(1)) * num_envs + env
        non_terminal = wp.float32(1.0) - dones[idx]
        delta = rewards[idx] + gamma * values[next_idx] * non_terminal - values[idx]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[idx] = gae
        returns[idx] = gae + values[idx]


@wp.kernel
def sum_and_sumsq_kernel(x: wp.array[wp.float32], count: wp.int32, stats: wp.array[wp.float32]):
    i = wp.tid()
    if i < count:
        v = x[i]
        wp.atomic_add(stats, 0, v)
        wp.atomic_add(stats, 1, v * v)


@wp.kernel
def normalize_kernel(x: wp.array[wp.float32], mean: wp.float32, inv_std: wp.float32, count: wp.int32):
    i = wp.tid()
    if i < count:
        x[i] = (x[i] - mean) * inv_std


@wp.kernel
def concat_obs_action_kernel(
    obs: wp.array2d[wp.float32],
    actions: wp.array2d[wp.float32],
    obs_dim: wp.int32,
    action_dim: wp.int32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    if col < obs_dim:
        out[row, col] = obs[row, col]
    else:
        out[row, col] = actions[row, col - obs_dim]


@wp.kernel
def min_q_target_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    q1: wp.array2d[wp.float32],
    q2: wp.array2d[wp.float32],
    next_log_probs: wp.array[wp.float32],
    gamma: wp.float32,
    alpha: wp.float32,
    targets: wp.array[wp.float32],
):
    i = wp.tid()
    min_q = wp.min(q1[i, 0], q2[i, 0])
    targets[i] = rewards[i] + gamma * (wp.float32(1.0) - dones[i]) * (min_q - alpha * next_log_probs[i])


@wp.kernel
def sac_critic_loss_kernel(
    q1: wp.array2d[wp.float32],
    q2: wp.array2d[wp.float32],
    targets: wp.array[wp.float32],
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    d1 = q1[i, 0] - targets[i]
    d2 = q2[i, 0] - targets[i]
    wp.atomic_add(loss, 0, wp.float32(0.5) * (d1 * d1 + d2 * d2) / wp.float32(batch_size))


@wp.kernel
def sac_actor_loss_kernel(
    q1: wp.array2d[wp.float32],
    q2: wp.array2d[wp.float32],
    log_probs: wp.array[wp.float32],
    batch_size: wp.int32,
    alpha: wp.float32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    min_q = wp.min(q1[i, 0], q2[i, 0])
    wp.atomic_add(loss, 0, (alpha * log_probs[i] - min_q) / wp.float32(batch_size))


@wp.kernel
def sac_alpha_loss_kernel(
    log_probs: wp.array[wp.float32],
    log_alpha: wp.array[wp.float32],
    batch_size: wp.int32,
    target_entropy: wp.float32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    # Detach log_probs by passing an array computed outside this tape.
    wp.atomic_add(loss, 0, -log_alpha[0] * (log_probs[i] + target_entropy) / wp.float32(batch_size))


@wp.kernel
def replay_store_kernel(
    obs_src: wp.array2d[wp.float32],
    actions_src: wp.array2d[wp.float32],
    rewards_src: wp.array[wp.float32],
    dones_src: wp.array[wp.float32],
    next_obs_src: wp.array2d[wp.float32],
    start: wp.int32,
    capacity: wp.int32,
    row_count: wp.int32,
    obs_dim: wp.int32,
    action_dim: wp.int32,
    obs_dst: wp.array2d[wp.float32],
    actions_dst: wp.array2d[wp.float32],
    rewards_dst: wp.array[wp.float32],
    dones_dst: wp.array[wp.float32],
    next_obs_dst: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    if row >= row_count:
        return

    dst = (start + row) % capacity
    if col < obs_dim:
        obs_dst[dst, col] = obs_src[row, col]
        next_obs_dst[dst, col] = next_obs_src[row, col]
    if col < action_dim:
        actions_dst[dst, col] = actions_src[row, col]
    if col == 0:
        rewards_dst[dst] = rewards_src[row]
        dones_dst[dst] = dones_src[row]


@wp.kernel
def replay_sample_kernel(
    obs_src: wp.array2d[wp.float32],
    actions_src: wp.array2d[wp.float32],
    rewards_src: wp.array[wp.float32],
    dones_src: wp.array[wp.float32],
    next_obs_src: wp.array2d[wp.float32],
    size: wp.int32,
    seed: wp.int32,
    obs_dim: wp.int32,
    action_dim: wp.int32,
    obs_out: wp.array2d[wp.float32],
    actions_out: wp.array2d[wp.float32],
    rewards_out: wp.array[wp.float32],
    dones_out: wp.array[wp.float32],
    next_obs_out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    rng = wp.rand_init(seed, row)
    src = wp.int32(wp.floor(wp.randf(rng) * wp.float32(size)))
    src = wp.min(src, size - wp.int32(1))

    if col < obs_dim:
        obs_out[row, col] = obs_src[src, col]
        next_obs_out[row, col] = next_obs_src[src, col]
    if col < action_dim:
        actions_out[row, col] = actions_src[src, col]
    if col == 0:
        rewards_out[row] = rewards_src[src]
        dones_out[row] = dones_src[src]


@wp.kernel
def copy_1d_kernel(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_2d_kernel(src: wp.array2d[wp.float32], dst: wp.array2d[wp.float32]):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


@wp.kernel
def soft_update_1d_kernel(src: wp.array[wp.float32], tau: wp.float32, dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = (wp.float32(1.0) - tau) * dst[i] + tau * src[i]


@wp.kernel
def soft_update_2d_kernel(src: wp.array2d[wp.float32], tau: wp.float32, dst: wp.array2d[wp.float32]):
    i, j = wp.tid()
    dst[i, j] = (wp.float32(1.0) - tau) * dst[i, j] + tau * src[i, j]


@wp.kernel
def adam_step_1d_kernel(
    param: wp.array[wp.float32],
    grad: wp.array[wp.float32],
    m: wp.array[wp.float32],
    v: wp.array[wp.float32],
    lr: wp.float32,
    beta1: wp.float32,
    beta2: wp.float32,
    beta1_correction: wp.float32,
    beta2_correction: wp.float32,
    eps: wp.float32,
    weight_decay: wp.float32,
):
    i = wp.tid()
    g = grad[i] + weight_decay * param[i]
    mi = beta1 * m[i] + (wp.float32(1.0) - beta1) * g
    vi = beta2 * v[i] + (wp.float32(1.0) - beta2) * g * g
    m[i] = mi
    v[i] = vi
    param[i] = param[i] - lr * (mi / beta1_correction) / (wp.sqrt(vi / beta2_correction) + eps)
    grad[i] = wp.float32(0.0)


@wp.kernel
def adam_step_2d_kernel(
    param: wp.array2d[wp.float32],
    grad: wp.array2d[wp.float32],
    m: wp.array2d[wp.float32],
    v: wp.array2d[wp.float32],
    lr: wp.float32,
    beta1: wp.float32,
    beta2: wp.float32,
    beta1_correction: wp.float32,
    beta2_correction: wp.float32,
    eps: wp.float32,
    weight_decay: wp.float32,
):
    i, j = wp.tid()
    g = grad[i, j] + weight_decay * param[i, j]
    mi = beta1 * m[i, j] + (wp.float32(1.0) - beta1) * g
    vi = beta2 * v[i, j] + (wp.float32(1.0) - beta2) * g * g
    m[i, j] = mi
    v[i, j] = vi
    param[i, j] = param[i, j] - lr * (mi / beta1_correction) / (wp.sqrt(vi / beta2_correction) + eps)
    grad[i, j] = wp.float32(0.0)
