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

DENSE_TILE_BATCH = 128
DENSE_TILE_IN = 16
DENSE_TILE_OUT = 16
DENSE_TILE_BLOCK_DIM = 256
DENSE_BIAS_TILE_BATCH = 256
PPO_LOG_STD_PARTIAL_BATCH = 256


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


@wp.func
def _activation_grad_from_output(y: wp.float32, activation: wp.int32) -> wp.float32:
    if activation == ACTIVATION_TANH:
        return wp.float32(1.0) - y * y
    if activation == ACTIVATION_RELU:
        if y > wp.float32(0.0):
            return wp.float32(1.0)
        return wp.float32(0.0)
    if activation == ACTIVATION_ELU:
        if y > wp.float32(0.0):
            return wp.float32(1.0)
        return y + wp.float32(1.0)
    return wp.float32(1.0)


@wp.kernel
def zero_scalar_kernel(x: wp.array[wp.float32]):
    x[0] = wp.float32(0.0)


@wp.kernel
def adam_step_prepare_kernel(
    step_count: wp.array[wp.int32], beta1: wp.float32, beta2: wp.float32, corrections: wp.array[wp.float32]
):
    step_count[0] = step_count[0] + wp.int32(1)
    step = wp.float32(step_count[0])
    corrections[0] = wp.float32(1.0) - wp.pow(beta1, step)
    corrections[1] = wp.float32(1.0) - wp.pow(beta2, step)


@wp.kernel
def seed_counter_increment_kernel(counter: wp.array[wp.int32], delta: wp.int32):
    modulus = wp.int64(2147483647)
    value = (wp.int64(counter[0]) + wp.int64(delta)) % modulus
    counter[0] = wp.int32(value)


@wp.kernel
def pack_ppo_update_stats_kernel(
    policy_loss: wp.array[wp.float32],
    value_loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
    stats: wp.array[wp.float32],
):
    stats[0] = policy_loss[0]
    stats[1] = value_loss[0]
    stats[2] = approx_kl[0]
    stats[3] = clip_fraction[0]


@wp.kernel
def zero_2d_tail_rows_kernel(start_row: wp.int32, x: wp.array2d[wp.float32]):
    row, col = wp.tid()
    x[start_row + row, col] = wp.float32(0.0)


@wp.kernel
def zero_ppo_actor_stats_kernel(
    partial_count: wp.int32,
    action_dim: wp.int32,
    loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
    log_std_grad_partials: wp.array2d[wp.float32],
):
    i = wp.tid()
    if i == wp.int32(0):
        loss[0] = wp.float32(0.0)
        approx_kl[0] = wp.float32(0.0)
        clip_fraction[0] = wp.float32(0.0)
    if i < partial_count * action_dim:
        partial = i / action_dim
        action = i - partial * action_dim
        log_std_grad_partials[partial, action] = wp.float32(0.0)


@wp.kernel
def zero_ppo_loss_stats_kernel(
    loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
):
    loss[0] = wp.float32(0.0)
    approx_kl[0] = wp.float32(0.0)
    clip_fraction[0] = wp.float32(0.0)


@wp.kernel
def dense_activation_grad_kernel(
    y: wp.array2d[wp.float32],
    grad_y: wp.array2d[wp.float32],
    activation: wp.int32,
    grad_pre: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    grad_pre[row, col] = grad_y[row, col] * _activation_grad_from_output(y[row, col], activation)


@wp.kernel
def dense_weight_bias_grad_kernel(
    x: wp.array2d[wp.float32],
    grad_pre: wp.array2d[wp.float32],
    batch_size: wp.int32,
    weight_grad: wp.array2d[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    in_col, out_col = wp.tid()
    total = wp.float32(0.0)
    bias_total = wp.float32(0.0)
    for row in range(batch_size):
        g = grad_pre[row, out_col]
        total = total + x[row, in_col] * g
        if in_col == wp.int32(0):
            bias_total = bias_total + g
    weight_grad[in_col, out_col] = total
    if in_col == wp.int32(0):
        bias_grad[out_col] = bias_total


@wp.kernel
def dense_input_grad_kernel(
    grad_pre: wp.array2d[wp.float32],
    weight: wp.array2d[wp.float32],
    out_dim: wp.int32,
    grad_x: wp.array2d[wp.float32],
):
    row, in_col = wp.tid()
    total = wp.float32(0.0)
    for out_col in range(out_dim):
        total = total + grad_pre[row, out_col] * weight[in_col, out_col]
    grad_x[row, in_col] = total


@wp.kernel
def dense_weight_grad_tiled_kernel(
    x: wp.array2d[wp.float32],
    grad_pre: wp.array2d[wp.float32],
    batch_size: wp.int32,
    weight_grad: wp.array2d[wp.float32],
):
    in_tile, out_tile = wp.tid()
    total = wp.tile_zeros(shape=(DENSE_TILE_IN, DENSE_TILE_OUT), dtype=wp.float32)
    batch_tiles = (batch_size + DENSE_TILE_BATCH - wp.int32(1)) // DENSE_TILE_BATCH
    for tile in range(batch_tiles):
        x_tile = wp.tile_load(
            x,
            shape=(DENSE_TILE_BATCH, DENSE_TILE_IN),
            offset=(tile * DENSE_TILE_BATCH, in_tile * DENSE_TILE_IN),
        )
        grad_tile = wp.tile_load(
            grad_pre,
            shape=(DENSE_TILE_BATCH, DENSE_TILE_OUT),
            offset=(tile * DENSE_TILE_BATCH, out_tile * DENSE_TILE_OUT),
        )
        wp.tile_matmul(wp.tile_transpose(x_tile), grad_tile, total)
    wp.tile_store(weight_grad, total, offset=(in_tile * DENSE_TILE_IN, out_tile * DENSE_TILE_OUT))


@wp.kernel
def dense_bias_grad_kernel(
    grad_pre: wp.array2d[wp.float32],
    batch_size: wp.int32,
    bias_grad: wp.array[wp.float32],
):
    out_col = wp.tid()
    total = wp.float32(0.0)
    for row in range(batch_size):
        total = total + grad_pre[row, out_col]
    bias_grad[out_col] = total


@wp.kernel
def dense_bias_partial_grad_kernel(
    grad_pre: wp.array2d[wp.float32],
    batch_size: wp.int32,
    partial_grad: wp.array2d[wp.float32],
):
    batch_tile, out_col = wp.tid()
    start = batch_tile * DENSE_BIAS_TILE_BATCH
    total = wp.float32(0.0)
    for i in range(DENSE_BIAS_TILE_BATCH):
        row = start + i
        if row < batch_size:
            total = total + grad_pre[row, out_col]
    partial_grad[batch_tile, out_col] = total


@wp.kernel
def dense_bias_reduce_grad_kernel(
    partial_grad: wp.array2d[wp.float32],
    partial_count: wp.int32,
    bias_grad: wp.array[wp.float32],
):
    out_col = wp.tid()
    total = wp.float32(0.0)
    for i in range(partial_count):
        total = total + partial_grad[i, out_col]
    bias_grad[out_col] = total


@wp.kernel
def dense_input_grad_tiled_kernel(
    grad_pre: wp.array2d[wp.float32],
    weight: wp.array2d[wp.float32],
    out_dim: wp.int32,
    grad_x: wp.array2d[wp.float32],
):
    batch_tile, in_tile = wp.tid()
    total = wp.tile_zeros(shape=(DENSE_TILE_BATCH, DENSE_TILE_IN), dtype=wp.float32)
    out_tiles = (out_dim + DENSE_TILE_OUT - wp.int32(1)) // DENSE_TILE_OUT
    for tile in range(out_tiles):
        grad_tile = wp.tile_load(
            grad_pre,
            shape=(DENSE_TILE_BATCH, DENSE_TILE_OUT),
            offset=(batch_tile * DENSE_TILE_BATCH, tile * DENSE_TILE_OUT),
        )
        weight_tile = wp.tile_load(
            weight,
            shape=(DENSE_TILE_IN, DENSE_TILE_OUT),
            offset=(in_tile * DENSE_TILE_IN, tile * DENSE_TILE_OUT),
        )
        wp.tile_matmul(grad_tile, wp.tile_transpose(weight_tile), total)
    wp.tile_store(grad_x, total, offset=(batch_tile * DENSE_TILE_BATCH, in_tile * DENSE_TILE_IN))


@wp.kernel
def fill_eps_kernel(seed: wp.int32, eps: wp.array2d[wp.float32]):
    row, col = wp.tid()
    flat = row * eps.shape[1] + col
    rng = wp.rand_init(seed, flat)
    eps[row, col] = wp.randn(rng)


@wp.kernel
def fill_eps_seed_counter_kernel(seed_counter: wp.array[wp.int32], seed_offset: wp.int32, eps: wp.array2d[wp.float32]):
    row, col = wp.tid()
    flat = row * eps.shape[1] + col
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
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
    ratios: wp.array[wp.float32],
):
    i = wp.tid()
    log_ratio = new_log_probs[i] - old_log_probs[i]
    ratio = wp.exp(log_ratio)
    ratios[i] = ratio
    clipped = _clip(ratio, wp.float32(1.0) - clip_ratio, wp.float32(1.0) + clip_ratio)
    unclipped_obj = ratio * advantages[i]
    clipped_obj = clipped * advantages[i]
    obj = wp.min(unclipped_obj, clipped_obj)
    wp.atomic_add(loss, 0, (-obj - entropy_coeff * entropy[i]) / wp.float32(batch_size))
    wp.atomic_add(approx_kl, 0, ((ratio - wp.float32(1.0)) - log_ratio) / wp.float32(batch_size))
    if wp.abs(ratio - wp.float32(1.0)) > clip_ratio:
        wp.atomic_add(clip_fraction, 0, wp.float32(1.0) / wp.float32(batch_size))


@wp.kernel
def reduce_ppo_log_std_grad_kernel(
    log_std_grad_partials: wp.array2d[wp.float32],
    partial_count: wp.int32,
    log_std_grad: wp.array[wp.float32],
):
    action = wp.tid()
    total = wp.float32(0.0)
    for partial in range(partial_count):
        total = total + log_std_grad_partials[partial, action]
    log_std_grad[action] = total


@wp.kernel
def ppo_actor_loss_backward_kernel(
    policy_out: wp.array2d[wp.float32],
    log_std_param: wp.array[wp.float32],
    actions: wp.array2d[wp.float32],
    old_log_probs: wp.array[wp.float32],
    advantages: wp.array[wp.float32],
    clip_ratio: wp.float32,
    entropy_coeff: wp.float32,
    action_dim: wp.int32,
    state_dependent_std: wp.int32,
    squash: wp.int32,
    log_std_min: wp.float32,
    log_std_max: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    approx_kl: wp.array[wp.float32],
    clip_fraction: wp.array[wp.float32],
    ratios: wp.array[wp.float32],
    policy_out_grad: wp.array2d[wp.float32],
    log_std_grad_partials: wp.array2d[wp.float32],
):
    row = wp.tid()
    total_log_prob = wp.float32(0.0)
    total_entropy = wp.float32(0.0)
    for j in range(action_dim):
        mean = policy_out[row, j]
        raw_log_std = log_std_param[j]
        if state_dependent_std != 0:
            raw_log_std = policy_out[row, action_dim + j]
        log_std = _clip(raw_log_std, log_std_min, log_std_max)
        a = actions[row, j]
        if squash != 0:
            a = _atanh_clamped(a)
        total_log_prob = total_log_prob + _normal_log_prob(a, mean, log_std)
        if squash != 0:
            action = actions[row, j]
            total_log_prob = total_log_prob - wp.log(wp.float32(1.0) - action * action + wp.float32(TANH_EPS))
        total_entropy = total_entropy + wp.float32(0.5 * LOG_2PI_E) + log_std

    log_ratio = total_log_prob - old_log_probs[row]
    ratio = wp.exp(log_ratio)
    ratios[row] = ratio
    clipped = _clip(ratio, wp.float32(1.0) - clip_ratio, wp.float32(1.0) + clip_ratio)
    adv = advantages[row]
    pg_loss_unclipped = -adv * ratio
    pg_loss_clipped = -adv * clipped
    pg_loss = wp.max(pg_loss_unclipped, pg_loss_clipped)
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    wp.atomic_add(loss, 0, (pg_loss - entropy_coeff * total_entropy) * inv_batch)
    wp.atomic_add(approx_kl, 0, ((ratio - wp.float32(1.0)) - log_ratio) * inv_batch)
    clipped_branch = pg_loss_clipped > pg_loss_unclipped
    outside_clip = ratio <= wp.float32(1.0) - clip_ratio or ratio >= wp.float32(1.0) + clip_ratio
    d_log_prob = -adv * ratio * inv_batch
    if clipped_branch and outside_clip:
        d_log_prob = wp.float32(0.0)
    if wp.abs(ratio - wp.float32(1.0)) > clip_ratio:
        wp.atomic_add(clip_fraction, 0, inv_batch)

    for j in range(action_dim):
        mean = policy_out[row, j]
        raw_log_std = log_std_param[j]
        if state_dependent_std != 0:
            raw_log_std = policy_out[row, action_dim + j]
        log_std = _clip(raw_log_std, log_std_min, log_std_max)
        std = wp.exp(log_std)
        var = std * std
        a = actions[row, j]
        if squash != 0:
            a = _atanh_clamped(a)
        diff = a - mean
        policy_out_grad[row, j] = d_log_prob * diff / var
        raw_log_std_active = raw_log_std >= log_std_min and raw_log_std <= log_std_max
        d_log_std = wp.float32(0.0)
        if raw_log_std_active:
            d_log_std = d_log_prob * (diff * diff / var - wp.float32(1.0)) - entropy_coeff * inv_batch
        if state_dependent_std != 0:
            policy_out_grad[row, action_dim + j] = d_log_std
        else:
            partial = row / PPO_LOG_STD_PARTIAL_BATCH
            wp.atomic_add(log_std_grad_partials, partial, j, d_log_std)


@wp.kernel
def mirrored_action_mse_grad_kernel(
    policy_out: wp.array2d[wp.float32],
    mirrored_policy_out: wp.array2d[wp.float32],
    action_mirror_src: wp.array[wp.int32],
    action_mirror_sign: wp.array[wp.float32],
    action_dim: wp.int32,
    coeff: wp.float32,
    batch_size: wp.int32,
    policy_out_grad: wp.array2d[wp.float32],
    loss: wp.array[wp.float32],
):
    row = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    row_loss = wp.float32(0.0)
    for action in range(action_dim):
        target = action_mirror_sign[action] * mirrored_policy_out[row, action_mirror_src[action]]
        delta = policy_out[row, action] - target
        policy_out_grad[row, action] = policy_out_grad[row, action] + coeff * delta * inv_batch
        row_loss = row_loss + wp.float32(0.5) * coeff * delta * delta * inv_batch
    wp.atomic_add(loss, 0, row_loss)


@wp.func
def _ppo_value_loss_term(
    value: wp.float32,
    old_value: wp.float32,
    target: wp.float32,
    value_loss_coeff: wp.float32,
    value_clip_range: wp.float32,
) -> wp.float32:
    delta = value - target
    loss_sq = delta * delta
    if value_clip_range > wp.float32(0.0):
        value_error = value - old_value
        clipped_value = old_value + _clip(value_error, -value_clip_range, value_clip_range)
        clipped_delta = clipped_value - target
        clipped_loss_sq = clipped_delta * clipped_delta
        loss_sq = wp.max(loss_sq, clipped_loss_sq)
    return wp.float32(0.5) * value_loss_coeff * loss_sq


@wp.func
def _ppo_value_grad_term(
    value: wp.float32,
    old_value: wp.float32,
    target: wp.float32,
    value_loss_coeff: wp.float32,
    value_clip_range: wp.float32,
) -> wp.float32:
    delta = value - target
    if value_clip_range > wp.float32(0.0):
        value_error = value - old_value
        clipped_error = _clip(value_error, -value_clip_range, value_clip_range)
        clipped_delta = old_value + clipped_error - target
        if clipped_delta * clipped_delta > delta * delta:
            if value_error >= -value_clip_range and value_error <= value_clip_range:
                return value_loss_coeff * clipped_delta
            return wp.float32(0.0)
    return value_loss_coeff * delta


@wp.kernel
def value_loss_kernel(
    values: wp.array2d[wp.float32],
    old_values: wp.array[wp.float32],
    returns: wp.array[wp.float32],
    value_loss_coeff: wp.float32,
    value_clip_range: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
):
    i = wp.tid()
    loss_term = _ppo_value_loss_term(values[i, 0], old_values[i], returns[i], value_loss_coeff, value_clip_range)
    wp.atomic_add(loss, 0, loss_term / wp.float32(batch_size))


@wp.kernel
def value_loss_grad_kernel(
    values: wp.array2d[wp.float32],
    old_values: wp.array[wp.float32],
    returns: wp.array[wp.float32],
    value_loss_coeff: wp.float32,
    value_clip_range: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    value_grad: wp.array2d[wp.float32],
):
    i = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    value = values[i, 0]
    loss_term = _ppo_value_loss_term(value, old_values[i], returns[i], value_loss_coeff, value_clip_range)
    value_grad[i, 0] = (
        _ppo_value_grad_term(value, old_values[i], returns[i], value_loss_coeff, value_clip_range) * inv_batch
    )
    wp.atomic_add(loss, 0, loss_term * inv_batch)


@wp.kernel
def value_column_loss_grad_kernel(
    values: wp.array2d[wp.float32],
    value_col: wp.int32,
    old_values: wp.array[wp.float32],
    returns: wp.array[wp.float32],
    value_loss_coeff: wp.float32,
    value_clip_range: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    output_grad: wp.array2d[wp.float32],
):
    i = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    value = values[i, value_col]
    loss_term = _ppo_value_loss_term(value, old_values[i], returns[i], value_loss_coeff, value_clip_range)
    output_grad[i, value_col] = (
        _ppo_value_grad_term(value, old_values[i], returns[i], value_loss_coeff, value_clip_range) * inv_batch
    )
    wp.atomic_add(loss, 0, loss_term * inv_batch)


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
    row = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    row_loss = wp.float32(0.0)
    for action in range(action_dim):
        target = action_mirror_sign[action] * mirrored_policy_out[row, action_mirror_src[action]]
        delta = policy_out[row, action] - target
        row_loss = row_loss + wp.float32(0.5) * coeff * delta * delta * inv_batch
    wp.atomic_add(loss, 0, row_loss)


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
def value_symmetry_loss_grad_kernel(
    values: wp.array2d[wp.float32],
    mirrored_values: wp.array2d[wp.float32],
    coeff: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    value_grad: wp.array2d[wp.float32],
):
    row = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    delta = values[row, 0] - mirrored_values[row, 0]
    value_grad[row, 0] = value_grad[row, 0] + coeff * delta * inv_batch
    wp.atomic_add(loss, 0, wp.float32(0.5) * coeff * delta * delta * inv_batch)


@wp.kernel
def value_column_symmetry_loss_grad_kernel(
    values: wp.array2d[wp.float32],
    value_col: wp.int32,
    mirrored_values: wp.array2d[wp.float32],
    coeff: wp.float32,
    batch_size: wp.int32,
    loss: wp.array[wp.float32],
    output_grad: wp.array2d[wp.float32],
):
    row = wp.tid()
    inv_batch = wp.float32(1.0) / wp.float32(batch_size)
    delta = values[row, value_col] - mirrored_values[row, value_col]
    output_grad[row, value_col] = output_grad[row, value_col] + coeff * delta * inv_batch
    wp.atomic_add(loss, 0, wp.float32(0.5) * coeff * delta * delta * inv_batch)


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
def trajectory_priority_kernel(
    advantages: wp.array[wp.float32],
    num_steps: wp.int32,
    num_envs: wp.int32,
    priorities: wp.array[wp.float32],
):
    env = wp.tid()
    total = wp.float32(0.0)
    for step in range(num_steps):
        total = total + wp.abs(advantages[step * num_envs + env])
    priorities[env] = total


@wp.kernel
def trajectory_priority_weight_kernel(
    priorities: wp.array[wp.float32],
    priority_alpha: wp.float32,
    weights: wp.array[wp.float32],
    total_weight: wp.array[wp.float32],
):
    env = wp.tid()
    weight = wp.pow(wp.max(priorities[env], wp.float32(0.0)) + wp.float32(1.0e-6), priority_alpha)
    weights[env] = weight
    wp.atomic_add(total_weight, 0, weight)


@wp.kernel
def sample_trajectory_env_ids_kernel(
    priority_weights: wp.array[wp.float32],
    total_weight: wp.array[wp.float32],
    num_envs: wp.int32,
    seed: wp.int32,
    priority_beta: wp.float32,
    use_priority: wp.int32,
    env_ids: wp.array[wp.int32],
    importance_weights: wp.array[wp.float32],
):
    segment = wp.tid()
    rng = wp.rand_init(seed, segment)
    u = wp.randf(rng)
    env = wp.min(wp.int32(wp.floor(u * wp.float32(num_envs))), num_envs - wp.int32(1))
    importance = wp.float32(1.0)

    total = total_weight[0]
    if use_priority != wp.int32(0) and total > wp.float32(0.0):
        target = u * total
        cumulative = wp.float32(0.0)
        found = wp.int32(0)
        env = num_envs - wp.int32(1)
        for candidate in range(num_envs):
            cumulative = cumulative + priority_weights[candidate]
            if found == wp.int32(0) and target <= cumulative:
                env = candidate
                found = wp.int32(1)
        if priority_beta > wp.float32(0.0):
            prob = priority_weights[env] / total
            correction = wp.max(wp.float32(num_envs) * prob, wp.float32(1.0e-6))
            importance = wp.pow(correction, -priority_beta)

    env_ids[segment] = env
    importance_weights[segment] = importance


@wp.kernel
def sample_trajectory_env_ids_seed_counter_kernel(
    priority_weights: wp.array[wp.float32],
    total_weight: wp.array[wp.float32],
    num_envs: wp.int32,
    seed_counter: wp.array[wp.int32],
    seed_offset: wp.int32,
    priority_beta: wp.float32,
    use_priority: wp.int32,
    env_ids: wp.array[wp.int32],
    importance_weights: wp.array[wp.float32],
):
    segment = wp.tid()
    seed = wp.int32((wp.int64(seed_counter[0]) + wp.int64(seed_offset)) % wp.int64(2147483647))
    rng = wp.rand_init(seed, segment)
    u = wp.randf(rng)
    env = wp.min(wp.int32(wp.floor(u * wp.float32(num_envs))), num_envs - wp.int32(1))
    importance = wp.float32(1.0)

    total = total_weight[0]
    if use_priority != wp.int32(0) and total > wp.float32(0.0):
        target = u * total
        cumulative = wp.float32(0.0)
        found = wp.int32(0)
        env = num_envs - wp.int32(1)
        for candidate in range(num_envs):
            cumulative = cumulative + priority_weights[candidate]
            if found == wp.int32(0) and target <= cumulative:
                env = candidate
                found = wp.int32(1)
        if priority_beta > wp.float32(0.0):
            prob = priority_weights[env] / total
            correction = wp.max(wp.float32(num_envs) * prob, wp.float32(1.0e-6))
            importance = wp.pow(correction, -priority_beta)

    env_ids[segment] = env
    importance_weights[segment] = importance


@wp.kernel
def weight_trajectory_advantages_kernel(
    trajectory_weights: wp.array[wp.float32],
    segment_count: wp.int32,
    advantages: wp.array[wp.float32],
):
    row = wp.tid()
    step = row / segment_count
    segment = row - step * segment_count
    advantages[row] = advantages[row] * trajectory_weights[segment]


@wp.kernel
def gather_trajectory_minibatch_kernel(
    env_ids: wp.array[wp.int32],
    src_num_envs: wp.int32,
    segment_count: wp.int32,
    obs_dim: wp.int32,
    action_dim: wp.int32,
    obs_src: wp.array2d[wp.float32],
    actions_src: wp.array2d[wp.float32],
    log_probs_src: wp.array[wp.float32],
    advantages_src: wp.array[wp.float32],
    returns_src: wp.array[wp.float32],
    values_src: wp.array[wp.float32],
    obs_dst: wp.array2d[wp.float32],
    actions_dst: wp.array2d[wp.float32],
    log_probs_dst: wp.array[wp.float32],
    advantages_dst: wp.array[wp.float32],
    returns_dst: wp.array[wp.float32],
    old_values_dst: wp.array[wp.float32],
):
    row, col = wp.tid()
    step = row / segment_count
    segment = row - step * segment_count
    env = env_ids[segment]
    src_row = step * src_num_envs + env
    if col < obs_dim:
        obs_dst[row, col] = obs_src[src_row, col]
    if col < action_dim:
        actions_dst[row, col] = actions_src[src_row, col]
    if col == 0:
        log_probs_dst[row] = log_probs_src[src_row]
        advantages_dst[row] = advantages_src[src_row]
        returns_dst[row] = returns_src[src_row]
        old_values_dst[row] = values_src[src_row]


@wp.kernel
def scatter_trajectory_ratios_kernel(
    env_ids: wp.array[wp.int32],
    src_num_envs: wp.int32,
    segment_count: wp.int32,
    ratios_src: wp.array[wp.float32],
    ratios_dst: wp.array[wp.float32],
):
    row = wp.tid()
    step = row / segment_count
    segment = row - step * segment_count
    env = env_ids[segment]
    dst_row = step * src_num_envs + env
    ratios_dst[dst_row] = ratios_src[row]


@wp.kernel
def scatter_trajectory_values_kernel(
    env_ids: wp.array[wp.int32],
    src_num_envs: wp.int32,
    segment_count: wp.int32,
    values_src: wp.array2d[wp.float32],
    value_col: wp.int32,
    values_dst: wp.array[wp.float32],
):
    row = wp.tid()
    step = row / segment_count
    segment = row - step * segment_count
    env = env_ids[segment]
    dst_row = step * src_num_envs + env
    values_dst[dst_row] = values_src[row, value_col]


@wp.kernel
def compute_gae_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    values: wp.array[wp.float32],
    num_steps: wp.int32,
    num_envs: wp.int32,
    gamma: wp.float32,
    gae_lambda: wp.float32,
    reward_clip: wp.float32,
    advantages: wp.array[wp.float32],
    returns: wp.array[wp.float32],
):
    env = wp.tid()
    gae = wp.float32(0.0)
    for t_rev in range(num_steps):
        t = num_steps - wp.int32(1) - t_rev
        idx = t * num_envs + env
        next_idx = (t + wp.int32(1)) * num_envs + env
        reward = rewards[idx]
        if reward_clip > wp.float32(0.0):
            reward = _clip(reward, -reward_clip, reward_clip)
        non_terminal = wp.float32(1.0) - dones[idx]
        delta = reward + gamma * values[next_idx] * non_terminal - values[idx]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[idx] = gae
        returns[idx] = gae + values[idx]


@wp.kernel
def compute_vtrace_returns_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    values: wp.array[wp.float32],
    ratios: wp.array[wp.float32],
    num_steps: wp.int32,
    num_envs: wp.int32,
    gamma: wp.float32,
    gae_lambda: wp.float32,
    rho_clip: wp.float32,
    c_clip: wp.float32,
    reward_clip: wp.float32,
    advantages: wp.array[wp.float32],
    returns: wp.array[wp.float32],
):
    env = wp.tid()
    trace = wp.float32(0.0)
    for t_rev in range(num_steps):
        t = num_steps - wp.int32(1) - t_rev
        idx = t * num_envs + env
        next_idx = (t + wp.int32(1)) * num_envs + env
        reward = rewards[idx]
        if reward_clip > wp.float32(0.0):
            reward = _clip(reward, -reward_clip, reward_clip)
        non_terminal = wp.float32(1.0) - dones[idx]
        rho = ratios[idx]
        c = ratios[idx]
        if rho_clip > wp.float32(0.0):
            rho = wp.min(rho, rho_clip)
        if c_clip > wp.float32(0.0):
            c = wp.min(c, c_clip)
        delta = rho * (reward + gamma * values[next_idx] * non_terminal - values[idx])
        trace = delta + gamma * gae_lambda * c * trace * non_terminal
        advantages[idx] = trace
        returns[idx] = trace + values[idx]


@wp.kernel
def sum_and_sumsq_kernel(x: wp.array[wp.float32], count: wp.int32, stats: wp.array[wp.float32]):
    i = wp.tid()
    if i < count:
        v = x[i]
        wp.atomic_add(stats, 0, v)
        wp.atomic_add(stats, 1, v * v)


@wp.kernel
def rollout_reward_done_success_sums_kernel(
    rewards: wp.array[wp.float32],
    dones: wp.array[wp.float32],
    successes: wp.array[wp.float32],
    count: wp.int32,
    sums: wp.array[wp.float32],
):
    i = wp.tid()
    if i < count:
        wp.atomic_add(sums, 0, rewards[i])
        wp.atomic_add(sums, 1, dones[i])
        wp.atomic_add(sums, 2, successes[i])


@wp.kernel
def normalize_kernel(x: wp.array[wp.float32], mean: wp.float32, inv_std: wp.float32, count: wp.int32):
    i = wp.tid()
    if i < count:
        x[i] = (x[i] - mean) * inv_std


@wp.kernel
def normalize_from_stats_kernel(x: wp.array[wp.float32], stats: wp.array[wp.float32], count: wp.int32, eps: wp.float32):
    i = wp.tid()
    if i < count:
        inv_count = wp.float32(1.0) / wp.float32(count)
        mean = stats[0] * inv_count
        var = wp.max(stats[1] * inv_count - mean * mean, wp.float32(0.0))
        inv_std = wp.float32(1.0) / wp.sqrt(var + eps)
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
def grad_sumsq_1d_kernel(grad: wp.array[wp.float32], grad_sumsq: wp.array[wp.float32]):
    i = wp.tid()
    g = grad[i]
    wp.atomic_add(grad_sumsq, 0, g * g)


@wp.kernel
def grad_sumsq_2d_kernel(grad: wp.array2d[wp.float32], grad_sumsq: wp.array[wp.float32]):
    i, j = wp.tid()
    g = grad[i, j]
    wp.atomic_add(grad_sumsq, 0, g * g)


@wp.func
def _grad_clip_scale(grad_sumsq: wp.array[wp.float32], max_grad_norm: wp.float32) -> wp.float32:
    scale = wp.float32(1.0)
    if max_grad_norm > wp.float32(0.0):
        norm = wp.sqrt(grad_sumsq[0])
        if norm > max_grad_norm:
            scale = max_grad_norm / (norm + wp.float32(1.0e-6))
    return scale


@wp.kernel
def adam_step_1d_kernel(
    param: wp.array[wp.float32],
    grad: wp.array[wp.float32],
    m: wp.array[wp.float32],
    v: wp.array[wp.float32],
    grad_sumsq: wp.array[wp.float32],
    step_corrections: wp.array[wp.float32],
    lr: wp.float32,
    lr_scale: wp.array[wp.float32],
    beta1: wp.float32,
    beta2: wp.float32,
    eps: wp.float32,
    weight_decay: wp.float32,
    max_grad_norm: wp.float32,
):
    i = wp.tid()
    beta1_correction = step_corrections[0]
    beta2_correction = step_corrections[1]
    g = _grad_clip_scale(grad_sumsq, max_grad_norm) * grad[i] + weight_decay * param[i]
    mi = beta1 * m[i] + (wp.float32(1.0) - beta1) * g
    vi = beta2 * v[i] + (wp.float32(1.0) - beta2) * g * g
    m[i] = mi
    v[i] = vi
    step_lr = lr * lr_scale[0]
    param[i] = param[i] - step_lr * (mi / beta1_correction) / (wp.sqrt(vi / beta2_correction) + eps)
    grad[i] = wp.float32(0.0)


@wp.kernel
def adam_step_2d_kernel(
    param: wp.array2d[wp.float32],
    grad: wp.array2d[wp.float32],
    m: wp.array2d[wp.float32],
    v: wp.array2d[wp.float32],
    grad_sumsq: wp.array[wp.float32],
    step_corrections: wp.array[wp.float32],
    lr: wp.float32,
    lr_scale: wp.array[wp.float32],
    beta1: wp.float32,
    beta2: wp.float32,
    eps: wp.float32,
    weight_decay: wp.float32,
    max_grad_norm: wp.float32,
):
    i, j = wp.tid()
    beta1_correction = step_corrections[0]
    beta2_correction = step_corrections[1]
    g = _grad_clip_scale(grad_sumsq, max_grad_norm) * grad[i, j] + weight_decay * param[i, j]
    mi = beta1 * m[i, j] + (wp.float32(1.0) - beta1) * g
    vi = beta2 * v[i, j] + (wp.float32(1.0) - beta2) * g * g
    m[i, j] = mi
    v[i, j] = vi
    step_lr = lr * lr_scale[0]
    param[i, j] = param[i, j] - step_lr * (mi / beta1_correction) / (wp.sqrt(vi / beta2_correction) + eps)
    grad[i, j] = wp.float32(0.0)


@wp.kernel
def optimizer_step_count_kernel(step_count: wp.array[wp.int32]):
    step_count[0] = step_count[0] + wp.int32(1)


@wp.kernel
def ppo_lr_scale_kernel(
    iteration: wp.array[wp.int32],
    num_samples: wp.int32,
    anneal_lr: wp.int32,
    anneal_timesteps: wp.int32,
    min_lr_ratio: wp.float32,
    actor_lr_scale: wp.array[wp.float32],
    critic_lr_scale: wp.array[wp.float32],
):
    scale = wp.float32(1.0)
    if anneal_lr != wp.int32(0) and anneal_timesteps > wp.int32(0):
        progress = wp.float32(iteration[0]) * wp.float32(num_samples) / wp.float32(anneal_timesteps)
        progress = wp.min(wp.max(progress, wp.float32(0.0)), wp.float32(1.0))
        min_ratio = wp.min(wp.max(min_lr_ratio, wp.float32(0.0)), wp.float32(1.0))
        scale = min_ratio + wp.float32(0.5) * (wp.float32(1.0) - min_ratio) * (
            wp.float32(1.0) + wp.cos(wp.float32(3.141592653589793) * progress)
        )
    actor_lr_scale[0] = scale
    critic_lr_scale[0] = scale


@wp.kernel
def muon_step_1d_kernel(
    param: wp.array[wp.float32],
    grad: wp.array[wp.float32],
    momentum_buffer: wp.array[wp.float32],
    grad_sumsq: wp.array[wp.float32],
    lr: wp.float32,
    lr_scale: wp.array[wp.float32],
    momentum: wp.float32,
    weight_decay: wp.float32,
    max_grad_norm: wp.float32,
):
    i = wp.tid()
    g = _grad_clip_scale(grad_sumsq, max_grad_norm) * grad[i]
    m = momentum * momentum_buffer[i] + g
    momentum_buffer[i] = m
    update = g + momentum * m
    step_lr = lr * lr_scale[0]
    param[i] = param[i] * (wp.float32(1.0) - step_lr * weight_decay) - step_lr * update
    grad[i] = wp.float32(0.0)


@wp.kernel
def muon_nesterov_2d_kernel(
    grad: wp.array2d[wp.float32],
    momentum_buffer: wp.array2d[wp.float32],
    grad_sumsq: wp.array[wp.float32],
    momentum: wp.float32,
    max_grad_norm: wp.float32,
    update: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    g = _grad_clip_scale(grad_sumsq, max_grad_norm) * grad[i, j]
    m = momentum * momentum_buffer[i, j] + g
    momentum_buffer[i, j] = m
    update[i, j] = g + momentum * m


@wp.kernel
def muon_normalize_2d_kernel(x: wp.array2d[wp.float32], norm_sumsq: wp.array[wp.float32], eps: wp.float32):
    i, j = wp.tid()
    inv_norm = wp.float32(1.0) / wp.max(wp.sqrt(norm_sumsq[0]), eps)
    x[i, j] = x[i, j] * inv_norm


@wp.kernel
def muon_gram_tall_kernel(x: wp.array2d[wp.float32], rows: wp.int32, gram: wp.array2d[wp.float32]):
    i, j = wp.tid()
    total = wp.float32(0.0)
    for row in range(rows):
        total = total + x[row, i] * x[row, j]
    gram[i, j] = total


@wp.kernel
def muon_gram_wide_kernel(x: wp.array2d[wp.float32], cols: wp.int32, gram: wp.array2d[wp.float32]):
    i, j = wp.tid()
    total = wp.float32(0.0)
    for col in range(cols):
        total = total + x[i, col] * x[j, col]
    gram[i, j] = total


@wp.kernel
def muon_poly_kernel(
    gram: wp.array2d[wp.float32],
    coeff_b: wp.float32,
    coeff_c: wp.float32,
    poly: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    total = wp.float32(0.0)
    dim = gram.shape[0]
    for k in range(dim):
        total = total + gram[i, k] * gram[k, j]
    poly[i, j] = coeff_c * total + coeff_b * gram[i, j]


@wp.kernel
def muon_ns_tall_kernel(
    x: wp.array2d[wp.float32],
    poly: wp.array2d[wp.float32],
    coeff_a: wp.float32,
    cols: wp.int32,
    dst: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    total = wp.float32(0.0)
    for k in range(cols):
        total = total + x[i, k] * poly[k, j]
    dst[i, j] = coeff_a * x[i, j] + total


@wp.kernel
def muon_ns_wide_kernel(
    x: wp.array2d[wp.float32],
    poly: wp.array2d[wp.float32],
    coeff_a: wp.float32,
    rows: wp.int32,
    dst: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    total = wp.float32(0.0)
    for k in range(rows):
        total = total + poly[i, k] * x[k, j]
    dst[i, j] = coeff_a * x[i, j] + total


@wp.kernel
def muon_step_2d_kernel(
    param: wp.array2d[wp.float32],
    grad: wp.array2d[wp.float32],
    update: wp.array2d[wp.float32],
    lr: wp.float32,
    lr_scale: wp.array[wp.float32],
    weight_decay: wp.float32,
    scale: wp.float32,
):
    i, j = wp.tid()
    step_lr = lr * lr_scale[0]
    param[i, j] = param[i, j] * (wp.float32(1.0) - step_lr * weight_decay) - step_lr * scale * update[i, j]
    grad[i, j] = wp.float32(0.0)
