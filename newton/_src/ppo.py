# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-file PPO implementation backed by Warp kernels.

Provides a complete Proximal Policy Optimization trainer for continuous
action spaces using diagonal-Gaussian policies.  All computation runs on
GPU via Warp -- no PyTorch dependency.  The tiled GEMM kernels from
:mod:`newton._src.onnx_runtime` are reused for the MLP forward pass, and
``wp.Tape`` handles automatic differentiation of the network layers.

Typical workflow::

    from newton._src.ppo import ActorCritic, PPOTrainer, export_actor_to_onnx

    ac = ActorCritic(obs_dim=48, act_dim=12, device="cuda:0")
    trainer = PPOTrainer(ac)
    trainer.train(env, total_timesteps=1_000_000)
    export_actor_to_onnx(ac.actor, obs_dim=48, path="policy.onnx")
"""

from __future__ import annotations

import functools
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import onnx
import warp as wp
from onnx import TensorProto, helper, numpy_helper

from newton._src.onnx_runtime import (
    _TILE_THREADS,
    _bias_add_kernel,
    _ceil_div,
    _elu_kernel,
    _make_tiled_gemm_transB_kernel,
    _pick_tile_sizes,
    _relu_kernel,
    _tanh_kernel,
)

# ---------------------------------------------------------------------------
# Running observation normalizer (Welford's online algorithm, on-device)
# ---------------------------------------------------------------------------


@wp.kernel
def _welford_update_kernel(
    obs: wp.array2d[float],
    count: wp.array[float],
    mean: wp.array[float],
    var: wp.array[float],
    batch_size: int,
):
    """Update running mean/var with a batch of observations (one thread per feature)."""
    j = wp.tid()
    for i in range(batch_size):
        x = obs[i, j]
        count[j] = count[j] + 1.0
        delta = x - mean[j]
        mean[j] = mean[j] + delta / count[j]
        delta2 = x - mean[j]
        var[j] = var[j] + delta * delta2


@wp.kernel
def _normalize_obs_kernel(
    obs: wp.array2d[float],
    mean: wp.array[float],
    inv_std: wp.array[float],
    out: wp.array2d[float],
):
    """Normalize observations: out = clamp((obs - mean) * inv_std, -10, 10)."""
    i, j = wp.tid()
    out[i, j] = wp.clamp((obs[i, j] - mean[j]) * inv_std[j], -10.0, 10.0)


@wp.kernel
def _compute_inv_std_kernel(var: wp.array[float], count: wp.array[float], inv_std: wp.array[float]):
    j = wp.tid()
    inv_std[j] = 1.0 / wp.sqrt(var[j] / wp.max(count[j], 1.0) + 1.0e-8)


class ObsNormalizer:
    """Running observation normalizer using Welford's online algorithm.

    Args:
        obs_dim: Observation vector dimension.
        device: Warp device string.
    """

    def __init__(self, obs_dim: int, device: str | None = None):
        self._device = wp.get_device(device)
        self._obs_dim = obs_dim
        d = self._device
        self.count = wp.zeros(obs_dim, dtype=wp.float32, device=d)
        self.mean = wp.zeros(obs_dim, dtype=wp.float32, device=d)
        self.var = wp.zeros(obs_dim, dtype=wp.float32, device=d)
        self.inv_std = wp.ones(obs_dim, dtype=wp.float32, device=d)

    def update_and_normalize(self, obs: wp.array, out: wp.array) -> None:
        """Update running statistics and write normalized obs to *out*."""
        batch = obs.shape[0]
        d = self._device
        wp.launch(
            _welford_update_kernel, dim=self._obs_dim, inputs=[obs, self.count, self.mean, self.var, batch], device=d
        )
        wp.launch(_compute_inv_std_kernel, dim=self._obs_dim, inputs=[self.var, self.count, self.inv_std], device=d)
        wp.launch(
            _normalize_obs_kernel, dim=(batch, self._obs_dim), inputs=[obs, self.mean, self.inv_std, out], device=d
        )

    def normalize(self, obs: wp.array, out: wp.array) -> None:
        """Normalize without updating statistics (for inference)."""
        batch = obs.shape[0]
        wp.launch(
            _normalize_obs_kernel,
            dim=(batch, self._obs_dim),
            inputs=[obs, self.mean, self.inv_std, out],
            device=self._device,
        )


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------


@wp.kernel
def _adamw_step_kernel(
    g: wp.array[float],
    m: wp.array[float],
    v: wp.array[float],
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    weight_decay: float,
    params: wp.array[float],
):
    i = wp.tid()
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i]
    mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))
    vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))
    params[i] = params[i] * (1.0 - lr * weight_decay) - lr * mhat / (wp.sqrt(vhat) + eps)


class AdamW:
    """AdamW optimizer (Adam with decoupled weight decay).

    Args:
        params: Flat ``wp.array[float]`` parameter arrays.
        lr: Learning rate.
        betas: Momentum coefficients ``(beta1, beta2)``.
        eps: Numerical stability term.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: list[wp.array],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [wp.zeros_like(p) for p in params]
        self.v = [wp.zeros_like(p) for p in params]

    def step(self, grads: list[wp.array]) -> None:
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            wp.launch(
                _adamw_step_kernel,
                dim=len(p),
                inputs=[g, m, v, self.lr, self.beta1, self.beta2, float(self.t), self.eps, self.weight_decay, p],
                device=p.device,
            )
        self.t += 1


# ---------------------------------------------------------------------------
# WarpMLP -- differentiable tiled MLP
# ---------------------------------------------------------------------------

_ACTIVATION_KERNELS = {
    "elu": _elu_kernel,
    "relu": _relu_kernel,
    "tanh": _tanh_kernel,
}


@wp.kernel
def _layer_norm_kernel(
    x: wp.array2d[float],
    gamma: wp.array[float],
    beta: wp.array[float],
    out: wp.array2d[float],
    width: int,
):
    """Per-row layer normalization: out[i] = gamma * (x[i] - mean) / std + beta."""
    i = wp.tid()
    # Mean
    mu = float(0.0)
    for j in range(width):
        mu = mu + x[i, j]
    mu = mu / float(width)
    # Variance
    var = float(0.0)
    for j in range(width):
        d = x[i, j] - mu
        var = var + d * d
    var = var / float(width)
    inv_std = 1.0 / wp.sqrt(var + 1.0e-5)
    # Normalize + affine
    for j in range(width):
        out[i, j] = gamma[j] * (x[i, j] - mu) * inv_std + beta[j]


def _orthogonal_init(shape: tuple[int, int], gain: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Orthogonal weight initialization (standard for PPO)."""
    rows, cols = shape
    rng = np.random.default_rng(seed)
    n = max(rows, cols)
    flat = rng.standard_normal((n, n)).astype(np.float32)
    q, r = np.linalg.qr(flat)
    q *= np.sign(np.diag(r))
    return q[:rows, :cols] * gain


class WarpMLP:
    """Dense MLP with tiled GEMM forward pass and ``wp.Tape`` autodiff.

    Weights are stored in ``(out_dim, in_dim)`` layout (transB=1 convention)
    so the same tiled kernels used by :class:`OnnxRuntime` are reused here.

    Args:
        layer_sizes: Sequence of layer widths, e.g. ``[48, 128, 128, 128, 12]``.
        activation: Activation function name (``"elu"``, ``"relu"``, ``"tanh"``).
        layer_norm: Apply layer normalization after each hidden layer
            (before the activation).  Stabilises training and reduces
            sensitivity to input scale.
        device: Warp device string.
        output_gain: Orthogonal init gain for the last layer (0.01 is common for policy heads).
        seed: RNG seed for weight initialization reproducibility.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "elu",
        layer_norm: bool = True,
        device: str | None = None,
        output_gain: float = 1.0,
        seed: int | None = None,
    ):
        self._device = wp.get_device(device)
        self._activation = activation
        self._layer_norm = layer_norm
        self._act_kernel = _ACTIVATION_KERNELS.get(activation)
        if self._act_kernel is None:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATION_KERNELS)}")

        self.weights: list[wp.array] = []
        self.biases: list[wp.array] = []
        # LayerNorm parameters (gamma, beta) per hidden layer -- not on the output layer
        self.ln_gammas: list[wp.array] = []
        self.ln_betas: list[wp.array] = []
        self._intermediates: list[wp.array] = []

        num_layers = len(layer_sizes) - 1
        for i in range(num_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            gain = output_gain if i == num_layers - 1 else math.sqrt(2.0)
            layer_seed = None if seed is None else seed + i
            w_np = _orthogonal_init((fan_out, fan_in), gain=gain, seed=layer_seed)
            b_np = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(wp.array(w_np, dtype=wp.float32, device=self._device, requires_grad=True))
            self.biases.append(wp.array(b_np, dtype=wp.float32, device=self._device, requires_grad=True))
            # LayerNorm on hidden layers only
            if layer_norm and i < num_layers - 1:
                self.ln_gammas.append(
                    wp.array(np.ones(fan_out, dtype=np.float32), dtype=wp.float32, device=self._device, requires_grad=True)
                )
                self.ln_betas.append(
                    wp.array(np.zeros(fan_out, dtype=np.float32), dtype=wp.float32, device=self._device, requires_grad=True)
                )

        self._layer_sizes = layer_sizes

    def alloc_intermediates(self, batch: int) -> None:
        """Pre-allocate intermediate buffers for a fixed batch size."""
        needed = len(self.weights)
        if len(self._intermediates) == needed and self._intermediates[0].shape[0] == batch:
            return
        self._intermediates = [
            wp.zeros((batch, self._layer_sizes[i + 1]), dtype=wp.float32, device=self._device, requires_grad=True)
            for i in range(needed)
        ]

    def forward(self, x: wp.array) -> wp.array:
        batch = x.shape[0]
        if not self._intermediates or self._intermediates[0].shape[0] != batch:
            self.alloc_intermediates(batch)

        inp = x
        ln_idx = 0
        for i, (w, b, out) in enumerate(zip(self.weights, self.biases, self._intermediates, strict=True)):
            M, K = batch, self._layer_sizes[i]
            N = self._layer_sizes[i + 1]
            tile_m, tile_n, tile_k = _pick_tile_sizes(M, N, K)
            grid = (_ceil_div(M, tile_m), _ceil_div(N, tile_n))
            kernel = _make_tiled_gemm_transB_kernel(tile_m, tile_n, tile_k)

            out.zero_()
            wp.launch_tiled(kernel, dim=list(grid), inputs=[inp, w, out], block_dim=_TILE_THREADS, device=self._device)
            wp.launch(_bias_add_kernel, dim=(M, N), inputs=[out, b, 1.0, 1.0], device=self._device)

            is_last = i == len(self.weights) - 1
            if not is_last:
                # LayerNorm (before activation)
                if self._layer_norm:
                    gamma = self.ln_gammas[ln_idx]
                    beta = self.ln_betas[ln_idx]
                    wp.launch(_layer_norm_kernel, dim=M, inputs=[out, gamma, beta, out, N], device=self._device)
                    ln_idx += 1
                # Activation
                if self._activation == "elu":
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out, 1.0], device=self._device)
                else:
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out], device=self._device)

            inp = out

        return self._intermediates[-1]

    def parameters(self) -> list[wp.array]:
        """Return flat list of all trainable parameter arrays."""
        params = []
        ln_idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            params.append(w.flatten())
            params.append(b)
            if self._layer_norm and i < len(self.weights) - 1:
                params.append(self.ln_gammas[ln_idx])
                params.append(self.ln_betas[ln_idx])
                ln_idx += 1
        return params

    def grad_arrays(self) -> list[wp.array]:
        """Return flat list of gradient arrays matching :meth:`parameters`."""
        grads = []
        ln_idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            grads.append(w.grad.flatten())
            grads.append(b.grad)
            if self._layer_norm and i < len(self.weights) - 1:
                grads.append(self.ln_gammas[ln_idx].grad)
                grads.append(self.ln_betas[ln_idx].grad)
                ln_idx += 1
        return grads


# ---------------------------------------------------------------------------
# Gaussian action sampling (on-device via wp.rand_init / wp.randn)
# ---------------------------------------------------------------------------

LOG2PI = wp.constant(float(math.log(2.0 * math.pi)))
LOG2 = wp.constant(float(math.log(2.0)))


@wp.kernel
def _increment_counter_kernel(counter: wp.array[int]):
    """Increment a single-element counter on device (graph-capture safe)."""
    counter[0] = counter[0] + 1


@wp.kernel
def _sample_actions_kernel(
    mean: wp.array2d[float],
    log_std: wp.array[float],
    rng_counter: wp.array[int],
    use_tanh: int,
    actions: wp.array2d[float],
    pre_tanh: wp.array2d[float],
    log_probs: wp.array[float],
):
    """Sample actions.  When *use_tanh* is set, actions are squashed via
    ``tanh`` and the log-probability includes the numerically stable
    Jacobian correction from PPO+."""
    i, j = wp.tid()
    act_dim = log_std.shape[0]
    step_offset = rng_counter[0]
    rng = wp.rand_init(42, step_offset * (mean.shape[0] * act_dim) + i * act_dim + j)
    noise = wp.randn(rng)
    std = wp.exp(log_std[j])
    u = mean[i, j] + std * noise
    pre_tanh[i, j] = u

    lp = -0.5 * (noise * noise + 2.0 * log_std[j] + LOG2PI)
    if use_tanh > 0:
        actions[i, j] = wp.tanh(u)
        # Numerically stable log |det d(tanh)/du| correction (PPO+ Eq. from Sec 4.1)
        lp = lp - 2.0 * (LOG2 - u - wp.log(1.0 + wp.exp(-2.0 * u)))
    else:
        actions[i, j] = u

    wp.atomic_add(log_probs, i, lp)


# ---------------------------------------------------------------------------
# Fused PPO loss + analytical gradient kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _ppo_fused_loss_and_grad_kernel(
    mean: wp.array2d[float],
    log_std: wp.array[float],
    pre_tanh_actions: wp.array2d[float],
    old_log_probs: wp.array[float],
    advantages: wp.array[float],
    values_flat: wp.array[float],
    returns: wp.array[float],
    log_alpha: wp.array[float],
    clip_ratio: float,
    value_coef: float,
    use_tanh: int,
    n: int,
    act_dim: int,
    loss: wp.array[float],
    grad_mean: wp.array2d[float],
    grad_values: wp.array[float],
    grad_log_std: wp.array[float],
    grad_log_alpha: wp.array[float],
):
    """Fused PPO loss forward + analytical gradient computation.

    Computes the PPO clipped surrogate loss, value loss, and entropy bonus
    in a single kernel.  The entropy coefficient ``alpha = exp(log_alpha)``
    is read from a device array so it can be auto-tuned.  When
    ``use_tanh > 0``, the log-probability includes the tanh Jacobian
    correction.

    Writes exact gradients w.r.t. ``mean``, ``values``, ``log_std``, and
    ``log_alpha``.  ``wp.Tape`` is used only for the MLP backward pass.
    """
    i = wp.tid()
    inv_n = 1.0 / float(n)
    alpha = wp.exp(log_alpha[0])

    # -- Log-prob under current policy --
    new_lp = float(0.0)
    for j in range(act_dim):
        std_j = wp.exp(log_std[j])
        u_j = pre_tanh_actions[i, j]
        z = (u_j - mean[i, j]) / std_j
        new_lp = new_lp - 0.5 * (z * z + 2.0 * log_std[j] + LOG2PI)
        if use_tanh > 0:
            new_lp = new_lp - 2.0 * (LOG2 - u_j - wp.log(1.0 + wp.exp(-2.0 * u_j)))

    # -- PPO clipped surrogate --
    log_ratio = wp.clamp(new_lp - old_log_probs[i], -20.0, 20.0)
    ratio = wp.exp(log_ratio)
    adv = advantages[i]
    surr1 = ratio * adv
    surr2 = wp.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    clipped = surr2 < surr1
    policy_loss = -wp.min(surr1, surr2)

    # -- Value loss --
    vdiff = values_flat[i] - returns[i]
    value_loss = 0.5 * vdiff * vdiff

    # -- Entropy (diagonal Gaussian): sum_j (0.5 * log(2*pi*e) + log_std_j) --
    entropy = float(0.0)
    for j in range(act_dim):
        entropy = entropy + 0.5 * (LOG2PI + 1.0) + log_std[j]

    # -- Total loss --
    total = (policy_loss + value_coef * value_loss - alpha * entropy) * inv_n
    wp.atomic_add(loss, 0, total)

    # -- Analytical gradients --
    d_policy_d_lp = 0.0
    if not clipped:
        d_policy_d_lp = -ratio * adv
    d_total_d_lp = d_policy_d_lp * inv_n

    grad_values[i] = value_coef * vdiff * inv_n

    for j in range(act_dim):
        std_j = wp.exp(log_std[j])
        z = (pre_tanh_actions[i, j] - mean[i, j]) / std_j

        # d(log_prob)/d(mean_ij) = z / std_j  (tanh correction is independent of mean)
        grad_mean[i, j] = d_total_d_lp * z / std_j

        # d(log_prob)/d(log_std_j) = z^2 - 1;  d(entropy)/d(log_std_j) = 1
        d_logstd = d_total_d_lp * (z * z - 1.0) - alpha * inv_n
        wp.atomic_add(grad_log_std, j, d_logstd)

    # -- Alpha gradient: d/d(log_alpha) of -alpha * entropy / n --
    # alpha_loss per sample = -alpha * (log_pi + target_entropy) but we fold
    # the entropy term from the PPO loss here: d(-alpha*entropy*inv_n)/d(log_alpha) = -alpha*entropy*inv_n
    # The auto-tune target is handled separately via target_entropy in the trainer.
    wp.atomic_add(grad_log_alpha, 0, -alpha * entropy * inv_n)


# ---------------------------------------------------------------------------
# Buffer copy / flatten / gather kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _copy_2d_to_3d_slice(src: wp.array2d[float], dst: wp.array3d[float], t: int):
    i, j = wp.tid()
    dst[t, i, j] = src[i, j]


@wp.kernel
def _copy_1d_to_2d_slice(src: wp.array[float], dst: wp.array2d[float], t: int):
    i = wp.tid()
    dst[t, i] = src[i]


@wp.kernel
def _flatten_3d_to_2d(src: wp.array3d[float], dst: wp.array2d[float], num_envs: int):
    t, e, d = wp.tid()
    dst[t * num_envs + e, d] = src[t, e, d]


@wp.kernel
def _flatten_2d_to_1d(src: wp.array2d[float], dst: wp.array[float], num_envs: int):
    t, e = wp.tid()
    dst[t * num_envs + e] = src[t, e]


@wp.kernel
def _gather_2d_offset(src: wp.array2d[float], indices: wp.array[int], offset: int, dst: wp.array2d[float]):
    i, j = wp.tid()
    dst[i, j] = src[indices[offset + i], j]


@wp.kernel
def _gather_1d_offset(src: wp.array[float], indices: wp.array[int], offset: int, dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[indices[offset + i]]


# ---------------------------------------------------------------------------
# Tiled block-sum reduction (graph-capture safe, zero alloc in hot path)
# ---------------------------------------------------------------------------


@functools.cache
def _make_block_sum_kernel(tile_size: int):
    @wp.kernel
    def _block_sum_kernel(
        data: wp.array[float],
        arr_length: int,
        partial_sums: wp.array[float],
        final_sum: wp.array[float],
    ):
        block_id = wp.tid()
        start = block_id * tile_size
        if start >= arr_length:
            return
        t = wp.tile_load(data, shape=tile_size, offset=start)
        num_threads_per_block = wp.block_dim()
        if start + tile_size > arr_length:
            num_iterations = (tile_size + num_threads_per_block - 1) // num_threads_per_block
            for ii in range(num_iterations):
                linear_index = ii % tile_size
                value = t[linear_index]
                if start + linear_index >= arr_length:
                    value = 0.0
                t[linear_index] = value
        tile_sum = wp.tile_sum(t)
        partial_sums[block_id] = tile_sum[0]
        _, tid_block = wp.tid()
        if block_id == 0 and tid_block == 0 and arr_length <= tile_size:
            final_sum[0] = tile_sum[0]

    return _block_sum_kernel


class _ArraySum:
    """Pre-allocated hierarchical tiled reduction for ``sum(array)``."""

    TILE_SIZE = 512

    def __init__(self, max_length: int, device: str | None = None):
        self._device = wp.get_device(device)
        ts = self.TILE_SIZE
        num_blocks = _ceil_div(max_length, ts)
        self._partial_a = wp.zeros(num_blocks, dtype=wp.float32, device=self._device)
        self._partial_b = wp.zeros(num_blocks, dtype=wp.float32, device=self._device)
        self._kernel = _make_block_sum_kernel(ts)

    def compute(self, data: wp.array, length: int, result: wp.array) -> None:
        """Reduce *data* [:length] into *result* [0]. All on device."""
        ts = self.TILE_SIZE
        arr_len = length
        num_blocks = _ceil_div(arr_len, ts)
        src = data
        flip = 0
        while True:
            partial = self._partial_a if flip == 0 else self._partial_b
            wp.launch_tiled(
                self._kernel,
                dim=num_blocks,
                inputs=[src, arr_len, partial, result],
                block_dim=ts,
                device=self._device,
            )
            if num_blocks == 1:
                break
            arr_len = num_blocks
            num_blocks = _ceil_div(arr_len, ts)
            src = partial
            flip = 1 - flip


@wp.kernel
def _sq_diff_kernel(src: wp.array[float], mean_val: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    d = src[i] - mean_val[0]
    dst[i] = d * d


@wp.kernel
def _normalize_adv_kernel(
    adv: wp.array[float], mean_val: wp.array[float], inv_std: wp.array[float], out: wp.array[float]
):
    i = wp.tid()
    out[i] = (adv[i] - mean_val[0]) * inv_std[0]


@wp.kernel
def _scale_scalar_kernel(src: wp.array[float], scale: float, dst: wp.array[float]):
    dst[0] = src[0] * scale


@wp.kernel
def _inv_std_kernel(var_sum: wp.array[float], n: float, out: wp.array[float]):
    out[0] = 1.0 / (wp.sqrt(var_sum[0] / n) + 1.0e-8)


# ---------------------------------------------------------------------------
# On-device shuffle via random-key sort
# ---------------------------------------------------------------------------


@wp.kernel
def _generate_random_keys_kernel(
    keys: wp.array[float],
    indices: wp.array[int],
    rng_counter: wp.array[int],
    n: int,
):
    """Generate random sort keys and identity indices for GPU shuffle."""
    i = wp.tid()
    if i < n:
        epoch_offset = rng_counter[0]
        rng = wp.rand_init(42, epoch_offset * n + i)
        keys[i] = wp.randf(rng)
        indices[i] = i


# ---------------------------------------------------------------------------
# Gradient clipping (fully on-device)
# ---------------------------------------------------------------------------


@wp.kernel
def _grad_norm_sq_kernel(g: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, g[i] * g[i])


@wp.kernel
def _grad_clip_if_needed_kernel(
    g: wp.array[float],
    norm_sq: wp.array[float],
    max_norm_sq: float,
    max_norm: float,
):
    i = wp.tid()
    nsq = norm_sq[0]
    if nsq > max_norm_sq:
        scale = max_norm / (wp.sqrt(nsq) + 1.0e-6)
        g[i] = g[i] * scale


# ---------------------------------------------------------------------------
# GAE kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _gae_kernel(
    rewards: wp.array2d[float],
    values: wp.array2d[float],
    dones: wp.array2d[float],
    last_values: wp.array[float],
    advantages: wp.array2d[float],
    returns: wp.array2d[float],
    gamma: float,
    gae_lambda: float,
    num_steps: int,
):
    env = wp.tid()
    last_gae = float(0.0)
    for t in range(num_steps - 1, -1, -1):
        if t == num_steps - 1:
            next_val = last_values[env]
        else:
            next_val = values[t + 1, env]
        next_non_terminal = 1.0 - dones[t, env]
        delta = rewards[t, env] + gamma * next_val * next_non_terminal - values[t, env]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t, env] = last_gae
        returns[t, env] = last_gae + values[t, env]


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------


def _clone_mlp_for_batch(source: WarpMLP, batch: int) -> WarpMLP:
    """Create a WarpMLP that shares weights/LN params but has its own intermediates."""
    clone = WarpMLP.__new__(WarpMLP)
    clone.__dict__.update(source.__dict__)
    clone._intermediates = []
    clone.alloc_intermediates(batch)
    clone.weights = source.weights
    clone.biases = source.biases
    clone.ln_gammas = source.ln_gammas
    clone.ln_betas = source.ln_betas
    return clone


class ActorCritic:
    """Gaussian actor-critic with separate MLP networks.

    The default constructor builds standard :class:`WarpMLP` networks.
    Pass pre-built ``actor`` / ``critic`` networks for full control over
    architecture.  Any object with ``forward``, ``parameters``,
    ``grad_arrays``, and ``alloc_intermediates`` methods works.

    Args:
        obs_dim: Observation vector dimension.
        act_dim: Action vector dimension.
        hidden_sizes: Hidden layer widths (ignored when *actor*/*critic* are supplied).
        activation: Activation function name.
        layer_norm: Apply layer normalization in the default MLP.
        init_log_std: Initial value for the learnable log standard deviation.
        bounded_actions: Apply ``tanh`` squashing to bound actions to [-1, 1].
        device: Warp device string.
        seed: RNG seed for weight initialization.
        actor: Pre-built actor network (overrides *hidden_sizes*/*activation*/*layer_norm*).
        critic: Pre-built critic network.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "elu",
        layer_norm: bool = True,
        init_log_std: float = 0.0,
        bounded_actions: bool = True,
        device: str | None = None,
        seed: int | None = None,
        actor: Any | None = None,
        critic: Any | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]
        self._device = wp.get_device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.bounded_actions = bounded_actions
        self._use_tanh_int = 1 if bounded_actions else 0

        if actor is not None:
            self.actor = actor
        else:
            actor_seed = None if seed is None else seed
            self.actor = WarpMLP(
                [obs_dim, *hidden_sizes, act_dim],
                activation=activation,
                layer_norm=layer_norm,
                device=device,
                output_gain=0.01,
                seed=actor_seed,
            )
        if critic is not None:
            self.critic = critic
        else:
            critic_seed = None if seed is None else seed + 1000
            self.critic = WarpMLP(
                [obs_dim, *hidden_sizes, 1],
                activation=activation,
                layer_norm=layer_norm,
                device=device,
                output_gain=1.0,
                seed=critic_seed,
            )
        self.log_std = wp.array(
            np.full(act_dim, init_log_std, dtype=np.float32),
            dtype=wp.float32,
            device=self._device,
            requires_grad=True,
        )

    def alloc_buffers(self, rollout_batch: int, minibatch_size: int) -> None:
        """Pre-allocate all internal buffers for known batch sizes."""
        d = self._device
        self.actor.alloc_intermediates(rollout_batch)
        self.critic.alloc_intermediates(rollout_batch)

        self._act_actions = wp.zeros((rollout_batch, self.act_dim), dtype=wp.float32, device=d)
        self._act_pre_tanh = wp.zeros((rollout_batch, self.act_dim), dtype=wp.float32, device=d)
        self._act_log_probs = wp.zeros(rollout_batch, dtype=wp.float32, device=d)

        # Minibatch-sized MLP clones (share weights/LN params, separate intermediates)
        self._actor_mb = _clone_mlp_for_batch(self.actor, minibatch_size)
        self._critic_mb = _clone_mlp_for_batch(self.critic, minibatch_size)

    def act(self, obs: wp.array, rng_counter: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Forward actor + critic, sample actions using on-device RNG.

        Returns:
            ``(actions, log_probs, values)`` -- all on device, pre-allocated.
            When ``bounded_actions`` is set, actions are in [-1, 1].
        """
        batch = obs.shape[0]
        mean = self.actor.forward(obs)
        values_2d = self.critic.forward(obs)

        self._act_actions.zero_()
        self._act_log_probs.zero_()

        wp.launch(
            _sample_actions_kernel,
            dim=(batch, self.act_dim),
            inputs=[
                mean,
                self.log_std,
                rng_counter,
                self._use_tanh_int,
                self._act_actions,
                self._act_pre_tanh,
                self._act_log_probs,
            ],
            device=self._device,
        )
        wp.launch(_increment_counter_kernel, dim=1, inputs=[rng_counter], device=self._device)

        return self._act_actions, self._act_log_probs, values_2d.flatten()

    def parameters(self) -> list[wp.array]:
        return self.actor.parameters() + self.critic.parameters() + [self.log_std]


# ---------------------------------------------------------------------------
# Rollout buffer with GAE
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-size on-device rollout storage with GAE computation.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Rollout horizon (steps per env before update).
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        device: Warp device string.
    """

    def __init__(self, num_envs: int, num_steps: int, obs_dim: int, act_dim: int, device: str | None = None):
        self._device = wp.get_device(device)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        d = self._device
        self.observations = wp.zeros((num_steps, num_envs, obs_dim), dtype=wp.float32, device=d)
        self.actions = wp.zeros((num_steps, num_envs, act_dim), dtype=wp.float32, device=d)
        self.pre_tanh_actions = wp.zeros((num_steps, num_envs, act_dim), dtype=wp.float32, device=d)
        self.log_probs = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.rewards = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.dones = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.values = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.advantages = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.returns = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)

        n = num_steps * num_envs
        self.flat_obs = wp.zeros((n, obs_dim), dtype=wp.float32, device=d)
        self.flat_actions = wp.zeros((n, act_dim), dtype=wp.float32, device=d)
        self.flat_pre_tanh = wp.zeros((n, act_dim), dtype=wp.float32, device=d)
        self.flat_log_probs = wp.zeros(n, dtype=wp.float32, device=d)
        self.flat_advantages = wp.zeros(n, dtype=wp.float32, device=d)
        self.flat_returns = wp.zeros(n, dtype=wp.float32, device=d)

    def insert(
        self,
        t: int,
        obs: wp.array,
        actions: wp.array,
        pre_tanh: wp.array,
        log_probs: wp.array,
        rewards: wp.array,
        dones: wp.array,
        values: wp.array,
    ) -> None:
        d = self._device
        wp.launch(_copy_2d_to_3d_slice, dim=(self.num_envs, self.obs_dim), inputs=[obs, self.observations, t], device=d)
        wp.launch(_copy_2d_to_3d_slice, dim=(self.num_envs, self.act_dim), inputs=[actions, self.actions, t], device=d)
        wp.launch(
            _copy_2d_to_3d_slice,
            dim=(self.num_envs, self.act_dim),
            inputs=[pre_tanh, self.pre_tanh_actions, t],
            device=d,
        )
        wp.launch(_copy_1d_to_2d_slice, dim=self.num_envs, inputs=[log_probs, self.log_probs, t], device=d)
        wp.launch(_copy_1d_to_2d_slice, dim=self.num_envs, inputs=[rewards, self.rewards, t], device=d)
        wp.launch(_copy_1d_to_2d_slice, dim=self.num_envs, inputs=[dones, self.dones, t], device=d)
        wp.launch(_copy_1d_to_2d_slice, dim=self.num_envs, inputs=[values, self.values, t], device=d)

    def compute_gae(self, last_values: wp.array, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        wp.launch(
            _gae_kernel,
            dim=self.num_envs,
            inputs=[
                self.rewards,
                self.values,
                self.dones,
                last_values,
                self.advantages,
                self.returns,
                gamma,
                gae_lambda,
                self.num_steps,
            ],
            device=self._device,
        )

    def flatten(self) -> None:
        d = self._device
        ne = self.num_envs
        wp.launch(
            _flatten_3d_to_2d,
            dim=(self.num_steps, ne, self.obs_dim),
            inputs=[self.observations, self.flat_obs, ne],
            device=d,
        )
        wp.launch(
            _flatten_3d_to_2d,
            dim=(self.num_steps, ne, self.act_dim),
            inputs=[self.actions, self.flat_actions, ne],
            device=d,
        )
        wp.launch(
            _flatten_3d_to_2d,
            dim=(self.num_steps, ne, self.act_dim),
            inputs=[self.pre_tanh_actions, self.flat_pre_tanh, ne],
            device=d,
        )
        wp.launch(
            _flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.log_probs, self.flat_log_probs, ne], device=d
        )
        wp.launch(
            _flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.advantages, self.flat_advantages, ne], device=d
        )
        wp.launch(_flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.returns, self.flat_returns, ne], device=d)

    def mean_reward(self) -> float:
        """Mean per-step reward (single device-to-host readback for logging)."""
        return float(self.rewards.numpy().mean())


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Proximal Policy Optimization trainer for continuous action spaces.

    Args:
        actor_critic: The :class:`ActorCritic` to train.
        num_envs: Number of parallel environments.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_ratio: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient (used when ``auto_entropy=False``).
        value_coef: Value loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        num_epochs: PPO epochs per rollout.
        num_minibatches: Number of mini-batches per epoch.
        weight_decay: AdamW weight decay.
        num_steps: Rollout horizon per update.
        seed: RNG seed for on-device random number generation.
        auto_entropy: Auto-tune the entropy coefficient (SAC-style).
            When ``True``, ``entropy_coef`` is used only as the initial value
            and ``log_alpha`` is optimized to reach a target entropy of
            ``-act_dim``.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        num_envs: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        num_epochs: int = 5,
        num_minibatches: int = 4,
        weight_decay: float = 1e-4,
        num_steps: int = 24,
        seed: int = 42,
        auto_entropy: bool = True,
    ):
        self.ac = actor_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm_sq = max_grad_norm * max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.num_steps = num_steps
        self.auto_entropy = auto_entropy

        device = actor_critic._device

        # Entropy coefficient: learnable log_alpha on device
        init_log_alpha = math.log(max(entropy_coef, 1e-8))
        self._log_alpha = wp.array([init_log_alpha], dtype=wp.float32, device=device)
        self._grad_log_alpha = wp.zeros(1, dtype=wp.float32, device=device)
        self._target_entropy = -float(actor_critic.act_dim)
        if auto_entropy:
            self._alpha_optimizer = AdamW([self._log_alpha], lr=lr, weight_decay=0.0)

        self._rng_counter = wp.array([seed], dtype=wp.int32, device=device)
        n_samples = num_steps * num_envs
        batch_size = n_samples // num_minibatches

        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_dim=actor_critic.obs_dim,
            act_dim=actor_critic.act_dim,
            device=str(device),
        )

        self.obs_normalizer = ObsNormalizer(actor_critic.obs_dim, device=str(device))
        self._norm_obs = wp.zeros((num_envs, actor_critic.obs_dim), dtype=wp.float32, device=device)

        actor_critic.alloc_buffers(num_envs, batch_size)

        self.optimizer = AdamW(actor_critic.parameters(), lr=lr, weight_decay=weight_decay)

        # Advantage normalization scratch (on-device, no readbacks)
        self._norm_adv = wp.zeros(n_samples, dtype=wp.float32, device=device)
        self._sq_diff = wp.zeros(n_samples, dtype=wp.float32, device=device)
        self._sum_result = wp.zeros(1, dtype=wp.float32, device=device)
        self._mean_buf = wp.zeros(1, dtype=wp.float32, device=device)
        self._var_result = wp.zeros(1, dtype=wp.float32, device=device)
        self._inv_std_buf = wp.zeros(1, dtype=wp.float32, device=device)
        self._sum_reducer = _ArraySum(n_samples, device=str(device))

        # Shuffle arrays (2x capacity required by radix_sort_pairs)
        self._shuffle_keys = wp.zeros(2 * n_samples, dtype=wp.float32, device=device)
        self._indices = wp.zeros(2 * n_samples, dtype=wp.int32, device=device)

        # Mini-batch gather buffers
        self._mb_obs = wp.zeros((batch_size, actor_critic.obs_dim), dtype=wp.float32, device=device, requires_grad=True)
        self._mb_act = wp.zeros((batch_size, actor_critic.act_dim), dtype=wp.float32, device=device)
        self._mb_pre_tanh = wp.zeros((batch_size, actor_critic.act_dim), dtype=wp.float32, device=device)
        self._mb_old_lp = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._mb_adv = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._mb_ret = wp.zeros(batch_size, dtype=wp.float32, device=device)

        # Loss and gradient scratch
        self._loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        self._grad_norm_sq = wp.zeros(1, dtype=wp.float32, device=device)
        self._grad_mean = wp.zeros((batch_size, actor_critic.act_dim), dtype=wp.float32, device=device)
        self._grad_values = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._grad_log_std = wp.zeros(actor_critic.act_dim, dtype=wp.float32, device=device)

        self._n_samples = n_samples
        self._batch_size = batch_size

        # Profiling (opt-in, near-zero overhead when disabled)
        self.profile = False
        self._evt_sim_start: wp.Event | None = None
        self._evt_sim_end: wp.Event | None = None
        self._evt_ppo_start: wp.Event | None = None
        self._evt_ppo_end: wp.Event | None = None
        self._time_sim_ms = 0.0
        self._time_ppo_ms = 0.0

    def enable_profiling(self) -> None:
        """Create CUDA timing events.  Call once before the training loop."""
        self.profile = True
        self._evt_sim_start = wp.Event(enable_timing=True)
        self._evt_sim_end = wp.Event(enable_timing=True)
        self._evt_ppo_start = wp.Event(enable_timing=True)
        self._evt_ppo_end = wp.Event(enable_timing=True)
        self._time_sim_ms = 0.0
        self._time_ppo_ms = 0.0

    def collect_rollouts(self, env: Any, obs: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Run the policy in *env* for ``num_steps`` and fill the buffer.

        Args:
            env: Vectorized environment.
            obs: Current observations.  Pass back the second return value
                to avoid resetting the environment between rollouts.

        Returns:
            ``(last_values, obs)`` for GAE bootstrapping and next call.
        """
        if obs is None:
            obs = env.reset()
        buf = self.buffer
        p = self.profile

        for t in range(buf.num_steps):
            if p:
                wp.record_event(self._evt_ppo_start)
            self.obs_normalizer.update_and_normalize(obs, self._norm_obs)
            actions, log_probs, values = self.ac.act(self._norm_obs, self._rng_counter)
            if p:
                wp.record_event(self._evt_ppo_end)
                self._time_ppo_ms += wp.get_event_elapsed_time(self._evt_ppo_start, self._evt_ppo_end)
                wp.record_event(self._evt_sim_start)

            next_obs, rewards, dones = env.step(actions)

            if p:
                wp.record_event(self._evt_sim_end)
                self._time_sim_ms += wp.get_event_elapsed_time(self._evt_sim_start, self._evt_sim_end)
                wp.record_event(self._evt_ppo_start)

            buf.insert(t, self._norm_obs, actions, self.ac._act_pre_tanh, log_probs, rewards, dones, values)

            if p:
                wp.record_event(self._evt_ppo_end)
                self._time_ppo_ms += wp.get_event_elapsed_time(self._evt_ppo_start, self._evt_ppo_end)

            obs = next_obs

        self.obs_normalizer.normalize(obs, self._norm_obs)
        _, _, last_values = self.ac.act(self._norm_obs, self._rng_counter)
        return last_values, obs

    def _normalize_advantages(self) -> None:
        """Normalize flat_advantages -> _norm_adv entirely on device."""
        n = self._n_samples
        d = self.ac._device
        src = self.buffer.flat_advantages

        # mean = sum(adv) / n
        self._sum_result.zero_()
        self._sum_reducer.compute(src, n, self._sum_result)
        wp.launch(_scale_scalar_kernel, dim=1, inputs=[self._sum_result, 1.0 / float(n), self._mean_buf], device=d)

        # var = sum((adv - mean)^2) / n
        wp.launch(_sq_diff_kernel, dim=n, inputs=[src, self._mean_buf, self._sq_diff], device=d)
        self._var_result.zero_()
        self._sum_reducer.compute(self._sq_diff, n, self._var_result)
        wp.launch(_inv_std_kernel, dim=1, inputs=[self._var_result, float(n), self._inv_std_buf], device=d)

        # out = (adv - mean) * inv_std
        wp.launch(
            _normalize_adv_kernel, dim=n, inputs=[src, self._mean_buf, self._inv_std_buf, self._norm_adv], device=d
        )

    def update(self) -> float:
        """Run PPO update epochs on the filled buffer.

        Returns:
            Average loss (single scalar readback for logging).
        """
        p = self.profile
        if p:
            wp.record_event(self._evt_ppo_start)

        self.buffer.flatten()
        self._normalize_advantages()

        n_samples = self._n_samples
        batch_size = self._batch_size
        device = self.ac._device

        for _epoch in range(self.num_epochs):
            wp.launch(
                _generate_random_keys_kernel,
                dim=n_samples,
                inputs=[self._shuffle_keys, self._indices, self._rng_counter, n_samples],
                device=device,
            )
            wp.launch(_increment_counter_kernel, dim=1, inputs=[self._rng_counter], device=device)
            wp.utils.radix_sort_pairs(self._shuffle_keys, self._indices, n_samples)

            for mb_idx in range(self.num_minibatches):
                offset = mb_idx * batch_size

                wp.launch(
                    _gather_2d_offset,
                    dim=(batch_size, self.ac.obs_dim),
                    inputs=[self.buffer.flat_obs, self._indices, offset, self._mb_obs],
                    device=device,
                )
                wp.launch(
                    _gather_2d_offset,
                    dim=(batch_size, self.ac.act_dim),
                    inputs=[self.buffer.flat_actions, self._indices, offset, self._mb_act],
                    device=device,
                )
                wp.launch(
                    _gather_2d_offset,
                    dim=(batch_size, self.ac.act_dim),
                    inputs=[self.buffer.flat_pre_tanh, self._indices, offset, self._mb_pre_tanh],
                    device=device,
                )
                wp.launch(
                    _gather_1d_offset,
                    dim=batch_size,
                    inputs=[self.buffer.flat_log_probs, self._indices, offset, self._mb_old_lp],
                    device=device,
                )
                wp.launch(
                    _gather_1d_offset,
                    dim=batch_size,
                    inputs=[self._norm_adv, self._indices, offset, self._mb_adv],
                    device=device,
                )
                wp.launch(
                    _gather_1d_offset,
                    dim=batch_size,
                    inputs=[self.buffer.flat_returns, self._indices, offset, self._mb_ret],
                    device=device,
                )

                # Forward MLPs under tape
                tape = wp.Tape()
                with tape:
                    mean = self.ac._actor_mb.forward(self._mb_obs)
                    values_2d = self.ac._critic_mb.forward(self._mb_obs)

                # Fused PPO loss + analytical gradients
                self._loss.zero_()
                self._grad_mean.zero_()
                self._grad_values.zero_()
                self._grad_log_std.zero_()
                self._grad_log_alpha.zero_()

                wp.launch(
                    _ppo_fused_loss_and_grad_kernel,
                    dim=batch_size,
                    inputs=[
                        mean,
                        self.ac.log_std,
                        self._mb_pre_tanh,
                        self._mb_old_lp,
                        self._mb_adv,
                        values_2d.flatten(),
                        self._mb_ret,
                        self._log_alpha,
                        self.clip_ratio,
                        self.value_coef,
                        self.ac._use_tanh_int,
                        batch_size,
                        self.ac.act_dim,
                        self._loss,
                        self._grad_mean,
                        self._grad_values,
                        self._grad_log_std,
                        self._grad_log_alpha,
                    ],
                    device=device,
                )

                # Backward through MLPs using analytical gradients
                tape.backward(grads={mean: self._grad_mean, values_2d: self._grad_values.reshape((batch_size, 1))})

                grads = self.ac.actor.grad_arrays() + self.ac.critic.grad_arrays() + [self._grad_log_std]
                self._clip_grad_norm(grads)
                self.optimizer.step(grads)

                # Auto-tune entropy coefficient
                if self.auto_entropy:
                    self._alpha_optimizer.step([self._grad_log_alpha])

                tape.zero()

        if p:
            wp.record_event(self._evt_ppo_end)
            self._time_ppo_ms += wp.get_event_elapsed_time(self._evt_ppo_start, self._evt_ppo_end)

    def _clip_grad_norm(self, grads: list[wp.array]) -> None:
        device = self.ac._device
        self._grad_norm_sq.zero_()
        for g in grads:
            wp.launch(_grad_norm_sq_kernel, dim=len(g), inputs=[g, self._grad_norm_sq], device=device)
        for g in grads:
            wp.launch(
                _grad_clip_if_needed_kernel,
                dim=len(g),
                inputs=[g, self._grad_norm_sq, self.max_grad_norm_sq, self.max_grad_norm],
                device=device,
            )

    def get_stats(self) -> dict[str, float]:
        """Read back training statistics from device arrays.

        Call this **after** :meth:`update` completes (outside any captured
        graph) to retrieve diagnostics.  All readbacks are in this single
        method so the training loop itself remains graph-capture safe.
        """
        stats = {
            "loss": float(self._loss.numpy()[0]),
            "mean_reward": self.buffer.mean_reward(),
            "alpha": float(np.exp(self._log_alpha.numpy()[0])),
        }
        if self.profile:
            stats["sim_ms"] = self._time_sim_ms
            stats["ppo_ms"] = self._time_ppo_ms
            self._time_sim_ms = 0.0
            self._time_ppo_ms = 0.0
        return stats

    def train(self, env: Any, total_timesteps: int, log_interval: int = 1) -> None:
        """Main training loop."""
        steps_per_update = self.buffer.num_envs * self.buffer.num_steps
        num_updates = total_timesteps // steps_per_update
        total_steps = 0

        obs = None
        for update_idx in range(num_updates):
            last_values, obs = self.collect_rollouts(env, obs)
            self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)
            self.update()
            total_steps += steps_per_update

            if (update_idx + 1) % log_interval == 0:
                stats = self.get_stats()
                print(
                    f"Update {update_idx + 1}/{num_updates} | steps={total_steps}"
                    f" | loss={stats['loss']:.4f} | mean_reward={stats['mean_reward']:.4f}"
                )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_actor_to_onnx(actor: WarpMLP, obs_dim: int, path: str) -> None:
    """Export a trained :class:`WarpMLP` actor to ONNX format.

    The exported model can be loaded by :class:`newton._src.onnx_runtime.OnnxRuntime`.

    Args:
        actor: The trained actor MLP.
        obs_dim: Observation dimension (input width).
        path: Output ``.onnx`` file path.
    """
    nodes = []
    initializers = []
    prev_output = "observation"

    activation_map = {"elu": "Elu", "relu": "Relu", "tanh": "Tanh"}
    onnx_act = activation_map.get(actor._activation, "Elu")

    ln_idx = 0
    for i, (w, b) in enumerate(zip(actor.weights, actor.biases, strict=True)):
        w_name = f"actor.{i * 2}.weight"
        b_name = f"actor.{i * 2}.bias"
        gemm_out = f"/actor/{i * 2}/Gemm_output_0"

        initializers.append(numpy_helper.from_array(w.numpy(), name=w_name))
        initializers.append(numpy_helper.from_array(b.numpy(), name=b_name))
        nodes.append(helper.make_node("Gemm", [prev_output, w_name, b_name], [gemm_out], alpha=1.0, beta=1.0, transB=1))
        prev_output = gemm_out

        is_last = i == len(actor.weights) - 1
        if not is_last:
            # LayerNorm (if present)
            if actor._layer_norm and ln_idx < len(actor.ln_gammas):
                gamma_name = f"actor.ln{ln_idx}.gamma"
                beta_name = f"actor.ln{ln_idx}.beta"
                ln_out = f"/actor/ln{ln_idx}/output"
                initializers.append(numpy_helper.from_array(actor.ln_gammas[ln_idx].numpy(), name=gamma_name))
                initializers.append(numpy_helper.from_array(actor.ln_betas[ln_idx].numpy(), name=beta_name))
                nodes.append(
                    helper.make_node("LayerNormalization", [prev_output, gamma_name, beta_name], [ln_out], epsilon=1e-5)
                )
                prev_output = ln_out
                ln_idx += 1
            # Activation
            act_out = f"/actor/{i * 2 + 1}/{onnx_act}_output_0"
            nodes.append(helper.make_node(onnx_act, [prev_output], [act_out]))
            prev_output = act_out

    nodes[-1].output[0] = "action"

    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch_size", obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch_size", actor._layer_sizes[-1]])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, path)


# ---------------------------------------------------------------------------
# Environment protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VecEnv(Protocol):
    """Minimal vectorized environment interface for PPO."""

    num_envs: int
    obs_dim: int
    act_dim: int

    def reset(self) -> wp.array:
        """Return initial observations ``(num_envs, obs_dim)``."""
        ...

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Return ``(obs, rewards, dones)`` -- all ``wp.array`` on device."""
        ...
