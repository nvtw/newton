# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-file PPO implementation backed by Warp kernels.

Provides a complete Proximal Policy Optimization trainer for continuous
action spaces using diagonal-Gaussian policies.  All computation runs on
GPU via Warp -- no PyTorch dependency.  The tiled GEMM kernels from
:mod:`newton._src.onnx_runtime` are reused for the MLP forward pass, and
``wp.Tape`` handles automatic differentiation.

**Zero-allocation hot path**: every buffer used during rollout collection,
GAE, advantage normalization, mini-batch shuffling, and the PPO update
loop is pre-allocated once.  The only host round-trip is a single scalar
readback for loss logging.  The pipeline is compatible with
``wp.ScopedCapture`` / CUDA graph capture.

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


def _orthogonal_init(shape: tuple[int, int], gain: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Orthogonal weight initialization (standard for PPO)."""
    rows, cols = shape
    rng = np.random.default_rng(seed)
    flat = rng.standard_normal((max(rows, cols), min(rows, cols))).astype(np.float32)
    q, r = np.linalg.qr(flat)
    q *= np.sign(np.diag(r))
    q = q[:rows, :cols] * gain
    return q


class WarpMLP:
    """Dense MLP with tiled GEMM forward pass and ``wp.Tape`` autodiff.

    Weights are stored in ``(out_dim, in_dim)`` layout (transB=1 convention)
    so the same tiled kernels used by :class:`OnnxRuntime` are reused here.

    Args:
        layer_sizes: Sequence of layer widths, e.g. ``[48, 128, 128, 128, 12]``.
        activation: Activation function name (``"elu"``, ``"relu"``, ``"tanh"``).
        device: Warp device string.
        output_gain: Orthogonal init gain for the last layer (0.01 is common for policy heads).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "elu",
        device: str | None = None,
        output_gain: float = 1.0,
        seed: int | None = None,
    ):
        self._device = wp.get_device(device)
        self._activation = activation
        self._act_kernel = _ACTIVATION_KERNELS.get(activation)
        if self._act_kernel is None:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATION_KERNELS)}")

        self.weights: list[wp.array] = []
        self.biases: list[wp.array] = []
        self._intermediates: list[wp.array] = []

        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            gain = output_gain if i == len(layer_sizes) - 2 else math.sqrt(2.0)
            layer_seed = None if seed is None else seed + i
            w_np = _orthogonal_init((fan_out, fan_in), gain=gain, seed=layer_seed)
            b_np = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(wp.array(w_np, dtype=wp.float32, device=self._device, requires_grad=True))
            self.biases.append(wp.array(b_np, dtype=wp.float32, device=self._device, requires_grad=True))

        self._layer_sizes = layer_sizes

    def alloc_intermediates(self, batch: int) -> None:
        """Pre-allocate intermediate buffers for a fixed batch size.

        Must be called once before the first :meth:`forward` call with that
        batch size.  Calling again with the same size is a no-op.
        """
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
                if self._activation == "elu":
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out, 1.0], device=self._device)
                else:
                    wp.launch(self._act_kernel, dim=(M, N), inputs=[out, out], device=self._device)

            inp = out

        return self._intermediates[-1]

    def parameters(self) -> list[wp.array]:
        """Return flat list of all trainable parameter arrays."""
        params = []
        for w, b in zip(self.weights, self.biases, strict=True):
            params.append(w.flatten())
            params.append(b)
        return params

    def grad_arrays(self) -> list[wp.array]:
        """Return flat list of gradient arrays matching :meth:`parameters`."""
        grads = []
        for w, b in zip(self.weights, self.biases, strict=True):
            grads.append(w.grad.flatten())
            grads.append(b.grad)
        return grads


# ---------------------------------------------------------------------------
# Gaussian distribution kernels (fully on-device via wp.rand_init / wp.randn)
# ---------------------------------------------------------------------------

LOG2PI = wp.constant(float(math.log(2.0 * math.pi)))


@wp.kernel
def _increment_counter_kernel(counter: wp.array[int]):
    """Increment a single-element counter on device (graph-capture safe)."""
    counter[0] = counter[0] + 1


@wp.kernel
def _sample_actions_kernel(
    mean: wp.array2d[float],
    log_std: wp.array[float],
    rng_counter: wp.array[int],
    actions: wp.array2d[float],
    log_probs: wp.array[float],
):
    """Sample from N(mean, diag(exp(log_std)^2)) using on-device RNG."""
    i, j = wp.tid()
    act_dim = log_std.shape[0]
    step_offset = rng_counter[0]
    rng = wp.rand_init(42, step_offset * (mean.shape[0] * act_dim) + i * act_dim + j)
    noise = wp.randn(rng)
    std = wp.exp(log_std[j])
    a = mean[i, j] + std * noise
    actions[i, j] = a
    lp = -0.5 * (noise * noise + 2.0 * log_std[j] + LOG2PI)
    wp.atomic_add(log_probs, i, lp)


@wp.kernel
def _log_prob_kernel(
    mean: wp.array2d[float],
    log_std: wp.array[float],
    actions: wp.array2d[float],
    log_probs: wp.array[float],
):
    """Compute log-prob of given actions under N(mean, diag(exp(log_std)^2))."""
    i, j = wp.tid()
    std = wp.exp(log_std[j])
    diff = actions[i, j] - mean[i, j]
    z = diff / std
    lp = -0.5 * (z * z + 2.0 * log_std[j] + LOG2PI)
    wp.atomic_add(log_probs, i, lp)


@wp.kernel
def _entropy_kernel(
    log_std: wp.array[float],
    entropy: wp.array[float],
    act_dim: int,
):
    """Gaussian entropy per environment: sum_j (0.5 * log(2*pi*e) + log_std_j)."""
    i = wp.tid()
    ent = float(0.0)
    for j in range(act_dim):
        ent = ent + 0.5 * (LOG2PI + 1.0) + log_std[j]
    entropy[i] = ent


# ---------------------------------------------------------------------------
# Buffer copy / flatten / gather kernels (device-to-device, zero alloc)
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
def _gather_2d_offset(
    src: wp.array2d[float],
    indices: wp.array[int],
    offset: int,
    dst: wp.array2d[float],
):
    """Gather rows: ``dst[i, j] = src[indices[offset + i], j]``."""
    i, j = wp.tid()
    dst[i, j] = src[indices[offset + i], j]


@wp.kernel
def _gather_1d_offset(
    src: wp.array[float],
    indices: wp.array[int],
    offset: int,
    dst: wp.array[float],
):
    """Gather elements: ``dst[i] = src[indices[offset + i]]``."""
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
        n = max_length
        num_blocks = _ceil_div(n, ts)
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
    """``dst[i] = (src[i] - mean[0])^2``."""
    i = wp.tid()
    d = src[i] - mean_val[0]
    dst[i] = d * d


@wp.kernel
def _normalize_adv_kernel(
    adv: wp.array[float],
    mean_val: wp.array[float],
    inv_std: wp.array[float],
    out: wp.array[float],
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
    """Conditionally scale gradient element if total norm exceeds threshold."""
    i = wp.tid()
    nsq = norm_sq[0]
    if nsq > max_norm_sq:
        scale = max_norm / (wp.sqrt(nsq) + 1.0e-6)
        g[i] = g[i] * scale


# ---------------------------------------------------------------------------
# PPO loss kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _ppo_loss_kernel(
    old_log_probs: wp.array[float],
    new_log_probs: wp.array[float],
    advantages: wp.array[float],
    new_values: wp.array[float],
    returns: wp.array[float],
    entropy: wp.array[float],
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    loss: wp.array[float],
    n: int,
):
    i = wp.tid()
    ratio = wp.exp(new_log_probs[i] - old_log_probs[i])
    adv = advantages[i]
    surr1 = ratio * adv
    surr2 = wp.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -wp.min(surr1, surr2)
    vdiff = new_values[i] - returns[i]
    value_loss = 0.5 * vdiff * vdiff
    ent_loss = -entropy[i]
    total = (policy_loss + value_coef * value_loss + entropy_coef * ent_loss) / float(n)
    wp.atomic_add(loss, 0, total)


# ---------------------------------------------------------------------------
# ActorCritic
# ---------------------------------------------------------------------------


class ActorCritic:
    """Gaussian actor-critic with separate MLP networks.

    Args:
        obs_dim: Observation vector dimension.
        act_dim: Action vector dimension.
        hidden_sizes: Hidden layer widths.
        activation: Activation function name.
        init_log_std: Initial value for the learnable log standard deviation.
        device: Warp device string.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "elu",
        init_log_std: float = 0.0,
        device: str | None = None,
        seed: int | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]
        self._device = wp.get_device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        actor_seed = None if seed is None else seed
        critic_seed = None if seed is None else seed + 1000
        self.actor = WarpMLP(
            [obs_dim, *hidden_sizes, act_dim], activation=activation, device=device, output_gain=0.01, seed=actor_seed
        )
        self.critic = WarpMLP(
            [obs_dim, *hidden_sizes, 1], activation=activation, device=device, output_gain=1.0, seed=critic_seed
        )
        self.log_std = wp.array(
            np.full(act_dim, init_log_std, dtype=np.float32),
            dtype=wp.float32,
            device=self._device,
            requires_grad=True,
        )

    def alloc_buffers(self, rollout_batch: int, minibatch_size: int) -> None:
        """Pre-allocate all internal buffers for known batch sizes.

        Must be called once before training.  After this, :meth:`act` and
        :meth:`evaluate` perform zero allocations.

        Args:
            rollout_batch: ``num_envs`` (batch size during rollout collection).
            minibatch_size: Mini-batch size during PPO update.
        """
        d = self._device
        self.actor.alloc_intermediates(rollout_batch)
        self.critic.alloc_intermediates(rollout_batch)

        self._rb = rollout_batch
        self._mb = minibatch_size

        self._act_actions = wp.zeros((rollout_batch, self.act_dim), dtype=wp.float32, device=d)
        self._act_log_probs = wp.zeros(rollout_batch, dtype=wp.float32, device=d)

        self._eval_log_probs = wp.zeros(minibatch_size, dtype=wp.float32, device=d, requires_grad=True)
        self._eval_entropy = wp.zeros(minibatch_size, dtype=wp.float32, device=d, requires_grad=True)

        self._actor_mb = WarpMLP.__new__(WarpMLP)
        self._actor_mb.__dict__.update(self.actor.__dict__)
        self._actor_mb._intermediates = []
        self._actor_mb.alloc_intermediates(minibatch_size)
        self._actor_mb.weights = self.actor.weights
        self._actor_mb.biases = self.actor.biases

        self._critic_mb = WarpMLP.__new__(WarpMLP)
        self._critic_mb.__dict__.update(self.critic.__dict__)
        self._critic_mb._intermediates = []
        self._critic_mb.alloc_intermediates(minibatch_size)
        self._critic_mb.weights = self.critic.weights
        self._critic_mb.biases = self.critic.biases

    def act(
        self,
        obs: wp.array,
        rng_counter: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Forward actor + critic, sample actions using on-device RNG.

        Uses pre-allocated buffers from :meth:`alloc_buffers`.

        Args:
            obs: Observations, shape ``(batch, obs_dim)``.
            rng_counter: Single-element ``wp.array[int]`` counter on device.
                Incremented after each call to produce fresh random samples.

        Returns:
            ``(actions, log_probs, values)`` -- all on device, pre-allocated.
        """
        batch = obs.shape[0]
        mean = self.actor.forward(obs)
        values_2d = self.critic.forward(obs)

        self._act_actions.zero_()
        self._act_log_probs.zero_()

        wp.launch(
            _sample_actions_kernel,
            dim=(batch, self.act_dim),
            inputs=[mean, self.log_std, rng_counter, self._act_actions, self._act_log_probs],
            device=self._device,
        )
        wp.launch(_increment_counter_kernel, dim=1, inputs=[rng_counter], device=self._device)

        values = values_2d.flatten()
        return self._act_actions, self._act_log_probs, values

    def evaluate(
        self,
        obs: wp.array,
        actions: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Recompute log_probs, entropy, and values for stored transitions.

        Uses pre-allocated buffers from :meth:`alloc_buffers`.

        Returns:
            ``(log_probs, entropy, values)``
        """
        batch = obs.shape[0]
        mean = self._actor_mb.forward(obs)
        values_2d = self._critic_mb.forward(obs)

        self._eval_log_probs.zero_()
        self._eval_entropy.zero_()

        wp.launch(
            _log_prob_kernel,
            dim=(batch, self.act_dim),
            inputs=[mean, self.log_std, actions, self._eval_log_probs],
            device=self._device,
        )

        wp.launch(
            _entropy_kernel,
            dim=batch,
            inputs=[self.log_std, self._eval_entropy, self.act_dim],
            device=self._device,
        )

        values = values_2d.flatten()
        return self._eval_log_probs, self._eval_entropy, values

    def parameters(self) -> list[wp.array]:
        return self.actor.parameters() + self.critic.parameters() + [self.log_std]

    def grad_arrays(self) -> list[wp.array]:
        return self.actor.grad_arrays() + self.critic.grad_arrays() + [self.log_std.grad]


# ---------------------------------------------------------------------------
# Rollout buffer with GAE (fully on-device, zero alloc)
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


class RolloutBuffer:
    """Fixed-size on-device rollout storage with GAE computation.

    All data stays on the Warp device.  Insert uses device-to-device copy
    kernels.  Zero allocations after construction.

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
        self.log_probs = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.rewards = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.dones = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.values = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.advantages = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)
        self.returns = wp.zeros((num_steps, num_envs), dtype=wp.float32, device=d)

        n = num_steps * num_envs
        self.flat_obs = wp.zeros((n, obs_dim), dtype=wp.float32, device=d)
        self.flat_actions = wp.zeros((n, act_dim), dtype=wp.float32, device=d)
        self.flat_log_probs = wp.zeros(n, dtype=wp.float32, device=d)
        self.flat_values = wp.zeros(n, dtype=wp.float32, device=d)
        self.flat_advantages = wp.zeros(n, dtype=wp.float32, device=d)
        self.flat_returns = wp.zeros(n, dtype=wp.float32, device=d)

    def insert(
        self,
        t: int,
        obs: wp.array,
        actions: wp.array,
        log_probs: wp.array,
        rewards: wp.array,
        dones: wp.array,
        values: wp.array,
    ) -> None:
        d = self._device
        wp.launch(_copy_2d_to_3d_slice, dim=(self.num_envs, self.obs_dim), inputs=[obs, self.observations, t], device=d)
        wp.launch(_copy_2d_to_3d_slice, dim=(self.num_envs, self.act_dim), inputs=[actions, self.actions, t], device=d)
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
            _flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.log_probs, self.flat_log_probs, ne], device=d
        )
        wp.launch(_flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.values, self.flat_values, ne], device=d)
        wp.launch(
            _flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.advantages, self.flat_advantages, ne], device=d
        )
        wp.launch(_flatten_2d_to_1d, dim=(self.num_steps, ne), inputs=[self.returns, self.flat_returns, ne], device=d)


# ---------------------------------------------------------------------------
# PPOTrainer (fully on-device, zero-allocation hot path)
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Proximal Policy Optimization trainer for continuous action spaces.

    Every buffer is pre-allocated in :meth:`__init__` / :meth:`_alloc`.
    The hot path (rollout collection, GAE, advantage normalization,
    mini-batch shuffling, PPO update, gradient clipping) performs **zero
    device memory allocations**, making it compatible with
    ``wp.ScopedCapture`` / CUDA graph capture.

    The only host round-trip is a single ``loss.numpy()`` readback for
    logging (which can be removed if logging is disabled).

    Args:
        actor_critic: The :class:`ActorCritic` to train.
        num_envs: Number of parallel environments.
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_ratio: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        num_epochs: PPO epochs per rollout.
        num_minibatches: Number of mini-batches per epoch.
        weight_decay: AdamW weight decay.
        num_steps: Rollout horizon per update.
        seed: RNG seed for on-device random number generation.
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
    ):
        self.ac = actor_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm_sq = max_grad_norm * max_grad_norm
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.num_steps = num_steps

        device = actor_critic._device

        # On-device RNG counter (graph-capture safe)
        self._rng_counter = wp.array([seed], dtype=wp.int32, device=device)
        n_samples = num_steps * num_envs
        batch_size = n_samples // num_minibatches

        # -- Rollout buffer --
        self.buffer = RolloutBuffer(
            num_envs=num_envs,
            num_steps=num_steps,
            obs_dim=actor_critic.obs_dim,
            act_dim=actor_critic.act_dim,
            device=str(device),
        )

        # -- ActorCritic internal buffers --
        actor_critic.alloc_buffers(num_envs, batch_size)

        # -- Optimizer --
        self.optimizer = AdamW(actor_critic.parameters(), lr=lr, weight_decay=weight_decay)

        # -- Advantage normalization scratch --
        self._norm_adv = wp.zeros(n_samples, dtype=wp.float32, device=device)
        self._sq_diff = wp.zeros(n_samples, dtype=wp.float32, device=device)
        self._sum_result = wp.zeros(1, dtype=wp.float32, device=device)
        self._mean_buf = wp.zeros(1, dtype=wp.float32, device=device)
        self._var_result = wp.zeros(1, dtype=wp.float32, device=device)
        self._inv_std_buf = wp.zeros(1, dtype=wp.float32, device=device)
        self._sum_reducer = _ArraySum(n_samples, device=str(device))

        # -- Shuffle arrays (2x capacity required by radix_sort_pairs) --
        self._shuffle_keys = wp.zeros(2 * n_samples, dtype=wp.float32, device=device)
        self._indices = wp.zeros(2 * n_samples, dtype=wp.int32, device=device)

        # -- Mini-batch gather buffers --
        self._mb_obs = wp.zeros((batch_size, actor_critic.obs_dim), dtype=wp.float32, device=device, requires_grad=True)
        self._mb_act = wp.zeros((batch_size, actor_critic.act_dim), dtype=wp.float32, device=device)
        self._mb_old_lp = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._mb_adv = wp.zeros(batch_size, dtype=wp.float32, device=device)
        self._mb_ret = wp.zeros(batch_size, dtype=wp.float32, device=device)

        # -- Loss and gradient scratch --
        self._loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        self._grad_norm_sq = wp.zeros(1, dtype=wp.float32, device=device)

        self._n_samples = n_samples
        self._batch_size = batch_size

    def collect_rollouts(self, env: Any, obs: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Run the policy in *env* for ``num_steps`` and fill the buffer.

        Args:
            env: Vectorized environment.
            obs: Current observations.  When ``None`` the environment is
                reset to obtain initial observations.

        Returns:
            ``(last_values, obs)`` -- critic bootstrap values and the
            current observation tensor (pass back on the next call to
            avoid resetting the environment).
        """
        if obs is None:
            obs = env.reset()
        buf = self.buffer

        for t in range(buf.num_steps):
            actions, log_probs, values = self.ac.act(obs, self._rng_counter)

            next_obs, rewards, dones = env.step(actions)
            buf.insert(t, obs, actions, log_probs, rewards, dones, values)
            obs = next_obs

        # Bootstrap value for last obs
        _, _, last_values = self.ac.act(obs, self._rng_counter)
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
        self.buffer.flatten()
        self._normalize_advantages()

        n_samples = self._n_samples
        batch_size = self._batch_size
        device = self.ac._device

        total_loss = 0.0
        num_updates = 0

        for _epoch in range(self.num_epochs):
            # GPU shuffle: generate random keys, then radix sort indices by key
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

                # Gather mini-batch on device using offset into shuffled indices
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

                self._loss.zero_()

                tape = wp.Tape()
                with tape:
                    new_lp, ent, vals = self.ac.evaluate(self._mb_obs, self._mb_act)
                    wp.launch(
                        _ppo_loss_kernel,
                        dim=batch_size,
                        inputs=[
                            self._mb_old_lp,
                            new_lp,
                            self._mb_adv,
                            vals,
                            self._mb_ret,
                            ent,
                            self.clip_ratio,
                            self.value_coef,
                            self.entropy_coef,
                            self._loss,
                            batch_size,
                        ],
                        device=device,
                    )

                tape.backward(self._loss)

                grads = self.ac.grad_arrays()
                self._clip_grad_norm(grads)
                self.optimizer.step(grads)
                tape.zero()

                total_loss += self._loss.numpy()[0]
                num_updates += 1

        return total_loss / max(num_updates, 1)

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

    def train(self, env: Any, total_timesteps: int, log_interval: int = 1) -> None:
        """Main training loop.

        Args:
            env: Vectorized environment with ``num_envs``, ``obs_dim``, ``act_dim``
                that returns ``wp.array`` tensors.
            total_timesteps: Total environment steps to collect.
            log_interval: Print stats every N updates.
        """
        steps_per_update = self.buffer.num_envs * self.buffer.num_steps
        num_updates = total_timesteps // steps_per_update
        total_steps = 0

        obs = None
        for update_idx in range(num_updates):
            last_values, obs = self.collect_rollouts(env, obs)
            self.buffer.compute_gae(last_values, self.gamma, self.gae_lambda)
            avg_loss = self.update()
            total_steps += steps_per_update

            if (update_idx + 1) % log_interval == 0:
                print(f"Update {update_idx + 1}/{num_updates} | steps={total_steps} | loss={avg_loss:.4f}")


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

    for i, (w, b) in enumerate(zip(actor.weights, actor.biases, strict=True)):
        w_name = f"actor.{i * 2}.weight"
        b_name = f"actor.{i * 2}.bias"
        gemm_out = f"/actor/{i * 2}/Gemm_output_0"

        w_np = w.numpy()
        b_np = b.numpy()
        initializers.append(numpy_helper.from_array(w_np, name=w_name))
        initializers.append(numpy_helper.from_array(b_np, name=b_name))

        nodes.append(helper.make_node("Gemm", [prev_output, w_name, b_name], [gemm_out], alpha=1.0, beta=1.0, transB=1))

        is_last = i == len(actor.weights) - 1
        if not is_last:
            act_out = f"/actor/{i * 2 + 1}/{onnx_act}_output_0"
            nodes.append(helper.make_node(onnx_act, [gemm_out], [act_out]))
            prev_output = act_out
        else:
            prev_output = gemm_out

    last_node = nodes[-1]
    last_node.output[0] = "action"

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
    """Minimal vectorized environment interface for PPO.

    All arrays must be ``wp.array`` on the training device.
    """

    num_envs: int
    obs_dim: int
    act_dim: int

    def reset(self) -> wp.array:
        """Return initial observations ``(num_envs, obs_dim)``."""
        ...

    def step(self, actions: wp.array) -> tuple[wp.array, wp.array, wp.array]:
        """Step the environment.

        Returns:
            ``(obs, rewards, dones)`` -- all ``wp.array`` on device.
        """
        ...
