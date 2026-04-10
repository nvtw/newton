# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neural network for PufferLib on Warp.

Provides :class:`SimpleMLP` — a three-layer MLP (encoder → ReLU → hidden →
ReLU → decoder) with no biases, matching C++ PufferLib's default policy
architecture.  Forward and backward passes are hand-written kernels with
``enable_backward=False`` for fast compile times.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.pufferlib.kernels import matmul

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# Forward / backward kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _relu_inplace_kernel(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    y[i, j] = wp.max(x[i, j], 0.0)


@wp.kernel
def relu_backward_mask_kernel(
    y: wp.array2d(dtype=float),
    grad_y: wp.array2d(dtype=float),
    masked: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if y[i, j] > 0.0:
        masked[i, j] = grad_y[i, j]
    else:
        masked[i, j] = 0.0


@wp.kernel
def _elu_kernel(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    v = x[i, j]
    if v > 0.0:
        y[i, j] = v
    else:
        y[i, j] = wp.exp(v) - 1.0


@wp.kernel
def _elu_backward_kernel(
    y: wp.array2d(dtype=float),
    grad_y: wp.array2d(dtype=float),
    masked: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    v = y[i, j]
    if v > 0.0:
        masked[i, j] = grad_y[i, j]
    else:
        masked[i, j] = grad_y[i, j] * (v + 1.0)


@wp.kernel
def _bias_add_kernel(x: wp.array2d(dtype=float), bias: wp.array(dtype=float, ndim=1), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    y[i, j] = x[i, j] + bias[j]


@wp.kernel
def _bias_grad_kernel(
    grad_out: wp.array2d(dtype=float),
    grad_bias: wp.array(dtype=float, ndim=1),
    B: int,
):
    """grad_bias[j] = sum_i grad_out[i, j]."""
    j = wp.tid()
    s = float(0.0)
    for i in range(B):
        s = s + grad_out[i, j]
    grad_bias[j] = s


@wp.kernel
def kaiming_init_kernel(w: wp.array(dtype=float, ndim=1), fan_in: int, seed: int, gain: float):
    """Matches C++ PufferLib puf_kaiming_init: Uniform(-gain/sqrt(fan_in), gain/sqrt(fan_in))."""
    i = wp.tid()
    state = wp.rand_init(seed, i)
    bound = gain / wp.sqrt(float(fan_in))
    w[i] = wp.randf(state, -bound, bound)


# ---------------------------------------------------------------------------
# SimpleMLP
# ---------------------------------------------------------------------------

SQRT2 = 1.4142135623730951


class SimpleMLP:
    """Encoder -> ReLU -> Hidden -> ReLU -> Decoder (no biases, matching C++ PufferLib).

    The decoder output has ``num_actions + 1`` columns: the first ``num_actions``
    are logits (or means for continuous) and the last column is the value head,
    matching C++ PufferLib's fused decoder layout.

    For continuous action spaces, set ``continuous=True`` and ``num_actions`` to
    the number of action dimensions.  This adds a trainable ``logstd`` parameter
    of shape ``(num_actions,)`` initialized to zeros, matching C++ PufferLib's
    ``DecoderWeights::logstd``.

    Args:
        obs_dim: Observation dimensionality.
        hidden: Hidden layer width.
        out_dim: Decoder output width (typically ``num_actions + 1``).
        max_batch: Maximum batch size (pre-allocates scratch buffers).
        device: Warp device string.
        seed: RNG seed for Kaiming initialization.
        continuous: Whether the action space is continuous.
        num_actions: Number of action dimensions (only used when ``continuous=True``).
    """

    def __init__(self, obs_dim: int, hidden: int, out_dim: int,
                 max_batch: int, device: str, seed: int = 42,
                 continuous: bool = False, num_actions: int = 0,
                 init_logstd: float = 0.0, activation: str = "relu",
                 num_hidden_layers: int = 2, use_bias: bool = False):
        self.device = device
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.continuous = continuous
        self._activation = activation
        self._num_hidden = num_hidden_layers
        self._use_bias = use_bias

        # Build weight list: [obs->H, H->H (×num_hidden-1), H->out]
        layer_dims = [(hidden, obs_dim)] + [(hidden, hidden)] * (num_hidden_layers - 1) + [(out_dim, hidden)]
        self._weights = []
        self._biases = []
        for i, (rows, cols) in enumerate(layer_dims):
            w = wp.zeros((rows, cols), dtype=float, device=device, requires_grad=True)
            gain = SQRT2 if i == 0 else 1.0
            wp.launch(kaiming_init_kernel, dim=w.size,
                      inputs=[w.flatten(), cols, seed + i, gain], device=device)
            self._weights.append(w)
            if use_bias:
                self._biases.append(wp.zeros(rows, dtype=float, device=device, requires_grad=True))
            else:
                self._biases.append(None)

        # Legacy aliases for backward compat with 2-layer code (w1, w2, w3)
        self.w1 = self._weights[0]
        self.w2 = self._weights[1]
        self.w3 = self._weights[-1]

        # Scalar std parameterization (RSL-RL default: std_type="scalar").
        # Unlike log_std, the entropy gradient w.r.t. std is 1/std which
        # naturally decreases as std grows — preventing entropy explosion.
        self.logstd = None  # kept for backward compat (will be None)
        self.std = None
        if continuous:
            init_std = float(math.exp(init_logstd)) if init_logstd != 0.0 else 1.0
            self.std = wp.full(num_actions, value=init_std, dtype=float, device=device, requires_grad=True)

        # Scratch buffers for forward/backward
        num_layers = len(self._weights)
        self._pre_acts = [wp.zeros((max_batch, d[0]), dtype=float, device=device) for d in layer_dims]
        self._acts = [wp.zeros((max_batch, d[0]), dtype=float, device=device) for d in layer_dims[:-1]]  # no activation on output
        self._out = wp.zeros((max_batch, out_dim), dtype=float, device=device)
        self._grad_weights = [wp.zeros_like(w) for w in self._weights]
        self._grad_biases = [wp.zeros(d[0], dtype=float, device=device) if self._biases[i] is not None else None
                             for i, d in enumerate(layer_dims)]
        self._grad_acts = [wp.zeros((max_batch, d[0]), dtype=float, device=device) for d in layer_dims[:-1]]
        self._grad_acts_masked = [wp.zeros((max_batch, d[0]), dtype=float, device=device) for d in layer_dims[:-1]]
        self._last_x = None
        self._last_B = 0

    def forward(self, x: wp.array, B: int) -> wp.array:
        self._last_x = x
        self._last_B = B
        d = self.device
        act_fwd = _elu_kernel if self._activation == "elu" else _relu_inplace_kernel
        num_layers = len(self._weights)

        prev = x
        for i in range(num_layers):
            pre = self._pre_acts[i][:B]
            matmul(prev, self._weights[i], pre, transpose_b=True)
            if self._biases[i] is not None:
                wp.launch(_bias_add_kernel, dim=(B, pre.shape[1]),
                          inputs=[pre, self._biases[i], pre], device=d)
            if i < num_layers - 1:
                act = self._acts[i][:B]
                wp.launch(act_fwd, dim=(B, self.hidden), inputs=[pre, act], device=d)
                prev = act
            else:
                # Last layer: no activation, write to _out
                wp.copy(self._out, pre, count=B * self.out_dim)
                prev = self._out[:B]
        return self._out

    def backward(self, grad_out: wp.array, B: int) -> list[wp.array]:
        """Compute gradients for all parameters.

        Returns gradients in the same order as :meth:`parameters`:
        [grad_w0, grad_w1, ..., grad_b0, grad_b1, ..., (grad_logstd excluded)].
        """
        d = self.device
        act_bwd = _elu_backward_kernel if self._activation == "elu" else relu_backward_mask_kernel
        num_layers = len(self._weights)

        grad = grad_out
        weight_grads = [None] * num_layers
        bias_grads = [None] * num_layers
        for i in range(num_layers - 1, -1, -1):
            inp = self._acts[i - 1][:B] if i > 0 else self._last_x

            # grad_w = grad^T @ input
            matmul(grad, inp, self._grad_weights[i], transpose_a=True)
            weight_grads[i] = self._grad_weights[i]

            # grad_bias = sum(grad, dim=0)
            if self._grad_biases[i] is not None:
                wp.launch(_bias_grad_kernel, dim=self._grad_biases[i].shape[0],
                          inputs=[grad, self._grad_biases[i], B], device=d)
                bias_grads[i] = self._grad_biases[i]

            if i > 0:
                grad_h = self._grad_acts[i - 1][:B]
                matmul(grad, self._weights[i], grad_h)
                act_out = self._acts[i - 1][:B]
                grad_masked = self._grad_acts_masked[i - 1][:B]
                wp.launch(act_bwd, dim=(B, self.hidden), inputs=[act_out, grad_h, grad_masked], device=d)
                grad = grad_masked

        # Return in same order as parameters(): weights, then biases
        result = list(weight_grads)
        for bg in bias_grads:
            if bg is not None:
                result.append(bg)
        return result

    def parameters(self):
        params = list(self._weights)
        for b in self._biases:
            if b is not None:
                params.append(b)
        if self.std is not None:
            params.append(self.std)
        return params
