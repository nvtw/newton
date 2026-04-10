# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neural network for PufferLib on Warp.

Provides :class:`SimpleMLP` — a three-layer MLP (encoder → ReLU → hidden →
ReLU → decoder) with no biases, matching C++ PufferLib's default policy
architecture.  Forward and backward passes are hand-written kernels with
``enable_backward=False`` for fast compile times.
"""

from __future__ import annotations

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
                 init_logstd: float = 0.0, activation: str = "relu"):
        self.device = device
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.continuous = continuous
        self._activation = activation

        self.w1 = wp.zeros((hidden, obs_dim), dtype=float, device=device, requires_grad=True)
        self.w2 = wp.zeros((hidden, hidden), dtype=float, device=device, requires_grad=True)
        self.w3 = wp.zeros((out_dim, hidden), dtype=float, device=device, requires_grad=True)

        self.logstd = None
        if continuous:
            self.logstd = wp.full(num_actions, value=init_logstd, dtype=float, device=device, requires_grad=True)

        wp.launch(kaiming_init_kernel, dim=self.w1.size,
                  inputs=[self.w1.flatten(), obs_dim, seed, SQRT2], device=device)
        wp.launch(kaiming_init_kernel, dim=self.w2.size,
                  inputs=[self.w2.flatten(), hidden, seed + 1, 1.0], device=device)
        wp.launch(kaiming_init_kernel, dim=self.w3.size,
                  inputs=[self.w3.flatten(), hidden, seed + 2, 1.0], device=device)

        self._pre_h1 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._h1 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._pre_h2 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._h2 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._out = wp.zeros((max_batch, out_dim), dtype=float, device=device)
        self._grad_w3 = wp.zeros_like(self.w3)
        self._grad_h2 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._grad_h2_masked = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._grad_w2 = wp.zeros_like(self.w2)
        self._grad_h1 = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._grad_h1_masked = wp.zeros((max_batch, hidden), dtype=float, device=device)
        self._grad_w1 = wp.zeros_like(self.w1)
        self._last_x = None
        self._last_B = 0

    def forward(self, x: wp.array, B: int) -> wp.array:
        self._last_x = x
        self._last_B = B
        d = self.device
        pre_h1 = self._pre_h1[:B]
        h1 = self._h1[:B]
        pre_h2 = self._pre_h2[:B]
        h2 = self._h2[:B]
        out = self._out[:B]
        act_fwd = _elu_kernel if self._activation == "elu" else _relu_inplace_kernel
        # h1 = act(x @ w1^T)
        matmul(x, self.w1, pre_h1, transpose_b=True)
        wp.launch(act_fwd, dim=(B, self.hidden), inputs=[pre_h1, h1], device=d)
        # h2 = act(h1 @ w2^T)
        matmul(h1, self.w2, pre_h2, transpose_b=True)
        wp.launch(act_fwd, dim=(B, self.hidden), inputs=[pre_h2, h2], device=d)
        # out = h2 @ w3^T
        matmul(h2, self.w3, out, transpose_b=True)
        return self._out

    def backward(self, grad_out: wp.array, B: int) -> list[wp.array]:
        d = self.device
        h2 = self._h2[:B]
        h1 = self._h1[:B]
        grad_h2 = self._grad_h2[:B]
        grad_h2_m = self._grad_h2_masked[:B]
        grad_h1 = self._grad_h1[:B]
        grad_h1_m = self._grad_h1_masked[:B]
        act_bwd = _elu_backward_kernel if self._activation == "elu" else relu_backward_mask_kernel
        # grad_w3 = grad_out^T @ h2
        matmul(grad_out, h2, self._grad_w3, transpose_a=True)
        # grad_h2 = grad_out @ w3
        matmul(grad_out, self.w3, grad_h2)
        wp.launch(act_bwd, dim=(B, self.hidden), inputs=[h2, grad_h2, grad_h2_m], device=d)
        # grad_w2 = grad_h2_masked^T @ h1
        matmul(grad_h2_m, h1, self._grad_w2, transpose_a=True)
        # grad_h1 = grad_h2_masked @ w2
        matmul(grad_h2_m, self.w2, grad_h1)
        wp.launch(act_bwd, dim=(B, self.hidden), inputs=[h1, grad_h1, grad_h1_m], device=d)
        # grad_w1 = grad_h1_masked^T @ x
        matmul(grad_h1_m, self._last_x, self._grad_w1, transpose_a=True)
        return [self._grad_w1, self._grad_w2, self._grad_w3]

    def parameters(self):
        params = [self.w1, self.w2, self.w3]
        if self.logstd is not None:
            params.append(self.logstd)
        return params
