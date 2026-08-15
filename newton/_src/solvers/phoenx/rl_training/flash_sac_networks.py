# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure-Warp reference backbone layers for FlashSAC."""

from __future__ import annotations

import numpy as np
import warp as wp

from .cublas import gemm_float32, is_cublas_available
from .kernels import copy_1d_kernel, copy_2d_kernel, soft_update_1d_kernel, soft_update_2d_kernel


@wp.kernel
def _unit_linear_kernel(
    x: wp.array2d[wp.float32],
    weight: wp.array2d[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    value = wp.float32(0.0)
    for inner in range(weight.shape[0]):
        value += x[row, inner] * weight[inner, col]
    out[row, col] = value


@wp.kernel
def _linear_input_grad_kernel(
    output_grad: wp.array2d[wp.float32],
    weight: wp.array2d[wp.float32],
    input_grad: wp.array2d[wp.float32],
):
    row, inner = wp.tid()
    value = wp.float32(0.0)
    for col in range(weight.shape[1]):
        value += output_grad[row, col] * weight[inner, col]
    input_grad[row, inner] = value


@wp.kernel
def _linear_weight_grad_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    weight_grad: wp.array2d[wp.float32],
):
    inner, col = wp.tid()
    value = wp.float32(0.0)
    for row in range(x.shape[0]):
        value += x[row, inner] * output_grad[row, col]
    weight_grad[inner, col] = value


@wp.kernel
def _batch_norm_backward_stats_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    eps: wp.float32,
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    sum_grad: wp.array[wp.float32],
    sum_grad_normalized: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    col = wp.tid()
    batch_mean = wp.float32(0.0)
    for row in range(x.shape[0]):
        batch_mean += x[row, col]
    batch_mean /= wp.float32(x.shape[0])
    variance = wp.float32(0.0)
    for row in range(x.shape[0]):
        delta = x[row, col] - batch_mean
        variance += delta * delta
    variance /= wp.float32(x.shape[0])
    batch_inv_std = wp.float32(1.0) / wp.sqrt(variance + eps)
    grad_sum = wp.float32(0.0)
    grad_normalized_sum = wp.float32(0.0)
    for row in range(x.shape[0]):
        normalized = (x[row, col] - batch_mean) * batch_inv_std
        grad_sum += output_grad[row, col]
        grad_normalized_sum += output_grad[row, col] * normalized
    mean[col] = batch_mean
    inv_std[col] = batch_inv_std
    sum_grad[col] = grad_sum
    sum_grad_normalized[col] = grad_normalized_sum
    scale_grad[col] = grad_normalized_sum
    bias_grad[col] = grad_sum


@wp.kernel
def _batch_norm_input_grad_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    sum_grad: wp.array[wp.float32],
    sum_grad_normalized: wp.array[wp.float32],
    input_grad: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    count = wp.float32(x.shape[0])
    normalized = (x[row, col] - mean[col]) * inv_std[col]
    input_grad[row, col] = (
        scale[col]
        * inv_std[col]
        * (output_grad[row, col] - sum_grad[col] / count - normalized * sum_grad_normalized[col] / count)
    )


@wp.kernel
def _batch_norm_inference_input_grad_kernel(
    output_grad: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
    eps: wp.float32,
    input_grad: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    input_grad[row, col] = output_grad[row, col] * scale[col] / wp.sqrt(running_variance[col] + eps)


@wp.kernel
def _batch_norm_inference_parameter_grad_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    running_mean: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
    eps: wp.float32,
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    col = wp.tid()
    inv_std = wp.float32(1.0) / wp.sqrt(running_variance[col] + eps)
    grad_scale = wp.float32(0.0)
    grad_bias = wp.float32(0.0)
    for row in range(x.shape[0]):
        grad_scale += output_grad[row, col] * (x[row, col] - running_mean[col]) * inv_std
        grad_bias += output_grad[row, col]
    scale_grad[col] = grad_scale
    bias_grad[col] = grad_bias


@wp.kernel
def _relu_grad_kernel(
    activated: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    input_grad: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    input_grad[row, col] = wp.where(activated[row, col] > wp.float32(0.0), output_grad[row, col], wp.float32(0.0))


@wp.kernel
def _add_kernel(a: wp.array2d[wp.float32], b: wp.array2d[wp.float32], out: wp.array2d[wp.float32]):
    row, col = wp.tid()
    out[row, col] = a[row, col] + b[row, col]


@wp.kernel
def _rms_norm_backward_stats_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
):
    row = wp.tid()
    mean_square = wp.float32(0.0)
    projected = wp.float32(0.0)
    for col in range(x.shape[1]):
        mean_square += x[row, col] * x[row, col]
        projected += output_grad[row, col] * scale[col] * x[row, col]
    inv_rms[row] = wp.float32(1.0) / wp.sqrt(mean_square / wp.float32(x.shape[1]) + eps)
    projection[row] = projected


@wp.kernel
def _rms_norm_input_grad_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
    input_grad: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    input_grad[row, col] = output_grad[row, col] * scale[col] * inv_rms[row] - (
        x[row, col] * inv_rms[row] * inv_rms[row] * inv_rms[row] * projection[row] / wp.float32(x.shape[1])
    )


@wp.kernel
def _rms_norm_scale_grad_kernel(
    x: wp.array2d[wp.float32],
    output_grad: wp.array2d[wp.float32],
    inv_rms: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
):
    col = wp.tid()
    value = wp.float32(0.0)
    for row in range(x.shape[0]):
        value += output_grad[row, col] * x[row, col] * inv_rms[row]
    scale_grad[col] = value


@wp.kernel
def _head_output_grad_kernel(
    output_grad: wp.array2d[wp.float32],
    head: wp.array2d[wp.float32],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    smooth_log_std: wp.bool,
    minimum: wp.float32,
    maximum: wp.float32,
    head_grad: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    value = output_grad[row, col + offset]
    if smooth_log_std:
        mapped = wp.tanh(head[row, col] + bias[col])
        value *= (maximum - minimum) * wp.float32(0.5) * (wp.float32(1.0) - mapped * mapped)
    head_grad[row, col] = value


@wp.kernel
def _bias_grad_kernel(output_grad: wp.array2d[wp.float32], bias_grad: wp.array[wp.float32]):
    col = wp.tid()
    value = wp.float32(0.0)
    for row in range(output_grad.shape[0]):
        value += output_grad[row, col]
    bias_grad[col] = value


@wp.kernel(enable_backward=False)
def _batch_moments_kernel(
    x: wp.array2d[wp.float32],
    count: wp.int32,
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
):
    col = wp.tid()
    total = wp.float32(0.0)
    for row in range(count):
        total += x[row, col]
    batch_mean = total / wp.float32(count)
    squared = wp.float32(0.0)
    for row in range(count):
        delta = x[row, col] - batch_mean
        squared += delta * delta
    mean[col] = batch_mean
    variance[col] = squared / wp.float32(count)


@wp.kernel(enable_backward=False)
def _update_running_moments_kernel(
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    count: wp.int32,
    momentum: wp.float32,
    running_mean: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
):
    col = wp.tid()
    correction = wp.float32(1.0)
    if count > 1:
        correction = wp.float32(count) / wp.float32(count - 1)
    running_mean[col] = (wp.float32(1.0) - momentum) * running_mean[col] + momentum * mean[col]
    running_variance[col] = (wp.float32(1.0) - momentum) * running_variance[col] + momentum * variance[col] * correction


@wp.kernel
def _batch_norm_kernel(
    x: wp.array2d[wp.float32],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    normalized = (x[row, col] - mean[col]) / wp.sqrt(variance[col] + eps)
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _batch_norm_training_kernel(
    x: wp.array2d[wp.float32],
    count: wp.int32,
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    total = wp.float32(0.0)
    for sample in range(count):
        total += x[sample, col]
    mean = total / wp.float32(count)
    squared = wp.float32(0.0)
    for sample in range(count):
        delta = x[sample, col] - mean
        squared += delta * delta
    variance = squared / wp.float32(count)
    normalized = (x[row, col] - mean) / wp.sqrt(variance + eps)
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _relu_kernel(x: wp.array2d[wp.float32], out: wp.array2d[wp.float32]):
    row, col = wp.tid()
    out[row, col] = wp.max(x[row, col], wp.float32(0.0))


@wp.kernel
def _residual_add_kernel(
    x: wp.array2d[wp.float32],
    residual: wp.array2d[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    out[row, col] = x[row, col] + residual[row, col]


@wp.kernel
def _rms_inv_kernel(
    x: wp.array2d[wp.float32],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
):
    row = wp.tid()
    squared = wp.float32(0.0)
    for col in range(x.shape[1]):
        squared += x[row, col] * x[row, col]
    inv_rms[row] = wp.float32(1.0) / wp.sqrt(squared / wp.float32(x.shape[1]) + eps)


@wp.kernel
def _rms_norm_kernel(
    x: wp.array2d[wp.float32],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    out[row, col] = x[row, col] * inv_rms[row] * scale[col]


@wp.kernel
def _head_bias_kernel(
    x: wp.array2d[wp.float32],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    out[row, col + offset] = x[row, col] + bias[col]


@wp.kernel
def _head_bias_log_std_kernel(
    x: wp.array2d[wp.float32],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    minimum: wp.float32,
    maximum: wp.float32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    raw = x[row, col] + bias[col]
    out[row, col + offset] = minimum + (maximum - minimum) * wp.float32(0.5) * (wp.float32(1.0) + wp.tanh(raw))


@wp.kernel
def _normalize_columns_kernel(weight: wp.array2d[wp.float32], eps: wp.float32):
    col = wp.tid()
    squared = wp.float32(0.0)
    for row in range(weight.shape[0]):
        squared += weight[row, col] * weight[row, col]
    inv_norm = wp.float32(1.0) / wp.sqrt(squared + eps)
    for row in range(weight.shape[0]):
        weight[row, col] = weight[row, col] * inv_norm


@wp.kernel
def _normalize_batch_affine_kernel(
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
):
    index = wp.tid()
    if index == 0:
        squared = wp.float32(0.0)
        for col in range(scale.shape[0]):
            squared += scale[col] * scale[col] + bias[col] * bias[col]
        factor = wp.sqrt(wp.float32(scale.shape[0])) / wp.sqrt(squared + eps)
        for col in range(scale.shape[0]):
            scale[col] = scale[col] * factor
            bias[col] = bias[col] * factor


@wp.kernel
def _normalize_scale_kernel(scale: wp.array[wp.float32], eps: wp.float32):
    index = wp.tid()
    if index == 0:
        squared = wp.float32(0.0)
        for col in range(scale.shape[0]):
            squared += scale[col] * scale[col]
        factor = wp.sqrt(wp.float32(scale.shape[0])) / wp.sqrt(squared + eps)
        for col in range(scale.shape[0]):
            scale[col] = scale[col] * factor


def _orthogonal(input_dim: int, output_dim: int, rng: np.random.Generator) -> np.ndarray:
    shape = (int(output_dim), int(input_dim))
    flat = rng.normal(size=shape).astype(np.float32)
    if shape[0] < shape[1]:
        q, _r = np.linalg.qr(flat.T)
        result = q.T
    else:
        q, _r = np.linalg.qr(flat)
        result = q
    return np.asarray(result, dtype=np.float32).reshape(shape).T.copy()


class _UnitBatchNorm:
    def __init__(self, width: int, device: wp.context.Device, *, momentum: float = 0.01, eps: float = 1.0e-5):
        self.width = int(width)
        self.device = device
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.scale = wp.ones(self.width, dtype=wp.float32, device=device, requires_grad=True)
        self.bias = wp.zeros(self.width, dtype=wp.float32, device=device, requires_grad=True)
        self.running_mean = wp.zeros(self.width, dtype=wp.float32, device=device)
        self.running_variance = wp.ones(self.width, dtype=wp.float32, device=device)

    def forward(self, x: wp.array2d[wp.float32], *, training: bool, requires_grad: bool) -> wp.array2d[wp.float32]:
        rows = int(x.shape[0])
        if training:
            mean = wp.empty(self.width, dtype=wp.float32, device=self.device)
            variance = wp.empty(self.width, dtype=wp.float32, device=self.device)
            wp.launch(
                _batch_moments_kernel,
                dim=self.width,
                inputs=[x, rows],
                outputs=[mean, variance],
                device=self.device,
            )
            wp.launch(
                _update_running_moments_kernel,
                dim=self.width,
                inputs=[mean, variance, rows, self.momentum],
                outputs=[self.running_mean, self.running_variance],
                device=self.device,
            )
        else:
            mean = self.running_mean
            variance = self.running_variance
        out = wp.empty(x.shape, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        wp.launch(
            _batch_norm_kernel,
            dim=x.shape,
            inputs=[x, mean, variance, self.scale, self.bias, self.eps],
            outputs=[out],
            device=self.device,
        )
        return out

    def parameters(self) -> list[wp.array]:
        return [self.scale, self.bias]

    def state_arrays(self) -> list[wp.array]:
        return [self.scale, self.bias, self.running_mean, self.running_variance]


class NetworkFlashSAC:
    """Reference FlashSAC embedder, residual encoder, RMS norm, and heads."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        actor_heads: bool,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.output_dim = int(output_dim)
        self.actor_heads = bool(actor_heads)
        self.device = wp.get_device(device)
        self.log_std_min = -10.0
        self.log_std_max = 2.0
        self.default_training = False
        self.reference_batch_norm = True
        self._use_cublas = is_cublas_available(self.device)
        self.layer_sizes = (
            self.input_dim,
            self.hidden_dim,
            *((self.hidden_dim * 4, self.hidden_dim) * self.num_blocks),
            self.output_dim,
        )
        rng = np.random.default_rng(seed)
        self.embed_weight = self._weight(self.input_dim, self.hidden_dim, rng)
        self.block_weights: list[tuple[wp.array2d[wp.float32], wp.array2d[wp.float32]]] = []
        for _ in range(self.num_blocks):
            self.block_weights.append(
                (
                    self._weight(self.hidden_dim, self.hidden_dim * 4, rng),
                    self._weight(self.hidden_dim * 4, self.hidden_dim, rng),
                )
            )
        head_count = 2 if self.actor_heads else 1
        head_width = self.output_dim // 2 if self.actor_heads else self.output_dim
        self.head_weights = [self._weight(self.hidden_dim, head_width, rng) for _ in range(head_count)]
        self.head_biases = [
            wp.zeros(head_width, dtype=wp.float32, device=self.device, requires_grad=True) for _ in range(head_count)
        ]
        self.input_norm = _UnitBatchNorm(self.input_dim, self.device)
        self.block_norms = [
            (_UnitBatchNorm(self.hidden_dim * 4, self.device), _UnitBatchNorm(self.hidden_dim, self.device))
            for _ in range(self.num_blocks)
        ]
        self.rms_scale = wp.ones(self.hidden_dim, dtype=wp.float32, device=self.device, requires_grad=True)
        self.weights = [self.embed_weight]
        for w1, w2 in self.block_weights:
            self.weights.extend((w1, w2))
        self.weights.extend(self.head_weights)
        self.biases = self.head_biases
        self._manual_input: wp.array2d[wp.float32] | None = None
        self._manual_cache: dict[str, object] | None = None
        self._forward_rows = 0
        self._forward_buffers: dict[str, object] = {}
        self.normalize_parameters()

    def _weight(self, input_dim: int, output_dim: int, rng: np.random.Generator) -> wp.array2d[wp.float32]:
        return wp.array(
            _orthogonal(input_dim, output_dim, rng),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

    def parameters(self) -> list[wp.array]:
        params: list[wp.array] = list(self.weights)
        params.extend(self.head_biases)
        params.extend(self.input_norm.parameters())
        for norm1, norm2 in self.block_norms:
            params.extend(norm1.parameters())
            params.extend(norm2.parameters())
        params.append(self.rms_scale)
        return params

    def state_arrays(self) -> list[wp.array]:
        arrays: list[wp.array] = list(self.weights)
        arrays.extend(self.head_biases)
        arrays.extend(self.input_norm.state_arrays())
        for norm1, norm2 in self.block_norms:
            arrays.extend(norm1.state_arrays())
            arrays.extend(norm2.state_arrays())
        arrays.append(self.rms_scale)
        return arrays

    def _linear(
        self, x: wp.array2d[wp.float32], weight: wp.array2d[wp.float32], *, requires_grad: bool
    ) -> wp.array2d[wp.float32]:
        out = wp.empty(
            (int(x.shape[0]), int(weight.shape[1])),
            dtype=wp.float32,
            device=self.device,
            requires_grad=requires_grad,
        )
        if self._use_cublas:
            gemm_float32(x, weight, out, int(x.shape[0]), int(weight.shape[1]), int(weight.shape[0]))
        else:
            wp.launch(_unit_linear_kernel, dim=out.shape, inputs=[x, weight], outputs=[out], device=self.device)
        return out

    def _linear_into(
        self,
        x: wp.array2d[wp.float32],
        weight: wp.array2d[wp.float32],
        out: wp.array2d[wp.float32],
    ) -> None:
        if self._use_cublas:
            gemm_float32(x, weight, out, int(x.shape[0]), int(weight.shape[1]), int(weight.shape[0]))
        else:
            wp.launch(_unit_linear_kernel, dim=out.shape, inputs=[x, weight], outputs=[out], device=self.device)

    def reserve_forward_buffers(self, batch_size: int) -> None:
        """Reserve persistent inference buffers for one fixed batch size."""

        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        if rows == self._forward_rows:
            return
        blocks: list[tuple[wp.array2d[wp.float32], ...]] = []
        for _ in range(self.num_blocks):
            blocks.append(
                (
                    wp.empty((rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                    wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                )
            )
        head_width = self.output_dim // 2 if self.actor_heads else self.output_dim
        head_count = 2 if self.actor_heads else 1
        self._forward_buffers = {
            "input_normalized": wp.empty((rows, self.input_dim), dtype=wp.float32, device=self.device),
            "embed": wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "blocks": blocks,
            "inv_rms": wp.empty(rows, dtype=wp.float32, device=self.device),
            "normalized": wp.empty((rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "heads": [wp.empty((rows, head_width), dtype=wp.float32, device=self.device) for _ in range(head_count)],
            "output": wp.empty((rows, self.output_dim), dtype=wp.float32, device=self.device),
        }
        self._forward_rows = rows

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve reusable inference buffers for actor sampling."""

        self.reserve_forward_buffers(batch_size)

    def _forward_reuse(self, x: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        self.reserve_forward_buffers(int(x.shape[0]))
        buffers = self._forward_buffers
        input_normalized = buffers["input_normalized"]
        hidden = buffers["embed"]
        blocks = buffers["blocks"]
        inv_rms = buffers["inv_rms"]
        normalized = buffers["normalized"]
        heads = buffers["heads"]
        out = buffers["output"]
        if not isinstance(input_normalized, wp.array) or not isinstance(hidden, wp.array):
            raise RuntimeError("FlashSAC forward buffers are invalid")
        if not isinstance(inv_rms, wp.array) or not isinstance(normalized, wp.array) or not isinstance(out, wp.array):
            raise RuntimeError("FlashSAC forward buffers are invalid")
        if not isinstance(blocks, list) or not isinstance(heads, list):
            raise RuntimeError("FlashSAC forward buffers are invalid")
        wp.launch(
            _batch_norm_kernel,
            dim=input_normalized.shape,
            inputs=[
                x,
                self.input_norm.running_mean,
                self.input_norm.running_variance,
                self.input_norm.scale,
                self.input_norm.bias,
                self.input_norm.eps,
            ],
            outputs=[input_normalized],
            device=self.device,
        )
        self._linear_into(input_normalized, self.embed_weight, hidden)
        for block_index, ((w1, w2), (norm1, norm2)) in enumerate(
            zip(self.block_weights, self.block_norms, strict=True)
        ):
            residual = hidden
            pre1, normed1, activated1, pre2, normed2, activated2, block_out = blocks[block_index]
            self._linear_into(residual, w1, pre1)
            wp.launch(
                _batch_norm_kernel,
                dim=pre1.shape,
                inputs=[pre1, norm1.running_mean, norm1.running_variance, norm1.scale, norm1.bias, norm1.eps],
                outputs=[normed1],
                device=self.device,
            )
            wp.launch(_relu_kernel, dim=normed1.shape, inputs=[normed1], outputs=[activated1], device=self.device)
            self._linear_into(activated1, w2, pre2)
            wp.launch(
                _batch_norm_kernel,
                dim=pre2.shape,
                inputs=[pre2, norm2.running_mean, norm2.running_variance, norm2.scale, norm2.bias, norm2.eps],
                outputs=[normed2],
                device=self.device,
            )
            wp.launch(_relu_kernel, dim=normed2.shape, inputs=[normed2], outputs=[activated2], device=self.device)
            wp.launch(
                _residual_add_kernel,
                dim=block_out.shape,
                inputs=[activated2, residual],
                outputs=[block_out],
                device=self.device,
            )
            hidden = block_out
        wp.launch(
            _rms_inv_kernel,
            dim=hidden.shape[0],
            inputs=[hidden, 1.0e-6],
            outputs=[inv_rms],
            device=self.device,
        )
        wp.launch(
            _rms_norm_kernel,
            dim=hidden.shape,
            inputs=[hidden, self.rms_scale, inv_rms],
            outputs=[normalized],
            device=self.device,
        )
        offset = 0
        for head_index, (head, weight, bias) in enumerate(zip(heads, self.head_weights, self.head_biases, strict=True)):
            self._linear_into(normalized, weight, head)
            if self.actor_heads and head_index == 1:
                wp.launch(
                    _head_bias_log_std_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset, self.log_std_min, self.log_std_max],
                    outputs=[out],
                    device=self.device,
                )
            else:
                wp.launch(
                    _head_bias_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset],
                    outputs=[out],
                    device=self.device,
                )
            offset += int(head.shape[1])
        return out

    def _forward(
        self,
        x: wp.array2d[wp.float32],
        *,
        training: bool,
        requires_grad: bool,
    ) -> wp.array2d[wp.float32]:
        hidden = self.input_norm.forward(x, training=training, requires_grad=requires_grad)
        hidden = self._linear(hidden, self.embed_weight, requires_grad=requires_grad)
        for (w1, w2), (norm1, norm2) in zip(self.block_weights, self.block_norms, strict=True):
            residual = hidden
            hidden = self._linear(hidden, w1, requires_grad=requires_grad)
            hidden = norm1.forward(hidden, training=training, requires_grad=requires_grad)
            activated = wp.empty_like(hidden, requires_grad=requires_grad)
            wp.launch(_relu_kernel, dim=hidden.shape, inputs=[hidden], outputs=[activated], device=self.device)
            hidden = self._linear(activated, w2, requires_grad=requires_grad)
            hidden = norm2.forward(hidden, training=training, requires_grad=requires_grad)
            activated = wp.empty_like(hidden, requires_grad=requires_grad)
            wp.launch(_relu_kernel, dim=hidden.shape, inputs=[hidden], outputs=[activated], device=self.device)
            hidden = wp.empty_like(activated, requires_grad=requires_grad)
            wp.launch(
                _residual_add_kernel,
                dim=hidden.shape,
                inputs=[activated, residual],
                outputs=[hidden],
                device=self.device,
            )
        normalized = wp.empty_like(hidden, requires_grad=requires_grad)
        inv_rms = wp.empty(hidden.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            _rms_inv_kernel,
            dim=hidden.shape[0],
            inputs=[hidden, 1.0e-6],
            outputs=[inv_rms],
            device=self.device,
        )
        wp.launch(
            _rms_norm_kernel,
            dim=hidden.shape,
            inputs=[hidden, self.rms_scale, inv_rms],
            outputs=[normalized],
            device=self.device,
        )
        out = wp.empty(
            (int(x.shape[0]), self.output_dim),
            dtype=wp.float32,
            device=self.device,
            requires_grad=requires_grad,
        )
        offset = 0
        for head_index, (weight, bias) in enumerate(zip(self.head_weights, self.head_biases, strict=True)):
            head = self._linear(normalized, weight, requires_grad=requires_grad)
            if self.actor_heads and head_index == 1:
                wp.launch(
                    _head_bias_log_std_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset, self.log_std_min, self.log_std_max],
                    outputs=[out],
                    device=self.device,
                )
            else:
                wp.launch(
                    _head_bias_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset],
                    outputs=[out],
                    device=self.device,
                )
            offset += int(head.shape[1])
        return out

    def forward(
        self,
        x: wp.array2d[wp.float32],
        *,
        requires_grad: bool = True,
        training: bool | None = None,
    ) -> wp.array2d[wp.float32]:
        training = self.default_training if training is None else bool(training)
        return self._forward(x, training=training, requires_grad=requires_grad)

    def forward_reuse(self, x: wp.array2d[wp.float32]) -> wp.array2d[wp.float32]:
        return self._forward_reuse(x)

    def _linear_backward(
        self,
        x: wp.array2d[wp.float32],
        weight: wp.array2d[wp.float32],
        output_grad: wp.array2d[wp.float32],
    ) -> wp.array2d[wp.float32]:
        input_grad = wp.empty(x.shape, dtype=wp.float32, device=self.device)
        if self._use_cublas:
            gemm_float32(
                output_grad,
                weight,
                input_grad,
                int(x.shape[0]),
                int(weight.shape[0]),
                int(weight.shape[1]),
                transpose_rhs=True,
            )
            gemm_float32(
                x,
                output_grad,
                weight.grad,
                int(weight.shape[0]),
                int(weight.shape[1]),
                int(x.shape[0]),
                transpose_lhs=True,
            )
        else:
            wp.launch(
                _linear_input_grad_kernel,
                dim=x.shape,
                inputs=[output_grad, weight],
                outputs=[input_grad],
                device=self.device,
            )
            wp.launch(
                _linear_weight_grad_kernel,
                dim=weight.shape,
                inputs=[x, output_grad],
                outputs=[weight.grad],
                device=self.device,
            )
        return input_grad

    def _batch_norm_backward(
        self,
        norm: _UnitBatchNorm,
        x: wp.array2d[wp.float32],
        output_grad: wp.array2d[wp.float32],
        *,
        training: bool,
    ) -> wp.array2d[wp.float32]:
        input_grad = wp.empty(x.shape, dtype=wp.float32, device=self.device)
        if training:
            mean = wp.empty(norm.width, dtype=wp.float32, device=self.device)
            inv_std = wp.empty(norm.width, dtype=wp.float32, device=self.device)
            sum_grad = wp.empty(norm.width, dtype=wp.float32, device=self.device)
            sum_grad_normalized = wp.empty(norm.width, dtype=wp.float32, device=self.device)
            wp.launch(
                _batch_norm_backward_stats_kernel,
                dim=norm.width,
                inputs=[x, output_grad, norm.eps],
                outputs=[mean, inv_std, sum_grad, sum_grad_normalized, norm.scale.grad, norm.bias.grad],
                device=self.device,
            )
            wp.launch(
                _batch_norm_input_grad_kernel,
                dim=x.shape,
                inputs=[x, output_grad, norm.scale, mean, inv_std, sum_grad, sum_grad_normalized],
                outputs=[input_grad],
                device=self.device,
            )
        else:
            wp.launch(
                _batch_norm_inference_input_grad_kernel,
                dim=x.shape,
                inputs=[output_grad, norm.scale, norm.running_variance, norm.eps],
                outputs=[input_grad],
                device=self.device,
            )
            wp.launch(
                _batch_norm_inference_parameter_grad_kernel,
                dim=norm.width,
                inputs=[x, output_grad, norm.running_mean, norm.running_variance, norm.eps],
                outputs=[norm.scale.grad, norm.bias.grad],
                device=self.device,
            )
        return input_grad

    def forward_manual(self, x: wp.array2d[wp.float32], *, training: bool = True) -> wp.array2d[wp.float32]:
        """Evaluate a pass and retain activations for explicit backpropagation."""

        manual_input = wp.empty(x.shape, dtype=wp.float32, device=self.device)
        wp.launch(copy_2d_kernel, dim=x.shape, inputs=[x], outputs=[manual_input], device=self.device)
        input_normalized = self.input_norm.forward(manual_input, training=training, requires_grad=False)
        hidden = self._linear(input_normalized, self.embed_weight, requires_grad=False)
        block_cache: list[tuple[wp.array2d[wp.float32], ...]] = []
        for (w1, w2), (norm1, norm2) in zip(self.block_weights, self.block_norms, strict=True):
            residual = hidden
            pre1 = self._linear(residual, w1, requires_grad=False)
            normed1 = norm1.forward(pre1, training=training, requires_grad=False)
            activated1 = wp.empty_like(normed1)
            wp.launch(_relu_kernel, dim=normed1.shape, inputs=[normed1], outputs=[activated1], device=self.device)
            pre2 = self._linear(activated1, w2, requires_grad=False)
            normed2 = norm2.forward(pre2, training=training, requires_grad=False)
            activated2 = wp.empty_like(normed2)
            wp.launch(_relu_kernel, dim=normed2.shape, inputs=[normed2], outputs=[activated2], device=self.device)
            hidden = wp.empty_like(activated2)
            wp.launch(
                _residual_add_kernel,
                dim=hidden.shape,
                inputs=[activated2, residual],
                outputs=[hidden],
                device=self.device,
            )
            block_cache.append((residual, pre1, normed1, activated1, pre2, normed2, activated2))
        rms_input = hidden
        normalized = wp.empty_like(hidden)
        inv_rms = wp.empty(hidden.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            _rms_inv_kernel,
            dim=hidden.shape[0],
            inputs=[hidden, 1.0e-6],
            outputs=[inv_rms],
            device=self.device,
        )
        wp.launch(
            _rms_norm_kernel,
            dim=hidden.shape,
            inputs=[hidden, self.rms_scale, inv_rms],
            outputs=[normalized],
            device=self.device,
        )
        output = wp.empty((int(x.shape[0]), self.output_dim), dtype=wp.float32, device=self.device)
        heads: list[wp.array2d[wp.float32]] = []
        offset = 0
        for head_index, (weight, bias) in enumerate(zip(self.head_weights, self.head_biases, strict=True)):
            head = self._linear(normalized, weight, requires_grad=False)
            heads.append(head)
            if self.actor_heads and head_index == 1:
                wp.launch(
                    _head_bias_log_std_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset, self.log_std_min, self.log_std_max],
                    outputs=[output],
                    device=self.device,
                )
            else:
                wp.launch(
                    _head_bias_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset],
                    outputs=[output],
                    device=self.device,
                )
            offset += int(head.shape[1])
        self._manual_input = manual_input
        self._manual_cache = {
            "input_normalized": input_normalized,
            "blocks": block_cache,
            "rms_input": rms_input,
            "normalized": normalized,
            "heads": heads,
            "training": training,
        }
        return output

    def backward_manual(
        self,
        output_grad: wp.array2d[wp.float32],
        *,
        input_grad: wp.array2d[wp.float32] | None = None,
    ) -> None:
        """Backpropagate with explicit reference normalization derivatives."""

        if self._manual_input is None or self._manual_cache is None:
            raise RuntimeError("forward_manual() must be called before backward_manual()")
        normalized = self._manual_cache["normalized"]
        rms_input = self._manual_cache["rms_input"]
        heads = self._manual_cache["heads"]
        blocks = self._manual_cache["blocks"]
        input_normalized = self._manual_cache["input_normalized"]
        training = bool(self._manual_cache["training"])
        if not isinstance(normalized, wp.array) or not isinstance(rms_input, wp.array):
            raise RuntimeError("FlashSAC manual activation cache is invalid")
        if not isinstance(input_normalized, wp.array) or not isinstance(heads, list) or not isinstance(blocks, list):
            raise RuntimeError("FlashSAC manual activation cache is invalid")

        normalized_grad = wp.zeros(normalized.shape, dtype=wp.float32, device=self.device)
        offset = 0
        for head_index, (head, weight, bias) in enumerate(zip(heads, self.head_weights, self.head_biases, strict=True)):
            head_grad = wp.empty(head.shape, dtype=wp.float32, device=self.device)
            wp.launch(
                _head_output_grad_kernel,
                dim=head.shape,
                inputs=[
                    output_grad,
                    head,
                    bias,
                    offset,
                    self.actor_heads and head_index == 1,
                    self.log_std_min,
                    self.log_std_max,
                ],
                outputs=[head_grad],
                device=self.device,
            )
            wp.launch(_bias_grad_kernel, dim=bias.shape, inputs=[head_grad], outputs=[bias.grad], device=self.device)
            head_input_grad = self._linear_backward(normalized, weight, head_grad)
            accumulated = wp.empty_like(normalized_grad)
            wp.launch(
                _add_kernel,
                dim=normalized_grad.shape,
                inputs=[normalized_grad, head_input_grad],
                outputs=[accumulated],
                device=self.device,
            )
            normalized_grad = accumulated
            offset += int(head.shape[1])

        hidden_grad = wp.empty(rms_input.shape, dtype=wp.float32, device=self.device)
        inv_rms = wp.empty(rms_input.shape[0], dtype=wp.float32, device=self.device)
        projection = wp.empty(rms_input.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            _rms_norm_backward_stats_kernel,
            dim=rms_input.shape[0],
            inputs=[rms_input, normalized_grad, self.rms_scale, 1.0e-6],
            outputs=[inv_rms, projection],
            device=self.device,
        )
        wp.launch(
            _rms_norm_input_grad_kernel,
            dim=rms_input.shape,
            inputs=[rms_input, normalized_grad, self.rms_scale, inv_rms, projection],
            outputs=[hidden_grad],
            device=self.device,
        )
        wp.launch(
            _rms_norm_scale_grad_kernel,
            dim=self.rms_scale.shape,
            inputs=[rms_input, normalized_grad, inv_rms],
            outputs=[self.rms_scale.grad],
            device=self.device,
        )

        for block_index in reversed(range(self.num_blocks)):
            residual, pre1, _normed1, activated1, pre2, _normed2, activated2 = blocks[block_index]
            norm1, norm2 = self.block_norms[block_index]
            w1, w2 = self.block_weights[block_index]
            norm2_grad = wp.empty_like(activated2)
            wp.launch(
                _relu_grad_kernel,
                dim=activated2.shape,
                inputs=[activated2, hidden_grad],
                outputs=[norm2_grad],
                device=self.device,
            )
            pre2_grad = self._batch_norm_backward(norm2, pre2, norm2_grad, training=training)
            activated1_grad = self._linear_backward(activated1, w2, pre2_grad)
            norm1_grad = wp.empty_like(activated1)
            wp.launch(
                _relu_grad_kernel,
                dim=activated1.shape,
                inputs=[activated1, activated1_grad],
                outputs=[norm1_grad],
                device=self.device,
            )
            pre1_grad = self._batch_norm_backward(norm1, pre1, norm1_grad, training=training)
            branch_grad = self._linear_backward(residual, w1, pre1_grad)
            combined = wp.empty_like(hidden_grad)
            wp.launch(
                _add_kernel,
                dim=hidden_grad.shape,
                inputs=[hidden_grad, branch_grad],
                outputs=[combined],
                device=self.device,
            )
            hidden_grad = combined

        input_normalized_grad = self._linear_backward(input_normalized, self.embed_weight, hidden_grad)
        manual_input_grad = self._batch_norm_backward(
            self.input_norm, self._manual_input, input_normalized_grad, training=training
        )
        if input_grad is not None:
            wp.launch(
                copy_2d_kernel,
                dim=input_grad.shape,
                inputs=[manual_input_grad],
                outputs=[input_grad],
                device=self.device,
            )
        self._manual_cache = None

    def normalize_parameters(self, eps: float = 1.0e-8) -> None:
        for weight in self.weights:
            wp.launch(_normalize_columns_kernel, dim=weight.shape[1], inputs=[weight, eps], device=self.device)
        norms = [self.input_norm]
        for norm1, norm2 in self.block_norms:
            norms.extend((norm1, norm2))
        for norm in norms:
            wp.launch(
                _normalize_batch_affine_kernel,
                dim=1,
                inputs=[norm.scale, norm.bias, eps],
                device=self.device,
            )
        wp.launch(_normalize_scale_kernel, dim=1, inputs=[self.rms_scale, eps], device=self.device)

    def normalize_weights(self, eps: float = 1.0e-8) -> None:
        self.normalize_parameters(eps)

    def copy_from(self, source: NetworkFlashSAC) -> None:
        if len(self.state_arrays()) != len(source.state_arrays()):
            raise ValueError("FlashSAC network structures do not match")
        for dst, src in zip(self.state_arrays(), source.state_arrays(), strict=True):
            if dst.ndim == 1:
                wp.launch(copy_1d_kernel, dim=dst.shape, inputs=[src], outputs=[dst], device=self.device)
            else:
                wp.launch(copy_2d_kernel, dim=dst.shape, inputs=[src], outputs=[dst], device=self.device)

    def soft_update_from(self, source: NetworkFlashSAC, tau: float) -> None:
        if len(self.parameters()) != len(source.parameters()):
            raise ValueError("FlashSAC network structures do not match")
        for dst, src in zip(self.parameters(), source.parameters(), strict=True):
            if dst.ndim == 1:
                wp.launch(
                    soft_update_1d_kernel,
                    dim=dst.shape,
                    inputs=[src, float(tau)],
                    outputs=[dst],
                    device=self.device,
                )
            else:
                wp.launch(
                    soft_update_2d_kernel,
                    dim=dst.shape,
                    inputs=[src, float(tau)],
                    outputs=[dst],
                    device=self.device,
                )
