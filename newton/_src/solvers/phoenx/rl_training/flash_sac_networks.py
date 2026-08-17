# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure-Warp reference backbone layers for FlashSAC."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .cublas import (
    gemm_float16,
    gemm_float16_output,
    gemm_float16_strided_batched,
    gemm_float16_strided_batched_output,
    gemm_float32,
    gemm_float32_strided_batched,
    is_cublas_available,
)
from .kernels import (
    copy_1d_kernel,
    copy_2d_kernel,
    population_copy_float16_3d_kernel,
    population_copy_float_2d_kernel,
    population_copy_float_3d_kernel,
    soft_update_1d_kernel,
    soft_update_2d_kernel,
    unit_normalize_weight_columns_tile_kernel,
)
from .kernels_bf16 import cast_2d_float_to_float16_kernel, cast_3d_float_to_float16_kernel

_TILE_REDUCTION_BLOCK_DIM = 256


@wp.kernel
def _copy_2d_same_kernel(src: wp.array2d[Any], dst: wp.array2d[Any]):
    row, col = wp.tid()
    dst[row, col] = src[row, col]


@wp.kernel
def _scale_1d_kernel(values: wp.array[wp.float32], scale: wp.array[wp.float32], found_inf: wp.array[wp.int32]):
    i = wp.tid()
    value = values[i] / scale[0]
    if not wp.isfinite(value):
        wp.atomic_max(found_inf, 0, wp.int32(1))
    values[i] = value


@wp.kernel
def _scale_2d_kernel(values: wp.array2d[wp.float32], scale: wp.array[wp.float32], found_inf: wp.array[wp.int32]):
    i, j = wp.tid()
    value = values[i, j] / scale[0]
    if not wp.isfinite(value):
        wp.atomic_max(found_inf, 0, wp.int32(1))
    values[i, j] = value


def _unscale_parameter_grads(
    parameters: list[wp.array], loss_scale: wp.array[wp.float32] | None, found_inf: wp.array[wp.int32] | None
) -> None:
    if loss_scale is None:
        return
    if found_inf is None:
        raise RuntimeError("scaled backward requires a found_inf buffer")
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if grad.ndim == 1:
            wp.launch(_scale_1d_kernel, dim=grad.shape, inputs=[grad, loss_scale, found_inf], device=grad.device)
        else:
            wp.launch(_scale_2d_kernel, dim=grad.shape, inputs=[grad, loss_scale, found_inf], device=grad.device)


@wp.kernel
def _unit_linear_kernel(
    x: wp.array2d[Any],
    weight: wp.array2d[Any],
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    value = wp.float32(0.0)
    for inner in range(weight.shape[0]):
        value += x[row, inner] * weight[inner, col]
    out[row, col] = value


@wp.kernel
def _linear_input_grad_kernel(
    output_grad: wp.array2d[Any],
    weight: wp.array2d[Any],
    input_grad: wp.array2d[Any],
):
    row, inner = wp.tid()
    value = wp.float32(0.0)
    for col in range(weight.shape[1]):
        value += wp.float32(output_grad[row, col]) * weight[inner, col]
    input_grad[row, inner] = value


@wp.kernel
def _linear_weight_grad_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    weight_grad: wp.array2d[Any],
):
    inner, col = wp.tid()
    value = wp.float32(0.0)
    for row in range(x.shape[0]):
        value += x[row, inner] * output_grad[row, col]
    weight_grad[inner, col] = value


@wp.kernel(enable_backward=False)
def _transpose_2d_tile_kernel(x: wp.array2d[Any], transposed: wp.array2d[Any]):
    row_group, column_group, _lane = wp.tid()
    tile = wp.tile_load(
        x,
        shape=(32, 32),
        offset=(row_group * 32, column_group * 32),
        storage="shared",
    )
    wp.tile_store(transposed, wp.tile_transpose(tile), offset=(column_group * 32, row_group * 32))


@wp.kernel(enable_backward=False)
def _batch_moments_transposed_tile_kernel(
    transposed: wp.array2d[Any],
    count: wp.int32,
    eps: wp.float32,
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
):
    column, lane = wp.tid()
    total = wp.float32(0.0)
    for row in range(lane, count, wp.block_dim()):
        total += wp.float32(transposed[column, row])
    batch_mean = wp.tile_sum(wp.tile(total))[0] / wp.float32(count)
    squared = wp.float32(0.0)
    for row in range(lane, count, wp.block_dim()):
        delta = wp.float32(transposed[column, row]) - batch_mean
        squared += delta * delta
    batch_variance = wp.tile_sum(wp.tile(squared))[0] / wp.float32(count)
    if lane == 0:
        mean[column] = batch_mean
        variance[column] = batch_variance
        inv_std[column] = wp.float32(1.0) / wp.sqrt(batch_variance + eps)


@wp.kernel(enable_backward=False)
def _batch_moments_tile_kernel(
    x: wp.array2d[Any],
    count: wp.int32,
    eps: wp.float32,
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
):
    col, lane = wp.tid()
    total = wp.float32(0.0)
    for row in range(lane, count, wp.block_dim()):
        total += wp.float32(x[row, col])
    batch_mean = wp.tile_sum(wp.tile(total))[0] / wp.float32(count)
    squared = wp.float32(0.0)
    for row in range(lane, count, wp.block_dim()):
        delta = wp.float32(x[row, col]) - batch_mean
        squared += delta * delta
    batch_variance = wp.tile_sum(wp.tile(squared))[0] / wp.float32(count)
    if lane == 0:
        mean[col] = batch_mean
        variance[col] = batch_variance
        inv_std[col] = wp.float32(1.0) / wp.sqrt(batch_variance + eps)


@wp.kernel(enable_backward=False)
def _batch_norm_backward_tile_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    sum_grad: wp.array[wp.float32],
    sum_grad_normalized: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    col, lane = wp.tid()
    grad_sum = wp.float32(0.0)
    grad_normalized_sum = wp.float32(0.0)
    for row in range(lane, x.shape[0], wp.block_dim()):
        grad = wp.float32(output_grad[row, col])
        grad_sum += grad
        grad_normalized_sum += grad * (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    reduced = wp.tile_sum(wp.tile(wp.vec2(grad_sum, grad_normalized_sum)), axis=1)
    if lane == 0:
        sum_grad[col] = reduced[0]
        sum_grad_normalized[col] = reduced[1]
        bias_grad[col] = reduced[0]
        scale_grad[col] = reduced[1]


@wp.kernel(enable_backward=False)
def _batch_norm_backward_amp_transposed_tile_kernel(
    transposed: wp.array2d[Any],
    transposed_output_grad: wp.array2d[wp.float32],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    mean_grad: wp.array[wp.float32],
    variance_grad: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    column, lane = wp.tid()
    bias_sum = wp.float32(0.0)
    scale_sum = wp.float32(0.0)
    centered_grad_sum = wp.float32(0.0)
    inv_std_grad_sum = wp.float32(0.0)
    for row in range(lane, transposed.shape[1], wp.block_dim()):
        grad = transposed_output_grad[column, row]
        centered = wp.float16(wp.float32(transposed[column, row]) - mean[column])
        normalized = wp.float32(centered) * inv_std[column]
        centered_grad = wp.float16(grad * scale[column] * inv_std[column])
        bias_sum += grad
        scale_sum += grad * normalized
        centered_grad_sum += wp.float32(centered_grad)
        inv_std_grad_sum += grad * scale[column] * wp.float32(centered)
    reduced = wp.tile_sum(wp.tile(wp.vec4(bias_sum, scale_sum, centered_grad_sum, inv_std_grad_sum)), axis=1)
    if lane == 0:
        bias_grad[column] = reduced[0]
        scale_grad[column] = reduced[1]
        mean_grad[column] = wp.float32(wp.float16(-wp.float32(wp.float16(reduced[2]))))
        inv_cube = inv_std[column] * inv_std[column] * inv_std[column]
        variance_grad[column] = wp.float32(wp.float16(reduced[3] * wp.float32(-0.5) * inv_cube))


@wp.kernel(enable_backward=False)
def _batch_norm_backward_amp_tile_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    mean_grad: wp.array[wp.float32],
    variance_grad: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    col, lane = wp.tid()
    bias_sum = wp.float32(0.0)
    scale_sum = wp.float32(0.0)
    centered_grad_sum = wp.float32(0.0)
    inv_std_grad_sum = wp.float32(0.0)
    for row in range(lane, x.shape[0], wp.block_dim()):
        grad = wp.float32(output_grad[row, col])
        centered = wp.float16(wp.float32(x[row, col]) - mean[col])
        normalized = wp.float32(centered) * inv_std[col]
        centered_grad = wp.float16(grad * scale[col] * inv_std[col])
        bias_sum += grad
        scale_sum += grad * normalized
        centered_grad_sum += wp.float32(centered_grad)
        inv_std_grad_sum += grad * scale[col] * wp.float32(centered)
    reduced = wp.tile_sum(wp.tile(wp.vec4(bias_sum, scale_sum, centered_grad_sum, inv_std_grad_sum)), axis=1)
    if lane == 0:
        bias_grad[col] = reduced[0]
        scale_grad[col] = reduced[1]
        mean_grad[col] = wp.float32(wp.float16(-wp.float32(wp.float16(reduced[2]))))
        inv_cube = inv_std[col] * inv_std[col] * inv_std[col]
        variance_grad[col] = wp.float32(wp.float16(reduced[3] * wp.float32(-0.5) * inv_cube))


@wp.kernel
def _batch_norm_input_grad_amp_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    mean_grad: wp.array[wp.float32],
    variance_grad: wp.array[wp.float32],
    input_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    count = wp.float32(x.shape[0])
    centered = wp.float16(wp.float32(x[row, col]) - mean[col])
    direct = wp.float16(wp.float32(output_grad[row, col]) * scale[col] * inv_std[col])
    variance_path = wp.float16(variance_grad[col] * wp.float32(2.0) * wp.float32(centered) / count)
    mean_path = wp.float16(mean_grad[col] / count)
    combined = wp.float16(wp.float32(direct) + wp.float32(variance_path))
    input_grad[row, col] = wp.float16(wp.float32(combined) + wp.float32(mean_path))


@wp.kernel(enable_backward=False)
def _transpose_population_2d_tile_kernel(
    x: wp.array2d[Any],
    rows: wp.int32,
    width: wp.int32,
    transposed: wp.array2d[Any],
):
    member_row_group, column_group, _lane = wp.tid()
    row_groups = (rows + wp.int32(31)) // wp.int32(32)
    transposed_stride = ((width + wp.int32(31)) // wp.int32(32)) * wp.int32(32)
    member = member_row_group // row_groups
    row_group = member_row_group - member * row_groups
    tile = wp.tile_load(
        x,
        shape=(32, 32),
        offset=(member * rows + row_group * 32, column_group * 32),
        storage="shared",
    )
    wp.tile_store(
        transposed,
        wp.tile_transpose(tile),
        offset=(member * transposed_stride + column_group * 32, row_group * 32),
    )


@wp.kernel(enable_backward=False)
def _batch_moments_transposed_population_tile_kernel(
    transposed: wp.array2d[Any],
    rows: wp.int32,
    width: wp.int32,
    eps: wp.float32,
    mean: wp.array2d[wp.float32],
    variance: wp.array2d[wp.float32],
    inv_std: wp.array2d[wp.float32],
):
    member_column, lane = wp.tid()
    member = member_column // width
    column = member_column - member * width
    transposed_stride = ((width + wp.int32(31)) // wp.int32(32)) * wp.int32(32)
    source_column = member * transposed_stride + column
    total = wp.float32(0.0)
    for row in range(lane, rows, wp.block_dim()):
        total += wp.float32(transposed[source_column, row])
    batch_mean = wp.tile_sum(wp.tile(total))[0] / wp.float32(rows)
    squared = wp.float32(0.0)
    for row in range(lane, rows, wp.block_dim()):
        delta = wp.float32(transposed[source_column, row]) - batch_mean
        squared += delta * delta
    batch_variance = wp.tile_sum(wp.tile(squared))[0] / wp.float32(rows)
    if lane == 0:
        mean[member, column] = batch_mean
        variance[member, column] = batch_variance
        inv_std[member, column] = wp.float32(1.0) / wp.sqrt(batch_variance + eps)


@wp.kernel(enable_backward=False)
def _batch_norm_backward_amp_population_tile_kernel(
    transposed: wp.array2d[Any],
    transposed_output_grad: wp.array2d[wp.float32],
    rows: wp.int32,
    width: wp.int32,
    mean: wp.array2d[wp.float32],
    inv_std: wp.array2d[wp.float32],
    scale: wp.array2d[wp.float32],
    mean_grad: wp.array2d[wp.float32],
    variance_grad: wp.array2d[wp.float32],
    scale_grad: wp.array2d[wp.float32],
    bias_grad: wp.array2d[wp.float32],
):
    member_column, lane = wp.tid()
    member = member_column // width
    column = member_column - member * width
    transposed_stride = ((width + wp.int32(31)) // wp.int32(32)) * wp.int32(32)
    source_column = member * transposed_stride + column
    bias_sum = wp.float32(0.0)
    scale_sum = wp.float32(0.0)
    centered_grad_sum = wp.float32(0.0)
    inv_std_grad_sum = wp.float32(0.0)
    for row in range(lane, rows, wp.block_dim()):
        grad = transposed_output_grad[source_column, row]
        centered = wp.float16(wp.float32(transposed[source_column, row]) - mean[member, column])
        normalized = wp.float32(centered) * inv_std[member, column]
        centered_grad = wp.float16(grad * scale[member, column] * inv_std[member, column])
        bias_sum += grad
        scale_sum += grad * normalized
        centered_grad_sum += wp.float32(centered_grad)
        inv_std_grad_sum += grad * scale[member, column] * wp.float32(centered)
    reduced = wp.tile_sum(wp.tile(wp.vec4(bias_sum, scale_sum, centered_grad_sum, inv_std_grad_sum)), axis=1)
    if lane == 0:
        bias_grad[member, column] = reduced[0]
        scale_grad[member, column] = reduced[1]
        mean_grad[member, column] = wp.float32(wp.float16(-wp.float32(wp.float16(reduced[2]))))
        inv_cube = inv_std[member, column] * inv_std[member, column] * inv_std[member, column]
        variance_grad[member, column] = wp.float32(wp.float16(reduced[3] * wp.float32(-0.5) * inv_cube))


@wp.kernel
def _batch_norm_backward_stats_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
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
        batch_mean += wp.float32(x[row, col])
    batch_mean /= wp.float32(x.shape[0])
    variance = wp.float32(0.0)
    for row in range(x.shape[0]):
        delta = wp.float32(x[row, col]) - batch_mean
        variance += delta * delta
    variance /= wp.float32(x.shape[0])
    batch_inv_std = wp.float32(1.0) / wp.sqrt(variance + eps)
    grad_sum = wp.float32(0.0)
    grad_normalized_sum = wp.float32(0.0)
    for row in range(x.shape[0]):
        normalized = (wp.float32(x[row, col]) - batch_mean) * batch_inv_std
        grad_sum += wp.float32(output_grad[row, col])
        grad_normalized_sum += wp.float32(output_grad[row, col]) * normalized
    mean[col] = batch_mean
    inv_std[col] = batch_inv_std
    sum_grad[col] = grad_sum
    sum_grad_normalized[col] = grad_normalized_sum
    scale_grad[col] = grad_normalized_sum
    bias_grad[col] = grad_sum


@wp.kernel
def _batch_norm_input_grad_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    sum_grad: wp.array[wp.float32],
    sum_grad_normalized: wp.array[wp.float32],
    input_grad: wp.array2d[Any],
):
    row, col = wp.tid()
    count = wp.float32(x.shape[0])
    normalized = (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    input_grad[row, col] = (
        scale[col]
        * inv_std[col]
        * (wp.float32(output_grad[row, col]) - sum_grad[col] / count - normalized * sum_grad_normalized[col] / count)
    )


@wp.kernel
def _batch_norm_inference_input_grad_kernel(
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
    eps: wp.float32,
    input_grad: wp.array2d[Any],
):
    row, col = wp.tid()
    input_grad[row, col] = wp.float32(output_grad[row, col]) * scale[col] / wp.sqrt(running_variance[col] + eps)


@wp.kernel
def _batch_norm_inference_parameter_grad_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
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
        grad_scale += wp.float32(output_grad[row, col]) * (wp.float32(x[row, col]) - running_mean[col]) * inv_std
        grad_bias += wp.float32(output_grad[row, col])
    scale_grad[col] = grad_scale
    bias_grad[col] = grad_bias


@wp.kernel(enable_backward=False)
def _batch_norm_inference_parameter_grad_tile_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    running_mean: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
    eps: wp.float32,
    scale_grad: wp.array[wp.float32],
    bias_grad: wp.array[wp.float32],
):
    col, lane = wp.tid()
    inv_std = wp.float32(1.0) / wp.sqrt(running_variance[col] + eps)
    grad_scale = wp.float32(0.0)
    grad_bias = wp.float32(0.0)
    for row in range(lane, x.shape[0], wp.block_dim()):
        grad = wp.float32(output_grad[row, col])
        grad_scale += grad * (wp.float32(x[row, col]) - running_mean[col]) * inv_std
        grad_bias += grad
    reduced = wp.tile_sum(wp.tile(wp.vec2(grad_scale, grad_bias)), axis=1)
    if lane == 0:
        scale_grad[col] = reduced[0]
        bias_grad[col] = reduced[1]


@wp.kernel(enable_backward=False)
def _rms_norm_backward_stats_tile_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
):
    row, lane = wp.tid()
    mean_square = wp.float32(0.0)
    projected = wp.float32(0.0)
    for col in range(lane, x.shape[1], wp.block_dim()):
        value = wp.float32(x[row, col])
        mean_square += value * value
        projected += wp.float32(output_grad[row, col]) * scale[col] * value
    reduced = wp.tile_sum(wp.tile(wp.vec2(mean_square, projected)), axis=1)
    if lane == 0:
        inv_rms[row] = wp.float32(1.0) / wp.sqrt(reduced[0] / wp.float32(x.shape[1]) + eps)
        projection[row] = reduced[1]


@wp.kernel(enable_backward=False)
def _rms_norm_scale_grad_tile_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    inv_rms: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
):
    col, lane = wp.tid()
    value = wp.float32(0.0)
    for row in range(lane, x.shape[0], wp.block_dim()):
        value += wp.float32(output_grad[row, col]) * wp.float32(x[row, col]) * inv_rms[row]
    reduced = wp.tile_sum(wp.tile(value))
    if lane == 0:
        scale_grad[col] = reduced[0]


@wp.kernel(enable_backward=False)
def _bias_grad_tile_kernel(output_grad: wp.array2d[Any], bias_grad: wp.array[wp.float32]):
    col, lane = wp.tid()
    value = wp.float32(0.0)
    for row in range(lane, output_grad.shape[0], wp.block_dim()):
        value += wp.float32(output_grad[row, col])
    reduced = wp.tile_sum(wp.tile(value))
    if lane == 0:
        bias_grad[col] = reduced[0]


@wp.kernel
def _relu_grad_kernel(
    activated: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    input_grad: wp.array2d[Any],
):
    row, col = wp.tid()
    input_grad[row, col] = wp.where(activated[row, col] > wp.float32(0.0), output_grad[row, col], wp.float32(0.0))


@wp.kernel
def _add_kernel(a: wp.array2d[Any], b: wp.array2d[Any], out: wp.array2d[Any]):
    row, col = wp.tid()
    out[row, col] = a[row, col] + b[row, col]


@wp.kernel
def _rms_norm_backward_stats_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
):
    row = wp.tid()
    mean_square = wp.float32(0.0)
    projected = wp.float32(0.0)
    for col in range(x.shape[1]):
        mean_square += wp.float32(x[row, col]) * wp.float32(x[row, col])
        projected += wp.float32(output_grad[row, col]) * scale[col] * wp.float32(x[row, col])
    inv_rms[row] = wp.float32(1.0) / wp.sqrt(mean_square / wp.float32(x.shape[1]) + eps)
    projection[row] = projected


@wp.kernel
def _rms_norm_input_grad_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
    input_grad: wp.array2d[Any],
):
    row, col = wp.tid()
    input_grad[row, col] = wp.float32(output_grad[row, col]) * scale[col] * inv_rms[row] - (
        x[row, col] * inv_rms[row] * inv_rms[row] * inv_rms[row] * projection[row] / wp.float32(x.shape[1])
    )


@wp.kernel
def _rms_norm_scale_grad_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    inv_rms: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
):
    col = wp.tid()
    value = wp.float32(0.0)
    for row in range(x.shape[0]):
        value += wp.float32(output_grad[row, col]) * wp.float32(x[row, col]) * inv_rms[row]
    scale_grad[col] = value


@wp.kernel
def _head_output_grad_kernel(
    output_grad: wp.array2d[Any],
    head: wp.array2d[Any],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    smooth_log_std: wp.bool,
    minimum: wp.float32,
    maximum: wp.float32,
    head_grad: wp.array2d[Any],
):
    row, col = wp.tid()
    value = output_grad[row, col + offset]
    if smooth_log_std:
        mapped = wp.tanh(head[row, col] + bias[col])
        value *= (maximum - minimum) * wp.float32(0.5) * (wp.float32(1.0) - mapped * mapped)
    head_grad[row, col] = value


@wp.kernel
def _bias_grad_kernel(output_grad: wp.array2d[Any], bias_grad: wp.array[wp.float32]):
    col = wp.tid()
    value = wp.float32(0.0)
    for row in range(output_grad.shape[0]):
        value += wp.float32(output_grad[row, col])
    bias_grad[col] = value


@wp.kernel(enable_backward=False)
def _batch_moments_kernel(
    x: wp.array2d[Any],
    count: wp.int32,
    eps: wp.float32,
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
):
    col = wp.tid()
    total = wp.float32(0.0)
    for row in range(count):
        total += wp.float32(x[row, col])
    batch_mean = total / wp.float32(count)
    squared = wp.float32(0.0)
    for row in range(count):
        delta = wp.float32(x[row, col]) - batch_mean
        squared += delta * delta
    mean[col] = batch_mean
    variance[col] = squared / wp.float32(count)
    inv_std[col] = wp.float32(1.0) / wp.sqrt(wp.float32(wp.float16(variance[col] + eps)))


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
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    normalized = (wp.float32(x[row, col]) - mean[col]) / wp.sqrt(variance[col] + eps)
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _batch_norm_relu_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    normalized = (wp.float32(x[row, col]) - mean[col]) / wp.sqrt(variance[col] + eps)
    out[row, col] = wp.max(normalized * scale[col] + bias[col], wp.float32(0.0))


@wp.kernel
def _batch_norm_inv_std_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    normalized = (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _round_batch_moments_f16_kernel(
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    eps: wp.float32,
    inv_std: wp.array[wp.float32],
):
    col = wp.tid()
    mean[col] = wp.float32(wp.float16(mean[col]))
    variance[col] = wp.float32(wp.float16(variance[col]))
    inv_std[col] = wp.float32(1.0) / wp.sqrt(wp.float32(wp.float16(variance[col] + eps)))


@wp.kernel
def _batch_norm_inv_std_amp_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    centered = wp.float16(wp.float32(x[row, col]) - mean[col])
    normalized = wp.float32(centered) * inv_std[col]
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _batch_norm_inv_std_relu_amp_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    centered = wp.float16(wp.float32(x[row, col]) - mean[col])
    normalized = wp.float32(centered) * inv_std[col]
    out[row, col] = wp.max(normalized * scale[col] + bias[col], wp.float32(0.0))


@wp.kernel
def _batch_norm_inv_std_amp_dual_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    relu: bool,
    out: wp.array2d[wp.float32],
    out_f16: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    centered = wp.float16(wp.float32(x[row, col]) - mean[col])
    normalized = wp.float32(centered) * inv_std[col]
    value = normalized * scale[col] + bias[col]
    if relu:
        value = wp.max(value, wp.float32(0.0))
    out[row, col] = value
    out_f16[row, col] = wp.float16(value)


@wp.kernel
def _batch_norm_amp_dual_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    relu: bool,
    out: wp.array2d[wp.float32],
    out_f16: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = (wp.float32(x[row, col]) - mean[col]) / wp.sqrt(variance[col] + eps)
    value = value * scale[col] + bias[col]
    if relu:
        value = wp.max(value, wp.float32(0.0))
    out[row, col] = value
    out_f16[row, col] = wp.float16(value)


@wp.kernel
def _batch_norm_inv_std_relu_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    normalized = (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    out[row, col] = wp.max(normalized * scale[col] + bias[col], wp.float32(0.0))


@wp.kernel
def _batch_norm_training_kernel(
    x: wp.array2d[Any],
    count: wp.int32,
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[Any],
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
    normalized = (wp.float32(x[row, col]) - mean) / wp.sqrt(variance + eps)
    out[row, col] = normalized * scale[col] + bias[col]


@wp.kernel
def _relu_kernel(x: wp.array2d[Any], out: wp.array2d[Any]):
    row, col = wp.tid()
    out[row, col] = wp.max(x[row, col], wp.float32(0.0))


@wp.kernel
def _residual_add_kernel(
    x: wp.array2d[Any],
    residual: wp.array2d[Any],
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    out[row, col] = x[row, col] + residual[row, col]


@wp.kernel
def _rms_inv_kernel(
    x: wp.array2d[Any],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
):
    row = wp.tid()
    squared = wp.float32(0.0)
    for col in range(x.shape[1]):
        squared += x[row, col] * x[row, col]
    inv_rms[row] = wp.float32(1.0) / wp.sqrt(squared / wp.float32(x.shape[1]) + eps)


@wp.kernel(enable_backward=False)
def _rms_inv_tile_kernel(
    x: wp.array2d[Any],
    eps: wp.float32,
    inv_rms: wp.array[wp.float32],
):
    row, lane = wp.tid()
    squared = wp.float32(0.0)
    for col in range(lane, x.shape[1], wp.block_dim()):
        value = wp.float32(x[row, col])
        squared += value * value
    reduced = wp.tile_sum(wp.tile(squared))
    if lane == 0:
        inv_rms[row] = wp.float32(1.0) / wp.sqrt(reduced[0] / wp.float32(x.shape[1]) + eps)


@wp.kernel
def _rms_norm_kernel(
    x: wp.array2d[Any],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    out[row, col] = x[row, col] * inv_rms[row] * scale[col]


@wp.kernel
def _head_bias_kernel(
    x: wp.array2d[Any],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    out[row, col + offset] = wp.float32(x[row, col]) + bias[col]


@wp.kernel
def _head_bias_log_std_kernel(
    x: wp.array2d[Any],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    minimum: wp.float32,
    maximum: wp.float32,
    out: wp.array2d[Any],
):
    row, col = wp.tid()
    raw = wp.float32(x[row, col]) + bias[col]
    out[row, col + offset] = minimum + (maximum - minimum) * wp.float32(0.5) * (wp.float32(1.0) + wp.tanh(raw))


@wp.kernel
def _head_bias_amp_kernel(
    x: wp.array2d[Any],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    value = wp.float16(wp.float32(x[row, col]) + bias[col])
    out[row, col + offset] = wp.float32(value)


@wp.kernel
def _head_bias_log_std_amp_kernel(
    x: wp.array2d[Any],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    minimum: wp.float32,
    maximum: wp.float32,
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    raw = wp.float16(wp.float32(x[row, col]) + bias[col])
    tanh_raw = wp.float16(wp.tanh(wp.float32(raw)))
    unit = wp.float16(wp.float32(1.0) + wp.float32(tanh_raw))
    scaled = wp.float16(wp.float32(wp.float16((maximum - minimum) * wp.float32(0.5))) * wp.float32(unit))
    log_std = wp.float16(minimum + wp.float32(scaled))
    out[row, col + offset] = wp.float32(log_std)


@wp.kernel
def _batch_norm_f16_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = (wp.float32(x[row, col]) - mean[col]) / wp.sqrt(variance[col] + eps)
    out[row, col] = wp.float16(value * scale[col] + bias[col])


@wp.kernel
def _batch_norm_inv_std_f16_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    out[row, col] = wp.float16(value * scale[col] + bias[col])


@wp.kernel
def _batch_norm_relu_f16_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    variance: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = (wp.float32(x[row, col]) - mean[col]) / wp.sqrt(variance[col] + eps)
    out[row, col] = wp.float16(wp.max(value * scale[col] + bias[col], wp.float32(0.0)))


@wp.kernel
def _batch_norm_inv_std_relu_f16_kernel(
    x: wp.array2d[Any],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = (wp.float32(x[row, col]) - mean[col]) * inv_std[col]
    out[row, col] = wp.float16(wp.max(value * scale[col] + bias[col], wp.float32(0.0)))


@wp.kernel
def _residual_add_f16_kernel(
    x: wp.array2d[wp.float16],
    residual: wp.array2d[wp.float16],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    out[row, col] = wp.float16(wp.float32(x[row, col]) + wp.float32(residual[row, col]))


@wp.kernel
def _residual_add_mixed_f32_kernel(
    x: wp.array2d[Any],
    residual: wp.array2d[Any],
    out: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    out[row, col] = wp.float32(x[row, col]) + wp.float32(residual[row, col])


@wp.kernel
def _residual_add_mixed_dual_kernel(
    x: wp.array2d[Any],
    residual: wp.array2d[Any],
    out: wp.array2d[wp.float32],
    out_f16: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.float32(x[row, col]) + wp.float32(residual[row, col])
    out[row, col] = value
    out_f16[row, col] = wp.float16(value)


@wp.kernel
def _add_mixed_f16_kernel(
    a: wp.array2d[Any],
    b: wp.array2d[Any],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    out[row, col] = wp.float16(wp.float32(a[row, col]) + wp.float32(b[row, col]))


@wp.kernel
def _rms_norm_f16_kernel(
    x: wp.array2d[wp.float16],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()

    out[row, col] = wp.float16(wp.float32(x[row, col]) * inv_rms[row] * scale[col])


@wp.kernel
def _rms_norm_dual_kernel(
    x: wp.array2d[Any],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
    out_f16: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.float32(x[row, col]) * inv_rms[row] * scale[col]
    out[row, col] = value
    out_f16[row, col] = wp.float16(value)


@wp.kernel
def _relu_grad_f16_kernel(
    activated: wp.array2d[wp.float16],
    output_grad: wp.array2d[wp.float16],
    input_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.where(
        wp.float32(activated[row, col]) > wp.float32(0.0), wp.float32(output_grad[row, col]), wp.float32(0.0)
    )
    input_grad[row, col] = wp.float16(value)


@wp.kernel
def _add_f16_kernel(
    a: wp.array2d[wp.float16],
    b: wp.array2d[wp.float16],
    out: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    out[row, col] = wp.float16(wp.float32(a[row, col]) + wp.float32(b[row, col]))


@wp.kernel
def _batch_norm_input_grad_f16_kernel(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    mean: wp.array[wp.float32],
    inv_std: wp.array[wp.float32],
    sum_grad: wp.array[wp.float32],
    sum_grad_normalized: wp.array[wp.float32],
    input_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    count = wp.float32(x.shape[0])
    centered = wp.float32(x[row, col]) - mean[col]
    normalized = centered * inv_std[col]
    value = (
        scale[col]
        * inv_std[col]
        / count
        * (count * wp.float32(output_grad[row, col]) - sum_grad[col] - normalized * sum_grad_normalized[col])
    )
    input_grad[row, col] = wp.float16(value)


@wp.kernel
def _batch_norm_inference_input_grad_f16_kernel(
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    running_variance: wp.array[wp.float32],
    eps: wp.float32,
    input_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.float32(output_grad[row, col]) * scale[col] / wp.sqrt(running_variance[col] + eps)
    input_grad[row, col] = wp.float16(value)


@wp.kernel
def _rms_norm_input_grad_f16_kernel(
    x: wp.array2d[wp.float16],
    output_grad: wp.array2d[wp.float16],
    scale: wp.array[wp.float32],
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
    input_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.float32(output_grad[row, col]) * scale[col] * inv_rms[row] - wp.float32(x[row, col]) * projection[
        row
    ] * inv_rms[row] * inv_rms[row] * inv_rms[row] / wp.float32(x.shape[1])
    input_grad[row, col] = wp.float16(value)


@wp.kernel
def _head_output_grad_f16_kernel(
    output_grad: wp.array2d[wp.float32],
    head: wp.array2d[wp.float16],
    bias: wp.array[wp.float32],
    offset: wp.int32,
    smooth_log_std: wp.bool,
    minimum: wp.float32,
    maximum: wp.float32,
    head_grad: wp.array2d[wp.float16],
):
    row, col = wp.tid()
    value = wp.float16(output_grad[row, col + offset])
    if smooth_log_std:
        raw = wp.float16(wp.float32(head[row, col]) + bias[col])
        t = wp.float16(wp.tanh(wp.float32(raw)))
        one_minus_square = wp.float16(wp.float32(1.0) - wp.float32(wp.float16(wp.float32(t) * wp.float32(t))))
        scale = wp.float16((maximum - minimum) * wp.float32(0.5))
        derivative = wp.float16(wp.float32(scale) * wp.float32(one_minus_square))
        value = wp.float16(wp.float32(value) * wp.float32(derivative))
    head_grad[row, col] = value


@wp.kernel
def _normalize_columns_kernel(weight: wp.array2d[Any], eps: wp.float32):
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


@wp.kernel(enable_backward=False)
def _normalize_batch_affine_tile_kernel(
    scale: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    eps: wp.float32,
):
    lane = wp.tid()
    squared = wp.float32(0.0)
    for col in range(lane, scale.shape[0], wp.block_dim()):
        squared += scale[col] * scale[col] + bias[col] * bias[col]
    total = wp.tile_sum(wp.tile(squared))[0]
    factor = wp.sqrt(wp.float32(scale.shape[0])) / wp.sqrt(total + eps)
    for col in range(lane, scale.shape[0], wp.block_dim()):
        scale[col] *= factor
        bias[col] *= factor


@wp.kernel(enable_backward=False)
def _normalize_scale_tile_kernel(scale: wp.array[wp.float32], eps: wp.float32):
    lane = wp.tid()
    squared = wp.float32(0.0)
    for col in range(lane, scale.shape[0], wp.block_dim()):
        squared += scale[col] * scale[col]
    total = wp.tile_sum(wp.tile(squared))[0]
    factor = wp.sqrt(wp.float32(scale.shape[0])) / wp.sqrt(total + eps)
    for col in range(lane, scale.shape[0], wp.block_dim()):
        scale[col] *= factor


def _launch_rms_inv(x: wp.array2d[Any], eps: float, out: wp.array[wp.float32]) -> None:
    if x.device.is_cuda:
        wp.launch(
            _rms_inv_tile_kernel,
            dim=(x.shape[0], _TILE_REDUCTION_BLOCK_DIM),
            inputs=[x, eps],
            outputs=[out],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=x.device,
        )
    else:
        wp.launch(_rms_inv_kernel, dim=x.shape[0], inputs=[x, eps], outputs=[out], device=x.device)


def _launch_bias_grad(output_grad: wp.array2d[Any], bias_grad: wp.array[wp.float32]) -> None:
    if output_grad.device.is_cuda:
        wp.launch(
            _bias_grad_tile_kernel,
            dim=(bias_grad.shape[0], _TILE_REDUCTION_BLOCK_DIM),
            inputs=[output_grad],
            outputs=[bias_grad],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=output_grad.device,
        )
    else:
        wp.launch(
            _bias_grad_kernel, dim=bias_grad.shape, inputs=[output_grad], outputs=[bias_grad], device=output_grad.device
        )


def _launch_rms_backward_stats(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    scale: wp.array[wp.float32],
    eps: float,
    inv_rms: wp.array[wp.float32],
    projection: wp.array[wp.float32],
) -> None:
    if x.device.is_cuda:
        wp.launch(
            _rms_norm_backward_stats_tile_kernel,
            dim=(x.shape[0], _TILE_REDUCTION_BLOCK_DIM),
            inputs=[x, output_grad, scale, eps],
            outputs=[inv_rms, projection],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=x.device,
        )
    else:
        wp.launch(
            _rms_norm_backward_stats_kernel,
            dim=x.shape[0],
            inputs=[x, output_grad, scale, eps],
            outputs=[inv_rms, projection],
            device=x.device,
        )


def _launch_rms_scale_grad(
    x: wp.array2d[Any],
    output_grad: wp.array2d[Any],
    inv_rms: wp.array[wp.float32],
    scale_grad: wp.array[wp.float32],
) -> None:
    if x.device.is_cuda:
        wp.launch(
            _rms_norm_scale_grad_tile_kernel,
            dim=(scale_grad.shape[0], _TILE_REDUCTION_BLOCK_DIM),
            inputs=[x, output_grad, inv_rms],
            outputs=[scale_grad],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=x.device,
        )
    else:
        wp.launch(
            _rms_norm_scale_grad_kernel,
            dim=scale_grad.shape,
            inputs=[x, output_grad, inv_rms],
            outputs=[scale_grad],
            device=x.device,
        )


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


class _BatchNormScratch:
    def __init__(self, rows: int, width: int, input_dtype: Any, device: wp.context.Device):
        self.transposed = wp.empty((width, rows), dtype=input_dtype, device=device)
        self.transposed_grad = wp.empty((width, rows), dtype=wp.float32, device=device)
        self.mean = wp.empty(width, dtype=wp.float32, device=device)
        self.variance = wp.empty(width, dtype=wp.float32, device=device)
        self.inv_std = wp.empty(width, dtype=wp.float32, device=device)
        self.mean_grad = wp.empty(width, dtype=wp.float32, device=device)
        self.variance_grad = wp.empty(width, dtype=wp.float32, device=device)


class _PopulationBatchNormScratch:
    def __init__(self, count: int, rows: int, width: int, input_dtype: Any, device: wp.context.Device):
        self.count = int(count)
        self.rows = int(rows)
        self.width = int(width)
        self.padded_width = ((self.width + 31) // 32) * 32
        self.transposed = wp.empty((self.count * self.padded_width, self.rows), dtype=input_dtype, device=device)
        self.transposed_grad = wp.empty((self.count * self.padded_width, self.rows), dtype=wp.float32, device=device)
        self.mean = wp.empty((self.count, self.width), dtype=wp.float32, device=device)
        self.variance = wp.empty((self.count, self.width), dtype=wp.float32, device=device)
        self.inv_std = wp.empty((self.count, self.width), dtype=wp.float32, device=device)
        self.mean_grad = wp.empty((self.count, self.width), dtype=wp.float32, device=device)
        self.variance_grad = wp.empty((self.count, self.width), dtype=wp.float32, device=device)

    def bind(self, norms: tuple[_UnitBatchNorm, ...]) -> None:
        for member, norm in enumerate(norms):
            scratch = object.__new__(_BatchNormScratch)
            start = member * self.padded_width
            scratch.transposed = self.transposed[start : start + self.width]
            scratch.transposed_grad = self.transposed_grad[start : start + self.width]
            scratch.mean = self.mean[member]
            scratch.variance = self.variance[member]
            scratch.inv_std = self.inv_std[member]
            scratch.mean_grad = self.mean_grad[member]
            scratch.variance_grad = self.variance_grad[member]
            norm._scratch[(self.rows, scratch.transposed.dtype)] = scratch

    def arrays(self) -> tuple[wp.array[Any], ...]:
        return (
            self.transposed,
            self.transposed_grad,
            self.mean,
            self.variance,
            self.inv_std,
            self.mean_grad,
            self.variance_grad,
        )


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
        self.last_mean: wp.array[wp.float32] | None = None
        self.last_variance: wp.array[wp.float32] | None = None
        self.last_scratch: _BatchNormScratch | None = None
        self._scratch: dict[tuple[int, Any], _BatchNormScratch] = {}

    def scratch(self, rows: int, input_dtype: Any = wp.float32) -> _BatchNormScratch:
        key = (int(rows), input_dtype)
        value = self._scratch.get(key)
        if value is None:
            value = _BatchNormScratch(rows, self.width, input_dtype, self.device)
            self._scratch[key] = value
        return value

    def forward(
        self,
        x: wp.array2d[Any],
        *,
        training: bool,
        requires_grad: bool,
        relu: bool = False,
        output_dtype: Any = wp.float32,
    ) -> wp.array2d[Any]:
        rows = int(x.shape[0])
        if training:
            scratch = self.scratch(rows, x.dtype)
            mean = scratch.mean
            variance = scratch.variance
            if self.device.is_cuda:
                wp.launch(
                    _transpose_2d_tile_kernel,
                    dim=((rows + 31) // 32, (self.width + 31) // 32, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[x],
                    outputs=[scratch.transposed],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
                wp.launch(
                    _batch_moments_transposed_tile_kernel,
                    dim=(self.width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[scratch.transposed, rows, self.eps],
                    outputs=[mean, variance, scratch.inv_std],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
            else:
                wp.launch(
                    _batch_moments_kernel,
                    dim=self.width,
                    inputs=[x, rows, self.eps],
                    outputs=[mean, variance, scratch.inv_std],
                    device=self.device,
                )
            self.last_scratch = scratch
            self.last_mean = mean
            self.last_variance = variance
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
        out = wp.empty(x.shape, dtype=output_dtype, device=self.device, requires_grad=requires_grad)
        if output_dtype == wp.float16:
            kernel = (
                _batch_norm_inv_std_relu_f16_kernel
                if training and relu
                else _batch_norm_inv_std_f16_kernel
                if training
                else _batch_norm_relu_f16_kernel
                if relu
                else _batch_norm_f16_kernel
            )
        else:
            kernel = (
                _batch_norm_inv_std_relu_kernel
                if training and relu
                else _batch_norm_inv_std_kernel
                if training
                else _batch_norm_relu_kernel
                if relu
                else _batch_norm_kernel
            )
        wp.launch(
            kernel,
            dim=x.shape,
            inputs=(
                [x, mean, scratch.inv_std, self.scale, self.bias]
                if training
                else [x, mean, variance, self.scale, self.bias, self.eps]
            ),
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
        contraction_dtype: str = "float32",
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.output_dim = int(output_dim)
        self.actor_heads = bool(actor_heads)
        self.device = wp.get_device(device)
        if contraction_dtype not in {"float32", "float16"}:
            raise ValueError("contraction_dtype must be float32 or float16")
        if contraction_dtype == "float16" and not self.device.is_cuda:
            raise ValueError("float16 contractions require a CUDA device")
        self.contraction_dtype = contraction_dtype
        self.activation_dtype = wp.float16 if contraction_dtype == "float16" else wp.float32
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
        self._fp16_weights: dict[int, wp.array2d[wp.float16]] = {}
        self._fp16_inputs: dict[tuple[int, tuple[int, ...]], wp.array2d[wp.float16]] = {}
        self.normalize_parameters()
        self.refresh_contraction_weights()

    def _weight(self, input_dim: int, output_dim: int, rng: np.random.Generator) -> wp.array2d[wp.float32]:
        return wp.array(
            _orthogonal(input_dim, output_dim, rng),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

    def refresh_contraction_weights(self) -> None:
        """Refresh setup-owned FP16 mirrors after FP32 parameter mutation."""

        if self.contraction_dtype != "float16":
            return
        current: dict[int, wp.array2d[wp.float16]] = {}
        for weight in self.weights:
            mirror = self._fp16_weights.get(int(weight.ptr))
            if mirror is None or mirror.shape != weight.shape:
                mirror = wp.empty(weight.shape, dtype=wp.float16, device=self.device)
            wp.launch(
                cast_2d_float_to_float16_kernel,
                dim=weight.shape,
                inputs=[weight],
                outputs=[mirror],
                device=self.device,
            )
            current[int(weight.ptr)] = mirror
        self._fp16_weights = current

    def _contraction_weight(self, weight: wp.array2d[wp.float32]) -> wp.array2d[wp.float16]:
        mirror = self._fp16_weights.get(int(weight.ptr))
        if mirror is None:
            raise RuntimeError("FP16 contraction weights have not been refreshed")
        return mirror

    def _contraction_input(self, value: wp.array2d[Any]) -> wp.array2d[wp.float16]:
        if value.dtype == wp.float16:
            return value
        key = (int(value.ptr), tuple(int(size) for size in value.shape))
        mirror = self._fp16_inputs.get(key)
        if mirror is None:
            mirror = wp.empty(value.shape, dtype=wp.float16, device=self.device)
            self._fp16_inputs[key] = mirror
        wp.launch(
            cast_2d_float_to_float16_kernel,
            dim=value.shape,
            inputs=[value],
            outputs=[mirror],
            device=self.device,
        )
        return mirror

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

    def _linear(self, x: wp.array2d[Any], weight: wp.array2d[wp.float32], *, requires_grad: bool) -> wp.array2d[Any]:
        out = wp.empty(
            (int(x.shape[0]), int(weight.shape[1])),
            dtype=self.activation_dtype,
            device=self.device,
            requires_grad=requires_grad,
        )
        self._linear_into(x, weight, out)
        return out

    def _linear_into(
        self,
        x: wp.array2d[Any],
        weight: wp.array2d[wp.float32],
        out: wp.array2d[Any],
        *,
        use_amp: bool = True,
    ) -> None:
        if self._use_cublas:
            if self.contraction_dtype == "float16" and use_amp:
                contraction_input = self._contraction_input(x)
                gemm_float16_output(
                    contraction_input,
                    self._contraction_weight(weight),
                    out,
                    int(x.shape[0]),
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                )
            else:
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

    def reserve_training_buffers(self, batch_size: int) -> None:
        """Reserve persistent BatchNorm reduction workspaces for training."""

        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        self.input_norm.scratch(rows, wp.float32)
        for first, second in self.block_norms:
            first.scratch(rows, self.activation_dtype)
            second.scratch(rows, self.activation_dtype)

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
        self._linear_into(input_normalized, self.embed_weight, hidden, use_amp=False)
        for block_index, ((w1, w2), (norm1, norm2)) in enumerate(
            zip(self.block_weights, self.block_norms, strict=True)
        ):
            residual = hidden
            pre1, _normed1, activated1, pre2, _normed2, activated2, block_out = blocks[block_index]
            self._linear_into(residual, w1, pre1, use_amp=False)
            wp.launch(
                _batch_norm_relu_kernel,
                dim=pre1.shape,
                inputs=[pre1, norm1.running_mean, norm1.running_variance, norm1.scale, norm1.bias, norm1.eps],
                outputs=[activated1],
                device=self.device,
            )
            self._linear_into(activated1, w2, pre2, use_amp=False)
            wp.launch(
                _batch_norm_relu_kernel,
                dim=pre2.shape,
                inputs=[pre2, norm2.running_mean, norm2.running_variance, norm2.scale, norm2.bias, norm2.eps],
                outputs=[activated2],
                device=self.device,
            )
            wp.launch(
                _residual_add_kernel,
                dim=block_out.shape,
                inputs=[activated2, residual],
                outputs=[block_out],
                device=self.device,
            )
            hidden = block_out
        _launch_rms_inv(hidden, 1.0e-6, inv_rms)
        wp.launch(
            _rms_norm_kernel,
            dim=hidden.shape,
            inputs=[hidden, self.rms_scale, inv_rms],
            outputs=[normalized],
            device=self.device,
        )
        offset = 0
        for head_index, (head, weight, bias) in enumerate(zip(heads, self.head_weights, self.head_biases, strict=True)):
            self._linear_into(normalized, weight, head, use_amp=False)
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
        hidden = self.input_norm.forward(x, training=training, requires_grad=requires_grad, output_dtype=wp.float32)
        hidden = self._linear(hidden, self.embed_weight, requires_grad=requires_grad)
        for (w1, w2), (norm1, norm2) in zip(self.block_weights, self.block_norms, strict=True):
            residual = hidden
            hidden = self._linear(hidden, w1, requires_grad=requires_grad)
            activated = norm1.forward(
                hidden,
                training=training,
                requires_grad=requires_grad,
                relu=True,
                output_dtype=self.activation_dtype,
            )
            hidden = self._linear(activated, w2, requires_grad=requires_grad)
            activated = norm2.forward(
                hidden,
                training=training,
                requires_grad=requires_grad,
                relu=True,
                output_dtype=self.activation_dtype,
            )
            hidden = wp.empty_like(activated, requires_grad=requires_grad)
            wp.launch(
                _residual_add_f16_kernel if self.contraction_dtype == "float16" else _residual_add_kernel,
                dim=hidden.shape,
                inputs=[activated, residual],
                outputs=[hidden],
                device=self.device,
            )
        normalized = wp.empty_like(hidden, requires_grad=requires_grad)
        inv_rms = wp.empty(hidden.shape[0], dtype=wp.float32, device=self.device)
        _launch_rms_inv(hidden, 1.0e-6, inv_rms)
        wp.launch(
            _rms_norm_f16_kernel if self.contraction_dtype == "float16" else _rms_norm_kernel,
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
                    _head_bias_log_std_amp_kernel if self.contraction_dtype == "float16" else _head_bias_log_std_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset, self.log_std_min, self.log_std_max],
                    outputs=[out],
                    device=self.device,
                )
            else:
                wp.launch(
                    _head_bias_amp_kernel if self.contraction_dtype == "float16" else _head_bias_kernel,
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
        x: wp.array2d[Any],
        weight: wp.array2d[wp.float32],
        output_grad: wp.array2d[Any],
    ) -> wp.array2d[Any]:
        input_grad = wp.empty(x.shape, dtype=x.dtype, device=self.device)
        if self._use_cublas:
            if self.contraction_dtype == "float16":
                mirror = self._contraction_weight(weight)
                contraction_output_grad = self._contraction_input(output_grad)
                input_gemm = gemm_float16_output if x.dtype == wp.float16 else gemm_float16
                input_gemm(
                    contraction_output_grad,
                    mirror,
                    input_grad,
                    int(x.shape[0]),
                    int(weight.shape[0]),
                    int(weight.shape[1]),
                    transpose_rhs=True,
                )
                contraction_input = self._contraction_input(x)
                gemm_float16(
                    contraction_input,
                    contraction_output_grad,
                    weight.grad,
                    int(weight.shape[0]),
                    int(weight.shape[1]),
                    int(x.shape[0]),
                    transpose_lhs=True,
                )
            else:
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
        amp_staged: bool = False,
        input_grad: wp.array2d[wp.float32] | None = None,
    ) -> wp.array2d[wp.float32]:
        if input_grad is None:
            input_grad = wp.empty(x.shape, dtype=x.dtype, device=self.device)
        if training and amp_staged and self.device.is_cuda and norm.last_scratch is not None:
            scratch = norm.last_scratch
            wp.launch(
                _transpose_2d_tile_kernel,
                dim=((x.shape[0] + 31) // 32, (norm.width + 31) // 32, _TILE_REDUCTION_BLOCK_DIM),
                inputs=[output_grad],
                outputs=[scratch.transposed_grad],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                _batch_norm_backward_amp_transposed_tile_kernel,
                dim=(norm.width, _TILE_REDUCTION_BLOCK_DIM),
                inputs=[scratch.transposed, scratch.transposed_grad, norm.last_mean, scratch.inv_std, norm.scale],
                outputs=[scratch.mean_grad, scratch.variance_grad, norm.scale.grad, norm.bias.grad],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                _batch_norm_input_grad_amp_kernel,
                dim=x.shape,
                inputs=[
                    x,
                    output_grad,
                    norm.scale,
                    norm.last_mean,
                    scratch.inv_std,
                    scratch.mean_grad,
                    scratch.variance_grad,
                ],
                outputs=[input_grad],
                device=self.device,
            )
            return input_grad
        if training:
            if self.device.is_cuda and norm.last_scratch is not None:
                mean = norm.last_mean
                scratch = norm.last_scratch
                inv_std = scratch.inv_std
                sum_grad = norm.bias.grad
                sum_grad_normalized = norm.scale.grad
                wp.launch(
                    _batch_norm_backward_tile_kernel,
                    dim=(norm.width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[x, output_grad, mean, inv_std],
                    outputs=[sum_grad, sum_grad_normalized, norm.scale.grad, norm.bias.grad],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
            else:
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
                _batch_norm_input_grad_f16_kernel if x.dtype == wp.float16 else _batch_norm_input_grad_kernel,
                dim=x.shape,
                inputs=[x, output_grad, norm.scale, mean, inv_std, sum_grad, sum_grad_normalized],
                outputs=[input_grad],
                device=self.device,
            )
        else:
            wp.launch(
                _batch_norm_inference_input_grad_f16_kernel
                if x.dtype == wp.float16
                else _batch_norm_inference_input_grad_kernel,
                dim=x.shape,
                inputs=[output_grad, norm.scale, norm.running_variance, norm.eps],
                outputs=[input_grad],
                device=self.device,
            )
            if self.device.is_cuda:
                wp.launch(
                    _batch_norm_inference_parameter_grad_tile_kernel,
                    dim=(norm.width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[x, output_grad, norm.running_mean, norm.running_variance, norm.eps],
                    outputs=[norm.scale.grad, norm.bias.grad],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
            else:
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
        input_normalized = self.input_norm.forward(
            manual_input, training=training, requires_grad=False, output_dtype=wp.float32
        )
        hidden = self._linear(input_normalized, self.embed_weight, requires_grad=False)
        block_cache: list[tuple[wp.array2d[wp.float32], ...]] = []
        for (w1, w2), (norm1, norm2) in zip(self.block_weights, self.block_norms, strict=True):
            residual = hidden
            pre1 = self._linear(residual, w1, requires_grad=False)
            activated1 = norm1.forward(
                pre1, training=training, requires_grad=False, relu=True, output_dtype=self.activation_dtype
            )
            normed1 = activated1
            pre2 = self._linear(activated1, w2, requires_grad=False)
            activated2 = norm2.forward(
                pre2, training=training, requires_grad=False, relu=True, output_dtype=self.activation_dtype
            )
            normed2 = activated2
            hidden = wp.empty_like(activated2)
            wp.launch(
                _residual_add_f16_kernel if self.contraction_dtype == "float16" else _residual_add_kernel,
                dim=hidden.shape,
                inputs=[activated2, residual],
                outputs=[hidden],
                device=self.device,
            )
            block_cache.append((residual, pre1, normed1, activated1, pre2, normed2, activated2))
        rms_input = hidden
        normalized = wp.empty_like(hidden)
        inv_rms = wp.empty(hidden.shape[0], dtype=wp.float32, device=self.device)
        _launch_rms_inv(hidden, 1.0e-6, inv_rms)
        wp.launch(
            _rms_norm_f16_kernel if self.contraction_dtype == "float16" else _rms_norm_kernel,
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
                    _head_bias_log_std_amp_kernel if self.contraction_dtype == "float16" else _head_bias_log_std_kernel,
                    dim=head.shape,
                    inputs=[head, bias, offset, self.log_std_min, self.log_std_max],
                    outputs=[output],
                    device=self.device,
                )
            else:
                wp.launch(
                    _head_bias_amp_kernel if self.contraction_dtype == "float16" else _head_bias_kernel,
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
        loss_scale: wp.array[wp.float32] | None = None,
        found_inf: wp.array[wp.int32] | None = None,
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

        normalized_grad = wp.zeros(normalized.shape, dtype=self.activation_dtype, device=self.device)
        offset = 0
        for head_index, (head, weight, bias) in enumerate(zip(heads, self.head_weights, self.head_biases, strict=True)):
            head_grad = wp.empty(head.shape, dtype=self.activation_dtype, device=self.device)
            wp.launch(
                _head_output_grad_f16_kernel if self.contraction_dtype == "float16" else _head_output_grad_kernel,
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
            _launch_bias_grad(head_grad, bias.grad)
            head_input_grad = self._linear_backward(normalized, weight, head_grad)
            accumulated = wp.empty_like(normalized_grad)
            wp.launch(
                _add_f16_kernel if self.contraction_dtype == "float16" else _add_kernel,
                dim=normalized_grad.shape,
                inputs=[normalized_grad, head_input_grad],
                outputs=[accumulated],
                device=self.device,
            )
            normalized_grad = accumulated
            offset += int(head.shape[1])

        hidden_grad = wp.empty(rms_input.shape, dtype=self.activation_dtype, device=self.device)
        inv_rms = wp.empty(rms_input.shape[0], dtype=wp.float32, device=self.device)
        projection = wp.empty(rms_input.shape[0], dtype=wp.float32, device=self.device)
        _launch_rms_backward_stats(rms_input, normalized_grad, self.rms_scale, 1.0e-6, inv_rms, projection)
        wp.launch(
            _rms_norm_input_grad_f16_kernel if self.contraction_dtype == "float16" else _rms_norm_input_grad_kernel,
            dim=rms_input.shape,
            inputs=[rms_input, normalized_grad, self.rms_scale, inv_rms, projection],
            outputs=[hidden_grad],
            device=self.device,
        )
        _launch_rms_scale_grad(rms_input, normalized_grad, inv_rms, self.rms_scale.grad)

        for block_index in reversed(range(self.num_blocks)):
            residual, pre1, _normed1, activated1, pre2, _normed2, activated2 = blocks[block_index]
            norm1, norm2 = self.block_norms[block_index]
            w1, w2 = self.block_weights[block_index]
            norm2_grad = wp.empty_like(activated2)
            wp.launch(
                _relu_grad_f16_kernel if self.contraction_dtype == "float16" else _relu_grad_kernel,
                dim=activated2.shape,
                inputs=[activated2, hidden_grad],
                outputs=[norm2_grad],
                device=self.device,
            )
            pre2_grad = self._batch_norm_backward(norm2, pre2, norm2_grad, training=training)
            activated1_grad = self._linear_backward(activated1, w2, pre2_grad)
            norm1_grad = wp.empty_like(activated1)
            wp.launch(
                _relu_grad_f16_kernel if self.contraction_dtype == "float16" else _relu_grad_kernel,
                dim=activated1.shape,
                inputs=[activated1, activated1_grad],
                outputs=[norm1_grad],
                device=self.device,
            )
            pre1_grad = self._batch_norm_backward(norm1, pre1, norm1_grad, training=training)
            branch_grad = self._linear_backward(residual, w1, pre1_grad)
            combined = wp.empty_like(hidden_grad)
            wp.launch(
                _add_f16_kernel if self.contraction_dtype == "float16" else _add_kernel,
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
        _unscale_parameter_grads(self.parameters(), loss_scale, found_inf)
        self._manual_cache = None

    def normalize_parameters(self, eps: float = 1.0e-8) -> None:
        for weight in self.weights:
            if self.device.is_cuda:
                wp.launch(
                    unit_normalize_weight_columns_tile_kernel,
                    dim=(weight.shape[1], _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[weight, eps],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
            else:
                wp.launch(_normalize_columns_kernel, dim=weight.shape[1], inputs=[weight, eps], device=self.device)
        norms = [self.input_norm]
        for norm1, norm2 in self.block_norms:
            norms.extend((norm1, norm2))
        for norm in norms:
            if self.device.is_cuda:
                wp.launch(
                    _normalize_batch_affine_tile_kernel,
                    dim=_TILE_REDUCTION_BLOCK_DIM,
                    inputs=[norm.scale, norm.bias, eps],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=self.device,
                )
            else:
                wp.launch(
                    _normalize_batch_affine_kernel,
                    dim=1,
                    inputs=[norm.scale, norm.bias, eps],
                    device=self.device,
                )
        if self.device.is_cuda:
            wp.launch(
                _normalize_scale_tile_kernel,
                dim=_TILE_REDUCTION_BLOCK_DIM,
                inputs=[self.rms_scale, eps],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
        else:
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


class EnsembleNetworkFlashSAC:
    """Fuse dense contractions across compatible reference networks."""

    def __init__(self, first: NetworkFlashSAC, second: NetworkFlashSAC | None = None, *additional: NetworkFlashSAC):
        self.networks = (first, *(() if second is None else (second,)), *additional)
        if any(
            network.device != first.device
            or network.layer_sizes != first.layer_sizes
            or network.actor_heads != first.actor_heads
            or network.contraction_dtype != first.contraction_dtype
            for network in self.networks
        ):
            raise ValueError("FlashSAC critic structures do not match")
        self.ensemble_count = len(self.networks)
        self.actor_heads = first.actor_heads
        self.device = first.device
        self.input_dim = first.input_dim
        self.hidden_dim = first.hidden_dim
        self.num_blocks = first.num_blocks
        self.output_dim = first.output_dim
        self.contraction_dtype = first.contraction_dtype
        self.activation_dtype = first.activation_dtype
        self.embed_weight = self._stack_weights(*(network.embed_weight for network in self.networks))
        for ensemble_index, network in enumerate(self.networks):
            network.embed_weight = self.embed_weight[ensemble_index]
        self.block_weights: list[tuple[wp.array3d[wp.float32], wp.array3d[wp.float32]]] = []
        for block_index in range(self.num_blocks):
            w1 = self._stack_weights(*(network.block_weights[block_index][0] for network in self.networks))
            w2 = self._stack_weights(*(network.block_weights[block_index][1] for network in self.networks))
            for ensemble_index, network in enumerate(self.networks):
                network.block_weights[block_index] = (w1[ensemble_index], w2[ensemble_index])
            self.block_weights.append((w1, w2))
        self.head_weights: list[wp.array3d[wp.float32]] = []
        for head_index in range(len(first.head_weights)):
            weight = self._stack_weights(*(network.head_weights[head_index] for network in self.networks))
            for ensemble_index, network in enumerate(self.networks):
                network.head_weights[head_index] = weight[ensemble_index]
            self.head_weights.append(weight)
        self.head_biases: list[wp.array2d[wp.float32]] = []
        for head_index in range(len(first.head_biases)):
            bias = self._stack_vectors(*(network.head_biases[head_index] for network in self.networks))
            for ensemble_index, network in enumerate(self.networks):
                network.head_biases[head_index] = bias[ensemble_index]
            self.head_biases.append(bias)
        self.rms_scale = self._stack_vectors(*(network.rms_scale for network in self.networks))
        for ensemble_index, network in enumerate(self.networks):
            network.rms_scale = self.rms_scale[ensemble_index]
            network.biases = network.head_biases
        for network in self.networks:
            network.weights = [network.embed_weight]
            for w1, w2 in network.block_weights:
                network.weights.extend((w1, w2))
            network.weights.extend(network.head_weights)
        self.weights = [self.embed_weight]
        for w1, w2 in self.block_weights:
            self.weights.extend((w1, w2))
        self.weights.extend(self.head_weights)
        self._fp16_weights: dict[int, wp.array3d[wp.float16]] = {}
        self._fp16_inputs_2d: dict[tuple[int, tuple[int, ...]], wp.array2d[wp.float16]] = {}
        self._fp16_inputs_3d: dict[tuple[int, tuple[int, ...]], wp.array3d[wp.float16]] = {}
        self._workspace_rows = 0
        self._workspace: dict[str, object] = {}
        self._population_norm_states: dict[int, dict[str, wp.array[Any]]] = {}
        self._population_norm_scratch: dict[int, _PopulationBatchNormScratch] = {}
        self._population_flat_views: dict[int, wp.array2d[Any]] = {}
        self._stack_norm_group(tuple(network.input_norm for network in self.networks))
        for block_index in range(self.num_blocks):
            self._stack_norm_group(tuple(network.block_norms[block_index][0] for network in self.networks))
            self._stack_norm_group(tuple(network.block_norms[block_index][1] for network in self.networks))
        self._population_parameters = [*self.weights, *self.head_biases]
        for state in self._population_norm_states.values():
            self._population_parameters.extend((state["scale"], state["bias"]))
        self._population_parameters.append(self.rms_scale)
        self._population_state_arrays = [*self.weights, *self.head_biases]
        for state in self._population_norm_states.values():
            self._population_state_arrays.extend(state.values())
        self._population_state_arrays.append(self.rms_scale)

    def _stack_norm_group(self, norms: tuple[_UnitBatchNorm, ...]) -> None:
        state: dict[str, wp.array[Any]] = {}
        for name, requires_grad in (
            ("scale", True),
            ("bias", True),
            ("running_mean", False),
            ("running_variance", False),
        ):
            stacked = wp.array(
                np.stack(tuple(getattr(norm, name).numpy() for norm in norms)),
                dtype=wp.float32,
                device=self.device,
                requires_grad=requires_grad,
            )
            for member, norm in enumerate(norms):
                setattr(norm, name, stacked[member])
            state[name] = stacked
        self._population_norm_states[id(norms[0])] = state

    def reserve_buffers(self, batch_size: int) -> None:
        """Reserve fixed-address population forward and reduction buffers."""

        rows = int(batch_size)
        if rows <= 0:
            raise ValueError("batch_size must be positive")
        if rows == self._workspace_rows:
            return
        count = self.ensemble_count
        blocks: list[tuple[wp.array3d[Any], ...]] = []
        for _ in range(self.num_blocks):
            blocks.append(
                (
                    wp.empty((count, rows, self.hidden_dim * 4), dtype=self.activation_dtype, device=self.device),
                    wp.empty((count, rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=self.activation_dtype, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                )
            )
        head_width = self.output_dim // 2 if self.actor_heads else self.output_dim
        backward_blocks: list[tuple[wp.array3d[Any], ...]] = []
        for block_index in range(self.num_blocks):
            residual_dtype = self.activation_dtype if block_index == 0 else wp.float32
            backward_blocks.append(
                (
                    wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=self.activation_dtype, device=self.device),
                    wp.empty((count, rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((count, rows, self.hidden_dim * 4), dtype=wp.float32, device=self.device),
                    wp.empty((count, rows, self.hidden_dim * 4), dtype=self.activation_dtype, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=residual_dtype, device=self.device),
                    wp.empty((count, rows, self.hidden_dim), dtype=residual_dtype, device=self.device),
                )
            )
        backward = {
            "output_grad": wp.empty((count, rows, self.output_dim), dtype=wp.float32, device=self.device),
            "normalized_grad": wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "head_grads": [
                wp.empty((count, rows, head_width), dtype=self.activation_dtype, device=self.device)
                for _ in self.head_weights
            ],
            "head_input_grads": [
                wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device)
                for _ in self.head_weights
            ],
            "normalized_accum": wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "rms_grad": wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "rms_inv": wp.empty((count, rows), dtype=wp.float32, device=self.device),
            "rms_projection": wp.empty((count, rows), dtype=wp.float32, device=self.device),
            "blocks": backward_blocks,
            "input_normalized_grad": wp.empty((count, rows, self.input_dim), dtype=wp.float32, device=self.device),
            "input_grad": wp.empty((count, rows, self.input_dim), dtype=wp.float32, device=self.device),
        }
        self._workspace = {
            "input_normalized": wp.empty((count, rows, self.input_dim), dtype=wp.float32, device=self.device),
            "embed": wp.empty((count, rows, self.hidden_dim), dtype=self.activation_dtype, device=self.device),
            "blocks": blocks,
            "inv_rms": wp.empty((count, rows), dtype=wp.float32, device=self.device),
            "normalized": wp.empty((count, rows, self.hidden_dim), dtype=wp.float32, device=self.device),
            "heads": [
                wp.empty((count, rows, head_width), dtype=self.activation_dtype, device=self.device)
                for _ in self.head_weights
            ],
            "output": wp.empty((count, rows, self.output_dim), dtype=wp.float32, device=self.device),
            "backward": backward,
        }
        population_scratch: list[_PopulationBatchNormScratch] = []
        for network in self.networks:
            network.input_norm.scratch(rows, wp.float32)
        for block_index in range(self.num_blocks):
            for norm_index in range(2):
                norms = tuple(network.block_norms[block_index][norm_index] for network in self.networks)
                scratch = _PopulationBatchNormScratch(count, rows, norms[0].width, self.activation_dtype, self.device)
                scratch.bind(norms)
                self._population_norm_scratch[id(norms[0])] = scratch
                population_scratch.append(scratch)
        flat_values: list[wp.array3d[Any]] = []
        for pre1, _activated1, pre2, _activated2, hidden in blocks:
            flat_values.extend((pre1, pre2, hidden))
        for (
            norm2_grad,
            _pre2_grad,
            _activated1_grad,
            norm1_grad,
            _pre1_grad,
            _branch_grad,
            _combined,
        ) in backward_blocks:
            flat_values.extend((norm1_grad, norm2_grad))
        self._population_flat_views = {
            int(value.ptr): value.reshape((count * rows, int(value.shape[2]))) for value in flat_values
        }
        self._workspace["population_norm_scratch"] = [scratch.arrays() for scratch in population_scratch]
        if self.contraction_dtype == "float16":
            mirror_sources = [self._workspace["input_normalized"], self._workspace["normalized"]]
            for _pre1, activated1, _pre2, _activated2, hidden in blocks:
                mirror_sources.extend((activated1, hidden))
            for source in mirror_sources:
                self._contraction_input(source, refresh=False)
        self._workspace_rows = rows
        self.refresh_contraction_weights()

    def _stack_vectors(self, *values: wp.array[wp.float32]) -> wp.array2d[wp.float32]:
        return wp.array(
            np.stack(tuple(value.numpy() for value in values)),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

    def population_parameters(self) -> list[wp.array]:
        """Return setup-owned trainable arrays with a leading population dimension."""

        return list(self._population_parameters)

    def population_state_arrays(self) -> list[wp.array]:
        """Return complete setup-owned network state with a leading population dimension."""

        return list(self._population_state_arrays)

    def copy_population_member(
        self,
        source_index: wp.array[wp.int32],
        destination_index: wp.array[wp.int32],
    ) -> None:
        """Copy one complete network member using device-resident indices."""

        for value in self._population_state_arrays:
            if value.ndim == 2:
                wp.launch(
                    population_copy_float_2d_kernel,
                    dim=value.shape[1],
                    inputs=[value, source_index, destination_index],
                    device=self.device,
                )
            else:
                wp.launch(
                    population_copy_float_3d_kernel,
                    dim=value.shape[1:],
                    inputs=[value, source_index, destination_index],
                    device=self.device,
                )
        for value in self._fp16_weights.values():
            wp.launch(
                population_copy_float16_3d_kernel,
                dim=value.shape[1:],
                inputs=[value, source_index, destination_index],
                device=self.device,
            )

    def _stack_weights(self, *weights: wp.array2d[wp.float32]) -> wp.array3d[wp.float32]:
        return wp.array(
            np.stack(tuple(weight.numpy() for weight in weights)),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

    def refresh_contraction_weights(self) -> None:
        """Refresh persistent fused FP16 mirrors after master-weight mutation."""

        if self.contraction_dtype != "float16":
            return
        current: dict[int, wp.array3d[wp.float16]] = {}
        for weight in self.weights:
            mirror = self._fp16_weights.get(int(weight.ptr))
            if mirror is None or mirror.shape != weight.shape:
                mirror = wp.empty(weight.shape, dtype=wp.float16, device=self.device)
            wp.launch(
                cast_3d_float_to_float16_kernel,
                dim=weight.shape,
                inputs=[weight],
                outputs=[mirror],
                device=self.device,
            )
            current[int(weight.ptr)] = mirror
        self._fp16_weights = current

    def _contraction_weight(self, weight: wp.array3d[wp.float32]) -> wp.array3d[wp.float16]:
        mirror = self._fp16_weights.get(int(weight.ptr))
        if mirror is None:
            raise RuntimeError("Fused FP16 contraction weights have not been refreshed")
        return mirror

    def _contraction_input(
        self,
        value: wp.array2d[Any] | wp.array3d[Any],
        *,
        refresh: bool = True,
    ) -> wp.array2d[wp.float16] | wp.array3d[wp.float16]:
        if value.dtype == wp.float16:
            return value
        key = (int(value.ptr), tuple(int(size) for size in value.shape))
        if value.ndim == 2:
            mirror = self._fp16_inputs_2d.get(key)
            if mirror is None:
                mirror = wp.empty(value.shape, dtype=wp.float16, device=self.device)
                self._fp16_inputs_2d[key] = mirror
            kernel = cast_2d_float_to_float16_kernel
        else:
            mirror = self._fp16_inputs_3d.get(key)
            if mirror is None:
                mirror = wp.empty(value.shape, dtype=wp.float16, device=self.device)
                self._fp16_inputs_3d[key] = mirror
            kernel = cast_3d_float_to_float16_kernel
        if refresh:
            wp.launch(kernel, dim=value.shape, inputs=[value], outputs=[mirror], device=self.device)
        return mirror

    def _linear(
        self,
        x: wp.array2d[Any] | wp.array3d[Any],
        weight: wp.array3d[wp.float32],
        *,
        broadcast_input: bool = False,
        contraction_input: wp.array2d[wp.float16] | wp.array3d[wp.float16] | None = None,
        out: wp.array3d[Any] | None = None,
    ) -> wp.array3d[Any]:
        rows = int(x.shape[-2])
        inner = int(weight.shape[1])
        cols = int(weight.shape[2])
        if out is None:
            out = wp.empty((self.ensemble_count, rows, cols), dtype=self.activation_dtype, device=self.device)
        if self.contraction_dtype == "float16":
            x = contraction_input if contraction_input is not None else self._contraction_input(x)
            gemm_float16_strided_batched_output(
                x,
                self._contraction_weight(weight),
                out,
                rows,
                cols,
                inner,
                self.ensemble_count,
                broadcast_lhs=broadcast_input,
            )
        else:
            gemm_float32_strided_batched(
                x, weight, out, rows, cols, inner, self.ensemble_count, broadcast_lhs=broadcast_input
            )
        return out

    def _norm_into(
        self,
        x: wp.array2d[wp.float32] | wp.array3d[wp.float32],
        out: wp.array3d[Any],
        norms: tuple[_UnitBatchNorm, ...],
        *,
        training: bool,
        broadcast_input: bool = False,
        relu: bool = False,
        mirror_output: bool = True,
    ) -> wp.array3d[wp.float16] | None:
        out_f16 = (
            self._contraction_input(out, refresh=False)
            if self.contraction_dtype == "float16" and mirror_output
            else None
        )
        population_scratch = (
            self._population_norm_scratch.get(id(norms[0]))
            if training and not broadcast_input and self.contraction_dtype == "float16" and self.device.is_cuda
            else None
        )
        if population_scratch is not None:
            rows = int(x.shape[1])
            width = int(x.shape[2])
            flat_source = self._population_flat_views[int(x.ptr)]
            wp.launch(
                _transpose_population_2d_tile_kernel,
                dim=(
                    self.ensemble_count * ((rows + 31) // 32),
                    (width + 31) // 32,
                    _TILE_REDUCTION_BLOCK_DIM,
                ),
                inputs=[flat_source, rows, width],
                outputs=[population_scratch.transposed],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
            wp.launch(
                _batch_moments_transposed_population_tile_kernel,
                dim=(self.ensemble_count * width, _TILE_REDUCTION_BLOCK_DIM),
                inputs=[population_scratch.transposed, rows, width, norms[0].eps],
                outputs=[population_scratch.mean, population_scratch.variance, population_scratch.inv_std],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
        for ensemble_index, norm in enumerate(norms):
            source = x if broadcast_input else x[ensemble_index]
            rows = int(source.shape[0])
            if training:
                scratch = norm.scratch(rows, source.dtype)
                mean = scratch.mean
                variance = scratch.variance
                if population_scratch is None:
                    if self.device.is_cuda:
                        wp.launch(
                            _transpose_2d_tile_kernel,
                            dim=((rows + 31) // 32, (norm.width + 31) // 32, _TILE_REDUCTION_BLOCK_DIM),
                            inputs=[source],
                            outputs=[scratch.transposed],
                            block_dim=_TILE_REDUCTION_BLOCK_DIM,
                            device=self.device,
                        )
                        wp.launch(
                            _batch_moments_transposed_tile_kernel,
                            dim=(norm.width, _TILE_REDUCTION_BLOCK_DIM),
                            inputs=[scratch.transposed, rows, norm.eps],
                            outputs=[mean, variance, scratch.inv_std],
                            block_dim=_TILE_REDUCTION_BLOCK_DIM,
                            device=self.device,
                        )
                    else:
                        wp.launch(
                            _batch_moments_kernel,
                            dim=norm.width,
                            inputs=[source, rows, norm.eps],
                            outputs=[mean, variance, scratch.inv_std],
                            device=self.device,
                        )
                if self.contraction_dtype == "float16":
                    wp.launch(
                        _round_batch_moments_f16_kernel,
                        dim=norm.width,
                        inputs=[mean, variance, norm.eps],
                        outputs=[scratch.inv_std],
                        device=self.device,
                    )
                norm.last_scratch = scratch
                norm.last_mean = mean
                norm.last_variance = variance
                wp.launch(
                    _update_running_moments_kernel,
                    dim=norm.width,
                    inputs=[mean, variance, rows, norm.momentum],
                    outputs=[norm.running_mean, norm.running_variance],
                    device=self.device,
                )
            else:
                mean = norm.running_mean
                variance = norm.running_variance
            if out_f16 is not None:
                kernel = _batch_norm_inv_std_amp_dual_kernel if training else _batch_norm_amp_dual_kernel
                wp.launch(
                    kernel,
                    dim=source.shape,
                    inputs=(
                        [source, mean, scratch.inv_std, norm.scale, norm.bias, relu]
                        if training
                        else [source, mean, variance, norm.scale, norm.bias, norm.eps, relu]
                    ),
                    outputs=[out[ensemble_index], out_f16[ensemble_index]],
                    device=self.device,
                )
            else:
                kernel = (
                    _batch_norm_inv_std_relu_kernel
                    if training and relu
                    else _batch_norm_inv_std_kernel
                    if training
                    else _batch_norm_relu_kernel
                    if relu
                    else _batch_norm_kernel
                )
                wp.launch(
                    kernel,
                    dim=source.shape,
                    inputs=(
                        [source, mean, scratch.inv_std, norm.scale, norm.bias]
                        if training
                        else [source, mean, variance, norm.scale, norm.bias, norm.eps]
                    ),
                    outputs=[out[ensemble_index]],
                    device=self.device,
                )
        return out_f16

    def _forward(
        self,
        x: wp.array2d[wp.float32] | wp.array3d[wp.float32],
        *,
        training: bool,
        retain_activations: bool,
    ) -> wp.array3d[wp.float32]:
        if x.ndim == 3 and int(x.shape[0]) != self.ensemble_count:
            raise ValueError("Population input leading dimension must match the ensemble")
        self.reserve_buffers(int(x.shape[-2]))
        workspace = self._workspace
        input_normalized = workspace["input_normalized"]
        if not isinstance(input_normalized, wp.array):
            raise RuntimeError("population workspace is invalid")
        input_normalized_f16 = self._norm_into(
            x,
            input_normalized,
            tuple(network.input_norm for network in self.networks),
            training=training,
            broadcast_input=x.ndim == 2,
        )
        hidden = self._linear(
            input_normalized, self.embed_weight, contraction_input=input_normalized_f16, out=workspace["embed"]
        )
        hidden_f16 = hidden if self.contraction_dtype == "float16" else None
        block_caches: list[list[tuple[wp.array2d[wp.float32], ...]]] = [[] for _ in self.networks]
        pair_block_cache: list[tuple[wp.array3d[wp.float32], ...]] = []
        for block_index, (w1, w2) in enumerate(self.block_weights):
            pre1_buffer, activated1, pre2_buffer, activated2, block_hidden = workspace["blocks"][block_index]
            residual = hidden
            residual_f16 = hidden_f16
            pre1 = self._linear(residual, w1, contraction_input=residual_f16, out=pre1_buffer)
            first_norms = tuple(network.block_norms[block_index][0] for network in self.networks)
            second_norms = tuple(network.block_norms[block_index][1] for network in self.networks)
            activated1_f16 = self._norm_into(pre1, activated1, first_norms, training=training, relu=True)
            normed1 = activated1
            pre2 = self._linear(activated1, w2, contraction_input=activated1_f16, out=pre2_buffer)
            self._norm_into(
                pre2,
                activated2,
                second_norms,
                training=training,
                relu=True,
                mirror_output=False,
            )
            normed2 = activated2
            hidden = block_hidden
            hidden_f16 = self._contraction_input(hidden, refresh=False) if self.contraction_dtype == "float16" else None
            for ensemble_index in range(self.ensemble_count):
                kernel = _residual_add_mixed_dual_kernel if hidden_f16 is not None else _residual_add_kernel
                wp.launch(
                    kernel,
                    dim=normed2[ensemble_index].shape,
                    inputs=[activated2[ensemble_index], residual[ensemble_index]],
                    outputs=(
                        [hidden[ensemble_index], hidden_f16[ensemble_index]]
                        if hidden_f16 is not None
                        else [hidden[ensemble_index]]
                    ),
                    device=self.device,
                )
                block_caches[ensemble_index].append(
                    (
                        residual[ensemble_index],
                        pre1[ensemble_index],
                        normed1[ensemble_index],
                        activated1[ensemble_index],
                        pre2[ensemble_index],
                        normed2[ensemble_index],
                        activated2[ensemble_index],
                    )
                )
            pair_block_cache.append((residual, pre1, normed1, activated1, pre2, normed2, activated2))
        normalized = workspace["normalized"]
        normalized_f16 = (
            self._contraction_input(normalized, refresh=False) if self.contraction_dtype == "float16" else None
        )
        for ensemble_index, network in enumerate(self.networks):
            inv_rms = workspace["inv_rms"][ensemble_index]
            _launch_rms_inv(hidden[ensemble_index], 1.0e-6, inv_rms)
            kernel = _rms_norm_dual_kernel if normalized_f16 is not None else _rms_norm_kernel
            wp.launch(
                kernel,
                dim=hidden[ensemble_index].shape,
                inputs=[hidden[ensemble_index], network.rms_scale, inv_rms],
                outputs=(
                    [normalized[ensemble_index], normalized_f16[ensemble_index]]
                    if normalized_f16 is not None
                    else [normalized[ensemble_index]]
                ),
                device=self.device,
            )
        heads = [
            self._linear(normalized, weight, contraction_input=normalized_f16, out=workspace["heads"][head_index])
            for head_index, weight in enumerate(self.head_weights)
        ]
        output = workspace["output"]
        for ensemble_index, network in enumerate(self.networks):
            offset = 0
            for head_index, head in enumerate(heads):
                if self.actor_heads and head_index == 1:
                    kernel = (
                        _head_bias_log_std_amp_kernel
                        if self.contraction_dtype == "float16"
                        else _head_bias_log_std_kernel
                    )
                    inputs = [
                        head[ensemble_index],
                        network.head_biases[head_index],
                        offset,
                        network.log_std_min,
                        network.log_std_max,
                    ]
                else:
                    kernel = _head_bias_amp_kernel if self.contraction_dtype == "float16" else _head_bias_kernel
                    inputs = [head[ensemble_index], network.head_biases[head_index], offset]
                wp.launch(
                    kernel,
                    dim=head[ensemble_index].shape,
                    inputs=inputs,
                    outputs=[output[ensemble_index]],
                    device=self.device,
                )
                offset += int(head.shape[2])
            if retain_activations:
                network._manual_input = x
                network._manual_cache = {
                    "input_normalized": input_normalized[ensemble_index],
                    "blocks": block_caches[ensemble_index],
                    "rms_input": hidden[ensemble_index],
                    "normalized": normalized[ensemble_index],
                    "heads": [head[ensemble_index] for head in heads],
                    "training": training,
                }
        if retain_activations:
            self._manual_cache = {
                "input": x,
                "input_normalized": input_normalized,
                "blocks": pair_block_cache,
                "rms_input": hidden,
                "normalized": normalized,
                "heads": heads[0] if not self.actor_heads else heads,
                "training": training,
            }
        return output

    def forward(
        self, x: wp.array2d[wp.float32], *, training: bool = True
    ) -> tuple[wp.array2d[wp.float32], wp.array2d[wp.float32]]:
        """Evaluate both critics with batched dense contractions."""

        if self.ensemble_count != 2:
            raise RuntimeError("forward() requires exactly two critics; use forward_all() for a larger ensemble")
        output = self._forward(x, training=training, retain_activations=False)
        return output[0], output[1]

    def forward_all(
        self, x: wp.array2d[wp.float32] | wp.array3d[wp.float32], *, training: bool = True
    ) -> wp.array3d[wp.float32]:
        """Evaluate every critic with shared-input batched contractions."""

        return self._forward(x, training=training, retain_activations=False)

    def forward_manual(
        self, x: wp.array2d[wp.float32], *, training: bool = True
    ) -> tuple[wp.array2d[wp.float32], wp.array2d[wp.float32]]:
        """Evaluate both critics and retain per-member backward activations."""

        if self.ensemble_count != 2:
            raise RuntimeError("forward_manual() currently requires exactly two critics")
        output = self._forward(x, training=training, retain_activations=True)
        return output[0], output[1]

    def forward_all_manual(
        self, x: wp.array2d[wp.float32] | wp.array3d[wp.float32], *, training: bool = True
    ) -> wp.array3d[wp.float32]:
        """Evaluate every critic and retain stacked backward activations."""

        return self._forward(x, training=training, retain_activations=True)

    def _linear_backward_pair(
        self,
        x: wp.array3d[Any],
        weight: wp.array3d[wp.float32],
        output_grad: wp.array3d[Any],
        input_grad: wp.array3d[Any] | None = None,
    ) -> wp.array3d[Any]:
        rows = int(x.shape[1])
        input_width = int(weight.shape[1])
        output_width = int(weight.shape[2])
        if input_grad is None:
            input_grad = wp.empty_like(x)
        if self.contraction_dtype == "float16":
            mirror = self._contraction_weight(weight)
            contraction_output_grad = self._contraction_input(output_grad)
            input_gemm = gemm_float16_strided_batched_output if x.dtype == wp.float16 else gemm_float16_strided_batched
            input_gemm(
                contraction_output_grad,
                mirror,
                input_grad,
                rows,
                input_width,
                output_width,
                self.ensemble_count,
                transpose_rhs=True,
            )
            contraction_input = self._contraction_input(x, refresh=False)
            gemm_float16_strided_batched(
                contraction_input,
                contraction_output_grad,
                weight.grad,
                input_width,
                output_width,
                rows,
                self.ensemble_count,
                transpose_lhs=True,
            )
        else:
            gemm_float32_strided_batched(
                output_grad,
                weight,
                input_grad,
                rows,
                input_width,
                output_width,
                self.ensemble_count,
                transpose_rhs=True,
            )
            gemm_float32_strided_batched(
                x, output_grad, weight.grad, input_width, output_width, rows, self.ensemble_count, transpose_lhs=True
            )
        return input_grad

    def _norm_backward_pair(
        self,
        x: wp.array3d[wp.float32],
        output_grad: wp.array3d[wp.float32],
        norms: tuple[_UnitBatchNorm, ...],
        *,
        training: bool,
        input_grad: wp.array3d[wp.float32] | None = None,
    ) -> wp.array3d[wp.float32]:
        if input_grad is None:
            input_grad = wp.empty_like(x)
        population_scratch = (
            self._population_norm_scratch.get(id(norms[0]))
            if training and self.contraction_dtype == "float16" and self.device.is_cuda
            else None
        )
        if population_scratch is not None:
            rows = int(x.shape[1])
            width = int(x.shape[2])
            flat_output_grad = self._population_flat_views[int(output_grad.ptr)]
            wp.launch(
                _transpose_population_2d_tile_kernel,
                dim=(
                    self.ensemble_count * ((rows + 31) // 32),
                    (width + 31) // 32,
                    _TILE_REDUCTION_BLOCK_DIM,
                ),
                inputs=[flat_output_grad, rows, width],
                outputs=[population_scratch.transposed_grad],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
            state = self._population_norm_states[id(norms[0])]
            wp.launch(
                _batch_norm_backward_amp_population_tile_kernel,
                dim=(self.ensemble_count * width, _TILE_REDUCTION_BLOCK_DIM),
                inputs=[
                    population_scratch.transposed,
                    population_scratch.transposed_grad,
                    rows,
                    width,
                    population_scratch.mean,
                    population_scratch.inv_std,
                    state["scale"],
                ],
                outputs=[
                    population_scratch.mean_grad,
                    population_scratch.variance_grad,
                    state["scale"].grad,
                    state["bias"].grad,
                ],
                block_dim=_TILE_REDUCTION_BLOCK_DIM,
                device=self.device,
            )
            for ensemble_index, norm in enumerate(norms):
                scratch = norm.last_scratch
                if scratch is None:
                    raise RuntimeError("BatchNorm forward scratch is missing")
                wp.launch(
                    _batch_norm_input_grad_amp_kernel,
                    dim=x[ensemble_index].shape,
                    inputs=[
                        x[ensemble_index],
                        output_grad[ensemble_index],
                        norm.scale,
                        scratch.mean,
                        scratch.inv_std,
                        scratch.mean_grad,
                        scratch.variance_grad,
                    ],
                    outputs=[input_grad[ensemble_index]],
                    device=self.device,
                )
            return input_grad
        for ensemble_index, (network, norm) in enumerate(zip(self.networks, norms, strict=True)):
            member_grad = network._batch_norm_backward(
                norm,
                x[ensemble_index],
                output_grad[ensemble_index],
                training=training,
                amp_staged=self.contraction_dtype == "float16",
                input_grad=input_grad[ensemble_index],
            )
            if member_grad.ptr != input_grad[ensemble_index].ptr:
                raise RuntimeError("BatchNorm backward did not use the reserved destination")
        return input_grad

    def backward_manual(
        self,
        first_output_grad: wp.array2d[wp.float32],
        second_output_grad: wp.array2d[wp.float32],
        *,
        first_input_grad: wp.array2d[wp.float32] | None = None,
        second_input_grad: wp.array2d[wp.float32] | None = None,
        loss_scale: wp.array[wp.float32] | None = None,
        found_inf: wp.array[wp.int32] | None = None,
    ) -> None:
        """Backpropagate both critics with batched dense contractions."""

        cache = getattr(self, "_manual_cache", None)
        if cache is None:
            raise RuntimeError("forward_manual() must be called before backward_manual()")
        output_grad = self._workspace["backward"]["output_grad"]
        wp.launch(
            copy_2d_kernel,
            dim=first_output_grad.shape,
            inputs=[first_output_grad],
            outputs=[output_grad[0]],
            device=self.device,
        )
        wp.launch(
            copy_2d_kernel,
            dim=second_output_grad.shape,
            inputs=[second_output_grad],
            outputs=[output_grad[1]],
            device=self.device,
        )
        self.backward_all_manual(
            output_grad,
            input_grads=(first_input_grad, second_input_grad),
            loss_scale=loss_scale,
            found_inf=found_inf,
        )

    def backward_all_manual(
        self,
        output_grad: wp.array3d[wp.float32],
        *,
        input_grads: tuple[wp.array2d[wp.float32] | None, ...] | None = None,
        loss_scale: wp.array[wp.float32] | None = None,
        found_inf: wp.array[wp.int32] | None = None,
    ) -> None:
        """Backpropagate every critic with population-batched contractions."""

        cache = getattr(self, "_manual_cache", None)
        if cache is None:
            raise RuntimeError("forward_all_manual() must be called before backward_all_manual()")
        if output_grad.shape[0] != self.ensemble_count:
            raise ValueError("output_grad leading dimension must match the critic ensemble")
        input_grads = input_grads if input_grads is not None else (None,) * self.ensemble_count
        if len(input_grads) != self.ensemble_count:
            raise ValueError("input_grads length must match the critic ensemble")
        normalized = cache["normalized"]
        rms_input = cache["rms_input"]
        heads = cache["heads"]
        blocks = cache["blocks"]
        input_normalized = cache["input_normalized"]
        manual_input = cache["input"]
        training = bool(cache["training"])
        backward = self._workspace["backward"]

        normalized_grad = backward["normalized_grad"]
        normalized_grad.zero_()
        if not self.actor_heads:
            heads = (heads,)
        offset = 0
        for head_index, (head, weight) in enumerate(zip(heads, self.head_weights, strict=True)):
            head_grad = backward["head_grads"][head_index]
            for ensemble_index, network in enumerate(self.networks):
                bias = network.head_biases[head_index]
                wp.launch(
                    _head_output_grad_f16_kernel if self.contraction_dtype == "float16" else _head_output_grad_kernel,
                    dim=head[ensemble_index].shape,
                    inputs=[
                        output_grad[ensemble_index],
                        head[ensemble_index],
                        bias,
                        offset,
                        self.actor_heads and head_index == 1,
                        network.log_std_min,
                        network.log_std_max,
                    ],
                    outputs=[head_grad[ensemble_index]],
                    device=self.device,
                )
                _launch_bias_grad(head_grad[ensemble_index], bias.grad)
            head_input_grad = self._linear_backward_pair(
                normalized, weight, head_grad, backward["head_input_grads"][head_index]
            )
            if self.actor_heads:
                accumulated = backward["normalized_accum"]
                for ensemble_index in range(self.ensemble_count):
                    wp.launch(
                        _add_f16_kernel if normalized_grad.dtype == wp.float16 else _add_kernel,
                        dim=normalized_grad[ensemble_index].shape,
                        inputs=[normalized_grad[ensemble_index], head_input_grad[ensemble_index]],
                        outputs=[accumulated[ensemble_index]],
                        device=self.device,
                    )
                normalized_grad = accumulated
            else:
                normalized_grad = head_input_grad
            offset += int(head.shape[2])
        hidden_grad = normalized_grad
        rms_grad = backward["rms_grad"]
        for ensemble_index, network in enumerate(self.networks):
            inv_rms = backward["rms_inv"][ensemble_index]
            projection = backward["rms_projection"][ensemble_index]
            _launch_rms_backward_stats(
                rms_input[ensemble_index], hidden_grad[ensemble_index], network.rms_scale, 1.0e-6, inv_rms, projection
            )
            wp.launch(
                _rms_norm_input_grad_kernel,
                dim=rms_input[ensemble_index].shape,
                inputs=[rms_input[ensemble_index], hidden_grad[ensemble_index], network.rms_scale, inv_rms, projection],
                outputs=[rms_grad[ensemble_index]],
                device=self.device,
            )
            _launch_rms_scale_grad(
                rms_input[ensemble_index], hidden_grad[ensemble_index], inv_rms, network.rms_scale.grad
            )
        hidden_grad = rms_grad
        for block_index in reversed(range(self.num_blocks)):
            residual, pre1, _normed1, activated1, pre2, _normed2, activated2 = blocks[block_index]
            norm2_grad, pre2_grad, activated1_grad, norm1_grad, pre1_grad, branch_grad, combined = backward["blocks"][
                block_index
            ]
            for ensemble_index in range(self.ensemble_count):
                wp.launch(
                    _relu_grad_kernel,
                    dim=activated2[ensemble_index].shape,
                    inputs=[activated2[ensemble_index], hidden_grad[ensemble_index]],
                    outputs=[norm2_grad[ensemble_index]],
                    device=self.device,
                )
            pre2_grad = self._norm_backward_pair(
                pre2,
                norm2_grad,
                tuple(network.block_norms[block_index][1] for network in self.networks),
                training=training,
                input_grad=pre2_grad,
            )
            activated1_grad = self._linear_backward_pair(
                activated1, self.block_weights[block_index][1], pre2_grad, activated1_grad
            )
            for ensemble_index in range(self.ensemble_count):
                wp.launch(
                    _relu_grad_kernel,
                    dim=activated1[ensemble_index].shape,
                    inputs=[activated1[ensemble_index], activated1_grad[ensemble_index]],
                    outputs=[norm1_grad[ensemble_index]],
                    device=self.device,
                )
            pre1_grad = self._norm_backward_pair(
                pre1,
                norm1_grad,
                tuple(network.block_norms[block_index][0] for network in self.networks),
                training=training,
                input_grad=pre1_grad,
            )
            branch_grad = self._linear_backward_pair(
                residual, self.block_weights[block_index][0], pre1_grad, branch_grad
            )
            for ensemble_index in range(self.ensemble_count):
                wp.launch(
                    _add_mixed_f16_kernel
                    if self.contraction_dtype == "float16" and residual.dtype == wp.float16
                    else _add_kernel,
                    dim=hidden_grad[ensemble_index].shape,
                    inputs=[hidden_grad[ensemble_index], branch_grad[ensemble_index]],
                    outputs=[combined[ensemble_index]],
                    device=self.device,
                )
            hidden_grad = combined
        input_normalized_grad = self._linear_backward_pair(
            input_normalized, self.embed_weight, hidden_grad, backward["input_normalized_grad"]
        )
        for ensemble_index, (network, destination) in enumerate(zip(self.networks, input_grads, strict=True)):
            member_grad = network._batch_norm_backward(
                network.input_norm,
                manual_input if manual_input.ndim == 2 else manual_input[ensemble_index],
                input_normalized_grad[ensemble_index],
                training=training,
                input_grad=backward["input_grad"][ensemble_index],
            )
            if destination is not None:
                wp.launch(
                    copy_2d_kernel,
                    dim=destination.shape,
                    inputs=[member_grad],
                    outputs=[destination],
                    device=self.device,
                )
            network._manual_cache = None
        for network in self.networks:
            _unscale_parameter_grads(network.parameters(), loss_scale, found_inf)
        self._manual_cache = None
