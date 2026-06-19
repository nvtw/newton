# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from .kernels import (
    ACTIVATION_ELU,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    DENSE_TILE_BATCH,
    DENSE_TILE_IN,
    DENSE_TILE_OUT,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def cast_2d_float_to_bfloat16_kernel(src: wp.array2d[wp.float32], dst: wp.array2d[wp.bfloat16]):
    row, col = wp.tid()
    dst[row, col] = wp.bfloat16(src[row, col])


@wp.kernel
def dense_weight_grad_bf16_tiled_kernel(
    x: wp.array2d[wp.bfloat16],
    grad_pre: wp.array2d[wp.bfloat16],
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


@wp.kernel
def dense_forward_bf16_tiled_kernel(
    x: wp.array2d[wp.bfloat16],
    weight: wp.array2d[wp.bfloat16],
    in_dim: wp.int32,
    y: wp.array2d[wp.float32],
):
    batch_tile, out_tile = wp.tid()
    total = wp.tile_zeros(shape=(DENSE_TILE_BATCH, DENSE_TILE_OUT), dtype=wp.float32)
    in_tiles = (in_dim + DENSE_TILE_IN - wp.int32(1)) // DENSE_TILE_IN
    for tile in range(in_tiles):
        x_tile = wp.tile_load(
            x,
            shape=(DENSE_TILE_BATCH, DENSE_TILE_IN),
            offset=(batch_tile * DENSE_TILE_BATCH, tile * DENSE_TILE_IN),
        )
        weight_tile = wp.tile_load(
            weight,
            shape=(DENSE_TILE_IN, DENSE_TILE_OUT),
            offset=(tile * DENSE_TILE_IN, out_tile * DENSE_TILE_OUT),
        )
        wp.tile_matmul(x_tile, weight_tile, total)
    wp.tile_store(y, total, offset=(batch_tile * DENSE_TILE_BATCH, out_tile * DENSE_TILE_OUT))


@wp.kernel
def dense_bias_activation_kernel(y: wp.array2d[wp.float32], bias: wp.array[wp.float32], activation: wp.int32):
    row, col = wp.tid()
    y[row, col] = _activation(y[row, col] + bias[col], activation)
