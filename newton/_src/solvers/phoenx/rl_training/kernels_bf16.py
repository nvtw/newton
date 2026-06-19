# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warp as wp

from .kernels import DENSE_TILE_BATCH, DENSE_TILE_IN, DENSE_TILE_OUT

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
