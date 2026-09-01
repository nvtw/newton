# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental batched sparse PCG with a multilevel additive Schwarz preconditioner.

The global matrix is stored as row-contiguous 3x3 BSR. Dense matrices only
exist inside the small Schwarz subdomains. The hierarchy uses injection
restriction: every 16 nodes become one node on the next level, and corrections
from all levels are added during prolongation.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kamino._src.linalg.conjugate import BatchedLinearOperator, CGSolver
from .kamino._src.linalg.conjugate_fused import warp_allreduce_sum

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_CLUSTER_NODE_COUNT = 16
_CLUSTER_DOF_COUNT = 3 * _CLUSTER_NODE_COUNT
_MAX_LEVEL_COUNT = 8


@wp.func_native("__syncthreads();")
def _block_sync(): ...


@wp.func_native(
    r"""
#if defined(__CUDA_ARCH__)
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const int local_row = warp * 8;
    const int base = cluster * 48;
    float totals[8] = {};

    for (int k = lane * 8; k < 48; k += 256) {
        const uint4 rv = *reinterpret_cast<const uint4*>(residual.data + base + k);
        uint4 weights[8];
        #pragma unroll
        for (int output = 0; output < 8; ++output) {
            weights[output] = __ldcs(reinterpret_cast<const uint4*>(
                factors.data + (base + local_row + output) * 48 + k));
        }
        const unsigned* packed_r = reinterpret_cast<const unsigned*>(&rv);
        #pragma unroll
        for (int word = 0; word < 4; ++word) {
            float value = __uint_as_float(packed_r[word] << 16);
            #pragma unroll
            for (int output = 0; output < 8; ++output) {
                const unsigned packed_w = reinterpret_cast<const unsigned*>(&weights[output])[word];
                totals[output] = fmaf(value, __uint_as_float(packed_w << 16), totals[output]);
            }
            value = __uint_as_float(packed_r[word] & 0xffff0000u);
            #pragma unroll
            for (int output = 0; output < 8; ++output) {
                const unsigned packed_w = reinterpret_cast<const unsigned*>(&weights[output])[word];
                totals[output] = fmaf(value, __uint_as_float(packed_w & 0xffff0000u), totals[output]);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        #pragma unroll
        for (int output = 0; output < 8; ++output)
            totals[output] += __shfl_down_sync(0xffffffffu, totals[output], offset);
    }
    if (lane == 0) {
        #pragma unroll
        for (int output = 0; output < 8; ++output)
            result.data[base + local_row + output] = totals[output];
    }
#endif
"""
)
def _bf16_cluster_gemv(
    factors: wp.array2d[wp.bfloat16],
    residual: wp.array[wp.bfloat16],
    result: wp.array2d[wp.float32],
    cluster: int,
    thread: int,
): ...


@wp.func_native(
    r"""
#if defined(__CUDA_ARCH__)
    constexpr int ROWS = 48;
    constexpr int PADDED_LD = 52;
    __shared__ __align__(16) float shared_data[ROWS * PADDED_LD + ROWS];
    float* shared_matrix = shared_data;
    float* shared_residual = shared_data + ROWS * PADDED_LD;
    const int base = cluster * ROWS;

    #if __CUDA_ARCH__ >= 800
    for (int copy = thread; copy < ROWS * 12; copy += blockDim.x) {
        const int row = copy / 12;
        const int segment = copy - row * 12;
        float* dst = shared_matrix + row * PADDED_LD + segment * 4;
        const float* src = factors.data + (base + row) * ROWS + segment * 4;
        const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared), "l"(src));
    }
    #else
    for (int index = thread; index < ROWS * ROWS; index += blockDim.x) {
        const int row = index / ROWS;
        const int col = index - row * ROWS;
        shared_matrix[row * PADDED_LD + col] = factors.data[base * ROWS + index];
    }
    #endif

    if (thread < ROWS) {
        const int slot = cluster * 16 + thread / 3;
        const int component = thread % 3;
        const int begin = slot_fine_begin.data[slot];
        const int count = slot_fine_count.data[slot];
        float total = 0.0f;
        for (int index = 0; index < count; ++index)
            total += residual.data[3 * (begin + index) + component];
        shared_residual[thread] = total;
    }
    #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    #endif
    __syncthreads();

    if (thread < ROWS) {
        float total = 0.0f;
        #pragma unroll
        for (int col = 0; col < ROWS; ++col)
            total = fmaf(shared_matrix[thread * PADDED_LD + col], shared_residual[col], total);
        result.data[base + thread] = total;
    }
#endif
"""
)
def _fp32_async_cluster_gemv(
    factors: wp.array2d[wp.float32],
    residual: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    result: wp.array2d[wp.float32],
    cluster: int,
    thread: int,
): ...


@wp.kernel
def _insert_bsr_blocks_kernel(
    input_row: wp.array[wp.int32],
    input_col: wp.array[wp.int32],
    input_value: wp.array[wp.mat33f],
    input_count: wp.array[wp.int32],
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    overflow: wp.array[wp.int32],
):
    block = wp.tid()
    if block >= input_count[0]:
        return
    row = input_row[block]
    slot = wp.atomic_add(row_nnz, row, 1)
    capacity = row_offsets[row + 1] - row_offsets[row]
    if slot >= capacity:
        wp.atomic_sub(row_nnz, row, 1)
        overflow[0] = 1
        return
    output = row_offsets[row] + slot
    col_indices[output] = input_col[block]
    values[output] = input_value[block]


@wp.kernel
def _bsr_gemv_kernel(
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    world_row_offsets: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    alpha: wp.float32,
    beta: wp.float32,
):
    world, local_row = wp.tid()
    if local_row >= world_row_count[world] or not world_active[world]:
        return

    row = world_row_offsets[world] + local_row
    result = wp.vec3f(0.0)
    row_begin = row_offsets[row]
    for local_block in range(row_nnz[row]):
        block = row_begin + local_block
        col = col_indices[block]
        x_col = wp.vec3f(x[3 * col], x[3 * col + 1], x[3 * col + 2])
        result += values[block] * x_col

    base = 3 * row
    y[base] = alpha * result[0] + beta * y[base]
    y[base + 1] = alpha * result[1] + beta * y[base + 1]
    y[base + 2] = alpha * result[2] + beta * y[base + 2]


@wp.kernel
def _bsr_spmv_dot_kernel(
    row_offsets: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    world_row_offsets: wp.array[wp.int32],
    world_row_count: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    p: wp.array[wp.float32],
    ap: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
):
    world, chunk, thread = wp.tid()
    local_row = chunk * wp.block_dim() + thread
    contribution = wp.float32(0.0)
    if local_row < world_row_count[world] and world_active[world]:
        row = world_row_offsets[world] + local_row
        result = wp.vec3f(0.0)
        row_begin = row_offsets[row]
        for local_block in range(row_nnz[row]):
            block = row_begin + local_block
            col = col_indices[block]
            p_col = wp.vec3f(p[3 * col], p[3 * col + 1], p[3 * col + 2])
            result += values[block] * p_col
        base = 3 * row
        ap[base] = result[0]
        ap[base + 1] = result[1]
        ap[base + 2] = result[2]
        contribution = p[base] * result[0] + p[base + 1] * result[1] + p[base + 2] * result[2]
    chunk_sum = wp.tile_sum(wp.tile(contribution))[0]
    if thread == 0:
        wp.atomic_add(p_ap, world, chunk_sum)


@wp.kernel
def _cg_update_xr_save_kernel(
    tolerance: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    rz_old: wp.array[wp.float32],
    rz_new: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
    p: wp.array[wp.float32],
    ap: wp.array[wp.float32],
    x: wp.array[wp.float32],
    r: wp.array[wp.float32],
    vector_offsets: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
):
    world, local_dof = wp.tid()
    if local_dof >= dimensions[world]:
        return
    current_rz = rz_new[world]
    alpha = wp.where(
        residual[world] > tolerance[world] and p_ap[world] > 0.0,
        current_rz / p_ap[world],
        wp.float32(0.0),
    )
    dof = vector_offsets[world] + local_dof
    x[dof] += alpha * p[dof]
    r[dof] -= alpha * ap[dof]
    if local_dof == 0:
        rz_old[world] = current_rz


@wp.kernel
def _cg_update_p_reset_kernel(
    tolerance: wp.array[wp.float32],
    residual: wp.array[wp.float32],
    rz_old: wp.array[wp.float32],
    rz_new: wp.array[wp.float32],
    z: wp.array[wp.float32],
    p: wp.array[wp.float32],
    p_ap: wp.array[wp.float32],
    vector_offsets: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
):
    world, local_dof = wp.tid()
    if local_dof >= dimensions[world]:
        return
    beta = wp.where(
        residual[world] > tolerance[world] and rz_old[world] > 0.0,
        rz_new[world] / rz_old[world],
        wp.float32(0.0),
    )
    dof = vector_offsets[world] + local_dof
    p[dof] = z[dof] + beta * p[dof]
    if local_dof == 0:
        p_ap[world] = 0.0


@wp.kernel
def _accumulate_iterations_kernel(
    pass_iterations: wp.array[wp.int32],
    total_iterations: wp.array[wp.int32],
):
    world = wp.tid()
    total_iterations[world] += pass_iterations[world]


@wp.kernel
def _scatter_hierarchy_kernel(
    block_rows: wp.array[wp.int32],
    block_slots: wp.array[wp.int32],
    row_nnz: wp.array[wp.int32],
    col_indices: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    ancestors: wp.array2d[wp.int32],
    factors: wp.array2d[wp.float32],
):
    block = wp.tid()
    row = block_rows[block]
    if block_slots[block] >= row_nnz[row]:
        return
    col = col_indices[block]
    value = values[block]

    for level in range(_MAX_LEVEL_COUNT):
        row_node = ancestors[row, level]
        col_node = ancestors[col, level]
        if row_node < 0 or col_node < 0:
            break
        cluster = row_node // _CLUSTER_NODE_COUNT
        if cluster != col_node // _CLUSTER_NODE_COUNT:
            continue
        local_row = 3 * (row_node % _CLUSTER_NODE_COUNT)
        local_col = 3 * (col_node % _CLUSTER_NODE_COUNT)
        matrix_row = cluster * _CLUSTER_DOF_COUNT + local_row
        for i in range(3):
            for j in range(3):
                wp.atomic_add(factors, matrix_row + i, local_col + j, value[i, j])


@wp.kernel
def _initialize_cluster_kernel(
    cluster_dof_count: wp.array[wp.int32],
    regularization: wp.float32,
    factors: wp.array2d[wp.float32],
):
    cluster, row, col = wp.tid()
    value = wp.float32(0.0)
    if row == col:
        value = wp.where(row < cluster_dof_count[cluster], regularization, wp.float32(1.0))
    factors[cluster * _CLUSTER_DOF_COUNT + row, col] = value


@wp.kernel
def _factorize_cluster_kernel(factors: wp.array2d[wp.float32]):
    cluster, thread = wp.tid()
    factor = wp.tile_load(
        factors,
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    block_dim = wp.block_dim()
    element_count = _CLUSTER_DOF_COUNT * _CLUSTER_DOF_COUNT
    wp.tile_cholesky_inplace(factor)
    inverse = wp.tile_zeros(
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        dtype=wp.float32,
        storage="shared",
    )
    for iteration in range((element_count + 127) // 128):
        index = thread + iteration * block_dim
        if index < element_count:
            row = index // _CLUSTER_DOF_COUNT
            col = index % _CLUSTER_DOF_COUNT
            inverse[row, col] = wp.where(row == col, wp.float32(1.0), wp.float32(0.0))
    wp.tile_lower_solve_inplace(factor, inverse)
    wp.tile_upper_solve_inplace(wp.tile_transpose(factor), inverse)
    wp.tile_store(factors, inverse, offset=(cluster * _CLUSTER_DOF_COUNT, 0))


@wp.kernel
def _convert_factors_bf16_kernel(
    factors: wp.array2d[wp.float32],
    factors_bf16: wp.array2d[wp.bfloat16],
):
    row, col = wp.tid()
    factors_bf16[row, col] = wp.bfloat16(factors[row, col])


@wp.kernel
def _restrict_residual_kernel(
    r: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    slot_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.float32],
):
    slot, component = wp.tid()
    count = slot_fine_count[slot]
    value = wp.float32(0.0)
    if count > 0 and world_active[slot_world[slot]]:
        begin = slot_fine_begin[slot]
        for i in range(count):
            value += r[3 * (begin + i) + component]
    hierarchy_r[3 * slot + component, 0] = value


@wp.kernel
def _restrict_residual_bf16_kernel(
    r: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    slot_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.bfloat16],
):
    slot, component = wp.tid()
    count = slot_fine_count[slot]
    value = wp.float32(0.0)
    if count > 0 and world_active[slot_world[slot]]:
        begin = slot_fine_begin[slot]
        for i in range(count):
            value += r[3 * (begin + i) + component]
    row = 3 * slot + component
    hierarchy_r[row, 0] = wp.bfloat16(value)
    for col in range(1, 8):
        hierarchy_r[row, col] = wp.bfloat16(0.0)


@wp.kernel
def _restrict_residual_bf16_vector_kernel(
    r: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    slot_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array[wp.bfloat16],
):
    slot, component = wp.tid()
    count = slot_fine_count[slot]
    value = wp.float32(0.0)
    if count > 0 and world_active[slot_world[slot]]:
        begin = slot_fine_begin[slot]
        for i in range(count):
            value += r[3 * (begin + i) + component]
    hierarchy_r[3 * slot + component] = wp.bfloat16(value)


@wp.kernel
def _restrict_residual_tensor_f32_kernel(
    r: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    slot_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.float32],
):
    slot, component = wp.tid()
    count = slot_fine_count[slot]
    value = wp.float32(0.0)
    if count > 0 and world_active[slot_world[slot]]:
        begin = slot_fine_begin[slot]
        for i in range(count):
            value += r[3 * (begin + i) + component]
    row = 3 * slot + component
    hierarchy_r[row, 0] = value
    for col in range(1, 8):
        hierarchy_r[row, col] = 0.0


@wp.kernel
def _apply_cluster_inverse_kernel(
    factors: wp.array2d[wp.float32],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.float32],
    hierarchy_z: wp.array2d[wp.float32],
):
    cluster, dof = wp.tid()
    if not world_active[cluster_world[cluster]]:
        return
    value = wp.float32(0.0)
    row = cluster * _CLUSTER_DOF_COUNT + dof
    for col in range(_CLUSTER_DOF_COUNT):
        value += factors[row, col] * hierarchy_r[cluster * _CLUSTER_DOF_COUNT + col, 0]
    hierarchy_z[row, 0] = value


@wp.kernel(enable_backward=False)
def _apply_cluster_inverse_bf16_kernel(
    factors: wp.array2d[wp.bfloat16],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.bfloat16],
    hierarchy_z: wp.array2d[wp.float32],
):
    cluster, _thread = wp.tid()
    if not world_active[cluster_world[cluster]]:
        return
    matrix = wp.tile_load(
        factors,
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    rhs = wp.tile_load(
        hierarchy_r,
        shape=(_CLUSTER_DOF_COUNT, 8),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    result = wp.tile_zeros(shape=(_CLUSTER_DOF_COUNT, 8), dtype=wp.float32, storage="shared")
    wp.tile_matmul(matrix, rhs, result)
    wp.tile_store(hierarchy_z, result, offset=(cluster * _CLUSTER_DOF_COUNT, 0))


@wp.kernel(enable_backward=False)
def _apply_cluster_inverse_tensor_f32_kernel(
    factors: wp.array2d[wp.float32],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array2d[wp.float32],
    hierarchy_z: wp.array2d[wp.float32],
):
    cluster, _thread = wp.tid()
    if not world_active[cluster_world[cluster]]:
        return
    matrix = wp.tile_load(
        factors,
        shape=(_CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    rhs = wp.tile_load(
        hierarchy_r,
        shape=(_CLUSTER_DOF_COUNT, 8),
        offset=(cluster * _CLUSTER_DOF_COUNT, 0),
        storage="shared",
    )
    result = wp.tile_zeros(shape=(_CLUSTER_DOF_COUNT, 8), dtype=wp.float32, storage="shared")
    wp.tile_matmul(matrix, rhs, result)
    wp.tile_store(hierarchy_z, result, offset=(cluster * _CLUSTER_DOF_COUNT, 0))


@wp.kernel(enable_backward=False)
def _apply_cluster_inverse_bf16_vector_kernel(
    factors: wp.array2d[wp.bfloat16],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_r: wp.array[wp.bfloat16],
    hierarchy_z: wp.array2d[wp.float32],
):
    cluster, thread = wp.tid()
    if world_active[cluster_world[cluster]]:
        _bf16_cluster_gemv(factors, hierarchy_r, hierarchy_z, cluster, thread)


@wp.kernel(enable_backward=False)
def _apply_cluster_inverse_fp32_async_kernel(
    factors: wp.array2d[wp.float32],
    cluster_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    residual: wp.array[wp.float32],
    slot_fine_begin: wp.array[wp.int32],
    slot_fine_count: wp.array[wp.int32],
    hierarchy_z: wp.array2d[wp.float32],
):
    cluster, thread = wp.tid()
    if world_active[cluster_world[cluster]]:
        _fp32_async_cluster_gemv(
            factors,
            residual,
            slot_fine_begin,
            slot_fine_count,
            hierarchy_z,
            cluster,
            thread,
        )


@wp.kernel
def _prolongate_kernel(
    ancestors: wp.array2d[wp.int32],
    fine_node_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_z: wp.array2d[wp.float32],
    z: wp.array[wp.float32],
):
    fine_node, component = wp.tid()
    if not world_active[fine_node_world[fine_node]]:
        z[3 * fine_node + component] = 0.0
        return
    value = wp.float32(0.0)
    for level in range(_MAX_LEVEL_COUNT):
        node = ancestors[fine_node, level]
        if node < 0:
            break
        value += hierarchy_z[3 * node + component, 0]
    z[3 * fine_node + component] = value


@wp.kernel
def _prolongate_bf16_kernel(
    ancestors: wp.array2d[wp.int32],
    fine_node_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    hierarchy_z: wp.array2d[wp.float32],
    z: wp.array[wp.float32],
):
    fine_node, component = wp.tid()
    if not world_active[fine_node_world[fine_node]]:
        z[3 * fine_node + component] = 0.0
        return
    value = wp.float32(0.0)
    for level in range(_MAX_LEVEL_COUNT):
        node = ancestors[fine_node, level]
        if node < 0:
            break
        value += hierarchy_z[3 * node + component, 0]
    z[3 * fine_node + component] = value


@functools.cache
def _make_one_block_pcg_kernel(block_dim: int):
    warps_per_block = block_dim // 32

    @wp.func
    def reduce_sum(value: wp.float32, thread: wp.int32) -> wp.float32:
        warp_value = warp_allreduce_sum(value)
        warp_sums = wp.tile_zeros(shape=wp.static(warps_per_block), dtype=wp.float32, storage="shared")
        wp.tile_scatter_add(warp_sums, thread // 32, warp_value, thread % 32 == 0, False)
        result = wp.float32(0.0)
        for warp in range(wp.static(warps_per_block)):
            result += warp_sums[warp]
        return result

    @wp.func
    def apply_mas(
        world: wp.int32,
        thread: wp.int32,
        scalar_begin: wp.int32,
        scalar_count: wp.int32,
        cluster_begin: wp.int32,
        cluster_count: wp.int32,
        ancestors: wp.array2d[wp.int32],
        slot_fine_begin: wp.array[wp.int32],
        slot_fine_count: wp.array[wp.int32],
        factors: wp.array2d[wp.float32],
        r: wp.array[wp.float32],
        hierarchy_r: wp.array2d[wp.float32],
        hierarchy_z: wp.array2d[wp.float32],
        z: wp.array[wp.float32],
    ):
        hierarchy_begin = cluster_begin * _CLUSTER_DOF_COUNT
        hierarchy_count = cluster_count * _CLUSTER_DOF_COUNT
        q = thread
        while q < hierarchy_count:
            hierarchy_dof = hierarchy_begin + q
            slot = hierarchy_dof // 3
            component = hierarchy_dof % 3
            count = slot_fine_count[slot]
            value = wp.float32(0.0)
            if count > 0:
                begin = slot_fine_begin[slot]
                for i in range(count):
                    value += r[3 * (begin + i) + component]
            hierarchy_r[hierarchy_dof, 0] = value
            q += wp.static(block_dim)
        _block_sync()

        q = thread
        while q < hierarchy_count:
            hierarchy_dof = hierarchy_begin + q
            cluster = hierarchy_dof // _CLUSTER_DOF_COUNT
            local_dof = hierarchy_dof % _CLUSTER_DOF_COUNT
            value = wp.float32(0.0)
            factor_row = cluster * _CLUSTER_DOF_COUNT + local_dof
            for col in range(_CLUSTER_DOF_COUNT):
                value += factors[factor_row, col] * hierarchy_r[cluster * _CLUSTER_DOF_COUNT + col, 0]
            hierarchy_z[hierarchy_dof, 0] = value
            q += wp.static(block_dim)
        _block_sync()

        q = thread
        while q < scalar_count:
            fine_node = (scalar_begin + q) // 3
            component = q % 3
            value = wp.float32(0.0)
            for level in range(_MAX_LEVEL_COUNT):
                slot = ancestors[fine_node, level]
                if slot < 0:
                    break
                value += hierarchy_z[3 * slot + component, 0]
            z[scalar_begin + q] = value
            q += wp.static(block_dim)
        _block_sync()

    @wp.kernel
    def one_block_pcg_kernel(
        row_offsets: wp.array[wp.int32],
        row_nnz: wp.array[wp.int32],
        col_indices: wp.array[wp.int32],
        values: wp.array[wp.mat33f],
        world_row_offsets: wp.array[wp.int32],
        world_row_count: wp.array[wp.int32],
        world_scalar_offsets: wp.array[wp.int32],
        world_cluster_offsets: wp.array[wp.int32],
        world_cluster_count: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        ancestors: wp.array2d[wp.int32],
        slot_fine_begin: wp.array[wp.int32],
        slot_fine_count: wp.array[wp.int32],
        factors: wp.array2d[wp.float32],
        hierarchy_r: wp.array2d[wp.float32],
        hierarchy_z: wp.array2d[wp.float32],
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        r: wp.array[wp.float32],
        z: wp.array[wp.float32],
        p: wp.array[wp.float32],
        ap: wp.array[wp.float32],
        atol: wp.array[wp.float32],
        rtol: wp.array[wp.float32],
        max_iterations: wp.array[wp.int32],
        out_iterations: wp.array[wp.int32],
        out_residual: wp.array[wp.float32],
    ):
        world, thread = wp.tid()
        if not world_active[world]:
            if thread == 0:
                out_iterations[world] = 0
                out_residual[world] = 0.0
            return

        row_begin = world_row_offsets[world]
        row_count = world_row_count[world]
        scalar_begin = world_scalar_offsets[world]
        scalar_count = 3 * row_count
        cluster_begin = world_cluster_offsets[world]
        cluster_count = world_cluster_count[world]

        bb_local = wp.float32(0.0)
        q = thread
        while q < scalar_count:
            value = b[scalar_begin + q]
            x[scalar_begin + q] = 0.0
            r[scalar_begin + q] = value
            bb_local += value * value
            q += wp.static(block_dim)
        bb = reduce_sum(bb_local, thread)
        tolerance = wp.max(rtol[world] * rtol[world] * bb, atol[world] * atol[world])

        apply_mas(
            world,
            thread,
            scalar_begin,
            scalar_count,
            cluster_begin,
            cluster_count,
            ancestors,
            slot_fine_begin,
            slot_fine_count,
            factors,
            r,
            hierarchy_r,
            hierarchy_z,
            z,
        )
        rz_local = wp.float32(0.0)
        rr_local = wp.float32(0.0)
        q = thread
        while q < scalar_count:
            index = scalar_begin + q
            p[index] = z[index]
            rz_local += r[index] * z[index]
            rr_local += r[index] * r[index]
            q += wp.static(block_dim)
        rz = reduce_sum(rz_local, thread)
        rr = reduce_sum(rr_local, thread)
        _block_sync()

        iteration = wp.int32(0)
        for _iteration in range(max_iterations[world]):
            if rr <= tolerance:
                break
            p_ap_local = wp.float32(0.0)
            local_row = thread
            while local_row < row_count:
                row = row_begin + local_row
                result = wp.vec3f(0.0)
                block_begin = row_offsets[row]
                for local_block in range(row_nnz[row]):
                    block = block_begin + local_block
                    col = col_indices[block]
                    p_col = wp.vec3f(p[3 * col], p[3 * col + 1], p[3 * col + 2])
                    result += values[block] * p_col
                base = 3 * row
                ap[base] = result[0]
                ap[base + 1] = result[1]
                ap[base + 2] = result[2]
                p_ap_local += p[base] * result[0] + p[base + 1] * result[1] + p[base + 2] * result[2]
                local_row += wp.static(block_dim)
            p_ap = reduce_sum(p_ap_local, thread)
            alpha = wp.where(p_ap > 0.0, rz / p_ap, wp.float32(0.0))

            q = thread
            while q < scalar_count:
                index = scalar_begin + q
                x[index] += alpha * p[index]
                r[index] -= alpha * ap[index]
                q += wp.static(block_dim)
            _block_sync()

            apply_mas(
                world,
                thread,
                scalar_begin,
                scalar_count,
                cluster_begin,
                cluster_count,
                ancestors,
                slot_fine_begin,
                slot_fine_count,
                factors,
                r,
                hierarchy_r,
                hierarchy_z,
                z,
            )
            rz_new_local = wp.float32(0.0)
            rr_local = wp.float32(0.0)
            q = thread
            while q < scalar_count:
                index = scalar_begin + q
                rz_new_local += r[index] * z[index]
                rr_local += r[index] * r[index]
                q += wp.static(block_dim)
            rz_new = reduce_sum(rz_new_local, thread)
            rr = reduce_sum(rr_local, thread)
            beta = wp.where(rz > 0.0, rz_new / rz, wp.float32(0.0))
            q = thread
            while q < scalar_count:
                index = scalar_begin + q
                p[index] = z[index] + beta * p[index]
                q += wp.static(block_dim)
            _block_sync()
            rz = rz_new
            iteration += 1

        if thread == 0:
            out_iterations[world] = iteration
            out_residual[world] = rr

    return one_block_pcg_kernel


@dataclass(frozen=True)
class BatchedBSRMatrix:
    """Batched square matrices in row-contiguous 3x3 BSR storage."""

    row_offsets: wp.array
    row_nnz: wp.array
    col_indices: wp.array
    values: wp.array
    block_rows: wp.array
    block_slots: wp.array
    world_row_offsets: wp.array
    world_row_count: wp.array
    world_scalar_offsets: wp.array
    active_dims: wp.array
    fine_node_world: wp.array
    overflow: wp.array
    world_count: int
    total_row_count: int
    total_scalar_count: int
    max_row_count: int
    max_nnz_count: int
    device: wp.Device

    @classmethod
    def from_host(
        cls,
        row_offsets: Sequence[np.ndarray],
        col_indices: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        *,
        row_capacities: Sequence[np.ndarray] | None = None,
        device: wp.DeviceLike = None,
    ) -> BatchedBSRMatrix:
        """Upload independent local-indexed 3x3 BSR matrices."""
        if not (len(row_offsets) == len(col_indices) == len(values)) or not row_offsets:
            raise ValueError("row_offsets, col_indices, and values must have equal nonzero lengths")

        row_counts: list[int] = []
        global_rows = [0]
        packed_row_offsets = [0]
        packed_row_nnz: list[np.ndarray] = []
        packed_cols: list[int] = []
        packed_values: list[np.ndarray] = []
        packed_block_rows: list[int] = []
        packed_block_slots: list[int] = []
        node_world: list[np.ndarray] = []

        if row_capacities is not None and len(row_capacities) != len(row_offsets):
            raise ValueError("row_capacities must have one array per world")

        for world, (row_data, col_data, block_data) in enumerate(zip(row_offsets, col_indices, values, strict=True)):
            rows = np.asarray(row_data, dtype=np.int32)
            cols = np.asarray(col_data, dtype=np.int32)
            blocks = np.asarray(block_data, dtype=np.float32)
            row_count = rows.size - 1
            if row_count <= 0 or rows[0] != 0 or rows[-1] != cols.size:
                raise ValueError(f"invalid BSR row offsets for world {world}")
            if blocks.shape != (cols.size, 3, 3):
                raise ValueError(f"values for world {world} must have shape ({cols.size}, 3, 3)")
            if np.any(rows[1:] < rows[:-1]) or np.any(cols < 0) or np.any(cols >= row_count):
                raise ValueError(f"invalid BSR indices for world {world}")

            row_nnz = np.diff(rows).astype(np.int32)
            capacities = row_nnz if row_capacities is None else np.asarray(row_capacities[world], dtype=np.int32)
            if capacities.shape != (row_count,) or np.any(capacities < row_nnz):
                raise ValueError(f"row capacities for world {world} must cover every active row entry")

            row_base = global_rows[-1]
            row_counts.append(row_count)
            global_rows.append(row_base + row_count)
            packed_row_nnz.append(row_nnz)
            for local_row in range(row_count):
                begin = int(rows[local_row])
                count = int(row_nnz[local_row])
                capacity = int(capacities[local_row])
                for slot in range(capacity):
                    packed_block_rows.append(row_base + local_row)
                    packed_block_slots.append(slot)
                    if slot < count:
                        packed_cols.append(row_base + int(cols[begin + slot]))
                        packed_values.append(blocks[begin + slot])
                    else:
                        packed_cols.append(row_base + local_row)
                        packed_values.append(np.zeros((3, 3), dtype=np.float32))
                packed_row_offsets.append(packed_row_offsets[-1] + capacity)
            node_world.append(np.full(row_count, world, dtype=np.int32))

        device = wp.get_device(device)
        row_counts_np = np.asarray(row_counts, dtype=np.int32)
        global_rows_np = np.asarray(global_rows, dtype=np.int32)
        scalar_offsets = 3 * global_rows_np[:-1]
        return cls(
            row_offsets=wp.array(np.asarray(packed_row_offsets, dtype=np.int32), device=device),
            row_nnz=wp.array(np.concatenate(packed_row_nnz), device=device),
            col_indices=wp.array(np.asarray(packed_cols, dtype=np.int32), device=device),
            values=wp.array(np.asarray(packed_values, dtype=np.float32), dtype=wp.mat33f, device=device),
            block_rows=wp.array(np.asarray(packed_block_rows, dtype=np.int32), device=device),
            block_slots=wp.array(np.asarray(packed_block_slots, dtype=np.int32), device=device),
            world_row_offsets=wp.array(global_rows_np[:-1], device=device),
            world_row_count=wp.array(row_counts_np, device=device),
            world_scalar_offsets=wp.array(scalar_offsets, device=device),
            active_dims=wp.array(3 * row_counts_np, device=device),
            fine_node_world=wp.array(np.concatenate(node_world), device=device),
            overflow=wp.zeros(1, dtype=wp.int32, device=device),
            world_count=len(row_counts),
            total_row_count=int(global_rows_np[-1]),
            total_scalar_count=3 * int(global_rows_np[-1]),
            max_row_count=int(row_counts_np.max()),
            max_nnz_count=len(packed_cols),
            device=device,
        )

    @property
    def storage_bytes(self) -> int:
        """Return GPU bytes used by the sparse matrix data and metadata."""
        arrays = (
            self.row_offsets,
            self.row_nnz,
            self.col_indices,
            self.values,
            self.block_rows,
            self.block_slots,
            self.world_row_offsets,
            self.world_row_count,
            self.world_scalar_offsets,
            self.active_dims,
            self.fine_node_world,
            self.overflow,
        )
        return sum(array.size * wp.types.type_size_in_bytes(array.dtype) for array in arrays)

    def begin_assembly(self) -> None:
        """Clear active row lengths before a device-side pattern rebuild."""
        self.row_nnz.zero_()
        self.overflow.zero_()

    def insert_blocks(
        self,
        rows: wp.array,
        cols: wp.array,
        values: wp.array,
        count: wp.array,
    ) -> None:
        """Insert global-indexed BSR blocks into fixed row-capacity storage.

        Duplicate entries are legal and are accumulated naturally by SpMV and
        preconditioner assembly. ``count`` is a one-element device array, so
        the active contribution count may change inside a captured graph.
        """
        if rows.shape != cols.shape or rows.shape[0] != values.shape[0]:
            raise ValueError("rows, cols, and values must have equal capacities")
        wp.launch(
            _insert_bsr_blocks_kernel,
            dim=rows.shape[0],
            inputs=[
                rows,
                cols,
                values,
                count,
                self.row_offsets,
                self.row_nnz,
                self.col_indices,
                self.values,
                self.overflow,
            ],
            device=self.device,
        )

    def gemv(
        self,
        x: wp.array,
        y: wp.array,
        world_active: wp.array,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Compute ``y = alpha * A * x + beta * y``."""
        wp.launch(
            _bsr_gemv_kernel,
            dim=(self.world_count, self.max_row_count),
            inputs=[
                self.row_offsets,
                self.row_nnz,
                self.col_indices,
                self.values,
                self.world_row_offsets,
                self.world_row_count,
                world_active,
                x,
                y,
                wp.float32(alpha),
                wp.float32(beta),
            ],
            device=self.device,
        )


class MASPreconditioner:
    """Multilevel additive Schwarz preconditioner for :class:`BatchedBSRMatrix`."""

    def __init__(
        self,
        matrix: BatchedBSRMatrix,
        *,
        regularization: float = 1.0e-6,
        apply_mode: str = "fp32_async",
    ):
        if not matrix.device.is_cuda:
            raise ValueError("MASPreconditioner currently requires a CUDA device")
        self.matrix = matrix
        self.device = matrix.device
        self.regularization = float(regularization)
        if apply_mode not in ("fp32", "fp32_async", "fp32_tile", "bf16_tile", "bf16_vector"):
            raise ValueError("apply_mode must be 'fp32', 'fp32_async', 'fp32_tile', 'bf16_tile', or 'bf16_vector'")
        self.apply_mode = apply_mode
        metadata = self._build_hierarchy_metadata(matrix)
        self.level_count = metadata["level_count"]
        self.cluster_count = metadata["cluster_count"]

        self.ancestors = wp.array(metadata["ancestors"], device=self.device)
        self.cluster_dof_count = wp.array(metadata["cluster_dof_count"], device=self.device)
        self.cluster_world = wp.array(metadata["cluster_world"], device=self.device)
        self.world_cluster_offsets = wp.array(metadata["world_cluster_offsets"], device=self.device)
        self.world_cluster_count = wp.array(metadata["world_cluster_count"], device=self.device)
        self.slot_fine_begin = wp.array(metadata["slot_fine_begin"], device=self.device)
        self.slot_fine_count = wp.array(metadata["slot_fine_count"], device=self.device)
        self.slot_world = wp.array(metadata["slot_world"], device=self.device)
        self.factors = wp.zeros(
            (self.cluster_count * _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
            dtype=wp.float32,
            device=self.device,
        )
        hierarchy_scalar_count = self.cluster_count * _CLUSTER_DOF_COUNT
        self.hierarchy_r = wp.zeros((hierarchy_scalar_count, 1), dtype=wp.float32, device=self.device)
        self.hierarchy_z = wp.zeros_like(self.hierarchy_r)
        self.factors_bf16 = None
        self.hierarchy_r_tensor = None
        self.hierarchy_z_tensor = None
        self.hierarchy_r_bf16_vector = None
        if self.apply_mode in ("fp32_tile", "bf16_tile"):
            # Eight RHS columns make the matrix-vector product a tensor-core
            # friendly 48x48 @ 48x8 GEMM. Only column zero carries data.
            tensor_dtype = wp.bfloat16 if self.apply_mode == "bf16_tile" else wp.float32
            self.hierarchy_r_tensor = wp.zeros((hierarchy_scalar_count, 8), dtype=tensor_dtype, device=self.device)
            self.hierarchy_z_tensor = wp.zeros((hierarchy_scalar_count, 8), dtype=wp.float32, device=self.device)
        if self.apply_mode in ("bf16_tile", "bf16_vector"):
            self.factors_bf16 = wp.zeros(
                (self.cluster_count * _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
                dtype=wp.bfloat16,
                device=self.device,
            )
        if self.apply_mode == "bf16_vector":
            self.hierarchy_r_bf16_vector = wp.zeros(hierarchy_scalar_count, dtype=wp.bfloat16, device=self.device)
        self.refit()

    @staticmethod
    def _build_hierarchy_metadata(matrix: BatchedBSRMatrix) -> dict[str, np.ndarray | int]:
        row_counts = matrix.world_row_count.numpy()
        row_bases = np.concatenate(([0], np.cumsum(row_counts[:-1], dtype=np.int64))).astype(np.int32)
        ancestors = np.full((matrix.total_row_count, _MAX_LEVEL_COUNT), -1, dtype=np.int32)
        cluster_dof_count: list[int] = []
        cluster_world: list[int] = []
        slot_fine_begin: list[int] = []
        slot_fine_count: list[int] = []
        slot_world: list[int] = []
        world_cluster_offsets: list[int] = []
        world_cluster_count: list[int] = []
        cluster_count = 0
        max_level_count = 0

        for world, (fine_base, fine_count) in enumerate(zip(row_bases, row_counts, strict=True)):
            world_cluster_begin = cluster_count
            world_cluster_offsets.append(world_cluster_begin)
            level_size = int(fine_count)
            level = 0
            fine_indices = np.arange(int(fine_count), dtype=np.int32)
            while True:
                max_level_count = max(max_level_count, level + 1)
                cluster_count_level = (level_size + _CLUSTER_NODE_COUNT - 1) // _CLUSTER_NODE_COUNT
                level_cluster_begin = cluster_count
                fine_span = _CLUSTER_NODE_COUNT**level
                for cluster_local in range(cluster_count_level):
                    node_begin = cluster_local * _CLUSTER_NODE_COUNT
                    node_count = min(_CLUSTER_NODE_COUNT, level_size - node_begin)
                    cluster_dof_count.append(3 * node_count)
                    cluster_world.append(world)
                    for local in range(_CLUSTER_NODE_COUNT):
                        level_node = node_begin + local
                        begin = int(fine_base) + level_node * fine_span
                        count = max(0, min(fine_span, int(fine_base + fine_count) - begin))
                        slot_fine_begin.append(begin if count else 0)
                        slot_fine_count.append(count)
                        slot_world.append(world)
                level_nodes = fine_indices // fine_span
                ancestors[fine_base : fine_base + fine_count, level] = (
                    level_cluster_begin + level_nodes // _CLUSTER_NODE_COUNT
                ) * _CLUSTER_NODE_COUNT + level_nodes % _CLUSTER_NODE_COUNT
                cluster_count += cluster_count_level
                if level_size <= _CLUSTER_NODE_COUNT or level + 1 >= _MAX_LEVEL_COUNT:
                    break
                level_size = cluster_count_level
                level += 1
            world_cluster_count.append(cluster_count - world_cluster_begin)

        return {
            "ancestors": ancestors,
            "cluster_dof_count": np.asarray(cluster_dof_count, dtype=np.int32),
            "cluster_world": np.asarray(cluster_world, dtype=np.int32),
            "slot_fine_begin": np.asarray(slot_fine_begin, dtype=np.int32),
            "slot_fine_count": np.asarray(slot_fine_count, dtype=np.int32),
            "slot_world": np.asarray(slot_world, dtype=np.int32),
            "world_cluster_offsets": np.asarray(world_cluster_offsets, dtype=np.int32),
            "world_cluster_count": np.asarray(world_cluster_count, dtype=np.int32),
            "level_count": max_level_count,
            "cluster_count": cluster_count,
        }

    @property
    def storage_bytes(self) -> int:
        """Return GPU bytes used by the hierarchy, factors, and work vectors."""
        arrays = [
            self.ancestors,
            self.cluster_dof_count,
            self.cluster_world,
            self.world_cluster_offsets,
            self.world_cluster_count,
            self.slot_fine_begin,
            self.slot_fine_count,
            self.slot_world,
            self.factors,
            self.hierarchy_r,
            self.hierarchy_z,
        ]
        if self.apply_mode in ("fp32_tile", "bf16_tile"):
            arrays.extend((self.hierarchy_r_tensor, self.hierarchy_z_tensor))
        if self.apply_mode in ("bf16_tile", "bf16_vector"):
            arrays.append(self.factors_bf16)
        if self.apply_mode == "bf16_vector":
            arrays.append(self.hierarchy_r_bf16_vector)
        return sum(array.size * wp.types.type_size_in_bytes(array.dtype) for array in arrays)

    def refit(self) -> None:
        """Reassemble and refactor numeric blocks after sparse values or pattern change."""
        wp.launch(
            _initialize_cluster_kernel,
            dim=(self.cluster_count, _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
            inputs=[self.cluster_dof_count, wp.float32(self.regularization), self.factors],
            device=self.device,
        )
        wp.launch(
            _scatter_hierarchy_kernel,
            dim=self.matrix.max_nnz_count,
            inputs=[
                self.matrix.block_rows,
                self.matrix.block_slots,
                self.matrix.row_nnz,
                self.matrix.col_indices,
                self.matrix.values,
                self.ancestors,
                self.factors,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            _factorize_cluster_kernel,
            dim=self.cluster_count,
            inputs=[self.factors],
            block_dim=128,
            device=self.device,
        )
        if self.apply_mode in ("bf16_tile", "bf16_vector"):
            wp.launch(
                _convert_factors_bf16_kernel,
                dim=(self.cluster_count * _CLUSTER_DOF_COUNT, _CLUSTER_DOF_COUNT),
                inputs=[self.factors, self.factors_bf16],
                device=self.device,
            )

    def apply(self, r: wp.array, z: wp.array, world_active: wp.array) -> None:
        """Apply the preconditioner, ``z = M^-1 r``."""
        if self.apply_mode == "fp32_async":
            wp.launch_tiled(
                _apply_cluster_inverse_fp32_async_kernel,
                dim=self.cluster_count,
                inputs=[
                    self.factors,
                    self.cluster_world,
                    world_active,
                    r,
                    self.slot_fine_begin,
                    self.slot_fine_count,
                    self.hierarchy_z,
                ],
                block_dim=256,
                device=self.device,
            )
            wp.launch(
                _prolongate_kernel,
                dim=(self.matrix.total_row_count, 3),
                inputs=[self.ancestors, self.matrix.fine_node_world, world_active, self.hierarchy_z, z],
                device=self.device,
            )
            return
        if self.apply_mode == "bf16_vector":
            wp.launch(
                _restrict_residual_bf16_vector_kernel,
                dim=(self.cluster_count * _CLUSTER_NODE_COUNT, 3),
                inputs=[
                    r,
                    self.slot_fine_begin,
                    self.slot_fine_count,
                    self.slot_world,
                    world_active,
                    self.hierarchy_r_bf16_vector,
                ],
                device=self.device,
            )
            wp.launch_tiled(
                _apply_cluster_inverse_bf16_vector_kernel,
                dim=self.cluster_count,
                inputs=[
                    self.factors_bf16,
                    self.cluster_world,
                    world_active,
                    self.hierarchy_r_bf16_vector,
                    self.hierarchy_z,
                ],
                block_dim=192,
                device=self.device,
            )
            wp.launch(
                _prolongate_kernel,
                dim=(self.matrix.total_row_count, 3),
                inputs=[self.ancestors, self.matrix.fine_node_world, world_active, self.hierarchy_z, z],
                device=self.device,
            )
            return
        if self.apply_mode in ("fp32_tile", "bf16_tile"):
            restrict_kernel = (
                _restrict_residual_bf16_kernel
                if self.apply_mode == "bf16_tile"
                else _restrict_residual_tensor_f32_kernel
            )
            wp.launch(
                restrict_kernel,
                dim=(self.cluster_count * _CLUSTER_NODE_COUNT, 3),
                inputs=[
                    r,
                    self.slot_fine_begin,
                    self.slot_fine_count,
                    self.slot_world,
                    world_active,
                    self.hierarchy_r_tensor,
                ],
                device=self.device,
            )
            apply_kernel = (
                _apply_cluster_inverse_bf16_kernel
                if self.apply_mode == "bf16_tile"
                else _apply_cluster_inverse_tensor_f32_kernel
            )
            factors = self.factors_bf16 if self.apply_mode == "bf16_tile" else self.factors
            wp.launch_tiled(
                apply_kernel,
                dim=self.cluster_count,
                inputs=[
                    factors,
                    self.cluster_world,
                    world_active,
                    self.hierarchy_r_tensor,
                    self.hierarchy_z_tensor,
                ],
                block_dim=128,
                device=self.device,
            )
            wp.launch(
                _prolongate_bf16_kernel,
                dim=(self.matrix.total_row_count, 3),
                inputs=[self.ancestors, self.matrix.fine_node_world, world_active, self.hierarchy_z_tensor, z],
                device=self.device,
            )
            return
        wp.launch(
            _restrict_residual_kernel,
            dim=(self.cluster_count * _CLUSTER_NODE_COUNT, 3),
            inputs=[r, self.slot_fine_begin, self.slot_fine_count, self.slot_world, world_active, self.hierarchy_r],
            device=self.device,
        )
        wp.launch(
            _apply_cluster_inverse_kernel,
            dim=(self.cluster_count, _CLUSTER_DOF_COUNT),
            inputs=[
                self.factors,
                self.cluster_world,
                world_active,
                self.hierarchy_r,
                self.hierarchy_z,
            ],
            device=self.device,
        )
        wp.launch(
            _prolongate_kernel,
            dim=(self.matrix.total_row_count, 3),
            inputs=[self.ancestors, self.matrix.fine_node_world, world_active, self.hierarchy_z, z],
            device=self.device,
        )


class _BSRFastCGSolver(CGSolver):
    """CG specialization that fuses BSR SpMV with its dot reduction."""

    def __init__(self, matrix: BatchedBSRMatrix, *args, **kwargs):
        self.bsr_matrix = matrix
        super().__init__(*args, **kwargs)

    def _allocate(self):
        super()._allocate()
        self.p_ap = wp.zeros(self.n_worlds, dtype=wp.float32, device=self.device)

    def solve(self, *args, **kwargs):
        self.p_ap.zero_()
        return super().solve(*args, **kwargs)

    def do_iteration(self, p, Ap, rz_old, rz_new, z, x, r, r_norm_sq, active_dims, world_active):
        chunks = (self.bsr_matrix.max_row_count + 255) // 256
        wp.launch_tiled(
            _bsr_spmv_dot_kernel,
            dim=(self.n_worlds, chunks),
            inputs=[
                self.bsr_matrix.row_offsets,
                self.bsr_matrix.row_nnz,
                self.bsr_matrix.col_indices,
                self.bsr_matrix.values,
                self.bsr_matrix.world_row_offsets,
                self.bsr_matrix.world_row_count,
                world_active,
                p,
                Ap,
                self.p_ap,
            ],
            block_dim=256,
            device=self.device,
        )
        wp.launch(
            _cg_update_xr_save_kernel,
            dim=(self.n_worlds, self.maxdims),
            inputs=[
                self.atol_sq,
                r_norm_sq,
                rz_old,
                rz_new,
                self.p_ap,
                p,
                Ap,
                x,
                r,
                self.vio,
                active_dims,
            ],
            device=self.device,
        )
        self.update_rr_rz(r, z, self.r_repeated, active_dims, world_active)
        wp.launch(
            _cg_update_p_reset_kernel,
            dim=(self.n_worlds, self.maxdims),
            inputs=[
                self.atol_sq,
                r_norm_sq,
                rz_old,
                rz_new,
                z,
                p,
                self.p_ap,
                self.vio,
                active_dims,
            ],
            device=self.device,
        )


class BatchedMASPCG:
    """Float32 batched PCG using sparse BSR and multilevel additive Schwarz."""

    def __init__(
        self,
        matrix: BatchedBSRMatrix,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-5,
        max_iterations: int | None = None,
        use_cuda_graph: bool = True,
        loop_granularity: int = 1,
        regularization: float = 1.0e-6,
        one_block_max_rows: int = 0,
        fused_spmv_min_rows: int = 8192,
        mas_apply_mode: str = "fp32_async",
        refinement_passes: int = 1,
    ):
        self.matrix = matrix
        if refinement_passes < 1:
            raise ValueError("refinement_passes must be at least one")
        self.refinement_passes = int(refinement_passes)
        if mas_apply_mode in ("fp32_tile", "bf16_tile", "bf16_vector") and one_block_max_rows > 0:
            raise ValueError("mixed-precision preconditioning is incompatible with one-block PCG")
        if refinement_passes > 1 and one_block_max_rows > 0:
            raise ValueError("reliable refinement is incompatible with one-block PCG")
        self.preconditioner = MASPreconditioner(
            matrix,
            regularization=regularization,
            apply_mode=mas_apply_mode,
        )
        self.world_active = wp.full(matrix.world_count, True, dtype=wp.bool, device=matrix.device)
        max_iterations = max_iterations or max(16, 3 * matrix.max_row_count)
        maxiter = wp.full(matrix.world_count, max_iterations, dtype=wp.int32, device=matrix.device)
        self.max_iterations = maxiter
        self.atol = wp.full(matrix.world_count, float(atol), dtype=wp.float32, device=matrix.device)
        self.rtol = wp.full(matrix.world_count, float(rtol), dtype=wp.float32, device=matrix.device)

        def gemv(x, y, active, alpha, beta):
            matrix.gemv(x, y, active, alpha, beta)

        def matvec(x, y, active):
            matrix.gemv(x, y, active)

        operator = BatchedLinearOperator(
            gemv,
            matrix.world_count,
            3 * matrix.max_row_count,
            matrix.active_dims,
            matrix.device,
            wp.float32,
            matvec_fn=matvec,
            vio=matrix.world_scalar_offsets,
            total_vec_size=matrix.total_scalar_count,
        )

        def precondition(r, z, active):
            self.preconditioner.apply(r, z, active)

        inverse = BatchedLinearOperator(
            precondition,
            matrix.world_count,
            3 * matrix.max_row_count,
            matrix.active_dims,
            matrix.device,
            wp.float32,
            matvec_fn=precondition,
            vio=matrix.world_scalar_offsets,
            total_vec_size=matrix.total_scalar_count,
        )
        solver_kwargs = {
            "world_active": self.world_active,
            "atol": float(atol),
            "rtol": float(rtol),
            "maxiter": maxiter,
            "Mi": inverse,
            "use_graph": use_cuda_graph,
            "loop_granularity": loop_granularity,
        }
        if matrix.max_row_count >= fused_spmv_min_rows:
            self._solver = _BSRFastCGSolver(matrix, operator, **solver_kwargs)
        else:
            self._solver = CGSolver(operator, **solver_kwargs)
        self._one_block = matrix.max_row_count <= one_block_max_rows
        self.iterations = self._solver.cur_iter
        self.residual = self._solver.residual
        self._total_iterations = None
        if self.refinement_passes > 1:
            self._total_iterations = wp.zeros(matrix.world_count, dtype=wp.int32, device=matrix.device)
            self.iterations = self._total_iterations
        if self._one_block:
            self._one_block_kernel = _make_one_block_pcg_kernel(256)
            self._fused_r = wp.zeros(matrix.total_scalar_count, dtype=wp.float32, device=matrix.device)
            self._fused_z = wp.zeros_like(self._fused_r)
            self._fused_p = wp.zeros_like(self._fused_r)
            self._fused_ap = wp.zeros_like(self._fused_r)
            self.iterations = wp.zeros(matrix.world_count, dtype=wp.int32, device=matrix.device)
            self.residual = wp.zeros(matrix.world_count, dtype=wp.float32, device=matrix.device)

    @property
    def storage_bytes(self) -> int:
        """Return matrix and preconditioner storage, excluding PCG work vectors."""
        return self.matrix.storage_bytes + self.preconditioner.storage_bytes

    def solve(
        self,
        b: wp.array,
        x: wp.array,
        *,
        world_active: wp.array | None = None,
        refit: bool = True,
    ):
        """Solve ``A x = b`` and return per-world iteration and residual arrays."""
        active = self.world_active if world_active is None else world_active
        if refit:
            self.preconditioner.refit()
        if self._one_block:
            wp.launch_tiled(
                self._one_block_kernel,
                dim=self.matrix.world_count,
                inputs=[
                    self.matrix.row_offsets,
                    self.matrix.row_nnz,
                    self.matrix.col_indices,
                    self.matrix.values,
                    self.matrix.world_row_offsets,
                    self.matrix.world_row_count,
                    self.matrix.world_scalar_offsets,
                    self.preconditioner.world_cluster_offsets,
                    self.preconditioner.world_cluster_count,
                    active,
                    self.preconditioner.ancestors,
                    self.preconditioner.slot_fine_begin,
                    self.preconditioner.slot_fine_count,
                    self.preconditioner.factors,
                    self.preconditioner.hierarchy_r,
                    self.preconditioner.hierarchy_z,
                    b,
                    x,
                    self._fused_r,
                    self._fused_z,
                    self._fused_p,
                    self._fused_ap,
                    self.atol,
                    self.rtol,
                    self.max_iterations,
                    self.iterations,
                    self.residual,
                ],
                block_dim=256,
                device=self.matrix.device,
            )
            return self.iterations, self.residual
        if self.refinement_passes == 1:
            return self._solver.solve(b, x, world_active=active)
        self._total_iterations.zero_()
        for _pass in range(self.refinement_passes):
            self._solver.solve(b, x, world_active=active)
            wp.launch(
                _accumulate_iterations_kernel,
                dim=self.matrix.world_count,
                inputs=[self._solver.cur_iter, self._total_iterations],
                device=self.matrix.device,
            )
        return self.iterations, self.residual

    def capture(
        self,
        b: wp.array,
        x: wp.array,
        *,
        refit: bool = True,
        zero_initial_guess: bool = True,
    ) -> wp.Graph:
        """Capture numeric refit and the device-conditional PCG solve as one graph."""
        with wp.ScopedCapture(self.matrix.device) as capture:
            if zero_initial_guess:
                x.zero_()
            self.solve(b, x, refit=refit)
        return capture.graph
