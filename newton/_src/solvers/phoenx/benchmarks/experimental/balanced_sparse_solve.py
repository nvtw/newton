# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental warp-fused balanced sparse articulation solve."""

import warp as wp

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem

_BLOCK_SIZE = 6


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _sync_warp(): ...


@wp.func
def _factor_pivot(
    pivot: wp.int32,
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    pred_diag_ptr: wp.array[wp.int32],
    pred_diag_slot: wp.array[wp.int32],
    pred_off_ptr: wp.array[wp.int32],
    pred_off_slot_ik: wp.array[wp.int32],
    pred_off_slot_jk: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    block_diag: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    factor_off: wp.array3d[wp.float32],
):
    pivot_size = block_sizes[pivot]

    pred_diag_start = pred_diag_ptr[pivot]
    pred_diag_end = pred_diag_ptr[pivot + wp.int32(1)]
    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            value = wp.float32(0.0)
            if wp.int32(i) < pivot_size and wp.int32(j) < pivot_size:
                value = block_diag[pivot, i, j]
                for pred in range(pred_diag_start, pred_diag_end):
                    pred_slot = pred_diag_slot[pred]
                    for k in range(_BLOCK_SIZE):
                        value -= factor_off[pred_slot, i, k] * factor_off[pred_slot, j, k]
            elif i == j:
                value = wp.float32(1.0)
            factor_diag[pivot, i, j] = value

    for k in range(_BLOCK_SIZE):
        if wp.int32(k) < pivot_size:
            diag = factor_diag[pivot, k, k]
            if diag < wp.float32(1.0e-20):
                diag = wp.float32(1.0e-20)
            diag = wp.sqrt(diag)
            factor_diag[pivot, k, k] = diag
            inv_diag = wp.float32(1.0) / diag

            for i in range(_BLOCK_SIZE):
                if wp.int32(i) > wp.int32(k) and wp.int32(i) < pivot_size:
                    factor_diag[pivot, i, k] = factor_diag[pivot, i, k] * inv_diag

            for j in range(_BLOCK_SIZE):
                if wp.int32(j) > wp.int32(k) and wp.int32(j) < pivot_size:
                    l_jk = factor_diag[pivot, j, k]
                    for i in range(_BLOCK_SIZE):
                        if wp.int32(i) >= wp.int32(j) and wp.int32(i) < pivot_size:
                            factor_diag[pivot, i, j] -= factor_diag[pivot, i, k] * l_jk

    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            if wp.int32(i) < pivot_size and wp.int32(j) < pivot_size:
                if wp.int32(i) < wp.int32(j):
                    factor_diag[pivot, i, j] = wp.float32(0.0)
            elif i == j:
                factor_diag[pivot, i, j] = wp.float32(1.0)
            else:
                factor_diag[pivot, i, j] = wp.float32(0.0)

    col_start = l_col_ptr[pivot]
    col_end = l_col_ptr[pivot + wp.int32(1)]
    for ptr in range(col_start, col_end):
        row_pivot = l_row_idx[ptr]
        row_size = block_sizes[row_pivot]
        pred_off_start = pred_off_ptr[ptr]
        pred_off_end = pred_off_ptr[ptr + wp.int32(1)]

        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                value = wp.float32(0.0)
                if wp.int32(i) < row_size and wp.int32(j) < pivot_size:
                    value = factor_off[ptr, i, j]
                    for pred in range(pred_off_start, pred_off_end):
                        slot_ik = pred_off_slot_ik[pred]
                        slot_jk = pred_off_slot_jk[pred]
                        for k in range(_BLOCK_SIZE):
                            value -= factor_off[slot_ik, i, k] * factor_off[slot_jk, j, k]
                factor_off[ptr, i, j] = value

        for i in range(_BLOCK_SIZE):
            if wp.int32(i) < row_size:
                for j in range(_BLOCK_SIZE):
                    if wp.int32(j) < pivot_size:
                        value = factor_off[ptr, i, j]
                        for k in range(_BLOCK_SIZE):
                            if wp.int32(k) < wp.int32(j):
                                value -= factor_diag[pivot, j, k] * factor_off[ptr, i, k]
                        factor_off[ptr, i, j] = value / factor_diag[pivot, j, j]
                    else:
                        factor_off[ptr, i, j] = wp.float32(0.0)
            else:
                for j in range(_BLOCK_SIZE):
                    factor_off[ptr, i, j] = wp.float32(0.0)


@wp.func
def _forward_pivot(
    pivot: wp.int32,
    l_row_ptr: wp.array[wp.int32],
    l_col_idx: wp.array[wp.int32],
    l_csr_to_csc: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    factor_off: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    block_rhs: wp.array2d[wp.float32],
    block_y: wp.array2d[wp.float32],
):
    pivot_size = block_sizes[pivot]
    row_start = l_row_ptr[pivot]
    row_end = l_row_ptr[pivot + wp.int32(1)]

    for r in range(_BLOCK_SIZE):
        if wp.int32(r) < pivot_size:
            value = block_rhs[pivot, r]
            for row_ptr in range(row_start, row_end):
                col = l_col_idx[row_ptr]
                slot = l_csr_to_csc[row_ptr]
                col_size = block_sizes[col]
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < col_size:
                        value -= factor_off[slot, r, c] * block_y[col, c]
            for c in range(_BLOCK_SIZE):
                if wp.int32(c) < wp.int32(r):
                    value -= factor_diag[pivot, r, c] * block_y[pivot, c]
            block_y[pivot, r] = value / factor_diag[pivot, r, r]
        else:
            block_y[pivot, r] = wp.float32(0.0)


@wp.func
def _backward_pivot(
    pivot: wp.int32,
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    factor_off: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    block_y: wp.array2d[wp.float32],
    block_solution: wp.array2d[wp.float32],
):
    pivot_size = block_sizes[pivot]
    col_start = l_col_ptr[pivot]
    col_end = l_col_ptr[pivot + wp.int32(1)]

    for rev in range(_BLOCK_SIZE):
        r = _BLOCK_SIZE - 1 - rev
        if wp.int32(r) < pivot_size:
            value = block_y[pivot, r]
            for ptr in range(col_start, col_end):
                row_pivot = l_row_idx[ptr]
                row_size = block_sizes[row_pivot]
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < row_size:
                        value -= factor_off[ptr, c, r] * block_solution[row_pivot, c]
            for c in range(_BLOCK_SIZE):
                if wp.int32(c) > wp.int32(r) and wp.int32(c) < pivot_size:
                    value -= factor_diag[pivot, c, r] * block_solution[pivot, c]
            block_solution[pivot, r] = value / factor_diag[pivot, r, r]
        else:
            block_solution[pivot, r] = wp.float32(0.0)


@wp.kernel
def _factor_fused_kernel(
    block_off: wp.array3d[wp.float32],
    n_off_to_l: wp.array[wp.int32],
    block_nnz: wp.int32,
    block_l_nnz: wp.int32,
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    pred_diag_ptr: wp.array[wp.int32],
    pred_diag_slot: wp.array[wp.int32],
    pred_off_ptr: wp.array[wp.int32],
    pred_off_slot_ik: wp.array[wp.int32],
    pred_off_slot_jk: wp.array[wp.int32],
    level_ptr: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    block_diag: wp.array3d[wp.float32],
    num_levels: wp.int32,
    factor_diag: wp.array3d[wp.float32],
    factor_off: wp.array3d[wp.float32],
):
    lane = wp.tid()
    slot = lane
    while slot < block_l_nnz:
        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                factor_off[slot, i, j] = wp.float32(0.0)
        slot += wp.int32(32)
    _sync_warp()
    slot = lane
    while slot < block_nnz:
        dst = n_off_to_l[slot]
        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                factor_off[dst, i, j] = block_off[slot, i, j]
        slot += wp.int32(32)
    _sync_warp()

    reverse_level = wp.int32(0)
    while reverse_level < num_levels:
        level = num_levels - wp.int32(1) - reverse_level
        offset = level_ptr[level]
        count = level_ptr[level + wp.int32(1)] - offset
        local = lane
        while local < count:
            _factor_pivot(
                level_pivots[offset + local],
                l_col_ptr,
                l_row_idx,
                pred_diag_ptr,
                pred_diag_slot,
                pred_off_ptr,
                pred_off_slot_ik,
                pred_off_slot_jk,
                block_sizes,
                block_diag,
                factor_diag,
                factor_off,
            )
            local += wp.int32(32)
        _sync_warp()
        reverse_level += wp.int32(1)


@wp.kernel
def _solve_fused_kernel(
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    l_row_ptr: wp.array[wp.int32],
    l_col_idx: wp.array[wp.int32],
    l_csr_to_csc: wp.array[wp.int32],
    level_ptr: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    factor_off: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    block_rhs: wp.array2d[wp.float32],
    num_levels: wp.int32,
    block_y: wp.array2d[wp.float32],
    block_solution: wp.array2d[wp.float32],
):
    lane = wp.tid()
    reverse_level = wp.int32(0)
    while reverse_level < num_levels:
        level = num_levels - wp.int32(1) - reverse_level
        offset = level_ptr[level]
        count = level_ptr[level + wp.int32(1)] - offset
        local = lane
        while local < count:
            _forward_pivot(
                level_pivots[offset + local],
                l_row_ptr,
                l_col_idx,
                l_csr_to_csc,
                block_sizes,
                factor_off,
                factor_diag,
                block_rhs,
                block_y,
            )
            local += wp.int32(32)
        _sync_warp()
        reverse_level += wp.int32(1)

    level = wp.int32(0)
    while level < num_levels:
        offset = level_ptr[level]
        count = level_ptr[level + wp.int32(1)] - offset
        local = lane
        while local < count:
            _backward_pivot(
                level_pivots[offset + local],
                l_col_ptr,
                l_row_idx,
                block_sizes,
                factor_off,
                factor_diag,
                block_y,
                block_solution,
            )
            local += wp.int32(32)
        _sync_warp()
        level += wp.int32(1)


def solve_balanced_sparse_matrix(system: ArticulationDeviceSystem, *, device=None) -> None:
    """Factor and solve a balanced sparse system in four launches."""
    wp.launch(
        _factor_fused_kernel,
        dim=32,
        block_dim=32,
        inputs=[
            system.block_off,
            system.block_n_off_to_l,
            wp.int32(system.block_nnz),
            wp.int32(system.block_l_nnz),
            system.block_l_col_ptr,
            system.block_l_row_idx,
            system.block_pred_diag_ptr,
            system.block_pred_diag_slot,
            system.block_pred_off_ptr,
            system.block_pred_off_slot_ik,
            system.block_pred_off_slot_jk,
            system.block_level_ptr,
            system.block_level_pivots,
            system.block_sizes,
            system.block_diag,
            wp.int32(system.block_num_levels),
        ],
        outputs=[system.block_factor_diag, system.block_factor_off],
        device=device,
    )
    solve_balanced_sparse_factors(system, device=device)


def solve_balanced_sparse_factors(system: ArticulationDeviceSystem, *, device=None) -> None:
    """Solve existing balanced sparse factors in three launches."""
    system.gather_block_rhs(device=device)
    wp.launch(
        _solve_fused_kernel,
        dim=32,
        block_dim=32,
        inputs=[
            system.block_l_col_ptr,
            system.block_l_row_idx,
            system.block_l_row_ptr,
            system.block_l_col_idx,
            system.block_l_csr_to_csc,
            system.block_level_ptr,
            system.block_level_pivots,
            system.block_sizes,
            system.block_factor_off,
            system.block_factor_diag,
            system.block_rhs,
            wp.int32(system.block_num_levels),
        ],
        outputs=[system.block_y, system.block_solution],
        device=device,
    )
    system.scatter_block_solution(device=device)
