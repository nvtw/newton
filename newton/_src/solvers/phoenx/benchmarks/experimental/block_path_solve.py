# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental one-thread block-path articulation solve."""

import warp as wp

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem

_BLOCK_SIZE = 6


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _sync_warp(): ...


@wp.kernel
def _solve_block_path_kernel(
    block_sizes: wp.array[wp.int32],
    block_diag: wp.array3d[wp.float32],
    block_off: wp.array3d[wp.float32],
    block_rhs: wp.array2d[wp.float32],
    block_count: wp.int32,
    factor_diag: wp.array3d[wp.float32],
    factor_off: wp.array3d[wp.float32],
    block_y: wp.array2d[wp.float32],
    block_solution: wp.array2d[wp.float32],
):
    lane = wp.tid()
    entries = wp.int32(_BLOCK_SIZE * _BLOCK_SIZE)

    for pivot in range(block_count):
        pivot_size = block_sizes[pivot]
        entry = lane
        while entry < entries:
            i = entry / wp.int32(_BLOCK_SIZE)
            j = entry - i * wp.int32(_BLOCK_SIZE)
            value = wp.float32(0.0)
            if i < pivot_size and j < pivot_size:
                value = block_diag[pivot, i, j]
                if pivot > wp.int32(0):
                    for k in range(_BLOCK_SIZE):
                        value -= factor_off[pivot - wp.int32(1), i, k] * factor_off[pivot - wp.int32(1), j, k]
            elif i == j:
                value = wp.float32(1.0)
            factor_diag[pivot, i, j] = value
            entry += wp.int32(32)
        _sync_warp()

        for k in range(_BLOCK_SIZE):
            if lane == wp.int32(k) and lane < pivot_size:
                factor_diag[pivot, k, k] = wp.sqrt(wp.max(factor_diag[pivot, k, k], wp.float32(1.0e-20)))
            _sync_warp()
            if lane > wp.int32(k) and lane < pivot_size:
                factor_diag[pivot, lane, k] /= factor_diag[pivot, k, k]
            _sync_warp()
            entry = lane
            while entry < entries:
                i = entry / wp.int32(_BLOCK_SIZE)
                j = entry - i * wp.int32(_BLOCK_SIZE)
                if i >= j and j > wp.int32(k) and i < pivot_size:
                    factor_diag[pivot, i, j] -= factor_diag[pivot, i, k] * factor_diag[pivot, j, k]
                entry += wp.int32(32)
            _sync_warp()

        entry = lane
        while entry < entries:
            i = entry / wp.int32(_BLOCK_SIZE)
            j = entry - i * wp.int32(_BLOCK_SIZE)
            if i < j:
                factor_diag[pivot, i, j] = wp.float32(0.0)
            entry += wp.int32(32)
        _sync_warp()

        if pivot + wp.int32(1) < block_count:
            next_size = block_sizes[pivot + wp.int32(1)]
            for j in range(_BLOCK_SIZE):
                if lane < next_size:
                    value = wp.float32(0.0)
                    if wp.int32(j) < pivot_size:
                        value = block_off[pivot, lane, j]
                        for k in range(_BLOCK_SIZE):
                            if wp.int32(k) < wp.int32(j):
                                value -= factor_diag[pivot, j, k] * factor_off[pivot, lane, k]
                        value /= factor_diag[pivot, j, j]
                    factor_off[pivot, lane, j] = value
                _sync_warp()

    if lane == wp.int32(0):
        for pivot in range(block_count):
            pivot_size = block_sizes[pivot]
            for r in range(_BLOCK_SIZE):
                if wp.int32(r) < pivot_size:
                    value = block_rhs[pivot, r]
                    if pivot > wp.int32(0):
                        previous_size = block_sizes[pivot - wp.int32(1)]
                        for c in range(_BLOCK_SIZE):
                            if wp.int32(c) < previous_size:
                                value -= factor_off[pivot - wp.int32(1), r, c] * block_y[pivot - wp.int32(1), c]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) < wp.int32(r):
                            value -= factor_diag[pivot, r, c] * block_y[pivot, c]
                    block_y[pivot, r] = value / factor_diag[pivot, r, r]
                else:
                    block_y[pivot, r] = wp.float32(0.0)

        for reverse in range(block_count):
            pivot = block_count - wp.int32(1) - reverse
            pivot_size = block_sizes[pivot]
            for reverse_row in range(_BLOCK_SIZE):
                r = _BLOCK_SIZE - 1 - reverse_row
                if wp.int32(r) < pivot_size:
                    value = block_y[pivot, r]
                    if pivot + wp.int32(1) < block_count:
                        next_size = block_sizes[pivot + wp.int32(1)]
                        for c in range(_BLOCK_SIZE):
                            if wp.int32(c) < next_size:
                                value -= factor_off[pivot, c, r] * block_solution[pivot + wp.int32(1), c]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) > wp.int32(r) and wp.int32(c) < pivot_size:
                            value -= factor_diag[pivot, c, r] * block_solution[pivot, c]
                    block_solution[pivot, r] = value / factor_diag[pivot, r, r]
                else:
                    block_solution[pivot, r] = wp.float32(0.0)


def solve_block_path_matrix(system: ArticulationDeviceSystem, *, device=None) -> None:
    """Factor and solve a naturally ordered block-path system."""
    system.gather_block_rhs(device=device)
    wp.launch(
        _solve_block_path_kernel,
        dim=32,
        block_dim=32,
        inputs=[
            system.block_sizes,
            system.block_diag,
            system.block_off,
            system.block_rhs,
            wp.int32(system.block_count),
        ],
        outputs=[
            system.block_factor_diag,
            system.block_factor_off,
            system.block_y,
            system.block_solution,
        ],
        device=device,
    )
    system.scatter_block_solution(device=device)
