# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental factor-2 Galerkin block-GS path correction."""

import warp as wp

from newton._src.solvers.phoenx.articulations.device import ArticulationDeviceSystem

_BLOCK_SIZE = 6
_BLOCK_DIM = 128


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func
def _coarse_left(fine: wp.int32, fine_blocks: wp.int32) -> wp.int32:
    if fine + wp.int32(1) == fine_blocks and fine % wp.int32(2) == wp.int32(1):
        return (fine + wp.int32(1)) / wp.int32(2)
    return fine / wp.int32(2)


@wp.func
def _accumulate_coarse_edge(
    coarse_row: wp.int32,
    coarse_col: wp.int32,
    weight: wp.float32,
    fine_edge: wp.int32,
    rows: wp.int32,
    fine_off: wp.array3d[wp.float32],
    coarse_diag: wp.array3d[wp.float32],
    coarse_off: wp.array3d[wp.float32],
):
    for r in range(_BLOCK_SIZE):
        if wp.int32(r) < rows:
            for c in range(_BLOCK_SIZE):
                if wp.int32(c) < rows:
                    value = weight * fine_off[fine_edge, r, c]
                    if coarse_row == coarse_col:
                        wp.atomic_add(coarse_diag, coarse_row, r, c, value)
                        wp.atomic_add(coarse_diag, coarse_row, c, r, value)
                    elif coarse_row > coarse_col:
                        wp.atomic_add(coarse_off, coarse_col, r, c, value)
                    else:
                        wp.atomic_add(coarse_off, coarse_row, c, r, value)


@wp.kernel
def _assemble_coarse_path_kernel(
    fine_diag: wp.array3d[wp.float32],
    fine_off: wp.array3d[wp.float32],
    fine_rhs: wp.array2d[wp.float32],
    fine_blocks: wp.int32,
    coarse_blocks: wp.int32,
    rows: wp.int32,
    coarse_diag: wp.array3d[wp.float32],
    coarse_off: wp.array3d[wp.float32],
    coarse_rhs: wp.array2d[wp.float32],
    coarse_solution: wp.array2d[wp.float32],
):
    lane = wp.tid()
    if lane < coarse_blocks:
        for r in range(_BLOCK_SIZE):
            coarse_rhs[lane, r] = wp.float32(0.0)
            coarse_solution[lane, r] = wp.float32(0.0)
            for c in range(_BLOCK_SIZE):
                coarse_diag[lane, r, c] = wp.float32(0.0)
                if lane + wp.int32(1) < coarse_blocks:
                    coarse_off[lane, r, c] = wp.float32(0.0)
    _sync_threads()

    if lane < fine_blocks:
        left = _coarse_left(lane, fine_blocks)
        has_right = lane % wp.int32(2) == wp.int32(1) and lane + wp.int32(1) < fine_blocks
        left_weight = wp.float32(0.5) if has_right else wp.float32(1.0)
        right_weight = wp.float32(0.5)
        for r in range(_BLOCK_SIZE):
            if wp.int32(r) < rows:
                wp.atomic_add(coarse_rhs, left, r, left_weight * fine_rhs[lane, r])
                if has_right:
                    wp.atomic_add(coarse_rhs, left + wp.int32(1), r, right_weight * fine_rhs[lane, r])
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < rows:
                        value = fine_diag[lane, r, c]
                        wp.atomic_add(coarse_diag, left, r, c, left_weight * left_weight * value)
                        if has_right:
                            wp.atomic_add(
                                coarse_diag,
                                left + wp.int32(1),
                                r,
                                c,
                                right_weight * right_weight * value,
                            )
                            wp.atomic_add(coarse_off, left, r, c, left_weight * right_weight * value)

    if lane + wp.int32(1) < fine_blocks:
        row = lane + wp.int32(1)
        col = lane
        row_left = _coarse_left(row, fine_blocks)
        col_left = _coarse_left(col, fine_blocks)
        row_has_right = row % wp.int32(2) == wp.int32(1) and row + wp.int32(1) < fine_blocks
        col_has_right = col % wp.int32(2) == wp.int32(1) and col + wp.int32(1) < fine_blocks
        row_weight = wp.float32(0.5) if row_has_right else wp.float32(1.0)
        col_weight = wp.float32(0.5) if col_has_right else wp.float32(1.0)
        _accumulate_coarse_edge(
            row_left,
            col_left,
            row_weight * col_weight,
            lane,
            rows,
            fine_off,
            coarse_diag,
            coarse_off,
        )
        if col_has_right:
            _accumulate_coarse_edge(
                row_left,
                col_left + wp.int32(1),
                row_weight * wp.float32(0.5),
                lane,
                rows,
                fine_off,
                coarse_diag,
                coarse_off,
            )
        if row_has_right:
            _accumulate_coarse_edge(
                row_left + wp.int32(1),
                col_left,
                wp.float32(0.5) * col_weight,
                lane,
                rows,
                fine_off,
                coarse_diag,
                coarse_off,
            )
        if row_has_right and col_has_right:
            _accumulate_coarse_edge(
                row_left + wp.int32(1),
                col_left + wp.int32(1),
                wp.float32(0.25),
                lane,
                rows,
                fine_off,
                coarse_diag,
                coarse_off,
            )


@wp.kernel
def _solve_coarse_path_kernel(
    coarse_diag: wp.array3d[wp.float32],
    coarse_off: wp.array3d[wp.float32],
    coarse_rhs: wp.array2d[wp.float32],
    coarse_blocks: wp.int32,
    rows: wp.int32,
    color_sweeps: wp.int32,
    factor_diag: wp.array3d[wp.float32],
    work: wp.array2d[wp.float32],
    solution: wp.array2d[wp.float32],
):
    node = wp.tid()
    if node < coarse_blocks:
        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                value = wp.float32(0.0)
                if wp.int32(i) < rows and wp.int32(j) < rows:
                    value = coarse_diag[node, i, j]
                elif i == j:
                    value = wp.float32(1.0)
                factor_diag[node, i, j] = value
        for k in range(_BLOCK_SIZE):
            if wp.int32(k) < rows:
                diag = wp.sqrt(wp.max(factor_diag[node, k, k], wp.float32(1.0e-20)))
                factor_diag[node, k, k] = diag
                for i in range(_BLOCK_SIZE):
                    if wp.int32(i) > wp.int32(k) and wp.int32(i) < rows:
                        factor_diag[node, i, k] /= diag
                for j in range(_BLOCK_SIZE):
                    if wp.int32(j) > wp.int32(k) and wp.int32(j) < rows:
                        l_jk = factor_diag[node, j, k]
                        for i in range(_BLOCK_SIZE):
                            if wp.int32(i) >= wp.int32(j) and wp.int32(i) < rows:
                                factor_diag[node, i, j] -= factor_diag[node, i, k] * l_jk
    _sync_threads()

    sweep = wp.int32(0)
    while sweep < color_sweeps:
        color = sweep % wp.int32(2)
        if node < coarse_blocks and node % wp.int32(2) == color:
            for r in range(_BLOCK_SIZE):
                if wp.int32(r) < rows:
                    value = coarse_rhs[node, r]
                    if node > wp.int32(0):
                        for c in range(_BLOCK_SIZE):
                            if wp.int32(c) < rows:
                                value -= coarse_off[node - wp.int32(1), r, c] * solution[node - wp.int32(1), c]
                    if node + wp.int32(1) < coarse_blocks:
                        for c in range(_BLOCK_SIZE):
                            if wp.int32(c) < rows:
                                value -= coarse_off[node, c, r] * solution[node + wp.int32(1), c]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) < wp.int32(r):
                            value -= factor_diag[node, r, c] * work[node, c]
                    work[node, r] = value / factor_diag[node, r, r]
            for reverse in range(_BLOCK_SIZE):
                r = _BLOCK_SIZE - 1 - reverse
                if wp.int32(r) < rows:
                    value = work[node, r]
                    for c in range(_BLOCK_SIZE):
                        if wp.int32(c) > wp.int32(r) and wp.int32(c) < rows:
                            value -= factor_diag[node, c, r] * solution[node, c]
                    solution[node, r] = value / factor_diag[node, r, r]
        _sync_threads()
        sweep += wp.int32(1)


@wp.kernel
def _prolongate_coarse_path_kernel(
    coarse_solution: wp.array2d[wp.float32],
    fine_blocks: wp.int32,
    rows: wp.int32,
    fine_solution: wp.array[wp.float32],
):
    fine_row = wp.tid()
    fine = fine_row / rows
    row = fine_row - fine * rows
    if fine >= fine_blocks:
        return
    left = _coarse_left(fine, fine_blocks)
    has_right = fine % wp.int32(2) == wp.int32(1) and fine + wp.int32(1) < fine_blocks
    value = coarse_solution[left, row]
    if has_right:
        value = wp.float32(0.5) * (value + coarse_solution[left + wp.int32(1), row])
    fine_solution[fine_row] = value


class CoarsePathSolver:
    """Reusable buffers for an experimental factor-2 path correction."""

    def __init__(self, fine_blocks: int, rows: int, color_sweeps: int, device):
        self.fine_blocks = fine_blocks
        self.coarse_blocks = (fine_blocks + 2) // 2
        self.rows = rows
        self.color_sweeps = color_sweeps
        shape3 = (self.coarse_blocks, _BLOCK_SIZE, _BLOCK_SIZE)
        self.diag = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.off = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.factor = wp.zeros(shape3, dtype=wp.float32, device=device)
        self.rhs = wp.zeros((self.coarse_blocks, _BLOCK_SIZE), dtype=wp.float32, device=device)
        self.work = wp.zeros_like(self.rhs)
        self.solution = wp.zeros_like(self.rhs)

    def solve(self, system: ArticulationDeviceSystem, *, device=None) -> None:
        system.gather_block_rhs(device=device)
        wp.launch(
            _assemble_coarse_path_kernel,
            dim=_BLOCK_DIM,
            block_dim=_BLOCK_DIM,
            inputs=[
                system.block_diag,
                system.block_off,
                system.block_rhs,
                wp.int32(self.fine_blocks),
                wp.int32(self.coarse_blocks),
                wp.int32(self.rows),
            ],
            outputs=[self.diag, self.off, self.rhs, self.solution],
            device=device,
        )
        wp.launch(
            _solve_coarse_path_kernel,
            dim=_BLOCK_DIM,
            block_dim=_BLOCK_DIM,
            inputs=[
                self.diag,
                self.off,
                self.rhs,
                wp.int32(self.coarse_blocks),
                wp.int32(self.rows),
                wp.int32(self.color_sweeps),
            ],
            outputs=[self.factor, self.work, self.solution],
            device=device,
        )
        wp.launch(
            _prolongate_coarse_path_kernel,
            dim=self.fine_blocks * self.rows,
            inputs=[self.solution, wp.int32(self.fine_blocks), wp.int32(self.rows)],
            outputs=[system.solution],
            device=device,
        )
