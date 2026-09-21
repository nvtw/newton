# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parallel cyclic reduction for block-tridiagonal PhoenX systems."""

from __future__ import annotations

from ctypes import sizeof

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

BS = 16
E = BS * BS
_REFINEMENT_RELATIVE_TOLERANCE = 1.0e-4

_GET_ARRAY_PTR = "return (uint64_t)arr.data;"


@wp.func_native(_GET_ARRAY_PTR)
def _get_array_ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...
@wp.func
def _offset_ptr(a: wp.array[wp.float32], o: int) -> wp.uint64:
    return _get_array_ptr(a) + wp.uint64(o * wp.static(sizeof(wp.float32._type_)))


@wp.kernel(enable_backward=False)
def _factor_blocks_kernel(level: wp.int32, n: wp.int32, blocks: wp.array[wp.float32], factors: wp.array[wp.float32]):
    i, _lane = wp.tid()
    base = (level * n + i) * wp.int32(E)
    src = wp.array(ptr=_offset_ptr(blocks, base), shape=(BS, BS), dtype=wp.float32)
    dst = wp.array(ptr=_offset_ptr(factors, base), shape=(BS, BS), dtype=wp.float32)
    t = wp.tile_load(src, shape=(BS, BS), storage="shared")
    wp.tile_cholesky_inplace(t)
    wp.tile_store(dst, t)


@wp.kernel(enable_backward=False)
def _reduce_blocks_kernel(
    level: wp.int32,
    stride: wp.int32,
    n: wp.int32,
    segment_begin: wp.array[wp.int32],
    segment_end: wp.array[wp.int32],
    blocks: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    factors: wp.array[wp.float32],
    alpha_out: wp.array[wp.float32],
    gamma_out: wp.array[wp.float32],
):
    i, _lane = wp.tid()
    begin = segment_begin[i]
    end = segment_end[i]
    base = (level * n + i) * wp.int32(E)
    out = ((level + 1) * n + i) * wp.int32(E)
    bm = wp.array(ptr=_offset_ptr(blocks, base), shape=(BS, BS), dtype=wp.float32)
    bn = wp.array(ptr=_offset_ptr(blocks, out), shape=(BS, BS), dtype=wp.float32)
    ln = wp.array(ptr=_offset_ptr(lower, out), shape=(BS, BS), dtype=wp.float32)
    rn = wp.array(ptr=_offset_ptr(upper, out), shape=(BS, BS), dtype=wp.float32)
    ao = wp.array(
        ptr=_offset_ptr(alpha_out, level * n * wp.int32(E) + i * wp.int32(E)), shape=(BS, BS), dtype=wp.float32
    )
    go = wp.array(
        ptr=_offset_ptr(gamma_out, level * n * wp.int32(E) + i * wp.int32(E)), shape=(BS, BS), dtype=wp.float32
    )
    bnew = wp.tile_load(bm, shape=(BS, BS), storage="shared")
    lnew = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    rnew = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    az = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    gz = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    if i - stride >= begin:
        lm = wp.array(ptr=_offset_ptr(lower, base), shape=(BS, BS), dtype=wp.float32)
        fb = wp.array(
            ptr=_offset_ptr(factors, (level * n + i - stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32
        )
        rm = wp.array(ptr=_offset_ptr(upper, (level * n + i - stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        ll = wp.array(ptr=_offset_ptr(lower, (level * n + i - stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        lt = wp.tile_load(lm, shape=(BS, BS))
        ft = wp.tile_load(fb, shape=(BS, BS))
        at = wp.tile_transpose(lt)
        wp.tile_lower_solve_inplace(ft, at)
        wp.tile_upper_solve_inplace(wp.tile_transpose(ft), at)
        az = wp.tile_transpose(at)
        wp.tile_matmul(az, wp.tile_load(rm, shape=(BS, BS)), bnew, alpha=-1.0)
        wp.tile_matmul(az, wp.tile_load(ll, shape=(BS, BS)), lnew, alpha=-1.0)
    if i + stride < end:
        rm = wp.array(ptr=_offset_ptr(upper, base), shape=(BS, BS), dtype=wp.float32)
        fb = wp.array(
            ptr=_offset_ptr(factors, (level * n + i + stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32
        )
        lm = wp.array(ptr=_offset_ptr(lower, (level * n + i + stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        rr = wp.array(ptr=_offset_ptr(upper, (level * n + i + stride) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        rt = wp.tile_load(rm, shape=(BS, BS))
        ft = wp.tile_load(fb, shape=(BS, BS))
        gt = wp.tile_transpose(rt)
        wp.tile_lower_solve_inplace(ft, gt)
        wp.tile_upper_solve_inplace(wp.tile_transpose(ft), gt)
        gz = wp.tile_transpose(gt)
        wp.tile_matmul(gz, wp.tile_load(lm, shape=(BS, BS)), bnew, alpha=-1.0)
        wp.tile_matmul(gz, wp.tile_load(rr, shape=(BS, BS)), rnew, alpha=-1.0)
    wp.tile_store(bn, bnew)
    wp.tile_store(ln, lnew)
    wp.tile_store(rn, rnew)
    wp.tile_store(ao, az)
    wp.tile_store(go, gz)


@wp.kernel(enable_backward=False)
def _reduce_rhs_kernel(
    level: wp.int32,
    stride: wp.int32,
    n: wp.int32,
    solve: wp.bool,
    segment_begin: wp.array[wp.int32],
    segment_end: wp.array[wp.int32],
    alpha: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    factors: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    x: wp.array[wp.float32],
):
    i, _lane = wp.tid()
    begin = segment_begin[i]
    end = segment_end[i]
    base = (level * n + i) * BS
    dm = wp.array(ptr=_offset_ptr(rhs, base), shape=(BS, 1), dtype=wp.float32)
    v = wp.tile_load(dm, shape=(BS, 1), storage="shared")
    if i - stride >= begin:
        am = wp.array(ptr=_offset_ptr(alpha, (level * n + i) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        dl = wp.array(ptr=_offset_ptr(rhs, (level * n + i - stride) * BS), shape=(BS, 1), dtype=wp.float32)
        wp.tile_matmul(wp.tile_load(am, shape=(BS, BS)), wp.tile_load(dl, shape=(BS, 1)), v, alpha=-1.0)
    if i + stride < end:
        gm = wp.array(ptr=_offset_ptr(gamma, (level * n + i) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        dr = wp.array(ptr=_offset_ptr(rhs, (level * n + i + stride) * BS), shape=(BS, 1), dtype=wp.float32)
        wp.tile_matmul(wp.tile_load(gm, shape=(BS, BS)), wp.tile_load(dr, shape=(BS, 1)), v, alpha=-1.0)
    if solve:
        fb = wp.array(ptr=_offset_ptr(factors, ((level + 1) * n + i) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        xm = wp.array(ptr=_offset_ptr(x, i * BS), shape=(BS, 1), dtype=wp.float32)
        factor = wp.tile_load(fb, shape=(BS, BS))
        wp.tile_lower_solve_inplace(factor, v)
        wp.tile_upper_solve_inplace(wp.tile_transpose(factor), v)
        wp.tile_store(xm, v)
    else:
        dn = wp.array(ptr=_offset_ptr(rhs, ((level + 1) * n + i) * BS), shape=(BS, 1), dtype=wp.float32)
        wp.tile_store(dn, v)


# Iterative refinement needs the cancellation accuracy of an FP64 residual,
# while native FP64 products dominate this small kernel on GPUs with limited FP64 throughput.
# Keep the rounded sum in ``hi`` and accumulate both its rounding error and
# the FMA-recovered product error in ``lo``. Explicit CUDA rounding prevents
# compiler contraction from erasing either error term.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __fadd_rn(value, -__fmul_rn(left, right));
#else
    return value - left * right;
#endif
""")
def _compensated_product_hi(value: wp.float32, left: wp.float32, right: wp.float32) -> wp.float32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    float product = __fmul_rn(left, right);
    float term = -product;
    float recovered = __fsub_rn(updated, value);
    float sum_error = __fadd_rn(
        __fsub_rn(value, __fsub_rn(updated, recovered)),
        __fsub_rn(term, recovered));
    float product_error = __fmaf_rn(left, right, -product);
    return __fsub_rn(__fadd_rn(error, sum_error), product_error);
#else
    float product = left * right;
    float term = -product;
    float recovered = updated - value;
    float sum_error = (value - (updated - recovered)) + (term - recovered);
    float product_error = fma(left, right, -product);
    return (error + sum_error) - product_error;
#endif
""")
def _compensated_product_lo(
    value: wp.float32,
    error: wp.float32,
    left: wp.float32,
    right: wp.float32,
    updated: wp.float32,
) -> wp.float32: ...


@wp.func
def _subtract_compensated_product(value: wp.vec2f, left: wp.float32, right: wp.float32) -> wp.vec2f:
    updated = _compensated_product_hi(value[0], left, right)
    error = _compensated_product_lo(value[0], value[1], left, right, updated)
    return wp.vec2f(updated, error)


@wp.func
def _row_residual(
    index: wp.int32,
    segment_begin: wp.array[wp.int32],
    segment_end: wp.array[wp.int32],
    blocks: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    source: wp.array[wp.float32],
    x: wp.array[wp.float32],
) -> wp.float32:
    block = index // wp.int32(BS)
    row = index % wp.int32(BS)
    begin = segment_begin[block]
    end = segment_end[block]
    value = wp.vec2f(source[index], wp.float32(0.0))
    base = block * wp.int32(E) + row * wp.int32(BS)
    for column in range(BS):
        value = _subtract_compensated_product(value, blocks[base + column], x[block * wp.int32(BS) + column])
    if block > begin:
        for column in range(BS):
            value = _subtract_compensated_product(
                value, lower[base + column], x[(block - wp.int32(1)) * wp.int32(BS) + column]
            )
    if block + wp.int32(1) < end:
        for column in range(BS):
            value = _subtract_compensated_product(
                value, upper[base + column], x[(block + wp.int32(1)) * wp.int32(BS) + column]
            )
    return value[0] + value[1]


@wp.kernel(enable_backward=False)
def _residual_kernel(
    n: wp.int32,
    segment_begin: wp.array[wp.int32],
    segment_end: wp.array[wp.int32],
    blocks: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    source: wp.array[wp.float32],
    x: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
):
    index = wp.tid()
    if index < n * wp.int32(BS):
        rhs[index] = wp.float32(_row_residual(index, segment_begin, segment_end, blocks, lower, upper, source, x))


@wp.kernel(enable_backward=False)
def _add_kernel(n: wp.int32, x: wp.array[wp.float32], delta: wp.array[wp.float32]):
    i = wp.tid()
    if i < n:
        x[i] += delta[i]


@wp.kernel(enable_backward=False)
def _add_norm_kernel(
    task_group: wp.array[wp.int32],
    x: wp.array[wp.float32],
    delta: wp.array[wp.float32],
    norms: wp.array[wp.float32],
):
    task, row = wp.tid()
    index = task * wp.int32(BS) + row
    updated = x[index] + delta[index]
    x[index] = updated
    group = task_group[task]
    wp.atomic_add(norms, group * wp.int32(2), delta[index] * delta[index])
    wp.atomic_add(norms, group * wp.int32(2) + wp.int32(1), updated * updated)


@wp.kernel(enable_backward=False)
def _check_norm_kernel(
    group_count: wp.int32,
    norms: wp.array[wp.float32],
    refine: wp.array[wp.int32],
):
    group = wp.tid()
    if group < group_count:
        correction_squared = norms[group * wp.int32(2)]
        solution_squared = norms[group * wp.int32(2) + wp.int32(1)]
        if correction_squared > wp.float32(_REFINEMENT_RELATIVE_TOLERANCE * _REFINEMENT_RELATIVE_TOLERANCE) * wp.max(
            solution_squared, wp.float32(1.0e-20)
        ):
            wp.atomic_max(refine, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _pack_level_zero_kernel(
    task_mechanism: wp.array[wp.int32],
    task_tile: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    diagonal_panel: wp.array[wp.int32],
    lower_panel: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    blocks: wp.array[wp.float32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
):
    task, lane = wp.tid()
    mechanism = task_mechanism[task]
    tile = task_tile[task]
    dimension = dimensions[mechanism]
    tile_begin = tile * wp.int32(BS)
    diagonal = wp.array(ptr=_offset_ptr(matrix, diagonal_panel[task] * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
    block = wp.array(ptr=_offset_ptr(blocks, task * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
    diagonal_value = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    for iteration in range((E + wp.block_dim() - 1) // wp.block_dim()):
        index = lane + iteration * wp.block_dim()
        if index < wp.int32(E):
            row = index // wp.int32(BS)
            column = index % wp.int32(BS)
            active = tile_begin + row < dimension and tile_begin + column < dimension
            element = wp.float32(0.0)
            if active:
                if row >= column:
                    element = diagonal[row, column]
                else:
                    element = diagonal[column, row]
            elif row == column:
                element = wp.float32(1.0)
            diagonal_value[row, column] = element
    wp.tile_store(block, diagonal_value)
    lower_block = wp.array(ptr=_offset_ptr(lower, task * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
    value = wp.tile_zeros(shape=(BS, BS), dtype=wp.float32, storage="shared")
    panel = lower_panel[task]
    if panel >= wp.int32(0):
        source = wp.array(ptr=_offset_ptr(matrix, panel * wp.int32(E)), shape=(BS, BS), dtype=wp.float32)
        value = wp.tile_load(source, shape=(BS, BS))
        upper_block = wp.array(
            ptr=_offset_ptr(upper, (task - wp.int32(1)) * wp.int32(E)), shape=(BS, BS), dtype=wp.float32
        )
        wp.tile_store(upper_block, wp.tile_transpose(value))
    wp.tile_store(lower_block, value)


@wp.kernel(enable_backward=False)
def _initialize_rhs_kernel(
    task_mechanism: wp.array[wp.int32],
    task_tile: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    rhs: wp.array[wp.float32],
    source: wp.array[wp.float32],
    workspace: wp.array[wp.float32],
):
    task, row = wp.tid()
    mechanism = task_mechanism[task]
    local_row = task_tile[task] * wp.int32(BS) + row
    value = wp.float32(0.0)
    if local_row < dimensions[mechanism]:
        vector_offset = vector_offsets[mechanism]
        value = rhs[vector_offset + permutation[vector_offset + local_row]]
    source[task * wp.int32(BS) + row] = value
    workspace[task * wp.int32(BS) + row] = value


@wp.kernel(enable_backward=False)
def _unpermute_solution_kernel(
    task_mechanism: wp.array[wp.int32],
    task_tile: wp.array[wp.int32],
    dimensions: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    accumulated: wp.array[wp.float32],
    solution: wp.array[wp.float32],
):
    task, row = wp.tid()
    mechanism = task_mechanism[task]
    local_row = task_tile[task] * wp.int32(BS) + row
    if local_row < dimensions[mechanism]:
        vector_offset = vector_offsets[mechanism]
        original = permutation[vector_offset + local_row]
        solution[vector_offset + original] = accumulated[task * wp.int32(BS) + row]


class BlockTridiagonalPCR:
    """Batched, iteratively refined solve for block-tridiagonal panel graphs."""

    def __init__(
        self,
        panel_tables: list[np.ndarray],
        mechanisms: np.ndarray,
        dimensions: tuple[int, ...],
        vector_offsets: np.ndarray,
        permutation: np.ndarray,
        device: wp.DeviceLike,
    ):
        self.device = wp.get_device(device)
        self.mechanisms = np.asarray(mechanisms, dtype=np.int32)
        task_mechanism = []
        task_tile = []
        task_group = []
        diagonal_panel = []
        lower_panel = []
        segment_begin = []
        segment_end = []
        begin = 0
        max_tiles = 0
        for group, mechanism in enumerate(self.mechanisms):
            table = panel_tables[int(mechanism)]
            count = table.shape[0]
            end = begin + count
            max_tiles = max(max_tiles, count)
            for tile in range(count):
                task_mechanism.append(int(mechanism))
                task_tile.append(tile)
                task_group.append(group)
                diagonal_panel.append(int(table[tile, tile]))
                lower_panel.append(int(table[tile, tile - 1]) if tile else -1)
                segment_begin.append(begin)
                segment_end.append(end)
            begin = end
        self.task_count = begin
        self.reductions = max(0, (max_tiles - 1).bit_length())
        self.task_mechanism = wp.array(task_mechanism, dtype=wp.int32, device=self.device)
        self.task_tile = wp.array(task_tile, dtype=wp.int32, device=self.device)
        self.task_group = wp.array(task_group, dtype=wp.int32, device=self.device)
        self.diagonal_panel = wp.array(diagonal_panel, dtype=wp.int32, device=self.device)
        self.lower_panel = wp.array(lower_panel, dtype=wp.int32, device=self.device)
        self.segment_begin = wp.array(segment_begin, dtype=wp.int32, device=self.device)
        self.segment_end = wp.array(segment_end, dtype=wp.int32, device=self.device)
        self.dimensions = wp.array(dimensions, dtype=wp.int32, device=self.device)
        self.vector_offsets = wp.array(vector_offsets, dtype=wp.int32, device=self.device)
        self.permutation = wp.array(permutation, dtype=wp.int32, device=self.device)
        levels = self.reductions + 1
        self.blocks = wp.zeros(levels * self.task_count * E, dtype=wp.float32, device=self.device)
        self.lower = wp.zeros_like(self.blocks)
        self.upper = wp.zeros_like(self.blocks)
        self.factors = wp.zeros_like(self.blocks)
        self.alpha = wp.zeros(self.reductions * self.task_count * E, dtype=wp.float32, device=self.device)
        self.gamma = wp.zeros_like(self.alpha)
        self.rhs = wp.zeros(levels * self.task_count * BS, dtype=wp.float32, device=self.device)
        self.source = wp.zeros(self.task_count * BS, dtype=wp.float32, device=self.device)
        self.delta = wp.zeros_like(self.source)
        self.accumulated = wp.zeros_like(self.source)
        self.needs_refinement = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.refinement_norms = wp.zeros(2 * len(self.mechanisms), dtype=wp.float32, device=self.device)

    def compute(self, matrix: wp.array[wp.float32]) -> None:
        """Pack and reduce the current block-tridiagonal matrices."""
        wp.launch_tiled(
            _pack_level_zero_kernel,
            dim=self.task_count,
            block_dim=128,
            inputs=[
                self.task_mechanism,
                self.task_tile,
                self.dimensions,
                self.diagonal_panel,
                self.lower_panel,
                matrix,
                self.blocks,
                self.lower,
                self.upper,
            ],
            device=self.device,
        )
        for level in range(self.reductions):
            wp.launch_tiled(
                _factor_blocks_kernel,
                dim=self.task_count,
                block_dim=128,
                inputs=[level, self.task_count, self.blocks, self.factors],
                device=self.device,
            )
            wp.launch_tiled(
                _reduce_blocks_kernel,
                dim=self.task_count,
                block_dim=128,
                inputs=[
                    level,
                    1 << level,
                    self.task_count,
                    self.segment_begin,
                    self.segment_end,
                    self.blocks,
                    self.lower,
                    self.upper,
                    self.factors,
                    self.alpha,
                    self.gamma,
                ],
                device=self.device,
            )
        wp.launch_tiled(
            _factor_blocks_kernel,
            dim=self.task_count,
            block_dim=128,
            inputs=[self.reductions, self.task_count, self.blocks, self.factors],
            device=self.device,
        )

    def _solve_workspace(self) -> None:
        for level in range(self.reductions):
            wp.launch_tiled(
                _reduce_rhs_kernel,
                dim=self.task_count,
                block_dim=128,
                inputs=[
                    level,
                    1 << level,
                    self.task_count,
                    level == self.reductions - 1,
                    self.segment_begin,
                    self.segment_end,
                    self.alpha,
                    self.gamma,
                    self.factors,
                    self.rhs,
                    self.delta,
                ],
                device=self.device,
            )

    def _refine_twice(self) -> None:
        """Apply two additional mixed-precision residual corrections."""
        wp.launch(
            _residual_kernel,
            dim=self.task_count * BS,
            inputs=[
                self.task_count,
                self.segment_begin,
                self.segment_end,
                self.blocks,
                self.lower,
                self.upper,
                self.source,
                self.accumulated,
                self.rhs,
            ],
            device=self.device,
        )
        self._solve_workspace()
        wp.launch(
            _add_kernel,
            dim=self.task_count * BS,
            inputs=[self.task_count * BS, self.accumulated, self.delta],
            device=self.device,
        )
        wp.launch(
            _residual_kernel,
            dim=self.task_count * BS,
            inputs=[
                self.task_count,
                self.segment_begin,
                self.segment_end,
                self.blocks,
                self.lower,
                self.upper,
                self.source,
                self.accumulated,
                self.rhs,
            ],
            device=self.device,
        )
        self._solve_workspace()
        wp.launch(
            _add_kernel,
            dim=self.task_count * BS,
            inputs=[self.task_count * BS, self.accumulated, self.delta],
            device=self.device,
        )

    def solve(
        self,
        rhs: wp.array[wp.float32],
        solution: wp.array[wp.float32],
        *,
        refine: bool = True,
    ) -> None:
        """Solve the block systems, optionally omitting residual refinement.

        The unrefined path still performs the complete parallel cyclic
        reduction. It is intended for conservative intermediate projections
        that are followed by a fully refined solve.
        """
        wp.launch(
            _initialize_rhs_kernel,
            dim=(self.task_count, BS),
            inputs=[
                self.task_mechanism,
                self.task_tile,
                self.dimensions,
                self.vector_offsets,
                self.permutation,
                rhs,
                self.source,
                self.rhs,
            ],
            device=self.device,
        )
        self._solve_workspace()
        wp.copy(self.accumulated, self.delta)
        if not refine:
            wp.launch(
                _unpermute_solution_kernel,
                dim=(self.task_count, BS),
                inputs=[
                    self.task_mechanism,
                    self.task_tile,
                    self.dimensions,
                    self.vector_offsets,
                    self.permutation,
                    self.accumulated,
                    solution,
                ],
                device=self.device,
            )
            return
        wp.launch(
            _residual_kernel,
            dim=self.task_count * BS,
            inputs=[
                self.task_count,
                self.segment_begin,
                self.segment_end,
                self.blocks,
                self.lower,
                self.upper,
                self.source,
                self.accumulated,
                self.rhs,
            ],
            device=self.device,
        )
        self._solve_workspace()
        wp.launch(
            _add_kernel,
            dim=self.task_count * BS,
            inputs=[self.task_count * BS, self.accumulated, self.delta],
            device=self.device,
        )
        wp.launch(
            _residual_kernel,
            dim=self.task_count * BS,
            inputs=[
                self.task_count,
                self.segment_begin,
                self.segment_end,
                self.blocks,
                self.lower,
                self.upper,
                self.source,
                self.accumulated,
                self.rhs,
            ],
            device=self.device,
        )
        self._solve_workspace()
        self.needs_refinement.zero_()
        self.refinement_norms.zero_()
        wp.launch(
            _add_norm_kernel,
            dim=(self.task_count, BS),
            inputs=[self.task_group, self.accumulated, self.delta, self.refinement_norms],
            device=self.device,
        )
        wp.launch(
            _check_norm_kernel,
            dim=len(self.mechanisms),
            inputs=[len(self.mechanisms), self.refinement_norms, self.needs_refinement],
            device=self.device,
        )
        if self.device.is_capturing and wp.is_conditional_graph_supported():
            wp.capture_if(self.needs_refinement, on_true=self._refine_twice)
        else:
            self._refine_twice()
        wp.launch(
            _unpermute_solution_kernel,
            dim=(self.task_count, BS),
            inputs=[
                self.task_mechanism,
                self.task_tile,
                self.dimensions,
                self.vector_offsets,
                self.permutation,
                self.accumulated,
                solution,
            ],
            device=self.device,
        )


__all__ = ["BlockTridiagonalPCR"]
