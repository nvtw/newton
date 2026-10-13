# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-pattern compact panel Cholesky for PhoenX mechanisms."""

from __future__ import annotations

import functools
from ctypes import sizeof
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import _block_sync, factor_partial_panel_row
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_schedule import (
    PersistentFactorSchedule,
    PersistentProductFactorSchedule,
    PersistentPushSolveSchedule,
)
from newton._src.solvers.phoenx.articulations.fixed_pattern_pcr import BlockTridiagonalPCR

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

GROUPED_RHS_ITEMS_PER_TASK = 4
GROUPED_RHS_ITEM_WIDTH = 3
_MIN_PCR_TILE_COUNT = 8

_GET_ARRAY_PTR = """return (uint64_t)arr.data;"""


@wp.func_native(_GET_ARRAY_PTR)
def _get_float_array_ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...


@wp.func
def _get_float_array_offset_ptr(arr: wp.array[wp.float32], start: int) -> wp.uint64:
    return _get_float_array_ptr(arr) + wp.uint64(start * wp.static(sizeof(wp.float32._type_)))


@wp.func_native(
    r"""
#if defined(__CUDA_ARCH__)
    const int subgroup = (threadIdx.x & 31) >> 4;
    const unsigned mask = 0xffffu << (subgroup << 4);
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset, 16);
    }
#endif
    return value;
"""
)
def _sum_group_16(value: wp.float32) -> wp.float32: ...


@wp.func
def _compute_grouped_rhs_gram_item(
    item: wp.int32,
    dimension: wp.int32,
    item_workspace_stride: wp.int32,
    rhs: wp.array[wp.float32],
    solution: wp.array[wp.float32],
    gram: wp.array2d[wp.float32],
):
    begin = item * item_workspace_stride
    gram00 = wp.float32(0.0)
    gram01 = wp.float32(0.0)
    gram02 = wp.float32(0.0)
    gram11 = wp.float32(0.0)
    gram12 = wp.float32(0.0)
    gram22 = wp.float32(0.0)
    for row in range(dimension):
        offset = begin + row * wp.int32(GROUPED_RHS_ITEM_WIDTH)
        rhs0 = rhs[offset]
        rhs1 = rhs[offset + wp.int32(1)]
        rhs2 = rhs[offset + wp.int32(2)]
        solution0 = solution[offset]
        solution1 = solution[offset + wp.int32(1)]
        solution2 = solution[offset + wp.int32(2)]
        gram00 += rhs0 * solution0
        gram01 += rhs0 * solution1
        gram02 += rhs0 * solution2
        gram11 += rhs1 * solution1
        gram12 += rhs1 * solution2
        gram22 += rhs2 * solution2
    gram[0, item] = gram00
    gram[1, item] = gram01
    gram[2, item] = gram02
    gram[3, item] = gram11
    gram[4, item] = gram12
    gram[5, item] = gram22


@dataclass(frozen=True)
class FixedPanelSymbolic:
    """Immutable compact storage and task metadata."""

    matrix_row: np.ndarray
    matrix_column: np.ndarray
    matrix_storage: np.ndarray
    diagonal_storage: np.ndarray
    panel_index: np.ndarray
    panel_table_offset: np.ndarray
    tile_counts: tuple[int, ...]
    panel_count: int


def build_fixed_panel_symbolic(
    dimensions: tuple[int, ...],
    mechanism_row_start: np.ndarray,
    permutation: np.ndarray,
    row_bodies: tuple[frozenset[int], ...],
    block_size: int,
) -> FixedPanelSymbolic:
    """Build RCM-permuted symbolic fill and compact panel addresses."""
    matrix_row: list[int] = []
    matrix_column: list[int] = []
    matrix_storage: list[int] = []
    diagonal_storage = np.empty(len(permutation), dtype=np.int32)
    panel_tables: list[np.ndarray] = []
    tile_counts: list[int] = []
    panel_count = 0
    tile_elements = block_size * block_size
    symbolic_cache: dict[tuple[int, bytes, tuple[tuple[int, ...], ...]], tuple] = {}

    for mechanism, dimension in enumerate(dimensions):
        vector_offset = int(mechanism_row_start[mechanism])
        local_permutation = np.asarray(
            permutation[vector_offset : vector_offset + dimension],
            dtype=np.int32,
        )

        body_labels: dict[int, int] = {}
        row_signature: list[tuple[int, ...]] = []
        for local_row in range(dimension):
            labels: list[int] = []
            for body in sorted(row_bodies[vector_offset + local_row]):
                if body not in body_labels:
                    body_labels[body] = len(body_labels)
                labels.append(body_labels[body])
            row_signature.append(tuple(labels))
        cache_key = (dimension, local_permutation.tobytes(), tuple(row_signature))
        cached = symbolic_cache.get(cache_key)
        if cached is None:
            body_rows: dict[int, list[int]] = {}
            active_pairs = {(row, row) for row in range(dimension)}
            for local_row, labels in enumerate(row_signature):
                for body in labels:
                    body_rows.setdefault(body, []).append(local_row)
            for rows in body_rows.values():
                for row in rows:
                    for column in rows:
                        active_pairs.add((max(row, column), min(row, column)))
            active_pair_key = tuple(sorted(active_pairs))
            inverse_permutation = np.empty(dimension, dtype=np.int32)
            inverse_permutation[local_permutation] = np.arange(dimension, dtype=np.int32)
            permuted_entries: set[tuple[int, int]] = set()
            for original_row, original_column in active_pair_key:
                row = int(inverse_permutation[original_row])
                column = int(inverse_permutation[original_column])
                if row < column:
                    row, column = column, row
                permuted_entries.add((row, column))

            tile_count = (dimension + block_size - 1) // block_size
            pattern = np.zeros((tile_count, tile_count), dtype=bool)
            for row, column in permuted_entries:
                pattern[row // block_size, column // block_size] = True
            np.fill_diagonal(pattern, True)
            for column in range(tile_count):
                for row in range(column + 1, tile_count):
                    if pattern[row, column]:
                        continue
                    for inner in range(column):
                        if pattern[row, inner] and pattern[column, inner]:
                            pattern[row, column] = True
                            break

            panel_table = np.full((tile_count, tile_count), -1, dtype=np.int32)
            local_panel_count = 0
            for column in range(tile_count):
                for row in range(column, tile_count):
                    if pattern[row, column]:
                        panel_table[row, column] = local_panel_count
                        local_panel_count += 1

            entries = sorted(permuted_entries)
            local_matrix_row = np.asarray([local_permutation[row] for row, _ in entries], dtype=np.int32)
            local_matrix_column = np.asarray([local_permutation[column] for _, column in entries], dtype=np.int32)
            local_matrix_storage = np.asarray(
                [
                    int(panel_table[row // block_size, column // block_size]) * tile_elements
                    + (row % block_size) * block_size
                    + column % block_size
                    for row, column in entries
                ],
                dtype=np.int32,
            )
            local_diagonal_storage = np.empty(dimension, dtype=np.int32)
            for original_row in range(dimension):
                row = int(inverse_permutation[original_row])
                panel = int(panel_table[row // block_size, row // block_size])
                local_diagonal_storage[original_row] = (
                    panel * tile_elements + (row % block_size) * block_size + row % block_size
                )
            cached = (
                tile_count,
                panel_table,
                local_panel_count,
                local_matrix_row,
                local_matrix_column,
                local_matrix_storage,
                local_diagonal_storage,
            )
            symbolic_cache[cache_key] = cached

        (
            tile_count,
            local_panel_table,
            local_panel_count,
            local_matrix_row,
            local_matrix_column,
            local_matrix_storage,
            local_diagonal_storage,
        ) = cached
        storage_offset = panel_count * tile_elements
        matrix_row.extend((vector_offset + local_matrix_row).tolist())
        matrix_column.extend((vector_offset + local_matrix_column).tolist())
        matrix_storage.extend((storage_offset + local_matrix_storage).tolist())
        diagonal_storage[vector_offset : vector_offset + dimension] = storage_offset + local_diagonal_storage
        panel_tables.append(np.where(local_panel_table >= 0, local_panel_table + panel_count, -1))
        tile_counts.append(tile_count)
        panel_count += local_panel_count

    table_offsets = [0]
    for table in panel_tables:
        table_offsets.append(table_offsets[-1] + table.size)
    return FixedPanelSymbolic(
        matrix_row=np.asarray(matrix_row, dtype=np.int32),
        matrix_column=np.asarray(matrix_column, dtype=np.int32),
        matrix_storage=np.asarray(matrix_storage, dtype=np.int32),
        diagonal_storage=diagonal_storage,
        panel_index=np.concatenate([table.ravel() for table in panel_tables]).astype(np.int32),
        panel_table_offset=np.asarray(table_offsets[:-1], dtype=np.int32),
        tile_counts=tuple(tile_counts),
        panel_count=panel_count,
    )


def _make_cooperative_factor_kernel(block_size: int):
    """Create a one-block-per-mechanism factor kernel."""
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def factor_cooperative(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        tile_adjacency_offset: wp.array[wp.int32],
        forward_start: wp.array[wp.int32],
        forward_panel: wp.array[wp.int32],
        backward_start: wp.array[wp.int32],
        backward_tile: wp.array[wp.int32],
        backward_panel: wp.array[wp.int32],
        offdiag_update_start: wp.array[wp.int32],
        offdiag_update_left: wp.array[wp.int32],
        offdiag_update_right: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = mechanisms[task]
        dimension = dimensions[mechanism]
        tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        adjacency_offset = tile_adjacency_offset[mechanism]

        for tile_k in range(tile_count):
            k = tile_k * block_size
            tile_slot = adjacency_offset + tile_k
            diagonal_panel = panel_index[table_offset + tile_k * tile_count + tile_k]
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(matrix, diagonal_panel * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size), storage="shared")
            if k + block_size > dimension:
                for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                    index = (lane + iteration * wp.block_dim()) % tile_elements
                    row = index // block_size
                    column = index % block_size
                    value = diagonal[row, column]
                    if k + row >= dimension or k + column >= dimension:
                        value = wp.where(row == column, wp.float32(1.0), wp.float32(0.0))
                    diagonal[row, column] = value

            for entry in range(forward_start[tile_slot], forward_start[tile_slot + wp.int32(1)]):
                previous_panel = forward_panel[entry]
                previous_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, previous_panel * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                previous = wp.tile_load(previous_matrix, shape=(block_size, block_size))
                wp.tile_matmul(previous, wp.tile_transpose(previous), diagonal, alpha=-1.0)
            wp.tile_cholesky_inplace(diagonal)
            diagonal_factor = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            wp.tile_store(diagonal_factor, diagonal)
            _block_sync()

            panel_begin = backward_start[tile_slot]
            panel_end = backward_start[tile_slot + wp.int32(1)]
            for panel_entry in range(panel_begin, panel_end):
                tile_i = backward_tile[panel_entry]
                panel_id = backward_panel[panel_entry]
                i = tile_i * block_size
                if i + block_size > dimension:
                    factor_partial_panel_row(
                        dimension - i,
                        panel_id,
                        diagonal_panel,
                        offdiag_update_start[panel_entry],
                        offdiag_update_start[panel_entry + wp.int32(1)],
                        offdiag_update_left,
                        offdiag_update_right,
                        matrix,
                        factor,
                        lane,
                        wp.int32(block_size),
                    )
                else:
                    panel_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(matrix, panel_id * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    panel = wp.tile_load(panel_matrix, shape=(block_size, block_size), storage="shared")
                    for update in range(
                        offdiag_update_start[panel_entry],
                        offdiag_update_start[panel_entry + wp.int32(1)],
                    ):
                        left_panel = offdiag_update_left[update]
                        right_panel = offdiag_update_right[update]
                        left_matrix = wp.array(
                            ptr=_get_float_array_offset_ptr(factor, left_panel * tile_elements),
                            shape=(block_size, block_size),
                            dtype=wp.float32,
                        )
                        right_matrix = wp.array(
                            ptr=_get_float_array_offset_ptr(factor, right_panel * tile_elements),
                            shape=(block_size, block_size),
                            dtype=wp.float32,
                        )
                        left = wp.tile_load(left_matrix, shape=(block_size, block_size))
                        right = wp.tile_load(right_matrix, shape=(block_size, block_size))
                        wp.tile_matmul(left, wp.tile_transpose(right), panel, alpha=-1.0)
                    transposed = wp.tile_transpose(panel)
                    wp.tile_lower_solve_inplace(diagonal, transposed)
                    panel_factor = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    wp.tile_store(panel_factor, wp.tile_transpose(transposed))
                _block_sync()

    return factor_cooperative


@wp.func_native("""
#if defined(__CUDA_ARCH__)
unsigned mask = width == 32 ? 0xffffffffu : (0xffffu << ((threadIdx.x & 31) & 16));
return __shfl_sync(mask, value, source, width);
#else
return value;
#endif
""")
def _shuffle_panel_value(value: wp.float32, source: wp.int32, width: wp.int32) -> wp.float32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp(width == 32 ? 0xffffffffu : (0xffffu << ((threadIdx.x & 31) & 16)));
#endif
""")
def sync_panel_subgroup(width: wp.int32): ...


def packed_lower_size(width: int) -> int:
    """Return the float count of one packed lower-triangular register panel."""
    return width * (width + 1) // 2


@wp.func
def _packed_lower_index(row: wp.int32, column: wp.int32) -> wp.int32:
    return row * (row + wp.int32(1)) / wp.int32(2) + column


@functools.cache
def make_subgroup_panel_solve_func(width: int, packed: bool = False):
    """Return a register solve of one permuted right-hand-side value per lane.

    ``packed`` reads a row-major lower triangle of :func:`packed_lower_size`
    floats instead of a full panel; the arithmetic is unchanged.
    """
    Row = wp.types.vector(length=width, dtype=wp.float32)

    @wp.func
    def entry(row: wp.int32, column: wp.int32) -> wp.int32:
        if wp.static(packed):
            return _packed_lower_index(row, column)
        return row * wp.int32(width) + column

    @wp.func
    def solve_panel(
        factor: wp.array[wp.float32],
        base: wp.int32,
        dimension: wp.int32,
        lane: wp.int32,
        value: wp.float32,
    ) -> wp.float32:
        diagonal = wp.float32(1.0)
        lower = Row(0.0)
        if lane < dimension:
            diagonal = factor[base + entry(lane, lane)]
            for column in range(wp.static(width)):
                if column < lane:
                    lower[column] = factor[base + entry(lane, column)]
        for column in range(wp.static(width)):
            if column < dimension:
                pivot = wp.float32(0.0)
                if lane == column:
                    value = value * (wp.float32(1.0) / diagonal)
                    pivot = value
                solved = _shuffle_panel_value(pivot, column, wp.static(width))
                if lane > column and lane < dimension:
                    value = value - lower[column] * solved
        upper = Row(0.0)
        if lane < dimension:
            for column in range(wp.static(width)):
                if column > lane and column < dimension:
                    upper[column] = factor[base + entry(column, lane)]
        if dimension < wp.static(width):
            # Match the existing tail solve's ascending dot-product order.
            solved_values = Row(0.0)
            for reverse in range(wp.static(width)):
                column = wp.static(width - 1) - reverse
                if column < dimension:
                    pivot = wp.float32(0.0)
                    if lane == column:
                        row_value = value
                        for solved_row in range(wp.static(width)):
                            if solved_row > column and solved_row < dimension:
                                row_value = row_value - upper[solved_row] * solved_values[solved_row]
                        value = row_value / diagonal
                        pivot = value
                    solved_values[column] = _shuffle_panel_value(pivot, column, wp.static(width))
        else:
            # Full tiles use cuSolverDx's reciprocal/update ordering.
            for reverse in range(wp.static(width)):
                column = wp.static(width - 1) - reverse
                pivot = wp.float32(0.0)
                if lane == column:
                    value = value * (wp.float32(1.0) / diagonal)
                    pivot = value
                solved = _shuffle_panel_value(pivot, column, wp.static(width))
                if lane < column:
                    value = value - upper[column] * solved
        return value

    return solve_panel


@functools.cache
def _make_subgroup_factor_kernel(width: int):
    """Factor a one-panel mechanism in registers within a subgroup.

    Lane ``i`` owns row ``i`` of the lower triangle. Rows and columns past
    the mechanism dimension are identity padding. A nonempty ``packed_factor``
    receives the task-ordered lower triangles read by packed subgroup solves;
    ``write_panel`` also stores the cooperative tile layout with a zero upper
    triangle for other panel consumers.
    """
    Row = wp.types.vector(length=width, dtype=wp.float32)

    @wp.kernel(enable_backward=False, module="unique")
    def factor_subgroup(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        tables: wp.array[wp.int32],
        panels: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
        packed_factor: wp.array[wp.float32],
        write_panel: wp.bool,
    ):
        task, lane = wp.tid()
        mechanism = mechanisms[task]
        dimension = dimensions[mechanism]
        base = panels[tables[mechanism]] * wp.static(width * width)
        row = Row(0.0)
        for column in range(wp.static(width)):
            if lane < dimension:
                if column <= lane:
                    row[column] = matrix[base + lane * wp.static(width) + column]
            elif column == lane:
                row[column] = wp.float32(1.0)
        # Right-looking Cholesky: column k is scaled by its pivot, then
        # broadcast one entry at a time to update the trailing rows.
        for k in range(wp.static(width)):
            pivot = wp.float32(0.0)
            if lane == k:
                pivot = wp.sqrt(row[k])
                row[k] = pivot
            pivot = _shuffle_panel_value(pivot, k, wp.static(width))
            if lane > k:
                row[k] = row[k] / pivot
            for column in range(wp.static(width)):
                if column > k:
                    trailing = _shuffle_panel_value(row[k], column, wp.static(width))
                    if lane >= column:
                        row[column] = row[column] - row[k] * trailing
        if packed_factor.shape[0] > 0:
            packed_base = task * wp.static(packed_lower_size(width)) + _packed_lower_index(lane, wp.int32(0))
            for column in range(wp.static(width)):
                if column <= lane:
                    packed_factor[packed_base + column] = row[column]
        if write_panel:
            for column in range(wp.static(width)):
                value = wp.float32(0.0)
                if column <= lane:
                    value = row[column]
                factor[base + lane * wp.static(width) + column] = value

    return factor_subgroup


@functools.cache
def _make_subgroup_solve_kernel(width: int, packed: bool = False):
    """Solve a one-panel mechanism in registers within a subgroup."""
    solve_panel = make_subgroup_panel_solve_func(width, packed)

    @wp.kernel(enable_backward=False, module="unique")
    def solve(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        offsets: wp.array[wp.int32],
        tables: wp.array[wp.int32],
        panels: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = mechanisms[task]
        dimension = dimensions[mechanism]
        start = offsets[mechanism]
        base = panels[tables[mechanism]] * wp.static(width * width)
        if wp.static(packed):
            base = task * wp.static(packed_lower_size(width))
        value = wp.float32(0.0)
        if lane < dimension:
            value = rhs[start + permutation[start + lane]]
        value = solve_panel(factor, base, dimension, lane, value)
        if lane < dimension:
            solution[start + permutation[start + lane]] = value

    return solve


def _make_cooperative_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel
    def solve(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        tile_adjacency_offset: wp.array[wp.int32],
        forward_start: wp.array[wp.int32],
        forward_tile: wp.array[wp.int32],
        forward_panel: wp.array[wp.int32],
        backward_start: wp.array[wp.int32],
        backward_tile: wp.array[wp.int32],
        backward_panel: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = mechanisms[task]
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        tile_count = tile_counts[mechanism]
        workspace_offset = workspace_offsets[mechanism]
        table_offset = panel_table_offset[mechanism]
        adjacency_offset = tile_adjacency_offset[mechanism]
        workspace_dimension = tile_count * block_size
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, workspace_offset),
            shape=(workspace_dimension, 1),
            dtype=wp.float32,
        )
        solution_permuted_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(solution_permuted, workspace_offset),
            shape=(workspace_dimension, 1),
            dtype=wp.float32,
        )

        for tile_i in range(tile_count):
            i = tile_i * block_size
            right_hand_side = wp.tile_zeros(shape=(block_size, 1), dtype=wp.float32, storage="shared")
            for iteration in range((block_size + wp.block_dim() - 1) // wp.block_dim()):
                row = lane + iteration * wp.block_dim()
                active = row < block_size and i + row < dimension
                value = wp.float32(0.0)
                if active:
                    value = rhs[vector_offset + permutation[vector_offset + i + row]]
                wp.tile_scatter_masked(right_hand_side, row, 0, value, active)
            tile_slot = adjacency_offset + tile_i
            for entry in range(forward_start[tile_slot], forward_start[tile_slot + wp.int32(1)]):
                tile_j = forward_tile[entry]
                factor_panel = forward_panel[entry]
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                left = wp.tile_load(factor_matrix, shape=(block_size, block_size))
                previous = wp.tile_load(
                    intermediate_matrix,
                    shape=(block_size, 1),
                    offset=(tile_j * block_size, 0),
                )
                wp.tile_matmul(left, previous, right_hand_side, alpha=-1.0)
            diagonal_panel = panel_index[table_offset + tile_i * tile_count + tile_i]
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            wp.tile_lower_solve_inplace(diagonal, right_hand_side)
            wp.tile_store(intermediate_matrix, right_hand_side, offset=(i, 0))

        last_full_tile = tile_count - wp.int32(1)
        tail_rows = dimension - last_full_tile * block_size
        if tail_rows < block_size:
            if lane == wp.int32(0):
                diagonal_panel = panel_index[table_offset + last_full_tile * tile_count + last_full_tile]
                factor_offset = diagonal_panel * tile_elements
                row = tail_rows - wp.int32(1)
                while row >= wp.int32(0):
                    value = intermediate[workspace_offset + last_full_tile * block_size + row]
                    column = row + wp.int32(1)
                    while column < tail_rows:
                        value -= (
                            factor[factor_offset + column * block_size + row]
                            * solution_permuted[workspace_offset + last_full_tile * block_size + column]
                        )
                        column += wp.int32(1)
                    value /= factor[factor_offset + row * block_size + row]
                    solution_permuted[workspace_offset + last_full_tile * block_size + row] = value
                    original = permutation[vector_offset + last_full_tile * block_size + row]
                    solution[vector_offset + original] = value
                    row -= wp.int32(1)
            _block_sync()
            last_full_tile -= wp.int32(1)

        for reverse_tile in range(last_full_tile + wp.int32(1)):
            tile_i = last_full_tile - reverse_tile
            i = tile_i * block_size
            right_hand_side = wp.tile_load(
                intermediate_matrix,
                shape=(block_size, 1),
                offset=(i, 0),
            )
            diagonal_panel = panel_index[table_offset + tile_i * tile_count + tile_i]
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            tile_slot = adjacency_offset + tile_i
            for entry in range(backward_start[tile_slot], backward_start[tile_slot + wp.int32(1)]):
                tile_j = backward_tile[entry]
                factor_panel = backward_panel[entry]
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                lower = wp.tile_load(factor_matrix, shape=(block_size, block_size))
                solved = wp.tile_load(
                    solution_permuted_matrix,
                    shape=(block_size, 1),
                    offset=(tile_j * block_size, 0),
                )
                wp.tile_matmul(wp.tile_transpose(lower), solved, right_hand_side, alpha=-1.0)
            wp.tile_upper_solve_inplace(wp.tile_transpose(diagonal), right_hand_side)
            wp.tile_store(solution_permuted_matrix, right_hand_side, offset=(i, 0))
            for iteration in range((block_size + wp.block_dim() - 1) // wp.block_dim()):
                row = lane + iteration * wp.block_dim()
                if row < block_size:
                    original = permutation[vector_offset + i + row]
                    solution[vector_offset + original] = right_hand_side[row, 0]

    return solve


def _make_grouped_rhs_batch_solve_kernel(block_size: int):
    """Create one solve task for a fixed group of three-column RHS items."""
    item_width = GROUPED_RHS_ITEM_WIDTH
    items_per_task = GROUPED_RHS_ITEMS_PER_TASK
    rhs_tile_width = item_width * items_per_task
    tile_elements = block_size * rhs_tile_width
    factor_tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def solve_grouped_rhs_batch(
        task_mechanism: wp.array[wp.int32],
        task_item: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        tile_adjacency_offset: wp.array[wp.int32],
        forward_start: wp.array[wp.int32],
        forward_tile: wp.array[wp.int32],
        forward_panel: wp.array[wp.int32],
        backward_start: wp.array[wp.int32],
        backward_tile: wp.array[wp.int32],
        backward_panel: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        item_workspace_stride: wp.int32,
        task_workspace_stride: wp.int32,
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
        gram: wp.array2d[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = task_mechanism[task]
        task_active = mechanism >= wp.int32(0)
        if not task_active:
            mechanism = wp.int32(0)
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        tile_count = wp.int32(0)
        if task_active:
            tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        adjacency_offset = tile_adjacency_offset[mechanism]
        workspace_dimension = task_workspace_stride // wp.int32(rhs_tile_width)
        task_offset = task * task_workspace_stride
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, task_offset),
            shape=(workspace_dimension, rhs_tile_width),
            dtype=wp.float32,
        )
        solution_permuted_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(solution_permuted, task_offset),
            shape=(workspace_dimension, rhs_tile_width),
            dtype=wp.float32,
        )
        gram00 = wp.float32(0.0)
        gram01 = wp.float32(0.0)
        gram02 = wp.float32(0.0)
        gram11 = wp.float32(0.0)
        gram12 = wp.float32(0.0)
        gram22 = wp.float32(0.0)
        for tile_i in range(tile_count):
            i = tile_i * block_size
            right_hand_side = wp.tile_zeros(
                shape=(block_size, rhs_tile_width),
                dtype=wp.float32,
                storage="shared",
            )
            for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                index = (lane + iteration * wp.block_dim()) % tile_elements
                row = index // rhs_tile_width
                column = index % rhs_tile_width
                item_slot = column // wp.int32(item_width)
                item_column = column - item_slot * wp.int32(item_width)
                item = task_item[task * wp.int32(items_per_task) + item_slot]
                item_active = i + row < dimension and item_column < wp.int32(3) and item >= wp.int32(0)
                value = wp.float32(0.0)
                if item_active:
                    original = permutation[vector_offset + i + row]
                    value = rhs[item * item_workspace_stride + original * wp.int32(item_width) + item_column]
                wp.tile_scatter_masked(right_hand_side, row, column, value, item_active)
            tile_slot = adjacency_offset + tile_i
            for entry in range(forward_start[tile_slot], forward_start[tile_slot + wp.int32(1)]):
                tile_j = forward_tile[entry]
                factor_panel = forward_panel[entry]
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * factor_tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                lower = wp.tile_load(factor_matrix, shape=(block_size, block_size))
                previous = wp.tile_load(
                    intermediate_matrix,
                    shape=(block_size, rhs_tile_width),
                    offset=(tile_j * block_size, 0),
                )
                wp.tile_matmul(lower, previous, right_hand_side, alpha=-1.0)
            diagonal_panel = panel_index[table_offset + tile_i * tile_count + tile_i]
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * factor_tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            wp.tile_lower_solve_inplace(diagonal, right_hand_side)
            wp.tile_store(intermediate_matrix, right_hand_side, offset=(i, 0))

        for reverse_tile in range(tile_count):
            tile_i = tile_count - wp.int32(1) - reverse_tile
            i = tile_i * block_size
            right_hand_side = wp.tile_load(
                intermediate_matrix,
                shape=(block_size, rhs_tile_width),
                offset=(i, 0),
            )
            diagonal_panel = panel_index[table_offset + tile_i * tile_count + tile_i]
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * factor_tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            tile_slot = adjacency_offset + tile_i
            for entry in range(backward_start[tile_slot], backward_start[tile_slot + wp.int32(1)]):
                tile_j = backward_tile[entry]
                factor_panel = backward_panel[entry]
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * factor_tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                lower = wp.tile_load(factor_matrix, shape=(block_size, block_size))
                solved = wp.tile_load(
                    solution_permuted_matrix,
                    shape=(block_size, rhs_tile_width),
                    offset=(tile_j * block_size, 0),
                )
                wp.tile_matmul(wp.tile_transpose(lower), solved, right_hand_side, alpha=-1.0)
            wp.tile_upper_solve_inplace(wp.tile_transpose(diagonal), right_hand_side)
            wp.tile_store(solution_permuted_matrix, right_hand_side, offset=(i, 0))
            if wp.static(block_size == 16):
                gram_item_slot = lane // wp.int32(block_size)
                gram_row = lane - gram_item_slot * wp.int32(block_size)
                gram_item = task_item[task * wp.int32(items_per_task) + gram_item_slot]
                if task_active and gram_item >= wp.int32(0) and i + gram_row < dimension:
                    original = permutation[vector_offset + i + gram_row]
                    rhs_offset = gram_item * item_workspace_stride + original * wp.int32(item_width)
                    rhs0 = rhs[rhs_offset]
                    rhs1 = rhs[rhs_offset + wp.int32(1)]
                    rhs2 = rhs[rhs_offset + wp.int32(2)]
                    column = gram_item_slot * wp.int32(item_width)
                    solution0 = right_hand_side[gram_row, column]
                    solution1 = right_hand_side[gram_row, column + wp.int32(1)]
                    solution2 = right_hand_side[gram_row, column + wp.int32(2)]
                    gram00 += rhs0 * solution0
                    gram01 += rhs0 * solution1
                    gram02 += rhs0 * solution2
                    gram11 += rhs1 * solution1
                    gram12 += rhs1 * solution2
                    gram22 += rhs2 * solution2
            for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                index = (lane + iteration * wp.block_dim()) % tile_elements
                row = index // rhs_tile_width
                column = index % rhs_tile_width
                item_slot = column // wp.int32(item_width)
                item_column = column - item_slot * wp.int32(item_width)
                item = task_item[task * wp.int32(items_per_task) + item_slot]
                if i + row < dimension and item_column < wp.int32(3) and item >= wp.int32(0):
                    original = permutation[vector_offset + i + row]
                    solution[item * item_workspace_stride + original * wp.int32(item_width) + item_column] = (
                        right_hand_side[row, column]
                    )

        if wp.static(block_size == 16):
            gram00 = _sum_group_16(gram00)
            gram01 = _sum_group_16(gram01)
            gram02 = _sum_group_16(gram02)
            gram11 = _sum_group_16(gram11)
            gram12 = _sum_group_16(gram12)
            gram22 = _sum_group_16(gram22)
            gram_item_slot = lane // wp.int32(block_size)
            if lane == gram_item_slot * wp.int32(block_size):
                gram_item = task_item[task * wp.int32(items_per_task) + gram_item_slot]
                if task_active and gram_item >= wp.int32(0):
                    gram[0, gram_item] = gram00
                    gram[1, gram_item] = gram01
                    gram[2, gram_item] = gram02
                    gram[3, gram_item] = gram11
                    gram[4, gram_item] = gram12
                    gram[5, gram_item] = gram22

        else:
            _block_sync()
            if lane < wp.int32(items_per_task):
                gram_item = task_item[task * wp.int32(items_per_task) + lane]
                if task_active and gram_item >= wp.int32(0):
                    _compute_grouped_rhs_gram_item(
                        gram_item,
                        dimension,
                        item_workspace_stride,
                        rhs,
                        solution,
                        gram,
                    )

    return solve_grouped_rhs_batch


class FixedPatternGroupedRHSBatch:
    """Group narrow RHS items without crossing mechanisms."""

    def __init__(self, panel: FixedPatternPanelLLT, item_capacity: int, task_capacity: int):
        if item_capacity < 0 or task_capacity < 0:
            raise ValueError("grouped RHS capacities must be nonnegative")
        self.panel = panel
        self.item_capacity = max(1, int(item_capacity))
        self.task_capacity = max(1, int(task_capacity))
        max_padded_dimension = max(panel.symbolic.tile_counts) * panel.block_size
        self.item_workspace_stride = max_padded_dimension * GROUPED_RHS_ITEM_WIDTH
        self.task_workspace_stride = max_padded_dimension * GROUPED_RHS_ITEM_WIDTH * GROUPED_RHS_ITEMS_PER_TASK
        device = panel.device
        self.task_mechanism = wp.full(self.task_capacity, -1, dtype=wp.int32, device=device)
        self.task_item = wp.full(
            self.task_capacity * GROUPED_RHS_ITEMS_PER_TASK,
            -1,
            dtype=wp.int32,
            device=device,
        )
        self.rhs = wp.zeros(
            self.item_capacity * self.item_workspace_stride,
            dtype=wp.float32,
            device=device,
        )
        task_storage_size = self.task_capacity * self.task_workspace_stride
        self.intermediate = wp.zeros(task_storage_size, dtype=wp.float32, device=device)
        # Descending back substitution overwrites each forward tile only after its last read.
        self.solution_permuted = self.intermediate
        self.solution = wp.zeros_like(self.rhs)
        self.gram = wp.zeros((6, self.item_capacity), dtype=wp.float32, device=device)

    def solve(self) -> None:
        """Solve all runtime-selected groups."""
        panel = self.panel
        wp.launch_tiled(
            panel._solve_grouped_rhs_batch,
            dim=self.task_capacity,
            block_dim=panel._grouped_rhs_block_dim,
            inputs=[
                self.task_mechanism,
                self.task_item,
                panel.dimension,
                panel.vector_offset,
                panel.panel_table_offset,
                panel.tile_count,
                panel.panel_index,
                panel.tile_adjacency_offset,
                panel.forward_start,
                panel.forward_tile,
                panel.forward_panel,
                panel.backward_start,
                panel.backward_tile,
                panel.backward_panel,
                panel.permutation,
                panel.factor,
                wp.int32(self.item_workspace_stride),
                wp.int32(self.task_workspace_stride),
                self.rhs,
                self.intermediate,
                self.solution_permuted,
                self.solution,
                self.gram,
            ],
            device=panel.device,
        )


def _find_block_tridiagonal_mechanisms(panel_tables: list[np.ndarray], block_size: int) -> np.ndarray:
    """Return long panel graphs supported by the parallel cyclic solver."""
    if block_size != 16:
        return np.empty(0, dtype=np.int32)
    mechanisms = []
    for mechanism, table in enumerate(panel_tables):
        tile_count = table.shape[0]
        if tile_count < _MIN_PCR_TILE_COUNT:
            continue
        expected = np.eye(tile_count, dtype=bool)
        expected[np.arange(1, tile_count), np.arange(tile_count - 1)] = True
        if np.array_equal(table >= 0, expected):
            mechanisms.append(mechanism)
    return np.asarray(mechanisms, dtype=np.int32)


# Atomic task queues pay off when a few mechanisms leave most of the
# per-mechanism block grid idle. With many mechanisms the grid is already busy:
# a forest of 50 G1 trees runs 24% faster without queues.
_QUEUE_MAX_MECHANISMS_PER_SM = 1.0 / 8.0


def _use_task_queue(ready_width: int, mechanism_count: int, device: wp.Device) -> bool:
    """Whether dependency-ready panel queues beat one block per mechanism."""
    return ready_width > mechanism_count and mechanism_count < _QUEUE_MAX_MECHANISMS_PER_SM * device.sm_count


class FixedPatternPanelLLT:
    """Factor and solve fixed-topology mechanism matrices in compact panels."""

    def __init__(
        self,
        dimensions: tuple[int, ...],
        mechanism_row_start: np.ndarray,
        permutation: np.ndarray,
        row_bodies: tuple[frozenset[int], ...],
        *,
        block_size: int = 16,
        device: wp.DeviceLike = None,
        use_subgroup_solve: bool = True,
    ):
        self.dimensions = dimensions
        self.block_size = block_size
        self.device = wp.get_device(device)
        self.symbolic = build_fixed_panel_symbolic(
            dimensions,
            mechanism_row_start,
            permutation,
            row_bodies,
            block_size,
        )
        vector_offsets = np.asarray(mechanism_row_start[:-1], dtype=np.int32)
        padded_dimensions = np.asarray(
            [((dimension + block_size - 1) // block_size) * block_size for dimension in dimensions],
            dtype=np.int32,
        )
        workspace_offsets = np.zeros(len(dimensions), dtype=np.int32)
        if len(dimensions) > 1:
            workspace_offsets[1:] = np.cumsum(padded_dimensions[:-1])
        self.dimension = wp.array(dimensions, dtype=wp.int32, device=self.device)
        self.vector_offset = wp.array(vector_offsets, dtype=wp.int32, device=self.device)
        self.workspace_offset = wp.array(workspace_offsets, dtype=wp.int32, device=self.device)
        small_mechanisms = np.flatnonzero(padded_dimensions == block_size).astype(np.int32)
        large_mechanisms = np.flatnonzero(padded_dimensions > block_size).astype(np.int32)
        self.large_mechanism = wp.array(large_mechanisms, dtype=wp.int32, device=self.device)
        self.permutation = wp.array(permutation, dtype=wp.int32, device=self.device)
        self.panel_table_offset = wp.array(
            self.symbolic.panel_table_offset,
            dtype=wp.int32,
            device=self.device,
        )
        self.tile_count = wp.array(self.symbolic.tile_counts, dtype=wp.int32, device=self.device)
        self.panel_index = wp.array(self.symbolic.panel_index, dtype=wp.int32, device=self.device)
        storage_size = self.symbolic.panel_count * block_size * block_size
        self.matrix = wp.zeros(storage_size, dtype=wp.float32, device=self.device)
        self.factor = wp.zeros_like(self.matrix)
        workspace_size = int(np.sum(padded_dimensions))
        self.intermediate = wp.zeros(workspace_size, dtype=wp.float32, device=self.device)
        self.solution_permuted = wp.zeros(workspace_size, dtype=wp.float32, device=self.device)
        workspace_rhs_index = np.full(workspace_size, -1, dtype=np.int32)
        for mechanism, dimension in enumerate(dimensions):
            vector_offset = int(vector_offsets[mechanism])
            workspace_offset = int(workspace_offsets[mechanism])
            for local_row in range(dimension):
                workspace_rhs_index[workspace_offset + local_row] = vector_offset + int(
                    permutation[vector_offset + local_row]
                )

        panel_tables = []
        offset = 0
        for tile_count in self.symbolic.tile_counts:
            size = tile_count * tile_count
            panel_tables.append(self.symbolic.panel_index[offset : offset + size].reshape(tile_count, tile_count))
            offset += size

        pcr_mechanisms = _find_block_tridiagonal_mechanisms(panel_tables, block_size)
        self._pcr = (
            BlockTridiagonalPCR(
                panel_tables,
                pcr_mechanisms,
                dimensions,
                vector_offsets,
                permutation,
                self.device,
            )
            if pcr_mechanisms.size
            else None
        )

        tile_adjacency_offset: list[int] = []
        forward_start = [0]
        forward_tile: list[int] = []
        forward_panel: list[int] = []
        backward_start = [0]
        backward_tile: list[int] = []
        backward_panel: list[int] = []
        tile_offset = 0
        for table in panel_tables:
            tile_adjacency_offset.append(tile_offset)
            for tile_i in range(table.shape[0]):
                for tile_j in range(tile_i):
                    panel = int(table[tile_i, tile_j])
                    if panel >= 0:
                        forward_tile.append(tile_j)
                        forward_panel.append(panel)
                forward_start.append(len(forward_panel))
                for tile_j in range(tile_i + 1, table.shape[0]):
                    panel = int(table[tile_j, tile_i])
                    if panel >= 0:
                        backward_tile.append(tile_j)
                        backward_panel.append(panel)
                backward_start.append(len(backward_panel))
            tile_offset += table.shape[0]
        self.tile_adjacency_offset = wp.array(tile_adjacency_offset, dtype=wp.int32, device=self.device)
        self.forward_start = wp.array(forward_start, dtype=wp.int32, device=self.device)
        self.forward_tile = wp.array(forward_tile, dtype=wp.int32, device=self.device)
        self.forward_panel = wp.array(forward_panel, dtype=wp.int32, device=self.device)
        self.backward_start = wp.array(backward_start, dtype=wp.int32, device=self.device)
        self.backward_tile = wp.array(backward_tile, dtype=wp.int32, device=self.device)
        self.backward_panel = wp.array(backward_panel, dtype=wp.int32, device=self.device)

        all_narrow = True
        offdiag_update_start = [0]
        offdiag_update_left: list[int] = []
        offdiag_update_right: list[int] = []
        for table in panel_tables:
            for tile_k in range(table.shape[0]):
                panel_rows = [tile_i for tile_i in range(tile_k + 1, table.shape[0]) if table[tile_i, tile_k] >= 0]
                all_narrow = all_narrow and len(panel_rows) <= 1
                for tile_i in panel_rows:
                    for tile_j in range(tile_k):
                        left_panel = int(table[tile_i, tile_j])
                        right_panel = int(table[tile_k, tile_j])
                        if left_panel >= 0 and right_panel >= 0:
                            offdiag_update_left.append(left_panel)
                            offdiag_update_right.append(right_panel)
                    offdiag_update_start.append(len(offdiag_update_left))
        use_cooperative_factor = all_narrow or len(dimensions) >= self.device.sm_count
        all_mechanisms = np.arange(len(dimensions), dtype=np.int32)
        cooperative_factor_mechanisms = (
            np.setdiff1d(all_mechanisms, pcr_mechanisms, assume_unique=True)
            if use_cooperative_factor
            else np.empty(0, dtype=np.int32)
        )
        grouped_rhs_factor_mechanisms = pcr_mechanisms if use_cooperative_factor else np.empty(0, dtype=np.int32)
        self.cooperative_factor_mechanism = wp.array(
            cooperative_factor_mechanisms,
            dtype=wp.int32,
            device=self.device,
        )
        self.grouped_rhs_factor_mechanism = wp.array(
            grouped_rhs_factor_mechanisms,
            dtype=wp.int32,
            device=self.device,
        )
        self._grouped_rhs_factor_required = False
        self.offdiag_update_start = wp.array(offdiag_update_start, dtype=wp.int32, device=self.device)
        self.offdiag_update_left = wp.array(offdiag_update_left, dtype=wp.int32, device=self.device)
        self.offdiag_update_right = wp.array(offdiag_update_right, dtype=wp.int32, device=self.device)

        self._product_factor_schedule = None
        self._use_product_factor = False
        if not all_narrow and len(dimensions) < self.device.sm_count:
            product_schedule = PersistentProductFactorSchedule(
                panel_tables,
                dimensions,
                self.symbolic.panel_count,
                block_size,
                self.device,
            )
            self._use_product_factor = product_schedule.product_count > 0 and _use_task_queue(
                product_schedule.max_ready_count, len(dimensions), self.device
            )
            if self._use_product_factor:
                self._product_factor_schedule = product_schedule

        self._persistent_schedule = None
        self._persistent_schedule_without_pcr = None
        if not self._use_product_factor and not use_cooperative_factor:
            self._persistent_schedule = PersistentFactorSchedule(
                panel_tables,
                dimensions,
                self.symbolic.panel_count,
                block_size,
                self.device,
            )
            # PCR solves its block-tridiagonal mechanisms from its own factor;
            # their panel LLT is needed only by the grouped contact response.
            other_mechanisms = np.setdiff1d(all_mechanisms, pcr_mechanisms, assume_unique=True)
            if pcr_mechanisms.size and other_mechanisms.size:
                other_tables = [panel_tables[mechanism] for mechanism in other_mechanisms]
                self._persistent_schedule_without_pcr = PersistentFactorSchedule(
                    other_tables,
                    tuple(dimensions[mechanism] for mechanism in other_mechanisms),
                    sum(int(np.count_nonzero(table >= 0)) for table in other_tables),
                    block_size,
                    self.device,
                )

        remaining_large_mechanisms = np.setdiff1d(large_mechanisms, pcr_mechanisms, assume_unique=True)
        mechanism_count = len(remaining_large_mechanisms)
        self._push_forward_schedule = None
        self._push_backward_schedule = None
        self._use_push_solve = False
        if 0 < mechanism_count < self.device.sm_count:
            push_forward_schedule = PersistentPushSolveSchedule(
                panel_tables,
                remaining_large_mechanisms,
                workspace_rhs_index,
                block_size,
                self.device,
                forward=True,
            )
            push_backward_schedule = PersistentPushSolveSchedule(
                panel_tables,
                remaining_large_mechanisms,
                workspace_rhs_index,
                block_size,
                self.device,
                forward=False,
            )
            push_ready_width = min(
                push_forward_schedule.max_ready_count,
                push_backward_schedule.max_ready_count,
            )
            self._use_push_solve = _use_task_queue(push_ready_width, mechanism_count, self.device)
            if self._use_push_solve:
                self._push_forward_schedule = push_forward_schedule
                self._push_backward_schedule = push_backward_schedule
        cooperative_mechanisms = (
            small_mechanisms
            if self._use_push_solve
            else np.setdiff1d(
                np.arange(len(dimensions), dtype=np.int32),
                pcr_mechanisms,
                assume_unique=True,
            )
        )
        # Single-panel systems need no panel traversal or global intermediate
        # vectors. Pack multiple independent subgroups into each CUDA block.
        subgroup_mechanisms = np.empty(0, dtype=np.int32)
        mathdx_solver = wp.get_module(__name__).options.get("enable_mathdx_solver")
        if mathdx_solver is None:
            mathdx_solver = getattr(wp.config, "enable_mathdx_solver", False)
        use_subgroups = use_subgroup_solve and mathdx_solver and self.device.is_cuda and block_size in (16, 32)
        if use_subgroups:
            subgroup_mechanisms = np.intersect1d(cooperative_mechanisms, small_mechanisms, assume_unique=True)
            cooperative_mechanisms = np.setdiff1d(cooperative_mechanisms, subgroup_mechanisms, assume_unique=True)
        self.subgroup_mechanism = wp.array(subgroup_mechanisms, dtype=wp.int32, device=self.device)
        # One-panel mechanisms factor in registers like their solves; the tile
        # factor keeps only multi-panel mechanisms.
        subgroup_factor_mechanisms = (
            np.intersect1d(cooperative_factor_mechanisms, small_mechanisms, assume_unique=True)
            if use_subgroups
            else np.empty(0, dtype=np.int32)
        )
        self.subgroup_factor_mechanism = wp.array(subgroup_factor_mechanisms, dtype=wp.int32, device=self.device)
        self._tile_factor_mechanism = wp.array(
            np.setdiff1d(cooperative_factor_mechanisms, subgroup_factor_mechanisms, assume_unique=True),
            dtype=wp.int32,
            device=self.device,
        )
        # When both use the same task order, solves read the factor kernel's
        # packed lower triangles, about half the bytes of full panels.
        self.subgroup_packed = subgroup_mechanisms.size > 0 and np.array_equal(
            subgroup_factor_mechanisms, subgroup_mechanisms
        )
        self.packed_factor = wp.zeros(
            subgroup_mechanisms.size * packed_lower_size(block_size) if self.subgroup_packed else 0,
            dtype=wp.float32,
            device=self.device,
        )
        self._write_subgroup_panels = not self.subgroup_packed
        self._factor_subgroup = _make_subgroup_factor_kernel(block_size) if subgroup_factor_mechanisms.size else None
        self._solve_subgroup = (
            _make_subgroup_solve_kernel(block_size, self.subgroup_packed) if subgroup_mechanisms.size else None
        )
        self.cooperative_mechanism = wp.array(cooperative_mechanisms, dtype=wp.int32, device=self.device)
        self._factor_cooperative = _make_cooperative_factor_kernel(block_size)
        self._solve_cooperative = _make_cooperative_solve_kernel(block_size)
        self._solve_grouped_rhs_batch = _make_grouped_rhs_batch_solve_kernel(block_size)
        self._cooperative_solve_block_dim = 128 if block_size == 32 else 64
        # Chains reduced by PCR are independent of the panel LLT mechanisms;
        # a side stream forks them into a parallel graph branch.
        self._pcr_stream = (
            wp.Stream(self.device)
            if self._pcr is not None and self.device.is_cuda and pcr_mechanisms.size < len(dimensions)
            else None
        )
        self._grouped_rhs_block_dim = block_size * GROUPED_RHS_ITEMS_PER_TASK

    @property
    def subgroup_factor(self) -> wp.array[wp.float32]:
        """Factor storage read by subgroup solves: packed triangles or full panels."""
        return self.packed_factor if self.subgroup_packed else self.factor

    def create_grouped_rhs_batch(
        self,
        item_capacity: int,
        task_capacity: int,
    ) -> FixedPatternGroupedRHSBatch:
        """Allocate runtime-selected groups of narrow RHS items."""
        self._grouped_rhs_factor_required = True
        return FixedPatternGroupedRHSBatch(self, item_capacity, task_capacity)

    def compute(self) -> None:
        """Factor narrow mechanisms cooperatively or use the ready queue."""
        if self._pcr_stream is not None:
            with wp.ScopedStream(self._pcr_stream):
                self._pcr.compute(self.matrix)
        if self.subgroup_factor_mechanism.size > 0:
            wp.launch(
                self._factor_subgroup,
                dim=(self.subgroup_factor_mechanism.size, self.block_size),
                inputs=[
                    self.subgroup_factor_mechanism,
                    self.dimension,
                    self.panel_table_offset,
                    self.panel_index,
                    self.matrix,
                    self.factor,
                    self.packed_factor,
                    wp.bool(self._write_subgroup_panels or self._grouped_rhs_factor_required),
                ],
                block_dim=128,
                device=self.device,
            )
        factor_mechanisms = [self._tile_factor_mechanism]
        if self._grouped_rhs_factor_required:
            factor_mechanisms.append(self.grouped_rhs_factor_mechanism)
        for mechanisms in factor_mechanisms:
            if mechanisms.size == 0:
                continue
            wp.launch_tiled(
                self._factor_cooperative,
                dim=mechanisms.size,
                block_dim=128,
                inputs=[
                    mechanisms,
                    self.dimension,
                    self.panel_table_offset,
                    self.tile_count,
                    self.panel_index,
                    self.tile_adjacency_offset,
                    self.forward_start,
                    self.forward_panel,
                    self.backward_start,
                    self.backward_tile,
                    self.backward_panel,
                    self.offdiag_update_start,
                    self.offdiag_update_left,
                    self.offdiag_update_right,
                    self.matrix,
                    self.factor,
                ],
                device=self.device,
            )
        if (
            self.cooperative_factor_mechanism.size == 0
            and self.grouped_rhs_factor_mechanism.size == 0
            and self._use_product_factor
        ):
            assert self._product_factor_schedule is not None
            self._product_factor_schedule.compute(
                self.matrix,
                self.factor,
            )
        elif self.cooperative_factor_mechanism.size == 0 and self.grouped_rhs_factor_mechanism.size == 0:
            assert self._persistent_schedule is not None
            schedule = self._persistent_schedule
            if self._pcr is not None and not self._grouped_rhs_factor_required:
                schedule = self._persistent_schedule_without_pcr
            if schedule is not None:
                schedule.compute(self.matrix, self.factor)
        # Grouped contact response still consumes the LLT factor; PCR replaces
        # only the latency-bound scalar right-hand-side solve.
        if self._pcr_stream is not None:
            wp.wait_stream(self._pcr_stream)
        elif self._pcr is not None:
            self._pcr.compute(self.matrix)

    def solve(
        self,
        rhs: wp.array[wp.float32],
        solution: wp.array[wp.float32],
        *,
        refine: bool = True,
    ) -> None:
        """Solve all mechanism blocks and unpermute the result.

        The refine option affects only long block-tridiagonal mechanisms using
        PCR; all other factor paths retain their existing solve.
        """
        if self._pcr_stream is not None:
            with wp.ScopedStream(self._pcr_stream):
                self._pcr.reduce(rhs, refine=refine)
        if self.subgroup_mechanism.size > 0:
            wp.launch(
                self._solve_subgroup,
                dim=(self.subgroup_mechanism.size, self.block_size),
                inputs=[
                    self.subgroup_mechanism,
                    self.dimension,
                    self.vector_offset,
                    self.panel_table_offset,
                    self.panel_index,
                    self.permutation,
                    self.subgroup_factor,
                    rhs,
                    solution,
                ],
                block_dim=128,
                device=self.device,
            )
        if self.cooperative_mechanism.size > 0:
            wp.launch_tiled(
                self._solve_cooperative,
                dim=self.cooperative_mechanism.size,
                block_dim=self._cooperative_solve_block_dim,
                inputs=[
                    self.cooperative_mechanism,
                    self.dimension,
                    self.vector_offset,
                    self.workspace_offset,
                    self.panel_table_offset,
                    self.tile_count,
                    self.panel_index,
                    self.tile_adjacency_offset,
                    self.forward_start,
                    self.forward_tile,
                    self.forward_panel,
                    self.backward_start,
                    self.backward_tile,
                    self.backward_panel,
                    self.permutation,
                    self.factor,
                    rhs,
                    self.intermediate,
                    self.solution_permuted,
                    solution,
                ],
                device=self.device,
            )
        if self._use_push_solve:
            assert self._push_forward_schedule is not None
            assert self._push_backward_schedule is not None
            self._push_forward_schedule.solve(
                self.dimension,
                self.vector_offset,
                self.workspace_offset,
                self.panel_table_offset,
                self.tile_count,
                self.panel_index,
                self.permutation,
                self.factor,
                rhs,
                self.intermediate,
                self.solution_permuted,
                solution,
            )
            self._push_backward_schedule.solve(
                self.dimension,
                self.vector_offset,
                self.workspace_offset,
                self.panel_table_offset,
                self.tile_count,
                self.panel_index,
                self.permutation,
                self.factor,
                rhs,
                self.intermediate,
                self.solution_permuted,
                solution,
            )
        if self._pcr_stream is not None:
            wp.wait_stream(self._pcr_stream)
            self._pcr.finish(solution, refine=refine)
        elif self._pcr is not None:
            self._pcr.solve(rhs, solution, refine=refine)


__all__ = ["FixedPanelSymbolic", "FixedPatternGroupedRHSBatch", "FixedPatternPanelLLT", "build_fixed_panel_symbolic"]
