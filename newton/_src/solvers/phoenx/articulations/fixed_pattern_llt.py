# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-pattern compact panel Cholesky for PhoenX mechanisms."""

from __future__ import annotations

from ctypes import sizeof
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import factor_partial_panel_row
from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_schedule import PersistentFactorSchedule

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


_GET_ARRAY_PTR = """return (uint64_t)arr.data;"""


@wp.func_native(_GET_ARRAY_PTR)
def _get_float_array_ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...


@wp.func
def _get_float_array_offset_ptr(arr: wp.array[wp.float32], start: int) -> wp.uint64:
    return _get_float_array_ptr(arr) + wp.uint64(start * wp.static(sizeof(wp.float32._type_)))


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

    for mechanism, dimension in enumerate(dimensions):
        vector_offset = int(mechanism_row_start[mechanism])
        local_permutation = np.asarray(
            permutation[vector_offset : vector_offset + dimension],
            dtype=np.int32,
        )
        inverse_permutation = np.empty(dimension, dtype=np.int32)
        inverse_permutation[local_permutation] = np.arange(dimension, dtype=np.int32)

        body_rows: dict[int, list[int]] = {}
        active_pairs = {(row, row) for row in range(dimension)}
        for local_row in range(dimension):
            for body in row_bodies[vector_offset + local_row]:
                body_rows.setdefault(body, []).append(local_row)
        for rows in body_rows.values():
            for row in rows:
                for column in rows:
                    active_pairs.add((max(row, column), min(row, column)))

        permuted_entries: set[tuple[int, int]] = set()
        for original_row, original_column in active_pairs:
            row = int(inverse_permutation[original_row])
            column = int(inverse_permutation[original_column])
            if row < column:
                row, column = column, row
            permuted_entries.add((row, column))

        tile_count = (dimension + block_size - 1) // block_size
        tile_counts.append(tile_count)
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
        for column in range(tile_count):
            for row in range(column, tile_count):
                if pattern[row, column]:
                    panel_table[row, column] = panel_count
                    panel_count += 1
        panel_tables.append(panel_table)

        for row, column in sorted(permuted_entries):
            panel = int(panel_table[row // block_size, column // block_size])
            storage = panel * block_size * block_size + (row % block_size) * block_size + column % block_size
            matrix_row.append(vector_offset + int(local_permutation[row]))
            matrix_column.append(vector_offset + int(local_permutation[column]))
            matrix_storage.append(storage)
        for original_row in range(dimension):
            row = int(inverse_permutation[original_row])
            panel = int(panel_table[row // block_size, row // block_size])
            diagonal_storage[vector_offset + original_row] = (
                panel * block_size * block_size + (row % block_size) * block_size + row % block_size
            )

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


def _make_factor_kernels(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel
    def factor_diagonal(
        task_mechanism: wp.array[wp.int32],
        tile_k: int,
        dimensions: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = task_mechanism[task]
        dimension = dimensions[mechanism]
        tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        k = tile_k * block_size
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

        for tile_j in range(tile_k):
            previous_panel = panel_index[table_offset + tile_k * tile_count + tile_j]
            if previous_panel < 0:
                continue
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

    @wp.kernel
    def factor_panel(
        task_mechanism: wp.array[wp.int32],
        task_tile_i: wp.array[wp.int32],
        tile_k: int,
        dimensions: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = task_mechanism[task]
        tile_i = task_tile_i[task]
        dimension = dimensions[mechanism]
        tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        i = tile_i * block_size
        if i + block_size > dimension:
            factor_partial_panel_row(
                dimension,
                tile_i,
                wp.int32(tile_k),
                table_offset,
                tile_count,
                panel_index,
                matrix,
                factor,
                lane,
                wp.int32(block_size),
            )
        else:
            panel_id = panel_index[table_offset + tile_i * tile_count + tile_k]
            diagonal_panel = panel_index[table_offset + tile_k * tile_count + tile_k]
            panel_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(matrix, panel_id * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            panel = wp.tile_load(panel_matrix, shape=(block_size, block_size), storage="shared")
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size), storage="shared")
            for tile_j in range(tile_k):
                left_panel = panel_index[table_offset + tile_i * tile_count + tile_j]
                right_panel = panel_index[table_offset + tile_k * tile_count + tile_j]
                if left_panel < 0 or right_panel < 0:
                    continue
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

    return factor_diagonal, factor_panel


def _make_aligned_solve_kernel(block_size: int):
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
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, workspace_offset),
            shape=(dimension, 1),
            dtype=wp.float32,
        )
        solution_permuted_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(solution_permuted, workspace_offset),
            shape=(dimension, 1),
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

        for reverse_tile in range(tile_count):
            tile_i = tile_count - 1 - reverse_tile
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
            if i + block_size > dimension:
                for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                    index = (lane + iteration * wp.block_dim()) % tile_elements
                    row = index // block_size
                    column = index % block_size
                    if i + row >= dimension:
                        diagonal[row, column] = wp.where(row == column, wp.float32(1.0), wp.float32(0.0))
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
                if row < block_size and i + row < dimension:
                    original = permutation[vector_offset + i + row]
                    solution[vector_offset + original] = right_hand_side[row, 0]

    return solve


def _make_forward_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel
    def solve_forward(
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
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = mechanisms[task]
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        tile_count = tile_counts[mechanism]
        workspace_offset = workspace_offsets[mechanism]
        workspace_dimension = tile_count * block_size
        table_offset = panel_table_offset[mechanism]
        adjacency_offset = tile_adjacency_offset[mechanism]
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, workspace_offset),
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

    return solve_forward


def _make_partial_backward_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def solve_partial_backward(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        mechanism = mechanisms[wp.tid()]
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        workspace_offset = workspace_offsets[mechanism]
        tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        tile_i = tile_count - wp.int32(1)
        i = tile_i * block_size
        active_rows = dimension - i
        diagonal_panel = panel_index[table_offset + tile_i * tile_count + tile_i]
        factor_offset = diagonal_panel * tile_elements

        row = active_rows - wp.int32(1)
        while row >= wp.int32(0):
            value = intermediate[workspace_offset + i + row]
            column = row + wp.int32(1)
            while column < active_rows:
                value -= (
                    factor[factor_offset + column * block_size + row] * solution_permuted[workspace_offset + i + column]
                )
                column += wp.int32(1)
            value /= factor[factor_offset + row * block_size + row]
            solution_permuted[workspace_offset + i + row] = value
            original = permutation[vector_offset + i + row]
            solution[vector_offset + original] = value
            row -= wp.int32(1)

    return solve_partial_backward


def _make_backward_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel
    def solve_backward(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        tile_adjacency_offset: wp.array[wp.int32],
        backward_start: wp.array[wp.int32],
        backward_tile: wp.array[wp.int32],
        backward_panel: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
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
        workspace_dimension = tile_count * block_size
        table_offset = panel_table_offset[mechanism]
        adjacency_offset = tile_adjacency_offset[mechanism]
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

        reverse_count = tile_count - wp.int32(1)
        for reverse_tile in range(reverse_count):
            tile_i = tile_count - 2 - reverse_tile
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
                if row < block_size and i + row < dimension:
                    original = permutation[vector_offset + i + row]
                    solution[vector_offset + original] = right_hand_side[row, 0]

    return solve_backward


def _make_small_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def solve_small(
        mechanisms: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        mechanism = mechanisms[wp.tid()]
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        workspace_offset = workspace_offsets[mechanism]
        table_offset = panel_table_offset[mechanism]
        diagonal_panel = panel_index[table_offset]
        factor_offset = diagonal_panel * tile_elements

        row = wp.int32(0)
        while row < dimension:
            original = permutation[vector_offset + row]
            value = rhs[vector_offset + original]
            column = wp.int32(0)
            while column < row:
                value -= factor[factor_offset + row * block_size + column] * intermediate[workspace_offset + column]
                column += wp.int32(1)
            value /= factor[factor_offset + row * block_size + row]
            intermediate[workspace_offset + row] = value
            row += wp.int32(1)

        row = dimension - wp.int32(1)
        while row >= wp.int32(0):
            value = intermediate[workspace_offset + row]
            column = row + wp.int32(1)
            while column < dimension:
                value -= (
                    factor[factor_offset + column * block_size + row] * solution_permuted[workspace_offset + column]
                )
                column += wp.int32(1)
            value /= factor[factor_offset + row * block_size + row]
            solution_permuted[workspace_offset + row] = value
            original = permutation[vector_offset + row]
            solution[vector_offset + original] = value
            row -= wp.int32(1)

    return solve_small


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
        partial_large_mechanisms = np.flatnonzero(
            (padded_dimensions > block_size) & (padded_dimensions != np.asarray(dimensions))
        ).astype(np.int32)
        aligned_large_mechanisms = np.flatnonzero(
            (padded_dimensions > block_size) & (padded_dimensions == np.asarray(dimensions))
        ).astype(np.int32)
        self.small_mechanism = wp.array(small_mechanisms, dtype=wp.int32, device=self.device)
        self.large_mechanism = wp.array(large_mechanisms, dtype=wp.int32, device=self.device)
        self.partial_large_mechanism = wp.array(partial_large_mechanisms, dtype=wp.int32, device=self.device)
        self.aligned_large_mechanism = wp.array(aligned_large_mechanisms, dtype=wp.int32, device=self.device)
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

        panel_tables = []
        offset = 0
        for tile_count in self.symbolic.tile_counts:
            size = tile_count * tile_count
            panel_tables.append(self.symbolic.panel_index[offset : offset + size].reshape(tile_count, tile_count))
            offset += size

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

        self._persistent_schedule = PersistentFactorSchedule(
            panel_tables,
            self.symbolic.panel_count,
            block_size,
            self.device,
        )
        self._solve_small = _make_small_solve_kernel(block_size)
        self._solve_aligned = _make_aligned_solve_kernel(block_size)
        self._solve_forward_partial = _make_forward_solve_kernel(block_size)
        self._solve_partial_backward = _make_partial_backward_solve_kernel(block_size)
        self._solve_backward_partial = _make_backward_solve_kernel(block_size)

    def compute(self) -> None:
        """Factor all mechanisms through one persistent atomic panel queue."""
        self._persistent_schedule.compute(
            self.dimension,
            self.panel_table_offset,
            self.tile_count,
            self.panel_index,
            self.matrix,
            self.factor,
        )

    def solve(self, rhs: wp.array[wp.float32], solution: wp.array[wp.float32]) -> None:
        """Solve all mechanism blocks and unpermute the result."""
        if self.small_mechanism.size > 0:
            wp.launch(
                self._solve_small,
                dim=self.small_mechanism.size,
                inputs=[
                    self.small_mechanism,
                    self.dimension,
                    self.vector_offset,
                    self.workspace_offset,
                    self.panel_table_offset,
                    self.panel_index,
                    self.permutation,
                    self.factor,
                    rhs,
                    self.intermediate,
                    self.solution_permuted,
                    solution,
                ],
                device=self.device,
            )
        if self.aligned_large_mechanism.size > 0:
            wp.launch_tiled(
                self._solve_aligned,
                dim=self.aligned_large_mechanism.size,
                block_dim=256,
                inputs=[
                    self.aligned_large_mechanism,
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
        if self.partial_large_mechanism.size > 0:
            wp.launch_tiled(
                self._solve_forward_partial,
                dim=self.partial_large_mechanism.size,
                block_dim=256,
                inputs=[
                    self.partial_large_mechanism,
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
                    self.permutation,
                    self.factor,
                    rhs,
                    self.intermediate,
                ],
                device=self.device,
            )
            wp.launch(
                self._solve_partial_backward,
                dim=self.partial_large_mechanism.size,
                inputs=[
                    self.partial_large_mechanism,
                    self.dimension,
                    self.vector_offset,
                    self.workspace_offset,
                    self.panel_table_offset,
                    self.tile_count,
                    self.panel_index,
                    self.permutation,
                    self.factor,
                    self.intermediate,
                    self.solution_permuted,
                    solution,
                ],
                device=self.device,
            )
            wp.launch_tiled(
                self._solve_backward_partial,
                dim=self.partial_large_mechanism.size,
                block_dim=256,
                inputs=[
                    self.partial_large_mechanism,
                    self.dimension,
                    self.vector_offset,
                    self.workspace_offset,
                    self.panel_table_offset,
                    self.tile_count,
                    self.panel_index,
                    self.tile_adjacency_offset,
                    self.backward_start,
                    self.backward_tile,
                    self.backward_panel,
                    self.permutation,
                    self.factor,
                    self.intermediate,
                    self.solution_permuted,
                    solution,
                ],
                device=self.device,
            )


__all__ = ["FixedPanelSymbolic", "FixedPatternPanelLLT", "build_fixed_panel_symbolic"]
