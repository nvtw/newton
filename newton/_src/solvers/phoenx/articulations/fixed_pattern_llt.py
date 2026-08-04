# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-pattern compact panel Cholesky for PhoenX mechanisms."""

from __future__ import annotations

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

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

GROUPED_RHS_ITEMS_PER_TASK = 8

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
                    diagonal_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    panel = wp.tile_load(panel_matrix, shape=(block_size, block_size), storage="shared")
                    diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size), storage="shared")
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
    item_width = 4
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
        self.item_workspace_stride = max_padded_dimension * 4
        self.task_workspace_stride = max_padded_dimension * 4 * GROUPED_RHS_ITEMS_PER_TASK
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
        self.solution_permuted = wp.zeros_like(self.intermediate)
        self.solution = wp.zeros_like(self.rhs)

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
            ],
            device=panel.device,
        )


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
        cooperative_factor_mechanisms = (
            np.arange(len(dimensions), dtype=np.int32) if use_cooperative_factor else np.empty(0, dtype=np.int32)
        )
        self.cooperative_factor_mechanism = wp.array(
            cooperative_factor_mechanisms,
            dtype=wp.int32,
            device=self.device,
        )
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
            self._use_product_factor = product_schedule.product_count > 0 and product_schedule.max_ready_count > len(
                dimensions
            )
            if self._use_product_factor:
                self._product_factor_schedule = product_schedule

        self._persistent_schedule = None
        if not self._use_product_factor and not use_cooperative_factor:
            self._persistent_schedule = PersistentFactorSchedule(
                panel_tables,
                dimensions,
                self.symbolic.panel_count,
                block_size,
                self.device,
            )

        mechanism_count = len(large_mechanisms)
        self._push_forward_schedule = None
        self._push_backward_schedule = None
        self._use_push_solve = False
        if 0 < mechanism_count < self.device.sm_count:
            push_forward_schedule = PersistentPushSolveSchedule(
                panel_tables,
                large_mechanisms,
                workspace_rhs_index,
                block_size,
                self.device,
                forward=True,
            )
            push_backward_schedule = PersistentPushSolveSchedule(
                panel_tables,
                large_mechanisms,
                workspace_rhs_index,
                block_size,
                self.device,
                forward=False,
            )
            push_ready_width = min(
                push_forward_schedule.max_ready_count,
                push_backward_schedule.max_ready_count,
            )
            self._use_push_solve = push_ready_width > mechanism_count
            if self._use_push_solve:
                self._push_forward_schedule = push_forward_schedule
                self._push_backward_schedule = push_backward_schedule
        cooperative_mechanisms = (
            small_mechanisms if self._use_push_solve else np.arange(len(dimensions), dtype=np.int32)
        )
        self.cooperative_mechanism = wp.array(cooperative_mechanisms, dtype=wp.int32, device=self.device)
        self._factor_cooperative = _make_cooperative_factor_kernel(block_size)
        self._solve_cooperative = _make_cooperative_solve_kernel(block_size)
        self._solve_grouped_rhs_batch = _make_grouped_rhs_batch_solve_kernel(block_size)
        self._cooperative_solve_block_dim = 128 if block_size == 32 else 64
        self._grouped_rhs_block_dim = 128

    def create_grouped_rhs_batch(
        self,
        item_capacity: int,
        task_capacity: int,
    ) -> FixedPatternGroupedRHSBatch:
        """Allocate runtime-selected groups of narrow RHS items."""
        return FixedPatternGroupedRHSBatch(self, item_capacity, task_capacity)

    def compute(self) -> None:
        """Factor narrow mechanisms cooperatively or use the ready queue."""
        if self.cooperative_factor_mechanism.size > 0:
            wp.launch_tiled(
                self._factor_cooperative,
                dim=self.cooperative_factor_mechanism.size,
                block_dim=128,
                inputs=[
                    self.cooperative_factor_mechanism,
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
        elif self._use_product_factor:
            assert self._product_factor_schedule is not None
            self._product_factor_schedule.compute(
                self.matrix,
                self.factor,
            )
        else:
            assert self._persistent_schedule is not None
            self._persistent_schedule.compute(
                self.matrix,
                self.factor,
            )

    def solve(self, rhs: wp.array[wp.float32], solution: wp.array[wp.float32]) -> None:
        """Solve all mechanism blocks and unpermute the result."""
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


__all__ = ["FixedPanelSymbolic", "FixedPatternGroupedRHSBatch", "FixedPatternPanelLLT", "build_fixed_panel_symbolic"]
