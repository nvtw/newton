# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-pattern compact panel Cholesky for PhoenX mechanisms."""

from __future__ import annotations

from ctypes import sizeof
from dataclasses import dataclass

import numpy as np
import warp as wp

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
        i = tile_i * block_size
        k = tile_k * block_size
        if i + block_size > dimension or k + block_size > dimension:
            for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                index = (lane + iteration * wp.block_dim()) % tile_elements
                row = index // block_size
                column = index % block_size
                if i + row >= dimension or k + column >= dimension:
                    panel[row, column] = wp.float32(0.0)
                if k + row >= dimension or k + column >= dimension:
                    diagonal[row, column] = wp.where(row == column, wp.float32(1.0), wp.float32(0.0))

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


def _make_solve_kernel(block_size: int):
    tile_elements = block_size * block_size

    @wp.kernel
    def solve(
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        mechanism, lane = wp.tid()
        dimension = dimensions[mechanism]
        vector_offset = vector_offsets[mechanism]
        tile_count = tile_counts[mechanism]
        table_offset = panel_table_offset[mechanism]
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, vector_offset),
            shape=(dimension, 1),
            dtype=wp.float32,
        )
        solution_permuted_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(solution_permuted, vector_offset),
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
            for tile_j in range(tile_i):
                factor_panel = panel_index[table_offset + tile_i * tile_count + tile_j]
                if factor_panel < 0:
                    continue
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
            for tile_j in range(tile_i + 1, tile_count):
                factor_panel = panel_index[table_offset + tile_j * tile_count + tile_i]
                if factor_panel < 0:
                    continue
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
        self.dimension = wp.array(dimensions, dtype=wp.int32, device=self.device)
        self.vector_offset = wp.array(vector_offsets, dtype=wp.int32, device=self.device)
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
        row_count = int(mechanism_row_start[-1])
        self.intermediate = wp.zeros(row_count, dtype=wp.float32, device=self.device)
        self.solution_permuted = wp.zeros(row_count, dtype=wp.float32, device=self.device)

        panel_tables = []
        offset = 0
        for tile_count in self.symbolic.tile_counts:
            size = tile_count * tile_count
            panel_tables.append(self.symbolic.panel_index[offset : offset + size].reshape(tile_count, tile_count))
            offset += size
        self._persistent_schedule = PersistentFactorSchedule(
            panel_tables,
            self.symbolic.panel_count,
            block_size,
            self.device,
        )
        self._solve = _make_solve_kernel(block_size)

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
        wp.launch_tiled(
            self._solve,
            dim=len(self.dimensions),
            block_dim=256,
            inputs=[
                self.dimension,
                self.vector_offset,
                self.panel_table_offset,
                self.tile_count,
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


__all__ = ["FixedPanelSymbolic", "FixedPatternPanelLLT", "build_fixed_panel_symbolic"]
