# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed triangular tiles for large batched bilateral LLT systems."""

import numpy as np
import warp as wp

__all__ = []

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func_native("return (uint64_t)arr.data;")
def _get_float_array_ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...


@wp.func
def _get_float_array_offset_ptr(arr: wp.array[wp.float32], start: wp.int64) -> wp.uint64:
    return _get_float_array_ptr(arr) + wp.uint64(start) * wp.uint64(4)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _block_sync(): ...


@wp.func
def _slot(base: wp.int64, row: wp.int32, column: wp.int32) -> wp.int64:
    return base + wp.int64(row) * wp.int64(row + 1) // wp.int64(2) + wp.int64(column)


@wp.func
def factor_partial_panel_row(
    active_rows: wp.int32,
    tile_i: wp.int32,
    tile_k: wp.int32,
    tile_count: wp.int32,
    table_offset: wp.int32,
    slot_base: wp.int64,
    pattern: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    factor: wp.array[wp.float32],
    lane: wp.int32,
    block_size: wp.int32,
):
    if lane >= active_rows:
        return
    tile_elements = block_size * block_size
    panel_offset = _slot(slot_base, tile_i, tile_k) * wp.int64(tile_elements)
    diagonal_offset = _slot(slot_base, tile_k, tile_k) * wp.int64(tile_elements)
    column = wp.int32(0)
    while column < block_size:
        value = matrix[panel_offset + wp.int64(lane * block_size + column)]
        tile_j = wp.int32(0)
        while tile_j < tile_k:
            if (
                pattern[table_offset + tile_i * tile_count + tile_j] != 0
                and pattern[table_offset + tile_k * tile_count + tile_j] != 0
            ):
                left_offset = _slot(slot_base, tile_i, tile_j) * wp.int64(tile_elements) + wp.int64(lane * block_size)
                right_offset = _slot(slot_base, tile_k, tile_j) * wp.int64(tile_elements) + wp.int64(
                    column * block_size
                )
                inner = wp.int32(0)
                while inner < block_size:
                    value -= factor[left_offset + wp.int64(inner)] * factor[right_offset + wp.int64(inner)]
                    inner += wp.int32(1)
            tile_j += wp.int32(1)
        inner = wp.int32(0)
        while inner < column:
            value -= (
                factor[diagonal_offset + wp.int64(column * block_size + inner)]
                * factor[panel_offset + wp.int64(lane * block_size + inner)]
            )
            inner += wp.int32(1)
        value /= factor[diagonal_offset + wp.int64(column * block_size + column)]
        factor[panel_offset + wp.int64(lane * block_size + column)] = value
        column += wp.int32(1)


def _make_cooperative_factor_kernel(block_size: int):
    """Create a one-block-per-mechanism factor kernel."""
    tile_elements = wp.constant(wp.int32(block_size * block_size))
    tile_elements64 = wp.constant(wp.int64(block_size * block_size))

    @wp.kernel(enable_backward=False)
    def factor_cooperative(
        dimensions: wp.array[wp.int32],
        slot_offsets: wp.array[wp.int64],
        panel_table_offset: wp.array[wp.int32],
        pattern: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = task
        dimension = dimensions[mechanism]
        tile_count = (dimension + block_size - 1) // block_size
        slot_base = slot_offsets[mechanism]
        table_offset = panel_table_offset[mechanism]

        for tile_k in range(tile_count):
            k = tile_k * block_size
            diagonal_panel = _slot(slot_base, tile_k, tile_k)
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(matrix, diagonal_panel * tile_elements64),
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
                if pattern[table_offset + tile_k * tile_count + tile_j] == 0:
                    continue
                previous_panel = _slot(slot_base, tile_k, tile_j)
                previous_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, previous_panel * tile_elements64),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                previous = wp.tile_load(previous_matrix, shape=(block_size, block_size))
                wp.tile_matmul(previous, wp.tile_transpose(previous), diagonal, alpha=-1.0)
            wp.tile_cholesky_inplace(diagonal)
            diagonal_factor = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements64),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            wp.tile_store(diagonal_factor, diagonal)
            _block_sync()

            for tile_i in range(tile_k + 1, tile_count):
                panel_id = _slot(slot_base, tile_i, tile_k)
                if pattern[table_offset + tile_i * tile_count + tile_k] == 0:
                    skipped = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements64),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    zeros = wp.tile_zeros(shape=(block_size, block_size), dtype=wp.float32)
                    wp.tile_store(skipped, zeros)
                    continue
                i = tile_i * block_size
                if i + block_size > dimension:
                    # Clear padded rows explicitly, including after storage reuse.
                    for element in range(lane, tile_elements, wp.block_dim()):
                        if element // block_size >= dimension - i:
                            factor[panel_id * tile_elements64 + wp.int64(element)] = wp.float32(0.0)
                    factor_partial_panel_row(
                        dimension - i,
                        tile_i,
                        tile_k,
                        tile_count,
                        table_offset,
                        slot_base,
                        pattern,
                        matrix,
                        factor,
                        lane,
                        wp.int32(block_size),
                    )
                else:
                    panel_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(matrix, panel_id * tile_elements64),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    panel = wp.tile_load(panel_matrix, shape=(block_size, block_size), storage="shared")
                    for tile_j in range(tile_k):
                        if (
                            pattern[table_offset + tile_i * tile_count + tile_j] == 0
                            or pattern[table_offset + tile_k * tile_count + tile_j] == 0
                        ):
                            continue
                        left_panel = _slot(slot_base, tile_i, tile_j)
                        right_panel = _slot(slot_base, tile_k, tile_j)
                        left_matrix = wp.array(
                            ptr=_get_float_array_offset_ptr(factor, left_panel * tile_elements64),
                            shape=(block_size, block_size),
                            dtype=wp.float32,
                        )
                        right_matrix = wp.array(
                            ptr=_get_float_array_offset_ptr(factor, right_panel * tile_elements64),
                            shape=(block_size, block_size),
                            dtype=wp.float32,
                        )
                        left = wp.tile_load(left_matrix, shape=(block_size, block_size))
                        right = wp.tile_load(right_matrix, shape=(block_size, block_size))
                        wp.tile_matmul(left, wp.tile_transpose(right), panel, alpha=-1.0)
                    transposed = wp.tile_transpose(panel)
                    wp.tile_lower_solve_inplace(diagonal, transposed)
                    panel_factor = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements64),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    wp.tile_store(panel_factor, wp.tile_transpose(transposed))
                _block_sync()

    return factor_cooperative


def _make_cooperative_solve_kernel(block_size: int):
    tile_elements64 = wp.constant(wp.int64(block_size * block_size))

    @wp.kernel
    def solve(
        dimensions: wp.array[wp.int32],
        active_dimensions: wp.array[wp.int32],
        slot_offsets: wp.array[wp.int64],
        errors: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        pattern: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ):
        task, lane = wp.tid()
        mechanism = task
        dimension = dimensions[mechanism]
        active_dimension = active_dimensions[mechanism]
        if active_dimension == 0 or dimension == 0:
            return
        if active_dimension != dimension:
            if lane == 0:
                wp.atomic_add(errors, 0, 1)
            return
        slot_base = slot_offsets[mechanism]
        vector_offset = vector_offsets[mechanism]
        tile_count = (dimension + block_size - 1) // block_size
        workspace_offset = workspace_offsets[mechanism]
        table_offset = panel_table_offset[mechanism]
        workspace_dimension = tile_count * block_size
        intermediate_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(intermediate, wp.int64(workspace_offset)),
            shape=(workspace_dimension, 1),
            dtype=wp.float32,
        )
        solution_permuted_matrix = wp.array(
            ptr=_get_float_array_offset_ptr(solution_permuted, wp.int64(workspace_offset)),
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
            for tile_j in range(tile_i):
                if pattern[table_offset + tile_i * tile_count + tile_j] == 0:
                    continue
                factor_panel = _slot(slot_base, tile_i, tile_j)
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * tile_elements64),
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
            diagonal_panel = _slot(slot_base, tile_i, tile_i)
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements64),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            wp.tile_lower_solve_inplace(diagonal, right_hand_side)
            wp.tile_store(intermediate_matrix, right_hand_side, offset=(i, 0))

        # Tail entries stay zero across every solve, not only first allocation.
        for row in range(dimension + lane, workspace_dimension, wp.block_dim()):
            solution_permuted[workspace_offset + row] = wp.float32(0.0)
        _block_sync()
        last_full_tile = tile_count - wp.int32(1)
        tail_rows = dimension - last_full_tile * block_size
        if tail_rows < block_size:
            if lane == wp.int32(0):
                diagonal_panel = _slot(slot_base, last_full_tile, last_full_tile)
                factor_offset = diagonal_panel * tile_elements64
                row = tail_rows - wp.int32(1)
                while row >= wp.int32(0):
                    value = intermediate[workspace_offset + last_full_tile * block_size + row]
                    column = row + wp.int32(1)
                    while column < tail_rows:
                        value -= (
                            factor[factor_offset + wp.int64(column * block_size + row)]
                            * solution_permuted[workspace_offset + last_full_tile * block_size + column]
                        )
                        column += wp.int32(1)
                    value /= factor[factor_offset + wp.int64(row * block_size + row)]
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
            diagonal_panel = _slot(slot_base, tile_i, tile_i)
            diagonal_matrix = wp.array(
                ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements64),
                shape=(block_size, block_size),
                dtype=wp.float32,
            )
            diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
            for tile_j in range(tile_i + 1, tile_count):
                if pattern[table_offset + tile_j * tile_count + tile_i] == 0:
                    continue
                factor_panel = _slot(slot_base, tile_j, tile_i)
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, factor_panel * tile_elements64),
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


class _PackedLLT:
    def __init__(self, dimensions, device, tile_pattern, tile_pattern_offsets, permutation, vector_offsets):
        self.device = wp.get_device(device)
        dims = np.asarray(dimensions, dtype=np.int64)
        if dims.ndim != 1 or np.any(dims < 0) or np.any(dims > np.iinfo(np.int32).max):
            raise ValueError("Expected nonnegative int32 dimensions")
        tiles = (dims + 31) // 32
        slots = tiles * (tiles + 1) // 2
        self.factor_size = int(slots.sum() * 1024)
        workspace_sizes = tiles * 32
        workspace_size = int(workspace_sizes.sum())
        if workspace_size >= 2**31:
            raise ValueError("Vector workspace exceeds int32 indexing capacity")
        self.num_worlds = len(dims)
        self.dimensions = wp.array(dims.astype(np.int32), dtype=wp.int32, device=self.device)
        self.slot_offsets = wp.array(np.cumsum(np.r_[0, slots])[:-1], dtype=wp.int64, device=self.device)
        self.workspace_offsets = wp.array(
            np.cumsum(np.r_[0, workspace_sizes])[:-1].astype(np.int32), dtype=wp.int32, device=self.device
        )
        self.tile_pattern = tile_pattern
        self.tile_pattern_offsets = tile_pattern_offsets
        self.permutation = permutation
        self.vector_offsets = vector_offsets
        self.intermediate = wp.zeros(workspace_size, dtype=wp.float32, device=self.device)
        self.solution_permuted = wp.zeros_like(self.intermediate)
        self.errors = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.factor_kernel = _make_cooperative_factor_kernel(32)
        self.solve_kernel = _make_cooperative_solve_kernel(32)
        for array in (tile_pattern, tile_pattern_offsets, permutation, vector_offsets):
            if array.device != self.device or array.dtype != wp.int32:
                raise ValueError("Metadata must be int32 arrays on the selected device")
        if tile_pattern_offsets.size != self.num_worlds or vector_offsets.size != self.num_worlds:
            raise ValueError("Expected one pattern and vector offset per world")

    def factor(self, matrix, factor):
        if matrix.size < self.factor_size or factor.size < self.factor_size:
            raise ValueError("Insufficient fixed-slot matrix/factor storage")
        if self.num_worlds:
            wp.launch_tiled(
                self.factor_kernel,
                dim=self.num_worlds,
                block_dim=128,
                inputs=[
                    self.dimensions,
                    self.slot_offsets,
                    self.tile_pattern_offsets,
                    self.tile_pattern,
                    matrix,
                    factor,
                ],
                device=self.device,
            )

    def solve(self, factor, rhs, out, active_dim=None):
        if self.num_worlds:
            wp.launch_tiled(
                self.solve_kernel,
                dim=self.num_worlds,
                block_dim=128,
                inputs=[
                    self.dimensions,
                    self.dimensions if active_dim is None else active_dim,
                    self.slot_offsets,
                    self.errors,
                    self.vector_offsets,
                    self.workspace_offsets,
                    self.tile_pattern_offsets,
                    self.tile_pattern,
                    self.permutation,
                    factor,
                    rhs,
                    self.intermediate,
                    self.solution_permuted,
                    out,
                ],
                device=self.device,
            )


@wp.func
def packed_element_offset(base: wp.int64, row: wp.int32, col: wp.int32) -> wp.int64:
    """Address a lower-triangular 32-by-32 tile, including diagonal tile upper entries."""
    return _slot(base, row // 32, col // 32) * wp.int64(1024) + wp.int64((row % 32) * 32 + col % 32)


@wp.kernel
def _gather_intermediate(
    dimensions: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    workspace_offsets: wp.array[wp.int32],
    intermediate: wp.array[wp.float32],
    dense: wp.array[wp.float32],
):
    world, row = wp.tid()
    if row < dimensions[world]:
        dense[vector_offsets[world] + row] = intermediate[workspace_offsets[world] + row]


@wp.kernel
def _transfer(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    slot_offsets: wp.array[wp.int64],
    dense: wp.array[wp.float32],
    packed: wp.array[wp.float32],
    to_packed: bool,
):
    world, element = wp.tid()
    n = dimensions[world]
    if n == 0:
        return
    width = ((n + 31) // 32) * 32
    row, col = element // width, element % width
    if row >= width:
        return
    if row // 32 >= col // 32:
        offset = packed_element_offset(slot_offsets[world], row, col)
        if to_packed:
            value = wp.float32(0.0)
            if row < n and col < n:
                value = dense[wp.int64(matrix_offsets[world]) + wp.int64(row) * wp.int64(n) + wp.int64(col)]
            packed[offset] = value
        elif row < n and col < n:
            dense[wp.int64(matrix_offsets[world]) + wp.int64(row) * wp.int64(n) + wp.int64(col)] = packed[offset]
    elif not to_packed and row < n and col < n:
        dense[wp.int64(matrix_offsets[world]) + wp.int64(row) * wp.int64(n) + wp.int64(col)] = wp.float32(0.0)
