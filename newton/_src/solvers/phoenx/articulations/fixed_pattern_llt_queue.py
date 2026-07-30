# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent atomic task scheduling for fixed-pattern panel Cholesky."""

from ctypes import sizeof

import warp as wp

_GET_ARRAY_PTR = """return (uint64_t)arr.data;"""


@wp.func_native(_GET_ARRAY_PTR)
def _get_float_array_ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...


@wp.func
def _get_float_array_offset_ptr(arr: wp.array[wp.float32], start: int) -> wp.uint64:
    return _get_float_array_ptr(arr) + wp.uint64(start * wp.static(sizeof(wp.float32._type_)))


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
"""
)
def _block_sync(): ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
__syncthreads();
if (threadIdx.x == 0) {
    __threadfence();
}
__syncthreads();
#endif
"""
)
def _block_fence(): ...


@wp.func
def factor_partial_panel_row(
    dimension: wp.int32,
    tile_i: wp.int32,
    tile_k: wp.int32,
    table_offset: wp.int32,
    tile_count: wp.int32,
    panel_index: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    factor: wp.array[wp.float32],
    lane: wp.int32,
    block_size: wp.int32,
):
    """Factor one active row of a partial off-diagonal panel."""
    i = tile_i * block_size
    if i + lane >= dimension:
        return
    tile_elements = block_size * block_size
    panel_id = panel_index[table_offset + tile_i * tile_count + tile_k]
    diagonal_panel = panel_index[table_offset + tile_k * tile_count + tile_k]
    panel_offset = panel_id * tile_elements
    diagonal_offset = diagonal_panel * tile_elements

    column = wp.int32(0)
    while column < block_size:
        value = matrix[panel_offset + lane * block_size + column]
        tile_j = wp.int32(0)
        while tile_j < tile_k:
            left_panel = panel_index[table_offset + tile_i * tile_count + tile_j]
            right_panel = panel_index[table_offset + tile_k * tile_count + tile_j]
            if left_panel >= wp.int32(0) and right_panel >= wp.int32(0):
                left_offset = left_panel * tile_elements + lane * block_size
                right_offset = right_panel * tile_elements + column * block_size
                inner = wp.int32(0)
                while inner < block_size:
                    value -= factor[left_offset + inner] * factor[right_offset + inner]
                    inner += wp.int32(1)
            tile_j += wp.int32(1)

        inner = wp.int32(0)
        while inner < column:
            value -= (
                factor[diagonal_offset + column * block_size + inner] * factor[panel_offset + lane * block_size + inner]
            )
            inner += wp.int32(1)
        value /= factor[diagonal_offset + column * block_size + column]
        factor[panel_offset + lane * block_size + column] = value
        column += wp.int32(1)


@wp.kernel(enable_backward=False)
def initialize_factor_queue(
    initial_task: wp.array[wp.int32],
    remaining_initial: wp.array[wp.int32],
    queue: wp.array[wp.int32],
    head: wp.array[wp.int32],
    tail: wp.array[wp.int32],
    completed: wp.array[wp.int32],
    remaining: wp.array[wp.int32],
):
    index = wp.tid()
    queue[index] = wp.int32(-1)
    remaining[index] = remaining_initial[index]
    if index < initial_task.shape[0]:
        queue[index] = initial_task[index]
    if index == 0:
        head[0] = wp.int32(0)
        tail[0] = initial_task.shape[0]
        completed[0] = wp.int32(0)


def make_persistent_factor_kernel(block_size: int):
    """Create a dependency-driven panel factorization worker kernel."""
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def factor_persistent(
        task_mechanism: wp.array[wp.int32],
        task_tile_i: wp.array[wp.int32],
        task_tile_k: wp.array[wp.int32],
        task_panel_count: wp.array[wp.int32],
        task_next_diagonal: wp.array[wp.int32],
        task_owner_diagonal: wp.array[wp.int32],
        task_update_start: wp.array[wp.int32],
        update_left_panel: wp.array[wp.int32],
        update_right_panel: wp.array[wp.int32],
        dimensions: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
        queue: wp.array[wp.int32],
        head: wp.array[wp.int32],
        tail: wp.array[wp.int32],
        completed: wp.array[wp.int32],
        remaining: wp.array[wp.int32],
        worker_task: wp.array[wp.int32],
    ):
        worker, lane = wp.tid()
        task_count = task_mechanism.shape[0]

        while True:
            if lane == 0:
                claimed_task = wp.int32(-1)
                while claimed_task == wp.int32(-1):
                    queue_head = wp.atomic_add(head, wp.int32(0), wp.int32(0))
                    queue_tail = wp.atomic_add(tail, wp.int32(0), wp.int32(0))
                    if queue_head < queue_tail:
                        observed = wp.atomic_cas(
                            head,
                            wp.int32(0),
                            queue_head,
                            queue_head + wp.int32(1),
                        )
                        if observed == queue_head:
                            while claimed_task == wp.int32(-1):
                                claimed_task = wp.atomic_add(queue, queue_head, wp.int32(0))
                    elif wp.atomic_add(completed, wp.int32(0), wp.int32(0)) == task_count:
                        claimed_task = wp.int32(-2)
                worker_task[worker] = claimed_task
            _block_sync()

            task = worker_task[worker]
            if task < 0:
                break

            mechanism = task_mechanism[task]
            tile_i = task_tile_i[task]
            tile_k = task_tile_k[task]
            dimension = dimensions[mechanism]
            tile_count = tile_counts[mechanism]
            table_offset = panel_table_offset[mechanism]
            i = tile_i * block_size
            k = tile_k * block_size

            if tile_i == tile_k:
                diagonal_panel = panel_index[table_offset + tile_k * tile_count + tile_k]
                diagonal_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(matrix, diagonal_panel * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                diagonal = wp.tile_load(
                    diagonal_matrix,
                    shape=(block_size, block_size),
                    storage="shared",
                )
                if k + block_size > dimension:
                    for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                        index = (lane + iteration * wp.block_dim()) % tile_elements
                        row = index // block_size
                        column = index % block_size
                        value = diagonal[row, column]
                        if k + row >= dimension or k + column >= dimension:
                            value = wp.where(row == column, wp.float32(1.0), wp.float32(0.0))
                        diagonal[row, column] = value

                for update in range(task_update_start[task], task_update_start[task + wp.int32(1)]):
                    previous_panel = update_left_panel[update]
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
            else:
                if i + block_size > dimension:
                    factor_partial_panel_row(
                        dimension,
                        tile_i,
                        tile_k,
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
                    for update in range(task_update_start[task], task_update_start[task + wp.int32(1)]):
                        left_panel = update_left_panel[update]
                        right_panel = update_right_panel[update]
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
            _block_fence()
            if lane == 0:
                if tile_i == tile_k:
                    panel_count = task_panel_count[task]
                    if panel_count:
                        first_slot = wp.atomic_add(tail, wp.int32(0), panel_count)
                        for panel_offset in range(panel_count):
                            wp.atomic_exch(
                                queue,
                                first_slot + panel_offset,
                                task + wp.int32(1) + panel_offset,
                            )
                    else:
                        next_diagonal = task_next_diagonal[task]
                        if next_diagonal >= 0:
                            slot = wp.atomic_add(tail, wp.int32(0), wp.int32(1))
                            wp.atomic_exch(queue, slot, next_diagonal)
                else:
                    owner_diagonal = task_owner_diagonal[task]
                    old_remaining = wp.atomic_sub(
                        remaining,
                        owner_diagonal,
                        wp.int32(1),
                    )
                    if old_remaining == wp.int32(1):
                        next_diagonal = task_next_diagonal[owner_diagonal]
                        if next_diagonal >= 0:
                            slot = wp.atomic_add(tail, wp.int32(0), wp.int32(1))
                            wp.atomic_exch(queue, slot, next_diagonal)
                wp.atomic_add(completed, wp.int32(0), wp.int32(1))
            _block_sync()

    return factor_persistent


__all__ = ["initialize_factor_queue", "make_persistent_factor_kernel"]
