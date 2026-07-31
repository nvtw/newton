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
    active_rows: wp.int32,
    panel_id: wp.int32,
    diagonal_panel: wp.int32,
    update_begin: wp.int32,
    update_end: wp.int32,
    update_left_panel: wp.array[wp.int32],
    update_right_panel: wp.array[wp.int32],
    matrix: wp.array[wp.float32],
    factor: wp.array[wp.float32],
    lane: wp.int32,
    block_size: wp.int32,
):
    """Factor one partial panel row using precomputed symbolic addresses."""
    if lane >= active_rows:
        return
    tile_elements = block_size * block_size
    panel_offset = panel_id * tile_elements
    diagonal_offset = diagonal_panel * tile_elements

    column = wp.int32(0)
    while column < block_size:
        value = matrix[panel_offset + lane * block_size + column]
        update = update_begin
        while update < update_end:
            left_offset = update_left_panel[update] * tile_elements + lane * block_size
            right_offset = update_right_panel[update] * tile_elements + column * block_size
            inner = wp.int32(0)
            while inner < block_size:
                value -= factor[left_offset + inner] * factor[right_offset + inner]
                inner += wp.int32(1)
            update += wp.int32(1)

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
def initialize_panel_queue(
    initial_task: wp.array[wp.int32],
    remaining_initial: wp.array[wp.int32],
    queue: wp.array[wp.int32],
    head: wp.array[wp.int32],
    tail: wp.array[wp.int32],
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


@wp.kernel(enable_backward=False)
def initialize_product_factor_queue(
    initial_task: wp.array[wp.int32],
    remaining_initial: wp.array[wp.int32],
    column_remaining_initial: wp.array[wp.int32],
    queue: wp.array[wp.int32],
    head: wp.array[wp.int32],
    tail: wp.array[wp.int32],
    remaining: wp.array[wp.int32],
    column_remaining: wp.array[wp.int32],
):
    """Reset factor/product dependencies and seed ready factor panels."""
    index = wp.tid()
    queue[index] = wp.int32(-1)
    remaining[index] = remaining_initial[index]
    column_remaining[index] = column_remaining_initial[index]
    if index < initial_task.shape[0]:
        queue[index] = initial_task[index]
    if index == 0:
        head[0] = wp.int32(0)
        tail[0] = initial_task.shape[0]


@wp.kernel(enable_backward=False)
def initialize_push_solve_queue(
    initial_task: wp.array[wp.int32],
    remaining_initial: wp.array[wp.int32],
    workspace_rhs_index: wp.array[wp.int32],
    rhs: wp.array[wp.float32],
    initialize_rhs: wp.bool,
    queue: wp.array[wp.int32],
    head: wp.array[wp.int32],
    tail: wp.array[wp.int32],
    remaining: wp.array[wp.int32],
    intermediate: wp.array[wp.float32],
):
    """Reset a push-solve queue and optionally gather the permuted RHS."""
    index = wp.tid()
    if index < queue.shape[0]:
        queue[index] = wp.int32(-1)
        remaining[index] = remaining_initial[index]
        if index < initial_task.shape[0]:
            queue[index] = initial_task[index]
    if initialize_rhs and index < workspace_rhs_index.shape[0]:
        rhs_index = workspace_rhs_index[index]
        value = wp.float32(0.0)
        if rhs_index >= wp.int32(0):
            value = rhs[rhs_index]
        intermediate[index] = value
    if index == 0:
        head[0] = wp.int32(0)
        tail[0] = initial_task.shape[0]


def make_persistent_factor_kernel(block_size: int):
    """Create a dependency-driven panel factorization worker kernel."""
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def factor_persistent(
        task_panel: wp.array[wp.int32],
        task_diagonal_panel: wp.array[wp.int32],
        task_active_rows: wp.array[wp.int32],
        task_panel_count: wp.array[wp.int32],
        task_next_diagonal: wp.array[wp.int32],
        task_owner_diagonal: wp.array[wp.int32],
        task_update_start: wp.array[wp.int32],
        update_left_panel: wp.array[wp.int32],
        update_right_panel: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
        queue: wp.array[wp.int32],
        head: wp.array[wp.int32],
        tail: wp.array[wp.int32],
        remaining: wp.array[wp.int32],
        worker_task: wp.array[wp.int32],
    ):
        worker, lane = wp.tid()
        task_count = task_panel.shape[0]

        while True:
            if lane == 0:
                queue_slot = wp.atomic_add(head, wp.int32(0), wp.int32(1))
                claimed_task = wp.int32(-2)
                if queue_slot < task_count:
                    claimed_task = wp.int32(-1)
                    while claimed_task == wp.int32(-1):
                        claimed_task = wp.atomic_add(queue, queue_slot, wp.int32(0))
                worker_task[worker] = claimed_task
            _block_sync()

            task = worker_task[worker]
            if task < 0:
                break

            panel_id = task_panel[task]
            diagonal_panel = task_diagonal_panel[task]
            active_rows = task_active_rows[task]

            if panel_id == diagonal_panel:
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
                if active_rows < block_size:
                    for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                        index = (lane + iteration * wp.block_dim()) % tile_elements
                        row = index // block_size
                        column = index % block_size
                        value = diagonal[row, column]
                        if row >= active_rows or column >= active_rows:
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
                if active_rows < block_size:
                    factor_partial_panel_row(
                        active_rows,
                        panel_id,
                        diagonal_panel,
                        task_update_start[task],
                        task_update_start[task + wp.int32(1)],
                        update_left_panel,
                        update_right_panel,
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
                if panel_id == diagonal_panel:
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
            _block_sync()

    return factor_persistent


def make_persistent_product_factor_kernel(block_size: int):
    """Create a deterministic factor graph with independent product tasks."""
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def factor_product_persistent(
        task_kind: wp.array[wp.int32],
        panel_count: wp.int32,
        task_panel: wp.array[wp.int32],
        task_diagonal_panel: wp.array[wp.int32],
        task_active_rows: wp.array[wp.int32],
        task_owner_diagonal: wp.array[wp.int32],
        factor_product_start: wp.array[wp.int32],
        factor_product: wp.array[wp.int32],
        diagonal_child_start: wp.array[wp.int32],
        diagonal_child: wp.array[wp.int32],
        column_product_start: wp.array[wp.int32],
        column_product_count: wp.array[wp.int32],
        product_left_panel: wp.array[wp.int32],
        product_right_panel: wp.array[wp.int32],
        product_target_factor: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
        contribution: wp.array[wp.float32],
        queue: wp.array[wp.int32],
        head: wp.array[wp.int32],
        tail: wp.array[wp.int32],
        remaining: wp.array[wp.int32],
        column_remaining: wp.array[wp.int32],
        worker_task: wp.array[wp.int32],
    ):
        worker, lane = wp.tid()
        task_count = task_kind.shape[0]

        while True:
            if lane == 0:
                queue_slot = wp.atomic_add(head, wp.int32(0), wp.int32(1))
                claimed_task = wp.int32(-2)
                if queue_slot < task_count:
                    claimed_task = wp.int32(-1)
                    while claimed_task == wp.int32(-1):
                        claimed_task = wp.atomic_add(queue, queue_slot, wp.int32(0))
                worker_task[worker] = claimed_task
            _block_sync()

            task = worker_task[worker]
            if task < 0:
                break

            if task_kind[task] == wp.int32(0):
                panel_id = task_panel[task]
                diagonal_panel = task_diagonal_panel[task]
                active_rows = task_active_rows[task]
                panel_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(matrix, panel_id * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                panel = wp.tile_load(panel_matrix, shape=(block_size, block_size), storage="shared")
                for product_entry in range(
                    factor_product_start[task],
                    factor_product_start[task + wp.int32(1)],
                ):
                    product_task = factor_product[product_entry]
                    product_index = product_task - panel_count
                    product_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(contribution, product_index * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    product = wp.tile_load(product_matrix, shape=(block_size, block_size))
                    panel = wp.tile_map(wp.sub, panel, product)

                if panel_id == diagonal_panel:
                    if active_rows < block_size:
                        for iteration in range((tile_elements + wp.block_dim() - 1) // wp.block_dim()):
                            index = (lane + iteration * wp.block_dim()) % tile_elements
                            row = index // block_size
                            column = index % block_size
                            value = panel[row, column]
                            if row >= active_rows or column >= active_rows:
                                value = wp.where(row == column, wp.float32(1.0), wp.float32(0.0))
                            panel[row, column] = value
                    wp.tile_cholesky_inplace(panel)
                    factor_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    wp.tile_store(factor_matrix, panel)
                else:
                    diagonal_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size), storage="shared")
                    transposed = wp.tile_transpose(panel)
                    wp.tile_lower_solve_inplace(diagonal, transposed)
                    factor_matrix = wp.array(
                        ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements),
                        shape=(block_size, block_size),
                        dtype=wp.float32,
                    )
                    wp.tile_store(factor_matrix, wp.tile_transpose(transposed))
            else:
                product_index = task - panel_count
                left_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(
                        factor,
                        product_left_panel[product_index] * tile_elements,
                    ),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                right_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(
                        factor,
                        product_right_panel[product_index] * tile_elements,
                    ),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                product_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(contribution, product_index * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                left = wp.tile_load(left_matrix, shape=(block_size, block_size))
                right = wp.tile_load(right_matrix, shape=(block_size, block_size))
                product = wp.tile_zeros(shape=(block_size, block_size), dtype=wp.float32, storage="shared")
                wp.tile_matmul(left, wp.tile_transpose(right), product)
                wp.tile_store(product_matrix, product)

            _block_fence()
            if lane == 0:
                if task_kind[task] == wp.int32(0):
                    panel_id = task_panel[task]
                    diagonal_panel = task_diagonal_panel[task]
                    if panel_id == diagonal_panel:
                        for child_entry in range(
                            diagonal_child_start[task],
                            diagonal_child_start[task + wp.int32(1)],
                        ):
                            child = diagonal_child[child_entry]
                            old_remaining = wp.atomic_sub(remaining, child, wp.int32(1))
                            if old_remaining == wp.int32(1):
                                slot = wp.atomic_add(tail, wp.int32(0), wp.int32(1))
                                wp.atomic_exch(queue, slot, child)
                    else:
                        owner = task_owner_diagonal[task]
                        old_column_remaining = wp.atomic_sub(column_remaining, owner, wp.int32(1))
                        if old_column_remaining == wp.int32(1):
                            product_count = column_product_count[owner]
                            if product_count > wp.int32(0):
                                first_slot = wp.atomic_add(tail, wp.int32(0), product_count)
                                first_task = column_product_start[owner]
                                for output in range(product_count):
                                    wp.atomic_exch(queue, first_slot + output, first_task + output)
                else:
                    product_index = task - panel_count
                    target = product_target_factor[product_index]
                    old_remaining = wp.atomic_sub(remaining, target, wp.int32(1))
                    if old_remaining == wp.int32(1):
                        slot = wp.atomic_add(tail, wp.int32(0), wp.int32(1))
                        wp.atomic_exch(queue, slot, target)
            _block_sync()

    return factor_product_persistent


def make_persistent_push_solve_kernel(block_size: int):
    """Create a deterministic panel-update kernel for one triangular direction."""
    tile_elements = block_size * block_size

    @wp.kernel(enable_backward=False)
    def solve_push_persistent(
        task_kind: wp.array[wp.int32],
        forward: wp.bool,
        task_mechanism: wp.array[wp.int32],
        task_source_tile: wp.array[wp.int32],
        task_panel: wp.array[wp.int32],
        task_output_start: wp.array[wp.int32],
        task_output_count: wp.array[wp.int32],
        task_owner_diagonal: wp.array[wp.int32],
        dependency_start: wp.array[wp.int32],
        dependency_task: wp.array[wp.int32],
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
        contribution: wp.array[wp.float32],
        queue: wp.array[wp.int32],
        head: wp.array[wp.int32],
        tail: wp.array[wp.int32],
        remaining: wp.array[wp.int32],
        worker_task: wp.array[wp.int32],
    ):
        worker, lane = wp.tid()
        task_count = task_kind.shape[0]

        while True:
            if lane == 0:
                queue_slot = wp.atomic_add(head, wp.int32(0), wp.int32(1))
                claimed_task = wp.int32(-2)
                if queue_slot < task_count:
                    claimed_task = wp.int32(-1)
                    while claimed_task == wp.int32(-1):
                        claimed_task = wp.atomic_add(queue, queue_slot, wp.int32(0))
                worker_task[worker] = claimed_task
            _block_sync()

            task = worker_task[worker]
            if task < 0:
                break

            mechanism = task_mechanism[task]
            source_tile = task_source_tile[task]
            dimension = dimensions[mechanism]
            vector_offset = vector_offsets[mechanism]
            workspace_offset = workspace_offsets[mechanism]
            tile_count = tile_counts[mechanism]
            table_offset = panel_table_offset[mechanism]
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

            if task_kind[task] == wp.int32(0):
                i = source_tile * block_size
                right_hand_side = wp.tile_zeros(shape=(block_size, 1), dtype=wp.float32, storage="shared")
                for iteration in range((block_size + wp.block_dim() - 1) // wp.block_dim()):
                    row = lane + iteration * wp.block_dim()
                    active = row < block_size
                    value = wp.float32(0.0)
                    if active:
                        value = intermediate[workspace_offset + i + row]
                        for dependency in range(
                            dependency_start[task],
                            dependency_start[task + wp.int32(1)],
                        ):
                            update_task = dependency_task[dependency]
                            value -= contribution[update_task * block_size + row]
                    wp.tile_scatter_masked(right_hand_side, row, 0, value, active)

                diagonal_panel = panel_index[table_offset + source_tile * tile_count + source_tile]
                diagonal_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, diagonal_panel * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                diagonal = wp.tile_load(diagonal_matrix, shape=(block_size, block_size))
                if forward:
                    wp.tile_lower_solve_inplace(diagonal, right_hand_side)
                    wp.tile_store(intermediate_matrix, right_hand_side, offset=(i, 0))
                else:
                    wp.tile_upper_solve_inplace(wp.tile_transpose(diagonal), right_hand_side)
                    wp.tile_store(solution_permuted_matrix, right_hand_side, offset=(i, 0))
                    for iteration in range((block_size + wp.block_dim() - 1) // wp.block_dim()):
                        row = lane + iteration * wp.block_dim()
                        if row < block_size and i + row < dimension:
                            original = permutation[vector_offset + i + row]
                            solution[vector_offset + original] = right_hand_side[row, 0]
            else:
                panel_id = task_panel[task]
                factor_matrix = wp.array(
                    ptr=_get_float_array_offset_ptr(factor, panel_id * tile_elements),
                    shape=(block_size, block_size),
                    dtype=wp.float32,
                )
                lower = wp.tile_load(factor_matrix, shape=(block_size, block_size))
                product = wp.tile_zeros(shape=(block_size, 1), dtype=wp.float32, storage="shared")
                if forward:
                    solved = wp.tile_load(
                        intermediate_matrix,
                        shape=(block_size, 1),
                        offset=(source_tile * block_size, 0),
                    )
                    wp.tile_matmul(lower, solved, product)
                else:
                    solved = wp.tile_load(
                        solution_permuted_matrix,
                        shape=(block_size, 1),
                        offset=(source_tile * block_size, 0),
                    )
                    wp.tile_matmul(wp.tile_transpose(lower), solved, product)
                for iteration in range((block_size + wp.block_dim() - 1) // wp.block_dim()):
                    row = lane + iteration * wp.block_dim()
                    if row < block_size:
                        contribution[task * block_size + row] = product[row, 0]

            _block_fence()
            if lane == 0:
                if task_kind[task] == wp.int32(0):
                    output_count = task_output_count[task]
                    if output_count > wp.int32(0):
                        first_slot = wp.atomic_add(tail, wp.int32(0), output_count)
                        first_task = task_output_start[task]
                        for output in range(output_count):
                            wp.atomic_exch(queue, first_slot + output, first_task + output)
                else:
                    owner_diagonal = task_owner_diagonal[task]
                    old_remaining = wp.atomic_sub(remaining, owner_diagonal, wp.int32(1))
                    if old_remaining == wp.int32(1):
                        slot = wp.atomic_add(tail, wp.int32(0), wp.int32(1))
                        wp.atomic_exch(queue, slot, owner_diagonal)
            _block_sync()

    return solve_push_persistent


__all__ = [
    "initialize_panel_queue",
    "initialize_product_factor_queue",
    "initialize_push_solve_queue",
    "make_persistent_factor_kernel",
    "make_persistent_product_factor_kernel",
    "make_persistent_push_solve_kernel",
]
