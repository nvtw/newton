# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host metadata and launches for persistent panel factorization."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import (
    initialize_panel_queue,
    initialize_product_factor_queue,
    initialize_push_solve_queue,
    make_persistent_factor_kernel,
    make_persistent_product_factor_kernel,
    make_persistent_push_solve_kernel,
)


class PersistentFactorSchedule:
    """Run one atomic-ready queue across all mechanism panel tasks."""

    def __init__(
        self,
        panel_tables: list[np.ndarray],
        dimensions: tuple[int, ...],
        panel_count: int,
        block_size: int,
        device: wp.Device,
    ):
        task_panel: list[int] = []
        task_diagonal_panel: list[int] = []
        task_active_rows: list[int] = []
        task_panel_count: list[int] = []
        task_next_diagonal: list[int] = []
        task_owner_diagonal: list[int] = []
        task_update_start: list[int] = [0]
        update_left_panel: list[int] = []
        update_right_panel: list[int] = []
        remaining_initial: list[int] = []
        initial_task: list[int] = []

        def append_symbolic_updates(table: np.ndarray, tile_i: int, tile_k: int) -> None:
            """Record only numerical panel products present in the symbolic factor."""
            for tile_j in range(tile_k):
                left_panel = int(table[tile_i, tile_j])
                right_panel = int(table[tile_k, tile_j])
                if left_panel >= 0 and right_panel >= 0:
                    update_left_panel.append(left_panel)
                    update_right_panel.append(right_panel)
            task_update_start.append(len(update_left_panel))

        for mechanism, table in enumerate(panel_tables):
            previous_diagonal = -1
            for tile_k in range(table.shape[0]):
                panel_rows = [tile_i for tile_i in range(tile_k + 1, table.shape[0]) if table[tile_i, tile_k] >= 0]
                diagonal_task = len(task_panel)
                if previous_diagonal < 0:
                    initial_task.append(diagonal_task)
                else:
                    task_next_diagonal[previous_diagonal] = diagonal_task
                task_panel.append(int(table[tile_k, tile_k]))
                task_diagonal_panel.append(int(table[tile_k, tile_k]))
                task_active_rows.append(min(block_size, dimensions[mechanism] - tile_k * block_size))
                task_panel_count.append(len(panel_rows))
                task_next_diagonal.append(-1)
                task_owner_diagonal.append(diagonal_task)
                remaining_initial.append(len(panel_rows))
                append_symbolic_updates(table, tile_k, tile_k)
                for tile_i in panel_rows:
                    task_panel.append(int(table[tile_i, tile_k]))
                    task_diagonal_panel.append(int(table[tile_k, tile_k]))
                    task_active_rows.append(min(block_size, dimensions[mechanism] - tile_i * block_size))
                    task_panel_count.append(0)
                    task_next_diagonal.append(-1)
                    task_owner_diagonal.append(diagonal_task)
                    remaining_initial.append(0)
                    append_symbolic_updates(table, tile_i, tile_k)
                previous_diagonal = diagonal_task

        if len(task_panel) != panel_count:
            raise RuntimeError("persistent task graph must contain one task per symbolic panel")

        def task_array(values: list[int]) -> wp.array[wp.int32]:
            return wp.array(values, dtype=wp.int32, device=device)

        self.task_panel = task_array(task_panel)
        self.task_diagonal_panel = task_array(task_diagonal_panel)
        self.task_active_rows = task_array(task_active_rows)
        self.task_panel_count = task_array(task_panel_count)
        self.task_next_diagonal = task_array(task_next_diagonal)
        self.task_owner_diagonal = task_array(task_owner_diagonal)
        self.task_update_start = task_array(task_update_start)
        self.update_left_panel = task_array(update_left_panel)
        self.update_right_panel = task_array(update_right_panel)
        self.remaining_initial = task_array(remaining_initial)
        self.initial_task = task_array(initial_task)
        self.queue = wp.full(panel_count, -1, dtype=wp.int32, device=device)
        self.head = wp.zeros(1, dtype=wp.int32, device=device)
        self.tail = wp.zeros(1, dtype=wp.int32, device=device)
        self.remaining = wp.zeros(panel_count, dtype=wp.int32, device=device)
        worker_count = min(panel_count, max(1, 4 * device.sm_count))
        self.worker_task = wp.full(worker_count, -1, dtype=wp.int32, device=device)
        self.kernel = make_persistent_factor_kernel(block_size)
        self.block_dim = 128 if block_size == 32 else 64
        self.device = device

    def compute(
        self,
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ) -> None:
        """Initialize and drain the dependency-ready panel queue."""
        wp.launch(
            initialize_panel_queue,
            dim=self.task_panel.size,
            inputs=[
                self.initial_task,
                self.remaining_initial,
                self.queue,
                self.head,
                self.tail,
                self.remaining,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self.kernel,
            dim=self.worker_task.size,
            block_dim=self.block_dim,
            inputs=[
                self.task_panel,
                self.task_diagonal_panel,
                self.task_active_rows,
                self.task_panel_count,
                self.task_next_diagonal,
                self.task_owner_diagonal,
                self.task_update_start,
                self.update_left_panel,
                self.update_right_panel,
                matrix,
                factor,
                self.queue,
                self.head,
                self.tail,
                self.remaining,
                self.worker_task,
            ],
            device=self.device,
        )


class PersistentProductFactorSchedule:
    """Factor panels after deterministic, independently scheduled products."""

    def __init__(
        self,
        panel_tables: list[np.ndarray],
        dimensions: tuple[int, ...],
        panel_count: int,
        block_size: int,
        device: wp.Device,
    ):
        task_panel = [-1] * panel_count
        task_diagonal_panel = [-1] * panel_count
        task_active_rows = [0] * panel_count
        task_owner_diagonal = [-1] * panel_count
        diagonal_children: list[list[int]] = [[] for _ in range(panel_count)]
        target_products: list[list[int]] = [[] for _ in range(panel_count)]
        column_product_start = [-1] * panel_count
        column_product_count = [0] * panel_count
        product_left_panel: list[int] = []
        product_right_panel: list[int] = []
        product_target_factor: list[int] = []

        for mechanism, table in enumerate(panel_tables):
            for tile_k in range(table.shape[0]):
                diagonal = int(table[tile_k, tile_k])
                task_panel[diagonal] = diagonal
                task_diagonal_panel[diagonal] = diagonal
                task_active_rows[diagonal] = min(block_size, dimensions[mechanism] - tile_k * block_size)
                panel_rows = [tile_i for tile_i in range(tile_k + 1, table.shape[0]) if table[tile_i, tile_k] >= 0]
                for tile_i in panel_rows:
                    panel = int(table[tile_i, tile_k])
                    task_panel[panel] = panel
                    task_diagonal_panel[panel] = diagonal
                    task_active_rows[panel] = min(block_size, dimensions[mechanism] - tile_i * block_size)
                    task_owner_diagonal[panel] = diagonal
                    diagonal_children[diagonal].append(panel)

                first_product = panel_count + len(product_left_panel)
                for right_position, tile_j in enumerate(panel_rows):
                    right_panel = int(table[tile_j, tile_k])
                    for tile_i in panel_rows[right_position:]:
                        left_panel = int(table[tile_i, tile_k])
                        target_panel = int(table[tile_i, tile_j])
                        if target_panel < 0:
                            raise RuntimeError("symbolic Cholesky fill omitted a product target panel")
                        product_task = panel_count + len(product_left_panel)
                        product_left_panel.append(left_panel)
                        product_right_panel.append(right_panel)
                        product_target_factor.append(target_panel)
                        target_products[target_panel].append(product_task)
                column_product_start[diagonal] = first_product
                column_product_count[diagonal] = len(product_left_panel) - (first_product - panel_count)

        if any(panel < 0 for panel in task_panel):
            raise RuntimeError("product factor schedule did not cover every symbolic panel")
        product_count = len(product_left_panel)
        task_count = panel_count + product_count
        task_kind = [0] * panel_count + [1] * product_count
        task_panel.extend([-1] * product_count)
        task_diagonal_panel.extend([-1] * product_count)
        task_active_rows.extend([0] * product_count)
        task_owner_diagonal.extend([-1] * product_count)
        column_product_start.extend([-1] * product_count)
        column_product_count.extend([0] * product_count)

        factor_product_start = [0]
        factor_product: list[int] = []
        diagonal_child_start = [0]
        diagonal_child: list[int] = []
        remaining_initial = [0] * task_count
        column_remaining_initial = [0] * task_count
        initial_task: list[int] = []
        for task in range(task_count):
            if task < panel_count:
                factor_product.extend(target_products[task])
                diagonal_child.extend(diagonal_children[task])
                remaining_initial[task] = len(target_products[task]) + int(task_owner_diagonal[task] >= 0)
                column_remaining_initial[task] = len(diagonal_children[task])
                if remaining_initial[task] == 0:
                    initial_task.append(task)
            factor_product_start.append(len(factor_product))
            diagonal_child_start.append(len(diagonal_child))

        ready = initial_task.copy()
        remaining_work = remaining_initial.copy()
        column_work = column_remaining_initial.copy()
        max_ready_count = 0
        completed_count = 0
        while ready:
            max_ready_count = max(max_ready_count, len(ready))
            next_ready: list[int] = []
            for task in ready:
                completed_count += 1
                if task < panel_count:
                    diagonal = task_diagonal_panel[task]
                    if task == diagonal:
                        for child in diagonal_children[task]:
                            remaining_work[child] -= 1
                            if remaining_work[child] == 0:
                                next_ready.append(child)
                    else:
                        owner = task_owner_diagonal[task]
                        column_work[owner] -= 1
                        if column_work[owner] == 0:
                            begin = column_product_start[owner]
                            next_ready.extend(range(begin, begin + column_product_count[owner]))
                else:
                    target = product_target_factor[task - panel_count]
                    remaining_work[target] -= 1
                    if remaining_work[target] == 0:
                        next_ready.append(target)
            ready = next_ready
        if completed_count != task_count:
            raise RuntimeError("product factor task graph must be acyclic")
        self.max_ready_count = max_ready_count
        self.product_count = product_count

        def task_array(values: list[int]) -> wp.array[wp.int32]:
            return wp.array(values, dtype=wp.int32, device=device)

        self.task_kind = task_array(task_kind)
        self.task_panel = task_array(task_panel)
        self.task_diagonal_panel = task_array(task_diagonal_panel)
        self.task_active_rows = task_array(task_active_rows)
        self.task_owner_diagonal = task_array(task_owner_diagonal)
        self.factor_product_start = task_array(factor_product_start)
        self.factor_product = task_array(factor_product)
        self.diagonal_child_start = task_array(diagonal_child_start)
        self.diagonal_child = task_array(diagonal_child)
        self.column_product_start = task_array(column_product_start)
        self.column_product_count = task_array(column_product_count)
        self.product_left_panel = task_array(product_left_panel)
        self.product_right_panel = task_array(product_right_panel)
        self.product_target_factor = task_array(product_target_factor)
        self.remaining_initial = task_array(remaining_initial)
        self.column_remaining_initial = task_array(column_remaining_initial)
        self.initial_task = task_array(initial_task)
        self.contribution = wp.zeros(product_count * block_size * block_size, dtype=wp.float32, device=device)
        self.queue = wp.full(task_count, -1, dtype=wp.int32, device=device)
        self.head = wp.zeros(1, dtype=wp.int32, device=device)
        self.tail = wp.zeros(1, dtype=wp.int32, device=device)
        self.remaining = wp.zeros(task_count, dtype=wp.int32, device=device)
        self.column_remaining = wp.zeros(task_count, dtype=wp.int32, device=device)
        worker_count = min(task_count, max(1, 4 * device.sm_count))
        self.worker_task = wp.full(worker_count, -1, dtype=wp.int32, device=device)
        self.panel_count = panel_count
        self.kernel = make_persistent_product_factor_kernel(block_size)
        self.block_dim = 128 if block_size == 32 else 64
        self.device = device

    def compute(
        self,
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ) -> None:
        """Initialize and drain deterministic factor and product tasks."""
        wp.launch(
            initialize_product_factor_queue,
            dim=self.task_kind.size,
            inputs=[
                self.initial_task,
                self.remaining_initial,
                self.column_remaining_initial,
                self.queue,
                self.head,
                self.tail,
                self.remaining,
                self.column_remaining,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self.kernel,
            dim=self.worker_task.size,
            block_dim=self.block_dim,
            inputs=[
                self.task_kind,
                wp.int32(self.panel_count),
                self.task_panel,
                self.task_diagonal_panel,
                self.task_active_rows,
                self.task_owner_diagonal,
                self.factor_product_start,
                self.factor_product,
                self.diagonal_child_start,
                self.diagonal_child,
                self.column_product_start,
                self.column_product_count,
                self.product_left_panel,
                self.product_right_panel,
                self.product_target_factor,
                matrix,
                factor,
                self.contribution,
                self.queue,
                self.head,
                self.tail,
                self.remaining,
                self.column_remaining,
                self.worker_task,
            ],
            device=self.device,
        )


class PersistentPushSolveSchedule:
    """Run deterministic triangular panel updates through one ready queue."""

    def __init__(
        self,
        panel_tables: list[np.ndarray],
        mechanisms: np.ndarray,
        workspace_rhs_index: np.ndarray,
        block_size: int,
        device: wp.Device,
        *,
        forward: bool,
    ):
        task_kind: list[int] = []
        task_mechanism: list[int] = []
        task_source_tile: list[int] = []
        task_panel: list[int] = []
        task_output_start: list[int] = []
        task_output_count: list[int] = []
        task_owner_diagonal: list[int] = []
        remaining_initial: list[int] = []
        initial_task: list[int] = []

        for mechanism_np in mechanisms:
            mechanism = int(mechanism_np)
            table = panel_tables[mechanism]
            tile_count = table.shape[0]
            sources = list(range(tile_count)) if forward else list(range(tile_count - 1, -1, -1))
            outgoing: dict[int, list[tuple[int, int]]] = {}
            diagonal_task: dict[int, int] = {}
            task_cursor = len(task_kind)
            for source in sources:
                if forward:
                    updates = [
                        (target, int(table[target, source]))
                        for target in range(source + 1, tile_count)
                        if table[target, source] >= 0
                    ]
                else:
                    updates = [
                        (target, int(table[source, target])) for target in range(source) if table[source, target] >= 0
                    ]
                outgoing[source] = updates
                diagonal_task[source] = task_cursor
                task_cursor += 1 + len(updates)

            for source in sources:
                updates = outgoing[source]
                diagonal = diagonal_task[source]
                if diagonal != len(task_kind):
                    raise RuntimeError("push-solve tasks must keep each diagonal with its outgoing panels")
                if forward:
                    dependency_count = sum(table[source, predecessor] >= 0 for predecessor in range(source))
                else:
                    dependency_count = sum(table[successor, source] >= 0 for successor in range(source + 1, tile_count))
                task_kind.append(0)
                task_mechanism.append(mechanism)
                task_source_tile.append(source)
                task_panel.append(int(table[source, source]))
                task_output_start.append(diagonal + 1)
                task_output_count.append(len(updates))
                task_owner_diagonal.append(-1)
                remaining_initial.append(dependency_count)
                if dependency_count == 0:
                    initial_task.append(diagonal)
                for target, panel in updates:
                    task_kind.append(1)
                    task_mechanism.append(mechanism)
                    task_source_tile.append(source)
                    task_panel.append(panel)
                    task_output_start.append(-1)
                    task_output_count.append(0)
                    task_owner_diagonal.append(diagonal_task[target])
                    remaining_initial.append(0)

        owner_dependencies: list[list[int]] = [[] for _ in task_kind]
        for task, owner in enumerate(task_owner_diagonal):
            if owner >= 0:
                owner_dependencies[owner].append(task)
        dependency_start = [0]
        dependency_task: list[int] = []
        for task, dependencies in enumerate(owner_dependencies):
            if task_kind[task] == 0:
                dependencies.sort(key=task_source_tile.__getitem__)
                dependency_task.extend(dependencies)
            dependency_start.append(len(dependency_task))

        ready = initial_task.copy()
        remaining_work = remaining_initial.copy()
        max_ready_count = 0
        completed_count = 0
        while ready:
            max_ready_count = max(max_ready_count, len(ready))
            next_ready: list[int] = []
            for task in ready:
                completed_count += 1
                if task_kind[task] == 0:
                    begin = task_output_start[task]
                    next_ready.extend(range(begin, begin + task_output_count[task]))
                else:
                    owner = task_owner_diagonal[task]
                    remaining_work[owner] -= 1
                    if remaining_work[owner] == 0:
                        next_ready.append(owner)
            ready = next_ready
        if completed_count != len(task_kind):
            raise RuntimeError("push triangular solve task graph must be acyclic")
        self.max_ready_count = max_ready_count

        def task_array(values: list[int]) -> wp.array[wp.int32]:
            return wp.array(values, dtype=wp.int32, device=device)

        task_count = len(task_kind)
        self.task_kind = task_array(task_kind)
        self.task_mechanism = task_array(task_mechanism)
        self.task_source_tile = task_array(task_source_tile)
        self.task_panel = task_array(task_panel)
        self.task_output_start = task_array(task_output_start)
        self.task_output_count = task_array(task_output_count)
        self.task_owner_diagonal = task_array(task_owner_diagonal)
        self.dependency_start = task_array(dependency_start)
        self.dependency_task = task_array(dependency_task)
        self.remaining_initial = task_array(remaining_initial)
        self.initial_task = task_array(initial_task)
        self.workspace_rhs_index = wp.array(workspace_rhs_index, dtype=wp.int32, device=device)
        self.contribution = wp.zeros(task_count * block_size, dtype=wp.float32, device=device)
        self.queue = wp.full(task_count, -1, dtype=wp.int32, device=device)
        self.head = wp.zeros(1, dtype=wp.int32, device=device)
        self.tail = wp.zeros(1, dtype=wp.int32, device=device)
        self.remaining = wp.zeros(task_count, dtype=wp.int32, device=device)
        worker_count = min(task_count, max(1, 2 * device.sm_count))
        self.worker_task = wp.full(worker_count, -1, dtype=wp.int32, device=device)
        self.kernel = make_persistent_push_solve_kernel(block_size)
        self.block_dim = 128 if block_size == 32 else 64
        self.forward = forward
        self.device = device

    def solve(
        self,
        dimensions: wp.array[wp.int32],
        vector_offsets: wp.array[wp.int32],
        workspace_offsets: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        permutation: wp.array[wp.int32],
        factor: wp.array[wp.float32],
        rhs: wp.array[wp.float32],
        intermediate: wp.array[wp.float32],
        solution_permuted: wp.array[wp.float32],
        solution: wp.array[wp.float32],
    ) -> None:
        """Initialize and drain deterministic diagonal and panel-update tasks."""
        initialization_dim = self.task_kind.size
        if self.forward:
            initialization_dim = max(initialization_dim, self.workspace_rhs_index.size)
        wp.launch(
            initialize_push_solve_queue,
            dim=initialization_dim,
            inputs=[
                self.initial_task,
                self.remaining_initial,
                self.workspace_rhs_index,
                rhs,
                wp.bool(self.forward),
                self.queue,
                self.head,
                self.tail,
                self.remaining,
                intermediate,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self.kernel,
            dim=self.worker_task.size,
            block_dim=self.block_dim,
            inputs=[
                self.task_kind,
                wp.bool(self.forward),
                self.task_mechanism,
                self.task_source_tile,
                self.task_panel,
                self.task_output_start,
                self.task_output_count,
                self.task_owner_diagonal,
                self.dependency_start,
                self.dependency_task,
                dimensions,
                vector_offsets,
                workspace_offsets,
                panel_table_offset,
                tile_counts,
                panel_index,
                permutation,
                factor,
                intermediate,
                solution_permuted,
                solution,
                self.contribution,
                self.queue,
                self.head,
                self.tail,
                self.remaining,
                self.worker_task,
            ],
            device=self.device,
        )


__all__ = [
    "PersistentFactorSchedule",
    "PersistentProductFactorSchedule",
    "PersistentPushSolveSchedule",
]
