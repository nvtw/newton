# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host metadata and launches for persistent panel factorization."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt_queue import (
    initialize_factor_queue,
    make_persistent_factor_kernel,
)


class PersistentFactorSchedule:
    """Run one atomic-ready queue across all mechanism panel tasks."""

    def __init__(
        self,
        panel_tables: list[np.ndarray],
        panel_count: int,
        block_size: int,
        device: wp.Device,
    ):
        task_mechanism: list[int] = []
        task_tile_i: list[int] = []
        task_tile_k: list[int] = []
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
                diagonal_task = len(task_mechanism)
                if previous_diagonal < 0:
                    initial_task.append(diagonal_task)
                else:
                    task_next_diagonal[previous_diagonal] = diagonal_task
                task_mechanism.append(mechanism)
                task_tile_i.append(tile_k)
                task_tile_k.append(tile_k)
                task_panel_count.append(len(panel_rows))
                task_next_diagonal.append(-1)
                task_owner_diagonal.append(diagonal_task)
                remaining_initial.append(len(panel_rows))
                append_symbolic_updates(table, tile_k, tile_k)
                for tile_i in panel_rows:
                    task_mechanism.append(mechanism)
                    task_tile_i.append(tile_i)
                    task_tile_k.append(tile_k)
                    task_panel_count.append(0)
                    task_next_diagonal.append(-1)
                    task_owner_diagonal.append(diagonal_task)
                    remaining_initial.append(0)
                    append_symbolic_updates(table, tile_i, tile_k)
                previous_diagonal = diagonal_task

        if len(task_mechanism) != panel_count:
            raise RuntimeError("persistent task graph must contain one task per symbolic panel")

        def task_array(values: list[int]) -> wp.array[wp.int32]:
            return wp.array(values, dtype=wp.int32, device=device)

        self.task_mechanism = task_array(task_mechanism)
        self.task_tile_i = task_array(task_tile_i)
        self.task_tile_k = task_array(task_tile_k)
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
        self.completed = wp.zeros(1, dtype=wp.int32, device=device)
        self.remaining = wp.zeros(panel_count, dtype=wp.int32, device=device)
        worker_count = min(panel_count, max(1, 2 * device.sm_count))
        self.worker_task = wp.full(worker_count, -1, dtype=wp.int32, device=device)
        self.kernel = make_persistent_factor_kernel(block_size)
        self.device = device

    def compute(
        self,
        dimensions: wp.array[wp.int32],
        panel_table_offset: wp.array[wp.int32],
        tile_counts: wp.array[wp.int32],
        panel_index: wp.array[wp.int32],
        matrix: wp.array[wp.float32],
        factor: wp.array[wp.float32],
    ) -> None:
        """Initialize and drain the dependency-ready panel queue."""
        wp.launch(
            initialize_factor_queue,
            dim=self.task_mechanism.size,
            inputs=[
                self.initial_task,
                self.remaining_initial,
                self.queue,
                self.head,
                self.tail,
                self.completed,
                self.remaining,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            self.kernel,
            dim=self.worker_task.size,
            block_dim=128,
            inputs=[
                self.task_mechanism,
                self.task_tile_i,
                self.task_tile_k,
                self.task_panel_count,
                self.task_next_diagonal,
                self.task_owner_diagonal,
                self.task_update_start,
                self.update_left_panel,
                self.update_right_panel,
                dimensions,
                panel_table_offset,
                tile_counts,
                panel_index,
                matrix,
                factor,
                self.queue,
                self.head,
                self.tail,
                self.completed,
                self.remaining,
                self.worker_task,
            ],
            device=self.device,
        )


__all__ = ["PersistentFactorSchedule"]
