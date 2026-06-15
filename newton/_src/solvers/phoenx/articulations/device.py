# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp storage and kernels for PhoenX articulation DVI systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .topology import ArticulationTopology


@dataclass
class ArticulationDeviceSystem:
    """Device buffers for full-coordinate articulation matrix assembly."""

    total_rows: int
    active_joint_count: int
    body1: wp.array
    body2: wp.array
    row_to_active_block: wp.array
    jacobian: wp.array
    violation: wp.array
    matrix: wp.array
    rhs: wp.array
    solution: wp.array

    @classmethod
    def from_topology(cls, topology: ArticulationTopology, device) -> ArticulationDeviceSystem:
        """Allocate reusable device buffers for ``topology``."""
        total_rows = int(topology.total_rows)
        rows_alloc = max(total_rows, 1)
        body1_np = (
            topology.active_body1.astype(np.int32)
            if topology.active_joint_count > 0
            else np.full(1, -1, dtype=np.int32)
        )
        body2_np = (
            topology.active_body2.astype(np.int32)
            if topology.active_joint_count > 0
            else np.full(1, -1, dtype=np.int32)
        )
        row_to_block_np = (
            topology.row_to_active_block.astype(np.int32) if total_rows > 0 else np.zeros(1, dtype=np.int32)
        )

        return cls(
            total_rows=total_rows,
            active_joint_count=int(topology.active_joint_count),
            body1=wp.array(body1_np, dtype=wp.int32, device=device),
            body2=wp.array(body2_np, dtype=wp.int32, device=device),
            row_to_active_block=wp.array(row_to_block_np, dtype=wp.int32, device=device),
            jacobian=wp.zeros((rows_alloc, 12), dtype=wp.float32, device=device),
            violation=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            matrix=wp.zeros((rows_alloc, rows_alloc), dtype=wp.float32, device=device),
            rhs=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            solution=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
        )

    def assemble_dense_matrix(
        self,
        inverse_mass: wp.array,
        inverse_inertia_world: wp.array,
        *,
        device=None,
    ) -> None:
        """Launch dense ``J W J^T`` assembly into :attr:`matrix`."""
        if self.total_rows <= 0:
            return
        wp.launch(
            _assemble_dense_articulation_matrix_kernel,
            dim=(self.total_rows, self.total_rows),
            inputs=[
                self.jacobian,
                self.body1,
                self.body2,
                self.row_to_active_block,
                inverse_mass,
                inverse_inertia_world,
                wp.int32(self.total_rows),
            ],
            outputs=[self.matrix],
            device=device,
        )


@wp.kernel
def _assemble_dense_articulation_matrix_kernel(
    jacobian: wp.array2d[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    total_rows: wp.int32,
    matrix: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    if row >= total_rows or col >= total_rows:
        return

    block_row = row_to_active_block[row]
    block_col = row_to_active_block[col]
    row_body1 = body1[block_row]
    row_body2 = body2[block_row]
    col_body1 = body1[block_col]
    col_body2 = body2[block_col]

    value = wp.float32(0.0)
    if row_body1 >= 0 and row_body1 == col_body1:
        value += _body_metric_dot(row, col, 0, 0, row_body1, jacobian, inverse_mass, inverse_inertia_world)
    if row_body1 >= 0 and row_body1 == col_body2:
        value += _body_metric_dot(row, col, 0, 6, row_body1, jacobian, inverse_mass, inverse_inertia_world)
    if row_body2 >= 0 and row_body2 == col_body1:
        value += _body_metric_dot(row, col, 6, 0, row_body2, jacobian, inverse_mass, inverse_inertia_world)
    if row_body2 >= 0 and row_body2 == col_body2:
        value += _body_metric_dot(row, col, 6, 6, row_body2, jacobian, inverse_mass, inverse_inertia_world)

    matrix[row, col] = value


@wp.func
def _body_metric_dot(
    row: wp.int32,
    col: wp.int32,
    row_offset: wp.int32,
    col_offset: wp.int32,
    body: wp.int32,
    jacobian: wp.array2d[wp.float32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
    row_lin = wp.vec3(
        jacobian[row, row_offset + 0],
        jacobian[row, row_offset + 1],
        jacobian[row, row_offset + 2],
    )
    row_ang = wp.vec3(
        jacobian[row, row_offset + 3],
        jacobian[row, row_offset + 4],
        jacobian[row, row_offset + 5],
    )
    col_lin = wp.vec3(
        jacobian[col, col_offset + 0],
        jacobian[col, col_offset + 1],
        jacobian[col, col_offset + 2],
    )
    col_ang = wp.vec3(
        jacobian[col, col_offset + 3],
        jacobian[col, col_offset + 4],
        jacobian[col, col_offset + 5],
    )
    value = inverse_mass[body] * wp.dot(row_lin, col_lin)
    value += wp.dot(row_ang, inverse_inertia_world[body] * col_ang)
    return value
