# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp storage and kernels for PhoenX articulation DVI systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer, read_int, read_vec3
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _OFF_AXIS_LOCAL1,
    _OFF_BODY1,
    _OFF_BODY2,
    _OFF_JOINT_MODE,
    _OFF_LA1_B1,
    _OFF_LA1_B2,
    _OFF_LA2_B1,
    _OFF_LA2_B2,
    _OFF_LA3_B1,
    _OFF_LA3_B2,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)

from .symbolic import BlockSparseSymbolic, compute_block_sparse_symbolic
from .topology import ArticulationTopology

_BLOCK_SIZE = 6


@dataclass
class ArticulationDeviceSystem:
    """Device buffers for full-coordinate articulation matrix assembly."""

    total_rows: int
    active_joint_count: int
    body1: wp.array
    body2: wp.array
    active_joint_indices: wp.array
    active_block_offsets: wp.array
    row_to_active_block: wp.array
    jacobian: wp.array
    violation: wp.array
    matrix: wp.array
    rhs: wp.array
    solution: wp.array
    block_size: int
    block_count: int
    block_nnz: int
    block_pivot_order: wp.array
    block_sizes: wp.array
    block_n_off_row_idx: wp.array
    block_n_off_col_idx: wp.array
    block_diag: wp.array
    block_off: wp.array

    @classmethod
    def from_topology(
        cls,
        topology: ArticulationTopology,
        device,
        symbolic: BlockSparseSymbolic | None = None,
    ) -> ArticulationDeviceSystem:
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
        active_joint_indices_np = (
            topology.active_joint_indices.astype(np.int32)
            if topology.active_joint_count > 0
            else np.full(1, -1, dtype=np.int32)
        )
        active_block_offsets_np = (
            topology.active_block_offsets.astype(np.int32)
            if topology.active_joint_count > 0
            else np.zeros(1, dtype=np.int32)
        )
        if symbolic is None:
            symbolic = compute_block_sparse_symbolic(
                topology.active_body1,
                topology.active_body2,
                topology.active_row_counts,
            )
        block_count = int(symbolic.num_blocks)
        block_nnz = int(symbolic.nnz_n)
        block_pivot_order_np = symbolic.pivot_order.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        block_sizes_np = symbolic.block_sizes.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        block_n_off_row_idx_np = (
            symbolic.n_off_row_idx.astype(np.int32) if block_nnz > 0 else np.zeros(1, dtype=np.int32)
        )
        block_n_off_col_idx_np = (
            symbolic.n_off_col_idx.astype(np.int32) if block_nnz > 0 else np.zeros(1, dtype=np.int32)
        )

        return cls(
            total_rows=total_rows,
            active_joint_count=int(topology.active_joint_count),
            body1=wp.array(body1_np, dtype=wp.int32, device=device),
            body2=wp.array(body2_np, dtype=wp.int32, device=device),
            active_joint_indices=wp.array(active_joint_indices_np, dtype=wp.int32, device=device),
            active_block_offsets=wp.array(active_block_offsets_np, dtype=wp.int32, device=device),
            row_to_active_block=wp.array(row_to_block_np, dtype=wp.int32, device=device),
            jacobian=wp.zeros((rows_alloc, 12), dtype=wp.float32, device=device),
            violation=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            matrix=wp.zeros((rows_alloc, rows_alloc), dtype=wp.float32, device=device),
            rhs=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            solution=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            block_size=_BLOCK_SIZE,
            block_count=block_count,
            block_nnz=block_nnz,
            block_pivot_order=wp.array(block_pivot_order_np, dtype=wp.int32, device=device),
            block_sizes=wp.array(block_sizes_np, dtype=wp.int32, device=device),
            block_n_off_row_idx=wp.array(block_n_off_row_idx_np, dtype=wp.int32, device=device),
            block_n_off_col_idx=wp.array(block_n_off_col_idx_np, dtype=wp.int32, device=device),
            block_diag=wp.zeros((max(block_count, 1), _BLOCK_SIZE, _BLOCK_SIZE), dtype=wp.float32, device=device),
            block_off=wp.zeros((max(block_nnz, 1), _BLOCK_SIZE, _BLOCK_SIZE), dtype=wp.float32, device=device),
        )

    def populate_from_adbs_constraints(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        *,
        device=None,
    ) -> None:
        """Populate compact DVI rows from initialized ADBS joint columns."""
        if self.total_rows <= 0 or self.active_joint_count <= 0:
            return
        wp.launch(
            _populate_adbs_articulation_rows_kernel,
            dim=self.active_joint_count,
            inputs=[
                constraints,
                bodies,
                self.active_joint_indices,
                self.active_block_offsets,
                wp.int32(self.active_joint_count),
            ],
            outputs=[self.jacobian, self.violation],
            device=device,
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

    def assemble_block_sparse_matrix(
        self,
        inverse_mass: wp.array,
        inverse_inertia_world: wp.array,
        *,
        diagonal_regularization: float = 0.0,
        device=None,
    ) -> None:
        """Launch block-sparse ``J W J^T`` assembly in symbolic pivot order."""
        if self.total_rows <= 0 or self.block_count <= 0:
            return
        wp.launch(
            _assemble_block_sparse_articulation_diag_kernel,
            dim=self.block_count,
            inputs=[
                self.jacobian,
                self.body1,
                self.body2,
                self.row_to_active_block,
                self.active_block_offsets,
                self.block_pivot_order,
                self.block_sizes,
                inverse_mass,
                inverse_inertia_world,
                wp.float32(diagonal_regularization),
                wp.int32(self.block_count),
            ],
            outputs=[self.block_diag],
            device=device,
        )
        if self.block_nnz <= 0:
            return
        wp.launch(
            _assemble_block_sparse_articulation_off_kernel,
            dim=self.block_nnz,
            inputs=[
                self.jacobian,
                self.body1,
                self.body2,
                self.row_to_active_block,
                self.active_block_offsets,
                self.block_pivot_order,
                self.block_sizes,
                self.block_n_off_row_idx,
                self.block_n_off_col_idx,
                inverse_mass,
                inverse_inertia_world,
                wp.int32(self.block_nnz),
            ],
            outputs=[self.block_off],
            device=device,
        )

    def compute_residual(
        self,
        bodies: BodyContainer,
        *,
        dt: float,
        alpha: float = 0.0,
        recovery_speed: float = -1.0,
        device=None,
    ) -> None:
        """Compute ``-(J v + phi / (dt + alpha))`` into :attr:`rhs`."""
        if self.total_rows <= 0:
            return
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if dt + alpha <= 0.0:
            raise ValueError(f"dt + alpha must be positive, got {dt + alpha}")
        wp.launch(
            _compute_articulation_residual_kernel,
            dim=self.total_rows,
            inputs=[
                self.jacobian,
                self.violation,
                self.body1,
                self.body2,
                self.row_to_active_block,
                bodies,
                wp.float32(dt),
                wp.float32(alpha),
                wp.float32(recovery_speed),
                wp.int32(self.total_rows),
            ],
            outputs=[self.rhs],
            device=device,
        )

    def apply_solution(
        self,
        bodies: BodyContainer,
        inverse_mass: wp.array,
        inverse_inertia_world: wp.array,
        *,
        device=None,
    ) -> None:
        """Apply ``M^-1 J^T solution`` to body velocities."""
        if self.total_rows <= 0:
            return
        wp.launch(
            _apply_articulation_solution_kernel,
            dim=self.total_rows,
            inputs=[
                self.jacobian,
                self.solution,
                self.body1,
                self.body2,
                self.row_to_active_block,
                inverse_mass,
                inverse_inertia_world,
                wp.int32(self.total_rows),
            ],
            outputs=[bodies.velocity, bodies.angular_velocity],
            device=device,
        )


@wp.kernel
def _populate_adbs_articulation_rows_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    active_joint_indices: wp.array[wp.int32],
    active_block_offsets: wp.array[wp.int32],
    active_joint_count: wp.int32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    block = wp.tid()
    if block >= active_joint_count:
        return

    cid = active_joint_indices[block]
    row0 = active_block_offsets[block]
    row_count = active_block_offsets[block + 1] - row0
    _clear_joint_rows(row0, row_count, jacobian, violation)

    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    p1 = bodies.position[b1]
    p2 = bodies.position[b2]

    r1_b1 = wp.quat_rotate(q1, read_vec3(constraints, _OFF_LA1_B1, cid))
    r1_b2 = wp.quat_rotate(q2, read_vec3(constraints, _OFF_LA1_B2, cid))
    r2_b1 = wp.quat_rotate(q1, read_vec3(constraints, _OFF_LA2_B1, cid))
    r2_b2 = wp.quat_rotate(q2, read_vec3(constraints, _OFF_LA2_B2, cid))

    a1_b1 = p1 + r1_b1
    a1_b2 = p2 + r1_b2
    a2_b1 = p1 + r2_b1
    a2_b2 = p2 + r2_b2

    axis_fallback = wp.quat_rotate(q1, read_vec3(constraints, _OFF_AXIS_LOCAL1, cid))
    axis_parent = _safe_normalize(a2_b1 - a1_b1, axis_fallback)
    axis_child = _safe_normalize(a2_b2 - a1_b2, axis_parent)
    tangent0, tangent1 = _orthonormal_pair(axis_parent)
    swing_error = wp.cross(axis_parent, axis_child)

    if mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_CABLE:
        _fill_spherical_rows(row0, a1_b1, a1_b2, r1_b1, r1_b2, jacobian, violation)
    elif mode == JOINT_MODE_REVOLUTE:
        _fill_spherical_rows(row0, a1_b1, a1_b2, r1_b1, r1_b2, jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 3, tangent0, tangent1, swing_error, jacobian, violation)
    elif mode == JOINT_MODE_PRISMATIC:
        r3_b1 = wp.quat_rotate(q1, read_vec3(constraints, _OFF_LA3_B1, cid))
        r3_b2 = wp.quat_rotate(q2, read_vec3(constraints, _OFF_LA3_B2, cid))
        parent_twist = _safe_project_tangent(r3_b1 - r1_b1, axis_parent, tangent0)
        child_twist = _safe_project_tangent(r3_b2 - r1_b2, axis_parent, parent_twist)
        twist_error = wp.dot(wp.cross(parent_twist, child_twist), axis_parent)
        _fill_perpendicular_position_rows(row0, tangent0, tangent1, a1_b2 - a1_b1, r1_b1, r1_b2, jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 2, tangent0, tangent1, swing_error, jacobian, violation)
        _fill_angular_row(row0 + 4, axis_parent, twist_error, jacobian, violation)
    elif mode == JOINT_MODE_FIXED:
        r3_b1 = wp.quat_rotate(q1, read_vec3(constraints, _OFF_LA3_B1, cid))
        r3_b2 = wp.quat_rotate(q2, read_vec3(constraints, _OFF_LA3_B2, cid))
        parent_twist = _safe_project_tangent(r3_b1 - r1_b1, axis_parent, tangent0)
        child_twist = _safe_project_tangent(r3_b2 - r1_b2, axis_parent, parent_twist)
        twist_error = wp.dot(wp.cross(parent_twist, child_twist), axis_parent)
        _fill_spherical_rows(row0, a1_b1, a1_b2, r1_b1, r1_b2, jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 3, tangent0, tangent1, swing_error, jacobian, violation)
        _fill_angular_row(row0 + 5, axis_parent, twist_error, jacobian, violation)
    elif mode == JOINT_MODE_CYLINDRICAL:
        _fill_perpendicular_position_rows(row0, tangent0, tangent1, a1_b2 - a1_b1, r1_b1, r1_b2, jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 2, tangent0, tangent1, swing_error, jacobian, violation)
    elif mode == JOINT_MODE_UNIVERSAL:
        _fill_spherical_rows(row0, a1_b1, a1_b2, r1_b1, r1_b2, jacobian, violation)
        _fill_angular_row(row0 + 3, axis_parent, wp.float32(0.0), jacobian, violation)
    elif mode == JOINT_MODE_PLANAR:
        _fill_spatial_row(row0, axis_parent, r1_b1, r1_b2, wp.dot(a1_b2 - a1_b1, axis_parent), jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 1, tangent0, tangent1, swing_error, jacobian, violation)


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

    matrix[row, col] = _articulation_matrix_entry(
        row,
        col,
        jacobian,
        body1,
        body2,
        row_to_active_block,
        inverse_mass,
        inverse_inertia_world,
    )


@wp.kernel
def _assemble_block_sparse_articulation_diag_kernel(
    jacobian: wp.array2d[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    active_block_offsets: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    diagonal_regularization: wp.float32,
    block_count: wp.int32,
    block_diag: wp.array3d[wp.float32],
):
    pivot = wp.tid()
    if pivot >= block_count:
        return

    active_block = pivot_order[pivot]
    row0 = active_block_offsets[active_block]
    block_size = block_sizes[pivot]
    for i in range(_BLOCK_SIZE):
        row = row0 + wp.int32(i)
        for j in range(_BLOCK_SIZE):
            value = wp.float32(0.0)
            if wp.int32(i) < block_size and wp.int32(j) < block_size:
                col = row0 + wp.int32(j)
                value = _articulation_matrix_entry(
                    row,
                    col,
                    jacobian,
                    body1,
                    body2,
                    row_to_active_block,
                    inverse_mass,
                    inverse_inertia_world,
                )
                if i == j:
                    value += diagonal_regularization
            block_diag[pivot, i, j] = value


@wp.kernel
def _assemble_block_sparse_articulation_off_kernel(
    jacobian: wp.array2d[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    active_block_offsets: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    n_off_row_idx: wp.array[wp.int32],
    n_off_col_idx: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    block_nnz: wp.int32,
    block_off: wp.array3d[wp.float32],
):
    slot = wp.tid()
    if slot >= block_nnz:
        return

    row_pivot = n_off_row_idx[slot]
    col_pivot = n_off_col_idx[slot]
    row_active_block = pivot_order[row_pivot]
    col_active_block = pivot_order[col_pivot]
    row0 = active_block_offsets[row_active_block]
    col0 = active_block_offsets[col_active_block]
    row_block_size = block_sizes[row_pivot]
    col_block_size = block_sizes[col_pivot]
    for i in range(_BLOCK_SIZE):
        row = row0 + wp.int32(i)
        for j in range(_BLOCK_SIZE):
            value = wp.float32(0.0)
            if wp.int32(i) < row_block_size and wp.int32(j) < col_block_size:
                col = col0 + wp.int32(j)
                value = _articulation_matrix_entry(
                    row,
                    col,
                    jacobian,
                    body1,
                    body2,
                    row_to_active_block,
                    inverse_mass,
                    inverse_inertia_world,
                )
            block_off[slot, i, j] = value


@wp.kernel
def _compute_articulation_residual_kernel(
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    bodies: BodyContainer,
    dt: wp.float32,
    alpha: wp.float32,
    recovery_speed: wp.float32,
    total_rows: wp.int32,
    rhs: wp.array[wp.float32],
):
    row = wp.tid()
    if row >= total_rows:
        return

    block = row_to_active_block[row]
    b1 = body1[block]
    b2 = body2[block]

    residual = wp.float32(0.0)
    if b1 >= 0:
        residual += _jacobian_velocity_dot(row, 0, b1, jacobian, bodies)
    if b2 >= 0:
        residual += _jacobian_velocity_dot(row, 6, b2, jacobian, bodies)

    baumgarte = violation[row] / (dt + alpha)
    if recovery_speed > wp.float32(0.0):
        baumgarte = wp.clamp(baumgarte, -recovery_speed, recovery_speed)

    rhs[row] = -(residual + baumgarte)


@wp.kernel
def _apply_articulation_solution_kernel(
    jacobian: wp.array2d[wp.float32],
    solution: wp.array[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    total_rows: wp.int32,
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
):
    row = wp.tid()
    if row >= total_rows:
        return

    block = row_to_active_block[row]
    lam = solution[row]
    b1 = body1[block]
    b2 = body2[block]
    if b1 >= 0:
        _apply_body_delta(row, 0, b1, lam, jacobian, inverse_mass, inverse_inertia_world, velocity, angular_velocity)
    if b2 >= 0:
        _apply_body_delta(row, 6, b2, lam, jacobian, inverse_mass, inverse_inertia_world, velocity, angular_velocity)


@wp.func
def _articulation_matrix_entry(
    row: wp.int32,
    col: wp.int32,
    jacobian: wp.array2d[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
) -> wp.float32:
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
    return value


@wp.func
def _clear_joint_rows(
    row0: wp.int32,
    row_count: wp.int32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    local = wp.int32(0)
    while local < row_count:
        row = row0 + local
        violation[row] = wp.float32(0.0)
        col = wp.int32(0)
        while col < wp.int32(12):
            jacobian[row, col] = wp.float32(0.0)
            col += wp.int32(1)
        local += wp.int32(1)


@wp.func
def _fill_spherical_rows(
    row0: wp.int32,
    parent_anchor_world: wp.vec3f,
    child_anchor_world: wp.vec3f,
    parent_r: wp.vec3f,
    child_r: wp.vec3f,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    error = child_anchor_world - parent_anchor_world
    _fill_spatial_row(row0 + 0, wp.vec3f(1.0, 0.0, 0.0), parent_r, child_r, error[0], jacobian, violation)
    _fill_spatial_row(row0 + 1, wp.vec3f(0.0, 1.0, 0.0), parent_r, child_r, error[1], jacobian, violation)
    _fill_spatial_row(row0 + 2, wp.vec3f(0.0, 0.0, 1.0), parent_r, child_r, error[2], jacobian, violation)


@wp.func
def _fill_perpendicular_position_rows(
    row0: wp.int32,
    tangent0: wp.vec3f,
    tangent1: wp.vec3f,
    anchor_error: wp.vec3f,
    parent_r: wp.vec3f,
    child_r: wp.vec3f,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    _fill_spatial_row(row0 + 0, tangent0, parent_r, child_r, wp.dot(anchor_error, tangent0), jacobian, violation)
    _fill_spatial_row(row0 + 1, tangent1, parent_r, child_r, wp.dot(anchor_error, tangent1), jacobian, violation)


@wp.func
def _fill_perpendicular_rotation_rows(
    row0: wp.int32,
    tangent0: wp.vec3f,
    tangent1: wp.vec3f,
    swing_error: wp.vec3f,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    _fill_angular_row(row0 + 0, tangent0, wp.dot(swing_error, tangent0), jacobian, violation)
    _fill_angular_row(row0 + 1, tangent1, wp.dot(swing_error, tangent1), jacobian, violation)


@wp.func
def _fill_spatial_row(
    row: wp.int32,
    axis: wp.vec3f,
    parent_r: wp.vec3f,
    child_r: wp.vec3f,
    error: wp.float32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    parent_ang = -wp.cross(parent_r, axis)
    child_ang = wp.cross(child_r, axis)

    jacobian[row, 0] = -axis[0]
    jacobian[row, 1] = -axis[1]
    jacobian[row, 2] = -axis[2]
    jacobian[row, 3] = parent_ang[0]
    jacobian[row, 4] = parent_ang[1]
    jacobian[row, 5] = parent_ang[2]
    jacobian[row, 6] = axis[0]
    jacobian[row, 7] = axis[1]
    jacobian[row, 8] = axis[2]
    jacobian[row, 9] = child_ang[0]
    jacobian[row, 10] = child_ang[1]
    jacobian[row, 11] = child_ang[2]
    violation[row] = error


@wp.func
def _fill_angular_row(
    row: wp.int32,
    axis: wp.vec3f,
    error: wp.float32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    jacobian[row, 3] = -axis[0]
    jacobian[row, 4] = -axis[1]
    jacobian[row, 5] = -axis[2]
    jacobian[row, 9] = axis[0]
    jacobian[row, 10] = axis[1]
    jacobian[row, 11] = axis[2]
    violation[row] = error


@wp.func
def _safe_normalize(v: wp.vec3f, fallback: wp.vec3f) -> wp.vec3f:
    length2 = wp.dot(v, v)
    if length2 > wp.float32(1.0e-20):
        return v / wp.sqrt(length2)

    fallback_length2 = wp.dot(fallback, fallback)
    if fallback_length2 > wp.float32(1.0e-20):
        return fallback / wp.sqrt(fallback_length2)

    return wp.vec3f(1.0, 0.0, 0.0)


@wp.func
def _safe_project_tangent(v: wp.vec3f, axis: wp.vec3f, fallback: wp.vec3f) -> wp.vec3f:
    projected = v - wp.dot(v, axis) * axis
    return _safe_normalize(projected, fallback)


@wp.func
def _orthonormal_pair(axis: wp.vec3f):
    seed = wp.vec3f(1.0, 0.0, 0.0)
    if wp.abs(axis[0]) < wp.abs(axis[1]):
        if wp.abs(axis[0]) < wp.abs(axis[2]):
            seed = wp.vec3f(1.0, 0.0, 0.0)
        else:
            seed = wp.vec3f(0.0, 0.0, 1.0)
    elif wp.abs(axis[1]) < wp.abs(axis[2]):
        seed = wp.vec3f(0.0, 1.0, 0.0)
    else:
        seed = wp.vec3f(0.0, 0.0, 1.0)

    tangent0 = _safe_normalize(wp.cross(axis, seed), wp.vec3f(0.0, 1.0, 0.0))
    tangent1 = wp.cross(axis, tangent0)
    return tangent0, tangent1


@wp.func
def _jacobian_velocity_dot(
    row: wp.int32,
    offset: wp.int32,
    body: wp.int32,
    jacobian: wp.array2d[wp.float32],
    bodies: BodyContainer,
) -> wp.float32:
    lin = wp.vec3(
        jacobian[row, offset + 0],
        jacobian[row, offset + 1],
        jacobian[row, offset + 2],
    )
    ang = wp.vec3(
        jacobian[row, offset + 3],
        jacobian[row, offset + 4],
        jacobian[row, offset + 5],
    )
    return wp.dot(lin, bodies.velocity[body]) + wp.dot(ang, bodies.angular_velocity[body])


@wp.func
def _apply_body_delta(
    row: wp.int32,
    offset: wp.int32,
    body: wp.int32,
    lam: wp.float32,
    jacobian: wp.array2d[wp.float32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[wp.mat33f],
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
):
    lin = wp.vec3(
        jacobian[row, offset + 0],
        jacobian[row, offset + 1],
        jacobian[row, offset + 2],
    )
    ang = wp.vec3(
        jacobian[row, offset + 3],
        jacobian[row, offset + 4],
        jacobian[row, offset + 5],
    )
    delta_lin = inverse_mass[body] * lam * lin
    delta_ang = inverse_inertia_world[body] * (lam * ang)
    wp.atomic_add(velocity, body, delta_lin)
    wp.atomic_add(angular_velocity, body, delta_ang)


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
