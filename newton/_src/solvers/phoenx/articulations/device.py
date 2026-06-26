# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp storage and kernels for PhoenX articulation DVI systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, inertia_sym6, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    pd_coefficients,
    read_float,
    read_int,
    read_quat,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    _CLAMP_MAX,
    _CLAMP_MIN,
    _CLAMP_NONE,
    _OFF_AXIS_LOCAL1,
    _OFF_BODY1,
    _OFF_BODY2,
    _OFF_CLAMP,
    _OFF_D6_LIMIT_COUNT,
    _OFF_D6_LIMIT_LOWER,
    _OFF_D6_LIMIT_UPPER,
    _OFF_DAMPING_DRIVE,
    _OFF_DAMPING_LIMIT,
    _OFF_DAMPING_RATIO_LIMIT,
    _OFF_DRIVE_MODE,
    _OFF_HERTZ_LIMIT,
    _OFF_INV_INITIAL_ORIENTATION,
    _OFF_JOINT_MODE,
    _OFF_LA1_B1,
    _OFF_LA1_B2,
    _OFF_LA2_B1,
    _OFF_LA2_B2,
    _OFF_LA3_B1,
    _OFF_LA3_B2,
    _OFF_MAX_FORCE_DRIVE,
    _OFF_MAX_VALUE,
    _OFF_MIN_VALUE,
    _OFF_PREVIOUS_QUATERNION_ANGLE,
    _OFF_REVOLUTION_COUNTER,
    _OFF_STIFFNESS_DRIVE,
    _OFF_STIFFNESS_LIMIT,
    _OFF_TARGET,
    _OFF_TARGET_VELOCITY,
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)
from newton._src.solvers.phoenx.helpers.math_helpers import (
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_BOOST_PRISMATIC_DRIVE,
    PHOENX_BOOST_PRISMATIC_LIMIT,
    PHOENX_BOOST_REVOLUTE_DRIVE,
    PHOENX_BOOST_REVOLUTE_LIMIT,
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
    velocity_target: wp.array
    row_regularization: wp.array
    solution_lower: wp.array
    solution_upper: wp.array
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
    block_rhs: wp.array
    block_solution: wp.array
    block_l_nnz: int
    block_num_levels: int
    block_level_ptr_host: tuple[int, ...]
    block_factor_diag: wp.array
    block_factor_off: wp.array
    block_y: wp.array
    block_l_col_ptr: wp.array
    block_l_row_idx: wp.array
    block_l_row_ptr: wp.array
    block_l_col_idx: wp.array
    block_l_csr_to_csc: wp.array
    block_n_off_to_l: wp.array
    block_pred_diag_ptr: wp.array
    block_pred_diag_slot: wp.array
    block_pred_off_ptr: wp.array
    block_pred_off_slot_ik: wp.array
    block_pred_off_slot_jk: wp.array
    block_level_ptr: wp.array
    block_level_pivots: wp.array

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
        block_l_nnz = int(symbolic.nnz_l)
        block_l_col_ptr_np = symbolic.l_col_ptr.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        block_l_row_idx_np = symbolic.l_row_idx.astype(np.int32) if block_l_nnz > 0 else np.zeros(1, dtype=np.int32)
        block_l_row_ptr_np = symbolic.l_row_ptr.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        block_l_col_idx_np = symbolic.l_col_idx.astype(np.int32) if block_l_nnz > 0 else np.zeros(1, dtype=np.int32)
        block_l_csr_to_csc_np = (
            symbolic.l_csr_to_csc.astype(np.int32) if block_l_nnz > 0 else np.zeros(1, dtype=np.int32)
        )
        block_n_off_to_l_np = symbolic.n_off_to_l.astype(np.int32) if block_nnz > 0 else np.zeros(1, dtype=np.int32)
        block_pred_diag_ptr_np = (
            symbolic.pred_diag_ptr.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        )
        block_pred_diag_slot_np = symbolic.pred_diag_slot.astype(np.int32)
        block_pred_off_ptr_np = (
            symbolic.pred_off_ptr.astype(np.int32) if block_l_nnz > 0 else np.zeros(1, dtype=np.int32)
        )
        block_pred_off_slot_ik_np = symbolic.pred_off_slot_ik.astype(np.int32)
        block_pred_off_slot_jk_np = symbolic.pred_off_slot_jk.astype(np.int32)
        block_level_ptr_np = symbolic.level_ptr.astype(np.int32)
        block_level_pivots_np = (
            symbolic.level_pivots.astype(np.int32) if block_count > 0 else np.zeros(1, dtype=np.int32)
        )
        block_num_levels = int(symbolic.num_levels)

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
            velocity_target=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            row_regularization=wp.zeros(rows_alloc, dtype=wp.float32, device=device),
            solution_lower=wp.full(rows_alloc, -1.0e20, dtype=wp.float32, device=device),
            solution_upper=wp.full(rows_alloc, 1.0e20, dtype=wp.float32, device=device),
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
            block_rhs=wp.zeros((max(block_count, 1), _BLOCK_SIZE), dtype=wp.float32, device=device),
            block_solution=wp.zeros((max(block_count, 1), _BLOCK_SIZE), dtype=wp.float32, device=device),
            block_l_nnz=block_l_nnz,
            block_num_levels=block_num_levels,
            block_level_ptr_host=tuple(int(v) for v in block_level_ptr_np),
            block_factor_diag=wp.zeros(
                (max(block_count, 1), _BLOCK_SIZE, _BLOCK_SIZE), dtype=wp.float32, device=device
            ),
            block_factor_off=wp.zeros((max(block_l_nnz, 1), _BLOCK_SIZE, _BLOCK_SIZE), dtype=wp.float32, device=device),
            block_y=wp.zeros((max(block_count, 1), _BLOCK_SIZE), dtype=wp.float32, device=device),
            block_l_col_ptr=wp.array(block_l_col_ptr_np, dtype=wp.int32, device=device),
            block_l_row_idx=wp.array(block_l_row_idx_np, dtype=wp.int32, device=device),
            block_l_row_ptr=wp.array(block_l_row_ptr_np, dtype=wp.int32, device=device),
            block_l_col_idx=wp.array(block_l_col_idx_np, dtype=wp.int32, device=device),
            block_l_csr_to_csc=wp.array(block_l_csr_to_csc_np, dtype=wp.int32, device=device),
            block_n_off_to_l=wp.array(block_n_off_to_l_np, dtype=wp.int32, device=device),
            block_pred_diag_ptr=wp.array(block_pred_diag_ptr_np, dtype=wp.int32, device=device),
            block_pred_diag_slot=wp.array(block_pred_diag_slot_np, dtype=wp.int32, device=device),
            block_pred_off_ptr=wp.array(block_pred_off_ptr_np, dtype=wp.int32, device=device),
            block_pred_off_slot_ik=wp.array(block_pred_off_slot_ik_np, dtype=wp.int32, device=device),
            block_pred_off_slot_jk=wp.array(block_pred_off_slot_jk_np, dtype=wp.int32, device=device),
            block_level_ptr=wp.array(block_level_ptr_np, dtype=wp.int32, device=device),
            block_level_pivots=wp.array(block_level_pivots_np, dtype=wp.int32, device=device),
        )

    def populate_from_adbs_constraints(
        self,
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        *,
        dt: float = 0.0,
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
                wp.float32(dt),
                wp.int32(self.active_joint_count),
            ],
            outputs=[
                self.jacobian,
                self.violation,
                self.velocity_target,
                self.row_regularization,
                self.solution_lower,
                self.solution_upper,
            ],
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
                self.row_regularization,
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
                self.row_regularization,
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
                self.velocity_target,
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

    def gather_block_rhs(self, *, device=None) -> None:
        """Pack flat RHS rows into padded pivot-order block storage."""
        if self.total_rows <= 0 or self.block_count <= 0:
            return
        wp.launch(
            _gather_articulation_rhs_blocks_kernel,
            dim=self.block_count,
            inputs=[
                self.rhs,
                self.active_block_offsets,
                self.block_pivot_order,
                self.block_sizes,
                wp.int32(self.block_count),
            ],
            outputs=[self.block_rhs],
            device=device,
        )

    def scatter_block_solution(self, *, device=None) -> None:
        """Unpack padded pivot-order block solution into flat rows."""
        if self.total_rows <= 0 or self.block_count <= 0:
            return
        wp.launch(
            _scatter_articulation_solution_blocks_kernel,
            dim=self.block_count,
            inputs=[
                self.block_solution,
                self.active_block_offsets,
                self.block_pivot_order,
                self.block_sizes,
                wp.int32(self.block_count),
            ],
            outputs=[self.solution],
            device=device,
        )

    def factor_block_sparse_matrix(self, *, device=None) -> None:
        """Factor the assembled block-sparse matrix in place."""
        if self.block_count <= 0:
            return
        if self.block_l_nnz > 0:
            wp.launch(
                _zero_articulation_factor_off_kernel,
                dim=self.block_l_nnz,
                inputs=[wp.int32(self.block_l_nnz)],
                outputs=[self.block_factor_off],
                device=device,
            )
        if self.block_nnz > 0:
            wp.launch(
                _copy_articulation_n_off_to_factor_kernel,
                dim=self.block_nnz,
                inputs=[self.block_off, self.block_n_off_to_l, wp.int32(self.block_nnz)],
                outputs=[self.block_factor_off],
                device=device,
            )
        for level in range(self.block_num_levels - 1, -1, -1):
            level_offset = self.block_level_ptr_host[level]
            level_count = self.block_level_ptr_host[level + 1] - level_offset
            if level_count <= 0:
                continue
            wp.launch(
                _factor_articulation_cholesky_level_kernel,
                dim=level_count,
                inputs=[
                    self.block_l_col_ptr,
                    self.block_l_row_idx,
                    self.block_pred_diag_ptr,
                    self.block_pred_diag_slot,
                    self.block_pred_off_ptr,
                    self.block_pred_off_slot_ik,
                    self.block_pred_off_slot_jk,
                    self.block_level_pivots,
                    self.block_sizes,
                    self.block_diag,
                    wp.int32(level_count),
                    wp.int32(level_offset),
                ],
                outputs=[self.block_factor_diag, self.block_factor_off],
                device=device,
            )

    def solve_block_sparse_factors(self, *, device=None) -> None:
        """Solve using the most recent device block factorization."""
        if self.block_count <= 0:
            return
        for level in range(self.block_num_levels - 1, -1, -1):
            level_offset = self.block_level_ptr_host[level]
            level_count = self.block_level_ptr_host[level + 1] - level_offset
            if level_count <= 0:
                continue
            wp.launch(
                _forward_substitute_articulation_level_kernel,
                dim=level_count,
                inputs=[
                    self.block_l_row_ptr,
                    self.block_l_col_idx,
                    self.block_l_csr_to_csc,
                    self.block_level_pivots,
                    self.block_sizes,
                    self.block_factor_off,
                    self.block_factor_diag,
                    self.block_rhs,
                    wp.int32(level_count),
                    wp.int32(level_offset),
                ],
                outputs=[self.block_y],
                device=device,
            )
        for level in range(self.block_num_levels):
            level_offset = self.block_level_ptr_host[level]
            level_count = self.block_level_ptr_host[level + 1] - level_offset
            if level_count <= 0:
                continue
            wp.launch(
                _backward_substitute_articulation_level_kernel,
                dim=level_count,
                inputs=[
                    self.block_l_col_ptr,
                    self.block_l_row_idx,
                    self.block_level_pivots,
                    self.block_sizes,
                    self.block_factor_off,
                    self.block_factor_diag,
                    self.block_y,
                    wp.int32(level_count),
                    wp.int32(level_offset),
                ],
                outputs=[self.block_solution],
                device=device,
            )

    def solve_block_sparse_matrix(self, *, device=None) -> None:
        """Factor the current block matrix and solve the current RHS."""
        self.factor_block_sparse_matrix(device=device)
        self.gather_block_rhs(device=device)
        self.solve_block_sparse_factors(device=device)
        self.scatter_block_solution(device=device)

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
                self.solution_lower,
                self.solution_upper,
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
    dt: wp.float32,
    active_joint_count: wp.int32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
    velocity_target: wp.array[wp.float32],
    row_regularization: wp.array[wp.float32],
    solution_lower: wp.array[wp.float32],
    solution_upper: wp.array[wp.float32],
):
    block = wp.tid()
    if block >= active_joint_count:
        return

    cid = active_joint_indices[block]
    row0 = active_block_offsets[block]
    row_count = active_block_offsets[block + 1] - row0
    _clear_joint_rows(
        row0, row_count, jacobian, violation, velocity_target, row_regularization, solution_lower, solution_upper
    )

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
        axis_lock = _safe_normalize(wp.quat_rotate(q1, read_vec3(constraints, _OFF_AXIS_LOCAL1, cid)), axis_parent)
        _fill_spherical_rows(row0, a1_b1, a1_b2, r1_b1, r1_b2, jacobian, violation)
        _fill_angular_row(row0 + 3, axis_lock, wp.float32(0.0), jacobian, violation)
    elif mode == JOINT_MODE_PLANAR:
        _fill_spatial_row(row0, axis_parent, r1_b1, r1_b2, wp.dot(a1_b2 - a1_b1, axis_parent), jacobian, violation)
        _fill_perpendicular_rotation_rows(row0 + 1, tangent0, tangent1, swing_error, jacobian, violation)

    equality_rows = _adbs_equality_row_count(mode)
    axial_rows = wp.int32(0)
    if (mode == JOINT_MODE_REVOLUTE or mode == JOINT_MODE_PRISMATIC) and row_count > equality_rows:
        axial_rows = wp.int32(1)
        axial_row = row0 + equality_rows
        drive_mode = read_int(constraints, _OFF_DRIVE_MODE, cid)
        stiffness_drive = read_float(constraints, _OFF_STIFFNESS_DRIVE, cid)
        damping_drive = read_float(constraints, _OFF_DAMPING_DRIVE, cid)
        target = read_float(constraints, _OFF_TARGET, cid)
        target_velocity = read_float(constraints, _OFF_TARGET_VELOCITY, cid)
        max_force_drive = read_float(constraints, _OFF_MAX_FORCE_DRIVE, cid)
        min_value = read_float(constraints, _OFF_MIN_VALUE, cid)
        max_value = read_float(constraints, _OFF_MAX_VALUE, cid)
        hertz_limit = read_float(constraints, _OFF_HERTZ_LIMIT, cid)
        damping_ratio_limit = read_float(constraints, _OFF_DAMPING_RATIO_LIMIT, cid)
        stiffness_limit = read_float(constraints, _OFF_STIFFNESS_LIMIT, cid)
        damping_limit = read_float(constraints, _OFF_DAMPING_LIMIT, cid)
        drive_active = drive_mode != DRIVE_MODE_OFF and (stiffness_drive > 0.0 or damping_drive > 0.0)

        if mode == JOINT_MODE_REVOLUTE:
            axis_drive = _safe_normalize(wp.quat_rotate(q1, read_vec3(constraints, _OFF_AXIS_LOCAL1, cid)), axis_parent)
            inv_init = read_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid)
            diff = q2 * inv_init * wp.quat_inverse(q1)
            new_q_angle = extract_rotation_angle(diff, axis_drive)
            old_counter = read_int(constraints, _OFF_REVOLUTION_COUNTER, cid)
            old_prev = read_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
            new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
            write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, new_counter)
            write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
            value = revolution_tracker_angle(new_counter, new_prev)
            axial_error, active, clamp = _axial_row_error(value, target, min_value, max_value, drive_mode, drive_active)
            write_int(constraints, _OFF_CLAMP, cid, clamp)
            if active:
                _fill_angular_row(axial_row, axis_drive, axial_error, jacobian, violation)
                if clamp == _CLAMP_NONE:
                    _apply_axial_drive_row_softness(
                        axial_row,
                        mode,
                        value,
                        target,
                        target_velocity,
                        drive_mode,
                        stiffness_drive,
                        damping_drive,
                        max_force_drive,
                        dt,
                        b1,
                        b2,
                        jacobian,
                        bodies,
                        violation,
                        velocity_target,
                        row_regularization,
                        solution_lower,
                        solution_upper,
                    )
                else:
                    _apply_axial_limit_row_softness(
                        axial_row,
                        mode,
                        axial_error,
                        clamp,
                        hertz_limit,
                        damping_ratio_limit,
                        stiffness_limit,
                        damping_limit,
                        dt,
                        b1,
                        b2,
                        jacobian,
                        bodies,
                        violation,
                        velocity_target,
                        row_regularization,
                        solution_lower,
                        solution_upper,
                    )
        elif mode == JOINT_MODE_PRISMATIC:
            axis_drive = _safe_normalize(wp.quat_rotate(q1, read_vec3(constraints, _OFF_AXIS_LOCAL1, cid)), axis_parent)
            value = wp.dot(axis_drive, a1_b2 - a1_b1)
            axial_error, active, clamp = _axial_row_error(value, target, min_value, max_value, drive_mode, drive_active)
            write_int(constraints, _OFF_CLAMP, cid, clamp)
            if active:
                _fill_spatial_row(axial_row, axis_drive, r1_b1, r1_b2, axial_error, jacobian, violation)
                if clamp == _CLAMP_NONE:
                    _apply_axial_drive_row_softness(
                        axial_row,
                        mode,
                        value,
                        target,
                        target_velocity,
                        drive_mode,
                        stiffness_drive,
                        damping_drive,
                        max_force_drive,
                        dt,
                        b1,
                        b2,
                        jacobian,
                        bodies,
                        violation,
                        velocity_target,
                        row_regularization,
                        solution_lower,
                        solution_upper,
                    )
                else:
                    _apply_axial_limit_row_softness(
                        axial_row,
                        mode,
                        axial_error,
                        clamp,
                        hertz_limit,
                        damping_ratio_limit,
                        stiffness_limit,
                        damping_limit,
                        dt,
                        b1,
                        b2,
                        jacobian,
                        bodies,
                        violation,
                        velocity_target,
                        row_regularization,
                        solution_lower,
                        solution_upper,
                    )

    d6_row0 = row0 + equality_rows + axial_rows
    d6_row_count = row_count - equality_rows - axial_rows
    if d6_row_count > wp.int32(0) and (mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL):
        _fill_d6_angular_limit_rows(d6_row0, d6_row_count, mode, q1, q2, constraints, cid, jacobian, violation)


@wp.kernel
def _assemble_dense_articulation_matrix_kernel(
    jacobian: wp.array2d[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[inertia_sym6],
    row_regularization: wp.array[wp.float32],
    total_rows: wp.int32,
    matrix: wp.array2d[wp.float32],
):
    row, col = wp.tid()
    if row >= total_rows or col >= total_rows:
        return

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
    if row == col:
        value += row_regularization[row]
    matrix[row, col] = value


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
    inverse_inertia_world: wp.array[inertia_sym6],
    row_regularization: wp.array[wp.float32],
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
                    value += row_regularization[row]
                    value += diagonal_regularization * wp.max(wp.abs(value), wp.float32(1.0))
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
    inverse_inertia_world: wp.array[inertia_sym6],
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
    velocity_target: wp.array[wp.float32],
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

    rhs[row] = -(residual - velocity_target[row] + baumgarte)


@wp.kernel
def _gather_articulation_rhs_blocks_kernel(
    rhs: wp.array[wp.float32],
    active_block_offsets: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    block_count: wp.int32,
    block_rhs: wp.array2d[wp.float32],
):
    pivot = wp.tid()
    if pivot >= block_count:
        return

    active_block = pivot_order[pivot]
    row0 = active_block_offsets[active_block]
    block_size = block_sizes[pivot]
    for i in range(_BLOCK_SIZE):
        value = wp.float32(0.0)
        if wp.int32(i) < block_size:
            value = rhs[row0 + wp.int32(i)]
        block_rhs[pivot, i] = value


@wp.kernel
def _scatter_articulation_solution_blocks_kernel(
    block_solution: wp.array2d[wp.float32],
    active_block_offsets: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    block_count: wp.int32,
    solution: wp.array[wp.float32],
):
    pivot = wp.tid()
    if pivot >= block_count:
        return

    active_block = pivot_order[pivot]
    row0 = active_block_offsets[active_block]
    block_size = block_sizes[pivot]
    for i in range(_BLOCK_SIZE):
        if wp.int32(i) < block_size:
            solution[row0 + wp.int32(i)] = block_solution[pivot, i]


@wp.kernel
def _zero_articulation_factor_off_kernel(
    block_l_nnz: wp.int32,
    factor_off: wp.array3d[wp.float32],
):
    slot = wp.tid()
    if slot >= block_l_nnz:
        return
    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            factor_off[slot, i, j] = wp.float32(0.0)


@wp.kernel
def _copy_articulation_n_off_to_factor_kernel(
    block_off: wp.array3d[wp.float32],
    n_off_to_l: wp.array[wp.int32],
    block_nnz: wp.int32,
    factor_off: wp.array3d[wp.float32],
):
    slot = wp.tid()
    if slot >= block_nnz:
        return

    dst = n_off_to_l[slot]
    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            factor_off[dst, i, j] = block_off[slot, i, j]


@wp.kernel
def _factor_articulation_cholesky_level_kernel(
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    pred_diag_ptr: wp.array[wp.int32],
    pred_diag_slot: wp.array[wp.int32],
    pred_off_ptr: wp.array[wp.int32],
    pred_off_slot_ik: wp.array[wp.int32],
    pred_off_slot_jk: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    block_diag: wp.array3d[wp.float32],
    level_count: wp.int32,
    level_offset: wp.int32,
    factor_diag: wp.array3d[wp.float32],
    factor_off: wp.array3d[wp.float32],
):
    local = wp.tid()
    if local >= level_count:
        return

    pivot = level_pivots[level_offset + local]
    pivot_size = block_sizes[pivot]

    pred_diag_start = pred_diag_ptr[pivot]
    pred_diag_end = pred_diag_ptr[pivot + wp.int32(1)]
    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            value = wp.float32(0.0)
            if wp.int32(i) < pivot_size and wp.int32(j) < pivot_size:
                value = block_diag[pivot, i, j]
                for pred in range(pred_diag_start, pred_diag_end):
                    pred_slot = pred_diag_slot[pred]
                    for k in range(_BLOCK_SIZE):
                        value -= factor_off[pred_slot, i, k] * factor_off[pred_slot, j, k]
            elif i == j:
                value = wp.float32(1.0)
            factor_diag[pivot, i, j] = value

    for k in range(_BLOCK_SIZE):
        if wp.int32(k) < pivot_size:
            diag = factor_diag[pivot, k, k]
            if diag < wp.float32(1.0e-20):
                diag = wp.float32(1.0e-20)
            diag = wp.sqrt(diag)
            factor_diag[pivot, k, k] = diag
            inv_diag = wp.float32(1.0) / diag

            for i in range(_BLOCK_SIZE):
                if wp.int32(i) > wp.int32(k) and wp.int32(i) < pivot_size:
                    factor_diag[pivot, i, k] = factor_diag[pivot, i, k] * inv_diag

            for j in range(_BLOCK_SIZE):
                if wp.int32(j) > wp.int32(k) and wp.int32(j) < pivot_size:
                    l_jk = factor_diag[pivot, j, k]
                    for i in range(_BLOCK_SIZE):
                        if wp.int32(i) >= wp.int32(j) and wp.int32(i) < pivot_size:
                            factor_diag[pivot, i, j] -= factor_diag[pivot, i, k] * l_jk

    for i in range(_BLOCK_SIZE):
        for j in range(_BLOCK_SIZE):
            if wp.int32(i) < pivot_size and wp.int32(j) < pivot_size:
                if wp.int32(i) < wp.int32(j):
                    factor_diag[pivot, i, j] = wp.float32(0.0)
            elif i == j:
                factor_diag[pivot, i, j] = wp.float32(1.0)
            else:
                factor_diag[pivot, i, j] = wp.float32(0.0)

    col_start = l_col_ptr[pivot]
    col_end = l_col_ptr[pivot + wp.int32(1)]
    for ptr in range(col_start, col_end):
        row_pivot = l_row_idx[ptr]
        row_size = block_sizes[row_pivot]
        pred_off_start = pred_off_ptr[ptr]
        pred_off_end = pred_off_ptr[ptr + wp.int32(1)]

        for i in range(_BLOCK_SIZE):
            for j in range(_BLOCK_SIZE):
                value = wp.float32(0.0)
                if wp.int32(i) < row_size and wp.int32(j) < pivot_size:
                    value = factor_off[ptr, i, j]
                    for pred in range(pred_off_start, pred_off_end):
                        slot_ik = pred_off_slot_ik[pred]
                        slot_jk = pred_off_slot_jk[pred]
                        for k in range(_BLOCK_SIZE):
                            value -= factor_off[slot_ik, i, k] * factor_off[slot_jk, j, k]
                factor_off[ptr, i, j] = value

        for i in range(_BLOCK_SIZE):
            if wp.int32(i) < row_size:
                for j in range(_BLOCK_SIZE):
                    if wp.int32(j) < pivot_size:
                        value = factor_off[ptr, i, j]
                        for k in range(_BLOCK_SIZE):
                            if wp.int32(k) < wp.int32(j):
                                value -= factor_diag[pivot, j, k] * factor_off[ptr, i, k]
                        factor_off[ptr, i, j] = value / factor_diag[pivot, j, j]
                    else:
                        factor_off[ptr, i, j] = wp.float32(0.0)
            else:
                for j in range(_BLOCK_SIZE):
                    factor_off[ptr, i, j] = wp.float32(0.0)


@wp.kernel
def _forward_substitute_articulation_level_kernel(
    l_row_ptr: wp.array[wp.int32],
    l_col_idx: wp.array[wp.int32],
    l_csr_to_csc: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    factor_off: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    block_rhs: wp.array2d[wp.float32],
    level_count: wp.int32,
    level_offset: wp.int32,
    block_y: wp.array2d[wp.float32],
):
    local = wp.tid()
    if local >= level_count:
        return

    pivot = level_pivots[level_offset + local]
    pivot_size = block_sizes[pivot]
    row_start = l_row_ptr[pivot]
    row_end = l_row_ptr[pivot + wp.int32(1)]

    for r in range(_BLOCK_SIZE):
        if wp.int32(r) < pivot_size:
            value = block_rhs[pivot, r]
            for row_ptr in range(row_start, row_end):
                col = l_col_idx[row_ptr]
                slot = l_csr_to_csc[row_ptr]
                col_size = block_sizes[col]
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < col_size:
                        value -= factor_off[slot, r, c] * block_y[col, c]
            for c in range(_BLOCK_SIZE):
                if wp.int32(c) < wp.int32(r):
                    value -= factor_diag[pivot, r, c] * block_y[pivot, c]
            block_y[pivot, r] = value / factor_diag[pivot, r, r]
        else:
            block_y[pivot, r] = wp.float32(0.0)


@wp.kernel
def _backward_substitute_articulation_level_kernel(
    l_col_ptr: wp.array[wp.int32],
    l_row_idx: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    factor_off: wp.array3d[wp.float32],
    factor_diag: wp.array3d[wp.float32],
    block_y: wp.array2d[wp.float32],
    level_count: wp.int32,
    level_offset: wp.int32,
    block_solution: wp.array2d[wp.float32],
):
    local = wp.tid()
    if local >= level_count:
        return

    pivot = level_pivots[level_offset + local]
    pivot_size = block_sizes[pivot]
    col_start = l_col_ptr[pivot]
    col_end = l_col_ptr[pivot + wp.int32(1)]

    for rev in range(_BLOCK_SIZE):
        r = _BLOCK_SIZE - 1 - rev
        if wp.int32(r) < pivot_size:
            value = block_y[pivot, r]
            for ptr in range(col_start, col_end):
                row_pivot = l_row_idx[ptr]
                row_size = block_sizes[row_pivot]
                for c in range(_BLOCK_SIZE):
                    if wp.int32(c) < row_size:
                        value -= factor_off[ptr, c, r] * block_solution[row_pivot, c]
            for c in range(_BLOCK_SIZE):
                if wp.int32(c) > wp.int32(r) and wp.int32(c) < pivot_size:
                    value -= factor_diag[pivot, c, r] * block_solution[pivot, c]
            block_solution[pivot, r] = value / factor_diag[pivot, r, r]
        else:
            block_solution[pivot, r] = wp.float32(0.0)


@wp.kernel
def _apply_articulation_solution_kernel(
    jacobian: wp.array2d[wp.float32],
    solution: wp.array[wp.float32],
    solution_lower: wp.array[wp.float32],
    solution_upper: wp.array[wp.float32],
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    row_to_active_block: wp.array[wp.int32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[inertia_sym6],
    total_rows: wp.int32,
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
):
    row = wp.tid()
    if row >= total_rows:
        return

    block = row_to_active_block[row]
    lam = wp.clamp(solution[row], solution_lower[row], solution_upper[row])
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
    inverse_inertia_world: wp.array[inertia_sym6],
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
def _d6_limit_axis_local(
    constraints: ConstraintContainer,
    cid: wp.int32,
    mode: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    if mode == JOINT_MODE_BALL_SOCKET:
        if slot == wp.int32(0):
            return read_vec3(constraints, _OFF_AXIS_LOCAL1, cid)
        if slot == wp.int32(1):
            return read_vec3(constraints, _OFF_LA2_B1, cid)
        return read_vec3(constraints, _OFF_LA2_B2, cid)
    if slot == wp.int32(0):
        return read_vec3(constraints, _OFF_LA2_B1, cid)
    if slot == wp.int32(1):
        return read_vec3(constraints, _OFF_LA2_B2, cid)
    return wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def _fill_d6_angular_limit_row(
    row: wp.int32,
    axis_local: wp.vec3f,
    lower: wp.float32,
    upper: wp.float32,
    diff: wp.quatf,
    q1: wp.quatf,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    if lower > upper:
        return

    axis = _safe_normalize(wp.quat_rotate(q1, axis_local), wp.vec3f(1.0, 0.0, 0.0))
    angle = extract_rotation_angle(diff, axis)
    error = wp.float32(0.0)
    active = False
    if angle > upper:
        error = angle - upper
        active = True
    elif angle < lower:
        error = angle - lower
        active = True

    if active:
        _fill_angular_row(row, axis, error, jacobian, violation)


@wp.func
def _fill_d6_angular_limit_rows(
    row0: wp.int32,
    row_count: wp.int32,
    mode: wp.int32,
    q1: wp.quatf,
    q2: wp.quatf,
    constraints: ConstraintContainer,
    cid: wp.int32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
):
    count = read_int(constraints, _OFF_D6_LIMIT_COUNT, cid)
    lower = read_vec3(constraints, _OFF_D6_LIMIT_LOWER, cid)
    upper = read_vec3(constraints, _OFF_D6_LIMIT_UPPER, cid)
    inv_init = read_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid)
    diff = q2 * inv_init * wp.quat_inverse(q1)

    if row_count > wp.int32(0) and count > wp.int32(0):
        _fill_d6_angular_limit_row(
            row0,
            _d6_limit_axis_local(constraints, cid, mode, wp.int32(0)),
            lower[0],
            upper[0],
            diff,
            q1,
            jacobian,
            violation,
        )
    if row_count > wp.int32(1) and count > wp.int32(1):
        _fill_d6_angular_limit_row(
            row0 + wp.int32(1),
            _d6_limit_axis_local(constraints, cid, mode, wp.int32(1)),
            lower[1],
            upper[1],
            diff,
            q1,
            jacobian,
            violation,
        )
    if row_count > wp.int32(2) and count > wp.int32(2):
        _fill_d6_angular_limit_row(
            row0 + wp.int32(2),
            _d6_limit_axis_local(constraints, cid, mode, wp.int32(2)),
            lower[2],
            upper[2],
            diff,
            q1,
            jacobian,
            violation,
        )


@wp.func
def _axial_row_error(
    value: wp.float32,
    target: wp.float32,
    min_value: wp.float32,
    max_value: wp.float32,
    drive_mode: wp.int32,
    drive_active: wp.bool,
):
    if min_value <= max_value:
        if value > max_value:
            return value - max_value, True, _CLAMP_MAX
        if value < min_value:
            return value - min_value, True, _CLAMP_MIN

    if drive_active:
        drive_error = wp.float32(0.0)
        if drive_mode == DRIVE_MODE_POSITION:
            drive_error = value - target
        return drive_error, True, _CLAMP_NONE

    return wp.float32(0.0), False, _CLAMP_NONE


@wp.func
def _apply_axial_drive_row_softness(
    row: wp.int32,
    mode: wp.int32,
    value: wp.float32,
    target: wp.float32,
    target_velocity: wp.float32,
    drive_mode: wp.int32,
    stiffness_drive: wp.float32,
    damping_drive: wp.float32,
    max_force_drive: wp.float32,
    dt: wp.float32,
    b1: wp.int32,
    b2: wp.int32,
    jacobian: wp.array2d[wp.float32],
    bodies: BodyContainer,
    violation: wp.array[wp.float32],
    velocity_target: wp.array[wp.float32],
    row_regularization: wp.array[wp.float32],
    solution_lower: wp.array[wp.float32],
    solution_upper: wp.array[wp.float32],
):
    if max_force_drive > wp.float32(0.0) and dt > wp.float32(0.0):
        max_impulse = max_force_drive * dt
        solution_lower[row] = -max_impulse
        solution_upper[row] = max_impulse

    if drive_mode == DRIVE_MODE_OFF:
        return
    if stiffness_drive <= wp.float32(0.0) and damping_drive <= wp.float32(0.0):
        return

    if dt <= wp.float32(0.0):
        velocity_target[row] = target_velocity
        return

    drive_error = wp.float32(0.0)
    if drive_mode == DRIVE_MODE_POSITION:
        drive_error = value - target

    eff_inv = _joint_row_effective_inverse(row, b1, b2, jacobian, bodies.inverse_mass, bodies.inverse_inertia_world)
    boost = PHOENX_BOOST_REVOLUTE_DRIVE
    if mode == JOINT_MODE_PRISMATIC:
        boost = PHOENX_BOOST_PRISMATIC_DRIVE
    gamma, bias, _eff_mass_soft = pd_coefficients(stiffness_drive, damping_drive, drive_error, eff_inv, dt, boost)
    row_regularization[row] = gamma
    velocity_target[row] = -(bias - target_velocity)
    violation[row] = wp.float32(0.0)


@wp.func
def _apply_axial_limit_row_softness(
    row: wp.int32,
    mode: wp.int32,
    limit_error: wp.float32,
    clamp: wp.int32,
    hertz_limit: wp.float32,
    damping_ratio_limit: wp.float32,
    stiffness_limit: wp.float32,
    damping_limit: wp.float32,
    dt: wp.float32,
    b1: wp.int32,
    b2: wp.int32,
    jacobian: wp.array2d[wp.float32],
    bodies: BodyContainer,
    violation: wp.array[wp.float32],
    velocity_target: wp.array[wp.float32],
    row_regularization: wp.array[wp.float32],
    solution_lower: wp.array[wp.float32],
    solution_upper: wp.array[wp.float32],
):
    if clamp == _CLAMP_MAX:
        solution_upper[row] = wp.float32(0.0)
    elif clamp == _CLAMP_MIN:
        solution_lower[row] = wp.float32(0.0)

    if dt <= wp.float32(0.0):
        return

    if stiffness_limit > wp.float32(0.0) or damping_limit > wp.float32(0.0):
        eff_inv = _joint_row_effective_inverse(row, b1, b2, jacobian, bodies.inverse_mass, bodies.inverse_inertia_world)
        boost = PHOENX_BOOST_REVOLUTE_LIMIT
        if mode == JOINT_MODE_PRISMATIC:
            boost = PHOENX_BOOST_PRISMATIC_LIMIT
        gamma, bias, _eff_mass_soft = pd_coefficients(stiffness_limit, damping_limit, limit_error, eff_inv, dt, boost)
        row_regularization[row] = gamma
        velocity_target[row] = -bias
    else:
        bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(hertz_limit, damping_ratio_limit, dt)
        velocity_target[row] = -limit_error * bias_rate

    violation[row] = wp.float32(0.0)


@wp.func
def _joint_row_effective_inverse(
    row: wp.int32,
    b1: wp.int32,
    b2: wp.int32,
    jacobian: wp.array2d[wp.float32],
    inverse_mass: wp.array[wp.float32],
    inverse_inertia_world: wp.array[inertia_sym6],
) -> wp.float32:
    value = wp.float32(0.0)
    if b1 >= wp.int32(0):
        value += _body_metric_dot(row, row, wp.int32(0), wp.int32(0), b1, jacobian, inverse_mass, inverse_inertia_world)
    if b2 >= wp.int32(0):
        value += _body_metric_dot(row, row, wp.int32(6), wp.int32(6), b2, jacobian, inverse_mass, inverse_inertia_world)
    if b1 >= wp.int32(0) and b1 == b2:
        value += _body_metric_dot(row, row, wp.int32(0), wp.int32(6), b1, jacobian, inverse_mass, inverse_inertia_world)
        value += _body_metric_dot(row, row, wp.int32(6), wp.int32(0), b1, jacobian, inverse_mass, inverse_inertia_world)
    return value


@wp.func
def _adbs_equality_row_count(mode: wp.int32) -> wp.int32:
    if mode == JOINT_MODE_REVOLUTE:
        return wp.int32(5)
    if mode == JOINT_MODE_PRISMATIC:
        return wp.int32(5)
    if mode == JOINT_MODE_BALL_SOCKET:
        return wp.int32(3)
    if mode == JOINT_MODE_FIXED:
        return wp.int32(6)
    if mode == JOINT_MODE_CABLE:
        return wp.int32(3)
    if mode == JOINT_MODE_UNIVERSAL:
        return wp.int32(4)
    if mode == JOINT_MODE_CYLINDRICAL:
        return wp.int32(4)
    if mode == JOINT_MODE_PLANAR:
        return wp.int32(3)
    return wp.int32(0)


@wp.func
def _clear_joint_rows(
    row0: wp.int32,
    row_count: wp.int32,
    jacobian: wp.array2d[wp.float32],
    violation: wp.array[wp.float32],
    velocity_target: wp.array[wp.float32],
    row_regularization: wp.array[wp.float32],
    solution_lower: wp.array[wp.float32],
    solution_upper: wp.array[wp.float32],
):
    local = wp.int32(0)
    while local < row_count:
        row = row0 + local
        violation[row] = wp.float32(0.0)
        velocity_target[row] = wp.float32(0.0)
        row_regularization[row] = wp.float32(0.0)
        solution_lower[row] = wp.float32(-1.0e20)
        solution_upper[row] = wp.float32(1.0e20)
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
    inverse_inertia_world: wp.array[inertia_sym6],
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
    delta_ang = mat33_from_sym6(inverse_inertia_world[body]) * (lam * ang)
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
    inverse_inertia_world: wp.array[inertia_sym6],
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
    value += wp.dot(row_ang, mat33_from_sym6(inverse_inertia_world[body]) * col_ang)
    return value
