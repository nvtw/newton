# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse DVI solve path for Kamino dual systems."""

from __future__ import annotations

import warp as wp

from ...dynamics.dual import DualProblem
from . import sparse_kernels
from .kernels import (
    _initialize_dvi_status,
    _scatter_bilateral_solution,
    _set_dvi_bilateral_active_dim,
    _set_dvi_direct_status_iterations,
)
from .sparse_kernels import (
    _build_sparse_bilateral_block,
    _build_sparse_bilateral_rhs,
    _compute_dvi_sparse_solution_vectors,
    _set_dvi_sparse_status_iterations,
    _set_sparse_bilateral_diagonal,
    _solve_dvi_sparse_contacts_offset_update,
    _solve_dvi_sparse_jacobi_update,
    _solve_dvi_sparse_limits_offset_update,
    _solve_dvi_sparse_unilateral_jacobi_update,
    _solve_dvi_sparse_unilateral_offset_update,
    _sparse_delassus_gemv_rows,
    _zero_bilateral_lambdas,
)

wp.set_module_options({"enable_backward": False})

int32 = wp.int32


_SPARSE_DELASSUS_ROWS_JOINTS = 0
_SPARSE_DELASSUS_ROWS_UNILATERAL = 1


def solve_sparse(solver, problem: DualProblem) -> None:
    """Solve a sparse Kamino DVI problem without materializing dense Delassus."""
    if solver._has_contact_block_preconditioner and solver._size.max_of_max_contacts > 0:
        _compute_sparse_contact_block_inverse(solver, problem)

    if solver._bilateral_solver is not None and solver._data.bilateral_operator is not None:
        _solve_sparse_with_bilateral_direct_block(solver, problem)
    else:
        _solve_sparse_jacobi(solver, problem)


def _solve_sparse_jacobi(solver, problem: DualProblem) -> None:
    state = solver._data.state
    problem.delassus.diagonal(state.scratch)

    for iteration in range(solver._max_iterations):
        problem.delassus.matvec(
            x=solver._data.solution.lambdas,
            y=state.v_aug,
            world_mask=state.world_mask,
        )
        wp.launch(
            kernel=_solve_dvi_sparse_jacobi_update,
            dim=(solver._size.num_worlds, solver._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.njc,
                problem.data.nl,
                problem.data.nc,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.cio,
                problem.data.mu,
                state.scratch,
                problem.data.P,
                problem.data.v_f,
                state.v_aug,
                state.contact_block_inv,
                iteration,
                solver._data.config,
                solver._data.solution.lambdas,
            ],
            device=solver.device,
        )

    problem.delassus.matvec(
        x=solver._data.solution.lambdas,
        y=state.v_aug,
        world_mask=state.world_mask,
    )
    wp.launch(
        kernel=_compute_dvi_sparse_solution_vectors,
        dim=(solver._size.num_worlds, solver._size.max_of_max_total_cts),
        inputs=[
            problem.data.dim,
            problem.data.vio,
            problem.data.v_f,
            state.s,
            state.v_aug,
            solver._data.solution.v_plus,
        ],
        device=solver.device,
    )
    wp.launch(
        kernel=_set_dvi_sparse_status_iterations,
        dim=solver._size.num_worlds,
        inputs=[
            problem.data.dim,
            solver._data.config,
            solver._data.status,
        ],
        device=solver.device,
    )


def _compute_sparse_solution_vectors(solver, problem: DualProblem) -> None:
    state = solver._data.state
    problem.delassus.matvec(
        x=solver._data.solution.lambdas,
        y=state.v_aug,
        world_mask=state.world_mask,
    )
    wp.launch(
        kernel=_compute_dvi_sparse_solution_vectors,
        dim=(solver._size.num_worlds, solver._size.max_of_max_total_cts),
        inputs=[
            problem.data.dim,
            problem.data.vio,
            problem.data.v_f,
            state.s,
            state.v_aug,
            solver._data.solution.v_plus,
        ],
        device=solver.device,
    )


def _sparse_delassus_regularization(problem: DualProblem) -> wp.array[wp.float32] | None:
    combined_regularization = getattr(problem.delassus, "_combined_regularization", None)
    if combined_regularization is not None:
        return combined_regularization
    return getattr(problem.delassus, "_eta", None)


def _sparse_delassus_matvec_rows(solver, problem: DualProblem, row_kind: int) -> None:
    delassus = problem.delassus
    state = solver._data.state
    regularization = _sparse_delassus_regularization(problem)
    transpose_matrix = getattr(delassus, "_transpose_op_matrix", None)
    body_space = getattr(delassus, "_vec_temp_body_space", None)
    bsm = getattr(delassus, "bsm", None)

    if (
        regularization is None
        or transpose_matrix is None
        or body_space is None
        or bsm is None
        or getattr(delassus, "ATy_op", None) is None
    ):
        delassus.matvec(
            x=solver._data.solution.lambdas,
            y=state.v_aug,
            world_mask=state.world_mask,
        )
        return

    if getattr(delassus, "_needs_update", False):
        delassus.update()

    delassus.ATy_op(
        transpose_matrix,
        solver._data.solution.lambdas,
        body_space,
        state.world_mask,
    )
    state.v_aug.zero_()
    wp.launch(
        kernel=_sparse_delassus_gemv_rows,
        dim=(bsm.num_matrices, bsm.max_of_num_nzb),
        inputs=[
            bsm.dims,
            bsm.num_nzb,
            bsm.nzb_start,
            bsm.nzb_coords,
            bsm.nzb_values,
            bsm.row_start,
            bsm.col_start,
            problem.data.dim,
            problem.data.njc,
            row_kind,
            regularization,
            body_space,
            state.v_aug,
            solver._data.solution.lambdas,
            state.world_mask,
        ],
        device=solver.device,
    )


def _sparse_delassus_update_unilateral_rows(
    solver,
    problem: DualProblem,
    block_iteration: int,
    contact_iteration: int,
) -> None:
    if _sparse_delassus_update_unilateral_offsets(solver, problem, block_iteration, contact_iteration):
        return

    state = solver._data.state
    _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_UNILATERAL)
    wp.launch(
        kernel=_solve_dvi_sparse_unilateral_jacobi_update,
        dim=(solver._size.num_worlds, solver._size.max_of_max_total_cts),
        inputs=[
            problem.data.dim,
            problem.data.vio,
            problem.data.njc,
            problem.data.nl,
            problem.data.nc,
            problem.data.lcgo,
            problem.data.ccgo,
            problem.data.cio,
            problem.data.mu,
            state.scratch,
            problem.data.P,
            problem.data.v_f,
            state.v_aug,
            state.contact_block_inv,
            block_iteration,
            contact_iteration,
            solver._data.config,
            solver._data.solution.lambdas,
        ],
        device=solver.device,
    )


def _sparse_delassus_update_unilateral_offsets(
    solver,
    problem: DualProblem,
    block_iteration: int,
    contact_iteration: int,
) -> bool:
    delassus = problem.delassus
    state = solver._data.state
    regularization = _sparse_delassus_regularization(problem)
    transpose_matrix = getattr(delassus, "_transpose_op_matrix", None)
    body_space = getattr(delassus, "_vec_temp_body_space", None)
    bsm = getattr(delassus, "bsm", None)
    jacobians = getattr(delassus, "_jacobians", None)
    limits = getattr(delassus, "_limits", None)
    contacts = getattr(delassus, "_contacts", None)
    limit_offsets = getattr(jacobians, "_J_cts_limit_nzb_offsets", None)
    contact_offsets = getattr(jacobians, "_J_cts_contact_nzb_offsets", None)
    has_limits = limits is not None and limits.model_max_limits_host > 0 and limit_offsets is not None
    has_contacts = contacts is not None and contacts.model_max_contacts_host > 0 and contact_offsets is not None

    if (
        regularization is None
        or transpose_matrix is None
        or body_space is None
        or bsm is None
        or getattr(delassus, "ATy_op", None) is None
        or not (has_limits or has_contacts)
    ):
        return False

    if getattr(delassus, "_needs_update", False):
        delassus.update()

    delassus.ATy_op(
        transpose_matrix,
        solver._data.solution.lambdas,
        body_space,
        state.world_mask,
    )

    if has_limits and has_contacts:
        # Fuse the two independent sweeps (disjoint lambda outputs, shared
        # body_space) into one launch to remove a per-iteration kernel launch.
        limits_capacity = limits.model_max_limits_host
        wp.launch(
            kernel=_solve_dvi_sparse_unilateral_offset_update,
            dim=limits_capacity + contacts.model_max_contacts_host,
            inputs=[
                limits_capacity,
                bsm.num_nzb,
                bsm.nzb_start,
                bsm.nzb_coords,
                bsm.nzb_values,
                bsm.row_start,
                bsm.col_start,
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                limit_offsets,
                problem.data.nl,
                problem.data.lcgo,
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                contact_offsets,
                problem.data.nc,
                problem.data.ccgo,
                problem.data.cio,
                problem.data.mu,
                state.contact_block_inv,
                problem.data.vio,
                state.scratch,
                problem.data.P,
                problem.data.v_f,
                regularization,
                body_space,
                block_iteration,
                contact_iteration,
                solver._data.config,
                solver._data.solution.lambdas,
            ],
            device=solver.device,
        )
        return True

    if has_limits:
        wp.launch(
            kernel=_solve_dvi_sparse_limits_offset_update,
            dim=limits.model_max_limits_host,
            inputs=[
                bsm.num_nzb,
                bsm.nzb_start,
                bsm.nzb_coords,
                bsm.nzb_values,
                bsm.row_start,
                bsm.col_start,
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                limit_offsets,
                problem.data.vio,
                problem.data.nl,
                problem.data.lcgo,
                state.scratch,
                problem.data.P,
                problem.data.v_f,
                regularization,
                body_space,
                block_iteration,
                contact_iteration,
                solver._data.config,
                solver._data.solution.lambdas,
            ],
            device=solver.device,
        )

    if has_contacts:
        wp.launch(
            kernel=_solve_dvi_sparse_contacts_offset_update,
            dim=contacts.model_max_contacts_host,
            inputs=[
                bsm.num_nzb,
                bsm.nzb_start,
                bsm.nzb_coords,
                bsm.nzb_values,
                bsm.row_start,
                bsm.col_start,
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                contact_offsets,
                problem.data.vio,
                problem.data.nc,
                problem.data.ccgo,
                problem.data.cio,
                problem.data.mu,
                state.scratch,
                problem.data.P,
                problem.data.v_f,
                regularization,
                body_space,
                state.contact_block_inv,
                block_iteration,
                contact_iteration,
                solver._data.config,
                solver._data.solution.lambdas,
            ],
            device=solver.device,
        )

    return True


def _compute_sparse_contact_block_inverse(solver, problem: DualProblem) -> None:
    jacobian = problem.delassus.constraint_jacobian
    wp.launch(
        kernel=sparse_kernels._compute_sparse_contact_block_inverse,
        dim=(solver._size.num_worlds, solver._size.max_of_max_contacts),
        inputs=[
            problem.delassus.model.info.bodies_offset,
            problem.delassus.model.bodies.inv_m_i,
            problem.delassus.data.bodies.inv_I_i,
            jacobian.nzb_start,
            jacobian.num_nzb,
            jacobian.nzb_coords,
            jacobian.nzb_values,
            problem.data.nc,
            problem.data.ccgo,
            problem.data.cio,
            problem.data.vio,
            problem.data.P,
            solver._data.config,
            jacobian.max_of_num_nzb,
            solver._data.state.contact_block_inv,
        ],
        device=solver.device,
    )


def _factor_sparse_bilateral_block(solver, problem: DualProblem) -> None:
    operator = solver._data.bilateral_operator
    state = solver._data.state
    operator.info.dim = operator.info.maxdim
    operator.mat.zero_()
    state.bilateral_preconditioner.zero_()
    problem.delassus.diagonal(state.scratch)

    jacobian = problem.delassus.constraint_jacobian
    if solver._bilateral_nzb_pairs is None:
        _build_sparse_bilateral_pairs(solver, problem)
    wp.launch(
        kernel=_set_sparse_bilateral_diagonal,
        dim=(solver._size.num_worlds, solver._size.max_of_num_joint_cts),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            operator.info.mio,
            operator.info.vio,
            state.scratch,
            operator.mat,
            state.bilateral_preconditioner,
        ],
        device=solver.device,
    )
    pair_wid, pair_row, pair_col, pair_bid, pair_i, pair_j = solver._bilateral_nzb_pairs
    if pair_wid.size > 0:
        wp.launch(
            kernel=_build_sparse_bilateral_block,
            dim=pair_wid.size,
            inputs=[
                problem.delassus.model.bodies.inv_m_i,
                problem.delassus.data.bodies.inv_I_i,
                pair_wid,
                pair_row,
                pair_col,
                pair_bid,
                pair_i,
                pair_j,
                jacobian.nzb_values,
                problem.data.njc,
                operator.info.mio,
                operator.info.vio,
                state.bilateral_preconditioner,
                operator.mat,
            ],
            device=solver.device,
        )
    solver._bilateral_solver.compute(A=operator.mat)


def _build_sparse_bilateral_pairs(solver, problem: DualProblem) -> None:
    """Cache joint Jacobian block pairs that contribute to the bilateral matrix."""
    jacobian = problem.delassus.constraint_jacobian
    counts = problem.delassus.joint_constraint_nzb_count.numpy().tolist()
    starts = jacobian.nzb_start.numpy().tolist()
    coords = jacobian.nzb_coords.numpy()
    joint_counts = problem.data.njc.numpy().tolist()
    body_offsets = problem.delassus.model.info.bodies_offset.numpy().tolist()

    pair_wid: list[int] = []
    pair_row: list[int] = []
    pair_col: list[int] = []
    pair_bid: list[int] = []
    pair_i: list[int] = []
    pair_j: list[int] = []
    for wid, count in enumerate(counts):
        start = starts[wid]
        njc = joint_counts[wid]
        for local_i in range(count):
            nzb_i = start + local_i
            row = int(coords[nzb_i, 0])
            body_col = int(coords[nzb_i, 1])
            if row >= njc:
                continue
            for local_j in range(count):
                nzb_j = start + local_j
                col = int(coords[nzb_j, 0])
                if row < col < njc and body_col == int(coords[nzb_j, 1]):
                    pair_wid.append(wid)
                    pair_row.append(row)
                    pair_col.append(col)
                    pair_bid.append(body_offsets[wid] + body_col // 6)
                    pair_i.append(nzb_i)
                    pair_j.append(nzb_j)

    solver._bilateral_nzb_pairs = tuple(
        wp.array(values, dtype=int32, device=solver.device)
        for values in (pair_wid, pair_row, pair_col, pair_bid, pair_i, pair_j)
    )


def _solve_sparse_bilateral_block(solver, problem: DualProblem, active_dim: wp.array[int32] | None = None) -> None:
    operator = solver._data.bilateral_operator
    state = solver._data.state
    wp.launch(
        kernel=_zero_bilateral_lambdas,
        dim=(solver._size.num_worlds, solver._size.max_of_num_joint_cts),
        inputs=[
            problem.data.njc,
            problem.data.vio,
            solver._data.solution.lambdas,
        ],
        device=solver.device,
    )
    _sparse_delassus_matvec_rows(solver, problem, _SPARSE_DELASSUS_ROWS_JOINTS)
    wp.launch(
        kernel=_build_sparse_bilateral_rhs,
        dim=(solver._size.num_worlds, solver._size.max_of_num_joint_cts),
        inputs=[
            problem.data.vio,
            problem.data.njc,
            problem.data.v_f,
            state.v_aug,
            operator.info.vio,
            state.bilateral_preconditioner,
            state.bilateral_rhs,
        ],
        device=solver.device,
    )
    full_dim = operator.info.dim
    if active_dim is not None:
        operator.info.dim = active_dim
    try:
        solver._bilateral_solver.solve(b=state.bilateral_rhs, x=state.bilateral_solution)
    finally:
        operator.info.dim = full_dim
    wp.launch(
        kernel=_scatter_bilateral_solution,
        dim=(solver._size.num_worlds, solver._size.max_of_num_joint_cts),
        inputs=[
            problem.data.vio,
            problem.data.njc,
            operator.info.vio,
            state.bilateral_preconditioner,
            state.bilateral_solution,
            solver._data.solution.lambdas,
        ],
        device=solver.device,
    )


def _solve_sparse_with_bilateral_direct_block(solver, problem: DualProblem) -> None:
    state = solver._data.state
    _factor_sparse_bilateral_block(solver, problem)
    _solve_sparse_bilateral_block(solver, problem)
    if not solver._has_unilateral_constraints:
        _compute_sparse_solution_vectors(solver, problem)
        return

    wp.launch(
        kernel=_initialize_dvi_status,
        dim=solver._size.num_worlds,
        inputs=[
            solver._data.config,
            solver._data.status,
        ],
        device=solver.device,
    )
    wp.launch(
        kernel=_set_dvi_bilateral_active_dim,
        dim=solver._size.num_worlds,
        inputs=[
            problem.data.njc,
            problem.data.nl,
            problem.data.nc,
            state.bilateral_active_dim,
        ],
        device=solver.device,
    )

    for block_iteration in range(solver._max_block_iterations):
        for contact_iteration in range(solver._max_contact_iterations):
            _sparse_delassus_update_unilateral_rows(solver, problem, block_iteration, contact_iteration)

        if solver._should_solve_bilateral_after_block(block_iteration):
            _solve_sparse_bilateral_block(solver, problem, active_dim=state.bilateral_active_dim)

    _solve_sparse_bilateral_block(solver, problem, active_dim=state.bilateral_active_dim)
    wp.launch(
        kernel=_set_dvi_direct_status_iterations,
        dim=solver._size.num_worlds,
        inputs=[
            problem.data.nl,
            problem.data.nc,
            solver._data.config,
            solver._data.status,
        ],
        device=solver.device,
    )
    _compute_sparse_solution_vectors(solver, problem)
