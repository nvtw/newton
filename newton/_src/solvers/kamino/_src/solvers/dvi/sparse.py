# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse DVI solve path for Kamino dual systems."""

from __future__ import annotations

import warp as wp

from ...dynamics.dual import DualProblem
from . import sparse_kernels
from .kernels import _initialize_dvi_status, _scatter_bilateral_solution, _set_dvi_direct_status_iterations
from .sparse_kernels import (
    _build_sparse_bilateral_block,
    _build_sparse_bilateral_rhs,
    _compute_dvi_sparse_solution_vectors,
    _set_dvi_sparse_status_iterations,
    _set_sparse_bilateral_diagonal,
    _solve_dvi_sparse_jacobi_update,
    _solve_dvi_sparse_unilateral_jacobi_update,
    _zero_bilateral_lambdas,
)

wp.set_module_options({"enable_backward": False})


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
    wp.launch(
        kernel=_build_sparse_bilateral_block,
        dim=(solver._size.num_worlds, jacobian.max_of_num_nzb * jacobian.max_of_num_nzb),
        inputs=[
            problem.delassus.model.info.bodies_offset,
            problem.delassus.model.bodies.inv_m_i,
            problem.delassus.data.bodies.inv_I_i,
            problem.delassus.joint_constraint_nzb_count,
            jacobian.nzb_start,
            jacobian.nzb_coords,
            jacobian.nzb_values,
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
    solver._bilateral_solver.compute(A=operator.mat)


def _solve_sparse_bilateral_block(solver, problem: DualProblem) -> None:
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
    problem.delassus.matvec(
        x=solver._data.solution.lambdas,
        y=state.v_aug,
        world_mask=state.world_mask,
    )
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
    solver._bilateral_solver.solve(b=state.bilateral_rhs, x=state.bilateral_solution)
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

    for block_iteration in range(solver._max_block_iterations):
        for contact_iteration in range(solver._max_contact_iterations):
            problem.delassus.matvec(
                x=solver._data.solution.lambdas,
                y=state.v_aug,
                world_mask=state.world_mask,
            )
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

        if block_iteration + 1 < solver._max_block_iterations:
            _solve_sparse_bilateral_block(solver, problem)

    _solve_sparse_bilateral_block(solver, problem)
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
