# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import float32, int32, mat33f, vec3f, vec6f
from ..padmm.math import project_to_coulomb_cone
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})


@wp.func
def _project_contact_diagonal_update(
    lambda_old: vec3f,
    v_c: vec3f,
    D_diag: vec3f,
    regularization: float32,
    omega: float32,
    mu: float32,
) -> vec3f:
    lambda_arg = lambda_old
    if D_diag.x > FLOAT32_EPS:
        lambda_arg.x = lambda_old.x - omega * v_c.x / (D_diag.x + regularization)
    if D_diag.y > FLOAT32_EPS:
        lambda_arg.y = lambda_old.y - omega * v_c.y / (D_diag.y + regularization)
    if D_diag.z > FLOAT32_EPS:
        lambda_arg.z = lambda_old.z - omega * v_c.z / (D_diag.z + regularization)
    return project_to_coulomb_cone(lambda_arg, mu)


@wp.func
def _contact_trace_preconditioner(D_diag: vec3f) -> vec3f:
    D_eff = (D_diag.x + D_diag.y + D_diag.z) / float32(3.0)
    return vec3f(D_eff, D_eff, D_eff)


@wp.kernel
def _zero_bilateral_lambdas(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    solution_lambdas[problem_vio[wid] + row] = 0.0


@wp.kernel
def _build_sparse_bilateral_rhs(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    bilateral_vio: wp.array[int32],
    bilateral_P: wp.array[float32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    rhs = -(state_v_aug[pvio + row] + problem_v_f[pvio + row])
    bilateral_rhs[bvio + row] = bilateral_P[bvio + row] * rhs


@wp.kernel
def _build_sparse_bilateral_block(
    # Inputs:
    model_info_bodies_offset: wp.array[int32],
    model_bodies_inv_m_i: wp.array[float32],
    data_bodies_inv_I_i: wp.array[mat33f],
    jacobian_cts_num_joint_nzb: wp.array[int32],
    jacobian_cts_nzb_start: wp.array[int32],
    jacobian_cts_nzb_coords: wp.array2d[int32],
    jacobian_cts_nzb_values: wp.array[vec6f],
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    problem_diag: wp.array[float32],
    # Outputs:
    bilateral_D: wp.array[float32],
    bilateral_P: wp.array[float32],
):
    wid, tid = wp.tid()

    njc = problem_njc[wid]
    if njc == 0:
        return

    num_nzb = jacobian_cts_num_joint_nzb[wid]
    if num_nzb == 0:
        return

    block_id_i = tid // num_nzb
    block_id_j = tid % num_nzb
    if block_id_i >= num_nzb:
        return

    nzb_start = jacobian_cts_nzb_start[wid]
    global_block_id_i = nzb_start + block_id_i
    global_block_id_j = nzb_start + block_id_j

    block_coords_i = jacobian_cts_nzb_coords[global_block_id_i]
    block_coords_j = jacobian_cts_nzb_coords[global_block_id_j]
    if block_coords_i[1] != block_coords_j[1]:
        return

    row = block_coords_i[0]
    col = block_coords_j[0]
    if row > col or row >= njc or col >= njc:
        return

    block_i = jacobian_cts_nzb_values[global_block_id_i]
    block_j = jacobian_cts_nzb_values[global_block_id_j]
    Jv_i = vec3f(block_i[0], block_i[1], block_i[2])
    Jv_j = vec3f(block_j[0], block_j[1], block_j[2])
    Jw_i = vec3f(block_i[3], block_i[4], block_i[5])
    Jw_j = vec3f(block_j[3], block_j[4], block_j[5])

    bid_k = model_info_bodies_offset[wid] + block_coords_i[1] // int32(6)
    inv_m_k = model_bodies_inv_m_i[bid_k]
    inv_I_k = data_bodies_inv_I_i[bid_k]
    D_ij = inv_m_k * wp.dot(Jv_i, Jv_j) + wp.dot(Jw_i, inv_I_k @ Jw_j)

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    p_row = wp.sqrt(1.0 / (wp.abs(problem_diag[pvio + row]) + FLOAT32_EPS))
    p_col = wp.sqrt(1.0 / (wp.abs(problem_diag[pvio + col]) + FLOAT32_EPS))
    val = p_row * D_ij * p_col

    bmio = bilateral_mio[wid]
    wp.atomic_add(bilateral_D, bmio + njc * row + col, val)
    if row != col:
        wp.atomic_add(bilateral_D, bmio + njc * col + row, val)
    else:
        bilateral_P[bvio + row] = p_row


@wp.kernel
def _set_sparse_bilateral_diagonal(
    # Inputs:
    problem_njc: wp.array[int32],
    problem_vio: wp.array[int32],
    bilateral_mio: wp.array[int32],
    bilateral_vio: wp.array[int32],
    problem_diag: wp.array[float32],
    # Outputs:
    bilateral_D: wp.array[float32],
    bilateral_P: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    bmio = bilateral_mio[wid]
    diag = wp.abs(problem_diag[pvio + row])
    p = wp.sqrt(1.0 / (diag + FLOAT32_EPS))
    bilateral_P[bvio + row] = p
    bilateral_D[bmio + njc * row + row] = p * diag * p + float32(7.0e-7)


@wp.kernel
def _solve_dvi_sparse_jacobi_update(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    cfg = solver_config[wid]
    if iteration >= cfg.max_iterations:
        return

    vio = problem_vio[wid]
    v_i = state_v_aug[vio + tid] + problem_v_f[vio + tid]
    P_i = problem_P[vio + tid]
    D_ii_raw = wp.abs(problem_diag[vio + tid]) * P_i * P_i
    D_ii = D_ii_raw + cfg.regularization + FLOAT32_EPS

    njc = problem_njc[wid]
    if tid < njc:
        solution_lambdas[vio + tid] += -cfg.omega * v_i / D_ii
        return

    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    lid = tid - lcgo
    if lid >= 0 and lid < nl:
        lambda_limit_old = solution_lambdas[vio + tid]
        lambda_limit_new = lambda_limit_old
        if D_ii_raw > FLOAT32_EPS:
            lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
        solution_lambdas[vio + tid] = lambda_limit_new
        return

    nc = problem_nc[wid]
    ccgo = problem_ccgo[wid]
    contact_row = tid - ccgo
    if contact_row < 0 or contact_row >= int32(3) * nc or contact_row % int32(3) != int32(0):
        return

    cid = contact_row // int32(3)
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]

    v_t0 = state_v_aug[ccio_v + 0] + problem_v_f[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1] + problem_v_f[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + problem_v_f[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    P_0 = problem_P[ccio_v + 0]
    P_1 = problem_P[ccio_v + 1]
    P_2 = problem_P[ccio_v + 2]
    D_00 = wp.abs(problem_diag[ccio_v + 0]) * P_0 * P_0
    D_11 = wp.abs(problem_diag[ccio_v + 1]) * P_1 * P_1
    D_22 = wp.abs(problem_diag[ccio_v + 2]) * P_2 * P_2

    lambda_contact_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    lambda_projected = _project_contact_diagonal_update(
        lambda_contact_old,
        v_c,
        _contact_trace_preconditioner(vec3f(D_00, D_11, D_22)),
        cfg.regularization,
        cfg.contact_jacobi_omega,
        mu_c,
    )
    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_contact_old)

    solution_lambdas[ccio_v + 0] = lambda_contact_new.x
    solution_lambdas[ccio_v + 1] = lambda_contact_new.y
    solution_lambdas[ccio_v + 2] = lambda_contact_new.z


@wp.kernel
def _solve_dvi_sparse_unilateral_jacobi_update(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_diag: wp.array[float32],
    problem_P: wp.array[float32],
    problem_v_f: wp.array[float32],
    state_v_aug: wp.array[float32],
    block_iteration: int32,
    contact_iteration: int32,
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    cfg = solver_config[wid]
    if block_iteration >= cfg.block_iterations or contact_iteration >= cfg.contact_iterations:
        return

    njc = problem_njc[wid]
    if tid < njc:
        return

    vio = problem_vio[wid]
    v_i = state_v_aug[vio + tid] + problem_v_f[vio + tid]
    P_i = problem_P[vio + tid]
    D_ii_raw = wp.abs(problem_diag[vio + tid]) * P_i * P_i
    D_ii = D_ii_raw + cfg.regularization + FLOAT32_EPS

    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    lid = tid - lcgo
    if lid >= 0 and lid < nl:
        lambda_limit_old = solution_lambdas[vio + tid]
        lambda_limit_new = lambda_limit_old
        if D_ii_raw > FLOAT32_EPS:
            lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
        solution_lambdas[vio + tid] = lambda_limit_new
        return

    nc = problem_nc[wid]
    ccgo = problem_ccgo[wid]
    contact_row = tid - ccgo
    if contact_row < 0 or contact_row >= int32(3) * nc or contact_row % int32(3) != int32(0):
        return

    cid = contact_row // int32(3)
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]

    v_t0 = state_v_aug[ccio_v + 0] + problem_v_f[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1] + problem_v_f[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + problem_v_f[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    P_0 = problem_P[ccio_v + 0]
    P_1 = problem_P[ccio_v + 1]
    P_2 = problem_P[ccio_v + 2]
    D_00 = wp.abs(problem_diag[ccio_v + 0]) * P_0 * P_0
    D_11 = wp.abs(problem_diag[ccio_v + 1]) * P_1 * P_1
    D_22 = wp.abs(problem_diag[ccio_v + 2]) * P_2 * P_2

    lambda_contact_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    lambda_projected = _project_contact_diagonal_update(
        lambda_contact_old,
        v_c,
        _contact_trace_preconditioner(vec3f(D_00, D_11, D_22)),
        cfg.regularization,
        cfg.contact_jacobi_omega,
        mu_c,
    )
    lambda_contact_new = lambda_contact_old + cfg.contact_jacobi_relaxation * (lambda_projected - lambda_contact_old)

    solution_lambdas[ccio_v + 0] = lambda_contact_new.x
    solution_lambdas[ccio_v + 1] = lambda_contact_new.y
    solution_lambdas[ccio_v + 2] = lambda_contact_new.z


@wp.kernel
def _compute_dvi_sparse_solution_vectors(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_v_f: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    v_i = problem_vio[wid] + tid
    v_plus = state_v_aug[v_i] + problem_v_f[v_i]
    solution_v_plus[v_i] = v_plus
    state_v_aug[v_i] = v_plus
    state_s[v_i] = 0.0


@wp.kernel
def _set_dvi_sparse_status_iterations(
    # Inputs:
    problem_dim: wp.array[int32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    status = solver_status[wid]
    if problem_dim[wid] == int32(0):
        status.iterations = int32(0)
    else:
        status.iterations = solver_config[wid].max_iterations
    solver_status[wid] = status
