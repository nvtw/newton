# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import float32, int32, vec3f
from ..padmm.math import project_to_coulomb_cone, project_to_coulomb_dual_cone
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False})


@wp.func
def _compute_row_velocity(
    ncts: int32,
    mio: int32,
    vio: int32,
    row: int32,
    D: wp.array[float32],
    v_f: wp.array[float32],
    lambdas: wp.array[float32],
) -> float32:
    v = v_f[vio + row]
    m_i = mio + ncts * row
    for j in range(ncts):
        v += D[m_i + j] * lambdas[vio + j]
    return v


@wp.func
def _contact_velocity_aug(
    ncts: int32,
    mio: int32,
    vio: int32,
    ccgo: int32,
    cio: int32,
    cid: int32,
    D: wp.array[float32],
    v_f: wp.array[float32],
    lambdas: wp.array[float32],
    mu: wp.array[float32],
) -> vec3f:
    ccio = ccgo + 3 * cid
    v_t0 = _compute_row_velocity(ncts, mio, vio, ccio + 0, D, v_f, lambdas)
    v_t1 = _compute_row_velocity(ncts, mio, vio, ccio + 1, D, v_f, lambdas)
    v_n = _compute_row_velocity(ncts, mio, vio, ccio + 2, D, v_f, lambdas)
    vt_norm = wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    return vec3f(v_t0, v_t1, v_n + mu[cio + cid] * vt_norm)


@wp.kernel
def _reset_dvi_solver_data(
    # Inputs:
    world_mask: wp.array[wp.bool],
    problem_vio: wp.array[int32],
    problem_maxdim: wp.array[int32],
    # Outputs:
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()
    if not world_mask[wid] or tid >= problem_maxdim[wid]:
        return
    v_i = problem_vio[wid] + tid
    solution_lambdas[v_i] = 0.0
    solution_v_plus[v_i] = 0.0


@wp.kernel
def _copy_bilateral_block(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    bilateral_mio: wp.array[int32],
    # Outputs:
    bilateral_D: wp.array[float32],
):
    wid, tid = wp.tid()

    njc = problem_njc[wid]
    if njc == 0 or tid >= njc * njc:
        return

    ncts = problem_dim[wid]
    pmio = problem_mio[wid]
    bmio = bilateral_mio[wid]
    row = tid // njc
    col = tid - row * njc

    val = problem_D[pmio + ncts * row + col]
    if row == col:
        val += solver_config[wid].regularization
    bilateral_D[bmio + njc * row + col] = val


@wp.kernel
def _build_bilateral_rhs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    bilateral_vio: wp.array[int32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    ncts = problem_dim[wid]
    pmio = problem_mio[wid]
    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]

    rhs = -problem_v_f[pvio + row]
    for col in range(njc, ncts):
        rhs -= problem_D[pmio + ncts * row + col] * solution_lambdas[pvio + col]
    bilateral_rhs[bvio + row] = rhs


@wp.kernel
def _scatter_bilateral_solution(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    bilateral_vio: wp.array[int32],
    bilateral_solution: wp.array[float32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    solution_lambdas[problem_vio[wid] + row] = bilateral_solution[bilateral_vio[wid] + row]


@wp.kernel
def _build_bilateral_free_velocity_rhs(
    # Inputs:
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_v_f: wp.array[float32],
    bilateral_vio: wp.array[int32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    if row >= njc:
        return

    bilateral_rhs[bilateral_vio[wid] + row] = problem_v_f[problem_vio[wid] + row]


@wp.kernel
def _build_bilateral_column_rhs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    schur_col: int32,
    bilateral_vio: wp.array[int32],
    # Outputs:
    bilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    njc = problem_njc[wid]
    ncts = problem_dim[wid]
    udim = ncts - njc
    bvio = bilateral_vio[wid]
    if row >= njc:
        return

    rhs = float32(0.0)
    if schur_col < udim:
        rhs = problem_D[problem_mio[wid] + ncts * row + njc + schur_col]
    bilateral_rhs[bvio + row] = rhs


@wp.kernel
def _build_reduced_unilateral_rhs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    bilateral_vio: wp.array[int32],
    bilateral_solution: wp.array[float32],
    unilateral_vio: wp.array[int32],
    # Outputs:
    unilateral_rhs: wp.array[float32],
):
    wid, row = wp.tid()

    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    udim = ncts - njc
    uvio = unilateral_vio[wid]
    if row >= udim:
        return

    pmio = problem_mio[wid]
    pvio = problem_vio[wid]
    bvio = bilateral_vio[wid]
    full_row = njc + row

    rhs = problem_v_f[pvio + full_row]
    for col in range(njc):
        rhs -= problem_D[pmio + ncts * full_row + col] * bilateral_solution[bvio + col]
    unilateral_rhs[uvio + row] = rhs


@wp.kernel
def _write_reduced_unilateral_column(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_D: wp.array[float32],
    schur_col: int32,
    bilateral_vio: wp.array[int32],
    bilateral_solution: wp.array[float32],
    unilateral_maxdim: wp.array[int32],
    unilateral_mio: wp.array[int32],
    # Outputs:
    unilateral_D: wp.array[float32],
):
    wid, row = wp.tid()

    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    udim = ncts - njc
    umax = unilateral_maxdim[wid]
    if row >= udim or schur_col >= udim:
        return

    pmio = problem_mio[wid]
    bmio = bilateral_vio[wid]
    umio = unilateral_mio[wid]
    full_row = njc + row
    full_col = njc + schur_col

    val = problem_D[pmio + ncts * full_row + full_col]
    for e in range(njc):
        val -= problem_D[pmio + ncts * full_row + e] * bilateral_solution[bmio + e]
    unilateral_D[umio + umax * row + schur_col] = val


@wp.kernel
def _gather_unilateral_solution(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    unilateral_vio: wp.array[int32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    unilateral_solution: wp.array[float32],
):
    wid, row = wp.tid()

    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    udim = ncts - njc
    if row >= udim:
        return

    unilateral_solution[unilateral_vio[wid] + row] = solution_lambdas[problem_vio[wid] + njc + row]


@wp.kernel
def _scatter_unilateral_solution(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    unilateral_vio: wp.array[int32],
    unilateral_solution: wp.array[float32],
    # Outputs:
    solution_lambdas: wp.array[float32],
):
    wid, row = wp.tid()

    ncts = problem_dim[wid]
    njc = problem_njc[wid]
    udim = ncts - njc
    if row >= udim:
        return

    solution_lambdas[problem_vio[wid] + njc + row] = unilateral_solution[unilateral_vio[wid] + row]


@wp.func
def _compute_reduced_row_velocity(
    udim: int32,
    umax: int32,
    umio: int32,
    uvio: int32,
    row: int32,
    D: wp.array[float32],
    rhs: wp.array[float32],
    lambdas: wp.array[float32],
) -> float32:
    v = rhs[uvio + row]
    m_i = umio + umax * row
    for j in range(udim):
        v += D[m_i + j] * lambdas[uvio + j]
    return v


@wp.func
def _reduced_contact_velocity_aug(
    udim: int32,
    umax: int32,
    umio: int32,
    uvio: int32,
    contact_offset: int32,
    cio: int32,
    cid: int32,
    D: wp.array[float32],
    rhs: wp.array[float32],
    lambdas: wp.array[float32],
    mu: wp.array[float32],
) -> vec3f:
    ccio = contact_offset + 3 * cid
    v_t0 = _compute_reduced_row_velocity(udim, umax, umio, uvio, ccio + 0, D, rhs, lambdas)
    v_t1 = _compute_reduced_row_velocity(udim, umax, umio, uvio, ccio + 1, D, rhs, lambdas)
    v_n = _compute_reduced_row_velocity(udim, umax, umio, uvio, ccio + 2, D, rhs, lambdas)
    vt_norm = wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    return vec3f(v_t0, v_t1, v_n + mu[cio + cid] * vt_norm)


@wp.kernel
def _solve_dvi_reduced_unilateral_pgs(
    # Inputs:
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    unilateral_maxdim: wp.array[int32],
    unilateral_mio: wp.array[int32],
    unilateral_vio: wp.array[int32],
    unilateral_D: wp.array[float32],
    unilateral_rhs: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    unilateral_solution: wp.array[float32],
):
    wid = wp.tid()

    nl = problem_nl[wid]
    nc = problem_nc[wid]
    udim = nl + 3 * nc
    umax = unilateral_maxdim[wid]
    umio = unilateral_mio[wid]
    uvio = unilateral_vio[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = int32(0)
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if udim == 0:
        status.converged = int32(1)
        status.iterations = int32(1)
        solver_status[wid] = status
        return

    done = int32(0)
    for iteration in range(cfg.max_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            for l in range(nl):
                v_i = _compute_reduced_row_velocity(
                    udim, umax, umio, uvio, l, unilateral_D, unilateral_rhs, unilateral_solution
                )
                D_ii_raw = wp.abs(unilateral_D[umio + umax * l + l])
                lambda_limit_old = unilateral_solution[uvio + l]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                unilateral_solution[uvio + l] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            contact_offset = nl
            for cid in range(nc):
                ccio = contact_offset + 3 * cid
                v_c = _reduced_contact_velocity_aug(
                    udim,
                    umax,
                    umio,
                    uvio,
                    contact_offset,
                    cio,
                    cid,
                    unilateral_D,
                    unilateral_rhs,
                    unilateral_solution,
                    problem_mu,
                )
                D_00 = wp.abs(unilateral_D[umio + umax * (ccio + 0) + (ccio + 0)])
                D_11 = wp.abs(unilateral_D[umio + umax * (ccio + 1) + (ccio + 1)])
                D_22 = wp.abs(unilateral_D[umio + umax * (ccio + 2) + (ccio + 2)])
                D_kk_raw = wp.max(vec3f(D_00, D_11, D_22))
                lambda_contact_old = vec3f(
                    unilateral_solution[uvio + ccio + 0],
                    unilateral_solution[uvio + ccio + 1],
                    unilateral_solution[uvio + ccio + 2],
                )
                lambda_contact_new = lambda_contact_old
                if D_kk_raw > FLOAT32_EPS:
                    lambda_contact_arg = lambda_contact_old - (cfg.omega / (D_kk_raw + cfg.regularization)) * v_c
                    lambda_contact_new = project_to_coulomb_cone(lambda_contact_arg, problem_mu[cio + cid])
                unilateral_solution[uvio + ccio + 0] = lambda_contact_new.x
                unilateral_solution[uvio + ccio + 1] = lambda_contact_new.y
                unilateral_solution[uvio + ccio + 2] = lambda_contact_new.z
                lambda_delta = lambda_contact_new - lambda_contact_old
                max_step = wp.max(max_step, wp.max(wp.abs(lambda_delta)))
                max_velocity = wp.max(max_velocity, wp.max(wp.abs(v_c)))
                max_complementarity = wp.max(max_complementarity, wp.abs(wp.dot(lambda_contact_new, v_c)))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

    solver_status[wid] = status


@wp.kernel
def _compute_dvi_status_residuals(
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
    solver_config: wp.array[DVIConfigStruct],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = solver_status[wid]
    if status.iterations == 0:
        status.iterations = int32(1)

    r_b = float32(0.0)
    r_p = float32(0.0)
    r_d = float32(0.0)
    r_c = float32(0.0)

    for jid in range(njc):
        v_j = state_v_aug[vio + jid]
        r_b = wp.max(r_b, wp.abs(v_j))

    for lid in range(nl):
        lcio = vio + lcgo + lid
        lambda_l = solution_lambdas[lcio]
        v_l = state_v_aug[lcio]
        r_p = wp.max(r_p, wp.abs(lambda_l - wp.max(0.0, lambda_l)))
        r_d = wp.max(r_d, wp.abs(v_l - wp.max(0.0, v_l)))
        r_c = wp.max(r_c, wp.abs(lambda_l * v_l))

    for cid in range(nc):
        ccio = vio + ccgo + 3 * cid
        mu_c = problem_mu[cio + cid]
        lambda_c = vec3f(solution_lambdas[ccio], solution_lambdas[ccio + 1], solution_lambdas[ccio + 2])
        v_c = vec3f(state_v_aug[ccio], state_v_aug[ccio + 1], state_v_aug[ccio + 2])
        lambda_proj = project_to_coulomb_cone(lambda_c, mu_c)
        v_proj = project_to_coulomb_dual_cone(v_c, mu_c)
        r_p = wp.max(r_p, wp.max(wp.abs(lambda_c - lambda_proj)))
        r_d = wp.max(r_d, wp.max(wp.abs(v_c - v_proj)))
        r_c = wp.max(r_c, wp.abs(wp.dot(lambda_c, v_c)))

    status.r_b = r_b
    status.r_p = r_p
    status.r_d = wp.max(r_d, r_b)
    status.r_c = r_c
    status.converged = int32(0)
    if ncts == 0 or (r_b <= cfg.tolerance and r_p <= cfg.tolerance and r_d <= cfg.tolerance and r_c <= cfg.tolerance):
        status.converged = int32(1)
    solver_status[wid] = status


@wp.kernel
def _solve_dvi_pgs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_njc: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    solution_lambdas: wp.array[float32],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    njc = problem_njc[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = int32(0)
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if ncts == 0:
        status.converged = int32(1)
        solver_status[wid] = status
        return

    done = int32(0)
    for iteration in range(cfg.max_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            # Equality constraints use scalar Gauss-Seidel updates. This keeps the
            # solve on Kamino's Delassus system while avoiding a separate subsystem.
            for i in range(njc):
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii = wp.abs(problem_D[mio + ncts * i + i]) + cfg.regularization + FLOAT32_EPS
                delta = -cfg.omega * v_i / D_ii
                solution_lambdas[vio + i] += delta
                max_step = wp.max(max_step, wp.abs(delta))
                max_velocity = wp.max(max_velocity, wp.abs(v_i))

            for l in range(nl):
                i = lcgo + l
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii_raw = wp.abs(problem_D[mio + ncts * i + i])
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                solution_lambdas[vio + i] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            for cid in range(nc):
                ccio = ccgo + 3 * cid
                v_c = _contact_velocity_aug(
                    ncts,
                    mio,
                    vio,
                    ccgo,
                    cio,
                    cid,
                    problem_D,
                    problem_v_f,
                    solution_lambdas,
                    problem_mu,
                )
                D_00 = wp.abs(problem_D[mio + ncts * (ccio + 0) + (ccio + 0)])
                D_11 = wp.abs(problem_D[mio + ncts * (ccio + 1) + (ccio + 1)])
                D_22 = wp.abs(problem_D[mio + ncts * (ccio + 2) + (ccio + 2)])
                D_kk_raw = wp.max(vec3f(D_00, D_11, D_22))
                lambda_contact_old = vec3f(
                    solution_lambdas[vio + ccio + 0],
                    solution_lambdas[vio + ccio + 1],
                    solution_lambdas[vio + ccio + 2],
                )
                lambda_contact_new = lambda_contact_old
                if D_kk_raw > FLOAT32_EPS:
                    lambda_contact_arg = lambda_contact_old - (cfg.omega / (D_kk_raw + cfg.regularization)) * v_c
                    lambda_contact_new = project_to_coulomb_cone(lambda_contact_arg, problem_mu[cio + cid])
                solution_lambdas[vio + ccio + 0] = lambda_contact_new.x
                solution_lambdas[vio + ccio + 1] = lambda_contact_new.y
                solution_lambdas[vio + ccio + 2] = lambda_contact_new.z
                lambda_delta = lambda_contact_new - lambda_contact_old
                max_step = wp.max(max_step, wp.max(wp.abs(lambda_delta)))
                max_velocity = wp.max(max_velocity, wp.max(wp.abs(v_c)))
                max_complementarity = wp.max(max_complementarity, wp.abs(wp.dot(lambda_contact_new, v_c)))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

    solver_status[wid] = status


@wp.kernel
def _solve_dvi_unilateral_pgs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    solution_lambdas: wp.array[float32],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    cio = problem_cio[wid]
    cfg = solver_config[wid]

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = int32(0)
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if ncts == 0 or (nl == 0 and nc == 0):
        status.converged = int32(1)
        status.iterations = int32(1)
        solver_status[wid] = status
        return

    done = int32(0)
    for iteration in range(cfg.max_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            for l in range(nl):
                i = lcgo + l
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii_raw = wp.abs(problem_D[mio + ncts * i + i])
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                solution_lambdas[vio + i] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            for cid in range(nc):
                ccio = ccgo + 3 * cid
                v_c = _contact_velocity_aug(
                    ncts,
                    mio,
                    vio,
                    ccgo,
                    cio,
                    cid,
                    problem_D,
                    problem_v_f,
                    solution_lambdas,
                    problem_mu,
                )
                D_00 = wp.abs(problem_D[mio + ncts * (ccio + 0) + (ccio + 0)])
                D_11 = wp.abs(problem_D[mio + ncts * (ccio + 1) + (ccio + 1)])
                D_22 = wp.abs(problem_D[mio + ncts * (ccio + 2) + (ccio + 2)])
                D_kk_raw = wp.max(vec3f(D_00, D_11, D_22))
                lambda_contact_old = vec3f(
                    solution_lambdas[vio + ccio + 0],
                    solution_lambdas[vio + ccio + 1],
                    solution_lambdas[vio + ccio + 2],
                )
                lambda_contact_new = lambda_contact_old
                if D_kk_raw > FLOAT32_EPS:
                    lambda_contact_arg = lambda_contact_old - (cfg.omega / (D_kk_raw + cfg.regularization)) * v_c
                    lambda_contact_new = project_to_coulomb_cone(lambda_contact_arg, problem_mu[cio + cid])
                solution_lambdas[vio + ccio + 0] = lambda_contact_new.x
                solution_lambdas[vio + ccio + 1] = lambda_contact_new.y
                solution_lambdas[vio + ccio + 2] = lambda_contact_new.z
                lambda_delta = lambda_contact_new - lambda_contact_old
                max_step = wp.max(max_step, wp.max(wp.abs(lambda_delta)))
                max_velocity = wp.max(max_velocity, wp.max(wp.abs(v_c)))
                max_complementarity = wp.max(max_complementarity, wp.abs(wp.dot(lambda_contact_new, v_c)))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

    solver_status[wid] = status


@wp.kernel
def _initialize_dvi_status(
    # Inputs:
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
):
    wid = wp.tid()
    cfg = solver_config[wid]
    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = cfg.contact_iterations
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)
    solver_status[wid] = status


@wp.kernel
def _solve_dvi_limits_pgs(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nl: wp.array[int32],
    problem_lcgo: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    # Outputs:
    solver_status: wp.array[DVIStatus],
    solution_lambdas: wp.array[float32],
):
    wid = wp.tid()

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    nl = problem_nl[wid]
    lcgo = problem_lcgo[wid]
    cfg = solver_config[wid]

    status = DVIStatus()
    status.converged = int32(0)
    status.iterations = cfg.contact_iterations
    status.r_p = float32(0.0)
    status.r_d = float32(0.0)
    status.r_c = float32(0.0)
    status.r_b = float32(0.0)

    if ncts == 0 or nl == 0:
        status.converged = int32(1)
        solver_status[wid] = status
        return

    done = int32(0)
    for iteration in range(cfg.max_iterations):
        if done == 0:
            max_step = float32(0.0)
            max_velocity = float32(0.0)
            max_complementarity = float32(0.0)

            for l in range(nl):
                i = lcgo + l
                v_i = _compute_row_velocity(ncts, mio, vio, i, problem_D, problem_v_f, solution_lambdas)
                D_ii_raw = wp.abs(problem_D[mio + ncts * i + i])
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = lambda_limit_old
                if D_ii_raw > FLOAT32_EPS:
                    lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / (D_ii_raw + cfg.regularization))
                solution_lambdas[vio + i] = lambda_limit_new
                max_step = wp.max(max_step, wp.abs(lambda_limit_new - lambda_limit_old))
                max_velocity = wp.max(max_velocity, wp.abs(wp.min(v_i, 0.0)))
                max_complementarity = wp.max(max_complementarity, wp.abs(lambda_limit_new * v_i))

            status.iterations = iteration + int32(1)
            status.r_p = max_step
            status.r_d = max_velocity
            status.r_c = max_complementarity
            if max_step <= cfg.tolerance:
                status.converged = int32(1)
                done = int32(1)

    if done == 0:
        status.converged = int32(0)

    solver_status[wid] = status


@wp.kernel
def _compute_dvi_contact_jacobi_delta(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_mu: wp.array[float32],
    problem_D: wp.array[float32],
    solver_config: wp.array[DVIConfigStruct],
    state_v_aug: wp.array[float32],
    # Outputs:
    solution_lambdas: wp.array[float32],
    state_scratch: wp.array[float32],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    if cid >= nc:
        return

    ncts = problem_dim[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    ccio = ccgo + int32(3) * cid
    ccio_v = vio + ccio
    mu_c = problem_mu[problem_cio[wid] + cid]
    cfg = solver_config[wid]

    v_t0 = state_v_aug[ccio_v + 0]
    v_t1 = state_v_aug[ccio_v + 1]
    v_n = state_v_aug[ccio_v + 2] + mu_c * wp.sqrt(v_t0 * v_t0 + v_t1 * v_t1)
    v_c = vec3f(v_t0, v_t1, v_n)

    D_block = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    trace = float32(0.0)
    for row in range(3):
        row_i = ccio + row
        for col in range(3):
            col_i = ccio + col
            val = problem_D[mio + ncts * row_i + col_i]
            if row == col:
                trace += wp.abs(val)
                val += cfg.regularization
            D_block[row, col] = val

    lambda_old = vec3f(
        solution_lambdas[ccio_v + 0],
        solution_lambdas[ccio_v + 1],
        solution_lambdas[ccio_v + 2],
    )
    lambda_new = lambda_old
    if trace > FLOAT32_EPS:
        lambda_arg = lambda_old - cfg.omega * (wp.inverse(D_block) * v_c)
        lambda_new = project_to_coulomb_cone(lambda_arg, mu_c)

    delta = lambda_new - lambda_old
    solution_lambdas[ccio_v + 0] = lambda_new.x
    solution_lambdas[ccio_v + 1] = lambda_new.y
    solution_lambdas[ccio_v + 2] = lambda_new.z
    state_scratch[ccio_v + 0] = delta.x
    state_scratch[ccio_v + 1] = delta.y
    state_scratch[ccio_v + 2] = delta.z


@wp.kernel
def _apply_dvi_contact_jacobi_delta(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_D: wp.array[float32],
    state_scratch: wp.array[float32],
    # Outputs:
    state_v_aug: wp.array[float32],
):
    wid, row = wp.tid()

    ncts = problem_dim[wid]
    if row >= ncts:
        return

    nc = problem_nc[wid]
    if nc == 0:
        return

    vio = problem_vio[wid]
    mio = problem_mio[wid]
    ccgo = problem_ccgo[wid]
    row_mio = mio + ncts * row

    dv = float32(0.0)
    for cid in range(nc):
        ccio = ccgo + int32(3) * cid
        dv += problem_D[row_mio + ccio + 0] * state_scratch[vio + ccio + 0]
        dv += problem_D[row_mio + ccio + 1] * state_scratch[vio + ccio + 1]
        dv += problem_D[row_mio + ccio + 2] * state_scratch[vio + ccio + 2]

    state_v_aug[vio + row] += dv


@wp.kernel
def _compute_dvi_solution_vectors(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_mio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_D: wp.array[float32],
    problem_v_f: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    mio = problem_mio[wid]
    vio = problem_vio[wid]
    v_i = _compute_row_velocity(ncts, mio, vio, tid, problem_D, problem_v_f, solution_lambdas)
    solution_v_plus[vio + tid] = v_i
    state_v_aug[vio + tid] = v_i
    state_s[vio + tid] = 0.0


@wp.kernel
def _compute_dvi_desaxce_corrections(
    # Inputs:
    problem_nc: wp.array[int32],
    problem_ccgo: wp.array[int32],
    problem_cio: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_mu: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, cid = wp.tid()

    nc = problem_nc[wid]
    if cid >= nc:
        return

    vio = problem_vio[wid]
    ccgo = problem_ccgo[wid]
    ccio = ccgo + 3 * cid
    vt0 = solution_v_plus[vio + ccio]
    vt1 = solution_v_plus[vio + ccio + 1]
    s_n = problem_mu[problem_cio[wid] + cid] * wp.sqrt(vt0 * vt0 + vt1 * vt1)
    state_s[vio + ccio + 2] = s_n
    state_v_aug[vio + ccio + 2] = solution_v_plus[vio + ccio + 2] + s_n


@wp.kernel
def _unprecondition_dvi_solution(
    # Inputs:
    problem_dim: wp.array[int32],
    problem_vio: wp.array[int32],
    problem_P: wp.array[float32],
    # Outputs:
    state_s: wp.array[float32],
    state_v_aug: wp.array[float32],
    solution_lambdas: wp.array[float32],
    solution_v_plus: wp.array[float32],
):
    wid, tid = wp.tid()

    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    vio = problem_vio[wid]
    v_i = vio + tid
    P_i = problem_P[v_i]
    solution_lambdas[v_i] = P_i * solution_lambdas[v_i]
    solution_v_plus[v_i] = solution_v_plus[v_i] / P_i
    state_v_aug[v_i] = state_v_aug[v_i] / P_i
    state_s[v_i] = state_s[v_i] / P_i
