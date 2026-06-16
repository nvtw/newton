# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ...core.math import FLOAT32_EPS
from ...core.types import float32, int32, vec3f
from ..padmm.math import project_to_coulomb_cone
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
                D_ii = wp.abs(problem_D[mio + ncts * i + i]) + cfg.regularization + FLOAT32_EPS
                lambda_limit_old = solution_lambdas[vio + i]
                lambda_limit_new = wp.max(0.0, lambda_limit_old - cfg.omega * v_i / D_ii)
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
                D_kk = wp.max(vec3f(D_00, D_11, D_22)) + cfg.regularization + FLOAT32_EPS
                lambda_contact_old = vec3f(
                    solution_lambdas[vio + ccio + 0],
                    solution_lambdas[vio + ccio + 1],
                    solution_lambdas[vio + ccio + 2],
                )
                lambda_contact_arg = lambda_contact_old - (cfg.omega / D_kk) * v_c
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
