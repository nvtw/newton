# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Projected contact row updates shared by rigid and cloth contact paths."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    VELOCITY_ROWS3_PROJECT_CONTACT_CONE,
    RigidFrameRows3,
    RigidFrameRows3State,
    RigidFrameRows3Update,
    VelocityRows3Op,
    block_solve_rigid_frame_rows3_contact,
    block_solve_velocity_rows3_op,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal_lambda,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)

__all__ = [
    "contact_frame_velocity_update",
    "contact_frame_velocity_update_no_soft_pd",
    "contact_project_velocity_update",
    "contact_project_velocity_update_no_soft_pd",
]


def _make_contact_project_velocity_update(has_soft_contact_pd: bool):
    @wp.func
    def impl(
        cc: ContactContainer,
        k: wp.int32,
        normal: wp.vec3f,
        tangent1: wp.vec3f,
        tangent2: wp.vec3f,
        jv_n: wp.float32,
        jv_t1: wp.float32,
        jv_t2: wp.float32,
        eff_n: wp.float32,
        eff_t1: wp.float32,
        eff_t2: wp.float32,
        bias_n: wp.float32,
        bias_t1: wp.float32,
        bias_t2: wp.float32,
        mu_s: wp.float32,
        mu_k: wp.float32,
        mass_coeff_n: wp.float32,
        impulse_coeff_n: wp.float32,
        sor_boost: wp.float32,
        pd_eff_soft_n: wp.float32,
        pd_gamma_n: wp.float32,
        pd_bias_n: wp.float32,
    ) -> wp.vec3f:
        """Solve/project one contact's normal + two friction rows.

        The returned vector is the incremental impulse applied on side 2.
        """
        lam_n_old = cc_get_normal_lambda(cc, k)
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)

        k_inv_n = eff_n
        rhs_n = jv_n + bias_n
        normal_mass_coeff = mass_coeff_n
        normal_impulse_coeff = impulse_coeff_n
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n > wp.float32(0.0):
                k_inv_n = pd_eff_soft_n
                rhs_n = jv_n - pd_bias_n + pd_gamma_n * lam_n_old
                normal_mass_coeff = wp.float32(1.0)
                normal_impulse_coeff = wp.float32(0.0)

        op = VelocityRows3Op()
        op.k_inv = wp.vec3f(k_inv_n, eff_t1, eff_t2)
        op.residual = wp.vec3f(rhs_n, jv_t1 + bias_t1, jv_t2 + bias_t2)
        op.lambda_old = wp.vec3f(lam_n_old, lam_t1_old, lam_t2_old)
        op.mass_coeff = wp.vec3f(normal_mass_coeff, wp.float32(1.0), wp.float32(1.0))
        op.impulse_coeff = wp.vec3f(normal_impulse_coeff, wp.float32(0.0), wp.float32(0.0))
        op.lambda_min = wp.vec3f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        op.lambda_max = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        op.projection_mode = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
        op.friction_static = mu_s
        op.friction_kinetic = mu_k

        projection = block_solve_velocity_rows3_op(op, sor_boost)

        cc_set_normal_lambda(cc, k, projection.lambda_new[0])
        cc_set_tangent1_lambda(cc, k, projection.lambda_new[1])
        cc_set_tangent2_lambda(cc, k, projection.lambda_new[2])

        return projection.delta[0] * normal + projection.delta[1] * tangent1 + projection.delta[2] * tangent2

    return impl


def _make_contact_frame_velocity_update(has_soft_contact_pd: bool):
    @wp.func
    def impl(
        cc: ContactContainer,
        k: wp.int32,
        normal: wp.vec3f,
        tangent1: wp.vec3f,
        tangent2: wp.vec3f,
        r0: wp.vec3f,
        r1: wp.vec3f,
        v0: wp.vec3f,
        w0: wp.vec3f,
        v1: wp.vec3f,
        w1: wp.vec3f,
        inv_mass0: wp.float32,
        inv_mass1: wp.float32,
        inv_inertia0: wp.mat33f,
        inv_inertia1: wp.mat33f,
        eff_n: wp.float32,
        eff_t1: wp.float32,
        eff_t2: wp.float32,
        bias_n: wp.float32,
        bias_t1: wp.float32,
        bias_t2: wp.float32,
        mu_s: wp.float32,
        mu_k: wp.float32,
        mass_coeff_n: wp.float32,
        impulse_coeff_n: wp.float32,
        sor_boost: wp.float32,
        pd_eff_soft_n: wp.float32,
        pd_gamma_n: wp.float32,
        pd_bias_n: wp.float32,
    ) -> RigidFrameRows3Update:
        """Solve/project/apply one rigid contact with compact frame rows."""
        lam_n_old = cc_get_normal_lambda(cc, k)
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)

        k_inv_n = eff_n
        normal_bias = bias_n
        normal_mass_coeff = mass_coeff_n
        normal_impulse_coeff = impulse_coeff_n
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n > wp.float32(0.0):
                k_inv_n = pd_eff_soft_n
                normal_bias = -pd_bias_n + pd_gamma_n * lam_n_old
                normal_mass_coeff = wp.float32(1.0)
                normal_impulse_coeff = wp.float32(0.0)

        rows = RigidFrameRows3()
        rows.axis0 = normal
        rows.axis1 = tangent1
        rows.axis2 = tangent2
        rows.r0 = r0
        rows.r1 = r1
        rows.mode = wp.vec3f(wp.float32(1.0), wp.float32(1.0), wp.float32(0.0))
        rows.k_inv = wp.vec3f(k_inv_n, eff_t1, eff_t2)
        rows.bias = wp.vec3f(normal_bias, bias_t1, bias_t2)
        rows.lambda_old = wp.vec3f(lam_n_old, lam_t1_old, lam_t2_old)
        rows.mass_coeff = wp.vec3f(normal_mass_coeff, wp.float32(1.0), wp.float32(1.0))
        rows.impulse_coeff = wp.vec3f(normal_impulse_coeff, wp.float32(0.0), wp.float32(0.0))
        rows.lambda_min = wp.vec3f(wp.float32(0.0), -BLOCK_LAMBDA_INF, -BLOCK_LAMBDA_INF)
        rows.lambda_max = wp.vec3f(BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF, BLOCK_LAMBDA_INF)
        rows.projection_mode = VELOCITY_ROWS3_PROJECT_CONTACT_CONE
        rows.friction_static = mu_s
        rows.friction_kinetic = mu_k

        state = RigidFrameRows3State()
        state.v_a = v0
        state.w_a = w0
        state.v_b = v1
        state.w_b = w1
        state.inv_m_a = inv_mass0
        state.inv_m_b = inv_mass1
        state.inv_i_a = inv_inertia0
        state.inv_i_b = inv_inertia1

        update = block_solve_rigid_frame_rows3_contact(rows, state, sor_boost)
        cc_set_normal_lambda(cc, k, update.lambda_new[0])
        cc_set_tangent1_lambda(cc, k, update.lambda_new[1])
        cc_set_tangent2_lambda(cc, k, update.lambda_new[2])
        return update

    return impl


contact_frame_velocity_update = _make_contact_frame_velocity_update(has_soft_contact_pd=True)
contact_frame_velocity_update_no_soft_pd = _make_contact_frame_velocity_update(has_soft_contact_pd=False)
contact_project_velocity_update = _make_contact_project_velocity_update(has_soft_contact_pd=True)
contact_project_velocity_update_no_soft_pd = _make_contact_project_velocity_update(has_soft_contact_pd=False)
