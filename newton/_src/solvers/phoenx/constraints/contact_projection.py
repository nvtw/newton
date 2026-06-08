# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Projected contact row updates shared by rigid and cloth contact paths."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    block_project_accumulated_bounded_1,
    block_project_friction_delta_sor_2,
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
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n > wp.float32(0.0):
                d_lam_n_us = -pd_eff_soft_n * (jv_n - pd_bias_n + pd_gamma_n * lam_n_old)
                normal_projection = block_project_accumulated_bounded_1(
                    d_lam_n_us,
                    lam_n_old,
                    wp.float32(1.0),
                    wp.float32(0.0),
                    sor_boost,
                    wp.float32(0.0),
                    BLOCK_LAMBDA_INF,
                )
            else:
                d_lam_n_us = -eff_n * (jv_n + bias_n)
                normal_projection = block_project_accumulated_bounded_1(
                    d_lam_n_us,
                    lam_n_old,
                    mass_coeff_n,
                    impulse_coeff_n,
                    sor_boost,
                    wp.float32(0.0),
                    BLOCK_LAMBDA_INF,
                )
        else:
            d_lam_n_us = -eff_n * (jv_n + bias_n)
            normal_projection = block_project_accumulated_bounded_1(
                d_lam_n_us,
                lam_n_old,
                mass_coeff_n,
                impulse_coeff_n,
                sor_boost,
                wp.float32(0.0),
                BLOCK_LAMBDA_INF,
            )

        d_lam_n = normal_projection.delta
        lam_n_new = normal_projection.lambda_new

        fric_limit_static = mu_s * lam_n_new
        fric_limit_kinetic = mu_k * lam_n_new
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)
        tangent_projection = block_project_friction_delta_sor_2(
            lam_t1_old,
            lam_t2_old,
            -eff_t1 * (jv_t1 + bias_t1),
            -eff_t2 * (jv_t2 + bias_t2),
            sor_boost,
            fric_limit_static,
            fric_limit_kinetic,
        )

        d_lam_t1 = tangent_projection.delta[0]
        d_lam_t2 = tangent_projection.delta[1]

        cc_set_normal_lambda(cc, k, lam_n_new)
        cc_set_tangent1_lambda(cc, k, tangent_projection.lambda_new[0])
        cc_set_tangent2_lambda(cc, k, tangent_projection.lambda_new[1])

        return d_lam_n * normal + d_lam_t1 * tangent1 + d_lam_t2 * tangent2

    return impl


contact_project_velocity_update = _make_contact_project_velocity_update(has_soft_contact_pd=True)
contact_project_velocity_update_no_soft_pd = _make_contact_project_velocity_update(has_soft_contact_pd=False)
