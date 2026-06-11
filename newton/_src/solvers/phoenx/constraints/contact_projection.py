# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Projected contact row updates shared by rigid and cloth contact paths."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    RigidFrameRows3Update,
    block_project_friction_delta_sor_2,
    block_solve_accumulated_inverse_bounded_1,
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


@wp.func
def _friction_normal_lambda(
    lambda_n: wp.float32,
    eff_n: wp.float32,
    bias_n: wp.float32,
    mass_coeff_n: wp.float32,
    sor_boost: wp.float32,
) -> wp.float32:
    """Normal load for Coulomb friction, excluding Baumgarte correction."""
    load = lambda_n + mass_coeff_n * eff_n * bias_n * sor_boost
    return wp.clamp(load, wp.float32(0.0), lambda_n)


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

        normal_update = block_solve_accumulated_inverse_bounded_1(
            k_inv_n,
            rhs_n,
            lam_n_old,
            normal_mass_coeff,
            normal_impulse_coeff,
            sor_boost,
            wp.float32(0.0),
            BLOCK_LAMBDA_INF,
        )
        lambda_n_friction = normal_update.lambda_new
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n <= wp.float32(0.0):
                lambda_n_friction = _friction_normal_lambda(
                    normal_update.lambda_new,
                    k_inv_n,
                    bias_n,
                    normal_mass_coeff,
                    sor_boost,
                )
        else:
            lambda_n_friction = _friction_normal_lambda(
                normal_update.lambda_new,
                k_inv_n,
                bias_n,
                normal_mass_coeff,
                sor_boost,
            )

        d_lambda_t1 = -(eff_t1 * (jv_t1 + bias_t1))
        d_lambda_t2 = -(eff_t2 * (jv_t2 + bias_t2))
        tangents = block_project_friction_delta_sor_2(
            lam_t1_old,
            lam_t2_old,
            d_lambda_t1,
            d_lambda_t2,
            sor_boost,
            mu_s * lambda_n_friction,
            mu_k * lambda_n_friction,
        )

        cc_set_normal_lambda(cc, k, normal_update.lambda_new)
        cc_set_tangent1_lambda(cc, k, tangents.lambda_new[0])
        cc_set_tangent2_lambda(cc, k, tangents.lambda_new[1])

        return normal_update.delta * normal + tangents.delta[0] * tangent1 + tangents.delta[1] * tangent2

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

        rel = v1 - v0 + wp.cross(w1, r1) - wp.cross(w0, r0)
        jv_n = wp.dot(normal, rel)
        jv_t1 = wp.dot(tangent1, rel)
        jv_t2 = wp.dot(tangent2, rel)

        normal_update = block_solve_accumulated_inverse_bounded_1(
            k_inv_n,
            jv_n + normal_bias,
            lam_n_old,
            normal_mass_coeff,
            normal_impulse_coeff,
            sor_boost,
            wp.float32(0.0),
            BLOCK_LAMBDA_INF,
        )
        lambda_n_friction = normal_update.lambda_new
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n <= wp.float32(0.0):
                lambda_n_friction = _friction_normal_lambda(
                    normal_update.lambda_new,
                    k_inv_n,
                    bias_n,
                    normal_mass_coeff,
                    sor_boost,
                )
        else:
            lambda_n_friction = _friction_normal_lambda(
                normal_update.lambda_new,
                k_inv_n,
                bias_n,
                normal_mass_coeff,
                sor_boost,
            )

        d_lambda_t1 = -(eff_t1 * (jv_t1 + bias_t1))
        d_lambda_t2 = -(eff_t2 * (jv_t2 + bias_t2))
        tangents = block_project_friction_delta_sor_2(
            lam_t1_old,
            lam_t2_old,
            d_lambda_t1,
            d_lambda_t2,
            sor_boost,
            mu_s * lambda_n_friction,
            mu_k * lambda_n_friction,
        )

        impulse = normal_update.delta * normal + tangents.delta[0] * tangent1 + tangents.delta[1] * tangent2
        update = RigidFrameRows3Update()
        update.v_a = v0 - inv_mass0 * impulse
        update.v_b = v1 + inv_mass1 * impulse
        update.w_a = w0 - inv_inertia0 @ wp.cross(r0, impulse)
        update.w_b = w1 + inv_inertia1 @ wp.cross(r1, impulse)
        update.lambda_new = wp.vec3f(normal_update.lambda_new, tangents.lambda_new[0], tangents.lambda_new[1])
        update.delta = wp.vec3f(normal_update.delta, tangents.delta[0], tangents.delta[1])

        cc_set_normal_lambda(cc, k, normal_update.lambda_new)
        cc_set_tangent1_lambda(cc, k, tangents.lambda_new[0])
        cc_set_tangent2_lambda(cc, k, tangents.lambda_new[1])
        return update

    return impl


contact_frame_velocity_update = _make_contact_frame_velocity_update(has_soft_contact_pd=True)
contact_frame_velocity_update_no_soft_pd = _make_contact_frame_velocity_update(has_soft_contact_pd=False)
contact_project_velocity_update = _make_contact_project_velocity_update(has_soft_contact_pd=True)
contact_project_velocity_update_no_soft_pd = _make_contact_project_velocity_update(has_soft_contact_pd=False)
