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
    "contact_project_coupled_velocity_update_no_soft_pd",
    "contact_project_friction_metric",
    "contact_project_normal_velocity_update",
    "contact_project_normal_velocity_update_no_soft_pd",
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


def _make_contact_project_normal_velocity_update(has_soft_contact_pd: bool):
    @wp.func
    def impl(
        cc: ContactContainer,
        k: wp.int32,
        normal: wp.vec3f,
        jv_n: wp.float32,
        eff_n: wp.float32,
        bias_n: wp.float32,
        mass_coeff_n: wp.float32,
        impulse_coeff_n: wp.float32,
        sor_boost: wp.float32,
        pd_eff_soft_n: wp.float32,
        pd_gamma_n: wp.float32,
        pd_bias_n: wp.float32,
    ) -> wp.vec3f:
        """Solve one normal row without evaluating unused tangent rows."""
        lambda_old = cc_get_normal_lambda(cc, k)
        inverse_response = eff_n
        rhs = jv_n + bias_n
        normal_mass_coeff = mass_coeff_n
        normal_impulse_coeff = impulse_coeff_n
        if wp.static(has_soft_contact_pd):
            if pd_eff_soft_n > wp.float32(0.0):
                inverse_response = pd_eff_soft_n
                rhs = jv_n - pd_bias_n + pd_gamma_n * lambda_old
                normal_mass_coeff = wp.float32(1.0)
                normal_impulse_coeff = wp.float32(0.0)

        update = block_solve_accumulated_inverse_bounded_1(
            inverse_response,
            rhs,
            lambda_old,
            normal_mass_coeff,
            normal_impulse_coeff,
            sor_boost,
            wp.float32(0.0),
            BLOCK_LAMBDA_INF,
        )
        cc_set_normal_lambda(cc, k, update.lambda_new)
        return update.delta * normal

    return impl


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


def _make_contact_project_friction_metric(scalar_type):
    relative_tolerance = 2.0e-7 if scalar_type == wp.float32 else 1.0e-7

    @wp.func
    def impl(
        mobility_t1: wp.float32,
        mobility_t1t2: wp.float32,
        mobility_t2: wp.float32,
        rhs_t1: wp.float32,
        rhs_t2: wp.float32,
        lambda_t1_old: wp.float32,
        lambda_t2_old: wp.float32,
        static_radius: wp.float32,
        dynamic_radius: wp.float32,
    ) -> wp.vec3f:
        """Minimize tangent kinetic energy within the Coulomb disk.

        Sliding friction must oppose the resulting slip velocity. For an
        anisotropic mobility, radial clamping of the unconstrained impulse
        does not satisfy this condition. The disk boundary minimizer solves
        (K + alpha I) lambda = K lambda_old - velocity, with alpha >= 0.
        """
        scale = wp.max(mobility_t1, mobility_t2)
        result = wp.vec2f(0.0)
        root_iterations = wp.int32(0)
        if scale > wp.float32(0.0) and static_radius > wp.float32(0.0):
            a = scalar_type(mobility_t1) / scalar_type(scale)
            c = scalar_type(mobility_t2) / scalar_type(scale)
            b = scalar_type(mobility_t1t2) / scalar_type(scale)
            v1 = scalar_type(rhs_t1) / scalar_type(scale)
            v2 = scalar_type(rhs_t2) / scalar_type(scale)
            target1 = a * scalar_type(lambda_t1_old) + b * scalar_type(lambda_t2_old) - v1
            target2 = b * scalar_type(lambda_t1_old) + c * scalar_type(lambda_t2_old) - v2
            determinant = scalar_type(a) * scalar_type(c) - scalar_type(b) * scalar_type(b)
            if determinant > scalar_type(0.0):
                result = wp.vec2f(
                    wp.float32(
                        (scalar_type(c) * scalar_type(target1) - scalar_type(b) * scalar_type(target2)) / determinant
                    ),
                    wp.float32(
                        (scalar_type(a) * scalar_type(target2) - scalar_type(b) * scalar_type(target1)) / determinant
                    ),
                )
            else:
                # A rank-one mobility has one movable tangent direction.
                inverse_trace_squared = scalar_type(1.0) / ((a + c) * (a + c))
                result = wp.vec2f(
                    wp.float32((a * target1 + b * target2) * inverse_trace_squared),
                    wp.float32((b * target1 + c * target2) * inverse_trace_squared),
                )
            if wp.length_sq(result) > static_radius * static_radius:
                result = wp.vec2f(0.0)
                if dynamic_radius > wp.float32(0.0):
                    lower = scalar_type(0.0)
                    radius = scalar_type(dynamic_radius)
                    target_x = scalar_type(target1)
                    target_y = scalar_type(target2)
                    upper = wp.sqrt(target_x * target_x + target_y * target_y) / radius
                    alpha = upper
                    old_x = scalar_type(lambda_t1_old)
                    old_y = scalar_type(lambda_t2_old)
                    old_length_sq = old_x * old_x + old_y * old_y
                    if old_length_sq > scalar_type(0.0):
                        old_length = wp.sqrt(old_length_sq)
                        old_k_old = old_x * (a * old_x + b * old_y) + old_y * (b * old_x + c * old_y)
                        estimate = (old_x * target_x + old_y * target_y) / (
                            radius * old_length
                        ) - old_k_old / old_length_sq
                        if estimate > lower and estimate < upper:
                            alpha = estimate
                    # The norm bound supplies a feasible upper bracket. Newton
                    # usually converges in a few iterations; bisection safeguards
                    # nearly singular blocks without changing their mobility.
                    for _iteration in range(40):
                        root_iterations += wp.int32(1)
                        aa = scalar_type(a) + alpha
                        cc = scalar_type(c) + alpha
                        bb = scalar_type(b)
                        det = aa * cc - bb * bb
                        x = (cc * target_x - bb * target_y) / det
                        y = (aa * target_y - bb * target_x) / det
                        length = wp.sqrt(x * x + y * y)
                        error = length - radius
                        # Newton approaches this convex decreasing norm from
                        # below its root after the first step. Accept convergence
                        # from either side; otherwise positive roundoff can force
                        # all forty iterations despite an accurate solution.
                        if wp.abs(error) <= scalar_type(relative_tolerance) * radius:
                            boundary_scale = radius / length
                            result = wp.vec2f(wp.float32(x * boundary_scale), wp.float32(y * boundary_scale))
                            break
                        if error > scalar_type(0.0):
                            lower = alpha
                        else:
                            upper = alpha
                            result = wp.vec2f(wp.float32(x), wp.float32(y))
                        derivative = -(cc * x * x - scalar_type(2.0) * bb * x * y + aa * y * y) / (det * length)
                        proposal = alpha - error / derivative
                        if proposal <= lower or proposal >= upper:
                            proposal = scalar_type(0.5) * (lower + upper)
                        alpha = proposal
        return wp.vec3f(result[0], result[1], wp.float32(root_iterations))

    return impl


_contact_project_friction_metric_float32 = _make_contact_project_friction_metric(wp.float32)
_contact_project_friction_metric_float64 = _make_contact_project_friction_metric(wp.float64)


@wp.func
def contact_project_friction_metric(
    mobility_t1: wp.float32,
    mobility_t1t2: wp.float32,
    mobility_t2: wp.float32,
    rhs_t1: wp.float32,
    rhs_t2: wp.float32,
    lambda_t1_old: wp.float32,
    lambda_t2_old: wp.float32,
    static_radius: wp.float32,
    dynamic_radius: wp.float32,
) -> wp.vec3f:
    """Use float64 only when the tangent mobility is ill-conditioned."""
    scale = wp.max(mobility_t1, mobility_t2)
    well_conditioned = wp.bool(False)
    if scale > wp.float32(0.0):
        a = mobility_t1 / scale
        b = mobility_t1t2 / scale
        c = mobility_t2 / scale
        # Bound the determinant cancellation before choosing float32.
        # Tiny positive eigenvalues retain the precise physical solve.
        well_conditioned = a * c - b * b > wp.float32(0.01) * (a + c) * (a + c)
    if well_conditioned:
        return _contact_project_friction_metric_float32(
            mobility_t1,
            mobility_t1t2,
            mobility_t2,
            rhs_t1,
            rhs_t2,
            lambda_t1_old,
            lambda_t2_old,
            static_radius,
            dynamic_radius,
        )
    return _contact_project_friction_metric_float64(
        mobility_t1,
        mobility_t1t2,
        mobility_t2,
        rhs_t1,
        rhs_t2,
        lambda_t1_old,
        lambda_t2_old,
        static_radius,
        dynamic_radius,
    )


def _make_contact_project_coupled_velocity_update(has_soft_contact_pd: bool):
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
        mobility_nt1: wp.float32,
        mobility_nt2: wp.float32,
        mobility_t1t2: wp.float32,
    ) -> wp.vec3f:
        """Solve a normal row, then its coupled tangent block.

        Cross mobilities account for the velocity change from the normal
        impulse before Coulomb friction acts. The returned world-space
        impulse acts on side 2.
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

        # The normal impulse changes tangential velocity before friction acts.
        rhs_t1 = jv_t1 + bias_t1 + mobility_nt1 * normal_update.delta
        rhs_t2 = jv_t2 + bias_t2 + mobility_nt2 * normal_update.delta
        mobility_t1 = wp.float32(1.0) / eff_t1 if eff_t1 > wp.float32(0.0) else wp.float32(0.0)
        mobility_t2 = wp.float32(1.0) / eff_t2 if eff_t2 > wp.float32(0.0) else wp.float32(0.0)
        friction = contact_project_friction_metric(
            mobility_t1,
            mobility_t1t2,
            mobility_t2,
            rhs_t1,
            rhs_t2,
            lam_t1_old,
            lam_t2_old,
            mu_s * lambda_n_friction,
            mu_k * lambda_n_friction,
        )
        wp.atomic_add(cc.derived, cc.derived.shape[0] - 1, wp.int32(friction[2]), wp.float32(1.0))
        d_lambda_t1 = friction[0] - lam_t1_old
        d_lambda_t2 = friction[1] - lam_t2_old
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


contact_project_normal_velocity_update = _make_contact_project_normal_velocity_update(has_soft_contact_pd=True)
contact_project_normal_velocity_update_no_soft_pd = _make_contact_project_normal_velocity_update(
    has_soft_contact_pd=False
)
contact_frame_velocity_update = _make_contact_frame_velocity_update(has_soft_contact_pd=True)
contact_frame_velocity_update_no_soft_pd = _make_contact_frame_velocity_update(has_soft_contact_pd=False)
contact_project_velocity_update = _make_contact_project_velocity_update(has_soft_contact_pd=True)
contact_project_velocity_update_no_soft_pd = _make_contact_project_velocity_update(has_soft_contact_pd=False)

contact_project_coupled_velocity_update_no_soft_pd = _make_contact_project_coupled_velocity_update(
    has_soft_contact_pd=False
)

contact_project_coupled_velocity_update = _make_contact_project_coupled_velocity_update(has_soft_contact_pd=True)
