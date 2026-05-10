# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth-aware contact prepare + iterate.

The Box2D-v3 soft PGS row math is identical to the rigid-only contact
solver in :mod:`constraint_contact`; the only difference is *how* each
side's velocity-at-contact-point and effective-mass terms are loaded,
and *how* the impulse is scattered. Both pieces are abstracted via
the unified endpoint helpers in :mod:`contact_endpoint`.

Result: rigid-rigid through this path produces byte-equivalent
impulses to :func:`contact_iterate_at` (algebraically verified by
:mod:`tests.test_contact_endpoint_helpers`); cloth-rigid + cloth-cloth
go through the same row math with barycentric-weighted state reads
and impulse scatters.

Phase 5 dispatches the cloth-supporting kernel variants
(``cloth_support=True``) to these wp.funcs; rigid-only kernels
(``cloth_support=False``) keep the existing
:func:`contact_iterate_at` / :func:`contact_prepare_for_iteration_at`
binaries byte-identical.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.cloth_collision import SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_side0_kind,
    contact_get_side0_nodes_extra,
    contact_get_side1_kind,
    contact_get_side1_nodes_extra,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    pd_coefficients,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
    cc_get_side0_bary,
    cc_get_side1_bary,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_normal_lambda,
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.constraints.constraint_container import constraint_bodies_make
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    contact_get_body1,
    contact_get_body2,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_apply_impulse,
    contact_endpoint_inv_mass_along,
    contact_endpoint_set_access_mode,
    contact_endpoint_velocity_at_point,
)
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_config import PHOENX_BOOST_CONTACT_NORMAL

__all__ = [
    "contact_iterate_at_cloth_aware",
    "contact_iterate_cloth_aware",
    "contact_prepare_for_iteration_at_cloth_aware",
    "contact_prepare_for_iteration_cloth_aware",
]


@wp.func
def _side_world_contact_point(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    cc: ContactContainer,
    k: wp.int32,
    is_side1: wp.bool,
    margin: wp.float32,
    n: wp.vec3f,
) -> wp.vec3f:
    """World-space contact point on this side, including the per-shape
    surface margin shift along the normal.

    Mirrors the projection :func:`contact_prepare_for_iteration_at` /
    :func:`contact_iterate_at` use for the rigid case
    (``position + quat_rotate(orient, local_p - body_com) + margin * n``)
    and adds a barycentric branch for the cloth case
    (``bary . particle_positions + margin * n``).

    Sign of the margin shift follows the existing rigid convention:

    * side 0: ``+margin * n`` (push toward side 1 along ``n``)
    * side 1: ``-margin * n`` (push toward side 0)
    """
    sign = wp.float32(-1.0) if is_side1 else wp.float32(1.0)
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        anchor = (
            bary[0] * particles.position[p_a]
            + bary[1] * particles.position[p_b]
            + bary[2] * particles.position[p_c]
        )
        return anchor + (sign * margin) * n
    b = nodes[0]
    if b < 0:
        # Static-anchor world shape: the local_p stored in CC is in
        # the "world body" frame == identity, so the shape transform
        # is identity. Read local_p and add the margin shift.
        local_p = cc_get_local_p1(cc, k) if is_side1 else cc_get_local_p0(cc, k)
        return local_p + (sign * margin) * n
    body_com = bodies.body_com[b]
    orient = bodies.orientation[b]
    pos = bodies.position[b]
    local_p = cc_get_local_p1(cc, k) if is_side1 else cc_get_local_p0(cc, k)
    return pos + wp.quat_rotate(orient, local_p - body_com) + (sign * margin) * n


@wp.func
def contact_prepare_for_iteration_at_cloth_aware(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    """Cloth-aware contact prepare: same Box2D-v3 row plumbing as
    :func:`contact_prepare_for_iteration_at`, but state reads + warm-
    start scatter dispatch via the endpoint helpers so cloth tris
    work transparently alongside rigid bodies.

    See :mod:`constraint_contact_cloth` module docstring for the
    rationale.
    """
    _ = base_offset

    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    side0_nodes = wp.vec3i(body_pair.b1, side0_extra[0], side0_extra[1])
    side1_nodes = wp.vec3i(body_pair.b2, side1_extra[0], side1_extra[1])

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    dt_substep = wp.float32(1.0) / idt
    bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    friction_bias_factor = wp.float32(0.08)
    friction_slop = wp.float32(0.001)
    max_push_speed = wp.float32(2.0)
    max_approach_speed = wp.float32(10.0)

    mu_s_col = contact_get_friction(constraints, cid)

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        bary0 = cc_get_side0_bary(cc, k)
        bary1 = cc_get_side1_bary(cc, k)
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]

        p0_world = _side_world_contact_point(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies,
            cc, k, False, margin0, n,
        )
        p1_world = _side_world_contact_point(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies,
            cc, k, True, margin1, n,
        )

        # Effective mass per row (n / t1 / t2): sum of per-side
        # contributions. Reduces to ``effective_mass_scalar``'s
        # denominator in the rigid-rigid case (verified algebraically
        # in tests/test_contact_endpoint_helpers.py).
        inv_n = (
            contact_endpoint_inv_mass_along(side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world, n)
            + contact_endpoint_inv_mass_along(side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world, n)
        )
        inv_t1 = (
            contact_endpoint_inv_mass_along(side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world, t1_dir)
            + contact_endpoint_inv_mass_along(side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world, t1_dir)
        )
        inv_t2 = (
            contact_endpoint_inv_mass_along(side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world, t2_dir)
            + contact_endpoint_inv_mass_along(side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world, t2_dir)
        )
        eff_n = wp.float32(0.0)
        if inv_n > wp.float32(1.0e-12):
            eff_n = wp.float32(1.0) / inv_n
        eff_t1 = wp.float32(0.0)
        if inv_t1 > wp.float32(1.0e-12):
            eff_t1 = wp.float32(1.0) / inv_t1
        eff_t2 = wp.float32(0.0)
        if inv_t2 > wp.float32(1.0e-12):
            eff_t2 = wp.float32(1.0) / inv_t2

        # Speculative-vs-penetrating Baumgarte bias on the normal row.
        # Sign convention: ``effective_gap > 0`` means separated
        # (no penetration). Same as the rigid path.
        effective_gap = wp.dot(p1_world - p0_world, n)

        # Load-scaled correction (warm-started normal load).
        lam_n_ws = cc_get_normal_lambda(cc, k)
        lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
        load_boost = wp.min(wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0))

        if effective_gap > wp.float32(0.0):
            bias_val = effective_gap * idt
        else:
            bias_val = effective_gap * bias_rate
        bias_val = wp.clamp(bias_val, -max_push_speed, max_approach_speed)

        # Friction-row drift biases (sticky-friction). Cloth
        # contacts get the same drift logic as rigid -- the
        # barycentric weights make the cloth side track tangential
        # particle motion automatically.
        p_diff = p1_world - p0_world
        drift_t1_raw = wp.dot(p_diff, t1_dir)
        drift_t2_raw = wp.dot(p_diff, t2_dir)
        drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
        drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
        bias_t1_val = friction_bias_factor * drift_t1 * idt * load_boost
        bias_t2_val = friction_bias_factor * drift_t2 * idt * load_boost

        cc_set_eff_n(cc, k, eff_n)
        cc_set_eff_t1(cc, k, eff_t1)
        cc_set_eff_t2(cc, k, eff_t2)
        cc_set_bias(cc, k, bias_val)
        cc_set_bias_t1(cc, k, bias_t1_val)
        cc_set_bias_t2(cc, k, bias_t2_val)

        # Soft-contact PD plumbing (per-contact stiffness/damping
        # arrays). Identical to the rigid path.
        stiffness_arr_len = contacts.rigid_contact_stiffness.shape[0]
        damping_arr_len = contacts.rigid_contact_damping.shape[0]
        if stiffness_arr_len > k or damping_arr_len > k:
            k_n = wp.float32(0.0)
            c_n = wp.float32(0.0)
            if stiffness_arr_len > k:
                k_n = contacts.rigid_contact_stiffness[k]
            if damping_arr_len > k:
                c_n = contacts.rigid_contact_damping[k]
            if (k_n > wp.float32(0.0) or c_n > wp.float32(0.0)) and eff_n > wp.float32(0.0):
                eff_inv_n = wp.float32(1.0) / eff_n
                pd_gamma_n, pd_bias_n, pd_eff_soft_n = pd_coefficients(
                    k_n, c_n, -effective_gap, eff_inv_n, dt_substep, PHOENX_BOOST_CONTACT_NORMAL
                )
                cc_set_pd_gamma(cc, k, pd_gamma_n)
                cc_set_pd_bias(cc, k, pd_bias_n)
                cc_set_pd_eff_soft(cc, k, pd_eff_soft_n)
            else:
                cc_set_pd_eff_soft(cc, k, wp.float32(0.0))
        else:
            cc_set_pd_eff_soft(cc, k, wp.float32(0.0))

        # Warm-start scatter via endpoint helpers. Side 0 gets ``-imp``,
        # side 1 gets ``+imp``; the helpers pick the right path
        # (bodies vs particles) and apply mass-weighted impulses.
        lam_n = cc_get_normal_lambda(cc, k)
        lam_t1 = cc_get_tangent1_lambda(cc, k)
        lam_t2 = cc_get_tangent2_lambda(cc, k)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        contact_endpoint_apply_impulse(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world, -imp,
        )
        contact_endpoint_apply_impulse(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world, imp,
        )


@wp.func
def contact_iterate_at_cloth_aware(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
):
    """Cloth-aware contact iterate: same Box2D-v3 PGS row math as
    :func:`contact_iterate_at`, dispatched per side via the endpoint
    helpers.
    """
    _ = base_offset

    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    side0_nodes = wp.vec3i(body_pair.b1, side0_extra[0], side0_extra[1])
    side1_nodes = wp.vec3i(body_pair.b2, side1_extra[0], side1_extra[1])

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    dt_substep = wp.float32(1.0) / idt
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        bary0 = cc_get_side0_bary(cc, k)
        bary1 = cc_get_side1_bary(cc, k)
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]

        p0_world = _side_world_contact_point(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies,
            cc, k, False, margin0, n,
        )
        p1_world = _side_world_contact_point(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies,
            cc, k, True, margin1, n,
        )

        eff_n = cc_get_eff_n(cc, k)
        eff_t1 = cc_get_eff_t1(cc, k)
        eff_t2 = cc_get_eff_t2(cc, k)
        bias_val = cc_get_bias(cc, k)
        bias_t1_val = cc_get_bias_t1(cc, k)
        bias_t2_val = cc_get_bias_t2(cc, k)
        is_speculative = bias_val > wp.float32(0.0)
        if not use_bias:
            if not is_speculative:
                bias_val = wp.float32(0.0)
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)

        # Per-side velocity at the contact point (rigid: v + omega x r;
        # cloth: bary . v). Project relative velocity onto n / t1 / t2.
        v0_at_p = contact_endpoint_velocity_at_point(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world,
        )
        v1_at_p = contact_endpoint_velocity_at_point(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world,
        )
        vel_rel = v1_at_p - v0_at_p
        jv_n = wp.dot(vel_rel, n)
        jv_t1 = wp.dot(vel_rel, t1_dir)
        jv_t2 = wp.dot(vel_rel, t2_dir)

        # Normal row (Box2D-v3 soft / rigid speculative / soft-contact PD).
        pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
        lam_n_old = cc_get_normal_lambda(cc, k)
        if pd_eff_soft_n > wp.float32(0.0):
            pd_gamma_n = cc_get_pd_gamma(cc, k)
            pd_bias_n = cc_get_pd_bias(cc, k)
            d_lam_n_us = -pd_eff_soft_n * (jv_n - pd_bias_n + pd_gamma_n * lam_n_old)
            lam_n_new = wp.max(lam_n_old + d_lam_n_us, wp.float32(0.0))
            d_lam_n = lam_n_new - lam_n_old
        else:
            if is_speculative:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
            elif use_bias:
                mass_coeff_n = mass_coeff
                impulse_coeff_n = impulse_coeff
            else:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
            d_lam_n_us = -eff_n * (jv_n + bias_val)
            d_lam_n = mass_coeff_n * d_lam_n_us - impulse_coeff_n * lam_n_old
            lam_n_new = wp.max(lam_n_old + d_lam_n, wp.float32(0.0))
            d_lam_n = lam_n_new - lam_n_old

        fric_limit_static = mu_s * lam_n_new
        fric_limit_kinetic = mu_k * lam_n_new

        d_lam_t1 = -eff_t1 * (jv_t1 + bias_t1_val)
        d_lam_t2 = -eff_t2 * (jv_t2 + bias_t2_val)
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)
        lam_t1_raw = lam_t1_old + d_lam_t1
        lam_t2_raw = lam_t2_old + d_lam_t2
        lam_t_sq = lam_t1_raw * lam_t1_raw + lam_t2_raw * lam_t2_raw
        static_limit_sq = fric_limit_static * fric_limit_static
        if lam_t_sq > static_limit_sq and lam_t_sq > wp.float32(1.0e-30):
            inv_mag = fric_limit_kinetic / wp.sqrt(lam_t_sq)
            lam_t1_new = lam_t1_raw * inv_mag
            lam_t2_new = lam_t2_raw * inv_mag
        else:
            lam_t1_new = lam_t1_raw
            lam_t2_new = lam_t2_raw
        d_lam_t1 = lam_t1_new - lam_t1_old
        d_lam_t2 = lam_t2_new - lam_t2_old

        cc_set_normal_lambda(cc, k, lam_n_new)
        cc_set_tangent1_lambda(cc, k, lam_t1_new)
        cc_set_tangent2_lambda(cc, k, lam_t2_new)

        # Scatter the row impulse to both sides via the endpoint helper.
        # Side 0 gets ``-imp``, side 1 gets ``+imp``; the helper picks
        # rigid (v += J*invM, w += invI . (r x J)) or cloth
        # (v_i += bary_i * J * invM_i) per kind.
        imp = d_lam_n * n + d_lam_t1 * t1_dir + d_lam_t2 * t2_dir
        contact_endpoint_apply_impulse(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies, p0_world, -imp,
        )
        contact_endpoint_apply_impulse(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies, p1_world, imp,
        )


# ---------------------------------------------------------------------------
# Public entry points -- mirrors :func:`contact_prepare_for_iteration` /
# :func:`contact_iterate` but forwards to the cloth-aware ``*_at_cloth_aware``
# variants. Reads body1/body2 + side*_nodes_extra from the contact column,
# flips access modes per side, then dispatches.
# ---------------------------------------------------------------------------


@wp.func
def contact_prepare_for_iteration_cloth_aware(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    # Velocity-level constraint: flip every node on each side to
    # VELOCITY_LEVEL so any prior position-level write (cloth iterate)
    # is finite-diffed into velocity before we read it.
    contact_endpoint_set_access_mode(
        side0_kind, wp.vec3i(b1, side0_extra[0], side0_extra[1]),
        bodies, particles, num_bodies, ACCESS_MODE_VELOCITY_LEVEL, idt,
    )
    contact_endpoint_set_access_mode(
        side1_kind, wp.vec3i(b2, side1_extra[0], side1_extra[1]),
        bodies, particles, num_bodies, ACCESS_MODE_VELOCITY_LEVEL, idt,
    )
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at_cloth_aware(
        constraints, cid, 0, bodies, particles, num_bodies, body_pair, idt, cc, contacts,
    )


@wp.func
def contact_iterate_cloth_aware(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    contact_endpoint_set_access_mode(
        side0_kind, wp.vec3i(b1, side0_extra[0], side0_extra[1]),
        bodies, particles, num_bodies, ACCESS_MODE_VELOCITY_LEVEL, idt,
    )
    contact_endpoint_set_access_mode(
        side1_kind, wp.vec3i(b2, side1_extra[0], side1_extra[1]),
        bodies, particles, num_bodies, ACCESS_MODE_VELOCITY_LEVEL, idt,
    )
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at_cloth_aware(
        constraints, cid, 0, bodies, particles, num_bodies, body_pair, idt, cc, contacts, use_bias,
    )
