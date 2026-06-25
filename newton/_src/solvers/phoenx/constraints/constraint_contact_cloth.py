# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact prepare/iterate factories for rigid and deformable endpoints.

``wp.static`` keeps rigid-only kernels on the body-register path and routes
deformable contacts through endpoint helpers.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import MOTION_DYNAMIC, BodyContainer
from newton._src.solvers.phoenx.cloth_collision import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_count1,
    contact_get_count2,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_side0_counts_extra,
    contact_get_side0_kind,
    contact_get_side0_nodes_extra,
    contact_get_side0_slots_extra,
    contact_get_side1_counts_extra,
    contact_get_side1_kind,
    contact_get_side1_nodes_extra,
    contact_get_side1_slots_extra,
    contact_get_slot1,
    contact_get_slot2,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    constraint_bodies_make,
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
    cc_get_r0,
    cc_get_r1,
    cc_get_side0_bary,
    cc_get_side1_bary,
    cc_get_start_gap,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_local_p0,
    cc_set_local_p1,
    cc_set_normal_lambda,
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_apply_impulse_cached as contact_endpoint_apply_impulse,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_inv_mass_along_cached as contact_endpoint_inv_mass_along,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_set_access_mode_cached as contact_endpoint_set_access_mode,
)
from newton._src.solvers.phoenx.constraints.contact_endpoint import (
    contact_endpoint_velocity_at_point_cached as contact_endpoint_velocity_at_point,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_velocity_update,
    contact_project_velocity_update_no_soft_pd,
)
from newton._src.solvers.phoenx.helpers.math_helpers import (
    apply_pair_spatial_impulse,
    apply_pair_velocity_impulse,
    effective_mass_scalar,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    set_particle_access_mode_with_slot,
    write_angular_velocity_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_config import PHOENX_BOOST_CONTACT_NORMAL


@wp.func
def _contact_uses_stale_anchor_start_gap(contacts: ContactViews, k: wp.int32) -> bool:
    shape0 = contacts.rigid_contact_shape0[k]
    shape1 = contacts.rigid_contact_shape1[k]
    shape_type_count = contacts.shape_type.shape[0]
    uses_start_gap = False
    if shape0 >= wp.int32(0) and shape0 < shape_type_count:
        type0 = contacts.shape_type[shape0]
        uses_start_gap = (
            uses_start_gap
            or type0 == wp.int32(GeoType.MESH)
            or type0 == wp.int32(GeoType.HFIELD)
            or type0 == wp.int32(GeoType.TETRAHEDRON)
        )
    if shape1 >= wp.int32(0) and shape1 < shape_type_count:
        type1 = contacts.shape_type[shape1]
        uses_start_gap = (
            uses_start_gap
            or type1 == wp.int32(GeoType.MESH)
            or type1 == wp.int32(GeoType.HFIELD)
            or type1 == wp.int32(GeoType.TETRAHEDRON)
        )
    return uses_start_gap


__all__ = [
    "contact_cached_warmstart_lean",
    "contact_iterate",
    "contact_iterate_at",
    "contact_iterate_at_cloth_aware",
    "contact_iterate_at_lean",
    "contact_iterate_cloth_aware",
    "contact_iterate_lean",
    "contact_iterate_lean_no_sleep",
    "contact_iterate_lean_no_sleep_no_soft_pd",
    "contact_iterate_lean_no_soft_pd",
    "contact_iterate_no_sleep",
    "contact_iterate_no_sleep_no_soft_pd",
    "contact_iterate_no_soft_pd",
    "contact_prepare_for_iteration",
    "contact_prepare_for_iteration_at",
    "contact_prepare_for_iteration_at_cloth_aware",
    "contact_prepare_for_iteration_at_lean",
    "contact_prepare_for_iteration_cloth_aware",
    "contact_prepare_for_iteration_lean",
    "contact_prepare_for_iteration_lean_no_soft_pd",
    "contact_prepare_for_iteration_no_soft_pd",
]


@wp.func
def _soft_tet_endpoint_set_access_mode_for_column(
    nodes: wp.vec4i,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    slots: wp.vec4i,
    counts: wp.vec4i,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
    cc: ContactContainer,
    contact_first: wp.int32,
    contact_count: wp.int32,
    is_side1: wp.bool,
):
    # A contact column can contain multiple contact points. Use the union
    # of nonzero barycentric weights so access modes cover exactly the
    # particles the column can read or write.
    use_a = bool(False)
    use_b = bool(False)
    use_c = bool(False)
    use_d = bool(False)
    for i in range(contact_count):
        k = contact_first + i
        bary = cc_get_side1_bary(cc, k) if is_side1 else cc_get_side0_bary(cc, k)
        weight_a = bary[0]
        weight_b = bary[1]
        weight_c = bary[2]
        weight_d = wp.float32(1.0) - weight_a - weight_b - weight_c
        if weight_a != wp.float32(0.0):
            use_a = bool(True)
        if weight_b != wp.float32(0.0):
            use_b = bool(True)
        if weight_c != wp.float32(0.0):
            use_c = bool(True)
        if weight_d != wp.float32(0.0):
            use_d = bool(True)
    if use_a and nodes[0] >= wp.int32(0):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[0] - num_bodies, slots[0], new_access_mode, inv_dt
        )
    if use_b and nodes[1] >= wp.int32(0):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[1] - num_bodies, slots[1], new_access_mode, inv_dt
        )
    if use_c and nodes[2] >= wp.int32(0):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[2] - num_bodies, slots[2], new_access_mode, inv_dt
        )
    if use_d and nodes[3] >= wp.int32(0):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[3] - num_bodies, slots[3], new_access_mode, inv_dt
        )


@wp.func
def _side_world_contact_point(
    kind: wp.int32,
    nodes: wp.vec4i,
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
    """World-space contact point on this side incl. ``+/-margin*n`` shift.

    Rigid: ``pos + quat_rotate(orient, local_p - body_com) + sign*margin*n``.
    Cloth: ``bary . particle_positions + sign*margin*n`` (3 nodes).
    Soft-tet: 4-node barycentric anchor (4th weight derived).
    Side 0 uses ``+margin*n`` (push toward side 1); side 1 uses ``-margin*n``.
    """
    sign = wp.float32(-1.0) if is_side1 else wp.float32(1.0)
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        anchor = (
            bary[0] * particles.position[p_a] + bary[1] * particles.position[p_b] + bary[2] * particles.position[p_c]
        )
        return anchor + (sign * margin) * n
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        weight_a = bary[0]
        weight_b = bary[1]
        weight_c = bary[2]
        weight_d = wp.float32(1.0) - weight_a - weight_b - weight_c
        anchor = wp.vec3f(0.0, 0.0, 0.0)
        if weight_a != wp.float32(0.0):
            p_a = nodes[0] - num_bodies
            anchor = anchor + weight_a * particles.position[p_a]
        if weight_b != wp.float32(0.0):
            p_b = nodes[1] - num_bodies
            anchor = anchor + weight_b * particles.position[p_b]
        if weight_c != wp.float32(0.0):
            p_c = nodes[2] - num_bodies
            anchor = anchor + weight_c * particles.position[p_c]
        if weight_d != wp.float32(0.0):
            p_d = nodes[3] - num_bodies
            anchor = anchor + weight_d * particles.position[p_d]
        return anchor + (sign * margin) * n
    b = nodes[0]
    local_p = cc_get_local_p1(cc, k) if is_side1 else cc_get_local_p0(cc, k)
    if b < 0:
        # Static-anchor world shape: shape transform is identity.
        return local_p + (sign * margin) * n
    body_com = bodies.body_com[b]
    orient = bodies.orientation[b]
    pos = bodies.position[b]
    return pos + wp.quat_rotate(orient, local_p - body_com) + (sign * margin) * n


# ---------------------------------------------------------------------------
# Factories: one source per phase, ``cloth_support`` selects the side I/O
# strategy at codegen via ``wp.static``. The rigid path keeps hoisted body
# state + register-cached v/w accumulators; the cloth-aware path routes per
# side through the endpoint helpers (barycentric for cloth tris).
# ---------------------------------------------------------------------------


def _make_contact_prepare_for_iteration_at(
    cloth_support: bool, has_mass_splitting: bool = True, has_soft_contact_pd: bool = True
):
    @wp.func
    def impl(
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
        copy_state: CopyStateContainer,
        parallel_id: wp.int32,
    ):
        """Prepare one contact column for PGS.

        Re-projects each contact's body-frame anchors to world space,
        computes lever arms, effective masses for n/t1/t2 (cached in
        ``cc.derived``), Baumgarte + friction biases, and applies the
        warm-start impulse. Rigid path batches the warm-start scatter;
        cloth-aware scatters per-side via endpoint helpers.
        """
        _ = base_offset

        b1 = body_pair.b1
        b2 = body_pair.b2

        contact_first = contact_get_contact_first(constraints, cid)
        contact_count = contact_get_contact_count(constraints, cid)
        if contact_count == 0:
            return

        dt_substep = wp.float32(1.0) / idt
        bias_rate, _mass_coeff, _impulse_coeff = soft_constraint_coefficients(
            DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
        )

        # Friction bias + speed caps. Scale-invariant; numeric caps would
        # need scaling for scene_scale >> 1.
        friction_bias_factor = wp.float32(0.08)
        friction_slop = wp.float32(0.001)
        max_push_speed = wp.float32(2.0)
        max_approach_speed = wp.float32(10.0)
        slip_threshold = wp.float32(0.002)

        mu_s_col = contact_get_friction(constraints, cid)

        if wp.static(cloth_support):
            side0_kind = contact_get_side0_kind(constraints, cid)
            side1_kind = contact_get_side1_kind(constraints, cid)
            side0_extra = contact_get_side0_nodes_extra(constraints, cid)
            side1_extra = contact_get_side1_nodes_extra(constraints, cid)
            side0_nodes = wp.vec4i(b1, side0_extra[0], side0_extra[1], side0_extra[2])
            side1_nodes = wp.vec4i(b2, side1_extra[0], side1_extra[1], side1_extra[2])
            side0_slots_extra = contact_get_side0_slots_extra(constraints, cid)
            side1_slots_extra = contact_get_side1_slots_extra(constraints, cid)
            side0_counts_extra = contact_get_side0_counts_extra(constraints, cid)
            side1_counts_extra = contact_get_side1_counts_extra(constraints, cid)
            side0_slots = wp.vec4i(
                contact_get_slot1(constraints, cid), side0_slots_extra[0], side0_slots_extra[1], side0_slots_extra[2]
            )
            side1_slots = wp.vec4i(
                contact_get_slot2(constraints, cid), side1_slots_extra[0], side1_slots_extra[1], side1_slots_extra[2]
            )
            side0_counts = wp.vec4i(
                contact_get_count1(constraints, cid),
                side0_counts_extra[0],
                side0_counts_extra[1],
                side0_counts_extra[2],
            )
            side1_counts = wp.vec4i(
                contact_get_count2(constraints, cid),
                side1_counts_extra[0],
                side1_counts_extra[1],
                side1_counts_extra[2],
            )
        else:
            # Mass-splitting fast path: ``highest_index_in_use[0] == 0`` means
            # slot lookup would return (-1, 1) identity. Bypass the
            # get_state_index inlining + inv_factor multiplies entirely.
            # Sets ``slot1 = slot2 = -1`` so the warm-start scatter at the end
            # takes the matching fast-path writeback.
            orientation1 = bodies.orientation[b1]
            orientation2 = bodies.orientation[b2]
            position1 = bodies.position[b1]
            position2 = bodies.position[b2]
            body_com1 = bodies.body_com[b1]
            body_com2 = bodies.body_com[b2]
            if wp.static(not has_mass_splitting):
                # Compile-time lean path: slot lookup dead-code-eliminated.
                slot1 = wp.int32(-1)
                slot2 = wp.int32(-1)
                inv_mass1 = bodies.inverse_mass[b1]
                inv_mass2 = bodies.inverse_mass[b2]
                inv_inertia1 = bodies.inverse_inertia_world[b1]
                inv_inertia2 = bodies.inverse_inertia_world[b2]
            else:
                slot1 = contact_get_slot1(constraints, cid)
                slot2 = contact_get_slot2(constraints, cid)
                inv_factor1 = contact_get_count1(constraints, cid)
                inv_factor2 = contact_get_count2(constraints, cid)
                inv_factor1_f = wp.float32(inv_factor1)
                inv_factor2_f = wp.float32(inv_factor2)
                inv_mass1 = bodies.inverse_mass[b1] * inv_factor1_f
                inv_mass2 = bodies.inverse_mass[b2] * inv_factor2_f
                inv_inertia1 = bodies.inverse_inertia_world[b1] * inv_factor1_f
                inv_inertia2 = bodies.inverse_inertia_world[b2] * inv_factor2_f
            # Batched warm-start accumulators (one velocity scatter at end).
            total_lin_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
            total_ang_imp_on_b1 = wp.vec3f(0.0, 0.0, 0.0)
            total_ang_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)

        for i in range(contact_count):
            k = contact_first + i

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)
            margin0 = contacts.rigid_contact_margin0[k]
            margin1 = contacts.rigid_contact_margin1[k]

            if wp.static(cloth_support):
                bary0 = cc_get_side0_bary(cc, k)
                bary1 = cc_get_side1_bary(cc, k)
                p0_world = _side_world_contact_point(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    num_bodies,
                    cc,
                    k,
                    False,
                    margin0,
                    n,
                )
                p1_world = _side_world_contact_point(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    num_bodies,
                    cc,
                    k,
                    True,
                    margin1,
                    n,
                )
                side0_is_deformable = side0_kind == wp.int32(
                    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE
                ) or side0_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON)
                side1_is_deformable = side1_kind == wp.int32(
                    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE
                ) or side1_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON)

                inv0_n = contact_endpoint_inv_mass_along(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side0_slots,
                    side0_counts,
                    p0_world,
                    n,
                )
                inv0_t1 = inv0_n
                inv0_t2 = inv0_n
                if not side0_is_deformable:
                    inv0_t1 = contact_endpoint_inv_mass_along(
                        side0_kind,
                        side0_nodes,
                        bary0,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        parallel_id,
                        side0_slots,
                        side0_counts,
                        p0_world,
                        t1_dir,
                    )
                    inv0_t2 = contact_endpoint_inv_mass_along(
                        side0_kind,
                        side0_nodes,
                        bary0,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        parallel_id,
                        side0_slots,
                        side0_counts,
                        p0_world,
                        t2_dir,
                    )

                inv1_n = contact_endpoint_inv_mass_along(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side1_slots,
                    side1_counts,
                    p1_world,
                    n,
                )
                inv1_t1 = inv1_n
                inv1_t2 = inv1_n
                if not side1_is_deformable:
                    inv1_t1 = contact_endpoint_inv_mass_along(
                        side1_kind,
                        side1_nodes,
                        bary1,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        parallel_id,
                        side1_slots,
                        side1_counts,
                        p1_world,
                        t1_dir,
                    )
                    inv1_t2 = contact_endpoint_inv_mass_along(
                        side1_kind,
                        side1_nodes,
                        bary1,
                        bodies,
                        particles,
                        copy_state,
                        num_bodies,
                        parallel_id,
                        side1_slots,
                        side1_counts,
                        p1_world,
                        t2_dir,
                    )

                inv_n = inv0_n + inv1_n
                inv_t1 = inv0_t1 + inv1_t1
                inv_t2 = inv0_t2 + inv1_t2
                eff_n = wp.float32(0.0)
                if inv_n > wp.float32(1.0e-12):
                    eff_n = wp.float32(1.0) / inv_n
                eff_t1 = wp.float32(0.0)
                if inv_t1 > wp.float32(1.0e-12):
                    eff_t1 = wp.float32(1.0) / inv_t1
                eff_t2 = wp.float32(0.0)
                if inv_t2 > wp.float32(1.0e-12):
                    eff_t2 = wp.float32(1.0) / inv_t2
                effective_gap = wp.dot(p1_world - p0_world, n)
                p_diff = p1_world - p0_world
            else:
                local_p0 = cc_get_local_p0(cc, k)
                local_p1 = cc_get_local_p1(cc, k)
                p0_world = position1 + wp.quat_rotate(orientation1, local_p0 - body_com1) + margin0 * n
                p1_world = position2 + wp.quat_rotate(orientation2, local_p1 - body_com2) - margin1 * n
                r1 = p0_world - position1
                r2 = p1_world - position2
                eff_n = effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                eff_t1 = effective_mass_scalar(t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                eff_t2 = effective_mass_scalar(t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                effective_gap = wp.dot(p1_world - p0_world, n)
                p_diff = p1_world - p0_world

            # Load-scaled correction (warm-started normal load).
            lam_n_ws = cc_get_normal_lambda(cc, k)
            lam_n_ref = wp.float32(1.0) / wp.max(eff_n * idt, wp.float32(1.0e-6))
            load_boost = wp.min(wp.float32(1.0) + lam_n_ws / lam_n_ref, wp.float32(4.0))

            # Speculative (>0) vs penetrating (<0) Baumgarte bias. Rows
            # generated from stale anchors keep their generation-time gap as an
            # activation barrier; box/plane primitive stacks use the live
            # substep gap so dense stacks can become load-bearing within TGS.
            solver_gap = effective_gap
            use_start_gap = _contact_uses_stale_anchor_start_gap(contacts, k)
            if use_start_gap:
                start_gap = cc_get_start_gap(cc, k)
                if start_gap > wp.float32(0.0) and solver_gap < start_gap:
                    solver_gap = start_gap
            if solver_gap > wp.float32(0.0):
                bias_val = solver_gap * idt
            else:
                bias_val = effective_gap * bias_rate
            bias_val = wp.clamp(bias_val, -max_push_speed, max_approach_speed)

            drift_t1_raw = wp.dot(p_diff, t1_dir)
            drift_t2_raw = wp.dot(p_diff, t2_dir)

            # Sticky-friction break (rigid-only path). Re-reads fresh
            # narrow-phase anchors and zeros tangent lambdas on Coulomb
            # saturation. Cloth contacts regenerate anchors each frame.
            if wp.static(not cloth_support):
                drift_sq = drift_t1_raw * drift_t1_raw + drift_t2_raw * drift_t2_raw
                fresh_n = contacts.rigid_contact_normal[k]
                normal_aligned = wp.dot(n, fresh_n)
                lam_t1_prev = cc_get_tangent1_lambda(cc, k)
                lam_t2_prev = cc_get_tangent2_lambda(cc, k)
                lam_n_prev = cc_get_normal_lambda(cc, k)
                fric_limit_prev = mu_s_col * lam_n_prev
                cone_margin = wp.float32(0.98)
                lam_t_mag_sq = lam_t1_prev * lam_t1_prev + lam_t2_prev * lam_t2_prev
                coulomb_saturated = lam_n_prev > wp.float32(0.0) and lam_t_mag_sq >= (cone_margin * fric_limit_prev) * (
                    cone_margin * fric_limit_prev
                )
                if drift_sq > slip_threshold * slip_threshold or normal_aligned < wp.float32(0.95) or coulomb_saturated:
                    fresh_lp0 = contacts.rigid_contact_point0[k]
                    fresh_lp1 = contacts.rigid_contact_point1[k]
                    cc_set_local_p0(cc, k, fresh_lp0)
                    cc_set_local_p1(cc, k, fresh_lp1)
                    if coulomb_saturated:
                        cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                        cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
                    p0_world = position1 + wp.quat_rotate(orientation1, fresh_lp0 - body_com1) + margin0 * n
                    p1_world = position2 + wp.quat_rotate(orientation2, fresh_lp1 - body_com2) - margin1 * n
                    r1 = p0_world - position1
                    r2 = p1_world - position2
                    eff_n = effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                    eff_t1 = effective_mass_scalar(t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                    eff_t2 = effective_mass_scalar(t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                    p_diff = p1_world - p0_world
                    drift_t1_raw = wp.dot(p_diff, t1_dir)
                    drift_t2_raw = wp.dot(p_diff, t2_dir)

            if (use_start_gap and solver_gap > wp.float32(0.0)) or effective_gap > wp.float32(0.002):
                # Far speculative rows are normal velocity caps, not manifolds.
                cc_set_normal_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
                bias_t1_val = wp.float32(0.0)
                bias_t2_val = wp.float32(0.0)
            else:
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
            if wp.static(not cloth_support):
                cc_set_r0(cc, k, r1)
                cc_set_r1(cc, k, r2)

            # Soft-contact PD normal row (per-contact stiffness/damping).
            # Scenes without stiffness/damping arrays use a specialised iterate
            # variant and skip these per-contact scratch writes entirely.
            if wp.static(has_soft_contact_pd):
                stiffness_arr_len = contacts.rigid_contact_stiffness.shape[0]
                damping_arr_len = contacts.rigid_contact_damping.shape[0]
                if stiffness_arr_len > k or damping_arr_len > k:
                    k_n = wp.float32(0.0)
                    c_n = wp.float32(0.0)
                    if stiffness_arr_len > k:
                        k_n = contacts.rigid_contact_stiffness[k]
                    if damping_arr_len > k:
                        c_n = contacts.rigid_contact_damping[k]
                    if (
                        (k_n > wp.float32(0.0) or c_n > wp.float32(0.0))
                        and eff_n > wp.float32(0.0)
                        and solver_gap <= wp.float32(0.0)
                    ):
                        eff_inv_n = wp.float32(1.0) / eff_n
                        # ``-effective_gap`` flips sign so spring depth is +ve
                        # for penetration; matches the lam_n >= 0 clamp.
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

            # Warm-start impulse scatter. Rigid path accumulates into
            # body-level totals; cloth-aware scatters per side.
            lam_n = cc_get_normal_lambda(cc, k)
            lam_t1 = cc_get_tangent1_lambda(cc, k)
            lam_t2 = cc_get_tangent2_lambda(cc, k)
            imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
            if wp.static(cloth_support):
                contact_endpoint_apply_impulse(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side0_slots,
                    side0_counts,
                    p0_world,
                    -imp,
                )
                contact_endpoint_apply_impulse(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side1_slots,
                    side1_counts,
                    p1_world,
                    imp,
                )
            else:
                total_lin_imp_on_b2 += imp
                total_ang_imp_on_b1 += wp.cross(r1, imp)
                total_ang_imp_on_b2 += wp.cross(r2, imp)

        if wp.static(not cloth_support):
            # Warm-start scatter. ``slot1`` / ``slot2`` carry the mass-
            # splitting state captured at the top of this scope: -1 means
            # the load took the disabled fast-path; >=0 means a slot lookup
            # was done. The fast-path here bypasses ``read_*_unified`` /
            # ``write_*_unified`` inlining for the rigid hot loop.
            if wp.static(not has_mass_splitting):
                # Compile-time lean writeback: direct SoA only.
                v1_cur = bodies.velocity[b1]
                v2_cur = bodies.velocity[b2]
                w1_cur = bodies.angular_velocity[b1]
                w2_cur = bodies.angular_velocity[b2]
                v1_new, v2_new, w1_new, w2_new = apply_pair_spatial_impulse(
                    v1_cur,
                    v2_cur,
                    w1_cur,
                    w2_cur,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                    total_lin_imp_on_b2,
                    total_ang_imp_on_b1,
                    total_ang_imp_on_b2,
                )
                bodies.velocity[b1] = v1_new
                bodies.velocity[b2] = v2_new
                bodies.angular_velocity[b1] = w1_new
                bodies.angular_velocity[b2] = w2_new
            elif slot1 < wp.int32(0) and slot2 < wp.int32(0):
                v1_cur = bodies.velocity[b1]
                v2_cur = bodies.velocity[b2]
                w1_cur = bodies.angular_velocity[b1]
                w2_cur = bodies.angular_velocity[b2]
                v1_new, v2_new, w1_new, w2_new = apply_pair_spatial_impulse(
                    v1_cur,
                    v2_cur,
                    w1_cur,
                    w2_cur,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                    total_lin_imp_on_b2,
                    total_ang_imp_on_b1,
                    total_ang_imp_on_b2,
                )
                bodies.velocity[b1] = v1_new
                bodies.velocity[b2] = v2_new
                bodies.angular_velocity[b1] = w1_new
                bodies.angular_velocity[b2] = w2_new
            else:
                if slot1 < wp.int32(0):
                    v1_cur = bodies.velocity[b1]
                    w1_cur = bodies.angular_velocity[b1]
                else:
                    v1_cur = copy_state.velocity[slot1]
                    w1_cur = copy_state.angular_velocity[slot1]
                if slot2 < wp.int32(0):
                    v2_cur = bodies.velocity[b2]
                    w2_cur = bodies.angular_velocity[b2]
                else:
                    v2_cur = copy_state.velocity[slot2]
                    w2_cur = copy_state.angular_velocity[slot2]
                v1_new, v2_new, w1_new, w2_new = apply_pair_spatial_impulse(
                    v1_cur,
                    v2_cur,
                    w1_cur,
                    w2_cur,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                    total_lin_imp_on_b2,
                    total_ang_imp_on_b1,
                    total_ang_imp_on_b2,
                )
                write_velocity_unified(bodies, particles, copy_state, b1, slot1, num_bodies, v1_new)
                write_velocity_unified(bodies, particles, copy_state, b2, slot2, num_bodies, v2_new)
                write_angular_velocity_unified(bodies, copy_state, b1, slot1, w1_new)
                write_angular_velocity_unified(bodies, copy_state, b2, slot2, w2_new)

    return impl


@wp.func
def contact_cached_warmstart_lean(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    """Apply cached rigid contact warm-start impulses without rebuilding J."""
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    total_lin_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b1 = wp.vec3f(0.0, 0.0, 0.0)
    total_ang_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
    for i in range(contact_count):
        k = contact_first + i
        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        r1 = cc_get_r0(cc, k)
        r2 = cc_get_r1(cc, k)
        lam_n = cc_get_normal_lambda(cc, k)
        lam_t1 = cc_get_tangent1_lambda(cc, k)
        lam_t2 = cc_get_tangent2_lambda(cc, k)
        imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
        total_lin_imp_on_b2 += imp
        total_ang_imp_on_b1 += wp.cross(r1, imp)
        total_ang_imp_on_b2 += wp.cross(r2, imp)

    v1_cur = bodies.velocity[b1]
    v2_cur = bodies.velocity[b2]
    w1_cur = bodies.angular_velocity[b1]
    w2_cur = bodies.angular_velocity[b2]
    v1_new, v2_new, w1_new, w2_new = apply_pair_spatial_impulse(
        v1_cur,
        v2_cur,
        w1_cur,
        w2_cur,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        total_lin_imp_on_b2,
        total_ang_imp_on_b1,
        total_ang_imp_on_b2,
    )
    bodies.velocity[b1] = v1_new
    bodies.velocity[b2] = v2_new
    bodies.angular_velocity[b1] = w1_new
    bodies.angular_velocity[b2] = w2_new


def _make_contact_iterate_at(
    cloth_support: bool, has_mass_splitting: bool = True, use_bias: bool = True, has_soft_contact_pd: bool = True
):
    @wp.func
    def impl(
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
        copy_state: CopyStateContainer,
        parallel_id: wp.int32,
        sor_boost: wp.float32,
    ):
        """One PGS sweep over every contact of one shape pair.

        Sequential Gauss-Seidel within the pair; per-contact normal row
        first (clamp lam_n >= 0), then two tangent rows under the
        two-regime circular Coulomb cone. ``use_bias=False`` is the
        relax pass; speculative normal rows always keep the gap bias.
        """
        _ = base_offset

        b1 = body_pair.b1
        b2 = body_pair.b2

        contact_first = contact_get_contact_first(constraints, cid)
        contact_count = contact_get_contact_count(constraints, cid)
        if contact_count == 0:
            return

        mu_s = contact_get_friction(constraints, cid)
        mu_k = contact_get_friction_dynamic(constraints, cid)

        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if wp.static(use_bias):
            dt_substep = wp.float32(1.0) / idt
            _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
                DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
            )

        if wp.static(cloth_support):
            side0_kind = contact_get_side0_kind(constraints, cid)
            side1_kind = contact_get_side1_kind(constraints, cid)
            side0_extra = contact_get_side0_nodes_extra(constraints, cid)
            side1_extra = contact_get_side1_nodes_extra(constraints, cid)
            side0_nodes = wp.vec4i(b1, side0_extra[0], side0_extra[1], side0_extra[2])
            side1_nodes = wp.vec4i(b2, side1_extra[0], side1_extra[1], side1_extra[2])
            side0_slots_extra = contact_get_side0_slots_extra(constraints, cid)
            side1_slots_extra = contact_get_side1_slots_extra(constraints, cid)
            side0_counts_extra = contact_get_side0_counts_extra(constraints, cid)
            side1_counts_extra = contact_get_side1_counts_extra(constraints, cid)
            side0_slots = wp.vec4i(
                contact_get_slot1(constraints, cid), side0_slots_extra[0], side0_slots_extra[1], side0_slots_extra[2]
            )
            side1_slots = wp.vec4i(
                contact_get_slot2(constraints, cid), side1_slots_extra[0], side1_slots_extra[1], side1_slots_extra[2]
            )
            side0_counts = wp.vec4i(
                contact_get_count1(constraints, cid),
                side0_counts_extra[0],
                side0_counts_extra[1],
                side0_counts_extra[2],
            )
            side1_counts = wp.vec4i(
                contact_get_count2(constraints, cid),
                side1_counts_extra[0],
                side1_counts_extra[1],
                side1_counts_extra[2],
            )
        else:
            # Mass-splitting contact columns read slot/count values stamped by
            # the graph build; no-slot endpoints are cached as ``(-1, 1)``.
            if wp.static(not has_mass_splitting):
                # Compile-time lean: slot lookup dead-code-eliminated.
                v1 = bodies.velocity[b1]
                v2 = bodies.velocity[b2]
                w1 = bodies.angular_velocity[b1]
                w2 = bodies.angular_velocity[b2]
                inv_mass1 = bodies.inverse_mass[b1]
                inv_mass2 = bodies.inverse_mass[b2]
                inv_inertia1 = bodies.inverse_inertia_world[b1]
                inv_inertia2 = bodies.inverse_inertia_world[b2]
                slot1 = wp.int32(-1)
                slot2 = wp.int32(-1)
            else:
                slot1 = contact_get_slot1(constraints, cid)
                slot2 = contact_get_slot2(constraints, cid)
                inv_factor1 = contact_get_count1(constraints, cid)
                inv_factor2 = contact_get_count2(constraints, cid)
                if slot1 < wp.int32(0):
                    v1 = bodies.velocity[b1]
                    w1 = bodies.angular_velocity[b1]
                else:
                    v1 = copy_state.velocity[slot1]
                    w1 = copy_state.angular_velocity[slot1]
                if slot2 < wp.int32(0):
                    v2 = bodies.velocity[b2]
                    w2 = bodies.angular_velocity[b2]
                else:
                    v2 = copy_state.velocity[slot2]
                    w2 = copy_state.angular_velocity[slot2]
                inv_factor1_f = wp.float32(inv_factor1)
                inv_factor2_f = wp.float32(inv_factor2)
                inv_mass1 = bodies.inverse_mass[b1] * inv_factor1_f
                inv_mass2 = bodies.inverse_mass[b2] * inv_factor2_f
                inv_inertia1 = bodies.inverse_inertia_world[b1] * inv_factor1_f
                inv_inertia2 = bodies.inverse_inertia_world[b2] * inv_factor2_f

        for i in range(contact_count):
            k = contact_first + i

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)

            if wp.static(cloth_support):
                margin0 = contacts.rigid_contact_margin0[k]
                margin1 = contacts.rigid_contact_margin1[k]
                bary0 = cc_get_side0_bary(cc, k)
                bary1 = cc_get_side1_bary(cc, k)
                p0_world = _side_world_contact_point(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    num_bodies,
                    cc,
                    k,
                    False,
                    margin0,
                    n,
                )
                p1_world = _side_world_contact_point(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    num_bodies,
                    cc,
                    k,
                    True,
                    margin1,
                    n,
                )
                v0_at_p = contact_endpoint_velocity_at_point(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side0_slots,
                    side0_counts,
                    p0_world,
                )
                v1_at_p = contact_endpoint_velocity_at_point(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side1_slots,
                    side1_counts,
                    p1_world,
                )
                vel_rel = v1_at_p - v0_at_p
            else:
                r1 = cc_get_r0(cc, k)
                r2 = cc_get_r1(cc, k)
                vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)

            jv_n = wp.dot(vel_rel, n)
            jv_t1 = wp.dot(vel_rel, t1_dir)
            jv_t2 = wp.dot(vel_rel, t2_dir)

            eff_n = cc_get_eff_n(cc, k)
            eff_t1 = cc_get_eff_t1(cc, k)
            eff_t2 = cc_get_eff_t2(cc, k)
            bias_val = cc_get_bias(cc, k)
            speculative_bias = bias_val
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)
            if wp.static(use_bias):
                bias_t1_val = cc_get_bias_t1(cc, k)
                bias_t2_val = cc_get_bias_t2(cc, k)
            is_speculative = speculative_bias > wp.float32(0.0)
            if is_speculative and wp.static(not use_bias):
                continue
            if wp.static(not use_bias):
                bias_val = wp.float32(0.0)

            # Normal/friction rows: optional soft-contact PD, penetrating
            # main solve, or rigid relax.
            pd_eff_soft_n = wp.float32(0.0)
            pd_gamma_n = wp.float32(0.0)
            pd_bias_n = wp.float32(0.0)
            if wp.static(has_soft_contact_pd):
                pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
                if pd_eff_soft_n > wp.float32(0.0):
                    pd_gamma_n = cc_get_pd_gamma(cc, k)
                    pd_bias_n = cc_get_pd_bias(cc, k)

            if is_speculative:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
                if speculative_bias <= idt * wp.float32(0.002):
                    mu_s_eff = mu_s
                    mu_k_eff = mu_k
                else:
                    mu_s_eff = wp.float32(0.0)
                    mu_k_eff = wp.float32(0.0)
            elif wp.static(use_bias):
                mass_coeff_n = mass_coeff
                impulse_coeff_n = impulse_coeff
                mu_s_eff = mu_s
                mu_k_eff = mu_k
            else:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
                mu_s_eff = mu_s
                mu_k_eff = mu_k

            if wp.static(has_soft_contact_pd):
                imp = contact_project_velocity_update(
                    cc,
                    k,
                    n,
                    t1_dir,
                    t2_dir,
                    jv_n,
                    jv_t1,
                    jv_t2,
                    eff_n,
                    eff_t1,
                    eff_t2,
                    bias_val,
                    bias_t1_val,
                    bias_t2_val,
                    mu_s_eff,
                    mu_k_eff,
                    mass_coeff_n,
                    impulse_coeff_n,
                    sor_boost,
                    pd_eff_soft_n,
                    pd_gamma_n,
                    pd_bias_n,
                )
            else:
                imp = contact_project_velocity_update_no_soft_pd(
                    cc,
                    k,
                    n,
                    t1_dir,
                    t2_dir,
                    jv_n,
                    jv_t1,
                    jv_t2,
                    eff_n,
                    eff_t1,
                    eff_t2,
                    bias_val,
                    bias_t1_val,
                    bias_t2_val,
                    mu_s_eff,
                    mu_k_eff,
                    mass_coeff_n,
                    impulse_coeff_n,
                    sor_boost,
                    wp.float32(0.0),
                    wp.float32(0.0),
                    wp.float32(0.0),
                )
            if wp.static(cloth_support):
                contact_endpoint_apply_impulse(
                    side0_kind,
                    side0_nodes,
                    bary0,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side0_slots,
                    side0_counts,
                    p0_world,
                    -imp,
                )
                contact_endpoint_apply_impulse(
                    side1_kind,
                    side1_nodes,
                    bary1,
                    bodies,
                    particles,
                    copy_state,
                    num_bodies,
                    parallel_id,
                    side1_slots,
                    side1_counts,
                    p1_world,
                    imp,
                )
            else:
                v1, v2, w1, w2 = apply_pair_velocity_impulse(
                    v1,
                    v2,
                    w1,
                    w2,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                    r1,
                    r2,
                    imp,
                )

        if wp.static(not cloth_support):
            # Mass-splitting fast-path writeback (matches the load gate above).
            if wp.static(not has_mass_splitting) or (slot1 < wp.int32(0) and slot2 < wp.int32(0)):
                bodies.velocity[b1] = v1
                bodies.velocity[b2] = v2
                bodies.angular_velocity[b1] = w1
                bodies.angular_velocity[b2] = w2
            else:
                write_velocity_unified(bodies, particles, copy_state, b1, slot1, num_bodies, v1)
                write_velocity_unified(bodies, particles, copy_state, b2, slot2, num_bodies, v2)
                write_angular_velocity_unified(bodies, copy_state, b1, slot1, w1)
                write_angular_velocity_unified(bodies, copy_state, b2, slot2, w2)

    return impl


# Bind factory outputs as module-level names BEFORE the entry-point
# ``@wp.func`` wrappers below reference them.

contact_prepare_for_iteration_at = _make_contact_prepare_for_iteration_at(cloth_support=False)
contact_prepare_for_iteration_at_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_soft_contact_pd=False
)
contact_prepare_for_iteration_at_lean = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False
)
contact_prepare_for_iteration_at_lean_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False, has_soft_contact_pd=False
)
contact_prepare_for_iteration_at_cloth_aware = _make_contact_prepare_for_iteration_at(cloth_support=True)
contact_iterate_at = _make_contact_iterate_at(cloth_support=False, use_bias=True)
contact_relax_at = _make_contact_iterate_at(cloth_support=False, use_bias=False)
contact_iterate_at_no_soft_pd = _make_contact_iterate_at(cloth_support=False, use_bias=True, has_soft_contact_pd=False)
contact_relax_at_no_soft_pd = _make_contact_iterate_at(cloth_support=False, use_bias=False, has_soft_contact_pd=False)
contact_iterate_at_lean = _make_contact_iterate_at(cloth_support=False, has_mass_splitting=False, use_bias=True)
contact_relax_at_lean = _make_contact_iterate_at(cloth_support=False, has_mass_splitting=False, use_bias=False)
contact_iterate_at_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=True, has_soft_contact_pd=False
)
contact_relax_at_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=False, has_soft_contact_pd=False
)
contact_iterate_at_cloth_aware = _make_contact_iterate_at(cloth_support=True, use_bias=True)
contact_relax_at_cloth_aware = _make_contact_iterate_at(cloth_support=True, use_bias=False)


# Entry-point wrappers: read header, flip access modes, then call the
# ``*_at`` variant. Module-level wp.funcs so Warp resolves them from globals.


@wp.func
def contact_prepare_for_iteration(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    # Access-mode flip is the caller's responsibility now. The dispatcher routes
    # rigid-only scenes here (``cloth_support=False`` branch), where no position-
    # level writers exist, so the flip is provably a no-op.
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at(
        constraints,
        cid,
        0,
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        copy_state,
        parallel_id,
    )


@wp.func
def contact_prepare_for_iteration_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at_no_soft_pd(
        constraints,
        cid,
        0,
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        copy_state,
        parallel_id,
    )


@wp.func
def contact_prepare_for_iteration_lean(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    """Mass-splitting-free entry. Used by the multi-world fast-tail
    kernel where mass splitting is rejected at construction."""
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at_lean(
        constraints,
        cid,
        0,
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        copy_state,
        parallel_id,
    )


@wp.func
def contact_prepare_for_iteration_lean_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at_lean_no_soft_pd(
        constraints,
        cid,
        0,
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        copy_state,
        parallel_id,
    )


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
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    side0_nodes = wp.vec4i(b1, side0_extra[0], side0_extra[1], side0_extra[2])
    side1_nodes = wp.vec4i(b2, side1_extra[0], side1_extra[1], side1_extra[2])
    side0_slots_extra = contact_get_side0_slots_extra(constraints, cid)
    side1_slots_extra = contact_get_side1_slots_extra(constraints, cid)
    side0_counts_extra = contact_get_side0_counts_extra(constraints, cid)
    side1_counts_extra = contact_get_side1_counts_extra(constraints, cid)
    side0_slots = wp.vec4i(
        contact_get_slot1(constraints, cid), side0_slots_extra[0], side0_slots_extra[1], side0_slots_extra[2]
    )
    side1_slots = wp.vec4i(
        contact_get_slot2(constraints, cid), side1_slots_extra[0], side1_slots_extra[1], side1_slots_extra[2]
    )
    side0_counts = wp.vec4i(
        contact_get_count1(constraints, cid), side0_counts_extra[0], side0_counts_extra[1], side0_counts_extra[2]
    )
    side1_counts = wp.vec4i(
        contact_get_count2(constraints, cid), side1_counts_extra[0], side1_counts_extra[1], side1_counts_extra[2]
    )
    if side0_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        _soft_tet_endpoint_set_access_mode_for_column(
            side0_nodes,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side0_slots,
            side0_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
            cc,
            contact_first,
            contact_count,
            False,
        )
    else:
        contact_endpoint_set_access_mode(
            side0_kind,
            side0_nodes,
            bodies,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side0_slots,
            side0_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
        )
    if side1_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        _soft_tet_endpoint_set_access_mode_for_column(
            side1_nodes,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side1_slots,
            side1_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
            cc,
            contact_first,
            contact_count,
            True,
        )
    else:
        contact_endpoint_set_access_mode(
            side1_kind,
            side1_nodes,
            bodies,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side1_slots,
            side1_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
        )
    body_pair = constraint_bodies_make(b1, b2)
    contact_prepare_for_iteration_at_cloth_aware(
        constraints,
        cid,
        0,
        bodies,
        particles,
        num_bodies,
        body_pair,
        idt,
        cc,
        contacts,
        copy_state,
        parallel_id,
    )


@wp.func
def contact_iterate(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    if not _contact_iterate_guard_allows(bodies, b1, b2, num_bodies):
        return
    # Access-mode flip is the caller's responsibility now (dispatcher only
    # routes rigid-only scenes here, so the flip is provably a no-op).
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    if not _contact_iterate_guard_allows(bodies, b1, b2, num_bodies):
        return
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def _contact_iterate_guard_allows(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    num_bodies: wp.int32,
) -> wp.bool:
    # Backup safety for the sleep-transition frame. Contacts produced
    # before the sleeping pass can survive into this step; if both
    # endpoints are frozen, skip the row.
    if b1 >= 0 and b1 < num_bodies and b2 >= 0 and b2 < num_bodies:
        frozen1 = (bodies.motion_type[b1] != MOTION_DYNAMIC) or (bodies.island_root[b1] >= wp.int32(0))
        frozen2 = (bodies.motion_type[b2] != MOTION_DYNAMIC) or (bodies.island_root[b2] >= wp.int32(0))
        if frozen1 and frozen2:
            return False
    return True


@wp.func
def contact_iterate_no_sleep(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_no_sleep_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_lean(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    if not _contact_iterate_guard_allows(bodies, b1, b2, num_bodies):
        return
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_lean(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_lean(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_lean_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    if not _contact_iterate_guard_allows(bodies, b1, b2, num_bodies):
        return
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_lean_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_lean_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_lean_no_sleep(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_lean(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_lean(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )


@wp.func
def contact_iterate_lean_no_sleep_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_lean_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_lean_no_soft_pd(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
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
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    side0_nodes = wp.vec4i(b1, side0_extra[0], side0_extra[1], side0_extra[2])
    side1_nodes = wp.vec4i(b2, side1_extra[0], side1_extra[1], side1_extra[2])
    side0_slots_extra = contact_get_side0_slots_extra(constraints, cid)
    side1_slots_extra = contact_get_side1_slots_extra(constraints, cid)
    side0_counts_extra = contact_get_side0_counts_extra(constraints, cid)
    side1_counts_extra = contact_get_side1_counts_extra(constraints, cid)
    side0_slots = wp.vec4i(
        contact_get_slot1(constraints, cid), side0_slots_extra[0], side0_slots_extra[1], side0_slots_extra[2]
    )
    side1_slots = wp.vec4i(
        contact_get_slot2(constraints, cid), side1_slots_extra[0], side1_slots_extra[1], side1_slots_extra[2]
    )
    side0_counts = wp.vec4i(
        contact_get_count1(constraints, cid), side0_counts_extra[0], side0_counts_extra[1], side0_counts_extra[2]
    )
    side1_counts = wp.vec4i(
        contact_get_count2(constraints, cid), side1_counts_extra[0], side1_counts_extra[1], side1_counts_extra[2]
    )
    if side0_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        _soft_tet_endpoint_set_access_mode_for_column(
            side0_nodes,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side0_slots,
            side0_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
            cc,
            contact_first,
            contact_count,
            False,
        )
    else:
        contact_endpoint_set_access_mode(
            side0_kind,
            side0_nodes,
            bodies,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side0_slots,
            side0_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
        )
    if side1_kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        _soft_tet_endpoint_set_access_mode_for_column(
            side1_nodes,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side1_slots,
            side1_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
            cc,
            contact_first,
            contact_count,
            True,
        )
    else:
        contact_endpoint_set_access_mode(
            side1_kind,
            side1_nodes,
            bodies,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            side1_slots,
            side1_counts,
            ACCESS_MODE_VELOCITY_LEVEL,
            idt,
        )
    body_pair = constraint_bodies_make(b1, b2)
    if use_bias:
        contact_iterate_at_cloth_aware(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
    else:
        contact_relax_at_cloth_aware(
            constraints,
            cid,
            0,
            bodies,
            particles,
            num_bodies,
            body_pair,
            idt,
            cc,
            contacts,
            copy_state,
            parallel_id,
            sor_boost,
        )
