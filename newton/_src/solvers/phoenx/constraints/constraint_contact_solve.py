# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact prepare/iterate factories for rigid and deformable endpoints.

``wp.static`` keeps rigid-only kernels on the body-register path and routes
deformable contacts through endpoint helpers.
"""

from __future__ import annotations

import functools

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_orientation,
    body_load_vw,
    body_store_vw,
    contact_point_weight,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    _contact_uses_stale_anchor_start_gap,
    contact_angular_friction,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_count1,
    contact_get_count2,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_original_contact_first,
    contact_get_packed_inv_inertia1,
    contact_get_packed_inv_inertia2,
    contact_get_packed_inv_mass1,
    contact_get_packed_inv_mass2,
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
    contact_owned_elsewhere,
    contact_pd_coefficients,
    contact_set_packed_body_properties,
    contact_side_witness,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    constraint_bodies_make,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_friction_anchor0,
    cc_get_friction_anchor1,
    cc_get_mobility_nt1,
    cc_get_mobility_nt2,
    cc_get_mobility_t1t2,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
    cc_get_r0,
    cc_get_r1,
    cc_get_side0_bary,
    cc_get_side1_bary,
    cc_get_surface_target,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_prepare_surface_target,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_friction_anchor0,
    cc_set_friction_anchor1,
    cc_set_mobility_nt1,
    cc_set_mobility_nt2,
    cc_set_mobility_t1t2,
    cc_set_normal_lambda,
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
    cc_soft_coefficients,
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
from newton._src.solvers.phoenx.constraints.contact_patch_friction import (
    contact_patch_effective_mass,
    contact_patch_project_velocity_update,
)
from newton._src.solvers.phoenx.constraints.contact_projection import (
    contact_project_coupled_velocity_update,
    contact_project_coupled_velocity_update_mobility,
    contact_project_coupled_velocity_update_mobility_no_soft_pd,
    contact_project_coupled_velocity_update_no_soft_pd,
    contact_project_normal_delta_load,
    contact_project_normal_delta_load_no_soft_pd,
    contact_project_normal_velocity_update,
    contact_project_normal_velocity_update_no_soft_pd,
    contact_project_tangent_delta,
    contact_project_velocity_update,
    contact_project_velocity_update_no_soft_pd,
)
from newton._src.solvers.phoenx.constraints.contact_tgs_cooperative import shuffle_vec
from newton._src.solvers.phoenx.constraints.contact_warm_start import store_row_warm_start
from newton._src.solvers.phoenx.contact_endpoints import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON,
)
from newton._src.solvers.phoenx.helpers.math_helpers import (
    apply_pair_spatial_impulse,
    apply_pair_velocity_impulse,
    contact_mobility_scalar,
    effective_mass_scalar,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    set_particle_access_mode_with_slot,
    write_angular_velocity_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "contact_cached_warmstart_lean",
    "contact_iterate",
    "contact_iterate_at",
    "contact_iterate_at_cloth_aware",
    "contact_iterate_at_lean",
    "contact_iterate_cloth_aware",
    "contact_iterate_lean",
    "contact_iterate_lean_mobility",
    "contact_iterate_lean_no_sleep",
    "contact_iterate_lean_no_sleep_mobility",
    "contact_iterate_lean_no_sleep_no_soft_pd",
    "contact_iterate_lean_no_sleep_no_soft_pd_mobility",
    "contact_iterate_lean_no_soft_pd",
    "contact_iterate_lean_no_soft_pd_mobility",
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


#: Maximum speed [m/s] added by penetration recovery.
_CONTACT_MAX_PUSH_SPEED = wp.constant(wp.float32(2.0))
#: Friction anchors: largest drift [m] the positional bias corrects, drift [m]
#: past which the anchor follows the contact, and the gap [m] below which a
#: speculative row already carries friction.
_FRICTION_SLOP = wp.constant(wp.float32(0.001))
_FRICTION_SLIP = wp.constant(wp.float32(0.002))
_FRICTION_GAP = wp.constant(wp.float32(0.002))


@wp.func
def contact_normal_bias(
    gap: wp.float32, idt: wp.float32, bias_rate: wp.float32, length_scale: wp.float32
) -> wp.float32:
    """Normal velocity bias of a contact row with signed ``gap``, in scene units.

    A separated (speculative) row admits the approach speed that closes the
    gap within the substep; a penetrating row pushes out at the soft-contact
    rate. ``length_scale`` is the scene's length units per metre.
    """
    if gap > wp.float32(0.0):
        return gap * idt
    return wp.max(gap * bias_rate, -_CONTACT_MAX_PUSH_SPEED * length_scale)


# The uncapped default contact hertz is Nyquist-clamped whenever idt <= 2*hertz.
# In that regime the Box2D mass/impulse coefficients are independent of dt.
_DEFAULT_CONTACT_MASS_COEFF = wp.constant(wp.float32(0.9417003989219666))
_DEFAULT_CONTACT_IMPULSE_COEFF = wp.constant(wp.float32(0.05829954519867897))


@wp.func
def _default_contact_solver_coefficients(cc: ContactContainer, idt: wp.float32):
    """Return default contact mass/impulse coefficients."""
    if cc.soft_capped == 0 and idt <= wp.float32(2.0) * DEFAULT_HERTZ_CONTACT:
        return _DEFAULT_CONTACT_MASS_COEFF, _DEFAULT_CONTACT_IMPULSE_COEFF
    _bias_rate, mass_coeff, impulse_coeff = cc_soft_coefficients(cc, idt)
    return mass_coeff, impulse_coeff


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
    contacts: ContactViews,
    source_k: wp.int32,
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
    local_p, extra = contact_side_witness(contacts, source_k, wp.int32(1) if is_side1 else wp.int32(0))
    margin = margin + extra
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


@wp.kernel(enable_backward=False)
def rebase_ordinary_contact_relax_kernel(
    columns: ContactColumnContainer,
    cc: ContactContainer,
    bodies: BodyContainer,
    column_count: wp.array[wp.int32],
    mass_splitting: wp.bool,
    canonical_owner: wp.array[wp.int32],
    element_ids_by_color: wp.array[wp.int32],
    contact_offset: wp.int32,
    packed_headers: wp.bool,
    rows_per_column: wp.int32,
    rebase_levers: wp.bool,
    advance_gap: wp.bool,
    idt: wp.float32,
    column_stride: wp.int32,
):
    """Update prepared rows for the substep's integrated rigid motion.

    ``rebase_levers`` moves each prepared impulse point with the bodies' translation,
    weighted like :func:`contact_point_weight`. ``advance_gap`` moves each row's gap by the normal displacement of
    the material points at the impulse point and refreshes its normal bias, so a
    following substep that reuses the prepared rows neither admits an approach
    that was already spent nor keeps pushing out a resolved penetration.
    """
    first_column, lane = wp.tid()
    for cid in range(first_column, column_count[0], column_stride):
        owner_cid = cid
        if packed_headers:
            owner_cid = element_ids_by_color[cid] - contact_offset
            if owner_cid < wp.int32(0):
                continue
        if canonical_owner[owner_cid] >= wp.int32(0):
            continue
        b1 = contact_get_body1(columns, cid)
        b2 = contact_get_body2(columns, cid)
        shift1 = bodies.position_prev_substep[b1] - bodies.position[b1]
        shift2 = bodies.position_prev_substep[b2] - bodies.position[b2]
        first = contact_get_contact_first(columns, cid)
        count = contact_get_contact_count(columns, cid)
        if advance_gap and cc.gap.shape[0] > 0:
            # Substep rotation of each body; its translation is ``-shift``.
            rotation1 = body_load_orientation(bodies, b1) * wp.quat_inverse(bodies.orientation_prev_substep[b1])
            rotation2 = body_load_orientation(bodies, b2) * wp.quat_inverse(bodies.orientation_prev_substep[b2])
            bias_rate, _mass_coeff, _impulse_coeff = cc_soft_coefficients(cc, idt)
            for k in range(first + lane, first + count, rows_per_column):
                r1 = cc_get_r0(cc, k)
                r2 = cc_get_r1(cc, k)
                moved1 = wp.quat_rotate(rotation1, r1) - r1 - shift1
                moved2 = wp.quat_rotate(rotation2, r2) - r2 - shift2
                gap = cc.gap[k] + wp.dot(moved2 - moved1, cc_get_normal(cc, k))
                cc.gap[k] = gap
                cc_set_bias(cc, k, contact_normal_bias(gap, idt, bias_rate, cc.length_scale))
        if not rebase_levers:
            continue
        factor1 = wp.float32(1.0)
        factor2 = wp.float32(1.0)
        if mass_splitting:
            factor1 = wp.float32(contact_get_count1(columns, cid))
            factor2 = wp.float32(contact_get_count2(columns, cid))
        inverse_mass1 = bodies.inverse_mass[b1] * factor1
        inverse_mass2 = bodies.inverse_mass[b2] * factor2
        inverse_inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1]) * factor1
        inverse_inertia2 = mat33_from_sym6(bodies.inverse_inertia_world[b2]) * factor2
        # The common impulse point follows the bodies' translation weighted like their witnesses,
        # see :func:`contact_point_weight`.
        weight1 = contact_point_weight(bodies, b1, b2)
        point_shift = -(weight1 * shift1 + (wp.float32(1.0) - weight1) * shift2)
        # Rebase rows independently; no impulse or body velocity is modified.
        for k in range(first + lane, first + count, rows_per_column):
            r1 = cc_get_r0(cc, k) + shift1 + point_shift
            r2 = cc_get_r1(cc, k) + shift2 + point_shift
            cc_set_r0(cc, k, r1)
            cc_set_r1(cc, k, r2)
            normal = cc_get_normal(cc, k)
            tangent1 = cc_get_tangent1(cc, k)
            tangent2 = wp.cross(normal, tangent1)
            eff_n = effective_mass_scalar(
                normal, r1, r2, inverse_mass1, inverse_mass2, inverse_inertia1, inverse_inertia2
            )
            cc_set_eff_n(cc, k, eff_n)
            tangent_response1 = contact_mobility_scalar(
                tangent1, r1, r2, inverse_mass1, inverse_mass2, inverse_inertia1, inverse_inertia2
            )
            tangent_response2 = contact_mobility_scalar(
                tangent2, r1, r2, inverse_mass1, inverse_mass2, inverse_inertia1, inverse_inertia2
            )
            if not packed_headers:
                # Ordinary columns and reduced-coordinate consumers retain
                # reciprocal effective masses; packed rigid PGS uses mobility.
                tangent_response1 = wp.float32(1.0) / tangent_response1 if tangent_response1 > 0.0 else 0.0
                tangent_response2 = wp.float32(1.0) / tangent_response2 if tangent_response2 > 0.0 else 0.0
            cc_set_eff_t1(cc, k, tangent_response1)
            cc_set_eff_t2(cc, k, tangent_response2)
            rn1 = wp.cross(r1, normal)
            rn2 = wp.cross(r2, normal)
            rt11 = wp.cross(r1, tangent1)
            rt12 = wp.cross(r2, tangent1)
            rt21 = wp.cross(r1, tangent2)
            rt22 = wp.cross(r2, tangent2)
            response_t11 = inverse_inertia1 * rt11
            response_t12 = inverse_inertia2 * rt12
            response_t21 = inverse_inertia1 * rt21
            response_t22 = inverse_inertia2 * rt22
            cc_set_mobility_nt1(cc, k, wp.dot(rn1, response_t11) + wp.dot(rn2, response_t12))
            cc_set_mobility_nt2(cc, k, wp.dot(rn1, response_t21) + wp.dot(rn2, response_t22))
            cc_set_mobility_t1t2(cc, k, wp.dot(rt11, response_t21) + wp.dot(rt12, response_t22))
            if cc_get_pd_eff_soft(cc, k) > wp.float32(0.0) and eff_n > wp.float32(0.0):
                cc_set_pd_eff_soft(cc, k, wp.float32(1.0) / (wp.float32(1.0) / eff_n + cc_get_pd_gamma(cc, k)))


def _make_contact_prepare_for_iteration_at(
    cloth_support: bool,
    has_mass_splitting: bool = True,
    has_soft_contact_pd: bool = True,
    patch_friction: bool = False,
    packed_rows: bool = False,
    stage_body_properties: bool = False,
    pointwise_geometry: bool = False,
    tangent_mobility_layout: bool = False,
):
    if pointwise_geometry and (cloth_support or patch_friction or stage_body_properties):
        raise ValueError("Pointwise geometry requires rigid contacts without patches or shared header writes.")

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
        if wp.static(not packed_rows):
            _ = base_offset

        b1 = body_pair.b1
        b2 = body_pair.b2

        contact_first = contact_get_contact_first(constraints, cid)
        contact_count = contact_get_contact_count(constraints, cid)
        if contact_count == 0:
            return
        if wp.static(pointwise_geometry):
            contact_first = base_offset
            contact_count = wp.int32(1)

        dt_substep = wp.float32(1.0) / idt
        bias_rate, _mass_coeff, _impulse_coeff = cc_soft_coefficients(cc, idt)

        # Friction bias + speed caps. Scale-invariant; numeric caps would
        # need scaling for scene_scale >> 1.
        friction_bias_factor = wp.float32(0.08)
        friction_slop = _FRICTION_SLOP * cc.length_scale
        slip_threshold = _FRICTION_SLIP * cc.length_scale

        mu_s_col = contact_get_friction(constraints, cid)

        use_patch = False
        if wp.static(patch_friction):
            use_patch = constraints.patch.eligible[cid] != wp.int32(0)
            patch_normal_sum = wp.vec3f(0.0, 0.0, 0.0)
            patch_r0_sum = wp.vec3f(0.0, 0.0, 0.0)
            patch_r1_sum = wp.vec3f(0.0, 0.0, 0.0)
            patch_bias_world_sum = wp.vec3f(0.0, 0.0, 0.0)
            patch_normal_load = wp.float32(0.0)

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
            orientation1 = body_load_orientation(bodies, b1)
            orientation2 = body_load_orientation(bodies, b2)
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
                inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1))
                inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2))
            else:
                slot1 = contact_get_slot1(constraints, cid)
                slot2 = contact_get_slot2(constraints, cid)
                inv_factor1 = contact_get_count1(constraints, cid)
                inv_factor2 = contact_get_count2(constraints, cid)
                inv_factor1_f = wp.float32(inv_factor1)
                inv_factor2_f = wp.float32(inv_factor2)
                inv_mass1_raw = bodies.inverse_mass[b1]
                inv_mass2_raw = bodies.inverse_mass[b2]
                inv_inertia1_raw = body_load_inv_inertia_sym6(bodies, b1)
                inv_inertia2_raw = body_load_inv_inertia_sym6(bodies, b2)
                inv_mass1 = inv_mass1_raw * inv_factor1_f
                inv_mass2 = inv_mass2_raw * inv_factor2_f
                inv_inertia1 = mat33_from_sym6(inv_inertia1_raw) * inv_factor1_f
                inv_inertia2 = mat33_from_sym6(inv_inertia2_raw) * inv_factor2_f
                if wp.static(stage_body_properties):
                    contact_set_packed_body_properties(
                        constraints,
                        cid,
                        inv_mass1_raw,
                        inv_mass2_raw,
                        inv_inertia1_raw,
                        inv_inertia2_raw,
                    )
            # Batched warm-start accumulators (one velocity scatter at end).
            total_lin_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)
            total_ang_imp_on_b1 = wp.vec3f(0.0, 0.0, 0.0)
            total_ang_imp_on_b2 = wp.vec3f(0.0, 0.0, 0.0)

        for i in range(contact_count):
            k = contact_first + i
            source_k = k
            if wp.static(pointwise_geometry and packed_rows):
                # ``base_offset`` is one color-ordered row; map it back to the
                # immutable Newton input through the column's canonical range.
                source_k = contact_get_original_contact_first(constraints, cid) + (
                    k - contact_get_contact_first(constraints, cid)
                )
            elif wp.static(packed_rows):
                source_k = base_offset + i

            if wp.static(patch_friction):
                if use_patch:
                    cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                    cc_set_tangent2_lambda(cc, k, wp.float32(0.0))

            n = cc_get_normal(cc, k)
            t1_dir = cc_get_tangent1(cc, k)
            t2_dir = wp.cross(n, t1_dir)
            margin0 = contacts.rigid_contact_margin0[source_k]
            margin1 = contacts.rigid_contact_margin1[source_k]

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
                    contacts,
                    source_k,
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
                    contacts,
                    source_k,
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
                local_p0, extra0 = contact_side_witness(contacts, source_k, wp.int32(0))
                local_p1, extra1 = contact_side_witness(contacts, source_k, wp.int32(1))
                relative_p0 = wp.quat_rotate(orientation1, local_p0 - body_com1) + (margin0 + extra0) * n
                relative_p1 = wp.quat_rotate(orientation2, local_p1 - body_com2) - (margin1 + extra1) * n
                witness_delta = (position2 - position1) + (relative_p1 - relative_p0)
                # Apply equal and opposite impulses at a common point so
                # separated collision witnesses cannot create a friction couple.
                weight1 = contact_point_weight(bodies, b1, b2)
                # Avoid losing local precision by adding and subtracting world positions.
                r1 = relative_p0 + (wp.float32(1.0) - weight1) * witness_delta
                r2 = relative_p1 - weight1 * witness_delta
                # Only ``eff_n`` is consumed before the sticky-break anchor
                # decision (bias below). ``eff_t1`` / ``eff_t2``
                # are deferred until after that block so they are computed once
                # from the final ``r1`` / ``r2`` instead of twice when the
                # break re-projects fresh anchors. Bit-identical either way.
                eff_n = effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                effective_gap = wp.dot(witness_delta, n)
                p_diff = witness_delta

            if wp.static(has_mass_splitting and not cloth_support):
                if cc.normal_compliance_scale.shape[0] > k:
                    scale = wp.float32(1.0)
                    if inv_factor1 > wp.int32(1) or inv_factor2 > wp.int32(1):
                        physical_eff_n = effective_mass_scalar(
                            n,
                            r1,
                            r2,
                            inv_mass1_raw,
                            inv_mass2_raw,
                            mat33_from_sym6(inv_inertia1_raw),
                            mat33_from_sym6(inv_inertia2_raw),
                        )
                        if physical_eff_n > wp.float32(0.0):
                            scale = eff_n / physical_eff_n
                    cc.normal_compliance_scale[k] = scale

            # Recompute remaining separation from the current transformed
            # witnesses. A positive generation-time gap must not prevent a
            # speculative contact from activating during later substeps.
            solver_gap = effective_gap
            use_start_gap = _contact_uses_stale_anchor_start_gap(contacts, k)
            friction_inactive = (
                use_start_gap and solver_gap > wp.float32(0.0)
            ) or effective_gap > _FRICTION_GAP * cc.length_scale
            bias_val = contact_normal_bias(effective_gap, idt, bias_rate, cc.length_scale)
            if cc.gap.shape[0] > 0:
                cc.gap[k] = effective_gap

            if wp.static(not cloth_support):
                friction_r0 = wp.quat_rotate(orientation1, cc_get_friction_anchor0(cc, k) - body_com1)
                friction_r1 = wp.quat_rotate(orientation2, cc_get_friction_anchor1(cc, k) - body_com2)
                p_diff = (position2 - position1) + (friction_r1 - friction_r0)

            drift_t1_raw = wp.dot(p_diff, t1_dir)
            drift_t2_raw = wp.dot(p_diff, t2_dir)

            if wp.static(not cloth_support):
                drift_sq = drift_t1_raw * drift_t1_raw + drift_t2_raw * drift_t2_raw
                if friction_inactive or drift_sq > slip_threshold * slip_threshold:
                    # These are friction references, never collision witnesses.
                    # Rebase their displacement without discarding the contact's
                    # accumulated impulse. Proximity to the Coulomb limit alone
                    # does not imply sliding; the solve projects the impulse.
                    # A supported manifold can briefly unload individual witnesses
                    # within a step. Keep their reference while friction remains
                    # eligible, including the stricter stale-witness gap rule.
                    midpoint_r0 = relative_p0 + wp.float32(0.5) * witness_delta
                    midpoint_r1 = relative_p1 - wp.float32(0.5) * witness_delta
                    anchor0 = body_com1 + wp.quat_rotate_inv(orientation1, midpoint_r0)
                    anchor1 = body_com2 + wp.quat_rotate_inv(orientation2, midpoint_r1)
                    cc_set_friction_anchor0(cc, k, anchor0)
                    cc_set_friction_anchor1(cc, k, anchor1)
                    drift_t1_raw = wp.float32(0.0)
                    drift_t2_raw = wp.float32(0.0)
                if cc.surface_target.shape[0] > 0:
                    # Conveyor surfaces: friction drives the slip to the surface velocity. The
                    # anchors would hold the bodies together, so they follow the contact instead.
                    target = cc_prepare_surface_target(
                        cc, k, contacts.rigid_contact_surface_velocity, source_k, t1_dir, t2_dir, n
                    )
                    if target[0] * target[0] + target[1] * target[1] > wp.float32(0.0):
                        midpoint_r0 = relative_p0 + wp.float32(0.5) * witness_delta
                        midpoint_r1 = relative_p1 - wp.float32(0.5) * witness_delta
                        cc_set_friction_anchor0(cc, k, body_com1 + wp.quat_rotate_inv(orientation1, midpoint_r0))
                        cc_set_friction_anchor1(cc, k, body_com2 + wp.quat_rotate_inv(orientation2, midpoint_r1))
                        drift_t1_raw = wp.float32(0.0)
                        drift_t2_raw = wp.float32(0.0)

            if friction_inactive:
                # Far speculative rows are normal velocity caps, not manifolds.
                cc_set_normal_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
                bias_t1_val = wp.float32(0.0)
                bias_t2_val = wp.float32(0.0)
            else:
                drift_t1 = wp.clamp(drift_t1_raw, -friction_slop, friction_slop)
                drift_t2 = wp.clamp(drift_t2_raw, -friction_slop, friction_slop)
                bias_t1_val = friction_bias_factor * drift_t1 * idt
                bias_t2_val = friction_bias_factor * drift_t2 * idt

            if wp.static(not cloth_support):
                # Patch friction builds one coupled tangent mass at the shared
                # patch point, so per-point tangent masses are redundant.
                eff_t1 = wp.float32(0.0)
                eff_t2 = wp.float32(0.0)
                if wp.static(not patch_friction):
                    if wp.static(tangent_mobility_layout):
                        # Packed rigid point rows feed a coupled tangent
                        # metric, which uses mobility directly each sweep.
                        eff_t1 = contact_mobility_scalar(
                            t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
                        )
                        eff_t2 = contact_mobility_scalar(
                            t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2
                        )
                    else:
                        eff_t1 = effective_mass_scalar(t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                        eff_t2 = effective_mass_scalar(t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                elif not use_patch:
                    eff_t1 = effective_mass_scalar(t1_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
                    eff_t2 = effective_mass_scalar(t2_dir, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)

            cc_set_eff_n(cc, k, eff_n)
            cc_set_eff_t1(cc, k, eff_t1)
            cc_set_eff_t2(cc, k, eff_t2)
            if wp.static(not cloth_support and not patch_friction):
                rn1 = wp.cross(r1, n)
                rn2 = wp.cross(r2, n)
                rt11 = wp.cross(r1, t1_dir)
                rt12 = wp.cross(r2, t1_dir)
                rt21 = wp.cross(r1, t2_dir)
                rt22 = wp.cross(r2, t2_dir)
                response_t11 = inv_inertia1 * rt11
                response_t12 = inv_inertia2 * rt12
                response_t21 = inv_inertia1 * rt21
                response_t22 = inv_inertia2 * rt22
                cc_set_mobility_nt1(cc, k, wp.dot(rn1, response_t11) + wp.dot(rn2, response_t12))
                cc_set_mobility_nt2(cc, k, wp.dot(rn1, response_t21) + wp.dot(rn2, response_t22))
                cc_set_mobility_t1t2(cc, k, wp.dot(rt11, response_t21) + wp.dot(rt12, response_t22))
            cc_set_bias(cc, k, bias_val)
            cc_set_bias_t1(cc, k, bias_t1_val)
            cc_set_bias_t2(cc, k, bias_t2_val)
            if wp.static(not cloth_support):
                cc_set_r0(cc, k, r1)
                cc_set_r1(cc, k, r2)

            if wp.static(patch_friction):
                if use_patch:
                    patch_normal_sum += n
                    patch_r0_sum += r1
                    patch_r1_sum += r2
                    patch_bias_world_sum += bias_t1_val * t1_dir + bias_t2_val * t2_dir
                    patch_normal_load += wp.max(cc_get_normal_lambda(cc, k), wp.float32(0.0))

            # Soft-contact PD normal row (per-contact stiffness/damping).
            # Scenes without stiffness/damping arrays use a specialised iterate
            # variant and skip these per-contact scratch writes entirely.
            if wp.static(has_soft_contact_pd):
                if (
                    contacts.rigid_contact_stiffness.shape[0] > source_k
                    or contacts.rigid_contact_damping.shape[0] > source_k
                    or contacts.shape_compliance.shape[0] > 0
                ):
                    if eff_n > 0.0 and solver_gap <= 0.0:
                        pd_gamma_n, pd_bias_n, pd_eff_soft_n = contact_pd_coefficients(
                            contacts, source_k, -effective_gap, wp.float32(1.0) / eff_n, dt_substep, cc
                        )
                        cc_set_pd_gamma(cc, k, pd_gamma_n)
                        cc_set_pd_bias(cc, k, pd_bias_n)
                        cc_set_pd_eff_soft(cc, k, pd_eff_soft_n)
                    else:
                        cc_set_pd_eff_soft(cc, k, wp.float32(0.0))
                else:
                    cc_set_pd_eff_soft(cc, k, wp.float32(0.0))
                if contacts.restitution_target.shape[0] > 0:
                    impact = contacts.restitution_target[source_k]
                    if impact[0] > 0.0 or impact[1] != 0.0:
                        # By the end of this substep the swept witnesses have met.
                        cc_set_bias(cc, k, 0.0)
                    if impact[0] != 0.0 or impact[1] != 0.0:
                        # Impact imposes an outgoing speed independent of the resting contact spring.
                        cc_set_pd_gamma(cc, k, wp.float32(0.0))
                        cc_set_pd_bias(cc, k, impact[0])
                        cc_set_pd_eff_soft(cc, k, eff_n)

            if wp.static(pointwise_geometry):
                # No velocity reads precede this return. The colored cached
                # warm-start pass retains the original impulse summation order.
                return

            # Warm-start impulse scatter. Rigid path accumulates into
            # body-level totals; cloth-aware scatters per side.
            lam_n = cc_get_normal_lambda(cc, k)
            lam_t1 = cc_get_tangent1_lambda(cc, k)
            lam_t2 = cc_get_tangent2_lambda(cc, k)
            imp = lam_n * n + lam_t1 * t1_dir + lam_t2 * t2_dir
            if constraints.articulation_owner[cid] >= wp.int32(0):
                # The owning solver applies this cached load through its
                # constrained response after contact geometry is prepared.
                imp = wp.vec3f(0.0)
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

        if wp.static(patch_friction):
            if use_patch:
                inv_count = wp.float32(1.0) / wp.float32(contact_count)
                patch_normal_length_sq = wp.dot(patch_normal_sum, patch_normal_sum)
                patch_normal = cc_get_normal(cc, contact_first)
                if patch_normal_length_sq > wp.float32(1.0e-12):
                    patch_normal = patch_normal_sum / wp.sqrt(patch_normal_length_sq)
                patch_point0 = position1 + patch_r0_sum * inv_count
                patch_point1 = position2 + patch_r1_sum * inv_count
                patch_point = wp.float32(0.5) * (patch_point0 + patch_point1)
                patch_r0 = patch_point - position1
                patch_r1 = patch_point - position2
                tangent_seed = cc_get_tangent1(cc, contact_first)
                tangent1 = tangent_seed - wp.dot(tangent_seed, patch_normal) * patch_normal
                tangent_length_sq = wp.dot(tangent1, tangent1)
                if tangent_length_sq > wp.float32(1.0e-12):
                    tangent1 = tangent1 / wp.sqrt(tangent_length_sq)
                else:
                    sign = wp.float32(1.0)
                    if patch_normal[2] < wp.float32(0.0):
                        sign = wp.float32(-1.0)
                    a = wp.float32(-1.0) / (sign + patch_normal[2])
                    tangent1 = wp.vec3f(
                        wp.float32(1.0) + sign * patch_normal[0] * patch_normal[0] * a,
                        sign * patch_normal[0] * patch_normal[1] * a,
                        -sign * patch_normal[0],
                    )
                tangent2 = wp.cross(patch_normal, tangent1)
                patch_bias_world = patch_bias_world_sum * inv_count
                patch_bias = wp.vec2f(
                    wp.dot(patch_bias_world, tangent1),
                    wp.dot(patch_bias_world, tangent2),
                )
                effective_mass = contact_patch_effective_mass(
                    tangent1,
                    tangent2,
                    patch_r0,
                    patch_r1,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                )
                warm_impulse = constraints.patch.impulse_world[cid]
                warm_impulse = warm_impulse - wp.dot(warm_impulse, patch_normal) * patch_normal
                warm_magnitude = wp.length(warm_impulse)
                warm_limit = mu_s_col * patch_normal_load
                if warm_magnitude > warm_limit and warm_magnitude > wp.float32(1.0e-15):
                    warm_impulse *= warm_limit / warm_magnitude
                constraints.patch.impulse_world[cid] = warm_impulse
                constraints.patch.normal[cid] = patch_normal
                constraints.patch.tangent1[cid] = tangent1
                constraints.patch.r0[cid] = patch_r0
                constraints.patch.r1[cid] = patch_r1
                constraints.patch.effective_mass[cid] = effective_mass
                constraints.patch.bias[cid] = patch_bias
                total_lin_imp_on_b2 += warm_impulse
                total_ang_imp_on_b1 += wp.cross(patch_r0, warm_impulse)
                total_ang_imp_on_b2 += wp.cross(patch_r1, warm_impulse)

        if constraints.articulation_owner[cid] >= wp.int32(0):
            return

        if wp.static(not cloth_support):
            # Warm-start scatter. ``slot1`` / ``slot2`` carry the mass-
            # splitting state captured at the top of this scope: -1 means
            # the load took the disabled fast-path; >=0 means a slot lookup
            # was done. The fast-path here bypasses ``read_*_unified`` /
            # ``write_*_unified`` inlining for the rigid hot loop.
            if wp.static(not has_mass_splitting):
                # Compile-time lean writeback: direct SoA only.
                v1_cur, w1_cur = body_load_vw(bodies, b1)
                v2_cur, w2_cur = body_load_vw(bodies, b2)
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
                body_store_vw(bodies, b1, v1_new, w1_new)
                body_store_vw(bodies, b2, v2_new, w2_new)
            elif slot1 < wp.int32(0) and slot2 < wp.int32(0):
                v1_cur, w1_cur = body_load_vw(bodies, b1)
                v2_cur, w2_cur = body_load_vw(bodies, b2)
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
                body_store_vw(bodies, b1, v1_new, w1_new)
                body_store_vw(bodies, b2, v2_new, w2_new)
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
    if constraints.articulation_owner[cid] >= wp.int32(0):
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1))
    inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2))

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

    v1_cur, w1_cur = body_load_vw(bodies, b1)
    v2_cur, w2_cur = body_load_vw(bodies, b2)
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
    body_store_vw(bodies, b1, v1_new, w1_new)
    body_store_vw(bodies, b2, v2_new, w2_new)


@wp.func
def _contact_cached_warmstart_split(
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
    if constraints.articulation_owner[cid] >= wp.int32(0):
        return
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    slot1 = contact_get_slot1(constraints, cid)
    slot2 = contact_get_slot2(constraints, cid)
    count1 = wp.float32(contact_get_count1(constraints, cid))
    count2 = wp.float32(contact_get_count2(constraints, cid))
    inv_mass1 = bodies.inverse_mass[b1] * count1
    inv_mass2 = bodies.inverse_mass[b2] * count2
    inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1)) * count1
    inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2)) * count2

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

    if slot1 < wp.int32(0) and slot2 < wp.int32(0):
        v1_cur, w1_cur = body_load_vw(bodies, b1)
        v2_cur, w2_cur = body_load_vw(bodies, b2)
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
        body_store_vw(bodies, b1, v1_new, w1_new)
        body_store_vw(bodies, b2, v2_new, w2_new)
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


@functools.cache
def get_colored_point_prepare_kernel(has_soft_pd: bool):
    """One thread per color-ordered contact row: geometry only, no velocity writes.

    Each row stores its warm-start impulse, which is applied separately per
    body, so rows need no coloring.
    Rows of the mass-splitting overflow color scale body responses by their
    copy counts like the overflow sweep; regular colors use physical bodies.
    """
    prepare_split = _make_contact_prepare_for_iteration_at(
        cloth_support=False,
        has_mass_splitting=True,
        has_soft_contact_pd=has_soft_pd,
        packed_rows=True,
        pointwise_geometry=True,
        tangent_mobility_layout=True,
    )
    prepare_lean = _make_contact_prepare_for_iteration_at(
        cloth_support=False,
        has_mass_splitting=False,
        has_soft_contact_pd=has_soft_pd,
        packed_rows=True,
        pointwise_geometry=True,
        tangent_mobility_layout=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        columns: ContactColumnContainer,
        row_total: wp.array[wp.int32],
        row_slots: wp.array[wp.int32],
        color_starts: wp.array[wp.int32],
        overflow_color: wp.int32,
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        warm_linear: wp.array[wp.vec3f],
        warm_angular0: wp.array[wp.vec3f],
        warm_angular1: wp.array[wp.vec3f],
    ):
        k = wp.tid()
        if k >= row_total[0]:
            return
        slot = row_slots[k]
        b1 = contact_get_body1(columns, slot)
        b2 = contact_get_body2(columns, slot)
        pair = constraint_bodies_make(b1, b2)
        if overflow_color >= wp.int32(0) and slot >= color_starts[overflow_color]:
            if k == contact_get_contact_first(columns, slot):
                # The overflow sweep reads body properties staged in the header.
                contact_set_packed_body_properties(
                    columns,
                    slot,
                    bodies.inverse_mass[b1],
                    bodies.inverse_mass[b2],
                    body_load_inv_inertia_sym6(bodies, b1),
                    body_load_inv_inertia_sym6(bodies, b2),
                )
            prepare_split(
                columns, slot, k, bodies, particles, num_bodies, pair, idt, cc, contacts, copy_state, wp.int32(0)
            )
        else:
            prepare_lean(
                columns, slot, k, bodies, particles, num_bodies, pair, idt, cc, contacts, copy_state, wp.int32(0)
            )
        store_row_warm_start(cc, k, warm_linear, warm_angular0, warm_angular1)

    return kernel


@functools.cache
def _get_parallel_contact_prepare_kernel(has_mass_splitting: bool, has_soft_pd: bool):
    prepare_point = _make_contact_prepare_for_iteration_at(
        cloth_support=False,
        has_mass_splitting=has_mass_splitting,
        has_soft_contact_pd=has_soft_pd,
        pointwise_geometry=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        columns: ContactColumnContainer,
        num_columns: wp.array[wp.int32],
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        idt: wp.float32,
        cc: ContactContainer,
        contacts: ContactViews,
        copy_state: CopyStateContainer,
        column_stride: wp.int32,
        row_lanes: wp.int32,
    ):
        first_column, lane = wp.tid()
        for cid in range(first_column, num_columns[0], column_stride):
            count = contact_get_contact_count(columns, cid)
            first = contact_get_contact_first(columns, cid)
            pair = constraint_bodies_make(contact_get_body1(columns, cid), contact_get_body2(columns, cid))
            for offset in range(lane, count, row_lanes):
                prepare_point(
                    columns,
                    cid,
                    first + offset,
                    bodies,
                    particles,
                    num_bodies,
                    pair,
                    idt,
                    cc,
                    contacts,
                    copy_state,
                    wp.int32(0),
                )

    return kernel


@wp.func
def _speed_upper_bound(velocity: wp.vec3f) -> float:
    # The 1-norm bounds the Euclidean speed without a square root in the row loop.
    return wp.abs(velocity[0]) + wp.abs(velocity[1]) + wp.abs(velocity[2])


@wp.func
def _contact_group_is_separated(bounds: wp.vec4f, v1: wp.vec3f, v2: wp.vec3f, w1: wp.vec3f, w2: wp.vec3f) -> bool:
    """Test the minimum separation bias against the largest possible approach speed in a group."""
    if bounds[0] <= wp.float32(0.0):
        return False
    # Forming the point velocity adds spin to the full translation before subtracting v1.
    # Even equal large translations can round a marginally separated row into contact.
    translation_error = wp.float32(2.0e-6) * _speed_upper_bound(v1)
    approach = (
        _speed_upper_bound(v2 - v1)
        + wp.max(_speed_upper_bound(w1), wp.float32(1.0e-18)) * bounds[1]
        + wp.max(_speed_upper_bound(w2), wp.float32(1.0e-18)) * bounds[2]
        + translation_error
    )
    return bounds[0] > approach


#: Rows per column whose relaxation skip tests load ahead of the row loop.
_RELAX_PRELOADED_ROWS = 8


def _make_contact_iterate_at(
    cloth_support: bool,
    has_mass_splitting: bool = True,
    use_bias: bool = True,
    has_soft_contact_pd: bool = True,
    moving_surface_only: bool = False,
    patch_friction: bool = False,
    staged_body_properties: bool = False,
    cooperative_lanes: int = 1,
    *,
    frictionless_fast_path: bool = False,
    normal_first: bool = False,
    tangent_mobility_layout: bool = False,
):
    has_surface_target = has_soft_contact_pd or moving_surface_only
    # Moving-surface sweeps skip four-row groups that cannot touch this substep.
    separation_tests = (
        moving_surface_only
        and use_bias
        and not has_mass_splitting
        and not cloth_support
        and not patch_friction
        and cooperative_lanes == 1
    )
    if tangent_mobility_layout and normal_first:
        raise ValueError("The normal-first projector requires inverse tangent effective masses")
    if tangent_mobility_layout:
        coupled_project = (
            contact_project_coupled_velocity_update_mobility
            if has_soft_contact_pd
            else contact_project_coupled_velocity_update_mobility_no_soft_pd
        )
    else:
        coupled_project = (
            contact_project_coupled_velocity_update
            if has_soft_contact_pd
            else contact_project_coupled_velocity_update_no_soft_pd
        )

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
        relax pass; speculative normal rows always keep the gap bias. The
        dispatchers exclude reduced-owned columns before calling this
        single-sweep path; their multi-sweep path keeps its explicit guard.
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
        use_patch = False
        if wp.static(patch_friction):
            use_patch = constraints.patch.eligible[cid] != wp.int32(0)
            patch_normal_load = wp.float32(0.0)

        mass_coeff = wp.float32(1.0)
        impulse_coeff = wp.float32(0.0)
        if wp.static(use_bias):
            mass_coeff, impulse_coeff = _default_contact_solver_coefficients(cc, idt)

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
                v1, w1 = body_load_vw(bodies, b1)
                v2, w2 = body_load_vw(bodies, b2)
                inv_mass1 = bodies.inverse_mass[b1]
                inv_mass2 = bodies.inverse_mass[b2]
                inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1))
                inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2))
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
                if wp.static(staged_body_properties):
                    inv_mass1 = contact_get_packed_inv_mass1(constraints, cid) * inv_factor1_f
                    inv_mass2 = contact_get_packed_inv_mass2(constraints, cid) * inv_factor2_f
                    inv_inertia1 = mat33_from_sym6(contact_get_packed_inv_inertia1(constraints, cid)) * inv_factor1_f
                    inv_inertia2 = mat33_from_sym6(contact_get_packed_inv_inertia2(constraints, cid)) * inv_factor2_f
                else:
                    inv_mass1 = bodies.inverse_mass[b1] * inv_factor1_f
                    inv_mass2 = bodies.inverse_mass[b2] * inv_factor2_f
                    inv_inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1]) * inv_factor1_f
                    inv_inertia2 = mat33_from_sym6(bodies.inverse_inertia_world[b2]) * inv_factor2_f

        row_sor = sor_boost
        if wp.static(use_bias):
            if cc.sweep_relaxation.shape[0] > contact_first:
                row_sor = sor_boost * cc.sweep_relaxation[contact_first]

        if wp.static(separation_tests):
            # Load the bounds of the first three four-row groups together
            # with the column: tested one by one, each load would add its
            # latency to the column's chain.
            group_bounds0 = wp.vec4f(0.0)
            group_bounds1 = wp.vec4f(0.0)
            group_bounds2 = wp.vec4f(0.0)
            if cc.separation_bounds.shape[0] > 0 and contact_count > 0:
                last = contact_first + contact_count - 1
                group_bounds0 = cc.separation_bounds[contact_first]
                group_bounds1 = cc.separation_bounds[wp.min(contact_first + 4, last)]
                group_bounds2 = cc.separation_bounds[wp.min(contact_first + 8, last)]
        if wp.static(not use_bias):
            # Relaxation skips separated rows; test the first rows' biases
            # with parallel loads rather than one load per row.
            separated_rows = wp.int32(0)
            if contact_count > 0:
                for j in range(wp.static(_RELAX_PRELOADED_ROWS)):
                    row = contact_first + wp.min(j, contact_count - 1)
                    if cc_get_bias(cc, row) > wp.float32(0.0):
                        separated_rows = separated_rows | (wp.int32(1) << wp.int32(j))

        next_row = wp.int32(0)
        while next_row < contact_count:
            i = next_row
            next_row += 1
            if wp.static(separation_tests):
                if cc.separation_bounds.shape[0] > 0 and (i & 3) == 0 and row_sor >= 0.0:
                    bounds = group_bounds0
                    if i == 4:
                        bounds = group_bounds1
                    elif i == 8:
                        bounds = group_bounds2
                    elif i > 8:
                        bounds = cc.separation_bounds[contact_first + i]
                    if _contact_group_is_separated(bounds, v1, v2, w1, w2):
                        next_row = i + 4
                        continue
            active_point = True
            if wp.static(cooperative_lanes > 1):
                active_point = parallel_id == i % wp.static(cooperative_lanes)
            if active_point:
                k = contact_first + i
                if wp.static(not use_bias):
                    # Velocity relaxation skips separated rows. Check before
                    # loading their geometry or computing endpoint velocities.
                    separated = False
                    if i < wp.static(_RELAX_PRELOADED_ROWS):
                        separated = (separated_rows >> i) & wp.int32(1) != wp.int32(0)
                    else:
                        separated = cc_get_bias(cc, k) > wp.float32(0.0)
                    if separated:
                        continue
                n = cc_get_normal(cc, k)
                normal_only = False
                if wp.static(frictionless_fast_path and not cloth_support):
                    if not use_patch and mu_s == wp.float32(0.0) and mu_k == wp.float32(0.0):
                        # A stale tangent impulse must still be released by the
                        # general solve, even after friction becomes zero.
                        normal_only = cc_get_tangent1_lambda(cc, k) == wp.float32(0.0) and cc_get_tangent2_lambda(
                            cc, k
                        ) == wp.float32(0.0)
                t1_dir = wp.vec3f(0.0)
                t2_dir = wp.vec3f(0.0)
                if not normal_only:
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
                        contacts,
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
                        contacts,
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
                if wp.static(has_surface_target):
                    jv_n += cc_get_surface_target(cc, k)[2]
                jv_t1 = wp.float32(0.0)
                jv_t2 = wp.float32(0.0)
                eff_t1 = wp.float32(0.0)
                eff_t2 = wp.float32(0.0)
                if not normal_only:
                    if wp.static(not patch_friction):
                        jv_t1 = wp.dot(vel_rel, t1_dir)
                        jv_t2 = wp.dot(vel_rel, t2_dir)
                        eff_t1 = cc_get_eff_t1(cc, k)
                        eff_t2 = cc_get_eff_t2(cc, k)
                    elif not use_patch:
                        jv_t1 = wp.dot(vel_rel, t1_dir)
                        jv_t2 = wp.dot(vel_rel, t2_dir)
                        eff_t1 = cc_get_eff_t1(cc, k)
                        eff_t2 = cc_get_eff_t2(cc, k)
                    # Moving meshes are nonconvex and use point rows even in
                    # patch mode. Preserve their per-contact velocity field.
                    if wp.static(has_surface_target):
                        if not use_patch and cc.surface_target.shape[0] > 0:
                            target = cc_get_surface_target(cc, k)
                            jv_t1 += target[0]
                            jv_t2 += target[1]

                eff_n = cc_get_eff_n(cc, k)
                bias_val = cc_get_bias(cc, k)
                speculative_bias = bias_val
                friction_in_range = speculative_bias <= idt * _FRICTION_GAP * cc.length_scale
                if cc.gap.shape[0] > 0:
                    friction_in_range = cc.gap[k] <= _FRICTION_GAP * cc.length_scale
                bias_t1_val = wp.float32(0.0)
                bias_t2_val = wp.float32(0.0)
                if not normal_only:
                    if wp.static(use_bias and not patch_friction):
                        bias_t1_val = cc_get_bias_t1(cc, k)
                        bias_t2_val = cc_get_bias_t2(cc, k)
                    elif wp.static(use_bias):
                        if not use_patch:
                            bias_t1_val = cc_get_bias_t1(cc, k)
                            bias_t2_val = cc_get_bias_t2(cc, k)
                is_speculative = speculative_bias > wp.float32(0.0)
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
                    if friction_in_range:
                        mu_s_eff = mu_s
                        mu_k_eff = mu_k
                    else:
                        mu_s_eff = wp.float32(0.0)
                        mu_k_eff = wp.float32(0.0)
                elif wp.static(use_bias):
                    mass_coeff_n = mass_coeff
                    impulse_coeff_n = impulse_coeff
                    if wp.static(has_mass_splitting and not cloth_support and not patch_friction and not normal_first):
                        if cc.normal_compliance_scale.shape[0] > k and pd_eff_soft_n <= wp.float32(0.0):
                            scale = cc.normal_compliance_scale[k]
                            if scale > wp.float32(0.0) and scale < wp.float32(1.0):
                                # Keep gamma = (b/a) * W_physical unchanged
                                # when W_split preconditions the solve. The
                                # resulting coefficients are W_split/(W_split
                                # + gamma) and gamma/(W_split + gamma).
                                scaled_impulse = impulse_coeff * scale
                                denominator = mass_coeff + scaled_impulse
                                mass_coeff_n = mass_coeff / denominator
                                impulse_coeff_n = scaled_impulse / denominator
                    mu_s_eff = mu_s
                    mu_k_eff = mu_k
                else:
                    mass_coeff_n = wp.float32(1.0)
                    impulse_coeff_n = wp.float32(0.0)
                    mu_s_eff = mu_s
                    mu_k_eff = mu_k

                if normal_only:
                    if wp.static(has_soft_contact_pd):
                        imp = contact_project_normal_velocity_update(
                            cc,
                            k,
                            n,
                            jv_n,
                            eff_n,
                            bias_val,
                            mass_coeff_n,
                            impulse_coeff_n,
                            row_sor,
                            pd_eff_soft_n,
                            pd_gamma_n,
                            pd_bias_n,
                        )
                    else:
                        imp = contact_project_normal_velocity_update_no_soft_pd(
                            cc,
                            k,
                            n,
                            jv_n,
                            eff_n,
                            bias_val,
                            mass_coeff_n,
                            impulse_coeff_n,
                            row_sor,
                            pd_eff_soft_n,
                            pd_gamma_n,
                            pd_bias_n,
                        )
                else:
                    # Keep rigid lever/inertia expressions behind a standalone
                    # static branch: Warp does not prune a static term inside a
                    # mixed static/runtime boolean expression for cloth kernels.
                    handled_rigid_point = False
                    imp = wp.vec3(0.0)
                    if wp.static(not cloth_support):
                        if not use_patch:
                            handled_rigid_point = True
                            if wp.static(normal_first):
                                if wp.static(has_soft_contact_pd):
                                    normal_result = contact_project_normal_delta_load(
                                        cc,
                                        k,
                                        jv_n,
                                        eff_n,
                                        bias_val,
                                        mass_coeff_n,
                                        impulse_coeff_n,
                                        row_sor,
                                        pd_eff_soft_n,
                                        pd_gamma_n,
                                        pd_bias_n,
                                    )
                                else:
                                    normal_result = contact_project_normal_delta_load_no_soft_pd(
                                        cc,
                                        k,
                                        jv_n,
                                        eff_n,
                                        bias_val,
                                        mass_coeff_n,
                                        impulse_coeff_n,
                                        row_sor,
                                        pd_eff_soft_n,
                                        pd_gamma_n,
                                        pd_bias_n,
                                    )
                                normal_delta = normal_result[0]
                                load = normal_result[1]
                                tangent_delta = wp.vec2f(0.0)
                                if load == wp.float32(0.0) or (
                                    mu_s_eff == wp.float32(0.0) and mu_k_eff == wp.float32(0.0)
                                ):
                                    # Removing old tangent impulses is still a physical impulse.
                                    tangent_delta = wp.vec2f(
                                        -cc_get_tangent1_lambda(cc, k),
                                        -cc_get_tangent2_lambda(cc, k),
                                    )
                                    cc_set_tangent1_lambda(cc, k, wp.float32(0.0))
                                    cc_set_tangent2_lambda(cc, k, wp.float32(0.0))
                                else:
                                    mobility_nt1 = cc_get_mobility_nt1(cc, k)
                                    mobility_nt2 = cc_get_mobility_nt2(cc, k)
                                    mobility_t1t2 = cc_get_mobility_t1t2(cc, k)
                                    tangent_delta = contact_project_tangent_delta(
                                        cc,
                                        k,
                                        normal_delta,
                                        load,
                                        jv_t1,
                                        jv_t2,
                                        eff_t1,
                                        eff_t2,
                                        bias_t1_val,
                                        bias_t2_val,
                                        mu_s_eff,
                                        mu_k_eff,
                                        row_sor,
                                        mobility_nt1,
                                        mobility_nt2,
                                        mobility_t1t2,
                                    )
                                imp = normal_delta * n + tangent_delta[0] * t1_dir + tangent_delta[1] * t2_dir
                            else:
                                mobility_nt1 = cc_get_mobility_nt1(cc, k)
                                mobility_nt2 = cc_get_mobility_nt2(cc, k)
                                mobility_t1t2 = cc_get_mobility_t1t2(cc, k)
                                imp = coupled_project(
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
                                    row_sor,
                                    pd_eff_soft_n,
                                    pd_gamma_n,
                                    pd_bias_n,
                                    mobility_nt1,
                                    mobility_nt2,
                                    mobility_t1t2,
                                )
                    if not handled_rigid_point:
                        if wp.static(patch_friction):
                            if use_patch:
                                if wp.static(has_soft_contact_pd):
                                    imp = contact_project_normal_velocity_update(
                                        cc,
                                        k,
                                        n,
                                        jv_n,
                                        eff_n,
                                        bias_val,
                                        mass_coeff_n,
                                        impulse_coeff_n,
                                        row_sor,
                                        pd_eff_soft_n,
                                        pd_gamma_n,
                                        pd_bias_n,
                                    )
                                else:
                                    imp = contact_project_normal_velocity_update_no_soft_pd(
                                        cc,
                                        k,
                                        n,
                                        jv_n,
                                        eff_n,
                                        bias_val,
                                        mass_coeff_n,
                                        impulse_coeff_n,
                                        row_sor,
                                        wp.float32(0.0),
                                        wp.float32(0.0),
                                        wp.float32(0.0),
                                    )

                            else:
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
                                        row_sor,
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
                                        row_sor,
                                        wp.float32(0.0),
                                        wp.float32(0.0),
                                        wp.float32(0.0),
                                    )
                        else:
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
                                    row_sor,
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
                                    row_sor,
                                    wp.float32(0.0),
                                    wp.float32(0.0),
                                    wp.float32(0.0),
                                )
                if wp.static(patch_friction):
                    if use_patch:
                        lambda_n_load = cc_get_normal_lambda(cc, k)
                        if is_speculative and not friction_in_range:
                            lambda_n_load = wp.float32(0.0)
                        patch_normal_load += lambda_n_load

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
            if wp.static(cooperative_lanes > 1):
                if contact_count > wp.int32(1):
                    v1 = shuffle_vec(v1, i % wp.static(cooperative_lanes), wp.static(cooperative_lanes))
                    v2 = shuffle_vec(v2, i % wp.static(cooperative_lanes), wp.static(cooperative_lanes))
                    w1 = shuffle_vec(w1, i % wp.static(cooperative_lanes), wp.static(cooperative_lanes))
                    w2 = shuffle_vec(w2, i % wp.static(cooperative_lanes), wp.static(cooperative_lanes))

        if wp.static(patch_friction):
            if use_patch:
                patch_normal = constraints.patch.normal[cid]
                patch_tangent1 = constraints.patch.tangent1[cid]
                patch_tangent2 = wp.cross(patch_normal, patch_tangent1)
                patch_r0 = constraints.patch.r0[cid]
                patch_r1 = constraints.patch.r1[cid]
                patch_relative_velocity = v2 + wp.cross(w2, patch_r1) - v1 - wp.cross(w1, patch_r0)
                patch_bias = wp.vec2f(0.0, 0.0)
                if wp.static(use_bias):
                    patch_bias = constraints.patch.bias[cid]
                patch_impulse_old = constraints.patch.impulse_world[cid]
                patch_lambda_old = wp.vec2f(
                    wp.dot(patch_impulse_old, patch_tangent1),
                    wp.dot(patch_impulse_old, patch_tangent2),
                )
                patch_update = contact_patch_project_velocity_update(
                    patch_lambda_old,
                    wp.vec2f(
                        wp.dot(patch_relative_velocity, patch_tangent1),
                        wp.dot(patch_relative_velocity, patch_tangent2),
                    ),
                    patch_bias,
                    constraints.patch.effective_mass[cid],
                    patch_normal_load,
                    mu_s,
                    mu_k,
                    row_sor,
                )
                patch_impulse_delta = patch_update.delta[0] * patch_tangent1 + patch_update.delta[1] * patch_tangent2
                v1, v2, w1, w2 = apply_pair_velocity_impulse(
                    v1,
                    v2,
                    w1,
                    w2,
                    inv_mass1,
                    inv_mass2,
                    inv_inertia1,
                    inv_inertia2,
                    patch_r0,
                    patch_r1,
                    patch_impulse_delta,
                )
                constraints.patch.impulse_world[cid] = (
                    patch_update.lambda_new[0] * patch_tangent1 + patch_update.lambda_new[1] * patch_tangent2
                )

        if wp.static(not cloth_support and cooperative_lanes == 1):
            if constraints.angular_impulse.shape[0] > cid:
                w1, w2 = contact_angular_friction(
                    constraints, cid, cc, contact_first, contact_count, w1, w2, inv_inertia1, inv_inertia2
                )

        write_owner = True
        if wp.static(cooperative_lanes > 1):
            write_owner = parallel_id == 0
        if write_owner:
            if wp.static(not cloth_support):
                # Mass-splitting fast-path writeback (matches the load gate above).
                if wp.static(not has_mass_splitting) or (slot1 < wp.int32(0) and slot2 < wp.int32(0)):
                    body_store_vw(bodies, b1, v1, w1)
                    body_store_vw(bodies, b2, v2, w2)
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
contact_prepare_for_iteration_at_packed_rows = _make_contact_prepare_for_iteration_at(
    cloth_support=False, packed_rows=True, stage_body_properties=True, tangent_mobility_layout=True
)
contact_prepare_for_iteration_at_packed_rows_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False,
    has_soft_contact_pd=False,
    packed_rows=True,
    stage_body_properties=True,
    tangent_mobility_layout=True,
)
contact_prepare_for_iteration_at_packed_rows_lean = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False, packed_rows=True, tangent_mobility_layout=True
)
contact_prepare_for_iteration_at_packed_rows_lean_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False,
    has_mass_splitting=False,
    has_soft_contact_pd=False,
    packed_rows=True,
    tangent_mobility_layout=True,
)
contact_prepare_for_iteration_at_lean = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False
)
contact_prepare_for_iteration_at_lean_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False, has_soft_contact_pd=False
)
contact_prepare_for_iteration_at_patch_lean = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False, patch_friction=True
)
contact_prepare_for_iteration_at_patch_lean_no_soft_pd = _make_contact_prepare_for_iteration_at(
    cloth_support=False, has_mass_splitting=False, has_soft_contact_pd=False, patch_friction=True
)
contact_prepare_for_iteration_at_cloth_aware = _make_contact_prepare_for_iteration_at(cloth_support=True)
contact_iterate_at = _make_contact_iterate_at(cloth_support=False, use_bias=True)
contact_relax_at = _make_contact_iterate_at(cloth_support=False, use_bias=False)
contact_iterate_at_no_soft_pd = _make_contact_iterate_at(cloth_support=False, use_bias=True, has_soft_contact_pd=False)
contact_relax_at_no_soft_pd = _make_contact_iterate_at(cloth_support=False, use_bias=False, has_soft_contact_pd=False)
contact_iterate_at_packed = _make_contact_iterate_at(
    cloth_support=False, use_bias=True, staged_body_properties=True, tangent_mobility_layout=True
)
contact_relax_at_packed = _make_contact_iterate_at(
    cloth_support=False, use_bias=False, staged_body_properties=True, tangent_mobility_layout=True
)
contact_iterate_at_packed_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False,
    use_bias=True,
    has_soft_contact_pd=False,
    staged_body_properties=True,
    tangent_mobility_layout=True,
)
contact_relax_at_packed_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False,
    use_bias=False,
    has_soft_contact_pd=False,
    staged_body_properties=True,
    tangent_mobility_layout=True,
)
contact_iterate_at_lean = _make_contact_iterate_at(cloth_support=False, has_mass_splitting=False, use_bias=True)
contact_relax_at_lean = _make_contact_iterate_at(cloth_support=False, has_mass_splitting=False, use_bias=False)
contact_iterate_at_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=True, has_soft_contact_pd=False
)
contact_relax_at_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=False, has_soft_contact_pd=False
)
contact_iterate_at_lean_no_soft_pd_mobility = _make_contact_iterate_at(
    cloth_support=False,
    has_mass_splitting=False,
    use_bias=True,
    has_soft_contact_pd=False,
    tangent_mobility_layout=True,
)
contact_relax_at_lean_no_soft_pd_mobility = _make_contact_iterate_at(
    cloth_support=False,
    has_mass_splitting=False,
    use_bias=False,
    has_soft_contact_pd=False,
    tangent_mobility_layout=True,
)
contact_iterate_at_lean_mobility = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=True, tangent_mobility_layout=True
)
contact_relax_at_lean_mobility = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=False, tangent_mobility_layout=True
)
contact_iterate_at_patch_lean = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=True, patch_friction=True
)
contact_relax_at_patch_lean = _make_contact_iterate_at(
    cloth_support=False, has_mass_splitting=False, use_bias=False, patch_friction=True
)
contact_iterate_at_patch_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False,
    has_mass_splitting=False,
    use_bias=True,
    has_soft_contact_pd=False,
    patch_friction=True,
)
contact_relax_at_patch_lean_no_soft_pd = _make_contact_iterate_at(
    cloth_support=False,
    has_mass_splitting=False,
    use_bias=False,
    has_soft_contact_pd=False,
    patch_friction=True,
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
def contact_prepare_for_iteration_packed_rows(
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
    source_first = contact_get_original_contact_first(constraints, cid)
    contact_prepare_for_iteration_at_packed_rows(
        constraints,
        cid,
        source_first,
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
def contact_prepare_for_iteration_packed_rows_lean(
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
    source_first = contact_get_original_contact_first(constraints, cid)
    contact_prepare_for_iteration_at_packed_rows_lean(
        constraints,
        cid,
        source_first,
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
def contact_prepare_for_iteration_packed_rows_lean_no_soft_pd(
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
    source_first = contact_get_original_contact_first(constraints, cid)
    contact_prepare_for_iteration_at_packed_rows_lean_no_soft_pd(
        constraints,
        cid,
        source_first,
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
def contact_prepare_for_iteration_packed_rows_no_soft_pd(
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
    source_first = contact_get_original_contact_first(constraints, cid)
    contact_prepare_for_iteration_at_packed_rows_no_soft_pd(
        constraints,
        cid,
        source_first,
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
def contact_prepare_for_iteration_patch_lean(
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
    contact_prepare_for_iteration_at_patch_lean(
        constraints, cid, 0, bodies, particles, num_bodies, body_pair, idt, cc, contacts, copy_state, parallel_id
    )


@wp.func
def contact_prepare_for_iteration_patch_lean_no_soft_pd(
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
    contact_prepare_for_iteration_at_patch_lean_no_soft_pd(
        constraints, cid, 0, bodies, particles, num_bodies, body_pair, idt, cc, contacts, copy_state, parallel_id
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
def _contact_iterate_guard_allows(
    bodies: BodyContainer,
    b1: wp.int32,
    b2: wp.int32,
    num_bodies: wp.int32,
) -> wp.bool:
    # Contacts produced before a sleep transition can survive for one step.
    # The four body fields load together from clamped indices instead of in
    # a chain of short-circuit branches; out-of-range endpoints ignore them.
    last = wp.max(num_bodies - 1, 0)
    i1 = wp.clamp(b1, 0, last)
    i2 = wp.clamp(b2, 0, last)
    # A body is frozen when it is not dynamic or its island sleeps (root >= 0).
    frozen1 = wp.max(bodies.island_root[i1], wp.where(bodies.motion_type[i1] != MOTION_DYNAMIC, 0, -1))
    frozen2 = wp.max(bodies.island_root[i2], wp.where(bodies.motion_type[i2] != MOTION_DYNAMIC, 0, -1))
    in_range = b1 >= 0 and b1 < num_bodies and b2 >= 0 and b2 < num_bodies
    return not (in_range and wp.min(frozen1, frozen2) >= 0)


def _make_contact_iterate_entry(*, has_sleeping: bool, has_soft_contact_pd: bool, staged_body_properties: bool):
    if staged_body_properties:
        iterate_at = contact_iterate_at_packed if has_soft_contact_pd else contact_iterate_at_packed_no_soft_pd
        relax_at = contact_relax_at_packed if has_soft_contact_pd else contact_relax_at_packed_no_soft_pd
    else:
        iterate_at = contact_iterate_at if has_soft_contact_pd else contact_iterate_at_no_soft_pd
        relax_at = contact_relax_at if has_soft_contact_pd else contact_relax_at_no_soft_pd

    @wp.func
    def impl(
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
        if wp.static(has_sleeping):
            if not _contact_iterate_guard_allows(bodies, b1, b2, num_bodies):
                return
        body_pair = constraint_bodies_make(b1, b2)
        if use_bias:
            iterate_at(
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
            relax_at(
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

    return impl


contact_iterate = _make_contact_iterate_entry(has_sleeping=True, has_soft_contact_pd=True, staged_body_properties=False)
contact_iterate_no_soft_pd = _make_contact_iterate_entry(
    has_sleeping=True, has_soft_contact_pd=False, staged_body_properties=False
)
contact_iterate_no_sleep = _make_contact_iterate_entry(
    has_sleeping=False, has_soft_contact_pd=True, staged_body_properties=False
)
contact_iterate_no_sleep_no_soft_pd = _make_contact_iterate_entry(
    has_sleeping=False, has_soft_contact_pd=False, staged_body_properties=False
)
contact_iterate_packed = _make_contact_iterate_entry(
    has_sleeping=True, has_soft_contact_pd=True, staged_body_properties=True
)
contact_iterate_packed_no_soft_pd = _make_contact_iterate_entry(
    has_sleeping=True, has_soft_contact_pd=False, staged_body_properties=True
)
contact_iterate_packed_no_sleep = _make_contact_iterate_entry(
    has_sleeping=False, has_soft_contact_pd=True, staged_body_properties=True
)
contact_iterate_packed_no_sleep_no_soft_pd = _make_contact_iterate_entry(
    has_sleeping=False, has_soft_contact_pd=False, staged_body_properties=True
)


@wp.func
def contact_iterate_patch_lean_no_sleep(
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
        contact_iterate_at_patch_lean(
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
        contact_relax_at_patch_lean(
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
def contact_iterate_patch_lean_no_sleep_no_soft_pd(
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
        contact_iterate_at_patch_lean_no_soft_pd(
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
        contact_relax_at_patch_lean_no_soft_pd(
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
def contact_iterate_patch_multi(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    sweep = wp.int32(0)
    while sweep < num_sweeps:
        contact_iterate_patch_lean_no_sleep(
            constraints,
            cid,
            bodies,
            particles,
            num_bodies,
            idt,
            cc,
            contacts,
            use_bias,
            copy_state,
            parallel_id,
            sor_boost,
        )
        sweep += wp.int32(1)


@wp.func
def contact_iterate_patch_multi_no_soft_pd(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    copy_state: CopyStateContainer,
    parallel_id: wp.int32,
    sor_boost: wp.float32,
):
    sweep = wp.int32(0)
    while sweep < num_sweeps:
        contact_iterate_patch_lean_no_sleep_no_soft_pd(
            constraints,
            cid,
            bodies,
            particles,
            num_bodies,
            idt,
            cc,
            contacts,
            use_bias,
            copy_state,
            parallel_id,
            sor_boost,
        )
        sweep += wp.int32(1)


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


@functools.cache
def _make_contact_iterate_lean(
    *,
    has_sleeping: bool,
    has_soft_contact_pd: bool,
    tangent_mobility_layout: bool,
    moving_surface_only: bool = False,
    check_owner: bool = False,
):
    """Build a rigid column sweep that skips frozen columns behind one branch.

    With ``check_owner`` the same branch also skips columns that another
    solver owns, so the column fields and then the body fields of both tests
    load as two batches instead of a chain.
    """
    if moving_surface_only:
        options = {
            "cloth_support": False,
            "has_mass_splitting": False,
            "has_soft_contact_pd": False,
            "moving_surface_only": True,
            "tangent_mobility_layout": tangent_mobility_layout,
        }
        iterate_at = _make_contact_iterate_at(use_bias=True, **options)
        relax_at = _make_contact_iterate_at(use_bias=False, **options)
    elif has_soft_contact_pd:
        iterate_at = contact_iterate_at_lean_mobility if tangent_mobility_layout else contact_iterate_at_lean
        relax_at = contact_relax_at_lean_mobility if tangent_mobility_layout else contact_relax_at_lean
    else:
        iterate_at = (
            contact_iterate_at_lean_no_soft_pd_mobility
            if tangent_mobility_layout
            else contact_iterate_at_lean_no_soft_pd
        )
        relax_at = (
            contact_relax_at_lean_no_soft_pd_mobility if tangent_mobility_layout else contact_relax_at_lean_no_soft_pd
        )

    @wp.func
    def impl(
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
        owned = False
        if wp.static(check_owner):
            owned = contact_owned_elsewhere(constraints, cid)
        allowed = True
        if wp.static(has_sleeping):
            allowed = _contact_iterate_guard_allows(bodies, b1, b2, num_bodies)
        if owned or not allowed:
            return
        body_pair = constraint_bodies_make(b1, b2)
        if use_bias:
            iterate_at(
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
            relax_at(
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

    return impl


contact_iterate_lean_no_sleep_no_soft_pd = _make_contact_iterate_lean(
    has_sleeping=False, has_soft_contact_pd=False, tangent_mobility_layout=False
)
contact_iterate_lean_no_sleep_no_soft_pd_mobility = _make_contact_iterate_lean(
    has_sleeping=False, has_soft_contact_pd=False, tangent_mobility_layout=True
)
contact_iterate_lean_no_sleep_mobility = _make_contact_iterate_lean(
    has_sleeping=False, has_soft_contact_pd=True, tangent_mobility_layout=True
)
contact_iterate_lean_no_soft_pd_mobility = _make_contact_iterate_lean(
    has_sleeping=True, has_soft_contact_pd=False, tangent_mobility_layout=True
)
contact_iterate_lean_mobility = _make_contact_iterate_lean(
    has_sleeping=True, has_soft_contact_pd=True, tangent_mobility_layout=True
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


contact_iterate_at_lean_no_soft_pd_cooperative = _make_contact_iterate_at(
    cloth_support=False,
    has_mass_splitting=False,
    use_bias=True,
    has_soft_contact_pd=False,
    cooperative_lanes=4,
    tangent_mobility_layout=True,
)


@wp.func
def contact_iterate_lean_no_sleep_no_soft_pd_cooperative(
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
    sor_boost: wp.float32,
    lane: wp.int32,
):
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    contact_iterate_at_lean_no_soft_pd_cooperative(
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
        lane,
        sor_boost,
    )
