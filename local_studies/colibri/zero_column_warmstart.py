# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Skip exact-zero warm impulses while preserving accumulation order."""

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    BodyContainer,
    ContactColumnContainer,
    ContactContainer,
    ContactViews,
    CopyStateContainer,
    ParticleContainer,
    apply_pair_spatial_impulse,
    body_load_inv_inertia_sym6,
    body_load_vw,
    body_store_vw,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_r0,
    cc_get_r1,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_count1,
    contact_get_count2,
    contact_get_slot1,
    contact_get_slot2,
    mat33_from_sym6,
    write_angular_velocity_unified,
    write_velocity_unified,
)


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

    has_impulse = bool(False)
    for i in range(contact_count):
        k = contact_first + i
        if (
            cc_get_normal_lambda(cc, k) != 0.0
            or cc_get_tangent1_lambda(cc, k) != 0.0
            or cc_get_tangent2_lambda(cc, k) != 0.0
        ):
            has_impulse = True
            break
    if not has_impulse:
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


def install():
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels._contact_cached_warmstart_split = _contact_cached_warmstart_split
