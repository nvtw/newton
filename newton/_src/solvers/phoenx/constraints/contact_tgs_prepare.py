# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepare independent normal rows in parallel, then correlate each patch group."""

import warp as wp

from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_orientation,
    mat33_from_sym6,
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
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_r0,
    cc_set_r1,
)
from newton._src.solvers.phoenx.constraints.contact_static_ownership import static_owner
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, NormalRow, prepare_contact_tgs
from newton._src.solvers.phoenx.constraints.contact_tgs_friction import FrictionRow
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import partition_range
from newton._src.solvers.phoenx.constraints.contact_tgs_partition_cuda import CACHE_CAPACITY, partition_cached
from newton._src.solvers.phoenx.helpers.math_helpers import effective_mass_scalar


@wp.kernel(enable_backward=False)
def geometry(
    columns: ContactColumnContainer,
    state: ContactTGS,
    active: wp.array[int],
    bodies: BodyContainer,
    cc: ContactContainer,
    views: ContactViews,
    idt: float,
):
    """Rebase rigid normal rows without resetting or reapplying temporal impulses.

    Launch with a second dimension of 128. Dynamic pairs use their split
    response; static contacts use the physical response. Both lever arms
    refer to a common point, preserving paired angular impulse balance.
    """
    cid, lane = wp.tid()
    if cid < active[0]:
        first = contact_get_contact_first(columns, cid)
        count = contact_get_contact_count(columns, cid)
        a = contact_get_body1(columns, cid)
        b = contact_get_body2(columns, cid)
        orientation1 = body_load_orientation(bodies, a)
        orientation2 = body_load_orientation(bodies, b)
        position1 = bodies.position[a]
        position2 = bodies.position[b]
        body_com1 = bodies.body_com[a]
        body_com2 = bodies.body_com[b]
        if static_owner(columns, cid, bodies) >= 0:
            inv_mass1 = bodies.inverse_mass[a]
            inv_mass2 = bodies.inverse_mass[b]
            inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, a))
            inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b))
        else:
            factor1 = wp.float32(contact_get_count1(columns, cid))
            factor2 = wp.float32(contact_get_count2(columns, cid))
            inv_mass1 = bodies.inverse_mass[a] * factor1
            inv_mass2 = bodies.inverse_mass[b] * factor2
            inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, a)) * factor1
            inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b)) * factor2
        for offset in range(lane, count, 128):
            k = first + offset
            n = cc_get_normal(cc, k)
            local_p0 = views.rigid_contact_point0[k]
            local_p1 = views.rigid_contact_point1[k]
            margin0 = views.rigid_contact_margin0[k]
            margin1 = views.rigid_contact_margin1[k]
            p0_world = position1 + wp.quat_rotate(orientation1, local_p0 - body_com1) + margin0 * n
            p1_world = position2 + wp.quat_rotate(orientation2, local_p1 - body_com2) - margin1 * n
            impulse_point = wp.float32(0.5) * (p0_world + p1_world)
            r1 = impulse_point - position1
            r2 = impulse_point - position2
            eff_n = effective_mass_scalar(n, r1, r2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2)
            effective_gap = wp.dot(p1_world - p0_world, n)
            if effective_gap > wp.float32(0.0):
                bias = effective_gap * idt
            else:
                bias = effective_gap * state.gain * idt
            # Preserve the existing solver's approach/depenetration speed limits.
            bias = wp.clamp(bias, wp.float32(-2.0), wp.float32(10.0))
            cc_set_eff_n(cc, k, eff_n)
            cc_set_eff_t1(cc, k, wp.float32(0.0))
            cc_set_eff_t2(cc, k, wp.float32(0.0))
            cc_set_bias(cc, k, bias)
            cc_set_bias_t1(cc, k, wp.float32(0.0))
            cc_set_bias_t2(cc, k, wp.float32(0.0))
            cc_set_r0(cc, k, r1)
            cc_set_r1(cc, k, r2)
            row = NormalRow()
            row.normal = n
            row.r0 = r1
            row.r1 = r2
            row.effective_mass = eff_n
            row.bias = bias
            state.normal_rows[k] = row
            state.normals[k] = n


@wp.kernel(enable_backward=False)
def patches(
    columns: ContactColumnContainer,
    state: ContactTGS,
    active: wp.array[int],
    bodies: BodyContainer,
    cc: ContactContainer,
):
    cid = wp.tid()
    if cid < active[0]:
        first = contact_get_contact_first(columns, cid)
        count = contact_get_contact_count(columns, cid)
        if count > 0:
            refresh = state.last[first] != state.generation[0]
            prepare_contact_tgs(
                state,
                cc,
                first,
                count,
                bodies,
                contact_get_body1(columns, cid),
                contact_get_body2(columns, cid),
                contact_get_friction(columns, cid),
                contact_get_friction_dynamic(columns, cid),
            )
            if refresh:
                patch = state.current.group_first[first]
                while patch >= 0:
                    state.point_last[patch] = state.generation[0]
                    state.patch_column[patch] = cid
                    patch = state.current.patch_next[patch]


@wp.kernel(enable_backward=False)
def partition_groups(columns: ContactColumnContainer, state: ContactTGS, active: wp.array[int]):
    """Partition each active generation once; launch with dimensions (64, 128)."""
    block, lane = wp.tid()
    for cid in range(block, active[0], 64):
        first = contact_get_contact_first(columns, cid)
        count = contact_get_contact_count(columns, cid)
        if count > 0 and state.partition_last[first] != state.generation[0]:
            if count <= CACHE_CAPACITY:
                partition_cached(first, first, count, state.normals, 0.999, state.current, lane)
            elif lane == 0:
                partition_range(first, first, count, state.normals, 0.999, state.current)
            if lane == 0:
                state.partition_last[first] = state.generation[0]


@wp.kernel(enable_backward=False)
def friction_geometry(columns: ContactColumnContainer, state: ContactTGS, bodies: BodyContainer):
    patch, anchor = wp.tid()
    if state.point_last[patch] != state.generation[0] or anchor >= state.anchors.count[patch]:
        return
    cid = state.patch_column[patch]
    a = contact_get_body1(columns, cid)
    b = contact_get_body2(columns, cid)
    scale0 = wp.float32(contact_get_count1(columns, cid))
    scale1 = wp.float32(contact_get_count2(columns, cid))
    if static_owner(columns, cid, bodies) >= 0:
        scale0 = 1.0
        scale1 = 1.0
    m0 = bodies.inverse_mass[a] * scale0
    m1 = bodies.inverse_mass[b] * scale1
    i0 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, a)) * scale0
    i1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b)) * scale1
    pose0 = wp.transformf(bodies.position[a], body_load_orientation(bodies, a))
    pose1 = wp.transformf(bodies.position[b], body_load_orientation(bodies, b))
    normal = state.current.patch_normal[patch]
    t0 = wp.vec3f(0.0, -normal[2], normal[1])
    if wp.abs(normal[0]) >= 0.70710678:
        t0 = wp.vec3f(-normal[1], normal[0], 0.0)
    t0 = wp.normalize(t0)
    t1 = wp.cross(normal, t0)
    point0 = wp.transform_point(pose0, state.anchors.local0[patch, anchor])
    point1 = wp.transform_point(pose1, state.anchors.local1[patch, anchor])
    common = 0.5 * (point0 + point1)
    r0 = common - wp.transform_get_translation(pose0)
    r1 = common - wp.transform_get_translation(pose1)
    a0 = wp.cross(r0, t0)
    a1 = wp.cross(r1, t0)
    b0 = wp.cross(r0, t1)
    b1 = wp.cross(r1, t1)
    row = FrictionRow()
    row.t0 = t0
    row.t1 = t1
    row.r0 = r0
    row.r1 = r1
    row.error = point1 - point0
    row.response0 = m0 + m1 + wp.dot(a0, i0 @ a0) + wp.dot(a1, i1 @ a1)
    row.response1 = m0 + m1 + wp.dot(b0, i0 @ b0) + wp.dot(b1, i1 @ b1)
    state.friction_rows[patch, anchor] = row
