# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dispatch temporal rigid contact rows through physical or split body state."""

import functools

import warp as wp

from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_vw,
    body_store_vw,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_count1,
    contact_get_count2,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_slot1,
    contact_get_slot2,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, get_solve_contact_rows_tgs
from newton._src.solvers.phoenx.constraints.contact_tgs_cooperative import get_solve_rows_cooperative
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer


@functools.cache
def make_iterate(*, mass_splitting: bool, biased: bool, cooperative: bool = False, record_wrenches: bool = False):
    """Specialize rigid contact dispatch without adding branches to other solvers.

    Slots and copy counts must be stamped by the constraint graph. Static
    contacts belong to the separate physical-body sweep. After solving dynamic
    constraints, the caller averages each body's copies before integration.
    """

    solve_contact_rows_tgs = get_solve_contact_rows_tgs(record_wrenches)
    solve_rows_cooperative = get_solve_rows_cooperative(record_wrenches)

    @wp.func
    def iterate(
        columns: ContactColumnContainer,
        state: ContactTGS,
        cid: int,
        bodies: BodyContainer,
        cc: ContactContainer,
        copies: CopyStateContainer,
        idt: float,
        lane: int,
    ):
        size = contact_get_contact_count(columns, cid)
        if size == 0:
            return
        a = contact_get_body1(columns, cid)
        b = contact_get_body2(columns, cid)
        slot0 = int(-1)
        slot1 = int(-1)
        if wp.static(mass_splitting):
            slot0 = contact_get_slot1(columns, cid)
            slot1 = contact_get_slot2(columns, cid)
            factor0 = wp.float32(contact_get_count1(columns, cid))
            factor1 = wp.float32(contact_get_count2(columns, cid))
            if slot0 < 0:
                v0 = bodies.velocity[a]
                w0 = bodies.angular_velocity[a]
            else:
                v0 = copies.velocity[slot0]
                w0 = copies.angular_velocity[slot0]
            if slot1 < 0:
                v1 = bodies.velocity[b]
                w1 = bodies.angular_velocity[b]
            else:
                v1 = copies.velocity[slot1]
                w1 = copies.angular_velocity[slot1]
            m0 = bodies.inverse_mass[a] * factor0
            m1 = bodies.inverse_mass[b] * factor1
            i0 = mat33_from_sym6(bodies.inverse_inertia_world[a]) * factor0
            i1 = mat33_from_sym6(bodies.inverse_inertia_world[b]) * factor1
        else:
            v0, w0 = body_load_vw(bodies, a)
            v1, w1 = body_load_vw(bodies, b)
            m0 = bodies.inverse_mass[a]
            m1 = bodies.inverse_mass[b]
            i0 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, a))
            i1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b))
        if wp.static(cooperative):
            v0, v1, w0, w1 = solve_rows_cooperative(
                state,
                cc,
                contact_get_contact_first(columns, cid),
                size,
                bodies,
                a,
                b,
                v0,
                v1,
                w0,
                w1,
                m0,
                m1,
                i0,
                i1,
                contact_get_friction(columns, cid),
                contact_get_friction_dynamic(columns, cid),
                idt,
                wp.bool(wp.static(biased)),
                lane,
            )
            if lane != 0:
                return
        else:
            v0, v1, w0, w1 = solve_contact_rows_tgs(
                state,
                cc,
                contact_get_contact_first(columns, cid),
                size,
                bodies,
                a,
                b,
                v0,
                v1,
                w0,
                w1,
                m0,
                m1,
                i0,
                i1,
                contact_get_friction(columns, cid),
                contact_get_friction_dynamic(columns, cid),
                idt,
                wp.bool(wp.static(biased)),
            )
        if wp.static(not mass_splitting) or (slot0 < 0 and slot1 < 0):
            body_store_vw(bodies, a, v0, w0)
            body_store_vw(bodies, b, v1, w1)
        else:
            if slot0 < 0:
                bodies.velocity[a] = v0
                bodies.angular_velocity[a] = w0
            else:
                copies.velocity[slot0] = v0
                copies.angular_velocity[slot0] = w0
            if slot1 < 0:
                bodies.velocity[b] = v1
                bodies.angular_velocity[b] = w1
            else:
                copies.velocity[slot1] = v1
                copies.angular_velocity[slot1] = w1

    return iterate
