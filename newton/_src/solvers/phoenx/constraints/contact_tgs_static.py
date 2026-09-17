# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Solve static contacts on physical bodies after averaging dynamic constraints."""

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
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.constraints.contact_static_ownership import static_owner
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, get_solve_contact_rows_tgs
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer


@wp.kernel
def build_lists(
    columns: ContactColumnContainer,
    bodies: BodyContainer,
    active: wp.array[int],
    heads: wp.array[int],
    links: wp.array[int],
):
    """Build ordered lists from active contact columns, excluding dynamic pairs."""
    body = wp.tid()
    previous = int(-1)
    heads[body] = -1
    for cid in range(active[0]):
        if static_owner(columns, cid, bodies) == body:
            if previous < 0:
                heads[body] = cid
            else:
                links[previous] = cid
            links[cid] = -1
            previous = cid


@functools.cache
def get_sweep(phase: str, *, record_wrenches: bool = False):
    """Build a physical-body solve followed by broadcast to its averaged copies.

    Call only after averaging dynamic constraints. Preparation copies the
    average without reapplying retained impulses; iterate and relax process
    each static column once in its original order. Static endpoints are read
    only, so multiple dynamic owners may safely share the same static body.
    """
    if phase not in ("prepare", "iterate", "relax"):
        raise ValueError("Expected prepare, iterate, or relax phase")

    solve_contact_rows_tgs = get_solve_contact_rows_tgs(record_wrenches)

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        columns: ContactColumnContainer,
        state: ContactTGS,
        bodies: BodyContainer,
        cc: ContactContainer,
        copies: CopyStateContainer,
        heads: wp.array[int],
        links: wp.array[int],
        idt: float,
    ):
        body = wp.tid()
        cid = heads[body]
        if cid < 0:
            return
        count = copies.count_per_node[body]
        first = int(0)
        if body > 0:
            first = copies.section_end[body - 1]
        if count > 0:
            bodies.velocity[body] = copies.velocity[first]
            bodies.angular_velocity[body] = copies.angular_velocity[first]
        if wp.static(phase != "prepare"):
            while cid >= 0:
                a = contact_get_body1(columns, cid)
                b = contact_get_body2(columns, cid)
                v0, w0 = body_load_vw(bodies, a)
                v1, w1 = body_load_vw(bodies, b)
                m0 = bodies.inverse_mass[a]
                m1 = bodies.inverse_mass[b]
                i0 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, a))
                i1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b))
                v0, v1, w0, w1 = solve_contact_rows_tgs(
                    state,
                    cc,
                    contact_get_contact_first(columns, cid),
                    contact_get_contact_count(columns, cid),
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
                    wp.bool(wp.static(phase == "iterate")),
                )
                if a == body:
                    body_store_vw(bodies, body, v0, w0)
                else:
                    body_store_vw(bodies, body, v1, w1)
                cid = links[cid]
        for slot in range(first, first + count):
            copies.velocity[slot] = bodies.velocity[body]
            copies.angular_velocity[slot] = bodies.angular_velocity[body]

    return sweep
