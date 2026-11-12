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
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer
from newton._src.solvers.phoenx.constraints.contact_tgs import ContactTGS, get_solve_contact_rows_tgs
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

_NO_KEY = wp.constant(wp.int64(9223372036854775807))


@wp.kernel(enable_backward=False)
def _static_column_keys(
    columns: ContactColumnContainer,
    active: wp.array[int],
    keys: wp.array[wp.int64],
    values: wp.array[int],
):
    """Key each active static-owned column by ``(owner, column)``; others sort last."""
    cid = wp.tid()
    key = _NO_KEY
    if cid < active[0]:
        owner = columns.static_owner[cid]
        if owner >= 0:
            key = (wp.int64(owner) << wp.int64(32)) | wp.int64(cid)
    keys[cid] = key
    values[cid] = cid


@wp.kernel(enable_backward=False)
def _clear_static_heads(heads: wp.array[int]):
    heads[wp.tid()] = -1


@wp.kernel(enable_backward=False)
def _link_static_columns(
    active: wp.array[int],
    keys: wp.array[wp.int64],
    values: wp.array[int],
    heads: wp.array[int],
    links: wp.array[int],
):
    """Turn each owner's sorted run of columns into its ordered list."""
    position = wp.tid()
    if position >= active[0] or keys[position] == _NO_KEY:
        return
    owner = int(keys[position] >> wp.int64(32))
    cid = values[position]
    if position == 0 or int(keys[position - 1] >> wp.int64(32)) != owner:
        heads[owner] = cid
    following = -1
    if position + 1 < active[0] and keys[position + 1] != _NO_KEY:
        if int(keys[position + 1] >> wp.int64(32)) == owner:
            following = values[position + 1]
    links[cid] = following


def build_static_contact_lists(columns, active, heads, links, keys, values, device) -> None:
    """Order each dynamic body's static-owned contact columns into a list.

    Sorting ``(owner, column)`` keys keeps the work linear in the active
    columns and the order deterministic; ``keys`` and ``values`` are radix
    scratch of twice the column capacity.
    """
    capacity = links.shape[0]
    wp.launch(_static_column_keys, capacity, [columns, active, keys, values], device=device)
    wp.utils.radix_sort_pairs(keys, values, capacity)
    wp.launch(_clear_static_heads, heads.shape[0], [heads], device=device)
    wp.launch(_link_static_columns, capacity, [active, keys, values, heads, links], device=device)


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


@functools.cache
def get_average_and_soft_sweep(phase: str, *, soft_pd: bool, sleeping: bool, patch: bool):
    """Average each rigid body's copies, then solve its static contacts on its full mass.

    A static partner has infinite mass, so these rows need no mass-split copy.
    Eight lanes average a body's copies; the first lane then solves the body's
    static columns in their original order on that average (Gauss-Seidel), and
    the lanes write the result back to every copy. The impulse comes from the
    static world, so splitting the dynamic constraints keeps conserving their
    momentum. Replaces the rigid velocity-only copy average after a sweep;
    ``cached_prepare`` only applies the warm start of rows whose geometry the
    row-parallel prepare built.
    """
    from newton._src.solvers.phoenx.constraints import constraint_contact_solve as solve  # noqa: PLC0415
    from newton._src.solvers.phoenx.constraints.contact_tgs_cooperative import shuffle_vec  # noqa: PLC0415
    from newton._src.solvers.phoenx.mass_splitting.kernels import (  # noqa: PLC0415
        subgroup_average8_velocity,
        subgroup_broadcast8_velocity,
    )

    if phase not in ("prepare", "cached_prepare", "iterate", "relax"):
        raise ValueError("Expected prepare, cached_prepare, iterate, or relax phase")
    suffix = "" if soft_pd else "_no_soft_pd"
    if phase == "cached_prepare":
        # The row-parallel prepare already built these rows' geometry.
        if patch or sleeping:
            raise ValueError("Cached preparation requires sleepless point-friction contacts")
        prepare = solve.contact_cached_warmstart_lean
        iterate = None
    elif patch:
        prepare = getattr(solve, "contact_prepare_for_iteration_patch_lean" + suffix)
        iterate = getattr(solve, "contact_iterate_patch_lean_no_sleep" + suffix)
    else:
        prepare = getattr(solve, "contact_prepare_for_iteration_lean" + suffix)
        iterate = getattr(solve, ("contact_iterate_lean" if sleeping else "contact_iterate_lean_no_sleep") + suffix)

    @wp.kernel(enable_backward=False, module="unique")
    def average_and_sweep(
        columns: ContactColumnContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        num_bodies: wp.int32,
        cc: ContactContainer,
        contacts: ContactViews,
        copies: CopyStateContainer,
        heads: wp.array[int],
        links: wp.array[int],
        idt: float,
        sor_boost: float,
    ):
        tid = wp.tid()
        body = tid // 8
        lane = tid & 7
        if body >= num_bodies:
            return
        cid = heads[body]
        count = copies.count_per_node[body]
        if cid < 0 and count <= 1:
            return
        start = int(0)
        if body > 0:
            start = copies.section_end[body - 1]
        end = start + count
        v = wp.vec3f(0.0)
        w = wp.vec3f(0.0)
        if count > 0:
            v, w = subgroup_average8_velocity(copies, start, end, lane, tid)
        if cid >= 0:
            if lane == 0:
                if count > 0:
                    bodies.velocity[body] = v
                    bodies.angular_velocity[body] = w
                while cid >= 0:
                    if wp.static(phase in ("prepare", "cached_prepare")):
                        prepare(columns, cid, bodies, particles, num_bodies, idt, cc, contacts, copies, 0)
                    else:
                        iterate(
                            columns,
                            cid,
                            bodies,
                            particles,
                            num_bodies,
                            idt,
                            cc,
                            contacts,
                            wp.bool(wp.static(phase == "iterate")),
                            copies,
                            0,
                            sor_boost,
                        )
                    cid = links[cid]
                v = bodies.velocity[body]
                w = bodies.angular_velocity[body]
            if count == 0:
                return
            v = shuffle_vec(v, 0, 8)
            w = shuffle_vec(w, 0, 8)
        subgroup_broadcast8_velocity(copies, start, end, lane, v, w)

    return average_and_sweep
