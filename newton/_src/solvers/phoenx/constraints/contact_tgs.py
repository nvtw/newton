# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Persistent TGS contact state and paired friction for compatible rigid groups.

The caller supplies unique ordered-body/material groups and owns normal-row
preparation, physical mass-copy scaling, and the outer-step generation. Normal
and tangent multipliers remain applied throughout that outer step. Material
anchors persist through geometric correlation; multiplier warm starts do not.
"""

import functools

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer, body_load_orientation
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_eff_n,
    cc_get_normal,
    cc_get_normal_lambdas,
    cc_get_r0,
    cc_get_r1,
    cc_get_start_gap,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_normal_velocity_update_no_soft_pd
from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import PatchAnchors, prepare_group
from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import allocate as allocate_anchors
from newton._src.solvers.phoenx.constraints.contact_tgs_friction import FrictionRow, solve_cached
from newton._src.solvers.phoenx.constraints.contact_tgs_friction import solve as solve_patch
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import NormalPatches, partition_range
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import allocate as allocate_partition
from newton._src.solvers.phoenx.helpers.data_packing import reinterpret_float_as_int
from newton._src.solvers.phoenx.helpers.math_helpers import apply_pair_velocity_impulse


@wp.struct
class NormalRow:
    normal: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    effective_mass: float
    bias: float


@wp.struct
class ContactTGS:
    record_wrenches: int
    wrenches: wp.array[wp.spatial_vector]
    prepared_friction: int
    friction_rows: wp.array2d[FrictionRow]
    patch_column: wp.array[int]
    body_count: int
    gain: float
    generation: wp.array[int]
    point_last: wp.array[int]
    partition_last: wp.array[int]
    last: wp.array[int]
    keys: wp.array[wp.vec4i]
    previous_keys: wp.array[wp.vec4i]
    previous_active: wp.array[int]
    current_group_count: wp.array[int]
    current_groups: wp.array[int]
    previous_groups: wp.array[int]
    solve_first: wp.array[int]
    solve_next: wp.array[int]
    current: NormalPatches
    previous: NormalPatches
    anchors: PatchAnchors
    previous_anchors: PatchAnchors
    normals: wp.array[wp.vec3f]
    points: wp.array[wp.vec3f]
    gaps: wp.array[float]
    loads: wp.array[float]
    impulse: wp.array2d[wp.vec3f]
    normal_rows: wp.array[NormalRow]


def allocate_contact_tgs(capacity: int, body_count: int, substeps: int, device: wp.DeviceLike) -> ContactTGS:
    """Allocate persistent patch state using the caller's contact-point capacity."""
    if capacity < 0 or substeps < 1:
        raise ValueError("Contact capacity must be nonnegative and substeps positive")
    state = ContactTGS()
    state.wrenches = wp.zeros(capacity, dtype=wp.spatial_vector, device=device)
    state.body_count = body_count
    state.gain = min(0.8, 2 / substeps**0.5)
    state.generation = wp.zeros(1, dtype=int, device=device)
    state.point_last = wp.full(capacity, -1, dtype=int, device=device)
    state.partition_last = wp.full(capacity, -1, dtype=int, device=device)
    state.last = wp.full(capacity, -1, dtype=int, device=device)
    state.keys = wp.full(capacity, wp.vec4i(-1), dtype=wp.vec4i, device=device)
    state.previous_keys = wp.full(capacity, wp.vec4i(-1), dtype=wp.vec4i, device=device)
    state.previous_active = wp.zeros(1, dtype=int, device=device)
    state.current_group_count = wp.zeros(1, dtype=int, device=device)
    state.current_groups = wp.zeros(capacity, dtype=int, device=device)
    state.previous_groups = wp.zeros(capacity, dtype=int, device=device)
    state.solve_first = wp.full(capacity, -1, dtype=int, device=device)
    state.solve_next = wp.full(capacity, -1, dtype=int, device=device)
    state.current = allocate_partition(capacity, capacity, device)
    state.previous = allocate_partition(capacity, capacity, device)
    state.anchors = allocate_anchors(capacity, device)
    state.previous_anchors = allocate_anchors(capacity, device)
    state.normals = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    state.points = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    state.gaps = wp.zeros(capacity, dtype=float, device=device)
    state.loads = wp.zeros(capacity, dtype=float, device=device)
    state.impulse = wp.zeros((capacity, 2), dtype=wp.vec3f, device=device)
    state.normal_rows = wp.zeros(capacity, dtype=NormalRow, device=device)
    state.friction_rows = wp.zeros((capacity, 2), dtype=FrictionRow, device=device)
    state.patch_column = wp.zeros(capacity, dtype=int, device=device)
    return state


def snapshot_contact_tgs(state: ContactTGS) -> None:
    """Preserve material references and clear multipliers once per outer step."""
    for field in (
        "point_patch",
        "point_next",
        "patch_first",
        "patch_last",
        "patch_count",
        "patch_next",
        "patch_normal",
        "group_first",
        "group_count",
    ):
        wp.copy(getattr(state.previous, field), getattr(state.current, field))
    for field in ("count", "broken", "source", "normal0", "normal1", "local0", "local1"):
        wp.copy(getattr(state.previous_anchors, field), getattr(state.anchors, field))
    wp.copy(state.previous_groups, state.current_groups)
    wp.copy(state.previous_active, state.current_group_count)
    state.current_group_count.zero_()
    wp.copy(state.previous_keys, state.keys)
    state.keys.fill_(wp.vec4i(-1))
    state.impulse.zero_()
    if state.record_wrenches:
        state.wrenches.zero_()


@wp.kernel
def advance_contact_tgs_generation(state: ContactTGS):
    state.generation[0] += 1


@wp.func
def record_impulse(state: ContactTGS, point: int, common: wp.vec3f, impulse: wp.vec3f):
    # Store the impulse on endpoint 1 and its moment about the world origin.
    # Summing deltas preserves the wrench when the application point moves.
    if state.record_wrenches != 0:
        state.wrenches[point] += wp.spatial_vector(impulse, wp.cross(common, impulse))


@wp.func
def eligible(state: ContactTGS, cc: ContactContainer, first: int, count: int, a: int, b: int):
    return count > 0


@wp.func
def prepare_contact_tgs(
    state: ContactTGS,
    cc: ContactContainer,
    first: int,
    size: int,
    bodies: BodyContainer,
    a: int,
    b: int,
    mu_s: float,
    mu_d: float,
):
    if state.last[first] != state.generation[0]:
        state.keys[first] = wp.vec4i(a, b, reinterpret_float_as_int(mu_s), reinterpret_float_as_int(mu_d))
        state.solve_first[first] = -1
        group_slot = wp.atomic_add(state.current_group_count, 0, 1)
        state.current_groups[group_slot] = first
        # Normal rows remain active. Zero friction needs no patch history;
        # the material key prevents reuse if friction returns next generation.
        if mu_s == 0.0 and mu_d == 0.0:
            state.current.group_first[first] = -1
            state.current.group_count[first] = 0
            state.partition_last[first] = state.generation[0]
            state.last[first] = state.generation[0]
            return wp.vec3f(0.0), wp.vec3f(0.0), wp.vec3f(0.0)
        for k in range(first, first + size):
            state.normals[k] = cc_get_normal(cc, k)
            state.points[k] = bodies.position[a] + cc_get_r0(cc, k)
            state.gaps[k] = cc_get_start_gap(cc, k)
        if state.partition_last[first] != state.generation[0]:
            partition_range(first, first, size, state.normals, 0.999, state.current)
            state.partition_last[first] = state.generation[0]
        pose0 = wp.transformf(bodies.position[a], body_load_orientation(bodies, a))
        pose1 = wp.transformf(bodies.position[b], body_load_orientation(bodies, b))
        prepare_group(
            first,
            state.current,
            state.previous,
            state.keys,
            state.previous_keys,
            state.previous_active,
            state.previous_groups,
            True,
            pose0,
            pose1,
            state.points,
            state.gaps,
            # PhysX default factors (0.025 and 0.04) at a 0.01 m length scale.
            # These are this experimental mode's SI tolerances, not universal defaults.
            0.00025,
            0.0004,
            state.previous_anchors,
            state.anchors,
        )
        tail = int(-1)
        patch = state.current.group_first[first]
        while patch >= 0:
            if state.anchors.count[patch] > 0:
                if tail < 0:
                    state.solve_first[first] = patch
                else:
                    state.solve_next[tail] = patch
                state.solve_next[patch] = -1
                tail = patch
            patch = state.current.patch_next[patch]
        state.last[first] = state.generation[0]
    return wp.vec3f(0.0), wp.vec3f(0.0), wp.vec3f(0.0)


@functools.cache
def get_solve_contact_tgs(record_wrenches: bool = False):
    """Specialize the solve so disabled force reporting has no arithmetic cost."""

    @wp.func
    def solve_contact_tgs(
        state: ContactTGS,
        cc: ContactContainer,
        first: int,
        size: int,
        bodies: BodyContainer,
        a: int,
        b: int,
        v0: wp.vec3f,
        v1: wp.vec3f,
        w0: wp.vec3f,
        w1: wp.vec3f,
        m0: float,
        m1: float,
        i0: wp.mat33f,
        i1: wp.mat33f,
        mu_s: float,
        mu_d: float,
        idt: float,
        biased: bool,
    ):
        if state.solve_first[first] < 0:
            return v0, v1, w0, w1
        normal_impulses = cc_get_normal_lambdas(cc)
        pose0 = wp.transformf(bodies.position[a], body_load_orientation(bodies, a))
        pose1 = wp.transformf(bodies.position[b], body_load_orientation(bodies, b))
        patch = state.solve_first[first]
        while patch >= 0:
            prior0 = wp.vec3f(0.0)
            prior1 = wp.vec3f(0.0)
            if wp.static(record_wrenches):
                prior0 = state.impulse[patch, 0]
                prior1 = state.impulse[patch, 1]
            if biased and state.prepared_friction != 0:
                v0, v1, w0, w1 = solve_cached(
                    state.current,
                    state.anchors,
                    patch,
                    normal_impulses,
                    state.impulse,
                    pose0,
                    pose1,
                    v0,
                    v1,
                    w0,
                    w1,
                    m0,
                    m1,
                    i0,
                    i1,
                    mu_s,
                    mu_d,
                    idt,
                    state.gain,
                    biased,
                    state.friction_rows,
                )
            else:
                v0, v1, w0, w1 = solve_patch(
                    state.current,
                    state.anchors,
                    patch,
                    normal_impulses,
                    state.impulse,
                    pose0,
                    pose1,
                    v0,
                    v1,
                    w0,
                    w1,
                    m0,
                    m1,
                    i0,
                    i1,
                    mu_s,
                    mu_d,
                    idt,
                    state.gain,
                    biased,
                )
            if wp.static(record_wrenches):
                for j in range(state.anchors.count[patch]):
                    prior = prior0
                    if j == 1:
                        prior = prior1
                    common = bodies.position[a] + state.friction_rows[patch, j].r0
                    if not biased or state.prepared_friction == 0:
                        common = 0.5 * (
                            wp.transform_point(pose0, state.anchors.local0[patch, j])
                            + wp.transform_point(pose1, state.anchors.local1[patch, j])
                        )
                    record_impulse(state, state.current.patch_first[patch], common, state.impulse[patch, j] - prior)
            patch = state.solve_next[patch]
        return v0, v1, w0, w1

    return solve_contact_tgs


solve_contact_tgs = get_solve_contact_tgs()


@functools.cache
def get_solve_contact_rows_tgs(record_wrenches: bool = False):
    """Specialize the solve so disabled force reporting has no arithmetic cost."""
    solve_contact_tgs = get_solve_contact_tgs(record_wrenches)

    @wp.func
    def solve_contact_rows_tgs(
        state: ContactTGS,
        cc: ContactContainer,
        first: int,
        size: int,
        bodies: BodyContainer,
        a: int,
        b: int,
        v0: wp.vec3f,
        v1: wp.vec3f,
        w0: wp.vec3f,
        w1: wp.vec3f,
        m0: float,
        m1: float,
        i0: wp.mat33f,
        i1: wp.mat33f,
        mu_s: float,
        mu_d: float,
        idt: float,
        biased: bool,
    ):
        """Solve every normal row in order, then its compatible friction patches.

        Cached geometry is valid during biased substep iterations. Relaxation
        reads the common-point lever arms rebased after body integration. Both
        stages apply equal-and-opposite impulses through the same spatial point.
        """
        # Fetch the next immutable row before the dependent velocity update.
        next_row = NormalRow()
        if biased and size > 0:
            next_row = state.normal_rows[first]
        for k in range(first, first + size):
            if biased:
                row = next_row
                if k + 1 < first + size:
                    next_row = state.normal_rows[k + 1]
                normal = row.normal
                r0 = row.r0
                r1 = row.r1
                effective_mass = row.effective_mass
                bias = row.bias
            else:
                normal = cc_get_normal(cc, k)
                r0 = cc_get_r0(cc, k)
                r1 = cc_get_r1(cc, k)
                effective_mass = cc_get_eff_n(cc, k)
                bias = cc_get_bias(cc, k)
            relative = v1 + wp.cross(w1, r1) - v0 - wp.cross(w0, r0)
            speed = wp.dot(relative, normal)
            if not biased:
                if bias > 0.0:
                    continue
                bias = 0.0
            impulse = contact_project_normal_velocity_update_no_soft_pd(
                cc,
                k,
                normal,
                speed,
                effective_mass,
                bias,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            if wp.static(record_wrenches):
                record_impulse(state, k, bodies.position[a] + r0, impulse)
            v0, v1, w0, w1 = apply_pair_velocity_impulse(
                v0,
                v1,
                w0,
                w1,
                m0,
                m1,
                i0,
                i1,
                r0,
                r1,
                impulse,
            )
        return solve_contact_tgs(
            state, cc, first, size, bodies, a, b, v0, v1, w0, w1, m0, m1, i0, i1, mu_s, mu_d, idt, biased
        )

    return solve_contact_rows_tgs


solve_contact_rows_tgs = get_solve_contact_rows_tgs()


@wp.kernel
def export_contact_wrenches(
    count: wp.array[int],
    state: ContactTGS,
    bodies: BodyContainer,
    shape_body: wp.array[int],
    shape0: wp.array[int],
    sort_perm: wp.array[int],
    inverse_dt: float,
    force: wp.array[wp.spatial_vector],
):
    """Export outer-step average wrenches on body0, about its current COM."""
    k = wp.tid()
    if k >= wp.min(count[0], wp.min(force.shape[0], state.wrenches.shape[0])):
        return
    out_k = sort_perm[k]
    body0 = shape_body[shape0[out_k]]
    center = bodies.position[body0]
    wrench = state.wrenches[k]
    impulse = wp.spatial_top(wrench)
    moment = wp.spatial_bottom(wrench) - wp.cross(center, impulse)
    force[out_k] = -inverse_dt * wp.spatial_vector(impulse, moment)
