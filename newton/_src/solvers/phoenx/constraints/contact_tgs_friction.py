# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Source-style two-anchor friction with paired common-point impulses.

The caller owns impulse lifetime and body-copy scaling. Material references
persist through PatchAnchors; impulse storage is zeroed once per outer step.
"""

import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_anchors import PatchAnchors
from newton._src.solvers.phoenx.constraints.contact_tgs_partition import NormalPatches
from newton._src.solvers.phoenx.helpers.math_helpers import apply_pair_velocity_impulse


@wp.struct
class FrictionRow:
    t0: wp.vec3f
    t1: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    error: wp.vec3f
    response0: float
    response1: float


@wp.func
def solve(
    patches: NormalPatches,
    anchors: PatchAnchors,
    patch: int,
    normal_impulses: wp.array[wp.float32],
    impulses: wp.array2d[wp.vec3f],
    pose0: wp.transformf,
    pose1: wp.transformf,
    v0: wp.vec3f,
    v1: wp.vec3f,
    w0: wp.vec3f,
    w1: wp.vec3f,
    inv_mass0: wp.float32,
    inv_mass1: wp.float32,
    inv_inertia0: wp.mat33f,
    inv_inertia1: wp.mat33f,
    mu_static: wp.float32,
    mu_dynamic: wp.float32,
    inverse_dt: wp.float32,
    gain: wp.float32,
    biased: bool,
):
    count = anchors.count[patch]
    if count == 0:
        anchors.broken[patch] = 0
        return v0, v1, w0, w1
    load = float(0.0)
    point = patches.patch_first[patch]
    while point >= 0:
        load += wp.max(normal_impulses[point], 0.0)
        point = patches.point_next[point]
    if count > 0:
        load /= wp.float32(count)
    normal = patches.patch_normal[patch]
    t0 = wp.vec3f(0.0, -normal[2], normal[1])
    if wp.abs(normal[0]) >= 0.70710678:
        t0 = wp.vec3f(-normal[1], normal[0], 0.0)
    t0 = wp.normalize(t0)
    t1 = wp.cross(normal, t0)
    broken = int(0)
    for j in range(count):
        point0 = wp.transform_point(pose0, anchors.local0[patch, j])
        point1 = wp.transform_point(pose1, anchors.local1[patch, j])
        common = 0.5 * (point0 + point1)
        r0 = common - wp.transform_get_translation(pose0)
        r1 = common - wp.transform_get_translation(pose1)
        a0 = wp.cross(r0, t0)
        a1 = wp.cross(r1, t0)
        b0 = wp.cross(r0, t1)
        b1 = wp.cross(r1, t1)
        response0 = inv_mass0 + inv_mass1 + wp.dot(a0, inv_inertia0 @ a0) + wp.dot(a1, inv_inertia1 @ a1)
        response1 = inv_mass0 + inv_mass1 + wp.dot(b0, inv_inertia0 @ b0) + wp.dot(b1, inv_inertia1 @ b1)
        rhs = v1 + wp.cross(w1, r1) - v0 - wp.cross(w0, r0)
        if biased:
            rhs += (point1 - point0) * inverse_dt
        prior = impulses[patch, j]
        trial = wp.vec2f(wp.dot(prior, t0), wp.dot(prior, t1))
        if response0 > 0.0:
            trial[0] -= gain * wp.dot(rhs, t0) / response0
        if response1 > 0.0:
            trial[1] -= gain * wp.dot(rhs, t1) / response1
        magnitude = wp.length(trial)
        if magnitude > mu_static * load:
            broken = 1
            trial *= wp.min(mu_dynamic * load, magnitude) / magnitude
        updated = trial[0] * t0 + trial[1] * t1
        v0, v1, w0, w1 = apply_pair_velocity_impulse(
            v0, v1, w0, w1, inv_mass0, inv_mass1, inv_inertia0, inv_inertia1, r0, r1, updated - prior
        )
        impulses[patch, j] = updated
    anchors.broken[patch] = broken
    return v0, v1, w0, w1


@wp.func
def solve_cached(
    patches: NormalPatches,
    anchors: PatchAnchors,
    patch: int,
    normal_impulses: wp.array[wp.float32],
    impulses: wp.array2d[wp.vec3f],
    pose0: wp.transformf,
    pose1: wp.transformf,
    v0: wp.vec3f,
    v1: wp.vec3f,
    w0: wp.vec3f,
    w1: wp.vec3f,
    inv_mass0: wp.float32,
    inv_mass1: wp.float32,
    inv_inertia0: wp.mat33f,
    inv_inertia1: wp.mat33f,
    mu_static: wp.float32,
    mu_dynamic: wp.float32,
    inverse_dt: wp.float32,
    gain: wp.float32,
    biased: bool,
    rows: wp.array2d[FrictionRow],
):
    count = anchors.count[patch]
    if count == 0:
        anchors.broken[patch] = 0
        return v0, v1, w0, w1
    load = float(0.0)
    point = patches.patch_first[patch]
    while point >= 0:
        load += wp.max(normal_impulses[point], 0.0)
        point = patches.point_next[point]
    if count > 0:
        load /= wp.float32(count)
    broken = int(0)
    for j in range(count):
        row = rows[patch, j]
        t0 = row.t0
        t1 = row.t1
        r0 = row.r0
        r1 = row.r1
        response0 = row.response0
        response1 = row.response1
        rhs = v1 + wp.cross(w1, r1) - v0 - wp.cross(w0, r0)
        if biased:
            rhs += row.error * inverse_dt
        prior = impulses[patch, j]
        trial = wp.vec2f(wp.dot(prior, t0), wp.dot(prior, t1))
        if response0 > 0.0:
            trial[0] -= gain * wp.dot(rhs, t0) / response0
        if response1 > 0.0:
            trial[1] -= gain * wp.dot(rhs, t1) / response1
        magnitude = wp.length(trial)
        if magnitude > mu_static * load:
            broken = 1
            trial *= wp.min(mu_dynamic * load, magnitude) / magnitude
        updated = trial[0] * t0 + trial[1] * t1
        v0, v1, w0, w1 = apply_pair_velocity_impulse(
            v0, v1, w0, w1, inv_mass0, inv_mass1, inv_inertia0, inv_inertia1, r0, r1, updated - prior
        )
        impulses[patch, j] = updated
    anchors.broken[patch] = broken
    return v0, v1, w0, w1
