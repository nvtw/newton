# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe storage for coherent rigid contact-patch history.

This module neither selects friction forces nor changes contact geometry. Callers
supply coherent groups (world, ordered bodies, material ID), their point CSR and
injective contact matches. Retention requires complete, unambiguous membership.
A stored birth pose does not authorize constraining a newly entering/rolling
material point to an older configuration. Such membership changes rebirth here.
"""

from __future__ import annotations

import warp as wp


@wp.struct
class FrictionPatchHistory:
    """Fixed-capacity current/previous history and device validation scratch."""

    count: wp.array[wp.int32]
    point_count: wp.array[wp.int32]
    previous_count: wp.array[wp.int32]
    previous_point_count: wp.array[wp.int32]
    key: wp.array[wp.vec4i]
    previous_key: wp.array[wp.vec4i]
    normal: wp.array[wp.vec3f]
    previous_normal: wp.array[wp.vec3f]
    birth: wp.array2d[wp.float32]
    previous_birth: wp.array2d[wp.float32]
    age: wp.array[wp.int32]
    previous_age: wp.array[wp.int32]
    broken: wp.array[wp.int32]
    previous_broken: wp.array[wp.int32]
    member_count: wp.array[wp.int32]
    previous_member_count: wp.array[wp.int32]
    point_patch: wp.array[wp.int32]
    previous_point_patch: wp.array[wp.int32]
    candidate: wp.array[wp.int32]
    claims: wp.array[wp.int32]
    owners: wp.array[wp.int32]
    match_claims: wp.array[wp.int32]
    error: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _initialize_validation(s: FrictionPatchHistory, counts: wp.array[wp.int32]):
    i = wp.tid()
    if i == 0:
        error = wp.int32(0)
        if counts[0] < 0 or counts[0] > s.key.shape[0] or counts[1] < 0 or counts[1] > s.point_patch.shape[0]:
            error = wp.int32(1)
        s.error[0] = error
    if i < s.owners.shape[0]:
        s.owners[i] = -1
        s.match_claims[i] = 0
    if i < s.claims.shape[0]:
        s.claims[i] = 0


@wp.kernel(enable_backward=False)
def _validate_groups(
    s: FrictionPatchHistory,
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    members: wp.array[wp.int32],
    keys: wp.array[wp.vec4i],
    normals: wp.array[wp.vec3f],
    cosine: wp.float32,
    birth: wp.array2d[wp.float32],
):
    patch = wp.tid()
    if counts[0] < 0 or counts[0] > s.key.shape[0] or counts[1] < 0 or counts[1] > s.point_patch.shape[0]:
        return
    if patch >= counts[0]:
        return
    begin = offsets[patch]
    end = offsets[patch + 1]
    if begin < 0 or end <= begin or end > counts[1]:
        wp.atomic_or(s.error, 0, 2)
        return
    first = members[begin]
    if first < 0 or first >= counts[1]:
        wp.atomic_or(s.error, 0, 2)
        return
    key = keys[first]
    for j in range(14):
        if not wp.isfinite(birth[j, patch]):
            wp.atomic_or(s.error, 0, 8)
    normal = normals[first]
    for row in range(begin, end):
        point = members[row]
        if point < 0 or point >= counts[1]:
            wp.atomic_or(s.error, 0, 2)
        else:
            if wp.atomic_cas(s.owners, point, -1, patch) != -1:
                wp.atomic_or(s.error, 0, 2)
            other = keys[point]
            if other[0] != key[0] or other[1] != key[1] or other[2] != key[2] or other[3] != key[3]:
                wp.atomic_or(s.error, 0, 4)
            norm_squared = wp.length_sq(normals[point])
            # Validation tolerance only:32 FP32 eps for supplied unit normals.
            # Do not normalize/alter geometry or widen the angular grouping cone.
            if not wp.isfinite(norm_squared) or wp.abs(norm_squared - wp.float32(1.0)) > wp.float32(3.814697265625e-6):
                wp.atomic_or(s.error, 0, 4)
            alignment = wp.dot(normal, normals[point])
            if not wp.isfinite(alignment) or alignment < cosine:
                wp.atomic_or(s.error, 0, 4)


@wp.kernel(enable_backward=False)
def _validate_coverage(s: FrictionPatchHistory, counts: wp.array[wp.int32], offsets: wp.array[wp.int32]):
    point = wp.tid()
    if counts[0] < 0 or counts[0] > s.key.shape[0] or counts[1] < 0 or counts[1] > s.point_patch.shape[0]:
        return
    if point == 0:
        if offsets[0] != 0 or offsets[counts[0]] != counts[1]:
            wp.atomic_or(s.error, 0, 2)
    if point < counts[1] and s.owners[point] < 0:
        wp.atomic_or(s.error, 0, 2)


@wp.kernel(enable_backward=False)
def _snapshot(s: FrictionPatchHistory):
    i = wp.tid()
    if s.error[0] != 0:
        return
    if i == 0:
        s.previous_count[0] = s.count[0]
        s.previous_point_count[0] = s.point_count[0]
    if i < s.count[0]:
        s.previous_key[i] = s.key[i]
        s.previous_normal[i] = s.normal[i]
        s.previous_age[i] = s.age[i]
        s.previous_broken[i] = s.broken[i]
        s.previous_member_count[i] = s.member_count[i]
        for j in range(14):
            s.previous_birth[j, i] = s.birth[j, i]
    if i < s.point_count[0]:
        s.previous_point_patch[i] = s.point_patch[i]


@wp.kernel(enable_backward=False)
def _count_matches(s: FrictionPatchHistory, counts: wp.array[wp.int32], matches: wp.array[wp.int32]):
    point = wp.tid()
    if s.error[0] != 0 or point >= counts[1]:
        return
    match = matches[point]
    if match >= 0 and match < s.previous_point_count[0]:
        wp.atomic_add(s.match_claims, match, 1)


@wp.kernel(enable_backward=False)
def _correlate(
    s: FrictionPatchHistory,
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    members: wp.array[wp.int32],
    matches: wp.array[wp.int32],
    keys: wp.array[wp.vec4i],
    normals: wp.array[wp.vec3f],
    cosine: wp.float32,
):
    patch = wp.tid()
    if s.error[0] != 0 or patch >= counts[0]:
        return
    begin = offsets[patch]
    end = offsets[patch + 1]
    first = members[begin]
    old = wp.int32(-1)
    complete = wp.bool(True)
    for row in range(begin, end):
        point = members[row]
        match = matches[point]
        if match < 0 or match >= s.previous_point_count[0]:
            complete = False
        else:
            previous = s.previous_point_patch[match]
            if s.match_claims[match] != 1:
                complete = False
            if row == begin:
                old = previous
            elif previous != old:
                complete = False
    if old < 0 or old >= s.previous_count[0]:
        complete = False
    if complete:
        a = keys[first]
        b = s.previous_key[old]
        complete = (
            a[0] == b[0]
            and a[1] == b[1]
            and a[2] == b[2]
            and a[3] == b[3]
            and s.previous_member_count[old] == end - begin
            and s.previous_broken[old] == 0
            and wp.dot(normals[first], s.previous_normal[old]) >= cosine
        )
    if complete:
        s.candidate[patch] = old
        wp.atomic_add(s.claims, old, 1)
    else:
        s.candidate[patch] = -1


@wp.kernel(enable_backward=False)
def _commit(
    s: FrictionPatchHistory,
    counts: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    members: wp.array[wp.int32],
    keys: wp.array[wp.vec4i],
    normals: wp.array[wp.vec3f],
    birth: wp.array2d[wp.float32],
):
    i = wp.tid()
    if s.error[0] != 0:
        return
    if i == 0:
        s.count[0] = counts[0]
        s.point_count[0] = counts[1]
    if i < counts[1]:
        s.point_patch[i] = s.owners[i]
    if i < counts[0]:
        first = members[offsets[i]]
        s.key[i] = keys[first]
        s.normal[i] = normals[first]
        s.member_count[i] = offsets[i + 1] - offsets[i]
        old = s.candidate[i]
        retain = False
        if old >= 0:
            retain = s.claims[old] == 1
        s.age[i] = 0
        s.broken[i] = 0
        if retain:
            s.age[i] = s.previous_age[old] + 1
        for j in range(14):
            value = birth[j, i]
            if retain:
                value = s.previous_birth[j, old]
            s.birth[j, i] = value


def allocate_friction_patch_history(max_patches: int, max_points: int, device=None) -> FrictionPatchHistory:
    """Allocate all lifecycle memory before capture; capacities must be positive."""
    if max_patches < 1 or max_points < 1:
        raise ValueError("Patch and point capacities must be positive")
    s = FrictionPatchHistory()
    for name in ("count", "point_count", "previous_count", "previous_point_count", "error"):
        setattr(s, name, wp.zeros(1, dtype=wp.int32, device=device))
    for name in ("key", "previous_key"):
        setattr(s, name, wp.zeros(max_patches, dtype=wp.vec4i, device=device))
    for name in ("normal", "previous_normal"):
        setattr(s, name, wp.zeros(max_patches, dtype=wp.vec3f, device=device))
    for name in ("birth", "previous_birth"):
        setattr(s, name, wp.zeros((14, max_patches), dtype=wp.float32, device=device))
    for name in (
        "age",
        "previous_age",
        "broken",
        "previous_broken",
        "member_count",
        "previous_member_count",
        "candidate",
        "claims",
    ):
        setattr(s, name, wp.zeros(max_patches, dtype=wp.int32, device=device))
    for name in ("point_patch", "previous_point_patch", "owners", "match_claims"):
        setattr(s, name, wp.zeros(max_points, dtype=wp.int32, device=device))
    return s


def update_friction_patch_history(
    s, counts, offsets, members, keys, normals, matches, birth, *, normal_cosine, device=None
):
    """Update one contact generation using supplied CSR groups and birth poses.

    counts=[patches,points]; keys are (world,body0,body1,material_group).
    birth stores p0(3),p1(3),q0_xyzw(4),q1_xyzw(4) in FP32. Storage alone
    does not activate constraints or define a shared material anchor.
    Normals must be finite unit vectors. Materials/groups and matching are
    supplied by collision ingestion, never inferred from body IDs here. Matches
    must certify persistent material-point identity, not merely a reused
    collision-feature index; entering/rolling/reentering points require -1.
    Compatibility keys are not patch identity: CSR groups and lineage distinguish
    disjoint patches sharing the same key. Invoke once per contact generation,
    not once per solver substep.
    Device error bits:1 capacity,2 invalid CSR,4 incompatible group/nonunit
    normal,8 nonfinite birth reference. On error,
    live and previous history are untouched; check_friction_patch_history must
    be called outside capture before consuming invalid input.
    Caller sets broken only under its explicit coherent-history invalidation
    policy, not an inferred per-point clipping OR. age counts retained contact
    generations, not simulation substeps or elapsed physical time.
    """
    p, n = s.key.shape[0], s.point_patch.shape[0]
    if (
        counts.shape[0] != 2
        or offsets.shape[0] < p + 1
        or members.shape[0] < n
        or keys.shape[0] < n
        or normals.shape[0] < n
        or matches.shape[0] < n
        or birth.shape != (14, p)
    ):
        raise ValueError("Inputs must cover fixed patch/point capacities")
    if not 0 <= normal_cosine <= 1:
        raise ValueError("normal_cosine must lie in [0,1]")
    width = max(p, n)
    wp.launch(_initialize_validation, width, inputs=[s, counts], device=device)
    wp.launch(
        _validate_groups,
        p,
        inputs=[s, counts, offsets, members, keys, normals, wp.float32(normal_cosine), birth],
        device=device,
    )
    wp.launch(_validate_coverage, n, inputs=[s, counts, offsets], device=device)
    wp.launch(_snapshot, width, inputs=[s], device=device)
    wp.launch(_count_matches, n, inputs=[s, counts, matches], device=device)
    wp.launch(
        _correlate,
        p,
        inputs=[s, counts, offsets, members, matches, keys, normals, wp.float32(normal_cosine)],
        device=device,
    )
    wp.launch(_commit, width, inputs=[s, counts, offsets, members, keys, normals, birth], device=device)


def check_friction_patch_history(s) -> None:
    """Explicit post-launch host error check; never call inside CUDA capture."""
    error = int(s.error.numpy()[0])
    if error:
        raise ValueError(f"Invalid friction patch history input (device error bits {error})")
