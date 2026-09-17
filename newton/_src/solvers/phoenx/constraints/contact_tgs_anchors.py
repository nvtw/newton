# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Persistent two-anchor correlation for losslessly partitioned normal patches.

Caller supplies unique compatible-group keys (world, ordered bodies, material),
COM transforms, contact world points, and separate current/previous storage.
Only material anchor references persist across steps, not previous impulses.
"""

import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_partition import NormalPatches


@wp.struct
class PatchAnchors:
    count: wp.array[wp.int32]
    broken: wp.array[wp.int32]
    source: wp.array[wp.int32]
    normal0: wp.array[wp.vec3f]
    normal1: wp.array[wp.vec3f]
    local0: wp.array2d[wp.vec3f]
    local1: wp.array2d[wp.vec3f]
    claimed: wp.array[wp.int32]
    candidate_next: wp.array[wp.int32]


def allocate(capacity: int, device: wp.DeviceLike) -> PatchAnchors:
    """Allocate two material anchors and matching scratch per possible patch."""
    state = PatchAnchors()
    state.count = wp.zeros(capacity, dtype=wp.int32, device=device)
    state.broken = wp.zeros(capacity, dtype=wp.int32, device=device)
    state.source = wp.full(capacity, -1, dtype=wp.int32, device=device)
    state.claimed = wp.zeros(capacity, dtype=wp.int32, device=device)
    state.candidate_next = wp.full(capacity, -1, dtype=wp.int32, device=device)
    state.normal0 = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    state.normal1 = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    state.local0 = wp.zeros((capacity, 2), dtype=wp.vec3f, device=device)
    state.local1 = wp.zeros((capacity, 2), dtype=wp.vec3f, device=device)
    return state


@wp.func
def same_key(a: wp.vec4i, b: wp.vec4i):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


@wp.func
def prepare_group(
    group: int,
    current: NormalPatches,
    previous: NormalPatches,
    keys: wp.array[wp.vec4i],
    previous_keys: wp.array[wp.vec4i],
    previous_active: wp.array[wp.int32],
    pose0: wp.transformf,
    pose1: wp.transformf,
    points: wp.array[wp.vec3f],
    gaps: wp.array[wp.float32],
    correlation_distance: wp.float32,
    friction_offset: wp.float32,
    previous_anchors: PatchAnchors,
    current_anchors: PatchAnchors,
):
    key = keys[group]
    inv0 = wp.transform_inverse(pose0)
    inv1 = wp.transform_inverse(pose1)
    old_group = int(-1)
    for candidate in range(previous_active[0]):
        if same_key(key, previous_keys[candidate]):
            old_group = candidate
            break
    first_candidate = int(-1)
    last_candidate = int(-1)
    if old_group >= 0:
        patch = previous.group_first[old_group]
        while patch >= 0:
            previous_anchors.claimed[patch] = 0
            # Keep the original candidate order, excluding only history that
            # cannot match any current patch. Rebuild once per group instead
            # of traversing every broken/empty patch for every new patch.
            if previous_anchors.count[patch] > 0 and previous_anchors.broken[patch] == 0:
                previous_anchors.candidate_next[patch] = -1
                if last_candidate < 0:
                    first_candidate = patch
                else:
                    previous_anchors.candidate_next[last_candidate] = patch
                last_candidate = patch
            patch = previous.patch_next[patch]
    patch = current.group_first[group]
    while patch >= 0:
        normal = current.patch_normal[patch]
        selected = int(-1)
        if old_group >= 0:
            candidate = first_candidate
            while candidate >= 0:
                n = previous_anchors.count[candidate]
                valid = previous_anchors.claimed[candidate] == 0 and previous_anchors.broken[candidate] == 0
                valid = valid and n > 0 and n <= current.patch_count[patch]
                if valid:
                    old_normal0 = wp.transform_vector(pose0, previous_anchors.normal0[candidate])
                    old_normal1 = wp.transform_vector(pose1, previous_anchors.normal1[candidate])
                    valid = valid and wp.dot(normal, old_normal0) > 0.999
                    valid = valid and wp.dot(old_normal0, old_normal1) > 0.999
                    for j in range(n):
                        p0 = wp.transform_point(pose0, previous_anchors.local0[candidate, j])
                        p1 = wp.transform_point(pose1, previous_anchors.local1[candidate, j])
                        valid = valid and wp.abs(wp.dot(p1 - p0, old_normal0)) < correlation_distance
                if valid:
                    selected = candidate
                    break
                candidate = previous_anchors.candidate_next[candidate]
        n = int(0)
        if selected >= 0:
            previous_anchors.claimed[selected] = 1
            n = previous_anchors.count[selected]
            for j in range(n):
                current_anchors.local0[patch, j] = previous_anchors.local0[selected, j]
                current_anchors.local1[patch, j] = previous_anchors.local1[selected, j]
        low = wp.vec3f(1.0e20)
        high = wp.vec3f(-1.0e20)
        point = current.patch_first[patch]
        while point >= 0:
            low = wp.min(low, points[point])
            high = wp.max(high, points[point])
            point = current.point_next[point]
        if n == 2:
            span = current_anchors.local0[patch, 0] - current_anchors.local0[patch, 1]
            if 4.0 * wp.length_sq(span) < wp.length_sq(high - low):
                n = 0
                selected = -1
        old_count = n
        if n < 2:
            x0 = wp.vec3f(0.0)
            x1 = wp.vec3f(0.0)
            distance = float(0.0)
            if n == 1:
                x0 = wp.transform_point(pose0, current_anchors.local0[patch, 0])
            point = current.patch_first[patch]
            while point >= 0:
                if gaps[point] < friction_offset:
                    x = points[point]
                    if n == 0:
                        x0 = x
                        n = 1
                    elif n == 1:
                        distance = wp.length_sq(x - x0)
                        if distance > 1.0e-12:
                            x1 = x
                            n = 2
                    else:
                        d0 = wp.length_sq(x - x0)
                        d1 = wp.length_sq(x - x1)
                        if d0 > d1 and d0 > distance:
                            x1 = x
                            distance = d0
                        elif d1 >= d0 and d1 > distance:
                            x0 = x
                            distance = d1
                point = current.point_next[point]
            if old_count == 0 and n > 0:
                current_anchors.local0[patch, 0] = wp.transform_point(inv0, x0)
                current_anchors.local1[patch, 0] = wp.transform_point(inv1, x0)
            if n == 2:
                current_anchors.local0[patch, 1] = wp.transform_point(inv0, x1)
                current_anchors.local1[patch, 1] = wp.transform_point(inv1, x1)
        current_anchors.count[patch] = n
        current_anchors.source[patch] = selected
        current_anchors.broken[patch] = 0
        current_anchors.normal0[patch] = wp.transform_vector(inv0, normal)
        current_anchors.normal1[patch] = wp.transform_vector(inv1, normal)
        if selected >= 0:
            current_anchors.normal0[patch] = previous_anchors.normal0[selected]
            current_anchors.normal1[patch] = previous_anchors.normal1[selected]
        patch = current.patch_next[patch]


@wp.kernel(enable_backward=False)
def prepare(
    current: NormalPatches,
    previous: NormalPatches,
    keys: wp.array[wp.vec4i],
    previous_keys: wp.array[wp.vec4i],
    active: wp.array[wp.int32],
    previous_active: wp.array[wp.int32],
    poses: wp.array[wp.transformf],
    points: wp.array[wp.vec3f],
    gaps: wp.array[wp.float32],
    correlation_distance: wp.float32,
    friction_offset: wp.float32,
    previous_anchors: PatchAnchors,
    current_anchors: PatchAnchors,
):
    group = wp.tid()
    if group < active[0]:
        key = keys[group]
        prepare_group(
            group,
            current,
            previous,
            keys,
            previous_keys,
            previous_active,
            poses[key[1]],
            poses[key[2]],
            points,
            gaps,
            correlation_distance,
            friction_offset,
            previous_anchors,
            current_anchors,
        )
