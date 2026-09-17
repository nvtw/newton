# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Lossless GPU normal-patch partition within compatible contact groups.

Input groups must already separate world, ordered endpoints and materials.
Each contact belongs to exactly one normal patch. Sparse patch IDs are the
first member's original contact index, so no fixed per-pair limit drops rows.
Linked membership preserves original contact/history indices and avoids a
second permutation of the contact arrays. Temporal correlation is separate.
"""

import warp as wp


@wp.struct
class NormalPatches:
    point_patch: wp.array[wp.int32]
    point_next: wp.array[wp.int32]
    patch_first: wp.array[wp.int32]
    patch_last: wp.array[wp.int32]
    patch_count: wp.array[wp.int32]
    patch_next: wp.array[wp.int32]
    patch_normal: wp.array[wp.vec3f]
    group_first: wp.array[wp.int32]
    group_count: wp.array[wp.int32]


def allocate(point_capacity: int, group_capacity: int, device: wp.DeviceLike) -> NormalPatches:
    """Allocate disjoint contact and compatible-group partition storage."""
    state = NormalPatches()
    for name in ("point_patch", "point_next", "patch_first", "patch_last", "patch_next"):
        setattr(state, name, wp.full(point_capacity, -1, dtype=wp.int32, device=device))
    state.patch_count = wp.zeros(point_capacity, dtype=wp.int32, device=device)
    state.patch_normal = wp.zeros(point_capacity, dtype=wp.vec3f, device=device)
    state.group_first = wp.full(group_capacity, -1, dtype=wp.int32, device=device)
    state.group_count = wp.zeros(group_capacity, dtype=wp.int32, device=device)
    return state


@wp.func
def partition_range(
    group: int, first: int, count: int, normals: wp.array[wp.vec3f], cosine: wp.float32, state: NormalPatches
):
    state.group_first[group] = -1
    state.group_count[group] = 0
    last_patch = int(-1)
    for point in range(first, first + count):
        normal = normals[point]
        candidate = state.group_first[group]
        selected = int(-1)
        while candidate >= 0:
            if wp.dot(normal, state.patch_normal[candidate]) > cosine:
                selected = candidate
                break
            candidate = state.patch_next[candidate]
        if selected < 0:
            selected = point
            state.patch_first[selected] = point
            state.patch_last[selected] = -1
            state.patch_count[selected] = 0
            state.patch_next[selected] = -1
            state.patch_normal[selected] = normal
            if last_patch < 0:
                state.group_first[group] = selected
            else:
                state.patch_next[last_patch] = selected
            last_patch = selected
            state.group_count[group] += 1
        previous = state.patch_last[selected]
        if previous >= 0:
            state.point_next[previous] = point
        state.point_next[point] = -1
        state.point_patch[point] = selected
        state.patch_last[selected] = point
        state.patch_count[selected] += 1


@wp.kernel(enable_backward=False)
def partition(
    normals: wp.array[wp.vec3f],
    first: wp.array[wp.int32],
    count: wp.array[wp.int32],
    active_groups: wp.array[wp.int32],
    cosine: wp.float32,
    state: NormalPatches,
):
    group = wp.tid()
    state.group_first[group] = -1
    state.group_count[group] = 0
    if group < active_groups[0]:
        partition_range(group, first[group], count[group], normals, cosine, state)
