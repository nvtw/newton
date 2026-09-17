# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cache compatible-group normals in CUDA shared memory for exact first-fit partitioning.

The greedy order, representative normal, and original contact indices match
partition_range. Groups beyond the cache capacity use the original lossless
implementation; the cache size never limits the number of physical rows.
"""

import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_partition import NormalPatches, partition_range

CACHE_CAPACITY = 1024
BLOCK_SIZE = 128


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ wp::vec3 cache[1024];
    __shared__ int seeds[1024];
    __shared__ int tails[1024];
    __shared__ int sizes[1024];
    const wp::vec3* input = (const wp::vec3*)normals.data;
    for (int i = lane; i < count; i += 128)
        cache[i] = input[first + i];
    __syncthreads();
    if (lane == 0) {
        int* point_patch = (int*)state.point_patch.data;
        int* point_next = (int*)state.point_next.data;
        int* patch_first = (int*)state.patch_first.data;
        int* patch_last = (int*)state.patch_last.data;
        int* patch_count = (int*)state.patch_count.data;
        int* patch_next = (int*)state.patch_next.data;
        wp::vec3* patch_normal = (wp::vec3*)state.patch_normal.data;
        int* group_first = (int*)state.group_first.data;
        int* group_count = (int*)state.group_count.data;
        int used = 0;
        for (int i = 0; i < count; ++i) {
            int selected = -1;
            for (int j = 0; j < used; ++j) {
                if (wp::dot(cache[i], cache[seeds[j]]) > cosine) {
                    selected = j;
                    break;
                }
            }
            if (selected < 0) {
                selected = used++;
                seeds[selected] = i;
                tails[selected] = -1;
                sizes[selected] = 0;
            }
            if (tails[selected] >= 0)
                point_next[first + tails[selected]] = first + i;
            point_next[first + i] = -1;
            point_patch[first + i] = first + seeds[selected];
            tails[selected] = i;
            ++sizes[selected];
        }
        for (int j = 0; j < used; ++j) {
            int patch = first + seeds[j];
            patch_first[patch] = patch;
            patch_last[patch] = first + tails[j];
            patch_count[patch] = sizes[j];
            patch_next[patch] = j + 1 < used ? first + seeds[j + 1] : -1;
            patch_normal[patch] = cache[seeds[j]];
        }
        group_first[group] = used > 0 ? first : -1;
        group_count[group] = used;
    }
    __syncthreads();
#endif
""")
def partition_cached(
    group: int, first: int, count: int, normals: wp.array[wp.vec3f], cosine: float, state: NormalPatches, lane: int
): ...


@wp.kernel(enable_backward=False)
def partition_shared(
    normals: wp.array[wp.vec3f],
    first: wp.array[int],
    count: wp.array[int],
    active: wp.array[int],
    cosine: float,
    state: NormalPatches,
):
    group, lane = wp.tid()
    if group < active[0]:
        if count[group] <= CACHE_CAPACITY:
            partition_cached(group, first[group], count[group], normals, cosine, state, lane)
        elif lane == 0:
            partition_range(group, first[group], count[group], normals, cosine, state)
    elif lane == 0:
        state.group_first[group] = -1
        state.group_count[group] = 0
