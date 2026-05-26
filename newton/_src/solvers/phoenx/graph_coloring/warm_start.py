# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-frame warm-start cache for the greedy MIS partitioner.

Keyed by body-pair (stable across frames, unlike cid which the contact
matcher re-indexes). The partitioner seeds ``color_tags`` from the
cache at the top of ``build_csr``, validates against the current
adjacency, and skips MIS for entries whose cached colour survives.

Multi-contact pairs (N contact points between the same bodies) share
one cache entry, so N-1 of those re-MIS each step. Best case is near
100% cache hit for unique-body-pair scenes (joints, cloth tris, etc.).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    _COLOR_SHIFT,
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_get,
)

__all__ = [
    "WarmStartCache",
    "binary_search_warm_start_func",
    "seed_warm_start_kernel",
    "warm_start_cache_zeros",
    "warm_start_dedup_pairs_kernel",
    "warm_start_emit_pairs_kernel",
    "warm_start_invalidate_apply_kernel",
    "warm_start_invalidate_mark_kernel",
    "warm_start_key_func",
    "warm_start_mark_boundaries_kernel",
    "warm_start_reset_count_kernel",
]


@wp.struct
class WarmStartCache:
    """SoA storage for the body-pair -> colour cache, sized to
    ``max_num_interactions`` with live count in ``num_entries[0]``.
    Keys are sorted ascending after each ``build_csr`` so the seed
    kernel's binary search is well-defined.
    """

    #: Packed body-pair keys: ``(body_2nd_min << 32) | body_min``.
    keys: wp.array[wp.int64]
    #: Per-key cached colour, encoded ``c + 1`` to match ``color_tags``.
    colors: wp.array[wp.int32]
    #: Live entry count (length 1).
    num_entries: wp.array[wp.int32]


def warm_start_cache_zeros(
    capacity: int,
    device: wp.context.Devicelike = None,
) -> WarmStartCache:
    """Allocate a zero-initialised :class:`WarmStartCache` of capacity
    ``capacity``. ``num_entries`` starts at 0; the persist kernel
    populates the cache after the first ``build_csr``."""
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1 (got {capacity})")
    c = WarmStartCache()
    c.keys = wp.zeros(capacity, dtype=wp.int64, device=device)
    c.colors = wp.zeros(capacity, dtype=wp.int32, device=device)
    c.num_entries = wp.zeros(1, dtype=wp.int32, device=device)
    return c


_BODY_INF = wp.constant(wp.int32(0x7FFFFFFF))


@wp.func
def warm_start_key_func(el: ElementInteractionData) -> wp.int64:
    """Pack ``(body_2nd_min << 32) | body_min`` from the two smallest
    non-negative body endpoints. Missing slots use ``INT32_MAX`` so
    the key stays unique per body-pair signature."""
    b_min = _BODY_INF
    b_2nd = _BODY_INF
    for j in range(MAX_BODIES):
        b = element_interaction_data_get(el, j)
        if b < wp.int32(0):
            break
        if b < b_min:
            b_2nd = b_min
            b_min = b
        elif b < b_2nd:
            b_2nd = b
    return (wp.int64(b_2nd) << wp.int64(32)) | (wp.int64(b_min) & wp.int64(0xFFFFFFFF))


@wp.func
def binary_search_warm_start_func(
    keys: wp.array[wp.int64],
    num_entries: wp.int32,
    target: wp.int64,
) -> wp.int32:
    """Binary search for ``target`` in ``keys[:num_entries]``; returns
    the match index or -1. Capture-safe (no host roundtrip)."""
    lo = wp.int32(0)
    hi = num_entries - wp.int32(1)
    while lo <= hi:
        mid = (lo + hi) >> wp.int32(1)
        v = keys[mid]
        if v < target:
            lo = mid + wp.int32(1)
        elif v > target:
            hi = mid - wp.int32(1)
        else:
            return mid
    return wp.int32(-1)


@wp.kernel(enable_backward=False)
def seed_warm_start_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    cache_keys: wp.array[wp.int64],
    cache_colors: wp.array[wp.int32],
    cache_num_entries: wp.array[wp.int32],
    color_tags: wp.array[wp.int32],
    partition_data_concat: wp.array[wp.int64],
    skip_color_start_plus_one: wp.array[wp.int32],
    skip_color_end_plus_one: wp.array[wp.int32],
):
    """Seed ``color_tags[tid]`` from the cache via body-pair key
    lookup. Empty cache (``cache_num_entries[0] == 0``) is a no-op
    (cold-start).

    ``skip_color_{start,end}_plus_one[0]`` mark an inclusive
    encoded-colour range to skip this frame (forces those constraints
    to re-MIS). Empty range (start > end) disables skipping. Mirrors
    the colour into ``partition_data_concat`` for the greedy kernel.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    n = cache_num_entries[0]
    if n == wp.int32(0):
        # Empty cache: leave color_tags / partition_data_concat as-is
        # (adjacency_store_kernel already wrote them for the cold path).
        return
    el = elements[tid]
    key = warm_start_key_func(el)
    idx = binary_search_warm_start_func(cache_keys, n, key)
    if idx < wp.int32(0):
        return  # not in cache -> cold path (color_tags[tid] stays 0)
    cached_color = cache_colors[idx]
    if cached_color <= wp.int32(0):
        return  # corrupt entry (shouldn't happen); fall through
    s = skip_color_start_plus_one[0]
    e = skip_color_end_plus_one[0]
    if s <= e and cached_color >= s and cached_color <= e:
        return  # rotate-skip: re-MIS this colour this frame
    color_tags[tid] = cached_color
    # partition_data_concat encoding: high bits = color+1 (which is
    # exactly what color_tags stores), low bits = tid.
    partition_data_concat[tid] = (wp.int64(cached_color) << _COLOR_SHIFT) | wp.int64(tid)


@wp.kernel(enable_backward=False)
def warm_start_invalidate_mark_kernel(
    color_tags: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
    adjacency_section_end_indices: wp.array[wp.int32],
    vertex_to_adjacent_elements: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
    max_colored_partitions: wp.int32,
    invalid_mark: wp.array[wp.int32],
):
    """Pass 1 of warm-start validation. For each seeded ``tid``, scan
    neighbours; if any with higher tid shares the same colour, mark
    ``invalid_mark[tid]``. Tie-break by tid keeps both sides from
    losing. Race-free (reads-only on ``color_tags``). Overflow-bucket
    entries are exempt -- mass splitting handles their conflicts."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    my_color = color_tags[tid]
    if my_color == wp.int32(0):
        return  # not seeded, MIS handles
    # Overflow bucket: same-colour neighbours are expected; not a conflict.
    if max_colored_partitions >= wp.int32(0) and my_color == max_colored_partitions + wp.int32(1):
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < wp.int32(0):
            break
        start = wp.int32(0)
        if v > wp.int32(0):
            start = adjacency_section_end_indices[v - wp.int32(1)]
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            # Self isn't a conflict, and we tie-break by tid so each
            # conflict pair (lower, higher) marks only the higher.
            if neighbor <= tid:
                continue
            if color_tags[neighbor] == my_color:
                invalid_mark[tid] = wp.int32(1)
                return


_PERSIST_TAIL_KEY: int = 0x7FFFFFFFFFFFFFFF


@wp.kernel(enable_backward=False)
def warm_start_emit_pairs_kernel(
    elements: wp.array[ElementInteractionData],
    color_tags: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
    keys_out: wp.array[wp.int64],
    values_out: wp.array[wp.int32],
):
    """Emit ``(body_pair_key, cid)`` per active coloured constraint;
    inactive slots get ``INT64_MAX`` keys (sort pushes them to tail).
    ``values_out[i] = cid`` so dedup can recover the colour after sort."""
    tid = wp.tid()
    if tid >= keys_out.shape[0]:
        return
    n = num_elements[0]
    if tid >= n:
        keys_out[tid] = wp.int64(_PERSIST_TAIL_KEY)
        values_out[tid] = wp.int32(0)
        return
    color = color_tags[tid]
    if color == wp.int32(0):
        # Shouldn't happen after a clean build (overflow spill makes
        # sure every active element is coloured), but be defensive.
        keys_out[tid] = wp.int64(_PERSIST_TAIL_KEY)
        values_out[tid] = wp.int32(0)
        return
    keys_out[tid] = warm_start_key_func(elements[tid])
    values_out[tid] = tid


@wp.kernel(enable_backward=False)
def warm_start_mark_boundaries_kernel(
    sorted_keys: wp.array[wp.int64],
    num_elements: wp.array[wp.int32],
    is_boundary: wp.array[wp.int32],
):
    """Mark ``is_boundary[i] = 1`` iff ``sorted_keys[i]`` is the first
    occurrence of its key (within the active region). Feeds the
    prefix-scan that assigns destination cache slots."""
    tid = wp.tid()
    if tid >= is_boundary.shape[0]:
        return
    n = num_elements[0]
    if tid >= n:
        is_boundary[tid] = wp.int32(0)
        return
    key = sorted_keys[tid]
    if key == wp.int64(_PERSIST_TAIL_KEY):
        is_boundary[tid] = wp.int32(0)
        return
    if tid == wp.int32(0):
        is_boundary[tid] = wp.int32(1)
        return
    if sorted_keys[tid - wp.int32(1)] != key:
        is_boundary[tid] = wp.int32(1)
    else:
        is_boundary[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def warm_start_dedup_pairs_kernel(
    sorted_keys: wp.array[wp.int64],
    sorted_values: wp.array[wp.int32],
    is_boundary: wp.array[wp.int32],
    dest_idx: wp.array[wp.int32],
    color_tags: wp.array[wp.int32],
    num_elements: wp.array[wp.int32],
    out_keys: wp.array[wp.int64],
    out_colors: wp.array[wp.int32],
    out_num_entries: wp.array[wp.int32],
):
    """Compact unique keys into the cache: each boundary thread writes
    ``(key, color)`` to its scan-assigned slot and atomically updates
    ``out_num_entries`` to the final unique count. Duplicate keys
    (multi-contact body pairs) keep the sort's first-landed entry."""
    tid = wp.tid()
    if tid >= sorted_keys.shape[0]:
        return
    n = num_elements[0]
    if tid >= n:
        return
    if is_boundary[tid] == wp.int32(0):
        return
    cid = sorted_values[tid]
    dest = dest_idx[tid]
    out_keys[dest] = sorted_keys[tid]
    out_colors[dest] = color_tags[cid]
    # Single-thread write to num_entries: the LAST boundary slot's
    # dest+1 is the total unique count. Use atomic_max so any
    # straggler write is harmless.
    wp.atomic_max(out_num_entries, 0, dest + wp.int32(1))


@wp.kernel(enable_backward=False)
def warm_start_reset_count_kernel(out_num_entries: wp.array[wp.int32]):
    """Zero ``out_num_entries[0]`` before the dedup kernel's
    ``atomic_max`` updates."""
    if wp.tid() == wp.int32(0):
        out_num_entries[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def warm_start_invalidate_apply_kernel(
    invalid_mark: wp.array[wp.int32],
    color_tags: wp.array[wp.int32],
    partition_data_concat: wp.array[wp.int64],
    num_elements: wp.array[wp.int32],
    num_remaining: wp.array[wp.int32],
):
    """Pass 2 of warm-start validation: reset invalidated constraints
    and decrement ``num_remaining`` for the ones that survived.

    Three outcomes per element:

    * ``invalid_mark == 1``: previous colour was illegal under the
      new adjacency. Reset to 0 so the MIS loop picks it up.
    * ``invalid_mark == 0`` and ``color_tags != 0``: warm-start hit,
      colour kept. Decrement ``num_remaining`` (greedy uses this as
      its "elements still to colour" counter; for warm-started slots
      we're effectively pre-colouring, so they shouldn't be counted
      as remaining work).
    * ``color_tags == 0``: no warm-start, MIS will colour this one.
      ``num_remaining`` already counts it (initialised to
      ``num_elements``); no change.

    Also clears ``invalid_mark[tid]`` for next frame.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if invalid_mark[tid] != wp.int32(0):
        invalid_mark[tid] = wp.int32(0)  # clear for next frame
        color_tags[tid] = wp.int32(0)
        partition_data_concat[tid] = wp.int64(tid)
        return
    if color_tags[tid] != wp.int32(0):
        wp.atomic_sub(num_remaining, 0, 1)
