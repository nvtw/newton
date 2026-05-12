# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-frame warm-start cache for the greedy MIS partitioner.

Most of the constraint graph is unchanged between adjacent solver
steps -- the same bodies remain in contact, the same joints / cloth
triangles / soft tets persist. Recomputing the colouring from
scratch every step wastes work on constraints whose previous-frame
colour is still legal.

This module provides a body-pair-indexed cache of previously-assigned
colours. The partitioner reads it at the top of ``build_csr``,
validates entries against the current adjacency, and skips MIS for
constraints whose cached colour survives validation.

Why body-pair, not cid? Contacts get re-indexed every step by the
contact matcher -- ``cid=42`` last frame and ``cid=42`` this frame
are different physical constraints. Body indices, however, are
stable across frames (the builder assigns them once). Keying the
cache by the constraint's body-pair signature gives a stable handle
even as cids shuffle.

Multi-contact body pairs (e.g. ring-vs-ring with N contact points)
all hash to the same key; the cache stores only ONE colour per pair,
so N-1 of those constraints will be invalidated by the validation
pass and re-coloured by greedy MIS. The first seeded constraint
benefits from the warm-start; the others fall back to cold-start.
Worst case: zero speedup. Best case (joints, cloth tris, soft tets
with unique body signatures): near-100%% cache hit.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_get,
)

__all__ = [
    "WarmStartCache",
    "warm_start_cache_zeros",
    "warm_start_key_func",
    "binary_search_warm_start_func",
]


@wp.struct
class WarmStartCache:
    """SoA storage for the body-pair -> colour cache.

    Arrays are sized to ``max_num_interactions`` (upper bound on
    distinct body pairs == one per constraint). Actual count tracked
    by ``num_entries[0]``. ``keys`` are sorted ascending so device-side
    binary search is well-defined; the persist pipeline reorders into
    this layout after each ``build_csr``.
    """

    #: Packed body-pair keys. Layout: ``(body_2nd_min << 32) | body_min``.
    #: Sorted ascending after :func:`persist_warm_start_kernel`.
    keys: wp.array[wp.int64]
    #: Per-key cached colour. Same encoding as ``color_tags``: ``c + 1``
    #: where ``c`` is the colour index (``0`` would mean "no colour",
    #: which is never stored).
    colors: wp.array[wp.int32]
    #: Live entry count (length 1).
    num_entries: wp.array[wp.int32]


def warm_start_cache_zeros(
    capacity: int,
    device: wp.context.Devicelike = None,
) -> WarmStartCache:
    """Allocate a zero-initialised :class:`WarmStartCache`.

    Args:
        capacity: Maximum entries (matches the partitioner's
            ``max_num_interactions``).
        device: Warp device for the allocation.

    ``num_entries`` starts at 0 -- so the first build's lookup hits
    the empty path on every cid (cold start). The persist kernel then
    populates the cache for subsequent builds.
    """
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
    """Compute the body-pair cache key for one element.

    Layout: ``(body_2nd_min << 32) | (body_min & 0xFFFFFFFF)``. Uses
    the two smallest non-negative body endpoints. If the constraint
    has fewer than two real bodies, the missing slots get a sentinel
    ``INT32_MAX`` so the key is still unique-per-signature.

    Equivalent ``(body_min, body_2nd_min)`` lexicographic ordering --
    matches what the radix sort produces, so cached keys land in a
    deterministic order.
    """
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
    """Binary search for ``target`` in ``keys[:num_entries]``. Returns
    the index of the match or ``-1`` if not found.

    Sorted-array invariant is established by the persist pipeline
    (radix sort + boundary compaction). Safe to call from inside a
    captured graph -- no host roundtrip.
    """
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
