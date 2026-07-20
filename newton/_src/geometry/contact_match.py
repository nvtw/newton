# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic frame-to-frame contact matching on sorted shape-pair keys.

Current contacts claim the closest compatible prior contact within their shape
pair. Packed atomic minimum claims make the mapping injective and deterministic.
Sticky mode builds canonical contact geometry before the final deterministic
gather, avoiding separate replay history and copy passes.
"""

from __future__ import annotations

import warp as wp

from ..core.types import Devicelike
from .contact_sort import SORT_KEY_SENTINEL

MATCH_NOT_FOUND = wp.constant(wp.int32(-1))
"""Sentinel: no matching key found in last frame's contacts."""

MATCH_BROKEN = wp.constant(wp.int32(-2))
"""Sentinel: key found but position or normal threshold exceeded."""


# ------------------------------------------------------------------
# Warp helpers
# ------------------------------------------------------------------


# Sentinel value for unclaimed slots in ``_prev_claim``.  Larger than
# any packed (flipped_dist << 32 | key_low32) any kernel will ever
# produce, so the first ``atomic_min`` always wins.
_CLAIM_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


@wp.func
def _lower_bound_int64(
    lower: int,
    upper: int,
    target: wp.int64,
    keys: wp.array[wp.int64],
) -> int:
    """First index in ``keys[lower:upper]`` whose value is >= *target*.

    Returns ``upper`` if no such index exists.
    """
    left = lower
    right = upper
    while left < right:
        mid = left + (right - left) // 2
        if keys[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000u;
return i ^ mask;
""")
def _float_flip(f: float) -> wp.uint32:
    """Reinterpret a 32-bit float as a sortable ``uint32`` (Stereopsis trick).

    For non-negative floats this is a strictly monotone encoding, so
    comparing the resulting ``uint32`` orders the original floats
    correctly.  We only ever feed non-negative ``dist_sq`` values into
    it, so the negative branch is unused here but kept generic.
    """
    ...


@wp.func
def _pack_claim(dist_sq: float, key_low32: wp.int64) -> wp.int64:
    """Pack ``(dist_sq, sort_key_low32)`` into a single int64 for ``atomic_min``.

    High 32 bits: ``float_flip(dist_sq)`` — ascending by distance.
    Low 32 bits:  the low 32 bits of the contact's sort key — deterministic
        tie-break (smallest wins).  Using the sort key (rather than the
        unsorted thread id) keeps the resolution invariant under
        non-deterministic narrow-phase slot assignment: two new contacts
        racing for the same prev contact get the same packed claim
        regardless of which unsorted slot the narrow phase happened to
        hand them this run.

    Within a single shape pair the upper 40 bits of every contact's sort
    key are identical, so the low 32 bits hold the (shape_b LSBs +
    sort_sub_key) which uniquely identifies each contact in the pair as
    long as ``sort_sub_key`` is unique per contact within the pair.

    Note this is a *shared* assumption with the deterministic radix sort
    upstream, not a hard guarantee enforced by it.  The multi-contact
    and mesh/SDF paths build ``sort_sub_key`` from per-contact identifiers
    (clip-vertex slot, triangle/edge/vertex index) that are unique per
    pair by construction, but the reduced-contact path
    (``contact_reduction_global.export_reduced_contacts_kernel``)
    re-uses the original contact's fingerprint as ``sort_sub_key`` and
    only deduplicates by ``contact_id``, so two reduced contacts in the
    same pair can in principle land in different reduction slots and
    still share a fingerprint.  When that happens the deterministic
    sort and this tiebreak degrade together: the contacts are
    indistinguishable to either, and frame-to-frame matching becomes
    order-sensitive only to the same extent the sort itself does.  In
    other words, this scheme is no worse than what the upstream sort
    already provides.
    """
    flipped = wp.int64(_float_flip(dist_sq))
    return (flipped << wp.int64(32)) | (key_low32 & wp.int64(0xFFFFFFFF))


# ------------------------------------------------------------------
# Match kernel
# ------------------------------------------------------------------


@wp.struct
class _MatchData:
    """Bundled arrays for the contact match kernel."""

    # Previous frame (sorted) — pos/normal reuse ContactSorter scratch buffers.
    # ``prev_pos_world`` holds the world-space *midpoint* between shape 0's and
    # shape 1's contact points, saved by the previous frame's save kernel.
    prev_keys: wp.array[wp.int64]
    prev_pos_world: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]
    prev_count: wp.array[wp.int32]

    # Current frame (canonical in CollisionPipeline; arbitrary for direct callers).
    new_keys: wp.array[wp.int64]
    new_point0: wp.array[wp.vec3]
    new_point1: wp.array[wp.vec3]
    new_shape0: wp.array[wp.int32]
    new_shape1: wp.array[wp.int32]
    new_normal: wp.array[wp.vec3]
    new_count: wp.array[wp.int32]
    canonical_to_source: wp.array[wp.int32]
    use_permutation: int

    # Body transforms for world-space conversion
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Per-prev claim word, packed (float_flip(dist_sq) << 32 | key_low32),
    # where key_low32 is the low 32 bits of the racing new contact's sort
    # key (deterministic per contact, invariant under non-deterministic
    # narrow-phase slot assignment -- see ``_pack_claim``).
    # Initialised to _CLAIM_SENTINEL each frame; race with atomic_min.
    prev_claim: wp.array[wp.int64]

    # Per-new candidate prev index (final value resolved in pass 2).
    match_index: wp.array[wp.int32]

    # Thresholds
    pos_threshold_sq: float
    normal_dot_threshold: float


@wp.kernel(enable_backward=False)
def _match_contacts_kernel(data: _MatchData):
    """Pass 1: pick each new contact's closest prev candidate and stake
    a packed claim on it via ``wp.atomic_min``.
    """
    tid = wp.tid()
    n_new = data.new_count[0]
    if tid >= n_new:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    # Clamp against the prev_keys capacity. ``_save_sorted_state_kernel``
    # already clamps before writing, but the read-side clamp is cheap
    # defense in depth in case a future caller pre-populates
    # ``prev_count`` outside that save path.
    n_old = data.prev_count[0]
    cap_old = data.prev_keys.shape[0]
    if n_old > cap_old:
        n_old = cap_old
    if n_old == 0:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    source = tid
    if data.use_permutation != 0:
        source = data.canonical_to_source[tid]
    target_key = data.new_keys[source] if data.use_permutation != 0 else data.new_keys[tid]

    # World-space midpoint of the two contact points (symmetric in shape 0 /
    # shape 1).  Matches the quantity persisted by ``_save_sorted_state_kernel``
    # for the previous frame, so both sides of ``diff`` below measure the same
    # physical quantity.
    p0 = data.new_point0[source]
    bid0 = data.shape_body[data.new_shape0[source]]
    if bid0 == -1:
        p0w = p0
    else:
        p0w = wp.transform_point(data.body_q[bid0], p0)

    p1 = data.new_point1[source]
    bid1 = data.shape_body[data.new_shape1[source]]
    if bid1 == -1:
        p1w = p1
    else:
        p1w = wp.transform_point(data.body_q[bid1], p1)

    new_pos_w = 0.5 * (p0w + p1w)
    new_n = data.new_normal[source]

    # Binary search the [range_lo, range_hi) interval of prev contacts
    # sharing the same (shape_a, shape_b) pair.  We ignore sort_sub_key
    # because for multi-contact manifolds (e.g. box-box face-face) the
    # sub-key assignment is not stable across frames; matching by the
    # exact key would spuriously break stable contacts.  Pair counts are
    # small (<= a few manifold points), so a linear scan inside the
    # range is cheap.
    pair_prefix = target_key & wp.int64(~0x7FFFFF)
    pair_end = pair_prefix + wp.int64(0x800000)  # 1 << 23
    range_lo = _lower_bound_int64(0, n_old, pair_prefix, data.prev_keys)
    range_hi = _lower_bound_int64(range_lo, n_old, pair_end, data.prev_keys)

    if range_lo >= range_hi:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    # Closest-point match within the pair range, gated by normal dot.
    best_idx = int(-1)
    best_dist_sq = float(data.pos_threshold_sq)
    for old_idx in range(range_lo, range_hi):
        old_pos = data.prev_pos_world[old_idx]
        diff = new_pos_w - old_pos
        dist_sq = wp.dot(diff, diff)
        old_n = data.prev_normal[old_idx]
        ndot = wp.dot(new_n, old_n)

        if dist_sq <= best_dist_sq and ndot >= data.normal_dot_threshold:
            best_dist_sq = dist_sq
            best_idx = old_idx

    if best_idx >= 0:
        data.match_index[tid] = wp.int32(best_idx)
        # Race for ownership of prev[best_idx] with a single atomic_min.
        # Closest distance wins; ties resolved by smallest sort_key low
        # 32 bits.  Using the sort key (instead of ``tid``) keeps the
        # winner invariant under the non-deterministic unsorted slot
        # assignment that the narrow phase gives us via ``wp.atomic_add``.
        wp.atomic_min(data.prev_claim, best_idx, _pack_claim(best_dist_sq, target_key))
    else:
        # Pair range exists but no contact within thresholds.
        data.match_index[tid] = MATCH_BROKEN


@wp.kernel(enable_backward=False)
def _clear_prev_claim_kernel(
    prev_claim: wp.array[wp.int64],
    prev_count: wp.array[wp.int32],
):
    """Reset only the active prefix of the claim buffer to ``_CLAIM_SENTINEL``.

    Launched with ``capacity`` threads so the per-frame launch fits a
    static CUDA graph, but each thread guards on ``prev_count[0]`` so we
    only touch the (typically much smaller) range of slots that ``match``
    will actually race on.  Slots beyond ``prev_count`` are never read
    by either kernel, so leaving them stale is safe.
    """
    i = wp.tid()
    # Clamp to the prev_claim capacity in case ``prev_count`` was
    # populated from an overflowed frame -- the launch dim already
    # bounds ``i`` to capacity, but the explicit clamp makes the
    # bound-safety property local to the kernel.
    n_old = prev_count[0]
    cap = prev_claim.shape[0]
    if n_old > cap:
        n_old = cap
    if i < n_old:
        prev_claim[i] = _CLAIM_SENTINEL


@wp.kernel(enable_backward=False)
def _resolve_claims_kernel(
    match_index: wp.array[wp.int32],
    sort_keys: wp.array[wp.int64],
    prev_claim: wp.array[wp.int64],
    prev_was_matched: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
    canonical_to_source: wp.array[wp.int32],
    use_permutation: int,
    has_report: int,
):
    """Pass 2: keep winners, demote losers to :data:`MATCH_BROKEN`.

    The low 32 bits of ``prev_claim[cand]`` identify the winning
    contact by the low 32 bits of its sort key (deterministic per
    contact, invariant under unsorted slot reordering); everyone else
    who staked a claim on the same ``cand`` becomes :data:`MATCH_BROKEN`
    (no second-closest fallback).
    """
    tid = wp.tid()
    # Clamp against ``match_index`` capacity so an overflowed
    # ``new_count`` doesn't let threads in [capacity, new_count) read
    # past the end of the match_index / sort_keys buffers.
    n_new = new_count[0]
    cap = match_index.shape[0]
    if n_new > cap:
        n_new = cap
    if tid >= n_new:
        return

    cand = match_index[tid]
    if cand < wp.int32(0):
        return  # already MATCH_NOT_FOUND or MATCH_BROKEN

    winner_key_low = prev_claim[cand] & wp.int64(0xFFFFFFFF)
    source = tid
    if use_permutation != 0:
        source = canonical_to_source[tid]
    my_key_low = sort_keys[source] & wp.int64(0xFFFFFFFF)
    if winner_key_low == my_key_low:
        if has_report != 0:
            prev_was_matched[cand] = wp.int32(1)
    else:
        match_index[tid] = MATCH_BROKEN


# ------------------------------------------------------------------
# Save sorted state kernel
# ------------------------------------------------------------------


@wp.struct
class _SaveStateData:
    """Arrays used to retain matching keys, midpoints, and normals."""

    src_keys: wp.array[wp.int64]
    src_point0: wp.array[wp.vec3]
    src_point1: wp.array[wp.vec3]
    src_shape0: wp.array[wp.int32]
    src_shape1: wp.array[wp.int32]
    src_normal: wp.array[wp.vec3]
    src_count: wp.array[wp.int32]

    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    dst_keys: wp.array[wp.int64]
    dst_pos_world: wp.array[wp.vec3]  # world-space midpoint of point0 and point1
    dst_normal: wp.array[wp.vec3]
    dst_count: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _save_sorted_state_kernel(data: _SaveStateData):
    """Copy sorted contacts into the previous-frame buffers for next-frame matching.

    The persisted ``dst_pos_world`` is the world-space *midpoint* of the two
    contact points, so the next frame's match kernel compares a shape-symmetric
    quantity.

    ``dst_count`` is clamped against the destination buffer capacity. On
    narrow-phase overflow ``src_count[0]`` exceeds ``dst_keys.shape[0]``
    and the per-thread write below only fills ``[0, capacity)``; saving
    the raw inflated count would leave the ``[capacity, src_count)`` tail
    appearing valid to next frame's matcher (it'd binary-search into
    stale/garbage prev_keys and return bogus matches, which feed bogus
    warm-start impulses into the solver and make resting bodies kick
    spontaneously).
    """
    i = wp.tid()
    if i == 0:
        saved = data.src_count[0]
        cap = data.dst_keys.shape[0]
        if saved > cap:
            saved = cap
        data.dst_count[0] = saved
    if i < data.src_count[0]:
        data.dst_keys[i] = data.src_keys[i]

        p0 = data.src_point0[i]
        bid0 = data.shape_body[data.src_shape0[i]]
        if bid0 == -1:
            p0w = p0
        else:
            p0w = wp.transform_point(data.body_q[bid0], p0)

        p1 = data.src_point1[i]
        bid1 = data.shape_body[data.src_shape1[i]]
        if bid1 == -1:
            p1w = p1
        else:
            p1w = wp.transform_point(data.body_q[bid1], p1)

        data.dst_pos_world[i] = 0.5 * (p0w + p1w)
        data.dst_normal[i] = data.src_normal[i]


# ------------------------------------------------------------------
# Sticky canonical geometry overlay
# ------------------------------------------------------------------
#
# Sticky mode preserves only the fields that actually change across frames
# for a matched contact: the body-frame contact points (``point0``/``point1``)
# and offsets (``offset0``/``offset1``), plus the world-frame normal (which
# is already persisted for matching in ``prev_normal``, no extra allocation).
#
# Everything else is either key-derived or a per-shape constant that does
# not change between frames, so the new frame's values are already correct:
#
# - ``shape0`` / ``shape1``  : implied by the sort key; identical by
#   construction for matched contacts.
# - ``margin0`` / ``margin1``: ``radius_eff + margin``, per-shape constant.
# - ``stiffness`` / ``damping`` / ``friction``: per-shape constants, and
#   contact matching for hydroelastic contacts (the only path that writes
#   these) is not supported anyway.


@wp.struct
class _StickyOverlayData:
    """Prior canonical contacts plus current unsorted contact geometry."""

    match_index: wp.array[wp.int32]
    contact_count: wp.array[wp.int32]
    canonical_to_source: wp.array[wp.int32]
    previous_point0: wp.array[wp.vec3]
    previous_point1: wp.array[wp.vec3]
    previous_offset0: wp.array[wp.vec3]
    previous_offset1: wp.array[wp.vec3]
    previous_normal: wp.array[wp.vec3]
    current_point0: wp.array[wp.vec3]
    current_point1: wp.array[wp.vec3]
    current_offset0: wp.array[wp.vec3]
    current_offset1: wp.array[wp.vec3]
    current_normal: wp.array[wp.vec3]
    current_shape0: wp.array[wp.int32]
    current_shape1: wp.array[wp.int32]
    current_margin0: wp.array[wp.float32]
    current_margin1: wp.array[wp.float32]
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]
    overlay_point0: wp.array[wp.vec3]
    overlay_point1: wp.array[wp.vec3]
    overlay_offset0: wp.array[wp.vec3]
    overlay_offset1: wp.array[wp.vec3]
    overlay_normal: wp.array[wp.vec3]


@wp.kernel(enable_backward=False)
def _build_sticky_overlay_kernel(data: _StickyOverlayData):
    i = wp.tid()
    if i >= data.contact_count[0]:
        return
    source = data.canonical_to_source[i]
    use_previous = data.match_index[i] >= wp.int32(0)
    if use_previous:
        body0 = data.shape_body[data.current_shape0[source]]
        body1 = data.shape_body[data.current_shape1[source]]
        p0_world = data.current_point0[source]
        p1_world = data.current_point1[source]
        if body0 >= wp.int32(0):
            p0_world = wp.transform_point(data.body_q[body0], p0_world)
        if body1 >= wp.int32(0):
            p1_world = wp.transform_point(data.body_q[body1], p1_world)
        fresh_gap = wp.dot(p1_world - p0_world, data.current_normal[source]) - (
            data.current_margin0[source] + data.current_margin1[source]
        )
        use_previous = fresh_gap <= wp.float32(0.0)
    if use_previous:
        previous = data.match_index[i]
        data.overlay_point0[i] = data.previous_point0[previous]
        data.overlay_point1[i] = data.previous_point1[previous]
        data.overlay_offset0[i] = data.previous_offset0[previous]
        data.overlay_offset1[i] = data.previous_offset1[previous]
        data.overlay_normal[i] = data.previous_normal[previous]
    else:
        data.overlay_point0[i] = data.current_point0[source]
        data.overlay_point1[i] = data.current_point1[source]
        data.overlay_offset0[i] = data.current_offset0[source]
        data.overlay_offset1[i] = data.current_offset1[source]
        data.overlay_normal[i] = data.current_normal[source]


# ------------------------------------------------------------------
# Contact report kernels
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _collect_new_contacts_kernel(
    match_index: wp.array[wp.int32],
    contact_count: wp.array[wp.int32],
    new_indices: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
):
    """Collect indices of new or broken contacts (match_index < 0) after sorting."""
    i = wp.tid()
    # Clamp against the match_index capacity so an overflowed count
    # doesn't let threads read past the end of match_index.
    n_active = contact_count[0]
    cap = match_index.shape[0]
    if n_active > cap:
        n_active = cap
    if i >= n_active:
        return
    if match_index[i] < wp.int32(0):
        slot = wp.atomic_add(new_count, 0, wp.int32(1))
        new_indices[slot] = wp.int32(i)


@wp.kernel(enable_backward=False)
def _collect_broken_contacts_kernel(
    prev_was_matched: wp.array[wp.int32],
    prev_count: wp.array[wp.int32],
    broken_indices: wp.array[wp.int32],
    broken_count: wp.array[wp.int32],
):
    """Collect indices of old contacts that were not matched by any new contact."""
    i = wp.tid()
    # Clamp against ``prev_was_matched`` capacity so an overflowed
    # prev_count (set when the prior frame overflowed and the now-fixed
    # save path didn't clamp) doesn't OOB-read.
    n_old = prev_count[0]
    cap = prev_was_matched.shape[0]
    if n_old > cap:
        n_old = cap
    if i >= n_old:
        return
    if prev_was_matched[i] == wp.int32(0):
        slot = wp.atomic_add(broken_count, 0, wp.int32(1))
        broken_indices[slot] = wp.int32(i)


# ------------------------------------------------------------------
# ContactMatcher class
# ------------------------------------------------------------------


class ContactMatcher:
    """Match canonical contacts against retained keys, midpoints, and normals.

    Args:
        capacity: Maximum contact count.
        pos_threshold: Maximum midpoint distance [m] for a match.
        normal_dot_threshold: Minimum normal dot product for a match.
        contact_report: Whether to retain matched flags for reports.
        sticky: Whether to allocate canonical sticky geometry overlays.
        device: Device on which to allocate state.
    """

    def __init__(
        self,
        capacity: int,
        *,
        pos_threshold: float = 0.0005,
        normal_dot_threshold: float = 0.995,
        contact_report: bool = False,
        sticky: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            self._pos_threshold_sq = pos_threshold * pos_threshold
            self._normal_dot_threshold = normal_dot_threshold
            self._prev_pos_world = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_normal = wp.zeros(capacity, dtype=wp.vec3)

            # Sorted keys must survive across frames
            # (_sort_keys_copy is overwritten by _prepare_sort each frame).
            # Init with the sort-key sentinel so a debugger dump of the buffer
            # before the first save_sorted_state does not look like valid keys
            # for shape_a=0, shape_b=0, sub_key=0.
            self._prev_sorted_keys = wp.full(capacity, SORT_KEY_SENTINEL, dtype=wp.int64)
            self._prev_count = wp.zeros(1, dtype=wp.int32)

            # Per-prev claim word for the atomic_min race that keeps the
            # new→prev mapping injective (see module docstring).  Reset
            # to _CLAIM_SENTINEL each frame; the low 32 bits of the
            # surviving value identify the winning new contact by the low
            # 32 bits of its sort key (deterministic, invariant under
            # non-deterministic narrow-phase slot assignment -- see
            # ``_pack_claim``).
            self._prev_claim = wp.empty(capacity, dtype=wp.int64)

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)
            else:
                # Dummy single-element array so the Warp struct is always valid.
                self._prev_was_matched = wp.zeros(1, dtype=wp.int32)

            # Sticky mode also preserves the body-frame point/offset pairs
            # consumed after sorting. Shape indices, margins, and per-shape
            # properties are key-derived or constant for a matched contact.
            self._sticky = sticky
            if sticky:
                self._prev_point0 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_point1 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_offset0 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_offset1 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_normal_sticky = wp.zeros(capacity, dtype=wp.vec3)
            else:
                self._prev_point0 = None
                self._prev_point1 = None
                self._prev_offset0 = None
                self._prev_offset1 = None
                self._prev_normal_sticky = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_report(self) -> bool:
        """Whether the contact report buffers are allocated."""
        return self._has_report

    @property
    def is_sticky(self) -> bool:
        """Whether sticky-mode overlay buffers are allocated."""
        return self._sticky

    @property
    def prev_contact_count(self) -> wp.array[wp.int32]:
        """Device-side previous frame contact count (single-element int32)."""
        return self._prev_count

    def reset(self) -> None:
        """Clear cross-frame state so the next frame starts fresh.

        Use this after any discontinuity that invalidates the previous
        frame's contacts (RL episode reset, teleported bodies, scene
        reload).  After ``reset()`` the next :meth:`match` produces all
        :data:`MATCH_NOT_FOUND` and :meth:`build_report` reports zero broken
        contacts.  Zeroing ``_prev_count`` is sufficient because both kernels
        gate on it.
        """
        self._prev_count.zero_()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def match(
        self,
        sort_keys: wp.array[wp.int64],
        contact_count: wp.array[wp.int32],
        point0: wp.array[wp.vec3],
        point1: wp.array[wp.vec3],
        shape0: wp.array[wp.int32],
        shape1: wp.array[wp.int32],
        normal: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        shape_body: wp.array[wp.int32],
        match_index_out: wp.array[wp.int32],
        *,
        canonical_to_source: wp.array[wp.int32] | None = None,
        device: Devicelike = None,
    ) -> None:
        """Match current contacts against the previous sorted contacts.

        :class:`CollisionPipeline` calls this after deterministic sorting so
        current and previous pair searches are warp-coherent.

        Distance is measured between world-space contact midpoints
        (``0.5 * (world(point0) + world(point1))``) so the metric is symmetric
        in shape 0 / shape 1.

        Args:
            sort_keys: Current-frame int64 sort keys.
            contact_count: Single-element int array with active contact count.
            point0: Body-frame contact points on shape 0 (current frame).
            point1: Body-frame contact points on shape 1 (current frame).
            shape0: Shape indices for shape 0 (current frame).
            shape1: Shape indices for shape 1 (current frame).
            normal: Contact normals (current frame).
            body_q: Body transforms for the current frame.
            shape_body: Shape-to-body index map.
            match_index_out: Output int32 array to receive match results.
                Written directly (no intermediate copy).
            device: Device to launch on.
        """
        if self._has_report:
            self._prev_was_matched.zero_()

        # Reset only the active prefix of the claim buffer.  Launching
        # ``capacity`` threads keeps the call shape constant for graph
        # capture, but the kernel guards on ``prev_count`` so we touch
        # the minimum bytes — important for sparsely-loaded pipelines
        # where ``capacity >> prev_count``.
        wp.launch(
            _clear_prev_claim_kernel,
            dim=self._capacity,
            inputs=[self._prev_claim, self._prev_count],
            device=device,
        )

        data = _MatchData()
        data.prev_keys = self._prev_sorted_keys
        data.prev_pos_world = self._prev_pos_world
        data.prev_normal = self._prev_normal
        data.prev_count = self._prev_count
        data.new_keys = sort_keys
        data.new_point0 = point0
        data.new_point1 = point1
        data.new_shape0 = shape0
        data.new_shape1 = shape1
        data.new_normal = normal
        data.new_count = contact_count
        data.canonical_to_source = canonical_to_source if canonical_to_source is not None else match_index_out
        data.use_permutation = 1 if canonical_to_source is not None else 0
        data.body_q = body_q
        data.shape_body = shape_body
        data.match_index = match_index_out
        data.prev_claim = self._prev_claim
        data.pos_threshold_sq = self._pos_threshold_sq
        data.normal_dot_threshold = self._normal_dot_threshold

        wp.launch(_match_contacts_kernel, dim=self._capacity, inputs=[data], device=device)
        wp.launch(
            _resolve_claims_kernel,
            dim=self._capacity,
            inputs=[
                match_index_out,
                sort_keys,
                self._prev_claim,
                self._prev_was_matched,
                contact_count,
                canonical_to_source if canonical_to_source is not None else match_index_out,
                1 if canonical_to_source is not None else 0,
                1 if self._has_report else 0,
            ],
            device=device,
        )

    def build_sticky_overlay(
        self,
        contact_count: wp.array[wp.int32],
        match_index: wp.array[wp.int32],
        canonical_to_source: wp.array[wp.int32],
        *,
        previous_point0: wp.array[wp.vec3],
        previous_point1: wp.array[wp.vec3],
        previous_offset0: wp.array[wp.vec3],
        previous_offset1: wp.array[wp.vec3],
        previous_normal: wp.array[wp.vec3],
        current_point0: wp.array[wp.vec3],
        current_point1: wp.array[wp.vec3],
        current_offset0: wp.array[wp.vec3],
        current_offset1: wp.array[wp.vec3],
        current_normal: wp.array[wp.vec3],
        current_shape0: wp.array[wp.int32],
        current_shape1: wp.array[wp.int32],
        current_margin0: wp.array[wp.float32],
        current_margin1: wp.array[wp.float32],
        body_q: wp.array[wp.transform],
        shape_body: wp.array[wp.int32],
        device: Devicelike = None,
    ) -> None:
        """Build canonical sticky geometry before the full contact gather."""
        if not self._sticky:
            raise ValueError("sticky overlay requires sticky matching")
        data = _StickyOverlayData()
        data.match_index = match_index
        data.contact_count = contact_count
        data.canonical_to_source = canonical_to_source
        data.previous_point0 = previous_point0
        data.previous_point1 = previous_point1
        data.previous_offset0 = previous_offset0
        data.previous_offset1 = previous_offset1
        data.previous_normal = previous_normal
        data.current_point0 = current_point0
        data.current_point1 = current_point1
        data.current_offset0 = current_offset0
        data.current_offset1 = current_offset1
        data.current_normal = current_normal
        data.current_shape0 = current_shape0
        data.current_shape1 = current_shape1
        data.current_margin0 = current_margin0
        data.current_margin1 = current_margin1
        data.body_q = body_q
        data.shape_body = shape_body
        data.overlay_point0 = self._prev_point0
        data.overlay_point1 = self._prev_point1
        data.overlay_offset0 = self._prev_offset0
        data.overlay_offset1 = self._prev_offset1
        data.overlay_normal = self._prev_normal_sticky
        wp.launch(_build_sticky_overlay_kernel, dim=self._capacity, inputs=[data], device=device)

    @property
    def sticky_overlay_arrays(self) -> tuple[wp.array, wp.array, wp.array, wp.array, wp.array]:
        """Canonical sticky point, offset, and normal overlay arrays."""
        if not self._sticky:
            raise ValueError("sticky overlay requires sticky matching")
        return (
            self._prev_point0,
            self._prev_point1,
            self._prev_offset0,
            self._prev_offset1,
            self._prev_normal_sticky,
        )

    def save_sorted_state(
        self,
        sorted_keys: wp.array[wp.int64],
        contact_count: wp.array[wp.int32],
        sorted_point0: wp.array[wp.vec3],
        sorted_point1: wp.array[wp.vec3],
        sorted_shape0: wp.array[wp.int32],
        sorted_shape1: wp.array[wp.int32],
        sorted_normal: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        shape_body: wp.array[wp.int32],
        *,
        device: Devicelike = None,
    ) -> None:
        """Save canonical keys, world-space midpoints, and normals for matching."""
        data = _SaveStateData()
        data.src_keys = sorted_keys
        data.src_point0 = sorted_point0
        data.src_point1 = sorted_point1
        data.src_shape0 = sorted_shape0
        data.src_shape1 = sorted_shape1
        data.src_normal = sorted_normal
        data.src_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.dst_keys = self._prev_sorted_keys
        data.dst_pos_world = self._prev_pos_world
        data.dst_normal = self._prev_normal
        data.dst_count = self._prev_count

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

    def build_report(
        self,
        match_index: wp.array[wp.int32],
        contact_count: wp.array[wp.int32],
        new_indices: wp.array[wp.int32],
        new_count: wp.array[wp.int32],
        broken_indices: wp.array[wp.int32],
        broken_count: wp.array[wp.int32],
        *,
        device: Devicelike = None,
    ) -> None:
        """Build new/broken contact index lists (optional, post-sort).

        Must be called **after** :meth:`ContactSorter.sort_full` and **before**
        :meth:`save_sorted_state` (``save_sorted_state`` overwrites
        ``_prev_count``, which this method reads to bound the broken-contact
        enumeration).

        After this call, ``new_indices`` / ``new_count`` hold indices of
        contacts in the current sorted buffer that have no prior match
        (``match_index < 0``), and ``broken_indices`` / ``broken_count`` hold
        indices of old contacts that were not matched by any new contact.

        Args:
            match_index: Sorted match_index array (from :class:`Contacts`).
            contact_count: Single-element int array with active contact count.
            new_indices: Output array to receive new-contact indices.
            new_count: Single-element output counter for new contacts.
            broken_indices: Output array to receive broken-contact indices
                (indexing the previous frame's sorted buffer).
            broken_count: Single-element output counter for broken contacts.
            device: Device to launch on.
        """
        if not self._has_report:
            return

        new_count.zero_()
        broken_count.zero_()

        wp.launch(
            _collect_new_contacts_kernel,
            dim=self._capacity,
            inputs=[match_index, contact_count, new_indices, new_count],
            device=device,
        )
        wp.launch(
            _collect_broken_contacts_kernel,
            dim=self._capacity,
            inputs=[self._prev_was_matched, self._prev_count, broken_indices, broken_count],
            device=device,
        )
