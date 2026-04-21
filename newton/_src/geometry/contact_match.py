# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame contact matching via binary search on sorted contact keys.

Given the previous frame's sorted contacts (keys, world-space midpoints,
normals) and the current frame's unsorted contacts, this module finds
correspondences using the deterministic sort key from
:func:`~newton._src.geometry.contact_data.make_contact_sort_key`.

For each new contact the matcher binary-searches the previous frame's
sorted keys for the ``(shape_a, shape_b)`` pair range — ignoring the
``sort_sub_key`` bits — then picks the closest previous contact in that
range whose normal also passes the dot-product threshold.  "Closest" is
measured in world space between the contact *midpoints*
``0.5 * (world(point0) + world(point1))``, i.e. symmetric in shape 0 and
shape 1.  The result is a per-contact match index:

- ``>= 0``: index of the matched contact in the previous frame's sorted buffer.
- ``MATCH_NOT_FOUND (-1)``: shape pair has no prior contacts.
- ``MATCH_BROKEN (-2)``: shape pair exists but no contact within
  position/normal thresholds, *or* a closer new contact won the same
  prev contact in the uniqueness resolve pass.

Why ignore sort_sub_key
-----------------------
Multi-contact manifolds (e.g. box-box face-face) can rotate the
``sort_sub_key`` assignment across frames when their internal generation
order shifts (e.g. the Sutherland-Hodgman clip's starting vertex moves
by one slot), even though the physical contact points stay essentially
in place.  Matching on the full key would mark these contacts broken
every frame.  Pair counts are small (a few manifold points per pair),
so the linear scan inside the pair range is cheap.

One-to-one match via packed atomic_min
--------------------------------------
A pair-range scan can have multiple new contacts pick the same prev
contact as their closest.  To keep the mapping injective without
sorting or CAS retries, the matcher uses a single ``wp.atomic_min`` per
new contact on a per-prev ``int64`` claim word:

    claim = (float_flip(dist_sq) << 32) | tid

``float_flip`` reinterprets the non-negative ``dist_sq`` as a
sortable ``uint32``, so the high 32 bits order claims by ascending
distance; the low 32 bits hold the new contact index, breaking ties
deterministically (smallest ``tid`` wins).  After the match kernel
runs, a small finalize kernel reads ``prev_claim[best_idx]`` and
demotes any new contact whose ``tid`` does not appear in the low bits
to :data:`MATCH_BROKEN`.  Losers are *not* re-matched against a
second-closest prev (kept for simplicity and speed).

Cost: one ``int64[capacity]`` buffer, one ``wp.atomic_min`` per new
contact, and one short finalize kernel launch.  No ``atomic_cas``.

Memory efficiency
-----------------
The matcher reuses the :class:`ContactSorter`'s existing scratch buffers
(:attr:`ContactSorter.scratch_pos_world`, :attr:`ContactSorter.scratch_normal`)
to store previous-frame world-space contact midpoints and normals between
frames.  This works because matching runs *before* sorting each frame (so
the scratch data is still intact), and the save step runs *after* sorting
(overwriting the scratch with the new sorted data).  The only additional
per-contact allocation is the ``_prev_sorted_keys`` buffer (8 bytes/contact)
since the sorter's key buffer is overwritten by ``_prepare_sort`` each frame.

Per-frame call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.match(...)  # before ContactSorter.sort_full()
    sorter.sort_full(...)  # match_index is permuted with contacts
    matcher.replay_matched(...)  # sticky-only; overwrite matched rows
    matcher.build_report(...)  # optional; must precede save_sorted_state
    matcher.save_sorted_state(...)  # after sorting, replay, and report

The ordering matters: ``save_sorted_state`` overwrites ``_prev_count`` with
the current frame's count, while ``build_report`` reads the *old*
``_prev_count`` to bound the broken-contact enumeration, and sticky
``replay_matched`` must see the post-sort ``match_index`` and the pre-save
``_prev_*`` buffers it reads from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .contact_sort import SORT_KEY_SENTINEL

if TYPE_CHECKING:
    from .contact_sort import ContactSorter

MATCH_NOT_FOUND = wp.constant(wp.int32(-1))
"""Sentinel: no matching key found in last frame's contacts."""

MATCH_BROKEN = wp.constant(wp.int32(-2))
"""Sentinel: key found but position or normal threshold exceeded."""


# ------------------------------------------------------------------
# Warp helpers
# ------------------------------------------------------------------


# Sentinel value for unclaimed slots in ``_prev_claim``.  Larger than
# any packed (flipped_dist << 32 | tid) any kernel will ever produce,
# so the first ``atomic_min`` always wins.
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
def _pack_claim(dist_sq: float, tid: int) -> wp.int64:
    """Pack ``(dist_sq, tid)`` into a single int64 for ``atomic_min``.

    High 32 bits: ``float_flip(dist_sq)`` — ascending by distance.
    Low 32 bits:  ``tid`` — deterministic tie-break (smallest wins).
    """
    flipped = wp.int64(_float_flip(dist_sq))
    return (flipped << wp.int64(32)) | wp.int64(tid)


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

    # Current frame (unsorted).
    new_keys: wp.array[wp.int64]
    new_point0: wp.array[wp.vec3]
    new_point1: wp.array[wp.vec3]
    new_shape0: wp.array[wp.int32]
    new_shape1: wp.array[wp.int32]
    new_normal: wp.array[wp.vec3]
    new_count: wp.array[wp.int32]

    # Body transforms for world-space conversion
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Per-prev claim word, packed (float_flip(dist_sq) << 32 | tid).
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

    n_old = data.prev_count[0]
    if n_old == 0:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    target_key = data.new_keys[tid]

    # World-space midpoint of the two contact points (symmetric in shape 0 /
    # shape 1).  Matches the quantity persisted by ``_save_sorted_state_kernel``
    # for the previous frame, so both sides of ``diff`` below measure the same
    # physical quantity.
    p0 = data.new_point0[tid]
    bid0 = data.shape_body[data.new_shape0[tid]]
    if bid0 == -1:
        p0w = p0
    else:
        p0w = wp.transform_point(data.body_q[bid0], p0)

    p1 = data.new_point1[tid]
    bid1 = data.shape_body[data.new_shape1[tid]]
    if bid1 == -1:
        p1w = p1
    else:
        p1w = wp.transform_point(data.body_q[bid1], p1)

    new_pos_w = 0.5 * (p0w + p1w)
    new_n = data.new_normal[tid]

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
        # Closest distance wins; ties resolved by lowest tid.
        wp.atomic_min(data.prev_claim, best_idx, _pack_claim(best_dist_sq, tid))
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
    if i < prev_count[0]:
        prev_claim[i] = _CLAIM_SENTINEL


@wp.kernel(enable_backward=False)
def _resolve_claims_kernel(
    match_index: wp.array[wp.int32],
    prev_claim: wp.array[wp.int64],
    prev_was_matched: wp.array[wp.int32],
    new_count: wp.array[wp.int32],
    has_report: int,
):
    """Pass 2: keep winners, demote losers to :data:`MATCH_BROKEN`.

    The low 32 bits of ``prev_claim[cand]`` identify the winning
    ``tid``; everyone else who staked a claim on the same ``cand``
    becomes :data:`MATCH_BROKEN` (no second-closest fallback).
    """
    tid = wp.tid()
    if tid >= new_count[0]:
        return

    cand = match_index[tid]
    if cand < wp.int32(0):
        return  # already MATCH_NOT_FOUND or MATCH_BROKEN

    winner_tid = wp.int32(prev_claim[cand] & wp.int64(0xFFFFFFFF))
    if winner_tid == wp.int32(tid):
        if has_report != 0:
            prev_was_matched[cand] = wp.int32(1)
    else:
        match_index[tid] = MATCH_BROKEN


# ------------------------------------------------------------------
# Save sorted state kernel
# ------------------------------------------------------------------


@wp.struct
class _SaveStateData:
    """Bundled arrays for the save-sorted-state kernel.

    ``src_point0`` / ``src_point1`` and their shape indices are consumed every
    frame to compute the symmetric world-space midpoint written into
    ``dst_pos_world``.  The ``dst_point*_body`` columns are only written when
    ``has_sticky != 0``; when sticky is disabled the matcher passes dummy
    arrays for those slots and the kernel's ``if has_sticky`` guard skips the
    extra writes, so sticky-only columns need zero allocation in the
    non-sticky path.  The sticky mode does *not* persist ``offset0`` /
    ``offset1``: those are derived quantities (``margin * n_world`` rotated
    into the body frame) and are recomputed fresh in
    :func:`_replay_matched_kernel` from the persistent ``point0`` / ``point1``
    and ``normal``.
    """

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
    dst_point0_body: wp.array[wp.vec3]
    dst_point1_body: wp.array[wp.vec3]
    dst_count: wp.array[wp.int32]

    has_sticky: int


@wp.kernel(enable_backward=False)
def _save_sorted_state_kernel(data: _SaveStateData):
    """Copy sorted contacts into the previous-frame buffers for next-frame matching.

    The persisted ``dst_pos_world`` is the world-space *midpoint* of the two
    contact points, so the next frame's match kernel compares a shape-symmetric
    quantity.
    """
    i = wp.tid()
    if i == 0:
        data.dst_count[0] = data.src_count[0]
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

        if data.has_sticky != 0:
            data.dst_point0_body[i] = p0
            data.dst_point1_body[i] = p1


# ------------------------------------------------------------------
# Sticky-mode replay (matched rows only)
# ------------------------------------------------------------------
#
# Sticky mode pins the **creation-frame contact manifold** for matched
# rows so the solver sees the same body-local anchor pair and the same
# world-frame normal from frame to frame.  Concretely, three fields are
# overwritten from the previous frame's saved record:
#
# - ``point0`` / ``point1`` -- body-A and body-B local anchors.  These
#   rotate with their bodies through the current ``body_q`` each frame,
#   so the world-space anchors
#   ``bx_a = transform(X_wb_a, point0)``,
#   ``bx_b = transform(X_wb_b, point1)``
#   track the original surface location under arbitrary body motion.
#   This is exactly the same data model Jolt/Bullet/PhysX/PhoenX use
#   for persistent contact manifolds.
# - ``normal`` -- world-frame contact normal captured at creation.
#   Because the matcher only accepts a prev<->new pair when their
#   normals are within ``normal_dot_threshold``, the persistent normal
#   is guaranteed to be a small rotation of what the narrow phase
#   would have produced this frame.
#
# The fourth pair of contact-geometry fields, ``offset0`` / ``offset1``,
# is **not** persisted.  Those offsets are *derived*: the narrow phase
# writes ``offset_i = transform_vector(X_bw_i, margin_i * n_world)`` --
# i.e. the normal-direction thickness padding, re-expressed in the body
# frame as of that frame.  Persisting them directly and later
# transforming them back through the *current* body frame would
# reintroduce whichever normal was current when we saved, not the
# persistent normal we actually want to use.  Instead,
# :func:`_replay_matched_kernel` **recomputes** ``offset0`` /
# ``offset1`` on every frame from the persistent ``normal`` and the
# per-shape ``margin0`` / ``margin1`` using the *current* body
# transform, with exactly the same formula as
# :func:`~newton._src.sim.collide.write_contact`::
#
#     offset0 = transform_vector(X_bw_a,  margin0 * n_persistent)
#     offset1 = transform_vector(X_bw_b, -margin1 * n_persistent)
#
# This keeps the friction offsets self-consistent with the persistent
# normal: after rotating both quantities back to world space the
# offsets collapse to ``+/-margin * n_persistent`` by construction,
# contributing exactly zero tangential component to the solver's
# friction delta -- which is the invariant we need for stable
# static-friction behaviour in XPBD.
#
# The remaining non-geometric fields are either key-derived or
# per-shape constants and so are already identical for a matched
# contact on the new frame:
#
# - ``shape0`` / ``shape1``  : implied by the sort key; identical by
#   construction for matched contacts.
# - ``margin0`` / ``margin1``: ``radius_eff + margin``, per-shape constant
#   (and consumed below to recompute ``offset0`` / ``offset1``).
# - ``stiffness`` / ``damping`` / ``friction``: per-shape constants, and
#   contact matching for hydroelastic contacts (the only path that writes
#   these) is not supported anyway.


@wp.struct
class _ReplayData:
    """Bundled arrays for the sticky replay kernel."""

    match_index: wp.array[wp.int32]
    contact_count: wp.array[wp.int32]

    # Persistent (prev-frame) state
    prev_point0: wp.array[wp.vec3]
    prev_point1: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]

    # Per-contact inputs needed to recompute offsets fresh
    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    margin0: wp.array[float]
    margin1: wp.array[float]
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Outputs -- overwritten in-place for matched rows
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]

    # Diagnostic: single-element int32 atomic counter incremented every
    # time a matched row actually takes the replay branch (i.e. the
    # persistent manifold was committed byte-for-byte to point0/point1
    # and the persistent world-frame normal was written to normal).
    # Matched rows that fail the drift guard keep their fresh
    # narrow-phase data and are *not* counted.
    replayed_count: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _replay_matched_kernel(data: _ReplayData):
    tid = wp.tid()
    if tid >= data.contact_count[0]:
        return
    idx = data.match_index[tid]
    if idx < wp.int32(0):
        return  # MATCH_NOT_FOUND or MATCH_BROKEN -- keep new-frame data.

    # Resolve current body transforms once: needed below both to
    # evaluate the "keep deeper" guard and to recompute body-frame
    # friction offsets.
    bid0 = data.shape_body[data.shape0[tid]]
    if bid0 == -1:
        X_wb_a = wp.transform_identity()
    else:
        X_wb_a = data.body_q[bid0]

    bid1 = data.shape_body[data.shape1[tid]]
    if bid1 == -1:
        X_wb_b = wp.transform_identity()
    else:
        X_wb_b = data.body_q[bid1]

    thickness = data.margin0[tid] + data.margin1[tid]

    # "Drift guard": the solver penetration depth is
    #     d = dot(n, bx_b - bx_a) - thickness
    # with d < 0 meaning penetration.  Compute d for the fresh
    # narrow-phase manifold point and for the persistent manifold
    # point rotated through the current body transforms.
    #
    # If the persistent contact is *deeper* than the fresh one
    # (d_old < d_new), the persistent anchors have drifted into the
    # body: rotating the stored body-local anchors through the new
    # body_q no longer matches the actual surface.  Replaying such a
    # drifted contact hands XPBD a phantom penetration bigger than the
    # real one, which is what blows up pyramid stacks.  In that case
    # discard the persistent record and keep the fresh narrow-phase
    # values (effectively behaving like LATEST for that row, but with
    # the match bookkeeping still intact for the report).
    #
    # If the fresh contact is deeper (d_new < d_old), the persistent
    # anchor is the less penetrating of the two and is safe to use as
    # a sticky anchor -- it preserves frame-to-frame geometry without
    # over-correcting.  This is analogous to PhoenX's manifold
    # reduction keeping the "deepest" candidate per normal bin: we
    # keep whichever of the two candidates (fresh vs persistent)
    # reports the larger penetration.
    p0_new = data.point0[tid]
    p1_new = data.point1[tid]
    n_new = data.normal[tid]
    bx_a_new = wp.transform_point(X_wb_a, p0_new)
    bx_b_new = wp.transform_point(X_wb_b, p1_new)
    d_new = wp.dot(n_new, bx_b_new - bx_a_new) - thickness

    p0_old = data.prev_point0[idx]
    p1_old = data.prev_point1[idx]
    n_old = data.prev_normal[idx]
    bx_a_old = wp.transform_point(X_wb_a, p0_old)
    bx_b_old = wp.transform_point(X_wb_b, p1_old)
    d_old = wp.dot(n_old, bx_b_old - bx_a_old) - thickness

    if d_old < d_new:
        # Persistent contact has drifted to a deeper-than-real
        # penetration: its rotated anchors no longer reflect the
        # actual surface geometry.  Keep the fresh narrow-phase
        # manifold (already in contacts.point0/1/normal/offset0/1).
        return

    # Otherwise the persistent contact is shallower (or equal) than
    # the fresh one -- replay its body-local anchors and world-frame
    # normal to pin the manifold across frames.
    data.point0[tid] = p0_old
    data.point1[tid] = p1_old
    data.normal[tid] = n_old
    wp.atomic_add(data.replayed_count, 0, wp.int32(1))

    # Recompute body-frame friction offsets from (margin, n_persistent,
    # current body_q).  This matches the formula used by the narrow
    # phase's write_contact() so the offsets stay self-consistent with
    # the persistent normal after rotating back to world space.
    X_bw_a = wp.transform_inverse(X_wb_a)
    X_bw_b = wp.transform_inverse(X_wb_b)
    data.offset0[tid] = wp.transform_vector(X_bw_a, data.margin0[tid] * n_old)
    data.offset1[tid] = wp.transform_vector(X_bw_b, -data.margin1[tid] * n_old)


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
    if i >= contact_count[0]:
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
    if i >= prev_count[0]:
        return
    if prev_was_matched[i] == wp.int32(0):
        slot = wp.atomic_add(broken_count, 0, wp.int32(1))
        broken_indices[slot] = wp.int32(i)


# ------------------------------------------------------------------
# ContactMatcher class
# ------------------------------------------------------------------


class ContactMatcher:
    """Frame-to-frame contact matching using binary search on sorted keys.

    Internal helper owned by :class:`~newton.CollisionPipeline`.  All user-visible
    results (match index, new/broken index lists) are surfaced on the
    :class:`~newton.Contacts` container; this class only owns cross-frame state.

    Pre-allocates all buffers at construction time for CUDA graph capture
    compatibility.  See the module docstring for the per-frame call order and
    the ordering constraints between :meth:`match`, :meth:`replay_matched`,
    :meth:`build_report`, and :meth:`save_sorted_state`.

    Memory is minimised by reusing the sorter's existing scratch buffers for
    the previous-frame world-space contact midpoints and normals.  The matcher
    owns two small per-contact buffers in addition: the sorted key cache
    (8 bytes/contact) and the per-prev claim word used by the ``atomic_min``
    race that keeps new→prev injective (8 bytes/contact).  When
    ``contact_report`` is disabled, the ``prev_was_matched`` flag array is
    also skipped.

    .. note::
        Previous-frame state persists across :meth:`~newton.CollisionPipeline.collide`
        calls — that is the whole point.  In RL-style workflows where the user
        resets or teleports bodies between episodes, call :meth:`reset` after
        such discontinuities so the next frame starts fresh with all
        :data:`MATCH_NOT_FOUND`.

    Args:
        capacity: Maximum number of contacts (must match :class:`ContactSorter`).
        sorter: The :class:`ContactSorter` whose scratch buffers will be
            reused for storing previous-frame positions and normals.
        pos_threshold: World-space distance threshold [m] between the
            previous and current contact midpoints
            ``0.5 * (world(point0) + world(point1))``.  Contacts whose midpoint
            moved more than this between frames are considered broken.
        normal_dot_threshold: Minimum dot product between old and new contact
            normals.  Below this the contact is considered broken.
        contact_report: Allocate the ``prev_was_matched`` flag array needed
            to enumerate broken contacts in :meth:`build_report`.
        sticky: Allocate two extra per-contact ``wp.vec3`` buffers
            (``point0`` / ``point1``, body-frame) used by
            :meth:`replay_matched` to pin the matched-set contact
            anchor geometry from the frame a contact was first
            created.  The world-frame ``normal`` is persisted too but
            reuses the sorter's existing ``scratch_normal`` buffer (no
            extra allocation -- it already stores the previous frame's
            normals for the matching dot-product check).  The
            body-frame ``offset0`` / ``offset1`` are **not** persisted;
            they are recomputed fresh each frame inside
            :func:`_replay_matched_kernel` from ``margin_i *
            n_persistent`` rotated through the current body frames
            (see the module-level comment above that kernel for the
            rationale).  When ``False`` the ``_prev_point*``
            attributes are ``None`` and no extra kernel launches are
            added.
        device: Device to allocate on.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sorter: ContactSorter,
        pos_threshold: float = 0.005,
        normal_dot_threshold: float = 0.9,
        contact_report: bool = False,
        sticky: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            self._pos_threshold_sq = pos_threshold * pos_threshold
            self._normal_dot_threshold = normal_dot_threshold
            self._sorter = sorter

            # Only buffer we must own: sorted keys survive across frames
            # (_sort_keys_copy is overwritten by _prepare_sort each frame).
            # Init with the sort-key sentinel so a debugger dump of the buffer
            # before the first save_sorted_state does not look like valid keys
            # for shape_a=0, shape_b=0, sub_key=0.
            self._prev_sorted_keys = wp.full(capacity, SORT_KEY_SENTINEL, dtype=wp.int64)
            self._prev_count = wp.zeros(1, dtype=wp.int32)

            # Per-prev claim word for the atomic_min race that keeps the
            # new→prev mapping injective (see module docstring).  Reset
            # to _CLAIM_SENTINEL each frame; the low 32 bits of the
            # surviving value identify the winning new contact ``tid``.
            self._prev_claim = wp.empty(capacity, dtype=wp.int64)

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)
            else:
                # Dummy single-element array so the Warp struct is always valid.
                self._prev_was_matched = wp.zeros(1, dtype=wp.int32)

            # Sticky-mode buffers.  We persist the body-frame contact
            # anchor points (``point0`` / ``point1``) and the world-
            # frame contact normal across frames.  The normal reuses
            # the sorter's ``scratch_normal``, which already stores
            # the previous frame's sorted normal for the matcher's
            # dot-product gate, so no extra allocation is needed.
            # ``offset0`` / ``offset1`` are *not* persisted -- they
            # are derived from ``margin * n_world`` in the body frame
            # and are recomputed fresh each frame inside
            # :func:`_replay_matched_kernel` from the persistent
            # normal and the current body transforms.  Shape indices,
            # margins, and per-shape properties are either key-derived
            # or per-shape constants and so identical on the next
            # frame for a matched contact.  See the module-level
            # comment above ``_replay_matched_kernel`` for the full
            # rationale.
            self._sticky = sticky
            if sticky:
                self._prev_point0 = wp.zeros(capacity, dtype=wp.vec3)
                self._prev_point1 = wp.zeros(capacity, dtype=wp.vec3)
                # Diagnostic counter: how many matched rows took the
                # replay branch in the most recent ``replay_matched``
                # call.  Single-element device array so it can be read
                # from host via ``.numpy()`` without copying the whole
                # contacts buffer.
                self._replayed_count = wp.zeros(1, dtype=wp.int32)
            else:
                self._prev_point0 = None
                self._prev_point1 = None
                self._replayed_count = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_report(self) -> bool:
        """Whether the contact report buffers are allocated."""
        return self._has_report

    @property
    def is_sticky(self) -> bool:
        """Whether sticky-mode full-record buffers are allocated."""
        return self._sticky

    @property
    def prev_contact_count(self) -> wp.array:
        """Device-side previous frame contact count (single-element int32)."""
        return self._prev_count

    @property
    def replayed_count(self) -> wp.array | None:
        """Device-side count of matched rows whose anchors+normal were
        replayed byte-for-byte from the previous frame in the most recent
        :meth:`replay_matched` call.

        Returns ``None`` when sticky mode is disabled.  In sticky mode
        this equals the number of matched rows minus the drift-guard
        rejections, so a matched contact always falls into exactly one
        of three buckets: (a) replayed from the previous frame,
        (b) matched but kept-fresh because the persistent anchors had
        drifted deeper than the fresh manifold, (c) broken.
        """
        return self._replayed_count

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
        sort_keys: wp.array,
        contact_count: wp.array,
        point0: wp.array,
        point1: wp.array,
        shape0: wp.array,
        shape1: wp.array,
        normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        match_index_out: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Match current unsorted contacts against last frame's sorted contacts.

        Must be called **before** :meth:`ContactSorter.sort_full`.

        Distance is measured between world-space contact midpoints
        (``0.5 * (world(point0) + world(point1))``) so the metric is symmetric
        in shape 0 / shape 1.

        Args:
            sort_keys: Current frame's unsorted int64 sort keys.
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
        # Reuse sorter scratch buffers for prev-frame world-space data.
        data.prev_pos_world = self._sorter.scratch_pos_world
        data.prev_normal = self._sorter.scratch_normal
        data.prev_count = self._prev_count
        data.new_keys = sort_keys
        data.new_point0 = point0
        data.new_point1 = point1
        data.new_shape0 = shape0
        data.new_shape1 = shape1
        data.new_normal = normal
        data.new_count = contact_count
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
                self._prev_claim,
                self._prev_was_matched,
                contact_count,
                1 if self._has_report else 0,
            ],
            device=device,
        )

    def save_sorted_state(
        self,
        sorted_keys: wp.array,
        contact_count: wp.array,
        sorted_point0: wp.array,
        sorted_point1: wp.array,
        sorted_shape0: wp.array,
        sorted_shape1: wp.array,
        sorted_normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Save current frame's sorted contacts for next-frame matching.

        Must be called **after** :meth:`ContactSorter.sort_full`.  The
        world-space midpoint of ``sorted_point0``/``sorted_point1`` and the
        sorted normal are written into the sorter's scratch buffers
        (:attr:`ContactSorter.scratch_pos_world` /
        :attr:`ContactSorter.scratch_normal`), which are idle between frames.

        When the matcher was built with ``sticky=True``, the body-frame
        contact anchor points (``sorted_point0`` / ``sorted_point1``) are
        also persisted for :meth:`replay_matched` in the same kernel
        launch.  The world-frame normal is already persisted into the
        sorter's ``scratch_normal`` (needed anyway for the matcher's
        normal-dot gate) and is reused by :meth:`replay_matched`.  The
        body-frame ``offset0`` / ``offset1`` are *not* persisted -- they
        are recomputed from the persistent normal and current body
        transforms inside :func:`_replay_matched_kernel`.

        Args:
            sorted_keys: Sorted int64 keys (use :attr:`ContactSorter.sorted_keys_view`).
            contact_count: Single-element int array with active contact count.
            sorted_point0: Sorted body-frame contact points on shape 0.
            sorted_point1: Sorted body-frame contact points on shape 1.
            sorted_shape0: Sorted shape 0 indices.
            sorted_shape1: Sorted shape 1 indices.
            sorted_normal: Sorted contact normals.
            body_q: Body transforms (current frame).
            shape_body: Shape-to-body index map.
            device: Device to launch on.
        """
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
        # Write world-space midpoint and normal into the sorter's scratch buffers.
        data.dst_pos_world = self._sorter.scratch_pos_world
        data.dst_normal = self._sorter.scratch_normal
        data.dst_count = self._prev_count

        if self._sticky:
            data.dst_point0_body = self._prev_point0
            data.dst_point1_body = self._prev_point1
            data.has_sticky = 1
        else:
            # The struct requires a valid array for every field -- the
            # kernel guards with has_sticky==0 and never reads/writes them.
            data.dst_point0_body = self._sorter.scratch_pos_world
            data.dst_point1_body = self._sorter.scratch_pos_world
            data.has_sticky = 0

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

    def replay_matched(
        self,
        contact_count: wp.array,
        match_index: wp.array,
        *,
        point0: wp.array,
        point1: wp.array,
        normal: wp.array,
        offset0: wp.array,
        offset1: wp.array,
        shape0: wp.array,
        shape1: wp.array,
        margin0: wp.array,
        margin1: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        device: Devicelike = None,
    ) -> None:
        """Overwrite matched rows with the saved previous-frame contact manifold.

        Only valid when the matcher was constructed with ``sticky=True``.  Must
        run **after** :meth:`ContactSorter.sort_full` and **before**
        :meth:`save_sorted_state`.  Unmatched rows (``match_index < 0``) are
        left untouched so new contacts keep their fresh narrow-phase geometry.

        For each matched row the kernel restores the body-frame anchor pair
        ``point0`` / ``point1`` and the world-frame ``normal`` from the prior
        frame's saved record, then **recomputes** the body-frame friction
        offsets ``offset0`` / ``offset1`` from ``margin_i * n_persistent``
        rotated through the *current* body transforms -- the same formula
        used by the narrow phase.  This keeps offsets self-consistent with
        the persistent normal and avoids the spurious tangent error that
        would result from pairing a stale stored offset with a freshly
        rotated body.  See the module-level comment above
        :func:`_replay_matched_kernel` for the full rationale.

        Args:
            contact_count: Single-element int array with the active contact count.
            match_index: Sorted match_index array (from :class:`Contacts`).
            point0, point1: Current-frame sorted body-frame contact anchor
                points (overwritten on matched rows).
            normal: Current-frame sorted world-frame contact normal
                (overwritten on matched rows with the creation-frame normal).
            offset0, offset1: Current-frame sorted body-frame friction
                offsets (recomputed from persistent normal + current
                body transforms on matched rows).
            shape0, shape1: Sorted shape indices for the two sides.
            margin0, margin1: Sorted per-contact offset magnitudes
                ``radius_eff + margin`` for the two sides.
            body_q: Body transforms (current frame).
            shape_body: Shape-to-body index map.
            device: Device to launch on.
        """
        if not self._sticky:
            raise ValueError("replay_matched requires the matcher to be constructed with sticky=True")

        # Reset the diagnostic counter before the kernel launches: each
        # matched row that actually commits the replay atomically bumps
        # this, so after the launch it holds the number of 1:1 copies
        # carried forward from the previous frame (see module comment
        # above ``_replay_matched_kernel``).
        self._replayed_count.zero_()

        data = _ReplayData()
        data.match_index = match_index
        data.contact_count = contact_count
        data.prev_point0 = self._prev_point0
        data.prev_point1 = self._prev_point1
        # The persistent world-frame normal lives in the sorter's
        # scratch_normal (written by save_sorted_state and already used
        # by the matcher's dot-product gate); reuse it directly.
        data.prev_normal = self._sorter.scratch_normal
        data.shape0 = shape0
        data.shape1 = shape1
        data.margin0 = margin0
        data.margin1 = margin1
        data.body_q = body_q
        data.shape_body = shape_body
        data.point0 = point0
        data.point1 = point1
        data.normal = normal
        data.offset0 = offset0
        data.offset1 = offset1
        data.replayed_count = self._replayed_count

        wp.launch(_replay_matched_kernel, dim=self._capacity, inputs=[data], device=device)

    def build_report(
        self,
        match_index: wp.array,
        contact_count: wp.array,
        new_indices: wp.array,
        new_count: wp.array,
        broken_indices: wp.array,
        broken_count: wp.array,
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
