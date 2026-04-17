# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-frame contact matching via binary search on sorted contact keys.

Given the previous frame's sorted contacts (keys, world-space positions,
normals) and the current frame's unsorted contacts, this module finds
correspondences using the deterministic sort key from
:func:`~newton._src.geometry.contact_data.make_contact_sort_key`.

For each new contact the matcher performs a binary search on last frame's
sorted keys, then verifies candidates against world-space position distance
and normal-direction thresholds.  The result is a per-contact match index:

- ``>= 0``: index of the matched contact in the previous frame's sorted buffer.
- ``MATCH_NOT_FOUND (-1)``: new contact with no prior correspondence.
- ``MATCH_BROKEN (-2)``: key matched but position/normal thresholds exceeded.

Key-uniqueness assumption
-------------------------
Matching assumes the sort keys produced by
:func:`~newton._src.geometry.contact_data.make_contact_sort_key` are unique
per active contact.  ``make_contact_sort_key`` silently masks overflow, so
scenes that exceed the bit budgets documented there (e.g. more than ~524K
mesh-triangle contacts after multi-contact expansion) can collide two keys
onto the same value.  When that happens, two new contacts will binary-search
to the same old range and may pick the same ``best_idx``, producing
duplicate ``match_index`` values.  The invariant is inherited from the sort
key itself; if you need to diagnose duplicate matches, start there.

Memory efficiency
-----------------
The matcher reuses the :class:`ContactSorter`'s existing scratch buffers
(:attr:`ContactSorter.scratch_pos_world`, :attr:`ContactSorter.scratch_normal`)
to store previous-frame world-space positions and normals between frames.
This works because matching runs *before* sorting each frame (so the
scratch data is still intact), and the save step runs *after* sorting
(overwriting the scratch with the new sorted data).  The only additional
per-contact allocation is the ``_prev_sorted_keys`` buffer (8 bytes/contact)
since the sorter's key buffer is overwritten by ``_prepare_sort`` each frame.

Per-frame call order (inside :class:`~newton.CollisionPipeline`)::

    matcher.match(...)  # before ContactSorter.sort_full()
    sorter.sort_full(...)  # match_index is permuted with contacts
    matcher.build_report(...)  # optional; must precede save_sorted_state
    matcher.save_sorted_state(...)  # after sorting and report

The ``build_report`` / ``save_sorted_state`` ordering is load-bearing:
``save_sorted_state`` overwrites ``_prev_count`` with the current frame's
count, while ``build_report`` reads the *old* ``_prev_count`` to bound the
broken-contact enumeration.  Swapping the two silently narrows the
broken-contact scan to the current frame's active range.
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


@wp.func
def _binary_search_int64(
    lower: int,
    upper: int,
    target: wp.int64,
    keys: wp.array[wp.int64],
) -> int:
    """Lower-bound binary search on a sorted int64 array.

    Returns the index of the first occurrence of *target*, or ``-1``
    if *target* is not present in ``keys[lower:upper]``.
    """
    left = lower
    right = upper
    while left < right:
        mid = left + (right - left) // 2
        if keys[mid] < target:
            left = mid + 1
        else:
            right = mid
    if left >= upper or keys[left] != target:
        return -1
    return left


# ------------------------------------------------------------------
# Match kernel
# ------------------------------------------------------------------


@wp.struct
class _MatchData:
    """Bundled arrays for the contact match kernel."""

    # Previous frame (sorted) — pos/normal reuse ContactSorter scratch buffers
    prev_keys: wp.array[wp.int64]
    prev_pos_world: wp.array[wp.vec3]
    prev_normal: wp.array[wp.vec3]
    prev_count: wp.array[wp.int32]

    # Current frame (unsorted)
    new_keys: wp.array[wp.int64]
    new_point0: wp.array[wp.vec3]
    new_shape0: wp.array[wp.int32]
    new_normal: wp.array[wp.vec3]
    new_count: wp.array[wp.int32]

    # Body transforms for world-space conversion
    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    # Outputs
    match_index: wp.array[wp.int32]
    prev_was_matched: wp.array[wp.int32]

    # Thresholds
    pos_threshold_sq: float
    normal_dot_threshold: float
    has_report: int


@wp.kernel(enable_backward=False)
def _match_contacts_kernel(data: _MatchData):
    """Match each new contact against the previous frame's sorted contacts."""
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

    # Compute world-space position for the new contact.
    bid = data.shape_body[data.new_shape0[tid]]
    if bid == -1:
        new_pos_w = data.new_point0[tid]
    else:
        new_pos_w = wp.transform_point(data.body_q[bid], data.new_point0[tid])
    new_n = data.new_normal[tid]

    # Binary search for the first old contact with matching key.
    start = _binary_search_int64(0, n_old, target_key, data.prev_keys)

    if start == -1:
        data.match_index[tid] = MATCH_NOT_FOUND
        return

    # Linear scan through contacts sharing the same key.
    best_idx = int(-1)
    best_dist_sq = float(data.pos_threshold_sq)
    k = int(0)
    while start + k < n_old:
        old_idx = start + k
        if data.prev_keys[old_idx] != target_key:
            break

        old_pos = data.prev_pos_world[old_idx]
        diff = new_pos_w - old_pos
        dist_sq = wp.dot(diff, diff)
        old_n = data.prev_normal[old_idx]
        ndot = wp.dot(new_n, old_n)

        if dist_sq <= best_dist_sq and ndot >= data.normal_dot_threshold:
            best_dist_sq = dist_sq
            best_idx = old_idx

        k += 1

    if best_idx >= 0:
        data.match_index[tid] = wp.int32(best_idx)
        if data.has_report != 0:
            wp.atomic_max(data.prev_was_matched, best_idx, wp.int32(1))
    else:
        # Key matched but thresholds not met.
        data.match_index[tid] = MATCH_BROKEN


# ------------------------------------------------------------------
# Save sorted state kernel
# ------------------------------------------------------------------


@wp.struct
class _SaveStateData:
    """Bundled arrays for the save-sorted-state kernel."""

    src_keys: wp.array[wp.int64]
    src_point0: wp.array[wp.vec3]
    src_shape0: wp.array[wp.int32]
    src_normal: wp.array[wp.vec3]
    src_count: wp.array[wp.int32]

    body_q: wp.array[wp.transform]
    shape_body: wp.array[wp.int32]

    dst_keys: wp.array[wp.int64]
    dst_pos_world: wp.array[wp.vec3]
    dst_normal: wp.array[wp.vec3]
    dst_count: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _save_sorted_state_kernel(data: _SaveStateData):
    """Copy sorted contacts into the previous-frame buffers for next-frame matching."""
    i = wp.tid()
    if i == 0:
        data.dst_count[0] = data.src_count[0]
    if i < data.src_count[0]:
        data.dst_keys[i] = data.src_keys[i]
        bid = data.shape_body[data.src_shape0[i]]
        if bid == -1:
            data.dst_pos_world[i] = data.src_point0[i]
        else:
            data.dst_pos_world[i] = wp.transform_point(data.body_q[bid], data.src_point0[i])
        data.dst_normal[i] = data.src_normal[i]


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
    compatibility.  The typical per-frame call sequence is::

        matcher.match(...)  # before ContactSorter.sort_full()
        sorter.sort_full(...)  # match_index is permuted with contacts
        matcher.build_report(...)  # optional; must precede save_sorted_state
        matcher.save_sorted_state(...)  # after sorting and report

    The ``build_report`` → ``save_sorted_state`` order is required:
    ``save_sorted_state`` overwrites ``_prev_count`` with the current frame's
    count, while ``build_report`` reads the *old* ``_prev_count`` to bound
    the broken-contact enumeration.

    Memory is minimised by reusing the *sorter*'s existing scratch buffers
    for the previous-frame world-space positions and normals.  The only
    new per-contact allocation is the sorted-key cache (8 bytes/contact).
    When ``contact_report`` is disabled, the ``prev_was_matched`` flag
    array (4 bytes/contact) is also skipped.

    .. note::
        Previous-frame state persists across :meth:`~newton.CollisionPipeline.collide`
        calls — that is the whole point.  But in RL-style workflows where a
        user resets or teleports all bodies between episodes, the stale
        previous-frame data will produce spurious matches on the next frame.
        Call :meth:`reset` after such discontinuities to zero ``_prev_count``
        so the next frame starts fresh with all ``MATCH_NOT_FOUND``.

    .. note::
        Requires a CUDA device.  The underlying :class:`ContactSorter`
        depends on ``wp.utils.radix_sort_pairs``, which is CUDA-only in
        practice, so the matcher is only supported on CUDA devices.

    Args:
        capacity: Maximum number of contacts (must match :class:`ContactSorter`).
        sorter: The :class:`ContactSorter` whose scratch buffers will be
            reused for storing previous-frame positions and normals.
        pos_threshold: World-space distance threshold [m].  Contacts whose
            positions moved more than this between frames are considered broken.
        normal_dot_threshold: Minimum dot product between old and new contact
            normals.  Below this the contact is considered broken.
        contact_report: Allocate the ``prev_was_matched`` flag array needed
            to enumerate broken contacts in :meth:`build_report`.
        device: Device to allocate on.  Must be a CUDA device.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sorter: ContactSorter,
        pos_threshold: float = 0.02,
        normal_dot_threshold: float = 0.9,
        contact_report: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            resolved_device = wp.get_device()
            if not resolved_device.is_cuda:
                raise RuntimeError(
                    "ContactMatcher requires a CUDA device; got "
                    f"'{resolved_device}'.  The underlying ContactSorter relies "
                    "on wp.utils.radix_sort_pairs, which is CUDA-only."
                )

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

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)
            else:
                # Dummy single-element array so the Warp struct is always valid.
                self._prev_was_matched = wp.zeros(1, dtype=wp.int32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_report(self) -> bool:
        """Whether the contact report buffers are allocated."""
        return self._has_report

    @property
    def prev_contact_count(self) -> wp.array:
        """Device-side previous frame contact count (single-element int32)."""
        return self._prev_count

    def reset(self) -> None:
        """Clear cross-frame state so the next frame starts fresh.

        Use this after any discontinuity that invalidates the previous
        frame's contacts — e.g. resetting an RL environment, teleporting
        bodies, or loading a new scene.  After ``reset()`` the next call to
        :meth:`match` produces all :data:`MATCH_NOT_FOUND` and
        :meth:`build_report` reports zero broken contacts.

        Zeroing ``_prev_count`` is sufficient to short-circuit both paths;
        the previous-frame key and position buffers are not touched, but
        they will not be read because both kernels gate on ``_prev_count``
        / ``prev_count``.
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
        shape0: wp.array,
        normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        match_index_out: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Match current unsorted contacts against last frame's sorted contacts.

        Must be called **before** :meth:`ContactSorter.sort_full`.

        Args:
            sort_keys: Current frame's unsorted int64 sort keys.
            contact_count: Single-element int array with active contact count.
            point0: Body-frame contact points on shape 0 (current frame).
            shape0: Shape indices for shape 0 (current frame).
            normal: Contact normals (current frame).
            body_q: Body transforms for the current frame.
            shape_body: Shape-to-body index map.
            match_index_out: Output int32 array to receive match results.
                Written directly (no intermediate copy).
            device: Device to launch on.
        """
        if self._has_report:
            self._prev_was_matched.zero_()

        data = _MatchData()
        data.prev_keys = self._prev_sorted_keys
        # Reuse sorter scratch buffers for prev-frame world-space data.
        data.prev_pos_world = self._sorter.scratch_pos_world
        data.prev_normal = self._sorter.scratch_normal
        data.prev_count = self._prev_count
        data.new_keys = sort_keys
        data.new_point0 = point0
        data.new_shape0 = shape0
        data.new_normal = normal
        data.new_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.match_index = match_index_out
        data.prev_was_matched = self._prev_was_matched
        data.pos_threshold_sq = self._pos_threshold_sq
        data.normal_dot_threshold = self._normal_dot_threshold
        data.has_report = 1 if self._has_report else 0

        wp.launch(_match_contacts_kernel, dim=self._capacity, inputs=[data], device=device)

    def save_sorted_state(
        self,
        sorted_keys: wp.array,
        contact_count: wp.array,
        sorted_point0: wp.array,
        sorted_shape0: wp.array,
        sorted_normal: wp.array,
        body_q: wp.array,
        shape_body: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Save current frame's sorted contacts for next-frame matching.

        Must be called **after** :meth:`ContactSorter.sort_full`.

        World-space positions and normals are written into the sorter's scratch
        buffers (:attr:`ContactSorter.scratch_pos_world`,
        :attr:`ContactSorter.scratch_normal`), which are idle between frames.
        Sorted keys go to an owned buffer since the sorter's key buffer is
        overwritten by ``_prepare_sort`` each frame.

        Args:
            sorted_keys: Sorted int64 keys (use :attr:`ContactSorter.sorted_keys_view`).
            contact_count: Single-element int array with active contact count.
            sorted_point0: Sorted body-frame contact points on shape 0.
            sorted_shape0: Sorted shape 0 indices.
            sorted_normal: Sorted contact normals.
            body_q: Body transforms (current frame).
            shape_body: Shape-to-body index map.
            device: Device to launch on.
        """
        data = _SaveStateData()
        data.src_keys = sorted_keys
        data.src_point0 = sorted_point0
        data.src_shape0 = sorted_shape0
        data.src_normal = sorted_normal
        data.src_count = contact_count
        data.body_q = body_q
        data.shape_body = shape_body
        data.dst_keys = self._prev_sorted_keys
        # Write world-space positions and normals into sorter's scratch buffers.
        data.dst_pos_world = self._sorter.scratch_pos_world
        data.dst_normal = self._sorter.scratch_normal
        data.dst_count = self._prev_count

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

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
