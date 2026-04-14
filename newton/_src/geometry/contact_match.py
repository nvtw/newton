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
"""

from __future__ import annotations

import warp as wp

from ..core.types import Devicelike

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

    # Previous frame (sorted)
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

        if dist_sq < best_dist_sq and ndot > data.normal_dot_threshold:
            best_dist_sq = dist_sq
            best_idx = old_idx

        k += 1

    if best_idx >= 0:
        data.match_index[tid] = wp.int32(best_idx)
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

    Pre-allocates all buffers at construction time for CUDA graph capture
    compatibility.  The typical per-frame call sequence is::

        matcher.match(...)  # before ContactSorter.sort_full()
        sorter.sort_full(...)  # match_index is permuted with contacts
        matcher.save_sorted_state(...)  # after sorting
        matcher.build_report(...)  # optional

    Args:
        capacity: Maximum number of contacts (must match :class:`ContactSorter`).
        pos_threshold: World-space distance threshold [m].  Contacts whose
            positions moved more than this between frames are considered broken.
        normal_dot_threshold: Minimum dot product between old and new contact
            normals.  Below this the contact is considered broken.
        contact_report: Allocate buffers for new/broken contact index lists.
        device: Device to allocate on.
    """

    def __init__(
        self,
        capacity: int,
        *,
        pos_threshold: float = 0.02,
        normal_dot_threshold: float = 0.9,
        contact_report: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            self._pos_threshold_sq = pos_threshold * pos_threshold
            self._normal_dot_threshold = normal_dot_threshold

            # Previous frame sorted state.
            self._prev_sorted_keys = wp.zeros(capacity, dtype=wp.int64)
            self._prev_pos_world = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_normal = wp.zeros(capacity, dtype=wp.vec3)
            self._prev_count = wp.zeros(1, dtype=wp.int32)

            # Per-old-contact "was matched" flags (zeroed each frame).
            self._prev_was_matched = wp.zeros(capacity, dtype=wp.int32)

            # Contact report (optional).
            self._has_report = contact_report
            if contact_report:
                self._new_contact_indices = wp.zeros(capacity, dtype=wp.int32)
                self._new_contact_count = wp.zeros(1, dtype=wp.int32)
                self._broken_contact_indices = wp.zeros(capacity, dtype=wp.int32)
                self._broken_contact_count = wp.zeros(1, dtype=wp.int32)
            else:
                self._new_contact_indices = wp.zeros(0, dtype=wp.int32)
                self._new_contact_count = wp.zeros(0, dtype=wp.int32)
                self._broken_contact_indices = wp.zeros(0, dtype=wp.int32)
                self._broken_contact_count = wp.zeros(0, dtype=wp.int32)

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

    @property
    def new_contact_indices(self) -> wp.array:
        """Indices of new contacts in the current sorted buffer.

        Only valid after :meth:`build_report`.
        """
        return self._new_contact_indices

    @property
    def new_contact_count(self) -> wp.array:
        """Device-side count of new contacts (single-element int32).

        Only valid after :meth:`build_report`.
        """
        return self._new_contact_count

    @property
    def broken_contact_indices(self) -> wp.array:
        """Indices of broken contacts in the previous frame's sorted buffer.

        Only valid after :meth:`build_report`.
        """
        return self._broken_contact_indices

    @property
    def broken_contact_count(self) -> wp.array:
        """Device-side count of broken contacts (single-element int32).

        Only valid after :meth:`build_report`.
        """
        return self._broken_contact_count

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
        self._prev_was_matched.zero_()

        data = _MatchData()
        data.prev_keys = self._prev_sorted_keys
        data.prev_pos_world = self._prev_pos_world
        data.prev_normal = self._prev_normal
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
        data.dst_pos_world = self._prev_pos_world
        data.dst_normal = self._prev_normal
        data.dst_count = self._prev_count

        wp.launch(_save_sorted_state_kernel, dim=self._capacity, inputs=[data], device=device)

    def build_report(
        self,
        match_index: wp.array,
        contact_count: wp.array,
        *,
        device: Devicelike = None,
    ) -> None:
        """Build new/broken contact index lists (optional, post-sort).

        After this call, :attr:`new_contact_indices` / :attr:`new_contact_count`
        hold indices of contacts in the current sorted buffer that have no
        prior match (``match_index < 0``), and :attr:`broken_contact_indices` /
        :attr:`broken_contact_count` hold indices of old contacts that were not
        matched by any new contact.

        Args:
            match_index: Sorted match_index array (from :class:`Contacts`).
            contact_count: Single-element int array with active contact count.
            device: Device to launch on.
        """
        if not self._has_report:
            return

        self._new_contact_count.zero_()
        self._broken_contact_count.zero_()

        wp.launch(
            _collect_new_contacts_kernel,
            dim=self._capacity,
            inputs=[match_index, contact_count, self._new_contact_indices, self._new_contact_count],
            device=device,
        )
        wp.launch(
            _collect_broken_contacts_kernel,
            dim=self._capacity,
            inputs=[self._prev_was_matched, self._prev_count, self._broken_contact_indices, self._broken_contact_count],
            device=device,
        )
