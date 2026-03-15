# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contact warm-starting via sorted pair keys and binary search.

Provides:

* :func:`make_pair_key` -- symmetric ``int64`` key from two shape indices.
* :func:`binary_search_int64` -- GPU binary search in a sorted key array.
* :class:`WarmStarter` -- double-buffered cross-frame impulse transfer.

Pair keys are stored as ``int64`` (bit 63 is never set) so that Warp's
:func:`wp.utils.radix_sort_pairs` can sort them directly.
"""

from __future__ import annotations

import warp as wp

# Sentinel that sorts after any valid pair key (bit 63 clear).
_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


@wp.func
def make_pair_key(shape_a: wp.int32, shape_b: wp.int32) -> wp.int64:
    """Symmetric ``int64`` key encoding a pair of shape indices.

    ``make_pair_key(a, b) == make_pair_key(b, a)`` for all *a*, *b*.
    Bit 63 is always clear so that the key is a valid positive ``int64``.
    """
    lo = wp.min(shape_a, shape_b)
    hi = wp.max(shape_a, shape_b)
    return (wp.int64(lo) << wp.int64(32)) | wp.int64(hi)


@wp.func
def binary_search_int64(
    keys: wp.array(dtype=wp.int64),
    count: wp.int32,
    target: wp.int64,
) -> wp.int32:
    """Return the index of *target* in the sorted *keys* array, or ``-1``."""
    left = int(0)
    right = count - 1
    while left <= right:
        mid = (left + right) >> 1
        val = keys[mid]
        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_pair_keys_kernel(
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    pair_keys: wp.array(dtype=wp.int64),
    contact_count: wp.array(dtype=wp.int32),
):
    """Compute symmetric pair keys for each contact."""
    tid = wp.tid()
    if tid < contact_count[0]:
        pair_keys[tid] = make_pair_key(shape0[tid], shape1[tid])


@wp.kernel
def _init_indices_kernel(
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Initialise sort-value indices to ``tid`` for active, ``capacity`` for inactive."""
    tid = wp.tid()
    if tid < count[0]:
        indices[tid] = tid
    else:
        indices[tid] = capacity


@wp.kernel
def _pad_keys_kernel(
    keys: wp.array(dtype=wp.int64),
    count: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Pad unused key slots with sentinel so they sort to the end."""
    tid = wp.tid()
    if tid >= count[0] and tid < capacity:
        keys[tid] = _KEY_SENTINEL


@wp.kernel
def _transfer_impulses_kernel(
    curr_keys: wp.array(dtype=wp.int64),
    prev_keys: wp.array(dtype=wp.int64),
    prev_impulse_n: wp.array(dtype=wp.float32),
    prev_impulse_t1: wp.array(dtype=wp.float32),
    prev_impulse_t2: wp.array(dtype=wp.float32),
    prev_count: wp.array(dtype=wp.int32),
    out_impulse_n: wp.array(dtype=wp.float32),
    out_impulse_t1: wp.array(dtype=wp.float32),
    out_impulse_t2: wp.array(dtype=wp.float32),
    curr_count: wp.array(dtype=wp.int32),
):
    """For each current contact, binary-search the previous sorted keys and copy impulses.

    TODO: Newton's narrow phase can produce multiple contacts per shape pair
    (up to 4 for box-plane corners).  All contacts in a pair share the same
    pair key, so binary_search finds one previous match and copies its
    impulse to every current contact in that pair.  This is a reasonable
    approximation (contacts within a pair typically have similar normal
    impulse magnitudes) but is not exact.  A future improvement is to
    assign a monotonic sub-index within each pair group and encode it in
    the lower bits of the int64 key so that contacts can be matched 1:1.
    """
    tid = wp.tid()
    if tid >= curr_count[0]:
        return
    key = curr_keys[tid]
    idx = binary_search_int64(prev_keys, prev_count[0], key)
    if idx >= 0:
        out_impulse_n[tid] = prev_impulse_n[idx]
        out_impulse_t1[tid] = prev_impulse_t1[idx]
        out_impulse_t2[tid] = prev_impulse_t2[idx]
    else:
        out_impulse_n[tid] = 0.0
        out_impulse_t1[tid] = 0.0
        out_impulse_t2[tid] = 0.0


@wp.kernel
def _reorder_impulses_kernel(
    src_n: wp.array(dtype=wp.float32),
    src_t1: wp.array(dtype=wp.float32),
    src_t2: wp.array(dtype=wp.float32),
    perm: wp.array(dtype=wp.int32),
    dst_n: wp.array(dtype=wp.float32),
    dst_t1: wp.array(dtype=wp.float32),
    dst_t2: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
):
    """Reorder impulse arrays according to *perm* (produced by radix sort on keys)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    old = perm[tid]
    dst_n[tid] = src_n[old]
    dst_t1[tid] = src_t1[old]
    dst_t2[tid] = src_t2[old]


# ---------------------------------------------------------------------------
# WarmStarter
# ---------------------------------------------------------------------------


class WarmStarter:
    """Double-buffered cross-frame impulse transfer via sorted pair keys.

    At the start of each frame, call :meth:`begin_frame` to swap buffers,
    then :meth:`import_keys` + :meth:`sort` to prepare the current
    frame's contact keys, and :meth:`transfer_impulses` to seed
    accumulated impulses from the previous frame.  After the solver has
    converged, call :meth:`export_impulses` to snapshot the solved
    impulses for the next frame.

    Args:
        capacity: maximum number of contacts per frame.
        device: Warp device string or object.
    """

    def __init__(self, capacity: int, device: wp.context.Device | str | None = None):
        self.capacity = capacity
        self.device = wp.get_device(device)
        d = self.device

        # Double buffers: "curr" is the frame being built, "prev" is last frame.
        # Keys are 2*capacity because radix_sort_pairs needs scratch space.
        self.curr_keys = wp.zeros(2 * capacity, dtype=wp.int64, device=d)
        self.curr_impulse_n = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.curr_impulse_t1 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.curr_impulse_t2 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.curr_count = wp.zeros(1, dtype=wp.int32, device=d)
        self.curr_indices = wp.zeros(2 * capacity, dtype=wp.int32, device=d)

        self.prev_keys = wp.zeros(2 * capacity, dtype=wp.int64, device=d)
        self.prev_impulse_n = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_impulse_t1 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_impulse_t2 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_count = wp.zeros(1, dtype=wp.int32, device=d)

        # Temp buffer for reordering impulses after sort
        self._tmp_n = wp.zeros(capacity, dtype=wp.float32, device=d)
        self._tmp_t1 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self._tmp_t2 = wp.zeros(capacity, dtype=wp.float32, device=d)

    # -- frame lifecycle ----------------------------------------------------

    def begin_frame(self):
        """Swap current and previous buffers (pointer swap, O(1))."""
        (
            self.curr_keys, self.prev_keys,
            self.curr_impulse_n, self.prev_impulse_n,
            self.curr_impulse_t1, self.prev_impulse_t1,
            self.curr_impulse_t2, self.prev_impulse_t2,
            self.curr_count, self.prev_count,
        ) = (
            self.prev_keys, self.curr_keys,
            self.prev_impulse_n, self.curr_impulse_n,
            self.prev_impulse_t1, self.curr_impulse_t1,
            self.prev_impulse_t2, self.curr_impulse_t2,
            self.prev_count, self.curr_count,
        )

    def import_keys(
        self,
        shape0: wp.array,
        shape1: wp.array,
        contact_count: wp.array,
    ):
        """Compute pair keys from shape index arrays for the current frame.

        Args:
            shape0: ``int32`` array of shape-0 indices.
            shape1: ``int32`` array of shape-1 indices.
            contact_count: ``int32[1]`` device array with the active count.
        """
        d = self.device
        cap = self.capacity

        wp.copy(self.curr_count, contact_count)

        wp.launch(
            _compute_pair_keys_kernel,
            dim=cap,
            inputs=[shape0, shape1, self.curr_keys, self.curr_count],
            device=d,
        )

    def sort(self):
        """Radix-sort current keys (and track the permutation)."""
        d = self.device
        cap = self.capacity

        wp.launch(
            _init_indices_kernel,
            dim=cap,
            inputs=[self.curr_indices, self.curr_count, cap],
            device=d,
        )
        wp.launch(
            _pad_keys_kernel,
            dim=cap,
            inputs=[self.curr_keys, self.curr_count, cap],
            device=d,
        )

        wp.utils.radix_sort_pairs(self.curr_keys, self.curr_indices, cap)

    def transfer_impulses(
        self,
        out_impulse_n: wp.array,
        out_impulse_t1: wp.array,
        out_impulse_t2: wp.array,
    ):
        """Seed current-frame impulse columns from the previous frame.

        For each current contact whose pair key exists in the previous
        sorted keys, the corresponding accumulated impulse is copied.
        New pairs get zero.

        Args:
            out_impulse_n: target ``float32`` array for normal impulse.
            out_impulse_t1: target ``float32`` array for tangent impulse 1.
            out_impulse_t2: target ``float32`` array for tangent impulse 2.
        """
        d = self.device
        cap = self.capacity

        wp.launch(
            _transfer_impulses_kernel,
            dim=cap,
            inputs=[
                self.curr_keys,
                self.prev_keys,
                self.prev_impulse_n,
                self.prev_impulse_t1,
                self.prev_impulse_t2,
                self.prev_count,
                out_impulse_n,
                out_impulse_t1,
                out_impulse_t2,
                self.curr_count,
            ],
            device=d,
        )

    def export_impulses(
        self,
        src_impulse_n: wp.array,
        src_impulse_t1: wp.array,
        src_impulse_t2: wp.array,
    ):
        """Snapshot solved impulses so the next frame can warm-start.

        Must be called *after* the solver has finished and *before*
        :meth:`begin_frame` of the next frame.

        The impulses are reordered to match the sorted key order so
        that :meth:`transfer_impulses` can binary-search by key.

        Args:
            src_impulse_n: solved normal impulse column.
            src_impulse_t1: solved tangent impulse 1 column.
            src_impulse_t2: solved tangent impulse 2 column.
        """
        d = self.device
        cap = self.capacity

        # The sort permutation (curr_indices) maps sorted position -> original
        # position.  We reorder the impulses into sorted order.
        wp.launch(
            _reorder_impulses_kernel,
            dim=cap,
            inputs=[
                src_impulse_n, src_impulse_t1, src_impulse_t2,
                self.curr_indices,
                self._tmp_n, self._tmp_t1, self._tmp_t2,
                self.curr_count,
            ],
            device=d,
        )
        wp.copy(self.curr_impulse_n, self._tmp_n, count=cap)
        wp.copy(self.curr_impulse_t1, self._tmp_t1, count=cap)
        wp.copy(self.curr_impulse_t2, self._tmp_t2, count=cap)
