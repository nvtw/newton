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
* :func:`binary_search_lower_bound` -- GPU lower-bound search in a sorted key array.
* :class:`WarmStarter` -- double-buffered cross-frame impulse transfer.

Pair keys are stored as ``int64`` (bit 63 is never set) so that Warp's
:func:`wp.utils.radix_sort_pairs` can sort them directly.
"""

from __future__ import annotations

import warp as wp

# Sentinel that sorts after any valid pair key (bit 63 clear).
_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))

# Maximum contacts per bundle for primitive shape pairs (matches C# PhoenX).
MAX_BUNDLE_CONTACTS = 5

# Maximum contacts per bundle for mesh shape pairs.  Mesh contacts
# use voxel-binned keys for warm-start matching, but the PGS solver
# needs all contacts from the same shape pair processed sequentially
# in one bundle so that impulses couple properly across the contact
# manifold.  240 matches Newton's maximum reduced contacts per pair.
MAX_BUNDLE_CONTACTS_MESH = 240


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
def make_pair_key_voxel(
    shape_a: wp.int32,
    shape_b: wp.int32,
    offset0: wp.vec3,
    aabb_lower: wp.vec3,
    aabb_upper: wp.vec3,
    voxel_res: wp.vec3i,
) -> wp.int64:
    """Symmetric pair key with a spatial voxel bin in the lower 8 bits.

    Encodes ``(min_shape << 32) | (max_shape << 8) | voxel_bin``.
    The voxel bin is computed from the body-local contact offset within the
    shape's local AABB, modulo 256 to fit in 8 bits.
    """
    lo = wp.min(shape_a, shape_b)
    hi = wp.max(shape_a, shape_b)
    # Compute voxel bin from offset0 position in AABB
    extent = aabb_upper - aabb_lower
    cell_x = extent[0] / float(wp.max(voxel_res[0], 1))
    cell_y = extent[1] / float(wp.max(voxel_res[1], 1))
    cell_z = extent[2] / float(wp.max(voxel_res[2], 1))
    ix = wp.clamp(wp.int32((offset0[0] - aabb_lower[0]) / wp.max(cell_x, 1.0e-10)), 0, voxel_res[0] - 1)
    iy = wp.clamp(wp.int32((offset0[1] - aabb_lower[1]) / wp.max(cell_y, 1.0e-10)), 0, voxel_res[1] - 1)
    iz = wp.clamp(wp.int32((offset0[2] - aabb_lower[2]) / wp.max(cell_z, 1.0e-10)), 0, voxel_res[2] - 1)
    linear = ix * voxel_res[1] * voxel_res[2] + iy * voxel_res[2] + iz
    voxel_bin = linear % 256
    return (wp.int64(lo) << wp.int64(32)) | (wp.int64(hi) << wp.int64(8)) | wp.int64(voxel_bin)


@wp.func
def pair_key_without_voxel(key: wp.int64) -> wp.int64:
    """Strip the lower 8 voxel-bin bits from a voxel-extended pair key.

    Voxel keys: ``(min_shape << 32) | (max_shape << 8) | voxel_bin``.
    Plain keys: ``(min_shape << 32) | max_shape``.
    For plain keys the lower 8 bits are part of max_shape, but since
    we only use this to compare *within a sorted run*, it is safe to
    mask them — contacts from the same pair are contiguous after sort.
    """
    return key >> wp.int64(8)


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


@wp.func
def binary_search_lower_bound(
    keys: wp.array(dtype=wp.int64),
    count: wp.int32,
    target: wp.int64,
) -> wp.int32:
    """Return the index of the first element >= *target*, or *count* if none."""
    left = int(0)
    right = int(count)
    while left < right:
        mid = (left + right) >> 1
        if keys[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


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
def _compute_pair_keys_voxel_kernel(
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    offset0: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=wp.int32),
    shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
    shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
    shape_voxel_resolution: wp.array(dtype=wp.vec3i),
    pair_keys: wp.array(dtype=wp.int64),
    contact_count: wp.array(dtype=wp.int32),
):
    """Compute pair keys with voxel binning for mesh shape pairs."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    s0 = shape0[tid]
    s1 = shape1[tid]
    # Use voxel-binned key if either shape is a mesh (type 8)
    if shape_type[s0] == 8 or shape_type[s1] == 8:
        # Use shape with lower index for AABB lookup (the "min" shape)
        ref = wp.min(s0, s1)
        pair_keys[tid] = make_pair_key_voxel(
            s0,
            s1,
            offset0[tid],
            shape_collision_aabb_lower[ref],
            shape_collision_aabb_upper[ref],
            shape_voxel_resolution[ref],
        )
    else:
        pair_keys[tid] = make_pair_key(s0, s1)


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
def _break_uniform_keys_kernel(
    keys: wp.array(dtype=wp.int64),
    count: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Set the last key to 0 if all keys are sentinel (avoids radix sort hang)."""
    if count[0] == 0 and capacity > 1:
        keys[capacity - 1] = wp.int64(0)


@wp.kernel
def _copy_offset0_kernel(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
):
    """Copy offset0 values for active contacts."""
    tid = wp.tid()
    if tid < count[0]:
        dst[tid] = src[tid]


@wp.kernel
def _transfer_impulses_kernel(
    curr_keys: wp.array(dtype=wp.int64),
    curr_offset0: wp.array(dtype=wp.vec3),
    prev_keys: wp.array(dtype=wp.int64),
    prev_offset0: wp.array(dtype=wp.vec3),
    prev_impulse_n: wp.array(dtype=wp.float32),
    prev_impulse_t1: wp.array(dtype=wp.float32),
    prev_impulse_t2: wp.array(dtype=wp.float32),
    prev_count: wp.array(dtype=wp.int32),
    out_impulse_n: wp.array(dtype=wp.float32),
    out_impulse_t1: wp.array(dtype=wp.float32),
    out_impulse_t2: wp.array(dtype=wp.float32),
    curr_count: wp.array(dtype=wp.int32),
):
    """Per-point warm start: match current contacts to previous by pair key + nearest offset0."""
    tid = wp.tid()
    if tid >= curr_count[0]:
        return
    key = curr_keys[tid]
    # Lower-bound search to find first occurrence of key in prev_keys
    idx = binary_search_lower_bound(prev_keys, prev_count[0], key)
    if idx >= prev_count[0] or prev_keys[idx] != key:
        out_impulse_n[tid] = 0.0
        out_impulse_t1[tid] = 0.0
        out_impulse_t2[tid] = 0.0
        return
    # Scan the run of contacts with same key, find the one with closest offset0
    my_off = curr_offset0[tid]
    best_idx = int(idx)
    best_dist = float(1.0e30)
    s = int(idx)
    while s < prev_count[0]:
        if prev_keys[s] != key:
            break
        d = wp.length(prev_offset0[s] - my_off)
        if d < best_dist:
            best_dist = float(d)
            best_idx = int(s)
        s = s + 1
    out_impulse_n[tid] = prev_impulse_n[best_idx]
    out_impulse_t1[tid] = prev_impulse_t1[best_idx]
    out_impulse_t2[tid] = prev_impulse_t2[best_idx]


@wp.kernel
def _mark_bundle_heads_kernel(
    keys: wp.array(dtype=wp.int64),
    marks: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    capacity: int,
    max_bundle: int,
):
    """Mark each contact that starts a new bundle (1) or not (0).

    A contact is a bundle head when it is the first contact, its key
    differs from the previous contact's, or its offset within a
    same-key run is a multiple of *max_bundle*.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        if tid < capacity:
            marks[tid] = 0
        return
    if tid == 0:
        marks[tid] = 1
        return
    if keys[tid] != keys[tid - 1]:
        marks[tid] = 1
        return
    run_start = tid
    s = tid - 1
    while s >= 0 and keys[s] == keys[tid]:
        run_start = s
        s = s - 1
    offset_in_run = tid - run_start
    if offset_in_run % max_bundle == 0:
        marks[tid] = 1
    else:
        marks[tid] = 0


@wp.kernel
def _mark_bundle_heads_mesh_kernel(
    keys: wp.array(dtype=wp.int64),
    marks: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    capacity: int,
    max_bundle_mesh: int,
):
    """Bundle-head marking that groups mesh contacts by shape pair.

    Same as :func:`_mark_bundle_heads_kernel` but compares keys with
    voxel bits stripped so that all contacts from the same shape pair
    (regardless of voxel bin) land in one bundle, up to
    *max_bundle_mesh* contacts.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        if tid < capacity:
            marks[tid] = 0
        return
    if tid == 0:
        marks[tid] = 1
        return
    pair_curr = pair_key_without_voxel(keys[tid])
    pair_prev = pair_key_without_voxel(keys[tid - 1])
    if pair_curr != pair_prev:
        marks[tid] = 1
        return
    # Same pair — check offset within pair run
    run_start = tid
    s = tid - 1
    while s >= 0 and pair_key_without_voxel(keys[s]) == pair_curr:
        run_start = s
        s = s - 1
    offset_in_run = tid - run_start
    if offset_in_run % max_bundle_mesh == 0:
        marks[tid] = 1
    else:
        marks[tid] = 0


@wp.kernel
def _scatter_bundle_starts_kernel(
    prefix: wp.array(dtype=wp.int32),
    bundle_starts: wp.array(dtype=wp.int32),
    bundle_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
):
    """Scatter bundle start positions and write the sentinel + count."""
    tid = wp.tid()
    count = contact_count[0]
    if tid >= count:
        return
    idx = prefix[tid]
    is_head = False
    if tid == 0:
        is_head = idx == 1
    else:
        is_head = idx != prefix[tid - 1]
    if is_head:
        bundle_starts[idx - 1] = tid
    if tid == count - 1:
        bundle_count[0] = idx
        bundle_starts[idx] = count


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


@wp.kernel
def _reorder_offset0_kernel(
    src: wp.array(dtype=wp.vec3),
    perm: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
):
    """Reorder offset0 array according to *perm* (produced by radix sort on keys)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    dst[tid] = src[perm[tid]]


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
        self.curr_offset0 = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.curr_count = wp.zeros(1, dtype=wp.int32, device=d)
        self.curr_indices = wp.zeros(2 * capacity, dtype=wp.int32, device=d)

        self.prev_keys = wp.zeros(2 * capacity, dtype=wp.int64, device=d)
        self.prev_impulse_n = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_impulse_t1 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_impulse_t2 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self.prev_offset0 = wp.zeros(capacity, dtype=wp.vec3, device=d)
        self.prev_count = wp.zeros(1, dtype=wp.int32, device=d)

        # Temp buffers for reordering after sort
        self._tmp_n = wp.zeros(capacity, dtype=wp.float32, device=d)
        self._tmp_t1 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self._tmp_t2 = wp.zeros(capacity, dtype=wp.float32, device=d)
        self._tmp_offset0 = wp.zeros(capacity, dtype=wp.vec3, device=d)

        # Bundle metadata (built by build_bundles on GPU)
        # bundle_starts has capacity+1 entries to hold the sentinel end index.
        self.bundle_starts = wp.zeros(capacity + 1, dtype=wp.int32, device=d)
        self.bundle_count = wp.zeros(1, dtype=wp.int32, device=d)
        self._bundle_marks = wp.zeros(capacity, dtype=wp.int32, device=d)
        self._bundle_prefix = wp.zeros(capacity, dtype=wp.int32, device=d)

    # -- frame lifecycle ----------------------------------------------------

    def begin_frame(self):
        """Swap current and previous buffers (pointer swap, O(1))."""
        (
            self.curr_keys,
            self.prev_keys,
            self.curr_impulse_n,
            self.prev_impulse_n,
            self.curr_impulse_t1,
            self.prev_impulse_t1,
            self.curr_impulse_t2,
            self.prev_impulse_t2,
            self.curr_offset0,
            self.prev_offset0,
            self.curr_count,
            self.prev_count,
        ) = (
            self.prev_keys,
            self.curr_keys,
            self.prev_impulse_n,
            self.curr_impulse_n,
            self.prev_impulse_t1,
            self.curr_impulse_t1,
            self.prev_impulse_t2,
            self.curr_impulse_t2,
            self.prev_offset0,
            self.curr_offset0,
            self.prev_count,
            self.curr_count,
        )

    def set_mesh_data(
        self,
        shape_type: wp.array,
        shape_collision_aabb_lower: wp.array,
        shape_collision_aabb_upper: wp.array,
        shape_voxel_resolution: wp.array,
    ):
        """Provide mesh shape data for voxel-bucketed pair keys.

        Call once after :meth:`finalize` on the collision pipeline. When set,
        :meth:`import_keys` will use voxel-binned keys for mesh shape pairs.

        Args:
            shape_type: ``int32`` per-shape geometry type array.
            shape_collision_aabb_lower: ``vec3`` per-shape local AABB lower bounds.
            shape_collision_aabb_upper: ``vec3`` per-shape local AABB upper bounds.
            shape_voxel_resolution: ``vec3i`` per-shape voxel grid resolution.
        """
        self._shape_type = shape_type
        self._shape_collision_aabb_lower = shape_collision_aabb_lower
        self._shape_collision_aabb_upper = shape_collision_aabb_upper
        self._shape_voxel_resolution = shape_voxel_resolution

    def import_keys(
        self,
        shape0: wp.array,
        shape1: wp.array,
        contact_count: wp.array,
        offset0: wp.array | None = None,
    ):
        """Compute pair keys from shape index arrays for the current frame.

        Args:
            shape0: ``int32`` array of shape-0 indices.
            shape1: ``int32`` array of shape-1 indices.
            contact_count: ``int32[1]`` device array with the active count.
            offset0: ``vec3`` array of body-local contact offsets for per-point matching.
        """
        d = self.device
        cap = self.capacity

        wp.copy(self.curr_count, contact_count)

        if offset0 is not None and hasattr(self, "_shape_type") and self._shape_type is not None:
            # Use voxel-binned keys when mesh data is available
            wp.launch(
                _compute_pair_keys_voxel_kernel,
                dim=cap,
                inputs=[
                    shape0,
                    shape1,
                    offset0,
                    self._shape_type,
                    self._shape_collision_aabb_lower,
                    self._shape_collision_aabb_upper,
                    self._shape_voxel_resolution,
                    self.curr_keys,
                    self.curr_count,
                ],
                device=d,
            )
        else:
            wp.launch(
                _compute_pair_keys_kernel,
                dim=cap,
                inputs=[shape0, shape1, self.curr_keys, self.curr_count],
                device=d,
            )

        if offset0 is not None:
            wp.launch(
                _copy_offset0_kernel,
                dim=cap,
                inputs=[offset0, self.curr_offset0, self.curr_count],
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

    def build_bundles(self):
        """Split sorted contacts into bundles.

        Primitive contacts use bundles of up to ``MAX_BUNDLE_CONTACTS``
        (5).  When mesh data has been set via :meth:`set_mesh_data`,
        contacts are grouped by **shape pair** (ignoring voxel bins)
        with a limit of ``MAX_BUNDLE_CONTACTS_MESH`` (240) so that all
        mesh contacts on the same pair are solved sequentially in one
        bundle.

        Must be called after :meth:`sort`.  The entire operation runs on
        the GPU with no device-to-host sync, so it is safe inside a CUDA
        graph capture.
        """
        d = self.device
        cap = self.capacity

        self.bundle_count.zero_()

        if hasattr(self, "_shape_type") and self._shape_type is not None:
            # Mesh-aware: group by pair (strip voxel bits) so all
            # contacts from the same shape pair form one large bundle.
            # This puts each pair into a single graph-coloring element,
            # avoiding overflow and enabling true sequential GS within
            # the bundle — no mass splitting needed.
            wp.launch(
                _mark_bundle_heads_mesh_kernel,
                dim=cap,
                inputs=[
                    self.curr_keys,
                    self._bundle_marks,
                    self.curr_count,
                    cap,
                    MAX_BUNDLE_CONTACTS_MESH,
                ],
                device=d,
            )
        else:
            wp.launch(
                _mark_bundle_heads_kernel,
                dim=cap,
                inputs=[self.curr_keys, self._bundle_marks, self.curr_count, cap, MAX_BUNDLE_CONTACTS],
                device=d,
            )

        wp.utils.array_scan(self._bundle_marks, self._bundle_prefix, inclusive=True)

        wp.launch(
            _scatter_bundle_starts_kernel,
            dim=cap,
            inputs=[self._bundle_prefix, self.bundle_starts, self.bundle_count, self.curr_count],
            device=d,
        )

    def transfer_impulses(
        self,
        out_impulse_n: wp.array,
        out_impulse_t1: wp.array,
        out_impulse_t2: wp.array,
    ):
        """Seed current-frame impulse columns from the previous frame.

        For each current contact whose pair key exists in the previous
        sorted keys, the corresponding accumulated impulse is copied
        (matched by nearest body-local offset0 within the same pair).
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
                self.curr_offset0,
                self.prev_keys,
                self.prev_offset0,
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
        src_offset0: wp.array | None = None,
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
            src_offset0: body-local offset0 column for per-point matching.
        """
        d = self.device
        cap = self.capacity

        # The sort permutation (curr_indices) maps sorted position -> original
        # position.  We reorder the impulses into sorted order.
        wp.launch(
            _reorder_impulses_kernel,
            dim=cap,
            inputs=[
                src_impulse_n,
                src_impulse_t1,
                src_impulse_t2,
                self.curr_indices,
                self._tmp_n,
                self._tmp_t1,
                self._tmp_t2,
                self.curr_count,
            ],
            device=d,
        )
        wp.copy(self.curr_impulse_n, self._tmp_n, count=cap)
        wp.copy(self.curr_impulse_t1, self._tmp_t1, count=cap)
        wp.copy(self.curr_impulse_t2, self._tmp_t2, count=cap)

        if src_offset0 is not None:
            wp.launch(
                _reorder_offset0_kernel,
                dim=cap,
                inputs=[
                    src_offset0,
                    self.curr_indices,
                    self._tmp_offset0,
                    self.curr_count,
                ],
                device=d,
            )
            wp.copy(self.curr_offset0, self._tmp_offset0, count=cap)
