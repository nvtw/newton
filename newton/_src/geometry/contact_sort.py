# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contact sorting via radix sort on per-contact keys.

Provides the machinery to reorder contact arrays into a canonical,
deterministic order after the narrow-phase collision pipeline has
written contacts in GPU-scheduling-dependent order.

The sort always operates over the full pre-allocated buffer (for CUDA
graph capture compatibility).  Unused slots beyond ``contact_count``
are filled with ``0x7FFFFFFFFFFFFFFF`` so they sort to the end.
"""

from __future__ import annotations

import warp as wp

from ..core.types import Devicelike

# Sentinel key for unused contact slots.  ``radix_sort_pairs`` treats
# keys as signed int64, so ``0x7FFF…`` (max positive int64) sorts last.
SORT_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


@wp.kernel(enable_backward=False)
def _prepare_sort(
    contact_count: wp.array[int],
    sort_keys_src: wp.array[wp.int64],
    sort_keys_dst: wp.array[wp.int64],
    sort_indices: wp.array[wp.int32],
):
    """Copy active keys and init identity indices; fill unused slots with sentinel."""
    tid = wp.tid()
    if tid < contact_count[0]:
        sort_keys_dst[tid] = sort_keys_src[tid]
        sort_indices[tid] = wp.int32(tid)
    else:
        sort_keys_dst[tid] = SORT_KEY_SENTINEL
        sort_indices[tid] = wp.int32(tid)


@wp.kernel(enable_backward=False)
def _gather_int(src: wp.array[wp.int32], dst: wp.array[wp.int32], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]


@wp.kernel(enable_backward=False)
def _gather_float(src: wp.array[float], dst: wp.array[float], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]


@wp.kernel(enable_backward=False)
def _gather_vec3(src: wp.array[wp.vec3], dst: wp.array[wp.vec3], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]


@wp.kernel(enable_backward=False)
def _gather_vec2i(src: wp.array[wp.vec2i], dst: wp.array[wp.vec2i], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]


class ContactSorter:
    """Sort contact arrays into a deterministic canonical order.

    Pre-allocates double-buffer arrays and permutation indices at construction
    time so that the per-frame :meth:`sort_simple` / :meth:`sort_full` calls
    are allocation-free and fully CUDA-graph-capturable (no host synchronization).

    The radix sort always runs over the full *capacity* buffer.  Slots beyond
    the active ``contact_count`` are filled with a sentinel key
    (``0x7FFFFFFFFFFFFFFF``) so they sort to the end and the gather kernels
    skip them via the ``contact_count`` guard.
    """

    def __init__(self, capacity: int, *, per_contact_shape_properties: bool = False, device: Devicelike = None):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            # radix_sort_pairs uses the second half as scratch, so allocate 2x.
            self._sort_indices = wp.zeros(2 * capacity, dtype=wp.int32)
            self._sort_keys_copy = wp.zeros(2 * capacity, dtype=wp.int64)

            self._buf_int = wp.zeros(capacity, dtype=wp.int32)
            self._buf_float = wp.zeros(capacity, dtype=float)
            self._buf_vec3 = wp.zeros(capacity, dtype=wp.vec3)
            self._buf_vec2i = wp.zeros(capacity, dtype=wp.vec2i)

            self._has_shape_props = per_contact_shape_properties

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sort_simple(
        self,
        sort_keys: wp.array,
        contact_count: wp.array,
        *,
        contact_pair: wp.array,
        contact_position: wp.array,
        contact_normal: wp.array,
        contact_penetration: wp.array,
        contact_tangent: wp.array | None = None,
        device: Devicelike = None,
    ) -> None:
        """Sort contacts written through the simplified narrow-phase writer.

        Fully graph-capturable — no host synchronization.

        Args:
            sort_keys: Per-contact int64 sort keys (filled by the writer).
            contact_count: Single-element int array with the active contact count.
            contact_pair: vec2i shape pair array.
            contact_position: vec3 contact positions.
            contact_normal: vec3 contact normals.
            contact_penetration: float penetration depths.
            contact_tangent: Optional vec3 tangent array.
            device: Device to launch on.
        """
        self._sort_and_permute(sort_keys, contact_count, device=device)

        self._gather_array_vec3(contact_position, contact_count, device)
        self._gather_array_vec3(contact_normal, contact_count, device)
        self._gather_array_float(contact_penetration, contact_count, device)
        self._gather_array_vec2i(contact_pair, contact_count, device)
        if contact_tangent is not None and contact_tangent.shape[0] > 0:
            self._gather_array_vec3(contact_tangent, contact_count, device)

    def sort_full(
        self,
        sort_keys: wp.array,
        contact_count: wp.array,
        *,
        shape0: wp.array,
        shape1: wp.array,
        point0: wp.array,
        point1: wp.array,
        offset0: wp.array,
        offset1: wp.array,
        normal: wp.array,
        margin0: wp.array,
        margin1: wp.array,
        tids: wp.array,
        stiffness: wp.array | None = None,
        damping: wp.array | None = None,
        friction: wp.array | None = None,
        device: Devicelike = None,
    ) -> None:
        """Sort contacts written through the full collide.py writer.

        Fully graph-capturable — no host synchronization.

        Args:
            sort_keys: Per-contact int64 sort keys (filled by the writer).
            contact_count: Single-element int array with the active contact count.
            shape0: int array of first shape indices.
            shape1: int array of second shape indices.
            point0: vec3 body-frame contact points on shape 0.
            point1: vec3 body-frame contact points on shape 1.
            offset0: vec3 body-frame friction anchor offsets for shape 0.
            offset1: vec3 body-frame friction anchor offsets for shape 1.
            normal: vec3 contact normals.
            margin0: float surface thickness for shape 0.
            margin1: float surface thickness for shape 1.
            tids: int tid array.
            stiffness: Optional float per-contact stiffness.
            damping: Optional float per-contact damping.
            friction: Optional float per-contact friction.
            device: Device to launch on.
        """
        self._sort_and_permute(sort_keys, contact_count, device=device)

        self._gather_array_int(shape0, contact_count, device)
        self._gather_array_int(shape1, contact_count, device)
        self._gather_array_vec3(point0, contact_count, device)
        self._gather_array_vec3(point1, contact_count, device)
        self._gather_array_vec3(offset0, contact_count, device)
        self._gather_array_vec3(offset1, contact_count, device)
        self._gather_array_vec3(normal, contact_count, device)
        self._gather_array_float(margin0, contact_count, device)
        self._gather_array_float(margin1, contact_count, device)
        self._gather_array_int(tids, contact_count, device)
        if self._has_shape_props:
            if stiffness is not None and stiffness.shape[0] > 0:
                self._gather_array_float(stiffness, contact_count, device)
            if damping is not None and damping.shape[0] > 0:
                self._gather_array_float(damping, contact_count, device)
            if friction is not None and friction.shape[0] > 0:
                self._gather_array_float(friction, contact_count, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_and_permute(self, sort_keys: wp.array, contact_count: wp.array, *, device: Devicelike = None) -> None:
        """Prepare keys (sentinel-fill unused slots), then radix-sort over the full buffer."""
        n = self._capacity
        wp.launch(
            _prepare_sort,
            dim=n,
            inputs=[contact_count, sort_keys, self._sort_keys_copy, self._sort_indices],
            device=device,
        )
        wp.utils.radix_sort_pairs(self._sort_keys_copy, self._sort_indices, n)

    def _gather_array_int(self, arr: wp.array, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_int, arr, count=self._capacity)
        wp.launch(
            _gather_int, dim=self._capacity, inputs=[self._buf_int, arr, self._sort_indices, count], device=device
        )

    def _gather_array_float(self, arr: wp.array, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_float, arr, count=self._capacity)
        wp.launch(
            _gather_float, dim=self._capacity, inputs=[self._buf_float, arr, self._sort_indices, count], device=device
        )

    def _gather_array_vec3(self, arr: wp.array, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_vec3, arr, count=self._capacity)
        wp.launch(
            _gather_vec3, dim=self._capacity, inputs=[self._buf_vec3, arr, self._sort_indices, count], device=device
        )

    def _gather_array_vec2i(self, arr: wp.array, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_vec2i, arr, count=self._capacity)
        wp.launch(
            _gather_vec2i, dim=self._capacity, inputs=[self._buf_vec2i, arr, self._sort_indices, count], device=device
        )
