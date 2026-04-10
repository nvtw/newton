# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contact sorting via radix sort on per-contact keys.

Provides the machinery to reorder contact arrays into a canonical,
deterministic order after the narrow-phase collision pipeline has
written contacts in GPU-scheduling-dependent order.
"""

from __future__ import annotations

import warp as wp

from ..core.types import Devicelike


@wp.kernel(enable_backward=False)
def _init_identity_indices(indices: wp.array[wp.int32]):
    """Fill *indices* with the identity permutation [0, 1, …, N-1]."""
    i = wp.tid()
    indices[i] = wp.int32(i)


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
def _gather_int64(src: wp.array[wp.int64], dst: wp.array[wp.int64], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]


class ContactSorter:
    """Sort contact arrays into a deterministic canonical order.

    Pre-allocates double-buffer arrays and permutation indices at construction
    time so that the per-frame :meth:`sort` call is allocation-free.
    """

    def __init__(self, capacity: int, *, per_contact_shape_properties: bool = False, device: Devicelike = None):
        with wp.ScopedDevice(device):
            self._capacity = capacity
            self._sort_indices = wp.zeros(capacity, dtype=wp.int32)
            self._sort_keys_copy = wp.zeros(capacity, dtype=wp.int64)

            # Double buffers for gather (one per dtype used by contact arrays)
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
        n = min(int(contact_count.numpy()[0]), self._capacity)
        if n <= 0:
            return

        self._sort_and_permute(sort_keys, n, device=device)

        self._gather_array_vec3(contact_position, n, contact_count, device)
        self._gather_array_vec3(contact_normal, n, contact_count, device)
        self._gather_array_float(contact_penetration, n, contact_count, device)
        # contact_pair is vec2i — we gather each component via int views would be complex;
        # instead gather both ints of the pair via a dedicated approach.
        # Since vec2i has the same size as 2 ints, we treat each element pair separately.
        # Simpler: swap the array after gather using a vec3 buffer trick won't work.
        # Let's just use a dedicated kernel for vec2i.
        self._gather_array_vec2i(contact_pair, n, contact_count, device)
        if contact_tangent is not None and contact_tangent.shape[0] > 0:
            self._gather_array_vec3(contact_tangent, n, contact_count, device)

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
        n = min(int(contact_count.numpy()[0]), self._capacity)
        if n <= 0:
            return

        self._sort_and_permute(sort_keys, n, device=device)

        self._gather_array_int(shape0, n, contact_count, device)
        self._gather_array_int(shape1, n, contact_count, device)
        self._gather_array_vec3(point0, n, contact_count, device)
        self._gather_array_vec3(point1, n, contact_count, device)
        self._gather_array_vec3(offset0, n, contact_count, device)
        self._gather_array_vec3(offset1, n, contact_count, device)
        self._gather_array_vec3(normal, n, contact_count, device)
        self._gather_array_float(margin0, n, contact_count, device)
        self._gather_array_float(margin1, n, contact_count, device)
        self._gather_array_int(tids, n, contact_count, device)
        if self._has_shape_props:
            if stiffness is not None and stiffness.shape[0] > 0:
                self._gather_array_float(stiffness, n, contact_count, device)
            if damping is not None and damping.shape[0] > 0:
                self._gather_array_float(damping, n, contact_count, device)
            if friction is not None and friction.shape[0] > 0:
                self._gather_array_float(friction, n, contact_count, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_and_permute(self, sort_keys: wp.array, n: int, *, device: Devicelike = None) -> None:
        """Run radix sort on *sort_keys* and produce the permutation in ``_sort_indices``."""
        wp.launch(_init_identity_indices, dim=n, inputs=[self._sort_indices], device=device)
        wp.copy(self._sort_keys_copy, sort_keys, count=n)
        wp.utils.radix_sort_pairs(self._sort_keys_copy, self._sort_indices, n)

    def _gather_array_int(self, arr: wp.array, n: int, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_int, arr, count=n)
        wp.launch(_gather_int, dim=n, inputs=[self._buf_int, arr, self._sort_indices, count], device=device)

    def _gather_array_float(self, arr: wp.array, n: int, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_float, arr, count=n)
        wp.launch(_gather_float, dim=n, inputs=[self._buf_float, arr, self._sort_indices, count], device=device)

    def _gather_array_vec3(self, arr: wp.array, n: int, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_vec3, arr, count=n)
        wp.launch(_gather_vec3, dim=n, inputs=[self._buf_vec3, arr, self._sort_indices, count], device=device)

    def _gather_array_vec2i(self, arr: wp.array, n: int, count: wp.array, device: Devicelike) -> None:
        wp.copy(self._buf_vec2i, arr, count=n)
        wp.launch(_gather_vec2i, dim=n, inputs=[self._buf_vec2i, arr, self._sort_indices, count], device=device)


@wp.kernel(enable_backward=False)
def _gather_vec2i(src: wp.array[wp.vec2i], dst: wp.array[wp.vec2i], perm: wp.array[wp.int32], count: wp.array[int]):
    i = wp.tid()
    if i >= count[0]:
        return
    dst[i] = src[perm[i]]
