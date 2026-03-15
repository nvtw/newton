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

"""Column-major struct-of-arrays storage backed by a single Warp array.

Provides two layers:

* :class:`DataStore` -- bare column-major storage whose schema is derived
  from a ``@wp.struct`` type.  Individual fields can be accessed as typed
  ``wp.array`` views via :meth:`DataStore.column_of`.
* :class:`HandleStore` -- wraps a :class:`DataStore` and adds stable
  integer handles that survive compaction and reordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

_HANDLE_INVALID_BIT = 1 << 31

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DTYPE_NORMALIZE = {
    float: wp.float32,
    int: wp.int32,
    bool: wp.int32,
}


def _normalize_dtype(dtype):
    return _DTYPE_NORMALIZE.get(dtype, dtype)


@dataclass(frozen=True)
class FieldInfo:
    """Metadata for a single field inside a :class:`Schema`."""

    name: str
    dtype: type
    float_width: int
    col_offset: int


class Schema:
    """Column layout derived from a ``@wp.struct`` type.

    Iterates the struct's annotations in declaration order, computes each
    field's float-column offset, and stores the result so that
    :class:`DataStore` can build typed array views.

    Args:
        struct_type: a Warp struct type decorated with ``@wp.struct``.
    """

    def __init__(self, struct_type):
        self.struct_type = struct_type
        self.fields: dict[str, FieldInfo] = {}
        self.floats_per_row: int = 0

        annotations = struct_type.cls.__annotations__
        for name, raw_dtype in annotations.items():
            dtype = _normalize_dtype(raw_dtype)
            byte_size = wp.types.type_size_in_bytes(dtype)
            if byte_size % 4 != 0:
                raise ValueError(
                    f"Field '{name}' dtype {dtype} has size {byte_size} which is not a multiple of 4 bytes"
                )
            width = byte_size // 4
            self.fields[name] = FieldInfo(name, dtype, width, self.floats_per_row)
            self.floats_per_row += width

    def __repr__(self) -> str:
        lines = [f"Schema({self.struct_type.__name__}, floats_per_row={self.floats_per_row})"]
        for fi in self.fields.values():
            lines.append(f"  {fi.name}: {fi.dtype.__name__}  width={fi.float_width}  offset={fi.col_offset}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataStore (Tier 1)
# ---------------------------------------------------------------------------


class DataStore:
    """Column-major struct-of-arrays storage backed by a flat ``float32`` array.

    The schema is derived from a ``@wp.struct``.  Each field becomes a
    contiguous column of ``capacity`` elements in the backing store.
    Use :meth:`column_of` to obtain a typed ``wp.array`` view for any
    field.

    Args:
        struct_type: a ``@wp.struct`` type defining the per-row layout.
        capacity: maximum number of rows.
        device: Warp device string or object.
    """

    def __init__(
        self,
        struct_type,
        capacity: int,
        device: wp.context.Device | str | None = None,
    ):
        self.schema = Schema(struct_type)
        self.capacity = capacity
        self.device = wp.get_device(device)

        total_floats = capacity * self.schema.floats_per_row
        self.data = wp.zeros(total_floats, dtype=wp.float32, device=self.device)

        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)

    # -- field access -------------------------------------------------------

    def column_of(self, name: str) -> wp.array:
        """Return a typed ``wp.array`` view into the backing store for *name*.

        The returned array has shape ``(capacity,)`` and the dtype declared
        in the struct annotation.  It shares memory with :attr:`data` --
        writes through the view are visible in the backing store and vice
        versa.

        Args:
            name: field name as declared in the ``@wp.struct``.
        """
        field = self.schema.fields[name]
        byte_offset = field.col_offset * self.capacity * 4
        return wp.array(
            ptr=self.data.ptr + byte_offset,
            dtype=field.dtype,
            shape=(self.capacity,),
            device=self.device,
            copy=False,
        )

    def clear(self):
        """Zero the backing store and reset :attr:`count` to 0."""
        self.data.zero_()
        self.count.zero_()


# ---------------------------------------------------------------------------
# Compaction kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _build_active_keys_kernel(
    index_to_handle: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
    past_end: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Active rows get key 0, removed/unused rows get key 1."""
    tid = wp.tid()
    if tid < capacity:
        values[tid] = tid
        if tid < past_end[0]:
            h = index_to_handle[tid]
            if h >= 0:
                keys[tid] = 0
            else:
                keys[tid] = 1
        else:
            keys[tid] = 1


@wp.kernel
def _reorder_column_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    perm: wp.array(dtype=wp.int32),
    active_count: wp.array(dtype=wp.int32),
    capacity: int,
    col_floats: int,
):
    """Reorder *col_floats* contiguous columns using *perm*."""
    tid = wp.tid()
    if tid >= active_count[0]:
        return
    old_row = perm[tid]
    for c in range(col_floats):
        offset = c * capacity
        dst[offset + tid] = src[offset + old_row]


@wp.kernel
def _update_handle_map_kernel(
    perm: wp.array(dtype=wp.int32),
    index_to_handle: wp.array(dtype=wp.int32),
    handle_to_index: wp.array(dtype=wp.int32),
    active_count: wp.array(dtype=wp.int32),
):
    """After reorder: update handle_to_index and index_to_handle."""
    tid = wp.tid()
    if tid >= active_count[0]:
        return
    old_row = perm[tid]
    h = index_to_handle[old_row]
    if h >= 0:
        handle_to_index[h] = tid


@wp.kernel
def _rebuild_index_to_handle_kernel(
    handle_to_index: wp.array(dtype=wp.int32),
    index_to_handle: wp.array(dtype=wp.int32),
    handle_count: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Rebuild index_to_handle from handle_to_index after compaction."""
    tid = wp.tid()
    if tid < capacity:
        index_to_handle[tid] = -1
    if tid < handle_count[0]:
        idx = handle_to_index[tid]
        if idx >= 0:
            index_to_handle[idx] = tid


@wp.kernel
def _count_active_kernel(
    keys: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
    past_end: wp.array(dtype=wp.int32),
):
    """Count rows with key==0 (single-thread kernel, dim=1)."""
    n = past_end[0]
    total = int(0)
    for i in range(n):
        if keys[i] == 0:
            total = total + 1
    count_out[0] = total


# ---------------------------------------------------------------------------
# HandleStore (Tier 2)
# ---------------------------------------------------------------------------


class HandleStore:
    """Column-major storage with stable integer handles.

    Wraps a :class:`DataStore` and adds a handle indirection layer.
    Handles are non-negative integers that remain valid across
    :meth:`compact` calls.  Internally a free-list stack manages handle
    recycling.

    Args:
        struct_type: a ``@wp.struct`` type defining the per-row layout.
        capacity: maximum number of rows / handles.
        device: Warp device string or object.
    """

    def __init__(
        self,
        struct_type,
        capacity: int,
        device: wp.context.Device | str | None = None,
    ):
        self.store = DataStore(struct_type, capacity, device=device)
        self.capacity = capacity
        self.device = self.store.device

        self.handle_to_index = wp.full(capacity, -1, dtype=wp.int32, device=self.device)
        self.index_to_handle = wp.full(capacity, -1, dtype=wp.int32, device=self.device)

        # Free-list: initially all handle IDs are free (stack order 0..N-1)
        free_np = np.arange(capacity, dtype=np.int32)
        self.free_handles = wp.array(free_np, dtype=wp.int32, device=self.device)
        self.num_free = wp.array([capacity], dtype=wp.int32, device=self.device)

        # past_end tracks highest used storage index + 1 (may have holes)
        self.past_end = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Temp buffers for compaction (2x for radix_sort_pairs)
        self._sort_keys = wp.zeros(2 * capacity, dtype=wp.int32, device=self.device)
        self._sort_vals = wp.zeros(2 * capacity, dtype=wp.int32, device=self.device)
        self._reorder_buf = wp.zeros(
            capacity * self.store.schema.floats_per_row,
            dtype=wp.float32,
            device=self.device,
        )

    # -- convenience --------------------------------------------------------

    @property
    def schema(self) -> Schema:
        return self.store.schema

    @property
    def count(self) -> wp.array:
        return self.store.count

    def column_of(self, name: str) -> wp.array:
        """Shorthand for ``self.store.column_of(name)``."""
        return self.store.column_of(name)

    # -- allocation / removal (host-side) -----------------------------------

    def allocate(self) -> int:
        """Allocate one row and return its handle (host-side).

        Returns:
            A non-negative handle ID, or -1 if the store is full.
        """
        nf = self.num_free.numpy()
        if nf[0] <= 0:
            return -1

        nf[0] -= 1
        handle_id = int(self.free_handles.numpy()[nf[0]])
        row = int(self.past_end.numpy()[0])

        self.num_free.assign(wp.array(nf, dtype=wp.int32, device=self.device))

        h2i = self.handle_to_index
        i2h = self.index_to_handle

        # Write single-element updates
        h2i_np = h2i.numpy()
        i2h_np = i2h.numpy()
        h2i_np[handle_id] = row
        i2h_np[row] = handle_id
        h2i.assign(wp.array(h2i_np, dtype=wp.int32, device=self.device))
        i2h.assign(wp.array(i2h_np, dtype=wp.int32, device=self.device))

        pe = self.past_end.numpy()
        pe[0] = row + 1
        self.past_end.assign(wp.array(pe, dtype=wp.int32, device=self.device))

        cnt = self.store.count.numpy()
        cnt[0] += 1
        self.store.count.assign(wp.array(cnt, dtype=wp.int32, device=self.device))

        return handle_id

    def remove(self, handle_id: int):
        """Mark *handle_id* for removal (host-side).

        The storage row is not reclaimed until :meth:`compact` is called.
        """
        h2i_np = self.handle_to_index.numpy()
        row = h2i_np[handle_id]
        if row < 0:
            return

        h2i_np[handle_id] = -1
        self.handle_to_index.assign(wp.array(h2i_np, dtype=wp.int32, device=self.device))

        i2h_np = self.index_to_handle.numpy()
        i2h_np[row] = -1
        self.index_to_handle.assign(wp.array(i2h_np, dtype=wp.int32, device=self.device))

        # Push handle back onto free-list
        nf = self.num_free.numpy()
        fl = self.free_handles.numpy()
        fl[nf[0]] = handle_id
        nf[0] += 1
        self.free_handles.assign(wp.array(fl, dtype=wp.int32, device=self.device))
        self.num_free.assign(wp.array(nf, dtype=wp.int32, device=self.device))

        cnt = self.store.count.numpy()
        cnt[0] -= 1
        self.store.count.assign(wp.array(cnt, dtype=wp.int32, device=self.device))

    # -- compaction ---------------------------------------------------------

    def compact(self):
        """Close gaps left by :meth:`remove`, keeping handles valid.

        After compaction, active rows are contiguous in ``[0, count)``.
        """
        d = self.device
        cap = self.capacity
        schema = self.schema

        # 1. Build sort keys (0 = active, 1 = removed/unused)
        wp.launch(
            _build_active_keys_kernel,
            dim=cap,
            inputs=[self.index_to_handle, self._sort_keys, self._sort_vals, self.past_end, cap],
            device=d,
        )

        # 2. Count active rows before sorting (needed to set count)
        wp.launch(
            _count_active_kernel,
            dim=1,
            inputs=[self._sort_keys, self.store.count, self.past_end],
            device=d,
        )

        # 3. Stable sort: active rows (key=0) come first, preserving order
        wp.utils.radix_sort_pairs(self._sort_keys, self._sort_vals, cap)

        # 4. Reorder data columns using the permutation in _sort_vals
        #    Process all float columns as one contiguous block
        fpr = schema.floats_per_row
        src_view = wp.array(
            ptr=self.store.data.ptr,
            dtype=wp.float32,
            shape=(fpr * cap,),
            device=d,
            copy=False,
        )
        dst_view = wp.array(
            ptr=self._reorder_buf.ptr,
            dtype=wp.float32,
            shape=(fpr * cap,),
            device=d,
            copy=False,
        )
        wp.launch(
            _reorder_column_kernel,
            dim=cap,
            inputs=[src_view, dst_view, self._sort_vals, self.store.count, cap, fpr],
            device=d,
        )
        wp.copy(self.store.data, self._reorder_buf, count=fpr * cap)

        # 5. Update handle maps
        wp.launch(
            _update_handle_map_kernel,
            dim=cap,
            inputs=[self._sort_vals, self.index_to_handle, self.handle_to_index, self.store.count],
            device=d,
        )

        # Rebuild index_to_handle from the now-updated handle_to_index
        wp.launch(
            _rebuild_index_to_handle_kernel,
            dim=cap,
            inputs=[self.handle_to_index, self.index_to_handle, wp.array([cap], dtype=wp.int32, device=d), cap],
            device=d,
        )

        # 6. Update past_end = count (no holes after compaction)
        wp.copy(self.past_end, self.store.count)
