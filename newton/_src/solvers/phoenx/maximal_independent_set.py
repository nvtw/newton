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

"""GPU-parallel graph coloring via Luby-style maximal independent sets.

Colours a hypergraph of *elements* (contacts / constraints) that share
*nodes* (rigid bodies).  Two elements are adjacent when they reference at
least one common node.  The algorithm assigns colours (partitions) such
that no two adjacent elements share the same colour, enabling safe
parallel processing within each partition.

Determinism is guaranteed by a fixed random-priority permutation generated
at construction time with a murmur-hash-based shuffle identical to the C#
PhoenX reference implementation.

All kernel launches use fixed ``dim`` equal to the pre-allocated capacity
and guard with on-device counts, making the full pipeline graph-capturable
via :class:`wp.ScopedCapture`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

MAX_BODIES_PER_ELEMENT = 8

# Sentinel used by the sort-pad kernel so inactive slots sort to the end.
_SORT_SENTINEL = 0x7FFFFFFF

# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def _murmur_hash11(src: int) -> int:
    """Murmur-style hash -- mirrors the C# ``murmurHash11``."""
    mask = 0xFFFFFFFF
    M = 0x5BD1E995
    h = 1190494759
    src = int(src) & mask
    src = (src * M) & mask
    src ^= src >> 24
    src = (src * M) & mask
    h = (h * M) & mask
    h ^= src
    h ^= h >> 13
    h = (h * M) & mask
    h ^= h >> 15
    return h


def _deterministic_shuffle(length: int) -> np.ndarray:
    """Fisher-Yates shuffle driven by :func:`_murmur_hash11`.

    Returns a permutation of ``[0, length)`` identical to the C# reference.
    """
    arr = np.arange(length, dtype=np.int32)
    n = length
    while n > 1:
        k = _murmur_hash11(n) % n
        n -= 1
        arr[n], arr[k] = arr[k], arr[n]
    return arr


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


@wp.func
def _is_removed(
    partition_data: wp.array(dtype=wp.int32),
    i: int,
    color: int,
    removed_marker: wp.array(dtype=wp.int32),
    luby_base: int,
    luby_marker: int,
) -> bool:
    marker = removed_marker[i]
    if marker >= luby_base and marker < luby_marker:
        return True
    rem = (partition_data[i] & ~(1 << 30)) >> 26
    not_removed = (rem == 0) or (rem == color + 1)
    return not not_removed


@wp.func
def _get_priority(
    random_values: wp.array(dtype=wp.int32),
    i: int,
    section_marker: int,
    max_num_elements: int,
) -> int:
    r = random_values[i]
    if i >= section_marker:
        r = r + max_num_elements
    return r


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _prepare_kernel(
    adjacency_offsets: wp.array(dtype=wp.int32),
    partition_ends: wp.array(dtype=wp.int32),
    max_used_color: wp.array(dtype=wp.int32),
    node_count: wp.array(dtype=wp.int32),
    max_colors_plus_one: int,
):
    tid = wp.tid()
    if tid < node_count[0]:
        adjacency_offsets[tid] = 0
    if tid < max_colors_plus_one:
        partition_ends[tid] = 0
    if tid == 0:
        max_used_color[0] = -1


@wp.kernel
def _adjacency_count_kernel(
    elements: wp.array2d(dtype=wp.int32),
    adjacency_offsets: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= element_count[0]:
        return
    for j in range(MAX_BODIES_PER_ELEMENT):
        v = elements[tid, j]
        if v < 0:
            break
        wp.atomic_add(adjacency_offsets, v, 1)


@wp.kernel
def _adjacency_store_kernel(
    elements: wp.array2d(dtype=wp.int32),
    adjacency_offsets: wp.array(dtype=wp.int32),
    adj_elements: wp.array(dtype=wp.int32),
    partition_data: wp.array(dtype=wp.int32),
    removed_marker: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= element_count[0]:
        return
    for j in range(MAX_BODIES_PER_ELEMENT):
        v = elements[tid, j]
        if v < 0:
            break
        idx = wp.atomic_add(adjacency_offsets, v, 1)
        adj_elements[idx] = tid
    partition_data[tid] = (1 << 30) | tid
    removed_marker[tid] = -1


@wp.kernel
def _coloring_kernel(
    elements: wp.array2d(dtype=wp.int32),
    partition_data: wp.array(dtype=wp.int32),
    removed_marker: wp.array(dtype=wp.int32),
    random_values: wp.array(dtype=wp.int32),
    adjacency_offsets: wp.array(dtype=wp.int32),
    adj_elements: wp.array(dtype=wp.int32),
    partition_ends: wp.array(dtype=wp.int32),
    max_used_color: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
    section_marker: wp.array(dtype=wp.int32),
    max_num_elements: int,
    color: int,
    luby_base: int,
    luby_marker: int,
):
    tid = wp.tid()
    if tid >= element_count[0]:
        return
    if _is_removed(partition_data, tid, color, removed_marker, luby_base, luby_marker):
        return

    if max_used_color[0] != color:
        max_used_color[0] = color

    sec = section_marker[0]
    self_prio = _get_priority(random_values, tid, sec, max_num_elements)
    is_local_max = int(1)

    for j in range(MAX_BODIES_PER_ELEMENT):
        if is_local_max == 0:
            break
        v = elements[tid, j]
        if v < 0:
            break
        start = 0
        if v > 0:
            start = adjacency_offsets[v - 1]
        end = adjacency_offsets[v]
        for k in range(start, end):
            neighbor = adj_elements[k]
            if not _is_removed(partition_data, neighbor, color, removed_marker, luby_base, luby_marker):
                if _get_priority(random_values, neighbor, sec, max_num_elements) > self_prio:
                    is_local_max = 0

    if is_local_max != 0:
        removed_marker[tid] = luby_marker
        for j in range(MAX_BODIES_PER_ELEMENT):
            v = elements[tid, j]
            if v < 0:
                break
            start = 0
            if v > 0:
                start = adjacency_offsets[v - 1]
            end = adjacency_offsets[v]
            for k in range(start, end):
                neighbor = adj_elements[k]
                if not _is_removed(partition_data, neighbor, color, removed_marker, luby_base, luby_marker):
                    removed_marker[neighbor] = luby_marker
        partition_data[tid] = ((color + 1) << 26) | tid
        wp.atomic_add(partition_ends, color, 1)


@wp.kernel
def _finalize_pre_sort_kernel(
    partition_data: wp.array(dtype=wp.int32),
    partition_ends: wp.array(dtype=wp.int32),
    num_partitions: wp.array(dtype=wp.int32),
    has_additional: wp.array(dtype=wp.int32),
    element_to_partition: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
    max_colors: int,
    max_used_color: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid == 0:
        total = int(0)
        for i in range(max_colors + 1):
            total = total + partition_ends[i]
            partition_ends[i] = total
        num_partitions[0] = wp.min(max_colors, max_used_color[0] + 1)
        n_el = element_count[0]
        if total != n_el:
            has_additional[0] = 1
            partition_ends[max_colors] = n_el
        else:
            has_additional[0] = 0

    if tid < element_count[0]:
        element_to_partition[tid] = wp.min(max_colors, (partition_data[tid] >> 26) - 1)


@wp.kernel
def _sort_pad_kernel(
    partition_data: wp.array(dtype=wp.int32),
    sort_values: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
    capacity: int,
):
    """Fill slots beyond active count with sentinel so they sort to the end."""
    tid = wp.tid()
    if tid < capacity:
        if tid < element_count[0]:
            sort_values[tid] = tid
        else:
            partition_data[tid] = _SORT_SENTINEL
            sort_values[tid] = tid


@wp.kernel
def _finalize_post_sort_kernel(
    partition_data: wp.array(dtype=wp.int32),
    element_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < element_count[0]:
        partition_data[tid] = partition_data[tid] & ((1 << 26) - 1)


# ---------------------------------------------------------------------------
# Host-side driver
# ---------------------------------------------------------------------------


class GraphColoring:
    """Host driver for Luby-MIS graph coloring.

    Colours a hypergraph of *elements* (each referencing up to 8 *nodes*)
    into independent-set partitions suitable for parallel Gauss-Seidel.

    All internal arrays are pre-allocated at construction time and every
    kernel launch uses a fixed ``dim``, so the full :meth:`color` pipeline
    is safe to record inside a CUDA graph.

    Args:
        max_elements: upper bound on the number of elements (contacts +
            constraints).
        max_nodes: upper bound on the number of nodes (rigid bodies).
        max_colors: maximum number of colour partitions.
        luby_iterations: Luby rounds per colour (default 2).
        device: Warp device string or object.
    """

    def __init__(
        self,
        max_elements: int,
        max_nodes: int,
        max_colors: int = 16,
        luby_iterations: int = 2,
        device: wp.context.Device | str | None = None,
    ):
        self.device = wp.get_device(device)
        self.max_elements = max_elements
        self.max_nodes = max_nodes
        self.max_colors = max_colors
        self.luby_iterations = luby_iterations

        d = self.device

        # Adjacency CSR
        self.adjacency_offsets = wp.zeros(max_nodes, dtype=wp.int32, device=d)
        self._adjacency_offsets_tmp = wp.zeros(max_nodes, dtype=wp.int32, device=d)
        adj_capacity = max_elements * MAX_BODIES_PER_ELEMENT
        self.adj_elements = wp.zeros(adj_capacity, dtype=wp.int32, device=d)

        # Partition bookkeeping -- 2x capacity for radix_sort_pairs
        self.partition_data = wp.zeros(2 * max_elements, dtype=wp.int32, device=d)
        self.sort_values = wp.zeros(2 * max_elements, dtype=wp.int32, device=d)
        self.partition_ends = wp.zeros(max_colors + 1, dtype=wp.int32, device=d)
        self.num_partitions = wp.zeros(1, dtype=wp.int32, device=d)
        self.has_additional = wp.zeros(1, dtype=wp.int32, device=d)
        self.max_used_color = wp.zeros(1, dtype=wp.int32, device=d)
        self.element_to_partition = wp.zeros(max_elements, dtype=wp.int32, device=d)

        # Luby state
        self.removed_marker = wp.zeros(max_elements, dtype=wp.int32, device=d)

        # Deterministic random priorities
        perm = _deterministic_shuffle(max_elements)
        self.random_values = wp.array(perm, dtype=wp.int32, device=d)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def color(
        self,
        elements: wp.array,
        element_count: wp.array,
        node_count: wp.array,
        section_marker: wp.array | None = None,
    ):
        """Run the full MIS coloring pipeline.

        After this call, :attr:`partition_data` ``[0:N]`` contains element
        indices sorted by colour, :attr:`partition_ends` holds inclusive
        cumulative sizes, and :attr:`num_partitions` /
        :attr:`has_additional` describe the partition layout.

        Args:
            elements: ``int32`` array of shape ``(max_elements, 8)``.  Each
                row lists the node (body) IDs for one element; ``-1`` marks
                unused slots.
            element_count: device ``int32`` array of shape ``(1,)`` with
                the number of active elements.
            node_count: device ``int32`` array of shape ``(1,)`` with the
                number of active nodes.
            section_marker: optional device ``int32`` array of shape
                ``(1,)``.  Elements at index ``>= section_marker`` receive a
                priority offset so that constraints are preferred over
                contacts.  If ``None`` a marker equal to ``max_elements`` is
                used (no offset).
        """
        d = self.device
        me = self.max_elements
        mn = self.max_nodes

        if section_marker is None:
            section_marker = wp.full(1, me, dtype=wp.int32, device=d)

        # 1. Prepare: zero adjacency offsets, partition ends, max_used_color
        wp.launch(
            _prepare_kernel,
            dim=max(mn, self.max_colors + 1),
            inputs=[
                self.adjacency_offsets,
                self.partition_ends,
                self.max_used_color,
                node_count,
                self.max_colors + 1,
            ],
            device=d,
        )

        # 2. Adjacency count
        wp.launch(
            _adjacency_count_kernel,
            dim=me,
            inputs=[elements, self.adjacency_offsets, element_count],
            device=d,
        )

        # 3. Exclusive prefix scan (counts -> start offsets)
        wp.utils.array_scan(self.adjacency_offsets, self._adjacency_offsets_tmp, inclusive=False)

        # Copy scanned result back (store kernel will atomically advance these)
        wp.copy(self.adjacency_offsets, self._adjacency_offsets_tmp)

        # 4. Adjacency store (scatter element indices; init partition_data / removed_marker)
        wp.launch(
            _adjacency_store_kernel,
            dim=me,
            inputs=[
                elements,
                self.adjacency_offsets,
                self.adj_elements,
                self.partition_data,
                self.removed_marker,
                element_count,
            ],
            device=d,
        )

        # 5. Luby coloring loop
        for color in range(self.max_colors):
            luby_base = self.luby_iterations * color
            for luby in range(self.luby_iterations):
                luby_marker = self.luby_iterations * color + luby
                wp.launch(
                    _coloring_kernel,
                    dim=me,
                    inputs=[
                        elements,
                        self.partition_data,
                        self.removed_marker,
                        self.random_values,
                        self.adjacency_offsets,
                        self.adj_elements,
                        self.partition_ends,
                        self.max_used_color,
                        element_count,
                        section_marker,
                        me,
                        color,
                        luby_base,
                        luby_marker,
                    ],
                    device=d,
                )

        # 6. Finalize pre-sort (prefix-sum partition sizes, compute num_partitions)
        wp.launch(
            _finalize_pre_sort_kernel,
            dim=me,
            inputs=[
                self.partition_data,
                self.partition_ends,
                self.num_partitions,
                self.has_additional,
                self.element_to_partition,
                element_count,
                self.max_colors,
                self.max_used_color,
            ],
            device=d,
        )

        # 7. Sort by (color << 26 | element_index)
        #    Pad inactive slots with a sentinel so they sort to the end.
        wp.launch(
            _sort_pad_kernel,
            dim=me,
            inputs=[self.partition_data, self.sort_values, element_count, me],
            device=d,
        )
        wp.utils.radix_sort_pairs(self.partition_data, self.sort_values, me)

        # 8. Finalize post-sort (strip colour bits)
        wp.launch(
            _finalize_post_sort_kernel,
            dim=me,
            inputs=[self.partition_data, element_count],
            device=d,
        )
