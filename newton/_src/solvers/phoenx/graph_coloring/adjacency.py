# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vertex -> adjacent-element CSR built from ``ElementInteractionData``.

A small reusable helper that owns the two-array CSR ``(section_end_indices,
vertex_to_adjacent_elements)`` describing which constraint elements touch
each body. The PhoenX graph coloring pipeline constructs this for its
JP-MIS neighbour walk.

The class delegates to the existing kernels in
:mod:`graph_coloring.graph_coloring_common` to keep behaviour byte-for-byte
identical with the current coloring path; future cleanups may move those
kernel definitions over here.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import scan_variable_length

__all__ = ["ElementVertexAdjacency"]


@wp.kernel(enable_backward=False)
def _adjacency_zero_section_ends_kernel(
    section_end_indices: wp.array[wp.int32],
    max_num_nodes: wp.int32,
):
    tid = wp.tid()
    if tid < max_num_nodes:
        section_end_indices[tid] = wp.int32(0)


class ElementVertexAdjacency:
    """Build & own a vertex -> adjacent-element CSR.

    The store kernel from graph coloring also touches a couple of partition-
    scratch buffers as a side-effect; this class allocates throwaway versions
    of those so the standalone adjacency build doesn't require the caller to
    own coloring state. Coloring callers that already own those buffers can
    instead use :meth:`build_into` and pass theirs in.

    Usage::

        adj = ElementVertexAdjacency(max_num_interactions=N, max_num_nodes=M)
        adj.build(elements, num_elements)
        # adj.section_end_indices and adj.vertex_to_adjacent_elements are now
        # populated for [0, num_elements[0]).

    Fixed launch sizes; graph-capture safe.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.context.Devicelike = None,
    ) -> None:
        if max_num_interactions <= 0:
            raise ValueError(f"max_num_interactions must be > 0 (got {max_num_interactions})")
        if max_num_nodes <= 0:
            raise ValueError(f"max_num_nodes must be > 0 (got {max_num_nodes})")
        self._max_num_interactions = int(max_num_interactions)
        self._max_num_nodes = int(max_num_nodes)
        self._device = wp.get_device(device)

        self.section_end_indices: wp.array[wp.int32] = wp.zeros(
            self._max_num_nodes, dtype=wp.int32, device=self._device
        )
        self.vertex_to_adjacent_elements: wp.array[wp.int32] = wp.zeros(
            self._max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=self._device
        )
        # Throwaway scratch for the store kernel's coloring side-effects.
        self._scratch_partition_data_concat: wp.array[wp.int64] = wp.zeros(
            self._max_num_interactions, dtype=wp.int64, device=self._device
        )
        self._scratch_color_tags: wp.array[wp.int32] = wp.zeros(
            self._max_num_interactions, dtype=wp.int32, device=self._device
        )

    @property
    def max_num_interactions(self) -> int:
        return self._max_num_interactions

    @property
    def max_num_nodes(self) -> int:
        return self._max_num_nodes

    def build(
        self,
        elements: wp.array,  # wp.array[ElementInteractionData]
        num_elements: wp.array[wp.int32],
    ) -> None:
        """Build the CSR for ``elements[:num_elements[0]]`` into the
        owned ``section_end_indices`` / ``vertex_to_adjacent_elements``."""
        self.build_into(
            elements=elements,
            num_elements=num_elements,
            section_end_indices=self.section_end_indices,
            vertex_to_adjacent_elements=self.vertex_to_adjacent_elements,
            partition_data_concat=self._scratch_partition_data_concat,
            color_tags=self._scratch_color_tags,
        )

    def build_into(
        self,
        elements: wp.array,
        num_elements: wp.array[wp.int32],
        section_end_indices: wp.array[wp.int32],
        vertex_to_adjacent_elements: wp.array[wp.int32],
        partition_data_concat: wp.array[wp.int64],
        color_tags: wp.array[wp.int32],
    ) -> None:
        """Build into caller-owned buffers. Lets the graph coloring path
        reuse its existing partition scratch instead of allocating duplicates.

        The store kernel writes ``_UNPARTITIONED | tid`` into
        ``partition_data_concat[tid]`` and zero into ``color_tags[tid]`` --
        callers that want a clean adjacency-only build can ignore those
        side-effects.
        """
        wp.launch(
            _adjacency_zero_section_ends_kernel,
            dim=self._max_num_nodes,
            inputs=[section_end_indices, self._max_num_nodes],
            device=self._device,
        )
        wp.launch(
            partitioning_adjacency_count_kernel,
            dim=self._max_num_interactions,
            inputs=[section_end_indices, elements, num_elements],
            device=self._device,
        )
        scan_variable_length(section_end_indices, num_elements, inclusive=False)
        wp.launch(
            partitioning_adjacency_store_kernel,
            dim=self._max_num_interactions,
            inputs=[
                section_end_indices,
                vertex_to_adjacent_elements,
                partition_data_concat,
                color_tags,
                elements,
                num_elements,
            ],
            device=self._device,
        )
