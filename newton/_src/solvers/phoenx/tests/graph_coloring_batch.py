# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batch graph-coloring reference used by tests."""

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    _COLOR_SHIFT,
    _ID_MASK,
    _TAG_MASK,
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_add,
    element_interaction_data_contains,
    element_interaction_data_count,
    element_interaction_data_empty,
    element_interaction_data_get,
    element_interaction_data_make,
    partitioning_adjacency_count_kernel,
    partitioning_adjacency_store_kernel,
    partitioning_coloring_kernel,
    partitioning_prepare_kernel,
    set_int_array_kernel,
    vec8i,
)
from newton._src.solvers.phoenx.helpers.scan_and_sort import scan_variable_length, sort_variable_length_int64

__all__ = [
    "MAX_BODIES",
    "ContactPartitioner",
    "ElementInteractionData",
    "element_interaction_data_add",
    "element_interaction_data_contains",
    "element_interaction_data_count",
    "element_interaction_data_empty",
    "element_interaction_data_get",
    "element_interaction_data_make",
    "maximal_independent_set_partitioning",
    "vec8i",
]


@wp.kernel(enable_backward=False)
def partitioning_adjacency_finalize_pre_sort_kernel(
    partition_ends: wp.array[int],
    num_partitions: wp.array[int],
    has_additional_partition: wp.array[int],
    max_used_color: wp.array[int],
    max_num_partitions: int,
    partition_data_concat: wp.array[wp.int64],
    interaction_id_to_partition: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    # Serial header on thread 0; rest of the threads run the body in parallel.
    if tid == 0:
        added_to_set_counter = int(0)
        for i in range(max_num_partitions + 1):
            added_to_set_counter += partition_ends[i]
            partition_ends[i] = added_to_set_counter

        num_partitions[0] = wp.min(max_num_partitions, max_used_color[0] + 1)

        if added_to_set_counter != num_elements[0]:
            has_additional_partition[0] = 1
        else:
            has_additional_partition[0] = 0

        if has_additional_partition[0] == 1:
            partition_ends[max_num_partitions] = num_elements[0]

    if tid >= num_elements[0]:
        return

    # Strip bit 62 (unpartitioned marker) before extracting the colour.
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = int(tagged >> _COLOR_SHIFT)
    interaction_id_to_partition[tid] = wp.min(max_num_partitions, color_plus_one - 1)


@wp.kernel(enable_backward=False)
def partitioning_adjacency_finalize_post_sort_kernel(
    partition_data_concat: wp.array[wp.int64],
    partition_data_elements: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    partition_data_elements[tid] = int(partition_data_concat[tid] & _ID_MASK)


def maximal_independent_set_partitioning(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    max_num_nodes: int,
    partition_ends: wp.array[int],
    num_partitions: wp.array[int],
    has_additional_partition: wp.array[int],
    max_used_color: wp.array[int],
    max_num_partitions: int,
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
    partition_data_elements: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    # Radix-sort scratch (size 2*N).
    partition_data_concat_sort_values: wp.array[int],
    # Per-color loop counter (1-elem device array).
    color_arr: wp.array[int],
) -> None:
    max_num_interactions = partition_data_concat.shape[0] // 2

    prepare_dim = max(max_num_partitions + 1, max_num_nodes)
    wp.launch(
        partitioning_prepare_kernel,
        dim=prepare_dim,
        inputs=[partition_ends, max_used_color, adjacency_section_end_indices, max_num_partitions, max_num_nodes],
    )

    wp.launch(
        partitioning_adjacency_count_kernel,
        dim=max_num_interactions,
        inputs=[adjacency_section_end_indices, elements, num_elements],
    )

    scan_variable_length(adjacency_section_end_indices, num_elements, inclusive=False)

    wp.launch(
        partitioning_adjacency_store_kernel,
        dim=max_num_interactions,
        inputs=[
            adjacency_section_end_indices,
            vertex_to_adjacent_elements,
            partition_data_concat,
            color_tags,
            elements,
            num_elements,
        ],
    )

    # Jones-Plassmann: one MIS kernel launch per color.
    for color in range(max_num_partitions):
        wp.launch(set_int_array_kernel, dim=1, inputs=[color_arr, color])
        wp.launch(
            partitioning_coloring_kernel,
            dim=max_num_interactions,
            inputs=[
                partition_data_concat,
                partition_ends,
                max_used_color,
                packed_priorities,
                adjacency_section_end_indices,
                vertex_to_adjacent_elements,
                elements,
                num_elements,
                color_arr,
            ],
        )

    wp.launch(
        partitioning_adjacency_finalize_pre_sort_kernel,
        dim=max(1, max_num_interactions),
        inputs=[
            partition_ends,
            num_partitions,
            has_additional_partition,
            max_used_color,
            max_num_partitions,
            partition_data_concat,
            interaction_id_to_partition,
            num_elements,
        ],
    )

    sort_variable_length_int64(partition_data_concat, partition_data_concat_sort_values, num_elements)

    wp.launch(
        partitioning_adjacency_finalize_post_sort_kernel,
        dim=max_num_interactions,
        inputs=[partition_data_concat, partition_data_elements, num_elements],
    )


class ContactPartitioner:
    """Wrapper around :func:`maximal_independent_set_partitioning`. Pre-allocates
    scratch/output buffers and the Jones-Plassmann priority array."""

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        max_num_partitions: int,
        device: wp.DeviceLike = None,
        seed: int = 0,
    ) -> None:
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        self.max_num_partitions = max_num_partitions

        # JP priorities: permutation of [1, N] (uniqueness for tiebreak),
        # stored prepacked as (cost << 24) | (random & 0xFFFFFF). Cost is
        # refreshed each step from contact counts via the partitioner's
        # set_costs_from_contacts hook; this batch-mode helper leaves it
        # at 0 so cost-biasing is opt-in only.
        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        if priorities.max() >= (1 << 24):
            raise ValueError(f"max_num_interactions ({max_num_interactions}) exceeds the 2^24 packed-priority limit.")
        packed_init = (priorities & 0x00FFFFFF).astype(np.int32)
        self._packed_priorities = wp.from_numpy(packed_init, dtype=wp.int32, device=device)

        self._partition_ends = wp.zeros(max_num_partitions + 1, dtype=wp.int32, device=device)
        self._num_partitions = wp.zeros(1, dtype=wp.int32, device=device)
        self._has_additional_partition = wp.zeros(1, dtype=wp.int32, device=device)
        self._max_used_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(
            max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device
        )

        # 2*N ping-pong (radix sort). int64 packs (marker | color+1 | tid).
        self._partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
        self._partition_data_concat_sort_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
        self._partition_data_elements = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # color_tags is unused here (greedy variant only); kept for kernel signature.
        self._color_tags = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        self._color_arr = wp.zeros(1, dtype=wp.int32, device=device)

    def launch(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
        packed_priorities=None,
    ) -> None:
        """Run the colouring pipeline on ``elements[:num_elements[0]]``.

        ``packed_priorities`` overrides the partitioner's seeded
        priorities (used by tests to inject deterministic
        ``(cost, random)`` packed values). When ``None`` uses the
        internal permutation-only buffer (cost-biasing off).
        """
        if packed_priorities is None:
            packed_priorities = self._packed_priorities

        maximal_independent_set_partitioning(
            elements=elements,
            num_elements=num_elements,
            max_num_nodes=self.max_num_nodes,
            partition_ends=self._partition_ends,
            num_partitions=self._num_partitions,
            has_additional_partition=self._has_additional_partition,
            max_used_color=self._max_used_color,
            max_num_partitions=self.max_num_partitions,
            partition_data_concat=self._partition_data_concat,
            color_tags=self._color_tags,
            partition_data_elements=self._partition_data_elements,
            interaction_id_to_partition=self._interaction_id_to_partition,
            packed_priorities=packed_priorities,
            adjacency_section_end_indices=self._adjacency_section_end_indices,
            vertex_to_adjacent_elements=self._vertex_to_adjacent_elements,
            partition_data_concat_sort_values=self._partition_data_concat_sort_values,
            color_arr=self._color_arr,
        )

    # Results from the most recent launch.

    @property
    def num_partitions(self) -> wp.array:
        """Number of non-overflow partitions (device scalar)."""
        return self._num_partitions

    @property
    def has_additional_partition(self) -> wp.array:
        """1 if an overflow partition at index num_partitions is present, else 0."""
        return self._has_additional_partition

    @property
    def partition_ends(self) -> wp.array:
        """Exclusive end index into partition_data_concat per partition."""
        return self._partition_ends

    @property
    def partition_data_concat(self) -> wp.array:
        """Element ids grouped by partition. First num_elements[0] entries valid."""
        return self._partition_data_elements

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """Partition index per element."""
        return self._interaction_id_to_partition
