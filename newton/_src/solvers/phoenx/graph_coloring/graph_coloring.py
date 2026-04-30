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

    # Serial header, done by thread 0 only. No cross-phase dependency with the
    # parallel body below, so other threads can run the body concurrently.
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

    # Bit 62 (unpartitioned marker) also falls in the ">> 32" window; strip it
    # first so overflow elements get mapped to max_num_partitions (clamped).
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

    # Strip color bits and overflow marker, keeping only the element id in
    # the low 32 bits. Write into the int32 output buffer used by callers.
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
    random_values: wp.array[int],
    cost_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    # Scratch buffer required by Warp's key-value radix sort.
    partition_data_concat_sort_values: wp.array[int],
    # 1-element device array used to feed the current color into the coloring
    # kernel. Allocated by :class:`ContactPartitioner`; supply a fresh one when
    # calling this function directly.
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
                random_values,
                cost_values,
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

    # We widened partition_data_concat to int64 to fit (color+1) in bits 32..61;
    # signed int64 order == unsigned int64 order here because bit 63 stays clear.
    sort_variable_length_int64(partition_data_concat, partition_data_concat_sort_values, num_elements)

    wp.launch(
        partitioning_adjacency_finalize_post_sort_kernel,
        dim=max_num_interactions,
        inputs=[partition_data_concat, partition_data_elements, num_elements],
    )


class ContactPartitioner:
    """User-friendly wrapper around :func:`maximal_independent_set_partitioning`.

    The constructor allocates all scratch/output buffers once and generates the
    Jones-Plassmann random-priority array. Each call to :meth:`launch` runs the
    full colouring pipeline on caller-supplied ``elements`` and writes results
    into the owned buffers, accessible via the ``partition_*`` properties.

    C# equivalent: ``ContactPartitions`` / ``ContactPartitionsGpu`` from
    PhoenX ``MassSplitting/ContactPartitions.cs``.
    """

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

        # Pairwise-distinct Jones-Plassmann priorities. A permutation of [1, N]
        # guarantees uniqueness, which the algorithm needs to break ties
        # between neighbouring elements.
        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)
        self._cost_values = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        self._partition_ends = wp.zeros(max_num_partitions + 1, dtype=wp.int32, device=device)
        self._num_partitions = wp.zeros(1, dtype=wp.int32, device=device)
        self._has_additional_partition = wp.zeros(1, dtype=wp.int32, device=device)
        self._max_used_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(
            max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device
        )

        # 2*N ping-pong buffer required by Warp's radix sort. int64 so the
        # (unpartitioned_marker | color_plus_one | tid) packing is lossless
        # for any realistic partition count.
        self._partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
        self._partition_data_concat_sort_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
        # int32 element-id view, filled by the post-sort finalize kernel.
        self._partition_data_elements = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # int32 colour-tag mirror used by the greedy variant; the
        # batch partitioner runs round-based JP only, so this array is
        # never read here -- it just satisfies
        # ``partitioning_adjacency_store_kernel``'s signature.
        self._color_tags = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

        # 1-element device array feeding the coloring kernel's per-color loop.
        self._color_arr = wp.zeros(1, dtype=wp.int32, device=device)

    def launch(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
        cost_values=None,
    ) -> None:
        """Run the colouring pipeline on ``elements[:num_elements[0]]``.

        Args:
            elements: Per-interaction body lists. Capacity must be
                ``>= max_num_interactions``.
            num_elements: Single-element device array holding the active count.
            cost_values: Optional per-interaction JP cost. If omitted, all costs
                are zero and the colourer uses the seeded jitter only.
        """
        if cost_values is None:
            cost_values = self._cost_values

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
            random_values=self._random_values,
            cost_values=cost_values,
            adjacency_section_end_indices=self._adjacency_section_end_indices,
            vertex_to_adjacent_elements=self._vertex_to_adjacent_elements,
            partition_data_concat_sort_values=self._partition_data_concat_sort_values,
            color_arr=self._color_arr,
        )

    # ------------------------------------------------------------------
    # Results (populated by the most recent ``launch`` call).
    # ------------------------------------------------------------------

    @property
    def num_partitions(self) -> wp.array:
        """Number of non-overflow partitions used (device scalar array)."""
        return self._num_partitions

    @property
    def has_additional_partition(self) -> wp.array:
        """1 if an overflow partition at index ``num_partitions`` is present,
        else 0 (device scalar array)."""
        return self._has_additional_partition

    @property
    def partition_ends(self) -> wp.array:
        """Inclusive cumulative sum: ``partition_ends[i]`` is the exclusive end
        index into :attr:`partition_data_concat` of partition ``i``."""
        return self._partition_ends

    @property
    def partition_data_concat(self) -> wp.array:
        """Sorted, concatenated element ids grouped by partition (int32).
        Length ``max_num_interactions``; only the first ``num_elements[0]``
        entries are meaningful."""
        return self._partition_data_elements

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` holds the partition index that
        element ``i`` was assigned to."""
        return self._interaction_id_to_partition
