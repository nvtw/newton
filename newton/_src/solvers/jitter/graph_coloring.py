import warp as wp

from newton._src.solvers.jitter.scan_and_sort import scan_variable_length, sort_variable_length_int

MAX_BODIES = wp.constant(8)

vec8i = wp.vec(length=8, dtype=wp.int32)


@wp.struct
class ElementInteractionData:
    # Body slots. Inactive slots hold -1.
    # Index 0..1 are the primary pair; indices 2..7 are optional.
    bodies: vec8i


@wp.func
def element_interaction_data_empty() -> ElementInteractionData:
    d = ElementInteractionData()
    d.bodies = vec8i(-1, -1, -1, -1, -1, -1, -1, -1)
    return d


@wp.func
def element_interaction_data_make(
    body1: int, body2: int, body3: int, body4: int, body5: int, body6: int, body7: int, body8: int
) -> ElementInteractionData:
    d = ElementInteractionData()
    d.bodies = vec8i(body1, body2, body3, body4, body5, body6, body7, body8)
    return d


@wp.func
def element_interaction_data_get(d: ElementInteractionData, index: int) -> int:
    if index >= MAX_BODIES:
        return -1
    return d.bodies[index]


@wp.func
def element_interaction_data_add(d: ElementInteractionData, body_id: int) -> ElementInteractionData:
    # Returns updated struct; `added` is true if a free slot was found.
    # Warp funcs return a single value; callers check whether any slot changed by comparing counts.
    for i in range(MAX_BODIES):
        if d.bodies[i] < 0:
            d.bodies[i] = body_id
            return d
    return d


@wp.func
def element_interaction_data_count(d: ElementInteractionData) -> int:
    for i in range(MAX_BODIES):
        if d.bodies[i] < 0:
            return i
    return MAX_BODIES


@wp.func
def element_interaction_data_contains(d: ElementInteractionData, body_id: int) -> bool:
    for i in range(MAX_BODIES):
        b = d.bodies[i]
        if b < 0:
            return False
        if body_id == b:
            return True
    return False


@wp.kernel
def partitioning_prepare_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    max_num_partitions: int,
    # Remaining PartitioningArgs fields:
    max_num_nodes: wp.array[int],
):
    tid = wp.tid()

    if tid <= max_num_partitions:
        partition_ends[tid] = 0

    if tid == 0:
        max_used_color[0] = -1

    total_num_work_packages = max_num_nodes[0]
    if tid < total_num_work_packages:
        adjacency_section_end_indices[tid] = 0


# TODO: Is a bit heavy on atomics
@wp.kernel
def partitioning_adjacency_count_kernel(
    # ContactPartitionsGpu fields (unpacked):
    adjacency_section_end_indices: wp.array[int],
    # Remaining PartitioningArgs fields:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    total_num_work_packages = num_elements[0]
    if tid >= total_num_work_packages:
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        vertex = element_interaction_data_get(el, j)
        if vertex < 0:
            break
        wp.atomic_add(adjacency_section_end_indices, vertex, 1)


@wp.kernel
def partitioning_adjacency_store_kernel(
    # ContactPartitionsGpu fields (unpacked):
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    partition_data_concat: wp.array[int],
    removed_marker_array: wp.array[int],
    # Remaining PartitioningArgs fields:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        vertex = element_interaction_data_get(el, j)
        if vertex < 0:
            break
        index = wp.atomic_add(adjacency_section_end_indices, vertex, 1)
        vertex_to_adjacent_elements[index] = tid

    # Assign high value such that unpartitioned (overflow) elements end up at
    # the end when sorting. Bits 26..30 must remain zero -- there the
    # partition id (color) will be written.
    partition_data_concat[tid] = (1 << 30) | tid
    removed_marker_array[tid] = -1


@wp.func
def contact_partitions_is_removed(
    removed: wp.array[int],
    i: int,
    color: int,
    helper: wp.array[int],
    helper_low: int,
    helper_high: int,
) -> bool:
    if helper[i] >= helper_low and helper[i] < helper_high:
        return True

    rem = removed[i] & (~(1 << 30))
    rem = rem >> 26
    not_removed = rem == 0 or rem == color + 1
    return not not_removed


@wp.func
def contact_partitions_get_random_value(
    random_values: wp.array[int], i: int, section_marker: int, max_num_contacts: int
) -> int:
    r = random_values[i]
    if i >= section_marker:
        r += max_num_contacts
    return r


@wp.kernel
def partitioning_coloring_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_data_concat: wp.array[int],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    removed_marker_array: wp.array[int],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    # Remaining PartitioningArgs fields:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    color: int,
    luby_base: int,
    luby_marker: int,
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    color_copy = color
    section_marker = section_marker_single_el_arr[0]

    if contact_partitions_is_removed(
        partition_data_concat, tid, color_copy, removed_marker_array, luby_base, luby_marker
    ):
        return

    if max_used_color[0] != color_copy:
        max_used_color[0] = color_copy

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(random_values, tid, section_marker, max_num_contacts)
    el = elements[tid]

    for j in range(MAX_BODIES):
        if not is_local_max:
            break
        v = element_interaction_data_get(el, j)
        if v < 0:
            break

        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = 0
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if not contact_partitions_is_removed(
                partition_data_concat, neighbor, color_copy, removed_marker_array, luby_base, luby_marker
            ):
                if (
                    contact_partitions_get_random_value(random_values, neighbor, section_marker, max_num_contacts)
                    > self_prio
                ):
                    is_local_max = False
                    break

    if is_local_max:
        removed_marker_array[tid] = luby_marker
        for j in range(MAX_BODIES):
            v = element_interaction_data_get(el, j)
            if v < 0:
                break

            if v > 0:
                start = adjacency_section_end_indices[v - 1]
            else:
                start = 0
            end = adjacency_section_end_indices[v]
            for k in range(start, end):
                neighbor = vertex_to_adjacent_elements[k]
                if not contact_partitions_is_removed(
                    partition_data_concat, neighbor, color_copy, removed_marker_array, luby_base, luby_marker
                ):
                    removed_marker_array[neighbor] = luby_marker

        # Debug check from C#: partitionDataConcat[tid] should still be (1<<30)|tid
        # at this point. Skipped here (Warp has no host-side VALIDATION_ASSERT in
        # kernels); can be added via wp.printf if needed during bring-up.

        partition_data_concat[tid] = ((color_copy + 1) << 26) | tid

        wp.atomic_add(partition_ends, color_copy, 1)


@wp.kernel
def partitioning_adjacency_finalize_pre_sort_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_ends: wp.array[int],
    num_partitions: wp.array[int],
    has_additional_partition: wp.array[int],
    max_used_color: wp.array[int],
    max_num_partitions: int,
    partition_data_concat: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    # Remaining PartitioningArgs fields:
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
            # Debug check from C#: num_partitions[0] should equal
            # max_num_partitions here. Skipped (see coloring kernel note).

    if tid >= num_elements[0]:
        return

    interaction_id_to_partition[tid] = wp.min(max_num_partitions, (partition_data_concat[tid] >> 26) - 1)


@wp.kernel
def partitioning_adjacency_finalize_post_sort_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_data_concat: wp.array[int],
    # Remaining PartitioningArgs fields:
    num_elements: wp.array[int],
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    partition_data_concat[tid] &= (1 << 26) - 1


def maximal_independent_set_partitioning(
    # Inputs from caller (C#: method arguments):
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    max_num_nodes: int,
    section_marker_single_el_arr: wp.array[int],
    # ContactPartitionsGpu fields (C#: gpuCore / self.Data):
    partition_ends: wp.array[int],
    num_partitions: wp.array[int],
    has_additional_partition: wp.array[int],
    max_used_color: wp.array[int],
    max_num_partitions: int,
    partition_data_concat: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    removed_marker_array: wp.array[int],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    # Scratch buffer required by Warp's key-value radix sort.
    partition_data_concat_sort_values: wp.array[int],
) -> None:
    # Launch dims use host-side capacities (option (a) pattern).
    # max_num_interactions is the allocated capacity of elements/num_elements
    # and every per-element array. Radix sort requires a 2*N ping-pong buffer,
    # so partition_data_concat.shape[0] == 2 * max_num_interactions.
    max_num_interactions = partition_data_concat.shape[0] // 2

    prepare_dim = max(max_num_partitions + 1, max_num_nodes)
    wp.launch(
        partitioning_prepare_kernel,
        dim=prepare_dim,
        inputs=[partition_ends, max_used_color, adjacency_section_end_indices, max_num_partitions, num_elements],
    )

    wp.launch(
        partitioning_adjacency_count_kernel,
        dim=max_num_interactions,
        inputs=[adjacency_section_end_indices, elements, num_elements],
    )

    # C#: scan.exclusiveScan(gpuCore.adjacencySectionEndIndices, stream, maxNumNodes)
    scan_variable_length(adjacency_section_end_indices, num_elements, inclusive=False)

    wp.launch(
        partitioning_adjacency_store_kernel,
        dim=max_num_interactions,
        inputs=[
            adjacency_section_end_indices,
            vertex_to_adjacent_elements,
            partition_data_concat,
            removed_marker_array,
            elements,
            num_elements,
        ],
    )

    num_luby_iterations = 2

    for color in range(max_num_partitions):
        luby_base = num_luby_iterations * color
        for luby in range(num_luby_iterations):
            luby_marker = num_luby_iterations * color + luby

            wp.launch(
                partitioning_coloring_kernel,
                dim=max_num_interactions,
                inputs=[
                    partition_data_concat,
                    partition_ends,
                    max_used_color,
                    removed_marker_array,
                    random_values,
                    adjacency_section_end_indices,
                    vertex_to_adjacent_elements,
                    max_num_contacts,
                    elements,
                    num_elements,
                    section_marker_single_el_arr,
                    color,
                    luby_base,
                    luby_marker,
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

    # C#: sort.sort((uint*)partitionDataConcat.As<uint>(), 32, stream, numElements)
    # Keys-only uint32 sort in C#. We use int32 here because partition_data_concat
    # values are non-negative (top bit stays clear), so signed order == unsigned order.
    sort_variable_length_int(partition_data_concat, partition_data_concat_sort_values, num_elements)

    wp.launch(
        partitioning_adjacency_finalize_post_sort_kernel,
        dim=max_num_interactions,
        inputs=[partition_data_concat, num_elements],
    )

    # C# #if DEBUG ValidatePartitions(...) -- not translated.


class ContactPartitioner:
    """User-friendly wrapper around :func:`maximal_independent_set_partitioning`.

    The constructor allocates all scratch/output buffers once and generates the
    Luby MIS random-priority array. Each call to :meth:`launch` runs the full
    colouring pipeline on caller-supplied ``elements`` and writes results into
    the owned buffers, accessible via the ``partition_*`` properties.

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
        # C# sets maxNumContacts = maxNumInteractions in the ctor.
        self.max_num_contacts = max_num_interactions

        # Pairwise-distinct Luby MIS priorities. A permutation of [1, N]
        # guarantees uniqueness, which the algorithm needs to break ties
        # between neighbouring elements.
        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)

        # All elements belong to a single "section" by default -> marker past
        # the end disables the offset in `GetRandomValue`.
        self._section_marker = wp.array([max_num_interactions], dtype=wp.int32, device=device)

        self._partition_ends = wp.zeros(max_num_partitions + 1, dtype=wp.int32, device=device)
        self._num_partitions = wp.zeros(1, dtype=wp.int32, device=device)
        self._has_additional_partition = wp.zeros(1, dtype=wp.int32, device=device)
        self._max_used_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._interaction_id_to_partition = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._removed_marker_array = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device)

        # 2*N ping-pong buffer required by Warp's radix sort.
        self._partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
        self._partition_data_concat_sort_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)

    def launch(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
    ) -> None:
        """Run the colouring pipeline on ``elements[:num_elements[0]]``.

        Args:
            elements: Per-interaction body lists. Capacity must be
                ``>= max_num_interactions``.
            num_elements: Single-element device array holding the active count.
        """
        maximal_independent_set_partitioning(
            elements=elements,
            num_elements=num_elements,
            max_num_nodes=self.max_num_nodes,
            section_marker_single_el_arr=self._section_marker,
            partition_ends=self._partition_ends,
            num_partitions=self._num_partitions,
            has_additional_partition=self._has_additional_partition,
            max_used_color=self._max_used_color,
            max_num_partitions=self.max_num_partitions,
            partition_data_concat=self._partition_data_concat,
            interaction_id_to_partition=self._interaction_id_to_partition,
            removed_marker_array=self._removed_marker_array,
            random_values=self._random_values,
            adjacency_section_end_indices=self._adjacency_section_end_indices,
            vertex_to_adjacent_elements=self._vertex_to_adjacent_elements,
            max_num_contacts=self.max_num_contacts,
            partition_data_concat_sort_values=self._partition_data_concat_sort_values,
        )

    # ------------------------------------------------------------------
    # Results (populated by the most recent `launch` call).
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
        """Sorted, concatenated element ids grouped by partition. Length
        ``2 * max_num_interactions`` (ping-pong); only the first
        ``num_elements[0]`` entries are meaningful."""
        return self._partition_data_concat

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` holds the partition index that
        element ``i`` was assigned to."""
        return self._interaction_id_to_partition
