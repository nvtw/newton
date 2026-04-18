import warp as wp

from newton._src.solvers.jitter.scan_and_sort import scan_variable_length, sort_variable_length_int64

# Bit layout of partition_data_concat entries (wp.int64):
#   bits  0..31  : element id (tid)
#   bits 32..61  : color + 1  (partitioned tag, 1..~1e9)
#   bit  62      : unpartitioned marker (set by the adjacency-store kernel;
#                  cleared as soon as the element is assigned a color)
#
# We use int64 so that the encoding supports an essentially unlimited number
# of partitions. The previous int32 encoding packed color+1 into bits 26..30
# and used bit 30 as the unpartitioned marker; that made "color+1 == 16" alias
# the marker bit, silently double-assigning elements once more than 15
# partitions were needed.
_COLOR_SHIFT = wp.constant(wp.int64(32))
_ID_MASK = wp.constant(wp.int64((1 << 32) - 1))
_UNPARTITIONED = wp.constant(wp.int64(1 << 62))
# Mask for bits 0..61 (everything except the unpartitioned marker). Used in
# place of `~_UNPARTITIONED` because Warp's codegen does not reliably emit a
# 64-bit bitwise NOT for int64 constants.
_TAG_MASK = wp.constant(wp.int64((1 << 62) - 1))

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
    max_num_nodes: int,
):
    tid = wp.tid()

    if tid <= max_num_partitions:
        partition_ends[tid] = 0

    if tid == 0:
        max_used_color[0] = -1

    if tid < max_num_nodes:
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
    partition_data_concat: wp.array[wp.int64],
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

    # Assign a high value so unpartitioned (overflow) elements sort to the end.
    # Bits 32..61 stay zero until the coloring kernel fills in (color + 1).
    partition_data_concat[tid] = _UNPARTITIONED | wp.int64(tid)


@wp.func
def contact_partitions_is_removed(
    partition_data_concat: wp.array[wp.int64],
    i: int,
    color: int,
) -> bool:
    """Returns True iff element ``i`` is settled for the current Jones-Plassmann
    round at partition ``color``. An element is settled iff it was colored in
    an *earlier* round; elements colored this round are treated as conflicting
    so races between concurrent writers are resolved by priority.
    """
    rem = partition_data_concat[i] & _TAG_MASK
    rem = rem >> _COLOR_SHIFT
    # Unpartitioned OR just-committed-this-round → still "active" (conflicting).
    # Only elements colored in a previous round are removed.
    return rem != wp.int64(0) and rem != wp.int64(color + 1)


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
    partition_data_concat: wp.array[wp.int64],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    # Remaining PartitioningArgs fields:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    color: int,
):
    # Jones-Plassmann independent-set pass: a vertex joins partition `color`
    # iff its priority is strictly greater than all *uncolored* neighbours'
    # priorities. One kernel launch per color — no luby double-pass required.
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    color_copy = color
    section_marker = section_marker_single_el_arr[0]

    if contact_partitions_is_removed(partition_data_concat, tid, color_copy):
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
            if neighbor == tid:
                continue
            if contact_partitions_is_removed(partition_data_concat, neighbor, color_copy):
                continue
            if (
                contact_partitions_get_random_value(random_values, neighbor, section_marker, max_num_contacts)
                > self_prio
            ):
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)
        wp.atomic_add(partition_ends, color_copy, 1)


@wp.kernel
def partitioning_adjacency_finalize_pre_sort_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_ends: wp.array[int],
    num_partitions: wp.array[int],
    has_additional_partition: wp.array[int],
    max_used_color: wp.array[int],
    max_num_partitions: int,
    partition_data_concat: wp.array[wp.int64],
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

    if tid >= num_elements[0]:
        return

    # Bit 62 (unpartitioned marker) also falls in the ">> 32" window; strip it
    # first so overflow elements get mapped to max_num_partitions (clamped).
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = int(tagged >> _COLOR_SHIFT)
    interaction_id_to_partition[tid] = wp.min(max_num_partitions, color_plus_one - 1)


@wp.kernel
def partitioning_adjacency_finalize_post_sort_kernel(
    # ContactPartitionsGpu fields (unpacked):
    partition_data_concat: wp.array[wp.int64],
    partition_data_elements: wp.array[int],
    # Remaining PartitioningArgs fields:
    num_elements: wp.array[int],
):
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    # Strip color bits and overflow marker, keeping only the element id in
    # the low 32 bits. Write into the int32 output buffer used by callers.
    partition_data_elements[tid] = int(partition_data_concat[tid] & _ID_MASK)


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
    partition_data_concat: wp.array[wp.int64],
    partition_data_elements: wp.array[int],
    interaction_id_to_partition: wp.array[int],
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
        inputs=[partition_ends, max_used_color, adjacency_section_end_indices, max_num_partitions, max_num_nodes],
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
            elements,
            num_elements,
        ],
    )

    # Jones-Plassmann: one MIS kernel launch per color.
    for color in range(max_num_partitions):
        wp.launch(
            partitioning_coloring_kernel,
            dim=max_num_interactions,
            inputs=[
                partition_data_concat,
                partition_ends,
                max_used_color,
                random_values,
                adjacency_section_end_indices,
                vertex_to_adjacent_elements,
                max_num_contacts,
                elements,
                num_elements,
                section_marker_single_el_arr,
                color,
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
    # We widened partition_data_concat to int64 to fit (color+1) in bits 32..61;
    # signed int64 order == unsigned int64 order here because bit 63 stays clear.
    sort_variable_length_int64(partition_data_concat, partition_data_concat_sort_values, num_elements)

    wp.launch(
        partitioning_adjacency_finalize_post_sort_kernel,
        dim=max_num_interactions,
        inputs=[partition_data_concat, partition_data_elements, num_elements],
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
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device)

        # 2*N ping-pong buffer required by Warp's radix sort. int64 so the
        # (unpartitioned_marker | color_plus_one | tid) packing is lossless
        # for any realistic partition count.
        self._partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int64, device=device)
        self._partition_data_concat_sort_values = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)
        # int32 element-id view, filled by the post-sort finalize kernel.
        self._partition_data_elements = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)

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
            partition_data_elements=self._partition_data_elements,
            interaction_id_to_partition=self._interaction_id_to_partition,
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
        """Sorted, concatenated element ids grouped by partition (int32).
        Length ``max_num_interactions``; only the first ``num_elements[0]``
        entries are meaningful."""
        return self._partition_data_elements

    @property
    def interaction_id_to_partition(self) -> wp.array:
        """``interaction_id_to_partition[i]`` holds the partition index that
        element ``i`` was assigned to."""
        return self._interaction_id_to_partition
