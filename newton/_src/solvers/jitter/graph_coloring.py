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


def _make_partitioning_coloring_kernel(incremental: bool):
    """Factory for the Luby MIS coloring kernel.

    The ``incremental`` flag toggles how a winning element records its claim:

    - Batch (``incremental=False``): tag ``partition_data_concat[tid]`` with the
      current color/overflow bits in place; a subsequent radix sort groups
      elements per color.
    - Incremental (``incremental=True``): atomically append the element id to
      ``partition_data_concat`` at ``partition_end_cursor[0]++``; no sort is
      needed because only the currently-emitted partition is produced.

    In the incremental variant the color scalars (``color``, ``luby_base``,
    ``luby_marker``) are derived from the device-side ``color_arr[0]`` and the
    kernel-scalar ``luby`` (0 or 1), so no host-side counter is required.
    """

    @wp.kernel
    def kernel(
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
        # Batch scalars (unused when incremental=True; pass 0s):
        color: int,
        luby_base: int,
        luby_marker: int,
        # Incremental-only inputs (unused when incremental=False; pass dummy size-1 arrays):
        color_arr: wp.array[int],
        luby: int,
        partition_end_cursor: wp.array[int],
    ):
        tid = wp.tid()

        if tid >= num_elements[0]:
            return

        if wp.static(incremental):
            color_copy = color_arr[0]
            luby_base_local = 2 * color_copy
            luby_marker_local = 2 * color_copy + luby
        else:
            color_copy = color
            luby_base_local = luby_base
            luby_marker_local = luby_marker

        section_marker = section_marker_single_el_arr[0]

        if contact_partitions_is_removed(
            partition_data_concat, tid, color_copy, removed_marker_array, luby_base_local, luby_marker_local
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
                    partition_data_concat,
                    neighbor,
                    color_copy,
                    removed_marker_array,
                    luby_base_local,
                    luby_marker_local,
                ):
                    if (
                        contact_partitions_get_random_value(random_values, neighbor, section_marker, max_num_contacts)
                        > self_prio
                    ):
                        is_local_max = False
                        break

        if is_local_max:
            removed_marker_array[tid] = luby_marker_local
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
                        partition_data_concat,
                        neighbor,
                        color_copy,
                        removed_marker_array,
                        luby_base_local,
                        luby_marker_local,
                    ):
                        removed_marker_array[neighbor] = luby_marker_local

            if wp.static(incremental):
                # Mark this element as removed from future partitions (same
                # bitfield scheme as batch mode, used by is_removed()) and
                # also append its id to the next free output slot. Slots
                # [max_num_contacts, 2*max_num_contacts) form the output
                # ring, disjoint from the per-element state slots [0, N).
                partition_data_concat[tid] = ((color_copy + 1) << 26) | tid
                idx = wp.atomic_add(partition_end_cursor, 0, 1)
                partition_data_concat[max_num_contacts + idx] = tid
            else:
                partition_data_concat[tid] = ((color_copy + 1) << 26) | tid
                wp.atomic_add(partition_ends, color_copy, 1)

    return kernel


partitioning_coloring_kernel = _make_partitioning_coloring_kernel(incremental=False)
partitioning_coloring_append_kernel = _make_partitioning_coloring_kernel(incremental=True)


# Cache of size-1 scratch dummies keyed by device, used to satisfy the shared
# coloring-kernel signature in batch mode (where color_arr / partition_end_cursor
# are ignored by the kernel body via `wp.static(incremental)` branches).
_COLORING_BATCH_DUMMY_CACHE: dict = {}


def _get_batch_coloring_dummy(device) -> wp.array:
    key = str(device) if device is not None else "default"
    arr = _COLORING_BATCH_DUMMY_CACHE.get(key)
    if arr is None:
        arr = wp.zeros(1, dtype=wp.int32, device=device)
        _COLORING_BATCH_DUMMY_CACHE[key] = arr
    return arr


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


# ---------------------------------------------------------------------------
# Incremental-mode helper kernels. All launched with dim=1.
# ---------------------------------------------------------------------------


@wp.kernel
def _increment_color_kernel(color: wp.array[int]):
    color[0] = color[0] + 1


@wp.kernel
def _incremental_finalize_kernel(
    partition_end_cursor: wp.array[int],
    num_elements: wp.array[int],
    not_done: wp.array[int],
):
    not_done[0] = wp.where(partition_end_cursor[0] < num_elements[0], 1, 0)


@wp.kernel
def _slide_partition_start_kernel(
    partition_start: wp.array[int],
    partition_end_cursor: wp.array[int],
):
    partition_start[0] = partition_end_cursor[0]


@wp.kernel
def _reset_counters_kernel(
    color: wp.array[int],
    partition_start: wp.array[int],
    partition_end_cursor: wp.array[int],
    not_done: wp.array[int],
):
    color[0] = 0
    partition_start[0] = 0
    partition_end_cursor[0] = 0
    not_done[0] = 1


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

    dummy = _get_batch_coloring_dummy(partition_data_concat.device)

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
                    # incremental-only (unused in batch mode):
                    dummy,
                    0,
                    dummy,
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


def maximal_independent_set_partitioning_incremental_setup(
    # Inputs:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    max_num_nodes: int,
    # Adjacency buffers (owned by the incremental partitioner):
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    removed_marker_array: wp.array[int],
    partition_data_concat: wp.array[int],
    max_used_color: wp.array[int],
    partition_ends_dummy: wp.array[int],
    # Device counters that parameterise each subsequent step():
    color: wp.array[int],
    partition_start: wp.array[int],
    partition_end_cursor: wp.array[int],
    not_done: wp.array[int],
) -> None:
    """Build the adjacency graph and initialise device-side incremental counters.

    Must be called once before a run of
    :func:`maximal_independent_set_partitioning_incremental_step` calls.

    ``partition_ends_dummy`` is a size-1 array only used to satisfy the shared
    kernel signature of ``partitioning_prepare_kernel``; its contents are not
    read during incremental mode.
    """
    max_num_interactions = partition_data_concat.shape[0] // 2

    prepare_dim = max(1, max_num_nodes)
    wp.launch(
        partitioning_prepare_kernel,
        dim=prepare_dim,
        inputs=[partition_ends_dummy, max_used_color, adjacency_section_end_indices, 0, num_elements],
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
            removed_marker_array,
            elements,
            num_elements,
        ],
    )

    wp.launch(
        _reset_counters_kernel,
        dim=1,
        inputs=[color, partition_start, partition_end_cursor, not_done],
    )


def maximal_independent_set_partitioning_incremental_step(
    # Inputs:
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    max_num_contacts: int,
    # Adjacency / scratch buffers (shared with setup):
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    removed_marker_array: wp.array[int],
    partition_data_concat: wp.array[int],
    max_used_color: wp.array[int],
    random_values: wp.array[int],
    # Device counters:
    color: wp.array[int],
    partition_start: wp.array[int],
    partition_end_cursor: wp.array[int],
    not_done: wp.array[int],
    # Size-1 dummy for the unused partition_ends kernel slot.
    partition_ends_dummy: wp.array[int],
) -> None:
    """Emit one partition. Results live in ``partition_data_concat[partition_start[0]:partition_end_cursor[0]]``.

    Sets ``not_done[0] = 0`` once every element has been assigned.
    """
    max_num_interactions = partition_data_concat.shape[0] // 2

    wp.launch(
        _slide_partition_start_kernel,
        dim=1,
        inputs=[partition_start, partition_end_cursor],
    )

    # Two Luby passes per color, matching the batch variant.
    for luby in range(2):
        wp.launch(
            partitioning_coloring_append_kernel,
            dim=max_num_interactions,
            inputs=[
                partition_data_concat,
                partition_ends_dummy,
                max_used_color,
                removed_marker_array,
                random_values,
                adjacency_section_end_indices,
                vertex_to_adjacent_elements,
                max_num_contacts,
                elements,
                num_elements,
                section_marker_single_el_arr,
                # batch-only scalars (ignored in incremental):
                0,
                0,
                0,
                # incremental inputs:
                color,
                luby,
                partition_end_cursor,
            ],
        )

    wp.launch(_increment_color_kernel, dim=1, inputs=[color])

    wp.launch(
        _incremental_finalize_kernel,
        dim=1,
        inputs=[partition_end_cursor, num_elements, not_done],
    )


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


class ContactPartitionerIncremental:
    """One-partition-at-a-time variant of :class:`ContactPartitioner`.

    Usage::

        p = ContactPartitionerIncremental(max_num_interactions, max_num_nodes)
        p.reset(elements, num_elements)
        while True:
            p.launch(elements, num_elements)
            # partition ids live in p.partition_data[p.partition_start[0] : p.partition_end[0]]
            # (read back only after the device work has completed)
            if int(p.not_done.numpy()[0]) == 0:
                break

    All mutable state between calls lives in size-1 ``wp.array[int]`` device
    scalars (``color``, ``partition_start``, ``partition_end``, ``not_done``),
    so the reset+launch sequence contains no host-side variables that would
    block a caller from wrapping it in a captured graph.

    There is no ``max_num_partitions`` cap: the caller drives the loop via
    ``not_done`` and is responsible for bounded iteration if desired.
    """

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.DeviceLike = None,
        seed: int = 0,
    ) -> None:
        self.max_num_interactions = max_num_interactions
        self.max_num_nodes = max_num_nodes
        self.max_num_contacts = max_num_interactions

        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(max_num_interactions).astype(np.int32) + 1
        self._random_values = wp.from_numpy(priorities, dtype=wp.int32, device=device)

        self._section_marker = wp.array([max_num_interactions], dtype=wp.int32, device=device)

        self._max_used_color = wp.zeros(1, dtype=wp.int32, device=device)
        self._removed_marker_array = wp.zeros(max_num_interactions, dtype=wp.int32, device=device)
        self._adjacency_section_end_indices = wp.zeros(max_num_nodes, dtype=wp.int32, device=device)
        self._vertex_to_adjacent_elements = wp.zeros(max_num_interactions * int(MAX_BODIES), dtype=wp.int32, device=device)

        # 2*N capacity to mirror the batch partitioner's ping-pong layout;
        # only the first max_num_interactions slots are written in incremental mode.
        self._partition_data_concat = wp.zeros(2 * max_num_interactions, dtype=wp.int32, device=device)

        # Size-1 dummy satisfying the shared coloring-kernel signature and the
        # prepare-kernel's partition_ends slot. Not read in incremental mode.
        self._partition_ends_dummy = wp.zeros(1, dtype=wp.int32, device=device)

        # Per-step device counters.
        self._color = wp.zeros(1, dtype=wp.int32, device=device)
        self._partition_start = wp.zeros(1, dtype=wp.int32, device=device)
        self._partition_end_cursor = wp.zeros(1, dtype=wp.int32, device=device)
        self._not_done = wp.zeros(1, dtype=wp.int32, device=device)

    def reset(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
    ) -> None:
        """Build the adjacency graph and clear all per-step counters."""
        maximal_independent_set_partitioning_incremental_setup(
            elements=elements,
            num_elements=num_elements,
            max_num_nodes=self.max_num_nodes,
            adjacency_section_end_indices=self._adjacency_section_end_indices,
            vertex_to_adjacent_elements=self._vertex_to_adjacent_elements,
            removed_marker_array=self._removed_marker_array,
            partition_data_concat=self._partition_data_concat,
            max_used_color=self._max_used_color,
            partition_ends_dummy=self._partition_ends_dummy,
            color=self._color,
            partition_start=self._partition_start,
            partition_end_cursor=self._partition_end_cursor,
            not_done=self._not_done,
        )

    def launch(
        self,
        elements: wp.array[ElementInteractionData],
        num_elements: wp.array[int],
    ) -> None:
        """Emit the next partition.

        After the launch completes, the partition's element ids occupy
        ``self.partition_data[self.partition_start[0] : self.partition_end[0]]``.
        """
        maximal_independent_set_partitioning_incremental_step(
            elements=elements,
            num_elements=num_elements,
            section_marker_single_el_arr=self._section_marker,
            max_num_contacts=self.max_num_contacts,
            adjacency_section_end_indices=self._adjacency_section_end_indices,
            vertex_to_adjacent_elements=self._vertex_to_adjacent_elements,
            removed_marker_array=self._removed_marker_array,
            partition_data_concat=self._partition_data_concat,
            max_used_color=self._max_used_color,
            random_values=self._random_values,
            color=self._color,
            partition_start=self._partition_start,
            partition_end_cursor=self._partition_end_cursor,
            not_done=self._not_done,
            partition_ends_dummy=self._partition_ends_dummy,
        )

    # ------------------------------------------------------------------
    # Device-array results (size-1 scalars unless noted).
    # ------------------------------------------------------------------

    @property
    def not_done(self) -> wp.array:
        """1 while elements remain unassigned, 0 once every element has been emitted."""
        return self._not_done

    @property
    def partition_start(self) -> wp.array:
        """Start index (inclusive) of the most recently emitted partition in :attr:`partition_data`."""
        return self._partition_start

    @property
    def partition_end(self) -> wp.array:
        """End index (exclusive) of the most recently emitted partition in :attr:`partition_data`."""
        return self._partition_end_cursor

    @property
    def partition_data(self) -> wp.array:
        """Output ring of emitted element ids. Slice ``[start, end)`` after each launch.

        Internally this is a view into the upper half of the 2*N
        ``partition_data_concat`` buffer; the lower half holds per-element
        state used by the coloring kernel and must not be read by callers.
        """
        return self._partition_data_concat[self.max_num_interactions :]

    @property
    def color(self) -> wp.array:
        """Current color counter (equals the number of partitions emitted so far)."""
        return self._color
