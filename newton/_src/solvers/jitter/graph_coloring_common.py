"""Shared primitives for the batch and incremental graph-coloring partitioners.

This module holds the data structures, bit-packing constants, adjacency-building
kernels, and Jones-Plassmann MIS kernel that are used by both
:mod:`graph_coloring` (batch, budgeted, sorted output) and
:mod:`graph_coloring_incremental` (one partition per launch, no sort).

Separating these primitives keeps the two entry-point modules thin and makes
it explicit which kernels are safe to reuse.
"""

import warp as wp

# Bit layout of ``partition_data_concat`` entries (``wp.int64``):
#   bits  0..31  : element id (tid)
#   bits 32..61  : color + 1  (partitioned tag, 1..~1e9)
#   bit  62      : unpartitioned marker (set by the adjacency-store kernel;
#                  cleared as soon as the element is assigned a color)
#
# int64 so the encoding supports an essentially unlimited number of partitions.
# The previous int32 encoding packed color+1 into bits 26..30 and used bit 30
# as the unpartitioned marker; that made "color+1 == 16" alias the marker bit
# and silently double-assigned elements once more than 15 partitions were used.
_COLOR_SHIFT = wp.constant(wp.int64(32))
_ID_MASK = wp.constant(wp.int64((1 << 32) - 1))
_UNPARTITIONED = wp.constant(wp.int64(1 << 62))
# Mask for bits 0..61 (everything except the unpartitioned marker). Used in
# place of ``~_UNPARTITIONED`` because Warp's codegen does not reliably emit a
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
    # Returns updated struct; callers detect success by a count comparison.
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


# ---------------------------------------------------------------------------
# Adjacency-building kernels (shared by batch and incremental paths).
# ---------------------------------------------------------------------------


@wp.kernel
def partitioning_prepare_kernel(
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
    adjacency_section_end_indices: wp.array[int],
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
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    partition_data_concat: wp.array[wp.int64],
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


# ---------------------------------------------------------------------------
# Jones-Plassmann MIS kernel (shared).
# ---------------------------------------------------------------------------


@wp.func
def contact_partitions_is_removed(
    partition_data_concat: wp.array[wp.int64],
    i: int,
    color: int,
) -> bool:
    """Returns True iff element ``i`` is settled for the current Jones-Plassmann
    round at partition ``color``.

    An element is settled iff it was colored in an *earlier* round; elements
    colored this round are treated as conflicting so races between concurrent
    writers are resolved by priority.
    """
    rem = partition_data_concat[i] & _TAG_MASK
    rem = rem >> _COLOR_SHIFT
    # Unpartitioned OR just-committed-this-round -> still "active" (conflicting).
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
    partition_data_concat: wp.array[wp.int64],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    random_values: wp.array[int],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    max_num_contacts: int,
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    section_marker_single_el_arr: wp.array[int],
    color_arr: wp.array[int],
):
    # Jones-Plassmann independent-set pass: a vertex joins partition
    # ``color_arr[0]`` iff its priority is strictly greater than all
    # *uncolored* neighbours' priorities. ``color_arr`` lives on the device
    # so the launcher can iterate without host readbacks (graph-capture safe).
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    color_copy = color_arr[0]
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


# ---------------------------------------------------------------------------
# Incremental-path helper kernels.
# ---------------------------------------------------------------------------


@wp.kernel
def incremental_init_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
    num_elements: wp.array[int],
):
    # Single-thread launch: set up device-side loop state after adjacency is
    # built. Using a kernel rather than host writes keeps reset() graph-capture
    # friendly if the caller ever wants to capture reset + launch together.
    current_color[0] = 0
    num_remaining[0] = num_elements[0]
    partition_count[0] = 0


@wp.kernel
def incremental_fill_minus_one_kernel(
    arr: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    arr[tid] = -1


@wp.kernel
def incremental_zero_int_kernel(
    arr: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    arr[tid] = 0


@wp.kernel
def incremental_flag_kernel(
    partition_data_concat: wp.array[wp.int64],
    current_color: wp.array[int],
    flags: wp.array[int],
    num_elements: wp.array[int],
):
    # flags[tid] = 1 iff element tid was just committed to partition
    # current_color[0] by the most recent JP coloring pass. Elements from
    # earlier rounds have a higher "color+1" stored, unpartitioned elements
    # still carry the _UNPARTITIONED marker; masking with _TAG_MASK strips
    # the marker and the comparison identifies only this round's new entries.
    tid = wp.tid()
    n = num_elements[0]
    if tid >= n:
        flags[tid] = 0
        return

    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(current_color[0] + 1):
        flags[tid] = 1
    else:
        flags[tid] = 0


@wp.kernel
def incremental_compact_kernel(
    flags: wp.array[int],
    offsets: wp.array[int],
    current_color: wp.array[int],
    partition_element_ids: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    partition_count: wp.array[int],
    num_elements: wp.array[int],
):
    # Given flags (0/1 just-committed) and their exclusive prefix sum, write
    # the compacted element list for the current partition and record the
    # assigned color per element. One thread stores the total count.
    tid = wp.tid()
    n = num_elements[0]

    # Handle degenerate n == 0 case (no work) and publish zero count.
    if n == 0:
        if tid == 0:
            partition_count[0] = 0
        return

    if tid >= n:
        return

    if flags[tid] == 1:
        partition_element_ids[offsets[tid]] = tid
        interaction_id_to_partition[tid] = current_color[0]

    # Last active thread publishes the partition size.
    if tid == n - 1:
        partition_count[0] = offsets[tid] + flags[tid]


@wp.kernel
def incremental_advance_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
):
    num_remaining[0] = num_remaining[0] - partition_count[0]
    current_color[0] = current_color[0] + 1


@wp.kernel
def set_int_array_kernel(arr: wp.array[int], value: int):
    # Tiny helper used by batch launcher to push the host-side color counter
    # into the device array consumed by ``partitioning_coloring_kernel``.
    arr[0] = value
