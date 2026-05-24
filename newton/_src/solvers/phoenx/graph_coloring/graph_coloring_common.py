"""Shared primitives for the batch and incremental graph-coloring partitioners.

Holds data structures, bit-packing constants, adjacency-building kernels,
and the Jones-Plassmann MIS kernel.
"""

import warp as wp

# partition_data_concat (int64) layout:
#   bits  0..31 : element id
#   bits 32..61 : color + 1  (0 = uncoloured)
#   bit  62     : unpartitioned marker
_COLOR_SHIFT = wp.constant(wp.int64(32))
_ID_MASK = wp.constant(wp.int64((1 << 32) - 1))
_UNPARTITIONED = wp.constant(wp.int64(1 << 62))
# Bits 0..61. Avoids ~_UNPARTITIONED — Warp codegen doesn't reliably emit
# 64-bit bitwise NOT on int64 constants.
_TAG_MASK = wp.constant(wp.int64((1 << 62) - 1))

MAX_BODIES = wp.constant(8)

vec8i = wp.types.vector(length=8, dtype=wp.int32)


@wp.struct
class ElementInteractionData:
    # Body slots; -1 = inactive. Slots 0..1 = primary pair; 2..7 optional.
    # Width 8 fits every constraint type (joint=2, cloth-tri=3,
    # cloth-bending=4, soft-tet shear=4, soft-tet-vs-soft-tet contact=6,
    # cloth-cloth contact=6). Was 6 historically (see PERF_NOTES.md
    # 2026-05-12); re-widened to 8.
    bodies: vec8i


@wp.func
def element_interaction_data_empty() -> ElementInteractionData:
    d = ElementInteractionData()
    d.bodies = vec8i(-1, -1, -1, -1, -1, -1, -1, -1)
    return d


@wp.func
def element_interaction_data_make(
    body1: int,
    body2: int,
    body3: int,
    body4: int,
    body5: int,
    body6: int,
    body7: int,
    body8: int,
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


# Adjacency-building kernels (shared by batch and incremental paths).


@wp.kernel(enable_backward=False)
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


@wp.kernel(enable_backward=False)
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


@wp.kernel(enable_backward=False)
def partitioning_adjacency_store_kernel(
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
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

    # Mark unpartitioned (sorts to end). Coloring kernel fills bits 32..61.
    partition_data_concat[tid] = _UNPARTITIONED | wp.int64(tid)
    # Mirror of colour bits for the greedy kernel's 4-byte hot-path reads.
    color_tags[tid] = wp.int32(0)


# Jones-Plassmann MIS kernel (shared).


@wp.func
def contact_partitions_is_removed(
    partition_data_concat: wp.array[wp.int64],
    i: int,
    color: int,
) -> bool:
    """True iff element ``i`` was coloured in a *previous* round. Elements
    coloured this round count as conflicting (resolved by priority)."""
    rem = partition_data_concat[i] & _TAG_MASK
    rem = rem >> _COLOR_SHIFT
    return rem != wp.int64(0) and rem != wp.int64(color + 1)


#: Bit layout of the packed-priority int32:
#:
#: * bits 24..31 : cost  (contact-count bias; 0 for non-contact rows)
#: * bits  0..23 : random tiebreaker (unique JP/Luby permutation jitter)
#:
#: Cost in the high byte makes plain int32 lexicographic comparison
#: equivalent to (cost, random) lexicographic ordering -- contacts win
#: over non-contacts; ties broken by the random permutation. Capping
#: cost at 255 is fine for our scenes (contact column count rarely
#: exceeds a few dozen).
PACKED_PRIO_COST_SHIFT = wp.constant(wp.int32(24))
PACKED_PRIO_RANDOM_MASK = wp.constant(wp.int32(0x00FFFFFF))
PACKED_PRIO_COST_MAX = 0xFF  # host-side cap when filling


@wp.func
def contact_partitions_get_random_value(
    packed_priorities: wp.array[wp.int32],
    i: int,
) -> wp.int32:
    """Read the prepacked (cost << 24) | (random & 0xFFFFFF) priority.

    Replaces a 2-load + cast + shift + OR per access with a single int32
    load. Pack happens once per step in
    :func:`pack_priorities_kernel` (called from
    ``set_costs_from_contacts``); coloring kernels just read.
    """
    return packed_priorities[i]


@wp.kernel(enable_backward=False)
def pack_priorities_kernel(
    random_values: wp.array[wp.int32],
    cost_values: wp.array[wp.int32],
    packed_priorities: wp.array[wp.int32],
):
    """Pack ``(cost, random)`` into a single int32 priority.

    Writes ``packed[i] = (cost[i] << 24) | (random[i] & 0xFFFFFF)``.
    Caller is responsible for keeping ``cost_values[i] <= 255`` (the
    contact-count bias only goes through 8 bits) and
    ``random_values[i] < 2^24`` (16M-element headroom; the partitioner
    seeds with ``permutation(max_num_interactions) + 1``).
    """
    tid = wp.tid()
    if tid >= packed_priorities.shape[0]:
        return
    cost = cost_values[tid]
    rand = random_values[tid] & PACKED_PRIO_RANDOM_MASK
    packed_priorities[tid] = (cost << PACKED_PRIO_COST_SHIFT) | rand


@wp.kernel(enable_backward=False)
def partitioning_coloring_kernel(
    partition_data_concat: wp.array[wp.int64],
    partition_ends: wp.array[int],
    max_used_color: wp.array[int],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    color_arr: wp.array[int],
):
    # JP MIS pass: vertex joins color iff its priority > all uncoloured
    # neighbours'. color_arr lives on device for graph-capture safety.
    tid = wp.tid()

    if tid >= num_elements[0]:
        return

    color_copy = color_arr[0]

    if contact_partitions_is_removed(partition_data_concat, tid, color_copy):
        return

    if max_used_color[0] != color_copy:
        max_used_color[0] = color_copy

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(packed_priorities, tid)
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
            if contact_partitions_get_random_value(packed_priorities, neighbor) > self_prio:
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)
        wp.atomic_add(partition_ends, color_copy, 1)


# Greedy variant: int64 forbidden-colour mask. Overflow past 64 sets a flag.
GREEDY_MAX_COLORS = wp.constant(64)

#: Outer-iteration cap for the greedy ``capture_while`` build. Each outer
#: iteration runs ``NUM_INNER_WHILE_ITERATIONS`` (8) inner greedy launches,
#: so 16 outer = 128 inner launches max. Past this cap the remaining
#: uncoloured elements are force-spilled to the overflow colour (consumed
#: by mass-splitting copy states). Picked to cover the soft_body_drop
#: steady-contact scene (~64 inner launches naturally needed) and leave
#: headroom for moderately denser graphs without paying for the long
#: tail of stragglers that drives the per-step coloring cost up to
#: 52 % of GPU time on res=3 scenes.
MAX_GREEDY_OUTER_ITERS = wp.constant(16)
# All-ones; used in place of int64 unary NOT (see _TAG_MASK comment).
_FREE_COLOR_FLIP = wp.constant(wp.int64(-1))


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return ((int)__ffsll((long long)mask)) - 1;
#else
// CPU fallback: linear scan.
if (mask == 0) return -1;
int p = 0;
while ((mask & 1LL) == 0LL) { mask >>= 1; p++; }
return p;
#endif
""")
def _lowest_set_bit(mask: wp.int64) -> wp.int32:
    """Lowest set bit position in ``mask`` (0-indexed), -1 if zero. CUDA: __ffsll."""
    ...


@wp.kernel(enable_backward=False)
def partitioning_coloring_incremental_greedy_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    total_num_threads: wp.int32,
    num_remaining: wp.array[int],
    overflow_flag: wp.array[int],
    max_colored_partitions: wp.int32,
):
    """JP-MIS with greedy colour selection. Vertex commits iff highest
    priority among uncoloured neighbours and gets the smallest colour
    not used by coloured neighbours. Typically 2-3x fewer colours than
    vanilla JP. Forbidden mask is int64 (max GREEDY_MAX_COLORS=64
    colours); saturation routes to overflow (``max_colored_partitions
    >= 0``, mass splitting) or sets ``overflow_flag`` for JP fallback.
    """
    n = num_elements[0]
    soft_cap = max_colored_partitions >= wp.int32(0)
    saturation_limit = max_colored_partitions if soft_cap else wp.int32(GREEDY_MAX_COLORS)

    for tid in range(wp.tid(), n, total_num_threads):
        # 4-byte tag read (0 = uncoloured) — halves bandwidth vs int64.
        if color_tags[tid] != wp.int32(0):
            continue

        self_prio = contact_partitions_get_random_value(packed_priorities, tid)
        el = elements[tid]

        is_local_max = bool(True)
        forbidden_mask = wp.int64(0)

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
                # Read both color tag AND priority unconditionally so the
                # compiler can issue both loads up-front (hides global-
                # memory latency for the conditional path below). Cost is
                # one extra read for already-coloured neighbours; the
                # array is small enough to live in cache.
                ntag = color_tags[neighbor]
                neigh_prio = contact_partitions_get_random_value(packed_priorities, neighbor)
                if ntag == wp.int32(0):
                    # Uncoloured neighbour -> MIS priority tiebreak.
                    if neigh_prio > self_prio:
                        is_local_max = False
                        break
                else:
                    # Coloured neighbour -> forbid its colour.
                    ncolor = ntag - wp.int32(1)
                    if ncolor < GREEDY_MAX_COLORS:
                        forbidden_mask = forbidden_mask | (wp.int64(1) << wp.int64(ncolor))

        if is_local_max:
            # Smallest free colour = lowest 0-bit in forbidden mask.
            free_mask = forbidden_mask ^ _FREE_COLOR_FLIP
            c = _lowest_set_bit(free_mask)
            # Saturation routing:
            #   soft_cap on  -> route to overflow colour ``max_colored_partitions``
            #   soft_cap off -> route to ``GREEDY_MAX_COLORS - 1`` and flag overflow
            # Both paths still commit + decrement so capture_while exits.
            if c < wp.int32(0) or c >= saturation_limit:
                if soft_cap:
                    c = max_colored_partitions
                else:
                    overflow_flag[0] = 1
                    c = wp.int32(GREEDY_MAX_COLORS - wp.int32(1))
            color_tags[tid] = c + wp.int32(1)
            partition_data_concat[tid] = (wp.int64(c + wp.int32(1)) << _COLOR_SHIFT) | wp.int64(tid)
            wp.atomic_sub(num_remaining, 0, 1)


# ---------------------------------------------------------------------------
# Speculative greedy coloring (Çatalyürek-style, deterministic).
#
# Each round:
#   pick    : uncoloured tids pick smallest free colour given coloured nbrs
#   validate+commit : abort if any uncoloured nbr with higher priority picked
#                     the same colour; else stamp color_tags + atomic_sub.
#
# Vs strict JP-MIS this commits at multiple colours per round, so dense
# graphs (Kapla 67k constraints) drain in ~6-10 rounds x 2 kernels instead
# of ~80 single-kernel MIS rounds.
#
# Determinism: same fixed priority permutation as JP-MIS. The race between
# pick (round R) and validate+commit (round R) is benign under the priority
# tiebreak -- the lower-priority side always aborts regardless of read order.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def speculative_pick_kernel(
    color_tags: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    total_num_threads: wp.int32,
    num_remaining: wp.array[int],
    overflow_flag: wp.array[int],
    max_colored_partitions: wp.int32,
    tentative_color: wp.array[wp.int32],
    commit_decision: wp.array[wp.int32],
):
    """Phase 1: each uncoloured tid picks smallest free colour given
    coloured neighbours. Writes ``tentative_color[tid]`` (encoded
    ``c+1``) and clears ``commit_decision[tid]``.

    Sets ``overflow_flag[0]=1`` on saturation (>= GREEDY_MAX_COLORS
    distinct nbr colours) so the host wrapper triggers JP fallback --
    same contract as ``partitioning_coloring_incremental_greedy_kernel``.
    """
    if num_remaining[0] <= wp.int32(0):
        return
    n = num_elements[0]
    soft_cap = max_colored_partitions >= wp.int32(0)
    saturation_limit = max_colored_partitions if soft_cap else wp.int32(GREEDY_MAX_COLORS)

    for tid in range(wp.tid(), n, total_num_threads):
        # Reset commit decision for this round; coloured tids skip the pick.
        commit_decision[tid] = wp.int32(0)
        if color_tags[tid] != wp.int32(0):
            continue
        el = elements[tid]
        forbidden_mask = wp.int64(0)
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
                if neighbor == tid:
                    continue
                ntag = color_tags[neighbor]
                if ntag != wp.int32(0):
                    ncolor = ntag - wp.int32(1)
                    if ncolor < GREEDY_MAX_COLORS:
                        forbidden_mask = forbidden_mask | (wp.int64(1) << wp.int64(ncolor))
        free_mask = forbidden_mask ^ _FREE_COLOR_FLIP
        c = _lowest_set_bit(free_mask)
        if c < wp.int32(0) or c >= saturation_limit:
            if soft_cap:
                c = max_colored_partitions
            else:
                overflow_flag[0] = wp.int32(1)
                c = wp.int32(GREEDY_MAX_COLORS - wp.int32(1))
        tentative_color[tid] = c + wp.int32(1)


@wp.kernel(enable_backward=False)
def speculative_validate_commit_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
    tentative_color: wp.array[wp.int32],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[int],
    total_num_threads: wp.int32,
    num_remaining: wp.array[int],
    max_colored_partitions: wp.int32,
):
    """Phase 2+3 (fused): validate the tentative pick and commit if
    no conflict. Race-safe under priority tiebreak: when A and B
    share a body and both pick colour ``c``, the lower-priority side
    always aborts whether it reads A as "uncoloured + higher prio"
    (pre-commit) or "already coloured at my tentative" (post-commit).

    The overflow colour (``c == max_colored_partitions`` when
    ``max_colored_partitions >= 0``) is exempt from the neighbour
    conflict check: mass splitting resolves within-colour conflicts in
    that bucket via per-(body, partition) copy states, so multiple
    neighbouring elements are allowed to share it. Without this exemption
    the speculative path deadlocks on dense scenes that spill every
    element to the overflow bucket -- the first iteration colours one
    element per neighbourhood, then subsequent iterations see the
    coloured overflow neighbour and reject ``commit`` forever
    (``ntag == my_tent`` on every neighbour).
    """
    if num_remaining[0] <= wp.int32(0):
        return
    n = num_elements[0]
    # Encoded overflow tag (``c+1`` in the tentative_color encoding) when
    # the soft cap is configured; ``-1`` disables the exemption when
    # ``max_colored_partitions < 0`` (no overflow bucket).
    overflow_tent = wp.where(max_colored_partitions >= wp.int32(0), max_colored_partitions + wp.int32(1), wp.int32(-1))
    for tid in range(wp.tid(), n, total_num_threads):
        if color_tags[tid] != wp.int32(0):
            continue
        my_tent = tentative_color[tid]
        my_prio = contact_partitions_get_random_value(packed_priorities, tid)
        commit = bool(True)
        # Overflow picks skip neighbour validation: mass splitting
        # handles intra-bucket conflicts.
        is_overflow_pick = my_tent == overflow_tent
        if not is_overflow_pick:
            el = elements[tid]
            for j in range(MAX_BODIES):
                if not commit:
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
                    ntag = color_tags[neighbor]
                    if ntag != wp.int32(0):
                        if ntag == my_tent:
                            commit = False
                            break
                        continue
                    if tentative_color[neighbor] != my_tent:
                        continue
                    neigh_prio = contact_partitions_get_random_value(packed_priorities, neighbor)
                    if neigh_prio > my_prio:
                        commit = False
                        break
        if commit:
            color_tags[tid] = my_tent
            partition_data_concat[tid] = (wp.int64(my_tent) << _COLOR_SHIFT) | wp.int64(tid)
            wp.atomic_sub(num_remaining, 0, 1)


@wp.kernel(enable_backward=False)
def speculative_overflow_exit_kernel(
    overflow_flag: wp.array[int],
    num_remaining: wp.array[int],
):
    """Force the speculative ``capture_while`` to exit on saturation
    (>= GREEDY_MAX_COLORS distinct nbr colours). Otherwise a saturated
    constraint whose only valid colour is taken by a coloured nbr
    would loop forever. Zeroing num_remaining surfaces saturation to
    the host wrapper which runs JP fallback."""
    if overflow_flag[0] != wp.int32(0):
        num_remaining[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def greedy_overflow_spill_kernel(
    color_tags: wp.array[wp.int32],
    partition_data_concat: wp.array[wp.int64],
    num_elements: wp.array[wp.int32],
    max_colored_partitions: wp.int32,
):
    """Assign every still-uncoloured element to the overflow colour
    ``max_colored_partitions``. Run AFTER the greedy ``capture_while``
    exits; no-op when capture_while drained num_remaining naturally
    (everything already coloured).

    The overflow colour is consumed by mass splitting's copy-state
    machinery, so spilling here gives correct physics with potentially
    more colours than the optimal MIS would have found -- the tradeoff
    that lets us cap the outer iteration loop on dense graphs."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if color_tags[tid] != wp.int32(0):
        return
    # Uncoloured: stamp the overflow colour. Skip ``num_remaining`` here
    # because the iter-cap watcher already zeroed it; downstream
    # histogram / scatter only reads ``partition_data_concat``.
    color_tags[tid] = max_colored_partitions + wp.int32(1)
    partition_data_concat[tid] = (wp.int64(max_colored_partitions + wp.int32(1)) << _COLOR_SHIFT) | wp.int64(tid)


@wp.kernel(enable_backward=False)
def partitioning_coloring_incremental_kernel(
    partition_data_concat: wp.array[wp.int64],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[int],
    vertex_to_adjacent_elements: wp.array[int],
    elements: wp.array[ElementInteractionData],
    remaining_ids: wp.array[int],
    num_remaining: wp.array[int],
    color_arr: wp.array[int],
):
    """JP MIS pass driven from ``remaining_ids[0..num_remaining[0])`` (compact);
    full-warp utilisation. Neighbours still filtered via is_removed."""
    slot = wp.tid()

    if slot >= num_remaining[0]:
        return

    tid = remaining_ids[slot]
    color_copy = color_arr[0]

    # Elements in the compact list are -- by construction -- not yet
    # assigned, so the outer ``is_removed`` self-check that the classic
    # kernel runs is redundant here and is skipped.

    is_local_max = bool(True)

    self_prio = contact_partitions_get_random_value(packed_priorities, tid)
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
            if contact_partitions_get_random_value(packed_priorities, neighbor) > self_prio:
                is_local_max = False
                break

    if is_local_max:
        partition_data_concat[tid] = (wp.int64(color_copy + 1) << _COLOR_SHIFT) | wp.int64(tid)


# Incremental-path helper kernels.


@wp.kernel(enable_backward=False)
def incremental_init_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
    num_elements: wp.array[int],
):
    """Single-thread loop-state init."""
    current_color[0] = 0
    num_remaining[0] = num_elements[0]
    partition_count[0] = 0


@wp.kernel(enable_backward=False)
def incremental_init_csr_kernel(
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    num_colors: wp.array[int],
    color_starts: wp.array[int],
    num_elements: wp.array[int],
):
    """Single-thread CSR coloring state init."""
    current_color[0] = 0
    num_remaining[0] = num_elements[0]
    num_colors[0] = 0
    color_starts[0] = 0


@wp.kernel(enable_backward=False)
def warm_start_periodic_invalidate_kernel(
    invalidate_counter: wp.array[int],
    cache_num_entries: wp.array[int],
    num_colors: wp.array[int],
    skip_color_start_plus_one: wp.array[int],
    skip_color_end_plus_one: wp.array[int],
    invalidate_period: wp.int32,
    rotate_skip_width: wp.int32,
):
    """Single-thread pre-pass for warm-start cache stir.

    Two strategies for breaking the colouring lock-in (both may be
    enabled together; periodic-invalidate wins on its trigger step):

    * Periodic invalidate (``invalidate_period > 0``): zero the cache
      every Nth call -> next seed is a no-op -> MIS rebuilds cold.
    * Rotating skip-range (``rotate_skip_width > 0``): pick ``width``
      consecutive cached colours per step (round-robin via counter)
      whose seeded entries the seed kernel drops -> those colours
      re-MIS while the rest stay warm-started. Cost is linear in
      ``width``.
    """
    c = invalidate_counter[0] + wp.int32(1)
    invalidate_counter[0] = c

    # Default: no skip (start > end disables the range check in seed).
    skip_color_start_plus_one[0] = wp.int32(1)
    skip_color_end_plus_one[0] = wp.int32(0)

    if invalidate_period > wp.int32(0) and c % invalidate_period == wp.int32(0):
        cache_num_entries[0] = wp.int32(0)
        return

    if rotate_skip_width > wp.int32(0):
        nc = num_colors[0]
        if nc > wp.int32(0):
            # Skip cached colours ``[start, start+width)`` mod nc.
            # Encoded as +1 to match ``color_tags``.
            start = (c * rotate_skip_width) % nc
            width = rotate_skip_width
            if width > nc:
                width = nc
            skip_color_start_plus_one[0] = start + wp.int32(1)
            skip_color_end_plus_one[0] = start + width


@wp.kernel(enable_backward=False)
def incremental_begin_sweep_kernel(
    num_colors: wp.array[int],
    color_cursor: wp.array[int],
    sweep_direction: wp.array[int],
    advance_direction: wp.int32,
):
    """Copy num_colors into color_cursor for the sweep-time
    capture_while. When ``advance_direction != 0`` also toggles
    ``sweep_direction[0]`` between 0 and 1 (symmetric Gauss-Seidel
    pattern: every other sweep walks the colour list in reverse).
    """
    color_cursor[0] = num_colors[0]
    if advance_direction != wp.int32(0):
        sweep_direction[0] = wp.int32(1) - sweep_direction[0]


@wp.kernel(enable_backward=False)
def incremental_fill_minus_one_kernel(
    arr: wp.array[int],
    num_elements: wp.array[int],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    arr[tid] = -1


@wp.kernel(enable_backward=False)
def incremental_init_remaining_ids_kernel(
    remaining_ids: wp.array[int],
    num_elements: wp.array[int],
):
    """Init compact remaining-ids: ``remaining_ids[i] = i``."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    remaining_ids[tid] = tid


@wp.kernel(enable_backward=False)
def incremental_reset_loop_state_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_tags: wp.array[wp.int32],
    interaction_id_to_partition: wp.array[int],
    num_elements: wp.array[int],
):
    """Reset per-element partitioner state without touching adjacency.
    Skipped between PGS iterations since the constraint set is unchanged."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    partition_data_concat[tid] = _UNPARTITIONED | wp.int64(tid)
    color_tags[tid] = wp.int32(0)
    interaction_id_to_partition[tid] = -1


@wp.kernel(enable_backward=False)
def set_int_array_kernel(arr: wp.array[int], value: int):
    arr[0] = value


# Single-block tile-scan kernel block size. 1024 = largest reliable Warp block.
TILE_SCAN_BLOCK_DIM = wp.constant(1024)


@wp.kernel(enable_backward=False)
def incremental_tile_compact_remaining_and_advance_kernel(
    partition_data_concat: wp.array[wp.int64],
    remaining_ids: wp.array[int],
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    partition_count: wp.array[int],
    partition_element_ids: wp.array[int],
    interaction_id_to_partition: wp.array[int],
):
    """Fused compact + remaining-list update + colour advance, single-block
    tile-scan. In-place compacts remaining_ids by dropping committed elements;
    survivors pack leftward (write-index <= read-index, safe in-place)."""
    _block, lane = wp.tid()

    n = num_remaining[0]
    cc = current_color[0]

    # Running exclusive-prefix counts (tile_sum at each tile end).
    committed_running = int(0)
    survivor_running = int(0)

    offset = int(0)
    while offset < n:
        slot = offset + lane

        eid = int(0)
        committed_flag = int(0)
        survivor_flag = int(0)
        if slot < n:
            eid = remaining_ids[slot]
            tagged = partition_data_concat[eid] & _TAG_MASK
            color_plus_one = tagged >> _COLOR_SHIFT
            if color_plus_one == wp.int64(cc + 1):
                committed_flag = 1
            else:
                survivor_flag = 1

        # Block-wide tile scans (implicit sync; safe in-place compaction).
        committed_tile = wp.tile(committed_flag)
        committed_scan = wp.tile_scan_exclusive(committed_tile)
        committed_local = wp.untile(committed_scan)

        survivor_tile = wp.tile(survivor_flag)
        survivor_scan = wp.tile_scan_exclusive(survivor_tile)
        survivor_local = wp.untile(survivor_scan)

        if committed_flag == 1:
            out_idx = committed_running + committed_local
            partition_element_ids[out_idx] = eid
            interaction_id_to_partition[eid] = cc

        if survivor_flag == 1:
            out_idx = survivor_running + survivor_local
            remaining_ids[out_idx] = eid

        committed_total = wp.tile_sum(committed_tile)
        survivor_total = wp.tile_sum(survivor_tile)
        committed_running = committed_running + committed_total[0]
        survivor_running = survivor_running + survivor_total[0]
        offset = offset + TILE_SCAN_BLOCK_DIM

    # Publish totals and advance current_color (race-free; all lanes done).
    if lane == 0:
        partition_count[0] = committed_running
        num_remaining[0] = survivor_running
        current_color[0] = cc + 1


@wp.kernel(enable_backward=False)
def incremental_tile_compact_csr_and_advance_kernel(
    partition_data_concat: wp.array[wp.int64],
    remaining_ids: wp.array[int],
    current_color: wp.array[int],
    num_remaining: wp.array[int],
    num_colors: wp.array[int],
    element_ids_by_color: wp.array[int],
    color_starts: wp.array[int],
    interaction_id_to_partition: wp.array[int],
    max_colors: int,
    overflow_flag: wp.array[int],
    max_colored_partitions: wp.int32,
):
    """CSR variant of :func:`incremental_tile_compact_remaining_and_advance_kernel`.

    Same per-tile algorithm but committed elements are appended to
    the CSR layout at ``element_ids_by_color[color_starts[cc]..]``
    instead of a one-shot buffer. Lane 0 additionally writes
    ``color_starts[cc + 1]`` (so the next colour -- and the trailing
    ``num_colors`` sentinel slot -- starts at the right offset) and
    mirrors ``current_color`` into ``num_colors`` for the outer
    sweep-time ``capture_while``.

    ``max_colored_partitions``:

    * ``< 0`` -- disabled. Reaching ``cc == max_colors`` raises
      ``overflow_flag``.
    * ``>= 0`` -- soft cap. When ``cc == max_colored_partitions``,
      every still-uncoloured element in ``remaining_ids[0..n)`` gets
      stamped with colour ``max_colored_partitions`` (the overflow
      bucket) in one tile-scan pass, bypassing the MIS-correctness
      requirement that colours 0..K-1 obey. Mass splitting consumes
      the bucket and resolves the within-colour conflicts via copy
      states.
    """
    _block, lane = wp.tid()

    n = num_remaining[0]
    # Early-exit contract: tolerate extra launches past convergence so
    # the host side can safely unroll the capture-while body
    # ``NUM_INNER_WHILE_ITERATIONS`` times. When ``num_remaining == 0``
    # no element is still uncoloured and every per-colour counter
    # (``current_color``, ``num_colors``, ``color_starts``) has already
    # been written to its final value by the round that drained the
    # list. Returning without any writes keeps those counters stable
    # so the surplus launches are true no-ops.
    if n == 0:
        return
    cc = current_color[0]
    # Soft cap: when we hit the overflow bucket colour, dump every
    # uncoloured remaining element into it in one pass. This bypasses
    # the MIS-correctness requirement (colours 0..K-1) — mass splitting
    # resolves the within-bucket conflicts on copy states.
    if max_colored_partitions >= wp.int32(0) and cc >= max_colored_partitions:
        base = color_starts[max_colored_partitions]
        offset = int(0)
        while offset < n:
            slot = offset + lane
            if slot < n:
                eid = remaining_ids[slot]
                element_ids_by_color[base + slot] = eid
                interaction_id_to_partition[eid] = max_colored_partitions
                partition_data_concat[eid] = (
                    wp.int64(max_colored_partitions + wp.int32(1)) << _COLOR_SHIFT
                ) | wp.int64(eid)
            offset = offset + TILE_SCAN_BLOCK_DIM
        if lane == 0:
            color_starts[max_colored_partitions + wp.int32(1)] = base + n
            num_remaining[0] = 0
            current_color[0] = max_colored_partitions + wp.int32(1)
            num_colors[0] = max_colored_partitions + wp.int32(1)
        return
    # Overflow guard. ``color_starts`` is sized ``max_colors + 1`` so
    # the largest writable end-offset slot is ``color_starts[max_colors]``
    # -- i.e. valid ``cc`` values are ``0 .. max_colors - 1``. If the
    # coloring exhausts the budget, raise an overflow flag, force the
    # outer ``capture_while`` to terminate by zeroing ``num_remaining``,
    # and early-return before any buffer writes. The host side of
    # ``build_csr`` reads the flag after the capture_while exits and
    # raises a descriptive error so this cannot silently corrupt memory.
    if cc >= max_colors:
        if lane == 0:
            overflow_flag[0] = 1
            num_remaining[0] = 0
        return
    # Base offset into the CSR buffer for this colour. Every lane
    # captures it into a register up front so the later survivor scan's
    # implicit block sync double-duties as a read barrier for it.
    base = color_starts[cc]

    committed_running = int(0)
    survivor_running = int(0)

    offset = int(0)
    while offset < n:
        slot = offset + lane

        eid = int(0)
        committed_flag = int(0)
        survivor_flag = int(0)
        if slot < n:
            eid = remaining_ids[slot]
            tagged = partition_data_concat[eid] & _TAG_MASK
            color_plus_one = tagged >> _COLOR_SHIFT
            if color_plus_one == wp.int64(cc + 1):
                committed_flag = 1
            else:
                survivor_flag = 1

        committed_tile = wp.tile(committed_flag)
        committed_scan = wp.tile_scan_exclusive(committed_tile)
        committed_local = wp.untile(committed_scan)

        survivor_tile = wp.tile(survivor_flag)
        survivor_scan = wp.tile_scan_exclusive(survivor_tile)
        survivor_local = wp.untile(survivor_scan)

        if committed_flag == 1:
            out_idx = base + committed_running + committed_local
            element_ids_by_color[out_idx] = eid
            interaction_id_to_partition[eid] = cc

        if survivor_flag == 1:
            out_idx = survivor_running + survivor_local
            remaining_ids[out_idx] = eid

        committed_total = wp.tile_sum(committed_tile)
        survivor_total = wp.tile_sum(survivor_tile)
        committed_running = committed_running + committed_total[0]
        survivor_running = survivor_running + survivor_total[0]
        offset = offset + TILE_SCAN_BLOCK_DIM

    if lane == 0:
        # color_starts[cc+1] = next colour's start offset.
        color_starts[cc + 1] = base + committed_running
        num_remaining[0] = survivor_running
        current_color[0] = cc + 1
        # num_colors converges to final count when capture_while exits.
        num_colors[0] = cc + 1


# Greedy-mode helpers. Greedy assigns smallest unused colour, so commits within
# a round can land in any colour. Replaces the colour=round compaction with
# histogram -> scan -> deterministic scatter.


@wp.kernel(enable_backward=False)
def greedy_clear_int_kernel(arr: wp.array[int]):
    """Single-thread zero of ``arr[0]`` (used to clear fallback-decision flag)."""
    if wp.tid() == 0:
        arr[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def greedy_reset_init_kernel(
    overflow_flag: wp.array[int],
    color_count: wp.array[int],
    color_offsets: wp.array[int],
    max_colors: int,
):
    """Reset overflow flag, color_count, and color_offsets before each build."""
    tid = wp.tid()
    if tid < max_colors:
        color_count[tid] = wp.int32(0)
        color_offsets[tid] = wp.int32(0)
    if tid == 0:
        overflow_flag[0] = wp.int32(0)


@wp.kernel(enable_backward=False)
def greedy_color_histogram_kernel(
    partition_data_concat: wp.array[wp.int64],
    num_elements: wp.array[int],
    color_count: wp.array[int],
    interaction_id_to_partition: wp.array[int],
):
    """Per-colour histogram + per-element colour stamp. Reads
    partition_data_concat (not color_tags) so the JP-fallback path stays consistent."""
    tid = wp.tid()
    n = num_elements[0]
    if tid >= n:
        return
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(0):
        # Unreachable on a converged build; stamp -1 sentinel just in case.
        interaction_id_to_partition[tid] = wp.int32(-1)
        return
    c = wp.int32(color_plus_one - wp.int64(1))
    interaction_id_to_partition[tid] = c
    wp.atomic_add(color_count, c, 1)


@wp.kernel(enable_backward=False)
def greedy_count_and_scan_color_starts_kernel(
    color_count: wp.array[int],
    color_starts: wp.array[int],
    num_colors: wp.array[int],
    max_colors: int,
):
    """Single-thread exclusive scan over color_count + num_colors derivation.
    color_starts[c] = sum of color_count[0..c); num_colors = 1 + max c with count>0."""
    tid = wp.tid()
    if tid != 0:
        return
    running = wp.int32(0)
    last = wp.int32(-1)
    for c in range(max_colors + 1):
        color_starts[c] = running
        if c < max_colors:
            cnt = color_count[c]
            if cnt > 0:
                last = c
            running = running + cnt
    num_colors[0] = last + 1


@wp.kernel(enable_backward=False)
def greedy_scatter_elements_by_color_kernel(
    partition_data_concat: wp.array[wp.int64],
    color_starts: wp.array[int],
    color_offsets: wp.array[int],
    element_ids_by_color: wp.array[int],
    num_elements: wp.array[int],
):
    """Scatter each coloured element into its CSR slot:
    element_ids_by_color[color_starts[c] + atomic_add(color_offsets, c, 1)] = tid.
    Within-colour order is non-deterministic but irrelevant (PGS treats slice as
    independent set)."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    tagged = partition_data_concat[tid] & _TAG_MASK
    color_plus_one = tagged >> _COLOR_SHIFT
    if color_plus_one == wp.int64(0):
        return
    c = wp.int32(color_plus_one - wp.int64(1))
    slot = wp.atomic_add(color_offsets, c, 1)
    element_ids_by_color[color_starts[c] + slot] = tid
