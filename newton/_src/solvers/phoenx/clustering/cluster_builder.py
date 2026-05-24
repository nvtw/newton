# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parallel greedy K-neighbour clustering on the constraint adjacency graph.

Adapts Algorithm 4 of Ton-That, Kry & Andrews (2022),
*Parallel Block Neo-Hookean XPBD using Graph Clustering*, replacing the
paper's sequential BFS with a parallel MIS-based claim that is
deterministic by construction and graph-capture safe.

Inputs match the existing graph coloring partitioners: an
``ElementInteractionData`` buffer plus a device-side active count.
Outputs are a per-cluster ``vec4i`` of constraint ids (``-1`` for unused
slots), a dense ``num_clusters`` counter, and a per-element
``element_to_cluster`` map.

Determinism: seed selection is an MIS over a fixed permutation priority
(the same scheme as the graph coloring partitioner). Race resolution on
contested 2-hop neighbours uses ``atomic_min`` on the seed's constraint
id -- lower-id seed always wins regardless of thread schedule. The
``vec4i`` slot order is established by a second ``atomic_min`` pass
that bubble-sorts member ids ascending.

Graph-capture safety: every kernel launch uses a fixed dimension
(``max_num_interactions`` or ``max_num_nodes``); active-count gating
is via ``tid >= num_elements[0]``. The outer ``Ks_target`` schedule is a
static host-side loop (4 iterations); the inner round loop uses
``wp.capture_while`` with a hard upper bound.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.adjacency import ElementVertexAdjacency
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_get,
    vec8i,
)

__all__ = ["ConstraintClusterBuilder"]


#: Hard-coded cluster size cap. Picked so cluster members fit a single
#: ``vec4i``; unused slots are written as -1 in the output.
MAX_CLUSTER_SIZE = wp.constant(wp.int32(4))

#: Hard cap on the union of bodies referenced by all constraints in a
#: cluster. Caller-imposed: downstream block solves want the per-cluster
#: node footprint bounded so per-cluster scratch can be tile-allocated.
#: The cap also matches the widened ``ElementInteractionData`` slot
#: count (``MAX_BODIES`` = 8), so a cluster's body union packs into the
#: same struct used for individual constraints -- useful for feeding
#: clusters back through the existing graph coloring partitioner as
#: supernodal elements (downstream work).
MAX_BODIES_PER_CLUSTER = wp.constant(wp.int32(8))

#: "Empty" sentinel for the internal cluster-id arrays. INT32_MAX is the
#: smallest value larger than any valid constraint id, so ``atomic_min``
#: cleanly resolves "lower seed id wins" without -1 being treated as
#: smaller than all positive ids.
_EMPTY = wp.constant(wp.int32(0x7FFFFFFF))

#: Inner-round unroll factor. Each unrolled iteration is 5 kernel
#: launches (propose / grow / validate / commit / clear_pending).
_INNER_UNROLL = 4

#: Hard cap on host-unrolled rounds per Ks_target. ``_INNER_UNROLL *
#: _MAX_OUTER_ITERS`` rounds in total. K=4 cluster size + 1-hop MIS
#: means a typical tet-mesh constraint graph drains in well under 16
#: rounds; this gives headroom for adversarial cases. The loop is
#: unrolled on the host (no ``wp.capture_while``); converged rounds are
#: cheap because every kernel early-exits on committed elements.
_MAX_OUTER_ITERS = wp.constant(wp.int32(8))


# --- Init kernels ------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _cluster_init_state_kernel(
    num_elements: wp.array[wp.int32],
    # out
    cluster_id_of: wp.array[wp.int32],
    pending_id_of: wp.array[wp.int32],
    seed_committed: wp.array[wp.int32],
    element_to_cluster: wp.array[wp.int32],
    num_remaining: wp.array[wp.int32],
):
    """Seed everything to ``_EMPTY`` and prime ``num_remaining``."""
    tid = wp.tid()
    n = num_elements[0]
    if tid == 0:
        num_remaining[0] = n
    if tid >= cluster_id_of.shape[0]:
        return
    cluster_id_of[tid] = _EMPTY
    pending_id_of[tid] = _EMPTY
    seed_committed[tid] = wp.int32(0)
    element_to_cluster[tid] = wp.int32(-1)


@wp.kernel(enable_backward=False)
def _cluster_init_members_flat_kernel(
    cluster_members_flat: wp.array[wp.int32],
):
    """Init the member-id atomic_min targets to ``_EMPTY``."""
    tid = wp.tid()
    cluster_members_flat[tid] = _EMPTY


@wp.kernel(enable_backward=False)
def _cluster_init_num_clusters_kernel(
    num_clusters: wp.array[wp.int32],
    cluster_dense_id: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid == 0:
        num_clusters[0] = wp.int32(0)
    if tid < cluster_dense_id.shape[0]:
        cluster_dense_id[tid] = wp.int32(-1)


@wp.kernel(enable_backward=False)
def _cluster_set_ks_kernel(
    ks_value: wp.int32,
    # out
    ks_target: wp.array[wp.int32],
):
    if wp.tid() == 0:
        ks_target[0] = ks_value


@wp.kernel(enable_backward=False)
def _cluster_set_progress_kernel(
    value: wp.int32,
    progress_flag: wp.array[wp.int32],
):
    if wp.tid() == 0:
        progress_flag[0] = value


# --- Inner-round kernels -----------------------------------------------------


@wp.func
def _is_uncommitted(cluster_id_of: wp.array[wp.int32], i: wp.int32) -> wp.bool:
    return cluster_id_of[i] == _EMPTY


@wp.func
def _body_set_contains(body_set: vec8i, body: wp.int32) -> wp.bool:
    """Return True iff ``body`` appears in ``body_set``. ``-1`` slots
    end the meaningful prefix."""
    for s in range(int(MAX_BODIES_PER_CLUSTER)):
        v = body_set[s]
        if v < 0:
            return False
        if v == body:
            return True
    return False


@wp.func
def _body_overlap_with_seed(seed_set: vec8i, neigh_el: ElementInteractionData) -> wp.int32:
    """Count how many of ``neigh_el``'s bodies appear in ``seed_set``.
    Higher overlap means fewer new bodies are added when admitting the
    neighbour into the cluster -- exactly what minimises the chance of
    hitting the body cap. Used by validate to pick the top-3 admission
    candidates."""
    cnt = wp.int32(0)
    for b_idx in range(MAX_BODIES):
        b = element_interaction_data_get(neigh_el, b_idx)
        if b < 0:
            break
        if _body_set_contains(seed_set, b):
            cnt += wp.int32(1)
    return cnt


@wp.func
def _body_set_count_new(body_set: vec8i, neigh_el: ElementInteractionData) -> wp.int32:
    """How many of ``neigh_el``'s bodies are NOT already in ``body_set``."""
    new_count = wp.int32(0)
    for b_idx in range(MAX_BODIES):
        b = element_interaction_data_get(neigh_el, b_idx)
        if b < 0:
            break
        if not _body_set_contains(body_set, b):
            new_count += wp.int32(1)
    return new_count


@wp.func
def _body_set_add_all(body_set: vec8i, body_count: wp.int32, neigh_el: ElementInteractionData) -> vec8i:
    """Append ``neigh_el``'s not-yet-present bodies into ``body_set``.
    Caller is responsible for ensuring ``body_count + new_bodies <=
    MAX_BODIES_PER_CLUSTER`` BEFORE calling (this function does no
    overflow check)."""
    cnt = body_count
    for b_idx in range(MAX_BODIES):
        b = element_interaction_data_get(neigh_el, b_idx)
        if b < 0:
            break
        if not _body_set_contains(body_set, b):
            body_set[cnt] = b
            cnt += wp.int32(1)
    return body_set


@wp.func
def _body_set_init_from(seed_el: ElementInteractionData) -> vec8i:
    """Initialize a body-union scratch from the seed's own bodies."""
    bs = vec8i(-1, -1, -1, -1, -1, -1, -1, -1)
    n = wp.int32(0)
    for i in range(MAX_BODIES):
        b = element_interaction_data_get(seed_el, i)
        if b < 0:
            break
        bs[n] = b
        n += wp.int32(1)
    return bs


@wp.func
def _body_set_size_from(seed_el: ElementInteractionData) -> wp.int32:
    """Number of valid bodies in the seed element."""
    n = wp.int32(0)
    for i in range(MAX_BODIES):
        if element_interaction_data_get(seed_el, i) < 0:
            break
        n += wp.int32(1)
    return n


@wp.kernel(enable_backward=False)
def _cluster_propose_seeds_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    packed_priorities: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[wp.int32],
    vertex_to_adjacent_elements: wp.array[wp.int32],
    cluster_id_of: wp.array[wp.int32],
    ks_target: wp.array[wp.int32],
    # out
    pending_id_of: wp.array[wp.int32],
):
    """Each uncommitted element checks 1-hop uncommitted neighbours.
    If it strictly outranks them all in ``packed_priorities`` it becomes
    a tentative seed -- ``pending_id_of[tid] = tid``.

    Pure 1-hop MIS: tentative seeds are body-disjoint (so their direct
    1-hop neighbour sets are disjoint), but two seeds may still share a
    2-hop neighbour candidate. Grow phase resolves that contention via
    ``atomic_min`` on seed id.

    Special-case: at ``ks_target == 1`` every uncommitted element
    auto-becomes a singleton seed in a single round (no MIS needed --
    nothing to claim). That makes the mop-up pass converge in one body
    invocation regardless of graph density.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if not _is_uncommitted(cluster_id_of, tid):
        return

    if ks_target[0] <= wp.int32(1):
        pending_id_of[tid] = tid
        return

    self_prio = packed_priorities[tid]
    el = elements[tid]
    is_local_max = bool(True)

    for j in range(MAX_BODIES):
        if not is_local_max:
            break
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = wp.int32(0)
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if not _is_uncommitted(cluster_id_of, neighbor):
                continue
            if packed_priorities[neighbor] > self_prio:
                is_local_max = False
                break

    if is_local_max:
        # No race possible: only this thread writes pending_id_of[tid]
        # for the seed slot, and only when cluster_id_of[tid] == _EMPTY.
        pending_id_of[tid] = tid


@wp.kernel(enable_backward=False)
def _cluster_grow_seeds_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[wp.int32],
    vertex_to_adjacent_elements: wp.array[wp.int32],
    cluster_id_of: wp.array[wp.int32],
    ks_target: wp.array[wp.int32],
    # inout
    pending_id_of: wp.array[wp.int32],
):
    """For each fresh seed (``pending_id_of[tid] == tid``), scan 1-hop
    uncommitted neighbours and atomic-min-claim each one. NO per-seed
    early termination and NO body-cap check here: those decisions are
    deferred to :func:`_cluster_validate_seeds_kernel` so they can be
    made deterministically based on the FINAL settled state of
    ``pending_id_of``.

    Why deferred: any check that depends on the per-thread sequence of
    atomic_min OUTCOMES (e.g. updating a running body_set on each
    "tid < prev" win) is order-dependent across thread schedules. A
    claim that a seed initially wins can be stolen by a smaller-id seed
    later in the same kernel launch, leaving the per-thread state stale.
    The validate kernel instead reads only the post-settled
    ``pending_id_of`` -- that read is deterministic, and so are the
    K-1-smallest-by-id selection + body-cap filtering it performs.

    Type-agnostic: ``ElementInteractionData`` represents any constraint
    (contact, joint, soft-tet, cloth, ...) so clusters mix freely
    across constraint types.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if pending_id_of[tid] != tid:
        return
    if not _is_uncommitted(cluster_id_of, tid):
        return

    ks = ks_target[0]
    if ks <= wp.int32(1):
        # K=1 mop-up: seed already counts as a singleton cluster.
        return

    el = elements[tid]
    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = wp.int32(0)
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if not _is_uncommitted(cluster_id_of, neighbor):
                continue
            # Skip a neighbour that's also a fresh seed (it owns itself).
            if pending_id_of[neighbor] == neighbor:
                continue
            wp.atomic_min(pending_id_of, neighbor, tid)


@wp.kernel(enable_backward=False)
def _cluster_validate_seeds_kernel(
    elements: wp.array[ElementInteractionData],
    num_elements: wp.array[wp.int32],
    adjacency_section_end_indices: wp.array[wp.int32],
    vertex_to_adjacent_elements: wp.array[wp.int32],
    cluster_id_of: wp.array[wp.int32],
    pending_id_of: wp.array[wp.int32],
    ks_target: wp.array[wp.int32],
    # out
    seed_committed: wp.array[wp.int32],
    num_remaining: wp.array[wp.int32],
    progress_flag: wp.array[wp.int32],
):
    """Decide which of the seed's tentatively-claimed neighbours to keep.

    The grow kernel claims neighbours promiscuously via ``atomic_min``;
    here we re-walk in deterministic CSR order, identify the ``ks-1``
    smallest-id confirmed claims, then admit them in id-sorted order
    subject to the body-union budget. Any claim NOT admitted (either
    because it has too high an id or because admitting it would push
    the cluster's body union past ``MAX_BODIES_PER_CLUSTER``) is
    released by writing ``pending_id_of[n] = _EMPTY``. The release is
    race-free: only the seed that owned ``n`` after grow visits ``n``
    in validate.

    Ks_target == 1: every fresh seed commits unconditionally (mop-up).
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    if pending_id_of[tid] != tid:
        return

    ks = ks_target[0]

    if ks <= wp.int32(1):
        # Singleton commit. No neighbour walk needed.
        seed_committed[tid] = wp.int32(1)
        wp.atomic_sub(num_remaining, 0, wp.int32(1))
        wp.atomic_add(progress_flag, 0, wp.int32(1))
        return

    # Pass 1: find the ks-1 = 3 confirmed neighbours with MAXIMUM body
    # overlap with the seed (tiebreak by lower neighbour id for
    # determinism). The 3 slots are kept sorted DESCENDING by
    # (overlap, -id) so cand0 is the best candidate to admit first.
    # This is the only departure from a faithful port of Algorithm 4:
    # the paper sorts by Jaccard distance globally, which is equivalent
    # to "more overlap = closer". We use overlap-with-seed since the
    # seed's body set is what's checked against the body cap; admitting
    # a high-overlap neighbour adds few new bodies, freeing room for
    # more admissions before the cap fires.
    el = elements[tid]
    seed_body_set = _body_set_init_from(el)

    cand0 = _EMPTY
    cand0_ov = wp.int32(-1)
    cand1 = _EMPTY
    cand1_ov = wp.int32(-1)
    cand2 = _EMPTY
    cand2_ov = wp.int32(-1)

    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = wp.int32(0)
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if not _is_uncommitted(cluster_id_of, neighbor):
                continue
            if pending_id_of[neighbor] != tid:
                continue
            if neighbor == cand0 or neighbor == cand1 or neighbor == cand2:
                continue
            ov = _body_overlap_with_seed(seed_body_set, elements[neighbor])
            # Comparator: candidate beats slot if higher overlap, or
            # equal overlap with smaller id (deterministic tiebreak).
            beats_0 = ov > cand0_ov or (ov == cand0_ov and neighbor < cand0)
            if beats_0:
                cand2 = cand1
                cand2_ov = cand1_ov
                cand1 = cand0
                cand1_ov = cand0_ov
                cand0 = neighbor
                cand0_ov = ov
                continue
            beats_1 = ov > cand1_ov or (ov == cand1_ov and neighbor < cand1)
            if beats_1:
                cand2 = cand1
                cand2_ov = cand1_ov
                cand1 = neighbor
                cand1_ov = ov
                continue
            beats_2 = ov > cand2_ov or (ov == cand2_ov and neighbor < cand2)
            if beats_2:
                cand2 = neighbor
                cand2_ov = ov

    # Pass 2: body-cap admit decisions in id-sorted order. Track which
    # of the three candidates were admitted so the release pass can
    # exempt them.
    body_set = _body_set_init_from(el)
    body_count = _body_set_size_from(el)
    admit0 = _EMPTY
    admit1 = _EMPTY
    admit2 = _EMPTY
    confirmed = wp.int32(1)  # the seed itself

    if cand0 != _EMPTY:
        n_el = elements[cand0]
        nb = _body_set_count_new(body_set, n_el)
        if body_count + nb <= wp.int32(int(MAX_BODIES_PER_CLUSTER)):
            body_set = _body_set_add_all(body_set, body_count, n_el)
            body_count += nb
            admit0 = cand0
            confirmed += wp.int32(1)
    if cand1 != _EMPTY and confirmed < ks:
        n_el = elements[cand1]
        nb = _body_set_count_new(body_set, n_el)
        if body_count + nb <= wp.int32(int(MAX_BODIES_PER_CLUSTER)):
            body_set = _body_set_add_all(body_set, body_count, n_el)
            body_count += nb
            admit1 = cand1
            confirmed += wp.int32(1)
    if cand2 != _EMPTY and confirmed < ks:
        n_el = elements[cand2]
        nb = _body_set_count_new(body_set, n_el)
        if body_count + nb <= wp.int32(int(MAX_BODIES_PER_CLUSTER)):
            body_set = _body_set_add_all(body_set, body_count, n_el)
            body_count += nb
            admit2 = cand2
            confirmed += wp.int32(1)

    # Pass 3: release every claim that isn't an admitted one. Includes
    # (a) the 4th+-smallest excess claims past ks-1 and (b) the candidates
    # we rejected for the body-cap. Race-free: no other seed writes to
    # ``pending_id_of[neighbor]`` here because ``pending_id_of[neighbor]
    # == tid`` means we own n in this round.
    for j in range(MAX_BODIES):
        v = element_interaction_data_get(el, j)
        if v < 0:
            break
        if v > 0:
            start = adjacency_section_end_indices[v - 1]
        else:
            start = wp.int32(0)
        end = adjacency_section_end_indices[v]
        for k in range(start, end):
            neighbor = vertex_to_adjacent_elements[k]
            if neighbor == tid:
                continue
            if pending_id_of[neighbor] != tid:
                continue
            if neighbor == admit0 or neighbor == admit1 or neighbor == admit2:
                continue
            pending_id_of[neighbor] = _EMPTY

    if confirmed >= ks:
        seed_committed[tid] = wp.int32(1)
        wp.atomic_sub(num_remaining, 0, confirmed)
        wp.atomic_add(progress_flag, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def _cluster_commit_or_release_kernel(
    num_elements: wp.array[wp.int32],
    seed_committed: wp.array[wp.int32],
    # inout
    cluster_id_of: wp.array[wp.int32],
    pending_id_of: wp.array[wp.int32],
):
    """Per element: if its tentative parent seed committed, finalize
    ``cluster_id_of[tid]`` to that seed id. Either way clear
    ``pending_id_of[tid]`` for the next round."""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    parent = pending_id_of[tid]
    if parent != _EMPTY:
        if seed_committed[parent] == wp.int32(1):
            cluster_id_of[tid] = parent
        # Whether the parent committed or not, this round's pending
        # assignment is now resolved.
        pending_id_of[tid] = _EMPTY


@wp.kernel(enable_backward=False)
def _cluster_reset_committed_flags_kernel(
    num_elements: wp.array[wp.int32],
    # inout
    seed_committed: wp.array[wp.int32],
):
    """Clear committed flags so the next round starts fresh. We can't
    leave them set -- a future seed proposal at a different tid would
    see a stale ``1`` if we ever reused the slot. (We don't, because
    once committed cluster_id_of[tid] != _EMPTY blocks reproposal, but
    keeping this defensive is cheap.)"""
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    seed_committed[tid] = wp.int32(0)


# --- Compaction --------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _cluster_mark_seeds_kernel(
    num_elements: wp.array[wp.int32],
    cluster_id_of: wp.array[wp.int32],
    # out
    is_seed_bitmap: wp.array[wp.int32],
):
    """``is_seed_bitmap[tid] = 1`` iff tid is its own cluster's seed.
    An element is a seed iff its committed cluster id equals itself."""
    tid = wp.tid()
    if tid >= cluster_id_of.shape[0]:
        return
    if tid >= num_elements[0]:
        is_seed_bitmap[tid] = wp.int32(0)
        return
    if cluster_id_of[tid] == tid:
        is_seed_bitmap[tid] = wp.int32(1)
    else:
        is_seed_bitmap[tid] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _cluster_assign_dense_ids_kernel(
    num_elements: wp.array[wp.int32],
    is_seed_bitmap_scanned: wp.array[wp.int32],  # exclusive prefix sum
    cluster_id_of: wp.array[wp.int32],
    # out
    cluster_dense_id: wp.array[wp.int32],
    num_clusters: wp.array[wp.int32],
):
    """Use the exclusive-scan bitmap to assign each seed a dense
    ``[0, num_clusters)`` id, and record the final count."""
    tid = wp.tid()
    n = num_elements[0]
    if tid >= cluster_dense_id.shape[0]:
        return
    if tid >= n:
        cluster_dense_id[tid] = wp.int32(-1)
        return
    if cluster_id_of[tid] == tid:
        dense = is_seed_bitmap_scanned[tid]
        cluster_dense_id[tid] = dense
        if tid == n - wp.int32(1):
            num_clusters[0] = dense + wp.int32(1)
    else:
        cluster_dense_id[tid] = wp.int32(-1)
        if tid == n - wp.int32(1):
            # Last element is not a seed: total = scan[last]. (Exclusive
            # scan with N elements emits scan[N-1] = sum of first N-1
            # flags, which equals total only if the last flag is 0.)
            num_clusters[0] = is_seed_bitmap_scanned[tid]


@wp.kernel(enable_backward=False)
def _cluster_fill_element_to_cluster_kernel(
    num_elements: wp.array[wp.int32],
    cluster_id_of: wp.array[wp.int32],
    cluster_dense_id: wp.array[wp.int32],
    # out
    element_to_cluster: wp.array[wp.int32],
):
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    cid = cluster_id_of[tid]
    if cid == _EMPTY:
        element_to_cluster[tid] = wp.int32(-1)
    else:
        element_to_cluster[tid] = cluster_dense_id[cid]


@wp.kernel(enable_backward=False)
def _cluster_emit_members_kernel(
    num_elements: wp.array[wp.int32],
    element_to_cluster: wp.array[wp.int32],
    # inout (init'd to _EMPTY)
    cluster_members_flat: wp.array[wp.int32],
):
    """Bubble-write tid into ``cluster_members_flat[c*4 .. c*4+3]`` via
    repeated ``atomic_min``. The four ints end up sorted ascending; the
    rest stay ``_EMPTY`` (mapped to -1 by the finalize kernel).

    Bubble semantics: at each slot, ``atomic_min(slot, value)`` returns
    the previous slot content. The MIN of (prev, value) stays in the
    slot; the MAX gets propagated to the next slot. Repeat until we
    fall off the end. If we fall off, our id was the largest in the
    cluster -- that only happens if more than ``MAX_CLUSTER_SIZE``
    elements map to the same cluster, which the K=4 cap prevents.
    """
    tid = wp.tid()
    if tid >= num_elements[0]:
        return
    c = element_to_cluster[tid]
    if c < 0:
        return

    base = c * wp.int32(int(MAX_CLUSTER_SIZE))
    value = tid
    for slot in range(int(MAX_CLUSTER_SIZE)):
        prev = wp.atomic_min(cluster_members_flat, base + slot, value)
        if value < prev:
            # We displaced ``prev`` -- propagate ``prev`` (the larger
            # of {value, prev}) to the next slot.
            value = prev
        # else: ``value >= prev``. Slot still holds ``prev``; our
        # ``value`` is now the larger and must keep bubbling.


@wp.kernel(enable_backward=False)
def _cluster_finalize_members_kernel(
    num_clusters: wp.array[wp.int32],
    cluster_members_flat: wp.array[wp.int32],
    # out
    cluster_members: wp.array[wp.vec4i],
):
    """Pack the flat int32 view into ``vec4i``, mapping ``_EMPTY`` -> -1."""
    tid = wp.tid()
    if tid >= cluster_members.shape[0]:
        return
    if tid >= num_clusters[0]:
        cluster_members[tid] = wp.vec4i(-1, -1, -1, -1)
        return
    base = tid * wp.int32(int(MAX_CLUSTER_SIZE))
    v = wp.vec4i(-1, -1, -1, -1)
    for slot in range(int(MAX_CLUSTER_SIZE)):
        raw = cluster_members_flat[base + slot]
        if raw != _EMPTY:
            v[slot] = raw
    cluster_members[tid] = v


# --- Builder -----------------------------------------------------------------


class ConstraintClusterBuilder:
    """Greedy K-neighbour clustering on the constraint adjacency graph.

    Outputs (populated after each :meth:`build_clusters` call):

    :ivar cluster_members: ``vec4i`` per cluster; constraint ids sorted
        ascending, ``-1`` for unused slots. Length ``max_num_interactions``;
        only the first ``num_clusters[0]`` entries are valid.
    :ivar num_clusters: scalar device array, length 1.
    :ivar element_to_cluster: dense cluster id per element, ``-1`` for
        elements past ``num_elements[0]``. Length ``max_num_interactions``.

    All scratch is pre-allocated. ``build_clusters`` is fixed-size and
    graph-capture safe. Output is deterministic given the same
    ``packed_priorities``.
    """

    MAX_CLUSTER_SIZE: int = int(MAX_CLUSTER_SIZE)

    def __init__(
        self,
        max_num_interactions: int,
        max_num_nodes: int,
        device: wp.context.Devicelike = None,
        seed: int = 0,
    ) -> None:
        if max_num_interactions <= 0:
            raise ValueError(f"max_num_interactions must be > 0 (got {max_num_interactions})")
        if max_num_nodes <= 0:
            raise ValueError(f"max_num_nodes must be > 0 (got {max_num_nodes})")
        self._max_num_interactions = int(max_num_interactions)
        self._max_num_nodes = int(max_num_nodes)
        self._device = wp.get_device(device)

        # Owned adjacency (used when caller doesn't provide one).
        self._adjacency = ElementVertexAdjacency(
            max_num_interactions=self._max_num_interactions,
            max_num_nodes=self._max_num_nodes,
            device=self._device,
        )

        # Default priorities: unique permutation of [1, N]. Matches the
        # JP-MIS seeding scheme in graph_coloring.py; callers can
        # override per-call.
        import numpy as np  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        priorities = rng.permutation(self._max_num_interactions).astype(np.int32) + 1
        if priorities.max() >= (1 << 24):
            raise ValueError(
                f"max_num_interactions ({self._max_num_interactions}) exceeds the 2^24 packed-priority limit."
            )
        packed = (priorities & 0x00FFFFFF).astype(np.int32)
        self._packed_priorities: wp.array[wp.int32] = wp.from_numpy(packed, dtype=wp.int32, device=self._device)

        # Working state.
        n = self._max_num_interactions
        self._cluster_id_of: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)
        self._pending_id_of: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)
        self._seed_committed: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)
        self._num_remaining: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._progress_flag: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._ks_target: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self._device)

        # Compaction scratch.
        self._is_seed_bitmap: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)
        self._cluster_dense_id: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)
        self._cluster_members_flat: wp.array[wp.int32] = wp.zeros(
            n * self.MAX_CLUSTER_SIZE, dtype=wp.int32, device=self._device
        )

        # Outputs.
        self.cluster_members: wp.array[wp.vec4i] = wp.zeros(n, dtype=wp.vec4i, device=self._device)
        self.num_clusters: wp.array[wp.int32] = wp.zeros(1, dtype=wp.int32, device=self._device)
        self.element_to_cluster: wp.array[wp.int32] = wp.zeros(n, dtype=wp.int32, device=self._device)

    @property
    def max_num_interactions(self) -> int:
        return self._max_num_interactions

    @property
    def max_num_nodes(self) -> int:
        return self._max_num_nodes

    @property
    def packed_priorities(self) -> wp.array:
        """The built-in priority permutation. Exposed for tests / parity
        checks."""
        return self._packed_priorities

    @property
    def adjacency(self) -> ElementVertexAdjacency:
        return self._adjacency

    def build_clusters(
        self,
        elements: wp.array,  # wp.array[ElementInteractionData]
        num_elements: wp.array[wp.int32],
        packed_priorities: wp.array | None = None,
        adjacency: ElementVertexAdjacency | None = None,
    ) -> None:
        """Run the full clustering pipeline.

        Args:
            elements: ``ElementInteractionData`` buffer (length
                ``max_num_interactions``).
            num_elements: device scalar holding the active element count.
            packed_priorities: optional override for the priority array;
                must be length ``max_num_interactions`` int32. ``None``
                uses the builder's internal permutation.
            adjacency: optional externally-built adjacency. Caller is
                responsible for invoking ``adjacency.build()`` for the
                current ``elements`` / ``num_elements`` BEFORE this
                call. ``None`` triggers an internal adjacency build.
        """
        if packed_priorities is None:
            packed_priorities = self._packed_priorities
        if adjacency is None:
            self._adjacency.build(elements, num_elements)
            adj = self._adjacency
        else:
            adj = adjacency

        n = self._max_num_interactions

        # 1) Init.
        wp.launch(
            _cluster_init_state_kernel,
            dim=n,
            inputs=[num_elements],
            outputs=[
                self._cluster_id_of,
                self._pending_id_of,
                self._seed_committed,
                self.element_to_cluster,
                self._num_remaining,
            ],
            device=self._device,
        )
        wp.launch(
            _cluster_init_members_flat_kernel,
            dim=n * self.MAX_CLUSTER_SIZE,
            inputs=[self._cluster_members_flat],
            device=self._device,
        )
        wp.launch(
            _cluster_init_num_clusters_kernel,
            dim=n,
            outputs=[self.num_clusters, self._cluster_dense_id],
            device=self._device,
        )

        # 2) Outer loop over Ks_target (host-static).
        self._elements_handle = elements
        self._num_elements_handle = num_elements
        self._packed_prio_handle = packed_priorities
        self._adj_handle = adj
        # Host-unrolled outer loop over Ks_target. Replaces the previous
        # ``wp.capture_while(progress_flag, body)`` with a fixed-iteration
        # unroll: bracketed kernels already early-exit on committed
        # elements, and the captured graph's per-iteration launch
        # overhead is ~µs, much less than the CUDA conditional-graph
        # node overhead the capture_while emitted. See
        # ``feedback_avoid_conditional_while.md`` and PERF_NOTES.md for
        # the general rule. Iteration cap is :data:`_MAX_OUTER_ITERS`
        # times :data:`_INNER_UNROLL` = 32 sub-rounds; tet meshes drain
        # well under that.
        max_outer = int(_MAX_OUTER_ITERS)
        for ks in (self.MAX_CLUSTER_SIZE, 3, 2, 1):
            wp.launch(
                _cluster_set_ks_kernel,
                dim=1,
                inputs=[wp.int32(ks)],
                outputs=[self._ks_target],
                device=self._device,
            )
            for _ in range(max_outer):
                self._inner_round_body()

        # 3) Compaction: seed bitmap -> scan -> dense ids -> per-element.
        wp.launch(
            _cluster_mark_seeds_kernel,
            dim=n,
            inputs=[num_elements, self._cluster_id_of],
            outputs=[self._is_seed_bitmap],
            device=self._device,
        )
        # Exclusive scan via the existing scan utility. Reuses scan_variable_length
        # semantics (just calls wp.utils.array_scan on a full array).
        wp.utils.array_scan(self._is_seed_bitmap, self._is_seed_bitmap, inclusive=False)
        wp.launch(
            _cluster_assign_dense_ids_kernel,
            dim=n,
            inputs=[num_elements, self._is_seed_bitmap, self._cluster_id_of],
            outputs=[self._cluster_dense_id, self.num_clusters],
            device=self._device,
        )
        wp.launch(
            _cluster_fill_element_to_cluster_kernel,
            dim=n,
            inputs=[num_elements, self._cluster_id_of, self._cluster_dense_id],
            outputs=[self.element_to_cluster],
            device=self._device,
        )
        wp.launch(
            _cluster_emit_members_kernel,
            dim=n,
            inputs=[num_elements, self.element_to_cluster],
            outputs=[self._cluster_members_flat],
            device=self._device,
        )
        wp.launch(
            _cluster_finalize_members_kernel,
            dim=n,
            inputs=[self.num_clusters, self._cluster_members_flat],
            outputs=[self.cluster_members],
            device=self._device,
        )

    # --- capture_while bodies ------------------------------------------------

    def _inner_round_body(self) -> None:
        """Body of the per-Ks_target capture_while. Unrolls
        ``_INNER_UNROLL`` propose+grow+validate+commit rounds; the
        capture_while predicate is ``progress_flag``, reset at the top of
        every body invocation and bumped by the validate kernel each
        time a cluster commits.
        """
        # Clear progress_flag for this body invocation. If no cluster
        # commits across the unrolled rounds, progress_flag stays 0 and
        # capture_while exits.
        wp.launch(
            _cluster_set_progress_kernel,
            dim=1,
            inputs=[wp.int32(0), self._progress_flag],
            device=self._device,
        )

        n = self._max_num_interactions
        for _ in range(_INNER_UNROLL):
            wp.launch(
                _cluster_propose_seeds_kernel,
                dim=n,
                inputs=[
                    self._elements_handle,
                    self._num_elements_handle,
                    self._packed_prio_handle,
                    self._adj_handle.section_end_indices,
                    self._adj_handle.vertex_to_adjacent_elements,
                    self._cluster_id_of,
                    self._ks_target,
                ],
                outputs=[self._pending_id_of],
                device=self._device,
            )
            wp.launch(
                _cluster_grow_seeds_kernel,
                dim=n,
                inputs=[
                    self._elements_handle,
                    self._num_elements_handle,
                    self._adj_handle.section_end_indices,
                    self._adj_handle.vertex_to_adjacent_elements,
                    self._cluster_id_of,
                    self._ks_target,
                ],
                outputs=[self._pending_id_of],
                device=self._device,
            )
            wp.launch(
                _cluster_validate_seeds_kernel,
                dim=n,
                inputs=[
                    self._elements_handle,
                    self._num_elements_handle,
                    self._adj_handle.section_end_indices,
                    self._adj_handle.vertex_to_adjacent_elements,
                    self._cluster_id_of,
                    self._pending_id_of,
                    self._ks_target,
                ],
                outputs=[
                    self._seed_committed,
                    self._num_remaining,
                    self._progress_flag,
                ],
                device=self._device,
            )
            wp.launch(
                _cluster_commit_or_release_kernel,
                dim=n,
                inputs=[self._num_elements_handle, self._seed_committed],
                outputs=[self._cluster_id_of, self._pending_id_of],
                device=self._device,
            )
            wp.launch(
                _cluster_reset_committed_flags_kernel,
                dim=n,
                inputs=[self._num_elements_handle],
                outputs=[self._seed_committed],
                device=self._device,
            )
