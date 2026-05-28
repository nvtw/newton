# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-cid slot / count cache build.

Constraint iterates need the mass-splitting slot for each vertex they
touch. Looking that slot up via :func:`get_state_index` on every PGS
iteration -- 4 to 8 lookups per cid, 20+ iterations per frame -- adds up:
the lookup chain (body load -> ``slot_for_pid0[body]``) sits on the
critical path of the iterate kernel.

This module stamps a per-cid table once per frame, right after the
partitioner emits ``element_ids_by_color`` and ``color_starts``. The
iterate kernel then reads ``constraints.slot_cache[cid, v]`` instead of
calling :func:`get_state_index`, collapsing two dependent loads to one.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_get_body1,
    contact_get_body2,
    contact_get_side0_nodes_extra,
    contact_get_side1_nodes_extra,
    contact_set_count1,
    contact_set_count2,
    contact_set_side0_counts_extra,
    contact_set_side0_slots_extra,
    contact_set_side1_counts_extra,
    contact_set_side1_slots_extra,
    contact_set_slot1,
    contact_set_slot2,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_CLOTH_BENDING,
    CONSTRAINT_TYPE_CLOTH_TRIANGLE,
    CONSTRAINT_TYPE_SOFT_HEXAHEDRON,
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON,
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN,
    ConstraintContainer,
    constraint_get_type,
    read_int,
)
from newton._src.solvers.phoenx.mass_splitting.access import get_state_index
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer

#: ConstraintContainer body-field offsets per type. Must match the
#: schemas in :mod:`constraints.constraint_*` (kept in sync with
#: ``_<TYPE>_OFF_BODY*`` in :mod:`solver_phoenx_kernels`). The natural
#: body order matters because the iterate / prepare functions read
#: ``slot_cache[cid, v]`` indexed by the SAME ordinal that they read
#: ``body{v+1}`` from the constraint row -- if the cache stored
#: compacted (pinned-dropped) order, the slot for vertex ``v`` would
#: end up routed to the wrong body in pinned-cloth scenes.
_OFF_BODY1 = wp.constant(wp.int32(1))
_OFF_BODY2 = wp.constant(wp.int32(2))
_OFF_BODY3 = wp.constant(wp.int32(3))
_OFF_BODY4 = wp.constant(wp.int32(4))
_OFF_BODY5 = wp.constant(wp.int32(5))
_OFF_BODY6 = wp.constant(wp.int32(6))
_OFF_BODY7 = wp.constant(wp.int32(7))
_OFF_BODY8 = wp.constant(wp.int32(8))

__all__ = [
    "build_constraint_slot_cache",
    "build_slot_cache_kernel",
]


@wp.kernel(enable_backward=False)
def build_slot_cache_kernel(
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    copy_state: CopyStateContainer,
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    contact_offset: wp.int32,
    max_colored_partitions: wp.int32,
    ms_batch_size: wp.int32,
):
    """Stamp ``constraints.slot_cache[cid, v]`` and
    ``constraints.count_cache[cid, v]`` for every active cid.

    One thread per CSR position. The cid's eventual parallel_id is
    deterministic from its position in ``element_ids_by_color``:

    * Regular colour (``c < max_colored_partitions``): parallel_id = 0.
    * Overflow colour (``c == max_colored_partitions``): parallel_id =
      ``(csr_pos - color_starts[overflow]) / ms_batch_size``, matching
      the per-thread layout in ``_make_singleworld_persistent_kernel``.

    ``-1`` is left in the cache for vertex slots past the constraint's
    active count (``element_interaction_data_get`` returns ``-1`` past
    the populated entries), and for nodes outside the mass-splitting
    graph (``get_state_index`` returns ``(-1, 1)``).
    """
    csr_pos = wp.tid()
    if csr_pos >= num_active_constraints[0]:
        return
    cid = element_ids_by_color[csr_pos]
    # parallel_id discovery. ``max_colored_partitions < 0`` disables the
    # overflow bucket entirely (everything stays in regular colours), so
    # parallel_id is statically 0 for every cid.
    parallel_id = wp.int32(0)
    if max_colored_partitions >= wp.int32(0):
        overflow_start = color_starts[max_colored_partitions]
        if csr_pos >= overflow_start:
            parallel_id = (csr_pos - overflow_start) / ms_batch_size

    if cid >= contact_offset:
        local_cid = cid - contact_offset
        if local_cid >= wp.int32(0) and local_cid < contact_cols.data.shape[1]:
            b1 = contact_get_body1(contact_cols, local_cid)
            b2 = contact_get_body2(contact_cols, local_cid)
            extra1 = contact_get_side0_nodes_extra(contact_cols, local_cid)
            extra2 = contact_get_side1_nodes_extra(contact_cols, local_cid)
            slot1, count1 = get_state_index(copy_state, b1, parallel_id)
            slot2, count2 = get_state_index(copy_state, b2, parallel_id)
            slot1e0, count1e0 = get_state_index(copy_state, extra1[0], parallel_id)
            slot1e1, count1e1 = get_state_index(copy_state, extra1[1], parallel_id)
            slot1e2, count1e2 = get_state_index(copy_state, extra1[2], parallel_id)
            slot2e0, count2e0 = get_state_index(copy_state, extra2[0], parallel_id)
            slot2e1, count2e1 = get_state_index(copy_state, extra2[1], parallel_id)
            slot2e2, count2e2 = get_state_index(copy_state, extra2[2], parallel_id)
            contact_set_slot1(contact_cols, local_cid, slot1)
            contact_set_slot2(contact_cols, local_cid, slot2)
            contact_set_side0_slots_extra(contact_cols, local_cid, wp.vec3i(slot1e0, slot1e1, slot1e2))
            contact_set_side1_slots_extra(contact_cols, local_cid, wp.vec3i(slot2e0, slot2e1, slot2e2))
            contact_set_count1(contact_cols, local_cid, count1)
            contact_set_count2(contact_cols, local_cid, count2)
            contact_set_side0_counts_extra(contact_cols, local_cid, wp.vec3i(count1e0, count1e1, count1e2))
            contact_set_side1_counts_extra(contact_cols, local_cid, wp.vec3i(count2e0, count2e1, count2e2))
        return

    if cid >= constraints.slot_cache.shape[0]:
        return

    # Read bodies in the iterate's NATURAL order (per-ctype) so the
    # cache row aligns with the per-vertex slot lookup pattern in the
    # iterate / prepare functions. ``elements[cid]`` is compacted
    # (pinned particles dropped + reordered to a contiguous prefix) and
    # is the correct shape for the adjacency / partitioner, but the
    # WRONG shape for iterate-time slot lookup: e.g. a cloth triangle
    # with body2 pinned has ``elements[cid].bodies = (body1, body3,
    # -1, ...)`` while the iterate reads ``body_a, body_b, body_c`` at
    # constraint-row offsets ``BODY1, BODY2, BODY3``. Reading from the
    # constraint row preserves the natural order for every type.
    ctype = constraint_get_type(constraints, cid)
    if ctype == CONSTRAINT_TYPE_SOFT_TETRAHEDRON or ctype == CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN:
        b0 = read_int(constraints, _OFF_BODY1, cid)
        b1 = read_int(constraints, _OFF_BODY2, cid)
        b2 = read_int(constraints, _OFF_BODY3, cid)
        b3 = read_int(constraints, _OFF_BODY4, cid)
        slot0, count0 = get_state_index(copy_state, b0, parallel_id)
        slot1, count1 = get_state_index(copy_state, b1, parallel_id)
        slot2, count2 = get_state_index(copy_state, b2, parallel_id)
        slot3, count3 = get_state_index(copy_state, b3, parallel_id)
        constraints.slot_cache[cid, 0] = slot0
        constraints.slot_cache[cid, 1] = slot1
        constraints.slot_cache[cid, 2] = slot2
        constraints.slot_cache[cid, 3] = slot3
        constraints.count_cache[cid, 0] = count0
        constraints.count_cache[cid, 1] = count1
        constraints.count_cache[cid, 2] = count2
        constraints.count_cache[cid, 3] = count3
    elif ctype == CONSTRAINT_TYPE_CLOTH_TRIANGLE:
        b0 = read_int(constraints, _OFF_BODY1, cid)
        b1 = read_int(constraints, _OFF_BODY2, cid)
        b2 = read_int(constraints, _OFF_BODY3, cid)
        slot0, count0 = get_state_index(copy_state, b0, parallel_id)
        slot1, count1 = get_state_index(copy_state, b1, parallel_id)
        slot2, count2 = get_state_index(copy_state, b2, parallel_id)
        constraints.slot_cache[cid, 0] = slot0
        constraints.slot_cache[cid, 1] = slot1
        constraints.slot_cache[cid, 2] = slot2
        constraints.count_cache[cid, 0] = count0
        constraints.count_cache[cid, 1] = count1
        constraints.count_cache[cid, 2] = count2
    elif ctype == CONSTRAINT_TYPE_CLOTH_BENDING:
        b0 = read_int(constraints, _OFF_BODY1, cid)
        b1 = read_int(constraints, _OFF_BODY2, cid)
        b2 = read_int(constraints, _OFF_BODY3, cid)
        b3 = read_int(constraints, _OFF_BODY4, cid)
        slot0, count0 = get_state_index(copy_state, b0, parallel_id)
        slot1, count1 = get_state_index(copy_state, b1, parallel_id)
        slot2, count2 = get_state_index(copy_state, b2, parallel_id)
        slot3, count3 = get_state_index(copy_state, b3, parallel_id)
        constraints.slot_cache[cid, 0] = slot0
        constraints.slot_cache[cid, 1] = slot1
        constraints.slot_cache[cid, 2] = slot2
        constraints.slot_cache[cid, 3] = slot3
        constraints.count_cache[cid, 0] = count0
        constraints.count_cache[cid, 1] = count1
        constraints.count_cache[cid, 2] = count2
        constraints.count_cache[cid, 3] = count3
    elif ctype == CONSTRAINT_TYPE_SOFT_HEXAHEDRON:
        b0 = read_int(constraints, _OFF_BODY1, cid)
        b1 = read_int(constraints, _OFF_BODY2, cid)
        b2 = read_int(constraints, _OFF_BODY3, cid)
        b3 = read_int(constraints, _OFF_BODY4, cid)
        b4 = read_int(constraints, _OFF_BODY5, cid)
        b5 = read_int(constraints, _OFF_BODY6, cid)
        b6 = read_int(constraints, _OFF_BODY7, cid)
        b7 = read_int(constraints, _OFF_BODY8, cid)
        slot0, count0 = get_state_index(copy_state, b0, parallel_id)
        slot1, count1 = get_state_index(copy_state, b1, parallel_id)
        slot2, count2 = get_state_index(copy_state, b2, parallel_id)
        slot3, count3 = get_state_index(copy_state, b3, parallel_id)
        slot4, count4 = get_state_index(copy_state, b4, parallel_id)
        slot5, count5 = get_state_index(copy_state, b5, parallel_id)
        slot6, count6 = get_state_index(copy_state, b6, parallel_id)
        slot7, count7 = get_state_index(copy_state, b7, parallel_id)
        constraints.slot_cache[cid, 0] = slot0
        constraints.slot_cache[cid, 1] = slot1
        constraints.slot_cache[cid, 2] = slot2
        constraints.slot_cache[cid, 3] = slot3
        constraints.slot_cache[cid, 4] = slot4
        constraints.slot_cache[cid, 5] = slot5
        constraints.slot_cache[cid, 6] = slot6
        constraints.slot_cache[cid, 7] = slot7
        constraints.count_cache[cid, 0] = count0
        constraints.count_cache[cid, 1] = count1
        constraints.count_cache[cid, 2] = count2
        constraints.count_cache[cid, 3] = count3
        constraints.count_cache[cid, 4] = count4
        constraints.count_cache[cid, 5] = count5
        constraints.count_cache[cid, 6] = count6
        constraints.count_cache[cid, 7] = count7


def build_constraint_slot_cache(
    element_ids_by_color: wp.array,
    color_starts: wp.array,
    num_active_constraints: wp.array,
    copy_state: CopyStateContainer,
    constraints: ConstraintContainer,
    contact_cols: ContactColumnContainer,
    contact_offset: int,
    max_colored_partitions: int,
    ms_batch_size: int,
) -> None:
    """Launch :func:`build_slot_cache_kernel`. Capture-safe -- runs on
    the same stream as the rest of the per-frame build chain.

    Args:
        element_ids_by_color: Partitioner's CSR of active cids.
        color_starts: Partitioner's per-colour CSR offsets.
        num_active_constraints: Length-1 device array with the live
            cid count.
        copy_state: The world's mass-splitting copy state.
        constraints: Target -- ``slot_cache`` / ``count_cache`` get
            stamped for non-contact constraints.
        contact_cols: Target for contact endpoint slot/count caches.
        contact_offset: First global cid belonging to contact columns.
        max_colored_partitions: Soft cap from the solver config; ``-1``
            disables overflow.
        ms_batch_size: Mass-splitting overflow batch size (matches the
            iterate kernel's per-thread layout).
    """
    if constraints.slot_cache.shape[0] == 0:
        return
    # dim covers every CSR position so any constraint-container cid in
    # the partition is reached. The kernel itself early-exits past
    # ``num_active_constraints[0]`` and on contact cids that overshoot
    # the slot_cache row count.
    wp.launch(
        build_slot_cache_kernel,
        dim=element_ids_by_color.shape[0],
        inputs=[
            element_ids_by_color,
            color_starts,
            num_active_constraints,
            copy_state,
            constraints,
            contact_cols,
            wp.int32(contact_offset),
            wp.int32(max_colored_partitions),
            wp.int32(ms_batch_size),
        ],
        device=constraints.slot_cache.device,
    )
