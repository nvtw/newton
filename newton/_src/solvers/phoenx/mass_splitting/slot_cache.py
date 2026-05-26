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

from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    MAX_BODIES,
    ElementInteractionData,
    element_interaction_data_get,
)
from newton._src.solvers.phoenx.mass_splitting.access import get_state_index
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer

__all__ = [
    "build_constraint_slot_cache",
    "build_slot_cache_kernel",
]


@wp.kernel(enable_backward=False)
def build_slot_cache_kernel(
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    elements: wp.array[ElementInteractionData],
    copy_state: CopyStateContainer,
    constraints: ConstraintContainer,
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
    # ``cid`` can be either a constraint-container cid (joint / cloth /
    # soft-tet / soft-hex range) or a contact local_cid offset past the
    # constraint container's bounds. The slot cache is sized to the
    # constraint container, so contact entries skip the write -- contact
    # iterates still use :func:`get_state_index` directly.
    if cid >= constraints.slot_cache.shape[0]:
        return

    # parallel_id discovery. ``max_colored_partitions < 0`` disables the
    # overflow bucket entirely (everything stays in regular colours), so
    # parallel_id is statically 0 for every cid.
    parallel_id = wp.int32(0)
    if max_colored_partitions >= wp.int32(0):
        overflow_start = color_starts[max_colored_partitions]
        if csr_pos >= overflow_start:
            parallel_id = (csr_pos - overflow_start) / ms_batch_size

    el = elements[cid]
    for v in range(MAX_BODIES):
        body = element_interaction_data_get(el, v)
        if body < wp.int32(0):
            constraints.slot_cache[cid, v] = wp.int32(-1)
            constraints.count_cache[cid, v] = wp.int32(1)
            continue
        slot, count = get_state_index(copy_state, body, parallel_id)
        constraints.slot_cache[cid, v] = slot
        constraints.count_cache[cid, v] = count


def build_constraint_slot_cache(
    element_ids_by_color: wp.array,
    color_starts: wp.array,
    num_active_constraints: wp.array,
    elements: wp.array,
    copy_state: CopyStateContainer,
    constraints: ConstraintContainer,
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
        elements: Per-cid ``ElementInteractionData`` body lists (built
            by the adjacency kernel).
        copy_state: The world's mass-splitting copy state.
        constraints: Target -- ``slot_cache`` / ``count_cache`` get
            stamped.
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
            elements,
            copy_state,
            constraints,
            wp.int32(max_colored_partitions),
            wp.int32(ms_batch_size),
        ],
        device=constraints.slot_cache.device,
    )
