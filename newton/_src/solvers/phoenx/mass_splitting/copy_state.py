# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Per-(node, partition) TinyRigidState slot storage for Tonge mass splitting.

Mirrors the C# ``MassSplittingRigidBodyInteractionGraphGpu`` struct
(``CudaKernels/Common/MassSplittingTypes.cs:25``). A single
:class:`CopyStateContainer` holds:

* The flat slot SoA (``position``, ``orientation``, ``velocity``,
  ``angular_velocity``, ``access_mode``). Length ``capacity``. Each
  slot is one (node, partition-copy) tuple.

* Per-node section bounds (``section_end``). Length
  ``num_bodies + num_particles``. Slot range owned by node ``i`` is
  ``[section_end[i-1], section_end[i])`` (with ``section_end[-1] := 0``).

* Per-slot partition key (``partition_list``). Length ``capacity``. Used
  by ``get_state_index`` (Step 4) to binary-search a slot for a given
  ``(node_id, parallel_id)`` pair.

* Total slots in use (``highest_index_in_use``). Length 1. Written
  on-device during build; read by ``get_state_index`` to short-circuit
  bodies past the populated tail.

The C# layout stores everything as ``TinyRigidState`` records inside a
``Ptr<TinyRigidState>`` — we use one ``wp.array`` per field instead so
Warp's struct-of-array codegen stays clean and the orientation /
angular_velocity columns aren't paid for by particle slots that never
read them.

The buffer is sized for ``num_bodies + num_particles`` unified slots, so
both rigid-body copies and cloth-particle copies live in the same arena.
Particle slots leave ``orientation`` / ``angular_velocity`` untouched.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_NONE

__all__ = [
    "CopyStateContainer",
    "copy_state_container_zeros",
]


@wp.struct
class CopyStateContainer:
    """SoA storage for per-(node, partition_copy) TinyRigidState slots.

    All arrays except ``section_end`` are indexed by slot id in
    ``[0, capacity)``. ``section_end`` is indexed by unified node id in
    ``[0, num_bodies + num_particles)``.
    """

    #: Slot position [m] in world space. Body slots: COM position. Particle
    #: slots: particle position. Length: ``capacity``.
    position: wp.array[wp.vec3f]

    #: Slot orientation. Body slots: body orientation. Particle slots:
    #: unused (left at the broadcast-time default). Length: ``capacity``.
    orientation: wp.array[wp.quatf]

    #: Slot linear velocity [m/s]. Length: ``capacity``.
    velocity: wp.array[wp.vec3f]

    #: Slot angular velocity [rad/s]. Body slots only; particle slots
    #: stay at zero. Length: ``capacity``.
    angular_velocity: wp.array[wp.vec3f]

    #: Per-slot access mode (see :mod:`access_mode`). Length: ``capacity``.
    access_mode: wp.array[wp.int32]

    #: Inclusive end index of each node's slot run. Indexed by unified
    #: node id. Empty bodies inherit the prefix-max from the build step
    #: so ``section_end[i] - section_end[i-1] == 0``. Length:
    #: ``num_bodies + num_particles``.
    section_end: wp.array[wp.int32]

    #: Partition key for each slot. Length: ``capacity``.
    partition_list: wp.array[wp.int32]

    #: Total slot count actually populated by the last build (length 1).
    #: When zero the read/write helpers short-circuit to direct
    #: body/particle access — this is the disabled-fast-path probe.
    highest_index_in_use: wp.array[wp.int32]

    #: Per-node cache: the slot index for ``parallel_id == 0`` if the node
    #: has one (the first / smallest partition key always lives at
    #: ``section_end[node-1]`` by sort construction), else ``-1``.
    #: Stamped at the end of :func:`build_interaction_graph` and read by
    #: :func:`get_state_index` to skip the section_end + partition_list
    #: load chain on the hot path. Length: ``num_bodies + num_particles``.
    #: Generic across constraint types (rigid joints, cloth, soft-tet,
    #: contacts) -- every iterate that calls ``get_state_index(node, 0)``
    #: benefits.
    slot_for_pid0: wp.array[wp.int32]

    #: Per-node cache: total slot count = ``section_end[node] -
    #: section_end[node-1]`` (with ``section_end[-1] = 0``). Used as the
    #: ``inv_factor`` return value in :func:`get_state_index`. Stamped
    #: alongside ``slot_for_pid0``. Length: ``num_bodies + num_particles``.
    count_per_node: wp.array[wp.int32]


def copy_state_container_zeros(
    capacity: int,
    num_nodes: int,
    device: wp.context.Devicelike = None,
) -> CopyStateContainer:
    """Allocate a zero-initialised :class:`CopyStateContainer`.

    Args:
        capacity: Maximum number of (node, partition_copy) slots. An
            upper bound is ``sum_i(num_partitions_touched_by_node_i)``,
            which is at most ``num_active_constraints *
            max_endpoints_per_constraint``. Caller is responsible for
            sizing — overshoot is just wasted memory, undershoot raises
            from the build kernel (build clamps via ``wp.min``).
        num_nodes: ``num_bodies + num_particles``. Sets the length of
            ``section_end``.
        device: Warp device for the allocation.

    All slot fields start at zero / identity. ``highest_index_in_use``
    starts at zero so :func:`get_state_index` (Step 4) short-circuits to
    direct body/particle access until the first build runs.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1 (got {capacity})")
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1 (got {num_nodes})")
    c = CopyStateContainer()
    c.position = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    c.orientation = wp.zeros(capacity, dtype=wp.quatf, device=device)
    c.velocity = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    c.angular_velocity = wp.zeros(capacity, dtype=wp.vec3f, device=device)
    c.access_mode = wp.full(capacity, value=int(ACCESS_MODE_NONE), dtype=wp.int32, device=device)
    c.section_end = wp.zeros(num_nodes, dtype=wp.int32, device=device)
    c.partition_list = wp.zeros(capacity, dtype=wp.int32, device=device)
    c.highest_index_in_use = wp.zeros(1, dtype=wp.int32, device=device)
    # ``slot_for_pid0`` starts at -1 (no slot) so :func:`get_state_index`
    # returns the no-slot fallback until the first build runs.
    c.slot_for_pid0 = wp.full(num_nodes, value=-1, dtype=wp.int32, device=device)
    c.count_per_node = wp.zeros(num_nodes, dtype=wp.int32, device=device)
    return c
