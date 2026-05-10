# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side mass-splitting setup kernels.

Replaces the host-side ``MassSplitting.setup_from_coloring`` chain
(``.numpy()`` roundtrip on ``_elements``, Python loop over
``constraint_bodies``, host sort/dedup in ``InteractionGraph.build``)
with a pipeline of Warp kernels that runs entirely on device. This
is what unblocks ``wp.capture_launch`` for scenes that use
``PhoenXWorld(mass_split_max_partitions=K)``.

Mirrors the C# reference (``PhoenX/CudaKernels/MassSplitting/PartitioningKernels.cu``
``RecordAllInteractionsKernel`` plus
``PhoenX/CudaKernels/MassSplitting/MassSplittingKernels.cu``
``BuildInteractionGraphPrepareKernel`` /
``BuildInteractionGraphBuildKernel``) but adapted to Warp's
primitives:

* ``record_all_interactions_kernel`` -- walks the partitioner CSR
  and atomic-appends ``(body, partition_constraint_id)`` int64
  keys into a construction buffer. Also writes
  ``cid_to_partition_constraint_id[cid]`` so the iterate kernel
  can call ``read_state(graph, cid_to_partition_constraint_id[cid], body)``.
  ``partition_constraint_id`` is ``0`` for regular MIS partitions
  and ``j / batch_size`` for cids in the overflow bucket
  (Tonge-style batching, same as C# ``RecordAllInteractionsKernel``).

* ``mark_unique_keys_kernel`` -- after sorting the keys, tag each
  slot with ``1`` if it's the first occurrence of its key (or
  index 0), else ``0``. Tail past the active count is sentinel
  (``INT64_MAX`` from the sort mask) so the comparison naturally
  filters inactive slots.

* ``compact_unique_keys_kernel`` -- scatter unique keys into a
  dense buffer using the inclusive scan of the unique-flag as the
  output offset.

* ``build_partition_list_kernel`` -- walk the dense unique keys,
  write ``partition_list[k] = partition_id``, and stamp
  ``state_section_end_indices[rigid] = k + 1`` at body boundaries
  (where the *next* unique key changes rigid). Bodies with no
  interactions are filled by the subsequent prefix-max scan.

* ``prefix_max_scan_kernel`` -- single-thread inclusive prefix-max
  over ``state_section_end_indices``. Body counts are typically
  small (<10k); a sequential scan is fast enough and avoids the
  multi-pass parallel-prefix-max machinery the C# uses
  (``maxScan.inclusiveScan``).

* ``write_highest_index_kernel`` -- length-1 helper that copies
  ``num_bodies`` (the rigid count) into the graph's
  ``highest_index_in_use`` slot. The C# reference computes this
  per-build by scanning unique-key entries; we use the static
  rigid count which is a safe over-estimate (read_state's
  ``rigid_body_index >= highest_index_in_use[0]`` guard still
  works correctly).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
    MAX_BODIES,
    element_interaction_data_get,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphData,
    graph_get_state,
    graph_set_state,
    graph_state_section,
)
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_VELOCITY_LEVEL,
    TinyRigidState,
    tiny_rigid_state_from_body,
    tiny_rigid_state_set_access_mode,
    tiny_rigid_state_write_back,
)
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "average_and_broadcast_unified_kernel",
    "broadcast_to_copy_states_unified_kernel",
    "build_partition_list_kernel",
    "compact_unique_keys_kernel",
    "copy_state_into_unified_kernel",
    "mark_unique_keys_kernel",
    "prefix_max_scan_kernel",
    "record_all_interactions_kernel",
    "reset_construction_buffers_kernel",
    "write_highest_index_kernel",
]


_ACCESS_MODE_VELOCITY_LEVEL_C = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


# Bit layout of the int64 construction keys.
#
#     bits 32..63 : rigid body index (signed; non-negative for valid
#                   entries; negatives are filtered before append).
#     bits  0..31 : partition_constraint_id.
#
# The radix sort is unsigned, but rigid_body_index >= 0 keeps the sign
# bit clear. The tail past the active count is masked with
# ``INT64_MAX`` by ``sort_variable_length_int64`` so it sorts to the
# end and never matches a real key.

_KEY_RIGID_SHIFT = wp.constant(wp.int64(32))


@wp.func
def _pack_construction_key(rigid_body_id: wp.int32, partition_constraint_id: wp.int32) -> wp.int64:
    """``(rigid << 32) | partition``. Both inputs must be >= 0."""
    return (wp.int64(rigid_body_id) << _KEY_RIGID_SHIFT) | wp.int64(partition_constraint_id)


@wp.func
def _decompose_rigid(key: wp.int64) -> wp.int32:
    return wp.int32(key >> _KEY_RIGID_SHIFT)


@wp.func
def _decompose_partition(key: wp.int64) -> wp.int32:
    """Lower 32 bits of the key. The shift form mirrors the C#
    ``DecomposeKey`` helper -- avoids the int64 unary ~ that Warp
    codegen doesn't reliably emit."""
    return wp.int32(key & wp.int64(0xFFFFFFFF))


@wp.kernel(enable_backward=False)
def reset_construction_buffers_kernel(
    construction_count: wp.array[wp.int32],
    state_section_end_indices: wp.array[wp.int32],
    cid_to_partition_constraint_id: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
):
    """Single launch, dim = max(num_bodies, max_interactions). Resets:

    * ``construction_count[0] = 0``
    * ``state_section_end_indices[i] = 0`` for ``i < num_bodies``
    * ``cid_to_partition_constraint_id[cid] = 0`` for ``cid < num_active``
      (default to "regular partition, single copy" so cids that
      fall outside any partition table still produce a defined
      lookup; ``read_state`` will still fall back to the static
      path because the body has no entry for partition 0).
    """
    tid = wp.tid()
    if tid == 0:
        construction_count[0] = 0
    if tid < state_section_end_indices.shape[0]:
        state_section_end_indices[tid] = 0
    if tid < num_active_constraints[0]:
        cid_to_partition_constraint_id[tid] = 0


@wp.kernel(enable_backward=False)
def record_all_interactions_kernel(
    elements: wp.array[ElementInteractionData],
    interaction_id_to_partition: wp.array[wp.int32],
    element_ids_by_color: wp.array[wp.int32],
    color_starts: wp.array[wp.int32],
    num_active_constraints: wp.array[wp.int32],
    num_bodies: wp.int32,
    max_partitions: wp.int32,
    batch_size: wp.int32,
    construction_keys: wp.array[wp.int64],
    construction_count: wp.array[wp.int32],
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    """Per-CSR-slot kernel: dim = ``element_ids_by_color.shape[0]``.

    For each global slot ``i`` in the CSR:

    1. Skip if ``i >= sum_of_colour_sizes`` (i.e. past the active range).
    2. Look up ``cid = element_ids_by_color[i]`` and the colour
       ``c = interaction_id_to_partition[cid]``.
    3. ``partition_constraint_id`` =
       - ``0`` if ``c < max_partitions`` (regular MIS partition).
       - ``(i - color_starts[c]) / batch_size`` if ``c == max_partitions``
         (overflow bucket; multiple distinct constraint_ids drive
         multiple TinyRigidState copies per body).
    4. Stamp ``cid_to_partition_constraint_id[cid] = partition_constraint_id``.
    5. For each node slot in ``elements[cid]`` (rigid body OR
       particle in unified node space; -1 sentinels are skipped),
       atomic-append the int64 key
       ``(node << 32) | partition_constraint_id`` into
       ``construction_keys``. Both rigid bodies (``node < num_bodies``)
       and cloth particles (``node >= num_bodies``) get state copies
       in the InteractionGraph -- the C# convention
       (``RecordAllInteractionsKernel`` registers every entry of
       ``ElementInteractionData``, regardless of kind).
    """
    i = wp.tid()
    n_active = num_active_constraints[0]
    if n_active == 0:
        return
    # color_starts is exclusive prefix; the total active count is
    # color_starts[num_colors[0]]. We don't have num_colors here so
    # we use ``n_active`` as the cap (every active cid lands in
    # exactly one colour, so the CSR length == n_active).
    if i >= n_active:
        return

    cid = element_ids_by_color[i]
    if cid < wp.int32(0) or cid >= n_active:
        return
    c = interaction_id_to_partition[cid]
    if c < wp.int32(0):
        return

    if c < max_partitions:
        partition_constraint_id = wp.int32(0)
    else:
        # Overflow bucket. ``j = i - color_starts[max_partitions]``
        # is the position within the bucket. Tonge-style batching
        # produces ``ceil(j / batch_size)`` distinct constraint ids,
        # one per batch.
        j = i - color_starts[max_partitions]
        partition_constraint_id = j / batch_size

    cid_to_partition_constraint_id[cid] = partition_constraint_id

    el = elements[cid]
    for k in range(MAX_BODIES):
        node = element_interaction_data_get(el, k)
        if node < wp.int32(0):
            break
        # Both rigid bodies (``node < num_bodies``) and particles
        # (``node >= num_bodies``) get registered. Particles use
        # the unified node id directly so their section in the
        # InteractionGraph lives at indices >= num_bodies.
        slot = wp.atomic_add(construction_count, 0, wp.int32(1))
        if slot < construction_keys.shape[0]:
            construction_keys[slot] = _pack_construction_key(node, partition_constraint_id)


@wp.kernel(enable_backward=False)
def mark_unique_keys_kernel(
    sorted_keys: wp.array[wp.int64],
    construction_count: wp.array[wp.int32],
    unique_flags: wp.array[wp.int32],
):
    """After sort, mark ``unique_flags[k] = 1`` iff ``k == 0`` or
    ``keys[k] != keys[k-1]``. ``k >= count`` is masked off."""
    k = wp.tid()
    n = construction_count[0]
    if k >= n:
        unique_flags[k] = wp.int32(0)
        return
    if k == 0:
        unique_flags[k] = wp.int32(1)
        return
    if sorted_keys[k] != sorted_keys[k - 1]:
        unique_flags[k] = wp.int32(1)
    else:
        unique_flags[k] = wp.int32(0)


@wp.kernel(enable_backward=False)
def compact_unique_keys_kernel(
    sorted_keys: wp.array[wp.int64],
    unique_flags: wp.array[wp.int32],
    unique_offsets: wp.array[wp.int32],
    construction_count: wp.array[wp.int32],
    unique_keys: wp.array[wp.int64],
    unique_count: wp.array[wp.int32],
):
    """Scatter ``sorted_keys[k]`` to ``unique_keys[unique_offsets[k]-1]``
    when ``unique_flags[k] == 1``. Lane 0 also writes
    ``unique_count[0] = unique_offsets[count - 1]`` if the buffer is
    non-empty; with zero active keys it leaves ``unique_count`` at
    whatever the reset wrote (0)."""
    k = wp.tid()
    n = construction_count[0]
    if k >= n:
        return
    if unique_flags[k] == wp.int32(1):
        slot = unique_offsets[k] - wp.int32(1)
        unique_keys[slot] = sorted_keys[k]
    if k == n - 1:
        unique_count[0] = unique_offsets[k]


@wp.kernel(enable_backward=False)
def build_partition_list_kernel(
    unique_keys: wp.array[wp.int64],
    unique_count: wp.array[wp.int32],
    partition_list: wp.array[wp.int32],
    state_section_end_indices: wp.array[wp.int32],
):
    """For each unique key slot ``k``:

    * Decompose ``key -> (rigid, partition)``.
    * ``partition_list[k] = partition``.
    * If this is the last entry for ``rigid`` (i.e. ``k == count-1``
      OR the next entry decomposes to a different ``rigid``), set
      ``state_section_end_indices[rigid] = k + 1``. The subsequent
      prefix-max fills bodies with no interactions.
    """
    k = wp.tid()
    n = unique_count[0]
    if k >= n:
        return
    key = unique_keys[k]
    rigid = _decompose_rigid(key)
    partition = _decompose_partition(key)
    partition_list[k] = partition
    if k == n - 1:
        state_section_end_indices[rigid] = k + wp.int32(1)
    else:
        next_key = unique_keys[k + 1]
        next_rigid = _decompose_rigid(next_key)
        if rigid != next_rigid:
            state_section_end_indices[rigid] = k + wp.int32(1)


@wp.kernel(enable_backward=False)
def prefix_max_scan_kernel(
    state_section_end_indices: wp.array[wp.int32],
    num_nodes: wp.int32,
):
    """Single-thread inclusive prefix-max over
    ``state_section_end_indices[0..num_bodies)``. After this every
    body's slot holds the END index of its section -- bodies with
    no interactions inherit the previous body's end so the
    ``read_state`` section lookup
    ``[end_indices[i-1], end_indices[i])`` correctly returns an
    empty section for them.
    """
    if wp.tid() != 0:
        return
    running = wp.int32(0)
    n = wp.min(num_nodes, state_section_end_indices.shape[0])
    for i in range(n):
        v = state_section_end_indices[i]
        if v > running:
            running = v
        state_section_end_indices[i] = running


@wp.kernel(enable_backward=False)
def write_highest_index_kernel(
    highest_index_in_use: wp.array[wp.int32],
    num_nodes: wp.int32,
):
    """Set ``highest_index_in_use[0] = num_nodes``. The C# reference
    computes the actual max-node-with-interactions per-build; using
    the static ``num_nodes`` (= num_bodies + num_particles) is a
    safe over-estimate -- ``read_state`` falls back to the
    static-body path for any node whose section happens to be empty,
    regardless of where the high-water mark sits."""
    if wp.tid() != 0:
        return
    highest_index_in_use[0] = num_nodes


# ---------------------------------------------------------------------------
# Unified broadcast / average / write-back -- bodies + particles in one
# pass. The mass_splitting/kernels.py versions only handle rigid bodies;
# these unified variants also cover cloth particles (node_id >= num_bodies).
# Particles use identity orientation and zero angular velocity in the
# TinyRigidState; the broadcast / average / write_back paths still
# operate on velocity correctly.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def broadcast_to_copy_states_unified_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    body_velocity: wp.array[wp.vec3f],
    body_angular_velocity: wp.array[wp.vec3f],
    particles: ParticleContainer,
    num_bodies: wp.int32,
    dt: wp.float32,
):
    """One thread per node. ``node < num_bodies`` -> rigid body
    broadcast (read body store). ``node >= num_bodies`` -> particle
    broadcast (read particle store, identity orientation, zero
    angular velocity)."""
    node = wp.tid()
    if node >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, node)
    if start >= end:
        return
    if node < num_bodies:
        new_state = tiny_rigid_state_from_body(
            body_position[node],
            body_orientation[node],
            body_velocity[node],
            body_angular_velocity[node],
            dt,
        )
    else:
        p = node - num_bodies
        new_state = tiny_rigid_state_from_body(
            particles.position[p],
            wp.quat_identity(),
            particles.velocity[p],
            wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
            dt,
        )
    for i in range(start, end):
        graph_set_state(graph, i, new_state)


@wp.kernel(enable_backward=False)
def average_and_broadcast_unified_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """One thread per node. Same averaging logic as the rigid-only
    ``average_and_broadcast_kernel``; just reads pose from the
    appropriate store (body vs particle) for the velocity-level
    sync that ``tiny_rigid_state_set_access_mode`` needs.
    """
    node = wp.tid()
    if node >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, node)
    count = end - start
    if count <= wp.int32(1):
        return
    if node < num_bodies:
        node_pos = body_position[node]
        node_ori = body_orientation[node]
    else:
        p = node - num_bodies
        node_pos = particles.position[p]
        node_ori = wp.quat_identity()
    sum_vel = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    sum_ang_vel = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for i in range(start, end):
        s = graph_get_state(graph, i)
        s = tiny_rigid_state_set_access_mode(s, _ACCESS_MODE_VELOCITY_LEVEL_C, node_pos, node_ori, inv_dt)
        sum_vel = sum_vel + s.velocity
        sum_ang_vel = sum_ang_vel + s.angular_velocity
    avg_scale = wp.float32(1.0) / wp.float32(count)
    avg_vel = sum_vel * avg_scale
    avg_ang_vel = sum_ang_vel * avg_scale
    for i in range(start, end):
        s = graph_get_state(graph, i)
        s.velocity = avg_vel
        s.angular_velocity = avg_ang_vel
        s.access_mode = _ACCESS_MODE_VELOCITY_LEVEL_C
        graph_set_state(graph, i, s)


@wp.kernel(enable_backward=False)
def copy_state_into_unified_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    body_velocity: wp.array[wp.vec3f],
    body_angular_velocity: wp.array[wp.vec3f],
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """End-of-substep write-back: tiny_states[start] -> body store
    (for rigid nodes) or particle store (for particle nodes)."""
    node = wp.tid()
    if node >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, node)
    if start >= end:
        return
    state = graph_get_state(graph, start)
    if node < num_bodies:
        body_pos = body_position[node]
        body_orient = body_orientation[node]
        velocity, angular_velocity = tiny_rigid_state_write_back(state, body_pos, body_orient, inv_dt)
        body_velocity[node] = velocity
        body_angular_velocity[node] = angular_velocity
    else:
        p = node - num_bodies
        node_pos = particles.position[p]
        velocity, _angular_velocity = tiny_rigid_state_write_back(state, node_pos, wp.quat_identity(), inv_dt)
        particles.velocity[p] = velocity
