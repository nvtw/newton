# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Substep-loop Warp kernels for mass splitting.

Three kernels, all launched at ``dim=num_bodies`` with one thread
per body. They mirror the C# ``SolverKernels.cu`` mass-splitting
helpers: ``BroadcastRigidToCopyStatesKernel`` (lines 92-110),
``AverageAndBroadcastKernel`` (67-88), ``CopyStateIntoRigidsKernel``
(45-63).

Each kernel takes the body store as plain SoA arrays
(``position``, ``orientation``, ``velocity``,
``angular_velocity``) rather than PhoenX's ``BodyContainer`` struct.
This keeps the mass-splitting subtree free of PhoenX-internal
imports -- exactly what the user asked for ("isolated for now").
The integration step (a wrapper inside :class:`SolverPhoenX`) can
unpack the ``BodyContainer`` SoA into these args at the launch site.

## Per-substep ordering invariants

```
substep_start:
  broadcast_rigid_to_copy_states_kernel       # body -> tiny_states[*]
  prepare_iteration_kernel * num_partitions   # NOT IN THIS PORT (lives in solver)
  average_and_broadcast_kernel                # consensus

  for iter in range(solver_iterations):
    iterate_kernel * num_partitions           # NOT IN THIS PORT
    average_and_broadcast_kernel

substep_end:
  copy_state_into_rigids_kernel               # tiny_states[*, 0] -> body
```

If you skip any of the three substep-boundary kernels in this
module, the rest stops being momentum-conserving. See README.md
"Numerical / ordering invariants".
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphData,
    graph_get_state,
    graph_set_state,
    graph_state_section,
)
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_VELOCITY_LEVEL,
    tiny_rigid_state_from_body,
    tiny_rigid_state_set_access_mode,
    tiny_rigid_state_write_back,
)

__all__ = [
    "average_and_broadcast_kernel",
    "broadcast_rigid_to_copy_states_kernel",
    "copy_state_into_rigids_kernel",
]


_ACCESS_MODE_VELOCITY_LEVEL_C = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


# ---------------------------------------------------------------------------
# Broadcast: body store -> all per-partition copies
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def broadcast_rigid_to_copy_states_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    body_velocity: wp.array[wp.vec3f],
    body_angular_velocity: wp.array[wp.vec3f],
    dt: wp.float32,
):
    """Initialise every per-partition copy from the body's main state.

    One thread per body. For body ``b``:

    * Look up its slice ``[start, end)`` in ``tiny_states`` /
      ``partition_list``. If empty (static body or body that didn't
      register interactions), early-out.
    * Construct a :class:`~.state.TinyRigidState` from
      ``(position, orientation, velocity, angular_velocity)``
      integrated forward by ``dt`` (matches C# constructor at
      ``BodyTypes.cs:233-244``).
    * Write that same state into every slot of body ``b``'s slice.
      Subsequent iterate kernels mutate the slots independently;
      :func:`average_and_broadcast_kernel` recombines them.

    Mirrors C# ``MassSplittingRigidBodyInteractionGraphGpu::BroadcastRigidToCopyStates``
    (``MassSplittingTypes.cuh:247-267``) plus the dispatch in
    ``SolverKernels.cu:92-110``.
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    if start >= end:
        return
    new_state = tiny_rigid_state_from_body(
        body_position[rigid_body_index],
        body_orientation[rigid_body_index],
        body_velocity[rigid_body_index],
        body_angular_velocity[rigid_body_index],
        dt,
    )
    for i in range(start, end):
        graph_set_state(graph, i, new_state)


# ---------------------------------------------------------------------------
# Average and broadcast: per-partition copies -> consensus -> all copies
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def average_and_broadcast_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    inv_dt: wp.float32,
):
    """Replace every per-partition copy with the body's average.

    For body ``b`` whose section has ``count > 1`` slots:

    1. Sum ``velocity`` / ``angular_velocity`` across the slots,
       forcing each into velocity-level form on read so summands are
       commensurable. C# does this with
       ``state.SetAccessMode(VelocityLevel, body.VelState, invDt)``
       at ``MassSplittingTypes.cuh:226``.
    2. Divide by ``count``.
    3. Write the averaged ``(velocity, angular_velocity)`` back into
       every slot, also forcing :data:`ACCESS_MODE_VELOCITY_LEVEL` so
       the next iterate sweep starts from a known regime.

    Bodies with ``count <= 1`` are left as-is -- the single copy is
    already its own consensus.

    ``body_position`` / ``body_orientation`` are read for the
    velocity-level synchronisation: when a copy is currently in
    position-level form, the conversion needs the parent body's
    pose at the start of the substep as the reference frame.

    Mirrors ``MassSplittingRigidBodyInteractionGraphGpu::AverageAndBroadcast``
    (``MassSplittingTypes.cuh:206-245``).
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    count = end - start
    if count <= wp.int32(1):
        return
    body_pos = body_position[rigid_body_index]
    body_orient = body_orientation[rigid_body_index]
    sum_vel = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    sum_ang_vel = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for i in range(start, end):
        s = graph_get_state(graph, i)
        s = tiny_rigid_state_set_access_mode(s, _ACCESS_MODE_VELOCITY_LEVEL_C, body_pos, body_orient, inv_dt)
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


# ---------------------------------------------------------------------------
# Copy state into rigids: per-partition copy -> body store
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def copy_state_into_rigids_kernel(
    graph: InteractionGraphData,
    body_position: wp.array[wp.vec3f],
    body_orientation: wp.array[wp.quatf],
    body_velocity: wp.array[wp.vec3f],
    body_angular_velocity: wp.array[wp.vec3f],
    inv_dt: wp.float32,
):
    """End-of-substep write-back: copy slot 0 of every body's section
    back into the SoA body store.

    Only the first slot is read because :func:`average_and_broadcast_kernel`
    has already made all slots equal -- reading the others would be
    redundant work. Bodies with no interactions (``start == end``)
    are skipped.

    ``body_position`` / ``body_orientation`` are read for the same
    velocity-level sync reason as in the average kernel; the write
    back is to ``body_velocity`` / ``body_angular_velocity`` only,
    leaving the integration of pose to the caller's substep loop
    (Newton already integrates pose between substeps; we don't want
    to duplicate that here).

    Mirrors ``MassSplittingRigidBodyInteractionGraphGpu::WriteBack``
    (``MassSplittingTypes.cuh:282-300``).
    """
    rigid_body_index = wp.tid()
    if rigid_body_index >= graph.highest_index_in_use[0]:
        return
    start, end = graph_state_section(graph, rigid_body_index)
    if start >= end:
        return
    state = graph_get_state(graph, start)
    body_pos = body_position[rigid_body_index]
    body_orient = body_orientation[rigid_body_index]
    velocity, angular_velocity = tiny_rigid_state_write_back(state, body_pos, body_orient, inv_dt)
    body_velocity[rigid_body_index] = velocity
    body_angular_velocity[rigid_body_index] = angular_velocity


# ---------------------------------------------------------------------------
# Sentinel hint: kernels above are intentionally device-only.
# Host-side launchers live in ``mass_splitting.py``. Keeping kernel
# definitions and launch policy separate matches Newton's wider
# convention (``constraint_contact.py`` exposes ``contact_iterate_at_RR``,
# ``solver_phoenx.py`` decides the launch grid).
# ---------------------------------------------------------------------------
