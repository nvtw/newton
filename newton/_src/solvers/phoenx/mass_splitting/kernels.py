# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Broadcast / average / writeback substep-loop kernels for mass splitting.

Direct port of C# ``MassSplitting`` substep helpers
(``CudaKernels/Solver/SolverKernels.cs`` lines 44-110):

* :func:`launch_broadcast_rigid_to_copy_states` ↔
  ``BroadcastRigidToCopyStatesKernel`` (called once per substep, before
  the PGS prepare phase). Fan body / particle state into every
  ``(node, partition_copy)`` slot. Mirrors the C# ``TinyRigidState``
  copy constructor which forward-integrates ``position += dt *
  velocity`` and ``orientation = integrate(orientation, omega, dt)``,
  matching TGS-soft's "predicted position at substep end" semantics.

* :func:`launch_average_and_broadcast` ↔ ``AverageAndBroadcastKernel``
  (called between PGS iterations inside the overflow partition's
  iterate loop). Average linear and angular velocity across all of a
  node's slots, broadcast the average back. Position / orientation
  stay at the broadcast-time forward-integrated value — only velocity
  travels through the Jacobi reduction.

* :func:`launch_copy_state_into_rigids` ↔ ``CopyStateIntoRigidsKernel``
  (called once after the PGS phase, before the integrate kernel).
  Write the first slot's averaged velocity back into
  ``BodyContainer.velocity`` / ``ParticleContainer.velocity``.

All three kernels launch one thread per unified-node-id and gate on
``copy_state.highest_index_in_use[0] == 0`` for the mass-splitting-off
no-op path. Inside a captured CUDA graph, the disabled launch is one
zero-cost kernel-boundary plus per-thread integer compare.

Access-mode synchronization is intentionally NOT folded in here —
Step 4 of the plan will thread the access-mode flip through the slot
helpers once the constraint kernels are refactored to read/write copy
slots directly. Until then the broadcast assumes the body is at
``VELOCITY_LEVEL`` (true by default per ``body_container_zeros``).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    integrate_orientation,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.mass_splitting.access import (
    slot_synchronize_to_velocity_level,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "launch_average_and_broadcast",
    "launch_broadcast_rigid_to_copy_states",
    "launch_copy_state_into_rigids",
]


_ACCESS_MODE_STATIC = wp.constant(wp.int32(ACCESS_MODE_STATIC))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


@wp.func
def _section_range(copy_state: CopyStateContainer, node_id: wp.int32):
    """Return ``(start, end)`` slot indices for a node, or ``(0, 0)``
    when ``node_id`` has no slots.

    Centralised here so the start-index branch (``section_end[node-1]``
    vs ``0``) lives in one place. Adds one int load and one branch on
    the hot path — within the "small overhead acceptable when disabled"
    budget the user signed off on.
    """
    if node_id < 0 or node_id >= copy_state.section_end.shape[0]:
        return wp.int32(0), wp.int32(0)
    start = wp.int32(0)
    if node_id > 0:
        start = copy_state.section_end[node_id - 1]
    end = copy_state.section_end[node_id]
    return start, end


@wp.kernel(enable_backward=False)
def _broadcast_rigid_to_copy_states_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    dt: wp.float32,
):
    """Fan body / particle state into every owned slot.

    One thread per unified-node-id. ``node_id < num_bodies`` reads from
    ``bodies``; otherwise reads from ``particles`` at index
    ``node_id - num_bodies``. Slot fields are stamped with
    ``ACCESS_MODE_VELOCITY_LEVEL`` (or ``ACCESS_MODE_STATIC`` if the
    source is static / pinned).
    """
    node_id = wp.tid()
    # Disabled fast path: zero slots populated → no work.
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return
    start, end = _section_range(copy_state, node_id)
    if start >= end:
        return

    pos = wp.vec3f(0.0, 0.0, 0.0)
    orient = wp.quatf(0.0, 0.0, 0.0, 1.0)
    vel = wp.vec3f(0.0, 0.0, 0.0)
    ang = wp.vec3f(0.0, 0.0, 0.0)
    mode = _ACCESS_MODE_VELOCITY_LEVEL

    if node_id < num_bodies:
        # Rigid body source.
        body_mode = bodies.access_mode[node_id]
        if body_mode == _ACCESS_MODE_STATIC:
            mode = _ACCESS_MODE_STATIC
        body_pos = bodies.position[node_id]
        body_orient = bodies.orientation[node_id]
        vel = bodies.velocity[node_id]
        ang = bodies.angular_velocity[node_id]
        # TGS-soft predicted position at substep end: matches C#
        # TinyRigidState(VelState, dt) constructor in
        # ``BodyTypes.cs:233-244``.
        pos = body_pos + dt * vel
        orient = integrate_orientation(body_orient, ang, dt)
    else:
        # Particle source. No orientation / angular velocity.
        p = node_id - num_bodies
        p_mode = particles.access_mode[p]
        if p_mode == _ACCESS_MODE_STATIC:
            mode = _ACCESS_MODE_STATIC
        p_pos = particles.position[p]
        vel = particles.velocity[p]
        pos = p_pos + dt * vel
        # Orientation slot is unused for particles; leave the broadcast
        # default of identity. Same for angular velocity (vec3f(0)).

    for slot in range(start, end):
        copy_state.position[slot] = pos
        copy_state.orientation[slot] = orient
        copy_state.velocity[slot] = vel
        copy_state.angular_velocity[slot] = ang
        copy_state.access_mode[slot] = mode


@wp.kernel(enable_backward=False)
def _average_and_broadcast_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """Synchronise per-slot dual state to VELOCITY_LEVEL, then average
    velocity / angular_velocity across a node's slots and broadcast the
    result.

    Mirrors the C# pattern in
    ``MassSplittingRigidBodyInteractionGraph.AverageAndBroadcast``
    (``MassSplitting/MassSplittingRigidBodyInteractionGraph.cs:324-378``):
    each slot's ``SetAccessMode(VelocityLevel, ...)`` is called BEFORE
    the velocity sum. Position-level work done by constraint iterates
    is thus encoded as ``v = (slot.position - body.position_prev_substep)
    / dt`` and folded into the velocity average for free -- no separate
    position-averaging code needed. The synchronize anchor is the
    body / particle's ``position_prev_substep`` (substep-start snapshot).

    Bodies with a single slot (or zero) are a no-op for the average;
    bodies with N>1 slots get their N velocities averaged and
    broadcast. For particle nodes the angular_velocity sum is over
    zeros so the write is harmless on that field.
    """
    node_id = wp.tid()
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return
    start, end = _section_range(copy_state, node_id)
    count = end - start
    if count <= wp.int32(1):
        return

    # Synchronize every slot to VELOCITY_LEVEL first (C# pattern).
    # Position-level work gets encoded as velocity deltas relative to
    # the body / particle's substep-start snapshot.
    for slot in range(start, end):
        slot_synchronize_to_velocity_level(
            bodies, particles, copy_state, node_id, slot, num_bodies, inv_dt
        )

    sum_v = wp.vec3f(0.0, 0.0, 0.0)
    sum_w = wp.vec3f(0.0, 0.0, 0.0)
    for slot in range(start, end):
        sum_v = sum_v + copy_state.velocity[slot]
        sum_w = sum_w + copy_state.angular_velocity[slot]

    inv_count = wp.float32(1.0) / wp.float32(count)
    avg_v = sum_v * inv_count
    avg_w = sum_w * inv_count

    for slot in range(start, end):
        copy_state.velocity[slot] = avg_v
        copy_state.angular_velocity[slot] = avg_w


@wp.kernel(enable_backward=False)
def _copy_state_into_rigids_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """Synchronise slot[0] to VELOCITY_LEVEL and write velocity back to
    body / particle storage.

    Mirrors C# ``TinyRigidState.WriteBack`` (``TinyRigidState.cs:92``):
    ``SynchronizeVelAndPosStateUpdates(VelocityLevel, ...)`` is invoked
    first to fold any pending position-level state into velocity, then
    velocity / angular_velocity are copied out. Position is not copied
    back -- the standard solver integrate step advances it from
    ``position_prev_substep + velocity * dt`` next, which automatically
    yields the averaged position when mass splitting was engaged.

    Static nodes are skipped (slot's access_mode was stamped STATIC by
    broadcast for those).
    """
    node_id = wp.tid()
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return
    start, end = _section_range(copy_state, node_id)
    if start >= end:
        return
    if copy_state.access_mode[start] == _ACCESS_MODE_STATIC:
        return

    # Synchronize slot[0] to VELOCITY_LEVEL before writeback. Sibling
    # slots already match after the prior average + broadcast pass.
    slot_synchronize_to_velocity_level(
        bodies, particles, copy_state, node_id, start, num_bodies, inv_dt
    )

    vel = copy_state.velocity[start]
    if node_id < num_bodies:
        bodies.velocity[node_id] = vel
        bodies.angular_velocity[node_id] = copy_state.angular_velocity[start]
    else:
        particles.velocity[node_id - num_bodies] = vel


def launch_broadcast_rigid_to_copy_states(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: int,
    dt: float,
) -> None:
    """Launch :func:`_broadcast_rigid_to_copy_states_kernel`.

    ``copy_state.section_end.shape[0]`` is the unified-node count and
    drives the launch dim. Safe to call from inside ``wp.ScopedCapture``.
    """
    num_nodes = copy_state.section_end.shape[0]
    wp.launch(
        _broadcast_rigid_to_copy_states_kernel,
        dim=num_nodes,
        inputs=[copy_state, bodies, particles, wp.int32(num_bodies), wp.float32(dt)],
        device=copy_state.section_end.device,
    )


def launch_average_and_broadcast(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: int,
    inv_dt: float,
) -> None:
    """Launch :func:`_average_and_broadcast_kernel`.

    The kernel needs ``bodies`` + ``particles`` to read the substep-start
    snapshots (``position_prev_substep`` / ``orientation_prev_substep``)
    used as the synchronize anchor when flipping slot access mode to
    VELOCITY_LEVEL. ``inv_dt`` is the substep inverse-dt.
    """
    num_nodes = copy_state.section_end.shape[0]
    wp.launch(
        _average_and_broadcast_kernel,
        dim=num_nodes,
        inputs=[copy_state, bodies, particles, wp.int32(num_bodies), wp.float32(inv_dt)],
        device=copy_state.section_end.device,
    )


def launch_copy_state_into_rigids(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: int,
    inv_dt: float,
) -> None:
    """Launch :func:`_copy_state_into_rigids_kernel`.

    ``inv_dt`` is the substep inverse-dt -- used by the slot
    synchronize at writeback to encode any pending position-level state
    as velocity.
    """
    num_nodes = copy_state.section_end.shape[0]
    wp.launch(
        _copy_state_into_rigids_kernel,
        dim=num_nodes,
        inputs=[copy_state, bodies, particles, wp.int32(num_bodies), wp.float32(inv_dt)],
        device=copy_state.section_end.device,
    )
