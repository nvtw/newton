# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Broadcast / average / writeback substep-loop kernels for mass splitting.

Port of C# ``MassSplitting`` substep helpers
(``CudaKernels/Solver/SolverKernels.cs:44-110``):

* :func:`launch_broadcast_rigid_to_copy_states` -- once per substep,
  pre-PGS. Fans body / particle state into every ``(node,
  partition_copy)`` slot. Forward-integrates position by ``dt *
  velocity`` (TGS-soft predicted-end-of-substep semantics).
* :func:`launch_average_and_broadcast` -- scalar fallback between PGS
  iterations. Averages linear / angular velocity across a node's slots;
  position / orientation stay at broadcast-time values.
* :func:`launch_average_and_broadcast_grouped` -- CUDA fast path for
  the same operation, using one warp block for eight nodes while
  preserving scalar reduction order.
* :func:`launch_copy_state_into_rigids` -- once post-PGS. Writes
  slot[0]'s averaged velocity back to body / particle storage.

The scalar kernels launch one thread per unified-node-id and short-circuit
on empty copy-state ranges. The grouped average launches 32-thread blocks
that each process eight nodes, keeping one leader lane per node for the
ordered sum and using the remaining subgroup lanes for broadcast writes.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_POSITION_LEVEL,
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
    "launch_average_and_broadcast_grouped",
    "launch_broadcast_rigid_to_copy_states",
    "launch_copy_state_into_rigids",
]


_ACCESS_MODE_NONE = wp.constant(wp.int32(ACCESS_MODE_NONE))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_STATIC = wp.constant(wp.int32(ACCESS_MODE_STATIC))
_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


@wp.func
def _section_range(copy_state: CopyStateContainer, node_id: wp.int32):
    """Return ``(start, end)`` slot indices for a node, or ``(0, 0)``
    when ``node_id`` has no slots."""
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
    if node_id >= copy_state.section_end.shape[0]:
        return
    # Fast bail via the cached count. Empty nodes (count==0) skip the
    # whole broadcast loop. Mass-splitting-disabled scenes hit this
    # path with count==0 for every node, replacing the per-thread
    # ``highest_index_in_use[0]`` probe.
    count = copy_state.count_per_node[node_id]
    if count <= wp.int32(0):
        return
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    end = start + count

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
        # NOTE: unlike rigid bodies, ``cloth_predict_kernel`` already
        # advances ``particles.position`` to the predicted end-of-substep
        # value (``pos_prev_substep + dt * vel``) before this kernel
        # runs. The rigid branch above adds ``dt * vel`` itself because
        # ``_phoenx_apply_forces_and_gravity_kernel`` only updates
        # velocity. Adding ``dt * vel`` again here would double-integrate
        # the particle and inject a spurious position delta into every
        # slot -- which then makes cloth-tri / bending compute huge
        # corrections and oscillate. Use the already-predicted position
        # directly.
        p = node_id - num_bodies
        p_mode = particles.access_mode[p]
        if p_mode == _ACCESS_MODE_STATIC:
            mode = _ACCESS_MODE_STATIC
        pos = particles.position[p]
        vel = particles.velocity[p]
        # Orientation slot is unused for particles; leave the broadcast
        # default of identity. Same for angular velocity (vec3f(0)).

    is_body = node_id < num_bodies
    for slot in range(start, end):
        copy_state.position[slot] = pos
        copy_state.velocity[slot] = vel
        copy_state.access_mode[slot] = mode
        if is_body:
            copy_state.orientation[slot] = orient
            copy_state.angular_velocity[slot] = ang


@wp.func
def _sync_particle_slot_to_velocity_level(
    copy_state: CopyStateContainer,
    particles: ParticleContainer,
    particle_id: wp.int32,
    slot: wp.int32,
    inv_dt: wp.float32,
) -> wp.vec3f:
    """Particle-only SetAccessMode(VELOCITY_LEVEL) for one copy slot."""
    current = copy_state.access_mode[slot]
    if current == _ACCESS_MODE_POSITION_LEVEL:
        vel = (copy_state.position[slot] - particles.position_prev_substep[particle_id]) * inv_dt
        copy_state.velocity[slot] = vel
        copy_state.access_mode[slot] = _ACCESS_MODE_VELOCITY_LEVEL
        return vel
    if current == _ACCESS_MODE_NONE:
        copy_state.access_mode[slot] = _ACCESS_MODE_VELOCITY_LEVEL
    return copy_state.velocity[slot]


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
    if node_id >= copy_state.section_end.shape[0]:
        return
    # Fast bail via the cached count: single-slot / empty nodes are
    # a no-op for the average. One load instead of the section_end +
    # highest_index_in_use chain. Mass-splitting-disabled scenes hit
    # this path with count == 0 for every node.
    count = copy_state.count_per_node[node_id]
    if count <= wp.int32(1):
        return
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    end = start + count

    # Synchronize each slot to VELOCITY_LEVEL and accumulate it in the
    # same pass. Split rigid and particle nodes before the slot loop so
    # particle nodes do not execute the orientation / angular-velocity
    # access path.
    inv_count = wp.float32(1.0) / wp.float32(count)
    if node_id < num_bodies:
        sum_v = wp.vec3f(0.0, 0.0, 0.0)
        sum_w = wp.vec3f(0.0, 0.0, 0.0)
        for slot in range(start, end):
            slot_synchronize_to_velocity_level(bodies, particles, copy_state, node_id, slot, num_bodies, inv_dt)
            sum_v = sum_v + copy_state.velocity[slot]
            sum_w = sum_w + copy_state.angular_velocity[slot]

        avg_v = sum_v * inv_count
        avg_w = sum_w * inv_count
        for slot in range(start, end):
            copy_state.velocity[slot] = avg_v
            copy_state.angular_velocity[slot] = avg_w
        return

    particle_id = node_id - num_bodies
    sum_v = wp.vec3f(0.0, 0.0, 0.0)
    for slot in range(start, end):
        sum_v = sum_v + _sync_particle_slot_to_velocity_level(copy_state, particles, particle_id, slot, inv_dt)

    avg_v = sum_v * inv_count
    for slot in range(start, end):
        copy_state.velocity[slot] = avg_v


@wp.kernel(enable_backward=False)
def _average_and_broadcast_grouped_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """Generic average/broadcast with several nodes per warp block.

    Lane 0 of each four-lane subgroup computes that node's average in
    scalar slot order. The subgroup then broadcasts the scalar result
    back to the node's slots cooperatively. This preserves the serial
    reduction order while reducing the second slot loop.
    """
    block, lane = wp.tid()
    group = lane / _MASS_SPLITTING_GROUPED_LANES_PER_NODE_WP
    local = lane - group * _MASS_SPLITTING_GROUPED_LANES_PER_NODE_WP
    node_id = block * wp.int32(_MASS_SPLITTING_GROUPED_NODES_PER_BLOCK) + group

    count = wp.int32(0)
    start = wp.int32(0)
    end = wp.int32(0)
    if node_id < copy_state.section_end.shape[0]:
        count = copy_state.count_per_node[node_id]
        if count > wp.int32(1):
            if node_id > wp.int32(0):
                start = copy_state.section_end[node_id - wp.int32(1)]
            end = start + count

    avg_x = wp.float32(0.0)
    avg_y = wp.float32(0.0)
    avg_z = wp.float32(0.0)
    avg_wx = wp.float32(0.0)
    avg_wy = wp.float32(0.0)
    avg_wz = wp.float32(0.0)
    is_body = node_id < num_bodies

    if local == wp.int32(0) and count > wp.int32(1):
        sum_v = wp.vec3f(0.0, 0.0, 0.0)
        sum_w = wp.vec3f(0.0, 0.0, 0.0)
        if is_body:
            for slot in range(start, end):
                slot_synchronize_to_velocity_level(bodies, particles, copy_state, node_id, slot, num_bodies, inv_dt)
                sum_v = sum_v + copy_state.velocity[slot]
                sum_w = sum_w + copy_state.angular_velocity[slot]
        else:
            p = node_id - num_bodies
            for slot in range(start, end):
                sum_v = sum_v + _sync_particle_slot_to_velocity_level(copy_state, particles, p, slot, inv_dt)
        inv_count = wp.float32(1.0) / wp.float32(count)
        avg_v = sum_v * inv_count
        avg_x = avg_v[0]
        avg_y = avg_v[1]
        avg_z = avg_v[2]
        if is_body:
            avg_w = sum_w * inv_count
            avg_wx = avg_w[0]
            avg_wy = avg_w[1]
            avg_wz = avg_w[2]

    group_leader = group * _MASS_SPLITTING_GROUPED_LANES_PER_NODE_WP
    avg_x = wp.tile(avg_x)[group_leader]
    avg_y = wp.tile(avg_y)[group_leader]
    avg_z = wp.tile(avg_z)[group_leader]
    avg_wx = wp.tile(avg_wx)[group_leader]
    avg_wy = wp.tile(avg_wy)[group_leader]
    avg_wz = wp.tile(avg_wz)[group_leader]

    if count <= wp.int32(1):
        return
    avg_v = wp.vec3f(avg_x, avg_y, avg_z)
    avg_w = wp.vec3f(avg_wx, avg_wy, avg_wz)
    slot = start + local
    while slot < end:
        copy_state.velocity[slot] = avg_v
        if is_body:
            copy_state.angular_velocity[slot] = avg_w
        slot = slot + _MASS_SPLITTING_GROUPED_LANES_PER_NODE_WP


@wp.kernel(enable_backward=False)
def _copy_state_into_rigids_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """Synchronise slot[0] to VELOCITY_LEVEL and write back to body /
    particle storage.

    Mirrors C# ``TinyRigidState.WriteBack`` (``TinyRigidState.cs:92``):
    ``SynchronizeVelAndPosStateUpdates(VelocityLevel, ...)`` is invoked
    first to fold any pending position-level state into velocity, then
    velocity / angular_velocity are copied out.

    Rigid bodies: position is **not** copied back -- the
    :func:`_integrate_velocities_kernel` step that runs next advances
    ``bodies.position += bodies.velocity * dt`` starting from
    ``bodies.position_prev_substep``, which yields exactly the right
    final position (C# template: ``body.Position +=
    body.Velocity * dt`` after WriteBack).

    Particles: there is no equivalent rigid-style integrate step that
    runs after writeback for particles, so we fold the integrate into
    the writeback for the particle range: ``particles.position =
    particles.position_prev_substep + dt * particles.velocity``. This
    is the exact analogue of the rigid pattern (substep-start anchor +
    dt * post-writeback velocity = final position) and works for both
    single-slot (writeback's slot-sync extracted velocity from the
    iterate's POSITION_LEVEL output) and multi-slot (average-and-
    broadcast already averaged the velocity) cases. Without this the
    particle would be stuck at the ``cloth_predict`` predicted
    position (cube falls through pinned cloth under mass splitting).

    Static nodes are skipped (slot's access_mode was stamped STATIC by
    broadcast for those).
    """
    node_id = wp.tid()
    if node_id >= copy_state.section_end.shape[0]:
        return
    # Fast bail via the cached count: empty / unbuilt nodes do nothing.
    count = copy_state.count_per_node[node_id]
    if count <= wp.int32(0):
        return
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    if copy_state.access_mode[start] == _ACCESS_MODE_STATIC:
        return

    # Synchronize slot[0] to VELOCITY_LEVEL before writeback. Sibling
    # slots already match after the prior average + broadcast pass.
    slot_synchronize_to_velocity_level(bodies, particles, copy_state, node_id, start, num_bodies, inv_dt)

    vel = copy_state.velocity[start]
    if node_id < num_bodies:
        bodies.velocity[node_id] = vel
        bodies.angular_velocity[node_id] = copy_state.angular_velocity[start]
    else:
        p = node_id - num_bodies
        particles.velocity[p] = vel
        # Fold the rigid-style integrate into the writeback for particles:
        # particle has no later ``pos += vel * dt`` step. Use the
        # substep-start anchor + the just-written velocity to recover the
        # final position. Identical to ``body.Position += body.Velocity *
        # dt`` (where body.Position at this point is the substep-start
        # value) in C# ``Integrate`` running after ``CopyStateIntoRigids``.
        dt = wp.float32(1.0) / inv_dt
        particles.position[p] = particles.position_prev_substep[p] + dt * vel


_MASS_SPLITTING_GROUPED_NODES_PER_BLOCK: int = 8
_MASS_SPLITTING_GROUPED_LANES_PER_NODE: int = 4
_MASS_SPLITTING_GROUPED_LANES_PER_NODE_WP = wp.constant(wp.int32(_MASS_SPLITTING_GROUPED_LANES_PER_NODE))

# Per-node mass-splitting kernels iterate sequentially over a small
# slot range and have no cross-thread sync, so they follow the same
# rule as :data:`solver_phoenx._SINGLEWORLD_BLOCK_DIM`: one warp per
# block maximises blocks-in-flight per SM and hides the global memory
# latency on the slot reads / writes.
_MASS_SPLITTING_PER_NODE_BLOCK_DIM: int = 32


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
        block_dim=_MASS_SPLITTING_PER_NODE_BLOCK_DIM,
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
        block_dim=_MASS_SPLITTING_PER_NODE_BLOCK_DIM,
        device=copy_state.section_end.device,
    )


def launch_average_and_broadcast_grouped(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: int,
    inv_dt: float,
) -> None:
    """Launch grouped generic average/broadcast.

    Uses one 32-thread block for eight nodes, four lanes per node.
    """
    num_nodes = copy_state.section_end.shape[0]
    num_blocks = (num_nodes + _MASS_SPLITTING_GROUPED_NODES_PER_BLOCK - 1) // _MASS_SPLITTING_GROUPED_NODES_PER_BLOCK
    wp.launch_tiled(
        _average_and_broadcast_grouped_kernel,
        dim=(num_blocks,),
        inputs=[copy_state, bodies, particles, wp.int32(num_bodies), wp.float32(inv_dt)],
        block_dim=32,
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
        block_dim=_MASS_SPLITTING_PER_NODE_BLOCK_DIM,
        device=copy_state.section_end.device,
    )
