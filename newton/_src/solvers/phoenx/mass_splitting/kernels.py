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
    num_bodies: wp.int32,
):
    """Average velocity / angular_velocity across a node's slots and
    broadcast the result.

    Mirrors the C# ``MassSplittingRigidBodyInteractionGraphGpu``
    ``AverageAndBroadcast`` method (``MassSplittingTypes.cs:206``):
    bodies with a single slot (or zero) are a no-op; bodies with N>1
    slots average their per-slot velocity / angular_velocity, scale by
    ``1/N``, and write the average to every slot. Position /
    orientation stay at the broadcast-time forward-integrated value.

    For particle nodes the angular_velocity sum is over zeros so the
    write is a no-op for them on that field — harmless.
    """
    node_id = wp.tid()
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return
    start, end = _section_range(copy_state, node_id)
    count = end - start
    # No copies (count==0) or only one copy (count==1, averaging a
    # singleton is a no-op): nothing to do.
    if count <= wp.int32(1):
        return

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
        # Force the access mode back to VELOCITY_LEVEL: any constraint
        # iterate that left a slot at POSITION_LEVEL gets unified back
        # to the velocity dual after the average.
        copy_state.access_mode[slot] = _ACCESS_MODE_VELOCITY_LEVEL


@wp.kernel(enable_backward=False)
def _copy_state_into_rigids_kernel(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
):
    """Write the first slot's velocity back to body / particle.

    All slots carry the same averaged velocity after
    :func:`launch_average_and_broadcast`, so the first slot is
    canonical. Mirrors C# ``WriteBack`` (``MassSplittingTypes.cs:282``)
    + ``TinyRigidState.WriteBack`` (``BodyTypes.cs:309``): only velocity
    travels back; position / orientation are advanced separately by the
    solver's integrate step.

    Static nodes are intentionally skipped (their slot would have been
    stamped ``ACCESS_MODE_STATIC`` by broadcast).
    """
    node_id = wp.tid()
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return
    start, end = _section_range(copy_state, node_id)
    if start >= end:
        return
    if copy_state.access_mode[start] == _ACCESS_MODE_STATIC:
        return

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
    num_bodies: int,
) -> None:
    """Launch :func:`_average_and_broadcast_kernel`."""
    num_nodes = copy_state.section_end.shape[0]
    wp.launch(
        _average_and_broadcast_kernel,
        dim=num_nodes,
        inputs=[copy_state, wp.int32(num_bodies)],
        device=copy_state.section_end.device,
    )


def launch_copy_state_into_rigids(
    copy_state: CopyStateContainer,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: int,
) -> None:
    """Launch :func:`_copy_state_into_rigids_kernel`."""
    num_nodes = copy_state.section_end.shape[0]
    wp.launch(
        _copy_state_into_rigids_kernel,
        dim=num_nodes,
        inputs=[copy_state, bodies, particles, wp.int32(num_bodies)],
        device=copy_state.section_end.device,
    )
