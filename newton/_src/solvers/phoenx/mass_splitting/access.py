# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Slot-aware read/write helpers for constraint kernels.

Port of C# ``ConstraintHelper.ReadState`` / ``WriteState``
(``CudaKernels/Constraints/ConstraintHelper.cs:153-232``). One kernel
serves both mass-splitting-off and -on: the helpers dispatch to direct
body/particle storage when the node has no copy-state slot for the
given ``parallel_id``, else to the matching :class:`CopyStateContainer`
slot.

Read helpers return ``(value, inv_factor, slot)``:

* ``inv_factor`` -- partition-copy count; the constraint kernel scales
  ``inv_mass`` / ``inv_inertia`` by this so a body in N slots sees
  ``1/N`` of its inertia per slot (Tonge effective-mass). ``1`` on the
  no-slot fallback. Static bodies' ``inv_mass == 0`` self-zero so the
  fallback is safe.
* ``slot`` -- index into the SoA arrays for the matching
  ``write_*_unified``. ``-1`` => fell through to direct storage.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    synchronize_pose_velocity,
    synchronize_position_velocity,
)
from newton._src.solvers.phoenx.body import BodyContainer, body_set_access_mode
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer, particle_set_access_mode

__all__ = [
    "get_state_index",
    "read_angular_velocity_unified",
    "read_orientation_unified",
    "read_particle_position_with_slot",
    "read_particle_velocity_unified",
    "read_position_unified",
    "read_position_with_slot",
    "read_velocity_unified",
    "set_access_mode_unified",
    "set_access_mode_with_slot",
    "set_particle_access_mode_unified",
    "set_particle_access_mode_with_slot",
    "slot_synchronize_to_velocity_level",
    "write_angular_velocity_unified",
    "write_orientation_unified",
    "write_particle_position_with_slot",
    "write_particle_velocity_with_slot",
    "write_position_unified",
    "write_velocity_unified",
]


@wp.func
def _binary_search_partition_list(
    partition_list: wp.array[wp.int32],
    start: wp.int32,
    end: wp.int32,
    target: wp.int32,
) -> wp.int32:
    """Binary search ``partition_list[start:end]`` for ``target``. Returns the
    slot index where the target lives, or -1 if not present.

    Direct port of ``MassSplittingTypes.cs:142-160`` ``BinarySearch``.
    Partition lists are always sorted ascending by build construction
    (radix sort on packed ``(node, partition)`` keys).
    """
    lo = start
    hi = end - wp.int32(1)
    while lo <= hi:
        mid = (lo + hi) >> wp.int32(1)
        v = partition_list[mid]
        if v < target:
            lo = mid + wp.int32(1)
        elif v > target:
            hi = mid - wp.int32(1)
        else:
            return mid
    return wp.int32(-1)


@wp.func
def get_state_index(
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
):
    """Locate the slot for ``(node_id, parallel_id)``.

    Returns ``(slot, inv_factor)``. ``slot == -1`` (with ``inv_factor
    == 1``) means no slot: caller falls back to direct body / particle
    storage. Hit when mass splitting is disabled, the node is out of
    range, or ``parallel_id`` isn't among the node's allocated keys.
    ``slot >= 0`` => index into the SoA arrays with ``inv_factor =
    count`` (total slots for this node, used for Tonge ``1/N`` mass
    scaling).

    Hot path: ``parallel_id == 0`` (regular colours) reads the cached
    ``slot_for_pid0`` / ``count_per_node`` arrays stamped by
    :func:`build_interaction_graph` -- one load each, no dependent
    section_end + partition_list lookup chain. Generic across all
    constraint types (joints, contacts, cloth, soft-tet, soft-hex).
    """
    if node_id < wp.int32(0) or node_id >= copy_state.section_end.shape[0]:
        return wp.int32(-1), wp.int32(1)
    # parallel_id == 0 (regular colour) is the common case: hit the
    # per-node cache. Both the "no slot" (mass splitting disabled or
    # node not in graph) and "single slot" cases resolve in one load
    # of ``slot_for_pid0``.
    if parallel_id == wp.int32(0):
        slot = copy_state.slot_for_pid0[node_id]
        if slot < wp.int32(0):
            return wp.int32(-1), wp.int32(1)
        return slot, copy_state.count_per_node[node_id]
    # Slow path: overflow colour (parallel_id > 0). Binary-search the
    # node's partition list.
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return wp.int32(-1), wp.int32(1)
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    end = copy_state.section_end[node_id]
    count = end - start
    if count == wp.int32(0):
        return wp.int32(-1), wp.int32(1)
    local = _binary_search_partition_list(copy_state.partition_list, start, end, parallel_id)
    if local < wp.int32(0):
        return wp.int32(-1), wp.int32(1)
    return local, count


# -----------------------------------------------------------------------------
# Read helpers. Each returns (value, inv_factor, slot).
# -----------------------------------------------------------------------------


@wp.func
def read_velocity_unified(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Read linear velocity for a unified-index node.

    Returns ``(velocity, inv_factor, slot)``. See module docstring for
    the contract; the constraint kernel scales its impulse by
    ``inv_factor``.
    """
    slot, inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        if node_id < num_bodies:
            return bodies.velocity[node_id], inv_factor, slot
        return particles.velocity[node_id - num_bodies], inv_factor, slot
    return copy_state.velocity[slot], inv_factor, slot


@wp.func
def read_particle_velocity_unified(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    particle_id: wp.int32,
    parallel_id: wp.int32,
):
    """Read linear velocity for a known particle endpoint."""
    slot, inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        return particles.velocity[particle_id], inv_factor, slot
    return copy_state.velocity[slot], inv_factor, slot


@wp.func
def read_angular_velocity_unified(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Read angular velocity for a unified-index node.

    Particles have no angular DOF; this helper is body-only. The
    constraint kernel knows from the unified index whether the node
    is a body or a particle and only calls this for body nodes.
    """
    slot, inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        return bodies.angular_velocity[node_id], inv_factor, slot
    return copy_state.angular_velocity[slot], inv_factor, slot


@wp.func
def read_angular_velocity_with_slot(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    """Read angular velocity using a precomputed slot index.

    Companion to :func:`read_velocity_unified`: callers that just paid
    for ``get_state_index`` can pass the resulting ``slot`` here to
    skip the second lookup. ``slot < 0`` means "no slot, use direct
    body storage".
    """
    if slot < wp.int32(0):
        return bodies.angular_velocity[node_id]
    return copy_state.angular_velocity[slot]


@wp.func
def read_position_with_slot(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    num_bodies: wp.int32,
) -> wp.vec3f:
    """Read position using a precomputed slot. Companion to
    :func:`get_state_index` so callers can do one slot lookup per
    node and reuse it for the access-mode flip + read + write."""
    if slot < wp.int32(0):
        if node_id < num_bodies:
            return bodies.position[node_id]
        return particles.position[node_id - num_bodies]
    return copy_state.position[slot]


@wp.func
def read_particle_position_with_slot(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    particle_id: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    """Read a particle position using a precomputed copy-state slot."""
    if slot < wp.int32(0):
        return particles.position[particle_id]
    return copy_state.position[slot]


@wp.func
def set_particle_access_mode_with_slot(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    particle_id: wp.int32,
    slot: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Particle-only slot-aware access-mode flip.

    This is the particle branch of :func:`set_access_mode_with_slot` for
    constraints whose endpoints are known particles (soft-tets / cloth).
    It preserves the same dual-state synchronization while skipping the
    rigid-body orientation path.
    """
    if slot < wp.int32(0):
        particle_set_access_mode(particles, particle_id, new_access_mode, inv_dt)
        return

    current = copy_state.access_mode[slot]
    if current == new_access_mode:
        return
    pos_prev = particles.position_prev_substep[particle_id]
    pos_new, vel_new, mode_new = synchronize_position_velocity(
        copy_state.position[slot],
        copy_state.velocity[slot],
        pos_prev,
        current,
        new_access_mode,
        inv_dt,
    )
    copy_state.position[slot] = pos_new
    copy_state.velocity[slot] = vel_new
    copy_state.access_mode[slot] = mode_new


@wp.func
def set_particle_access_mode_unified(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    particle_id: wp.int32,
    parallel_id: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Slot-aware access-mode flip for a known particle endpoint."""
    slot, _inv_factor = get_state_index(copy_state, node_id, parallel_id)
    set_particle_access_mode_with_slot(particles, copy_state, particle_id, slot, new_access_mode, inv_dt)


@wp.func
def set_access_mode_with_slot(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    num_bodies: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Slot-aware access-mode flip given a precomputed slot. Identical
    semantics to :func:`set_access_mode_unified` minus the redundant
    :func:`get_state_index` call. ``slot < 0`` falls through to the
    direct body / particle helper.
    """
    if slot < wp.int32(0):
        if node_id < num_bodies:
            body_set_access_mode(bodies, node_id, new_access_mode, inv_dt)
        else:
            particle_set_access_mode(particles, node_id - num_bodies, new_access_mode, inv_dt)
        return

    current = copy_state.access_mode[slot]
    if current == new_access_mode:
        return
    if node_id < num_bodies:
        pos_prev = bodies.position_prev_substep[node_id]
        orient_prev = bodies.orientation_prev_substep[node_id]
        pos_new, orient_new, vel_new, omega_new, mode_new = synchronize_pose_velocity(
            copy_state.position[slot],
            copy_state.orientation[slot],
            copy_state.velocity[slot],
            copy_state.angular_velocity[slot],
            pos_prev,
            orient_prev,
            current,
            new_access_mode,
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.orientation[slot] = orient_new
        copy_state.velocity[slot] = vel_new
        copy_state.angular_velocity[slot] = omega_new
        copy_state.access_mode[slot] = mode_new
    else:
        p = node_id - num_bodies
        pos_prev = particles.position_prev_substep[p]
        pos_new, vel_new, mode_new = synchronize_position_velocity(
            copy_state.position[slot],
            copy_state.velocity[slot],
            pos_prev,
            current,
            new_access_mode,
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.velocity[slot] = vel_new
        copy_state.access_mode[slot] = mode_new


@wp.func
def read_position_unified(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Read position for a unified-index node.

    For body nodes returns the slot's predicted position (forward-
    integrated at broadcast time) or the body's current position when
    the helper falls through. For particles, same semantics.
    """
    slot, inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        if node_id < num_bodies:
            return bodies.position[node_id], inv_factor, slot
        return particles.position[node_id - num_bodies], inv_factor, slot
    return copy_state.position[slot], inv_factor, slot


@wp.func
def read_orientation_unified(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Read orientation for a unified-index body node (particles have no
    orientation DOF)."""
    slot, inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        return bodies.orientation[node_id], inv_factor, slot
    return copy_state.orientation[slot], inv_factor, slot


# -----------------------------------------------------------------------------
# Write helpers. Each takes the slot index returned by the matching read; the
# helper routes to copy_state when slot >= 0 and to body/particle otherwise.
# -----------------------------------------------------------------------------


@wp.func
def write_velocity_unified(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    num_bodies: wp.int32,
    value: wp.vec3f,
):
    """Write linear velocity back to the slot or direct storage."""
    if slot < wp.int32(0):
        if node_id < num_bodies:
            bodies.velocity[node_id] = value
        else:
            particles.velocity[node_id - num_bodies] = value
        return
    copy_state.velocity[slot] = value


@wp.func
def write_angular_velocity_unified(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    value: wp.vec3f,
):
    """Write angular velocity back to the slot or direct storage (body-only)."""
    if slot < wp.int32(0):
        bodies.angular_velocity[node_id] = value
        return
    copy_state.angular_velocity[slot] = value


@wp.func
def write_particle_position_with_slot(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    particle_id: wp.int32,
    slot: wp.int32,
    value: wp.vec3f,
):
    """Write a particle position using a precomputed copy-state slot."""
    if slot < wp.int32(0):
        particles.position[particle_id] = value
        return
    copy_state.position[slot] = value


@wp.func
def write_particle_velocity_with_slot(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    particle_id: wp.int32,
    slot: wp.int32,
    value: wp.vec3f,
):
    """Write linear velocity for a known particle endpoint."""
    if slot < wp.int32(0):
        particles.velocity[particle_id] = value
        return
    copy_state.velocity[slot] = value


@wp.func
def write_position_unified(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    num_bodies: wp.int32,
    value: wp.vec3f,
):
    """Write position back to the slot or direct storage."""
    if slot < wp.int32(0):
        if node_id < num_bodies:
            bodies.position[node_id] = value
        else:
            particles.position[node_id - num_bodies] = value
        return
    copy_state.position[slot] = value


@wp.func
def write_orientation_unified(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    value: wp.quatf,
):
    """Write orientation back to the slot or direct storage (body-only)."""
    if slot < wp.int32(0):
        bodies.orientation[node_id] = value
        return
    copy_state.orientation[slot] = value


# -----------------------------------------------------------------------------
# Slot-aware access-mode flip. Mirrors Jitter2
# ``TinyRigidState.SetAccessMode`` (MassSplitting/TinyRigidState.cs:108) and
# the synchronize-on-flip semantics from
# ``SynchronizeVelAndPosStateUpdates``.
#
# Every constraint prepare / iterate must route the access-mode flip through
# this helper. The unified pattern: read state via ``read_*_unified``,
# compute, write via ``write_*_unified``, and synchronise via this helper
# whenever switching dual representation. Mass splitting is then automatic
# — the average / writeback kernels also call SetAccessMode(VelocityLevel)
# on each slot, which encodes any POSITION_LEVEL work as velocity deltas
# that the standard velocity average aggregates correctly.
# -----------------------------------------------------------------------------


@wp.func
def slot_synchronize_to_velocity_level(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    slot: wp.int32,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
):
    """Synchronise one slot to VELOCITY_LEVEL in place.

    Mirrors Jitter2's
    ``TinyRigidState.SynchronizeVelAndPosStateUpdates(VelocityLevel, ...)``
    used inside ``MassSplittingRigidBodyInteractionGraph.AverageAndBroadcast``
    and ``WriteBack``. If the slot is already at VELOCITY_LEVEL (or
    STATIC), no-op.

    The snapshot anchor for the synchronize is the body's (or particle's)
    ``position_prev_substep`` -- the substep-start state captured by the
    predict kernel before the broadcast.
    """
    if slot < wp.int32(0):
        return
    current = copy_state.access_mode[slot]
    # Fast path: VELOCITY_LEVEL / STATIC slots are no-ops on the
    # ``->VELOCITY_LEVEL`` flip. The original helper still read +
    # wrote all 5 fields with their unchanged values; skip those
    # writes so the average pass that follows doesn't pay for them.
    if current == wp.int32(1):  # ACCESS_MODE_VELOCITY_LEVEL
        return
    if node_id < num_bodies:
        pos_prev = bodies.position_prev_substep[node_id]
        orient_prev = bodies.orientation_prev_substep[node_id]
        pos_new, orient_new, vel_new, omega_new, mode_new = synchronize_pose_velocity(
            copy_state.position[slot],
            copy_state.orientation[slot],
            copy_state.velocity[slot],
            copy_state.angular_velocity[slot],
            pos_prev,
            orient_prev,
            current,
            wp.int32(1),  # ACCESS_MODE_VELOCITY_LEVEL
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.orientation[slot] = orient_new
        copy_state.velocity[slot] = vel_new
        copy_state.angular_velocity[slot] = omega_new
        copy_state.access_mode[slot] = mode_new
    else:
        p = node_id - num_bodies
        pos_prev = particles.position_prev_substep[p]
        pos_new, vel_new, mode_new = synchronize_position_velocity(
            copy_state.position[slot],
            copy_state.velocity[slot],
            pos_prev,
            current,
            wp.int32(1),  # ACCESS_MODE_VELOCITY_LEVEL
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.velocity[slot] = vel_new
        copy_state.access_mode[slot] = mode_new


@wp.func
def set_access_mode_unified(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Slot-aware access-mode flip used by constraint prepare / iterate.

    Routes through the per-(node, parallel_id) slot when one exists --
    synchronising the slot's dual state via the same arithmetic the
    body / particle helpers use, anchored on the body / particle's
    ``position_prev_substep`` (and ``orientation_prev_substep`` for
    bodies). When no slot exists (mass splitting off, or the node has
    no slot for this ``parallel_id``), falls through to
    :func:`body_set_access_mode` / :func:`particle_set_access_mode` --
    preserving the rigid-only path verbatim.

    Direct port of Jitter2's pattern in
    ``FemTetPBD.PrepareForIteration / Iterate``: every per-vertex
    access call is ``state.SetAccessMode(...)`` whether ``state`` is
    the body's VelState (no MS) or one of the slot's TinyRigidState
    (MS engaged).
    """
    slot, _inv_factor = get_state_index(copy_state, node_id, parallel_id)
    if slot < wp.int32(0):
        if node_id < num_bodies:
            body_set_access_mode(bodies, node_id, new_access_mode, inv_dt)
        else:
            particle_set_access_mode(particles, node_id - num_bodies, new_access_mode, inv_dt)
        return

    current = copy_state.access_mode[slot]
    if current == new_access_mode:
        return
    if node_id < num_bodies:
        pos_prev = bodies.position_prev_substep[node_id]
        orient_prev = bodies.orientation_prev_substep[node_id]
        pos_new, orient_new, vel_new, omega_new, mode_new = synchronize_pose_velocity(
            copy_state.position[slot],
            copy_state.orientation[slot],
            copy_state.velocity[slot],
            copy_state.angular_velocity[slot],
            pos_prev,
            orient_prev,
            current,
            new_access_mode,
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.orientation[slot] = orient_new
        copy_state.velocity[slot] = vel_new
        copy_state.angular_velocity[slot] = omega_new
        copy_state.access_mode[slot] = mode_new
    else:
        p = node_id - num_bodies
        pos_prev = particles.position_prev_substep[p]
        pos_new, vel_new, mode_new = synchronize_position_velocity(
            copy_state.position[slot],
            copy_state.velocity[slot],
            pos_prev,
            current,
            new_access_mode,
            inv_dt,
        )
        copy_state.position[slot] = pos_new
        copy_state.velocity[slot] = vel_new
        copy_state.access_mode[slot] = mode_new
