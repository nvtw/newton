# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Slot-aware read/write helpers for constraint kernels.

Direct port of the C# ``ConstraintHelper.ReadState`` /
``ConstraintHelper.WriteState`` pattern
(``CudaKernels/Constraints/ConstraintHelper.cs:153-232``).

Purpose: keep ONE set of constraint kernels that work both with mass
splitting OFF and ON. The constraint kernel reads / writes through
these helpers, which dispatch to either:

* The direct body / particle storage (``BodyContainer`` /
  ``ParticleContainer``) when the node has no copy state slot for
  this ``parallel_id``. This is the disabled-fast-path AND the
  static-body / not-in-graph fallback.
* A specific slot in :class:`CopyStateContainer` when the body
  participates in mass splitting and the ``parallel_id`` hits one of
  its allocated slots.

The helpers return ``(value, inv_factor, slot)``:

* ``inv_factor = count`` — the number of partition copies this node
  has. ``1`` when the helper fell through to direct storage. The
  constraint kernel scales ``inv_mass`` and ``inv_inertia`` by
  ``inv_factor`` so a body in N slots sees ``1/N`` of its inertia
  per slot — the Tonge mass-split effective-mass relation. Static
  bodies (``inv_mass == 0``) zero out the scaled inv mass / inertia
  regardless of ``inv_factor``, so the no-slot fallback's
  ``inv_factor=1`` is safe even if the body is static.
* ``slot`` — the copy-state slot index for the matching
  :func:`write_*_unified` call. ``-1`` means "fell through to
  direct"; the writer then routes back to the body / particle.

This file does NOT yet refactor any constraint kernel — it lands the
helpers + unit tests so the routing contract is locked in before the
larger kernel refactor (Step 4b / Step 6 of the plan).
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
    "add_position_correction_unified_all_slots",
    "get_state_index",
    "read_angular_velocity_unified",
    "read_orientation_unified",
    "read_position_unified",
    "read_velocity_unified",
    "set_access_mode_unified",
    "slot_synchronize_to_velocity_level",
    "write_angular_velocity_unified",
    "write_orientation_unified",
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

    Returns ``(slot, inv_factor)``:

    * ``slot == -1`` — no slot. Caller falls back to direct
      body / particle storage. ``inv_factor == 1`` (no mass scaling).
      Hit when: mass splitting disabled (``highest_index_in_use[0] ==
      0``), ``node_id`` out of range, the node has zero slots, or the
      requested ``parallel_id`` isn't among the node's allocated
      partition keys.
    * ``slot >= 0`` — slot index into the SoA arrays. ``inv_factor ==
      count`` — total slot count of this node. The constraint kernel
      scales ``inv_mass`` / ``inv_inertia`` by ``inv_factor`` so each
      slot sees ``mass / count`` — the Tonge mass-split effective-mass.
    """
    # Disabled fast path: zero slots populated. Single int load + compare.
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        return wp.int32(-1), wp.int32(1)
    if node_id < wp.int32(0) or node_id >= copy_state.section_end.shape[0]:
        return wp.int32(-1), wp.int32(1)
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    end = copy_state.section_end[node_id]
    count = end - start
    if count == wp.int32(0):
        # Node in range but has no slots (static body / not in graph). The
        # constraint kernel can still read its velocity from direct storage,
        # though typically static bodies are filtered before getting here.
        return wp.int32(-1), wp.int32(1)
    # Fast path for ``parallel_id == 0``: partition keys are emitted as
    # sorted (ascending) ints, and ``0`` (regular colour) is always the
    # smallest. If present, it lives at ``partition_list[start]``;
    # otherwise the body is overflow-only. Skipping the binary search
    # here saves the bulk of the lookup cost for regular-colour
    # iterates (which are the majority of contacts in dense scenes).
    if parallel_id == wp.int32(0):
        if copy_state.partition_list[start] == wp.int32(0):
            return start, count
        return wp.int32(-1), wp.int32(1)
    local = _binary_search_partition_list(copy_state.partition_list, start, end, parallel_id)
    if local < wp.int32(0):
        # Slots exist but none match this parallel_id (this constraint isn't
        # in the partition this body belongs to). Fall through to direct
        # storage so the read still returns the body's velocity. Mass
        # splitting only affects bodies/parallel_ids that actually appear
        # together in the interaction graph.
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


@wp.func
def add_position_correction_unified_all_slots(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    node_id: wp.int32,
    num_bodies: wp.int32,
    idt: wp.float32,
    dx: wp.vec3f,
):
    """Apply an XPBD position-level *delta* ``dx`` to every slot of
    ``node_id`` so :func:`launch_average_and_broadcast` preserves it
    (instead of diluting by ``1/N``).

    Velocity-level Tonge constraints (contacts) rely on the per-slot
    impulse being scaled by ``inv_factor`` so the post-average
    velocity matches the unsplit single-iter result. XPBD
    position-level constraints (cloth-triangle / cloth-bending /
    soft-tet elasticity) fall through that math because the ``1/N``
    mass scaling cancels out of the per-slot ``dx`` (XPBD lambda is
    ``N x`` bigger with ``inv_mass * N``; ``dx = lambda * grad *
    inv_mass * N`` gives ``dx_per_slot = dx_unsplit``). Without
    compensation, after averaging the particle would only see
    ``dx_unsplit / N``.

    Per-slot encoding follows that slot's current access mode so
    *both* the cloth contribution and any concurrent velocity-level
    contact contribution survive:

    * ``ACCESS_MODE_VELOCITY_LEVEL`` slot: encode ``dx`` as a velocity
      delta ``+= dx * idt``. The slot's existing ``vel`` (contact
      impulses) is preserved.
    * ``ACCESS_MODE_POSITION_LEVEL`` slot: add ``dx`` to ``position``
      directly. The slot's existing position-level state is preserved.
    * ``ACCESS_MODE_STATIC``: skip (pinned / world anchor).
    * ``ACCESS_MODE_NONE``: stamp ``VELOCITY_LEVEL`` and encode as
      velocity, mirroring how :func:`particle_set_access_mode` treats
      the uninitialised state.

    No-slot fallback (``highest_index_in_use[0] == 0`` or no slots for
    this node) writes the delta directly to body / particle storage.
    """
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        if node_id < num_bodies:
            bodies.position[node_id] = bodies.position[node_id] + dx
        else:
            p = node_id - num_bodies
            particles.position[p] = particles.position[p] + dx
        return
    if node_id < wp.int32(0) or node_id >= copy_state.section_end.shape[0]:
        return
    start = wp.int32(0)
    if node_id > wp.int32(0):
        start = copy_state.section_end[node_id - wp.int32(1)]
    end = copy_state.section_end[node_id]
    if start >= end:
        if node_id < num_bodies:
            bodies.position[node_id] = bodies.position[node_id] + dx
        else:
            p = node_id - num_bodies
            particles.position[p] = particles.position[p] + dx
        return
    dvel = dx * idt
    _NONE = wp.int32(0)
    _VEL = wp.int32(1)
    _POS = wp.int32(2)
    _STATIC = wp.int32(3)
    for s in range(start, end):
        mode = copy_state.access_mode[s]
        if mode == _STATIC:
            continue
        if mode == _POS:
            copy_state.position[s] = copy_state.position[s] + dx
        else:
            # VELOCITY_LEVEL (or NONE → stamp as VELOCITY_LEVEL).
            copy_state.velocity[s] = copy_state.velocity[s] + dvel
            if mode == _NONE:
                copy_state.access_mode[s] = _VEL


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
