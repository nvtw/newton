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

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "get_state_index",
    "read_angular_velocity_unified",
    "read_orientation_unified",
    "read_position_unified",
    "read_velocity_unified",
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
