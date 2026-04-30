# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle SoA storage for :class:`PhoenXWorld`.

Sibling of :class:`~newton._src.solvers.phoenx.body.BodyContainer`, but
strictly smaller: particles are point masses with no orientation /
angular velocity / inertia. Cloth nodes, soft-body nodes, fluid
markers, and tracer points all map onto this same SoA -- exactly the
particle abstraction the rest of Newton uses
(``Model.particle_q`` / ``particle_qd`` / ``particle_inv_mass``).

This module is the *data* half of the unified body-or-particle
indexing scheme; the *access* half lives in
:mod:`~newton._src.solvers.phoenx.body_or_particle`. The
two together let constraint kernels address either kind of "thing"
through a single integer index, the same way the
:func:`~newton._src.solvers.phoenx.constraints.contact_ingest`
dispatcher already addresses joints + contacts via a unified ``cid``
threshold (``cid < num_joints`` -> joint, otherwise contact).

Convention -- struct-of-arrays, one ``wp.array`` per field, one row
per particle. Same as ``BodyContainer`` so kernels can be written
the same way.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "PARTICLE_FLAG_FIXED",
    "PARTICLE_FLAG_NONE",
    "ParticleContainer",
    "particle_container_zeros",
    "particle_predict_position",
    "particle_recover_velocity",
]


# Particle flags. Mirrors Newton's :class:`~newton.ParticleFlags`
# convention (a bitmask) but kept narrow for now -- only the
# load-bearing flag for cloth (pinning) is exposed at this layer.
# Stored as ``int32`` in the container; kernels mask in the hot path.

#: No flags set. Particle is fully dynamic.
PARTICLE_FLAG_NONE = wp.constant(0)

#: Particle is *fixed* in place: its pose is held constant
#: regardless of forces / impulses applied to it. Cloth pinning
#: (corner of a flag pinned to a pole, hem stitched to a body) is
#: the canonical use case. The solver treats fixed particles like
#: static rigid bodies (``inverse_mass = 0`` shorthand): impulses
#: are still computed against them so neighbouring particles feel
#: the constraint, but the particle itself is never displaced.
PARTICLE_FLAG_FIXED = wp.constant(1)


@wp.struct
class ParticleContainer:
    """Struct-of-arrays storage for a batch of particles.

    Each field is a 1-D ``wp.array`` of length ``num_particles``
    (must match across fields). Constructed on the host with
    :func:`particle_container_zeros`; kernels read / write the
    fields directly, e.g. ``particles.position[i]``.

    Indexing convention
    -------------------
    The :class:`ParticleContainer` is indexed by a *particle slot*
    ``i_p in [0, num_particles)``, the same way the
    :class:`~newton._src.solvers.phoenx.body.BodyContainer` is
    indexed by a body slot. The unified body-or-particle index
    used at the constraint level (see
    :mod:`newton._src.solvers.phoenx.body_or_particle`) maps
    ``unified_index = num_bodies + i_p``; constraint kernels that
    care about both kinds of "thing" go through that helper rather
    than indexing this container directly.
    """

    #: Particle position in world space [m]. Mirrors
    #: ``Model.particle_q``.
    position: wp.array[wp.vec3f]

    #: Particle velocity in world space [m/s]. Mirrors
    #: ``Model.particle_qd``.
    velocity: wp.array[wp.vec3f]

    #: Inverse mass [1/kg]. ``0.0`` marks the particle as
    #: infinitely massive (pinned / kinematic-equivalent). Mirrors
    #: ``Model.particle_inv_mass``.
    inverse_mass: wp.array[wp.float32]

    #: External force accumulator [N]. The solver-side substep
    #: pre-pass adds this onto ``velocity`` (gravity / wind /
    #: picking) before constraint iterations and clears it after.
    #: Same role :attr:`BodyContainer.force` plays for rigid bodies.
    force: wp.array[wp.vec3f]

    #: Bitmask of :data:`PARTICLE_FLAG_*` constants. Hot-path
    #: kernels mask in this field per particle to gate behaviour
    #: (fixed particle skips the velocity update etc.).
    flags: wp.array[wp.int32]

    #: World id (multi-world layout). One particle belongs to
    #: exactly one world; the solver's per-world dispatch reads
    #: this to filter. ``0`` for single-world scenes.
    world_id: wp.array[wp.int32]

    #: Position at substep entry (pre-predict). Written by the
    #: substep-entry predict kernel (Velocity -> Position
    #: transition); read by the substep-exit recover kernel to
    #: derive ``velocity = (position - position_substep_start) *
    #: inv_dt``. Mirrors the C# ``TinyRigidState`` access-mode
    #: pattern -- the iterate operates on
    #: :attr:`position` directly, and the access-mode flip back to
    #: velocity-level at the end of the substep automatically
    #: recovers velocity from the position delta. See
    #: :func:`particle_predict_position` /
    #: :func:`particle_recover_velocity`.
    position_substep_start: wp.array[wp.vec3f]


def particle_container_zeros(
    num_particles: int,
    device: wp.DeviceLike = None,
) -> ParticleContainer:
    """Allocate a zero-initialised :class:`ParticleContainer`.

    Defaults match the canonical "fully dynamic, no external force,
    in world 0" particle: ``inverse_mass = 0`` (caller must set this
    -- particles are inert until the user supplies a mass), every
    other field zeroed. The caller fills in real positions /
    velocities / inverse masses via array assignment, same way
    ``body_container_zeros`` is used.

    Args:
        num_particles: Particle slot capacity. Must be ``>= 1``;
            zero-particle scenes pass a length-1 sentinel and gate
            on ``num_particles == 0`` at the call site.
        device: Warp device the buffers live on.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1 (got {num_particles}); pass 1 for empty-sentinel allocation")
    p = ParticleContainer()
    p.position = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.velocity = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.inverse_mass = wp.zeros(num_particles, dtype=wp.float32, device=device)
    p.force = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.flags = wp.zeros(num_particles, dtype=wp.int32, device=device)
    p.world_id = wp.zeros(num_particles, dtype=wp.int32, device=device)
    p.position_substep_start = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    return p


# ---------------------------------------------------------------------------
# Per-substep predict / recover helpers (access-mode pattern).
# ---------------------------------------------------------------------------
#
# Particles operate at position-level inside the cloth iterate (the
# iterate writes :attr:`ParticleContainer.position` directly), but the
# physics state at the substep boundaries is velocity-level (Newton's
# ``Model.particle_qd`` is a velocity, the integrator's ``v += g*dt``
# step works on velocity). So each substep does two transitions:
#
# 1. Substep entry: Velocity-level -> Position-level.
#    Apply forces / gravity to ``velocity``, snapshot the pre-predict
#    position into ``position_substep_start``, then advance
#    ``position`` by ``velocity * dt``. The cloth iterate sees the
#    advanced position and projects it onto constraint manifolds.
#
# 2. Substep exit: Position-level -> Velocity-level.
#    Recover ``velocity = (position - position_substep_start) * inv_dt``
#    so the next substep starts from a consistent velocity-level state.
#
# These mirror the C# ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
# transitions for rigid bodies (BodyTypes.cs:268-304); ours are
# simpler because particles don't have orientation.


@wp.func
def particle_predict_position(
    position: wp.vec3f,
    velocity: wp.vec3f,
    dt: wp.float32,
):
    """Substep-entry transition: Velocity-level -> Position-level.

    Returns ``(position_advanced, position_substep_start)``. The
    caller writes ``position_advanced`` back to
    :attr:`ParticleContainer.position` and snapshots
    ``position_substep_start`` into
    :attr:`ParticleContainer.position_substep_start`. ``velocity``
    must already include the per-substep gravity / external-force
    contribution; this helper does *not* mutate it.

    Mirrors the C# ``Velocity -> Position`` branch of
    ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
    (BodyTypes.cs:294-301): ``Position = bodyState.Position +
    dt * Velocity``.
    """
    return position + dt * velocity, position


@wp.func
def particle_recover_velocity(
    position: wp.vec3f,
    position_substep_start: wp.vec3f,
    inv_dt: wp.float32,
) -> wp.vec3f:
    """Substep-exit transition: Position-level -> Velocity-level.

    Mirrors the C# ``Position -> Velocity`` branch of
    ``TinyRigidState.SynchronizeVelAndPosStateUpdates``
    (BodyTypes.cs:282-290): ``Velocity = (Position -
    bodyState.Position) * invDt``. The cloth iterate has already
    written the constraint-projected position into ``position``;
    this helper just folds the delta over the substep into the
    velocity field.

    Returns the new velocity by value; caller writes it back to
    :attr:`ParticleContainer.velocity`.
    """
    return (position - position_substep_start) * inv_dt
