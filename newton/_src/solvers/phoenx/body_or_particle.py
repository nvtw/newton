# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified body-or-particle indexing for :class:`PhoenXWorld`.

PhoenX has two kinds of "things" constraint kernels can operate on:

* **Rigid bodies** -- six DOFs (position + orientation), the
  :class:`~newton._src.solvers.phoenx.body.BodyContainer` SoA.
* **Particles** -- three DOFs (position only; orientation collapses
  to identity, angular velocity collapses to zero). The
  :class:`~newton._src.solvers.phoenx.particle.ParticleContainer`
  SoA.

Cloth distance constraints, soft-body tetrahedra, particle-rigid
attachments, and rigid-rigid contacts all want to read "the
position of the thing at index ``i``" without caring whether ``i``
is a body or a particle. This module is the access layer that
makes that work -- the same way the joint-or-contact dispatcher in
``solver_phoenx_kernels.py:789, 824, ...`` lets one cid space cover
both kinds of constraint via a threshold compare.

## Index convention

```
body indices    : [0,         num_bodies)
particle indices: [num_bodies, num_bodies + num_particles)
```

A unified index ``i`` resolves to a body slot when
``i < num_bodies`` and to a particle slot at ``i - num_bodies``
otherwise. ``num_bodies`` is a warp-uniform constant baked into
the kernel launch -- the threshold compare predicts perfectly and
inlines into a single ``cmp + bra`` after Warp's compile pass. For
*typed* kernels (rigid-rigid contact never sees a particle index,
particle-particle distance never sees a body index) the wrong
branch is dead code and Warp eliminates it entirely.

## What's wasted for particles

Reading orientation / angular_velocity for a particle goes through
this module's :func:`get_orientation` / :func:`get_angular_velocity`,
which return ``wp.quat_identity()`` and ``wp.vec3f(0)``
respectively. The constraint math then operates on those
degenerate values -- correctly, since no rotation means no
angular impulse contribution. The cost is a few ALU ops on the
particle branch; no memory waste in
:class:`ParticleContainer` (the fields don't exist there).

## Access-mode compatibility

Both :class:`~newton._src.solvers.phoenx.body.BodyContainer` and
:class:`~newton._src.solvers.phoenx.particle.ParticleContainer`
carry per-entity ``access_mode`` / ``position_prev_substep``
(plus ``orientation_prev_substep`` for bodies) snapshots, so the
generic helpers in
:mod:`~newton._src.solvers.phoenx.access_mode` work uniformly
across both. The body case runs the full 6-DoF finite-diff;
the particle case is the 3-DoF subset. See
:func:`writeback_position_to_velocity` /
:func:`set_access_mode` for the unified-index entry points.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_POSITION_LEVEL,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_pose_velocity,
    synchronize_position_velocity,
)
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "BodyOrParticleStore",
    "body_set_access_mode",
    "get_angular_velocity",
    "get_inverse_mass",
    "get_orientation",
    "get_position",
    "get_velocity",
    "is_particle",
    "particle_set_access_mode",
    "set_access_mode",
    "set_position",
    "set_velocity",
    "writeback_position_to_velocity",
]


_ACCESS_MODE_VELOCITY_LEVEL = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))
_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_ACCESS_MODE_STATIC = wp.constant(wp.int32(ACCESS_MODE_STATIC))


@wp.struct
class BodyOrParticleStore:
    """Bundle of :class:`BodyContainer` + :class:`ParticleContainer`
    + the body count, kernel-visible.

    Constraint kernels that want to address either kind of "thing"
    take this struct as a single argument -- the unified accessors
    branch on ``num_bodies`` internally so the kernel signature
    stays a one-liner regardless of how many particle / body roles
    a constraint plays. Kernels that only ever address rigid bodies
    can keep the existing ``bodies: BodyContainer`` parameter; this
    struct is opt-in.

    Attributes:
        bodies: Rigid-body SoA. Length = ``num_bodies``.
        particles: Particle SoA. Length =
            ``num_particles``. May be a length-1 sentinel for scenes
            with no particles -- the unified-index branch is gated
            on the threshold so an unused particle container costs
            nothing.
        num_bodies: Plus-one of the highest body index in use.
            Unified indices ``< num_bodies`` resolve to bodies;
            ``>= num_bodies`` to particles.
    """

    bodies: BodyContainer
    particles: ParticleContainer
    num_bodies: wp.int32


# ---------------------------------------------------------------------------
# Read accessors. Branch internally on the threshold.
# ---------------------------------------------------------------------------
#
# All five getters follow the same pattern:
#
#   if i < store.num_bodies:
#       return store.bodies.<field>[i]
#   return store.particles.<field>[i - store.num_bodies]   # or degenerate value
#
# Particles don't have orientation / angular velocity, so those
# accessors return identity / zero on the particle branch. Inlined
# by Warp; warp-uniform branch on ``num_bodies``.


@wp.func
def is_particle(store: BodyOrParticleStore, i: wp.int32) -> wp.bool:
    """``True`` iff ``i`` is a particle index."""
    return i >= store.num_bodies


@wp.func
def get_position(store: BodyOrParticleStore, i: wp.int32) -> wp.vec3f:
    """World-space position of the body / particle at unified index
    ``i``."""
    if i < store.num_bodies:
        return store.bodies.position[i]
    return store.particles.position[i - store.num_bodies]


@wp.func
def get_orientation(store: BodyOrParticleStore, i: wp.int32) -> wp.quatf:
    """World-space orientation. Returns identity for particles
    (which have no orientation degree of freedom)."""
    if i < store.num_bodies:
        return store.bodies.orientation[i]
    return wp.quatf(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))


@wp.func
def get_velocity(store: BodyOrParticleStore, i: wp.int32) -> wp.vec3f:
    """World-space linear velocity."""
    if i < store.num_bodies:
        return store.bodies.velocity[i]
    return store.particles.velocity[i - store.num_bodies]


@wp.func
def get_angular_velocity(store: BodyOrParticleStore, i: wp.int32) -> wp.vec3f:
    """World-space angular velocity. Returns zero for particles."""
    if i < store.num_bodies:
        return store.bodies.angular_velocity[i]
    return wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))


@wp.func
def get_inverse_mass(store: BodyOrParticleStore, i: wp.int32) -> wp.float32:
    """Inverse mass ``[1/kg]``. ``0.0`` marks the body / particle as
    infinitely massive (anchor / pinned)."""
    if i < store.num_bodies:
        return store.bodies.inverse_mass[i]
    return store.particles.inverse_mass[i - store.num_bodies]


# ---------------------------------------------------------------------------
# Write accessors. Same branch shape; the particle write is a no-op
# on the orientation / angular-velocity setters, which the
# constraint kernel must not call for particles anyway.
# ---------------------------------------------------------------------------


@wp.func
def set_position(store: BodyOrParticleStore, i: wp.int32, p: wp.vec3f):
    """Write ``p`` into the body's / particle's position slot and flip
    its access mode to ``POSITION_LEVEL``.

    The access-mode flip is the contract that guarantees any
    subsequent reader (another constraint kernel or the substep-end
    recovery) sees the position write reflected in ``velocity`` after
    the next ``synchronize_*`` flip. ``STATIC`` entities skip the
    flag update so pinned particles / bodies stay short-circuited.
    """
    if i < store.num_bodies:
        store.bodies.position[i] = p
        if store.bodies.access_mode[i] != _ACCESS_MODE_STATIC:
            store.bodies.access_mode[i] = _ACCESS_MODE_POSITION_LEVEL
    else:
        i_p = i - store.num_bodies
        store.particles.position[i_p] = p
        if store.particles.access_mode[i_p] != _ACCESS_MODE_STATIC:
            store.particles.access_mode[i_p] = _ACCESS_MODE_POSITION_LEVEL


@wp.func
def set_velocity(store: BodyOrParticleStore, i: wp.int32, v: wp.vec3f):
    """Write ``v`` into the body's / particle's velocity slot and
    flip its access mode to ``VELOCITY_LEVEL``.

    Symmetric counterpart to :func:`set_position`. The access-mode
    flip ensures the substep-end recovery treats this entity as
    velocity-authoritative; if a position-level constraint touches
    the same entity afterwards it must call its own
    ``set_position`` (which flips back) so the contract holds at
    every constraint boundary.
    """
    if i < store.num_bodies:
        store.bodies.velocity[i] = v
        if store.bodies.access_mode[i] != _ACCESS_MODE_STATIC:
            store.bodies.access_mode[i] = _ACCESS_MODE_VELOCITY_LEVEL
    else:
        i_p = i - store.num_bodies
        store.particles.velocity[i_p] = v
        if store.particles.access_mode[i_p] != _ACCESS_MODE_STATIC:
            store.particles.access_mode[i_p] = _ACCESS_MODE_VELOCITY_LEVEL


@wp.func
def body_set_access_mode(
    bodies: BodyContainer,
    b: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Switch one rigid body's access mode, integrating the dual fields.

    SoA wrapper around
    :func:`~newton._src.solvers.phoenx.access_mode.synchronize_pose_velocity`:
    reads the current dual state out of the body container, runs the
    Jitter2-style synchronize, and scatters the result back. No-op
    when the body is already in ``new_access_mode``.

    Mirrors C# ``TinyRigidState.SetAccessMode`` (TinyRigidState.cs:108):
    a thin SoA-aware wrapper over the value-based math.
    """
    p_new, q_new, v_new, w_new, mode_new = synchronize_pose_velocity(
        bodies.position[b],
        bodies.orientation[b],
        bodies.velocity[b],
        bodies.angular_velocity[b],
        bodies.position_prev_substep[b],
        bodies.orientation_prev_substep[b],
        bodies.access_mode[b],
        new_access_mode,
        inv_dt,
    )
    bodies.position[b] = p_new
    bodies.orientation[b] = q_new
    bodies.velocity[b] = v_new
    bodies.angular_velocity[b] = w_new
    bodies.access_mode[b] = mode_new


@wp.func
def particle_set_access_mode(
    particles: ParticleContainer,
    p: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Switch one particle's access mode, integrating position / velocity.

    SoA wrapper around
    :func:`~newton._src.solvers.phoenx.access_mode.synchronize_position_velocity`.
    Same role as :func:`body_set_access_mode` but for the 3-DoF
    particle case.
    """
    pos_new, vel_new, mode_new = synchronize_position_velocity(
        particles.position[p],
        particles.velocity[p],
        particles.position_prev_substep[p],
        particles.access_mode[p],
        new_access_mode,
        inv_dt,
    )
    particles.position[p] = pos_new
    particles.velocity[p] = vel_new
    particles.access_mode[p] = mode_new


@wp.func
def set_access_mode(
    store: BodyOrParticleStore,
    i: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Unified-index entry point for :func:`body_set_access_mode` /
    :func:`particle_set_access_mode`.

    Branches on ``is_particle(store, i)`` and dispatches to the
    matching SoA wrapper. Constraint kernels that operate on unified
    body-or-particle indices call this; specialised kernels that
    only ever touch one kind should call the typed helper directly
    to avoid the branch.
    """
    if is_particle(store, i):
        particle_set_access_mode(store.particles, i - store.num_bodies, new_access_mode, inv_dt)
        return
    body_set_access_mode(store.bodies, i, new_access_mode, inv_dt)


@wp.func
def writeback_position_to_velocity(
    store: BodyOrParticleStore,
    i: wp.int32,
    inv_dt: wp.float32,
):
    """Force one entity back to ``VELOCITY_LEVEL``.

    Substep-exit recovery: every entity ``i`` ends the substep in
    ``VELOCITY_LEVEL`` so the next substep starts from a
    velocity-consistent state. Equivalent to Jitter2's
    ``TinyRigidState.WriteBack`` (TinyRigidState.cs:92-100), which
    just calls ``SynchronizeVelAndPosStateUpdates(VelocityLevel,
    ...)`` before scattering the velocity back. The ``Position ->
    Velocity`` branch of
    :func:`~newton._src.solvers.phoenx.access_mode.synchronize_pose_velocity`
    runs the linear + quaternion finite-diff; entities already in
    ``VELOCITY_LEVEL`` short-circuit.
    """
    set_access_mode(store, i, _ACCESS_MODE_VELOCITY_LEVEL, inv_dt)
