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

## Mass-splitting / position-pass compatibility

The :class:`~newton._src.solvers.phoenx.mass_splitting.TinyRigidState`
struct has both orientation and angular-velocity fields. Particles
ride on the same struct: orientation = identity, angular_velocity
= 0. The XPBD finite-difference recovery in
:func:`~newton._src.solvers.phoenx.mass_splitting.state.tiny_rigid_state_synchronize`
runs the quaternion math through unchanged -- the
``identity * conj(identity) = identity`` algebra produces zero
angular velocity for particles, which is correct. The 28 bytes of
"unused" orientation + angular_velocity per particle TinyState
slot are the cost of the unified state struct; a separate
``TinyParticleState`` would save them but at the price of parallel
buffers + parallel kernels. Defer until profiling demands it.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "BodyOrParticleStore",
    "get_angular_velocity",
    "get_inverse_mass",
    "get_orientation",
    "get_position",
    "get_velocity",
    "is_particle",
    "set_position",
    "set_velocity",
]


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
    """Write ``p`` into the body's / particle's position slot.
    Mirrors :func:`get_position`."""
    if i < store.num_bodies:
        store.bodies.position[i] = p
    else:
        store.particles.position[i - store.num_bodies] = p


@wp.func
def set_velocity(store: BodyOrParticleStore, i: wp.int32, v: wp.vec3f):
    """Write ``v`` into the body's / particle's velocity slot."""
    if i < store.num_bodies:
        store.bodies.velocity[i] = v
    else:
        store.particles.velocity[i - store.num_bodies] = v
