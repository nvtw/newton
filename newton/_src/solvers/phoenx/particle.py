# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle SoA storage for PhoenX deformables.

World ids live outside the hot SoA; pinning is ``inverse_mass == 0``.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import (
    ACCESS_MODE_NONE,
    ACCESS_MODE_STATIC,
    ACCESS_MODE_VELOCITY_LEVEL,
    synchronize_position_velocity,
)

__all__ = [
    "ParticleContainer",
    "particle_container_zeros",
    "particle_set_access_mode",
]


@wp.struct
class ParticleContainer:
    """Struct-of-arrays storage for a batch of cloth particles.

    Each field is a 1-D ``wp.array`` of length ``num_particles``.
    """

    #: Particle position in world space [m].
    position: wp.array[wp.vec3f]

    #: Particle velocity in world space [m/s].
    velocity: wp.array[wp.vec3f]

    #: Inverse mass [1/kg]. ``0.0`` marks the particle as pinned.
    inverse_mass: wp.array[wp.float32]

    #: Position at substep entry (pre-predict). Read by the XPBD
    #: damping term as ``dx = pos - pos_prev_substep`` and by the
    #: :mod:`newton._src.solvers.phoenx.access_mode` synchronize helper
    #: as the finite-diff anchor for the position->velocity flip.
    position_prev_substep: wp.array[wp.vec3f]

    #: Per-particle access-mode tag (see
    #: :mod:`newton._src.solvers.phoenx.access_mode`). Predict sets
    #: ``VELOCITY_LEVEL`` for dynamic particles and ``STATIC`` for
    #: pinned (``inv_mass == 0``).
    access_mode: wp.array[wp.int32]


def particle_container_zeros(
    num_particles: int,
    device: wp.context.Devicelike = None,
) -> ParticleContainer:
    """Allocate a zero-initialised :class:`ParticleContainer`."""
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1 (got {num_particles})")
    p = ParticleContainer()
    p.position = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.velocity = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.inverse_mass = wp.zeros(num_particles, dtype=wp.float32, device=device)
    p.position_prev_substep = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    p.access_mode = wp.full(num_particles, value=int(ACCESS_MODE_VELOCITY_LEVEL), dtype=wp.int32, device=device)
    return p


@wp.func
def particle_set_access_mode(
    particles: ParticleContainer,
    p: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Lazy SoA wrapper around :func:`synchronize_position_velocity`.

    Same hot-path optimisation as :func:`body_set_access_mode`: gate
    every other field read behind a single ``access_mode[p]`` load so
    the no-op flip (current already matches new) costs one int read.
    """
    current = particles.access_mode[p]
    if current == new_access_mode:
        return
    if current == ACCESS_MODE_STATIC:
        return
    if current == ACCESS_MODE_NONE:
        particles.access_mode[p] = new_access_mode
        return
    pos_new, vel_new, mode_new = synchronize_position_velocity(
        particles.position[p],
        particles.velocity[p],
        particles.position_prev_substep[p],
        current,
        new_access_mode,
        inv_dt,
    )
    particles.position[p] = pos_new
    particles.velocity[p] = vel_new
    particles.access_mode[p] = mode_new
