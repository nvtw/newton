# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal particle SoA storage for PhoenX cloth.

Slim 4-field container sized for the standalone cloth pipeline -- no
access-mode tag, no world id, no force accumulator. Pinning is encoded
as ``inverse_mass == 0`` (the iterate's particle-mass-weighted projection
naturally leaves zero-inverse-mass nodes alone).
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "ParticleContainer",
    "particle_container_zeros",
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
    #: damping term as ``dx = pos - pos_prev_substep``.
    position_prev_substep: wp.array[wp.vec3f]


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
    return p
