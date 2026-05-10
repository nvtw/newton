# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified per-side helpers for cloth-aware rigid contacts.

The contact iterate (phase 5) reads each side's velocity-at-contact-
point, builds the effective inverse mass along the row direction,
solves a 1D PGS row, and scatters the resulting impulse back. This
module factors the *side*-specific bits (rigid body vs cloth-triangle
endpoint) out of the row math so:

* the rigid-rigid PGS row math (Box2D-v3 soft + warm-started lambdas
  + Coulomb cone) stays unchanged;
* cloth-rigid and cloth-cloth contacts share the same PGS row math;
* the per-side helpers below are the only place that branches on the
  endpoint kind.

The helpers are wp.func, value-based, and parameterised by:

* ``kind`` -- ``SHAPE_ENDPOINT_KIND_RIGID`` (0) or
  ``SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE`` (1)
* ``nodes`` -- ``vec3i`` of unified body-or-particle indices
  (``[0, num_bodies)`` for rigid bodies, ``[num_bodies, num_bodies +
  num_particles)`` for particles)
* ``bary`` -- barycentric weights ``(alpha, beta, gamma)`` against
  ``nodes[0..3]``; meaningful only when ``kind == CLOTH``

## Derivation cross-check vs Jitter2 ``RigidTetContact``

Jitter2's ``RigidTetContactManifold.PrepareForIteration``
(``RigidTetContactManifold.cs:522`` -- ``625``) computes the
effective mass for a contact with one rigid body (b1) + N deformable
nodes (3 for triangle, 4 for tet). For a row direction ``d`` (one of
``n / t1 / t2``):

::

    M_eff = 1 / (rigid_term + sum_i deformable_node_term_i)
    rigid_term = invFactor1 * (invMass1 + (r x d) . invInertia . (r x d))
    deformable_node_term_i = invFactor_i * bary_i^2 * invMass_i

With ``invFactor = 1`` (no mass-splitting), this matches our derivation
line-for-line.

Jitter2's per-contact iterate
(``RigidTetContactManifold.cs:771`` -- ``789``) computes
``Jv = side1.v_at_p . d - rigid.(v + omega x rw) . d`` (note the
relative-velocity sign convention: side2 - side1 in Jitter2 = our
side1 - side0). The PGS row gives ``lambda_dt = -M_eff * (Jv + bias)``
and impulses are scattered:

* ``v1 += -lambda_dt * d * invM1`` (rigid; -d on side 0)
* ``w1 += -invI . (r x d) * lambda_dt``
* ``v_node_i += +bary_i * d * lambda_dt * invM_i`` (cloth nodes; +d on side 1)

Our convention picks ``J_side0 = -d * lambda_dt`` and
``J_side1 = +d * lambda_dt`` so ``apply_endpoint_impulse`` takes the
already-signed impulse vector and writes back via the standard
mass-weighted formulas.

For the rigid-rigid degenerate case, ``M_eff_recip = side0_term +
side1_term`` reduces to the existing :func:`effective_mass_scalar`
(``inv_mass1 + inv_mass2 + (r1 x d).invI1.(r1 x d) + (r2 x d).invI2.(r2 x d)``).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.cloth_collision import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
)
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "contact_endpoint_apply_impulse",
    "contact_endpoint_inv_mass_along",
    "contact_endpoint_velocity_at_point",
]


@wp.func
def contact_endpoint_velocity_at_point(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    contact_point_world: wp.vec3f,
) -> wp.vec3f:
    """Velocity of this side at the contact point (world frame).

    Rigid: ``v + omega x r``, where ``r = p_world - bodies.position[b]``.
    Cloth: barycentric-weighted particle velocities.
    Static-anchor (``nodes[0] < 0``) returns zero.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        return (
            bary[0] * particles.velocity[p_a]
            + bary[1] * particles.velocity[p_b]
            + bary[2] * particles.velocity[p_c]
        )
    b = nodes[0]
    if b < 0:
        return wp.vec3f(0.0, 0.0, 0.0)
    r = contact_point_world - bodies.position[b]
    return bodies.velocity[b] + wp.cross(bodies.angular_velocity[b], r)


@wp.func
def contact_endpoint_inv_mass_along(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    contact_point_world: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    """This side's contribution to the row's effective inverse mass:
    ``J . M^-1 . J^T`` along ``direction``.

    Rigid: ``inv_mass + (r x d) . inv_inertia_world . (r x d)``.
    Cloth: ``sum_i bary_i^2 * inv_mass_i`` (no orientation, no
    angular term).
    Static-anchor / pinned-particle (``inv_mass == 0``) returns 0.

    The full contact's effective mass is ``1 / (side0_term +
    side1_term)``. Reduces to :func:`effective_mass_scalar`'s
    denominator in the rigid-rigid case (verified algebraically
    against the existing PhoenX rigid contact path).
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        return (
            bary[0] * bary[0] * particles.inverse_mass[p_a]
            + bary[1] * bary[1] * particles.inverse_mass[p_b]
            + bary[2] * bary[2] * particles.inverse_mass[p_c]
        )
    b = nodes[0]
    if b < 0:
        return wp.float32(0.0)
    inv_m = bodies.inverse_mass[b]
    if inv_m == wp.float32(0.0):
        return wp.float32(0.0)
    r = contact_point_world - bodies.position[b]
    rc = wp.cross(r, direction)
    inv_i = bodies.inverse_inertia_world[b]
    return inv_m + wp.dot(rc, inv_i @ rc)


@wp.func
def contact_endpoint_apply_impulse(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    contact_point_world: wp.vec3f,
    impulse: wp.vec3f,
):
    """Scatter a 3D impulse vector onto this side's nodes.

    The caller picks the impulse sign per side (typically
    ``J_side0 = -direction * lambda_dt`` and
    ``J_side1 = +direction * lambda_dt``).

    Rigid: ``v += J * inv_mass``,
    ``omega += inv_inertia_world . (r x J)``.
    Cloth: each node ``i`` gets ``v_i += bary_i * J * inv_mass_i``.
    Static-anchor / pinned-particle is a no-op.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        inv_m_a = particles.inverse_mass[p_a]
        inv_m_b = particles.inverse_mass[p_b]
        inv_m_c = particles.inverse_mass[p_c]
        if inv_m_a > wp.float32(0.0):
            particles.velocity[p_a] = particles.velocity[p_a] + bary[0] * impulse * inv_m_a
        if inv_m_b > wp.float32(0.0):
            particles.velocity[p_b] = particles.velocity[p_b] + bary[1] * impulse * inv_m_b
        if inv_m_c > wp.float32(0.0):
            particles.velocity[p_c] = particles.velocity[p_c] + bary[2] * impulse * inv_m_c
        return
    b = nodes[0]
    if b < 0:
        return
    inv_m = bodies.inverse_mass[b]
    if inv_m == wp.float32(0.0):
        return
    r = contact_point_world - bodies.position[b]
    inv_i = bodies.inverse_inertia_world[b]
    bodies.velocity[b] = bodies.velocity[b] + impulse * inv_m
    bodies.angular_velocity[b] = bodies.angular_velocity[b] + inv_i @ wp.cross(r, impulse)
