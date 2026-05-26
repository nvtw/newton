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
    SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    get_state_index,
    read_angular_velocity_unified,
    read_particle_velocity_unified,
    read_velocity_unified,
    set_access_mode_unified,
    set_particle_access_mode_unified,
    write_angular_velocity_unified,
    write_particle_velocity_with_slot,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "contact_endpoint_apply_impulse",
    "contact_endpoint_inv_mass_along",
    "contact_endpoint_set_access_mode",
    "contact_endpoint_velocity_at_point",
]


@wp.func
def contact_endpoint_velocity_at_point(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point_world: wp.vec3f,
) -> wp.vec3f:
    """Velocity of this side at the contact point (world frame).

    Rigid: ``v + omega x r``, where ``r = p_world - bodies.position[b]``.
    Cloth: barycentric-weighted particle velocities (3 nodes).
    Soft-tet: 4-node barycentric-weighted particle velocities. The 4th
    weight is derived as ``1 - bary[0] - bary[1] - bary[2]`` (the bary
    is stored as 3 floats; the iterate recovers the 4th on the fly).
    Static-anchor (``nodes[0] < 0``) returns zero.

    Mass-splitting aware: each velocity read routes through
    :func:`read_velocity_unified` (unified body-or-particle index) so
    the constraint sees the per-``parallel_id`` copy-state slot when
    one exists. Particles are first-class mass-splitting participants
    (same pattern as rigid bodies): cloth-triangle and soft-tet
    endpoints read every node via :func:`read_velocity_unified` too.
    Falls through to direct ``bodies.velocity`` / ``particles.velocity``
    reads when no slot is allocated (mass-splitting disabled, regular-
    colour constraint, or node outside the interaction graph).
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        v_a, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[0], nodes[0] - num_bodies, parallel_id)
        v_b, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[1], nodes[1] - num_bodies, parallel_id)
        v_c, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[2], nodes[2] - num_bodies, parallel_id)
        return bary[0] * v_a + bary[1] * v_b + bary[2] * v_c
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        v_a, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[0], nodes[0] - num_bodies, parallel_id)
        v_b, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[1], nodes[1] - num_bodies, parallel_id)
        v_c, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[2], nodes[2] - num_bodies, parallel_id)
        v_d, _, _ = read_particle_velocity_unified(particles, copy_state, nodes[3], nodes[3] - num_bodies, parallel_id)
        bary_d = wp.float32(1.0) - bary[0] - bary[1] - bary[2]
        return bary[0] * v_a + bary[1] * v_b + bary[2] * v_c + bary_d * v_d
    b = nodes[0]
    if b < 0:
        return wp.vec3f(0.0, 0.0, 0.0)
    v_lin, _, _ = read_velocity_unified(bodies, particles, copy_state, b, parallel_id, num_bodies)
    v_ang, _, _ = read_angular_velocity_unified(bodies, copy_state, b, parallel_id, num_bodies)
    # Position is read from ``bodies`` directly; ``copy_state.position``
    # is the broadcast-time snapshot and is intentionally not used for
    # lever arms (matches the rigid-rigid contact iterate's pattern).
    r = contact_point_world - bodies.position[b]
    return v_lin + wp.cross(v_ang, r)


@wp.func
def contact_endpoint_inv_mass_along(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point_world: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    """This side's contribution to the row's effective inverse mass:
    ``J . M^-1 . J^T`` along ``direction``.

    Rigid: ``inv_mass + (r x d) . inv_inertia_world . (r x d)``.
    Cloth: ``sum_i bary_i^2 * inv_mass_i`` (3 nodes; no orientation).
    Soft-tet: ``sum_i bary_i^2 * inv_mass_i`` over 4 nodes (4th weight
    derived as ``1 - sum(bary[0..2])``).
    Static-anchor / pinned-particle (``inv_mass == 0``) returns 0.

    Mass-splitting aware: each per-node ``inverse_mass`` is scaled by
    that ``(node, parallel_id)`` slot's ``inv_factor`` (Tonge:
    ``mass_per_slot = mass / N``, so ``inv_mass_per_slot = inv_mass *
    N``). Particles use the same pattern as rigid bodies. With
    splitting disabled or the node outside the interaction graph,
    ``inv_factor = 1`` and the result matches the unscaled legacy
    formula.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        _, inv_f_a = get_state_index(copy_state, nodes[0], parallel_id)
        _, inv_f_b = get_state_index(copy_state, nodes[1], parallel_id)
        _, inv_f_c = get_state_index(copy_state, nodes[2], parallel_id)
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        return (
            bary[0] * bary[0] * particles.inverse_mass[p_a] * wp.float32(inv_f_a)
            + bary[1] * bary[1] * particles.inverse_mass[p_b] * wp.float32(inv_f_b)
            + bary[2] * bary[2] * particles.inverse_mass[p_c] * wp.float32(inv_f_c)
        )
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        _, inv_f_a = get_state_index(copy_state, nodes[0], parallel_id)
        _, inv_f_b = get_state_index(copy_state, nodes[1], parallel_id)
        _, inv_f_c = get_state_index(copy_state, nodes[2], parallel_id)
        _, inv_f_d = get_state_index(copy_state, nodes[3], parallel_id)
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        p_d = nodes[3] - num_bodies
        bary_d = wp.float32(1.0) - bary[0] - bary[1] - bary[2]
        return (
            bary[0] * bary[0] * particles.inverse_mass[p_a] * wp.float32(inv_f_a)
            + bary[1] * bary[1] * particles.inverse_mass[p_b] * wp.float32(inv_f_b)
            + bary[2] * bary[2] * particles.inverse_mass[p_c] * wp.float32(inv_f_c)
            + bary_d * bary_d * particles.inverse_mass[p_d] * wp.float32(inv_f_d)
        )
    b = nodes[0]
    if b < 0:
        return wp.float32(0.0)
    inv_m = bodies.inverse_mass[b]
    if inv_m == wp.float32(0.0):
        return wp.float32(0.0)
    # Rigid side IS in the mass-splitting graph; scale ``inv_mass`` /
    # ``inv_inertia`` by the body's slot count so the per-slot row
    # sees ``mass / N`` (Tonge).
    _, inv_factor = get_state_index(copy_state, b, parallel_id)
    inv_f = wp.float32(inv_factor)
    r = contact_point_world - bodies.position[b]
    rc = wp.cross(r, direction)
    inv_i = bodies.inverse_inertia_world[b] * inv_f
    return inv_m * inv_f + wp.dot(rc, inv_i @ rc)


@wp.func
def contact_endpoint_apply_impulse(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    contact_point_world: wp.vec3f,
    impulse: wp.vec3f,
):
    """Scatter a 3D impulse vector onto this side's nodes.

    The caller picks the impulse sign per side (typically
    ``J_side0 = -direction * lambda_dt`` and
    ``J_side1 = +direction * lambda_dt``).

    Rigid: ``v += J * inv_mass``,
    ``omega += inv_inertia_world . (r x J)``.
    Cloth: each node ``i`` gets ``v_i += bary_i * J * inv_mass_i``
    (3 nodes).
    Soft-tet: same per-node formula over 4 nodes (4th weight derived
    as ``1 - sum(bary[0..2])``).
    Static-anchor / pinned-particle is a no-op.

    Mass-splitting aware: each read / write routes through
    :func:`read_velocity_unified` / :func:`write_velocity_unified` so
    the impulse lands in the correct ``(node, parallel_id)`` slot.
    ``inverse_mass`` / ``inverse_inertia_world`` are scaled by that
    slot's ``inv_factor`` so the per-slot velocity delta matches the
    Tonge ``mass_per_slot = mass / N`` accounting; after
    :func:`launch_average_and_broadcast` reduces the N slots back to
    the body / particle, the net impulse on the entity equals what a
    non-split run would have produced. Particles use the same pattern
    as rigid bodies.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        inv_m_a_raw = particles.inverse_mass[p_a]
        inv_m_b_raw = particles.inverse_mass[p_b]
        inv_m_c_raw = particles.inverse_mass[p_c]
        if inv_m_a_raw > wp.float32(0.0):
            v_a, inv_f_a, slot_a = read_particle_velocity_unified(particles, copy_state, nodes[0], p_a, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_a,
                slot_a,
                v_a + bary[0] * impulse * inv_m_a_raw * wp.float32(inv_f_a),
            )
        if inv_m_b_raw > wp.float32(0.0):
            v_b, inv_f_b, slot_b = read_particle_velocity_unified(particles, copy_state, nodes[1], p_b, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_b,
                slot_b,
                v_b + bary[1] * impulse * inv_m_b_raw * wp.float32(inv_f_b),
            )
        if inv_m_c_raw > wp.float32(0.0):
            v_c, inv_f_c, slot_c = read_particle_velocity_unified(particles, copy_state, nodes[2], p_c, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_c,
                slot_c,
                v_c + bary[2] * impulse * inv_m_c_raw * wp.float32(inv_f_c),
            )
        return
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        p_d = nodes[3] - num_bodies
        bary_d = wp.float32(1.0) - bary[0] - bary[1] - bary[2]
        inv_m_a_raw = particles.inverse_mass[p_a]
        inv_m_b_raw = particles.inverse_mass[p_b]
        inv_m_c_raw = particles.inverse_mass[p_c]
        inv_m_d_raw = particles.inverse_mass[p_d]
        if inv_m_a_raw > wp.float32(0.0):
            v_a, inv_f_a, slot_a = read_particle_velocity_unified(particles, copy_state, nodes[0], p_a, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_a,
                slot_a,
                v_a + bary[0] * impulse * inv_m_a_raw * wp.float32(inv_f_a),
            )
        if inv_m_b_raw > wp.float32(0.0):
            v_b, inv_f_b, slot_b = read_particle_velocity_unified(particles, copy_state, nodes[1], p_b, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_b,
                slot_b,
                v_b + bary[1] * impulse * inv_m_b_raw * wp.float32(inv_f_b),
            )
        if inv_m_c_raw > wp.float32(0.0):
            v_c, inv_f_c, slot_c = read_particle_velocity_unified(particles, copy_state, nodes[2], p_c, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_c,
                slot_c,
                v_c + bary[2] * impulse * inv_m_c_raw * wp.float32(inv_f_c),
            )
        if inv_m_d_raw > wp.float32(0.0):
            v_d, inv_f_d, slot_d = read_particle_velocity_unified(particles, copy_state, nodes[3], p_d, parallel_id)
            write_particle_velocity_with_slot(
                particles,
                copy_state,
                p_d,
                slot_d,
                v_d + bary_d * impulse * inv_m_d_raw * wp.float32(inv_f_d),
            )
        return
    b = nodes[0]
    if b < 0:
        return
    inv_m_raw = bodies.inverse_mass[b]
    if inv_m_raw == wp.float32(0.0):
        return
    v_lin, inv_f, slot = read_velocity_unified(bodies, particles, copy_state, b, parallel_id, num_bodies)
    v_ang, _, _ = read_angular_velocity_unified(bodies, copy_state, b, parallel_id, num_bodies)
    inv_f_f = wp.float32(inv_f)
    inv_m = inv_m_raw * inv_f_f
    inv_i = bodies.inverse_inertia_world[b] * inv_f_f
    r = contact_point_world - bodies.position[b]
    write_velocity_unified(bodies, particles, copy_state, b, slot, num_bodies, v_lin + impulse * inv_m)
    write_angular_velocity_unified(bodies, copy_state, b, slot, v_ang + inv_i @ wp.cross(r, impulse))


@wp.func
def contact_endpoint_set_access_mode(
    kind: wp.int32,
    nodes: wp.vec4i,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Flip every node on this side to ``new_access_mode``.

    Cloth: flip all three particles. Soft-tet: flip all four. Rigid:
    flip the body (if ``nodes[0] >= 0``; static-anchor world shapes
    are no-ops). Routes through :func:`set_access_mode_unified` so
    when mass splitting is engaged the per-slot dual state is
    synchronised in lockstep with the body / particle storage — same
    pattern used by every other PhoenX constraint kernel. Without
    splitting the unified helper falls through to direct body /
    particle access-mode flips.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        set_particle_access_mode_unified(
            particles, copy_state, nodes[0], nodes[0] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        set_particle_access_mode_unified(
            particles, copy_state, nodes[1], nodes[1] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        set_particle_access_mode_unified(
            particles, copy_state, nodes[2], nodes[2] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        return
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        set_particle_access_mode_unified(
            particles, copy_state, nodes[0], nodes[0] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        set_particle_access_mode_unified(
            particles, copy_state, nodes[1], nodes[1] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        set_particle_access_mode_unified(
            particles, copy_state, nodes[2], nodes[2] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        set_particle_access_mode_unified(
            particles, copy_state, nodes[3], nodes[3] - num_bodies, parallel_id, new_access_mode, inv_dt
        )
        return
    b = nodes[0]
    if b < 0:
        return
    set_access_mode_unified(bodies, particles, copy_state, b, parallel_id, num_bodies, new_access_mode, inv_dt)
