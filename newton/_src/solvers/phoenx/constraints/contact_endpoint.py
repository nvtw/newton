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

from newton._src.solvers.phoenx.body import MOTION_ARTICULATED, BodyContainer, mat33_from_sym6
from newton._src.solvers.phoenx.cloth_collision import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON,
)
from newton._src.solvers.phoenx.helpers.math_helpers import apply_body_spatial_impulse
from newton._src.solvers.phoenx.mass_splitting.access import (
    get_state_index,
    read_angular_velocity_with_slot,
    set_access_mode_with_slot,
    set_particle_access_mode_with_slot,
    write_angular_velocity_unified,
    write_particle_velocity_with_slot,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

_vec6 = wp.types.vector(length=6, dtype=wp.float32)


__all__ = [
    "contact_endpoint_apply_impulse",
    "contact_endpoint_apply_impulse_cached",
    "contact_endpoint_inv_mass_along",
    "contact_endpoint_inv_mass_along_cached",
    "contact_endpoint_set_access_mode",
    "contact_endpoint_set_access_mode_cached",
    "contact_endpoint_velocity_at_point",
    "contact_endpoint_velocity_at_point_cached",
]


@wp.func
def _read_particle_velocity_with_slot(
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    particle_id: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    if slot < wp.int32(0):
        return particles.velocity[particle_id]
    return copy_state.velocity[slot]


@wp.func
def _read_body_velocity_with_slot(
    bodies: BodyContainer,
    copy_state: CopyStateContainer,
    body_id: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    if slot < wp.int32(0):
        return bodies.velocity[body_id]
    return copy_state.velocity[slot]


@wp.func
def _articulation_pair_wrench_response(
    bodies: BodyContainer,
    body_slot0: wp.int32,
    wrench0: wp.spatial_vector,
    body_slot1: wp.int32,
    wrench1: wp.spatial_vector,
    apply: wp.bool,
) -> wp.float32:
    """Solve one articulation response to one or two world-origin wrenches."""
    data = bodies.reduced
    articulation = data.body_articulation[body_slot0]
    if articulation < wp.int32(0):
        return wp.float32(0.0)

    start = data.articulation_start[articulation]
    end = data.articulation_end[articulation]
    for joint in range(start, end):
        child = data.joint_child[joint]
        data.body_work[child] = wp.spatial_vector()
        data.body_acceleration[child] = wp.spatial_vector()

    for side in range(2):
        body_slot = body_slot0
        wrench = wrench0
        if side == 1:
            body_slot = body_slot1
            wrench = wrench1
        if body_slot >= wp.int32(0):
            target_body = body_slot - wp.int32(1)
            data.body_work[target_body] = data.body_work[target_body] - wrench

    for reverse in range(end - start):
        joint = end - wp.int32(1) - reverse
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        p = data.body_work[child]
        reduced_force = _vec6(0.0)
        d_inv_u = _vec6(0.0)

        for row in range(6):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                reduced_force[row] = -wp.dot(data.joint_s[dof], p)
                data.joint_work[dof] = reduced_force[row]
        for row in range(6):
            if wp.int32(row) < dof_count:
                for column in range(6):
                    if wp.int32(column) < dof_count:
                        d_inv_u[row] += data.joint_d_inv[joint, row, column] * reduced_force[column]

        propagated = p
        for column in range(6):
            if wp.int32(column) < dof_count:
                propagated += data.joint_u[dof_start + wp.int32(column)] * d_inv_u[column]
        if parent >= wp.int32(0):
            data.body_work[parent] = data.body_work[parent] + propagated

    effective_inverse_mass = wp.float32(0.0)
    for joint in range(start, end):
        parent = data.joint_parent[joint]
        child = data.joint_child[joint]
        dof_start = data.joint_qd_start[joint]
        dof_end = data.joint_qd_start[joint + wp.int32(1)]
        dof_count = dof_end - dof_start
        parent_acceleration = wp.spatial_vector()
        if parent >= wp.int32(0):
            parent_acceleration = data.body_acceleration[parent]

        rhs = _vec6(0.0)
        response = _vec6(0.0)
        for row in range(6):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                rhs[row] = data.joint_work[dof] - wp.dot(data.joint_u[dof], parent_acceleration)
        for row in range(6):
            if wp.int32(row) < dof_count:
                for column in range(6):
                    if wp.int32(column) < dof_count:
                        response[row] += data.joint_d_inv[joint, row, column] * rhs[column]

        child_acceleration = parent_acceleration
        for row in range(6):
            if wp.int32(row) < dof_count:
                dof = dof_start + wp.int32(row)
                data.generalized_response[dof] = response[row]
                effective_inverse_mass += data.joint_work[dof] * response[row]
                child_acceleration += data.joint_s[dof] * response[row]
        data.body_acceleration[child] = child_acceleration

    if apply:
        for joint in range(start, end):
            dof_start = data.joint_qd_start[joint]
            dof_end = data.joint_qd_start[joint + wp.int32(1)]
            for dof in range(dof_start, dof_end):
                data.joint_qd[dof] += data.generalized_response[dof]

        # Reconstruct every link twist because one impulse generally changes
        # all descendants and ancestors in the articulation.
        for joint in range(start, end):
            parent = data.joint_parent[joint]
            child = data.joint_child[joint]
            twist = wp.spatial_vector()
            if parent >= wp.int32(0):
                twist = data.body_acceleration[parent]
            dof_start = data.joint_qd_start[joint]
            dof_end = data.joint_qd_start[joint + wp.int32(1)]
            for dof in range(dof_start, dof_end):
                twist += data.joint_s[dof] * data.joint_qd[dof]
            data.body_acceleration[child] = twist
            slot = child + wp.int32(1)
            omega = wp.spatial_bottom(twist)
            bodies.angular_velocity[slot] = omega
            bodies.velocity[slot] = wp.spatial_top(twist) + wp.cross(omega, bodies.position[slot])

    return effective_inverse_mass


@wp.func
def _articulation_pair_response(
    bodies: BodyContainer,
    body_slot0: wp.int32,
    point0: wp.vec3f,
    impulse0: wp.vec3f,
    body_slot1: wp.int32,
    point1: wp.vec3f,
    impulse1: wp.vec3f,
    apply: wp.bool,
) -> wp.float32:
    """Solve one articulation response to one or two point impulses."""
    return _articulation_pair_wrench_response(
        bodies,
        body_slot0,
        wp.spatial_vector(impulse0, wp.cross(point0, impulse0)),
        body_slot1,
        wp.spatial_vector(impulse1, wp.cross(point1, impulse1)),
        apply,
    )


@wp.func
def _articulation_point_response(
    bodies: BodyContainer,
    body_slot: wp.int32,
    point: wp.vec3f,
    impulse: wp.vec3f,
    apply: wp.bool,
) -> wp.float32:
    """Return point-impulse inverse mass and optionally update generalized speed."""
    return _articulation_pair_response(
        bodies,
        body_slot,
        point,
        impulse,
        wp.int32(-1),
        wp.vec3f(0.0),
        wp.vec3f(0.0),
        apply,
    )


@wp.func
def contact_endpoint_velocity_at_point_cached(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    slots: wp.vec4i,
    counts: wp.vec4i,
    contact_point_world: wp.vec3f,
) -> wp.vec3f:
    """Velocity of one contact side using pre-stamped endpoint slots."""
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        v_a = _read_particle_velocity_with_slot(particles, copy_state, p_a, slots[0])
        v_b = _read_particle_velocity_with_slot(particles, copy_state, p_b, slots[1])
        v_c = _read_particle_velocity_with_slot(particles, copy_state, p_c, slots[2])
        return bary[0] * v_a + bary[1] * v_b + bary[2] * v_c
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        weight_a = bary[0]
        weight_b = bary[1]
        weight_c = bary[2]
        weight_d = wp.float32(1.0) - weight_a - weight_b - weight_c
        v = wp.vec3f(0.0, 0.0, 0.0)
        if weight_a != wp.float32(0.0):
            v = v + weight_a * _read_particle_velocity_with_slot(particles, copy_state, nodes[0] - num_bodies, slots[0])
        if weight_b != wp.float32(0.0):
            v = v + weight_b * _read_particle_velocity_with_slot(particles, copy_state, nodes[1] - num_bodies, slots[1])
        if weight_c != wp.float32(0.0):
            v = v + weight_c * _read_particle_velocity_with_slot(particles, copy_state, nodes[2] - num_bodies, slots[2])
        if weight_d != wp.float32(0.0):
            v = v + weight_d * _read_particle_velocity_with_slot(particles, copy_state, nodes[3] - num_bodies, slots[3])
        return v
    b = nodes[0]
    if b < 0:
        return wp.vec3f(0.0, 0.0, 0.0)
    slot = slots[0]
    v_lin = _read_body_velocity_with_slot(bodies, copy_state, b, slot)
    v_ang = read_angular_velocity_with_slot(bodies, copy_state, b, slot)
    r = contact_point_world - bodies.position[b]
    return v_lin + wp.cross(v_ang, r)


@wp.func
def contact_endpoint_inv_mass_along_cached(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    slots: wp.vec4i,
    counts: wp.vec4i,
    contact_point_world: wp.vec3f,
    direction: wp.vec3f,
) -> wp.float32:
    """Endpoint contribution to ``J M^-1 J^T`` using cached counts."""
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        return (
            bary[0] * bary[0] * particles.inverse_mass[p_a] * wp.float32(counts[0])
            + bary[1] * bary[1] * particles.inverse_mass[p_b] * wp.float32(counts[1])
            + bary[2] * bary[2] * particles.inverse_mass[p_c] * wp.float32(counts[2])
        )
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        weight_a = bary[0]
        weight_b = bary[1]
        weight_c = bary[2]
        weight_d = wp.float32(1.0) - weight_a - weight_b - weight_c
        result = wp.float32(0.0)
        if weight_a != wp.float32(0.0):
            result = result + weight_a * weight_a * particles.inverse_mass[nodes[0] - num_bodies] * wp.float32(
                counts[0]
            )
        if weight_b != wp.float32(0.0):
            result = result + weight_b * weight_b * particles.inverse_mass[nodes[1] - num_bodies] * wp.float32(
                counts[1]
            )
        if weight_c != wp.float32(0.0):
            result = result + weight_c * weight_c * particles.inverse_mass[nodes[2] - num_bodies] * wp.float32(
                counts[2]
            )
        if weight_d != wp.float32(0.0):
            result = result + weight_d * weight_d * particles.inverse_mass[nodes[3] - num_bodies] * wp.float32(
                counts[3]
            )
        return result
    b = nodes[0]
    if b < 0:
        return wp.float32(0.0)
    if bodies.motion_type[b] == MOTION_ARTICULATED:
        return _articulation_point_response(bodies, b, contact_point_world, direction, wp.bool(False))
    inv_m = bodies.inverse_mass[b]
    if inv_m == wp.float32(0.0):
        return wp.float32(0.0)
    inv_f = wp.float32(counts[0])
    r = contact_point_world - bodies.position[b]
    rc = wp.cross(r, direction)
    inv_i = mat33_from_sym6(bodies.inverse_inertia_world[b]) * inv_f
    return inv_m * inv_f + wp.dot(rc, inv_i @ rc)


@wp.func
def contact_endpoint_apply_impulse_cached(
    kind: wp.int32,
    nodes: wp.vec4i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    slots: wp.vec4i,
    counts: wp.vec4i,
    contact_point_world: wp.vec3f,
    impulse: wp.vec3f,
):
    """Scatter a contact impulse using pre-stamped endpoint slots/counts."""
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        inv_m_a_raw = particles.inverse_mass[p_a]
        inv_m_b_raw = particles.inverse_mass[p_b]
        inv_m_c_raw = particles.inverse_mass[p_c]
        if inv_m_a_raw > wp.float32(0.0):
            v_a = _read_particle_velocity_with_slot(particles, copy_state, p_a, slots[0])
            write_particle_velocity_with_slot(
                particles, copy_state, p_a, slots[0], v_a + bary[0] * impulse * inv_m_a_raw * wp.float32(counts[0])
            )
        if inv_m_b_raw > wp.float32(0.0):
            v_b = _read_particle_velocity_with_slot(particles, copy_state, p_b, slots[1])
            write_particle_velocity_with_slot(
                particles, copy_state, p_b, slots[1], v_b + bary[1] * impulse * inv_m_b_raw * wp.float32(counts[1])
            )
        if inv_m_c_raw > wp.float32(0.0):
            v_c = _read_particle_velocity_with_slot(particles, copy_state, p_c, slots[2])
            write_particle_velocity_with_slot(
                particles, copy_state, p_c, slots[2], v_c + bary[2] * impulse * inv_m_c_raw * wp.float32(counts[2])
            )
        return
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        weight_a = bary[0]
        weight_b = bary[1]
        weight_c = bary[2]
        weight_d = wp.float32(1.0) - weight_a - weight_b - weight_c
        p_a = nodes[0] - num_bodies
        p_b = nodes[1] - num_bodies
        p_c = nodes[2] - num_bodies
        p_d = nodes[3] - num_bodies
        inv_m_a_raw = particles.inverse_mass[p_a]
        inv_m_b_raw = particles.inverse_mass[p_b]
        inv_m_c_raw = particles.inverse_mass[p_c]
        inv_m_d_raw = particles.inverse_mass[p_d]
        if weight_a != wp.float32(0.0) and inv_m_a_raw > wp.float32(0.0):
            v_a = _read_particle_velocity_with_slot(particles, copy_state, p_a, slots[0])
            write_particle_velocity_with_slot(
                particles, copy_state, p_a, slots[0], v_a + weight_a * impulse * inv_m_a_raw * wp.float32(counts[0])
            )
        if weight_b != wp.float32(0.0) and inv_m_b_raw > wp.float32(0.0):
            v_b = _read_particle_velocity_with_slot(particles, copy_state, p_b, slots[1])
            write_particle_velocity_with_slot(
                particles, copy_state, p_b, slots[1], v_b + weight_b * impulse * inv_m_b_raw * wp.float32(counts[1])
            )
        if weight_c != wp.float32(0.0) and inv_m_c_raw > wp.float32(0.0):
            v_c = _read_particle_velocity_with_slot(particles, copy_state, p_c, slots[2])
            write_particle_velocity_with_slot(
                particles, copy_state, p_c, slots[2], v_c + weight_c * impulse * inv_m_c_raw * wp.float32(counts[2])
            )
        if weight_d != wp.float32(0.0) and inv_m_d_raw > wp.float32(0.0):
            v_d = _read_particle_velocity_with_slot(particles, copy_state, p_d, slots[3])
            write_particle_velocity_with_slot(
                particles, copy_state, p_d, slots[3], v_d + weight_d * impulse * inv_m_d_raw * wp.float32(counts[3])
            )
        return
    b = nodes[0]
    if b < 0:
        return
    if bodies.motion_type[b] == MOTION_ARTICULATED:
        _articulation_point_response(bodies, b, contact_point_world, impulse, wp.bool(True))
        return
    inv_m_raw = bodies.inverse_mass[b]
    if inv_m_raw == wp.float32(0.0):
        return
    slot = slots[0]
    inv_f_f = wp.float32(counts[0])
    v_lin = _read_body_velocity_with_slot(bodies, copy_state, b, slot)
    v_ang = read_angular_velocity_with_slot(bodies, copy_state, b, slot)
    inv_m = inv_m_raw * inv_f_f
    inv_i = mat33_from_sym6(bodies.inverse_inertia_world[b]) * inv_f_f
    r = contact_point_world - bodies.position[b]
    v_lin_new, v_ang_new = apply_body_spatial_impulse(
        v_lin,
        v_ang,
        inv_m,
        inv_i,
        impulse,
        wp.cross(r, impulse),
    )
    write_velocity_unified(bodies, particles, copy_state, b, slot, num_bodies, v_lin_new)
    write_angular_velocity_unified(bodies, copy_state, b, slot, v_ang_new)


@wp.func
def contact_endpoint_set_access_mode_cached(
    kind: wp.int32,
    nodes: wp.vec4i,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    slots: wp.vec4i,
    counts: wp.vec4i,
    new_access_mode: wp.int32,
    inv_dt: wp.float32,
):
    """Flip every node on this side using cached endpoint slots."""
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[0] - num_bodies, slots[0], new_access_mode, inv_dt
        )
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[1] - num_bodies, slots[1], new_access_mode, inv_dt
        )
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[2] - num_bodies, slots[2], new_access_mode, inv_dt
        )
        return
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_SOFT_TETRAHEDRON):
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[0] - num_bodies, slots[0], new_access_mode, inv_dt
        )
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[1] - num_bodies, slots[1], new_access_mode, inv_dt
        )
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[2] - num_bodies, slots[2], new_access_mode, inv_dt
        )
        set_particle_access_mode_with_slot(
            particles, copy_state, nodes[3] - num_bodies, slots[3], new_access_mode, inv_dt
        )
        return
    b = nodes[0]
    if b < 0:
        return
    set_access_mode_with_slot(bodies, particles, copy_state, b, slots[0], num_bodies, new_access_mode, inv_dt)


@wp.func
def _contact_endpoint_slot_counts(
    copy_state: CopyStateContainer,
    nodes: wp.vec4i,
    parallel_id: wp.int32,
):
    slot0, count0 = get_state_index(copy_state, nodes[0], parallel_id)
    slot1, count1 = get_state_index(copy_state, nodes[1], parallel_id)
    slot2, count2 = get_state_index(copy_state, nodes[2], parallel_id)
    slot3, count3 = get_state_index(copy_state, nodes[3], parallel_id)
    return wp.vec4i(slot0, slot1, slot2, slot3), wp.vec4i(count0, count1, count2, count3)


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
    slots, counts = _contact_endpoint_slot_counts(copy_state, nodes, parallel_id)
    return contact_endpoint_velocity_at_point_cached(
        kind, nodes, bary, bodies, particles, copy_state, num_bodies, parallel_id, slots, counts, contact_point_world
    )


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
    slots, counts = _contact_endpoint_slot_counts(copy_state, nodes, parallel_id)
    return contact_endpoint_inv_mass_along_cached(
        kind,
        nodes,
        bary,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        slots,
        counts,
        contact_point_world,
        direction,
    )


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
    slots, counts = _contact_endpoint_slot_counts(copy_state, nodes, parallel_id)
    contact_endpoint_apply_impulse_cached(
        kind,
        nodes,
        bary,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        slots,
        counts,
        contact_point_world,
        impulse,
    )


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
    slots, counts = _contact_endpoint_slot_counts(copy_state, nodes, parallel_id)
    contact_endpoint_set_access_mode_cached(
        kind, nodes, bodies, particles, copy_state, num_bodies, parallel_id, slots, counts, new_access_mode, inv_dt
    )
