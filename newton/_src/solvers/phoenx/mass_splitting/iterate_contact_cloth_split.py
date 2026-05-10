# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split-aware variant of the cloth-aware rigid contact iterate.

Mirrors :func:`~newton._src.solvers.phoenx.constraints.constraint_contact_cloth.contact_iterate_at_cloth_aware`
verbatim for the PGS row math; differences are confined to the body
I/O surface, same pattern as
:mod:`newton._src.solvers.phoenx.mass_splitting.iterate_contact_split`:

* For rigid endpoint sides (``kind != CLOTH_TRIANGLE`` and
  ``nodes[0] >= 0``), the velocity / angular-velocity reads come from
  the per-(body, partition) ``TinyRigidState`` copy via
  :func:`read_state`. The accumulated delta over the contact column's
  sequential GS gets scaled by ``1/inv_factor`` and committed via
  :func:`write_state`. Single-copy bodies (``inv_factor == 1``) match
  the unsplit baseline bit-for-bit.
* For cloth endpoint sides, particle reads / writes stay direct --
  particles aren't in the rigid mass-splitting graph (the C# pattern
  splits rigid masses only).
* Static rigid endpoints (``nodes[0] < 0``) and static-anchor bodies
  with ``inv_mass == 0`` follow the static fallback in
  :func:`read_state` (returns ``inv_factor = 0``, ``state_index = -1``);
  the caller treats them as infinite mass and skips the impulse, same
  as the unsplit version.

Endpoint helpers in :mod:`contact_endpoint` write to the body store
directly inside the per-contact loop. We can't reuse them as-is for
the split path; the body update needs to land in a local register
first so a single ``write_state`` at the end can commit the scaled
delta. This module ships register-flavoured replacements
(``_endpoint_velocity_at_point_split`` /
``_endpoint_apply_impulse_split``) that read / mutate the local
``rigid_v`` / ``rigid_w`` registers for rigid sides and fall through
to the existing direct-write path for cloth.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.cloth_collision import SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    ContactViews,
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
    contact_get_side0_kind,
    contact_get_side0_nodes_extra,
    contact_get_side1_kind,
    contact_get_side1_nodes_extra,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    soft_constraint_coefficients,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_bias,
    cc_get_bias_t1,
    cc_get_bias_t2,
    cc_get_eff_n,
    cc_get_eff_t1,
    cc_get_eff_t2,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
    cc_get_side0_bary,
    cc_get_side1_bary,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_normal_lambda,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.constraint_contact_cloth import (
    _side_world_contact_point,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphData,
)
from newton._src.solvers.phoenx.mass_splitting.read_state import read_state, write_state
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_VELOCITY_LEVEL,
    TinyRigidState,
)
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "contact_iterate_at_cloth_aware_split",
    "contact_iterate_cloth_aware_split",
]


_ACCESS_MODE_VELOCITY_LEVEL_C = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


@wp.func
def _safe_inv(inv_factor: wp.int32) -> wp.float32:
    """``1 / max(1, inv_factor)``. Static-body fallback returns 0."""
    if inv_factor <= wp.int32(0):
        return wp.float32(0.0)
    return wp.float32(1.0) / wp.float32(inv_factor)


@wp.func
def _read_particle_velocity(
    graph: InteractionGraphData,
    pcid: wp.int32,
    node_id: wp.int32,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    inv_dt: wp.float32,
) -> wp.vec3f:
    """Particle's velocity for ``(constraint, particle)``: copy state
    when registered, particle store otherwise (static fallback)."""
    p = node_id - num_bodies
    state, _inv_factor, idx = read_state(
        graph, pcid, node_id,
        particles.position[p], wp.quat_identity(),
        particles.velocity[p], wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        _ACCESS_MODE_VELOCITY_LEVEL_C, inv_dt,
    )
    if idx < wp.int32(0):
        return particles.velocity[p]
    return state.velocity


@wp.func
def _endpoint_velocity_at_point_split(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    rigid_v: wp.vec3f,
    rigid_w: wp.vec3f,
    contact_point_world: wp.vec3f,
    graph: InteractionGraphData,
    pcid: wp.int32,
    inv_dt: wp.float32,
) -> wp.vec3f:
    """Velocity at the contact point (world frame).

    Cloth side: bary-weighted particle velocities. Each particle's
    velocity comes from its per-(particle, partition) copy state via
    :func:`read_state` so multiple cids in the same partition that
    share a particle don't race -- ``inv_factor`` scaling at the
    write-state boundary keeps the accumulation conservative.
    Rigid dynamic side: ``rigid_v + rigid_w x r``, where the
    caller's registers are pre-loaded from the rigid copy state.
    Static / anchor side: zero.
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        v_a = _read_particle_velocity(graph, pcid, nodes[0], particles, num_bodies, inv_dt)
        v_b = _read_particle_velocity(graph, pcid, nodes[1], particles, num_bodies, inv_dt)
        v_c = _read_particle_velocity(graph, pcid, nodes[2], particles, num_bodies, inv_dt)
        return bary[0] * v_a + bary[1] * v_b + bary[2] * v_c
    b = nodes[0]
    if b < 0:
        return wp.vec3f(0.0, 0.0, 0.0)
    r = contact_point_world - bodies.position[b]
    return rigid_v + wp.cross(rigid_w, r)


@wp.func
def _apply_particle_impulse_split(
    graph: InteractionGraphData,
    pcid: wp.int32,
    node_id: wp.int32,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    bary_weight: wp.float32,
    impulse: wp.vec3f,
    inv_dt: wp.float32,
):
    """Per-particle impulse application: route through the
    particle's per-(partition) copy state when registered. Scales
    the delta by ``1 / inv_factor`` so multiple cids in the same
    partition that share this particle accumulate consistently
    (Tonge averaging across iterations); the
    :func:`average_and_broadcast_unified_kernel` reconstructs the
    consensus.

    Static / unregistered particles fall through to the direct
    ``particles.velocity[p]`` write -- inv_factor=0 means the
    static-body fallback wasn't part of any partition, so a direct
    write doesn't race with anything.
    """
    p = node_id - num_bodies
    inv_m = particles.inverse_mass[p]
    if inv_m <= wp.float32(0.0):
        return
    state, inv_factor, idx = read_state(
        graph, pcid, node_id,
        particles.position[p], wp.quat_identity(),
        particles.velocity[p], wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        _ACCESS_MODE_VELOCITY_LEVEL_C, inv_dt,
    )
    dv = bary_weight * impulse * inv_m
    if idx < wp.int32(0):
        # Static-fallback (particle not registered in graph). Write
        # directly; with ``batch_size=1`` no other concurrent writers.
        particles.velocity[p] = particles.velocity[p] + dv
        return
    # C# Tonge convention: multiply the per-copy delta by invFactor
    # so the post-averaging response equals an unsplit GS step.
    # See ``contact_iterate_at_split`` for the full derivation.
    inv_factor_count = wp.float32(wp.max(inv_factor, wp.int32(1)))
    state.velocity = state.velocity + inv_factor_count * dv
    write_state(graph, idx, state)


@wp.func
def _endpoint_apply_impulse_split(
    kind: wp.int32,
    nodes: wp.vec3i,
    bary: wp.vec3f,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    contact_point_world: wp.vec3f,
    impulse: wp.vec3f,
    rigid_v: wp.vec3f,
    rigid_w: wp.vec3f,
    graph: InteractionGraphData,
    pcid: wp.int32,
    inv_dt: wp.float32,
):
    """Apply a 3D impulse to this side's nodes.

    Cloth: routes each particle through its per-(particle, partition)
    copy state via :func:`_apply_particle_impulse_split` --
    ``read_state`` / mutate / ``write_state`` with ``1/inv_factor``
    scaling. Multiple cids in the same partition that share this
    particle accumulate via the iteration loop, with the
    :func:`average_and_broadcast_unified_kernel` reconstructing the
    consensus between iterations.

    Rigid: instead of writing to ``bodies.velocity[b]``, return the
    updated ``(rigid_v, rigid_w)`` registers so the caller can keep
    them in scope across the per-contact GS loop and commit the
    scaled delta to the rigid ``TinyRigidState`` copy at the end.

    Returns ``(rigid_v_new, rigid_w_new)`` regardless of kind
    (cloth and static return the unchanged registers).
    """
    if kind == wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE):
        _apply_particle_impulse_split(graph, pcid, nodes[0], particles, num_bodies, bary[0], impulse, inv_dt)
        _apply_particle_impulse_split(graph, pcid, nodes[1], particles, num_bodies, bary[1], impulse, inv_dt)
        _apply_particle_impulse_split(graph, pcid, nodes[2], particles, num_bodies, bary[2], impulse, inv_dt)
        return rigid_v, rigid_w
    b = nodes[0]
    if b < 0:
        return rigid_v, rigid_w
    inv_m = bodies.inverse_mass[b]
    if inv_m == wp.float32(0.0):
        return rigid_v, rigid_w
    r = contact_point_world - bodies.position[b]
    inv_i = bodies.inverse_inertia_world[b]
    new_v = rigid_v + impulse * inv_m
    new_w = rigid_w + inv_i @ wp.cross(r, impulse)
    return new_v, new_w


@wp.func
def contact_iterate_at_cloth_aware_split(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    """Cloth-aware contact iterate with rigid-side mass splitting.

    Same Box2D-v3 PGS row math as
    :func:`contact_iterate_at_cloth_aware` -- only difference is the
    body I/O for rigid sides routes through
    :func:`read_state` / :func:`write_state` with ``1/inv_factor``
    impulse scaling. Cloth and static-anchor sides keep the existing
    direct path."""
    _ = base_offset

    side0_kind = contact_get_side0_kind(constraints, cid)
    side1_kind = contact_get_side1_kind(constraints, cid)
    side0_extra = contact_get_side0_nodes_extra(constraints, cid)
    side1_extra = contact_get_side1_nodes_extra(constraints, cid)
    side0_nodes = wp.vec3i(body_pair.b1, side0_extra[0], side0_extra[1])
    side1_nodes = wp.vec3i(body_pair.b2, side1_extra[0], side1_extra[1])

    contact_first = contact_get_contact_first(constraints, cid)
    contact_count = contact_get_contact_count(constraints, cid)
    if contact_count == 0:
        return

    mu_s = contact_get_friction(constraints, cid)
    mu_k = contact_get_friction_dynamic(constraints, cid)

    dt_substep = wp.float32(1.0) / idt
    _bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        DEFAULT_HERTZ_CONTACT, DEFAULT_DAMPING_RATIO, dt_substep
    )

    pcid = cid_to_partition_constraint_id[cid]

    # Pre-read each side's rigid state into local registers.
    # ``side0_is_rigid_dyn`` true => rigid dynamic body with a
    # registered copy; the GS loop mutates ``v0``/``w0`` in place and
    # the trailing ``write_state`` commits ``v0_pre + (v0 - v0_pre) /
    # inv_factor``. Cloth + static-anchor + zero-inv-mass bodies skip
    # the read_state/write_state dance entirely.
    side0_is_rigid_dyn = (side0_kind != wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)) and (side0_nodes[0] >= wp.int32(0))
    side1_is_rigid_dyn = (side1_kind != wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)) and (side1_nodes[0] >= wp.int32(0))

    state0 = TinyRigidState()
    inv_factor0 = wp.int32(0)
    idx0 = wp.int32(-1)
    if side0_is_rigid_dyn:
        b0 = side0_nodes[0]
        state0, inv_factor0, idx0 = read_state(
            graph, pcid, b0,
            bodies.position[b0], bodies.orientation[b0],
            bodies.velocity[b0], bodies.angular_velocity[b0],
            _ACCESS_MODE_VELOCITY_LEVEL_C, idt,
        )

    state1 = TinyRigidState()
    inv_factor1 = wp.int32(0)
    idx1 = wp.int32(-1)
    if side1_is_rigid_dyn:
        b1 = side1_nodes[0]
        state1, inv_factor1, idx1 = read_state(
            graph, pcid, b1,
            bodies.position[b1], bodies.orientation[b1],
            bodies.velocity[b1], bodies.angular_velocity[b1],
            _ACCESS_MODE_VELOCITY_LEVEL_C, idt,
        )

    v0 = state0.velocity
    w0 = state0.angular_velocity
    v1 = state1.velocity
    w1 = state1.angular_velocity
    v0_pre = v0
    w0_pre = w0
    v1_pre = v1
    w1_pre = w1
    inv_factor0_f = _safe_inv(inv_factor0)
    inv_factor1_f = _safe_inv(inv_factor1)

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        bary0 = cc_get_side0_bary(cc, k)
        bary1 = cc_get_side1_bary(cc, k)
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]

        p0_world = _side_world_contact_point(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies,
            cc, k, False, margin0, n,
        )
        p1_world = _side_world_contact_point(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies,
            cc, k, True, margin1, n,
        )

        eff_n = cc_get_eff_n(cc, k)
        eff_t1 = cc_get_eff_t1(cc, k)
        eff_t2 = cc_get_eff_t2(cc, k)
        bias_val = cc_get_bias(cc, k)
        bias_t1_val = cc_get_bias_t1(cc, k)
        bias_t2_val = cc_get_bias_t2(cc, k)
        is_speculative = bias_val > wp.float32(0.0)
        if not use_bias:
            if not is_speculative:
                bias_val = wp.float32(0.0)
            bias_t1_val = wp.float32(0.0)
            bias_t2_val = wp.float32(0.0)

        v0_at_p = _endpoint_velocity_at_point_split(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies,
            v0, w0, p0_world, graph, pcid, idt,
        )
        v1_at_p = _endpoint_velocity_at_point_split(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies,
            v1, w1, p1_world, graph, pcid, idt,
        )
        vel_rel = v1_at_p - v0_at_p
        jv_n = wp.dot(vel_rel, n)
        jv_t1 = wp.dot(vel_rel, t1_dir)
        jv_t2 = wp.dot(vel_rel, t2_dir)

        pd_eff_soft_n = cc_get_pd_eff_soft(cc, k)
        lam_n_old = cc_get_normal_lambda(cc, k)
        if pd_eff_soft_n > wp.float32(0.0):
            pd_gamma_n = cc_get_pd_gamma(cc, k)
            pd_bias_n = cc_get_pd_bias(cc, k)
            d_lam_n_us = -pd_eff_soft_n * (jv_n - pd_bias_n + pd_gamma_n * lam_n_old)
            lam_n_new = wp.max(lam_n_old + d_lam_n_us, wp.float32(0.0))
            d_lam_n = lam_n_new - lam_n_old
        else:
            if is_speculative:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
            elif use_bias:
                mass_coeff_n = mass_coeff
                impulse_coeff_n = impulse_coeff
            else:
                mass_coeff_n = wp.float32(1.0)
                impulse_coeff_n = wp.float32(0.0)
            d_lam_n_us = -eff_n * (jv_n + bias_val)
            d_lam_n = mass_coeff_n * d_lam_n_us - impulse_coeff_n * lam_n_old
            lam_n_new = wp.max(lam_n_old + d_lam_n, wp.float32(0.0))
            d_lam_n = lam_n_new - lam_n_old

        fric_limit_static = mu_s * lam_n_new
        fric_limit_kinetic = mu_k * lam_n_new

        d_lam_t1 = -eff_t1 * (jv_t1 + bias_t1_val)
        d_lam_t2 = -eff_t2 * (jv_t2 + bias_t2_val)
        lam_t1_old = cc_get_tangent1_lambda(cc, k)
        lam_t2_old = cc_get_tangent2_lambda(cc, k)
        lam_t1_raw = lam_t1_old + d_lam_t1
        lam_t2_raw = lam_t2_old + d_lam_t2
        lam_t_sq = lam_t1_raw * lam_t1_raw + lam_t2_raw * lam_t2_raw
        static_limit_sq = fric_limit_static * fric_limit_static
        if lam_t_sq > static_limit_sq and lam_t_sq > wp.float32(1.0e-30):
            inv_mag = fric_limit_kinetic / wp.sqrt(lam_t_sq)
            lam_t1_new = lam_t1_raw * inv_mag
            lam_t2_new = lam_t2_raw * inv_mag
        else:
            lam_t1_new = lam_t1_raw
            lam_t2_new = lam_t2_raw
        d_lam_t1 = lam_t1_new - lam_t1_old
        d_lam_t2 = lam_t2_new - lam_t2_old

        cc_set_normal_lambda(cc, k, lam_n_new)
        cc_set_tangent1_lambda(cc, k, lam_t1_new)
        cc_set_tangent2_lambda(cc, k, lam_t2_new)

        imp = d_lam_n * n + d_lam_t1 * t1_dir + d_lam_t2 * t2_dir
        # Side 0 receives -imp; side 1 receives +imp. Cloth sides
        # write to particles directly in the helper; rigid sides
        # update local registers so the per-contact GS sequence
        # accumulates in scope.
        v0, w0 = _endpoint_apply_impulse_split(
            side0_kind, side0_nodes, bary0, bodies, particles, num_bodies,
            p0_world, -imp, v0, w0, graph, pcid, idt,
        )
        v1, w1 = _endpoint_apply_impulse_split(
            side1_kind, side1_nodes, bary1, bodies, particles, num_bodies,
            p1_world, imp, v1, w1, graph, pcid, idt,
        )

    # C# Tonge convention -- multiply the per-copy delta by
    # ``invFactor`` (count of partitions this node participates in).
    # See the matching docstring in
    # :func:`~newton._src.solvers.phoenx.mass_splitting.iterate_contact_split.contact_iterate_at_split`
    # for the full derivation; tl;dr: only one of the body's N
    # copies receives this contact's update, so multiplying here
    # makes the post-averaging response equal an unsplit GS step.
    if side0_is_rigid_dyn and idx0 >= wp.int32(0):
        inv_factor0_count = wp.float32(wp.max(inv_factor0, wp.int32(1)))
        state0.velocity = v0_pre + inv_factor0_count * (v0 - v0_pre)
        state0.angular_velocity = w0_pre + inv_factor0_count * (w0 - w0_pre)
        write_state(graph, idx0, state0)
    if side1_is_rigid_dyn and idx1 >= wp.int32(0):
        inv_factor1_count = wp.float32(wp.max(inv_factor1, wp.int32(1)))
        state1.velocity = v1_pre + inv_factor1_count * (v1 - v1_pre)
        state1.angular_velocity = w1_pre + inv_factor1_count * (w1 - w1_pre)
        write_state(graph, idx1, state1)


@wp.func
def contact_iterate_cloth_aware_split(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    num_bodies: wp.int32,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    """Public entry point mirroring
    :func:`~newton._src.solvers.phoenx.constraints.constraint_contact_cloth.contact_iterate_cloth_aware`."""
    # ``body_pair.b1 / b2`` are the cid header's primary body fields;
    # for cloth sides this is the first node of the cloth-tri triple,
    # matching how ``side*_nodes`` is constructed in the at-variant.
    body_pair = ConstraintBodies()
    body_pair.b1 = contact_get_body1(constraints, cid)
    body_pair.b2 = contact_get_body2(constraints, cid)
    contact_iterate_at_cloth_aware_split(
        constraints, cid, 0, bodies, particles, num_bodies,
        body_pair, idt, cc, contacts, use_bias, graph,
        cid_to_partition_constraint_id,
    )
