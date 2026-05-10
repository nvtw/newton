# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split-aware variant of the rigid-rigid contact iterate ``@wp.func``.

Mirrors :func:`~newton._src.solvers.phoenx.constraints.constraint_contact.contact_iterate_at`
**verbatim** for everything that doesn't touch the body store. The two
substantive differences:

* ``b1`` / ``b2`` velocity + angular-velocity reads route through
  :func:`~newton._src.solvers.phoenx.mass_splitting.read_state.read_state`,
  which returns the per-(body, partition) ``TinyRigidState`` copy.
* The final velocity / angular-velocity writes go to
  :func:`~newton._src.solvers.phoenx.mass_splitting.read_state.write_state`
  on the same copy.

Body-invariants -- ``inv_mass``, ``inv_inertia_world``, ``body_com`` --
stay in :class:`~newton._src.solvers.phoenx.body.BodyContainer`; they
don't move per partition and there's no point copying them.

Impulse scaling: the C# convention (``ConstraintHelper.cuh:153-233``)
divides every velocity delta a body sees by ``inv_factor`` (the count
of partitions the body participates in). For bodies with a single copy
``inv_factor == 1`` and the split path is bit-for-bit identical to the
direct-body-store path -- ``test_split_matches_unsplit_at_inv_factor_one``
locks that invariant down with a numerical regression.

**Prepare stays unsplit** -- the non-split
:func:`~newton._src.solvers.phoenx.constraints.constraint_contact.contact_prepare_for_iteration_at`
runs once per substep against the body store before
:func:`~newton._src.solvers.phoenx.mass_splitting.MassSplitting.broadcast`
fans the body state into the per-partition copies. The Phase C.2b
solver wiring runs in this order so the body store sees the
warm-start impulse, then broadcast initialises every copy with that
post-warm-start state. Iterate is the only sweep that benefits from
copy-state isolation (it's the one repeated across PGS iterations).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_CONTACT,
    ConstraintBodies,
    pd_coefficients,
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
    cc_get_local_p0,
    cc_get_local_p1,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_pd_bias,
    cc_get_pd_eff_soft,
    cc_get_pd_gamma,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
    cc_set_bias,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_eff_n,
    cc_set_eff_t1,
    cc_set_eff_t2,
    cc_set_normal_lambda,
    cc_set_pd_bias,
    cc_set_pd_eff_soft,
    cc_set_pd_gamma,
    cc_set_tangent1_lambda,
    cc_set_tangent2_lambda,
)
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    contact_get_body1,
    contact_get_body2,
    contact_get_contact_count,
    contact_get_contact_first,
    contact_get_friction,
    contact_get_friction_dynamic,
    ContactColumnContainer,
    ContactViews,
)
from newton._src.solvers.phoenx.helpers.math_helpers import (
    apply_pair_velocity_impulse,
    effective_mass_scalar,
)
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    InteractionGraphData,
)
from newton._src.solvers.phoenx.mass_splitting.read_state import read_state, write_state
from newton._src.solvers.phoenx.mass_splitting.state import (
    ACCESS_MODE_VELOCITY_LEVEL,
)
from newton._src.solvers.phoenx.solver_config import PHOENX_BOOST_CONTACT_NORMAL

__all__ = [
    "contact_iterate_at_split",
    "contact_iterate_split",
]


_ACCESS_MODE_VELOCITY_LEVEL_C = wp.constant(wp.int32(ACCESS_MODE_VELOCITY_LEVEL))


@wp.func
def _safe_inv(inv_factor: wp.int32) -> wp.float32:
    """``1 / max(1, inv_factor)``. ``inv_factor == 0`` flags a static
    body in :func:`read_state`; the caller treats it as
    infinite-mass and skips the impulse, so the value here is just a
    safe placeholder.
    """
    if inv_factor <= wp.int32(0):
        return wp.float32(0.0)
    return wp.float32(1.0) / wp.float32(inv_factor)



@wp.func
def contact_iterate_at_split(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    """Split-aware mirror of
    :func:`~newton._src.solvers.phoenx.constraints.constraint_contact.contact_iterate_at`.

    Body velocity / angular velocity reads + writes route through
    :func:`read_state` / :func:`write_state`. The per-pair sequential
    Gauss-Seidel loop is identical to the unsplit version (it
    operates on local registers); only the bracketing body I/O and
    the per-body ``1/inv_factor`` impulse scaling on the final
    write differ.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

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

    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]
    body_com1 = bodies.body_com[b1]
    body_com2 = bodies.body_com[b2]

    # Read the partition-local copies. The InteractionGraph keys are
    # ``(rigid, partition_constraint_id)`` where
    # ``partition_constraint_id`` is 0 for regular MIS partitions and
    # ``j / batch_size`` for cids in the overflow bucket. The setup
    # kernel pre-populates ``cid_to_partition_constraint_id[cid]``
    # with the right value so this lookup hits the matching graph
    # entry. Passing ``cid`` directly would always miss (the keys
    # use the partition-id space, not the global cid space) and
    # silently fall through to the static-body path.
    pcid = cid_to_partition_constraint_id[cid]
    state1, inv_factor1, idx1 = read_state(
        graph, pcid, b1,
        bodies.position[b1], bodies.orientation[b1],
        bodies.velocity[b1], bodies.angular_velocity[b1],
        _ACCESS_MODE_VELOCITY_LEVEL_C, idt,
    )
    state2, inv_factor2, idx2 = read_state(
        graph, pcid, b2,
        bodies.position[b2], bodies.orientation[b2],
        bodies.velocity[b2], bodies.angular_velocity[b2],
        _ACCESS_MODE_VELOCITY_LEVEL_C, idt,
    )
    inv_factor1_f = _safe_inv(inv_factor1)
    inv_factor2_f = _safe_inv(inv_factor2)

    orientation1 = state1.orientation
    orientation2 = state2.orientation
    v1 = state1.velocity
    v2 = state2.velocity
    w1 = state1.angular_velocity
    w2 = state2.angular_velocity

    # Capture the start-of-iteration values; the apply-impulse
    # helper updates ``v1`` / ``v2`` / ``w1`` / ``w2`` in place via
    # ``apply_pair_velocity_impulse``, which models a one-copy update.
    # The final write splits the accumulated delta by the inv_factor
    # per body (Tonge mass-splitting convention).
    v1_pre = v1
    v2_pre = v2
    w1_pre = w1
    w2_pre = w2

    for i in range(contact_count):
        k = contact_first + i

        n = cc_get_normal(cc, k)
        t1_dir = cc_get_tangent1(cc, k)
        t2_dir = wp.cross(n, t1_dir)
        local_p0 = cc_get_local_p0(cc, k)
        local_p1 = cc_get_local_p1(cc, k)
        margin0 = contacts.rigid_contact_margin0[k]
        margin1 = contacts.rigid_contact_margin1[k]
        r1 = wp.quat_rotate(orientation1, local_p0 - body_com1) + margin0 * n
        r2 = wp.quat_rotate(orientation2, local_p1 - body_com2) - margin1 * n
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

        vel_rel = v2 + wp.cross(w2, r2) - v1 - wp.cross(w1, r1)
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
        v1, v2, w1, w2 = apply_pair_velocity_impulse(
            v1, v2, w1, w2,
            inv_mass1, inv_mass2, inv_inertia1, inv_inertia2,
            r1, r2, imp,
        )

    # Tonge split: store ``v_pre + (v_final - v_pre) / inv_factor``.
    # With ``inv_factor == 1`` this collapses to ``v_final`` -- the
    # unsplit semantics. With ``inv_factor > 1`` each partition only
    # commits its share of the accumulated impulse to its copy, and
    # the AverageAndBroadcast pass reconstructs the consensus.
    state1.velocity = v1_pre + inv_factor1_f * (v1 - v1_pre)
    state2.velocity = v2_pre + inv_factor2_f * (v2 - v2_pre)
    state1.angular_velocity = w1_pre + inv_factor1_f * (w1 - w1_pre)
    state2.angular_velocity = w2_pre + inv_factor2_f * (w2 - w2_pre)

    write_state(graph, idx1, state1)
    write_state(graph, idx2, state2)



@wp.func
def contact_iterate_split(
    constraints: ContactColumnContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    cc: ContactContainer,
    contacts: ContactViews,
    use_bias: wp.bool,
    graph: InteractionGraphData,
    cid_to_partition_constraint_id: wp.array[wp.int32],
):
    """Convenience wrapper that mirrors
    :func:`~newton._src.solvers.phoenx.constraints.constraint_contact.contact_iterate`."""
    b1 = contact_get_body1(constraints, cid)
    b2 = contact_get_body2(constraints, cid)
    body_pair = ConstraintBodies()
    body_pair.b1 = b1
    body_pair.b2 = b2
    contact_iterate_at_split(
        constraints, cid, 0, bodies, body_pair, idt, cc, contacts, use_bias, graph,
        cid_to_partition_constraint_id,
    )
