# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block Neo-Hookean XPBD soft-tetrahedron constraint.

Projects hydrostatic and deviatoric stable Neo-Hookean constraints
jointly with a 2x2 Schur solve. Shared tensor helpers live in
:mod:`soft_body_math` so the tet and hex variants stay aligned.
"""

from __future__ import annotations

import enum

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN,
    ConstraintContainer,
    assert_constraint_header,
    read_float,
    read_int,
    read_mat33,
    write_float,
    write_int,
    write_mat33,
)
from newton._src.solvers.phoenx.constraints.soft_body_math import (
    neohookean_constraints_from_F,
    tet_chain_rule_gradients,
    tet_deformation_gradient,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.mass_splitting.access import (
    read_position_with_slot,
    set_access_mode_with_slot,
    write_position_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "SOFT_TET_NEOHOOKEAN_DWORDS",
    "SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET",
    "SoftBodyConstraintType",
    "SoftTetNeoHookeanData",
    "soft_tet_neohookean_init_rows_kernel",
    "soft_tet_neohookean_iterate_at",
    "soft_tet_neohookean_prepare_for_iteration_at",
    "soft_tet_neohookean_set_alpha_d",
    "soft_tet_neohookean_set_alpha_h",
    "soft_tet_neohookean_set_beta_d",
    "soft_tet_neohookean_set_beta_h",
    "soft_tet_neohookean_set_body1",
    "soft_tet_neohookean_set_body2",
    "soft_tet_neohookean_set_body3",
    "soft_tet_neohookean_set_body4",
    "soft_tet_neohookean_set_gamma",
    "soft_tet_neohookean_set_inv_rest",
    "soft_tet_neohookean_set_rest_volume",
    "soft_tet_neohookean_set_type",
]


# ---------------------------------------------------------------------------
# Public enum to select between ARAP and Block-Neo-Hookean soft-tet variants.
# Consumed by :meth:`PhoenXWorld.populate_soft_tetrahedra_from_model`.
# ---------------------------------------------------------------------------


class SoftBodyConstraintType(enum.IntEnum):
    """Selector for the PhoenX soft-body tetrahedron constraint variant.

    Maps 1:1 to the per-row constraint type tag stamped at dword 0 of the
    :class:`ConstraintContainer`. The ARAP variant uses
    :data:`CONSTRAINT_TYPE_SOFT_TETRAHEDRON`; the block Neo-Hookean variant
    uses :data:`CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN`.
    """

    #: Corotational ARAP shear row (Mueller / Kugelstadt polar-decomposition
    #: warm start). Single scalar constraint per element. Fast and stable
    #: for stiff materials but volumetric drift under high deformation.
    ARAP = 0
    #: Stable Neo-Hookean (Smith et al. 2018) with the block-coupled
    #: hydrostatic + deviatoric 2x2 Schur solve from Ton-That et al. 2024.
    #: Slightly heavier per-iteration but ~10x faster to converge under
    #: heavy deformation; volume-preserving by construction.
    BLOCK_NEOHOOKEAN = 1


_PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class SoftTetNeoHookeanData:
    """Per-constraint dword-layout schema for one block Neo-Hookean tet.

    Two compliance / multiplier rows -- one for the hydrostatic constraint
    ``C^H = det(F) - gamma`` and one for the deviatoric constraint
    ``C^D = sqrt(tr(F^T F))``. ``gamma`` is the rest-pose offset
    ``1 + mu / lambda`` that makes the combined energy gradient vanish
    on ``F = I``.
    """

    constraint_type: wp.int32
    body1: wp.int32  # particle index of node A
    body2: wp.int32  # particle index of node B
    body3: wp.int32  # particle index of node C
    body4: wp.int32  # particle index of node D

    inv_rest: wp.mat33f
    rest_volume: wp.float32

    #: ``1 + mu / lambda`` -- stable Neo-Hookean offset.
    gamma: wp.float32
    #: ``1 / (lambda * V_rest)`` -- hydrostatic compliance.
    alpha_h: wp.float32
    #: ``1 / (mu * V_rest)`` -- deviatoric compliance.
    alpha_d: wp.float32
    #: Macklin XPBD damping coefficient on the hydrostatic row [1/s]. The
    #: paper does not specify damping in the block form; we extend per-row
    #: damping for parity with the ARAP variant. ``0`` = bare XPBD.
    beta_h: wp.float32
    #: Macklin XPBD damping coefficient on the deviatoric row [1/s].
    beta_d: wp.float32

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32

    #: Hydrostatic XPBD multiplier accumulator (reset each substep).
    lambda_sum_h: wp.float32
    #: Deviatoric XPBD multiplier accumulator (reset each substep).
    lambda_sum_d: wp.float32

    #: Opt-in per-column wall-clock accumulator (microseconds).
    time_us: wp.float32


assert_constraint_header(SoftTetNeoHookeanData)


_OFF_BODY1 = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "body3"))
_OFF_BODY4 = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "body4"))
_OFF_INV_REST = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "inv_rest"))
_OFF_REST_VOLUME = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "rest_volume"))
_OFF_GAMMA = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "gamma"))
_OFF_ALPHA_H = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "alpha_h"))
_OFF_ALPHA_D = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "alpha_d"))
_OFF_BETA_H = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "beta_h"))
_OFF_BETA_D = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "beta_d"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "inv_mass_d"))
_OFF_LAMBDA_SUM_H = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "lambda_sum_h"))
_OFF_LAMBDA_SUM_D = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "lambda_sum_d"))
SOFT_TET_NEOHOOKEAN_TIME_US_OFFSET = wp.constant(dword_offset_of(SoftTetNeoHookeanData, "time_us"))

SOFT_TET_NEOHOOKEAN_DWORDS: int = num_dwords(SoftTetNeoHookeanData)


@wp.func
def soft_tet_neohookean_set_type(c: ConstraintContainer, cid: wp.int32):
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_SOFT_TETRAHEDRON_NEOHOOKEAN)


@wp.func
def soft_tet_neohookean_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def soft_tet_neohookean_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def soft_tet_neohookean_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


@wp.func
def soft_tet_neohookean_set_body4(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY4, cid, v)


@wp.func
def soft_tet_neohookean_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_INV_REST, cid, v)


@wp.func
def soft_tet_neohookean_set_rest_volume(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_VOLUME, cid, v)


@wp.func
def soft_tet_neohookean_set_gamma(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_GAMMA, cid, v)


@wp.func
def soft_tet_neohookean_set_alpha_h(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_H, cid, v)


@wp.func
def soft_tet_neohookean_set_alpha_d(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_D, cid, v)


@wp.func
def soft_tet_neohookean_set_beta_h(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_H, cid, v)


@wp.func
def soft_tet_neohookean_set_beta_d(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_D, cid, v)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
#: Floor for the deviatoric constraint magnitude (avoids division by zero
#: when the element fully collapses, F -> 0). 1e-12 keeps the gradient
#: finite without otherwise perturbing the solve at typical scales.
_DEV_EPS = wp.constant(wp.float32(1.0e-12))
#: Floor for the Schur-complement determinant during the 2x2 inverse.
#: With a strictly positive Tikhonov regulariser (alpha_tilde > 0 on the
#: diagonal) det(A) >= alpha_tilde^H * alpha_tilde^D > 0 in exact
#: arithmetic; the floor only guards against catastrophic FP cancellation.
_DET_FLOOR = wp.constant(wp.float32(1.0e-30))


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def soft_tet_neohookean_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Substep-entry prepare: flip each vertex to POSITION_LEVEL, cache
    inverse masses, and reset both XPBD multipliers to zero.

    Unlike the ARAP variant there is no polar-decomposition warm start to
    cold-start: the analytic Neo-Hookean gradient is rotation-aware by
    construction (the strain energy is a function of ``F^T F`` and
    ``det(F)`` only)."""
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    # Per-cid slot / count cache stamped by
    # :func:`build_constraint_slot_cache` -- see
    # ``soft_tetrahedron_prepare_for_iteration_at`` for the rationale.
    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]
    slot_d = constraints.slot_cache[cid, 3]
    inv_factor_a = constraints.count_cache[cid, 0]
    inv_factor_b = constraints.count_cache[cid, 1]
    inv_factor_c = constraints.count_cache[cid, 2]
    inv_factor_d = constraints.count_cache[cid, 3]

    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_d, slot_d, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p_a] * wp.float32(inv_factor_a))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p_b] * wp.float32(inv_factor_b))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p_c] * wp.float32(inv_factor_c))
    write_float(constraints, _OFF_INV_MASS_D, cid, particles.inverse_mass[p_d] * wp.float32(inv_factor_d))

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, wp.float32(0.0))


@wp.func
def soft_tet_neohookean_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    """One block Neo-Hookean PGS sweep on a soft-body tetrahedron.

    Evaluates ``(C^H, C^D)`` and the per-vertex gradients of both, then
    solves the 2x2 Schur-complement system::

        A . [d lambda^H; d lambda^D] = -([C^H; C^D] + alpha_tilde [lambda^H; lambda^D] + damping)

    with::

        A_{HH} = (1 + gamma_H) sum_v w_v ||g^H_v||^2 + alpha_tilde^H
        A_{DD} = (1 + gamma_D) sum_v w_v ||g^D_v||^2 + alpha_tilde^D
        A_{HD} = sum_v w_v g^H_v . g^D_v
        gamma_i = beta_i * dt

    The per-row damping treatment is approximate (Macklin damping in the
    block form would augment the full 2x2 with an off-diagonal coupling);
    in practice the paper runs with no damping and the off-diagonal
    coupling vanishes there. Defaults to ``beta_i = 0`` (bare XPBD) so the
    approximation only kicks in when the caller opts in to damping.

    Final position update::

        d x_v = w_v (g^H_v d lambda^H + g^D_v d lambda^D)
    """
    body_a = read_int(constraints, _OFF_BODY1, cid)
    body_b = read_int(constraints, _OFF_BODY2, cid)
    body_c = read_int(constraints, _OFF_BODY3, cid)
    body_d = read_int(constraints, _OFF_BODY4, cid)
    p_a = body_a - num_bodies
    p_b = body_b - num_bodies
    p_c = body_c - num_bodies
    p_d = body_d - num_bodies

    slot_a = constraints.slot_cache[cid, 0]
    slot_b = constraints.slot_cache[cid, 1]
    slot_c = constraints.slot_cache[cid, 2]
    slot_d = constraints.slot_cache[cid, 3]

    set_access_mode_with_slot(
        bodies, particles, copy_state, body_a, slot_a, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_b, slot_b, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_c, slot_c, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )
    set_access_mode_with_slot(
        bodies, particles, copy_state, body_d, slot_d, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt
    )

    inv_mass_a = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass_b = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass_c = read_float(constraints, _OFF_INV_MASS_C, cid)
    inv_mass_d = read_float(constraints, _OFF_INV_MASS_D, cid)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    gamma_offset = read_float(constraints, _OFF_GAMMA, cid)
    alpha_h = read_float(constraints, _OFF_ALPHA_H, cid)
    alpha_d = read_float(constraints, _OFF_ALPHA_D, cid)
    beta_h = read_float(constraints, _OFF_BETA_H, cid)
    beta_d = read_float(constraints, _OFF_BETA_D, cid)
    lambda_h = read_float(constraints, _OFF_LAMBDA_SUM_H, cid)
    lambda_d = read_float(constraints, _OFF_LAMBDA_SUM_D, cid)

    x_a = read_position_with_slot(bodies, particles, copy_state, body_a, slot_a, num_bodies)
    x_b = read_position_with_slot(bodies, particles, copy_state, body_b, slot_b, num_bodies)
    x_c = read_position_with_slot(bodies, particles, copy_state, body_c, slot_c, num_bodies)
    x_d = read_position_with_slot(bodies, particles, copy_state, body_d, slot_d, num_bodies)

    dx_a = x_a - particles.position_prev_substep[p_a]
    dx_b = x_b - particles.position_prev_substep[p_b]
    dx_c = x_c - particles.position_prev_substep[p_c]
    dx_d = x_d - particles.position_prev_substep[p_d]

    F = tet_deformation_gradient(x_a, x_b, x_c, x_d, inv_rest)

    c_h, c_d, dCH_dF, dCD_dF = neohookean_constraints_from_F(F, gamma_offset, _DEV_EPS)

    g_ha, g_hb, g_hc, g_hd = tet_chain_rule_gradients(dCH_dF, inv_rest)
    g_da, g_db, g_dc, g_dd = tet_chain_rule_gradients(dCD_dF, inv_rest)

    # Schur-complement matrix entries.
    a_hh = (
        inv_mass_a * wp.dot(g_ha, g_ha)
        + inv_mass_b * wp.dot(g_hb, g_hb)
        + inv_mass_c * wp.dot(g_hc, g_hc)
        + inv_mass_d * wp.dot(g_hd, g_hd)
    )
    a_dd = (
        inv_mass_a * wp.dot(g_da, g_da)
        + inv_mass_b * wp.dot(g_db, g_db)
        + inv_mass_c * wp.dot(g_dc, g_dc)
        + inv_mass_d * wp.dot(g_dd, g_dd)
    )
    a_hd = (
        inv_mass_a * wp.dot(g_ha, g_da)
        + inv_mass_b * wp.dot(g_hb, g_db)
        + inv_mass_c * wp.dot(g_hc, g_dc)
        + inv_mass_d * wp.dot(g_hd, g_dd)
    )

    idt_sq = idt * idt
    dt = wp.float32(1.0) / idt
    bias_h = idt_sq * alpha_h  # alpha_tilde^H = alpha^H / dt^2
    bias_d = idt_sq * alpha_d
    gamma_h = beta_h * dt
    gamma_d = beta_d * dt

    grad_h_dot_dx = wp.dot(g_ha, dx_a) + wp.dot(g_hb, dx_b) + wp.dot(g_hc, dx_c) + wp.dot(g_hd, dx_d)
    grad_d_dot_dx = wp.dot(g_da, dx_a) + wp.dot(g_db, dx_b) + wp.dot(g_dc, dx_c) + wp.dot(g_dd, dx_d)

    # Diagonal damping scaling (per-row Macklin). Off-diagonal stays
    # undamped -- exact only when beta == 0 (the paper's setting);
    # approximate but stable for small beta.
    A11 = (wp.float32(1.0) + gamma_h) * a_hh + bias_h
    A22 = (wp.float32(1.0) + gamma_d) * a_dd + bias_d
    A12 = a_hd

    b_h = c_h + bias_h * lambda_h + gamma_h * grad_h_dot_dx
    b_d = c_d + bias_d * lambda_d + gamma_d * grad_d_dot_dx

    det_a = A11 * A22 - A12 * A12
    if det_a < _DET_FLOOR:
        # Catastrophic loss of rank. Skip the update for this sweep;
        # subsequent sweeps will revisit once gradients have moved.
        return

    inv_det = wp.float32(1.0) / det_a
    # A^{-1} for symmetric 2x2: [A22, -A12; -A12, A11] / det.
    dlam_h = -(A22 * b_h - A12 * b_d) * inv_det
    dlam_d = -(-A12 * b_h + A11 * b_d) * inv_det

    dlam_h = dlam_h * sor_boost
    dlam_d = dlam_d * sor_boost

    x_a = x_a + inv_mass_a * (dlam_h * g_ha + dlam_d * g_da)
    x_b = x_b + inv_mass_b * (dlam_h * g_hb + dlam_d * g_db)
    x_c = x_c + inv_mass_c * (dlam_h * g_hc + dlam_d * g_dc)
    x_d = x_d + inv_mass_d * (dlam_h * g_hd + dlam_d * g_dd)

    write_position_unified(bodies, particles, copy_state, body_a, slot_a, num_bodies, x_a)
    write_position_unified(bodies, particles, copy_state, body_b, slot_b, num_bodies, x_b)
    write_position_unified(bodies, particles, copy_state, body_c, slot_c, num_bodies, x_c)
    write_position_unified(bodies, particles, copy_state, body_d, slot_d, num_bodies, x_d)

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, lambda_h + dlam_h)
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, lambda_d + dlam_d)


# ---------------------------------------------------------------------------
# Builder-side init kernel
# ---------------------------------------------------------------------------


@wp.kernel
def soft_tet_neohookean_init_rows_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    tet_indices: wp.array2d[wp.int32],
    particle_q: wp.array[wp.vec3f],
    tet_poses: wp.array[wp.mat33f],
    tet_materials: wp.array2d[wp.float32],
    default_beta_h: wp.float32,
    default_beta_d: wp.float32,
):
    """Stamp one block Neo-Hookean tetrahedron row from Newton mesh API.

    Mirrors :func:`soft_tet_init_rows_kernel` for the ARAP variant but
    computes the stable Neo-Hookean offsets/compliances instead::

        gamma   = 1 + mu / lambda
        alpha^H = 1 / (lambda * V_rest)
        alpha^D = 1 / (mu * V_rest)
    """
    t = wp.tid()
    cid = cid_offset + t

    pa = tet_indices[t, 0]
    pb = tet_indices[t, 1]
    pc = tet_indices[t, 2]
    pd = tet_indices[t, 3]

    soft_tet_neohookean_set_type(constraints, cid)
    soft_tet_neohookean_set_body1(constraints, cid, num_bodies + pa)
    soft_tet_neohookean_set_body2(constraints, cid, num_bodies + pb)
    soft_tet_neohookean_set_body3(constraints, cid, num_bodies + pc)
    soft_tet_neohookean_set_body4(constraints, cid, num_bodies + pd)

    xa = particle_q[pa]
    xb = particle_q[pb]
    xc = particle_q[pc]
    xd = particle_q[pd]

    e_ab = xb - xa
    e_ac = xc - xa
    e_ad = xd - xa
    det_dm = wp.dot(e_ab, wp.cross(e_ac, e_ad))
    rest_volume = wp.abs(det_dm) * (wp.float32(1.0) / wp.float32(6.0))
    soft_tet_neohookean_set_rest_volume(constraints, cid, rest_volume)
    soft_tet_neohookean_set_inv_rest(constraints, cid, tet_poses[t])

    k_mu = tet_materials[t, 0]
    if k_mu < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        k_mu = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR
    k_lambda = tet_materials[t, 1]
    if k_lambda < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        k_lambda = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR

    # Floor V_rest in the same way the ARAP variant does -- a zero rest
    # volume would NaN the compliances. Degenerate tets are an upstream
    # mesh problem but we keep the kernel finite.
    v_eff = rest_volume
    if v_eff < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        v_eff = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR

    gamma_offset = wp.float32(1.0) + k_mu / k_lambda
    alpha_h = wp.float32(1.0) / (k_lambda * v_eff)
    alpha_d = wp.float32(1.0) / (k_mu * v_eff)

    soft_tet_neohookean_set_gamma(constraints, cid, gamma_offset)
    soft_tet_neohookean_set_alpha_h(constraints, cid, alpha_h)
    soft_tet_neohookean_set_alpha_d(constraints, cid, alpha_d)
    soft_tet_neohookean_set_beta_h(constraints, cid, default_beta_h)
    soft_tet_neohookean_set_beta_d(constraints, cid, default_beta_d)

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, wp.float32(0.0))
