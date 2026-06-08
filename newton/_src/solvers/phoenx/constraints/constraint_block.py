# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small local block solves shared by PhoenX constraint iterates.

The per-constraint kernels still own geometry-specific fetch and scatter. This
module keeps the actual projected block update in one place:

    d_lambda = solve(K, rhs)
    lambda = project(lambda + d_lambda)

That is the common core for contacts, joints, and XPBD soft constraints.
"""

from __future__ import annotations

import warp as wp

__all__ = [
    "BLOCK_LAMBDA_INF",
    "block_project_delta_1",
    "block_project_delta_sor_1",
    "block_project_friction_circle_2",
    "block_project_friction_delta_sor_2",
    "block_project_identity_delta_1",
    "block_project_identity_delta_2",
    "block_solve_inverse_2",
    "block_solve_inverse_3",
    "block_solve_inverse_4",
    "block_solve_projected_xpbd_1",
    "block_solve_projected_xpbd_2",
    "block_solve_projected_xpbd_2_strict",
    "block_solve_symmetric_1",
    "block_solve_symmetric_2",
    "block_solve_symmetric_2_strict",
    "block_solve_xpbd_1",
    "block_solve_xpbd_2",
    "block_solve_xpbd_2_strict",
]


BLOCK_LAMBDA_INF = wp.constant(wp.float32(1.0e30))


@wp.func
def block_project_delta_1(
    lambda_old: wp.float32,
    d_lambda: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> wp.vec2f:
    """Project one accumulated multiplier into a box.

    Returns ``(projected_delta, lambda_new)``.
    """
    lambda_new = wp.clamp(lambda_old + d_lambda, lambda_min, lambda_max)
    return wp.vec2f(lambda_new - lambda_old, lambda_new)


@wp.func
def block_project_delta_sor_1(
    lambda_old: wp.float32,
    d_lambda_unscaled: wp.float32,
    sor_boost: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> wp.vec2f:
    """Apply SOR, then project one accumulated multiplier into a box.

    Returns (projected_delta, lambda_new).
    """
    return block_project_delta_1(lambda_old, d_lambda_unscaled * sor_boost, lambda_min, lambda_max)


@wp.func
def block_project_identity_delta_1(lambda_old: wp.float32, d_lambda: wp.float32) -> wp.vec2f:
    """Identity-project one accumulated multiplier.

    Returns (projected_delta, lambda_new).
    """
    lambda_new = lambda_old + d_lambda
    return wp.vec2f(d_lambda, lambda_new)


@wp.func
def block_project_identity_delta_2(
    lambda1_old: wp.float32,
    lambda2_old: wp.float32,
    d_lambda1: wp.float32,
    d_lambda2: wp.float32,
) -> wp.vec4f:
    """Identity-project two accumulated multipliers.

    Returns (projected_delta1, projected_delta2, lambda1_new, lambda2_new).
    """
    return wp.vec4f(d_lambda1, d_lambda2, lambda1_old + d_lambda1, lambda2_old + d_lambda2)


@wp.func
def block_project_friction_circle_2(
    lambda_t1_raw: wp.float32,
    lambda_t2_raw: wp.float32,
    static_limit: wp.float32,
    kinetic_limit: wp.float32,
) -> wp.vec2f:
    """Project a two-tangent contact multiplier onto the Coulomb disk.

    The static limit decides whether projection is needed. If projection is
    needed, the kinetic limit supplies the slip radius, matching PhoenX's
    existing contact update.
    """
    lambda_t_sq = lambda_t1_raw * lambda_t1_raw + lambda_t2_raw * lambda_t2_raw
    static_limit_sq = static_limit * static_limit
    lambda_t1_new = lambda_t1_raw
    lambda_t2_new = lambda_t2_raw
    if lambda_t_sq > static_limit_sq and lambda_t_sq > wp.float32(1.0e-30):
        inv_mag = kinetic_limit / wp.sqrt(lambda_t_sq)
        lambda_t1_new = lambda_t1_raw * inv_mag
        lambda_t2_new = lambda_t2_raw * inv_mag
    return wp.vec2f(lambda_t1_new, lambda_t2_new)


@wp.func
def block_project_friction_delta_sor_2(
    lambda_t1_old: wp.float32,
    lambda_t2_old: wp.float32,
    d_lambda_t1_unscaled: wp.float32,
    d_lambda_t2_unscaled: wp.float32,
    sor_boost: wp.float32,
    static_limit: wp.float32,
    kinetic_limit: wp.float32,
) -> wp.vec4f:
    """Apply SOR, then project a two-row friction update.

    Returns (projected_delta_t1, projected_delta_t2, lambda_t1_new,
    lambda_t2_new).
    """
    lambda_t1_raw = lambda_t1_old + d_lambda_t1_unscaled * sor_boost
    lambda_t2_raw = lambda_t2_old + d_lambda_t2_unscaled * sor_boost
    lambda_new = block_project_friction_circle_2(lambda_t1_raw, lambda_t2_raw, static_limit, kinetic_limit)
    return wp.vec4f(lambda_new[0] - lambda_t1_old, lambda_new[1] - lambda_t2_old, lambda_new[0], lambda_new[1])


@wp.func
def block_solve_symmetric_1(
    A11: wp.float32,
    rhs1: wp.float32,
    diag_floor: wp.float32,
) -> wp.float32:
    """Solve one unconstrained symmetric block row."""
    d1 = wp.float32(0.0)
    if A11 > diag_floor:
        d1 = -rhs1 / A11
    return d1


@wp.func
def block_solve_xpbd_1(
    A11: wp.float32,
    rhs1: wp.float32,
    sor_boost: wp.float32,
    diag_floor: wp.float32,
) -> wp.float32:
    """Solve one XPBD block row and apply SOR."""
    return block_solve_symmetric_1(A11, rhs1, diag_floor) * sor_boost


@wp.func
def block_solve_projected_xpbd_1(
    A11: wp.float32,
    rhs1: wp.float32,
    lambda_old: wp.float32,
    sor_boost: wp.float32,
    diag_floor: wp.float32,
) -> wp.vec2f:
    """Solve one XPBD row, apply SOR, then identity-project lambda."""
    d_lambda = block_solve_xpbd_1(A11, rhs1, sor_boost, diag_floor)
    return block_project_identity_delta_1(lambda_old, d_lambda)


@wp.func
def block_solve_symmetric_2(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    det_floor: wp.float32,
) -> wp.vec2f:
    """Solve a 2x2 symmetric block with scalar fallback.

    Matrix layout::

        [A11 A12]
        [A12 A22]

    Returns ``(-A^-1 rhs)``. If the coupled block is singular, falls back to
    independent scalar rows for the non-zero diagonal entries.
    """
    det_a = A11 * A22 - A12 * A12
    d1 = wp.float32(0.0)
    d2 = wp.float32(0.0)
    if det_a > det_floor:
        inv_det = wp.float32(1.0) / det_a
        d1 = -(A22 * rhs1 - A12 * rhs2) * inv_det
        d2 = -(-A12 * rhs1 + A11 * rhs2) * inv_det
    else:
        if A11 > wp.float32(0.0):
            d1 = -rhs1 / A11
        if A22 > wp.float32(0.0):
            d2 = -rhs2 / A22
    return wp.vec2f(d1, d2)


@wp.func
def block_solve_xpbd_2(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    sor_boost: wp.float32,
    det_floor: wp.float32,
) -> wp.vec2f:
    """Solve a two-row XPBD block and apply SOR."""
    d = block_solve_symmetric_2(A11, A12, A22, rhs1, rhs2, det_floor)
    return wp.vec2f(d[0] * sor_boost, d[1] * sor_boost)


@wp.func
def block_solve_projected_xpbd_2(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    lambda1_old: wp.float32,
    lambda2_old: wp.float32,
    sor_boost: wp.float32,
    det_floor: wp.float32,
) -> wp.vec4f:
    """Solve a two-row XPBD block, apply SOR, then identity-project lambdas."""
    d = block_solve_xpbd_2(A11, A12, A22, rhs1, rhs2, sor_boost, det_floor)
    return block_project_identity_delta_2(lambda1_old, lambda2_old, d[0], d[1])


@wp.func
def block_solve_symmetric_2_strict(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    det_floor: wp.float32,
) -> wp.vec2f:
    """Solve a 2x2 symmetric block, returning zero for singular blocks.

    Returns ``(-A^-1 rhs)`` when the determinant is usable.
    """
    det_a = A11 * A22 - A12 * A12
    d1 = wp.float32(0.0)
    d2 = wp.float32(0.0)
    if det_a >= det_floor:
        inv_det = wp.float32(1.0) / det_a
        d1 = -(A22 * rhs1 - A12 * rhs2) * inv_det
        d2 = -(-A12 * rhs1 + A11 * rhs2) * inv_det
    return wp.vec2f(d1, d2)


@wp.func
def block_solve_xpbd_2_strict(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    sor_boost: wp.float32,
    det_floor: wp.float32,
) -> wp.vec2f:
    """Solve a strict two-row XPBD block and apply SOR."""
    d = block_solve_symmetric_2_strict(A11, A12, A22, rhs1, rhs2, det_floor)
    return wp.vec2f(d[0] * sor_boost, d[1] * sor_boost)


@wp.func
def block_solve_projected_xpbd_2_strict(
    A11: wp.float32,
    A12: wp.float32,
    A22: wp.float32,
    rhs1: wp.float32,
    rhs2: wp.float32,
    lambda1_old: wp.float32,
    lambda2_old: wp.float32,
    sor_boost: wp.float32,
    det_floor: wp.float32,
) -> wp.vec4f:
    """Solve a strict two-row XPBD block, apply SOR, then identity-project lambdas."""
    d = block_solve_xpbd_2_strict(A11, A12, A22, rhs1, rhs2, sor_boost, det_floor)
    return block_project_identity_delta_2(lambda1_old, lambda2_old, d[0], d[1])


@wp.func
def block_solve_inverse_2(K_inv: wp.mat22f, rhs: wp.vec2f) -> wp.vec2f:
    """Solve ``K d_lambda = -rhs`` from a cached 2x2 inverse."""
    return -(K_inv @ rhs)


@wp.func
def block_solve_inverse_3(K_inv: wp.mat33f, rhs: wp.vec3f) -> wp.vec3f:
    """Solve ``K d_lambda = -rhs`` from a cached 3x3 inverse."""
    return -(K_inv @ rhs)


@wp.func
def block_solve_inverse_4(K_inv: wp.mat44f, rhs: wp.vec4f) -> wp.vec4f:
    """Solve ``K d_lambda = -rhs`` from a cached 4x4 inverse."""
    return -(K_inv @ rhs)
