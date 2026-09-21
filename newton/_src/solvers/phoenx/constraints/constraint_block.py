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
    "BlockScalarUpdate",
    "BlockVector2Update",
    "PositionRows1",
    "PositionRows2",
    "RigidFrameRows3Update",
    "block_position_delta_1",
    "block_position_delta_2",
    "block_position_delta_2d_2",
    "block_project_accumulated_bounded_1",
    "block_project_delta_1",
    "block_project_delta_sor_1",
    "block_project_friction_circle_2",
    "block_project_friction_delta_sor_2",
    "block_project_identity_delta_1",
    "block_project_identity_delta_2",
    "block_solve_accumulated_inverse_bounded_1",
    "block_solve_position_rows1",
    "block_solve_position_rows2",
    "block_solve_position_rows2_strict",
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


@wp.struct
class BlockScalarUpdate:
    delta: wp.float32
    lambda_new: wp.float32


@wp.struct
class BlockVector2Update:
    delta: wp.vec2f
    lambda_new: wp.vec2f


@wp.struct
class PositionRows1:
    """Prepared one-row position/XPBD block for shared PGS updates."""

    A11: wp.float32
    residual: wp.float32
    lambda_old: wp.float32
    diag_floor: wp.float32


@wp.struct
class PositionRows2:
    """Prepared two-row position/XPBD block for shared PGS updates."""

    A11: wp.float32
    A12: wp.float32
    A22: wp.float32
    residual: wp.vec2f
    lambda_old: wp.vec2f
    det_floor: wp.float32


@wp.struct
class RigidFrameRows3Update:
    v_a: wp.vec3f
    w_a: wp.vec3f
    v_b: wp.vec3f
    w_b: wp.vec3f
    lambda_new: wp.vec3f
    delta: wp.vec3f


@wp.func
def block_project_delta_1(
    lambda_old: wp.float32,
    d_lambda: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> BlockScalarUpdate:
    """Project one accumulated multiplier into a box."""
    lambda_new = wp.clamp(lambda_old + d_lambda, lambda_min, lambda_max)
    update = BlockScalarUpdate()
    update.delta = lambda_new - lambda_old
    update.lambda_new = lambda_new
    return update


@wp.func
def block_project_delta_sor_1(
    lambda_old: wp.float32,
    d_lambda_unscaled: wp.float32,
    sor_boost: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> BlockScalarUpdate:
    """Apply SOR, then project one accumulated multiplier into a box."""
    return block_project_delta_1(lambda_old, d_lambda_unscaled * sor_boost, lambda_min, lambda_max)


@wp.func
def block_project_identity_delta_1(lambda_old: wp.float32, d_lambda: wp.float32) -> BlockScalarUpdate:
    """Identity-project one accumulated multiplier."""
    update = BlockScalarUpdate()
    update.delta = d_lambda
    update.lambda_new = lambda_old + d_lambda
    return update


@wp.func
def block_project_identity_delta_2(
    lambda1_old: wp.float32,
    lambda2_old: wp.float32,
    d_lambda1: wp.float32,
    d_lambda2: wp.float32,
) -> BlockVector2Update:
    """Identity-project two accumulated multipliers."""
    update = BlockVector2Update()
    update.delta = wp.vec2f(d_lambda1, d_lambda2)
    update.lambda_new = wp.vec2f(lambda1_old + d_lambda1, lambda2_old + d_lambda2)
    return update


@wp.func
def block_project_accumulated_bounded_1(
    d_lambda_unsoft: wp.float32,
    lambda_old: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> BlockScalarUpdate:
    """Apply soft-constraint coefficients, SOR, then clamp one row."""
    d_lambda = mass_coeff * d_lambda_unsoft - impulse_coeff * lambda_old
    return block_project_delta_sor_1(lambda_old, d_lambda, sor_boost, lambda_min, lambda_max)


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
) -> BlockVector2Update:
    """Apply SOR, then project a two-row friction update."""
    lambda_t1_raw = lambda_t1_old + d_lambda_t1_unscaled * sor_boost
    lambda_t2_raw = lambda_t2_old + d_lambda_t2_unscaled * sor_boost
    lambda_new = block_project_friction_circle_2(lambda_t1_raw, lambda_t2_raw, static_limit, kinetic_limit)
    update = BlockVector2Update()
    update.delta = wp.vec2f(lambda_new[0] - lambda_t1_old, lambda_new[1] - lambda_t2_old)
    update.lambda_new = lambda_new
    return update


@wp.func
def block_position_delta_1(
    inv_mass: wp.float32,
    d_lambda: wp.float32,
    grad: wp.vec3f,
) -> wp.vec3f:
    """Mass-weighted position delta for one XPBD row."""
    return inv_mass * d_lambda * grad


@wp.func
def block_position_delta_2(
    inv_mass: wp.float32,
    d_lambda: wp.vec2f,
    grad1: wp.vec3f,
    grad2: wp.vec3f,
) -> wp.vec3f:
    """Mass-weighted position delta for a two-row XPBD block."""
    return inv_mass * (d_lambda[0] * grad1 + d_lambda[1] * grad2)


@wp.func
def block_position_delta_2d_2(
    inv_mass: wp.float32,
    d_lambda: wp.vec2f,
    grad1: wp.vec2f,
    grad2: wp.vec2f,
) -> wp.vec2f:
    """Mass-weighted 2D position delta for a two-row XPBD block."""
    return wp.vec2f(
        inv_mass * (d_lambda[0] * grad1[0] + d_lambda[1] * grad2[0]),
        inv_mass * (d_lambda[0] * grad1[1] + d_lambda[1] * grad2[1]),
    )


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
) -> BlockScalarUpdate:
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
) -> BlockVector2Update:
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
) -> BlockVector2Update:
    """Solve a strict two-row XPBD block, apply SOR, then identity-project lambdas."""
    d = block_solve_xpbd_2_strict(A11, A12, A22, rhs1, rhs2, sor_boost, det_floor)
    return block_project_identity_delta_2(lambda1_old, lambda2_old, d[0], d[1])


@wp.func
def block_solve_position_rows1(rows: PositionRows1, sor_boost: wp.float32) -> BlockScalarUpdate:
    """Solve/project one prepared position-level XPBD row."""
    return block_solve_projected_xpbd_1(rows.A11, rows.residual, rows.lambda_old, sor_boost, rows.diag_floor)


@wp.func
def block_solve_position_rows2(rows: PositionRows2, sor_boost: wp.float32) -> BlockVector2Update:
    """Solve/project two prepared position-level XPBD rows."""
    return block_solve_projected_xpbd_2(
        rows.A11,
        rows.A12,
        rows.A22,
        rows.residual[0],
        rows.residual[1],
        rows.lambda_old[0],
        rows.lambda_old[1],
        sor_boost,
        rows.det_floor,
    )


@wp.func
def block_solve_position_rows2_strict(rows: PositionRows2, sor_boost: wp.float32) -> BlockVector2Update:
    """Solve/project two strict prepared position-level XPBD rows."""
    return block_solve_projected_xpbd_2_strict(
        rows.A11,
        rows.A12,
        rows.A22,
        rows.residual[0],
        rows.residual[1],
        rows.lambda_old[0],
        rows.lambda_old[1],
        sor_boost,
        rows.det_floor,
    )


@wp.func
def block_solve_accumulated_inverse_bounded_1(
    K_inv: wp.float32,
    rhs: wp.float32,
    lambda_old: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    lambda_min: wp.float32,
    lambda_max: wp.float32,
) -> BlockScalarUpdate:
    """Solve a cached scalar inverse row and clamp its accumulator."""
    d_lambda_unsoft = -(K_inv * rhs)
    return block_project_accumulated_bounded_1(
        d_lambda_unsoft,
        lambda_old,
        mass_coeff,
        impulse_coeff,
        sor_boost,
        lambda_min,
        lambda_max,
    )
