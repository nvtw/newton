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
    "VELOCITY_BLOCK_PROJECT_IDENTITY",
    "VELOCITY_ROWS3_PROJECT_BOUNDS",
    "VELOCITY_ROWS3_PROJECT_CONTACT_CONE",
    "BlockScalarUpdate",
    "BlockVector2Update",
    "BlockVector3Update",
    "BlockVector4Update",
    "PositionRows1",
    "PositionRows2",
    "RigidFrameBlock1",
    "RigidFrameBlock2",
    "RigidFrameBlock3",
    "RigidFrameBlock3Mixed",
    "RigidFrameRows1Update",
    "RigidFrameRows2Update",
    "RigidFrameRows3",
    "RigidFrameRows3State",
    "RigidFrameRows3Update",
    "VelocityBlock1",
    "VelocityBlock2",
    "VelocityBlock3",
    "VelocityBlock4",
    "VelocityBlockProjection",
    "VelocityRows3",
    "VelocityRows3Op",
    "VelocityRows3Projection",
    "VelocityRows3Update",
    "block_position_delta_1",
    "block_position_delta_2",
    "block_position_delta_2d_2",
    "block_project_accumulated_1",
    "block_project_accumulated_2",
    "block_project_accumulated_3",
    "block_project_accumulated_4",
    "block_project_accumulated_bounded_1",
    "block_project_delta_1",
    "block_project_delta_sor_1",
    "block_project_friction_circle_2",
    "block_project_friction_delta_sor_2",
    "block_project_identity_delta_1",
    "block_project_identity_delta_2",
    "block_project_velocity_block2_unsoft",
    "block_project_velocity_block3_unsoft",
    "block_solve_accumulated_inverse_1",
    "block_solve_accumulated_inverse_2",
    "block_solve_accumulated_inverse_3",
    "block_solve_accumulated_inverse_4",
    "block_solve_accumulated_inverse_bounded_1",
    "block_solve_inverse_2",
    "block_solve_inverse_3",
    "block_solve_inverse_4",
    "block_solve_position_rows1",
    "block_solve_position_rows2",
    "block_solve_position_rows2_strict",
    "block_solve_projected_xpbd_1",
    "block_solve_projected_xpbd_2",
    "block_solve_projected_xpbd_2_strict",
    "block_solve_rigid_frame_block1",
    "block_solve_rigid_frame_block2",
    "block_solve_rigid_frame_block3",
    "block_solve_rigid_frame_block3_bounded",
    "block_solve_rigid_frame_block3_mixed",
    "block_solve_rigid_frame_block3_planar_bounded",
    "block_solve_rigid_frame_rows3",
    "block_solve_rigid_frame_rows3_angular",
    "block_solve_rigid_frame_rows3_contact",
    "block_solve_rigid_frame_rows3_uniform_projection",
    "block_solve_symmetric_1",
    "block_solve_symmetric_2",
    "block_solve_symmetric_2_strict",
    "block_solve_velocity_block1",
    "block_solve_velocity_block1_projected",
    "block_solve_velocity_block2",
    "block_solve_velocity_block2_projected",
    "block_solve_velocity_block3",
    "block_solve_velocity_block3_projected",
    "block_solve_velocity_block4",
    "block_solve_velocity_block4_projected",
    "block_solve_velocity_rows3",
    "block_solve_velocity_rows3_bounded",
    "block_solve_velocity_rows3_contact_cone",
    "block_solve_velocity_rows3_op",
    "block_solve_velocity_rows3_op_uniform_projection",
    "block_solve_xpbd_1",
    "block_solve_xpbd_2",
    "block_solve_xpbd_2_strict",
]


BLOCK_LAMBDA_INF = wp.constant(wp.float32(1.0e30))
VELOCITY_ROWS3_PROJECT_BOUNDS = wp.constant(wp.int32(0))
VELOCITY_ROWS3_PROJECT_CONTACT_CONE = wp.constant(wp.int32(1))
VELOCITY_BLOCK_PROJECT_IDENTITY = wp.constant(wp.int32(0))


@wp.struct
class BlockScalarUpdate:
    delta: wp.float32
    lambda_new: wp.float32


@wp.struct
class BlockVector2Update:
    delta: wp.vec2f
    lambda_new: wp.vec2f


@wp.struct
class BlockVector3Update:
    delta: wp.vec3f
    lambda_new: wp.vec3f


@wp.struct
class BlockVector4Update:
    delta: wp.vec4f
    lambda_new: wp.vec4f


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
class VelocityBlock1:
    """Prepared one-row dense velocity block with cached inverse mass."""

    k_inv: wp.float32
    residual: wp.float32
    lambda_old: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class VelocityBlock2:
    """Prepared two-row dense velocity block with cached inverse mass."""

    k_inv: wp.mat22f
    residual: wp.vec2f
    lambda_old: wp.vec2f
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class VelocityBlock3:
    """Prepared three-row dense velocity block with cached inverse mass."""

    k_inv: wp.mat33f
    residual: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class VelocityBlock4:
    """Prepared four-row dense velocity block with cached inverse mass."""

    k_inv: wp.mat44f
    residual: wp.vec4f
    lambda_old: wp.vec4f
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class VelocityBlockProjection:
    """Projection parameters for prepared dense velocity blocks."""

    mode: wp.int32


@wp.struct
class VelocityRows3:
    """Prepared three scalar velocity rows for shared PGS updates."""

    k_inv: wp.vec3f
    residual: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.vec3f
    impulse_coeff: wp.vec3f
    lambda_min: wp.vec3f
    lambda_max: wp.vec3f


@wp.struct
class VelocityRows3Projection:
    """Projection parameters for prepared three-row velocity blocks."""

    mode: wp.int32
    friction_static: wp.float32
    friction_kinetic: wp.float32


@wp.struct
class VelocityRows3Op:
    """Flattened descriptor for three scalar velocity rows plus projection.

    This is the sidecar-friendly form of ``(rows, projection)``: prepare code
    can populate these fields, and iterate code can solve/project without
    knowing whether the source was a contact, drive, motor, or limit row.
    """

    k_inv: wp.vec3f
    residual: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.vec3f
    impulse_coeff: wp.vec3f
    lambda_min: wp.vec3f
    lambda_max: wp.vec3f
    projection_mode: wp.int32
    friction_static: wp.float32
    friction_kinetic: wp.float32


@wp.struct
class VelocityRows3Update:
    delta: wp.vec3f
    lambda_new: wp.vec3f


@wp.struct
class RigidFrameBlock1:
    """Compact rigid one-row frame descriptor with a scalar solve.

    mode gates the row shape as (linear, cross, angular) so point,
    angular-direct, and COM-linear rows share the same local update form.
    """

    axis: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    mode: wp.vec3f
    k_inv: wp.float32
    bias: wp.float32
    lambda_old: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class RigidFrameBlock2:
    """Compact rigid two-row frame descriptor with a dense 2x2 solve."""

    axis0: wp.vec3f
    axis1: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    mode: wp.vec3f
    k_inv: wp.mat22f
    bias: wp.vec2f
    lambda_old: wp.vec2f
    mass_coeff: wp.float32
    impulse_coeff: wp.float32


@wp.struct
class RigidFrameBlock3:
    """Compact rigid three-row frame descriptor with a dense 3x3 solve.

    This is the block-coupled counterpart to :class:`RigidFrameRows3`.
    Contact rows can populate ``k_inv`` as a diagonal matrix, while joint
    point-lock rows can use the full inverse effective mass block. The residual,
    projection, and rigid body apply sequence stay identical.
    """

    axis0: wp.vec3f
    axis1: wp.vec3f
    axis2: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    mode: wp.vec3f
    k_inv: wp.mat33f
    bias: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.vec3f
    impulse_coeff: wp.vec3f
    lambda_min: wp.vec3f
    lambda_max: wp.vec3f
    projection_mode: wp.int32
    friction_static: wp.float32
    friction_kinetic: wp.float32


@wp.struct
class RigidFrameBlock3Mixed:
    """Dense 3-row frame descriptor with per-row residual/apply modes.

    ``linear_mode``, ``cross_mode``, and ``angular_mode`` gate each row
    independently. This covers point/contact rows ``(1, 1, 0)``, angular rows
    ``(0, 0, 1)``, and mixed blocks such as planar constraints where one row
    is linear and two rows are angular.
    """

    axis0: wp.vec3f
    axis1: wp.vec3f
    axis2: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    linear_mode: wp.vec3f
    cross_mode: wp.vec3f
    angular_mode: wp.vec3f
    k_inv: wp.mat33f
    bias: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.vec3f
    impulse_coeff: wp.vec3f
    lambda_min: wp.vec3f
    lambda_max: wp.vec3f
    projection_mode: wp.int32
    friction_static: wp.float32
    friction_kinetic: wp.float32


@wp.struct
class RigidFrameRows3:
    """Compact prepared rigid three-row frame descriptor.

    ``axis0..2`` define the row frame. ``r0``/``r1`` are offsets from body A/B
    centers to the row point. ``mode`` gates the branchless residual/apply
    shape: ``(linear, cross, angular)``. Contact rows use ``(1, 1, 0)`` and
    angular-direct joint rows use ``(0, 0, 1)``. Projection and soft-row
    coefficient fields mirror :class:`VelocityRows3Op` so the same PGS
    projection code handles contacts, drives, motors, and bounded rows.
    """

    axis0: wp.vec3f
    axis1: wp.vec3f
    axis2: wp.vec3f
    r0: wp.vec3f
    r1: wp.vec3f
    mode: wp.vec3f
    k_inv: wp.vec3f
    bias: wp.vec3f
    lambda_old: wp.vec3f
    mass_coeff: wp.vec3f
    impulse_coeff: wp.vec3f
    lambda_min: wp.vec3f
    lambda_max: wp.vec3f
    projection_mode: wp.int32
    friction_static: wp.float32
    friction_kinetic: wp.float32


@wp.struct
class RigidFrameRows3State:
    """Body-pair state consumed by :func:`block_solve_rigid_frame_rows3`."""

    v_a: wp.vec3f
    w_a: wp.vec3f
    v_b: wp.vec3f
    w_b: wp.vec3f
    inv_m_a: wp.float32
    inv_m_b: wp.float32
    inv_i_a: wp.mat33f
    inv_i_b: wp.mat33f


@wp.struct
class RigidFrameRows1Update:
    v_a: wp.vec3f
    w_a: wp.vec3f
    v_b: wp.vec3f
    w_b: wp.vec3f
    lambda_new: wp.float32
    delta: wp.float32


@wp.struct
class RigidFrameRows2Update:
    v_a: wp.vec3f
    w_a: wp.vec3f
    v_b: wp.vec3f
    w_b: wp.vec3f
    lambda_new: wp.vec2f
    delta: wp.vec2f


@wp.struct
class RigidFrameRows3Update:
    v_a: wp.vec3f
    w_a: wp.vec3f
    v_b: wp.vec3f
    w_b: wp.vec3f
    lambda_new: wp.vec3f
    delta: wp.vec3f


@wp.func
def _rows3_dot(row0: wp.vec3f, row1: wp.vec3f, row2: wp.vec3f, x: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(wp.dot(row0, x), wp.dot(row1, x), wp.dot(row2, x))


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
def block_project_accumulated_1(
    d_lambda_unsoft: wp.float32,
    lambda_old: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockScalarUpdate:
    """Apply soft-constraint coefficients, SOR, then identity-project one row."""
    d_lambda = (mass_coeff * d_lambda_unsoft - impulse_coeff * lambda_old) * sor_boost
    return block_project_identity_delta_1(lambda_old, d_lambda)


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
def block_project_accumulated_2(
    d_lambda_unsoft: wp.vec2f,
    lambda_old: wp.vec2f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector2Update:
    """Apply soft-constraint coefficients, SOR, then identity-project two rows."""
    d_lambda = (mass_coeff * d_lambda_unsoft - impulse_coeff * lambda_old) * sor_boost
    update = BlockVector2Update()
    update.delta = d_lambda
    update.lambda_new = lambda_old + d_lambda
    return update


@wp.func
def block_project_accumulated_3(
    d_lambda_unsoft: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector3Update:
    """Apply soft-constraint coefficients, SOR, then identity-project three rows."""
    d_lambda = (mass_coeff * d_lambda_unsoft - impulse_coeff * lambda_old) * sor_boost
    update = BlockVector3Update()
    update.delta = d_lambda
    update.lambda_new = lambda_old + d_lambda
    return update


@wp.func
def block_project_accumulated_4(
    d_lambda_unsoft: wp.vec4f,
    lambda_old: wp.vec4f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector4Update:
    """Apply soft-constraint coefficients, SOR, then identity-project four rows."""
    d_lambda = (mass_coeff * d_lambda_unsoft - impulse_coeff * lambda_old) * sor_boost
    update = BlockVector4Update()
    update.delta = d_lambda
    update.lambda_new = lambda_old + d_lambda
    return update


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
def block_solve_accumulated_inverse_1(
    K_inv: wp.float32,
    rhs: wp.float32,
    lambda_old: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockScalarUpdate:
    """Solve a cached scalar inverse row and identity-project its accumulator."""
    d_lambda_unsoft = -(K_inv * rhs)
    return block_project_accumulated_1(d_lambda_unsoft, lambda_old, mass_coeff, impulse_coeff, sor_boost)


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


@wp.func
def block_solve_velocity_rows3_bounded(rows: VelocityRows3, sor_boost: wp.float32) -> VelocityRows3Update:
    """Solve/project three independent bounded velocity rows."""
    row0 = block_solve_accumulated_inverse_bounded_1(
        rows.k_inv[0],
        rows.residual[0],
        rows.lambda_old[0],
        rows.mass_coeff[0],
        rows.impulse_coeff[0],
        sor_boost,
        rows.lambda_min[0],
        rows.lambda_max[0],
    )
    row1 = block_solve_accumulated_inverse_bounded_1(
        rows.k_inv[1],
        rows.residual[1],
        rows.lambda_old[1],
        rows.mass_coeff[1],
        rows.impulse_coeff[1],
        sor_boost,
        rows.lambda_min[1],
        rows.lambda_max[1],
    )
    row2 = block_solve_accumulated_inverse_bounded_1(
        rows.k_inv[2],
        rows.residual[2],
        rows.lambda_old[2],
        rows.mass_coeff[2],
        rows.impulse_coeff[2],
        sor_boost,
        rows.lambda_min[2],
        rows.lambda_max[2],
    )
    update = VelocityRows3Update()
    update.delta = wp.vec3f(row0.delta, row1.delta, row2.delta)
    update.lambda_new = wp.vec3f(row0.lambda_new, row1.lambda_new, row2.lambda_new)
    return update


@wp.func
def block_solve_velocity_rows3_contact_cone(
    rows: VelocityRows3,
    sor_boost: wp.float32,
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
) -> VelocityRows3Update:
    """Solve/project a contact block: normal row plus two tangent rows."""
    row0 = block_solve_accumulated_inverse_bounded_1(
        rows.k_inv[0],
        rows.residual[0],
        rows.lambda_old[0],
        rows.mass_coeff[0],
        rows.impulse_coeff[0],
        sor_boost,
        rows.lambda_min[0],
        rows.lambda_max[0],
    )
    d_lambda_t1 = (
        rows.mass_coeff[1] * (-(rows.k_inv[1] * rows.residual[1])) - rows.impulse_coeff[1] * rows.lambda_old[1]
    )
    d_lambda_t2 = (
        rows.mass_coeff[2] * (-(rows.k_inv[2] * rows.residual[2])) - rows.impulse_coeff[2] * rows.lambda_old[2]
    )
    tangents = block_project_friction_delta_sor_2(
        rows.lambda_old[1],
        rows.lambda_old[2],
        d_lambda_t1,
        d_lambda_t2,
        sor_boost,
        friction_static * row0.lambda_new,
        friction_kinetic * row0.lambda_new,
    )
    update = VelocityRows3Update()
    update.delta = wp.vec3f(row0.delta, tangents.delta[0], tangents.delta[1])
    update.lambda_new = wp.vec3f(row0.lambda_new, tangents.lambda_new[0], tangents.lambda_new[1])
    return update


@wp.func
def block_solve_velocity_rows3(
    rows: VelocityRows3,
    projection: VelocityRows3Projection,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    """Solve/project three prepared velocity rows from an explicit projection descriptor."""
    if projection.mode == VELOCITY_ROWS3_PROJECT_CONTACT_CONE:
        return block_solve_velocity_rows3_contact_cone(
            rows,
            sor_boost,
            projection.friction_static,
            projection.friction_kinetic,
        )
    return block_solve_velocity_rows3_bounded(rows, sor_boost)


@wp.func
def block_solve_velocity_rows3_op(op: VelocityRows3Op, sor_boost: wp.float32) -> VelocityRows3Update:
    """Solve/project one flattened three-row velocity operation."""
    rows = VelocityRows3()
    rows.k_inv = op.k_inv
    rows.residual = op.residual
    rows.lambda_old = op.lambda_old
    rows.mass_coeff = op.mass_coeff
    rows.impulse_coeff = op.impulse_coeff
    rows.lambda_min = op.lambda_min
    rows.lambda_max = op.lambda_max

    projection = VelocityRows3Projection()
    projection.mode = op.projection_mode
    projection.friction_static = op.friction_static
    projection.friction_kinetic = op.friction_kinetic
    return block_solve_velocity_rows3(rows, projection, sor_boost)


@wp.func
def block_solve_velocity_rows3_op_uniform_projection(
    op: VelocityRows3Op,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    """Solve/project one three-row op with branchless projection-mode selection.

    Rows 0-2 are always solved as scalar bounded rows. Rows 1-2 are also
    projected as a contact friction disk, then ``projection_mode`` selects the
    bounded or cone result with ``wp.where``. This is intentionally a separate
    entry point from :func:`block_solve_velocity_rows3_op`: it is useful for
    mixed-family megakernel experiments, but it does extra arithmetic for pure
    bounded joint rows.
    """
    row0 = block_solve_accumulated_inverse_bounded_1(
        op.k_inv[0],
        op.residual[0],
        op.lambda_old[0],
        op.mass_coeff[0],
        op.impulse_coeff[0],
        sor_boost,
        op.lambda_min[0],
        op.lambda_max[0],
    )

    d_lambda1 = op.mass_coeff[1] * (-(op.k_inv[1] * op.residual[1])) - op.impulse_coeff[1] * op.lambda_old[1]
    d_lambda2 = op.mass_coeff[2] * (-(op.k_inv[2] * op.residual[2])) - op.impulse_coeff[2] * op.lambda_old[2]
    lambda1_raw = op.lambda_old[1] + d_lambda1 * sor_boost
    lambda2_raw = op.lambda_old[2] + d_lambda2 * sor_boost

    lambda1_box = wp.clamp(lambda1_raw, op.lambda_min[1], op.lambda_max[1])
    lambda2_box = wp.clamp(lambda2_raw, op.lambda_min[2], op.lambda_max[2])
    lambda_friction = block_project_friction_circle_2(
        lambda1_raw,
        lambda2_raw,
        op.friction_static * row0.lambda_new,
        op.friction_kinetic * row0.lambda_new,
    )

    is_contact = op.projection_mode == VELOCITY_ROWS3_PROJECT_CONTACT_CONE
    lambda1_new = wp.where(is_contact, lambda_friction[0], lambda1_box)
    lambda2_new = wp.where(is_contact, lambda_friction[1], lambda2_box)

    update = VelocityRows3Update()
    update.delta = wp.vec3f(
        row0.delta,
        lambda1_new - op.lambda_old[1],
        lambda2_new - op.lambda_old[2],
    )
    update.lambda_new = wp.vec3f(row0.lambda_new, lambda1_new, lambda2_new)
    return update


@wp.func
def block_solve_rigid_frame_block1(
    rows: RigidFrameBlock1,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows1Update:
    """Solve/project/apply one compact rigid frame row."""
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = wp.dot(rows.axis, rel) + rows.bias

    projection = block_solve_accumulated_inverse_1(
        rows.k_inv,
        residual,
        rows.lambda_old,
        rows.mass_coeff,
        rows.impulse_coeff,
        sor_boost,
    )
    impulse = projection.delta * rows.axis

    update = RigidFrameRows1Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = projection.delta
    return update


@wp.func
def block_solve_rigid_frame_block2(
    rows: RigidFrameBlock2,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows2Update:
    """Solve/project/apply one compact rigid two-row frame block."""
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = wp.vec2f(wp.dot(rows.axis0, rel), wp.dot(rows.axis1, rel)) + rows.bias

    projection = block_solve_accumulated_inverse_2(
        rows.k_inv,
        residual,
        rows.lambda_old,
        rows.mass_coeff,
        rows.impulse_coeff,
        sor_boost,
    )
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1

    update = RigidFrameRows2Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def _rigid_frame_rows3_projection(
    rows: RigidFrameRows3,
    residual: wp.vec3f,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    op = VelocityRows3Op()
    op.k_inv = rows.k_inv
    op.residual = residual
    op.lambda_old = rows.lambda_old
    op.mass_coeff = rows.mass_coeff
    op.impulse_coeff = rows.impulse_coeff
    op.lambda_min = rows.lambda_min
    op.lambda_max = rows.lambda_max
    op.projection_mode = rows.projection_mode
    op.friction_static = rows.friction_static
    op.friction_kinetic = rows.friction_kinetic
    return block_solve_velocity_rows3_op(op, sor_boost)


@wp.func
def _rigid_frame_rows3_projection_uniform(
    rows: RigidFrameRows3,
    residual: wp.vec3f,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    op = VelocityRows3Op()
    op.k_inv = rows.k_inv
    op.residual = residual
    op.lambda_old = rows.lambda_old
    op.mass_coeff = rows.mass_coeff
    op.impulse_coeff = rows.impulse_coeff
    op.lambda_min = rows.lambda_min
    op.lambda_max = rows.lambda_max
    op.projection_mode = rows.projection_mode
    op.friction_static = rows.friction_static
    op.friction_kinetic = rows.friction_kinetic
    return block_solve_velocity_rows3_op_uniform_projection(op, sor_boost)


@wp.func
def block_solve_rigid_frame_rows3_contact(
    rows: RigidFrameRows3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply a compact contact-point three-row operation."""
    rel = state.v_b - state.v_a + wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0)
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias
    projection = _rigid_frame_rows3_projection(rows, residual, sor_boost)
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - state.inv_m_a * impulse
    update.v_b = state.v_b + state.inv_m_b * impulse
    update.w_a = state.w_a - state.inv_i_a @ wp.cross(rows.r0, impulse)
    update.w_b = state.w_b + state.inv_i_b @ wp.cross(rows.r1, impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_rigid_frame_rows3_angular(
    rows: RigidFrameRows3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply a compact angular-direct three-row operation."""
    rel = state.w_b - state.w_a
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias
    projection = _rigid_frame_rows3_projection(rows, residual, sor_boost)
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a
    update.v_b = state.v_b
    update.w_a = state.w_a - state.inv_i_a @ impulse
    update.w_b = state.w_b + state.inv_i_b @ impulse
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_rigid_frame_rows3(
    rows: RigidFrameRows3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply one compact rigid three-row frame operation.

    This is a branchless local PGS row-block shape for contact-like rows and
    angular-direct joint rows. It deliberately uses compact axes/offsets instead
    of a dense max-J sidecar, keeping memory traffic closer to typed kernels
    while still sharing the same residual/projection/apply sequence.
    """
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias

    op = VelocityRows3Op()
    op.k_inv = rows.k_inv
    op.residual = residual
    op.lambda_old = rows.lambda_old
    op.mass_coeff = rows.mass_coeff
    op.impulse_coeff = rows.impulse_coeff
    op.lambda_min = rows.lambda_min
    op.lambda_max = rows.lambda_max
    op.projection_mode = rows.projection_mode
    op.friction_static = rows.friction_static
    op.friction_kinetic = rows.friction_kinetic

    projection = block_solve_velocity_rows3_op(op, sor_boost)
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_rigid_frame_rows3_uniform_projection(
    rows: RigidFrameRows3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply one rigid three-row frame op without a projection branch."""
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias

    projection = _rigid_frame_rows3_projection_uniform(rows, residual, sor_boost)
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def _dense_velocity_rows3_projection_bounded(
    k_inv: wp.mat33f,
    residual: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.vec3f,
    impulse_coeff: wp.vec3f,
    lambda_min: wp.vec3f,
    lambda_max: wp.vec3f,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    d_lambda_unsoft = -(k_inv @ residual)
    d_lambda = wp.vec3f(
        (mass_coeff[0] * d_lambda_unsoft[0] - impulse_coeff[0] * lambda_old[0]) * sor_boost,
        (mass_coeff[1] * d_lambda_unsoft[1] - impulse_coeff[1] * lambda_old[1]) * sor_boost,
        (mass_coeff[2] * d_lambda_unsoft[2] - impulse_coeff[2] * lambda_old[2]) * sor_boost,
    )
    lambda_raw = lambda_old + d_lambda
    lambda_new = wp.vec3f(
        wp.clamp(lambda_raw[0], lambda_min[0], lambda_max[0]),
        wp.clamp(lambda_raw[1], lambda_min[1], lambda_max[1]),
        wp.clamp(lambda_raw[2], lambda_min[2], lambda_max[2]),
    )

    update = VelocityRows3Update()
    update.lambda_new = lambda_new
    update.delta = lambda_new - lambda_old
    return update


@wp.func
def _dense_velocity_rows3_projection_uniform(
    k_inv: wp.mat33f,
    residual: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.vec3f,
    impulse_coeff: wp.vec3f,
    lambda_min: wp.vec3f,
    lambda_max: wp.vec3f,
    projection_mode: wp.int32,
    friction_static: wp.float32,
    friction_kinetic: wp.float32,
    sor_boost: wp.float32,
) -> VelocityRows3Update:
    d_lambda_unsoft = -(k_inv @ residual)
    d_lambda = wp.vec3f(
        (mass_coeff[0] * d_lambda_unsoft[0] - impulse_coeff[0] * lambda_old[0]) * sor_boost,
        (mass_coeff[1] * d_lambda_unsoft[1] - impulse_coeff[1] * lambda_old[1]) * sor_boost,
        (mass_coeff[2] * d_lambda_unsoft[2] - impulse_coeff[2] * lambda_old[2]) * sor_boost,
    )
    lambda_raw = lambda_old + d_lambda

    lambda0_new = wp.clamp(lambda_raw[0], lambda_min[0], lambda_max[0])
    lambda1_box = wp.clamp(lambda_raw[1], lambda_min[1], lambda_max[1])
    lambda2_box = wp.clamp(lambda_raw[2], lambda_min[2], lambda_max[2])
    lambda_friction = block_project_friction_circle_2(
        lambda_raw[1],
        lambda_raw[2],
        friction_static * lambda0_new,
        friction_kinetic * lambda0_new,
    )

    is_contact = projection_mode == VELOCITY_ROWS3_PROJECT_CONTACT_CONE
    lambda1_new = wp.where(is_contact, lambda_friction[0], lambda1_box)
    lambda2_new = wp.where(is_contact, lambda_friction[1], lambda2_box)
    lambda_new = wp.vec3f(lambda0_new, lambda1_new, lambda2_new)

    update = VelocityRows3Update()
    update.lambda_new = lambda_new
    update.delta = lambda_new - lambda_old
    return update


@wp.func
def block_solve_rigid_frame_block3_bounded(
    rows: RigidFrameBlock3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply one dense frame op with independent bounded rows."""
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias

    projection = _dense_velocity_rows3_projection_bounded(
        rows.k_inv,
        residual,
        rows.lambda_old,
        rows.mass_coeff,
        rows.impulse_coeff,
        rows.lambda_min,
        rows.lambda_max,
        sor_boost,
    )
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def _mixed_frame_row_residual(
    axis: wp.vec3f,
    linear_mode: wp.float32,
    cross_mode: wp.float32,
    angular_mode: wp.float32,
    rel_linear: wp.vec3f,
    rel_cross: wp.vec3f,
    rel_angular: wp.vec3f,
) -> wp.float32:
    return wp.dot(axis, linear_mode * rel_linear + cross_mode * rel_cross + angular_mode * rel_angular)


@wp.func
def block_solve_rigid_frame_block3_mixed(
    rows: RigidFrameBlock3Mixed,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply one dense frame op with row-wise modes."""
    rel_linear = state.v_b - state.v_a
    rel_cross = wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0)
    rel_angular = state.w_b - state.w_a
    residual = (
        wp.vec3f(
            _mixed_frame_row_residual(
                rows.axis0,
                rows.linear_mode[0],
                rows.cross_mode[0],
                rows.angular_mode[0],
                rel_linear,
                rel_cross,
                rel_angular,
            ),
            _mixed_frame_row_residual(
                rows.axis1,
                rows.linear_mode[1],
                rows.cross_mode[1],
                rows.angular_mode[1],
                rel_linear,
                rel_cross,
                rel_angular,
            ),
            _mixed_frame_row_residual(
                rows.axis2,
                rows.linear_mode[2],
                rows.cross_mode[2],
                rows.angular_mode[2],
                rel_linear,
                rel_cross,
                rel_angular,
            ),
        )
        + rows.bias
    )

    projection = VelocityRows3Update()
    if rows.projection_mode == VELOCITY_ROWS3_PROJECT_CONTACT_CONE:
        projection = _dense_velocity_rows3_projection_uniform(
            rows.k_inv,
            residual,
            rows.lambda_old,
            rows.mass_coeff,
            rows.impulse_coeff,
            rows.lambda_min,
            rows.lambda_max,
            rows.projection_mode,
            rows.friction_static,
            rows.friction_kinetic,
            sor_boost,
        )
    else:
        projection = _dense_velocity_rows3_projection_bounded(
            rows.k_inv,
            residual,
            rows.lambda_old,
            rows.mass_coeff,
            rows.impulse_coeff,
            rows.lambda_min,
            rows.lambda_max,
            sor_boost,
        )

    d = projection.delta
    linear_impulse = (
        rows.linear_mode[0] * d[0] * rows.axis0
        + rows.linear_mode[1] * d[1] * rows.axis1
        + rows.linear_mode[2] * d[2] * rows.axis2
    )
    cross_impulse = (
        rows.cross_mode[0] * d[0] * rows.axis0
        + rows.cross_mode[1] * d[1] * rows.axis1
        + rows.cross_mode[2] * d[2] * rows.axis2
    )
    angular_impulse = (
        rows.angular_mode[0] * d[0] * rows.axis0
        + rows.angular_mode[1] * d[1] * rows.axis1
        + rows.angular_mode[2] * d[2] * rows.axis2
    )

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - state.inv_m_a * linear_impulse
    update.v_b = state.v_b + state.inv_m_b * linear_impulse
    update.w_a = state.w_a + state.inv_i_a @ (-wp.cross(rows.r0, cross_impulse) - angular_impulse)
    update.w_b = state.w_b + state.inv_i_b @ (wp.cross(rows.r1, cross_impulse) + angular_impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_rigid_frame_block3_planar_bounded(
    rows: RigidFrameBlock3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply a planar dense frame op with bounded rows.

    The row shape is fixed to one linear row along axis0 plus two angular
    rows along axis1 and axis2. It keeps planar joints on the same
    dense frame-block solve/projection path without carrying row-wise mode
    vectors through the hot iterate path.
    """
    rel_linear = state.v_b - state.v_a
    rel_angular = state.w_b - state.w_a
    residual = (
        wp.vec3f(
            wp.dot(rows.axis0, rel_linear),
            wp.dot(rows.axis1, rel_angular),
            wp.dot(rows.axis2, rel_angular),
        )
        + rows.bias
    )

    projection = _dense_velocity_rows3_projection_bounded(
        rows.k_inv,
        residual,
        rows.lambda_old,
        rows.mass_coeff,
        rows.impulse_coeff,
        rows.lambda_min,
        rows.lambda_max,
        sor_boost,
    )
    d = projection.delta
    linear_impulse = d[0] * rows.axis0
    angular_impulse = d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - state.inv_m_a * linear_impulse
    update.v_b = state.v_b + state.inv_m_b * linear_impulse
    update.w_a = state.w_a - state.inv_i_a @ angular_impulse
    update.w_b = state.w_b + state.inv_i_b @ angular_impulse
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_rigid_frame_block3(
    rows: RigidFrameBlock3,
    state: RigidFrameRows3State,
    sor_boost: wp.float32,
) -> RigidFrameRows3Update:
    """Solve/project/apply one rigid three-row frame op with a dense 3x3 block."""
    linear_scale = rows.mode[0]
    cross_scale = rows.mode[1]
    angular_scale = rows.mode[2]

    rel = (
        linear_scale * (state.v_b - state.v_a)
        + cross_scale * (wp.cross(state.w_b, rows.r1) - wp.cross(state.w_a, rows.r0))
        + angular_scale * (state.w_b - state.w_a)
    )
    residual = _rows3_dot(rows.axis0, rows.axis1, rows.axis2, rel) + rows.bias

    projection = _dense_velocity_rows3_projection_uniform(
        rows.k_inv,
        residual,
        rows.lambda_old,
        rows.mass_coeff,
        rows.impulse_coeff,
        rows.lambda_min,
        rows.lambda_max,
        rows.projection_mode,
        rows.friction_static,
        rows.friction_kinetic,
        sor_boost,
    )
    d = projection.delta
    impulse = d[0] * rows.axis0 + d[1] * rows.axis1 + d[2] * rows.axis2

    update = RigidFrameRows3Update()
    update.v_a = state.v_a - linear_scale * state.inv_m_a * impulse
    update.v_b = state.v_b + linear_scale * state.inv_m_b * impulse
    update.w_a = state.w_a + state.inv_i_a @ (-cross_scale * wp.cross(rows.r0, impulse) - angular_scale * impulse)
    update.w_b = state.w_b + state.inv_i_b @ (cross_scale * wp.cross(rows.r1, impulse) + angular_scale * impulse)
    update.lambda_new = projection.lambda_new
    update.delta = d
    return update


@wp.func
def block_solve_accumulated_inverse_2(
    K_inv: wp.mat22f,
    rhs: wp.vec2f,
    lambda_old: wp.vec2f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector2Update:
    """Solve a cached 2x2 inverse block and identity-project its accumulator."""
    d_lambda_unsoft = block_solve_inverse_2(K_inv, rhs)
    return block_project_accumulated_2(d_lambda_unsoft, lambda_old, mass_coeff, impulse_coeff, sor_boost)


@wp.func
def block_solve_accumulated_inverse_3(
    K_inv: wp.mat33f,
    rhs: wp.vec3f,
    lambda_old: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector3Update:
    """Solve a cached 3x3 inverse block and identity-project its accumulator."""
    d_lambda_unsoft = block_solve_inverse_3(K_inv, rhs)
    return block_project_accumulated_3(d_lambda_unsoft, lambda_old, mass_coeff, impulse_coeff, sor_boost)


@wp.func
def block_solve_accumulated_inverse_4(
    K_inv: wp.mat44f,
    rhs: wp.vec4f,
    lambda_old: wp.vec4f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
) -> BlockVector4Update:
    """Solve a cached 4x4 inverse block and identity-project its accumulator."""
    d_lambda_unsoft = block_solve_inverse_4(K_inv, rhs)
    return block_project_accumulated_4(d_lambda_unsoft, lambda_old, mass_coeff, impulse_coeff, sor_boost)


@wp.func
def block_project_velocity_block2_unsoft(
    block: VelocityBlock2,
    d_lambda_unsoft: wp.vec2f,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockVector2Update:
    """Project a prepared two-row dense velocity block from a precomputed solve."""
    return block_project_accumulated_2(
        d_lambda_unsoft,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_project_velocity_block3_unsoft(
    block: VelocityBlock3,
    d_lambda_unsoft: wp.vec3f,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockVector3Update:
    """Project a prepared three-row dense velocity block from a precomputed solve."""
    return block_project_accumulated_3(
        d_lambda_unsoft,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_solve_velocity_block1_projected(
    block: VelocityBlock1,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockScalarUpdate:
    """Solve/project one prepared dense velocity block from an explicit projection descriptor."""
    # Dense joint blocks currently use identity projection; the explicit
    # descriptor keeps their call shape aligned with contact/row blocks.
    return block_solve_accumulated_inverse_1(
        block.k_inv,
        block.residual,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_solve_velocity_block1(block: VelocityBlock1, sor_boost: wp.float32) -> BlockScalarUpdate:
    """Solve/project one prepared dense velocity block."""
    projection = VelocityBlockProjection()
    projection.mode = VELOCITY_BLOCK_PROJECT_IDENTITY
    return block_solve_velocity_block1_projected(block, projection, sor_boost)


@wp.func
def block_solve_velocity_block2_projected(
    block: VelocityBlock2,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockVector2Update:
    """Solve/project two prepared dense velocity rows from an explicit projection descriptor."""
    return block_solve_accumulated_inverse_2(
        block.k_inv,
        block.residual,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_solve_velocity_block2(block: VelocityBlock2, sor_boost: wp.float32) -> BlockVector2Update:
    """Solve/project two prepared dense velocity rows."""
    projection = VelocityBlockProjection()
    projection.mode = VELOCITY_BLOCK_PROJECT_IDENTITY
    return block_solve_velocity_block2_projected(block, projection, sor_boost)


@wp.func
def block_solve_velocity_block3_projected(
    block: VelocityBlock3,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockVector3Update:
    """Solve/project three prepared dense velocity rows from an explicit projection descriptor."""
    return block_solve_accumulated_inverse_3(
        block.k_inv,
        block.residual,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_solve_velocity_block3(block: VelocityBlock3, sor_boost: wp.float32) -> BlockVector3Update:
    """Solve/project three prepared dense velocity rows."""
    projection = VelocityBlockProjection()
    projection.mode = VELOCITY_BLOCK_PROJECT_IDENTITY
    return block_solve_velocity_block3_projected(block, projection, sor_boost)


@wp.func
def block_solve_velocity_block4_projected(
    block: VelocityBlock4,
    projection: VelocityBlockProjection,
    sor_boost: wp.float32,
) -> BlockVector4Update:
    """Solve/project four prepared dense velocity rows from an explicit projection descriptor."""
    return block_solve_accumulated_inverse_4(
        block.k_inv,
        block.residual,
        block.lambda_old,
        block.mass_coeff,
        block.impulse_coeff,
        sor_boost,
    )


@wp.func
def block_solve_velocity_block4(block: VelocityBlock4, sor_boost: wp.float32) -> BlockVector4Update:
    """Solve/project four prepared dense velocity rows."""
    projection = VelocityBlockProjection()
    projection.mode = VELOCITY_BLOCK_PROJECT_IDENTITY
    return block_solve_velocity_block4_projected(block, projection, sor_boost)


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
