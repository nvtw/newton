# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mixed XPBD soft-hexahedron constraint.

Uses the same split that works well in ``xpbd-fem`` for linear hexes:
an integrated trilinear strain energy plus a reduced center-point
volume row. Evaluating strain at the eight Gauss points prevents the
classic hourglass mode that a single center deformation gradient cannot
see, while the center volume row keeps the nearly-incompressible solve
cheap and robust.

Optionally, the integrated strain row can use an ARAP residual at each
Gauss point. This keeps the large-rotation behavior of the tetrahedral
ARAP path without falling back to a one-point hex strain evaluation.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import (
    block_position_delta_2,
    block_solve_projected_xpbd_2,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_SOFT_HEXAHEDRON,
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
    deformation_gradient_determinant_cofactor,
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
    "SOFT_HEX_DWORDS",
    "SOFT_HEX_STRAIN_MODEL_ARAP",
    "SOFT_HEX_STRAIN_MODEL_TRACE",
    "SOFT_HEX_TIME_US_OFFSET",
    "SoftHexahedronData",
    "soft_hex_init_rows_from_arrays_kernel",
    "soft_hexahedron_iterate_at",
    "soft_hexahedron_prepare_for_iteration_at",
    "soft_hexahedron_set_alpha_d",
    "soft_hexahedron_set_alpha_h",
    "soft_hexahedron_set_beta_d",
    "soft_hexahedron_set_beta_h",
    "soft_hexahedron_set_body1",
    "soft_hexahedron_set_body2",
    "soft_hexahedron_set_body3",
    "soft_hexahedron_set_body4",
    "soft_hexahedron_set_body5",
    "soft_hexahedron_set_body6",
    "soft_hexahedron_set_body7",
    "soft_hexahedron_set_body8",
    "soft_hexahedron_set_gamma",
    "soft_hexahedron_set_inv_rest",
    "soft_hexahedron_set_rest_volume",
    "soft_hexahedron_set_strain_model",
    "soft_hexahedron_set_type",
]


_PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))

SOFT_HEX_STRAIN_MODEL_TRACE: int = 0
SOFT_HEX_STRAIN_MODEL_ARAP: int = 1
_SOFT_HEX_STRAIN_MODEL_ARAP = wp.constant(wp.int32(SOFT_HEX_STRAIN_MODEL_ARAP))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class SoftHexahedronData:
    """Per-constraint dword layout for one mixed strain/volume hexahedron.

    Mirrors :class:`SoftTetNeoHookeanData` field-for-field with the
    4-body endpoint set extended to 8. The mandatory 3-int32 header
    (``constraint_type`` / ``body1`` / ``body2``) lives at dwords 0/1/2.
    """

    constraint_type: wp.int32
    body1: wp.int32  # particle index of corner 0 (-,-,-)
    body2: wp.int32  # particle index of corner 1 (+,-,-)
    body3: wp.int32  # particle index of corner 2 (+,+,-)
    body4: wp.int32  # particle index of corner 3 (-,+,-)
    body5: wp.int32  # particle index of corner 4 (-,-,+)
    body6: wp.int32  # particle index of corner 5 (+,-,+)
    body7: wp.int32  # particle index of corner 6 (+,+,+)
    body8: wp.int32  # particle index of corner 7 (-,+,+)
    strain_model: wp.int32  # 0: integrated trace strain, 1: integrated ARAP strain

    # inv_rest = J^{-T} where J is the rest Jacobian at the hex center.
    # The volume row uses center gradients; the strain row reuses the
    # same affine map across the eight Gauss points.
    inv_rest: wp.mat33f
    rest_volume: wp.float32  # V = 8 * det(J)

    #: ``1 + mu / lambda`` -- stable Neo-Hookean volume offset.
    gamma: wp.float32
    #: ``1 / (lambda * V_rest)`` -- volume compliance.
    alpha_h: wp.float32
    #: ``1 / (mu * V_rest)`` -- integrated strain compliance.
    alpha_d: wp.float32
    #: Macklin XPBD damping coefficient on the volume row [1/s].
    beta_h: wp.float32
    #: Macklin XPBD damping coefficient on the strain row [1/s].
    beta_d: wp.float32

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32
    inv_mass_e: wp.float32
    inv_mass_f: wp.float32
    inv_mass_g: wp.float32
    inv_mass_h: wp.float32

    #: Hydrostatic XPBD multiplier accumulator (reset each substep).
    lambda_sum_h: wp.float32
    #: Deviatoric XPBD multiplier accumulator (reset each substep).
    lambda_sum_d: wp.float32

    #: Opt-in per-column wall-clock accumulator (microseconds).
    time_us: wp.float32


assert_constraint_header(SoftHexahedronData)


_OFF_BODY1 = wp.constant(dword_offset_of(SoftHexahedronData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(SoftHexahedronData, "body2"))
_OFF_BODY3 = wp.constant(dword_offset_of(SoftHexahedronData, "body3"))
_OFF_BODY4 = wp.constant(dword_offset_of(SoftHexahedronData, "body4"))
_OFF_BODY5 = wp.constant(dword_offset_of(SoftHexahedronData, "body5"))
_OFF_BODY6 = wp.constant(dword_offset_of(SoftHexahedronData, "body6"))
_OFF_BODY7 = wp.constant(dword_offset_of(SoftHexahedronData, "body7"))
_OFF_BODY8 = wp.constant(dword_offset_of(SoftHexahedronData, "body8"))
_OFF_STRAIN_MODEL = wp.constant(dword_offset_of(SoftHexahedronData, "strain_model"))
_OFF_INV_REST = wp.constant(dword_offset_of(SoftHexahedronData, "inv_rest"))
_OFF_REST_VOLUME = wp.constant(dword_offset_of(SoftHexahedronData, "rest_volume"))
_OFF_GAMMA = wp.constant(dword_offset_of(SoftHexahedronData, "gamma"))
_OFF_ALPHA_H = wp.constant(dword_offset_of(SoftHexahedronData, "alpha_h"))
_OFF_ALPHA_D = wp.constant(dword_offset_of(SoftHexahedronData, "alpha_d"))
_OFF_BETA_H = wp.constant(dword_offset_of(SoftHexahedronData, "beta_h"))
_OFF_BETA_D = wp.constant(dword_offset_of(SoftHexahedronData, "beta_d"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_d"))
_OFF_INV_MASS_E = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_e"))
_OFF_INV_MASS_F = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_f"))
_OFF_INV_MASS_G = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_g"))
_OFF_INV_MASS_H = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_h"))
_OFF_LAMBDA_SUM_H = wp.constant(dword_offset_of(SoftHexahedronData, "lambda_sum_h"))
_OFF_LAMBDA_SUM_D = wp.constant(dword_offset_of(SoftHexahedronData, "lambda_sum_d"))
SOFT_HEX_TIME_US_OFFSET = wp.constant(dword_offset_of(SoftHexahedronData, "time_us"))

SOFT_HEX_DWORDS: int = num_dwords(SoftHexahedronData)


@wp.func
def soft_hexahedron_set_type(c: ConstraintContainer, cid: wp.int32):
    write_int(c, wp.int32(0), cid, CONSTRAINT_TYPE_SOFT_HEXAHEDRON)


@wp.func
def soft_hexahedron_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def soft_hexahedron_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def soft_hexahedron_set_body3(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY3, cid, v)


@wp.func
def soft_hexahedron_set_body4(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY4, cid, v)


@wp.func
def soft_hexahedron_set_body5(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY5, cid, v)


@wp.func
def soft_hexahedron_set_body6(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY6, cid, v)


@wp.func
def soft_hexahedron_set_body7(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY7, cid, v)


@wp.func
def soft_hexahedron_set_body8(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY8, cid, v)


@wp.func
def soft_hexahedron_set_strain_model(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_STRAIN_MODEL, cid, v)


@wp.func
def soft_hexahedron_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_INV_REST, cid, v)


@wp.func
def soft_hexahedron_set_rest_volume(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_VOLUME, cid, v)


@wp.func
def soft_hexahedron_set_gamma(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_GAMMA, cid, v)


@wp.func
def soft_hexahedron_set_alpha_h(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_H, cid, v)


@wp.func
def soft_hexahedron_set_alpha_d(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_D, cid, v)


@wp.func
def soft_hexahedron_set_beta_h(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_H, cid, v)


@wp.func
def soft_hexahedron_set_beta_d(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_D, cid, v)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
#: Floor for the energy-style XPBD strain row. The trace energy is 3 at
#: rest for a 3-D element, so this only guards fully collapsed elements.
_ENERGY_FLOOR = wp.constant(wp.float32(1.0e-4))
#: Floor for the Schur-complement determinant during the 2x2 inverse.
_DET_FLOOR = wp.constant(wp.float32(1.0e-30))
_ARAP_EPS = wp.constant(wp.float32(1.0e-8))
_MUELLER_ROT_EPS = wp.constant(wp.float32(1.0e-6))
_MUELLER_DENOM_EPS = wp.constant(wp.float32(1.0e-9))
_MUELLER_CENTER_ROT_ITERS = wp.constant(wp.int32(32))
_MUELLER_GAUSS_ROT_ITERS = wp.constant(wp.int32(16))

#: 1/8 factor folded into B-vector reconstruction.
_ONE_EIGHTH = wp.constant(wp.float32(0.125))
_GAUSS_ONE_OVER_SQRT_THREE = wp.constant(wp.float32(0.5773502691896257))


@wp.func
def _corner_sign(i: wp.int32) -> wp.vec3f:
    """Reference-cube corner sign triplet for the standard hex order."""
    if i == wp.int32(0):
        return wp.vec3f(-1.0, -1.0, -1.0)
    if i == wp.int32(1):
        return wp.vec3f(1.0, -1.0, -1.0)
    if i == wp.int32(2):
        return wp.vec3f(1.0, 1.0, -1.0)
    if i == wp.int32(3):
        return wp.vec3f(-1.0, 1.0, -1.0)
    if i == wp.int32(4):
        return wp.vec3f(-1.0, -1.0, 1.0)
    if i == wp.int32(5):
        return wp.vec3f(1.0, -1.0, 1.0)
    if i == wp.int32(6):
        return wp.vec3f(1.0, 1.0, 1.0)
    if i == wp.int32(7):
        return wp.vec3f(-1.0, 1.0, 1.0)
    return wp.vec3f(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


@wp.func
def _shape_gradient_ref(i: wp.int32, xi: wp.float32, eta: wp.float32, zeta: wp.float32) -> wp.vec3f:
    """Reference-space trilinear shape-function gradient at ``(xi, eta, zeta)``."""
    s = _corner_sign(i)
    return _ONE_EIGHTH * wp.vec3f(
        s[0] * (wp.float32(1.0) + s[1] * eta) * (wp.float32(1.0) + s[2] * zeta),
        s[1] * (wp.float32(1.0) + s[0] * xi) * (wp.float32(1.0) + s[2] * zeta),
        s[2] * (wp.float32(1.0) + s[0] * xi) * (wp.float32(1.0) + s[1] * eta),
    )


@wp.func
def _shape_gradient_at(
    inv_rest: wp.mat33f,
    i: wp.int32,
    xi: wp.float32,
    eta: wp.float32,
    zeta: wp.float32,
) -> wp.vec3f:
    """World-space trilinear shape-function gradient at a reference point."""
    return inv_rest * _shape_gradient_ref(i, xi, eta, zeta)


@wp.func
def _shape_gradient(inv_rest: wp.mat33f, i: wp.int32) -> wp.vec3f:
    """Center-point world-space shape-function gradient."""
    return _shape_gradient_at(
        inv_rest,
        i,
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )


@wp.func
def _compute_F_hex_from_gradients(
    x0: wp.vec3f,
    x1: wp.vec3f,
    x2: wp.vec3f,
    x3: wp.vec3f,
    x4: wp.vec3f,
    x5: wp.vec3f,
    x6: wp.vec3f,
    x7: wp.vec3f,
    b0: wp.vec3f,
    b1: wp.vec3f,
    b2: wp.vec3f,
    b3: wp.vec3f,
    b4: wp.vec3f,
    b5: wp.vec3f,
    b6: wp.vec3f,
    b7: wp.vec3f,
) -> wp.mat33f:
    """Deformation gradient from precomputed world-space gradients."""
    F00 = (
        x0[0] * b0[0]
        + x1[0] * b1[0]
        + x2[0] * b2[0]
        + x3[0] * b3[0]
        + x4[0] * b4[0]
        + x5[0] * b5[0]
        + x6[0] * b6[0]
        + x7[0] * b7[0]
    )
    F01 = (
        x0[0] * b0[1]
        + x1[0] * b1[1]
        + x2[0] * b2[1]
        + x3[0] * b3[1]
        + x4[0] * b4[1]
        + x5[0] * b5[1]
        + x6[0] * b6[1]
        + x7[0] * b7[1]
    )
    F02 = (
        x0[0] * b0[2]
        + x1[0] * b1[2]
        + x2[0] * b2[2]
        + x3[0] * b3[2]
        + x4[0] * b4[2]
        + x5[0] * b5[2]
        + x6[0] * b6[2]
        + x7[0] * b7[2]
    )
    F10 = (
        x0[1] * b0[0]
        + x1[1] * b1[0]
        + x2[1] * b2[0]
        + x3[1] * b3[0]
        + x4[1] * b4[0]
        + x5[1] * b5[0]
        + x6[1] * b6[0]
        + x7[1] * b7[0]
    )
    F11 = (
        x0[1] * b0[1]
        + x1[1] * b1[1]
        + x2[1] * b2[1]
        + x3[1] * b3[1]
        + x4[1] * b4[1]
        + x5[1] * b5[1]
        + x6[1] * b6[1]
        + x7[1] * b7[1]
    )
    F12 = (
        x0[1] * b0[2]
        + x1[1] * b1[2]
        + x2[1] * b2[2]
        + x3[1] * b3[2]
        + x4[1] * b4[2]
        + x5[1] * b5[2]
        + x6[1] * b6[2]
        + x7[1] * b7[2]
    )
    F20 = (
        x0[2] * b0[0]
        + x1[2] * b1[0]
        + x2[2] * b2[0]
        + x3[2] * b3[0]
        + x4[2] * b4[0]
        + x5[2] * b5[0]
        + x6[2] * b6[0]
        + x7[2] * b7[0]
    )
    F21 = (
        x0[2] * b0[1]
        + x1[2] * b1[1]
        + x2[2] * b2[1]
        + x3[2] * b3[1]
        + x4[2] * b4[1]
        + x5[2] * b5[1]
        + x6[2] * b6[1]
        + x7[2] * b7[1]
    )
    F22 = (
        x0[2] * b0[2]
        + x1[2] * b1[2]
        + x2[2] * b2[2]
        + x3[2] * b3[2]
        + x4[2] * b4[2]
        + x5[2] * b5[2]
        + x6[2] * b6[2]
        + x7[2] * b7[2]
    )
    return wp.mat33f(F00, F01, F02, F10, F11, F12, F20, F21, F22)


@wp.func
def _compute_F_hex_at(
    x0: wp.vec3f,
    x1: wp.vec3f,
    x2: wp.vec3f,
    x3: wp.vec3f,
    x4: wp.vec3f,
    x5: wp.vec3f,
    x6: wp.vec3f,
    x7: wp.vec3f,
    inv_rest: wp.mat33f,
    xi: wp.float32,
    eta: wp.float32,
    zeta: wp.float32,
) -> wp.mat33f:
    """Deformation gradient at a reference point.

    Spelled out as explicit accumulators so the compiler can hoist the
    ``B_i`` values once per corner.
    """
    b0 = _shape_gradient_at(inv_rest, wp.int32(0), xi, eta, zeta)
    b1 = _shape_gradient_at(inv_rest, wp.int32(1), xi, eta, zeta)
    b2 = _shape_gradient_at(inv_rest, wp.int32(2), xi, eta, zeta)
    b3 = _shape_gradient_at(inv_rest, wp.int32(3), xi, eta, zeta)
    b4 = _shape_gradient_at(inv_rest, wp.int32(4), xi, eta, zeta)
    b5 = _shape_gradient_at(inv_rest, wp.int32(5), xi, eta, zeta)
    b6 = _shape_gradient_at(inv_rest, wp.int32(6), xi, eta, zeta)
    b7 = _shape_gradient_at(inv_rest, wp.int32(7), xi, eta, zeta)
    return _compute_F_hex_from_gradients(x0, x1, x2, x3, x4, x5, x6, x7, b0, b1, b2, b3, b4, b5, b6, b7)


@wp.func
def _compute_F_hex(
    x0: wp.vec3f,
    x1: wp.vec3f,
    x2: wp.vec3f,
    x3: wp.vec3f,
    x4: wp.vec3f,
    x5: wp.vec3f,
    x6: wp.vec3f,
    x7: wp.vec3f,
    inv_rest: wp.mat33f,
) -> wp.mat33f:
    """Center-point deformation gradient."""
    return _compute_F_hex_at(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        inv_rest,
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )


@wp.func
def _trace_FtF(F: wp.mat33f) -> wp.float32:
    """Return ``trace(F^T F)``."""
    return (
        F[0, 0] * F[0, 0]
        + F[0, 1] * F[0, 1]
        + F[0, 2] * F[0, 2]
        + F[1, 0] * F[1, 0]
        + F[1, 1] * F[1, 1]
        + F[1, 2] * F[1, 2]
        + F[2, 0] * F[2, 0]
        + F[2, 1] * F[2, 1]
        + F[2, 2] * F[2, 2]
    )


@wp.func
def _integrated_strain_energy_gradients(
    x0: wp.vec3f,
    x1: wp.vec3f,
    x2: wp.vec3f,
    x3: wp.vec3f,
    x4: wp.vec3f,
    x5: wp.vec3f,
    x6: wp.vec3f,
    x7: wp.vec3f,
    inv_rest: wp.mat33f,
):
    """Eight-point trilinear trace energy and particle gradients.

    This is the runtime equivalent of xpbd-fem's prefactored
    incompressible Neo-Hookean strain term for linear hexes. The center
    ``inv_rest`` is treated as an affine map, which matches the regular
    grids this constraint is currently built for.
    """
    u = wp.float32(0.0)
    g0 = wp.vec3f(0.0, 0.0, 0.0)
    g1 = wp.vec3f(0.0, 0.0, 0.0)
    g2 = wp.vec3f(0.0, 0.0, 0.0)
    g3 = wp.vec3f(0.0, 0.0, 0.0)
    g4 = wp.vec3f(0.0, 0.0, 0.0)
    g5 = wp.vec3f(0.0, 0.0, 0.0)
    g6 = wp.vec3f(0.0, 0.0, 0.0)
    g7 = wp.vec3f(0.0, 0.0, 0.0)

    for p in range(8):
        q = _GAUSS_ONE_OVER_SQRT_THREE * _corner_sign(wp.int32(p))
        xi = q[0]
        eta = q[1]
        zeta = q[2]

        b0 = _shape_gradient_at(inv_rest, wp.int32(0), xi, eta, zeta)
        b1 = _shape_gradient_at(inv_rest, wp.int32(1), xi, eta, zeta)
        b2 = _shape_gradient_at(inv_rest, wp.int32(2), xi, eta, zeta)
        b3 = _shape_gradient_at(inv_rest, wp.int32(3), xi, eta, zeta)
        b4 = _shape_gradient_at(inv_rest, wp.int32(4), xi, eta, zeta)
        b5 = _shape_gradient_at(inv_rest, wp.int32(5), xi, eta, zeta)
        b6 = _shape_gradient_at(inv_rest, wp.int32(6), xi, eta, zeta)
        b7 = _shape_gradient_at(inv_rest, wp.int32(7), xi, eta, zeta)
        F = _compute_F_hex_from_gradients(x0, x1, x2, x3, x4, x5, x6, x7, b0, b1, b2, b3, b4, b5, b6, b7)
        dU_dF = F * (wp.float32(2.0) * _ONE_EIGHTH)
        u = u + _ONE_EIGHTH * _trace_FtF(F)
        g0 = g0 + dU_dF * b0
        g1 = g1 + dU_dF * b1
        g2 = g2 + dU_dF * b2
        g3 = g3 + dU_dF * b3
        g4 = g4 + dU_dF * b4
        g5 = g5 + dU_dF * b5
        g6 = g6 + dU_dF * b6
        g7 = g7 + dU_dF * b7

    if u < _ENERGY_FLOOR:
        u = _ENERGY_FLOOR
    return u, g0, g1, g2, g3, g4, g5, g6, g7


@wp.func
def _extract_rotation_mueller(F: wp.mat33f, q_init: wp.quatf, max_iters: wp.int32) -> wp.quatf:
    """Closest-rotation quaternion via Mueller's iterative extraction."""
    q = q_init
    for _ in range(max_iters):
        R = wp.quat_to_matrix(q)
        r0 = wp.vec3f(R[0, 0], R[1, 0], R[2, 0])
        r1 = wp.vec3f(R[0, 1], R[1, 1], R[2, 1])
        r2 = wp.vec3f(R[0, 2], R[1, 2], R[2, 2])
        f0 = wp.vec3f(F[0, 0], F[1, 0], F[2, 0])
        f1 = wp.vec3f(F[0, 1], F[1, 1], F[2, 1])
        f2 = wp.vec3f(F[0, 2], F[1, 2], F[2, 2])

        omega = wp.cross(r0, f0) + wp.cross(r1, f1) + wp.cross(r2, f2)
        denom = wp.abs(wp.dot(r0, f0) + wp.dot(r1, f1) + wp.dot(r2, f2)) + _MUELLER_DENOM_EPS
        omega = omega * (wp.float32(1.0) / denom)
        w = wp.length(omega)
        if w < _MUELLER_ROT_EPS:
            break
        dq = wp.quat_from_axis_angle(omega * (wp.float32(1.0) / w), w)
        q = wp.normalize(dq * q)
    return q


@wp.func
def _integrated_arap_constraint_gradients(
    x0: wp.vec3f,
    x1: wp.vec3f,
    x2: wp.vec3f,
    x3: wp.vec3f,
    x4: wp.vec3f,
    x5: wp.vec3f,
    x6: wp.vec3f,
    x7: wp.vec3f,
    inv_rest: wp.mat33f,
    q_init: wp.quatf,
):
    """Eight-point ARAP residual and particle gradients.

    Each Gauss point extracts its own closest rotation, so the ARAP
    strain row remains hourglass-aware instead of collapsing to a
    center-only deformation gradient.
    """
    c = wp.float32(0.0)
    g0 = wp.vec3f(0.0, 0.0, 0.0)
    g1 = wp.vec3f(0.0, 0.0, 0.0)
    g2 = wp.vec3f(0.0, 0.0, 0.0)
    g3 = wp.vec3f(0.0, 0.0, 0.0)
    g4 = wp.vec3f(0.0, 0.0, 0.0)
    g5 = wp.vec3f(0.0, 0.0, 0.0)
    g6 = wp.vec3f(0.0, 0.0, 0.0)
    g7 = wp.vec3f(0.0, 0.0, 0.0)

    for p in range(8):
        q = _GAUSS_ONE_OVER_SQRT_THREE * _corner_sign(wp.int32(p))
        xi = q[0]
        eta = q[1]
        zeta = q[2]

        b0 = _shape_gradient_at(inv_rest, wp.int32(0), xi, eta, zeta)
        b1 = _shape_gradient_at(inv_rest, wp.int32(1), xi, eta, zeta)
        b2 = _shape_gradient_at(inv_rest, wp.int32(2), xi, eta, zeta)
        b3 = _shape_gradient_at(inv_rest, wp.int32(3), xi, eta, zeta)
        b4 = _shape_gradient_at(inv_rest, wp.int32(4), xi, eta, zeta)
        b5 = _shape_gradient_at(inv_rest, wp.int32(5), xi, eta, zeta)
        b6 = _shape_gradient_at(inv_rest, wp.int32(6), xi, eta, zeta)
        b7 = _shape_gradient_at(inv_rest, wp.int32(7), xi, eta, zeta)
        F = _compute_F_hex_from_gradients(x0, x1, x2, x3, x4, x5, x6, x7, b0, b1, b2, b3, b4, b5, b6, b7)
        rotation = _extract_rotation_mueller(F, q_init, _MUELLER_GAUSS_ROT_ITERS)
        R = wp.quat_to_matrix(rotation)
        S = F - R

        s_norm_sq = (
            S[0, 0] * S[0, 0]
            + S[0, 1] * S[0, 1]
            + S[0, 2] * S[0, 2]
            + S[1, 0] * S[1, 0]
            + S[1, 1] * S[1, 1]
            + S[1, 2] * S[1, 2]
            + S[2, 0] * S[2, 0]
            + S[2, 1] * S[2, 1]
            + S[2, 2] * S[2, 2]
        )
        c_gp = wp.sqrt(s_norm_sq + _ARAP_EPS)
        dC_dF = (_ONE_EIGHTH / c_gp) * S
        c = c + _ONE_EIGHTH * c_gp

        g0 = g0 + dC_dF * b0
        g1 = g1 + dC_dF * b1
        g2 = g2 + dC_dF * b2
        g3 = g3 + dC_dF * b3
        g4 = g4 + dC_dF * b4
        g5 = g5 + dC_dF * b5
        g6 = g6 + dC_dF * b6
        g7 = g7 + dC_dF * b7

    return c, g0, g1, g2, g3, g4, g5, g6, g7


# ---------------------------------------------------------------------------
# Prepare + iterate
# ---------------------------------------------------------------------------


@wp.func
def soft_hexahedron_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Substep-entry prepare: flip access mode to POSITION_LEVEL on all
    8 corners, cache inverse masses, reset both XPBD multipliers to
    zero. No polar-decomposition warm start is needed: both rows are
    expressed directly in terms of ``F^T F`` / ``det(F)``.
    """
    body0 = read_int(constraints, _OFF_BODY1, cid)
    body1 = read_int(constraints, _OFF_BODY2, cid)
    body2 = read_int(constraints, _OFF_BODY3, cid)
    body3 = read_int(constraints, _OFF_BODY4, cid)
    body4 = read_int(constraints, _OFF_BODY5, cid)
    body5 = read_int(constraints, _OFF_BODY6, cid)
    body6 = read_int(constraints, _OFF_BODY7, cid)
    body7 = read_int(constraints, _OFF_BODY8, cid)
    p0 = body0 - num_bodies
    p1 = body1 - num_bodies
    p2 = body2 - num_bodies
    p3 = body3 - num_bodies
    p4 = body4 - num_bodies
    p5 = body5 - num_bodies
    p6 = body6 - num_bodies
    p7 = body7 - num_bodies

    # Per-cid slot / count cache stamped by
    # :func:`build_constraint_slot_cache`; soft-hex uses all 8 cache
    # columns (the schema's ``MAX_BODIES`` upper bound was chosen for
    # exactly this case).
    slot0 = constraints.slot_cache[cid, 0]
    slot1 = constraints.slot_cache[cid, 1]
    slot2 = constraints.slot_cache[cid, 2]
    slot3 = constraints.slot_cache[cid, 3]
    slot4 = constraints.slot_cache[cid, 4]
    slot5 = constraints.slot_cache[cid, 5]
    slot6 = constraints.slot_cache[cid, 6]
    slot7 = constraints.slot_cache[cid, 7]
    inv_factor0 = constraints.count_cache[cid, 0]
    inv_factor1 = constraints.count_cache[cid, 1]
    inv_factor2 = constraints.count_cache[cid, 2]
    inv_factor3 = constraints.count_cache[cid, 3]
    inv_factor4 = constraints.count_cache[cid, 4]
    inv_factor5 = constraints.count_cache[cid, 5]
    inv_factor6 = constraints.count_cache[cid, 6]
    inv_factor7 = constraints.count_cache[cid, 7]

    set_access_mode_with_slot(bodies, particles, copy_state, body0, slot0, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body1, slot1, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body2, slot2, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body3, slot3, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body4, slot4, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body5, slot5, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body6, slot6, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body7, slot7, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)

    write_float(constraints, _OFF_INV_MASS_A, cid, particles.inverse_mass[p0] * wp.float32(inv_factor0))
    write_float(constraints, _OFF_INV_MASS_B, cid, particles.inverse_mass[p1] * wp.float32(inv_factor1))
    write_float(constraints, _OFF_INV_MASS_C, cid, particles.inverse_mass[p2] * wp.float32(inv_factor2))
    write_float(constraints, _OFF_INV_MASS_D, cid, particles.inverse_mass[p3] * wp.float32(inv_factor3))
    write_float(constraints, _OFF_INV_MASS_E, cid, particles.inverse_mass[p4] * wp.float32(inv_factor4))
    write_float(constraints, _OFF_INV_MASS_F, cid, particles.inverse_mass[p5] * wp.float32(inv_factor5))
    write_float(constraints, _OFF_INV_MASS_G, cid, particles.inverse_mass[p6] * wp.float32(inv_factor6))
    write_float(constraints, _OFF_INV_MASS_H, cid, particles.inverse_mass[p7] * wp.float32(inv_factor7))

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, wp.float32(0.0))


@wp.func
def soft_hexahedron_iterate_at(
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
    """One mixed strain/volume XPBD sweep on a soft-body hexahedron.

    The default strain row integrates ``trace(F^T F)`` at the eight
    Gauss points. The optional ARAP row integrates ``||F - R||`` at the
    same points. The volume row uses the reduced center determinant.
    Both modes project strain and volume jointly with a 2x2 Schur solve.
    """
    body0 = read_int(constraints, _OFF_BODY1, cid)
    body1 = read_int(constraints, _OFF_BODY2, cid)
    body2 = read_int(constraints, _OFF_BODY3, cid)
    body3 = read_int(constraints, _OFF_BODY4, cid)
    body4 = read_int(constraints, _OFF_BODY5, cid)
    body5 = read_int(constraints, _OFF_BODY6, cid)
    body6 = read_int(constraints, _OFF_BODY7, cid)
    body7 = read_int(constraints, _OFF_BODY8, cid)
    p0 = body0 - num_bodies
    p1 = body1 - num_bodies
    p2 = body2 - num_bodies
    p3 = body3 - num_bodies
    p4 = body4 - num_bodies
    p5 = body5 - num_bodies
    p6 = body6 - num_bodies
    p7 = body7 - num_bodies

    slot0 = constraints.slot_cache[cid, 0]
    slot1 = constraints.slot_cache[cid, 1]
    slot2 = constraints.slot_cache[cid, 2]
    slot3 = constraints.slot_cache[cid, 3]
    slot4 = constraints.slot_cache[cid, 4]
    slot5 = constraints.slot_cache[cid, 5]
    slot6 = constraints.slot_cache[cid, 6]
    slot7 = constraints.slot_cache[cid, 7]

    set_access_mode_with_slot(bodies, particles, copy_state, body0, slot0, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body1, slot1, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body2, slot2, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body3, slot3, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body4, slot4, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body5, slot5, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body6, slot6, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)
    set_access_mode_with_slot(bodies, particles, copy_state, body7, slot7, num_bodies, _ACCESS_MODE_POSITION_LEVEL, idt)

    inv_mass0 = read_float(constraints, _OFF_INV_MASS_A, cid)
    inv_mass1 = read_float(constraints, _OFF_INV_MASS_B, cid)
    inv_mass2 = read_float(constraints, _OFF_INV_MASS_C, cid)
    inv_mass3 = read_float(constraints, _OFF_INV_MASS_D, cid)
    inv_mass4 = read_float(constraints, _OFF_INV_MASS_E, cid)
    inv_mass5 = read_float(constraints, _OFF_INV_MASS_F, cid)
    inv_mass6 = read_float(constraints, _OFF_INV_MASS_G, cid)
    inv_mass7 = read_float(constraints, _OFF_INV_MASS_H, cid)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    strain_model = read_int(constraints, _OFF_STRAIN_MODEL, cid)
    gamma_offset = read_float(constraints, _OFF_GAMMA, cid)
    alpha_h = read_float(constraints, _OFF_ALPHA_H, cid)
    alpha_d = read_float(constraints, _OFF_ALPHA_D, cid)
    beta_h = read_float(constraints, _OFF_BETA_H, cid)
    beta_d = read_float(constraints, _OFF_BETA_D, cid)

    x0 = read_position_with_slot(bodies, particles, copy_state, body0, slot0, num_bodies)
    x1 = read_position_with_slot(bodies, particles, copy_state, body1, slot1, num_bodies)
    x2 = read_position_with_slot(bodies, particles, copy_state, body2, slot2, num_bodies)
    x3 = read_position_with_slot(bodies, particles, copy_state, body3, slot3, num_bodies)
    x4 = read_position_with_slot(bodies, particles, copy_state, body4, slot4, num_bodies)
    x5 = read_position_with_slot(bodies, particles, copy_state, body5, slot5, num_bodies)
    x6 = read_position_with_slot(bodies, particles, copy_state, body6, slot6, num_bodies)
    x7 = read_position_with_slot(bodies, particles, copy_state, body7, slot7, num_bodies)

    dx0 = x0 - particles.position_prev_substep[p0]
    dx1 = x1 - particles.position_prev_substep[p1]
    dx2 = x2 - particles.position_prev_substep[p2]
    dx3 = x3 - particles.position_prev_substep[p3]
    dx4 = x4 - particles.position_prev_substep[p4]
    dx5 = x5 - particles.position_prev_substep[p5]
    dx6 = x6 - particles.position_prev_substep[p6]
    dx7 = x7 - particles.position_prev_substep[p7]

    # Reduced volume row: center-point determinant.
    F = _compute_F_hex(x0, x1, x2, x3, x4, x5, x6, x7, inv_rest)
    det_f, dJ_dF = deformation_gradient_determinant_cofactor(F)

    b0 = _shape_gradient(inv_rest, wp.int32(0))
    b1 = _shape_gradient(inv_rest, wp.int32(1))
    b2 = _shape_gradient(inv_rest, wp.int32(2))
    b3 = _shape_gradient(inv_rest, wp.int32(3))
    b4 = _shape_gradient(inv_rest, wp.int32(4))
    b5 = _shape_gradient(inv_rest, wp.int32(5))
    b6 = _shape_gradient(inv_rest, wp.int32(6))
    b7 = _shape_gradient(inv_rest, wp.int32(7))

    if strain_model == _SOFT_HEX_STRAIN_MODEL_ARAP:
        # ARAP uses a true volume constraint with rest target det(F)=1.
        # The ARAP strain row is already rotation-free, so it does not
        # need the stable Neo-Hookean gamma offset that balances the
        # trace-energy row at rest.
        c_h = det_f - wp.float32(1.0)
        g_h0 = dJ_dF * b0
        g_h1 = dJ_dF * b1
        g_h2 = dJ_dF * b2
        g_h3 = dJ_dF * b3
        g_h4 = dJ_dF * b4
        g_h5 = dJ_dF * b5
        g_h6 = dJ_dF * b6
        g_h7 = dJ_dF * b7
        center_rotation = _extract_rotation_mueller(
            F,
            wp.quatf(0.0, 0.0, 0.0, 1.0),
            _MUELLER_CENTER_ROT_ITERS,
        )
        c_d, g_d0, g_d1, g_d2, g_d3, g_d4, g_d5, g_d6, g_d7 = _integrated_arap_constraint_gradients(
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            inv_rest,
            center_rotation,
        )
    else:
        c_h = det_f - gamma_offset
        u_h = c_h * c_h
        dUH_dF = dJ_dF * (wp.float32(2.0) * c_h)
        g_h0 = dUH_dF * b0
        g_h1 = dUH_dF * b1
        g_h2 = dUH_dF * b2
        g_h3 = dUH_dF * b3
        g_h4 = dUH_dF * b4
        g_h5 = dUH_dF * b5
        g_h6 = dUH_dF * b6
        g_h7 = dUH_dF * b7

        # Full/integrated strain row: this is the xpbd-fem ingredient
        # that makes the 8-node hex respond to hourglass modes.
        u_d, g_d0, g_d1, g_d2, g_d3, g_d4, g_d5, g_d6, g_d7 = _integrated_strain_energy_gradients(
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            inv_rest,
        )

    # Schur-complement matrix entries.
    a_hh = (
        inv_mass0 * wp.dot(g_h0, g_h0)
        + inv_mass1 * wp.dot(g_h1, g_h1)
        + inv_mass2 * wp.dot(g_h2, g_h2)
        + inv_mass3 * wp.dot(g_h3, g_h3)
        + inv_mass4 * wp.dot(g_h4, g_h4)
        + inv_mass5 * wp.dot(g_h5, g_h5)
        + inv_mass6 * wp.dot(g_h6, g_h6)
        + inv_mass7 * wp.dot(g_h7, g_h7)
    )
    a_dd = (
        inv_mass0 * wp.dot(g_d0, g_d0)
        + inv_mass1 * wp.dot(g_d1, g_d1)
        + inv_mass2 * wp.dot(g_d2, g_d2)
        + inv_mass3 * wp.dot(g_d3, g_d3)
        + inv_mass4 * wp.dot(g_d4, g_d4)
        + inv_mass5 * wp.dot(g_d5, g_d5)
        + inv_mass6 * wp.dot(g_d6, g_d6)
        + inv_mass7 * wp.dot(g_d7, g_d7)
    )
    a_hd = (
        inv_mass0 * wp.dot(g_h0, g_d0)
        + inv_mass1 * wp.dot(g_h1, g_d1)
        + inv_mass2 * wp.dot(g_h2, g_d2)
        + inv_mass3 * wp.dot(g_h3, g_d3)
        + inv_mass4 * wp.dot(g_h4, g_d4)
        + inv_mass5 * wp.dot(g_h5, g_d5)
        + inv_mass6 * wp.dot(g_h6, g_d6)
        + inv_mass7 * wp.dot(g_h7, g_d7)
    )

    idt_sq = idt * idt
    dt = wp.float32(1.0) / idt
    bias_h = idt_sq * alpha_h
    bias_d = idt_sq * alpha_d
    gamma_h = beta_h * dt
    gamma_d = beta_d * dt

    grad_h_dot_dx = (
        wp.dot(g_h0, dx0)
        + wp.dot(g_h1, dx1)
        + wp.dot(g_h2, dx2)
        + wp.dot(g_h3, dx3)
        + wp.dot(g_h4, dx4)
        + wp.dot(g_h5, dx5)
        + wp.dot(g_h6, dx6)
        + wp.dot(g_h7, dx7)
    )
    grad_d_dot_dx = (
        wp.dot(g_d0, dx0)
        + wp.dot(g_d1, dx1)
        + wp.dot(g_d2, dx2)
        + wp.dot(g_d3, dx3)
        + wp.dot(g_d4, dx4)
        + wp.dot(g_d5, dx5)
        + wp.dot(g_d6, dx6)
        + wp.dot(g_d7, dx7)
    )

    A11 = (wp.float32(1.0) + gamma_h) * a_hh
    A22 = (wp.float32(1.0) + gamma_d) * a_dd
    A12 = a_hd

    dlam_h = wp.float32(0.0)
    dlam_d = wp.float32(0.0)
    lambda_h = read_float(constraints, _OFF_LAMBDA_SUM_H, cid)
    lambda_d = read_float(constraints, _OFF_LAMBDA_SUM_D, cid)

    if strain_model == _SOFT_HEX_STRAIN_MODEL_ARAP:
        A11 = A11 + bias_h
        A22 = A22 + bias_d
        rhs_h = c_h + bias_h * lambda_h + gamma_h * grad_h_dot_dx
        rhs_d = c_d + bias_d * lambda_d + gamma_d * grad_d_dot_dx

        update = block_solve_projected_xpbd_2(A11, A12, A22, rhs_h, rhs_d, lambda_h, lambda_d, sor_boost, _DET_FLOOR)
        dlam_h = update.delta[0]
        dlam_d = update.delta[1]
        lambda_h = update.lambda_new[0]
        lambda_d = update.lambda_new[1]
    else:
        # Energy XPBD uses ``-2 U`` as the right-hand side and scales
        # the compliance regulariser by the current energy. This
        # mirrors xpbd-fem's ``EnergyXpbdConstrainSimultaneous`` for the
        # two-row strain/volume block.
        A11 = A11 + wp.float32(2.0) * u_h * bias_h
        A22 = A22 + wp.float32(2.0) * u_d * bias_d
        rhs_h = -wp.float32(2.0) * u_h - gamma_h * grad_h_dot_dx
        rhs_d = -wp.float32(2.0) * u_d - gamma_d * grad_d_dot_dx

        det_a = A11 * A22 - A12 * A12
        if det_a < _DET_FLOOR:
            return
        inv_det = wp.float32(1.0) / det_a
        dlam_h = (A22 * rhs_h - A12 * rhs_d) * inv_det
        dlam_d = (-A12 * rhs_h + A11 * rhs_d) * inv_det

    if strain_model != _SOFT_HEX_STRAIN_MODEL_ARAP:
        dlam_h = dlam_h * sor_boost
        dlam_d = dlam_d * sor_boost

    dlam = wp.vec2f(dlam_h, dlam_d)
    x0 = x0 + block_position_delta_2(inv_mass0, dlam, g_h0, g_d0)
    x1 = x1 + block_position_delta_2(inv_mass1, dlam, g_h1, g_d1)
    x2 = x2 + block_position_delta_2(inv_mass2, dlam, g_h2, g_d2)
    x3 = x3 + block_position_delta_2(inv_mass3, dlam, g_h3, g_d3)
    x4 = x4 + block_position_delta_2(inv_mass4, dlam, g_h4, g_d4)
    x5 = x5 + block_position_delta_2(inv_mass5, dlam, g_h5, g_d5)
    x6 = x6 + block_position_delta_2(inv_mass6, dlam, g_h6, g_d6)
    x7 = x7 + block_position_delta_2(inv_mass7, dlam, g_h7, g_d7)

    write_position_unified(bodies, particles, copy_state, body0, slot0, num_bodies, x0)
    write_position_unified(bodies, particles, copy_state, body1, slot1, num_bodies, x1)
    write_position_unified(bodies, particles, copy_state, body2, slot2, num_bodies, x2)
    write_position_unified(bodies, particles, copy_state, body3, slot3, num_bodies, x3)
    write_position_unified(bodies, particles, copy_state, body4, slot4, num_bodies, x4)
    write_position_unified(bodies, particles, copy_state, body5, slot5, num_bodies, x5)
    write_position_unified(bodies, particles, copy_state, body6, slot6, num_bodies, x6)
    write_position_unified(bodies, particles, copy_state, body7, slot7, num_bodies, x7)

    if strain_model == _SOFT_HEX_STRAIN_MODEL_ARAP:
        write_float(constraints, _OFF_LAMBDA_SUM_H, cid, lambda_h)
        write_float(constraints, _OFF_LAMBDA_SUM_D, cid, lambda_d)


# ---------------------------------------------------------------------------
# Builder-side init: array-based stamping (no Newton ``Model`` dependency).
# ---------------------------------------------------------------------------


@wp.kernel
def soft_hex_init_rows_from_arrays_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    # [num_hexes, 8] particle indices in canonical 8-corner order.
    hex_indices: wp.array2d[wp.int32],
    # [num_particles] rest corner positions.
    particle_q: wp.array[wp.vec3f],
    # [num_hexes, 4] = (k_mu, k_lambda, beta_h, beta_d).
    hex_materials: wp.array2d[wp.float32],
    strain_model: wp.int32,
):
    """Stamp one soft-hexahedron row from caller-supplied arrays.

    Builds ``inv_rest = J^{-T}`` and ``rest_volume = 8 det(J)`` from
    the rest particle positions of the 8 corners. Computes the
    mixed strain/volume compliances::

        gamma   = 1 + mu / lambda
        alpha^H = 1 / (lambda * V_rest)
        alpha^D = 1 / (mu * V_rest)

    Body indices follow the unified convention: rigid bodies occupy
    ``[0, num_bodies)`` and particles occupy ``[num_bodies,
    num_bodies + num_particles)``. This kernel applies the
    ``+num_bodies`` shift. ``strain_model`` selects either integrated
    trace strain or integrated ARAP strain for all stamped hexes.
    """
    h = wp.tid()
    cid = cid_offset + h

    p0 = hex_indices[h, 0]
    p1 = hex_indices[h, 1]
    p2 = hex_indices[h, 2]
    p3 = hex_indices[h, 3]
    p4 = hex_indices[h, 4]
    p5 = hex_indices[h, 5]
    p6 = hex_indices[h, 6]
    p7 = hex_indices[h, 7]

    soft_hexahedron_set_type(constraints, cid)
    soft_hexahedron_set_body1(constraints, cid, num_bodies + p0)
    soft_hexahedron_set_body2(constraints, cid, num_bodies + p1)
    soft_hexahedron_set_body3(constraints, cid, num_bodies + p2)
    soft_hexahedron_set_body4(constraints, cid, num_bodies + p3)
    soft_hexahedron_set_body5(constraints, cid, num_bodies + p4)
    soft_hexahedron_set_body6(constraints, cid, num_bodies + p5)
    soft_hexahedron_set_body7(constraints, cid, num_bodies + p6)
    soft_hexahedron_set_body8(constraints, cid, num_bodies + p7)
    soft_hexahedron_set_strain_model(constraints, cid, strain_model)

    X0 = particle_q[p0]
    X1 = particle_q[p1]
    X2 = particle_q[p2]
    X3 = particle_q[p3]
    X4 = particle_q[p4]
    X5 = particle_q[p5]
    X6 = particle_q[p6]
    X7 = particle_q[p7]

    # Rest Jacobian J = (1/8) sum_i X_i (x) sign_i.
    s0 = _corner_sign(wp.int32(0))
    s1 = _corner_sign(wp.int32(1))
    s2 = _corner_sign(wp.int32(2))
    s3 = _corner_sign(wp.int32(3))
    s4 = _corner_sign(wp.int32(4))
    s5 = _corner_sign(wp.int32(5))
    s6 = _corner_sign(wp.int32(6))
    s7 = _corner_sign(wp.int32(7))

    j00 = _ONE_EIGHTH * (
        X0[0] * s0[0]
        + X1[0] * s1[0]
        + X2[0] * s2[0]
        + X3[0] * s3[0]
        + X4[0] * s4[0]
        + X5[0] * s5[0]
        + X6[0] * s6[0]
        + X7[0] * s7[0]
    )
    j01 = _ONE_EIGHTH * (
        X0[0] * s0[1]
        + X1[0] * s1[1]
        + X2[0] * s2[1]
        + X3[0] * s3[1]
        + X4[0] * s4[1]
        + X5[0] * s5[1]
        + X6[0] * s6[1]
        + X7[0] * s7[1]
    )
    j02 = _ONE_EIGHTH * (
        X0[0] * s0[2]
        + X1[0] * s1[2]
        + X2[0] * s2[2]
        + X3[0] * s3[2]
        + X4[0] * s4[2]
        + X5[0] * s5[2]
        + X6[0] * s6[2]
        + X7[0] * s7[2]
    )
    j10 = _ONE_EIGHTH * (
        X0[1] * s0[0]
        + X1[1] * s1[0]
        + X2[1] * s2[0]
        + X3[1] * s3[0]
        + X4[1] * s4[0]
        + X5[1] * s5[0]
        + X6[1] * s6[0]
        + X7[1] * s7[0]
    )
    j11 = _ONE_EIGHTH * (
        X0[1] * s0[1]
        + X1[1] * s1[1]
        + X2[1] * s2[1]
        + X3[1] * s3[1]
        + X4[1] * s4[1]
        + X5[1] * s5[1]
        + X6[1] * s6[1]
        + X7[1] * s7[1]
    )
    j12 = _ONE_EIGHTH * (
        X0[1] * s0[2]
        + X1[1] * s1[2]
        + X2[1] * s2[2]
        + X3[1] * s3[2]
        + X4[1] * s4[2]
        + X5[1] * s5[2]
        + X6[1] * s6[2]
        + X7[1] * s7[2]
    )
    j20 = _ONE_EIGHTH * (
        X0[2] * s0[0]
        + X1[2] * s1[0]
        + X2[2] * s2[0]
        + X3[2] * s3[0]
        + X4[2] * s4[0]
        + X5[2] * s5[0]
        + X6[2] * s6[0]
        + X7[2] * s7[0]
    )
    j21 = _ONE_EIGHTH * (
        X0[2] * s0[1]
        + X1[2] * s1[1]
        + X2[2] * s2[1]
        + X3[2] * s3[1]
        + X4[2] * s4[1]
        + X5[2] * s5[1]
        + X6[2] * s6[1]
        + X7[2] * s7[1]
    )
    j22 = _ONE_EIGHTH * (
        X0[2] * s0[2]
        + X1[2] * s1[2]
        + X2[2] * s2[2]
        + X3[2] * s3[2]
        + X4[2] * s4[2]
        + X5[2] * s5[2]
        + X6[2] * s6[2]
        + X7[2] * s7[2]
    )
    J = wp.mat33f(j00, j01, j02, j10, j11, j12, j20, j21, j22)

    det_j = wp.determinant(J)
    rest_volume = wp.float32(8.0) * wp.abs(det_j)
    soft_hexahedron_set_rest_volume(constraints, cid, rest_volume)

    inv_j = wp.inverse(J)
    inv_rest = wp.transpose(inv_j)
    soft_hexahedron_set_inv_rest(constraints, cid, inv_rest)

    k_mu = hex_materials[h, 0]
    if k_mu < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        k_mu = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR
    k_lambda = hex_materials[h, 1]
    if k_lambda < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        k_lambda = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR

    v_eff = rest_volume
    if v_eff < _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR:
        v_eff = _PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR

    gamma_offset = wp.float32(1.0) + k_mu / k_lambda
    alpha_h = wp.float32(1.0) / (k_lambda * v_eff)
    alpha_d = wp.float32(1.0) / (k_mu * v_eff)

    soft_hexahedron_set_gamma(constraints, cid, gamma_offset)
    soft_hexahedron_set_alpha_h(constraints, cid, alpha_h)
    soft_hexahedron_set_alpha_d(constraints, cid, alpha_d)
    soft_hexahedron_set_beta_h(constraints, cid, hex_materials[h, 2])
    soft_hexahedron_set_beta_d(constraints, cid, hex_materials[h, 3])

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_A, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_B, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_C, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_D, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_E, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_F, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_G, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_H, cid, wp.float32(0.0))
    write_float(constraints, SOFT_HEX_TIME_US_OFFSET, cid, wp.float32(0.0))
