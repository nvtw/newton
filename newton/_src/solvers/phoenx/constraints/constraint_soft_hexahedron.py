# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-level XPBD soft-body hexahedron (8-node trilinear ARAP).

Single per-hex XPBD row per PGS sweep, mirroring
:mod:`constraint_soft_tetrahedron`'s formulation but evaluated on a
trilinear 8-node hex at 1-point Gauss quadrature (the hex center):

    C        = V_rest * sqrt(||F - R||_F^2 + eps)
    F        = sum_i x_i (x) B_i      (3x3, B_i = world-space dN_i / dX at center)
    R        = polar(F)               (warm-started APD Newton, same as tet)

The B_i vectors are reconstructed inside the kernel from a stored
``inv_rest`` mat33f equal to the inverse-transpose of the rest Jacobian
at the hex center::

    J        = sum_i X_i (x) grad_ref N_i(0)   (3x3, rest Jacobian)
    inv_rest = J^{-T}
    B_i      = (1/8) inv_rest * (xi_i, eta_i, zeta_i)

where ``(xi_i, eta_i, zeta_i)`` in {-1, +1}^3 are the corner-sign
triplets of the standard isoparametric 8-node hex ordering. Storing only
``J^{-T}`` (mat33f) keeps the per-row footprint comparable to the tet
schema; the 8 B-vectors are re-derived from the constant corner-sign
table at iterate time (8 mat33-vec3 products per sweep, identical to
the tet's chain-rule cost).

Rest volume: ``V = 8 * det(J)``. For an axis-aligned cube of side L,
``J = (L/2) I`` and ``V = L^3`` as expected.

Per-vertex gradient via chain rule:

    dC/dF        = (V / c_norm) * (F - R)
    g_i = dC/dx_i = (V / c_norm) * (F - R) * B_i

i.e. one mat33-vec3 per corner. No sum-rule (each corner is symmetric
under the trilinear shape functions, unlike the tet's asymmetric
"edges-from-A" encoding).

The XPBD/PGS solve mirrors :func:`soft_tetrahedron_iterate_at`: same
Macklin damping anchor (``gamma_mu = beta_mu * dt``), same single-row
Schur denominator, same ``alpha_mu = 1 / k_mu`` convention with the
rest volume folded into ``C`` rather than ``alpha``.

Corner ordering (matches standard isoparametric convention used by
e.g. VTK_HEXAHEDRON, Abaqus C3D8, glTF / OBJ box meshes):

    i | (xi, eta, zeta)
    0 | (-1, -1, -1)
    1 | (+1, -1, -1)
    2 | (+1, +1, -1)
    3 | (-1, +1, -1)
    4 | (-1, -1, +1)
    5 | (+1, -1, +1)
    6 | (+1, +1, +1)
    7 | (-1, +1, +1)

Caller is responsible for stamping body1..body8 in this order. Mixed
orderings will not invert (det(J) flips sign and the absolute-value
volume + polar decomposition absorb the chirality), but the per-corner
gradient indexing assumes consecutive i = 0..7 follow this table.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_POSITION_LEVEL
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_SOFT_HEXAHEDRON,
    ConstraintContainer,
    assert_constraint_header,
    read_float,
    read_int,
    read_mat33,
    read_quat,
    write_float,
    write_int,
    write_mat33,
    write_quat,
)
from newton._src.solvers.phoenx.constraints.constraint_soft_tetrahedron import (
    _extract_rotation,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.mass_splitting.access import (
    get_state_index,
    read_position_with_slot,
    set_access_mode_with_slot,
    write_position_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "SOFT_HEX_DWORDS",
    "SOFT_HEX_TIME_US_OFFSET",
    "SoftHexahedronData",
    "soft_hex_init_rows_from_arrays_kernel",
    "soft_hexahedron_iterate_at",
    "soft_hexahedron_prepare_for_iteration_at",
    "soft_hexahedron_set_alpha_mu",
    "soft_hexahedron_set_beta_mu",
    "soft_hexahedron_set_body1",
    "soft_hexahedron_set_body2",
    "soft_hexahedron_set_body3",
    "soft_hexahedron_set_body4",
    "soft_hexahedron_set_body5",
    "soft_hexahedron_set_body6",
    "soft_hexahedron_set_body7",
    "soft_hexahedron_set_body8",
    "soft_hexahedron_set_inv_rest",
    "soft_hexahedron_set_rest_volume",
    "soft_hexahedron_set_type",
]


_PHOENX_SOFT_HEX_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class SoftHexahedronData:
    """Per-constraint dword layout for one 8-node soft hexahedron.

    Mirrors :class:`SoftTetrahedronData` field-for-field with the
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

    # inv_rest = J^{-T} where J is the rest Jacobian at the hex center.
    # B_i = (1/8) inv_rest * (xi_i, eta_i, zeta_i) is the world-space
    # shape-function gradient for corner i; the kernel reconstructs all
    # 8 B_i from this single mat33 + the static corner-sign table.
    inv_rest: wp.mat33f
    rest_volume: wp.float32  # V = 8 * det(J)

    alpha_mu: wp.float32  # 1 / Lame mu (shear compliance)
    beta_mu: wp.float32  # Macklin XPBD damping coefficient

    inv_mass_a: wp.float32
    inv_mass_b: wp.float32
    inv_mass_c: wp.float32
    inv_mass_d: wp.float32
    inv_mass_e: wp.float32
    inv_mass_f: wp.float32
    inv_mass_g: wp.float32
    inv_mass_h: wp.float32

    rotation: wp.quatf  # corotational warm-start (closest-rotation to F)
    lambda_sum_mu: wp.float32  # shear-row XPBD accumulator

    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
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
_OFF_INV_REST = wp.constant(dword_offset_of(SoftHexahedronData, "inv_rest"))
_OFF_REST_VOLUME = wp.constant(dword_offset_of(SoftHexahedronData, "rest_volume"))
_OFF_ALPHA_MU = wp.constant(dword_offset_of(SoftHexahedronData, "alpha_mu"))
_OFF_BETA_MU = wp.constant(dword_offset_of(SoftHexahedronData, "beta_mu"))
_OFF_INV_MASS_A = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_a"))
_OFF_INV_MASS_B = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_b"))
_OFF_INV_MASS_C = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_c"))
_OFF_INV_MASS_D = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_d"))
_OFF_INV_MASS_E = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_e"))
_OFF_INV_MASS_F = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_f"))
_OFF_INV_MASS_G = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_g"))
_OFF_INV_MASS_H = wp.constant(dword_offset_of(SoftHexahedronData, "inv_mass_h"))
_OFF_ROTATION = wp.constant(dword_offset_of(SoftHexahedronData, "rotation"))
_OFF_LAMBDA_SUM_MU = wp.constant(dword_offset_of(SoftHexahedronData, "lambda_sum_mu"))
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
def soft_hexahedron_set_inv_rest(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_INV_REST, cid, v)


@wp.func
def soft_hexahedron_set_rest_volume(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_VOLUME, cid, v)


@wp.func
def soft_hexahedron_set_alpha_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ALPHA_MU, cid, v)


@wp.func
def soft_hexahedron_set_beta_mu(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BETA_MU, cid, v)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_ACCESS_MODE_POSITION_LEVEL = wp.constant(wp.int32(ACCESS_MODE_POSITION_LEVEL))
_DET_F_EPS = wp.constant(wp.float32(1.0e-8))
_ARAP_EPS = wp.constant(wp.float32(1.0e-8))
#: Cold-start polar-decomposition iteration count, matches the tet's
#: prepare-time budget. APD's quadratic convergence covers a cold
#: (identity) start in ~6 iters even on heavily-sheared input.
_PREPARE_ROT_ITERS = wp.constant(wp.int32(6))
#: Refine polar-decomposition iteration count per PGS sweep. Same
#: budget as the tet iterate; the warm-started quaternion lands at
#: machine precision in 2-3 refines.
_ITERATE_ROT_ITERS = wp.constant(wp.int32(3))

#: 1/8 factor folded into B-vector reconstruction. ``B_i = ONE_EIGHTH *
#: inv_rest * (xi_i, eta_i, zeta_i)``.
_ONE_EIGHTH = wp.constant(wp.float32(0.125))


@wp.func
def _corner_sign(i: wp.int32) -> wp.vec3f:
    """Reference-cube corner sign triplet for the standard
    isoparametric 8-node hex ordering. Returns ``(xi_i, eta_i, zeta_i)``
    in {-1, +1}^3 for ``i`` in ``[0, 8)``; ``(0, 0, 0)`` for out-of-range.

    The branchless form would be ``((i&1)*2-1, ((i>>1)&1)*2-1,
    ((i>>2)&1)*2-1)`` for a Gray-code ordering, but the standard
    convention (matching VTK_HEXAHEDRON / Abaqus C3D8) is not pure
    binary -- corners 2 and 3 swap relative to a naive bit-pattern.
    Spell the table out so the corner numbering matches user
    expectations.
    """
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
def _shape_gradient(inv_rest: wp.mat33f, i: wp.int32) -> wp.vec3f:
    """World-space shape-function gradient for corner ``i`` at the hex
    center: ``B_i = (1/8) inv_rest * sign_i``."""
    return _ONE_EIGHTH * (inv_rest * _corner_sign(i))


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
    """Deformation gradient at the hex center: ``F = sum_i x_i (x) B_i``.

    Expanded explicitly rather than via a loop so the per-corner sign
    triplet folds into compile-time constant scalings of inv_rest's
    columns. After common subexpression elimination this evaluates to
    the same number of multiplies as ``sum_i (1/8) x_i * (sign_i^T
    inv_rest^T)``, but it's spelled out structurally for clarity.
    """
    # B_i = (1/8) inv_rest * sign_i.
    b0 = _shape_gradient(inv_rest, wp.int32(0))
    b1 = _shape_gradient(inv_rest, wp.int32(1))
    b2 = _shape_gradient(inv_rest, wp.int32(2))
    b3 = _shape_gradient(inv_rest, wp.int32(3))
    b4 = _shape_gradient(inv_rest, wp.int32(4))
    b5 = _shape_gradient(inv_rest, wp.int32(5))
    b6 = _shape_gradient(inv_rest, wp.int32(6))
    b7 = _shape_gradient(inv_rest, wp.int32(7))
    # F = sum_i x_i b_i^T. Each (x_i, b_i) contributes a rank-1 update.
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
    8 corners, cache inverse masses, reset XPBD warm starts, cold-start
    the polar-decomposition rotation against the substep-entry pose.

    Mirrors :func:`soft_tetrahedron_prepare_for_iteration_at` -- the
    rotation quaternion warm start is preserved across substeps and
    only refreshed via Newton iters here, never reset to identity.
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

    slot0, inv_factor0 = get_state_index(copy_state, body0, parallel_id)
    slot1, inv_factor1 = get_state_index(copy_state, body1, parallel_id)
    slot2, inv_factor2 = get_state_index(copy_state, body2, parallel_id)
    slot3, inv_factor3 = get_state_index(copy_state, body3, parallel_id)
    slot4, inv_factor4 = get_state_index(copy_state, body4, parallel_id)
    slot5, inv_factor5 = get_state_index(copy_state, body5, parallel_id)
    slot6, inv_factor6 = get_state_index(copy_state, body6, parallel_id)
    slot7, inv_factor7 = get_state_index(copy_state, body7, parallel_id)

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

    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, wp.float32(0.0))

    x0 = read_position_with_slot(bodies, particles, copy_state, body0, slot0, num_bodies)
    x1 = read_position_with_slot(bodies, particles, copy_state, body1, slot1, num_bodies)
    x2 = read_position_with_slot(bodies, particles, copy_state, body2, slot2, num_bodies)
    x3 = read_position_with_slot(bodies, particles, copy_state, body3, slot3, num_bodies)
    x4 = read_position_with_slot(bodies, particles, copy_state, body4, slot4, num_bodies)
    x5 = read_position_with_slot(bodies, particles, copy_state, body5, slot5, num_bodies)
    x6 = read_position_with_slot(bodies, particles, copy_state, body6, slot6, num_bodies)
    x7 = read_position_with_slot(bodies, particles, copy_state, body7, slot7, num_bodies)
    inv_rest = read_mat33(constraints, _OFF_INV_REST, cid)
    F = _compute_F_hex(x0, x1, x2, x3, x4, x5, x6, x7, inv_rest)
    rotation = read_quat(constraints, _OFF_ROTATION, cid)
    rotation = _extract_rotation(F, rotation, _PREPARE_ROT_ITERS)
    write_quat(constraints, _OFF_ROTATION, cid, rotation)


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
    """One PGS sweep on a soft-body hexahedron.

    Reads 8 corner positions, evaluates ``F`` and the warm-started
    closest-rotation ``R``, builds the rest-volume-weighted Frobenius
    constraint ``C = V_rest * ||F - R||_F``, applies one Schur-complement
    correction with Macklin damping, and writes back the 8 updated
    corner positions.

    Body fields are unified indices: ``i_p = body - num_bodies`` is the
    particle slot. Reads / writes route through the slot-aware unified
    helpers so mass splitting (when enabled) lands position-level work
    in the slot.
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

    slot0, _if0 = get_state_index(copy_state, body0, parallel_id)
    slot1, _if1 = get_state_index(copy_state, body1, parallel_id)
    slot2, _if2 = get_state_index(copy_state, body2, parallel_id)
    slot3, _if3 = get_state_index(copy_state, body3, parallel_id)
    slot4, _if4 = get_state_index(copy_state, body4, parallel_id)
    slot5, _if5 = get_state_index(copy_state, body5, parallel_id)
    slot6, _if6 = get_state_index(copy_state, body6, parallel_id)
    slot7, _if7 = get_state_index(copy_state, body7, parallel_id)

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
    rest_volume = read_float(constraints, _OFF_REST_VOLUME, cid)
    alpha_mu = read_float(constraints, _OFF_ALPHA_MU, cid)
    beta_mu = read_float(constraints, _OFF_BETA_MU, cid)
    rotation = read_quat(constraints, _OFF_ROTATION, cid)
    lambda_sum_mu = read_float(constraints, _OFF_LAMBDA_SUM_MU, cid)

    x0 = read_position_with_slot(bodies, particles, copy_state, body0, slot0, num_bodies)
    x1 = read_position_with_slot(bodies, particles, copy_state, body1, slot1, num_bodies)
    x2 = read_position_with_slot(bodies, particles, copy_state, body2, slot2, num_bodies)
    x3 = read_position_with_slot(bodies, particles, copy_state, body3, slot3, num_bodies)
    x4 = read_position_with_slot(bodies, particles, copy_state, body4, slot4, num_bodies)
    x5 = read_position_with_slot(bodies, particles, copy_state, body5, slot5, num_bodies)
    x6 = read_position_with_slot(bodies, particles, copy_state, body6, slot6, num_bodies)
    x7 = read_position_with_slot(bodies, particles, copy_state, body7, slot7, num_bodies)

    # XPBD Macklin damping anchor (velocity-projected).
    dx0 = x0 - particles.position_prev_substep[p0]
    dx1 = x1 - particles.position_prev_substep[p1]
    dx2 = x2 - particles.position_prev_substep[p2]
    dx3 = x3 - particles.position_prev_substep[p3]
    dx4 = x4 - particles.position_prev_substep[p4]
    dx5 = x5 - particles.position_prev_substep[p5]
    dx6 = x6 - particles.position_prev_substep[p6]
    dx7 = x7 - particles.position_prev_substep[p7]

    F = _compute_F_hex(x0, x1, x2, x3, x4, x5, x6, x7, inv_rest)
    rotation = _extract_rotation(F, rotation, _ITERATE_ROT_ITERS)
    R = wp.quat_to_matrix(rotation)

    S = F - R
    s00 = S[0, 0]
    s01 = S[0, 1]
    s02 = S[0, 2]
    s10 = S[1, 0]
    s11 = S[1, 1]
    s12 = S[1, 2]
    s20 = S[2, 0]
    s21 = S[2, 1]
    s22 = S[2, 2]
    s_norm_sq = (
        s00 * s00 + s01 * s01 + s02 * s02 + s10 * s10 + s11 * s11 + s12 * s12 + s20 * s20 + s21 * s21 + s22 * s22
    )
    c_norm = wp.sqrt(s_norm_sq + _ARAP_EPS)

    if c_norm < _DET_F_EPS:
        # Pure rotation; persist refined rotation and bail.
        write_quat(constraints, _OFF_ROTATION, cid, rotation)
        return

    # dC/dF = V_rest * S / c_norm.
    inv_c = wp.float32(1.0) / c_norm
    dCdF = (rest_volume * inv_c) * S
    c_arap = rest_volume * c_norm

    # Per-corner gradient: g_i = dC/dF * B_i.
    g0 = dCdF * _shape_gradient(inv_rest, wp.int32(0))
    g1 = dCdF * _shape_gradient(inv_rest, wp.int32(1))
    g2 = dCdF * _shape_gradient(inv_rest, wp.int32(2))
    g3 = dCdF * _shape_gradient(inv_rest, wp.int32(3))
    g4 = dCdF * _shape_gradient(inv_rest, wp.int32(4))
    g5 = dCdF * _shape_gradient(inv_rest, wp.int32(5))
    g6 = dCdF * _shape_gradient(inv_rest, wp.int32(6))
    g7 = dCdF * _shape_gradient(inv_rest, wp.int32(7))

    idt_sq = idt * idt
    dt = wp.float32(1.0) / idt
    bias_mu = idt_sq * alpha_mu
    gamma_mu = beta_mu * dt

    grad2_im = (
        inv_mass0 * wp.dot(g0, g0)
        + inv_mass1 * wp.dot(g1, g1)
        + inv_mass2 * wp.dot(g2, g2)
        + inv_mass3 * wp.dot(g3, g3)
        + inv_mass4 * wp.dot(g4, g4)
        + inv_mass5 * wp.dot(g5, g5)
        + inv_mass6 * wp.dot(g6, g6)
        + inv_mass7 * wp.dot(g7, g7)
    )
    grad_dot_dx = (
        wp.dot(g0, dx0)
        + wp.dot(g1, dx1)
        + wp.dot(g2, dx2)
        + wp.dot(g3, dx3)
        + wp.dot(g4, dx4)
        + wp.dot(g5, dx5)
        + wp.dot(g6, dx6)
        + wp.dot(g7, dx7)
    )
    denom = (wp.float32(1.0) + gamma_mu) * grad2_im + bias_mu

    if denom > wp.float32(0.0):
        d_lam = -(c_arap + bias_mu * lambda_sum_mu + gamma_mu * grad_dot_dx) / denom
        d_lam = d_lam * sor_boost
        x0 = x0 + (d_lam * inv_mass0) * g0
        x1 = x1 + (d_lam * inv_mass1) * g1
        x2 = x2 + (d_lam * inv_mass2) * g2
        x3 = x3 + (d_lam * inv_mass3) * g3
        x4 = x4 + (d_lam * inv_mass4) * g4
        x5 = x5 + (d_lam * inv_mass5) * g5
        x6 = x6 + (d_lam * inv_mass6) * g6
        x7 = x7 + (d_lam * inv_mass7) * g7
        lambda_sum_mu = lambda_sum_mu + d_lam

    write_position_unified(bodies, particles, copy_state, body0, slot0, num_bodies, x0)
    write_position_unified(bodies, particles, copy_state, body1, slot1, num_bodies, x1)
    write_position_unified(bodies, particles, copy_state, body2, slot2, num_bodies, x2)
    write_position_unified(bodies, particles, copy_state, body3, slot3, num_bodies, x3)
    write_position_unified(bodies, particles, copy_state, body4, slot4, num_bodies, x4)
    write_position_unified(bodies, particles, copy_state, body5, slot5, num_bodies, x5)
    write_position_unified(bodies, particles, copy_state, body6, slot6, num_bodies, x6)
    write_position_unified(bodies, particles, copy_state, body7, slot7, num_bodies, x7)

    write_quat(constraints, _OFF_ROTATION, cid, rotation)
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, lambda_sum_mu)


# ---------------------------------------------------------------------------
# Population helper: array-based init (no Newton ``Model`` dependency).
# Lets callers build a hex constraint from raw particle indices + rest
# corners + per-hex (k_mu, beta_mu) -- the soft-body equivalent of the
# minimal "pin a cube corner" scene the test example uses.
# ---------------------------------------------------------------------------


@wp.kernel
def soft_hex_init_rows_from_arrays_kernel(
    constraints: ConstraintContainer,
    cid_offset: wp.int32,
    num_bodies: wp.int32,
    # [num_hexes, 8] particle indices in canonical 8-corner order.
    hex_indices: wp.array2d[wp.int32],
    # [num_particles] rest corner positions (used for both J and V).
    particle_q: wp.array[wp.vec3f],
    # [num_hexes, 2] = (k_mu, beta_mu). Volumetric Lame is handled
    # implicitly by the ARAP residual; the explicit lambda row is the
    # tet's responsibility, not the hex's (matches the tet's
    # ``soft_tet_init_rows_kernel`` schedule).
    hex_materials: wp.array2d[wp.float32],
):
    """Stamp one soft-hexahedron row from caller-supplied arrays.

    Builds ``inv_rest = J^{-T}`` and ``rest_volume = 8 det(J)`` from the
    rest particle positions of the 8 corners. The caller is responsible
    for stamping ``hex_indices[h, 0..7]`` in canonical isoparametric
    order (see this module's docstring); any other ordering will
    produce a non-physical rest pose.

    Body indices follow the unified convention: rigid bodies occupy
    ``[0, num_bodies)`` and particles occupy ``[num_bodies, num_bodies
    + num_particles)``. This kernel applies the ``+num_bodies`` shift.
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

    X0 = particle_q[p0]
    X1 = particle_q[p1]
    X2 = particle_q[p2]
    X3 = particle_q[p3]
    X4 = particle_q[p4]
    X5 = particle_q[p5]
    X6 = particle_q[p6]
    X7 = particle_q[p7]

    # Rest Jacobian J = sum_i X_i (x) grad_ref N_i(0) = (1/8) sum_i X_i (x) sign_i.
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

    # inv_rest = J^{-T} (transpose of the inverse). The chain rule we
    # use is ``g_i = dC/dF * (1/8) inv_rest * sign_i``; storing inv(J)^T
    # collapses the two-step "compute inv(J), then transpose at use" into
    # one stored matrix.
    inv_j = wp.inverse(J)
    inv_rest = wp.transpose(inv_j)
    soft_hexahedron_set_inv_rest(constraints, cid, inv_rest)

    k_mu = hex_materials[h, 0]
    if k_mu < _PHOENX_SOFT_HEX_STIFFNESS_FLOOR:
        k_mu = _PHOENX_SOFT_HEX_STIFFNESS_FLOOR
    soft_hexahedron_set_alpha_mu(constraints, cid, wp.float32(1.0) / k_mu)
    soft_hexahedron_set_beta_mu(constraints, cid, hex_materials[h, 1])

    # XPBD warm starts start at zero / identity-rotation; iterate-time
    # APD Newton refines from there. Note ``wp.quatf()`` defaults to
    # ``(0, 0, 0, 0)`` not identity -- explicitly write identity.
    write_quat(constraints, _OFF_ROTATION, cid, wp.quatf(0.0, 0.0, 0.0, 1.0))
    write_float(constraints, _OFF_LAMBDA_SUM_MU, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_A, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_B, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_C, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_D, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_E, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_F, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_G, cid, wp.float32(0.0))
    write_float(constraints, _OFF_INV_MASS_H, cid, wp.float32(0.0))
    write_float(constraints, SOFT_HEX_TIME_US_OFFSET, cid, wp.float32(0.0))
