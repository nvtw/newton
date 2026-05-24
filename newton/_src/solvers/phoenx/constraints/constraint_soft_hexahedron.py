# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stable block Neo-Hookean XPBD soft-body hexahedron.

8-node trilinear hex sibling of :mod:`constraint_soft_tet_neohookean`.
Same Ton-That, Kry & Andrews 2024 block-coupled formulation, but the
deformation gradient is evaluated at 1-point Gauss quadrature (hex
center) on standard isoparametric trilinear shape functions::

    Psi_neo(F) = (mu / 2) (tr(F^T F) - 3) + (lambda / 2) (det(F) - gamma)^2
    gamma      = 1 + mu / lambda

split into the two scalar XPBD rows::

    C^H(x) = det(F) - gamma                 (hydrostatic / volume)
    C^D(x) = sqrt(tr(F^T F) + eps)          (deviatoric / shear)
    alpha^H = 1 / (lambda * V_rest)
    alpha^D = 1 / (mu * V_rest)

projected jointly via a 2x2 Schur complement -- one constraint with
two coupled multipliers, **not** two independent rows. The combined
solve is what makes the formulation robust on single elements with
under-determined pin patterns: ``C^H`` directly resists the volumetric
modes that pure ARAP can't see, ``C^D`` handles shear, and the coupling
matrix balances them so high stiffness no longer overshoots into
runaway deformation.

ARAP's polar-decomposition warm start is gone -- the Neo-Hookean
energy is rotation-invariant by construction (it's a function of
``F^T F`` and ``det(F)`` only), so there's no rotation to track.

Per-corner gradients use the same shape-gradient chain rule as the
ARAP variant lived under here previously:

    F        = sum_i x_i (x) B_i               (3x3, B_i world shape grad)
    B_i      = (1/8) inv_rest * (xi_i, eta_i, zeta_i)
    inv_rest = J^{-T} at the hex center
    g_i      = dC/dF * B_i                     (one mat33-vec3 per corner)

Storage uses one ``inv_rest`` mat33f equal to J^{-T} at the hex center;
the 8 ``B_i`` are reconstructed in-kernel from the constant corner-sign
table. Rest volume: ``V = 8 * det(J)`` (for an axis-aligned cube of
side ``L``, ``J = (L/2) I`` and ``V = L^3``).

Corner ordering (canonical isoparametric, matches VTK_HEXAHEDRON /
Abaqus C3D8):

    i | (xi, eta, zeta)
    0 | (-1, -1, -1)
    1 | (+1, -1, -1)
    2 | (+1, +1, -1)
    3 | (-1, +1, -1)
    4 | (-1, -1, +1)
    5 | (+1, -1, +1)
    6 | (+1, +1, +1)
    7 | (-1, +1, +1)
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
    write_float,
    write_int,
    write_mat33,
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
    "soft_hexahedron_set_type",
]


_PHOENX_NEOHOOKEAN_STIFFNESS_FLOOR = wp.constant(wp.float32(1.0e-6))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class SoftHexahedronData:
    """Per-constraint dword layout for one block Neo-Hookean hexahedron.

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

    # inv_rest = J^{-T} where J is the rest Jacobian at the hex center.
    # B_i = (1/8) inv_rest * (xi_i, eta_i, zeta_i).
    inv_rest: wp.mat33f
    rest_volume: wp.float32  # V = 8 * det(J)

    #: ``1 + mu / lambda`` -- stable Neo-Hookean offset.
    gamma: wp.float32
    #: ``1 / (lambda * V_rest)`` -- hydrostatic compliance.
    alpha_h: wp.float32
    #: ``1 / (mu * V_rest)`` -- deviatoric compliance.
    alpha_d: wp.float32
    #: Macklin XPBD damping coefficient on the hydrostatic row [1/s].
    beta_h: wp.float32
    #: Macklin XPBD damping coefficient on the deviatoric row [1/s].
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
#: Floor for the deviatoric constraint magnitude (avoids division by zero
#: when the element fully collapses, F -> 0).
_DEV_EPS = wp.constant(wp.float32(1.0e-12))
#: Floor for the Schur-complement determinant during the 2x2 inverse.
_DET_FLOOR = wp.constant(wp.float32(1.0e-30))

#: 1/8 factor folded into B-vector reconstruction.
_ONE_EIGHTH = wp.constant(wp.float32(0.125))


@wp.func
def _corner_sign(i: wp.int32) -> wp.vec3f:
    """Reference-cube corner sign triplet ``(xi_i, eta_i, zeta_i)``
    in ``{-1, +1}^3`` for the standard isoparametric 8-node hex
    ordering. ``(0, 0, 0)`` for out-of-range ``i``."""
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
    """World-space shape-function gradient ``B_i = (1/8) inv_rest * sign_i``."""
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

    Spelled out as 9 explicit accumulators so the compiler can hoist
    ``B_i`` once per corner; each entry is the sum of 8 ``x_i[a]*B_i[b]``
    products.
    """
    b0 = _shape_gradient(inv_rest, wp.int32(0))
    b1 = _shape_gradient(inv_rest, wp.int32(1))
    b2 = _shape_gradient(inv_rest, wp.int32(2))
    b3 = _shape_gradient(inv_rest, wp.int32(3))
    b4 = _shape_gradient(inv_rest, wp.int32(4))
    b5 = _shape_gradient(inv_rest, wp.int32(5))
    b6 = _shape_gradient(inv_rest, wp.int32(6))
    b7 = _shape_gradient(inv_rest, wp.int32(7))
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
    8 corners, cache inverse masses, reset both XPBD multipliers to
    zero. No polar-decomposition warm start -- the Neo-Hookean
    formulation is rotation-invariant (depends only on ``F^T F`` and
    ``det(F)``).
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
    """One block Neo-Hookean PGS sweep on a soft-body hexahedron.

    Evaluates ``(C^H, C^D)`` and the per-corner gradients of both,
    then solves the 2x2 Schur-complement system::

        A . [d lambda^H; d lambda^D] = -([C^H; C^D] + alpha_tilde [...] + damping)

        A_{HH} = (1 + gamma_H) sum_v w_v ||g^H_v||^2 + alpha_tilde^H
        A_{DD} = (1 + gamma_D) sum_v w_v ||g^D_v||^2 + alpha_tilde^D
        A_{HD} = sum_v w_v g^H_v . g^D_v
        gamma_i = beta_i * dt

    Final position update::

        d x_v = w_v (g^H_v d lambda^H + g^D_v d lambda^D)
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
    gamma_offset = read_float(constraints, _OFF_GAMMA, cid)
    alpha_h = read_float(constraints, _OFF_ALPHA_H, cid)
    alpha_d = read_float(constraints, _OFF_ALPHA_D, cid)
    beta_h = read_float(constraints, _OFF_BETA_H, cid)
    beta_d = read_float(constraints, _OFF_BETA_D, cid)
    lambda_h = read_float(constraints, _OFF_LAMBDA_SUM_H, cid)
    lambda_d = read_float(constraints, _OFF_LAMBDA_SUM_D, cid)

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

    F = _compute_F_hex(x0, x1, x2, x3, x4, x5, x6, x7, inv_rest)

    # Columns of F (Warp mat33 stores row-major).
    f0 = wp.vec3f(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3f(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3f(F[0, 2], F[1, 2], F[2, 2])

    # Hydrostatic: C^H = det(F) - gamma. d det(F) / dF = cofactor matrix
    # (columns are pairwise cross products of F's columns).
    cof0 = wp.cross(f1, f2)
    cof1 = wp.cross(f2, f0)
    cof2 = wp.cross(f0, f1)
    det_f = wp.dot(f0, cof0)
    c_h = det_f - gamma_offset
    dCH_dF = wp.mat33f(
        cof0[0],
        cof1[0],
        cof2[0],
        cof0[1],
        cof1[1],
        cof2[1],
        cof0[2],
        cof1[2],
        cof2[2],
    )

    # Deviatoric: C^D = sqrt(tr(F^T F) + eps). dC^D / dF = F / C^D.
    i_c = (
        f0[0] * f0[0]
        + f0[1] * f0[1]
        + f0[2] * f0[2]
        + f1[0] * f1[0]
        + f1[1] * f1[1]
        + f1[2] * f1[2]
        + f2[0] * f2[0]
        + f2[1] * f2[1]
        + f2[2] * f2[2]
    )
    c_d = wp.sqrt(i_c + _DEV_EPS)
    inv_cd = wp.float32(1.0) / c_d
    dCD_dF = F * inv_cd

    # Per-corner gradients: g_i = dC/dF * B_i (8 mat33-vec3 each row).
    b0 = _shape_gradient(inv_rest, wp.int32(0))
    b1 = _shape_gradient(inv_rest, wp.int32(1))
    b2 = _shape_gradient(inv_rest, wp.int32(2))
    b3 = _shape_gradient(inv_rest, wp.int32(3))
    b4 = _shape_gradient(inv_rest, wp.int32(4))
    b5 = _shape_gradient(inv_rest, wp.int32(5))
    b6 = _shape_gradient(inv_rest, wp.int32(6))
    b7 = _shape_gradient(inv_rest, wp.int32(7))

    g_h0 = dCH_dF * b0
    g_h1 = dCH_dF * b1
    g_h2 = dCH_dF * b2
    g_h3 = dCH_dF * b3
    g_h4 = dCH_dF * b4
    g_h5 = dCH_dF * b5
    g_h6 = dCH_dF * b6
    g_h7 = dCH_dF * b7

    g_d0 = dCD_dF * b0
    g_d1 = dCD_dF * b1
    g_d2 = dCD_dF * b2
    g_d3 = dCD_dF * b3
    g_d4 = dCD_dF * b4
    g_d5 = dCD_dF * b5
    g_d6 = dCD_dF * b6
    g_d7 = dCD_dF * b7

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

    A11 = (wp.float32(1.0) + gamma_h) * a_hh + bias_h
    A22 = (wp.float32(1.0) + gamma_d) * a_dd + bias_d
    A12 = a_hd

    b_h = c_h + bias_h * lambda_h + gamma_h * grad_h_dot_dx
    b_d = c_d + bias_d * lambda_d + gamma_d * grad_d_dot_dx

    det_a = A11 * A22 - A12 * A12
    if det_a < _DET_FLOOR:
        return

    inv_det = wp.float32(1.0) / det_a
    dlam_h = -(A22 * b_h - A12 * b_d) * inv_det
    dlam_d = -(-A12 * b_h + A11 * b_d) * inv_det

    dlam_h = dlam_h * sor_boost
    dlam_d = dlam_d * sor_boost

    x0 = x0 + inv_mass0 * (dlam_h * g_h0 + dlam_d * g_d0)
    x1 = x1 + inv_mass1 * (dlam_h * g_h1 + dlam_d * g_d1)
    x2 = x2 + inv_mass2 * (dlam_h * g_h2 + dlam_d * g_d2)
    x3 = x3 + inv_mass3 * (dlam_h * g_h3 + dlam_d * g_d3)
    x4 = x4 + inv_mass4 * (dlam_h * g_h4 + dlam_d * g_d4)
    x5 = x5 + inv_mass5 * (dlam_h * g_h5 + dlam_d * g_d5)
    x6 = x6 + inv_mass6 * (dlam_h * g_h6 + dlam_d * g_d6)
    x7 = x7 + inv_mass7 * (dlam_h * g_h7 + dlam_d * g_d7)

    write_position_unified(bodies, particles, copy_state, body0, slot0, num_bodies, x0)
    write_position_unified(bodies, particles, copy_state, body1, slot1, num_bodies, x1)
    write_position_unified(bodies, particles, copy_state, body2, slot2, num_bodies, x2)
    write_position_unified(bodies, particles, copy_state, body3, slot3, num_bodies, x3)
    write_position_unified(bodies, particles, copy_state, body4, slot4, num_bodies, x4)
    write_position_unified(bodies, particles, copy_state, body5, slot5, num_bodies, x5)
    write_position_unified(bodies, particles, copy_state, body6, slot6, num_bodies, x6)
    write_position_unified(bodies, particles, copy_state, body7, slot7, num_bodies, x7)

    write_float(constraints, _OFF_LAMBDA_SUM_H, cid, lambda_h + dlam_h)
    write_float(constraints, _OFF_LAMBDA_SUM_D, cid, lambda_d + dlam_d)


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
):
    """Stamp one soft-hexahedron row from caller-supplied arrays.

    Builds ``inv_rest = J^{-T}`` and ``rest_volume = 8 det(J)`` from
    the rest particle positions of the 8 corners. Computes the
    stable Neo-Hookean compliances::

        gamma   = 1 + mu / lambda
        alpha^H = 1 / (lambda * V_rest)
        alpha^D = 1 / (mu * V_rest)

    Body indices follow the unified convention: rigid bodies occupy
    ``[0, num_bodies)`` and particles occupy ``[num_bodies,
    num_bodies + num_particles)``. This kernel applies the
    ``+num_bodies`` shift.
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
