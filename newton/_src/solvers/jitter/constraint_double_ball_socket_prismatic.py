# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone prismatic-mode "double ball-socket" constraint.

Translational twin of :mod:`constraint_double_ball_socket`: locks 5 DoF
so the only free relative motion between two bodies is translation along
a shared axis. Rotation is locked entirely; the two translations
perpendicular to the slide axis are locked as well.

Geometric layout
----------------
The joint is defined by two anchors on the slide axis -- ``anchor1`` and
``anchor2`` -- plus a third, implicit anchor off the axis that the
constraint uses to tag rotation about the axis. The on-axis pair behaves
like the two ball-sockets of the plain double ball-socket (``DBS``), but
here we project both anchors onto the tangent plane ``(t1, t2)``
perpendicular to the slide axis, so each contributes 2 rows instead of
3 -- the axial component is deliberately left free. The third anchor is
auto-derived at ``initialize_kernel`` time as a small offset from
anchor 1 in a direction chosen to be safely off-axis; its role is to add
a scalar tangential row along ``t2`` that locks rotation about the slide
axis without any angle math.

Constraint rows (5 total):

=====  =========================  =======================
Idx    Anchor / projection         DoF locked
=====  =========================  =======================
1, 2   anchor 1, on ``t1, t2``    2 perpendicular slides
3, 4   anchor 2, on ``t1, t2``    2 rotations
                                  (pitch / yaw about axis)
5      anchor 3, scalar on ``t2`` rotation about slide axis
=====  =========================  =======================

The axial translation along the slide axis ``n_hat`` is the only DoF
left free.

4+1 Schur solve
---------------
The 5x5 effective mass block decomposes naturally into a 4x4 block
``K4`` for the four tangent rows at anchors 1 and 2 plus a scalar row
``d_scalar`` for anchor 3 with a 4-vector ``c`` coupling them. We
eliminate the scalar row first:

::

    s_scalar = d_scalar - c^T K4^{-1} c         # Schur complement
    lambda_3 = -s_scalar^{-1} (rhs_3 - c^T K4^{-1} rhs_4)
    lambda_4 = -K4^{-1} (rhs_4 + c lambda_3)

``K4`` is symmetric (from the per-anchor cross-coupling matrices
``B_{ij}``) so we cache a single ``K4^{-1}`` (4x4), the coupling vector
``c`` (4-vec), and ``s_scalar^{-1}`` (scalar). ``iterate`` is then three
small products plus the impulse application.

This is the same block structure as the prismatic branch in
:mod:`constraint_actuated_double_ball_socket`, stripped of the separate
axial drive / limit scalar row. Keeping it as a standalone module makes
the unactuated prismatic joint usable by itself, and gives limits /
motors that attach to it a clean, independent target to test against.

Storage / dispatch contract matches the other per-type constraint
modules in this directory.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    read_float,
    read_int,
    read_mat44,
    read_vec3,
    read_vec4,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat44,
    write_vec3,
    write_vec4,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import create_orthonormal

__all__ = [
    "DBS_PRISMATIC_DWORDS",
    "DoubleBallSocketPrismaticData",
    "double_ball_socket_prismatic_get_body1",
    "double_ball_socket_prismatic_get_body2",
    "double_ball_socket_prismatic_initialize_kernel",
    "double_ball_socket_prismatic_iterate",
    "double_ball_socket_prismatic_iterate_at",
    "double_ball_socket_prismatic_prepare_for_iteration",
    "double_ball_socket_prismatic_prepare_for_iteration_at",
    "double_ball_socket_prismatic_set_body1",
    "double_ball_socket_prismatic_set_body2",
    "double_ball_socket_prismatic_set_damping_ratio",
    "double_ball_socket_prismatic_set_hertz",
    "double_ball_socket_prismatic_world_wrench",
    "double_ball_socket_prismatic_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class DoubleBallSocketPrismaticData:
    """Per-constraint dword-layout schema for the standalone prismatic DBS.

    Same naming convention as :class:`DoubleBallSocketData`
    (``*_b1`` / ``*_b2`` = body frame; ``1`` / ``2`` / ``3`` = anchor
    index). The first field is the global ``constraint_type`` tag
    (header contract; see :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Three anchors in each body's local frame. ``anchor3`` is
    # auto-derived at initialize time as a small fixed offset from
    # ``anchor1`` perpendicular to the slide axis, stored once and
    # reused every substep.
    local_anchor1_b1: wp.vec3f
    local_anchor1_b2: wp.vec3f
    local_anchor2_b1: wp.vec3f
    local_anchor2_b2: wp.vec3f
    local_anchor3_b1: wp.vec3f
    local_anchor3_b2: wp.vec3f

    # World-space lever arms (refreshed in prepare).
    r1_b1: wp.vec3f
    r1_b2: wp.vec3f
    r2_b1: wp.vec3f
    r2_b2: wp.vec3f
    r3_b1: wp.vec3f
    r3_b2: wp.vec3f

    # Cached slide axis and tangent basis. Rebuilt every prepare from
    # the current anchor positions; written so iterate can read them
    # without re-normalising.
    axis_world: wp.vec3f
    t1: wp.vec3f
    t2: wp.vec3f

    # Box2D / Bepu soft-constraint knobs (shared across all 5 rows).
    hertz: wp.float32
    damping_ratio: wp.float32

    # Cached per-substep coefficients.
    bias_rate: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    # Positional biases projected onto the tangent basis (anchors 1,2)
    # and the scalar t2 row (anchor 3). Stored as vec3 with third
    # component zero where unused, for layout uniformity.
    bias1: wp.vec3f
    bias2: wp.vec3f
    bias3: wp.float32

    # 4+1 Schur cache. ``k4_inv`` is the 4x4 inverse of the tangent
    # effective-mass block; ``c_pris`` is the 4-vec coupling to the
    # anchor-3 scalar row; ``s_scalar_inv`` is 1 / (d - c^T K4^{-1} c).
    k4_inv: wp.mat44f
    c_pris: wp.vec4f
    s_scalar_inv: wp.float32

    # Accumulated PGS impulses. Anchor 1 / 2 carry the world-frame
    # tangent impulses so warm-starts survive tangent-basis drift
    # between substeps; we reproject them onto the current (t1, t2)
    # basis on entry to prepare. Anchor 3 is stored as a world vector
    # along the cached t2 for the same reason.
    accumulated_impulse1: wp.vec3f
    accumulated_impulse2: wp.vec3f
    accumulated_impulse3: wp.vec3f


assert_constraint_header(DoubleBallSocketPrismaticData)


_OFF_BODY1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "body2"))
_OFF_LA1_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor1_b1"))
_OFF_LA1_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor1_b2"))
_OFF_LA2_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor2_b1"))
_OFF_LA2_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor2_b2"))
_OFF_LA3_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor3_b1"))
_OFF_LA3_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "local_anchor3_b2"))
_OFF_R1_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r1_b1"))
_OFF_R1_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r1_b2"))
_OFF_R2_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r2_b1"))
_OFF_R2_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r2_b2"))
_OFF_R3_B1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r3_b1"))
_OFF_R3_B2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "r3_b2"))
_OFF_AXIS_WORLD = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "axis_world"))
_OFF_T1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "t2"))
_OFF_HERTZ = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "damping_ratio"))
_OFF_BIAS_RATE = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "bias_rate"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "impulse_coeff"))
_OFF_BIAS1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "bias2"))
_OFF_BIAS3 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "bias3"))
_OFF_K4_INV = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "k4_inv"))
_OFF_C_PRIS = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "c_pris"))
_OFF_S_SCALAR_INV = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "s_scalar_inv"))
_OFF_ACC_IMP1 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "accumulated_impulse1"))
_OFF_ACC_IMP2 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "accumulated_impulse2"))
_OFF_ACC_IMP3 = wp.constant(dword_offset_of(DoubleBallSocketPrismaticData, "accumulated_impulse3"))


DBS_PRISMATIC_DWORDS: int = num_dwords(DoubleBallSocketPrismaticData)


# ---------------------------------------------------------------------------
# Typed accessors (header + knobs only; the solver reads internal slots
# via the OFF_ constants directly).
# ---------------------------------------------------------------------------


@wp.func
def double_ball_socket_prismatic_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def double_ball_socket_prismatic_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def double_ball_socket_prismatic_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def double_ball_socket_prismatic_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def double_ball_socket_prismatic_set_hertz(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ, cid, v)


@wp.func
def double_ball_socket_prismatic_set_damping_ratio(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO, cid, v)


# ---------------------------------------------------------------------------
# Initialize kernel
# ---------------------------------------------------------------------------


@wp.kernel
def double_ball_socket_prismatic_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor1: wp.array[wp.vec3f],
    anchor2: wp.array[wp.vec3f],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
):
    """Pack one batch of prismatic-DBS descriptors into ``constraints``.

    ``anchor1`` and ``anchor2`` are world-space points on the slide
    axis at the moment of initialisation; the line between them defines
    the axis direction. A third off-axis anchor is derived automatically
    by offsetting ``anchor1`` a fixed distance (``1 m``) along an
    arbitrary direction perpendicular to the axis; the constraint itself
    is insensitive to the scale because the anchor-3 row is scalar
    (projected onto ``t2``). The anchor 3 offset is snapshot in body 1
    and body 2's local frames so that any initial drift between the
    two bodies gets absorbed into the warm-start, not into the bias.
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a1 = anchor1[tid]
    a2 = anchor2[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]

    la1_b1 = wp.quat_rotate_inv(orient1, a1 - pos1)
    la1_b2 = wp.quat_rotate_inv(orient2, a1 - pos2)
    la2_b1 = wp.quat_rotate_inv(orient1, a2 - pos1)
    la2_b2 = wp.quat_rotate_inv(orient2, a2 - pos2)

    # Slide axis (world frame) at init time.
    axis_vec = a2 - a1
    axis_len2 = wp.dot(axis_vec, axis_vec)
    if axis_len2 > 1.0e-20:
        n_hat = axis_vec / wp.sqrt(axis_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)

    # Third anchor: anchor 1 + 1 m * t, where t is an arbitrary world
    # direction perpendicular to the slide axis. Both bodies see the
    # same world-space point at init, so ``anchor3_b1`` and
    # ``anchor3_b2`` coincide at t = 0 and the Jacobian row is
    # consistent for both bodies.
    t_world = create_orthonormal(n_hat)
    a3 = a1 + t_world

    la3_b1 = wp.quat_rotate_inv(orient1, a3 - pos1)
    la3_b2 = wp.quat_rotate_inv(orient2, a3 - pos2)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET_PRISMATIC)
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)

    write_vec3(constraints, _OFF_LA1_B1, cid, la1_b1)
    write_vec3(constraints, _OFF_LA1_B2, cid, la1_b2)
    write_vec3(constraints, _OFF_LA2_B1, cid, la2_b1)
    write_vec3(constraints, _OFF_LA2_B2, cid, la2_b2)
    write_vec3(constraints, _OFF_LA3_B1, cid, la3_b1)
    write_vec3(constraints, _OFF_LA3_B2, cid, la3_b2)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    write_vec3(constraints, _OFF_R1_B1, cid, zero3)
    write_vec3(constraints, _OFF_R1_B2, cid, zero3)
    write_vec3(constraints, _OFF_R2_B1, cid, zero3)
    write_vec3(constraints, _OFF_R2_B2, cid, zero3)
    write_vec3(constraints, _OFF_R3_B1, cid, zero3)
    write_vec3(constraints, _OFF_R3_B2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, n_hat)
    write_vec3(constraints, _OFF_T1, cid, t_world)
    write_vec3(constraints, _OFF_T2, cid, wp.cross(n_hat, t_world))

    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_BIAS_RATE, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, 0.0)

    write_vec3(constraints, _OFF_BIAS1, cid, zero3)
    write_vec3(constraints, _OFF_BIAS2, cid, zero3)
    write_float(constraints, _OFF_BIAS3, cid, 0.0)

    eye4 = wp.identity(4, dtype=wp.float32)
    write_mat44(constraints, _OFF_K4_INV, cid, eye4)
    write_vec4(constraints, _OFF_C_PRIS, cid, wp.vec4f(0.0, 0.0, 0.0, 0.0))
    write_float(constraints, _OFF_S_SCALAR_INV, cid, 0.0)

    write_vec3(constraints, _OFF_ACC_IMP1, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP2, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)


# ---------------------------------------------------------------------------
# Per-iteration math (4+1 Schur)
# ---------------------------------------------------------------------------


@wp.func
def double_ball_socket_prismatic_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the standalone prismatic DBS.

    Mirrors the ``_prismatic_prepare_at`` path of the actuated DBS, but
    stops after the 4+1 Schur block: no axial drive / limit cache, no
    axial warm-start. The linear axis is left free.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
    la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
    la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
    la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)

    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    r2_b1 = wp.quat_rotate(orientation1, la2_b1)
    r2_b2 = wp.quat_rotate(orientation2, la2_b2)
    r3_b1 = wp.quat_rotate(orientation1, la3_b1)
    r3_b2 = wp.quat_rotate(orientation2, la3_b2)

    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)
    write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
    write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)
    write_vec3(constraints, base_offset + _OFF_R3_B1, cid, r3_b1)
    write_vec3(constraints, base_offset + _OFF_R3_B2, cid, r3_b2)

    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2
    p2_b1 = position1 + r2_b1
    p2_b2 = position2 + r2_b2
    p3_b1 = position1 + r3_b1
    p3_b2 = position2 + r3_b2

    # Slide axis on body 2 (consistent with the actuated DBS prismatic
    # mode -- axis rides body 2's orientation).
    axis_vec = p2_b2 - p1_b2
    axis_len2 = wp.dot(axis_vec, axis_vec)
    if axis_len2 > 1.0e-20:
        n_hat = axis_vec / wp.sqrt(axis_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # Tangent basis: aligned so t1 ~ anchor3 lever direction projected
    # onto the plane. This makes the anchor-3 scalar row the exact
    # tangential velocity gate for rotation about n_hat (same rationale
    # as the actuated DBS prismatic prepare).
    anchor3_offset_b1 = r3_b1 - r1_b1
    t1_raw = anchor3_offset_b1 - wp.dot(anchor3_offset_b1, n_hat) * n_hat
    t1_len2 = wp.dot(t1_raw, t1_raw)
    if t1_len2 > 1.0e-20:
        t1 = t1_raw / wp.sqrt(t1_len2)
    else:
        t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)
    cr3_b1 = wp.skew(r3_b1)
    cr3_b2 = wp.skew(r3_b2)

    eye3 = wp.identity(3, dtype=wp.float32)
    m_diag = (inv_mass1 + inv_mass2) * eye3

    # Per-anchor / cross-anchor 3x3 coupling blocks:
    #   B_{i,j} = (1/m1 + 1/m2) I
    #             + skew(ri_b1) I1^-1 skew(rj_b1)^T
    #             + skew(ri_b2) I2^-1 skew(rj_b2)^T
    b11 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1)) + cr1_b2 @ (
        inv_inertia2 @ wp.transpose(cr1_b2)
    )
    b22 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr2_b2 @ (
        inv_inertia2 @ wp.transpose(cr2_b2)
    )
    b33 = m_diag + cr3_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr3_b2 @ (
        inv_inertia2 @ wp.transpose(cr3_b2)
    )
    b12 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr1_b2 @ (
        inv_inertia2 @ wp.transpose(cr2_b2)
    )
    b13 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr1_b2 @ (
        inv_inertia2 @ wp.transpose(cr3_b2)
    )
    b23 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr2_b2 @ (
        inv_inertia2 @ wp.transpose(cr3_b2)
    )

    # Project to the tangent basis.
    b11_t1 = b11 @ t1
    b11_t2 = b11 @ t2
    b22_t1 = b22 @ t1
    b22_t2 = b22 @ t2
    b12_t1 = b12 @ t1
    b12_t2 = b12 @ t2
    b13_t2 = b13 @ t2
    b23_t2 = b23 @ t2

    # 4x4 K4: rows / cols = (a1 t1, a1 t2, a2 t1, a2 t2).
    k4_00 = wp.dot(t1, b11_t1)
    k4_01 = wp.dot(t1, b11_t2)
    k4_11 = wp.dot(t2, b11_t2)
    k4_02 = wp.dot(t1, b12_t1)
    k4_03 = wp.dot(t1, b12_t2)
    k4_12 = wp.dot(t2, b12_t1)
    k4_13 = wp.dot(t2, b12_t2)
    k4_22 = wp.dot(t1, b22_t1)
    k4_23 = wp.dot(t1, b22_t2)
    k4_33 = wp.dot(t2, b22_t2)

    k4 = wp.mat44f(
        k4_00, k4_01, k4_02, k4_03,
        k4_01, k4_11, k4_12, k4_13,
        k4_02, k4_12, k4_22, k4_23,
        k4_03, k4_13, k4_23, k4_33,
    )

    # Coupling vector c (4x1): anchor i tangent-row coupled with the
    # anchor-3 scalar row (which is along t2).
    c0 = wp.dot(t1, b13_t2)
    c1 = wp.dot(t2, b13_t2)
    c2 = wp.dot(t1, b23_t2)
    c3 = wp.dot(t2, b23_t2)
    c_pris = wp.vec4f(c0, c1, c2, c3)

    # d scalar (1x1): t2 . B33 . t2.
    b33_t2 = b33 @ t2
    d_scalar = wp.dot(t2, b33_t2)

    # Schur.
    k4_inv = wp.inverse(k4)
    k4_inv_c = k4_inv @ c_pris
    s_scalar = d_scalar - wp.dot(c_pris, k4_inv_c)
    if wp.abs(s_scalar) > 1.0e-20:
        s_scalar_inv = 1.0 / s_scalar
    else:
        s_scalar_inv = 0.0

    write_mat44(constraints, base_offset + _OFF_K4_INV, cid, k4_inv)
    write_vec4(constraints, base_offset + _OFF_C_PRIS, cid, c_pris)
    write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_BIAS_RATE, cid, bias_rate)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Positional bias: tangent drift at each anchor, scalar drift at
    # anchor 3 along t2. Translation along n_hat is *not* corrected --
    # that's the one free DoF.
    drift1 = p1_b2 - p1_b1
    drift2 = p2_b2 - p2_b1
    drift3 = p3_b2 - p3_b1
    bias1 = wp.vec3f(wp.dot(t1, drift1) * bias_rate, wp.dot(t2, drift1) * bias_rate, 0.0)
    bias2 = wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0)
    bias3 = wp.dot(t2, drift3) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)
    write_float(constraints, base_offset + _OFF_BIAS3, cid, bias3)

    # Warm start: re-project the cached tangent impulses onto the
    # *current* tangent basis, then apply them to the bodies.
    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)

    acc1_t1 = wp.dot(t1, acc_imp1_world)
    acc1_t2 = wp.dot(t2, acc_imp1_world)
    acc2_t1 = wp.dot(t1, acc_imp2_world)
    acc2_t2 = wp.dot(t2, acc_imp2_world)
    acc3_t2 = wp.dot(t2, acc_imp3_world)
    acc_imp1_world = acc1_t1 * t1 + acc1_t2 * t2
    acc_imp2_world = acc2_t1 * t1 + acc2_t2 * t2
    acc_imp3_world = acc3_t2 * t2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc_imp3_world)

    total_linear = acc_imp1_world + acc_imp2_world + acc_imp3_world
    velocity1 = bodies.velocity[b1] - inv_mass1 * total_linear
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (
        cr1_b1 @ acc_imp1_world + cr2_b1 @ acc_imp2_world + cr3_b1 @ acc_imp3_world
    )
    velocity2 = bodies.velocity[b2] + inv_mass2 * total_linear
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (
        cr1_b2 @ acc_imp1_world + cr2_b2 @ acc_imp2_world + cr3_b2 @ acc_imp3_world
    )

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def double_ball_socket_prismatic_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """PGS iterate for the standalone prismatic DBS.

    4+1 Schur-complement solve: eliminate the scalar anchor-3 row,
    then back-substitute the 4x4 tangent block. No axial row.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    r1_b1 = read_vec3(constraints, base_offset + _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b1 = read_vec3(constraints, base_offset + _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    r3_b1 = read_vec3(constraints, base_offset + _OFF_R3_B1, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)
    cr3_b1 = wp.skew(r3_b1)
    cr3_b2 = wp.skew(r3_b2)

    k4_inv = read_mat44(constraints, base_offset + _OFF_K4_INV, cid)
    c_pris = read_vec4(constraints, base_offset + _OFF_C_PRIS, cid)
    s_scalar_inv = read_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid)
    bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc1_tan = wp.vec2f(wp.dot(t1, acc_imp1_world), wp.dot(t2, acc_imp1_world))
    acc2_tan = wp.vec2f(wp.dot(t1, acc_imp2_world), wp.dot(t2, acc_imp2_world))
    acc4 = wp.vec4f(acc1_tan[0], acc1_tan[1], acc2_tan[0], acc2_tan[1])
    acc3_scalar = wp.dot(t2, acc_imp3_world)

    # Velocity Jacobian rows (jv_i = d . ((v2 + w2 x r_i_b2) - (v1 + w1 x r_i_b1))).
    jv1_world = velocity2 - cr1_b2 @ angular_velocity2 - velocity1 + cr1_b1 @ angular_velocity1
    jv2_world = velocity2 - cr2_b2 @ angular_velocity2 - velocity1 + cr2_b1 @ angular_velocity1
    jv3_world = velocity2 - cr3_b2 @ angular_velocity2 - velocity1 + cr3_b1 @ angular_velocity1

    jv4 = wp.vec4f(
        wp.dot(t1, jv1_world),
        wp.dot(t2, jv1_world),
        wp.dot(t1, jv2_world),
        wp.dot(t2, jv2_world),
    )
    jv3 = wp.dot(t2, jv3_world)

    bias4 = wp.vec4f(bias1[0], bias1[1], bias2[0], bias2[1])

    rhs4 = jv4 + bias4
    rhs3 = jv3 + bias3

    # Schur: scalar row first.
    k4_inv_rhs4 = k4_inv @ rhs4
    lam3_us = -s_scalar_inv * (rhs3 - wp.dot(c_pris, k4_inv_rhs4))
    lam3 = mass_coeff * lam3_us - impulse_coeff * acc3_scalar

    lam4_us = -(k4_inv @ (rhs4 + c_pris * lam3_us))
    lam4 = mass_coeff * lam4_us - impulse_coeff * acc4

    # World-frame tangent impulses per anchor.
    lam1_world = lam4[0] * t1 + lam4[1] * t2
    lam2_world = lam4[2] * t1 + lam4[3] * t2
    lam3_world = lam3 * t2

    total_linear = lam1_world + lam2_world + lam3_world
    velocity1 = velocity1 - inv_mass1 * total_linear
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (
        cr1_b1 @ lam1_world + cr2_b1 @ lam2_world + cr3_b1 @ lam3_world
    )
    velocity2 = velocity2 + inv_mass2 * total_linear
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (
        cr1_b2 @ lam1_world + cr2_b2 @ lam2_world + cr3_b2 @ lam3_world
    )

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world + lam1_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world + lam2_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc_imp3_world + lam3_world)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def double_ball_socket_prismatic_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """World-frame (force, torque) on body 2 from the lock rows.

    Sum of the three anchor impulses over the substep gives the net
    force; each impulse's moment about body 2's COM contributes to
    the torque, computed from the cached lever arms.
    """
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc3 = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    force = (acc1 + acc2 + acc3) * idt
    torque = (
        wp.cross(r1_b2, acc1 * idt)
        + wp.cross(r2_b2, acc2 * idt)
        + wp.cross(r3_b2, acc3 * idt)
    )
    return force, torque


@wp.func
def double_ball_socket_prismatic_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    b1 = double_ball_socket_prismatic_get_body1(constraints, cid)
    b2 = double_ball_socket_prismatic_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    double_ball_socket_prismatic_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def double_ball_socket_prismatic_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    b1 = double_ball_socket_prismatic_get_body1(constraints, cid)
    b2 = double_ball_socket_prismatic_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    double_ball_socket_prismatic_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def double_ball_socket_prismatic_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    return double_ball_socket_prismatic_world_wrench_at(constraints, cid, 0, idt)
