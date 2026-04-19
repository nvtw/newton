# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Hinge-style "double ball-socket" constraint.

Built on the same per-anchor math as :mod:`constraint_ball_socket` but
solved as a *single* coupled constraint with one Jacobian. Two
ball-and-socket anchors at the same body pair lock 5 DoF (3 translational
+ 2 of the 3 rotational); the third rotational DoF -- the rotation about
the line through the two anchors -- is free, which is exactly the hinge
behaviour without ever invoking quaternion math.

Why not just two independent ball-socket constraints?
-----------------------------------------------------
Stacking two 3-row ball-socket Jacobians would give a 6x6 effective
mass with rank 5: the row corresponding to "rotate about the axis
through both anchors" is identically zero. Inverting that 6x6 needs a
pseudo-inverse or a CFM/softness fudge that leaks compliance along the
hinge axis. We instead **drop one row analytically**: keep all 3
ball-socket equations at anchor 1, but only keep the 2 components of
anchor 2's positional error that are perpendicular to the hinge axis
n_hat = (P2_world(anchor=2) - P1_world(anchor=2)) / |...|. That gives a
genuinely rank-5 constraint with no parameter tuning and no compliance
along the hinge axis -- mathematically equivalent to the two-ball-socket
formulation, just non-redundant.

Layout / storage / dispatch contract is identical to the other
per-type constraint modules in this directory -- see the comments
in :mod:`constraint_ball_socket` for the full conventions. Only
deltas are noted below.

Schur-complement solve (avoiding a 5x5 inverse)
-----------------------------------------------
Warp's ``wp.inverse`` overloads cover only 2x2 / 3x3 / 4x4. The 5x5
effective mass

::

    K = [ A1     U  ]   in  R^{5x5},  with  A1 in R^{3x3},
        [ U^T    D  ]                       U  in R^{3x2},
                                            D  in R^{2x2}

(where ``A1`` is the standard ball-socket effective mass at anchor 1,
``D = T^T A2 T`` is the same at anchor 2 projected onto the tangent
basis ``T = [t1 | t2]`` perpendicular to ``n_hat``, and ``U = B T`` is
the cross-anchor coupling; see the derivation block below) is solved
by block-elimination using only one 3x3 and one 2x2 inverse:

::

    S       = D - U^T A1^{-1} U                  # 2x2 Schur complement
    lambda2 = -S^{-1}     ( b2 - U^T A1^{-1} b1 )
    lambda1 = -A1^{-1}    ( b1 + U lambda2 )

We cache ``A1_inv`` (3x3), ``S_inv`` (2x2) and the precomputed product
``UTAi = U^T A1^{-1}`` (2x3) in the constraint column so ``iterate`` is
just three small mat-vecs plus the impulse application -- no per-iter
inverses.

Mapping summary (deltas vs ball-socket only):

* tangent basis ``T``       -> ``wp.mat33f`` (col 0 = t1, col 1 = t2,
                               col 2 = n_hat; we ignore col 2 in the
                               solve but keep it for the world-frame
                               re-projection of the cached anchor-2
                               accumulated impulse).
* ``T^T v``                 -> 2-vec extracted as
                               ``wp.vec2f(wp.dot(t1, v), wp.dot(t2, v))``.
* ``T lambda2``             -> ``lambda2[0] * t1 + lambda2[1] * t2``.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    read_float,
    read_int,
    read_mat33,
    read_vec3,
    write_float,
    write_int,
    write_mat33,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import create_orthonormal

__all__ = [
    "DBS_DWORDS",
    "DEFAULT_LINEAR_BIAS",
    "DEFAULT_LINEAR_SOFTNESS",
    "DoubleBallSocketData",
    "double_ball_socket_get_a1_inv",
    "double_ball_socket_get_accumulated_impulse1",
    "double_ball_socket_get_accumulated_impulse2",
    "double_ball_socket_get_bias1",
    "double_ball_socket_get_bias2",
    "double_ball_socket_get_bias_factor",
    "double_ball_socket_get_body1",
    "double_ball_socket_get_body2",
    "double_ball_socket_get_local_anchor1_b1",
    "double_ball_socket_get_local_anchor1_b2",
    "double_ball_socket_get_local_anchor2_b1",
    "double_ball_socket_get_local_anchor2_b2",
    "double_ball_socket_get_r1_b1",
    "double_ball_socket_get_r1_b2",
    "double_ball_socket_get_r2_b1",
    "double_ball_socket_get_r2_b2",
    "double_ball_socket_get_s_inv",
    "double_ball_socket_get_softness",
    "double_ball_socket_get_t1",
    "double_ball_socket_get_t2",
    "double_ball_socket_get_ut_ai",
    "double_ball_socket_initialize_kernel",
    "double_ball_socket_iterate",
    "double_ball_socket_iterate_at",
    "double_ball_socket_prepare_for_iteration",
    "double_ball_socket_prepare_for_iteration_at",
    "double_ball_socket_set_a1_inv",
    "double_ball_socket_set_accumulated_impulse1",
    "double_ball_socket_set_accumulated_impulse2",
    "double_ball_socket_set_bias1",
    "double_ball_socket_set_bias2",
    "double_ball_socket_set_bias_factor",
    "double_ball_socket_set_body1",
    "double_ball_socket_set_body2",
    "double_ball_socket_set_local_anchor1_b1",
    "double_ball_socket_set_local_anchor1_b2",
    "double_ball_socket_set_local_anchor2_b1",
    "double_ball_socket_set_local_anchor2_b2",
    "double_ball_socket_set_r1_b1",
    "double_ball_socket_set_r1_b2",
    "double_ball_socket_set_r2_b1",
    "double_ball_socket_set_r2_b2",
    "double_ball_socket_set_s_inv",
    "double_ball_socket_set_softness",
    "double_ball_socket_set_t1",
    "double_ball_socket_set_t2",
    "double_ball_socket_set_ut_ai",
    "double_ball_socket_world_wrench",
    "double_ball_socket_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#
# The per-cid type tag (CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET) lives in
# :mod:`constraint_container` so the central allowlist of constraint
# types stays the single source of truth; we just re-import it here.

#: Same defaults as the single ball-socket; positional softness is mostly
#: there to keep the (already non-degenerate) 3x3 + 2x2 inverses well
#: conditioned in the rare case that a user puts the two anchors right
#: on top of each other (zero hinge length -- physically meaningless but
#: shouldn't crash the solver).
DEFAULT_LINEAR_SOFTNESS = wp.constant(wp.float32(0.00001))
DEFAULT_LINEAR_BIAS = wp.constant(wp.float32(0.2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class DoubleBallSocketData:
    """Per-constraint dword-layout schema for a fused two-anchor hinge.

    *Schema only* (same conventions as :class:`BallSocketData`). The
    runtime never instantiates this; we derive dword offsets from the
    field declarations and read/write the shared
    :class:`ConstraintContainer` via the typed accessors below.

    Field naming convention: ``*_b1`` / ``*_b2`` is the *body* the field
    belongs to; ``1`` / ``2`` (without ``_b``) is the *anchor* index.
    So ``local_anchor1_b1`` is "anchor 1, expressed in body 1's local
    frame", and ``r2_b2`` is "world-space lever arm of anchor 2 on body
    2". The hinge axis at runtime is implicit in the column-3 of the
    cached tangent-basis matrix (we store ``t1, t2`` separately so the
    anchor 2 -> tangent projections fold into 2 dot-products).

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Anchor 1 / anchor 2 in each body's local frame. Set once at
    # initialise; world-space lever arms r*_b* are recomputed each
    # substep in prepare_for_iteration.
    local_anchor1_b1: wp.vec3f
    local_anchor1_b2: wp.vec3f
    local_anchor2_b1: wp.vec3f
    local_anchor2_b2: wp.vec3f

    # World-space lever arms (per anchor, per body). Cached in
    # prepare_for_iteration and reused by iterate.
    r1_b1: wp.vec3f
    r1_b2: wp.vec3f
    r2_b1: wp.vec3f
    r2_b2: wp.vec3f

    # Tangent basis perpendicular to the hinge axis at the most recent
    # prepare_for_iteration. ``t1, t2, n_hat`` is right-handed; we don't
    # store n_hat explicitly because the world-frame re-projection of
    # the anchor-2 accumulated impulse only uses t1 and t2.
    t1: wp.vec3f
    t2: wp.vec3f

    bias_factor: wp.float32
    softness: wp.float32

    # Cached velocity-error bias vectors (anchor 1: 3-vector in world
    # frame; anchor 2: 2-vector in tangent basis stored in the first two
    # components of a vec3 with the third left at 0). Vec3 storage on
    # both for layout uniformity.
    bias1: wp.vec3f
    bias2: wp.vec3f

    # Cached effective-mass blocks for the Schur-complement solve.
    # See module docstring for the math; ``ut_ai`` here is the 2x3
    # product ``U^T A1^{-1}`` packed into the first two rows of a mat33
    # (third row = 0). Storing a true 2x3 would need bespoke read/write
    # helpers; reusing mat33 is the cheapest dword layout.
    a1_inv: wp.mat33f
    ut_ai: wp.mat33f
    s_inv: wp.mat33f  # Only top-left 2x2 used; rest stays 0.

    # Accumulated PGS impulses split per anchor.
    # ``accumulated_impulse1`` is the world-frame ball-socket impulse at
    # anchor 1 (3 dof). ``accumulated_impulse2`` is the *world-frame*
    # impulse at anchor 2 -- we keep it in world coordinates rather than
    # tangent-basis coordinates so the warm-start stays valid even when
    # the hinge axis swings between substeps and the tangent basis
    # rotates with it. The solve still works in tangent coordinates;
    # we re-project at the boundaries.
    accumulated_impulse1: wp.vec3f
    accumulated_impulse2: wp.vec3f


# Enforce the global constraint header contract (constraint_type / body1
# / body2 at dwords 0 / 1 / 2) at import time so a future field reorder
# fails loudly here instead of silently mis-tagging columns or
# scrambling body indices at runtime.
assert_constraint_header(DoubleBallSocketData)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(DoubleBallSocketData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(DoubleBallSocketData, "body2"))
_OFF_LA1_B1 = wp.constant(dword_offset_of(DoubleBallSocketData, "local_anchor1_b1"))
_OFF_LA1_B2 = wp.constant(dword_offset_of(DoubleBallSocketData, "local_anchor1_b2"))
_OFF_LA2_B1 = wp.constant(dword_offset_of(DoubleBallSocketData, "local_anchor2_b1"))
_OFF_LA2_B2 = wp.constant(dword_offset_of(DoubleBallSocketData, "local_anchor2_b2"))
_OFF_R1_B1 = wp.constant(dword_offset_of(DoubleBallSocketData, "r1_b1"))
_OFF_R1_B2 = wp.constant(dword_offset_of(DoubleBallSocketData, "r1_b2"))
_OFF_R2_B1 = wp.constant(dword_offset_of(DoubleBallSocketData, "r2_b1"))
_OFF_R2_B2 = wp.constant(dword_offset_of(DoubleBallSocketData, "r2_b2"))
_OFF_T1 = wp.constant(dword_offset_of(DoubleBallSocketData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(DoubleBallSocketData, "t2"))
_OFF_BIAS_FACTOR = wp.constant(dword_offset_of(DoubleBallSocketData, "bias_factor"))
_OFF_SOFTNESS = wp.constant(dword_offset_of(DoubleBallSocketData, "softness"))
_OFF_BIAS1 = wp.constant(dword_offset_of(DoubleBallSocketData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(DoubleBallSocketData, "bias2"))
_OFF_A1_INV = wp.constant(dword_offset_of(DoubleBallSocketData, "a1_inv"))
_OFF_UT_AI = wp.constant(dword_offset_of(DoubleBallSocketData, "ut_ai"))
_OFF_S_INV = wp.constant(dword_offset_of(DoubleBallSocketData, "s_inv"))
_OFF_ACC_IMP1 = wp.constant(dword_offset_of(DoubleBallSocketData, "accumulated_impulse1"))
_OFF_ACC_IMP2 = wp.constant(dword_offset_of(DoubleBallSocketData, "accumulated_impulse2"))

#: Total dword count of one fused-hinge constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
DBS_DWORDS: int = num_dwords(DoubleBallSocketData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def double_ball_socket_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def double_ball_socket_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def double_ball_socket_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def double_ball_socket_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def double_ball_socket_get_local_anchor1_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA1_B1, cid)


@wp.func
def double_ball_socket_set_local_anchor1_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA1_B1, cid, v)


@wp.func
def double_ball_socket_get_local_anchor1_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA1_B2, cid)


@wp.func
def double_ball_socket_set_local_anchor1_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA1_B2, cid, v)


@wp.func
def double_ball_socket_get_local_anchor2_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA2_B1, cid)


@wp.func
def double_ball_socket_set_local_anchor2_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA2_B1, cid, v)


@wp.func
def double_ball_socket_get_local_anchor2_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA2_B2, cid)


@wp.func
def double_ball_socket_set_local_anchor2_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA2_B2, cid, v)


@wp.func
def double_ball_socket_get_r1_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R1_B1, cid)


@wp.func
def double_ball_socket_set_r1_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R1_B1, cid, v)


@wp.func
def double_ball_socket_get_r1_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R1_B2, cid)


@wp.func
def double_ball_socket_set_r1_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R1_B2, cid, v)


@wp.func
def double_ball_socket_get_r2_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R2_B1, cid)


@wp.func
def double_ball_socket_set_r2_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R2_B1, cid, v)


@wp.func
def double_ball_socket_get_r2_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R2_B2, cid)


@wp.func
def double_ball_socket_set_r2_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R2_B2, cid, v)


@wp.func
def double_ball_socket_get_t1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_T1, cid)


@wp.func
def double_ball_socket_set_t1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_T1, cid, v)


@wp.func
def double_ball_socket_get_t2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_T2, cid)


@wp.func
def double_ball_socket_set_t2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_T2, cid, v)


@wp.func
def double_ball_socket_get_bias_factor(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_FACTOR, cid)


@wp.func
def double_ball_socket_set_bias_factor(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_FACTOR, cid, v)


@wp.func
def double_ball_socket_get_softness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_SOFTNESS, cid)


@wp.func
def double_ball_socket_set_softness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_SOFTNESS, cid, v)


@wp.func
def double_ball_socket_get_bias1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS1, cid)


@wp.func
def double_ball_socket_set_bias1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS1, cid, v)


@wp.func
def double_ball_socket_get_bias2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS2, cid)


@wp.func
def double_ball_socket_set_bias2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS2, cid, v)


@wp.func
def double_ball_socket_get_a1_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_A1_INV, cid)


@wp.func
def double_ball_socket_set_a1_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_A1_INV, cid, v)


@wp.func
def double_ball_socket_get_ut_ai(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_UT_AI, cid)


@wp.func
def double_ball_socket_set_ut_ai(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_UT_AI, cid, v)


@wp.func
def double_ball_socket_get_s_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_S_INV, cid)


@wp.func
def double_ball_socket_set_s_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_S_INV, cid, v)


@wp.func
def double_ball_socket_get_accumulated_impulse1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP1, cid)


@wp.func
def double_ball_socket_set_accumulated_impulse1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP1, cid, v)


@wp.func
def double_ball_socket_get_accumulated_impulse2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP2, cid)


@wp.func
def double_ball_socket_set_accumulated_impulse2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP2, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel; mirrors ball_socket_initialize_kernel)
# ---------------------------------------------------------------------------


@wp.kernel
def double_ball_socket_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor1: wp.array[wp.vec3f],
    anchor2: wp.array[wp.vec3f],
):
    """Pack one batch of fused-hinge descriptors into ``constraints``.

    Snapshots the two anchors (in world space) into each body's local
    frame and zero-inits the cached / accumulated state. Same role as
    :func:`ball_socket_initialize_kernel`, just for two anchors.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; this kernel only reads
            ``position`` / ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor1: World-space first anchor [num_in_batch] [m]. Defines
            the hinge endpoint that becomes "anchor 1" in the schema.
        anchor2: World-space second anchor [num_in_batch] [m]. The
            line ``anchor1 -> anchor2`` is the hinge axis.
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

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_DOUBLE_BALL_SOCKET)
    double_ball_socket_set_body1(constraints, cid, b1)
    double_ball_socket_set_body2(constraints, cid, b2)
    double_ball_socket_set_local_anchor1_b1(constraints, cid, la1_b1)
    double_ball_socket_set_local_anchor1_b2(constraints, cid, la1_b2)
    double_ball_socket_set_local_anchor2_b1(constraints, cid, la2_b1)
    double_ball_socket_set_local_anchor2_b2(constraints, cid, la2_b2)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    double_ball_socket_set_r1_b1(constraints, cid, zero3)
    double_ball_socket_set_r1_b2(constraints, cid, zero3)
    double_ball_socket_set_r2_b1(constraints, cid, zero3)
    double_ball_socket_set_r2_b2(constraints, cid, zero3)
    double_ball_socket_set_t1(constraints, cid, zero3)
    double_ball_socket_set_t2(constraints, cid, zero3)
    double_ball_socket_set_bias1(constraints, cid, zero3)
    double_ball_socket_set_bias2(constraints, cid, zero3)
    double_ball_socket_set_accumulated_impulse1(constraints, cid, zero3)
    double_ball_socket_set_accumulated_impulse2(constraints, cid, zero3)

    double_ball_socket_set_bias_factor(constraints, cid, DEFAULT_LINEAR_BIAS)
    double_ball_socket_set_softness(constraints, cid, DEFAULT_LINEAR_SOFTNESS)

    eye = wp.identity(3, dtype=wp.float32)
    double_ball_socket_set_a1_inv(constraints, cid, eye)
    double_ball_socket_set_ut_ai(constraints, cid, eye)
    double_ball_socket_set_s_inv(constraints, cid, eye)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Same two access levels as ball-socket: a composable ``*_at`` variant
# that takes ``base_offset`` + ``ConstraintBodies`` (so a future fused
# composite constraint can stack this block alongside others in one
# column), and thin direct wrappers.
#
# Symbol cheat-sheet (matches module-docstring derivation):
#   r{k}_{b}          : world-space lever arm of anchor k on body b
#   cr{k}_{b}         : skew([r{k}_{b}])
#   A{k}              : ball-socket effective mass at anchor k (3x3)
#   B                 : cross-anchor coupling (3x3)
#   T   = [t1 | t2]   : tangent basis perp to hinge axis (3x2)
#   U   = B T         : 3x2
#   D   = T^T A2 T    : 2x2 (anchor-2 effective mass projected on T)
#   S   = D - U^T A1^{-1} U          (Schur complement, 2x2)
#   K   = [[A1, U],[U^T, D]]         (full 5x5 effective mass)


@wp.func
def double_ball_socket_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the fused two-anchor hinge.

    Recomputes world-space lever arms, hinge axis + tangent basis, the
    blocks of the 5x5 effective mass, the bias vectors, and warm-starts
    the bodies with the cached accumulated impulse.

    See module docstring for the derivation; see
    :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
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

    # World-frame lever arms (rotate body-local anchors into world frame).
    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    r2_b1 = wp.quat_rotate(orientation1, la2_b1)
    r2_b2 = wp.quat_rotate(orientation2, la2_b2)

    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)
    write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
    write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)

    # World-frame anchor positions (per anchor, per body).
    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2
    p2_b1 = position1 + r2_b1
    p2_b2 = position2 + r2_b2

    # Hinge axis = direction from anchor 1 to anchor 2 on body 2 (we
    # could equally use body 1 or the midpoint -- body 2 matches the
    # orientation convention used elsewhere in the codebase). If both
    # anchors coincide the hinge degenerates to a ball-socket; fall
    # back to an arbitrary axis to keep the math finite (the Schur
    # solve still produces a valid 5-DoF lock in that case, just with
    # a wasted 6th equation).
    hinge_vec = p2_b2 - p1_b2
    hinge_len2 = wp.dot(hinge_vec, hinge_vec)
    if hinge_len2 > 1.0e-20:
        n_hat = hinge_vec / wp.sqrt(hinge_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)

    # Right-handed orthonormal basis (t1, t2, n_hat). create_orthonormal
    # gives one perpendicular; the cross-product gives the other.
    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    # Skew matrices for the per-anchor Jacobian blocks.
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    eye3 = wp.identity(3, dtype=wp.float32)
    softness = read_float(constraints, base_offset + _OFF_SOFTNESS, cid) * idt

    # A1 = standard ball-socket effective mass at anchor 1 (+ softness on diag).
    a1 = inv_mass1 * eye3
    a1 = a1 + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1))
    a1 = a1 + inv_mass2 * eye3
    a1 = a1 + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
    a1[0, 0] = a1[0, 0] + softness
    a1[1, 1] = a1[1, 1] + softness
    a1[2, 2] = a1[2, 2] + softness

    # A2 = same at anchor 2 (+ softness on diag, will be projected to 2x2 below).
    a2 = inv_mass1 * eye3
    a2 = a2 + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    a2 = a2 + inv_mass2 * eye3
    a2 = a2 + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))
    a2[0, 0] = a2[0, 0] + softness
    a2[1, 1] = a2[1, 1] + softness
    a2[2, 2] = a2[2, 2] + softness

    # B = cross-anchor coupling J1 M^{-1} J2^T.
    b_mat = (inv_mass1 + inv_mass2) * eye3
    b_mat = b_mat + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    b_mat = b_mat + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    # Anchor-2 projected onto the tangent basis. We pack T as a mat33
    # whose columns are (t1, t2, 0) so that the matrix products below
    # naturally pick out the 2D blocks; the third column is left at zero
    # which makes any contribution it would have made vanish.
    t_mat = wp.mat33f(
        t1[0], t2[0], 0.0,
        t1[1], t2[1], 0.0,
        t1[2], t2[2], 0.0,
    )
    tt = wp.transpose(t_mat)  # T^T as a mat33 with the 3rd row zero.

    # U = B T ; the 3rd column of U is zero by construction.
    u_mat = b_mat @ t_mat

    # D = T^T A2 T ; only the 2x2 top-left block is meaningful, but we
    # carry it as a mat33 to reuse the 3x3 inverse helpers we don't
    # actually call -- D itself is solved in 2x2 form below.
    d_mat = tt @ (a2 @ t_mat)

    # Schur complement: S = D - U^T A1^{-1} U.
    a1_inv = wp.inverse(a1)
    ut_ai = wp.transpose(u_mat) @ a1_inv  # 3x3 with last col / row zero.
    s_mat = d_mat - ut_ai @ u_mat

    # Extract / invert the 2x2 top-left block of S.
    s22 = wp.mat22f(
        s_mat[0, 0], s_mat[0, 1],
        s_mat[1, 0], s_mat[1, 1],
    )
    s22_inv = wp.inverse(s22)

    # Pack S^{-1} back into a mat33 (top-left 2x2 active; rest zero) so
    # we can store it in a single mat33 slot with the existing helpers.
    s_inv_packed = wp.mat33f(
        s22_inv[0, 0], s22_inv[0, 1], 0.0,
        s22_inv[1, 0], s22_inv[1, 1], 0.0,
        0.0,           0.0,           0.0,
    )

    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, ut_ai)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv_packed)

    # Bias terms (Baumgarte position drift -> velocity correction).
    # Anchor 1: full 3-vector. Anchor 2: project drift onto (t1, t2),
    # store as the first two components of a vec3 with the third 0.
    bias_factor = read_float(constraints, base_offset + _OFF_BIAS_FACTOR, cid)
    bias1 = (p1_b2 - p1_b1) * bias_factor * idt
    drift2 = p2_b2 - p2_b1
    bias2_t1 = wp.dot(t1, drift2) * bias_factor * idt
    bias2_t2 = wp.dot(t2, drift2) * bias_factor * idt
    bias2 = wp.vec3f(bias2_t1, bias2_t2, 0.0)
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)

    # Warm start: re-apply the cached accumulated impulses. Anchor 2 is
    # stored in *world frame* so the warm-start stays valid even when
    # the tangent basis swings; we simply project it back onto the
    # current basis on entry to iterate. Here the warm start just
    # applies the impulses straight to the bodies, no projection
    # needed.
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)

    velocity1 = bodies.velocity[b1] - inv_mass1 * (acc1 + acc2)
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1_b1 @ acc1 + cr2_b1 @ acc2)
    velocity2 = bodies.velocity[b2] + inv_mass2 * (acc1 + acc2)
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr1_b2 @ acc1 + cr2_b2 @ acc2)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def double_ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable PGS iteration step for the fused two-anchor hinge.

    Solves the 5x5 block-diagonal system via the Schur complement
    (3x3 + 2x2 inverses), then applies the resulting impulse to both
    bodies. See :func:`ball_socket_iterate_at` for the
    ``base_offset`` / ``body_pair`` contract.
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
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    ut_ai = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)
    s_inv_packed = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
    bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    softness = read_float(constraints, base_offset + _OFF_SOFTNESS, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    # Project the world-frame anchor-2 accumulated impulse onto the
    # current tangent basis so the softness term and the impulse update
    # below stay self-consistent in tangent coordinates.
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)

    # Velocity error at anchor 1 (jv1 = -v1 + cr1_b1*w1 + v2 - cr1_b2*w2),
    # full 3-vector.
    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    # Velocity error at anchor 2 in world frame, then projected onto
    # the tangent basis to give a 2-vector.
    jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
    jv2_t1 = wp.dot(t1, jv2_world)
    jv2_t2 = wp.dot(t2, jv2_world)

    # Soft-constraint contribution (matches single ball-socket: the
    # softness term shrinks the cumulative impulse toward zero by
    # adding ``softness * idt * acc`` to the right-hand side).
    softness_idt = softness * idt
    rhs1 = jv1 + bias1 + acc1 * softness_idt
    rhs2_t1 = jv2_t1 + bias2[0] + acc2_t1 * softness_idt
    rhs2_t2 = jv2_t2 + bias2[1] + acc2_t2 * softness_idt
    rhs2 = wp.vec2f(rhs2_t1, rhs2_t2)

    # Schur-complement solve:
    #   lambda2 = -S^{-1} ( rhs2 - U^T A1^{-1} rhs1 )
    #   lambda1 = -A1^{-1} ( rhs1 + U lambda2 )
    # ``ut_ai`` is the precomputed 3x3 with U^T A1^{-1} in the top-left
    # 2x3; multiplying by rhs1 then taking the first two components
    # gives the 2-vector U^T A1^{-1} rhs1.
    ut_ai_rhs1_3 = ut_ai @ rhs1
    ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

    s_inv_22 = wp.mat22f(
        s_inv_packed[0, 0], s_inv_packed[0, 1],
        s_inv_packed[1, 0], s_inv_packed[1, 1],
    )
    lam2 = -(s_inv_22 @ (rhs2 - ut_ai_rhs1))

    # Re-expand lambda2 from tangent coordinates to a world 3-vector
    # so we can fold it into the 3x3 anchor-1 equation and into the
    # impulse application below.
    lam2_world = lam2[0] * t1 + lam2[1] * t2

    # U lambda2 = B (T lambda2) = B lambda2_world. We don't have B
    # cached separately (we packed U = B T into ut_ai indirectly),
    # but we can reconstruct U lambda2 from the original Jacobian
    # decomposition: it is exactly the cross-block contribution of
    # the anchor-2 impulse to the anchor-1 row, i.e.
    #     U lambda2 = (m1^{-1} + m2^{-1}) lambda2_world
    #               + cr1_b1 I1^{-1} cr2_b1^T lambda2_world
    #               + cr1_b2 I2^{-1} cr2_b2^T lambda2_world
    # (Same structure as the B matrix in prepare_for_iteration.)
    u_lam2 = (inv_mass1 + inv_mass2) * lam2_world
    u_lam2 = u_lam2 + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_world))
    u_lam2 = u_lam2 + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_world))

    lam1 = -(a1_inv @ (rhs1 + u_lam2))

    # Total per-body linear impulse is lam1 + lam2_world (anchor-1 +
    # anchor-2 in world coordinates); torque contributions split per
    # anchor with their respective lever arms.
    total_lin = lam1 + lam2_world

    bodies.velocity[b1] = velocity1 - inv_mass1 * total_lin
    bodies.angular_velocity[b1] = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1 + cr2_b1 @ lam2_world)

    bodies.velocity[b2] = velocity2 + inv_mass2 * total_lin
    bodies.angular_velocity[b2] = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1 + cr2_b2 @ lam2_world)

    # Accumulate. Anchor 1 stores its own world-frame impulse; anchor
    # 2 stores the world-frame projection of its tangent impulse so
    # warm-starts survive tangent-basis rotation between substeps.
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)


@wp.func
def double_ball_socket_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2; see :func:`double_ball_socket_world_wrench`."""
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    force = (acc1 + acc2) * idt
    # Torque about body 2's COM: each anchor contributes its lever arm
    # x its own impulse (this preserves the per-anchor moment, not just
    # the moment of the resultant force).
    torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt)
    return force, torque


@wp.func
def double_ball_socket_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct prepare entry: reads body indices from the column header
    and forwards to :func:`double_ball_socket_prepare_for_iteration_at`
    with ``base_offset = 0``."""
    b1 = double_ball_socket_get_body1(constraints, cid)
    b2 = double_ball_socket_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    double_ball_socket_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def double_ball_socket_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct iterate entry; see :func:`double_ball_socket_iterate_at`."""
    b1 = double_ball_socket_get_body1(constraints, cid)
    b2 = double_ball_socket_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    double_ball_socket_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def double_ball_socket_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body 2.

    Force is the sum of the two anchor accumulated impulses divided by
    the substep ``dt`` (``idt = 1 / substep_dt``); torque is each
    impulse's moment about body 2's COM, using the cached lever arms
    ``r1_b2`` / ``r2_b2`` from the most recent ``prepare_for_iteration``.
    """
    return double_ball_socket_world_wrench_at(constraints, cid, 0, idt)
