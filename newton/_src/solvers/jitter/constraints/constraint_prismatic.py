# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Prismatic (sliding) joint constraint.

Locks 5 of the 6 relative DoF between two bodies:

* **3 rotational DoF** -- the bodies must keep the same relative
  orientation as at ``initialize`` time (``q0 = q2^* * q1``).
* **2 translational DoF** -- the bodies may only separate along a
  fixed slide axis ``n_hat`` (frozen in body 1's local frame at
  initialize); displacement in the plane perpendicular to ``n_hat``
  is forced to zero.

The remaining DoF -- translation along ``n_hat`` -- is free.

Why a 5x12 Jacobian instead of two cascaded constraints?
--------------------------------------------------------
Same motivation as :mod:`constraint_double_ball_socket`: solving the
joint as one fused 5-row constraint per PGS thread converges much
faster on chains than colouring a separate 3-row angular lock and a
2-row planar lock. The two blocks are coupled through "angular impulse
moves the anchor laterally" so block-diagonal solving (each block
independently) wastes iterations bouncing the cross-block residual
back and forth.

Schur-complement solve (avoiding a 5x5 inverse)
-----------------------------------------------
Warp's ``wp.inverse`` overloads only cover 2x2 / 3x3 / 4x4. The 5x5
effective mass

::

    K = [ K_rot          K_rot_trans ]   in R^{5x5}
        [ K_rot_trans^T  K_trans     ]

with ``K_rot in R^{3x3}``, ``K_trans in R^{2x2}``, ``K_rot_trans in
R^{3x2}`` is solved by block-elimination using only one 3x3 and one
2x2 inverse:

::

    S       = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans   # 2x2 Schur complement
    lambda2 = -S^{-1}     ( b2 - K_rot_trans^T K_rot^{-1} b1 )
    lambda1 = -K_rot^{-1} ( b1 + K_rot_trans lambda2 )

We cache ``K_rot_inv`` (3x3), ``S_inv`` (2x2 packed into a mat33),
and ``Kt_Ki = K_rot_trans^T K_rot^{-1}`` (2x3 packed into a mat33)
in the constraint column so ``iterate`` is just three small mat-vecs
plus the impulse application -- no per-iter inverses.

Block ordering (rotational first, translational second) was chosen
because:

* ``K_rot = I_1^{-1} + I_2^{-1}`` is the simpler matrix (no skew-r
  contractions needed).
* The 3-row angular constraint is the larger of the two blocks, so
  putting it as the 3x3 outer block of the Schur factorisation
  matches the natural inversion size that ``wp.inverse`` provides
  for free.

Hertz / damping (Box2D v3 / Bepu / Nordby)
------------------------------------------
The two blocks are physically distinct (angular vs linear), so they
get *separate* user-facing knobs ``hertz_angular``/``damping_angular``
and ``hertz_linear``/``damping_linear``. Each block runs through its
own :func:`soft_constraint_coefficients` triple
(``bias_rate, mass_coeff, impulse_coeff``); the iterate path scales
each block's PGS update independently.

Mapping summary (deltas vs ball-socket only):

* slide axis ``n_hat``      -> stored in **body 1's local frame** as
                               ``axis_local_b1`` (so it follows body 1
                               rigidly; world ``n_hat = R_1 axis_local``
                               is recomputed each substep).
* tangent basis ``T``       -> ``[t1 | t2]`` in *world frame*, recomputed
                               each prepare from world ``n_hat`` via
                               :func:`create_orthonormal`.
* rest orientation ``q0``   -> ``q2^* * q1`` snapshotted at initialize.
* rest anchor pair          -> per-body local-anchor offsets so the
                               world-frame "where the slider lives"
                               point can be reconstructed each substep.
* relative-rotation error   -> 3-vec from the quaternion error
                               (``2 * vec(q_e)`` linearisation, same
                               idiom as :mod:`constraint_hinge_angle`
                               but unprojected -- we lock all three
                               angular DoF).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_PRISMATIC,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    read_float,
    read_int,
    read_mat33,
    read_quat,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.helpers.math_helpers import (
    create_orthonormal,
    qmatrix_project_multiply_left_right,
)

__all__ = [
    "PR_DWORDS",
    "PrismaticData",
    "prismatic_get_accumulated_impulse_lin",
    "prismatic_get_accumulated_impulse_rot",
    "prismatic_get_axis_local_b1",
    "prismatic_get_bias_lin",
    "prismatic_get_bias_rate_angular",
    "prismatic_get_bias_rate_linear",
    "prismatic_get_bias_rot",
    "prismatic_get_body1",
    "prismatic_get_body2",
    "prismatic_get_damping_ratio_angular",
    "prismatic_get_damping_ratio_linear",
    "prismatic_get_hertz_angular",
    "prismatic_get_hertz_linear",
    "prismatic_get_impulse_coeff_angular",
    "prismatic_get_impulse_coeff_linear",
    "prismatic_get_j_rot",
    "prismatic_get_k_rot_inv",
    "prismatic_get_kt_ki",
    "prismatic_get_local_anchor_b1",
    "prismatic_get_local_anchor_b2",
    "prismatic_get_mass_coeff_angular",
    "prismatic_get_mass_coeff_linear",
    "prismatic_get_q0",
    "prismatic_get_r1",
    "prismatic_get_r2",
    "prismatic_get_s_inv",
    "prismatic_get_t1",
    "prismatic_get_t2",
    "prismatic_initialize_kernel",
    "prismatic_iterate",
    "prismatic_iterate_at",
    "prismatic_prepare_for_iteration",
    "prismatic_prepare_for_iteration_at",
    "prismatic_set_accumulated_impulse_lin",
    "prismatic_set_accumulated_impulse_rot",
    "prismatic_set_axis_local_b1",
    "prismatic_set_bias_lin",
    "prismatic_set_bias_rate_angular",
    "prismatic_set_bias_rate_linear",
    "prismatic_set_bias_rot",
    "prismatic_set_body1",
    "prismatic_set_body2",
    "prismatic_set_damping_ratio_angular",
    "prismatic_set_damping_ratio_linear",
    "prismatic_set_hertz_angular",
    "prismatic_set_hertz_linear",
    "prismatic_set_impulse_coeff_angular",
    "prismatic_set_impulse_coeff_linear",
    "prismatic_set_j_rot",
    "prismatic_set_k_rot_inv",
    "prismatic_set_kt_ki",
    "prismatic_set_local_anchor_b1",
    "prismatic_set_local_anchor_b2",
    "prismatic_set_mass_coeff_angular",
    "prismatic_set_mass_coeff_linear",
    "prismatic_set_q0",
    "prismatic_set_r1",
    "prismatic_set_r2",
    "prismatic_set_s_inv",
    "prismatic_set_t1",
    "prismatic_set_t2",
    "prismatic_world_error",
    "prismatic_world_error_at",
    "prismatic_world_wrench",
    "prismatic_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class PrismaticData:
    """Per-constraint dword-layout schema for a prismatic (slider) joint.

    *Schema only* (same conventions as :class:`BallSocketData` /
    :class:`DoubleBallSocketData`). Field order fixes dword offsets;
    runtime kernels read/write fields out of the shared
    :class:`ConstraintContainer`.

    The first three fields are the global constraint header
    (``constraint_type``, ``body1``, ``body2`` at dwords 0/1/2),
    enforced by :func:`assert_constraint_header`.
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Anchor point in each body's local frame. The two anchors are
    # snapshotted from a shared world-space point at initialize (so
    # they coincide at rest); their separation drift in the plane
    # perpendicular to the slide axis is what the 2-row translational
    # block locks.
    local_anchor_b1: wp.vec3f
    local_anchor_b2: wp.vec3f

    # Slide axis in body 1's local frame. World axis recomputed each
    # substep as ``n_hat = R_1 axis_local_b1``. Storing it body-local
    # makes the slider "ride" body 1 rigidly, which is the standard
    # Bullet/ODE convention.
    axis_local_b1: wp.vec3f

    # Rest-pose relative orientation ``q0 = q2^* * q1`` snapshotted at
    # initialize. The angular error is the rotation that maps current
    # ``q2^* q1`` back to ``q0``.
    q0: wp.quatf

    # World-space lever arms from each body's COM to the (shared) anchor
    # point. ``r1`` may diverge from ``r2`` between substeps as the
    # bodies translate along the slide axis; that lateral drift is
    # exactly what the 2-row translational block sees.
    r1: wp.vec3f
    r2: wp.vec3f

    # Tangent basis perpendicular to the world slide axis, recomputed
    # each prepare. ``t1, t2, n_hat`` is right-handed.
    t1: wp.vec3f
    t2: wp.vec3f

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation; see :func:`soft_constraint_coefficients`). Two pairs:
    # ``*_angular`` for the 3-row rotational lock, ``*_linear`` for the
    # 2-row perpendicular-translation lock. Setting hertz <= 0 -> rigid
    # for that block.
    hertz_angular: wp.float32
    damping_ratio_angular: wp.float32
    hertz_linear: wp.float32
    damping_ratio_linear: wp.float32

    # Cached per-substep soft-constraint coefficients (recomputed each
    # ``prepare_for_iteration`` from the current ``dt``). Per block.
    bias_rate_angular: wp.float32
    mass_coeff_angular: wp.float32
    impulse_coeff_angular: wp.float32
    bias_rate_linear: wp.float32
    mass_coeff_linear: wp.float32
    impulse_coeff_linear: wp.float32

    # Cached velocity-error bias vectors. ``bias_rot`` is a 3-vec
    # (one per angular axis); ``bias_lin`` packs the 2-vec
    # ``T^T (P2 - P1) * bias_rate_linear`` into the first two
    # components of a vec3 (third = 0) for storage uniformity.
    bias_rot: wp.vec3f
    bias_lin: wp.vec3f

    # Cached angular Jacobian (3x3, world frame). Same convention as
    # :mod:`constraint_hinge_angle`: rows are world-axis Jacobian rows,
    # so ``J^T (w1 - w2)`` is the per-quaternion-axis velocity error
    # and ``J^T lam`` is the world-frame angular impulse for an
    # impulse ``lam`` expressed in the quaternion-axis basis. Cached
    # because the solve, the impulse application, the warm-start, and
    # the cross-block all need it consistently.
    j_rot: wp.mat33f

    # Cached effective-mass blocks for the Schur-complement solve.
    # ``k_rot_inv`` is the 3x3 inverse of the angular block. ``kt_ki``
    # holds the 2x3 product ``K_rot_trans^T K_rot^{-1}`` packed into the
    # first two rows of a mat33 (third row = 0). ``s_inv`` holds the
    # 2x2 Schur-complement inverse packed into the top-left of a mat33.
    k_rot_inv: wp.mat33f
    kt_ki: wp.mat33f
    s_inv: wp.mat33f

    # Accumulated PGS impulses split per block.
    # ``accumulated_impulse_rot`` is the world-frame angular impulse
    # (3-dof). ``accumulated_impulse_lin`` is the world-frame linear
    # impulse projected from the 2D tangent solution back to a
    # 3-vector ``T lambda_lin``; storing it in world frame keeps the
    # warm-start valid even when the tangent basis rotates between
    # substeps.
    accumulated_impulse_rot: wp.vec3f
    accumulated_impulse_lin: wp.vec3f


assert_constraint_header(PrismaticData)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(PrismaticData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(PrismaticData, "body2"))
_OFF_LA_B1 = wp.constant(dword_offset_of(PrismaticData, "local_anchor_b1"))
_OFF_LA_B2 = wp.constant(dword_offset_of(PrismaticData, "local_anchor_b2"))
_OFF_AXIS_LOCAL_B1 = wp.constant(dword_offset_of(PrismaticData, "axis_local_b1"))
_OFF_Q0 = wp.constant(dword_offset_of(PrismaticData, "q0"))
_OFF_R1 = wp.constant(dword_offset_of(PrismaticData, "r1"))
_OFF_R2 = wp.constant(dword_offset_of(PrismaticData, "r2"))
_OFF_T1 = wp.constant(dword_offset_of(PrismaticData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(PrismaticData, "t2"))
_OFF_HERTZ_ANG = wp.constant(dword_offset_of(PrismaticData, "hertz_angular"))
_OFF_DAMPING_ANG = wp.constant(dword_offset_of(PrismaticData, "damping_ratio_angular"))
_OFF_HERTZ_LIN = wp.constant(dword_offset_of(PrismaticData, "hertz_linear"))
_OFF_DAMPING_LIN = wp.constant(dword_offset_of(PrismaticData, "damping_ratio_linear"))
_OFF_BIAS_RATE_ANG = wp.constant(dword_offset_of(PrismaticData, "bias_rate_angular"))
_OFF_MASS_COEFF_ANG = wp.constant(dword_offset_of(PrismaticData, "mass_coeff_angular"))
_OFF_IMPULSE_COEFF_ANG = wp.constant(dword_offset_of(PrismaticData, "impulse_coeff_angular"))
_OFF_BIAS_RATE_LIN = wp.constant(dword_offset_of(PrismaticData, "bias_rate_linear"))
_OFF_MASS_COEFF_LIN = wp.constant(dword_offset_of(PrismaticData, "mass_coeff_linear"))
_OFF_IMPULSE_COEFF_LIN = wp.constant(dword_offset_of(PrismaticData, "impulse_coeff_linear"))
_OFF_BIAS_ROT = wp.constant(dword_offset_of(PrismaticData, "bias_rot"))
_OFF_BIAS_LIN = wp.constant(dword_offset_of(PrismaticData, "bias_lin"))
_OFF_J_ROT = wp.constant(dword_offset_of(PrismaticData, "j_rot"))
_OFF_K_ROT_INV = wp.constant(dword_offset_of(PrismaticData, "k_rot_inv"))
_OFF_KT_KI = wp.constant(dword_offset_of(PrismaticData, "kt_ki"))
_OFF_S_INV = wp.constant(dword_offset_of(PrismaticData, "s_inv"))
_OFF_ACC_IMP_ROT = wp.constant(dword_offset_of(PrismaticData, "accumulated_impulse_rot"))
_OFF_ACC_IMP_LIN = wp.constant(dword_offset_of(PrismaticData, "accumulated_impulse_lin"))

#: Total dword count of one prismatic constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
PR_DWORDS: int = num_dwords(PrismaticData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def prismatic_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def prismatic_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def prismatic_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def prismatic_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def prismatic_get_local_anchor_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA_B1, cid)


@wp.func
def prismatic_set_local_anchor_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA_B1, cid, v)


@wp.func
def prismatic_get_local_anchor_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA_B2, cid)


@wp.func
def prismatic_set_local_anchor_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA_B2, cid, v)


@wp.func
def prismatic_get_axis_local_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_AXIS_LOCAL_B1, cid)


@wp.func
def prismatic_set_axis_local_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_AXIS_LOCAL_B1, cid, v)


@wp.func
def prismatic_get_q0(c: ConstraintContainer, cid: wp.int32) -> wp.quatf:
    return read_quat(c, _OFF_Q0, cid)


@wp.func
def prismatic_set_q0(c: ConstraintContainer, cid: wp.int32, v: wp.quatf):
    write_quat(c, _OFF_Q0, cid, v)


@wp.func
def prismatic_get_r1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R1, cid)


@wp.func
def prismatic_set_r1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R1, cid, v)


@wp.func
def prismatic_get_r2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R2, cid)


@wp.func
def prismatic_set_r2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R2, cid, v)


@wp.func
def prismatic_get_t1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_T1, cid)


@wp.func
def prismatic_set_t1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_T1, cid, v)


@wp.func
def prismatic_get_t2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_T2, cid)


@wp.func
def prismatic_set_t2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_T2, cid, v)


@wp.func
def prismatic_get_hertz_angular(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ_ANG, cid)


@wp.func
def prismatic_set_hertz_angular(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ_ANG, cid, v)


@wp.func
def prismatic_get_damping_ratio_angular(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_ANG, cid)


@wp.func
def prismatic_set_damping_ratio_angular(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_ANG, cid, v)


@wp.func
def prismatic_get_hertz_linear(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ_LIN, cid)


@wp.func
def prismatic_set_hertz_linear(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ_LIN, cid, v)


@wp.func
def prismatic_get_damping_ratio_linear(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_LIN, cid)


@wp.func
def prismatic_set_damping_ratio_linear(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_LIN, cid, v)


@wp.func
def prismatic_get_bias_rate_angular(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_RATE_ANG, cid)


@wp.func
def prismatic_set_bias_rate_angular(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_RATE_ANG, cid, v)


@wp.func
def prismatic_get_mass_coeff_angular(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF_ANG, cid)


@wp.func
def prismatic_set_mass_coeff_angular(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF_ANG, cid, v)


@wp.func
def prismatic_get_impulse_coeff_angular(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF_ANG, cid)


@wp.func
def prismatic_set_impulse_coeff_angular(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF_ANG, cid, v)


@wp.func
def prismatic_get_bias_rate_linear(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_RATE_LIN, cid)


@wp.func
def prismatic_set_bias_rate_linear(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_RATE_LIN, cid, v)


@wp.func
def prismatic_get_mass_coeff_linear(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF_LIN, cid)


@wp.func
def prismatic_set_mass_coeff_linear(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF_LIN, cid, v)


@wp.func
def prismatic_get_impulse_coeff_linear(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF_LIN, cid)


@wp.func
def prismatic_set_impulse_coeff_linear(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF_LIN, cid, v)


@wp.func
def prismatic_get_bias_rot(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_ROT, cid)


@wp.func
def prismatic_set_bias_rot(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_ROT, cid, v)


@wp.func
def prismatic_get_bias_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_LIN, cid)


@wp.func
def prismatic_set_bias_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_LIN, cid, v)


@wp.func
def prismatic_get_j_rot(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_J_ROT, cid)


@wp.func
def prismatic_set_j_rot(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_J_ROT, cid, v)


@wp.func
def prismatic_get_k_rot_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_K_ROT_INV, cid)


@wp.func
def prismatic_set_k_rot_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_K_ROT_INV, cid, v)


@wp.func
def prismatic_get_kt_ki(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_KT_KI, cid)


@wp.func
def prismatic_set_kt_ki(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_KT_KI, cid, v)


@wp.func
def prismatic_get_s_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_S_INV, cid)


@wp.func
def prismatic_set_s_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_S_INV, cid, v)


@wp.func
def prismatic_get_accumulated_impulse_rot(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP_ROT, cid)


@wp.func
def prismatic_set_accumulated_impulse_rot(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP_ROT, cid, v)


@wp.func
def prismatic_get_accumulated_impulse_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP_LIN, cid)


@wp.func
def prismatic_set_accumulated_impulse_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP_LIN, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel)
# ---------------------------------------------------------------------------


@wp.kernel
def prismatic_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor: wp.array[wp.vec3f],
    axis_world: wp.array[wp.vec3f],
    hertz_angular: wp.array[wp.float32],
    damping_ratio_angular: wp.array[wp.float32],
    hertz_linear: wp.array[wp.float32],
    damping_ratio_linear: wp.array[wp.float32],
):
    """Pack one batch of prismatic descriptors into ``constraints``.

    Snapshots the user's (anchor, slide axis) into body-local frames
    plus the rest-pose relative orientation ``q0 = q2^* q1`` so the
    runtime math operates entirely on quantities that move rigidly
    with the bodies.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; reads ``position`` /
            ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor point [num_in_batch] [m]. The
            point on the slide axis where both bodies' lever arms are
            measured from. Stored in each body's local frame; the
            two body-local copies coincide at rest by construction.
        axis_world: World-space slide axis direction [num_in_batch],
            **not assumed unit length** (we normalise here). Stored
            body-1 local so the slide direction "rides" body 1.
        hertz_angular: Per-constraint angular-lock stiffness [Hz];
            applied to the 3-row rotational block.
        damping_ratio_angular: Angular-lock damping ratio
            (1 = critically damped).
        hertz_linear: Per-constraint linear-lock stiffness [Hz];
            applied to the 2-row perpendicular-translation block.
        damping_ratio_linear: Linear-lock damping ratio.
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a_world = anchor[tid]
    n_world_unnorm = axis_world[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # Anchor in each body's local frame. Both are derived from the
    # same world point so they coincide initially; the perpendicular
    # drift that develops as the bodies move off-axis is what the
    # 2-row linear lock corrects.
    la_b1 = wp.quat_rotate_inv(q1, a_world - pos1)
    la_b2 = wp.quat_rotate_inv(q2, a_world - pos2)

    # Slide axis in body 1's local frame (unit length).
    n_world_len = wp.length(n_world_unnorm)
    inv_len = 1.0 / wp.max(n_world_len, 1.0e-30)
    n_world = n_world_unnorm * inv_len
    axis_local = wp.quat_rotate_inv(q1, n_world)

    # Rest-pose relative orientation. Same convention as
    # ``constraint_hinge_angle``: q0 expresses body-1's orientation
    # in body-2's frame at construction time.
    q0 = wp.quat_inverse(q2) * q1

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_PRISMATIC)
    prismatic_set_body1(constraints, cid, b1)
    prismatic_set_body2(constraints, cid, b2)
    prismatic_set_local_anchor_b1(constraints, cid, la_b1)
    prismatic_set_local_anchor_b2(constraints, cid, la_b2)
    prismatic_set_axis_local_b1(constraints, cid, axis_local)
    prismatic_set_q0(constraints, cid, q0)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    prismatic_set_r1(constraints, cid, zero3)
    prismatic_set_r2(constraints, cid, zero3)
    prismatic_set_t1(constraints, cid, zero3)
    prismatic_set_t2(constraints, cid, zero3)
    prismatic_set_bias_rot(constraints, cid, zero3)
    prismatic_set_bias_lin(constraints, cid, zero3)
    prismatic_set_accumulated_impulse_rot(constraints, cid, zero3)
    prismatic_set_accumulated_impulse_lin(constraints, cid, zero3)

    prismatic_set_hertz_angular(constraints, cid, hertz_angular[tid])
    prismatic_set_damping_ratio_angular(constraints, cid, damping_ratio_angular[tid])
    prismatic_set_hertz_linear(constraints, cid, hertz_linear[tid])
    prismatic_set_damping_ratio_linear(constraints, cid, damping_ratio_linear[tid])
    prismatic_set_bias_rate_angular(constraints, cid, 0.0)
    prismatic_set_mass_coeff_angular(constraints, cid, 1.0)
    prismatic_set_impulse_coeff_angular(constraints, cid, 0.0)
    prismatic_set_bias_rate_linear(constraints, cid, 0.0)
    prismatic_set_mass_coeff_linear(constraints, cid, 1.0)
    prismatic_set_impulse_coeff_linear(constraints, cid, 0.0)

    eye = wp.identity(3, dtype=wp.float32)
    prismatic_set_j_rot(constraints, cid, eye)
    prismatic_set_k_rot_inv(constraints, cid, eye)
    prismatic_set_kt_ki(constraints, cid, eye)
    prismatic_set_s_inv(constraints, cid, eye)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Symbol cheat-sheet (matches module-docstring derivation):
#   r1, r2            : world-space lever arms (body 1 / body 2 -> anchor)
#   cr1, cr2          : skew([r1]), skew([r2])
#   n_hat             : world-space slide axis (= R_1 axis_local_b1)
#   T = [t1 | t2]     : tangent basis perp to n_hat (3x2)
#   K_rot             : 3x3 angular block of the 5x5 effective mass
#                       = I_1^{-1} + I_2^{-1}
#   K_trans           : 2x2 perpendicular-translation block
#                       = T^T A T  where A is the standard ball-socket
#                                  effective mass at the anchor
#   K_rot_trans       : 3x2 cross block, derived in prepare below
#   S = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans  : 2x2 Schur


@wp.func
def prismatic_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the prismatic joint.

    Recomputes world-space lever arms + slide axis + tangent basis,
    builds the three blocks of the 5x5 effective mass, factors them
    via Schur complement, computes the position-error bias for both
    blocks, and warm-starts the bodies with the cached accumulated
    impulses.

    See module docstring for the derivation; see
    :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la_b1 = read_vec3(constraints, base_offset + _OFF_LA_B1, cid)
    la_b2 = read_vec3(constraints, base_offset + _OFF_LA_B2, cid)
    axis_local = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL_B1, cid)
    q0 = read_quat(constraints, base_offset + _OFF_Q0, cid)
    acc_rot_in = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_in = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)

    # World-frame lever arms and anchor positions.
    r1 = wp.quat_rotate(q1, la_b1)
    r2 = wp.quat_rotate(q2, la_b2)
    p1 = pos1 + r1
    p2 = pos2 + r2
    write_vec3(constraints, base_offset + _OFF_R1, cid, r1)
    write_vec3(constraints, base_offset + _OFF_R2, cid, r2)

    # World-frame slide axis (rides body 1) and right-handed tangent
    # basis perpendicular to it.
    n_hat = wp.quat_rotate(q1, axis_local)
    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    # Soft-constraint coefficients (Box2D v3 / Bepu / Nordby) for both
    # blocks, derived from the current substep dt = 1 / idt so the
    # joint behaviour stays time-step-independent.
    dt = 1.0 / idt
    hertz_ang = read_float(constraints, base_offset + _OFF_HERTZ_ANG, cid)
    damping_ang = read_float(constraints, base_offset + _OFF_DAMPING_ANG, cid)
    hertz_lin = read_float(constraints, base_offset + _OFF_HERTZ_LIN, cid)
    damping_lin = read_float(constraints, base_offset + _OFF_DAMPING_LIN, cid)
    bias_rate_ang, mass_coeff_ang, impulse_coeff_ang = soft_constraint_coefficients(hertz_ang, damping_ang, dt)
    bias_rate_lin, mass_coeff_lin, impulse_coeff_lin = soft_constraint_coefficients(hertz_lin, damping_lin, dt)
    write_float(constraints, base_offset + _OFF_BIAS_RATE_ANG, cid, bias_rate_ang)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_ANG, cid, mass_coeff_ang)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_ANG, cid, impulse_coeff_ang)
    write_float(constraints, base_offset + _OFF_BIAS_RATE_LIN, cid, bias_rate_lin)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_LIN, cid, mass_coeff_lin)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIN, cid, impulse_coeff_lin)

    # ---- Angular block: K_rot = I_1^{-1} + I_2^{-1}, J_rot = [0, +I, 0, -I] ----
    # Linearised quaternion-error Jacobian (same construction as
    # :func:`hinge_angle_prepare_for_iteration_at`). For the prismatic we
    # *don't* project onto a single hinge axis -- we keep all three rows.
    q1_inv = wp.quat_inverse(q1)
    quat_e = q0 * q1_inv * q2  # error quaternion (identity at rest).
    # Pick the short-rotation branch.
    if quat_e[3] < 0.0:
        sign = -1.0
    else:
        sign = 1.0
    err_vec = wp.vec3f(quat_e[0] * sign, quat_e[1] * sign, quat_e[2] * sign)

    # m0 = -1/2 * QMatrix.ProjectMultiplyLeftRight(q0 * q1^*, q2). Same
    # closed form as the hinge-angle code; the 1/2 accounts for the
    # unit-quat half-angle convention.
    qq = q0 * q1_inv
    m0 = qmatrix_project_multiply_left_right(qq, q2) * (-0.5 * sign)

    # The effective Jacobian for the angular block in the world frame
    # is m0^T (rows are world-frame angular-velocity coefficients of
    # the per-axis quaternion error). We then take the inertia-weighted
    # quadratic form to get the 3x3 K_rot effective mass.
    # NB: in the limit of small relative rotation m0 -> -1/2 * I, which
    # is exactly the unit-quaternion Jacobian, and K_rot collapses to
    # 1/4 * (I_1^{-1} + I_2^{-1}); the 1/4 is absorbed by the "twice"
    # in the velocity-error projection below so the units come out
    # right.
    inv_inertia_sum = inv_inertia1 + inv_inertia2
    j_rot = m0  # 3x3, rows = world-axis Jacobian.
    k_rot = wp.transpose(j_rot) @ (inv_inertia_sum @ j_rot)
    k_rot_inv = wp.inverse(k_rot)

    # Angular bias = bias_rate * positional (quaternion) error.
    bias_rot = err_vec * bias_rate_ang

    # ---- Linear block: K_trans = T^T A T where A is the ball-socket A ----
    eye3 = wp.identity(3, dtype=wp.float32)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)
    a_mat = inv_mass1 * eye3
    a_mat = a_mat + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    a_mat = a_mat + inv_mass2 * eye3
    a_mat = a_mat + cr2 @ (inv_inertia2 @ wp.transpose(cr2))

    # K_trans is the 2x2 ``T^T A T``. Rather than form a real 2x2 right
    # away we evaluate the four scalar entries explicitly so we can
    # carry them around as either ``mat22f`` (for inverse) or as part
    # of larger Schur expressions below.
    a_t1 = a_mat @ t1
    a_t2 = a_mat @ t2
    k_trans_00 = wp.dot(t1, a_t1)
    k_trans_01 = wp.dot(t1, a_t2)
    k_trans_10 = wp.dot(t2, a_t1)
    k_trans_11 = wp.dot(t2, a_t2)

    # ---- Cross block: K_rot_trans (3x2). The translational impulse
    # ``T lambda_lin`` produces an angular impulse via the lever arms;
    # symmetrically, the angular impulse rotates the bodies, displacing
    # the anchor laterally.
    #
    # Working it out:
    #   dot(jv_rot, J_rot^T) -> contributes I_1^{-1} J_rot^T to omega_1
    #   that omega_1 then contributes -[r_1]_x to v at the anchor
    #   project that v onto T to get the cross-block.
    # Algebraically:
    #   K_rot_trans = J_rot^T (I_1^{-1} (-cr1^T) + I_2^{-1} (cr2^T)) T
    #               (signs follow from J_rot = [0, +I, 0, -I] and the
    #                ball-socket linear Jacobian sign convention used
    #                throughout this codebase).
    #
    # NB: ``j_rot = m0`` carries the half-angle factor, so the cross
    # block here picks up the same factor automatically; we don't
    # multiply or divide by 2 anywhere.
    j_rot_t = wp.transpose(j_rot)
    cross_lin_to_rot_b1 = inv_inertia1 @ (wp.transpose(cr1) @ t1)  # 3-vec
    cross_lin_to_rot_b2 = inv_inertia2 @ (wp.transpose(cr2) @ t1)
    krt_col0 = j_rot_t @ (-cross_lin_to_rot_b1 + cross_lin_to_rot_b2)
    cross_lin_to_rot_b1 = inv_inertia1 @ (wp.transpose(cr1) @ t2)
    cross_lin_to_rot_b2 = inv_inertia2 @ (wp.transpose(cr2) @ t2)
    krt_col1 = j_rot_t @ (-cross_lin_to_rot_b1 + cross_lin_to_rot_b2)
    # K_rot_trans is 3 rows x 2 cols; we pack it column-by-column into
    # a mat33 with the third column zero so we can reuse the existing
    # mat33 read/write helpers.
    k_rot_trans = wp.mat33f(
        krt_col0[0],
        krt_col1[0],
        0.0,
        krt_col0[1],
        krt_col1[1],
        0.0,
        krt_col0[2],
        krt_col1[2],
        0.0,
    )

    # ---- Schur complement: S = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans
    # Compute kt_ki = K_rot_trans^T K_rot^{-1} (2x3) packed in a mat33.
    kt_ki_full = wp.transpose(k_rot_trans) @ k_rot_inv  # 3x3 with last row zero
    kt_ki = wp.mat33f(
        kt_ki_full[0, 0],
        kt_ki_full[0, 1],
        kt_ki_full[0, 2],
        kt_ki_full[1, 0],
        kt_ki_full[1, 1],
        kt_ki_full[1, 2],
        0.0,
        0.0,
        0.0,
    )
    # S = K_trans - kt_ki @ K_rot_trans  (2x2).
    kt_ki_krt = kt_ki_full @ k_rot_trans
    s00 = k_trans_00 - kt_ki_krt[0, 0]
    s01 = k_trans_01 - kt_ki_krt[0, 1]
    s10 = k_trans_10 - kt_ki_krt[1, 0]
    s11 = k_trans_11 - kt_ki_krt[1, 1]
    s22 = wp.mat22f(s00, s01, s10, s11)
    s22_inv = wp.inverse(s22)
    s_inv_packed = wp.mat33f(
        s22_inv[0, 0],
        s22_inv[0, 1],
        0.0,
        s22_inv[1, 0],
        s22_inv[1, 1],
        0.0,
        0.0,
        0.0,
        0.0,
    )

    write_mat33(constraints, base_offset + _OFF_J_ROT, cid, j_rot)
    write_mat33(constraints, base_offset + _OFF_K_ROT_INV, cid, k_rot_inv)
    write_mat33(constraints, base_offset + _OFF_KT_KI, cid, kt_ki)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv_packed)
    write_vec3(constraints, base_offset + _OFF_BIAS_ROT, cid, bias_rot)

    # Linear bias = T^T (P2 - P1) * bias_rate_linear, packed (b_t1, b_t2, 0).
    drift = p2 - p1
    bias_lin_t1 = wp.dot(t1, drift) * bias_rate_lin
    bias_lin_t2 = wp.dot(t2, drift) * bias_rate_lin
    bias_lin = wp.vec3f(bias_lin_t1, bias_lin_t2, 0.0)
    write_vec3(constraints, base_offset + _OFF_BIAS_LIN, cid, bias_lin)

    # Warm start: re-apply the cached accumulated impulses straight to
    # the bodies. ``acc_rot_in`` is the cached angular impulse expressed
    # in the *quaternion-axis* basis (same convention as
    # :mod:`constraint_hinge_angle`'s ``accumulated_impulse``); the
    # corresponding world-frame angular impulse on body 1 is
    # ``J^T acc_rot``. ``acc_lin_in`` is a world-frame linear impulse
    # (= T * lambda_lin from the previous solve), already reprojected
    # into world so we don't have to track tangent-basis rotation
    # between substeps.
    #
    # Per-block impulse application formulas (signs match iterate). The
    # angular Jacobian convention is the one used by
    # :mod:`constraint_hinge_angle`: ``jv = J^T (w1 - w2)`` for the
    # velocity error, but the reverse direction uses ``J`` (not
    # ``J^T``) to convert the constraint-space multiplier back into a
    # world-frame angular impulse. Mathematically ``eff = J^T M^{-1} J``
    # so ``J^T (M^{-1} J lam) = eff lam``, i.e. ``M^{-1} J lam`` is the
    # impulse-induced velocity change consistent with the velocity-error
    # projection above.
    #   linear acc -> v1 -= m1^{-1} acc_lin, v2 += m2^{-1} acc_lin
    #              -> w1 -= I1^{-1} (cr1 acc_lin), w2 += I2^{-1} (cr2 acc_lin)
    #   angular acc -> w1 += I1^{-1} (J acc_rot)
    #               -> w2 -= I2^{-1} (J acc_rot)
    j_rot_acc_rot = j_rot @ acc_rot_in

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc_lin_in
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1 @ acc_lin_in) + inv_inertia1 @ j_rot_acc_rot

    velocity2 = bodies.velocity[b2] + inv_mass2 * acc_lin_in
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr2 @ acc_lin_in) - inv_inertia2 @ j_rot_acc_rot

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def prismatic_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Composable PGS iteration step for the prismatic joint.

    Solves the 5x5 block-diagonal-with-cross-coupling system via the
    Schur complement (3x3 + 2x2 inverses), applies the resulting
    impulses to both bodies. See :func:`ball_socket_iterate_at` for
    the ``base_offset`` / ``body_pair`` contract.

    ``use_bias`` toggles the rigid-lock positional drift bias
    (Box2D v3 TGS-soft ``useBias`` flag): ``True`` during the main
    solve pass, ``False`` during the relax pass so the lock's
    ``Jv=0`` update does not re-inject position-error velocity.
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

    r1 = read_vec3(constraints, base_offset + _OFF_R1, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    j_rot = read_mat33(constraints, base_offset + _OFF_J_ROT, cid)
    k_rot_inv = read_mat33(constraints, base_offset + _OFF_K_ROT_INV, cid)
    kt_ki = read_mat33(constraints, base_offset + _OFF_KT_KI, cid)
    s_inv_packed = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
    if use_bias:
        bias_rot = read_vec3(constraints, base_offset + _OFF_BIAS_ROT, cid)
        bias_lin = read_vec3(constraints, base_offset + _OFF_BIAS_LIN, cid)
    else:
        bias_rot = wp.vec3f(0.0, 0.0, 0.0)
        bias_lin = wp.vec3f(0.0, 0.0, 0.0)

    mass_coeff_ang = read_float(constraints, base_offset + _OFF_MASS_COEFF_ANG, cid)
    impulse_coeff_ang = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_ANG, cid)
    mass_coeff_lin = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIN, cid)
    impulse_coeff_lin = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIN, cid)

    acc_rot_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)
    # Anchor-2 (translational) accumulated impulse is stored in world
    # frame; project onto the current tangent basis so the soft-PGS
    # update operates in the same coordinates as the solve.
    acc_lin_t1 = wp.dot(t1, acc_lin_world)
    acc_lin_t2 = wp.dot(t2, acc_lin_world)

    # ---- Velocity-error vectors ----
    # Angular block: jv_rot = J_rot^T (w1 - w2), 3-vec in the
    # quaternion-axis basis. Same convention as
    # :func:`hinge_angle_iterate_at`.
    j_rot_t = wp.transpose(j_rot)
    jv_rot = j_rot_t @ (angular_velocity1 - angular_velocity2)

    # Linear block: jv_lin = T^T (-v1 + cr1 w1 + v2 - cr2 w2), 2-vec.
    jv_lin_world = -velocity1 + cr1 @ angular_velocity1 + velocity2 - cr2 @ angular_velocity2
    jv_lin_t1 = wp.dot(t1, jv_lin_world)
    jv_lin_t2 = wp.dot(t2, jv_lin_world)

    # ---- Build right-hand sides (with bias) ----
    rhs_rot = jv_rot + bias_rot
    rhs_lin = wp.vec2f(jv_lin_t1 + bias_lin[0], jv_lin_t2 + bias_lin[1])

    # ---- Schur-complement solve ----
    # lambda_lin = -S^{-1} (rhs_lin - kt_ki @ rhs_rot)
    # lambda_rot = -K_rot_inv (rhs_rot + K_rot_trans @ lambda_lin)
    kt_ki_rhs_rot_full = kt_ki @ rhs_rot
    kt_ki_rhs_rot = wp.vec2f(kt_ki_rhs_rot_full[0], kt_ki_rhs_rot_full[1])

    s_inv_22 = wp.mat22f(
        s_inv_packed[0, 0],
        s_inv_packed[0, 1],
        s_inv_packed[1, 0],
        s_inv_packed[1, 1],
    )
    lam_lin_full = -(s_inv_22 @ (rhs_lin - kt_ki_rhs_rot))

    # Re-expand lam_lin from tangent coordinates to world 3-vector for
    # the impulse-application path and the back-substitution into the
    # angular row.
    lam_lin_world = lam_lin_full[0] * t1 + lam_lin_full[1] * t2

    # Back-substitute: the linear impulse ``lam_lin_world`` perturbs the
    # body angular velocities by
    #   delta_w1 = -I1^{-1} (cr1 lam_lin_world)
    #   delta_w2 = +I2^{-1} (cr2 lam_lin_world)
    # (signs match the impulse-application formulas below). Projecting
    # the resulting (delta_w1 - delta_w2) through ``J^T`` gives the
    # additive change to the angular RHS in the quaternion-axis basis.
    # The Schur back-sub then reads
    #   lam_rot = -K_rot_inv (rhs_rot + J^T (delta_w1 - delta_w2)).
    delta_w1_from_lin = -inv_inertia1 @ (cr1 @ lam_lin_world)
    delta_w2_from_lin = +inv_inertia2 @ (cr2 @ lam_lin_world)
    delta_rhs_rot_from_lin = j_rot_t @ (delta_w1_from_lin - delta_w2_from_lin)
    lam_rot = -(k_rot_inv @ (rhs_rot + delta_rhs_rot_from_lin))

    # ---- Box2D v3 / Bepu soft-constraint scaling ----
    # Each block scales independently:
    #   lam_block = mass_coeff * lam_block_unsoft - impulse_coeff * acc_block.
    # Setting (mass_coeff, impulse_coeff) = (1, 0) recovers a rigid
    # plain-PGS update.
    lam_rot_soft = mass_coeff_ang * lam_rot - impulse_coeff_ang * acc_rot_world
    lam_lin_t1_soft = mass_coeff_lin * lam_lin_full[0] - impulse_coeff_lin * acc_lin_t1
    lam_lin_t2_soft = mass_coeff_lin * lam_lin_full[1] - impulse_coeff_lin * acc_lin_t2
    lam_lin_world_soft = lam_lin_t1_soft * t1 + lam_lin_t2_soft * t2

    # ---- Apply impulses to bodies ----
    # World-frame angular impulse on body 1 is ``J lam_rot`` (NOT
    # ``J^T``); body 2 gets the opposite sign. The asymmetry vs the
    # velocity-error projection (``jv = J^T (w1-w2)``) is by design --
    # see the warm-start application in
    # :func:`prismatic_prepare_for_iteration_at` for the eff-consistency
    # argument. Linear impulse goes straight to the bodies (already in
    # world frame).
    j_rot_lam_rot = j_rot @ lam_rot_soft
    bodies.velocity[b1] = velocity1 - inv_mass1 * lam_lin_world_soft
    bodies.angular_velocity[b1] = (
        angular_velocity1 - inv_inertia1 @ (cr1 @ lam_lin_world_soft) + inv_inertia1 @ j_rot_lam_rot
    )

    bodies.velocity[b2] = velocity2 + inv_mass2 * lam_lin_world_soft
    bodies.angular_velocity[b2] = (
        angular_velocity2 + inv_inertia2 @ (cr2 @ lam_lin_world_soft) - inv_inertia2 @ j_rot_lam_rot
    )

    # ---- Accumulate impulses ----
    # ``acc_rot`` stays in the quaternion-axis basis (matches the
    # bias_rot/jv_rot basis used here); ``acc_lin`` is in world frame.
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid, acc_rot_world + lam_rot_soft)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid, acc_lin_world + lam_lin_world_soft)


@wp.func
def prismatic_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2; see :func:`prismatic_world_wrench`."""
    acc_rot = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    j_rot = read_mat33(constraints, base_offset + _OFF_J_ROT, cid)
    # Linear block: total constraint force on body 2 = lam_lin / dt.
    force = acc_lin * idt
    # Torque on body 2: world-frame angular block impulse (J acc_rot)
    # / dt + moment of the linear impulse about body 2's COM. Sign
    # conventions match the iterate path: body 2 receives +lam_lin and
    # -J acc_rot for the angular part.
    angular_impulse_world = j_rot @ acc_rot
    torque = wp.cross(r2, force) - angular_impulse_world * idt
    return force, torque


@wp.func
def prismatic_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct prepare entry: reads body indices from the column header
    and forwards to :func:`prismatic_prepare_for_iteration_at` with
    ``base_offset = 0``."""
    b1 = prismatic_get_body1(constraints, cid)
    b2 = prismatic_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    prismatic_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def prismatic_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Direct iterate entry; see :func:`prismatic_iterate_at`."""
    b1 = prismatic_get_body1(constraints, cid)
    b2 = prismatic_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    prismatic_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def prismatic_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body 2.

    Force is the linear constraint impulse (perpendicular-translation
    block accumulated impulse) divided by the substep ``dt``
    (``idt = 1 / substep_dt``); torque is that force's moment about
    body 2's COM plus the angular block accumulated impulse / dt.
    """
    return prismatic_world_wrench_at(constraints, cid, 0, idt)


@wp.func
def prismatic_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
) -> wp.spatial_vector:
    """Position-level constraint residual for a prismatic (slider) joint.

    Mirrors the prepare math: the angular residual is the
    short-rotation branch of ``quat_e.xyz`` (where ``quat_e = q0 *
    q1^* * q2``); the linear residual is the perpendicular component
    of the world-space anchor separation ``p2 - p1`` projected onto
    the current tangent basis ``(t1, t2)`` (the axial component
    along the slide axis is the free DoF and is intentionally
    dropped).

    Output: :class:`wp.spatial_vector` with ``spatial_top`` = linear
    residual reconstructed in world frame as
    ``t1 * dot(t1, p2 - p1) + t2 * dot(t2, p2 - p1)`` [m], and
    ``spatial_bottom`` = angular residual (``quat_e.xyz`` sign-fixed)
    [rad, half-sin approximation].
    """
    b1 = body_pair.b1
    b2 = body_pair.b2
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    la_b1 = read_vec3(constraints, base_offset + _OFF_LA_B1, cid)
    la_b2 = read_vec3(constraints, base_offset + _OFF_LA_B2, cid)
    axis_local = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL_B1, cid)
    q0 = read_quat(constraints, base_offset + _OFF_Q0, cid)

    r1 = wp.quat_rotate(q1, la_b1)
    r2 = wp.quat_rotate(q2, la_b2)
    p1 = pos1 + r1
    p2 = pos2 + r2

    n_hat = wp.quat_rotate(q1, axis_local)
    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    drift = p2 - p1
    lin = wp.dot(t1, drift) * t1 + wp.dot(t2, drift) * t2

    q1_inv = wp.quat_inverse(q1)
    quat_e = q0 * q1_inv * q2
    sign = wp.float32(1.0)
    if quat_e[3] < 0.0:
        sign = wp.float32(-1.0)
    ang = wp.vec3f(quat_e[0] * sign, quat_e[1] * sign, quat_e[2] * sign)
    return wp.spatial_vector(lin, ang)


@wp.func
def prismatic_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`prismatic_world_error_at`."""
    b1 = prismatic_get_body1(constraints, cid)
    b2 = prismatic_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return prismatic_world_error_at(constraints, cid, 0, bodies, body_pair)
