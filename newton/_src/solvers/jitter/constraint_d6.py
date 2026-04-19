# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""6-DoF "D6" generalised joint constraint.

The D6 is the most general two-body joint in this solver: it owns *all
six* relative DoF (3 angular + 3 linear) of body 2 with respect to body 1
and lets each axis independently behave as one of:

* **Rigid lock**       -- ``hertz=0``, ``max_force=inf`` (the default)
* **Soft lock**        -- ``hertz>0``, ``target_*=0``, ``max_force=inf``
* **Position drive**   -- ``hertz>0``, ``target_position!=0``, ``max_force=cap``
* **Velocity drive**   -- ``hertz>0``, ``target_velocity!=0``, ``max_force=cap``
* **Free**             -- ``hertz=0`` AND ``max_force=0`` (impulse clamped to 0)

A single ``add_d6(...)`` call therefore covers fixed welds, springs,
position/velocity-PD drives (PhysX D6 style), and partially-free
joints (sliders, hinges, point-on-plane, ...) without needing different
constraint classes. The whole 6-vector of impulses is solved with one
6x6 Schur complement per PGS iteration -- no row-by-row sweeps -- and
*then* per-axis force limits are applied to the accumulated impulse,
matching the convention used by :mod:`constraint_angular_motor` for
its 1-DoF case (and by every production-grade D6 / 6Dof2 joint in
PhysX, Bullet, and Bepu).

Implicit-PD drives via Box2D / Bepu / Nordby
--------------------------------------------
The implicit-PD drives are expressed in the same Box2D-v3 /
Bepu / Nordby ``(hertz, damping_ratio)`` parameterisation that the
existing constraints already use (see
:func:`soft_constraint_coefficients`). This is mathematically identical
to the implicit-Euler PD update with ``kp = m * (2 pi hertz)^2`` and
``kd = m * 2 (2 pi hertz) damping_ratio``, but it stays unconditionally
stable for any ``hertz, damping_ratio >= 0`` and any ``dt`` without the
caller having to know body masses or inertias up front.

Locking the implementation to this plumbing also means each axis
collapses to a single set of cached scalars
``(bias_rate, mass_coeff, impulse_coeff, max_lambda, target_vel)``
inside the kernel; the per-axis "mode" is *implicit* in those values
rather than carried as a runtime branch.

Body-1-local axis convention
----------------------------
The 3 angular axes and 3 linear axes are the body-1-local basis
``(e_x, e_y, e_z)``. After ``q1`` rotates them, the world-frame
linear axes are simply the *columns of* ``R_1``, and the world-frame
angular Jacobian rows fall out of the same ``QMatrix`` machinery that
:mod:`constraint_hinge_angle` and :mod:`constraint_prismatic` use --
no separate axis vectors stored in the schema.

This matches PhysX D6 / Bullet 6Dof2 / Bepu's "frame A is the local
reference" convention. ``target_position_*`` / ``target_velocity_*`` /
``max_force_*`` are all interpreted in this body-1-local frame.

6x6 Schur-complement solve (avoiding a 6x6 inverse)
---------------------------------------------------
Warp's ``wp.inverse`` overloads only cover 2x2 / 3x3 / 4x4. The 6x6
effective mass

::

    K = [ K_rot          K_rot_trans ]   in R^{6x6}
        [ K_rot_trans^T  K_trans     ]

with ``K_rot in R^{3x3}``, ``K_trans in R^{3x3}``, ``K_rot_trans in
R^{3x3}`` is solved by block-elimination using two 3x3 inverses:

::

    S       = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans   # 3x3 Schur complement
    lambda_lin = -S^{-1}     ( b_lin - K_rot_trans^T K_rot^{-1} b_rot )
    lambda_rot = -K_rot^{-1} ( b_rot + K_rot_trans lambda_lin )

We cache ``K_rot_inv`` (3x3), ``S_inv`` (3x3), and
``Kt_Ki = K_rot_trans^T K_rot^{-1}`` (3x3) in the constraint column so
``iterate`` is just three small mat-vecs per block plus the impulse
application -- no per-iter inverses.

This is structurally the same as :mod:`constraint_prismatic`'s
Schur factorisation, with the prismatic's 2x2 linear block promoted
to 3x3 (one extra scalar inverse, no new matrix sizes).

Position-target packing
-----------------------
Position targets ``target_position_ang in R^3`` (rotation vector,
body-1-local) and ``target_position_lin in R^3`` (offset, body-1-local)
are folded *into the rest pose* at init time:

* ``q0_eff = q0 * exp(0.5 * target_position_ang)``
* ``local_anchor_b1_eff = local_anchor_b1 + target_position_lin``

so the runtime kernel measures positional error against ``(q0_eff,
local_anchor_b1_eff)`` and the standard "drive towards rest pose"
formula then naturally drives the bodies to the user-specified target.
This avoids any runtime branching on "is there a position target" and
collapses position drives back into "soft lock at the user-specified
pose" -- the kernel sees no special case at all.

Per-axis force/torque limits
----------------------------
Per-axis ``max_force_*`` (linear, [N]) and ``max_force_ang`` (angular,
[N*m]) become per-substep impulse caps ``max_lambda = max_force * dt``.
After the 6x6 direct solve produces ``(lambda_rot, lambda_lin_world)``
we project the linear impulse into the body-1-local frame, soft-scale
each block independently, then clip the *accumulated* impulse to
``[-max_lambda, +max_lambda]`` per axis (Box2D / motor pattern). The
clipped delta is what gets applied to the bodies; convergence to the
constrained solution happens across PGS iterations the same way the
1-DoF angular motor's clamping converges.

Setting ``max_force = 0`` on an axis therefore yields a *free* axis
(impulse pinned to zero on every iteration) regardless of ``hertz``;
this is the recommended way to disable an axis individually.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_D6,
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
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import qmatrix_project_multiply_left_right

__all__ = [
    "D6_DWORDS",
    "D6Data",
    "d6_get_accumulated_impulse_lin",
    "d6_get_accumulated_impulse_rot",
    "d6_get_bias_lin",
    "d6_get_bias_rate_ang",
    "d6_get_bias_rate_lin",
    "d6_get_bias_rot",
    "d6_get_body1",
    "d6_get_body2",
    "d6_get_damping_ang",
    "d6_get_damping_lin",
    "d6_get_hertz_ang",
    "d6_get_hertz_lin",
    "d6_get_impulse_coeff_ang",
    "d6_get_impulse_coeff_lin",
    "d6_get_j_rot",
    "d6_get_k_rot_inv",
    "d6_get_kt_ki",
    "d6_get_local_anchor_b1",
    "d6_get_local_anchor_b2",
    "d6_get_mass_coeff_ang",
    "d6_get_mass_coeff_lin",
    "d6_get_max_force_ang",
    "d6_get_max_force_lin",
    "d6_get_max_lambda_ang",
    "d6_get_max_lambda_lin",
    "d6_get_q0",
    "d6_get_r1",
    "d6_get_r2",
    "d6_get_r1_basis",
    "d6_get_s_inv",
    "d6_get_target_vel_ang_local",
    "d6_get_target_vel_lin_local",
    "d6_initialize_kernel",
    "d6_iterate",
    "d6_iterate_at",
    "d6_prepare_for_iteration",
    "d6_prepare_for_iteration_at",
    "d6_set_accumulated_impulse_lin",
    "d6_set_accumulated_impulse_rot",
    "d6_set_bias_lin",
    "d6_set_bias_rate_ang",
    "d6_set_bias_rate_lin",
    "d6_set_bias_rot",
    "d6_set_body1",
    "d6_set_body2",
    "d6_set_damping_ang",
    "d6_set_damping_lin",
    "d6_set_hertz_ang",
    "d6_set_hertz_lin",
    "d6_set_impulse_coeff_ang",
    "d6_set_impulse_coeff_lin",
    "d6_set_j_rot",
    "d6_set_k_rot_inv",
    "d6_set_kt_ki",
    "d6_set_local_anchor_b1",
    "d6_set_local_anchor_b2",
    "d6_set_mass_coeff_ang",
    "d6_set_mass_coeff_lin",
    "d6_set_max_force_ang",
    "d6_set_max_force_lin",
    "d6_set_max_lambda_ang",
    "d6_set_max_lambda_lin",
    "d6_set_q0",
    "d6_set_r1",
    "d6_set_r2",
    "d6_set_r1_basis",
    "d6_set_s_inv",
    "d6_set_target_vel_ang_local",
    "d6_set_target_vel_lin_local",
    "d6_world_wrench",
    "d6_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class D6Data:
    """Per-constraint dword-layout schema for a 6-DoF generalised joint.

    *Schema only* (same conventions as :class:`PrismaticData`). Field
    order fixes dword offsets; runtime kernels read/write fields out of
    the shared :class:`ConstraintContainer` via the typed accessors
    below.

    The first three fields are the global constraint header
    (``constraint_type``, ``body1``, ``body2`` at dwords 0/1/2),
    enforced by :func:`assert_constraint_header`.
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Anchor in each body's local frame. Both anchors are derived from
    # the user's world-space anchor at init time and refer to the same
    # world point at rest by construction; the *linear position target*
    # (body-1-local) is folded into ``local_anchor_b1`` so that the
    # "drive to rest pose" math automatically tracks the user-specified
    # offset.
    local_anchor_b1: wp.vec3f
    local_anchor_b2: wp.vec3f

    # Rest-pose relative orientation ``q0 = q2^* * q1`` snapshotted at
    # init time, with the user-specified angular position target folded
    # in (``q0_eff = q0 * exp(0.5 * target_position_ang_local)``). The
    # angular error is the rotation that maps current ``q2^* q1`` back
    # to ``q0_eff``.
    q0: wp.quatf

    # Velocity setpoints (body-1-local frame). For a velocity-tracking
    # axis the runtime adds these to the velocity-error vector. Setting
    # all six to zero recovers a pure positional drive / lock.
    target_vel_ang_local: wp.vec3f
    target_vel_lin_local: wp.vec3f

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby) per
    # axis. Six independent (hertz, damping_ratio) pairs -- three for
    # the angular block, three for the linear block. Setting an axis's
    # ``hertz`` to <= 0 makes that axis a perfectly rigid plain-PGS
    # update (``mass_coeff = 1, impulse_coeff = 0, bias_rate = 0``).
    hertz_ang: wp.vec3f
    damping_ang: wp.vec3f
    hertz_lin: wp.vec3f
    damping_lin: wp.vec3f

    # User-facing per-axis force / torque caps. ``max_force_lin`` is in
    # newtons (per linear axis); ``max_force_ang`` is in newton*metres
    # (per angular axis). The init kernel converts these to per-substep
    # impulse caps ``max_lambda = max_force * dt`` inside
    # ``prepare_for_iteration``. ``max_force = 0`` -> *free* axis (the
    # accumulated impulse is clamped to 0 every iteration), which is
    # how to disable an axis individually.
    max_force_ang: wp.vec3f
    max_force_lin: wp.vec3f

    # World-space lever arms from each body's COM to the (shared) anchor
    # point. ``r1`` may diverge from ``r2`` between substeps as the
    # bodies translate; that drift is exactly what the linear block
    # sees.
    r1: wp.vec3f
    r2: wp.vec3f

    # Cached per-substep soft-constraint coefficients (recomputed each
    # ``prepare_for_iteration`` from the current ``dt``). Per axis;
    # ``bias_rate`` multiplies the position error to give a velocity
    # bias, ``mass_coeff`` softens the unsoftened ``effective_mass``
    # contribution, ``impulse_coeff`` damps the accumulated-impulse
    # term ("softness leak"). See :func:`soft_constraint_coefficients`.
    bias_rate_ang: wp.vec3f
    mass_coeff_ang: wp.vec3f
    impulse_coeff_ang: wp.vec3f
    bias_rate_lin: wp.vec3f
    mass_coeff_lin: wp.vec3f
    impulse_coeff_lin: wp.vec3f

    # Cached per-substep impulse caps in the linear / angular blocks
    # respectively (per-axis, body-1-local). Computed from the user's
    # ``max_force`` once per substep and consumed by ``iterate`` to
    # clamp the accumulated impulse.
    max_lambda_ang: wp.vec3f
    max_lambda_lin: wp.vec3f

    # Cached velocity-error bias vectors (3-vec each). Both expressed in
    # the *body-1-local* frame (= the frame the position targets and
    # velocity targets are stated in); ``iterate`` projects velocity
    # errors into this frame too so everything composes.
    bias_rot: wp.vec3f
    bias_lin: wp.vec3f

    # Cached body-1 rotation matrix R_1 (= world axes of the linear
    # block). Same as ``wp.quat_to_matrix(q1)``; cached because both
    # ``iterate`` and ``world_wrench`` need it.
    r1_basis: wp.mat33f

    # Cached angular Jacobian (3x3, world frame). Same convention as
    # :mod:`constraint_hinge_angle` / :mod:`constraint_prismatic`: rows
    # are world-axis Jacobian rows, so ``J^T (w1 - w2)`` is the per-
    # quaternion-axis velocity error and ``J lam`` is the world-frame
    # angular impulse for an impulse ``lam`` expressed in the
    # quaternion-axis basis. Cached because the solve, the impulse
    # application, the warm-start, and the cross-block all need it
    # consistently.
    j_rot: wp.mat33f

    # Cached effective-mass blocks for the Schur-complement solve.
    # ``k_rot_inv`` is the 3x3 inverse of the angular block.
    # ``kt_ki = K_rot_trans^T K_rot^{-1}`` (3x3, in body-1-local linear
    # axes for rows and quaternion-axis basis for cols, just like the
    # prismatic). ``s_inv`` is the 3x3 inverse of the Schur complement
    # ``S = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans``.
    k_rot_inv: wp.mat33f
    kt_ki: wp.mat33f
    s_inv: wp.mat33f

    # Accumulated PGS impulses split per block.
    # ``accumulated_impulse_rot`` is the angular impulse expressed in
    # the *quaternion-axis basis* (same convention as
    # :mod:`constraint_prismatic` / :mod:`constraint_hinge_angle`).
    # ``accumulated_impulse_lin`` is the linear impulse expressed in
    # the *body-1-local* frame -- it rides body 1 rigidly so the warm
    # start stays valid as body 1 rotates between substeps without
    # needing an extra reprojection. ``world_wrench`` rotates it into
    # world for reporting.
    accumulated_impulse_rot: wp.vec3f
    accumulated_impulse_lin: wp.vec3f


assert_constraint_header(D6Data)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(D6Data, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(D6Data, "body2"))
_OFF_LA_B1 = wp.constant(dword_offset_of(D6Data, "local_anchor_b1"))
_OFF_LA_B2 = wp.constant(dword_offset_of(D6Data, "local_anchor_b2"))
_OFF_Q0 = wp.constant(dword_offset_of(D6Data, "q0"))
_OFF_TARGET_VEL_ANG = wp.constant(dword_offset_of(D6Data, "target_vel_ang_local"))
_OFF_TARGET_VEL_LIN = wp.constant(dword_offset_of(D6Data, "target_vel_lin_local"))
_OFF_HERTZ_ANG = wp.constant(dword_offset_of(D6Data, "hertz_ang"))
_OFF_DAMPING_ANG = wp.constant(dword_offset_of(D6Data, "damping_ang"))
_OFF_HERTZ_LIN = wp.constant(dword_offset_of(D6Data, "hertz_lin"))
_OFF_DAMPING_LIN = wp.constant(dword_offset_of(D6Data, "damping_lin"))
_OFF_MAX_FORCE_ANG = wp.constant(dword_offset_of(D6Data, "max_force_ang"))
_OFF_MAX_FORCE_LIN = wp.constant(dword_offset_of(D6Data, "max_force_lin"))
_OFF_R1 = wp.constant(dword_offset_of(D6Data, "r1"))
_OFF_R2 = wp.constant(dword_offset_of(D6Data, "r2"))
_OFF_BIAS_RATE_ANG = wp.constant(dword_offset_of(D6Data, "bias_rate_ang"))
_OFF_MASS_COEFF_ANG = wp.constant(dword_offset_of(D6Data, "mass_coeff_ang"))
_OFF_IMPULSE_COEFF_ANG = wp.constant(dword_offset_of(D6Data, "impulse_coeff_ang"))
_OFF_BIAS_RATE_LIN = wp.constant(dword_offset_of(D6Data, "bias_rate_lin"))
_OFF_MASS_COEFF_LIN = wp.constant(dword_offset_of(D6Data, "mass_coeff_lin"))
_OFF_IMPULSE_COEFF_LIN = wp.constant(dword_offset_of(D6Data, "impulse_coeff_lin"))
_OFF_MAX_LAMBDA_ANG = wp.constant(dword_offset_of(D6Data, "max_lambda_ang"))
_OFF_MAX_LAMBDA_LIN = wp.constant(dword_offset_of(D6Data, "max_lambda_lin"))
_OFF_BIAS_ROT = wp.constant(dword_offset_of(D6Data, "bias_rot"))
_OFF_BIAS_LIN = wp.constant(dword_offset_of(D6Data, "bias_lin"))
_OFF_R1_BASIS = wp.constant(dword_offset_of(D6Data, "r1_basis"))
_OFF_J_ROT = wp.constant(dword_offset_of(D6Data, "j_rot"))
_OFF_K_ROT_INV = wp.constant(dword_offset_of(D6Data, "k_rot_inv"))
_OFF_KT_KI = wp.constant(dword_offset_of(D6Data, "kt_ki"))
_OFF_S_INV = wp.constant(dword_offset_of(D6Data, "s_inv"))
_OFF_ACC_IMP_ROT = wp.constant(dword_offset_of(D6Data, "accumulated_impulse_rot"))
_OFF_ACC_IMP_LIN = wp.constant(dword_offset_of(D6Data, "accumulated_impulse_lin"))

#: Total dword count of one D6 constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
D6_DWORDS: int = num_dwords(D6Data)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def d6_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def d6_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def d6_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def d6_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def d6_get_local_anchor_b1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA_B1, cid)


@wp.func
def d6_set_local_anchor_b1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA_B1, cid, v)


@wp.func
def d6_get_local_anchor_b2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LA_B2, cid)


@wp.func
def d6_set_local_anchor_b2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LA_B2, cid, v)


@wp.func
def d6_get_q0(c: ConstraintContainer, cid: wp.int32) -> wp.quatf:
    return read_quat(c, _OFF_Q0, cid)


@wp.func
def d6_set_q0(c: ConstraintContainer, cid: wp.int32, v: wp.quatf):
    write_quat(c, _OFF_Q0, cid, v)


@wp.func
def d6_get_target_vel_ang_local(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_TARGET_VEL_ANG, cid)


@wp.func
def d6_set_target_vel_ang_local(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_TARGET_VEL_ANG, cid, v)


@wp.func
def d6_get_target_vel_lin_local(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_TARGET_VEL_LIN, cid)


@wp.func
def d6_set_target_vel_lin_local(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_TARGET_VEL_LIN, cid, v)


@wp.func
def d6_get_hertz_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_HERTZ_ANG, cid)


@wp.func
def d6_set_hertz_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_HERTZ_ANG, cid, v)


@wp.func
def d6_get_damping_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_DAMPING_ANG, cid)


@wp.func
def d6_set_damping_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_DAMPING_ANG, cid, v)


@wp.func
def d6_get_hertz_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_HERTZ_LIN, cid)


@wp.func
def d6_set_hertz_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_HERTZ_LIN, cid, v)


@wp.func
def d6_get_damping_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_DAMPING_LIN, cid)


@wp.func
def d6_set_damping_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_DAMPING_LIN, cid, v)


@wp.func
def d6_get_max_force_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MAX_FORCE_ANG, cid)


@wp.func
def d6_set_max_force_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MAX_FORCE_ANG, cid, v)


@wp.func
def d6_get_max_force_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MAX_FORCE_LIN, cid)


@wp.func
def d6_set_max_force_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MAX_FORCE_LIN, cid, v)


@wp.func
def d6_get_r1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R1, cid)


@wp.func
def d6_set_r1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R1, cid, v)


@wp.func
def d6_get_r2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_R2, cid)


@wp.func
def d6_set_r2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_R2, cid, v)


@wp.func
def d6_get_bias_rate_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_RATE_ANG, cid)


@wp.func
def d6_set_bias_rate_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_RATE_ANG, cid, v)


@wp.func
def d6_get_mass_coeff_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MASS_COEFF_ANG, cid)


@wp.func
def d6_set_mass_coeff_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MASS_COEFF_ANG, cid, v)


@wp.func
def d6_get_impulse_coeff_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_IMPULSE_COEFF_ANG, cid)


@wp.func
def d6_set_impulse_coeff_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_IMPULSE_COEFF_ANG, cid, v)


@wp.func
def d6_get_bias_rate_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_RATE_LIN, cid)


@wp.func
def d6_set_bias_rate_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_RATE_LIN, cid, v)


@wp.func
def d6_get_mass_coeff_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MASS_COEFF_LIN, cid)


@wp.func
def d6_set_mass_coeff_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MASS_COEFF_LIN, cid, v)


@wp.func
def d6_get_impulse_coeff_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_IMPULSE_COEFF_LIN, cid)


@wp.func
def d6_set_impulse_coeff_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_IMPULSE_COEFF_LIN, cid, v)


@wp.func
def d6_get_max_lambda_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MAX_LAMBDA_ANG, cid)


@wp.func
def d6_set_max_lambda_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MAX_LAMBDA_ANG, cid, v)


@wp.func
def d6_get_max_lambda_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_MAX_LAMBDA_LIN, cid)


@wp.func
def d6_set_max_lambda_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_MAX_LAMBDA_LIN, cid, v)


@wp.func
def d6_get_bias_rot(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_ROT, cid)


@wp.func
def d6_set_bias_rot(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_ROT, cid, v)


@wp.func
def d6_get_bias_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS_LIN, cid)


@wp.func
def d6_set_bias_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS_LIN, cid, v)


@wp.func
def d6_get_r1_basis(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_R1_BASIS, cid)


@wp.func
def d6_set_r1_basis(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_R1_BASIS, cid, v)


@wp.func
def d6_get_j_rot(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_J_ROT, cid)


@wp.func
def d6_set_j_rot(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_J_ROT, cid, v)


@wp.func
def d6_get_k_rot_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_K_ROT_INV, cid)


@wp.func
def d6_set_k_rot_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_K_ROT_INV, cid, v)


@wp.func
def d6_get_kt_ki(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_KT_KI, cid)


@wp.func
def d6_set_kt_ki(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_KT_KI, cid, v)


@wp.func
def d6_get_s_inv(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_S_INV, cid)


@wp.func
def d6_set_s_inv(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_S_INV, cid, v)


@wp.func
def d6_get_accumulated_impulse_rot(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP_ROT, cid)


@wp.func
def d6_set_accumulated_impulse_rot(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP_ROT, cid, v)


@wp.func
def d6_get_accumulated_impulse_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACC_IMP_LIN, cid)


@wp.func
def d6_set_accumulated_impulse_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACC_IMP_LIN, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel)
# ---------------------------------------------------------------------------


@wp.kernel
def d6_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor: wp.array[wp.vec3f],
    target_position_ang_local: wp.array[wp.vec3f],
    target_position_lin_local: wp.array[wp.vec3f],
    target_velocity_ang_local: wp.array[wp.vec3f],
    target_velocity_lin_local: wp.array[wp.vec3f],
    hertz_ang: wp.array[wp.vec3f],
    damping_ang: wp.array[wp.vec3f],
    hertz_lin: wp.array[wp.vec3f],
    damping_lin: wp.array[wp.vec3f],
    max_force_ang: wp.array[wp.vec3f],
    max_force_lin: wp.array[wp.vec3f],
):
    """Pack one batch of D6 descriptors into ``constraints``.

    Snapshots the world-space anchor into both bodies' local frames,
    folds the user's *position targets* into the rest pose so the
    runtime kernel is unaware of them (it always drives towards the
    rest pose, which already includes any offsets), and stamps every
    cached field with a benign default.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; reads ``position`` /
            ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor point [num_in_batch] [m].
        target_position_ang_local: Per-axis angular position targets
            in the body-1-local frame [num_in_batch] [rad]. Interpreted
            as a *rotation vector* (axis-angle vector); the init kernel
            converts it to a quaternion ``q_target`` and folds it into
            the rest pose so ``q0_eff = q_target * q0``. Pass ``(0,0,0)``
            for "drive to rest pose".
        target_position_lin_local: Per-axis linear position targets in
            the body-1-local frame [num_in_batch] [m]. Folded into
            ``local_anchor_b1`` so the runtime drives body 2 to the
            offset point. Pass ``(0,0,0)`` for "anchor coincides".
        target_velocity_ang_local: Per-axis angular velocity setpoints
            [num_in_batch] [rad/s], in body-1-local frame.
        target_velocity_lin_local: Per-axis linear velocity setpoints
            [num_in_batch] [m/s], in body-1-local frame.
        hertz_ang: Per-axis angular soft-constraint Hz [num_in_batch].
            Each component <= 0 makes that axis rigid.
        damping_ang: Per-axis angular damping ratio [num_in_batch].
        hertz_lin: Per-axis linear soft-constraint Hz [num_in_batch].
        damping_lin: Per-axis linear damping ratio [num_in_batch].
        max_force_ang: Per-axis torque cap [num_in_batch] [N*m].
            ``0`` -> free axis (impulse pinned to 0 each iteration).
        max_force_lin: Per-axis force cap [num_in_batch] [N].
            ``0`` -> free axis.
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a_world = anchor[tid]
    tp_ang = target_position_ang_local[tid]
    tp_lin = target_position_lin_local[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # Anchor in each body's local frame -- same construction as the
    # prismatic. Both refer to the same world point at rest.
    la_b1_raw = wp.quat_rotate_inv(q1, a_world - pos1)
    la_b2 = wp.quat_rotate_inv(q2, a_world - pos2)

    # Linear position target in body-1-local frame -> just shift the
    # body-1 anchor by the target offset. The runtime then projects
    # the anchor drift onto body-1-local axes (via R_1) and the
    # "drive to rest pose" math automatically tracks the offset.
    la_b1 = la_b1_raw + tp_lin

    # Angular position target as a rotation vector (axis-angle). Build
    # ``q_target = exp(0.5 * tp_ang)``; fold it into the rest-pose
    # ``q0`` so ``q0_eff = q_target * q0``. Then the runtime's quat
    # error ``q0_eff * q1^* * q2`` is identity exactly when body 2 is
    # at the user-requested orientation relative to body 1.
    half_angle_vec = tp_ang * 0.5
    half_angle = wp.length(half_angle_vec)
    if half_angle < 1.0e-9:
        # Use the small-angle linearisation explicitly to keep the
        # tangent at exactly identity (no /0 from sin/half_angle).
        q_target = wp.quatf(half_angle_vec[0], half_angle_vec[1], half_angle_vec[2], 1.0)
        q_target = wp.normalize(q_target)
    else:
        s = wp.sin(half_angle) / half_angle
        q_target = wp.quatf(
            half_angle_vec[0] * s,
            half_angle_vec[1] * s,
            half_angle_vec[2] * s,
            wp.cos(half_angle),
        )

    q0_raw = wp.quat_inverse(q2) * q1
    q0 = q_target * q0_raw

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_D6)
    d6_set_body1(constraints, cid, b1)
    d6_set_body2(constraints, cid, b2)
    d6_set_local_anchor_b1(constraints, cid, la_b1)
    d6_set_local_anchor_b2(constraints, cid, la_b2)
    d6_set_q0(constraints, cid, q0)

    d6_set_target_vel_ang_local(constraints, cid, target_velocity_ang_local[tid])
    d6_set_target_vel_lin_local(constraints, cid, target_velocity_lin_local[tid])

    d6_set_hertz_ang(constraints, cid, hertz_ang[tid])
    d6_set_damping_ang(constraints, cid, damping_ang[tid])
    d6_set_hertz_lin(constraints, cid, hertz_lin[tid])
    d6_set_damping_lin(constraints, cid, damping_lin[tid])
    d6_set_max_force_ang(constraints, cid, max_force_ang[tid])
    d6_set_max_force_lin(constraints, cid, max_force_lin[tid])

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    d6_set_r1(constraints, cid, zero3)
    d6_set_r2(constraints, cid, zero3)
    d6_set_bias_rate_ang(constraints, cid, zero3)
    d6_set_mass_coeff_ang(constraints, cid, wp.vec3f(1.0, 1.0, 1.0))
    d6_set_impulse_coeff_ang(constraints, cid, zero3)
    d6_set_bias_rate_lin(constraints, cid, zero3)
    d6_set_mass_coeff_lin(constraints, cid, wp.vec3f(1.0, 1.0, 1.0))
    d6_set_impulse_coeff_lin(constraints, cid, zero3)
    d6_set_max_lambda_ang(constraints, cid, zero3)
    d6_set_max_lambda_lin(constraints, cid, zero3)
    d6_set_bias_rot(constraints, cid, zero3)
    d6_set_bias_lin(constraints, cid, zero3)
    d6_set_accumulated_impulse_rot(constraints, cid, zero3)
    d6_set_accumulated_impulse_lin(constraints, cid, zero3)

    eye = wp.identity(3, dtype=wp.float32)
    d6_set_r1_basis(constraints, cid, eye)
    d6_set_j_rot(constraints, cid, eye)
    d6_set_k_rot_inv(constraints, cid, eye)
    d6_set_kt_ki(constraints, cid, eye)
    d6_set_s_inv(constraints, cid, eye)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Symbol cheat-sheet (matches module-docstring derivation):
#   r1, r2            : world-space lever arms (body 1 / body 2 -> anchor)
#   cr1, cr2          : skew(r1), skew(r2)
#   R_1               : body 1's world rotation matrix (= columns are
#                       the world-frame body-1-local linear axes)
#   K_rot             : 3x3 angular block of the 6x6 effective mass
#                       = J^T (I_1^{-1} + I_2^{-1}) J   (J = m0)
#   K_trans           : 3x3 linear block (in body-1-local linear axes)
#                       = R_1^T A R_1
#                       where A = (m1^{-1} + m2^{-1}) I + cr1 I_1^{-1} cr1^T
#                                                       + cr2 I_2^{-1} cr2^T
#   K_rot_trans       : 3x3 cross block
#                       = J^T (-I_1^{-1} cr1^T + I_2^{-1} cr2^T) R_1
#   S = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans  : 3x3 Schur


@wp.func
def d6_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the D6 joint.

    Recomputes world-space lever arms + body-1 basis, builds the three
    blocks of the 6x6 effective mass, factors them via Schur
    complement, computes per-axis position-error biases for both blocks,
    materialises the per-axis soft-constraint coefficients and
    impulse caps, and warm-starts the bodies with the cached
    accumulated impulses.

    See module docstring for the derivation; see
    :func:`prismatic_prepare_for_iteration_at` for the
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
    q0 = read_quat(constraints, base_offset + _OFF_Q0, cid)
    acc_rot_in = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_local_in = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)

    hertz_ang = read_vec3(constraints, base_offset + _OFF_HERTZ_ANG, cid)
    damping_ang = read_vec3(constraints, base_offset + _OFF_DAMPING_ANG, cid)
    hertz_lin = read_vec3(constraints, base_offset + _OFF_HERTZ_LIN, cid)
    damping_lin = read_vec3(constraints, base_offset + _OFF_DAMPING_LIN, cid)
    max_force_ang = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_ANG, cid)
    max_force_lin = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_LIN, cid)

    # World-frame lever arms and anchor positions.
    r1 = wp.quat_rotate(q1, la_b1)
    r2 = wp.quat_rotate(q2, la_b2)
    p1 = pos1 + r1
    p2 = pos2 + r2
    write_vec3(constraints, base_offset + _OFF_R1, cid, r1)
    write_vec3(constraints, base_offset + _OFF_R2, cid, r2)

    # Body-1 rotation matrix R_1 -- the columns are the world-frame
    # axes of the linear block. Cached because every block of the
    # iterate path needs it (velocity projection, impulse application,
    # warm-start, world-wrench).
    r1_basis = wp.quat_to_matrix(q1)
    write_mat33(constraints, base_offset + _OFF_R1_BASIS, cid, r1_basis)

    # ---- Per-axis soft-constraint coefficients ----
    # We want to evaluate ``soft_constraint_coefficients`` on each of
    # the 6 axes independently because each axis carries its own
    # ``(hertz, damping_ratio)``. The helper is a plain ``wp.func`` so
    # we can call it three times per block; the result triples are
    # packed into vec3 fields for storage.
    dt = 1.0 / idt
    br_ang_x, mc_ang_x, ic_ang_x = soft_constraint_coefficients(hertz_ang[0], damping_ang[0], dt)
    br_ang_y, mc_ang_y, ic_ang_y = soft_constraint_coefficients(hertz_ang[1], damping_ang[1], dt)
    br_ang_z, mc_ang_z, ic_ang_z = soft_constraint_coefficients(hertz_ang[2], damping_ang[2], dt)
    br_lin_x, mc_lin_x, ic_lin_x = soft_constraint_coefficients(hertz_lin[0], damping_lin[0], dt)
    br_lin_y, mc_lin_y, ic_lin_y = soft_constraint_coefficients(hertz_lin[1], damping_lin[1], dt)
    br_lin_z, mc_lin_z, ic_lin_z = soft_constraint_coefficients(hertz_lin[2], damping_lin[2], dt)

    bias_rate_ang = wp.vec3f(br_ang_x, br_ang_y, br_ang_z)
    mass_coeff_ang = wp.vec3f(mc_ang_x, mc_ang_y, mc_ang_z)
    impulse_coeff_ang = wp.vec3f(ic_ang_x, ic_ang_y, ic_ang_z)
    bias_rate_lin = wp.vec3f(br_lin_x, br_lin_y, br_lin_z)
    mass_coeff_lin = wp.vec3f(mc_lin_x, mc_lin_y, mc_lin_z)
    impulse_coeff_lin = wp.vec3f(ic_lin_x, ic_lin_y, ic_lin_z)

    write_vec3(constraints, base_offset + _OFF_BIAS_RATE_ANG, cid, bias_rate_ang)
    write_vec3(constraints, base_offset + _OFF_MASS_COEFF_ANG, cid, mass_coeff_ang)
    write_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_ANG, cid, impulse_coeff_ang)
    write_vec3(constraints, base_offset + _OFF_BIAS_RATE_LIN, cid, bias_rate_lin)
    write_vec3(constraints, base_offset + _OFF_MASS_COEFF_LIN, cid, mass_coeff_lin)
    write_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_LIN, cid, impulse_coeff_lin)

    # Per-axis impulse caps. ``max_force = 0`` -> ``max_lambda = 0`` ->
    # the iterate-path clamp pins the accumulated impulse to 0 (free
    # axis). Any positive ``max_force`` becomes the standard
    # ``max_force * dt`` per-substep impulse cap.
    max_lambda_ang = max_force_ang * dt
    max_lambda_lin = max_force_lin * dt
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_ANG, cid, max_lambda_ang)
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_LIN, cid, max_lambda_lin)

    # ---- Angular block: K_rot = J^T (I_1^{-1} + I_2^{-1}) J ----
    # Linearised quaternion-error Jacobian (same construction as
    # :func:`prismatic_prepare_for_iteration_at`). For the D6 we keep
    # all three angular rows (no projection).
    q1_inv = wp.quat_inverse(q1)
    quat_e = q0 * q1_inv * q2  # error quaternion (identity at the
    # user-specified target pose, since q0 already absorbed the
    # angular position target).
    if quat_e[3] < 0.0:
        sign = -1.0
    else:
        sign = 1.0
    err_vec_ang = wp.vec3f(quat_e[0] * sign, quat_e[1] * sign, quat_e[2] * sign)

    # m0 = -1/2 * QMatrix.ProjectMultiplyLeftRight(q0 * q1^*, q2). Same
    # closed form used by hinge_angle / prismatic; the 1/2 accounts for
    # the unit-quat half-angle convention.
    qq = q0 * q1_inv
    m0 = qmatrix_project_multiply_left_right(qq, q2) * (-0.5 * sign)
    j_rot = m0  # 3x3, rows = world-axis Jacobian.
    j_rot_t = wp.transpose(j_rot)

    inv_inertia_sum = inv_inertia1 + inv_inertia2
    k_rot = j_rot_t @ (inv_inertia_sum @ j_rot)
    # Guard against fully-singular K_rot (free angular block + static
    # body): add a tiny diagonal so wp.inverse still returns finite
    # numbers. The per-axis ``max_lambda_ang = 0`` clamp will pin the
    # impulse to zero anyway, so the regularisation has no observable
    # effect on the physics; it just keeps NaNs out of the back-sub.
    eye3 = wp.identity(3, dtype=wp.float32)
    k_rot_reg = k_rot + eye3 * 1.0e-10
    k_rot_inv = wp.inverse(k_rot_reg)

    # Angular bias. ``bias_rate_ang`` is a 3-vec (one per axis); the
    # quaternion error is in the *quaternion-axis basis*, which lines
    # up with the Jacobian rows so the per-axis bias_rate scaling makes
    # physical sense per quaternion-axis component.
    bias_rot = wp.vec3f(
        err_vec_ang[0] * bias_rate_ang[0],
        err_vec_ang[1] * bias_rate_ang[1],
        err_vec_ang[2] * bias_rate_ang[2],
    )

    # ---- Linear block: K_trans = R_1^T A R_1 ----
    # ``A`` is the standard ball-socket effective mass at the (shared)
    # anchor; we then express it in the body-1-local frame so the
    # per-axis Hertz/damping coefficients act on body-1-local axes
    # (x = body-1 e_x, y = body-1 e_y, z = body-1 e_z).
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)
    a_mat = (inv_mass1 + inv_mass2) * eye3
    a_mat = a_mat + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    a_mat = a_mat + cr2 @ (inv_inertia2 @ wp.transpose(cr2))
    k_trans = wp.transpose(r1_basis) @ (a_mat @ r1_basis)

    # ---- Cross block: K_rot_trans (3x3) ----
    # Working it out (mirrors the prismatic derivation but with R_1
    # instead of [t1|t2] for the linear axes):
    #   J_rot^T contributes I_1^{-1} J_rot to omega_1 -> -cr1 v at the
    #   anchor -> project onto R_1.
    # Algebraically:
    #   K_rot_trans = J_rot^T (-I_1^{-1} cr1^T + I_2^{-1} cr2^T) R_1
    cross_b1 = inv_inertia1 @ wp.transpose(cr1)
    cross_b2 = inv_inertia2 @ wp.transpose(cr2)
    k_rot_trans = j_rot_t @ ((-cross_b1 + cross_b2) @ r1_basis)

    # ---- Schur complement: S = K_trans - K_rot_trans^T K_rot^{-1} K_rot_trans ----
    kt_ki = wp.transpose(k_rot_trans) @ k_rot_inv  # 3x3
    s_mat = k_trans - kt_ki @ k_rot_trans
    s_mat_reg = s_mat + eye3 * 1.0e-10  # see K_rot regularisation above
    s_inv = wp.inverse(s_mat_reg)

    write_mat33(constraints, base_offset + _OFF_J_ROT, cid, j_rot)
    write_mat33(constraints, base_offset + _OFF_K_ROT_INV, cid, k_rot_inv)
    write_mat33(constraints, base_offset + _OFF_KT_KI, cid, kt_ki)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv)
    write_vec3(constraints, base_offset + _OFF_BIAS_ROT, cid, bias_rot)

    # Linear bias = R_1^T (P2 - P1), per-axis-scaled by bias_rate_lin.
    # The drift expressed in body-1-local coordinates lines up with the
    # per-axis ``hertz_lin / damping_lin`` knobs so each linear axis
    # gets its own spring tuning.
    drift_world = p2 - p1
    drift_local = wp.transpose(r1_basis) @ drift_world
    bias_lin = wp.vec3f(
        drift_local[0] * bias_rate_lin[0],
        drift_local[1] * bias_rate_lin[1],
        drift_local[2] * bias_rate_lin[2],
    )
    write_vec3(constraints, base_offset + _OFF_BIAS_LIN, cid, bias_lin)

    # ---- Warm start ----
    # ``acc_rot_in`` is in the quaternion-axis basis; the corresponding
    # world-frame angular impulse on body 1 is ``J acc_rot``.
    # ``acc_lin_local_in`` is in body-1-local; the world-frame linear
    # impulse is ``R_1 acc_lin_local_in``. Sign conventions match
    # iterate (and prismatic).
    j_rot_acc_rot = j_rot @ acc_rot_in
    acc_lin_world = r1_basis @ acc_lin_local_in

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc_lin_world
    angular_velocity1 = (
        bodies.angular_velocity[b1]
        - inv_inertia1 @ (cr1 @ acc_lin_world)
        + inv_inertia1 @ j_rot_acc_rot
    )
    velocity2 = bodies.velocity[b2] + inv_mass2 * acc_lin_world
    angular_velocity2 = (
        bodies.angular_velocity[b2]
        + inv_inertia2 @ (cr2 @ acc_lin_world)
        - inv_inertia2 @ j_rot_acc_rot
    )

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def _clamp_per_axis(v: wp.vec3f, lo: wp.vec3f, hi: wp.vec3f) -> wp.vec3f:
    """Component-wise ``clamp``. Used for the per-axis impulse cap."""
    return wp.vec3f(
        wp.clamp(v[0], lo[0], hi[0]),
        wp.clamp(v[1], lo[1], hi[1]),
        wp.clamp(v[2], lo[2], hi[2]),
    )


@wp.func
def _hadamard(a: wp.vec3f, b: wp.vec3f) -> wp.vec3f:
    """Component-wise multiply -- the per-axis soft-coefficient scaling."""
    return wp.vec3f(a[0] * b[0], a[1] * b[1], a[2] * b[2])


@wp.func
def d6_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable PGS iteration step for the D6 joint.

    Solves the 6x6 block-coupled system via the Schur complement (two
    3x3 inverses, both already cached by ``prepare``), applies per-
    axis soft-coefficient scaling, clamps the *accumulated* impulse to
    the per-axis ``max_lambda``, then applies the (clamped delta)
    impulses to both bodies. See :func:`prismatic_iterate_at` for
    the ``base_offset`` / ``body_pair`` contract.

    The per-axis clamp on the accumulated impulse is the standard
    PhysX D6 / Bullet 6Dof2 / Bepu approach: it converges to the
    constrained solution across PGS iterations the same way the 1-DoF
    angular motor's clamp does, without needing an inner active-set
    QP. ``max_lambda = 0`` -> the impulse is pinned to 0 every
    iteration, which is how to mark an axis as *free*.
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
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    r1_basis = read_mat33(constraints, base_offset + _OFF_R1_BASIS, cid)
    j_rot = read_mat33(constraints, base_offset + _OFF_J_ROT, cid)
    k_rot_inv = read_mat33(constraints, base_offset + _OFF_K_ROT_INV, cid)
    kt_ki = read_mat33(constraints, base_offset + _OFF_KT_KI, cid)
    s_inv = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
    bias_rot = read_vec3(constraints, base_offset + _OFF_BIAS_ROT, cid)
    bias_lin = read_vec3(constraints, base_offset + _OFF_BIAS_LIN, cid)

    target_vel_ang = read_vec3(constraints, base_offset + _OFF_TARGET_VEL_ANG, cid)
    target_vel_lin = read_vec3(constraints, base_offset + _OFF_TARGET_VEL_LIN, cid)

    mass_coeff_ang = read_vec3(constraints, base_offset + _OFF_MASS_COEFF_ANG, cid)
    impulse_coeff_ang = read_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_ANG, cid)
    mass_coeff_lin = read_vec3(constraints, base_offset + _OFF_MASS_COEFF_LIN, cid)
    impulse_coeff_lin = read_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_LIN, cid)

    max_lambda_ang = read_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_ANG, cid)
    max_lambda_lin = read_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_LIN, cid)

    acc_rot = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_local = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)

    # ---- Velocity-error vectors (per-axis-local) ----
    # Angular block: ``jv_rot = J^T (w1 - w2)`` in quaternion-axis
    # basis. Subtract the (rotated) angular target velocity so a
    # velocity drive simply chases the setpoint. The angular target is
    # given in body-1-local; rotate it into the *world* angular-velocity
    # frame and then project through ``J^T`` so it lands in the same
    # quaternion-axis basis as the rest of ``jv_rot``.
    j_rot_t = wp.transpose(j_rot)
    target_w_world = r1_basis @ target_vel_ang
    jv_rot = j_rot_t @ (angular_velocity1 - angular_velocity2 - target_w_world)

    # Linear block: ``Cdot_world = -v1 + cr1 w1 + v2 - cr2 w2``;
    # express in body-1-local as ``R_1^T Cdot_world`` and subtract the
    # body-1-local velocity target.
    cdot_lin_world = -velocity1 + cr1 @ angular_velocity1 + velocity2 - cr2 @ angular_velocity2
    jv_lin = wp.transpose(r1_basis) @ cdot_lin_world - target_vel_lin

    # ---- Build right-hand sides (with bias) ----
    rhs_rot = jv_rot + bias_rot
    rhs_lin = jv_lin + bias_lin

    # ---- Schur-complement solve (unbounded, 6x6 -> two 3x3 mat-vecs) ----
    # lambda_lin_local = -S^{-1} (rhs_lin - kt_ki @ rhs_rot)
    # lambda_rot       = -K_rot_inv (rhs_rot + K_rot_trans @ lambda_lin_local)
    lam_lin_local_unb = -(s_inv @ (rhs_lin - kt_ki @ rhs_rot))

    # Back-substitute lam_lin_local_unb's contribution to the angular
    # RHS via the same physics-driven path used by prismatic_iterate_at:
    # the linear impulse perturbs the body angular velocities; we
    # project the perturbation through J^T to get the correction to the
    # angular RHS. (Algebraically equivalent to ``K_rot_trans @
    # lam_lin_local`` but constructed without re-deriving K_rot_trans.)
    lam_lin_world = r1_basis @ lam_lin_local_unb
    delta_w1 = -inv_inertia1 @ (cr1 @ lam_lin_world)
    delta_w2 = +inv_inertia2 @ (cr2 @ lam_lin_world)
    delta_rhs_rot = j_rot_t @ (delta_w1 - delta_w2)
    lam_rot_unb = -(k_rot_inv @ (rhs_rot + delta_rhs_rot))

    # ---- Per-axis Box2D / Bepu soft-constraint scaling ----
    # Each axis scales independently:
    #   lam_axis = mass_coeff_axis * lam_axis_unb - impulse_coeff_axis * acc_axis.
    # All mass_coeff = 1, all impulse_coeff = 0 -> rigid plain-PGS;
    # any axis with mass_coeff = 0 -> that axis contributes no impulse.
    lam_rot_soft = _hadamard(mass_coeff_ang, lam_rot_unb) - _hadamard(impulse_coeff_ang, acc_rot)
    lam_lin_local_soft = _hadamard(mass_coeff_lin, lam_lin_local_unb) - _hadamard(
        impulse_coeff_lin, acc_lin_local
    )

    # ---- Per-axis force-cap on the *accumulated* impulse ----
    # Standard motor pattern; works because we operate on the
    # accumulated impulse rather than per-iteration delta. Across PGS
    # iterations the clipped solution converges to the constrained
    # one. ``max_lambda = 0`` for an axis -> the clamp pins acc to 0
    # (free axis); ``max_lambda > 0`` -> standard force/torque cap.
    old_acc_rot = acc_rot
    old_acc_lin = acc_lin_local
    new_acc_rot = _clamp_per_axis(old_acc_rot + lam_rot_soft, -max_lambda_ang, max_lambda_ang)
    new_acc_lin = _clamp_per_axis(
        old_acc_lin + lam_lin_local_soft, -max_lambda_lin, max_lambda_lin
    )
    lam_rot = new_acc_rot - old_acc_rot
    lam_lin_local = new_acc_lin - old_acc_lin
    lam_lin_world_final = r1_basis @ lam_lin_local

    # ---- Apply impulses to bodies ----
    # Same convention as :mod:`constraint_prismatic`: world-frame
    # angular impulse on body 1 is ``J lam_rot`` (NOT ``J^T``); body 2
    # gets the opposite sign. The asymmetry vs the velocity-error
    # projection (``jv = J^T (w1-w2)``) is by design -- see the
    # warm-start comment in prismatic for the eff-consistency argument.
    j_rot_lam_rot = j_rot @ lam_rot
    bodies.velocity[b1] = velocity1 - inv_mass1 * lam_lin_world_final
    bodies.angular_velocity[b1] = (
        angular_velocity1
        - inv_inertia1 @ (cr1 @ lam_lin_world_final)
        + inv_inertia1 @ j_rot_lam_rot
    )

    bodies.velocity[b2] = velocity2 + inv_mass2 * lam_lin_world_final
    bodies.angular_velocity[b2] = (
        angular_velocity2
        + inv_inertia2 @ (cr2 @ lam_lin_world_final)
        - inv_inertia2 @ j_rot_lam_rot
    )

    # ---- Persist accumulated impulses ----
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid, new_acc_rot)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid, new_acc_lin)


@wp.func
def d6_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2; see :func:`d6_world_wrench`."""
    acc_rot = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_local = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    j_rot = read_mat33(constraints, base_offset + _OFF_J_ROT, cid)
    r1_basis = read_mat33(constraints, base_offset + _OFF_R1_BASIS, cid)

    # Linear constraint force on body 2 = (R_1 acc_lin_local) / dt;
    # body 2 receives the +ve linear impulse in iterate.
    force = (r1_basis @ acc_lin_local) * idt

    # Torque on body 2: world-frame angular impulse (J acc_rot) / dt
    # plus the moment of the linear impulse about body 2's COM. Sign
    # conventions match iterate: body 2 receives -J acc_rot for the
    # angular block.
    angular_impulse_world = j_rot @ acc_rot
    torque = wp.cross(r2, force) - angular_impulse_world * idt
    return force, torque


@wp.func
def d6_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct prepare entry: reads body indices from the column header
    and forwards to :func:`d6_prepare_for_iteration_at` with
    ``base_offset = 0``."""
    b1 = d6_get_body1(constraints, cid)
    b2 = d6_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    d6_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def d6_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct iterate entry; see :func:`d6_iterate_at`."""
    b1 = d6_get_body1(constraints, cid)
    b2 = d6_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    d6_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def d6_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body 2.

    Force is the linear constraint impulse rotated from body-1-local
    back into world frame, divided by the substep ``dt``
    (``idt = 1 / substep_dt``); torque is that force's moment about
    body 2's COM plus the angular block accumulated impulse / dt.

    Reports the *signed* per-axis force/torque after the per-axis
    clamp -- so a saturated axis reports exactly its ``+/- max_force``,
    a free axis reports ``0``.
    """
    return d6_world_wrench_at(constraints, cid, 0, idt)
