# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""6-DoF generalised ("D6") joint constraint.

A single D6 owns *all* 6 relative DoF (3 angular + 3 linear) between
body 1 and body 2 and configures each axis independently as one of:

* **Rigid lock**     -- ``hertz <= 0`` and ``max_force = +inf``. Plain
  PGS lock; the axis is welded.
* **Soft lock**      -- ``hertz > 0``, no targets. Spring-and-damper
  restoring the rest pose with the Box2D / Bepu / Nordby implicit
  formulation (see :func:`soft_constraint_coefficients`).
* **Position drive** -- ``hertz > 0``, ``target_position != 0``,
  ``max_force`` finite. Implicit-PD drive that pulls the axis towards
  the target with the soft-spring stiffness, capped at ``max_force``.
* **Velocity drive** -- ``hertz > 0``, ``target_velocity != 0``,
  ``max_force`` finite. Implicit-PD velocity-tracking drive ("motor"
  mode); ``hertz`` modulates how aggressively the impulse converges
  to the velocity setpoint.
* **Free**           -- ``max_force = 0``. The accumulated impulse is
  pinned to zero on every PGS pass regardless of ``hertz`` -- the
  canonical way to mark an axis as un-constrained without changing
  the matrix shape.

Position + velocity targets compose, yielding an over-damped spring
with a steady-state velocity (e.g. "open the door at 1 rad/s while
pulling it towards 90°").

The math closely follows Jolt's ``SixDOFConstraint``
(``Jolt/Physics/Constraints/SixDOFConstraint.cpp``) but is adapted to
this codebase's solver style (single dispatched ``wp.func`` per
constraint type, column-major dword storage, Box2D / Bepu soft
constraints with no separate "position constraint" pass, ...). The
biggest deviation is that we always use the *uniform* matrix shape
(full 6x6 with per-axis softness/cap knobs) rather than dispatching
to a smaller part when "all rotation locked" or "all translation
locked"; uniform shape lets the PGS dispatcher stay branchless and
keeps the warm-start state size constant across drive-mode switches.

Why a 6x6 fused constraint instead of three cascaded ones?
----------------------------------------------------------
Same motivation as :mod:`constraint_prismatic` and
:mod:`constraint_double_ball_socket`: solving the joint as one fused
PGS thread converges much faster on chains because the angular and
linear blocks are coupled (an angular impulse displaces the anchor
laterally), so block-diagonal solving wastes iterations bouncing the
cross-block residual. Single-thread fusion also lets the partitioner
colour one D6 per partition (instead of three or six), so dense
graphs of D6s (e.g. an articulated character) parallelise far better.

Schur-complement solve (avoiding a 6x6 inverse)
-----------------------------------------------
Warp's ``wp.inverse`` overloads only cover 2x2 / 3x3 / 4x4. The 6x6
effective mass is

::

    K = [ K_rot          K_rot_trans ]   in R^{6x6}
        [ K_rot_trans^T  K_lin       ]

with ``K_rot``, ``K_lin``, ``K_rot_trans`` each in R^{3x3}. We
factor it via block-elimination using only two 3x3 inverses:

::

    S       = K_lin - K_rot_trans^T K_rot^{-1} K_rot_trans  # 3x3 Schur
    lambda_lin = -S^{-1}     ( b_lin - K_rot_trans^T K_rot^{-1} b_rot )
    lambda_rot = -K_rot^{-1} ( b_rot + K_rot_trans lambda_lin )

We cache ``K_rot_inv``, ``S_inv``, and ``Kt_Ki = K_rot_trans^T
K_rot^{-1}`` (all 3x3) in the constraint column, so ``iterate`` is
just three small mat-vecs plus the impulse application -- no
per-iter inverses.

Constraint frame ("body-1 local")
---------------------------------
The 6 axes are interpreted in *body 1's local frame*: the joint
frame "rides" body 1 rigidly. The three world-frame axes are
recomputed each prepare as ``e_k = R_1 e_k_local`` where
``e_k_local`` is the body-1-local k-th axis (just the k-th column
of the identity at init time, but stored explicitly so a future
extension can reorient the constraint frame independently of body 1).
Position targets are folded into the rest pose / rest anchor at
:func:`d6_initialize_kernel` time so the runtime kernel doesn't
branch on "is there a position target" (the per-axis bias is just
``bias_rate * (current_pos - rest_pos_with_target)``).

Per-axis caps (free axes / drives)
----------------------------------
Each axis carries an independent ``max_lambda = max_force * dt``.
After the Schur solve produces the unconstrained ``lambda_*``
3-vectors, we project them into the body-1-local axis basis,
clamp each component independently against the per-axis cap, and
re-expand back to world. ``max_force = 0`` therefore zeros out the
component (free axis); ``max_force = +inf`` (passed as 1e30 by the
descriptor) leaves the component untouched (rigid).

Per-axis softness
-----------------
The Box2D v3 / Bepu / Nordby soft-constraint coefficients
(``bias_rate, mass_coeff, impulse_coeff``) are per-axis: each block
has 3 ``vec3f`` triples (one per axis). The PGS update scales each
component independently:

::

    lambda_local[k] = mass_coeff[k] * lambda_unsoft_local[k]
                      - impulse_coeff[k] * acc_local[k]

Setting (``mass_coeff[k]``, ``impulse_coeff[k]``) = (1, 0)
recovers a rigid plain-PGS update on axis ``k``.

Mapping summary (deltas vs prismatic):
* slide axes ``e1, e2, e3``  -> body-1-local frame (here just the
                               three columns of identity, but
                               carried explicitly for symmetry).
* rest orientation ``q0``    -> ``q2^* * q1`` snapshotted at
                               initialize, *with the angular
                               position targets folded in* via
                               left-multiplication by the
                               target-rotation quaternion in
                               body 1's frame.
* rest anchor                -> per-body local-anchor offsets *with
                               the linear position targets folded
                               into body 1's local anchor* so
                               "drive to rest pose" tracks the
                               user-specified offset automatically.
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
    read_int,
    read_mat33,
    read_quat,
    read_vec3,
    soft_constraint_coefficients,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import (
    qmatrix_project_multiply_left_right,
)

__all__ = [
    "D6_DWORDS",
    "D6Data",
    "d6_get_accumulated_impulse_lin",
    "d6_get_accumulated_impulse_rot",
    "d6_get_axes_local_b1",
    "d6_get_bias_lin",
    "d6_get_bias_rate_ang",
    "d6_get_bias_rate_lin",
    "d6_get_bias_rot",
    "d6_get_body1",
    "d6_get_body2",
    "d6_get_damping_ratio_ang",
    "d6_get_damping_ratio_lin",
    "d6_get_e_world",
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
    "d6_get_s_inv",
    "d6_get_target_velocity_ang",
    "d6_get_target_velocity_lin",
    "d6_initialize_kernel",
    "d6_iterate",
    "d6_iterate_at",
    "d6_prepare_for_iteration",
    "d6_prepare_for_iteration_at",
    "d6_set_accumulated_impulse_lin",
    "d6_set_accumulated_impulse_rot",
    "d6_set_axes_local_b1",
    "d6_set_bias_lin",
    "d6_set_bias_rate_ang",
    "d6_set_bias_rate_lin",
    "d6_set_bias_rot",
    "d6_set_body1",
    "d6_set_body2",
    "d6_set_damping_ratio_ang",
    "d6_set_damping_ratio_lin",
    "d6_set_e_world",
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
    "d6_set_s_inv",
    "d6_set_target_velocity_ang",
    "d6_set_target_velocity_lin",
    "d6_world_wrench",
    "d6_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class D6Data:
    """Per-constraint dword-layout schema for a D6 (6-DoF) joint.

    *Schema only* (same conventions as :class:`BallSocketData` /
    :class:`PrismaticData`). Field order fixes dword offsets;
    runtime kernels read/write fields out of the shared
    :class:`ConstraintContainer`.

    The first three fields are the global constraint header
    (``constraint_type``, ``body1``, ``body2`` at dwords 0/1/2),
    enforced by :func:`assert_constraint_header`.

    Per-axis knobs (one ``vec3f`` per (block, knob) pair, with the
    three components addressing the three body-1-local axes) are
    laid out so the kernel can load all three components of one knob
    in a single :func:`read_vec3` call. Splitting "angular" from
    "linear" into separate vec3 fields keeps the per-block soft-
    constraint plumbing symmetric with the rest of the solver.
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Anchor point in each body's local frame. The anchors are
    # snapshotted from a shared world-space ``anchor`` at initialize,
    # and the linear position targets are folded into ``local_anchor_b1``
    # so that the "rest pose" (the configuration the bias drives the
    # joint toward) reflects the user-specified target offsets.
    local_anchor_b1: wp.vec3f
    local_anchor_b2: wp.vec3f

    # Three body-1-local axes packed as the rows of a mat33. By default
    # this is the 3x3 identity (the constraint frame coincides with
    # body 1's local frame at init time); kept as a real matrix so a
    # future extension can let the user specify an oriented constraint
    # frame relative to body 1 without code changes.
    axes_local_b1: wp.mat33f

    # Rest-pose relative orientation ``q0 = q2^* * q1_target`` where
    # ``q1_target = q1 * q_target_in_b1`` folds the angular position
    # targets (interpreted in body-1's constraint frame) into the rest
    # pose. The angular bias drives the current ``q2^* q1`` back to
    # this ``q0``, so a non-zero target rotation just shifts the
    # equilibrium.
    q0: wp.quatf

    # Cached per-substep world-space lever arms from each body's COM
    # to the (shared) anchor point.
    r1: wp.vec3f
    r2: wp.vec3f

    # Cached world-space constraint axes (recomputed each prepare as
    # ``R_1 axes_local_b1``). Rows are the three world-frame axes.
    e_world: wp.mat33f

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation), one per axis. ``hertz_*[k] <= 0`` -> rigid for
    # axis ``k``. Persisted across substeps; the per-substep
    # coefficients below are recomputed every prepare.
    hertz_ang: wp.vec3f
    damping_ratio_ang: wp.vec3f
    hertz_lin: wp.vec3f
    damping_ratio_lin: wp.vec3f

    # Target relative angular and linear velocities, in body-1
    # constraint coordinates. Compose with the position targets that
    # were folded into ``q0`` / ``local_anchor_b1`` at initialize.
    target_velocity_ang: wp.vec3f
    target_velocity_lin: wp.vec3f

    # Maximum per-axis force/torque (in body-1 constraint coords).
    # ``max_force_*[k] = 0`` marks axis ``k`` as *free* (the per-axis
    # clamp pins ``acc_local[k] = 0`` every iteration). A finite
    # positive value caps the per-axis impulse at ``max_force * dt``.
    # ``+inf`` is mapped to ``1e30`` by the host-side packer for a
    # branch-free rigid-axis path.
    max_force_ang: wp.vec3f
    max_force_lin: wp.vec3f

    # Cached per-substep ``max_force * dt`` per axis (in constraint
    # coords). Recomputed every prepare from the current substep dt.
    max_lambda_ang: wp.vec3f
    max_lambda_lin: wp.vec3f

    # Cached per-substep soft-constraint coefficients per axis
    # (recomputed every prepare). One ``vec3f`` per (block, coeff).
    bias_rate_ang: wp.vec3f
    mass_coeff_ang: wp.vec3f
    impulse_coeff_ang: wp.vec3f
    bias_rate_lin: wp.vec3f
    mass_coeff_lin: wp.vec3f
    impulse_coeff_lin: wp.vec3f

    # Cached velocity-error bias vectors, expressed in body-1
    # constraint coordinates. ``bias_rot[k] = bias_rate_ang[k] *
    # angular_error_along_axis_k``; ``bias_lin[k] = bias_rate_lin[k] *
    # linear_drift_along_axis_k``.
    bias_rot: wp.vec3f
    bias_lin: wp.vec3f

    # Cached angular Jacobian (3x3, world frame). Same convention as
    # :mod:`constraint_prismatic`: rows are world-axis Jacobian rows
    # in the *quaternion-axis* basis (carries the half-angle factor
    # from the quaternion-error linearisation), so ``J^T (w1 - w2)``
    # is the per-axis angular velocity error in that basis. To
    # express the velocity error in the *constraint* axis basis (the
    # one the per-axis caps and biases live in), we further project
    # by the world-axis frame ``e_world``.
    j_rot: wp.mat33f

    # Cached effective-mass blocks for the Schur-complement solve.
    # ``k_rot_inv`` is the 3x3 inverse of the angular block (in
    # constraint coordinates). ``kt_ki`` is the 3x3 product
    # ``K_rot_trans^T K_rot^{-1}`` (also in constraint coords).
    # ``s_inv`` is the 3x3 Schur-complement inverse.
    k_rot_inv: wp.mat33f
    kt_ki: wp.mat33f
    s_inv: wp.mat33f

    # Accumulated PGS impulses in *world* frame (linear) /
    # *quaternion-axis* frame (angular), so warm-start is invariant
    # under the constraint frame rotating with body 1 between
    # substeps. The per-iter clamp projects them into the constraint
    # axis basis, clamps, and re-expands.
    accumulated_impulse_rot: wp.vec3f
    accumulated_impulse_lin: wp.vec3f


assert_constraint_header(D6Data)


# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(D6Data, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(D6Data, "body2"))
_OFF_LA_B1 = wp.constant(dword_offset_of(D6Data, "local_anchor_b1"))
_OFF_LA_B2 = wp.constant(dword_offset_of(D6Data, "local_anchor_b2"))
_OFF_AXES_LOCAL_B1 = wp.constant(dword_offset_of(D6Data, "axes_local_b1"))
_OFF_Q0 = wp.constant(dword_offset_of(D6Data, "q0"))
_OFF_R1 = wp.constant(dword_offset_of(D6Data, "r1"))
_OFF_R2 = wp.constant(dword_offset_of(D6Data, "r2"))
_OFF_E_WORLD = wp.constant(dword_offset_of(D6Data, "e_world"))
_OFF_HERTZ_ANG = wp.constant(dword_offset_of(D6Data, "hertz_ang"))
_OFF_DAMPING_ANG = wp.constant(dword_offset_of(D6Data, "damping_ratio_ang"))
_OFF_HERTZ_LIN = wp.constant(dword_offset_of(D6Data, "hertz_lin"))
_OFF_DAMPING_LIN = wp.constant(dword_offset_of(D6Data, "damping_ratio_lin"))
_OFF_TARGET_VEL_ANG = wp.constant(dword_offset_of(D6Data, "target_velocity_ang"))
_OFF_TARGET_VEL_LIN = wp.constant(dword_offset_of(D6Data, "target_velocity_lin"))
_OFF_MAX_FORCE_ANG = wp.constant(dword_offset_of(D6Data, "max_force_ang"))
_OFF_MAX_FORCE_LIN = wp.constant(dword_offset_of(D6Data, "max_force_lin"))
_OFF_MAX_LAMBDA_ANG = wp.constant(dword_offset_of(D6Data, "max_lambda_ang"))
_OFF_MAX_LAMBDA_LIN = wp.constant(dword_offset_of(D6Data, "max_lambda_lin"))
_OFF_BIAS_RATE_ANG = wp.constant(dword_offset_of(D6Data, "bias_rate_ang"))
_OFF_MASS_COEFF_ANG = wp.constant(dword_offset_of(D6Data, "mass_coeff_ang"))
_OFF_IMPULSE_COEFF_ANG = wp.constant(dword_offset_of(D6Data, "impulse_coeff_ang"))
_OFF_BIAS_RATE_LIN = wp.constant(dword_offset_of(D6Data, "bias_rate_lin"))
_OFF_MASS_COEFF_LIN = wp.constant(dword_offset_of(D6Data, "mass_coeff_lin"))
_OFF_IMPULSE_COEFF_LIN = wp.constant(dword_offset_of(D6Data, "impulse_coeff_lin"))
_OFF_BIAS_ROT = wp.constant(dword_offset_of(D6Data, "bias_rot"))
_OFF_BIAS_LIN = wp.constant(dword_offset_of(D6Data, "bias_lin"))
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
def d6_get_axes_local_b1(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_AXES_LOCAL_B1, cid)


@wp.func
def d6_set_axes_local_b1(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_AXES_LOCAL_B1, cid, v)


@wp.func
def d6_get_q0(c: ConstraintContainer, cid: wp.int32) -> wp.quatf:
    return read_quat(c, _OFF_Q0, cid)


@wp.func
def d6_set_q0(c: ConstraintContainer, cid: wp.int32, v: wp.quatf):
    write_quat(c, _OFF_Q0, cid, v)


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
def d6_get_e_world(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_E_WORLD, cid)


@wp.func
def d6_set_e_world(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_E_WORLD, cid, v)


@wp.func
def d6_get_hertz_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_HERTZ_ANG, cid)


@wp.func
def d6_set_hertz_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_HERTZ_ANG, cid, v)


@wp.func
def d6_get_damping_ratio_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_DAMPING_ANG, cid)


@wp.func
def d6_set_damping_ratio_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_DAMPING_ANG, cid, v)


@wp.func
def d6_get_hertz_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_HERTZ_LIN, cid)


@wp.func
def d6_set_hertz_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_HERTZ_LIN, cid, v)


@wp.func
def d6_get_damping_ratio_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_DAMPING_LIN, cid)


@wp.func
def d6_set_damping_ratio_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_DAMPING_LIN, cid, v)


@wp.func
def d6_get_target_velocity_ang(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_TARGET_VEL_ANG, cid)


@wp.func
def d6_set_target_velocity_ang(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_TARGET_VEL_ANG, cid, v)


@wp.func
def d6_get_target_velocity_lin(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_TARGET_VEL_LIN, cid)


@wp.func
def d6_set_target_velocity_lin(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_TARGET_VEL_LIN, cid, v)


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
    target_position_ang: wp.array[wp.vec3f],
    target_position_lin: wp.array[wp.vec3f],
    target_velocity_ang: wp.array[wp.vec3f],
    target_velocity_lin: wp.array[wp.vec3f],
    hertz_ang: wp.array[wp.vec3f],
    damping_ratio_ang: wp.array[wp.vec3f],
    hertz_lin: wp.array[wp.vec3f],
    damping_ratio_lin: wp.array[wp.vec3f],
    max_force_ang: wp.array[wp.vec3f],
    max_force_lin: wp.array[wp.vec3f],
):
    """Pack one batch of D6 descriptors into ``constraints``.

    Snapshots the user's world-space anchor + per-axis target poses
    into the rest configuration so the runtime kernel sees a single
    (rest pose, current pose) error rather than separate "lock" and
    "drive" terms. Specifically:

    * The linear position targets ``target_position_lin`` (in body 1's
      constraint frame) are added to body 1's local anchor so the
      anchor-pair drift the linear bias drives to zero already
      includes the target offset.
    * The angular position targets ``target_position_ang`` (XYZ Euler
      angles in body 1's constraint frame, applied as
      ``q_target = qx * qy * qz``) are folded into the rest
      orientation via ``q0 = q2^* * (q1 * q_target)`` so the angular
      bias drives ``q2^* q1`` back to the offset rest pose.

    The constraint frame is chosen as body 1's local frame at init
    time (``axes_local_b1`` = identity); a future extension can make
    this user-configurable without changing the runtime kernel.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; reads ``position`` /
            ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor point [num_in_batch] [m]. Both
            bodies' local anchors are derived from this so they
            coincide at rest (modulo the linear position target).
        target_position_ang: Per-axis angular position targets
            [num_in_batch] [rad], interpreted as XYZ Euler in body 1's
            constraint frame.
        target_position_lin: Per-axis linear position targets
            [num_in_batch] [m], interpreted in body 1's constraint
            frame.
        target_velocity_ang: Per-axis angular velocity targets
            [num_in_batch] [rad/s], interpreted in body 1's constraint
            frame.
        target_velocity_lin: Per-axis linear velocity targets
            [num_in_batch] [m/s], interpreted in body 1's constraint
            frame.
        hertz_ang: Per-axis angular soft-constraint stiffness
            [num_in_batch] [Hz]. ``hertz <= 0`` -> rigid axis.
        damping_ratio_ang: Per-axis angular damping ratios
            [num_in_batch] (1 = critical).
        hertz_lin: Per-axis linear stiffness [num_in_batch] [Hz].
        damping_ratio_lin: Per-axis linear damping ratios
            [num_in_batch].
        max_force_ang: Per-axis angular force cap [num_in_batch]
            [N*m]. ``0`` marks the axis as free; ``+inf`` (passed as
            ``1e30`` by the host packer) is "rigid / no cap".
        max_force_lin: Per-axis linear force cap [num_in_batch] [N].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a_world = anchor[tid]
    tp_ang = target_position_ang[tid]
    tp_lin = target_position_lin[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # Body-1 local anchor *with the linear position target folded in*
    # along the constraint frame (which, at init time, is body 1's
    # local frame -- identity ``axes_local_b1``). This lets the
    # runtime kernel treat "drive to a non-zero offset" the same way
    # as "rigid lock at the rest pose" -- the bias just measures
    # against the shifted rest anchor.
    la_b1_rest = wp.quat_rotate_inv(q1, a_world - pos1)
    la_b1 = la_b1_rest + tp_lin
    la_b2 = wp.quat_rotate_inv(q2, a_world - pos2)

    # Build the angular target rotation in body 1's constraint frame
    # as an XYZ Euler composition. We use a closed-form half-angle
    # quaternion product instead of ``wp.quat_from_axis_angle`` calls
    # to keep the kernel branchless and stay inside the per-thread
    # arithmetic budget.
    half_x = tp_ang[0] * 0.5
    half_y = tp_ang[1] * 0.5
    half_z = tp_ang[2] * 0.5
    qx = wp.quatf(wp.sin(half_x), 0.0, 0.0, wp.cos(half_x))
    qy = wp.quatf(0.0, wp.sin(half_y), 0.0, wp.cos(half_y))
    qz = wp.quatf(0.0, 0.0, wp.sin(half_z), wp.cos(half_z))
    q_target = qx * qy * qz

    # Rest-pose relative orientation with the angular target folded
    # into body 1: q0 = q2^* * (q1 * q_target). So the runtime
    # error quaternion ``q0 * q1^* * q2`` is identity exactly when
    # body 2's orientation matches the user-specified target offset
    # relative to body 1.
    q1_target = q1 * q_target
    q0 = wp.quat_inverse(q2) * q1_target

    # Constraint-frame axes are body 1's identity at init time. We
    # store them as a real matrix so a future extension can let the
    # user reorient the constraint frame relative to body 1 without
    # touching the runtime path -- the prepare just picks up the new
    # axes.
    eye3 = wp.identity(3, dtype=wp.float32)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_D6)
    d6_set_body1(constraints, cid, b1)
    d6_set_body2(constraints, cid, b2)
    d6_set_local_anchor_b1(constraints, cid, la_b1)
    d6_set_local_anchor_b2(constraints, cid, la_b2)
    d6_set_axes_local_b1(constraints, cid, eye3)
    d6_set_q0(constraints, cid, q0)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    d6_set_r1(constraints, cid, zero3)
    d6_set_r2(constraints, cid, zero3)
    d6_set_e_world(constraints, cid, eye3)
    d6_set_bias_rot(constraints, cid, zero3)
    d6_set_bias_lin(constraints, cid, zero3)
    d6_set_accumulated_impulse_rot(constraints, cid, zero3)
    d6_set_accumulated_impulse_lin(constraints, cid, zero3)

    d6_set_hertz_ang(constraints, cid, hertz_ang[tid])
    d6_set_damping_ratio_ang(constraints, cid, damping_ratio_ang[tid])
    d6_set_hertz_lin(constraints, cid, hertz_lin[tid])
    d6_set_damping_ratio_lin(constraints, cid, damping_ratio_lin[tid])
    d6_set_target_velocity_ang(constraints, cid, target_velocity_ang[tid])
    d6_set_target_velocity_lin(constraints, cid, target_velocity_lin[tid])
    d6_set_max_force_ang(constraints, cid, max_force_ang[tid])
    d6_set_max_force_lin(constraints, cid, max_force_lin[tid])

    d6_set_max_lambda_ang(constraints, cid, zero3)
    d6_set_max_lambda_lin(constraints, cid, zero3)
    d6_set_bias_rate_ang(constraints, cid, zero3)
    d6_set_mass_coeff_ang(constraints, cid, wp.vec3f(1.0, 1.0, 1.0))
    d6_set_impulse_coeff_ang(constraints, cid, zero3)
    d6_set_bias_rate_lin(constraints, cid, zero3)
    d6_set_mass_coeff_lin(constraints, cid, wp.vec3f(1.0, 1.0, 1.0))
    d6_set_impulse_coeff_lin(constraints, cid, zero3)

    d6_set_j_rot(constraints, cid, eye3)
    d6_set_k_rot_inv(constraints, cid, eye3)
    d6_set_kt_ki(constraints, cid, eye3)
    d6_set_s_inv(constraints, cid, eye3)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Symbol cheat-sheet (matches module-docstring derivation):
#   r1, r2            : world-space lever arms (body 1 / body 2 -> anchor)
#   cr1, cr2          : skew([r1]), skew([r2])
#   e_world (3x3)     : rows = world-frame constraint axes
#   E (3x3)           : same as e_world (just a more "matrix"-ish name)
#   J_rot (3x3)       : world-frame angular Jacobian in quaternion-axis basis
#                       (carries the half-angle factor)
#   K_rot (3x3)       : angular block of the 6x6 effective mass, expressed
#                       in body-1 *constraint* coordinates:
#                       K_rot = E J_rot^T (I_1^{-1} + I_2^{-1}) J_rot E^T
#   K_lin (3x3)       : translational block in constraint coords:
#                       K_lin = E A E^T  where A is the standard ball-socket
#                                        effective-mass at the anchor
#   K_rt (3x3)        : cross block (constraint coords). Derived in prepare
#                       below; comes from the angular impulse displacing
#                       the anchor laterally.
#   S = K_lin - K_rt^T K_rot^{-1} K_rt   : 3x3 Schur complement
#
# Soft-constraint scaling: per-axis triples (bias_rate, mass_coeff,
# impulse_coeff) live in *constraint coordinates*. A "rigid" axis just
# carries (mass_coeff[k], impulse_coeff[k]) = (1, 0) and a non-zero
# bias_rate; a "free" axis has max_force = 0 so the per-axis clamp
# pins its impulse to zero regardless of softness.


@wp.func
def _vec3_componentwise_clamp(v: wp.vec3f, m: wp.vec3f) -> wp.vec3f:
    """Clamp each component of ``v`` to ``[-m[i], m[i]]``.

    ``m`` is the per-axis impulse cap (``max_force * dt``). For a
    "rigid" axis the cap is 1e30 (unlimited); for a "free" axis it
    is 0 (impulse pinned to zero). The clamp is the only place
    per-axis caps appear in the iterate path -- everything else is
    componentwise-uniform soft PGS.
    """
    return wp.vec3f(
        wp.clamp(v[0], -m[0], m[0]),
        wp.clamp(v[1], -m[1], m[1]),
        wp.clamp(v[2], -m[2], m[2]),
    )


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

    Recomputes world-space lever arms + constraint axes, builds the
    three blocks of the 6x6 effective mass in *constraint*
    coordinates, factors them via Schur complement, computes the
    velocity-error bias for both blocks (with the per-axis position
    targets already folded into the rest pose at init time), caches
    the per-substep soft-constraint and impulse-cap coefficients,
    and warm-starts the bodies with the cached accumulated impulses.

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
    axes_local = read_mat33(constraints, base_offset + _OFF_AXES_LOCAL_B1, cid)
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

    # World-frame constraint axes: rows of ``e_world`` are the three
    # world-space axes ``e_k_world = R_1 axes_local_b1[k]``. We
    # construct the matrix row-by-row so each row really is the
    # rotated local axis (as opposed to e.g. ``R_1 axes_local^T``).
    e_row0 = wp.quat_rotate(q1, wp.vec3f(axes_local[0, 0], axes_local[0, 1], axes_local[0, 2]))
    e_row1 = wp.quat_rotate(q1, wp.vec3f(axes_local[1, 0], axes_local[1, 1], axes_local[1, 2]))
    e_row2 = wp.quat_rotate(q1, wp.vec3f(axes_local[2, 0], axes_local[2, 1], axes_local[2, 2]))
    e_world = wp.mat33f(
        e_row0[0], e_row0[1], e_row0[2],
        e_row1[0], e_row1[1], e_row1[2],
        e_row2[0], e_row2[1], e_row2[2],
    )
    write_mat33(constraints, base_offset + _OFF_E_WORLD, cid, e_world)

    # Soft-constraint coefficients per axis. Each component goes
    # through :func:`soft_constraint_coefficients` independently so
    # mixing rigid + soft + drive axes in one D6 falls out for free.
    dt = 1.0 / idt
    hertz_ang = read_vec3(constraints, base_offset + _OFF_HERTZ_ANG, cid)
    damping_ang = read_vec3(constraints, base_offset + _OFF_DAMPING_ANG, cid)
    hertz_lin = read_vec3(constraints, base_offset + _OFF_HERTZ_LIN, cid)
    damping_lin = read_vec3(constraints, base_offset + _OFF_DAMPING_LIN, cid)

    br_a0, mc_a0, ic_a0 = soft_constraint_coefficients(hertz_ang[0], damping_ang[0], dt)
    br_a1, mc_a1, ic_a1 = soft_constraint_coefficients(hertz_ang[1], damping_ang[1], dt)
    br_a2, mc_a2, ic_a2 = soft_constraint_coefficients(hertz_ang[2], damping_ang[2], dt)
    br_l0, mc_l0, ic_l0 = soft_constraint_coefficients(hertz_lin[0], damping_lin[0], dt)
    br_l1, mc_l1, ic_l1 = soft_constraint_coefficients(hertz_lin[1], damping_lin[1], dt)
    br_l2, mc_l2, ic_l2 = soft_constraint_coefficients(hertz_lin[2], damping_lin[2], dt)
    write_vec3(constraints, base_offset + _OFF_BIAS_RATE_ANG, cid, wp.vec3f(br_a0, br_a1, br_a2))
    write_vec3(constraints, base_offset + _OFF_MASS_COEFF_ANG, cid, wp.vec3f(mc_a0, mc_a1, mc_a2))
    write_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_ANG, cid, wp.vec3f(ic_a0, ic_a1, ic_a2))
    write_vec3(constraints, base_offset + _OFF_BIAS_RATE_LIN, cid, wp.vec3f(br_l0, br_l1, br_l2))
    write_vec3(constraints, base_offset + _OFF_MASS_COEFF_LIN, cid, wp.vec3f(mc_l0, mc_l1, mc_l2))
    write_vec3(constraints, base_offset + _OFF_IMPULSE_COEFF_LIN, cid, wp.vec3f(ic_l0, ic_l1, ic_l2))

    # Per-axis impulse caps for the iterate-path clamp. ``max_force *
    # dt`` is the impulse the axis can apply over one substep.
    max_force_ang = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_ANG, cid)
    max_force_lin = read_vec3(constraints, base_offset + _OFF_MAX_FORCE_LIN, cid)
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_ANG, cid, max_force_ang * dt)
    write_vec3(constraints, base_offset + _OFF_MAX_LAMBDA_LIN, cid, max_force_lin * dt)

    # ---- Angular Jacobian + error (same construction as prismatic) ----
    # quat_e = q0 * q1^* * q2 -- error quaternion (identity when
    # current ``q2^* q1`` matches the rest pose ``q0`` that was
    # snapshotted at init time, *with the angular position targets
    # already folded in*).
    q1_inv = wp.quat_inverse(q1)
    quat_e = q0 * q1_inv * q2
    if quat_e[3] < 0.0:
        sign = -1.0
    else:
        sign = 1.0
    err_vec_world = wp.vec3f(quat_e[0] * sign, quat_e[1] * sign, quat_e[2] * sign)

    # m0 = -1/2 * QMatrix.ProjectMultiplyLeftRight(q0 * q1^*, q2). The
    # 1/2 here pairs with the unit-quaternion half-angle convention
    # (same as :mod:`constraint_hinge_angle` and
    # :mod:`constraint_prismatic`).
    qq = q0 * q1_inv
    m0 = qmatrix_project_multiply_left_right(qq, q2) * (-0.5 * sign)
    j_rot = m0  # rows = world-axis Jacobian in quaternion-axis basis.
    write_mat33(constraints, base_offset + _OFF_J_ROT, cid, j_rot)

    # ---- 3x3 effective-mass blocks in *constraint coordinates* ----
    # E = e_world (rows = world axes, columns = constraint axes when
    # we transpose).
    eye3 = wp.identity(3, dtype=wp.float32)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

    # Standard ball-socket A in world coords:
    a_mat = inv_mass1 * eye3
    a_mat = a_mat + cr1 @ (inv_inertia1 @ wp.transpose(cr1))
    a_mat = a_mat + inv_mass2 * eye3
    a_mat = a_mat + cr2 @ (inv_inertia2 @ wp.transpose(cr2))

    # K_lin = E A E^T  (3x3 in constraint coords). With ``e_world``
    # row-major storing constraint axes as rows, ``E A E^T = e_world @
    # a_mat @ wp.transpose(e_world)``.
    k_lin = e_world @ (a_mat @ wp.transpose(e_world))

    # K_rot in constraint coords:
    #   K_rot = E (J_rot^T (InvI1 + InvI2) J_rot) E^T
    # The inner ``J_rot^T M J_rot`` is the 3x3 angular block in the
    # quaternion-axis basis (same as :mod:`constraint_prismatic`'s
    # ``k_rot``); we then sandwich by E to express it in constraint
    # axes (which is where the per-axis caps and biases live).
    inv_inertia_sum = inv_inertia1 + inv_inertia2
    j_rot_t = wp.transpose(j_rot)
    k_rot_quat = j_rot_t @ (inv_inertia_sum @ j_rot)
    k_rot = e_world @ (k_rot_quat @ wp.transpose(e_world))
    k_rot_inv = wp.inverse(k_rot)

    # K_rt (3x3 cross block in constraint coords). Derivation: the
    # *angular* impulse expressed in constraint axes ``a_local`` maps
    # to a world-frame angular impulse ``E^T a_local`` on body 1
    # (``-E^T a_local`` on body 2 modulo the half-angle linearisation
    # already baked into J_rot), which the angular Jacobian then
    # contracts against. The full closed form, mirroring
    # :mod:`constraint_prismatic`'s 3x2 cross block but extended to
    # all three constraint axes, is
    #
    #     K_rt[:, k] = E (J_rot^T (-InvI1 cr1^T + InvI2 cr2^T) E^T[:, k])
    #
    # i.e. column ``k`` of K_rt is the world-frame translational axis
    # ``E^T[:, k]`` ran through the (linear -> angular -> Jacobian
    # contraction -> projection back to constraint axes) chain.
    cross_b1 = inv_inertia1 @ (wp.transpose(cr1) @ wp.transpose(e_world))  # 3x3
    cross_b2 = inv_inertia2 @ (wp.transpose(cr2) @ wp.transpose(e_world))  # 3x3
    cross_world = j_rot_t @ (-cross_b1 + cross_b2)  # 3x3 in quaternion-axis basis
    k_rt = e_world @ cross_world  # 3x3 in constraint coords

    # ---- Schur complement: S = K_lin - K_rt^T K_rot^{-1} K_rt ----
    kt_ki = wp.transpose(k_rt) @ k_rot_inv
    s_mat = k_lin - kt_ki @ k_rt
    s_inv = wp.inverse(s_mat)

    write_mat33(constraints, base_offset + _OFF_K_ROT_INV, cid, k_rot_inv)
    write_mat33(constraints, base_offset + _OFF_KT_KI, cid, kt_ki)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv)

    # ---- Position-error biases in constraint coords ----
    # Angular error vector lives in the quaternion-axis basis; project
    # onto the world axes (rows of e_world) to get the per-constraint-
    # axis component, then scale by the per-axis bias_rate.
    err_local = e_world @ err_vec_world  # 3-vec in constraint coords
    bias_rate_ang_v = wp.vec3f(br_a0, br_a1, br_a2)
    bias_rot = wp.vec3f(
        err_local[0] * bias_rate_ang_v[0],
        err_local[1] * bias_rate_ang_v[1],
        err_local[2] * bias_rate_ang_v[2],
    )
    write_vec3(constraints, base_offset + _OFF_BIAS_ROT, cid, bias_rot)

    # Linear position error: per-axis projection of the world-space
    # anchor drift ``p2 - p1`` onto the constraint axes. The linear
    # position targets were already folded into ``la_b1`` at init
    # time so the rest configuration has ``p2 - p1 = 0`` exactly when
    # the bodies sit at their target offsets.
    drift = p2 - p1
    drift_local = e_world @ drift
    bias_rate_lin_v = wp.vec3f(br_l0, br_l1, br_l2)
    bias_lin = wp.vec3f(
        drift_local[0] * bias_rate_lin_v[0],
        drift_local[1] * bias_rate_lin_v[1],
        drift_local[2] * bias_rate_lin_v[2],
    )
    write_vec3(constraints, base_offset + _OFF_BIAS_LIN, cid, bias_lin)

    # ---- Warm start ----
    # Same impulse-application formulas as :mod:`constraint_prismatic`
    # (with the linear block extended from a 2-vec to a full 3-vec):
    #   linear acc -> v1 -= m1^{-1} acc_lin, v2 += m2^{-1} acc_lin
    #              -> w1 -= I1^{-1} (cr1 acc_lin), w2 += I2^{-1} (cr2 acc_lin)
    #   angular acc -> w1 += I1^{-1} (J acc_rot_quat)
    #               -> w2 -= I2^{-1} (J acc_rot_quat)
    j_rot_acc_rot = j_rot @ acc_rot_in

    velocity1 = bodies.velocity[b1] - inv_mass1 * acc_lin_in
    angular_velocity1 = (
        bodies.angular_velocity[b1]
        - inv_inertia1 @ (cr1 @ acc_lin_in)
        + inv_inertia1 @ j_rot_acc_rot
    )

    velocity2 = bodies.velocity[b2] + inv_mass2 * acc_lin_in
    angular_velocity2 = (
        bodies.angular_velocity[b2]
        + inv_inertia2 @ (cr2 @ acc_lin_in)
        - inv_inertia2 @ j_rot_acc_rot
    )

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


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

    Solves the 6x6 system via Schur complement (two 3x3 inverses, all
    cached in ``prepare``), applies per-axis softness scaling, clamps
    each axis impulse to its ``max_force * dt`` cap, and applies the
    resulting impulses to both bodies. See
    :func:`ball_socket_iterate_at` for the ``base_offset`` /
    ``body_pair`` contract.
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
    e_world = read_mat33(constraints, base_offset + _OFF_E_WORLD, cid)
    cr1 = wp.skew(r1)
    cr2 = wp.skew(r2)

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

    acc_rot_quat = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)

    # The persisted accumulated impulses live in world (linear) /
    # quaternion-axis (angular) frames so warm-start stays valid even
    # as the constraint frame rotates with body 1 between substeps.
    # Project them into the constraint axis basis to do the per-axis
    # soft-scaling and clamp on the same coordinates as the bias /
    # cap.
    acc_rot_local = e_world @ acc_rot_quat
    acc_lin_local = e_world @ acc_lin_world

    # ---- Velocity-error vectors in constraint coords ----
    # Angular: jv_rot_quat = J^T (w1 - w2), then E to project onto
    # constraint axes, then subtract the per-axis target velocity.
    j_rot_t = wp.transpose(j_rot)
    jv_rot_quat = j_rot_t @ (angular_velocity1 - angular_velocity2)
    jv_rot_local = e_world @ jv_rot_quat

    # Linear: jv_lin_world = -v1 + cr1 w1 + v2 - cr2 w2, then E.
    jv_lin_world = (
        -velocity1
        + cr1 @ angular_velocity1
        + velocity2
        - cr2 @ angular_velocity2
    )
    jv_lin_local = e_world @ jv_lin_world

    # ---- Right-hand sides (with bias and target velocity) ----
    # The target-velocity entries enter as a setpoint shift on the
    # velocity error: the ideal jv on a velocity-driven axis equals
    # the target, so we subtract it from jv before the Schur solve.
    rhs_rot = jv_rot_local - target_vel_ang + bias_rot
    rhs_lin = jv_lin_local - target_vel_lin + bias_lin

    # ---- Schur-complement solve ----
    # lambda_lin = -S^{-1} (rhs_lin - kt_ki @ rhs_rot)
    # lambda_rot = -K_rot_inv (rhs_rot + K_rt @ lambda_lin)
    #            = -K_rot_inv (rhs_rot + ((kt_ki K_rot)^T) @ lambda_lin)
    # We can avoid recomputing K_rt by going through ``kt_ki @ K_rot``
    # = K_rt^T, so K_rt = (kt_ki @ K_rot)^T -- but we also have direct
    # access to ``kt_ki`` only, not K_rt. Easier: cache the back-
    # substitution by re-deriving it from ``kt_ki`` and ``K_rot_inv``:
    # since ``K_rt = K_rot @ kt_ki^T`` (by symmetry of K), the back-
    # sub becomes
    #   lambda_rot = -K_rot_inv (rhs_rot + K_rot kt_ki^T lambda_lin)
    #              = -(K_rot_inv rhs_rot + kt_ki^T lambda_lin)
    # which avoids touching ``K_rot`` directly.
    lam_lin_unsoft_local = -(s_inv @ (rhs_lin - kt_ki @ rhs_rot))
    lam_rot_unsoft_local = -(k_rot_inv @ rhs_rot + wp.transpose(kt_ki) @ lam_lin_unsoft_local)

    # ---- Box2D v3 / Bepu soft-constraint scaling, per-axis ----
    # Each axis scales independently:
    #   lam_local[k] = mass_coeff[k] * lam_unsoft_local[k]
    #                  - impulse_coeff[k] * acc_local[k]
    # Setting (mass_coeff, impulse_coeff) = (1, 0) recovers a rigid
    # plain-PGS update.
    lam_rot_soft_local = wp.vec3f(
        mass_coeff_ang[0] * lam_rot_unsoft_local[0] - impulse_coeff_ang[0] * acc_rot_local[0],
        mass_coeff_ang[1] * lam_rot_unsoft_local[1] - impulse_coeff_ang[1] * acc_rot_local[1],
        mass_coeff_ang[2] * lam_rot_unsoft_local[2] - impulse_coeff_ang[2] * acc_rot_local[2],
    )
    lam_lin_soft_local = wp.vec3f(
        mass_coeff_lin[0] * lam_lin_unsoft_local[0] - impulse_coeff_lin[0] * acc_lin_local[0],
        mass_coeff_lin[1] * lam_lin_unsoft_local[1] - impulse_coeff_lin[1] * acc_lin_local[1],
        mass_coeff_lin[2] * lam_lin_unsoft_local[2] - impulse_coeff_lin[2] * acc_lin_local[2],
    )

    # ---- Per-axis impulse caps (free axes, drives) ----
    # Update the per-axis accumulated impulse, clamp to cap, and
    # back out the actual incremental ``lam_*`` that was applied.
    # ``max_force = 0`` (free axis) -> clamp pins the component to 0;
    # ``max_force = inf`` (rigid) -> clamp is a no-op; finite cap
    # gives a saturating drive.
    new_acc_rot_local = _vec3_componentwise_clamp(
        acc_rot_local + lam_rot_soft_local, max_lambda_ang
    )
    new_acc_lin_local = _vec3_componentwise_clamp(
        acc_lin_local + lam_lin_soft_local, max_lambda_lin
    )
    delta_acc_rot_local = new_acc_rot_local - acc_rot_local
    delta_acc_lin_local = new_acc_lin_local - acc_lin_local

    # ---- Project incremental impulses back to world / quaternion
    # axes for the impulse-application step. ``e_world`` rows are the
    # world axes, so ``E^T x_local`` is the world-frame vector
    # corresponding to constraint-coords ``x_local``.
    e_world_t = wp.transpose(e_world)
    delta_acc_rot_quat = e_world_t @ delta_acc_rot_local
    delta_acc_lin_world = e_world_t @ delta_acc_lin_local

    # ---- Apply impulses to bodies (same convention as prismatic) ----
    j_rot_lam_rot = j_rot @ delta_acc_rot_quat
    bodies.velocity[b1] = velocity1 - inv_mass1 * delta_acc_lin_world
    bodies.angular_velocity[b1] = (
        angular_velocity1
        - inv_inertia1 @ (cr1 @ delta_acc_lin_world)
        + inv_inertia1 @ j_rot_lam_rot
    )

    bodies.velocity[b2] = velocity2 + inv_mass2 * delta_acc_lin_world
    bodies.angular_velocity[b2] = (
        angular_velocity2
        + inv_inertia2 @ (cr2 @ delta_acc_lin_world)
        - inv_inertia2 @ j_rot_lam_rot
    )

    # ---- Persist the updated accumulated impulses ----
    # Re-expand the (clamped) per-axis local impulses back into the
    # warm-start storage frames so the next prepare's warm-start sees
    # exactly the same rotation-invariant impulse we just applied.
    new_acc_rot_quat = e_world_t @ new_acc_rot_local
    new_acc_lin_world = e_world_t @ new_acc_lin_local
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid, new_acc_rot_quat)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid, new_acc_lin_world)


@wp.func
def d6_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2; see :func:`d6_world_wrench`."""
    acc_rot_quat = read_vec3(constraints, base_offset + _OFF_ACC_IMP_ROT, cid)
    acc_lin_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP_LIN, cid)
    r2 = read_vec3(constraints, base_offset + _OFF_R2, cid)
    j_rot = read_mat33(constraints, base_offset + _OFF_J_ROT, cid)
    # Linear block: total constraint force on body 2 = acc_lin / dt.
    force = acc_lin_world * idt
    # Torque on body 2: world-frame angular impulse ``J acc_rot_quat``
    # (negated for body 2; sign matches the iterate path) divided by
    # ``dt``, plus the moment of the linear force about body 2's COM.
    angular_impulse_world = j_rot @ acc_rot_quat
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

    Force is the linear constraint impulse divided by the substep
    ``dt`` (``idt = 1 / substep_dt``); torque is that force's moment
    about body 2's COM plus the angular block accumulated impulse /
    dt. Per-axis caps show up here as saturated (``+/- max_force``)
    components in the reported wrench, decomposed in the world frame.
    """
    return d6_world_wrench_at(constraints, cid, 0, idt)
