# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unified ball-socket / revolute / prismatic / fixed / cable joint with
optional PD drive + limit. Runtime ``joint_mode`` picks the locked DoF set;
everything else (soft-constraint plumbing, warm-starting, Schur blocks) is shared.

* BALL_SOCKET — 3-row point lock at anchor1.
* REVOLUTE — 5-DoF hinge about ``anchor2 - anchor1``; 3x3 + 2x2 Schur. Drive/
  limit act on twist (rad, N·m).
* PRISMATIC — 5-DoF slider along the same axis; 4x4 + 1x1 Schur. Drive/limit
  act on slide (m, N).
* FIXED — 6-DoF weld (3+2+1).
* CABLE — soft FIXED with PD bend (anchor-2 tangents) and PD twist (anchor-3
  scalar). Gains rescaled by 1/rest_length^2.

Drive: PD; POSITION/VELOCITY chooses target vs target_velocity in the bias.
``max_force_drive=0`` means unlimited (POSITION) or disabled (VELOCITY).
Limit: unilateral [min, max]; ``min > max`` disables. Either limit PD gain > 0
selects the PD path; otherwise Box2D (hertz_limit, damping_ratio_limit).
Drive and limit share the same scalar row; limit always wins.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import MOTION_STATIC, BodyContainer, body_set_access_mode, mat33_from_sym6
from newton._src.solvers.phoenx.constraints.constraint_container import (
    _PD_NYQUIST_HEADROOM_MAX,
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    pd_coefficients,
    read_float,
    read_int,
    read_mat33,
    read_mat44,
    read_quat,
    read_vec3,
    read_vec4,
    read_vec6,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_mat44,
    write_quat,
    write_vec3,
    write_vec4,
    write_vec6,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.helpers.math_helpers import (
    create_orthonormal,
    extract_rotation_angle,
    inv_sym2,
    inv_sym3,
    mul_sym2,
    mul_sym3,
    revolution_tracker_angle,
    revolution_tracker_update,
    sym6_from_mat33_upper,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    read_angular_velocity_unified,
    read_velocity_unified,
    write_angular_velocity_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_BOOST_CABLE_BEND,
    PHOENX_BOOST_CABLE_TWIST,
    PHOENX_BOOST_PRISMATIC_DRIVE,
    PHOENX_BOOST_PRISMATIC_LIMIT,
    PHOENX_BOOST_REVOLUTE_DRIVE,
    PHOENX_BOOST_REVOLUTE_LIMIT,
    PHOENX_FRICTION_SLIP_VELOCITY,
)

__all__ = [
    "ADBS_DWORDS",
    "ADBS_TIME_US_OFFSET",
    "DRIVE_MODE_OFF",
    "DRIVE_MODE_POSITION",
    "DRIVE_MODE_VELOCITY",
    "JOINT_MODE_BALL_SOCKET",
    "JOINT_MODE_CABLE",
    "JOINT_MODE_CYLINDRICAL",
    "JOINT_MODE_FIXED",
    "JOINT_MODE_PLANAR",
    "JOINT_MODE_PRISMATIC",
    "JOINT_MODE_REVOLUTE",
    "JOINT_MODE_UNIVERSAL",
    "ActuatedDoubleBallSocketData",
    "actuated_double_ball_socket_cached_warmstart",
    "actuated_double_ball_socket_clear_reset_worlds",
    "actuated_double_ball_socket_initialize_kernel",
    "actuated_double_ball_socket_iterate",
    "actuated_double_ball_socket_iterate_at",
    "actuated_double_ball_socket_iterate_multi",
    "actuated_double_ball_socket_prepare_for_iteration",
    "actuated_double_ball_socket_prepare_for_iteration_at",
    "actuated_double_ball_socket_world_error",
    "actuated_double_ball_socket_world_error_at",
    "actuated_double_ball_socket_world_wrench",
    "actuated_double_ball_socket_world_wrench_at",
    "revolute_cached_warmstart",
    "revolute_iterate",
    "revolute_iterate_multi",
    "revolute_prepare_for_iteration",
]


# ---------------------------------------------------------------------------
# Joint-mode tags
# ---------------------------------------------------------------------------

#: Revolute (hinge) joint: locks 3 translational + 2 rotational DoF.
#: The free DoF is rotation about ``n_hat``.
JOINT_MODE_REVOLUTE = wp.constant(wp.int32(0))
#: Prismatic (slider) joint: locks 3 rotational + 2 translational DoF.
#: The free DoF is translation along ``n_hat``.
JOINT_MODE_PRISMATIC = wp.constant(wp.int32(1))
#: Ball-socket joint: locks 3 translational DoF at ``anchor1``; all
#: 3 rotational DoF are free. No ``anchor2``, no drive, no limit.
JOINT_MODE_BALL_SOCKET = wp.constant(wp.int32(2))
#: Fixed (weld) joint: locks all 6 relative DoFs. Implemented as
#: REVOLUTE's anchor-1 3-row point lock + anchor-2 tangent 2-row lock
#: + PRISMATIC's anchor-3 scalar 1-row lock, solved in block
#: Gauss-Seidel. No drive, no limit. All three anchors are snapshotted
#: in the column at init regardless of mode, so no extra state.
JOINT_MODE_FIXED = wp.constant(wp.int32(3))
#: Cable (soft fixed): rigid anchor-1 ball-socket + PD spring-damper
#: on anchor-2 tangent rows (``k_bend, d_bend``) + PD spring-damper on
#: anchor-3 scalar row (``k_twist, d_twist``). Block Gauss-Seidel
#: across the three blocks, independent per-block soft coefficients.
#: Converges to REVOLUTE as ``k_bend -> inf`` and to FIXED as
#: ``k_twist -> inf``.
#:
#: User gains in rotational SI units, rescaled to positional springs
#: via the lever arm ``rest_length``:
#:
#:   * ``k_bend`` [N*m/rad], ``d_bend`` [N*m*s/rad] -- anchor-2
#:     positional spring with ``k_pos = k_bend / rest_length^2``.
#:   * ``k_twist`` [N*m/rad], ``d_twist`` [N*m*s/rad] -- anchor-3
#:     scalar spring along ``t2`` with the same ``1/rest_length^2``
#:     rescale.
#:
#: Slot reuse (no schema growth): drive_* aliases bend_*, limit_*
#: aliases twist_*, ``s_inv`` mat33 packs the PD soft cache
#: (dwords 0..3 = K22_inv, 4 = gamma_bend, 5 = M_twist_soft,
#: 6 = gamma_twist), ``bias3`` carries the twist bias.
JOINT_MODE_CABLE = wp.constant(wp.int32(4))
#: Universal (Hooke) joint: locks anchor translation and one angular
#: twist axis. D6-dispatched universal joints may also carry angular
#: limit rows on their two free axes.
JOINT_MODE_UNIVERSAL = wp.constant(wp.int32(5))
JOINT_MODE_CYLINDRICAL = wp.constant(wp.int32(6))
JOINT_MODE_PLANAR = wp.constant(wp.int32(7))

# Per-anchor solve kinds for the unified D6 row engine. Each anchor block
# in :func:`_d6_iterate_rows_at` selects one; the math lives once in the
# shared ``_d6_solve_anchor*`` helpers.
_D6_ROW_SOLVE_SKIP = wp.constant(wp.int32(0))
_D6_ROW_SOLVE_HARD3 = wp.constant(wp.int32(1))  # anchor-1 sym6 point lock (ball/universal)
_D6_ROW_SOLVE_SOFT3 = wp.constant(wp.int32(2))  # anchor-1 mat33 Box2D-soft lock (cable)
_D6_ROW_SOLVE_PD2_TAN = wp.constant(wp.int32(4))  # anchor-2 PD tangent (cable bend)
_D6_ROW_SOLVE_HARD1_SCALAR = wp.constant(wp.int32(5))  # anchor-3 scalar twist lock (fixed)
_D6_ROW_SOLVE_PD1_SCALAR = wp.constant(wp.int32(6))  # anchor-3 PD scalar (cable twist)

# Axial drive / limit row kinds.
_D6_AXIAL_NONE = wp.constant(wp.int32(0))
_D6_AXIAL_ANGULAR = wp.constant(wp.int32(1))  # twist about n_hat (revolute/universal)
_D6_AXIAL_LINEAR = wp.constant(wp.int32(2))  # slide along n_hat (prismatic)


# ---------------------------------------------------------------------------
# Drive-mode tags
# ---------------------------------------------------------------------------

#: No actuation along the free DoF.
DRIVE_MODE_OFF = wp.constant(wp.int32(0))
#: PD spring-damper towards ``target`` (rad for revolute, m for
#: prismatic). Caller must supply ``stiffness_drive`` / ``damping_drive``
#: as SI gains [N/m, N*s/m] or [N*m/rad, N*m*s/rad].
DRIVE_MODE_POSITION = wp.constant(wp.int32(1))
#: PD velocity servo tracking ``target_velocity`` (rad/s or m/s).
#: The spring term is disabled (``stiffness_drive = 0``); caller must
#: supply ``damping_drive > 0`` [N*s/m or N*m*s/rad], which acts as the
#: proportional gain on velocity error. ``max_force_drive`` optionally
#: clamps the per-substep impulse (N*s or N*m*s). There is no rigid
#: pure-velocity-motor fallback when ``damping_drive == 0``;
#: :meth:`WorldBuilder.add_joint` rejects that up front.
DRIVE_MODE_VELOCITY = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Limit-clamp state tags (mirrors constraint_hinge_angle's _CLAMP_*).
# ---------------------------------------------------------------------------

_CLAMP_NONE = wp.constant(wp.int32(0))
_CLAMP_MAX = wp.constant(wp.int32(1))
_CLAMP_MIN = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class ActuatedDoubleBallSocketData:
    """Per-constraint dword-layout schema for the unified joint.

    Union over revolute / prismatic / ball-socket / fixed / cable.
    Mode-specific Schur caches live in dedicated slots; the rest is
    shared. ~80 dwords (~320 B per joint). See field-level ``#:``
    comments for individual slot semantics.
    """

    # ---- Header -------------------------------------------------------
    constraint_type: wp.int32
    body1: wp.int32
    body2: wp.int32

    # ---- Shared positional block -------------------------------------
    joint_mode: wp.int32
    local_anchor1_b1: wp.vec3f
    local_anchor1_b2: wp.vec3f
    local_anchor2_b1: wp.vec3f
    local_anchor2_b2: wp.vec3f
    # Runtime (per-substep) lever arms for the two shared anchors.
    r1_b1: wp.vec3f
    r1_b2: wp.vec3f
    r2_b1: wp.vec3f
    r2_b2: wp.vec3f
    # Runtime tangent basis perpendicular to the current world joint axis.
    t1: wp.vec3f
    t2: wp.vec3f
    # Positional soft-constraint knobs + cached per-substep coefficients.
    hertz: wp.float32
    damping_ratio: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32
    # Positional biases at anchors 1+2 (prismatic anchor-3 bias lives
    # in ``mode_extras`` -- mode-exclusive with the revolute tracker).
    # Revolute:  bias1 = world drift at a1; bias2 = a2 tangent drift (t1,t2,0).
    # Prismatic: bias1, bias2 = a1, a2 tangent drifts (t1,t2,0).
    bias1: wp.vec3f
    bias2: wp.vec3f
    # Mode-specific Schur cache, aliased onto one 27-dword block sized
    # for the larger mode (joint mode is fixed at construction).
    # Reads/writes go through :func:`_read_revo_*` / :func:`_read_pris_*`.
    #
    # Revolute  (27 used): [0..8] a1_inv mat33, [9..17] ut_ai mat33,
    #                      [18..26] s_inv mat33.
    # Prismatic (21 used, 6 unused tail): [0..15] a4_inv mat44,
    #                      [16..19] c_pris vec4, [20] s_scalar_inv.
    mode_cache: wp.types.vector(length=27, dtype=wp.float32)
    # Mode-specific extras, same alias trick. 16 dwords sized for the
    # larger (prismatic) layout.
    #
    # Prismatic (16 used): [0..2] local_anchor3_b1, [3..5] local_anchor3_b2,
    #     [6..8] r3_b1, [9..11] r3_b2, [12..14] accumulated_impulse3,
    #     [15] bias3.
    # Revolute  (6 used, 10 unused tail):
    #     [0..3] inv_initial_orientation (quat),
    #     [4] revolution_counter, [5] previous_quaternion_angle.
    mode_extras: wp.types.vector(length=16, dtype=wp.float32)
    # Warm-start accumulated impulses for the shared anchors. The
    # third (prismatic-only) impulse moved into ``mode_extras`` above.
    accumulated_impulse1: wp.vec3f
    accumulated_impulse2: wp.vec3f

    # ---- Actuator + limit block --------------------------------------
    # Body-1-local joint axis snapshot. Used by revolute for a
    # single-axis Jacobian (matching the standalone angular motor /
    # angular limit's PD path) and by the world_wrench helper. The
    # companion 5-DoF positional lock keeps body 2's axis parallel, so
    # one axis is both simpler and more numerically stable than the
    # old two-axis projection.
    axis_local1: wp.vec3f
    rest_length: wp.float32
    # NB: ``inv_initial_orientation``, ``revolution_counter``, and
    # ``previous_quaternion_angle`` (revolute twist-tracker scratch)
    # used to live here as separate fields. They've been folded into
    # the ``mode_extras`` alias block above so prismatic joints don't
    # carry 6 unused dwords.
    drive_mode: wp.int32
    # Setpoints: ``target`` is radians (revolute) or meters (prismatic);
    # ``target_velocity`` is rad/s or m/s.
    target: wp.float32
    target_velocity: wp.float32
    max_force_drive: wp.float32
    # Drive parameters: normal PD only. ``stiffness_drive`` = kp [N/m or
    # N*m/rad], ``damping_drive`` = kd [N*s/m or N*m*s/rad]. Both zero
    # disables the drive row regardless of ``drive_mode`` -- matches
    # Jitter2's LinearMotor / AngularMotor short-circuit. See
    # :func:`pd_coefficients` for the implicit-Euler math. The Nyquist
    # headroom multiplier on this row is a compile-time constant
    # in :mod:`solver_config` (per joint type / per row); column
    # storage avoided to keep the constraint footprint compact.
    stiffness_drive: wp.float32
    damping_drive: wp.float32
    # Joint-axis armature (rotor / leadscrew inertia) [kg*m^2 for revolute,
    # kg for prismatic]. Increases the *axial* effective inertia seen by
    # the drive and limit rows, but not the rigid 5-row positional lock.
    # Equivalent to MuJoCo's ``mjOption.armature`` / ``joint armature`` in
    # reduced coordinates: ``M_eff_axial = M_chain_axial + armature``.
    # Critical for stability of high-stiffness PD drives on chains where
    # an intermediate link has near-zero inertia about the joint axis
    # (e.g. humanoid waist-yaw / waist-roll links). ``0`` disables.
    armature: wp.float32
    # Joint-axis Coulomb friction limit [N*m for revolute, N for
    # prismatic]. Implemented as a saturated soft row on the same axial
    # Jacobian as the drive / limit rows.
    friction_coefficient: wp.float32
    # MuJoCo-style slip scale for the friction row. Positive values are
    # multiplied by the current axial inverse effective mass and friction
    # limit to get the slip velocity; non-positive values use
    # :data:`PHOENX_FRICTION_SLIP_VELOCITY` as a solver fallback.
    friction_slip_scale: wp.float32
    # Limit window: rad (revolute) or m (prismatic). ``min_value >
    # max_value`` disables the limit (matches the standalone
    # angular_limit / linear_limit sentinel).
    min_value: wp.float32
    max_value: wp.float32
    # Limit softness: dual parameterisation -- if either
    # ``stiffness_limit`` or ``damping_limit`` is strictly positive the
    # row uses the Jitter2 PD spring-damper path; otherwise it falls
    # back to Box2D ``(hertz_limit, damping_ratio_limit)``. Same
    # discriminator as the standalone angular_limit / linear_limit
    # (``stiffness > 0 or damping > 0 -> PD``).
    hertz_limit: wp.float32
    damping_ratio_limit: wp.float32
    stiffness_limit: wp.float32
    damping_limit: wp.float32
    # Cached per-substep coefficients (drive row is always PD):
    #   gamma_drive         -- 1 / (kd + kp*dt).
    #   bias_drive          -- Jitter2 ``beta * C / dt`` minus
    #                          ``target_velocity``.
    #   eff_mass_drive_soft -- 1 / (J M^-1 J^T + gamma_drive).
    # Cached scalar inverse effective mass for the axial row,
    # ``J M^-1 J^T`` (used by the Box2D limit path).
    eff_inv_axial: wp.float32
    bias_drive: wp.float32
    gamma_drive: wp.float32
    eff_mass_drive_soft: wp.float32
    # Aliased per-substep limit cache: 3 dwords shared between the
    # Box2D and PD limit formulations. The discriminator is
    # ``stiffness_limit > 0 or damping_limit > 0`` -> PD, else Box2D;
    # the choice is fixed once stiffness_limit / damping_limit are set
    # at construction, so the two layouts never collide.
    #
    # Box2D layout: [bias_limit_box2d, mass_coeff_limit, impulse_coeff_limit]
    # PD layout:    [pd_gamma_limit,   pd_beta_limit,    pd_mass_coeff_limit]
    limit_cache: wp.types.vector(length=3, dtype=wp.float32)
    clamp: wp.int32
    # Cached world-frame joint axis from the most recent prepare-pass.
    axis_world: wp.vec3f
    accumulated_impulse_drive: wp.float32
    accumulated_impulse_limit: wp.float32
    accumulated_impulse_friction: wp.float32

    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(ActuatedDoubleBallSocketData)


# Dword offsets derived once from the schema. Named per field.
_OFF_BODY1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body2"))
_OFF_JOINT_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "joint_mode"))
_OFF_LA1_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor1_b1"))
_OFF_LA1_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor1_b2"))
_OFF_LA2_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor2_b1"))
_OFF_LA2_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor2_b2"))
_OFF_R1_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r1_b1"))
_OFF_R1_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r1_b2"))
_OFF_R2_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r2_b1"))
_OFF_R2_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r2_b2"))
_OFF_T1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "t2"))
_OFF_HERTZ = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff"))
_OFF_BIAS1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias2"))
# Aliased mode-specific Schur cache. Revolute uses dwords [0..27),
# prismatic uses [0..21) of the same 27-dword block. Joint mode is
# fixed at construction so the two layouts never collide.
_OFF_MODE_CACHE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mode_cache"))
_OFF_A1_INV = wp.constant(int(_OFF_MODE_CACHE) + 0)
_OFF_UT_AI = wp.constant(int(_OFF_MODE_CACHE) + 9)
_OFF_S_INV = wp.constant(int(_OFF_MODE_CACHE) + 18)
# Compressed rigid-family Schur cache (BALL / REVOLUTE / FIXED / UNIVERSAL).
# Symmetric-aware packing of the same Schur quantities the mat33 layout
# above stored: a1_inv as sym6 (upper triangle), ut_ai as two vec3 rows
# (2x3, not symmetric), s_inv (the 2x2 swing Schur) as sym3 (m00, m01, m11).
# Laid out in dwords [0..15) of mode_cache, clear of FIXED's
# ``_OFF_S_SCALAR_INV`` (dword 20). Rigid modes never coexist with
# prismatic / cable on a cid, so this overlaps their layouts harmlessly.
_OFF_A1_INV_S6 = wp.constant(int(_OFF_MODE_CACHE) + 0)
_OFF_UT_AI_ROW0 = wp.constant(int(_OFF_MODE_CACHE) + 6)
_OFF_UT_AI_ROW1 = wp.constant(int(_OFF_MODE_CACHE) + 9)
_OFF_S_INV_S3 = wp.constant(int(_OFF_MODE_CACHE) + 12)
_OFF_S_SCALAR_INV = wp.constant(int(_OFF_MODE_CACHE) + 20)
# Prismatic coupled 4+1 Schur cache (the convergent slider formulation):
#   dwords 0..15 = a4_inv  (4x4 tangent-block inverse for a1+a2 tangents)
#   dwords 16..19 = c_pris (vec4 coupling of the 4 tangent rows to a3)
#   dword 20      = s_scalar_inv (a3 twist Schur scalar; reuses FIXED slot)
# Overlaps the rigid sym6/sym3 layout harmlessly -- prismatic never shares
# a cid with the rigid family.
_OFF_A4_INV = wp.constant(int(_OFF_MODE_CACHE) + 0)
_OFF_C_PRIS = wp.constant(int(_OFF_MODE_CACHE) + 16)
# Aliased mode-extras block. Prismatic packs anchor-3 / r3 / acc_imp3
# / bias3 (16 dwords); revolute packs the twist-tracker scratch
# (inv_initial_orientation + revolution_counter + previous_quaternion_angle
# = 6 dwords). Mutually exclusive, so we share the 16-dword block.
_OFF_MODE_EXTRAS = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mode_extras"))
# Prismatic-only fields, dwords 0..15 of mode_extras:
_OFF_LA3_B1 = wp.constant(int(_OFF_MODE_EXTRAS) + 0)
_OFF_LA3_B2 = wp.constant(int(_OFF_MODE_EXTRAS) + 3)
_OFF_R3_B1 = wp.constant(int(_OFF_MODE_EXTRAS) + 6)
_OFF_R3_B2 = wp.constant(int(_OFF_MODE_EXTRAS) + 9)
_OFF_ACC_IMP3 = wp.constant(int(_OFF_MODE_EXTRAS) + 12)
_OFF_BIAS3 = wp.constant(int(_OFF_MODE_EXTRAS) + 15)
# Revolute / universal fields, dwords 0..5 of mode_extras (10 unused tail):
_OFF_INV_INITIAL_ORIENTATION = wp.constant(int(_OFF_MODE_EXTRAS) + 0)
_OFF_REVOLUTION_COUNTER = wp.constant(int(_OFF_MODE_EXTRAS) + 4)
_OFF_PREVIOUS_QUATERNION_ANGLE = wp.constant(int(_OFF_MODE_EXTRAS) + 5)
# BALL / UNIVERSAL D6 angular limit aliases, dwords 6..15 of mode_extras.
_OFF_D6_LIMIT_LOWER = wp.constant(int(_OFF_MODE_EXTRAS) + 6)
_OFF_D6_LIMIT_UPPER = wp.constant(int(_OFF_MODE_EXTRAS) + 9)
_OFF_D6_LIMIT_COUNT = wp.constant(int(_OFF_MODE_EXTRAS) + 12)
_OFF_D6_LIMIT_EFF_INV = wp.constant(int(_OFF_MODE_EXTRAS) + 13)
# Cable-only PD soft-cache aliases over the existing ``s_inv`` mat33 slot
# (9 dwords). Cable never uses the 3+2 Schur, so the revolute / fixed
# layout for these dwords is free to reinterpret here.
#   dwords 0..3 = K22_soft inverse (2x2 packed: m00, m01, m10, m11)
#   dword 4     = gamma_bend       (PD softness coefficient, anchor-2 PD rows)
#   dword 5     = M_twist_soft     (PD softened effective mass for anchor-3 row)
#   dword 6     = gamma_twist      (PD softness coefficient, anchor-3 PD row)
#   dwords 7..8 = unused
_OFF_CABLE_K22_INV_00 = wp.constant(int(_OFF_S_INV) + 0)
_OFF_CABLE_K22_INV_01 = wp.constant(int(_OFF_S_INV) + 1)
_OFF_CABLE_K22_INV_10 = wp.constant(int(_OFF_S_INV) + 2)
_OFF_CABLE_K22_INV_11 = wp.constant(int(_OFF_S_INV) + 3)
_OFF_CABLE_GAMMA_BEND = wp.constant(int(_OFF_S_INV) + 4)
_OFF_CABLE_M_TWIST_SOFT = wp.constant(int(_OFF_S_INV) + 5)
_OFF_CABLE_GAMMA_TWIST = wp.constant(int(_OFF_S_INV) + 6)

_OFF_ACC_IMP1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse1"))
_OFF_ACC_IMP2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse2"))

_OFF_AXIS_LOCAL1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_local1"))
_OFF_REST_LENGTH = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "rest_length"))
_OFF_DRIVE_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "drive_mode"))
_OFF_TARGET = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target"))
_OFF_TARGET_VELOCITY = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target_velocity"))
_OFF_MAX_FORCE_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_force_drive"))
_OFF_STIFFNESS_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_drive"))
_OFF_DAMPING_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_drive"))
_OFF_ARMATURE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "armature"))
_OFF_FRICTION_COEFFICIENT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_coefficient"))
_OFF_FRICTION_SLIP_SCALE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_slip_scale"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_value"))
_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz_limit"))
_OFF_DAMPING_RATIO_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio_limit"))
_OFF_STIFFNESS_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_limit"))
_OFF_DAMPING_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_limit"))
_OFF_EFF_INV_AXIAL = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "eff_inv_axial"))
_OFF_BIAS_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_drive"))
_OFF_GAMMA_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "gamma_drive"))
_OFF_EFF_MASS_DRIVE_SOFT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "eff_mass_drive_soft"))
# Aliased Box2D / PD limit cache: 3 shared dwords. Layouts:
#   Box2D: [bias_limit_box2d, mass_coeff_limit, impulse_coeff_limit]
#   PD:    [pd_gamma_limit,   pd_beta_limit,    pd_mass_coeff_limit]
_OFF_LIMIT_CACHE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "limit_cache"))
_OFF_BIAS_LIMIT_BOX2D = wp.constant(int(_OFF_LIMIT_CACHE) + 0)
_OFF_MASS_COEFF_LIMIT = wp.constant(int(_OFF_LIMIT_CACHE) + 1)
_OFF_IMPULSE_COEFF_LIMIT = wp.constant(int(_OFF_LIMIT_CACHE) + 2)
_OFF_PD_GAMMA_LIMIT = wp.constant(int(_OFF_LIMIT_CACHE) + 0)
_OFF_PD_BETA_LIMIT = wp.constant(int(_OFF_LIMIT_CACHE) + 1)
_OFF_PD_MASS_COEFF_LIMIT = wp.constant(int(_OFF_LIMIT_CACHE) + 2)
_OFF_CLAMP = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "clamp"))
_OFF_AXIS_WORLD = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_world"))
_OFF_ACC_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_drive"))
_OFF_ACC_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_limit"))
_OFF_ACC_FRICTION = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_friction"))
ADBS_TIME_US_OFFSET = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "time_us"))

#: Total dword count of one unified joint constraint.
ADBS_DWORDS: int = num_dwords(ActuatedDoubleBallSocketData)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False, module="unique")
def actuated_double_ball_socket_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor1: wp.array[wp.vec3f],
    anchor2: wp.array[wp.vec3f],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
    joint_mode: wp.array[wp.int32],
    drive_mode: wp.array[wp.int32],
    target: wp.array[wp.float32],
    target_velocity: wp.array[wp.float32],
    max_force_drive: wp.array[wp.float32],
    stiffness_drive: wp.array[wp.float32],
    damping_drive: wp.array[wp.float32],
    min_value: wp.array[wp.float32],
    max_value: wp.array[wp.float32],
    hertz_limit: wp.array[wp.float32],
    damping_ratio_limit: wp.array[wp.float32],
    stiffness_limit: wp.array[wp.float32],
    damping_limit: wp.array[wp.float32],
    armature: wp.array[wp.float32],
    friction_coefficient: wp.array[wp.float32],
    friction_slip_scale: wp.array[wp.float32],
    d6_limit_axis0: wp.array[wp.vec3f],
    d6_limit_axis1: wp.array[wp.vec3f],
    d6_limit_axis2: wp.array[wp.vec3f],
    d6_limit_lower: wp.array[wp.vec3f],
    d6_limit_upper: wp.array[wp.vec3f],
    d6_limit_count: wp.array[wp.int32],
):
    """Pack one batch of unified joint descriptors.

    ``anchor1`` / ``anchor2`` are two world-space points on the joint
    axis: the line through them is the hinge axis (revolute) or slide
    axis (prismatic). Prismatic init auto-derives a third anchor
    ``a3 = anchor1 + |a2 - a1| * t_ref`` (``t_ref`` arbitrary unit
    perp to ``n_hat_init``) and snapshots it into both body frames.

    Args:
        constraints: Column-major constraint storage.
        bodies: Only ``position`` / ``orientation`` of referenced
            bodies are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1, body2: Body indices [num_in_batch].
        anchor1, anchor2: World-space anchors [m] defining the axis.
        hertz, damping_ratio: Positional Schur block soft-constraint
            knobs.
        joint_mode: :data:`JOINT_MODE_REVOLUTE` or
            :data:`JOINT_MODE_PRISMATIC`.
        drive_mode: :data:`DRIVE_MODE_OFF` / ``_POSITION`` / ``_VELOCITY``.
        target: Position setpoint [rad or m].
        target_velocity: Velocity setpoint [rad/s or m/s].
        max_force_drive: Drive impulse cap [N*m or N]; ``0`` disables.
        stiffness_drive, damping_drive: Drive PD gains in absolute SI
            units; both ``0`` disables the drive row. CABLE mode
            reuses these slots for ``bend_stiffness`` / ``bend_damping``.
        min_value, max_value: Limit window [rad or m]; ``min > max``
            disables the limit.
        hertz_limit, damping_ratio_limit: Box2D-style limit knobs;
            used iff ``stiffness_limit == damping_limit == 0``.
        stiffness_limit, damping_limit: PD limit gains (absolute SI).
            If either > 0 the limit uses the Jitter2 spring-damper
            path and the Box2D knobs are ignored. CABLE mode reuses
            these slots for ``twist_stiffness`` / ``twist_damping``.
        armature: Joint-axis armature [kg*m^2 for revolute, kg for
            prismatic]. Adds to the axial effective inertia for the
            drive and limit rows; ``0`` disables.
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a1_w = anchor1[tid]
    a2_w = anchor2[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]

    # ---- Anchor 1 / anchor 2 body-local snapshots (both modes) ------
    la1_b1 = wp.quat_rotate_inv(orient1, a1_w - pos1)
    la1_b2 = wp.quat_rotate_inv(orient2, a1_w - pos2)
    la2_b1 = wp.quat_rotate_inv(orient1, a2_w - pos1)
    la2_b2 = wp.quat_rotate_inv(orient2, a2_w - pos2)

    # ---- Joint axis snapshot ----------------------------------------
    axis_world = a2_w - a1_w
    axis_len2 = wp.dot(axis_world, axis_world)
    if axis_len2 > 1.0e-20:
        rest_length = wp.sqrt(axis_len2)
        n_hat_init = axis_world / rest_length
    else:
        rest_length = 1.0
        n_hat_init = wp.vec3f(1.0, 0.0, 0.0)

    axis_local1 = wp.quat_rotate_inv(orient1, n_hat_init)
    # Rest relative orientation used by the revolute twist tracker.
    # ``diff = q2 * inv_initial_orientation * q1^*`` is the identity at
    # finalize() time, so the revolution-counter starts in-branch at 0.
    # Matches the standalone angular motor / angular limit exactly.
    inv_initial_orientation = wp.quat_inverse(orient2) * orient1

    # ---- Anchor 3 auto-derivation (prismatic only) -------------------
    # Pick any unit perpendicular to the slide axis, offset anchor 1 by
    # ``rest_length`` along it. Body-local snapshot so the runtime math
    # can rotate anchor 3 with each body independently.
    t_ref_init = create_orthonormal(n_hat_init)
    a3_w = a1_w + rest_length * t_ref_init
    la3_b1 = wp.quat_rotate_inv(orient1, a3_w - pos1)
    la3_b2 = wp.quat_rotate_inv(orient2, a3_w - pos2)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET)
    mode = joint_mode[tid]

    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
    write_int(constraints, _OFF_JOINT_MODE, cid, mode)
    write_vec3(constraints, _OFF_LA1_B1, cid, la1_b1)
    write_vec3(constraints, _OFF_LA1_B2, cid, la1_b2)
    write_vec3(constraints, _OFF_LA2_B1, cid, la2_b1)
    write_vec3(constraints, _OFF_LA2_B2, cid, la2_b2)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    write_vec3(constraints, _OFF_R1_B1, cid, zero3)
    write_vec3(constraints, _OFF_R1_B2, cid, zero3)
    write_vec3(constraints, _OFF_R2_B1, cid, zero3)
    write_vec3(constraints, _OFF_R2_B2, cid, zero3)
    write_vec3(constraints, _OFF_T1, cid, zero3)
    write_vec3(constraints, _OFF_T2, cid, zero3)
    write_vec3(constraints, _OFF_BIAS1, cid, zero3)
    write_vec3(constraints, _OFF_BIAS2, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP1, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP2, cid, zero3)

    # ``mode_extras`` block is mode-aliased: REVOLUTE / UNIVERSAL store the
    # twist-tracker scratch (inv_initial_orientation, revolution_counter,
    # previous_quaternion_angle); PRISMATIC / FIXED / CABLE store the
    # anchor-3 snapshot + bias3 + acc_imp3. Writing both layouts
    # unconditionally would clobber the alias, so we branch.
    if mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE:
        write_vec3(constraints, _OFF_LA3_B1, cid, la3_b1)
        write_vec3(constraints, _OFF_LA3_B2, cid, la3_b2)
        write_vec3(constraints, _OFF_R3_B1, cid, zero3)
        write_vec3(constraints, _OFF_R3_B2, cid, zero3)
        write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, 0.0)
    else:
        # REVOLUTE / BALL_SOCKET / UNIVERSAL: zero out the anchor-3 slots
        # via the twist-tracker layout. BALL_SOCKET only reads this when
        # it carries D6 angular limit rows.
        write_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid, inv_initial_orientation)
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, 0)
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, 0.0)

    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, 0.0)

    # Defensive identity init of the aliased mode_cache (dwords 0..26).
    # Three eye3 writes blanket the whole block; the per-mode prepare
    # overwrites the slots it actually uses (rigid sym6 / swing sym3,
    # prismatic tangent sym3 blocks, cable PD inverses).
    eye3 = wp.identity(3, dtype=wp.float32)
    write_mat33(constraints, _OFF_A1_INV, cid, eye3)
    write_mat33(constraints, _OFF_UT_AI, cid, eye3)
    write_mat33(constraints, _OFF_S_INV, cid, eye3)
    write_float(constraints, _OFF_S_SCALAR_INV, cid, 0.0)

    # Actuator block. Twist-tracker init (inv_initial_orientation +
    # revolution_counter + previous_quaternion_angle) ran in the
    # mode-conditional block above since those fields share dwords
    # with the prismatic anchor-3 snapshot.
    write_vec3(constraints, _OFF_AXIS_LOCAL1, cid, axis_local1)
    write_float(constraints, _OFF_REST_LENGTH, cid, rest_length)
    write_int(constraints, _OFF_DRIVE_MODE, cid, drive_mode[tid])
    write_float(constraints, _OFF_TARGET, cid, target[tid])
    write_float(constraints, _OFF_TARGET_VELOCITY, cid, target_velocity[tid])
    write_float(constraints, _OFF_MAX_FORCE_DRIVE, cid, max_force_drive[tid])
    write_float(constraints, _OFF_STIFFNESS_DRIVE, cid, stiffness_drive[tid])
    write_float(constraints, _OFF_DAMPING_DRIVE, cid, damping_drive[tid])
    write_float(constraints, _OFF_ARMATURE, cid, armature[tid])
    write_float(constraints, _OFF_FRICTION_COEFFICIENT, cid, friction_coefficient[tid])
    write_float(constraints, _OFF_FRICTION_SLIP_SCALE, cid, friction_slip_scale[tid])
    write_float(constraints, _OFF_MIN_VALUE, cid, min_value[tid])
    write_float(constraints, _OFF_MAX_VALUE, cid, max_value[tid])
    write_float(constraints, _OFF_HERTZ_LIMIT, cid, hertz_limit[tid])
    write_float(constraints, _OFF_DAMPING_RATIO_LIMIT, cid, damping_ratio_limit[tid])
    write_float(constraints, _OFF_STIFFNESS_LIMIT, cid, stiffness_limit[tid])
    write_float(constraints, _OFF_DAMPING_LIMIT, cid, damping_limit[tid])
    write_float(constraints, _OFF_EFF_INV_AXIAL, cid, 0.0)
    write_float(constraints, _OFF_BIAS_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_GAMMA_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_EFF_MASS_DRIVE_SOFT, cid, 0.0)
    # ``limit_cache`` is mode-aliased Box2D vs PD; one zero-fill of
    # the 3 shared dwords covers both layouts. Prepare overwrites
    # them every substep based on the limit type.
    write_float(constraints, _OFF_LIMIT_CACHE + 0, cid, 0.0)
    write_float(constraints, _OFF_LIMIT_CACHE + 1, cid, 0.0)
    write_float(constraints, _OFF_LIMIT_CACHE + 2, cid, 0.0)
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, n_hat_init)
    write_float(constraints, _OFF_ACC_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_ACC_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_ACC_FRICTION, cid, 0.0)

    if mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL:
        count = d6_limit_count[tid]
        write_vec3(constraints, _OFF_D6_LIMIT_LOWER, cid, d6_limit_lower[tid])
        write_vec3(constraints, _OFF_D6_LIMIT_UPPER, cid, d6_limit_upper[tid])
        write_int(constraints, _OFF_D6_LIMIT_COUNT, cid, count)
        write_vec3(constraints, _OFF_D6_LIMIT_EFF_INV, cid, zero3)
        if count > wp.int32(0):
            if mode == JOINT_MODE_BALL_SOCKET:
                write_vec3(constraints, _OFF_AXIS_LOCAL1, cid, wp.quat_rotate_inv(orient1, d6_limit_axis0[tid]))
                if count > wp.int32(1):
                    write_vec3(constraints, _OFF_LA2_B1, cid, wp.quat_rotate_inv(orient1, d6_limit_axis1[tid]))
                if count > wp.int32(2):
                    write_vec3(constraints, _OFF_LA2_B2, cid, wp.quat_rotate_inv(orient1, d6_limit_axis2[tid]))
            else:
                write_vec3(constraints, _OFF_LA2_B1, cid, wp.quat_rotate_inv(orient1, d6_limit_axis0[tid]))
                if count > wp.int32(1):
                    write_vec3(constraints, _OFF_LA2_B2, cid, wp.quat_rotate_inv(orient1, d6_limit_axis1[tid]))


# ---------------------------------------------------------------------------
# Runtime reset
# ---------------------------------------------------------------------------


@wp.func
def _adbs_constraint_world(bodies: BodyContainer, b1: wp.int32, b2: wp.int32) -> wp.int32:
    if b2 >= wp.int32(0) and b2 < bodies.world_id.shape[0] and bodies.motion_type[b2] != MOTION_STATIC:
        return bodies.world_id[b2]
    if b1 >= wp.int32(0) and b1 < bodies.world_id.shape[0] and bodies.motion_type[b1] != MOTION_STATIC:
        return bodies.world_id[b1]
    if b2 >= wp.int32(0) and b2 < bodies.world_id.shape[0]:
        return bodies.world_id[b2]
    if b1 >= wp.int32(0) and b1 < bodies.world_id.shape[0]:
        return bodies.world_id[b1]
    return wp.int32(-1)


@wp.kernel(enable_backward=False)
def _adbs_clear_reset_worlds_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: wp.int32,
    dones: wp.array[wp.float32],
):
    cid = wp.tid()
    if cid >= joint_count:
        return

    world = _adbs_constraint_world(
        bodies,
        read_int(constraints, _OFF_BODY1, cid),
        read_int(constraints, _OFF_BODY2, cid),
    )
    if world < wp.int32(0) or world >= dones.shape[0] or dones[world] <= wp.float32(0.5):
        return

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    write_vec3(constraints, _OFF_R1_B1, cid, zero3)
    write_vec3(constraints, _OFF_R1_B2, cid, zero3)
    write_vec3(constraints, _OFF_R2_B1, cid, zero3)
    write_vec3(constraints, _OFF_R2_B2, cid, zero3)
    write_vec3(constraints, _OFF_T1, cid, zero3)
    write_vec3(constraints, _OFF_T2, cid, zero3)
    write_float(constraints, _OFF_MASS_COEFF, cid, wp.float32(1.0))
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, wp.float32(0.0))
    write_vec3(constraints, _OFF_BIAS1, cid, zero3)
    write_vec3(constraints, _OFF_BIAS2, cid, zero3)

    for row in range(27):
        write_float(constraints, _OFF_MODE_CACHE + row, cid, wp.float32(0.0))

    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE:
        write_vec3(constraints, _OFF_R3_B1, cid, zero3)
        write_vec3(constraints, _OFF_R3_B2, cid, zero3)
        write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, wp.float32(0.0))
    else:
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, wp.int32(0))
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, wp.float32(0.0))
        if mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL:
            write_vec3(constraints, _OFF_D6_LIMIT_EFF_INV, cid, zero3)

    write_vec3(constraints, _OFF_ACC_IMP1, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP2, cid, zero3)
    write_float(constraints, _OFF_EFF_INV_AXIAL, cid, wp.float32(0.0))
    write_float(constraints, _OFF_BIAS_DRIVE, cid, wp.float32(0.0))
    write_float(constraints, _OFF_GAMMA_DRIVE, cid, wp.float32(0.0))
    write_float(constraints, _OFF_EFF_MASS_DRIVE_SOFT, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 0, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 1, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 2, cid, wp.float32(0.0))
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, zero3)
    write_float(constraints, _OFF_ACC_DRIVE, cid, wp.float32(0.0))
    write_float(constraints, _OFF_ACC_LIMIT, cid, wp.float32(0.0))
    write_float(constraints, _OFF_ACC_FRICTION, cid, wp.float32(0.0))


def actuated_double_ball_socket_clear_reset_worlds(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: int,
    dones: wp.array[wp.float32],
    device: wp.DeviceLike = None,
) -> None:
    """Clear ADBS runtime caches and warm starts for reset worlds."""
    count = max(0, min(int(joint_count), int(constraints.data.shape[1])))
    if count == 0:
        return
    wp.launch(
        _adbs_clear_reset_worlds_kernel,
        dim=count,
        inputs=[constraints, bodies, wp.int32(count), dones],
        device=device,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mass-splitting body-pair load/store helpers
#
# All joint iterates/prepares share the same access pattern: load (v, w,
# inv_mass, inv_inertia) for two bodies, do constraint math, write
# (v, w) back. With mass splitting the loads / stores route through the
# slot-aware unified helpers and inv_mass / inv_inertia are scaled by the
# per-body slot count (Tonge effective mass). Disabled-fast-path returns
# slot=-1 / inv_factor=1, so this collapses to the pre-mass-splitting
# bodies.* path without a branch.
#
# Joints connect bodies (never particles), but the unified helpers take
# a ParticleContainer parameter for the body/particle branch. We thread
# it through unchanged; the particle branch is unreachable for
# ``b < num_bodies`` and gets dead-code-eliminated by the runtime.
# ---------------------------------------------------------------------------


@wp.func
def _ms_load_body_pair(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    b1: wp.int32,
    b2: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Slot-aware load of body-pair kinematic state. Returns
    ``(v1, v2, w1, w2, inv_mass1, inv_mass2, inv_inertia1,
    inv_inertia2, slot1, slot2)``. Mass-splitting fast path
    (``highest_index_in_use[0] == 0``) bypasses copy_state + the
    Tonge ``inv_factor`` multiply (4 int reads + 4 FP muls saved
    per sweep).
    """
    if copy_state.highest_index_in_use[0] == wp.int32(0):
        # Mass splitting disabled: direct SoA, no copy_state touch.
        v1 = bodies.velocity[b1]
        v2 = bodies.velocity[b2]
        w1 = bodies.angular_velocity[b1]
        w2 = bodies.angular_velocity[b2]
        inv_mass1 = bodies.inverse_mass[b1]
        inv_mass2 = bodies.inverse_mass[b2]
        inv_inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1])
        inv_inertia2 = mat33_from_sym6(bodies.inverse_inertia_world[b2])
        return (
            v1,
            v2,
            w1,
            w2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            wp.int32(-1),
            wp.int32(-1),
        )
    v1, inv_factor1, slot1 = read_velocity_unified(bodies, particles, copy_state, b1, parallel_id, num_bodies)
    v2, inv_factor2, slot2 = read_velocity_unified(bodies, particles, copy_state, b2, parallel_id, num_bodies)
    w1, _wfb1, _wsb1 = read_angular_velocity_unified(bodies, copy_state, b1, parallel_id, num_bodies)
    w2, _wfb2, _wsb2 = read_angular_velocity_unified(bodies, copy_state, b2, parallel_id, num_bodies)
    inv_f1 = wp.float32(inv_factor1)
    inv_f2 = wp.float32(inv_factor2)
    inv_mass1 = bodies.inverse_mass[b1] * inv_f1
    inv_mass2 = bodies.inverse_mass[b2] * inv_f2
    inv_inertia1 = mat33_from_sym6(bodies.inverse_inertia_world[b1]) * inv_f1
    inv_inertia2 = mat33_from_sym6(bodies.inverse_inertia_world[b2]) * inv_f2
    return v1, v2, w1, w2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, slot1, slot2


@wp.func
def _ms_store_body_pair(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    b1: wp.int32,
    b2: wp.int32,
    slot1: wp.int32,
    slot2: wp.int32,
    num_bodies: wp.int32,
    v1: wp.vec3f,
    w1: wp.vec3f,
    v2: wp.vec3f,
    w2: wp.vec3f,
):
    """Slot-aware writeback paired with :func:`_ms_load_body_pair`.

    Fast path: when both slots are ``-1`` (load returned the disabled
    path) we know mass splitting is off for this pair — write directly
    to ``bodies.*`` without the 4 ``write_*_unified`` calls.
    """
    if slot1 < wp.int32(0) and slot2 < wp.int32(0):
        bodies.velocity[b1] = v1
        bodies.velocity[b2] = v2
        bodies.angular_velocity[b1] = w1
        bodies.angular_velocity[b2] = w2
        return
    write_velocity_unified(bodies, particles, copy_state, b1, slot1, num_bodies, v1)
    write_velocity_unified(bodies, particles, copy_state, b2, slot2, num_bodies, v2)
    write_angular_velocity_unified(bodies, copy_state, b1, slot1, w1)
    write_angular_velocity_unified(bodies, copy_state, b2, slot2, w2)


# ---------------------------------------------------------------------------
# Shared axial (drive + limit) iterate helper
# ---------------------------------------------------------------------------


@wp.func
def _axial_drive_limit_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    jv_axial: wp.float32,
    clamp: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    store_friction: wp.bool,
) -> wp.float32:
    """Scalar drive, limit, and friction PGS step for revolute/prismatic mode.

    Both modes apply a single scalar impulse ``axial_lam`` along the
    free DoF axis. Revolute applies it as an angular impulse about
    ``n_hat``; prismatic as a linear impulse along ``n_hat``. The
    actuator cache (drive PD scalars, limit Box2D *or* PD scalars, the
    warm-started accumulated impulses) is identical across both modes,
    so the iterate math collapses to a shared helper that returns the
    net ``lam_drive + lam_limit`` and lets the caller spread it onto
    the body velocities in the per-mode way.

    Args:
        constraints: Shared column-major constraint storage.
        cid: Constraint id.
        base_offset: Dword offset of the constraint within its column.
        jv_axial: Pre-step axial velocity residual. Revolute passes
            ``n . (w1 - w2)``; prismatic passes ``n . (v1_anchor -
            v2_anchor)``. See the per-mode callers for the sign
            conventions.
        clamp: Pre-computed limit clamp state (``_CLAMP_NONE`` / ``_CLAMP_MIN``
            / ``_CLAMP_MAX``) from prepare.
        store_friction: Whether to persist the friction accumulator for later
            warm-starting. Relax sweeps solve friction for velocity convergence
            but keep the bias-solve accumulator intact.

    Returns:
        Net per-iteration axial impulse ``lam_drive + lam_limit + lam_friction``.
    """
    # Single-fetch of every actuator scalar; the compiler hoists the
    # reads ahead of the branches so we don't pay for the ones that
    # aren't consumed by the active path.
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    # ``max_lambda_drive`` was a stored ``max_force_drive * dt``; recompute
    # inline since both inputs are already in registers (saves 1 dword/joint).
    max_lambda_drive = max_force_drive * (wp.float32(1.0) / idt)
    bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
    gamma_drive = read_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid)
    eff_mass_drive_soft = read_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)

    # ---- Drive row ----------------------------------------------------
    # The drive row is a PD spring-damper only; prepare writes
    # ``eff_mass_drive_soft == 0`` whenever both gains are zero (OFF
    # mode, POSITION mode with zero gains, or VELOCITY mode with
    # ``damping_drive == 0``), so the rigid pure-velocity-motor
    # fallback never reaches the iterate. The ``eff_mass_drive_soft ==
    # 0`` short-circuit below is the single disable gate.
    drive_active = drive_mode != DRIVE_MODE_OFF
    if eff_mass_drive_soft <= 0.0:
        drive_active = False

    lam_drive = float(0.0)
    if drive_active:
        # Jitter2 SpringConstraint iterate (negated once to match our
        # ``jv = n . (w1 - w2)`` convention, which is -1 * the
        # Jitter2 ``jv = n . (v2 - v1)`` convention):
        #   lam = -M_eff_soft * (jv - bias + gamma * acc)
        lam_drive = -eff_mass_drive_soft * (jv_axial - bias_drive + gamma_drive * acc_drive)
        lam_drive = lam_drive * sor_boost
        old_acc = acc_drive
        acc_drive = acc_drive + lam_drive
        if max_force_drive > 0.0:
            acc_drive = wp.clamp(acc_drive, -max_lambda_drive, max_lambda_drive)
        lam_drive = acc_drive - old_acc
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, acc_drive)

    # ---- Limit row ----------------------------------------------------
    lam_limit = float(0.0)
    if clamp != _CLAMP_NONE:
        stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
        damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
        acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
        pd_mode_limit = stiffness_limit > 0.0 or damping_limit > 0.0
        if pd_mode_limit:
            # Jitter2 PD: same iterate form as the drive row. ``beta``
            # is stored *positive* (``pd_coefficients`` returns it with
            # the Jitter2 sign); iterate subtracts it from ``jv`` so
            # we get ``lam = -M * (jv - bias_pd + gamma * acc)``.
            pd_mass = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid)
            pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid)
            pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid)
            if pd_mass > 0.0:
                lam_limit = -pd_mass * (jv_axial - pd_beta + pd_gamma * acc_limit)
        else:
            # Box2D soft-constraint path. ``bias_limit_box2d`` is
            # prefolded as ``-C * bias_rate`` so the PGS step targets
            # ``jv = -bias``.
            eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
            bias_box = read_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid)
            mc_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
            ic_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
            if eff_inv > 0.0:
                eff_axial = 1.0 / eff_inv
                lam_unsoft = -eff_axial * (jv_axial + bias_box)
                lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
        lam_limit = lam_limit * sor_boost
        old_acc = acc_limit
        acc_limit = acc_limit + lam_limit
        # Unilateral clamp: the limit only pushes back toward the
        # allowed range. Positive ``acc`` reduces the cumulative
        # position (right thing at max stop); negative ``acc``
        # increases it (right thing at min stop).
        if clamp == _CLAMP_MAX:
            acc_limit = wp.max(0.0, acc_limit)
        else:
            acc_limit = wp.min(0.0, acc_limit)
        lam_limit = acc_limit - old_acc
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, acc_limit)

    # ---- Coulomb friction row -----------------------------------------
    lam_friction = float(0.0)
    friction = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
    if friction > 0.0:
        eff_inv_friction = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
        max_lambda_friction = friction * (wp.float32(1.0) / idt)
        if eff_inv_friction > 0.0 and max_lambda_friction > 0.0:
            slip_velocity = PHOENX_FRICTION_SLIP_VELOCITY
            slip_scale = read_float(constraints, base_offset + _OFF_FRICTION_SLIP_SCALE, cid)
            if slip_scale > wp.float32(0.0):
                slip_velocity = slip_scale * eff_inv_friction * friction
            gamma_friction = slip_velocity / max_lambda_friction
            eff_mass_friction = wp.float32(1.0) / (eff_inv_friction + gamma_friction)
            lam_friction = -eff_mass_friction * (jv_axial + gamma_friction * acc_friction)
            lam_friction = lam_friction * sor_boost
            old_acc_friction = acc_friction
            acc_friction = wp.clamp(
                acc_friction + lam_friction,
                -max_lambda_friction,
                max_lambda_friction,
            )
            lam_friction = acc_friction - old_acc_friction
            if store_friction:
                write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, acc_friction)
    elif friction <= 0.0:
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, 0.0)

    return lam_drive + lam_limit + lam_friction


# ---------------------------------------------------------------------------
# Shared axial (drive + limit) prepare helper
# ---------------------------------------------------------------------------


@wp.func
def _axial_drive_limit_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    cumulative_value: wp.float32,
    eff_inv: wp.float32,
    dt: wp.float32,
    drive_boost: wp.float32,
    limit_boost: wp.float32,
) -> wp.float32:
    """Shared drive + limit prepare for the axial scalar row.

    Both revolute and prismatic drive / limit rows are scalar
    position-error PDs with optional Box2D fallback on the limit
    side. The only per-mode difference is what ``cumulative_value``
    means (cumulative_angle [rad] vs slide [m]) and how the caller
    spreads the warm-start axial impulse onto the body velocities
    (pure torque vs linear impulse with lever arm). This helper does
    everything in between: reads gain / target / limit scalars,
    computes drive_C, calls :func:`pd_coefficients`, decides the PD
    vs Box2D limit path, writes all coefficient slots, selects the
    clamp state, and returns the warm-start ``axial_imp =
    acc_drive + acc_limit`` (with ``acc_limit`` zeroed when the
    limit is inactive, matching the standalone modules).

    ``drive_boost`` / ``limit_boost`` are per-joint-type Nyquist
    headroom multipliers (e.g. :data:`PHOENX_BOOST_REVOLUTE_DRIVE`)
    threaded in from the per-mode caller; both are clamped to
    ``[1, _PD_NYQUIST_HEADROOM_MAX]`` inside :func:`pd_coefficients`.

    Pairs with :func:`_axial_drive_limit_iterate`.
    """
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    target = read_float(constraints, base_offset + _OFF_TARGET, cid)
    target_velocity = read_float(constraints, base_offset + _OFF_TARGET_VELOCITY, cid)
    # max_force_drive is read here for symmetry with the iterate path,
    # but the prepare row only consumes stiffness/damping. Suppress the
    # F841 by reading it under a noqa scope for documentation.
    stiffness_drive = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    damping_drive = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)

    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, eff_inv)

    # ---- Drive (PD only) ---------------------------------------------
    drive_C = float(0.0)
    if drive_mode == DRIVE_MODE_POSITION:
        drive_C = cumulative_value - target
    if stiffness_drive > 0.0 or damping_drive > 0.0:
        gamma_drive, bias_drive, eff_mass_drive_soft = pd_coefficients(
            stiffness_drive, damping_drive, drive_C, eff_inv, dt, drive_boost
        )
    else:
        gamma_drive = wp.float32(0.0)
        bias_drive = wp.float32(0.0)
        eff_mass_drive_soft = wp.float32(0.0)
    bias_drive = bias_drive - target_velocity
    write_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid, gamma_drive)
    write_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, eff_mass_drive_soft)
    write_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid, bias_drive)

    # ---- Limit (dual convention) -------------------------------------
    clamp = _CLAMP_NONE
    limit_C = float(0.0)
    if min_value <= max_value:
        if cumulative_value > max_value:
            clamp = _CLAMP_MAX
            limit_C = cumulative_value - max_value
        elif cumulative_value < min_value:
            clamp = _CLAMP_MIN
            limit_C = cumulative_value - min_value
    write_int(constraints, base_offset + _OFF_CLAMP, cid, clamp)

    # ``limit_cache`` is mode-aliased Box2D / PD: writing both layouts
    # would clobber the active one (same 3 dwords). Iterate gates on
    # ``stiffness_limit > 0 or damping_limit > 0`` to pick the layout,
    # so only the active triple is filled.
    if stiffness_limit > 0.0 or damping_limit > 0.0:
        pd_gamma_limit, pd_beta_limit, pd_m_soft = pd_coefficients(
            stiffness_limit, damping_limit, limit_C, eff_inv, dt, limit_boost
        )
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, pd_gamma_limit)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, pd_beta_limit)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, pd_m_soft)
    else:
        br_limit, mc_limit, ic_limit = soft_constraint_coefficients(hertz_limit, damping_ratio_limit, dt)
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, -limit_C * br_limit)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mc_limit)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, ic_limit)

    # Warm-start: sum of drive + limit accumulated impulses, with
    # ``acc_limit`` forcibly zeroed when the limit is inactive.
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    if clamp == _CLAMP_NONE:
        acc_limit = 0.0
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
    friction = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    if friction <= 0.0:
        acc_friction = 0.0
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, 0.0)
    return acc_drive + acc_limit + acc_friction


# ---------------------------------------------------------------------------
# Shared tangent-basis-from-anchor-3 helper
# ---------------------------------------------------------------------------


@wp.func
def _tangent_basis_from_anchor3(
    n_hat: wp.vec3f,
    r1_b1: wp.vec3f,
    r3_b1: wp.vec3f,
):
    """Anchor-3-aligned tangent basis perpendicular to ``n_hat``.

    ``t1`` is the component of the anchor-1 -> anchor-3 lever arm
    (body-1 frame, rotated into world) perpendicular to ``n_hat``;
    ``t2 = n_hat x t1``. Using anchor-3 as the reference (rather
    than an arbitrary ``create_orthonormal``) makes the scalar
    anchor-3 row the exact tangential velocity gate for rotation
    about ``n_hat`` -- needed for Gauss-Seidel convergence in chains
    with shared bodies. Falls back to ``create_orthonormal(n_hat)``
    if anchor-3 has drifted onto the slide axis (shouldn't happen in
    practice since ``rest_length > 0``).
    """
    anchor3_offset_b1 = r3_b1 - r1_b1
    t1_raw = anchor3_offset_b1 - wp.dot(anchor3_offset_b1, n_hat) * n_hat
    t1_len2 = wp.dot(t1_raw, t1_raw)
    if t1_len2 > 1.0e-20:
        t1 = t1_raw / wp.sqrt(t1_len2)
    else:
        t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    return t1, t2


@wp.func
def _d6_metric_anchor_block(
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    ri_b1: wp.vec3f,
    ri_b2: wp.vec3f,
    rj_b1: wp.vec3f,
    rj_b2: wp.vec3f,
) -> wp.mat33f:
    """Effective-mass block for two metric helper-point rows."""
    eye3 = wp.identity(3, dtype=wp.float32)
    cri_b1 = wp.skew(ri_b1)
    cri_b2 = wp.skew(ri_b2)
    crj_b1 = wp.skew(rj_b1)
    crj_b2 = wp.skew(rj_b2)
    return (
        (inv_mass1 + inv_mass2) * eye3
        + cri_b1 @ (inv_inertia1 @ wp.transpose(crj_b1))
        + cri_b2 @ (inv_inertia2 @ wp.transpose(crj_b2))
    )


@wp.func
def _d6_project_tangent_block(block: wp.mat33f, t1: wp.vec3f, t2: wp.vec3f) -> wp.vec4f:
    """Project a 3x3 metric block onto the tangent basis."""
    block_t1 = block @ t1
    block_t2 = block @ t2
    return wp.vec4f(
        wp.dot(t1, block_t1),
        wp.dot(t1, block_t2),
        wp.dot(t2, block_t1),
        wp.dot(t2, block_t2),
    )


@wp.func
def _d6_project_scalar_block(block: wp.mat33f, axis: wp.vec3f) -> wp.float32:
    """Project a 3x3 metric block onto a single scalar row."""
    return wp.dot(axis, block @ axis)


@wp.func
def _d6_reproject_tangent_impulse(impulse: wp.vec3f, t1: wp.vec3f, t2: wp.vec3f) -> wp.vec3f:
    """Rebuild a cached world impulse from tangent-basis components."""
    return wp.dot(t1, impulse) * t1 + wp.dot(t2, impulse) * t2


@wp.func
def _d6_reproject_scalar_impulse(impulse: wp.vec3f, axis: wp.vec3f) -> wp.vec3f:
    """Rebuild a cached world impulse from one scalar row component."""
    return wp.dot(axis, impulse) * axis


@wp.func
def _d6_anchor_relative_velocity(
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    r_b1: wp.vec3f,
    r_b2: wp.vec3f,
) -> wp.vec3f:
    """Relative velocity at a metric helper point, body 2 minus body 1."""
    return velocity2 + wp.cross(angular_velocity2, r_b2) - velocity1 - wp.cross(angular_velocity1, r_b1)


@wp.func
def _d6_pd_softness(
    k: wp.float32,
    d: wp.float32,
    eff_inv: wp.float32,
    dt: wp.float32,
    idt: wp.float32,
    boost: wp.float32,
):
    """Implicit-Euler PD softness used by soft D6 metric rows."""
    row_boost = wp.clamp(boost, wp.float32(1.0), _PD_NYQUIST_HEADROOM_MAX)
    bias_factor = wp.float32(0.0)
    gamma = wp.float32(0.0)
    m_soft = wp.float32(0.0)
    if (k > wp.float32(0.0)) or (d > wp.float32(0.0)):
        if eff_inv > wp.float32(0.0):
            k_max = row_boost / (eff_inv * dt * dt)
            k_clamped = wp.min(k, k_max)
        else:
            k_clamped = k
        denom = d + dt * k_clamped
        if denom > wp.float32(0.0):
            softness = wp.float32(1.0) / denom
            bias_factor = dt * k_clamped * softness
            gamma = softness * idt
            if eff_inv + gamma > wp.float32(0.0):
                m_soft = wp.float32(1.0) / (eff_inv + gamma)
    return bias_factor, gamma, m_soft


# ---------------------------------------------------------------------------
# Unified per-anchor solve blocks (iterate pass)
# ---------------------------------------------------------------------------
#
# One block per anchor, selected by a runtime ``solve_kind`` tag. These are
# the single definitions of each per-anchor numeric formula; the unified
# :func:`_d6_iterate_rows_at` body and the prismatic Schur path are the only
# callers. Each returns the updated body velocities.


@wp.func
def _d6_solve_anchor1_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    solve_kind: wp.int32,
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    r1_b1: wp.vec3f,
    r1_b2: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Anchor-1 point lock. ``HARD3`` uses the sym6-packed 3x3 inverse
    (ball / universal); ``SOFT3`` uses the mat33 inverse (cable Box2D-soft).
    """
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)

    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)

    acc1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    jv1 = _d6_anchor_relative_velocity(velocity1, angular_velocity1, velocity2, angular_velocity2, r1_b1, r1_b2)

    if solve_kind == _D6_ROW_SOLVE_SOFT3:
        a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
        lam1_us = -(a1_inv @ (jv1 + bias1))
        lam1 = mass_coeff * lam1_us - impulse_coeff * acc1_world
        lam1 = lam1 * sor_boost
    else:
        a1_inv_s6 = read_vec6(constraints, base_offset + _OFF_A1_INV_S6, cid)
        lam1_us = -(mul_sym3(a1_inv_s6, jv1 + bias1))
        lam1 = mass_coeff * lam1_us - impulse_coeff * acc1_world
        lam1 = lam1 * sor_boost

    velocity1 = velocity1 - inv_mass1 * lam1
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1)
    velocity2 = velocity2 + inv_mass2 * lam1
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1_world + lam1)

    return velocity1, angular_velocity1, velocity2, angular_velocity2


@wp.func
def _d6_solve_anchor2_tangent_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    solve_kind: wp.int32,
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Anchor-2 tangent 2-row block. ``HARD2_TAN`` uses the sym3 swing
    Schur inverse (rigid); ``PD2_TAN`` uses the cable bend PD inverse
    (``K22_soft`` + ``gamma_bend``, spring bias unconditional).
    """
    r2_b1 = read_vec3(constraints, base_offset + _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)

    jv2_world = _d6_anchor_relative_velocity(velocity1, angular_velocity1, velocity2, angular_velocity2, r2_b1, r2_b2)
    jv2_t1 = wp.dot(t1, jv2_world)
    jv2_t2 = wp.dot(t2, jv2_world)

    if solve_kind == _D6_ROW_SOLVE_PD2_TAN:
        # Cable bend: PD spring bias is unconditional (encodes k*theta,
        # not drift); only anchor-1 drift bias is gated by use_bias.
        k22_inv_00 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_00, cid)
        k22_inv_01 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_01, cid)
        k22_inv_10 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_10, cid)
        k22_inv_11 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_11, cid)
        gamma_bend = read_float(constraints, base_offset + _OFF_CABLE_GAMMA_BEND, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)

        rhs2_t1 = jv2_t1 + bias2[0] + gamma_bend * acc2_t1
        rhs2_t2 = jv2_t2 + bias2[1] + gamma_bend * acc2_t2

        lam2_t1 = -(k22_inv_00 * rhs2_t1 + k22_inv_01 * rhs2_t2)
        lam2_t2 = -(k22_inv_10 * rhs2_t1 + k22_inv_11 * rhs2_t2)
        lam2_t1 = lam2_t1 * sor_boost
        lam2_t2 = lam2_t2 * sor_boost
    else:
        s_inv_s3 = read_vec3(constraints, base_offset + _OFF_S_INV_S3, cid)
        if use_bias:
            bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
        else:
            bias2 = wp.vec3f(0.0, 0.0, 0.0)
        lam2_us = -(mul_sym2(s_inv_s3, wp.vec2f(jv2_t1 + bias2[0], jv2_t2 + bias2[1])))
        lam2 = mass_coeff * lam2_us - impulse_coeff * wp.vec2f(acc2_t1, acc2_t2)
        lam2 = lam2 * sor_boost
        lam2_t1 = lam2[0]
        lam2_t2 = lam2[1]

    lam2_world = lam2_t1 * t1 + lam2_t2 * t2

    velocity1 = velocity1 - inv_mass1 * lam2_world
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr2_b1 @ lam2_world)
    velocity2 = velocity2 + inv_mass2 * lam2_world
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr2_b2 @ lam2_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    return velocity1, angular_velocity1, velocity2, angular_velocity2


@wp.func
def _d6_solve_rigid_swing_coupled_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    r1_b1: wp.vec3f,
    r1_b2: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Anchor-1 point lock (3 rows) + anchor-2 swing tangent lock (2 rows)
    solved as one COUPLED 3+2 Schur block (REVOLUTE / FIXED).

    This is the convergence-critical path: when the hinge axis is
    perpendicular to gravity (cantilever chain) the anchor-2 swing lock is
    the only constraint resisting the gravity bending moment, and it
    converges dramatically faster coupled to the anchor-1 point lock than
    as an independent block-Gauss-Seidel block. ``ut_ai = U^T A1^-1`` and
    the 2x2 swing Schur are cached in prepare; here we back-substitute.
    """
    r2_b1 = read_vec3(constraints, base_offset + _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    a1_inv_s6 = read_vec6(constraints, base_offset + _OFF_A1_INV_S6, cid)
    s_inv_s3 = read_vec3(constraints, base_offset + _OFF_S_INV_S3, cid)
    ut_ai_row0 = read_vec3(constraints, base_offset + _OFF_UT_AI_ROW0, cid)
    ut_ai_row1 = read_vec3(constraints, base_offset + _OFF_UT_AI_ROW1, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
    bias2_tan = wp.vec2f(bias2[0], bias2[1])

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_tan = wp.vec2f(wp.dot(t1, acc2_world), wp.dot(t2, acc2_world))

    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
    jv2 = wp.vec2f(wp.dot(t1, jv2_world), wp.dot(t2, jv2_world))

    rhs1 = jv1 + bias1
    rhs2 = jv2 + bias2_tan

    ut_ai_rhs1 = wp.vec2f(wp.dot(ut_ai_row0, rhs1), wp.dot(ut_ai_row1, rhs1))
    lam2_us = -(mul_sym2(s_inv_s3, rhs2 - ut_ai_rhs1))
    lam2 = mass_coeff * lam2_us - impulse_coeff * acc2_tan
    lam2 = lam2 * sor_boost

    lam2_world = lam2[0] * t1 + lam2[1] * t2
    lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

    u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
    u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
    u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

    lam1_us = -(mul_sym3(a1_inv_s6, rhs1 + u_lam2_us))
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1
    lam1 = lam1 * sor_boost

    total_lin = lam1 + lam2_world
    velocity1 = velocity1 - inv_mass1 * total_lin
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1 + cr2_b1 @ lam2_world)
    velocity2 = velocity2 + inv_mass2 * total_lin
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1 + cr2_b2 @ lam2_world)

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    return velocity1, angular_velocity1, velocity2, angular_velocity2


@wp.func
def _d6_solve_prismatic_coupled_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    r1_b1: wp.vec3f,
    r1_b2: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Prismatic positional lock: anchor-1 tangent (2) + anchor-2 tangent
    (2) + anchor-3 scalar (1) solved as one COUPLED 4+1 Schur.

    The 4x4 ``a4_inv`` (the two tangent anchors) and the scalar ``c`` /
    ``s_scalar_inv`` coupling to the anchor-3 twist row are cached in
    prepare. Decoupling these into independent per-anchor blocks collapses
    a slider cantilever (the transverse locks under-converge); the coupled
    solve holds it rigid -- the prismatic analog of the rigid-swing Schur.
    """
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

    a4_inv = read_mat44(constraints, base_offset + _OFF_A4_INV, cid)
    c_pris = read_vec4(constraints, base_offset + _OFF_C_PRIS, cid)
    s_scalar_inv = read_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
        bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
        bias3 = wp.float32(0.0)

    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc4 = wp.vec4f(
        wp.dot(t1, acc_imp1_world),
        wp.dot(t2, acc_imp1_world),
        wp.dot(t1, acc_imp2_world),
        wp.dot(t2, acc_imp2_world),
    )
    acc3_scalar = wp.dot(t2, acc_imp3_world)

    jv1_world = velocity2 - cr1_b2 @ angular_velocity2 - velocity1 + cr1_b1 @ angular_velocity1
    jv2_world = velocity2 - cr2_b2 @ angular_velocity2 - velocity1 + cr2_b1 @ angular_velocity1
    jv3_world = velocity2 - cr3_b2 @ angular_velocity2 - velocity1 + cr3_b1 @ angular_velocity1

    jv4 = wp.vec4f(wp.dot(t1, jv1_world), wp.dot(t2, jv1_world), wp.dot(t1, jv2_world), wp.dot(t2, jv2_world))
    jv3 = wp.dot(t2, jv3_world)

    rhs4 = jv4 + wp.vec4f(bias1[0], bias1[1], bias2[0], bias2[1])
    rhs3 = jv3 + bias3

    # Schur: eliminate the a3 scalar row against the 4x4 tangent block.
    a4_inv_rhs4 = a4_inv @ rhs4
    lam3_us = -s_scalar_inv * (rhs3 - wp.dot(c_pris, a4_inv_rhs4))
    lam3 = mass_coeff * lam3_us - impulse_coeff * acc3_scalar
    lam3 = lam3 * sor_boost

    lam4_us = -(a4_inv @ (rhs4 + c_pris * lam3_us))
    lam4 = mass_coeff * lam4_us - impulse_coeff * acc4
    lam4 = lam4 * sor_boost

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

    return velocity1, angular_velocity1, velocity2, angular_velocity2


@wp.func
def _d6_solve_anchor3_scalar_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    solve_kind: wp.int32,
    velocity1: wp.vec3f,
    angular_velocity1: wp.vec3f,
    velocity2: wp.vec3f,
    angular_velocity2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Anchor-3 scalar 1-row block along ``t2``. ``HARD1_SCALAR`` uses the
    rigid twist Schur scalar; ``PD1_SCALAR`` uses the cable twist PD mass
    (``M_twist_soft`` + ``gamma_twist``, spring bias unconditional).
    """
    r3_b1 = read_vec3(constraints, base_offset + _OFF_R3_B1, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
    cr3_b1 = wp.skew(r3_b1)
    cr3_b2 = wp.skew(r3_b2)

    acc3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc3_scalar = wp.dot(t2, acc3_world)

    jv3_world = _d6_anchor_relative_velocity(velocity1, angular_velocity1, velocity2, angular_velocity2, r3_b1, r3_b2)
    jv3 = wp.dot(t2, jv3_world)

    if solve_kind == _D6_ROW_SOLVE_PD1_SCALAR:
        m_twist_soft = read_float(constraints, base_offset + _OFF_CABLE_M_TWIST_SOFT, cid)
        gamma_twist = read_float(constraints, base_offset + _OFF_CABLE_GAMMA_TWIST, cid)
        bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
        lam3 = -m_twist_soft * (jv3 + bias3 + gamma_twist * acc3_scalar)
        lam3 = lam3 * sor_boost
    else:
        s_scalar_inv = read_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid)
        if use_bias:
            bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
        else:
            bias3 = wp.float32(0.0)
        lam3_us = -(s_scalar_inv * (jv3 + bias3))
        lam3 = mass_coeff * lam3_us - impulse_coeff * acc3_scalar
        lam3 = lam3 * sor_boost

    lam3_world = lam3 * t2

    velocity1 = velocity1 - inv_mass1 * lam3_world
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr3_b1 @ lam3_world)
    velocity2 = velocity2 + inv_mass2 * lam3_world
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr3_b2 @ lam3_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3_world + lam3_world)

    return velocity1, angular_velocity1, velocity2, angular_velocity2


@wp.func
def _d6_limit_axis_local(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    joint_mode: wp.int32,
    slot: wp.int32,
) -> wp.vec3f:
    if joint_mode == JOINT_MODE_BALL_SOCKET:
        if slot == wp.int32(0):
            return read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
        if slot == wp.int32(1):
            return read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        return read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
    if slot == wp.int32(0):
        return read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
    if slot == wp.int32(1):
        return read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
    return wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def _d6_angular_limits_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    joint_mode: wp.int32,
    orientation1: wp.quatf,
    orientation2: wp.quatf,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    angular_velocity1: wp.vec3f,
    angular_velocity2: wp.vec3f,
    dt: wp.float32,
):
    count = read_int(constraints, base_offset + _OFF_D6_LIMIT_COUNT, cid)
    if count <= wp.int32(0):
        return angular_velocity1, angular_velocity2

    lower = read_vec3(constraints, base_offset + _OFF_D6_LIMIT_LOWER, cid)
    upper = read_vec3(constraints, base_offset + _OFF_D6_LIMIT_UPPER, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    bias_rate, _mc, _ic = soft_constraint_coefficients(hertz_limit, damping_ratio_limit, dt)
    inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
    diff = orientation2 * inv_init * wp.quat_inverse(orientation1)

    axis0 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(0)))
    axis1 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(1)))
    axis2 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(2)))

    bias0 = wp.float32(0.0)
    bias1 = wp.float32(0.0)
    bias2 = wp.float32(0.0)
    eff0 = wp.float32(0.0)
    eff1 = wp.float32(0.0)
    eff2 = wp.float32(0.0)

    if count > wp.int32(0) and lower[0] <= upper[0]:
        angle0 = extract_rotation_angle(diff, axis0)
        if angle0 > upper[0]:
            bias0 = -(angle0 - upper[0]) * bias_rate
        elif angle0 < lower[0]:
            bias0 = -(angle0 - lower[0]) * bias_rate
        eff0 = wp.dot(axis0, inv_inertia1 @ axis0) + wp.dot(axis0, inv_inertia2 @ axis0)
    if count > wp.int32(1) and lower[1] <= upper[1]:
        angle1 = extract_rotation_angle(diff, axis1)
        if angle1 > upper[1]:
            bias1 = -(angle1 - upper[1]) * bias_rate
        elif angle1 < lower[1]:
            bias1 = -(angle1 - lower[1]) * bias_rate
        eff1 = wp.dot(axis1, inv_inertia1 @ axis1) + wp.dot(axis1, inv_inertia2 @ axis1)
    if count > wp.int32(2) and lower[2] <= upper[2]:
        angle2 = extract_rotation_angle(diff, axis2)
        if angle2 > upper[2]:
            bias2 = -(angle2 - upper[2]) * bias_rate
        elif angle2 < lower[2]:
            bias2 = -(angle2 - lower[2]) * bias_rate
        eff2 = wp.dot(axis2, inv_inertia1 @ axis2) + wp.dot(axis2, inv_inertia2 @ axis2)

    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, wp.vec3f(bias0, bias1, bias2))
    write_vec3(constraints, base_offset + _OFF_D6_LIMIT_EFF_INV, cid, wp.vec3f(eff0, eff1, eff2))

    old_acc = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc = wp.vec3f(0.0, 0.0, 0.0)
    if bias0 != wp.float32(0.0):
        acc = acc + wp.dot(axis0, old_acc) * axis0
    if bias1 != wp.float32(0.0):
        acc = acc + wp.dot(axis1, old_acc) * axis1
    if bias2 != wp.float32(0.0):
        acc = acc + wp.dot(axis2, old_acc) * axis2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc)
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ acc
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ acc
    return angular_velocity1, angular_velocity2


@wp.func
def _d6_angular_limits_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    b1: wp.int32,
    joint_mode: wp.int32,
    w1: wp.vec3f,
    w2: wp.vec3f,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    count = read_int(constraints, base_offset + _OFF_D6_LIMIT_COUNT, cid)
    if count <= wp.int32(0):
        return w1, w2

    orientation1 = bodies.orientation[b1]
    axis0 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(0)))
    axis1 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(1)))
    axis2 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(2)))

    bias = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    eff_inv = read_vec3(constraints, base_offset + _OFF_D6_LIMIT_EFF_INV, cid)
    acc = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    old0 = wp.float32(0.0)
    old1 = wp.float32(0.0)
    old2 = wp.float32(0.0)
    if count > wp.int32(0):
        old0 = wp.dot(axis0, acc)
    if count > wp.int32(1):
        old1 = wp.dot(axis1, acc)
    if count > wp.int32(2):
        old2 = wp.dot(axis2, acc)

    dt = wp.float32(1.0) / idt
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    _br, mc, ic = soft_constraint_coefficients(hertz_limit, damping_ratio_limit, dt)

    new0 = old0
    new1 = old1
    new2 = old2
    if bias[0] != wp.float32(0.0) and eff_inv[0] > wp.float32(0.0):
        lam0 = mc * (-(wp.float32(1.0) / eff_inv[0]) * (wp.dot(axis0, w1 - w2) + bias[0])) - ic * old0
        new0 = old0 + lam0 * sor_boost
        if bias[0] < wp.float32(0.0):
            new0 = wp.max(wp.float32(0.0), new0)
        else:
            new0 = wp.min(wp.float32(0.0), new0)
    if bias[1] != wp.float32(0.0) and eff_inv[1] > wp.float32(0.0):
        lam1 = mc * (-(wp.float32(1.0) / eff_inv[1]) * (wp.dot(axis1, w1 - w2) + bias[1])) - ic * old1
        new1 = old1 + lam1 * sor_boost
        if bias[1] < wp.float32(0.0):
            new1 = wp.max(wp.float32(0.0), new1)
        else:
            new1 = wp.min(wp.float32(0.0), new1)
    if bias[2] != wp.float32(0.0) and eff_inv[2] > wp.float32(0.0):
        lam2 = mc * (-(wp.float32(1.0) / eff_inv[2]) * (wp.dot(axis2, w1 - w2) + bias[2])) - ic * old2
        new2 = old2 + lam2 * sor_boost
        if bias[2] < wp.float32(0.0):
            new2 = wp.max(wp.float32(0.0), new2)
        else:
            new2 = wp.min(wp.float32(0.0), new2)

    delta = (new0 - old0) * axis0 + (new1 - old1) * axis1 + (new2 - old2) * axis2
    new_acc = new0 * axis0 + new1 * axis1 + new2 * axis2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, new_acc)
    w1 = w1 + ii1 @ delta
    w2 = w2 - ii2 @ delta
    return w1, w2


# ---------------------------------------------------------------------------
# Prismatic (D6 linear-slider row layout)
# ---------------------------------------------------------------------------
#
# Rank-5 pure-points, 2+2+1 rows: anchor-1 tangent drift onto (t1,t2),
# anchor-2 tangent drift onto (t1,t2), and anchor-3 drift onto t2 to
# kill the last rotational DoF (rotation about n_hat).
#
# Solved as three INDEPENDENT block-Gauss-Seidel blocks, identical in
# shape to the rigid swing family: anchor-1 tangent 2x2 (sym3), anchor-2
# tangent 2x2 (sym3), anchor-3 twist 1x1. No cross-anchor coupling matrix
# and no ``wp.inverse(mat44f)`` -- the slider relies on the outer PGS
# sweeps to close the inter-anchor coupling, the same way revolute does.
# Three cheap sym2/scalar inverses per prepare; zero per-iter inverses.


@wp.func
def _d6_prepare_rows_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    mode_cfg: wp.int32,
):
    """Single unified D6 prepare body. No dispatch into per-mode
    formulations.

    Mirrors :func:`_d6_iterate_rows_at`: the three positional anchors and
    the axial row are prepared by data-driven blocks, derived once from
    ``mode_cfg``. Every mode caches independent per-anchor inverses
    (block-Gauss-Seidel); prismatic uses the same 2x2 / 1x1 shape as the
    rigid swing family rather than a coupled 4+1 Schur. Math is identical
    to the proven per-mode prepares -- only the dispatch is gone.
    """
    is_cable = mode_cfg == JOINT_MODE_CABLE
    is_prismatic = mode_cfg == JOINT_MODE_PRISMATIC
    has_swing = mode_cfg == JOINT_MODE_REVOLUTE or mode_cfg == JOINT_MODE_FIXED
    has_twist = mode_cfg == JOINT_MODE_FIXED
    has_axial = mode_cfg == JOINT_MODE_REVOLUTE or mode_cfg == JOINT_MODE_UNIVERSAL
    has_limits = mode_cfg == JOINT_MODE_BALL_SOCKET or mode_cfg == JOINT_MODE_UNIVERSAL

    # Anchors 2 / 3 levers + tangent basis are needed by every mode
    # except the anchor-1-only BALL/UNIVERSAL family.
    uses_three_anchors = is_cable or is_prismatic or has_twist
    uses_anchor2 = uses_three_anchors or has_swing

    b1 = body_pair.b1
    b2 = body_pair.b2

    orientation1 = bodies.orientation[b1]
    orientation2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    (
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    dt = wp.float32(1.0) / idt

    # ---- Anchor-1 levers (always) ------------------------------------
    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)
    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Anchor-2 / anchor-3 levers (modes that use them).
    r2_b1 = wp.vec3f(0.0, 0.0, 0.0)
    r2_b2 = wp.vec3f(0.0, 0.0, 0.0)
    r3_b1 = wp.vec3f(0.0, 0.0, 0.0)
    r3_b2 = wp.vec3f(0.0, 0.0, 0.0)
    p2_b1 = wp.vec3f(0.0, 0.0, 0.0)
    p2_b2 = wp.vec3f(0.0, 0.0, 0.0)
    p3_b1 = wp.vec3f(0.0, 0.0, 0.0)
    p3_b2 = wp.vec3f(0.0, 0.0, 0.0)
    if uses_anchor2:
        la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
        r2_b1 = wp.quat_rotate(orientation1, la2_b1)
        r2_b2 = wp.quat_rotate(orientation2, la2_b2)
        write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
        write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)
        p2_b1 = position1 + r2_b1
        p2_b2 = position2 + r2_b2
    if uses_three_anchors:
        la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
        la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
        r3_b1 = wp.quat_rotate(orientation1, la3_b1)
        r3_b2 = wp.quat_rotate(orientation2, la3_b2)
        write_vec3(constraints, base_offset + _OFF_R3_B1, cid, r3_b1)
        write_vec3(constraints, base_offset + _OFF_R3_B2, cid, r3_b2)
        p3_b1 = position1 + r3_b1
        p3_b2 = position2 + r3_b2

    # ---- Axis + tangent basis ----------------------------------------
    if uses_anchor2:
        hinge_vec = p2_b2 - p1_b2
        hinge_len2 = wp.dot(hinge_vec, hinge_vec)
        if hinge_len2 > 1.0e-20:
            n_hat = hinge_vec / wp.sqrt(hinge_len2)
        else:
            n_hat = wp.vec3f(1.0, 0.0, 0.0)
    else:
        n_hat = wp.quat_rotate(orientation1, read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid))
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # Tangent basis: anchor-3 gate when three anchors are used (FIXED /
    # PRISMATIC / CABLE), else the REVOLUTE create_orthonormal basis.
    if uses_three_anchors:
        t1, t2 = _tangent_basis_from_anchor3(n_hat, r1_b1, r3_b1)
    else:
        t1 = create_orthonormal(n_hat)
        t2 = wp.cross(n_hat, t1)
    if uses_anchor2:
        write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
        write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    # ---- Anchor-1 inverse + bias -------------------------------------
    a1 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r1_b1, r1_b2, r1_b1, r1_b2)
    if is_prismatic:
        # Tangent-only lock; the n_hat slide axis stays free. The coupled
        # 4+1 Schur (a4_inv / c_pris / s_scalar_inv) is assembled in the
        # prismatic block below -- here we only stash the anchor-1 tangent
        # drift bias, stored as (t1, t2, 0).
        drift1 = p1_b2 - p1_b1
        write_vec3(
            constraints,
            base_offset + _OFF_BIAS1,
            cid,
            wp.vec3f(wp.dot(t1, drift1) * bias_rate, wp.dot(t2, drift1) * bias_rate, 0.0),
        )
    elif is_cable:
        write_mat33(constraints, base_offset + _OFF_A1_INV, cid, wp.inverse(a1))
        write_vec3(constraints, base_offset + _OFF_BIAS1, cid, (p1_b2 - p1_b1) * bias_rate)
    else:
        write_vec6(constraints, base_offset + _OFF_A1_INV_S6, cid, inv_sym3(sym6_from_mat33_upper(a1)))
        write_vec3(constraints, base_offset + _OFF_BIAS1, cid, (p1_b2 - p1_b1) * bias_rate)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)

    if is_prismatic:
        # ---- PRISMATIC: COUPLED 4+1 Schur ----------------------------
        # Four tangent rows (a1, a2) coupled in one 4x4 (``a4_inv``) plus
        # the a3 scalar twist row, eliminated via the ``c_pris`` /
        # ``s_scalar_inv`` Schur. This is the convergent slider formulation
        # (decoupled per-anchor blocks collapse a slider cantilever, the
        # same failure the rigid-swing decoupling caused for hinges).
        # ``b11`` is the anchor-1 metric block ``a1`` computed above.
        cr2_b1 = wp.skew(r2_b1)
        cr2_b2 = wp.skew(r2_b2)
        cr3_b1 = wp.skew(r3_b1)
        cr3_b2 = wp.skew(r3_b2)

        b11 = a1
        b22 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r2_b1, r2_b2, r2_b1, r2_b2)
        b33 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r3_b1, r3_b2, r3_b1, r3_b2)
        b12 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r1_b1, r1_b2, r2_b1, r2_b2)
        b13 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r1_b1, r1_b2, r3_b1, r3_b2)
        b23 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r2_b1, r2_b2, r3_b1, r3_b2)

        d11 = _d6_project_tangent_block(b11, t1, t2)
        d22 = _d6_project_tangent_block(b22, t1, t2)
        d12 = _d6_project_tangent_block(b12, t1, t2)
        # K4 rows/cols = (a1 t1, a1 t2, a2 t1, a2 t2); symmetric.
        k4 = wp.mat44f(
            d11[0],
            d11[1],
            d12[0],
            d12[1],
            d11[1],
            d11[3],
            d12[2],
            d12[3],
            d12[0],
            d12[2],
            d22[0],
            d22[1],
            d12[1],
            d12[3],
            d22[1],
            d22[3],
        )
        # c: coupling of the 4 tangent rows to the a3 scalar row (along t2).
        b13_t2 = b13 @ t2
        b23_t2 = b23 @ t2
        c = wp.vec4f(wp.dot(t1, b13_t2), wp.dot(t2, b13_t2), wp.dot(t1, b23_t2), wp.dot(t2, b23_t2))
        d_scalar = _d6_project_scalar_block(b33, t2)

        a4_inv = wp.inverse(k4)
        s_scalar = d_scalar - wp.dot(c, a4_inv @ c)
        if wp.abs(s_scalar) > 1.0e-20:
            s_scalar_inv = 1.0 / s_scalar
        else:
            s_scalar_inv = 0.0
        write_mat44(constraints, base_offset + _OFF_A4_INV, cid, a4_inv)
        write_vec4(constraints, base_offset + _OFF_C_PRIS, cid, c)
        write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)

        drift2 = p2_b2 - p2_b1
        drift3 = p3_b2 - p3_b1
        write_vec3(
            constraints,
            base_offset + _OFF_BIAS2,
            cid,
            wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0),
        )
        write_float(constraints, base_offset + _OFF_BIAS3, cid, wp.dot(t2, drift3) * bias_rate)

        # Reproject + apply positional warm-start (tangent at a1/a2,
        # scalar along t2 at a3).
        acc1w = _d6_reproject_tangent_impulse(acc1, t1, t2)
        acc2w = _d6_reproject_tangent_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid), t1, t2)
        acc3w = _d6_reproject_scalar_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid), t2)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1w)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2w)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3w)
        total_linear = acc1w + acc2w + acc3w
        velocity1 = velocity1 - inv_mass1 * total_linear
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ acc1w + cr2_b1 @ acc2w + cr3_b1 @ acc3w)
        velocity2 = velocity2 + inv_mass2 * total_linear
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ acc1w + cr2_b2 @ acc2w + cr3_b2 @ acc3w)
    elif is_cable:
        # ---- CABLE: anchor-2 bend PD + anchor-3 twist PD -------------
        cr2_b1 = wp.skew(r2_b1)
        cr2_b2 = wp.skew(r2_b2)
        cr3_b1 = wp.skew(r3_b1)
        cr3_b2 = wp.skew(r3_b2)

        b22 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r2_b1, r2_b2, r2_b1, r2_b2)
        k22 = _d6_project_tangent_block(b22, t1, t2)
        rest_length = read_float(constraints, base_offset + _OFF_REST_LENGTH, cid)
        rest_len2 = rest_length * rest_length
        if rest_len2 > 1.0e-12:
            inv_rest2 = wp.float32(1.0) / rest_len2
        else:
            inv_rest2 = wp.float32(0.0)

        k_bend_user = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
        d_bend_user = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
        eff_inv_bend = wp.float32(0.5) * (k22[0] + k22[3])
        bias_factor_bend, gamma_bend, _m_bend_soft = _d6_pd_softness(
            k_bend_user * inv_rest2, d_bend_user * inv_rest2, eff_inv_bend, dt, idt, PHOENX_BOOST_CABLE_BEND
        )
        k22s_00 = k22[0] + gamma_bend
        k22s_11 = k22[3] + gamma_bend
        det_b = k22s_00 * k22s_11 - k22[1] * k22[2]
        if wp.abs(det_b) > wp.float32(1.0e-20):
            inv_det_b = wp.float32(1.0) / det_b
        else:
            inv_det_b = wp.float32(0.0)
        write_float(constraints, base_offset + _OFF_CABLE_K22_INV_00, cid, k22s_11 * inv_det_b)
        write_float(constraints, base_offset + _OFF_CABLE_K22_INV_01, cid, -k22[1] * inv_det_b)
        write_float(constraints, base_offset + _OFF_CABLE_K22_INV_10, cid, -k22[2] * inv_det_b)
        write_float(constraints, base_offset + _OFF_CABLE_K22_INV_11, cid, k22s_00 * inv_det_b)
        write_float(constraints, base_offset + _OFF_CABLE_GAMMA_BEND, cid, gamma_bend)

        drift2 = p2_b2 - p2_b1
        write_vec3(
            constraints,
            base_offset + _OFF_BIAS2,
            cid,
            wp.vec3f(wp.dot(t1, drift2) * bias_factor_bend * idt, wp.dot(t2, drift2) * bias_factor_bend * idt, 0.0),
        )

        b33 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r3_b1, r3_b2, r3_b1, r3_b2)
        eff_inv_twist = _d6_project_scalar_block(b33, t2)
        k_twist_user = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
        d_twist_user = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
        bias_factor_twist, gamma_twist, m_twist_soft = _d6_pd_softness(
            k_twist_user * inv_rest2, d_twist_user * inv_rest2, eff_inv_twist, dt, idt, PHOENX_BOOST_CABLE_TWIST
        )
        write_float(constraints, base_offset + _OFF_CABLE_M_TWIST_SOFT, cid, m_twist_soft)
        write_float(constraints, base_offset + _OFF_CABLE_GAMMA_TWIST, cid, gamma_twist)
        drift3 = p3_b2 - p3_b1
        write_float(constraints, base_offset + _OFF_BIAS3, cid, wp.dot(t2, drift3) * bias_factor_twist * idt)

        acc2w = _d6_reproject_tangent_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid), t1, t2)
        acc3w = _d6_reproject_scalar_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid), t2)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2w)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3w)
        total_linear = acc1 + acc2w + acc3w
        velocity1 = velocity1 - inv_mass1 * total_linear
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ acc1 + cr2_b1 @ acc2w + cr3_b1 @ acc3w)
        velocity2 = velocity2 + inv_mass2 * total_linear
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ acc1 + cr2_b2 @ acc2w + cr3_b2 @ acc3w)
    elif has_swing:
        # ---- RIGID swing (REVOLUTE / FIXED): anchor-1<->anchor-2 COUPLED
        # 3+2 Schur + optional a3 twist. The coupling (vs decoupled
        # block-Gauss-Seidel) is convergence-critical for stiff cantilever
        # hinge chains: the anchor-2 swing lock is the only constraint
        # resisting a gravity bending moment when the hinge axis is
        # perpendicular to gravity, and it converges far faster coupled to
        # the anchor-1 point lock. Cache ``ut_ai = U^T A1^-1`` and the 2x2
        # swing Schur ``S = T^T A2 T - U^T A1^-1 U``.
        cr2_b1 = wp.skew(r2_b1)
        cr2_b2 = wp.skew(r2_b2)
        a1_inv_s6 = read_vec6(constraints, base_offset + _OFF_A1_INV_S6, cid)
        a2 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r2_b1, r2_b2, r2_b1, r2_b2)
        # Cross-anchor coupling block B = A12 (anchor-1 rows vs anchor-2 rows).
        b_mat = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r1_b1, r1_b2, r2_b1, r2_b2)
        u_col0 = b_mat @ t1
        u_col1 = b_mat @ t2
        ut_ai_row0 = mul_sym3(a1_inv_s6, u_col0)
        ut_ai_row1 = mul_sym3(a1_inv_s6, u_col1)
        write_vec3(constraints, base_offset + _OFF_UT_AI_ROW0, cid, ut_ai_row0)
        write_vec3(constraints, base_offset + _OFF_UT_AI_ROW1, cid, ut_ai_row1)
        a2_t1 = a2 @ t1
        a2_t2 = a2 @ t2
        s00 = wp.dot(t1, a2_t1) - wp.dot(ut_ai_row0, u_col0)
        s01 = wp.dot(t1, a2_t2) - wp.dot(ut_ai_row0, u_col1)
        s11 = wp.dot(t2, a2_t2) - wp.dot(ut_ai_row1, u_col1)
        write_vec3(constraints, base_offset + _OFF_S_INV_S3, cid, inv_sym2(wp.vec3f(s00, s01, s11)))
        drift2 = p2_b2 - p2_b1
        write_vec3(
            constraints,
            base_offset + _OFF_BIAS2,
            cid,
            wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0),
        )

        acc2w = _d6_reproject_tangent_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid), t1, t2)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2w)
        acc3w = wp.vec3f(0.0, 0.0, 0.0)
        cr3_b1 = wp.skew(r3_b1)
        cr3_b2 = wp.skew(r3_b2)
        if has_twist:
            b33 = _d6_metric_anchor_block(inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, r3_b1, r3_b2, r3_b1, r3_b2)
            d_scalar = _d6_project_scalar_block(b33, t2)
            if wp.abs(d_scalar) > 1.0e-20:
                s_scalar_inv = 1.0 / d_scalar
            else:
                s_scalar_inv = 0.0
            write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)
            drift3 = p3_b2 - p3_b1
            write_float(constraints, base_offset + _OFF_BIAS3, cid, wp.dot(t2, drift3) * bias_rate)
            acc3w = _d6_reproject_scalar_impulse(read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid), t2)
            write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3w)

        total_linear = acc1 + acc2w + acc3w
        velocity1 = velocity1 - inv_mass1 * total_linear
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ acc1 + cr2_b1 @ acc2w + cr3_b1 @ acc3w)
        velocity2 = velocity2 + inv_mass2 * total_linear
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ acc1 + cr2_b2 @ acc2w + cr3_b2 @ acc3w)
    else:
        # ---- RIGID anchor-1-only warm-start (BALL / UNIVERSAL) -------
        velocity1 = velocity1 - inv_mass1 * acc1
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ acc1)
        velocity2 = velocity2 + inv_mass2 * acc1
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ acc1)

    # ---- Axial drive / limit prepare ---------------------------------
    if is_prismatic:
        b11 = a1
        eff_inv = wp.dot(n_hat, b11 @ n_hat)
        slide = wp.dot(n_hat, p1_b2 - p1_b1)
        axial_imp = _axial_drive_limit_prepare_at(
            constraints,
            cid,
            base_offset,
            slide,
            eff_inv,
            dt,
            PHOENX_BOOST_PRISMATIC_DRIVE,
            PHOENX_BOOST_PRISMATIC_LIMIT,
        )
        velocity1 = velocity1 + inv_mass1 * (n_hat * axial_imp)
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ wp.cross(r1_b1, n_hat * axial_imp)
        velocity2 = velocity2 - inv_mass2 * (n_hat * axial_imp)
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ wp.cross(r1_b2, n_hat * axial_imp)
    elif has_axial:
        eff_inv = wp.dot(n_hat, inv_inertia1 @ n_hat) + wp.dot(n_hat, inv_inertia2 @ n_hat)
        j1 = wp.quat_rotate(orientation1, read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid))
        inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
        diff = orientation2 * inv_init * wp.quat_inverse(orientation1)
        new_q_angle = extract_rotation_angle(diff, j1)
        old_counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
        old_prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
        new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
        write_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid, new_counter)
        write_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
        cumulative_angle = revolution_tracker_angle(new_counter, new_prev)
        axial_imp = _axial_drive_limit_prepare_at(
            constraints,
            cid,
            base_offset,
            cumulative_angle,
            eff_inv,
            dt,
            PHOENX_BOOST_REVOLUTE_DRIVE,
            PHOENX_BOOST_REVOLUTE_LIMIT,
        )
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_imp)
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_imp)

    if has_limits:
        count = read_int(constraints, base_offset + _OFF_D6_LIMIT_COUNT, cid)
        if count > wp.int32(0):
            angular_velocity1, angular_velocity2 = _d6_angular_limits_prepare_at(
                constraints,
                cid,
                base_offset,
                mode_cfg,
                orientation1,
                orientation2,
                inv_inertia1,
                inv_inertia2,
                angular_velocity1,
                angular_velocity2,
                dt,
            )

    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    )

    # Zero unused axial drive / limit state for the non-axial modes so
    # wrench helpers and cross-mode reads see a clean column (CABLE /
    # FIXED match the prior per-mode behavior).
    if is_cable or has_twist:
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, 0.0)
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, 0.0)
        write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, 0.0)
        write_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, 0.0)


@wp.func
def _d6_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    mode_cfg: wp.int32,
):
    """Unified D6 prepare pass."""
    _d6_prepare_rows_at(
        constraints,
        cid,
        base_offset,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        mode_cfg,
    )


@wp.func
def _d6_iterate_rows_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    mode_cfg: wp.int32,
):
    """Unified D6 metric row-engine iterate for all joint modes.

    One body, no dispatch into per-mode formulations. The differences
    (per-anchor solve kind, axial kind) are data, derived once from
    ``mode_cfg``:

    * per-anchor ``*_kind`` -- HARD3 / HARD1_TAN / HARD2_TAN /
      HARD1_SCALAR / SOFT3 / PD2_TAN / PD1_SCALAR / SKIP, selecting which
      cached inverse + softness the shared ``_d6_solve_anchor*`` helpers
      apply.
    * ``axial_kind`` -- ANGULAR (REVOLUTE/UNIVERSAL twist) / LINEAR
      (PRISMATIC slide) / NONE.

    All modes solve the positional rows as independent per-anchor blocks
    (block-Gauss-Seidel); there is no coupled multi-anchor inverse.
    """
    #
    # mode_cfg is uniform across a solve batch (one color = one mode), so
    # the branch below is warp-coherent: no divergence, registers bounded
    # by the taken branch -- revolute keeps its exact footprint.
    #
    # Every mode now solves the positional rows as INDEPENDENT per-anchor
    # blocks (block-Gauss-Seidel). Prismatic used to couple all five rows
    # into one 4+1 Schur (a mat44 inverse); it is now anchor-1 tangent +
    # anchor-2 tangent + anchor-3 twist, the same 2x2/2x2/1x1 shape as the
    # rigid swing family -- no mat44, no cross-anchor coupling matrix.
    has_swing = mode_cfg == JOINT_MODE_REVOLUTE or mode_cfg == JOINT_MODE_FIXED
    has_twist = mode_cfg == JOINT_MODE_FIXED
    is_cable = mode_cfg == JOINT_MODE_CABLE
    is_prismatic = mode_cfg == JOINT_MODE_PRISMATIC

    a1_kind = _D6_ROW_SOLVE_HARD3
    a2_kind = _D6_ROW_SOLVE_SKIP
    a3_kind = _D6_ROW_SOLVE_SKIP
    if is_cable:
        a1_kind = _D6_ROW_SOLVE_SOFT3
        a2_kind = _D6_ROW_SOLVE_PD2_TAN
        a3_kind = _D6_ROW_SOLVE_PD1_SCALAR
    elif is_prismatic:
        # Prismatic solves all 5 positional rows in one coupled 4+1 Schur
        # block (:func:`_d6_solve_prismatic_coupled_at`); the per-anchor
        # kinds stay SKIP so the generic anchor solves below don't fire.
        a1_kind = _D6_ROW_SOLVE_SKIP
        a2_kind = _D6_ROW_SOLVE_SKIP
        a3_kind = _D6_ROW_SOLVE_SKIP
    elif has_twist:
        # FIXED: anchor-1+anchor-2 via the coupled rigid-swing helper; the
        # anchor-3 twist scalar is solved as its own block below.
        a3_kind = _D6_ROW_SOLVE_HARD1_SCALAR

    axial_kind = _D6_AXIAL_NONE
    if is_prismatic:
        axial_kind = _D6_AXIAL_LINEAR
    elif mode_cfg == JOINT_MODE_REVOLUTE or mode_cfg == JOINT_MODE_UNIVERSAL:
        axial_kind = _D6_AXIAL_ANGULAR

    has_limits = mode_cfg == JOINT_MODE_BALL_SOCKET or mode_cfg == JOINT_MODE_UNIVERSAL

    b1 = body_pair.b1
    b2 = body_pair.b2

    (
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    r1_b1 = read_vec3(constraints, base_offset + _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)

    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    # ---- Positional group --------------------------------------------
    # Rigid swing (REVOLUTE / FIXED) solves anchor-1 + anchor-2 as one
    # COUPLED 3+2 Schur (convergence-critical, see
    # :func:`_d6_solve_rigid_swing_coupled_at`). Every other mode solves
    # independent per-anchor blocks (block-Gauss-Seidel): anchor-1 (SOFT3
    # cable / HARD1_TAN prismatic) then anchor-2 tangent. anchor-3 scalar
    # always follows as its own block.
    if has_swing:
        (
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
        ) = _d6_solve_rigid_swing_coupled_at(
            constraints,
            cid,
            base_offset,
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            r1_b1,
            r1_b2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
            use_bias,
        )
    elif is_prismatic:
        (
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
        ) = _d6_solve_prismatic_coupled_at(
            constraints,
            cid,
            base_offset,
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            r1_b1,
            r1_b2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
            use_bias,
        )
    else:
        (
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
        ) = _d6_solve_anchor1_at(
            constraints,
            cid,
            base_offset,
            a1_kind,
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            r1_b1,
            r1_b2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
            use_bias,
        )
        if a2_kind != _D6_ROW_SOLVE_SKIP:
            (
                velocity1,
                angular_velocity1,
                velocity2,
                angular_velocity2,
            ) = _d6_solve_anchor2_tangent_at(
                constraints,
                cid,
                base_offset,
                a2_kind,
                velocity1,
                angular_velocity1,
                velocity2,
                angular_velocity2,
                inv_mass1,
                inv_mass2,
                inv_inertia1,
                inv_inertia2,
                mass_coeff,
                impulse_coeff,
                sor_boost,
                use_bias,
            )
    if a3_kind != _D6_ROW_SOLVE_SKIP:
        (
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
        ) = _d6_solve_anchor3_scalar_at(
            constraints,
            cid,
            base_offset,
            a3_kind,
            velocity1,
            angular_velocity1,
            velocity2,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
            use_bias,
        )

    if axial_kind == _D6_AXIAL_LINEAR:
        # Linear drive / limit along n_hat at anchor 1 (PRISMATIC).
        n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
        clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
        v1_anchor = velocity1 + wp.cross(angular_velocity1, r1_b1)
        v2_anchor = velocity2 + wp.cross(angular_velocity2, r1_b2)
        jv_axial = wp.dot(n_hat, v1_anchor - v2_anchor)
        axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, sor_boost, use_bias)
        axial_imp = n_hat * axial_lam
        velocity1 = velocity1 + inv_mass1 * axial_imp
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ wp.cross(r1_b1, axial_imp)
        velocity2 = velocity2 - inv_mass2 * axial_imp
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ wp.cross(r1_b2, axial_imp)
    elif axial_kind == _D6_AXIAL_ANGULAR:
        # Angular twist drive / limit along n_hat (REVOLUTE/UNIVERSAL).
        n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
        clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
        jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)
        axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, sor_boost, use_bias)
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_lam)
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_lam)

    if has_limits:
        if read_int(constraints, base_offset + _OFF_D6_LIMIT_COUNT, cid) > wp.int32(0):
            angular_velocity1, angular_velocity2 = _d6_angular_limits_block(
                constraints,
                cid,
                base_offset,
                bodies,
                b1,
                mode_cfg,
                angular_velocity1,
                angular_velocity2,
                inv_inertia1,
                inv_inertia2,
                idt,
                sor_boost,
            )

    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    )


@wp.func
def _d6_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    mode_cfg: wp.int32,
):
    """Unified D6 PGS iteration step."""
    _d6_iterate_rows_at(
        constraints,
        cid,
        base_offset,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        sor_boost,
        use_bias,
        mode_cfg,
    )


@wp.func
def _d6_iterate_at_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
    mode_cfg: wp.int32,
):
    """Unified D6 multi-sweep iterate.

    Revolute takes the register-cached :func:`_revolute_iterate_at_multi`
    path; every other mode loops :func:`_d6_iterate_at` ``num_sweeps``
    times. Pass a compile-time constant for ``mode_cfg`` to fold the
    branch.
    """
    if mode_cfg == JOINT_MODE_REVOLUTE:
        _revolute_iterate_at_multi(
            constraints,
            cid,
            base_offset,
            bodies,
            particles,
            copy_state,
            num_bodies,
            parallel_id,
            body_pair,
            idt,
            sor_boost,
            use_bias,
            num_sweeps,
        )
    else:
        it = wp.int32(0)
        while it < num_sweeps:
            _d6_iterate_at(
                constraints,
                cid,
                base_offset,
                bodies,
                particles,
                copy_state,
                num_bodies,
                parallel_id,
                body_pair,
                idt,
                sor_boost,
                use_bias,
                mode_cfg,
            )
            it += 1


# ---------------------------------------------------------------------------
# Mode-dispatching entry points
# ---------------------------------------------------------------------------


@wp.func
def actuated_double_ball_socket_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass that dispatches on ``joint_mode``.

    Reads the per-constraint ``joint_mode`` tag and forwards it to the
    unified :func:`_d6_prepare_at`.
    """
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)
    _d6_prepare_at(
        constraints,
        cid,
        base_offset,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        joint_mode,
    )


@wp.func
def _revolute_iterate_at_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
):
    """Register-cached multi-sweep revolute iterate.

    Equivalent to calling :func:`_d6_iterate_rows_at` (revolute mode)
    ``num_sweeps`` times with the same arguments, but every per-cid constant
    (constraint body anchors, tangent basis, ``a1_inv``, ``ut_ai``,
    ``s_inv`` Schur complement, drift biases, soft-constraint
    coefficients, axial drive / limit params) and both bodies' full
    kinematic state are loaded ONCE and held in registers across
    sweeps. Accumulated impulses (``acc1``, ``acc2_world``, plus drive
    / limit ``acc``) update in registers and are written back at the
    end. Body velocities are written back once.

    Assumes the kernel wraps this in an outer loop that sweeps all
    colours per outer iteration -- ``num_sweeps`` trades some
    cross-colour PGS feedback for register caching. ``num_sweeps = 2``
    keeps 4 outer rounds (at the default ``solver_iterations = 8``)
    which the current test suite tolerates for joints.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    # ---- Body state (hoisted out of the sweep loop) ------------------
    (
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    # ---- Constraint constants ----------------------------------------
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

    a1_inv_s6 = read_vec6(constraints, base_offset + _OFF_A1_INV_S6, cid)
    s_inv_s3 = read_vec3(constraints, base_offset + _OFF_S_INV_S3, cid)
    ut_ai_row0 = read_vec3(constraints, base_offset + _OFF_UT_AI_ROW0, cid)
    ut_ai_row1 = read_vec3(constraints, base_offset + _OFF_UT_AI_ROW1, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
    bias2_tan = wp.vec2f(bias2[0], bias2[1])
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)

    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)

    # ---- Axial drive + limit constants -------------------------------
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    # ``max_lambda_drive`` derived from ``max_force_drive * dt`` -- see
    # the revolute iterate for rationale.
    max_lambda_drive = max_force_drive * (wp.float32(1.0) / idt)
    bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
    gamma_drive = read_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid)
    eff_mass_drive_soft = read_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)

    drive_active = drive_mode != DRIVE_MODE_OFF
    if eff_mass_drive_soft <= wp.float32(0.0):
        drive_active = False

    acc_limit = wp.float32(0.0)
    pd_mode_limit = False
    pd_mass = wp.float32(0.0)
    pd_gamma = wp.float32(0.0)
    pd_beta = wp.float32(0.0)
    eff_axial = wp.float32(0.0)
    bias_box = wp.float32(0.0)
    mc_limit = wp.float32(0.0)
    ic_limit = wp.float32(0.0)
    friction = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
    eff_inv_friction = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
    max_lambda_friction = friction * (wp.float32(1.0) / idt)
    friction_active = friction > wp.float32(0.0)
    if eff_inv_friction <= wp.float32(0.0) or max_lambda_friction <= wp.float32(0.0):
        friction_active = False
    gamma_friction = wp.float32(0.0)
    eff_mass_friction = wp.float32(0.0)
    if friction_active:
        slip_velocity = PHOENX_FRICTION_SLIP_VELOCITY
        slip_scale = read_float(constraints, base_offset + _OFF_FRICTION_SLIP_SCALE, cid)
        if slip_scale > wp.float32(0.0):
            slip_velocity = slip_scale * eff_inv_friction * friction
        gamma_friction = slip_velocity / max_lambda_friction
        eff_mass_friction = wp.float32(1.0) / (eff_inv_friction + gamma_friction)

    limit_active = clamp != _CLAMP_NONE
    if limit_active:
        acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
        stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
        damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
        pd_mode_limit = stiffness_limit > wp.float32(0.0) or damping_limit > wp.float32(0.0)
        if pd_mode_limit:
            pd_mass = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid)
            pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid)
            pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid)
        else:
            eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
            if eff_inv > wp.float32(0.0):
                eff_axial = wp.float32(1.0) / eff_inv
            bias_box = read_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid)
            mc_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
            ic_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)

    # ---- Sweep loop (all state register-resident) --------------------
    it = wp.int32(0)
    while it < num_sweeps:
        # COUPLED positional PGS: anchor-1 point lock (3 rows) +
        # anchor-2 swing tangent (2 rows) solved as one 3+2 Schur, with
        # ``ut_ai = U^T A1^-1`` back-substitution. The coupling is what
        # lets a stiff cantilever hinge chain hold its shape at low
        # iteration counts (see :func:`_d6_solve_rigid_swing_coupled_at`).
        acc2_t1 = wp.dot(t1, acc2_world)
        acc2_t2 = wp.dot(t2, acc2_world)
        acc2_tan = wp.vec2f(acc2_t1, acc2_t2)

        jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
        jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
        jv2 = wp.vec2f(wp.dot(t1, jv2_world), wp.dot(t2, jv2_world))

        rhs1 = jv1 + bias1
        rhs2 = jv2 + bias2_tan

        ut_ai_rhs1 = wp.vec2f(wp.dot(ut_ai_row0, rhs1), wp.dot(ut_ai_row1, rhs1))
        lam2_us = -(mul_sym2(s_inv_s3, rhs2 - ut_ai_rhs1))
        lam2 = mass_coeff * lam2_us - impulse_coeff * acc2_tan
        lam2 = lam2 * sor_boost

        lam2_world = lam2[0] * t1 + lam2[1] * t2
        lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

        u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
        u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
        u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

        lam1_us = -(mul_sym3(a1_inv_s6, rhs1 + u_lam2_us))
        lam1 = mass_coeff * lam1_us - impulse_coeff * acc1
        lam1 = lam1 * sor_boost

        total_lin = lam1 + lam2_world
        velocity1 = velocity1 - inv_mass1 * total_lin
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1 + cr2_b1 @ lam2_world)
        velocity2 = velocity2 + inv_mass2 * total_lin
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1 + cr2_b2 @ lam2_world)

        acc1 = acc1 + lam1
        acc2_world = acc2_world + lam2_world

        # Axial drive + limit scalar PGS
        jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)

        lam_drive = wp.float32(0.0)
        if drive_active:
            lam_drive = -eff_mass_drive_soft * (jv_axial - bias_drive + gamma_drive * acc_drive)
            lam_drive = lam_drive * sor_boost
            old_acc = acc_drive
            acc_drive = acc_drive + lam_drive
            if max_force_drive > wp.float32(0.0):
                acc_drive = wp.clamp(acc_drive, -max_lambda_drive, max_lambda_drive)
            lam_drive = acc_drive - old_acc

        lam_limit = wp.float32(0.0)
        if limit_active:
            if pd_mode_limit:
                if pd_mass > wp.float32(0.0):
                    lam_limit = -pd_mass * (jv_axial - pd_beta + pd_gamma * acc_limit)
            else:
                if eff_axial > wp.float32(0.0):
                    lam_unsoft = -eff_axial * (jv_axial + bias_box)
                    lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
            lam_limit = lam_limit * sor_boost
            old_acc_l = acc_limit
            acc_limit = acc_limit + lam_limit
            if clamp == _CLAMP_MAX:
                acc_limit = wp.max(wp.float32(0.0), acc_limit)
            else:
                acc_limit = wp.min(wp.float32(0.0), acc_limit)
            lam_limit = acc_limit - old_acc_l

        lam_friction = wp.float32(0.0)
        if friction_active:
            lam_friction = -eff_mass_friction * (jv_axial + gamma_friction * acc_friction)
            lam_friction = lam_friction * sor_boost
            old_acc_f = acc_friction
            acc_friction = wp.clamp(
                acc_friction + lam_friction,
                -max_lambda_friction,
                max_lambda_friction,
            )
            lam_friction = acc_friction - old_acc_f

        axial_lam = lam_drive + lam_limit + lam_friction
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_lam)
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_lam)

        it += 1

    # ---- Writeback ---------------------------------------------------
    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    )

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world)
    if drive_active:
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, acc_drive)
    if limit_active:
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, acc_limit)
    if friction_active and use_bias:
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, acc_friction)
    elif friction <= wp.float32(0.0):
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, 0.0)


@wp.func
def actuated_double_ball_socket_iterate_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
):
    """Multi-sweep ADBS iterate dispatcher.

    Revolute -- by far the most common mode in G1/H1 -- takes the
    register-cached :func:`_revolute_iterate_at_multi` path; every
    other mode falls back to a plain loop of per-sweep dispatches.
    The fallback sees no register-caching win but still honours the
    same ``num_sweeps`` contract, so the kernel can call this uniformly
    regardless of mode.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    joint_mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    _d6_iterate_at_multi(
        constraints,
        cid,
        0,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        sor_boost,
        use_bias,
        num_sweeps,
        joint_mode,
    )


@wp.func
def actuated_double_ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Composable PGS iteration step that dispatches on ``joint_mode``.

    ``use_bias`` is the Box2D v3 TGS-soft ``useBias`` flag. During the
    main solve pass pass ``True`` to apply positional drift correction
    via the prepared lock biases; during the relax pass pass ``False``
    so the anchor-lock rows enforce ``Jv = 0`` without re-injecting
    position-error velocity (axial drive / limit / motor targets stay
    on in both passes -- see the per-mode iterates).
    """
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)
    _d6_iterate_at(
        constraints,
        cid,
        base_offset,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        sor_boost,
        use_bias,
        joint_mode,
    )


@wp.func
def actuated_double_ball_socket_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench the joint applies on body 2.

    Sums the anchor impulses (converted to force via ``idt``) and the
    axial drive / limit contribution where applicable. Revolute reports
    the axial impulse as a torque about ``-n_hat``; prismatic reports
    it as a force along ``-n_hat`` (same sign convention as the
    iterate). Ball-socket has no anchor-2/anchor-3 rows and no axial
    block, so only the anchor-1 impulse contributes.
    """
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc3 = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
    acc_axial = acc_drive + acc_limit + acc_friction

    if joint_mode == JOINT_MODE_REVOLUTE:
        force = (acc1 + acc2) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt)
        # Axial block is a torque about -n_hat.
        torque = torque - n_hat * (acc_axial * idt)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
        # Axial block is a linear force along -n_hat.
        axial_force = n_hat * (acc_axial * idt)
        force = force - axial_force
        torque = torque - wp.cross(r1_b2, axial_force)
    elif joint_mode == JOINT_MODE_UNIVERSAL:
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt) - n_hat * (acc_axial * idt) - acc2 * idt
    elif joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_CABLE:
        # Same anchor layout (anchor-1 3-row + anchor-2 tangent 2-row +
        # anchor-3 scalar 1-row); no axial block. CABLE's PD softness
        # is already baked into the accumulated impulses, so the
        # wrench reflects the actual reaction the joint applied this
        # substep.
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    else:
        # Ball-socket: anchor-1 impulse plus optional D6 angular-limit torque.
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt)
        if read_int(constraints, base_offset + _OFF_D6_LIMIT_COUNT, cid) > wp.int32(0):
            torque = torque - acc2 * idt
    return force, torque


@wp.func
def actuated_double_ball_socket_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Direct prepare entry; see
    :func:`actuated_double_ball_socket_prepare_for_iteration_at`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    # Joint constraints are velocity-level. Flip both endpoints to
    # VELOCITY_LEVEL so any prior position-level write (e.g. a cloth
    # iterate touching a body that's also a cloth node, future use)
    # is finite-diffed into velocity before we read it. No-op when
    # the body is already VELOCITY_LEVEL or STATIC.
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_prepare_for_iteration_at(
        constraints, cid, 0, bodies, particles, copy_state, num_bodies, parallel_id, body_pair, idt
    )


@wp.func
def actuated_double_ball_socket_cached_warmstart(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    # The restored box3d_5 formulation has no separate cached-prepare path.
    # Re-running prepare is the conservative compatibility behaviour.
    actuated_double_ball_socket_prepare_for_iteration(
        constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
    )


@wp.func
def actuated_double_ball_socket_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Direct iterate entry; see
    :func:`actuated_double_ball_socket_iterate_at`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_iterate_at(
        constraints, cid, 0, bodies, particles, copy_state, num_bodies, parallel_id, body_pair, idt, sor_boost, use_bias
    )


@wp.func
def revolute_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    """Revolute-only iterate entry, skipping the ``joint_mode``
    dispatch.

    Equivalent to :func:`actuated_double_ball_socket_iterate` for
    revolute joints, but the ``read_int(_OFF_JOINT_MODE)`` and the
    branch into ball-socket / prismatic / fixed / cable code are
    removed. Used by the single-world solver kernels when the scene
    contains only revolute joints (or no joints at all): see
    :attr:`PhoenXWorld._all_joints_revolute`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    _d6_iterate_at(
        constraints,
        cid,
        0,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        sor_boost,
        use_bias,
        JOINT_MODE_REVOLUTE,
    )


@wp.func
def revolute_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Revolute-only prepare entry, skipping the ``joint_mode``
    dispatch. Counterpart of :func:`revolute_iterate`."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    _d6_prepare_at(
        constraints,
        cid,
        0,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        JOINT_MODE_REVOLUTE,
    )


@wp.func
def revolute_cached_warmstart(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    # The restored box3d_5 formulation has no separate cached-prepare path.
    # Re-running prepare is the conservative compatibility behaviour.
    revolute_prepare_for_iteration(constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt)


@wp.func
def revolute_iterate_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
    num_sweeps: wp.int32,
):
    """Revolute-only multi-sweep iterate entry.

    Equivalent to :func:`actuated_double_ball_socket_iterate_multi` for
    revolute joints, but skips the ``read_int(_OFF_JOINT_MODE)``
    global load and the joint-mode branch. Used by the multi-world
    fast-tail kernels when every joint is revolute.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_pair = constraint_bodies_make(b1, b2)
    _d6_iterate_at_multi(
        constraints,
        cid,
        0,
        bodies,
        particles,
        copy_state,
        num_bodies,
        parallel_id,
        body_pair,
        idt,
        sor_boost,
        use_bias,
        num_sweeps,
        JOINT_MODE_REVOLUTE,
    )


@wp.func
def actuated_double_ball_socket_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame (force, torque) this constraint exerts on body 2.

    Units: [N], [N*m]. See
    :func:`actuated_double_ball_socket_world_wrench_at` for details.
    """
    return actuated_double_ball_socket_world_wrench_at(constraints, cid, 0, idt)


@wp.func
def actuated_double_ball_socket_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
) -> wp.spatial_vector:
    """Position-level constraint residual for the unified joint.

    Covers REVOLUTE / PRISMATIC / BALL_SOCKET + optional actuator.

    * ``spatial_top``   = anchor 1 drift ``p1_b2 - p1_b1`` (all 3
      components in revolute / ball-socket; tangential only in
      prismatic -- axial is the free DoF).
    * ``spatial_bottom`` = ``(drift_t1_anchor2, drift_t2_anchor2,
      actuator_residual)``. Anchor-2 tangents are the extra 2
      positional rows in revolute / prismatic (zero in ball-socket).
      The actuator residual is
      ``cumulative_angle_or_slide - target`` (``DRIVE_MODE_POSITION``)
      plus ``- limit`` when clamped, else zero; drive and limit add
      when both active.

    Revolute uses the persisted revolution tracker; prismatic
    recomputes the slide from the current pose; ball-socket reports
    only anchor-1 drift.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]

    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)

    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    p1_b1 = pos1 + wp.quat_rotate(q1, la1_b1)
    p1_b2 = pos2 + wp.quat_rotate(q2, la1_b2)
    anchor1_drift = p1_b2 - p1_b1

    # Anchor 2 tangent drift (revolute / prismatic only). Project onto
    # the persisted tangent basis written by the last prepare pass; the
    # basis is stable across substeps.
    drift_t1 = wp.float32(0.0)
    drift_t2 = wp.float32(0.0)
    if joint_mode != JOINT_MODE_BALL_SOCKET and joint_mode != JOINT_MODE_UNIVERSAL:
        la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
        p2_b1 = pos1 + wp.quat_rotate(q1, la2_b1)
        p2_b2 = pos2 + wp.quat_rotate(q2, la2_b2)
        t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
        t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
        anchor2_drift = p2_b2 - p2_b1
        drift_t1 = wp.dot(t1, anchor2_drift)
        drift_t2 = wp.dot(t2, anchor2_drift)

    # Actuator residual (drive position error OR active limit C).
    actuator_err = wp.float32(0.0)
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    target = read_float(constraints, base_offset + _OFF_TARGET, cid)

    if joint_mode == JOINT_MODE_REVOLUTE:
        counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
        prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
        cumulative = revolution_tracker_angle(counter, prev)
        if drive_mode == DRIVE_MODE_POSITION:
            actuator_err = actuator_err + (cumulative - target)
        if min_value <= max_value:
            if cumulative > max_value:
                actuator_err = actuator_err + (cumulative - max_value)
            elif cumulative < min_value:
                actuator_err = actuator_err + (cumulative - min_value)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        # Recompute slide from anchors + rest_length (same expression
        # as the D6 linear-slider prepare rows). The axial sign matches the
        # prepare convention: slide > 0 when anchor 2 on body 2 has
        # moved past its rest position along the world axis.
        axis_local1 = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
        rest_length = read_float(constraints, base_offset + _OFF_REST_LENGTH, cid)
        la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
        p2_b1 = pos1 + wp.quat_rotate(q1, la2_b1)
        p2_b2 = pos2 + wp.quat_rotate(q2, la2_b2)
        n_hat = wp.quat_rotate(q1, axis_local1)
        slide = wp.dot(n_hat, p2_b2 - p2_b1) - rest_length
        if drive_mode == DRIVE_MODE_POSITION:
            actuator_err = actuator_err + (slide - target)
        if min_value <= max_value:
            if slide > max_value:
                actuator_err = actuator_err + (slide - max_value)
            elif slide < min_value:
                actuator_err = actuator_err + (slide - min_value)
    elif joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_CABLE:
        # Anchor-3 scalar drift along the persisted ``t2`` (the 6th
        # locked DoF). FIXED has no drive / limit; CABLE has no axial
        # drive / limit either (its bend / twist gains live in the
        # drive / limit slots but enter the iterate as PD soft
        # coefficients on the anchor-2 / anchor-3 rows). Reported in
        # the "actuator" slot for consistency with FIXED.
        la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
        la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
        p3_b1 = pos1 + wp.quat_rotate(q1, la3_b1)
        p3_b2 = pos2 + wp.quat_rotate(q2, la3_b2)
        t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
        actuator_err = wp.dot(t2, p3_b2 - p3_b1)

    return wp.spatial_vector(anchor1_drift, wp.vec3f(drift_t1, drift_t2, actuator_err))


@wp.func
def actuated_double_ball_socket_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`actuated_double_ball_socket_world_error_at`."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return actuated_double_ball_socket_world_error_at(constraints, cid, 0, bodies, body_pair)
