# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Store PhoenX joint columns and solve their unilateral rows.

Bilateral equalities are assembled per connected mechanism and solved by the
direct maximal-coordinate system. The PGS-facing routines in this module are
limited to joint limits and friction.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.access_mode import ACCESS_MODE_VELOCITY_LEVEL
from newton._src.solvers.phoenx.body import (
    MOTION_STATIC,
    BodyContainer,
    body_load_inv_inertia_sym6,
    body_load_orientation,
    body_load_vw,
    body_set_access_mode,
    body_store_vw,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_read_multiplier,
    constraint_read_multiplier_vec3,
    constraint_set_type,
    constraint_write_multiplier,
    constraint_write_multiplier_vec3,
    pd_coefficients,
    read_float,
    read_int,
    read_quat,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.helpers.math_helpers import (
    create_orthonormal,
    extract_rotation_angle,
    inv_sym3,
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
    PHOENX_BOOST_PRISMATIC_LIMIT,
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
    "actuated_double_ball_socket_clear_reset_worlds",
    "actuated_double_ball_socket_initialize_kernel",
    "actuated_double_ball_socket_prepare_inequality",
    "actuated_double_ball_socket_world_error",
    "actuated_double_ball_socket_world_error_at",
    "actuated_double_ball_socket_world_wrench",
    "actuated_double_ball_socket_world_wrench_at",
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
    shared. See field-level ``#:``
    comments for individual slot semantics.
    """

    # ---- Header -------------------------------------------------------
    constraint_type: wp.int32
    body1: wp.int32
    body2: wp.int32

    # ---- Shared positional block -------------------------------------
    joint_mode: wp.int32
    structural_direct: wp.int32
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
    # Prismatic (13 used): [0..2] local_anchor3_b1, [3..5] local_anchor3_b2,
    #     [6..8] r3_b1, [9..11] r3_b2, [15] bias3. Dwords [12..14]
    #     are free here; the third impulse lives in the multiplier sidecar.
    # Revolute  (6 used, 10 unused tail):
    #     [0..3] inv_initial_orientation (quat),
    #     [4] revolution_counter, [5] previous_quaternion_angle.
    mode_extras: wp.types.vector(length=16, dtype=wp.float32)
    # Mutable warm-start impulses live in the family-aliased
    # ``ConstraintContainer.multipliers`` sidecar.

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
    # Cached scalar inverse effective mass for the axial row,
    # ``J M^-1 J^T`` (used by the Box2D limit path).
    eff_inv_axial: wp.float32
    # Friction may need a direct-structural Schur correction while drives
    # retain their existing maximal-coordinate response.
    eff_inv_friction: wp.float32
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
    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(ActuatedDoubleBallSocketData)


# Dword offsets derived once from the schema. Named per field.
_OFF_BODY1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body2"))
_OFF_JOINT_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "joint_mode"))
_OFF_STRUCTURAL_DIRECT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "structural_direct"))
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

_OFF_AXIS_LOCAL1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_local1"))
_OFF_REST_LENGTH = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "rest_length"))
_OFF_DRIVE_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "drive_mode"))
_OFF_TARGET = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target"))
_OFF_TARGET_VELOCITY = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target_velocity"))
_OFF_MAX_FORCE_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_force_drive"))
_OFF_STIFFNESS_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_drive"))
_OFF_DAMPING_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_drive"))
_OFF_FRICTION_COEFFICIENT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_coefficient"))
_OFF_FRICTION_SLIP_SCALE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_slip_scale"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_value"))
_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz_limit"))
_OFF_DAMPING_RATIO_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio_limit"))
_OFF_STIFFNESS_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_limit"))
_OFF_DAMPING_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_limit"))
_OFF_EFF_INV_AXIAL = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "eff_inv_axial"))
_OFF_EFF_INV_FRICTION = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "eff_inv_friction"))
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
# Family-aliased mutable state in three aligned vec4 groups: impulse.xyz and
# its correlated limit/friction scalar in w.
_MUL_ACC_IMP1 = wp.constant(wp.int32(0))
_MUL_ACC_IMP2 = wp.constant(wp.int32(4))
_MUL_ACC_LIMIT = wp.constant(wp.int32(7))
_MUL_ACC_IMP3 = wp.constant(wp.int32(8))
_MUL_ACC_FRICTION = wp.constant(wp.int32(11))
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
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP1, cid, zero3)
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid, zero3)

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
        constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP3, cid, zero3)
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
    write_float(constraints, _OFF_FRICTION_COEFFICIENT, cid, friction_coefficient[tid])
    write_float(constraints, _OFF_FRICTION_SLIP_SCALE, cid, friction_slip_scale[tid])
    write_float(constraints, _OFF_MIN_VALUE, cid, min_value[tid])
    write_float(constraints, _OFF_MAX_VALUE, cid, max_value[tid])
    write_float(constraints, _OFF_HERTZ_LIMIT, cid, hertz_limit[tid])
    write_float(constraints, _OFF_DAMPING_RATIO_LIMIT, cid, damping_ratio_limit[tid])
    write_float(constraints, _OFF_STIFFNESS_LIMIT, cid, stiffness_limit[tid])
    write_float(constraints, _OFF_DAMPING_LIMIT, cid, damping_limit[tid])
    write_float(constraints, _OFF_EFF_INV_AXIAL, cid, 0.0)
    write_float(constraints, _OFF_EFF_INV_FRICTION, cid, 0.0)
    # ``limit_cache`` is mode-aliased Box2D vs PD; one zero-fill of
    # the 3 shared dwords covers both layouts. Prepare overwrites
    # them every substep based on the limit type.
    write_float(constraints, _OFF_LIMIT_CACHE + 0, cid, 0.0)
    write_float(constraints, _OFF_LIMIT_CACHE + 1, cid, 0.0)
    write_float(constraints, _OFF_LIMIT_CACHE + 2, cid, 0.0)
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, n_hat_init)
    constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, 0.0)
    constraint_write_multiplier(constraints, _MUL_ACC_FRICTION, cid, 0.0)

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
        constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, wp.float32(0.0))
    else:
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, wp.int32(0))
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, wp.float32(0.0))
        if mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL:
            write_vec3(constraints, _OFF_D6_LIMIT_EFF_INV, cid, zero3)

    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP1, cid, zero3)
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid, zero3)
    write_float(constraints, _OFF_EFF_INV_AXIAL, cid, wp.float32(0.0))
    write_float(constraints, _OFF_EFF_INV_FRICTION, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 0, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 1, cid, wp.float32(0.0))
    write_float(constraints, _OFF_LIMIT_CACHE + 2, cid, wp.float32(0.0))
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, zero3)
    constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, wp.float32(0.0))
    constraint_write_multiplier(constraints, _MUL_ACC_FRICTION, cid, wp.float32(0.0))


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
        v1, w1 = body_load_vw(bodies, b1)
        v2, w2 = body_load_vw(bodies, b2)
        inv_mass1 = bodies.inverse_mass[b1]
        inv_mass2 = bodies.inverse_mass[b2]
        inv_inertia1 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b1))
        inv_inertia2 = mat33_from_sym6(body_load_inv_inertia_sym6(bodies, b2))
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
        body_store_vw(bodies, b1, v1, w1)
        body_store_vw(bodies, b2, v2, w2)
        return
    write_velocity_unified(bodies, particles, copy_state, b1, slot1, num_bodies, v1)
    write_velocity_unified(bodies, particles, copy_state, b2, slot2, num_bodies, v2)
    write_angular_velocity_unified(bodies, copy_state, b1, slot1, w1)
    write_angular_velocity_unified(bodies, copy_state, b2, slot2, w2)


# ---------------------------------------------------------------------------
# Shared axial limit and friction iterate helper
# ---------------------------------------------------------------------------


@wp.func
def _axial_limit_friction_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    jv_axial: wp.float32,
    clamp: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    store_friction: wp.bool,
) -> wp.float32:
    """Drive-free axial PGS step for direct-owned scalar joints."""
    lam_limit = wp.float32(0.0)
    if clamp != _CLAMP_NONE:
        stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
        damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
        acc_limit = constraint_read_multiplier(constraints, _MUL_ACC_LIMIT, cid)
        if stiffness_limit > wp.float32(0.0) or damping_limit > wp.float32(0.0):
            pd_mass = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid)
            pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid)
            pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid)
            if pd_mass > wp.float32(0.0):
                lam_limit = -pd_mass * (jv_axial - pd_beta + pd_gamma * acc_limit)
        else:
            eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
            if eff_inv > wp.float32(0.0):
                bias_box = read_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid)
                mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
                impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
                lam_unsoft = -(jv_axial + bias_box) / eff_inv
                lam_limit = mass_coeff * lam_unsoft - impulse_coeff * acc_limit
        old_acc_limit = acc_limit
        acc_limit += lam_limit * sor_boost
        if clamp == _CLAMP_MAX:
            acc_limit = wp.max(wp.float32(0.0), acc_limit)
        else:
            acc_limit = wp.min(wp.float32(0.0), acc_limit)
        lam_limit = acc_limit - old_acc_limit
        constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, acc_limit)

    lam_friction = wp.float32(0.0)
    friction = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    acc_friction = constraint_read_multiplier(constraints, _MUL_ACC_FRICTION, cid)
    if friction > wp.float32(0.0):
        eff_inv_friction = read_float(constraints, base_offset + _OFF_EFF_INV_FRICTION, cid)
        max_lambda_friction = friction / idt
        if eff_inv_friction > wp.float32(0.0) and max_lambda_friction > wp.float32(0.0):
            slip_velocity = PHOENX_FRICTION_SLIP_VELOCITY
            slip_scale = read_float(constraints, base_offset + _OFF_FRICTION_SLIP_SCALE, cid)
            if slip_scale > wp.float32(0.0):
                slip_velocity = slip_scale * eff_inv_friction * friction
            gamma_friction = slip_velocity / max_lambda_friction
            effective_mass = wp.float32(1.0) / (eff_inv_friction + gamma_friction)
            lam_friction = -effective_mass * (jv_axial + gamma_friction * acc_friction) * sor_boost
            old_acc_friction = acc_friction
            acc_friction = wp.clamp(
                acc_friction + lam_friction,
                -max_lambda_friction,
                max_lambda_friction,
            )
            lam_friction = acc_friction - old_acc_friction
            if store_friction:
                constraint_write_multiplier(constraints, _MUL_ACC_FRICTION, cid, acc_friction)
    else:
        constraint_write_multiplier(constraints, _MUL_ACC_FRICTION, cid, wp.float32(0.0))

    return lam_limit + lam_friction


# ---------------------------------------------------------------------------
# Shared axial limit and friction prepare helper
# ---------------------------------------------------------------------------


@wp.func
def _axial_limit_friction_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    cumulative_value: wp.float32,
    eff_inv: wp.float32,
    eff_inv_friction: wp.float32,
    dt: wp.float32,
    limit_boost: wp.float32,
) -> wp.float32:
    """Prepare unilateral limit and friction state for one free axial row."""
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)

    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, eff_inv)
    write_float(constraints, base_offset + _OFF_EFF_INV_FRICTION, cid, eff_inv_friction)

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

    # Warm-start the active limit and friction impulses, with
    # ``acc_limit`` forcibly zeroed when the limit is inactive.
    acc_limit = constraint_read_multiplier(constraints, _MUL_ACC_LIMIT, cid)
    if clamp == _CLAMP_NONE:
        acc_limit = 0.0
        constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, 0.0)
    acc_friction = constraint_read_multiplier(constraints, _MUL_ACC_FRICTION, cid)
    friction = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    if friction <= 0.0:
        acc_friction = 0.0
        constraint_write_multiplier(constraints, _MUL_ACC_FRICTION, cid, 0.0)
    return acc_limit + acc_friction


# ---------------------------------------------------------------------------
# Shared tangent-basis-from-anchor-3 helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unified per-anchor solve blocks (iterate pass)
# ---------------------------------------------------------------------------
#
# One block per anchor, selected by a runtime ``solve_kind`` tag. These are
# the single definitions of each per-anchor numeric formula; the unified
# :func:`_d6_iterate_rows_at` body and the prismatic Schur path are the only
# callers. Each returns the updated body velocities.


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

    old_acc = constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid)
    acc = wp.vec3f(0.0, 0.0, 0.0)
    if bias0 != wp.float32(0.0):
        acc = acc + wp.dot(axis0, old_acc) * axis0
    if bias1 != wp.float32(0.0):
        acc = acc + wp.dot(axis1, old_acc) * axis1
    if bias2 != wp.float32(0.0):
        acc = acc + wp.dot(axis2, old_acc) * axis2
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid, acc)
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

    orientation1 = body_load_orientation(bodies, b1)
    axis0 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(0)))
    axis1 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(1)))
    axis2 = wp.quat_rotate(orientation1, _d6_limit_axis_local(constraints, cid, base_offset, joint_mode, wp.int32(2)))

    bias = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    eff_inv = read_vec3(constraints, base_offset + _OFF_D6_LIMIT_EFF_INV, cid)
    acc = constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid)
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
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid, new_acc)
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


# ---------------------------------------------------------------------------
# Mode-dispatching entry points
# ---------------------------------------------------------------------------


@wp.func
def actuated_double_ball_socket_prepare_inequality(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Prepare only free-axis limit and friction rows."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    orientation1 = body_load_orientation(bodies, b1)
    orientation2 = body_load_orientation(bodies, b2)
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

    la1_b1 = read_vec3(constraints, _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, _OFF_LA1_B2, cid)
    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    write_vec3(constraints, _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, _OFF_R1_B2, cid, r1_b2)
    axis = wp.normalize(wp.quat_rotate(orientation1, read_vec3(constraints, _OFF_AXIS_LOCAL1, cid)))
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, axis)
    dt = wp.float32(1.0) / idt

    if mode == JOINT_MODE_REVOLUTE or mode == JOINT_MODE_PRISMATIC:
        metric = _d6_metric_anchor_block(
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            r1_b1,
            r1_b2,
            r1_b1,
            r1_b2,
        )
        if mode == JOINT_MODE_PRISMATIC:
            eff_inv = wp.dot(axis, metric @ axis)
            slide = wp.dot(axis, position2 + r1_b2 - position1 - r1_b1)
            axial_impulse = _axial_limit_friction_prepare_at(
                constraints,
                cid,
                wp.int32(0),
                slide,
                eff_inv,
                eff_inv,
                dt,
                PHOENX_BOOST_PRISMATIC_LIMIT,
            )
            impulse = axis * axial_impulse
            velocity1 += inv_mass1 * impulse
            angular_velocity1 += inv_inertia1 @ wp.cross(r1_b1, impulse)
            velocity2 -= inv_mass2 * impulse
            angular_velocity2 -= inv_inertia2 @ wp.cross(r1_b2, impulse)
        else:
            eff_inv = wp.dot(axis, inv_inertia1 @ axis) + wp.dot(axis, inv_inertia2 @ axis)
            coupling = wp.cross(r1_b1, inv_inertia1 @ axis) + wp.cross(r1_b2, inv_inertia2 @ axis)
            metric_inverse = inv_sym3(sym6_from_mat33_upper(metric))
            eff_inv_friction = wp.max(
                wp.float32(0.0),
                eff_inv - wp.dot(coupling, mul_sym3(metric_inverse, coupling)),
            )
            inv_initial = read_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid)
            difference = orientation2 * inv_initial * wp.quat_inverse(orientation1)
            wrapped = extract_rotation_angle(difference, axis)
            old_counter = read_int(constraints, _OFF_REVOLUTION_COUNTER, cid)
            old_previous = read_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
            counter, previous = revolution_tracker_update(wrapped, old_counter, old_previous)
            write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, counter)
            write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, previous)
            coordinate = revolution_tracker_angle(counter, previous)
            axial_impulse = _axial_limit_friction_prepare_at(
                constraints,
                cid,
                wp.int32(0),
                coordinate,
                eff_inv,
                eff_inv_friction,
                dt,
                PHOENX_BOOST_REVOLUTE_LIMIT,
            )
            angular_velocity1 += inv_inertia1 @ (axis * axial_impulse)
            angular_velocity2 -= inv_inertia2 @ (axis * axial_impulse)
    elif mode == JOINT_MODE_BALL_SOCKET or mode == JOINT_MODE_UNIVERSAL:
        count = read_int(constraints, _OFF_D6_LIMIT_COUNT, cid)
        if count > wp.int32(0):
            angular_velocity1, angular_velocity2 = _d6_angular_limits_prepare_at(
                constraints,
                cid,
                wp.int32(0),
                mode,
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
    acc1 = constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP1, cid)
    acc2 = constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid)
    acc3 = constraint_read_multiplier_vec3(constraints, _MUL_ACC_IMP3, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    acc_limit = constraint_read_multiplier(constraints, _MUL_ACC_LIMIT, cid)
    acc_friction = constraint_read_multiplier(constraints, _MUL_ACC_FRICTION, cid)
    acc_axial = acc_limit + acc_friction

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
