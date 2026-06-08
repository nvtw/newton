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

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_block import (
    BLOCK_LAMBDA_INF,
    VelocityBlock1,
    VelocityBlock2,
    VelocityBlock3,
    VelocityBlock4,
    VelocityRows3,
    block_project_accumulated_2,
    block_project_accumulated_3,
    block_solve_inverse_2,
    block_solve_inverse_3,
    block_solve_velocity_block1,
    block_solve_velocity_block2,
    block_solve_velocity_block3,
    block_solve_velocity_block4,
    block_solve_velocity_rows3_bounded,
)
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
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_mat44,
    write_quat,
    write_vec3,
    write_vec4,
)
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.helpers.math_helpers import (
    apply_pair_angular_impulse,
    apply_pair_spatial_impulse,
    create_orthonormal,
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
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
#: Universal (Hooke) joint: locks 3 translational DoFs + 1 rotational
#: DoF about a user-specified axis. Two rotational DoFs perpendicular to
#: the locked axis are free. Composed from BALL_SOCKET (anchor-1 3-row
#: positional lock) + REVOLUTE's axial machinery used in *rigid-limit*
#: mode (``min_value == max_value == 0``, rigid ``hertz_limit``) to
#: enforce the 1-row angular lock about ``axis_local1``. Reuses the
#: revolute twist tracker for cumulative-angle position correction.
#:
#: This is the D6 universal pattern (3 lin locked + 2 ang locked + 1 ang
#: free → REVOLUTE; 3 lin locked + 1 ang locked + 2 ang free → UNIVERSAL).
#: Drives / friction on the two free angular DoFs are not yet stored in
#: the schema -- a Phase 2 follow-up may add them.
JOINT_MODE_UNIVERSAL = wp.constant(wp.int32(5))
#: Cylindrical joint: 1 linear free + 1 rotational free, both along the
#: same axis. The other 4 DoFs (2 linear + 2 rotational, all
#: perpendicular to the axis) are locked. Geometrically: a piston that
#: can both slide along and spin about the cylinder axis.
#:
#: Reuses PRISMATIC's anchor-1 + anchor-2 *tangent* prepare (4-row K4
#: block in ``mode_cache``'s 4x4 slot) but omits the anchor-3 scalar
#: lock that PRISMATIC uses to gate the rotation about ``n_hat``. The
#: Schur complement collapses: with no anchor-3 row the 4x4 K4 inverse
#: is the whole positional block, so no Schur math is needed and the
#: stored ``s_scalar_inv`` / ``c`` slots stay zero.
#:
#: Phase 2 MVP is kinematic-only: both free DoFs are unactuated (no
#: drive, no limit, no friction). The existing axial drive row stays
#: in the schema but is set to OFF by the adapter -- it costs only a
#: read in the iterate gate. A follow-up may add paired axial rows for
#: driving the slide and the spin independently.
JOINT_MODE_CYLINDRICAL = wp.constant(wp.int32(6))
#: Planar joint: 2 linear free (in-plane) + 1 rotational free (about
#: plane normal). The locked linear axis is parallel to the free
#: rotational axis (= the plane normal). Geometrically: a puck on an
#: air-hockey table, a wheeled mobile base on flat ground, planar
#: tiles in a puzzle.
#:
#: 3 constraint rows: 1-row linear lock at anchor 1 along ``n_hat``
#: (kills out-of-plane translation), 2-row angular lock at anchor 2
#: perpendicular to ``n_hat`` (kills off-axis rotations -- same form
#: as REVOLUTE's anchor-2 tangent rows). The 3x3 positional Schur is
#: solved directly via ``wp.inverse(mat33)``. No anchor-3 row --
#: rotation about ``n_hat`` is free.
#:
#: Phase 2 MVP is kinematic-only: no drives / limits / friction on
#: any of the 3 free DoFs. Adding drives on the in-plane translations
#: requires paired axial rows along ``t1, t2`` (Phase 2 follow-up).
JOINT_MODE_PLANAR = wp.constant(wp.int32(7))


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
    # Coulomb-friction row (independent of drive). Operates on the same
    # axial scalar (revolute twist or prismatic slide). Acts as a
    # regularized saturated damper toward zero relative velocity:
    # ``τ = clamp(-friction_eff_mass * jv_axial, -μ, +μ)``. ``μ`` and
    # the drive's clamped output add as independent impulses (matches
    # MuJoCo's ``dof_frictionloss + actuator`` decomposition).
    #   friction_coefficient -- μ [N·m for revolute, N for prismatic];
    #                           ``0`` disables.
    #   friction_gamma       -- regularization (cached at prepare;
    #                           :data:`PHOENX_FRICTION_SLIP_VELOCITY`
    #                           / (μ * dt) so the saturation slip
    #                           velocity equals the configured constant
    #                           regardless of joint impedance).
    #   friction_eff_mass    -- 1 / (eff_inv_axial + friction_gamma)
    #                           (cached at prepare).
    friction_coefficient: wp.float32
    friction_gamma: wp.float32
    friction_eff_mass: wp.float32
    accumulated_impulse_friction: wp.float32
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
_OFF_A4_INV = wp.constant(int(_OFF_MODE_CACHE) + 0)
_OFF_C_PRIS = wp.constant(int(_OFF_MODE_CACHE) + 16)
_OFF_S_SCALAR_INV = wp.constant(int(_OFF_MODE_CACHE) + 20)
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
# Revolute-only fields, dwords 0..5 of mode_extras (10 unused tail):
_OFF_INV_INITIAL_ORIENTATION = wp.constant(int(_OFF_MODE_EXTRAS) + 0)
_OFF_REVOLUTION_COUNTER = wp.constant(int(_OFF_MODE_EXTRAS) + 4)
_OFF_PREVIOUS_QUATERNION_ANGLE = wp.constant(int(_OFF_MODE_EXTRAS) + 5)
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
_OFF_FRICTION_COEFFICIENT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_coefficient"))
_OFF_FRICTION_GAMMA = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_gamma"))
_OFF_FRICTION_EFF_MASS = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "friction_eff_mass"))
_OFF_ACC_FRICTION = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_friction"))
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


@wp.func
def _read_s_inv_22(constraints: ConstraintContainer, base_offset: wp.int32, cid: wp.int32) -> wp.mat22f:
    """Read the live 2x2 Schur inverse packed at the front of ``S_INV``."""
    off = base_offset + _OFF_S_INV
    return wp.mat22f(
        read_float(constraints, off + wp.int32(0), cid),
        read_float(constraints, off + wp.int32(1), cid),
        read_float(constraints, off + wp.int32(2), cid),
        read_float(constraints, off + wp.int32(3), cid),
    )


@wp.func
def _write_s_inv_22(constraints: ConstraintContainer, base_offset: wp.int32, cid: wp.int32, s_inv: wp.mat22f):
    """Write only the live 2x2 Schur inverse entries."""
    off = base_offset + _OFF_S_INV
    write_float(constraints, off + wp.int32(0), cid, s_inv[0, 0])
    write_float(constraints, off + wp.int32(1), cid, s_inv[0, 1])
    write_float(constraints, off + wp.int32(2), cid, s_inv[1, 0])
    write_float(constraints, off + wp.int32(3), cid, s_inv[1, 1])


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
        friction_coefficient: Coulomb friction limit on the axial DoF
            [N*m for revolute, N for prismatic]. Acts independently of
            the drive: the total axial impulse is the sum of the
            clamped drive PD term and the clamped friction term
            (matches MuJoCo's ``dof_frictionloss + actuator``
            decomposition). ``0`` disables friction on that joint.
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

    # ``mode_extras`` block is mode-aliased: REVOLUTE / UNIVERSAL store
    # the twist-tracker scratch (inv_initial_orientation, revolution_counter,
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
        # REVOLUTE / BALL_SOCKET: zero out the anchor-3 slots
        # via the twist-tracker layout. BALL_SOCKET reads neither side
        # of the alias, so any consistent zero is fine.
        write_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid, inv_initial_orientation)
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, 0)
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, 0.0)

    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, 0.0)

    eye3 = wp.identity(3, dtype=wp.float32)
    write_mat33(constraints, _OFF_A1_INV, cid, eye3)
    write_mat33(constraints, _OFF_UT_AI, cid, eye3)
    write_mat33(constraints, _OFF_S_INV, cid, eye3)
    eye4 = wp.identity(4, dtype=wp.float32)
    write_mat44(constraints, _OFF_A4_INV, cid, eye4)
    write_vec4(constraints, _OFF_C_PRIS, cid, wp.vec4f(0.0, 0.0, 0.0, 0.0))
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
    write_float(constraints, _OFF_FRICTION_COEFFICIENT, cid, friction_coefficient[tid])
    write_float(constraints, _OFF_FRICTION_GAMMA, cid, 0.0)
    write_float(constraints, _OFF_FRICTION_EFF_MASS, cid, 0.0)
    write_float(constraints, _OFF_ACC_FRICTION, cid, 0.0)


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
        inv_inertia1 = bodies.inverse_inertia_world[b1]
        inv_inertia2 = bodies.inverse_inertia_world[b2]
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
    inv_inertia1 = bodies.inverse_inertia_world[b1] * inv_f1
    inv_inertia2 = bodies.inverse_inertia_world[b2] * inv_f2
    return v1, v2, w1, w2, inv_mass1, inv_mass2, inv_inertia1, inv_inertia2, slot1, slot2


@wp.func
def _ms_load_body_pair_lean(
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    b1: wp.int32,
    b2: wp.int32,
    parallel_id: wp.int32,
    num_bodies: wp.int32,
):
    """Mass-splitting-free variant of :func:`_ms_load_body_pair`.
    Direct SoA reads with no ``copy_state`` touch, no slot logic, no
    Tonge ``inv_factor`` multiply. Returns ``slot1 = slot2 = -1`` so
    the matching :func:`_ms_store_body_pair_lean` writeback fires.

    Used when ``has_mass_splitting=False`` is known at kernel-compile
    time (the common multi-world fast-tail case). Eliminates the
    dead-code slow path from the kernel binary.
    """
    return (
        bodies.velocity[b1],
        bodies.velocity[b2],
        bodies.angular_velocity[b1],
        bodies.angular_velocity[b2],
        bodies.inverse_mass[b1],
        bodies.inverse_mass[b2],
        bodies.inverse_inertia_world[b1],
        bodies.inverse_inertia_world[b2],
        wp.int32(-1),
        wp.int32(-1),
    )


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


@wp.func
def _ms_store_body_pair_lean(
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
    """Mass-splitting-free writeback. Direct SoA writes only."""
    bodies.velocity[b1] = v1
    bodies.velocity[b2] = v2
    bodies.angular_velocity[b1] = w1
    bodies.angular_velocity[b2] = w2


# ---------------------------------------------------------------------------
# Shared axial (drive + limit) iterate helper
# ---------------------------------------------------------------------------


@wp.struct
class AxialProjectionUpdate:
    delta: wp.float32
    acc_drive: wp.float32
    acc_limit: wp.float32
    acc_friction: wp.float32


@wp.func
def _axial_project_scalar_rows(
    jv_axial: wp.float32,
    clamp: wp.int32,
    sor_boost: wp.float32,
    drive_active: wp.bool,
    max_force_drive: wp.float32,
    max_lambda_drive: wp.float32,
    bias_drive: wp.float32,
    gamma_drive: wp.float32,
    eff_mass_drive_soft: wp.float32,
    acc_drive: wp.float32,
    limit_active: wp.bool,
    pd_mode_limit: wp.bool,
    pd_mass: wp.float32,
    pd_gamma: wp.float32,
    pd_beta: wp.float32,
    eff_axial: wp.float32,
    bias_box: wp.float32,
    mc_limit: wp.float32,
    ic_limit: wp.float32,
    acc_limit: wp.float32,
    friction_active: wp.bool,
    friction_eff_mass: wp.float32,
    friction_gamma: wp.float32,
    max_lambda_friction: wp.float32,
    acc_friction: wp.float32,
) -> AxialProjectionUpdate:
    """Project drive, limit, and friction as one shared 3-row block."""
    drive_k_inv = wp.float32(0.0)
    drive_rhs = wp.float32(0.0)
    drive_min = acc_drive
    drive_max = acc_drive
    if drive_active:
        drive_k_inv = eff_mass_drive_soft
        drive_rhs = jv_axial - bias_drive + gamma_drive * acc_drive
        drive_min = -BLOCK_LAMBDA_INF
        drive_max = BLOCK_LAMBDA_INF
        if max_force_drive > wp.float32(0.0):
            drive_min = -max_lambda_drive
            drive_max = max_lambda_drive

    limit_k_inv = wp.float32(0.0)
    limit_rhs = wp.float32(0.0)
    limit_mass_coeff = wp.float32(1.0)
    limit_impulse_coeff = wp.float32(0.0)
    limit_min = acc_limit
    limit_max = acc_limit
    if limit_active:
        if clamp == _CLAMP_MAX:
            limit_min = wp.float32(0.0)
            limit_max = BLOCK_LAMBDA_INF
        else:
            limit_min = -BLOCK_LAMBDA_INF
            limit_max = wp.float32(0.0)
        if pd_mode_limit:
            if pd_mass > wp.float32(0.0):
                limit_k_inv = pd_mass
                limit_rhs = jv_axial - pd_beta + pd_gamma * acc_limit
        else:
            if eff_axial > wp.float32(0.0):
                limit_k_inv = eff_axial
                limit_rhs = jv_axial + bias_box
                limit_mass_coeff = mc_limit
                limit_impulse_coeff = ic_limit

    friction_k_inv = wp.float32(0.0)
    friction_rhs = wp.float32(0.0)
    friction_min = acc_friction
    friction_max = acc_friction
    if friction_active:
        friction_k_inv = friction_eff_mass
        friction_rhs = jv_axial + friction_gamma * acc_friction
        friction_min = -max_lambda_friction
        friction_max = max_lambda_friction

    rows = VelocityRows3()
    rows.k_inv = wp.vec3f(drive_k_inv, limit_k_inv, friction_k_inv)
    rows.residual = wp.vec3f(drive_rhs, limit_rhs, friction_rhs)
    rows.lambda_old = wp.vec3f(acc_drive, acc_limit, acc_friction)
    rows.mass_coeff = wp.vec3f(wp.float32(1.0), limit_mass_coeff, wp.float32(1.0))
    rows.impulse_coeff = wp.vec3f(wp.float32(0.0), limit_impulse_coeff, wp.float32(0.0))
    rows.lambda_min = wp.vec3f(drive_min, limit_min, friction_min)
    rows.lambda_max = wp.vec3f(drive_max, limit_max, friction_max)

    projection = block_solve_velocity_rows3_bounded(rows, sor_boost)

    update = AxialProjectionUpdate()
    update.delta = projection.delta[0] + projection.delta[1] + projection.delta[2]
    update.acc_drive = projection.lambda_new[0]
    update.acc_limit = projection.lambda_new[1]
    update.acc_friction = projection.lambda_new[2]
    return update


@wp.func
def _axial_drive_limit_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    jv_axial: wp.float32,
    clamp: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
) -> wp.float32:
    """Scalar drive+limit PGS step for revolute/prismatic mode.

    Both modes apply a single scalar impulse ``axial_lam`` along the
    free DoF axis. Revolute applies it as an angular impulse about
    ``n_hat``; prismatic as a linear impulse along ``n_hat``. The
    actuator cache (drive PD scalars, limit Box2D *or* PD scalars, the
    warm-started accumulated impulses) is identical across both modes,
    so the iterate math collapses to a shared helper that returns the
    net drive + limit + friction impulse and lets the caller spread it
    onto the body velocities in the per-mode way.
    """
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
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
    limit_active = clamp != _CLAMP_NONE
    if limit_active:
        stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
        damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
        acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
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

    friction_eff_mass = read_float(constraints, base_offset + _OFF_FRICTION_EFF_MASS, cid)
    friction_active = friction_eff_mass > wp.float32(0.0)
    friction_gamma = wp.float32(0.0)
    acc_friction = wp.float32(0.0)
    max_lambda_friction = wp.float32(0.0)
    if friction_active:
        friction_coefficient = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
        friction_gamma = read_float(constraints, base_offset + _OFF_FRICTION_GAMMA, cid)
        acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
        max_lambda_friction = friction_coefficient * (wp.float32(1.0) / idt)

    update = _axial_project_scalar_rows(
        jv_axial,
        clamp,
        sor_boost,
        drive_active,
        max_force_drive,
        max_lambda_drive,
        bias_drive,
        gamma_drive,
        eff_mass_drive_soft,
        acc_drive,
        limit_active,
        pd_mode_limit,
        pd_mass,
        pd_gamma,
        pd_beta,
        eff_axial,
        bias_box,
        mc_limit,
        ic_limit,
        acc_limit,
        friction_active,
        friction_eff_mass,
        friction_gamma,
        max_lambda_friction,
        acc_friction,
    )
    if drive_active:
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, update.acc_drive)
    if limit_active:
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, update.acc_limit)
    if friction_active:
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, update.acc_friction)
    return update.delta


# ---------------------------------------------------------------------------
# Compound family iterates
#
# Each "family" packages everything between body-pair load and store
# inline: anchor data read + skew, bias read, the family's
# anchor1/anchor2 row block, optional anchor3 scalar block (block
# Gauss-Seidel), optional axial drive/limit. The static bool flags
# ``has_anchor3`` / ``has_axial_drive`` drive ``wp.static`` gates so
# unused branches DCE.
#
#   pivot  = REVOLUTE / FIXED       (anchor1 full 3-row + anchor2
#                                    tangent 2-row 3+2 Schur,
#                                    angular axial drive)
#   slide  = PRISMATIC / CYLINDRICAL (anchor1 tangent 2-row + anchor2
#                                    tangent 2-row 4-row direct,
#                                    linear axial drive)
#
# Each mode's per-mode iterate is a one-line wrapper into its family.
# ---------------------------------------------------------------------------


@wp.func
def _planar_3row_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    n_hat: wp.vec3f,
    t1: wp.vec3f,
    t2: wp.vec3f,
    bias_packed: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
):
    """PLANAR 3-row block: 1 linear lock along ``n_hat`` (relative
    COM motion) + 2 angular tangent locks along ``t1`` / ``t2``
    (relative angular motion). No anchor lever arms -- the linear
    impulse acts at the COM (pure force) and the angular impulse is a
    pure couple. Reads ``A1_INV`` (the 3x3 inverse of the planar K
    matrix, stored in the same slot REVOLUTE / FIXED use for their
    anchor-1 inverse) plus ``ACC_IMP1`` (linear acc along n_hat) /
    ``ACC_IMP2`` (angular acc along t1, t2)."""
    a3_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc3 = wp.vec3f(
        wp.dot(n_hat, acc_imp1_world),
        wp.dot(t1, acc_imp2_world),
        wp.dot(t2, acc_imp2_world),
    )

    v_rel = v2 - v1
    w_rel = w2 - w1
    jv3 = wp.vec3f(wp.dot(n_hat, v_rel), wp.dot(t1, w_rel), wp.dot(t2, w_rel))
    block3 = VelocityBlock3()
    block3.k_inv = a3_inv
    block3.residual = jv3 + bias_packed
    block3.lambda_old = acc3
    block3.mass_coeff = mass_coeff
    block3.impulse_coeff = impulse_coeff
    update3 = block_solve_velocity_block3(block3, sor_boost)
    lam3 = update3.delta
    lin_imp_world = lam3[0] * n_hat
    ang_imp_world = lam3[1] * t1 + lam3[2] * t2
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1, v2, w1, w2, im1, im2, ii1, ii2, lin_imp_world, ang_imp_world, ang_imp_world
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world + lin_imp_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world + ang_imp_world)
    return v1, v2, w1, w2


@wp.func
def _anchor1_standalone_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    cr1_b1: wp.mat33f,
    cr1_b2: wp.mat33f,
    bias1: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
):
    """3-row Box2D-soft anchor-1 positional lock, standalone (no
    anchor-2 coupling). Used by BALL_SOCKET, UNIVERSAL, and CABLE's
    point-lock block. Reads ``A1_INV`` / ``ACC_IMP1`` from the
    constraint; updates body velocities and writes the accumulator
    back."""
    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    jv1 = -v1 + cr1_b1 @ w1 + v2 - cr1_b2 @ w2
    block1 = VelocityBlock3()
    block1.k_inv = a1_inv
    block1.residual = jv1 + bias1
    block1.lambda_old = acc1
    block1.mass_coeff = mass_coeff
    block1.impulse_coeff = impulse_coeff
    update1 = block_solve_velocity_block3(block1, sor_boost)
    lam1 = update1.delta
    v1, v2, w1, w2 = apply_pair_spatial_impulse(v1, v2, w1, w2, im1, im2, ii1, ii2, lam1, cr1_b1 @ lam1, cr1_b2 @ lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, update1.lambda_new)
    return v1, v2, w1, w2


@wp.func
def _cable_anchor2_pd_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    t1: wp.vec3f,
    t2: wp.vec3f,
    cr2_b1: wp.mat33f,
    cr2_b2: wp.mat33f,
    sor_boost: wp.float32,
):
    """Cable anchor-2 2-row PD-soft tangent block (bend). Uses the
    ``lambda = -M_soft * (Jv + bias + gamma * acc)`` PD formulation
    (vs the Box2D-soft ``lambda = mass_coeff * lam_us - impulse_coeff *
    acc`` used for hard locks). PD bias is UNCONDITIONAL -- it encodes
    the spring force, not a drift correction, and zeroing it on the
    relax pass would cancel the spring."""
    k22_inv_00 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_00, cid)
    k22_inv_01 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_01, cid)
    k22_inv_10 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_10, cid)
    k22_inv_11 = read_float(constraints, base_offset + _OFF_CABLE_K22_INV_11, cid)
    gamma_bend = read_float(constraints, base_offset + _OFF_CABLE_GAMMA_BEND, cid)
    bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)
    jv2_world = -v1 + cr2_b1 @ w1 + v2 - cr2_b2 @ w2
    jv2_t1 = wp.dot(t1, jv2_world)
    jv2_t2 = wp.dot(t2, jv2_world)
    rhs2 = wp.vec2f(jv2_t1 + bias2[0] + gamma_bend * acc2_t1, jv2_t2 + bias2[1] + gamma_bend * acc2_t2)
    block2 = VelocityBlock2()
    block2.k_inv = wp.mat22f(k22_inv_00, k22_inv_01, k22_inv_10, k22_inv_11)
    block2.residual = rhs2
    block2.lambda_old = wp.vec2f(acc2_t1, acc2_t2)
    block2.mass_coeff = wp.float32(1.0)
    block2.impulse_coeff = wp.float32(0.0)
    update2 = block_solve_velocity_block2(block2, sor_boost)
    lam2_t1 = update2.delta[0]
    lam2_t2 = update2.delta[1]
    lam2_world = lam2_t1 * t1 + lam2_t2 * t2
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1, v2, w1, w2, im1, im2, ii1, ii2, lam2_world, cr2_b1 @ lam2_world, cr2_b2 @ lam2_world
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)
    return v1, v2, w1, w2


@wp.func
def _cable_anchor3_pd_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    t2: wp.vec3f,
    cr3_b1: wp.mat33f,
    cr3_b2: wp.mat33f,
    sor_boost: wp.float32,
):
    """Cable anchor-3 1-row PD-soft scalar block (twist). PD bias is
    UNCONDITIONAL (spring, not drift)."""
    m_twist_soft = read_float(constraints, base_offset + _OFF_CABLE_M_TWIST_SOFT, cid)
    gamma_twist = read_float(constraints, base_offset + _OFF_CABLE_GAMMA_TWIST, cid)
    bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
    acc3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc3_t2 = wp.dot(t2, acc3_world)
    jv3_world = -v1 + cr3_b1 @ w1 + v2 - cr3_b2 @ w2
    jv3_t2 = wp.dot(t2, jv3_world)
    block3 = VelocityBlock1()
    block3.k_inv = m_twist_soft
    block3.residual = jv3_t2 + bias3 + gamma_twist * acc3_t2
    block3.lambda_old = acc3_t2
    block3.mass_coeff = wp.float32(1.0)
    block3.impulse_coeff = wp.float32(0.0)
    update3 = block_solve_velocity_block1(block3, sor_boost)
    lam3 = update3.delta
    lam3_world = lam3 * t2
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1, v2, w1, w2, im1, im2, ii1, ii2, lam3_world, cr3_b1 @ lam3_world, cr3_b2 @ lam3_world
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3_world + lam3_world)
    return v1, v2, w1, w2


@wp.func
def _anchor1_anchor2_schur_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    t1: wp.vec3f,
    t2: wp.vec3f,
    cr1_b1: wp.mat33f,
    cr1_b2: wp.mat33f,
    cr2_b1: wp.mat33f,
    cr2_b2: wp.mat33f,
    bias1: wp.vec3f,
    bias2: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
):
    """3+2 Schur block: anchor-1 3-row positional lock + anchor-2 2-row
    tangent lock. Used by REVOLUTE / FIXED / UNIVERSAL / BALL_SOCKET
    (those last two zero the anchor-2 Schur slots at populate time so
    Block A degenerates to anchor-1 only). Reads ``A1_INV`` / ``UT_AI``
    / ``S_INV`` / ``ACC_IMP1`` / ``ACC_IMP2`` from the constraint;
    returns updated body velocities and writes the new accumulators
    back."""
    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    ut_ai = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)
    s_inv_22 = _read_s_inv_22(constraints, base_offset, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_tan = wp.vec2f(wp.dot(t1, acc2_world), wp.dot(t2, acc2_world))

    jv1 = -v1 + cr1_b1 @ w1 + v2 - cr1_b2 @ w2
    jv2_world = -v1 + cr2_b1 @ w1 + v2 - cr2_b2 @ w2
    jv2 = wp.vec2f(wp.dot(t1, jv2_world), wp.dot(t2, jv2_world))

    rhs1 = jv1 + bias1
    rhs2 = jv2 + wp.vec2f(bias2[0], bias2[1])
    ut_ai_rhs1_3 = ut_ai @ rhs1
    ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

    lam2_us = block_solve_inverse_2(s_inv_22, rhs2 - ut_ai_rhs1)
    update2 = block_project_accumulated_2(lam2_us, acc2_tan, mass_coeff, impulse_coeff, sor_boost)
    lam2 = update2.delta
    lam2_world = lam2[0] * t1 + lam2[1] * t2
    lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

    u_lam2_us = (im1 + im2) * lam2_us_world
    u_lam2_us = u_lam2_us + cr1_b1 @ (ii1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
    u_lam2_us = u_lam2_us + cr1_b2 @ (ii2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

    lam1_us = block_solve_inverse_3(a1_inv, rhs1 + u_lam2_us)
    update1 = block_project_accumulated_3(lam1_us, acc1, mass_coeff, impulse_coeff, sor_boost)
    lam1 = update1.delta
    total_lin = lam1 + lam2_world
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        total_lin,
        cr1_b1 @ lam1 + cr2_b1 @ lam2_world,
        cr1_b2 @ lam1 + cr2_b2 @ lam2_world,
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, update1.lambda_new)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    return v1, v2, w1, w2


@wp.func
def _anchor1_anchor2_tangent_4row_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    t1: wp.vec3f,
    t2: wp.vec3f,
    cr1_b1: wp.mat33f,
    cr1_b2: wp.mat33f,
    cr2_b1: wp.mat33f,
    cr2_b2: wp.mat33f,
    bias1: wp.vec3f,
    bias2: wp.vec3f,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
):
    """4-row tangent block: anchor-1 2-row tangent + anchor-2 2-row
    tangent. Used by PRISMATIC and CYLINDRICAL."""
    a4_inv = read_mat44(constraints, base_offset + _OFF_A4_INV, cid)
    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc1_tan = wp.vec2f(wp.dot(t1, acc_imp1_world), wp.dot(t2, acc_imp1_world))
    acc2_tan = wp.vec2f(wp.dot(t1, acc_imp2_world), wp.dot(t2, acc_imp2_world))
    acc4 = wp.vec4f(acc1_tan[0], acc1_tan[1], acc2_tan[0], acc2_tan[1])

    jv1_world = v2 - cr1_b2 @ w2 - v1 + cr1_b1 @ w1
    jv2_world = v2 - cr2_b2 @ w2 - v1 + cr2_b1 @ w1
    jv4 = wp.vec4f(
        wp.dot(t1, jv1_world),
        wp.dot(t2, jv1_world),
        wp.dot(t1, jv2_world),
        wp.dot(t2, jv2_world),
    )
    bias4 = wp.vec4f(bias1[0], bias1[1], bias2[0], bias2[1])
    block4 = VelocityBlock4()
    block4.k_inv = a4_inv
    block4.residual = jv4 + bias4
    block4.lambda_old = acc4
    block4.mass_coeff = mass_coeff
    block4.impulse_coeff = impulse_coeff
    update4 = block_solve_velocity_block4(block4, sor_boost)
    lam4 = update4.delta
    lam1_world = lam4[0] * t1 + lam4[1] * t2
    lam2_world = lam4[2] * t1 + lam4[3] * t2

    total_linear = lam1_world + lam2_world
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        total_linear,
        cr1_b1 @ lam1_world + cr2_b1 @ lam2_world,
        cr1_b2 @ lam1_world + cr2_b2 @ lam2_world,
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world + lam1_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world + lam2_world)

    return v1, v2, w1, w2


@wp.func
def _anchor3_scalar_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    t2: wp.vec3f,
    cr3_b1: wp.mat33f,
    cr3_b2: wp.mat33f,
    bias3: wp.float32,
    mass_coeff: wp.float32,
    impulse_coeff: wp.float32,
    sor_boost: wp.float32,
):
    """1-row anchor-3 scalar lock along ``t2`` (block Gauss-Seidel).
    Used by FIXED (locks rotation about anchor-3 axis) and PRISMATIC
    (locks rotation about slide axis)."""
    s3_inv = read_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid)
    acc3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc3_scalar = wp.dot(t2, acc3_world)
    jv3_world = -v1 + cr3_b1 @ w1 + v2 - cr3_b2 @ w2
    jv3 = wp.dot(t2, jv3_world)
    block3 = VelocityBlock1()
    block3.k_inv = s3_inv
    block3.residual = jv3 + bias3
    block3.lambda_old = acc3_scalar
    block3.mass_coeff = mass_coeff
    block3.impulse_coeff = impulse_coeff
    update3 = block_solve_velocity_block1(block3, sor_boost)
    lam3 = update3.delta
    lam3_world = lam3 * t2
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1, v2, w1, w2, im1, im2, ii1, ii2, lam3_world, cr3_b1 @ lam3_world, cr3_b2 @ lam3_world
    )
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3_world + lam3_world)
    return v1, v2, w1, w2


@wp.func
def _angular_axial_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    w1: wp.vec3f,
    w2: wp.vec3f,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    n_hat: wp.vec3f,
    clamp: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    """1-row angular axial drive + limit + friction about ``n_hat``.
    Used by REVOLUTE and UNIVERSAL. Pure couple -- updates angular
    velocities only."""
    jv_axial = wp.dot(n_hat, w1 - w2)
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, sor_boost)
    w1, w2 = apply_pair_angular_impulse(w1, w2, ii1, ii2, -n_hat * axial_lam, -n_hat * axial_lam)
    return w1, w2


@wp.func
def _linear_axial_block(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    im1: wp.float32,
    im2: wp.float32,
    ii1: wp.mat33f,
    ii2: wp.mat33f,
    r1_b1: wp.vec3f,
    r1_b2: wp.vec3f,
    n_hat: wp.vec3f,
    clamp: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
):
    """1-row linear axial drive + limit + friction along ``n_hat``,
    applied at anchor-1. Used by PRISMATIC and CYLINDRICAL. Linear
    impulse with anchor-1 lever arm -- updates linear and angular
    velocities."""
    v1_anchor = v1 + wp.cross(w1, r1_b1)
    v2_anchor = v2 + wp.cross(w2, r1_b2)
    jv_axial = wp.dot(n_hat, v1_anchor - v2_anchor)
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, sor_boost)
    axial_imp = n_hat * axial_lam
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        -axial_imp,
        -wp.cross(r1_b1, axial_imp),
        -wp.cross(r1_b2, axial_imp),
    )
    return v1, v2, w1, w2


# ---------------------------------------------------------------------------
# Shared anchor-1 positional prepare helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared pivot-family K-factor prepare
#
# REVOLUTE and FIXED both build the same 3+2 Schur factorisation of the
# anchor-1 full 3-row + anchor-2 2-row-tangent block. Extract the
# construction here and let both modes call it, then carry on with their
# mode-specific extras (axial drive vs anchor-3 scalar).
# ---------------------------------------------------------------------------


@wp.func
def _slide_anchor1_anchor2_K4_factor_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    cr1_b1: wp.mat33f,
    cr1_b2: wp.mat33f,
    cr2_b1: wp.mat33f,
    cr2_b2: wp.mat33f,
    t1: wp.vec3f,
    t2: wp.vec3f,
):
    """Build the 4-row tangent K-matrix factorisation shared by
    PRISMATIC and CYLINDRICAL. Computes the 3x3 anchor-anchor coupling
    blocks (``b11``, ``b22``, ``b12``), projects onto the tangent
    basis to assemble the 4x4 ``K4``, inverts it, and writes
    ``A4_INV`` back to the column. Returns ``(b11, b22, b12)`` so the
    caller can reuse them for axial-row effective mass or anchor-3
    coupling without recomputing the per-pair sums."""
    eye3 = wp.identity(3, dtype=wp.float32)
    m_diag = (inv_mass1 + inv_mass2) * eye3

    b11 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
    b22 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))
    b12 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    b11_t1 = b11 @ t1
    b11_t2 = b11 @ t2
    b22_t1 = b22 @ t1
    b22_t2 = b22 @ t2
    b12_t1 = b12 @ t1
    b12_t2 = b12 @ t2

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
        k4_00,
        k4_01,
        k4_02,
        k4_03,
        k4_01,
        k4_11,
        k4_12,
        k4_13,
        k4_02,
        k4_12,
        k4_22,
        k4_23,
        k4_03,
        k4_13,
        k4_23,
        k4_33,
    )
    a4_inv = wp.inverse(k4)
    write_mat44(constraints, base_offset + _OFF_A4_INV, cid, a4_inv)

    return b11, b22, b12


@wp.func
def _pivot_anchor1_anchor2_K_factor_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1: wp.mat33f,
    inv_inertia2: wp.mat33f,
    cr1_b1: wp.mat33f,
    cr1_b2: wp.mat33f,
    cr2_b1: wp.mat33f,
    cr2_b2: wp.mat33f,
    t1: wp.vec3f,
    t2: wp.vec3f,
):
    """Build the 3+2 Schur factorisation for the anchor-1 + anchor-2
    block shared by REVOLUTE and FIXED. Computes ``A1`` (anchor-1 3x3
    effective mass), ``A2`` (anchor-2 3x3), ``B`` (anchor-1/2
    cross-coupling), projects to the tangent basis, inverts the 2x2
    Schur complement, and writes ``A1_INV`` / ``UT_AI`` / ``S_INV``
    back to the column."""
    eye3 = wp.identity(3, dtype=wp.float32)

    a1 = inv_mass1 * eye3
    a1 = a1 + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1))
    a1 = a1 + inv_mass2 * eye3
    a1 = a1 + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))

    a2 = inv_mass1 * eye3
    a2 = a2 + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    a2 = a2 + inv_mass2 * eye3
    a2 = a2 + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    b_mat = (inv_mass1 + inv_mass2) * eye3
    b_mat = b_mat + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    b_mat = b_mat + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    t_mat = wp.mat33f(t1[0], t2[0], 0.0, t1[1], t2[1], 0.0, t1[2], t2[2], 0.0)
    tt = wp.transpose(t_mat)
    u_mat = b_mat @ t_mat
    d_mat = tt @ (a2 @ t_mat)

    a1_inv = wp.inverse(a1)
    ut_ai = wp.transpose(u_mat) @ a1_inv
    s_mat = d_mat - ut_ai @ u_mat
    s22 = wp.mat22f(s_mat[0, 0], s_mat[0, 1], s_mat[1, 0], s_mat[1, 1])
    s22_inv = wp.inverse(s22)

    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, ut_ai)
    _write_s_inv_22(constraints, base_offset, cid, s22_inv)


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

    # ---- Friction (Coulomb on axial row) -----------------------------
    # Soft saturated damper toward zero relative velocity, ``μ * dt``-
    # capped accumulated impulse. The regularization ``gamma`` is sized
    # so that the slip velocity at the saturation impulse equals the
    # globally-configured target (:data:`PHOENX_FRICTION_SLIP_VELOCITY`):
    # at convergence ``jv = -gamma * acc``, so at ``|acc| = μ * dt`` we
    # have ``|jv| = gamma * μ * dt``. Setting ``gamma = v_slip / (μ * dt)``
    # decouples the slip threshold from the joint impedance -- contrast
    # to scaling ``gamma`` against ``eff_inv`` directly, which gives
    # near-zero gamma on heavy joints and arbitrarily large slip on
    # light ones. Disabled when ``friction_coefficient <= 0``.
    friction_coefficient = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
    if friction_coefficient > 0.0 and eff_inv > 0.0:
        friction_gamma = PHOENX_FRICTION_SLIP_VELOCITY / (friction_coefficient * dt)
        friction_eff_mass = wp.float32(1.0) / (eff_inv + friction_gamma)
        write_float(constraints, base_offset + _OFF_FRICTION_GAMMA, cid, friction_gamma)
        write_float(constraints, base_offset + _OFF_FRICTION_EFF_MASS, cid, friction_eff_mass)
    else:
        write_float(constraints, base_offset + _OFF_FRICTION_EFF_MASS, cid, wp.float32(0.0))

    # Warm-start: sum of drive + limit + friction accumulated impulses,
    # with ``acc_limit`` forcibly zeroed when the limit is inactive.
    # The friction warm-start is naturally axially-aligned so it folds
    # into the same scalar impulse as drive / limit.
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
    if clamp == _CLAMP_NONE:
        acc_limit = 0.0
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    if friction_coefficient <= 0.0:
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


# ---------------------------------------------------------------------------
# Ball-socket mode math
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Revolute (hinge) mode math
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prismatic (slider) mode math
# ---------------------------------------------------------------------------
#
# Rank-5 pure-points, 2+2+1 rows: anchor-1 tangent drift onto (t1,t2),
# anchor-2 tangent drift onto (t1,t2), and anchor-3 drift onto t2 to
# kill the last rotational DoF (rotation about n_hat).
#
# The 5x5 effective-mass matrix is block-structured as
#     K = [ K4     c ]    K4 (4x4) = anchor-tangent pairs
#         [ c^T    d ]    c  (4)   = a3 cross-coupling, d (scalar)
#
# Schur-eliminate the scalar row first:
#     s_inv = 1 / (d - c^T K4^{-1} c)
#     lam3  = -s_inv * (rhs3 - c^T K4^{-1} rhs4)
#     lam4  = -K4^{-1} * (rhs4 + c * lam3)
#
# One ``wp.inverse(mat44f)`` per prepare; zero per-iter inverses.


# ---------------------------------------------------------------------------
# Cylindrical mode math
# ---------------------------------------------------------------------------
#
# CYLINDRICAL is PRISMATIC minus the anchor-3 scalar row. The
# anchor-1 + anchor-2 tangent block (2 lin perpendicular to ``n_hat`` +
# 2 ang perpendicular to ``n_hat``, 4 rows total) is identical to
# PRISMATIC. The 1-row anchor-3 scalar lock that PRISMATIC uses to gate
# rotation about ``n_hat`` is dropped; both linear translation along
# ``n_hat`` and rotation about ``n_hat`` are free DoFs.
#
# Schur math collapses: with no anchor-3 row there's no need for the
# Schur complement. We invert the 4x4 K4 directly and store it in
# ``mode_cache``'s ``A4_INV`` slot. The PRISMATIC-aliased ``c`` and
# ``s_scalar_inv`` slots stay zero -- the iterate doesn't read them.
#
# Phase 2 MVP: no drive / limit / friction on the two free DoFs (the
# existing axial row stays in the schema but is set to OFF by the
# adapter, costing one short-circuit read in the iterate). A follow-up
# can add paired axial drive rows for translation and rotation along
# ``n_hat`` independently.


# ---------------------------------------------------------------------------
# Planar mode math
# ---------------------------------------------------------------------------
#
# 3 constraint rows operating directly on relative velocity (no anchor
# lever arms -- PLANAR constrains the bodies' rigid motion, not the
# position of a designated joint anchor):
#   1. n_hat · (v_com_2 - v_com_1) = 0  -- locks relative translation
#      along the plane normal (kills out-of-plane motion).
#   2. t1 · (ω_2 - ω_1) = 0  -- locks relative rotation about t1.
#   3. t2 · (ω_2 - ω_1) = 0  -- locks relative rotation about t2.
#
# 3 free DoFs: 2 in-plane translations + 1 rotation about ``n_hat``.
#
# The K3 matrix is block-diagonal in the (linear, angular_t1, angular_t2)
# basis because pure-couple angular impulses produce no linear velocity
# at the COM and pure-force linear impulses produce no angular velocity
# at the COM. The 2x2 angular sub-block can still cross-couple if
# ``inv_inertia`` is non-diagonal in the (t1, t2) basis. We invert the
# full 3x3 with ``wp.inverse(mat33)`` for simplicity.
#
# Phase 2 MVP is kinematic-only -- the schema's axial drive row is kept
# OFF by the adapter; the iterate doesn't apply any axial impulse.
# Drives on the in-plane translations / about-normal rotation would
# need paired axial rows (Phase 2 follow-up).


@wp.func
def _planar_prepare_at(
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
    """Planar prepare: linear lock along ``n_hat`` (relative COM
    velocity) + 2-row angular lock perpendicular to ``n_hat`` (relative
    angular velocity). 3x3 K matrix inverted directly.

    Drift correction is built into the bias: positional Z-drift between
    the two bodies' joint anchors and orientation drift between the
    bodies' n_hat axes (Box2D soft-constraint biases)."""
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

    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
    la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)

    # Snapshot anchor1 lever arms (used for z-drift bias computation).
    # PLANAR's linear constraint operates on relative COM velocity, not
    # relative anchor velocity, so the impulse application is at COM
    # (no skew matrices needed in the iterate's velocity update).
    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)

    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2

    # Plane normal: in body 1's frame, stored as la2_b1 - la1_b1 at init
    # (the init kernel sets anchor2 = anchor1 + normal_axis). Recover it
    # in world frame by rotating la_diff into body 1's frame. Using
    # body 1 (rather than mean of body 1 and body 2) keeps n_hat
    # well-defined when the two bodies' relative orientation drifts.
    n_hat_body1 = la2_b1 - la1_b1
    n_hat_len2 = wp.dot(n_hat_body1, n_hat_body1)
    if n_hat_len2 > 1.0e-20:
        n_hat = wp.quat_rotate(orientation1, n_hat_body1 / wp.sqrt(n_hat_len2))
    else:
        n_hat = wp.vec3f(0.0, 0.0, 1.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # In-plane tangent basis perpendicular to n_hat. Used for the two
    # angular constraint rows ``t1 · ω_rel = 0`` and ``t2 · ω_rel = 0``.
    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    # K3 entries:
    #   row 0 (linear, along n_hat):
    #     n_hat · ((inv_mass1 + inv_mass2) * I) · n_hat = inv_mass1 + inv_mass2
    #     (no angular contribution -- pure-couple impulse produces no
    #     anchor-velocity change at COM; pure-force impulse produces no
    #     angular velocity change either at COM).
    #   rows 1, 2 (angular, along t1, t2):
    #     t · (inv_inertia1 + inv_inertia2) · t.
    #     (Inertia is the only contribution since the impulse is a pure
    #     couple -- no linear coupling.)
    #   Cross-coupling 0-1 / 0-2 is ZERO (linear ⟂ angular at COM).
    #   Cross-coupling 1-2 = t1 · (inv_inertia1 + inv_inertia2) · t2 (may
    #   be non-zero if the inertia tensor is non-diagonal in the t1/t2
    #   basis).
    inv_I_sum = inv_inertia1 + inv_inertia2
    k00 = inv_mass1 + inv_mass2
    inv_I_t1 = inv_I_sum @ t1
    inv_I_t2 = inv_I_sum @ t2
    k11 = wp.dot(t1, inv_I_t1)
    k22 = wp.dot(t2, inv_I_t2)
    k12 = wp.dot(t1, inv_I_t2)
    k3 = wp.mat33f(
        k00,
        0.0,
        0.0,
        0.0,
        k11,
        k12,
        0.0,
        k12,
        k22,
    )
    a3_inv = wp.inverse(k3)
    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a3_inv)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Bias: positional drift between the joint anchors projected onto
    # n_hat (Z-drift), plus orientation drift between body 1's and body
    # 2's n_hat-axes (the "pitch/roll" angle). The orientation drift is
    # tracked via the relative rotation's small-angle log projected onto
    # t1, t2 (the angular constraint axes). At init the relative
    # orientation is identity so this term starts at zero.
    drift1 = p1_b2 - p1_b1
    bias_lin_n = wp.dot(n_hat, drift1) * bias_rate
    # n_hat in body 2's frame (after rotation drift).
    n_hat_body2 = la2_b2 - la1_b2
    n_hat_body2_len2 = wp.dot(n_hat_body2, n_hat_body2)
    if n_hat_body2_len2 > 1.0e-20:
        n_hat_2_world = wp.quat_rotate(orientation2, n_hat_body2 / wp.sqrt(n_hat_body2_len2))
    else:
        n_hat_2_world = n_hat
    # Cross product gives the perpendicular rotation that would align
    # body 2's n_hat to body 1's n_hat. Projected onto t1, t2 it is the
    # small-angle "pitch / roll" drift.
    ang_drift = wp.cross(n_hat_2_world, n_hat)
    bias_ang_t1 = wp.dot(t1, ang_drift) * bias_rate
    bias_ang_t2 = wp.dot(t2, ang_drift) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, wp.vec3f(bias_lin_n, bias_ang_t1, bias_ang_t2))

    # Warm-start: ``acc_imp1`` stores the linear impulse along n_hat,
    # ``acc_imp2`` stores the angular impulse along (t1, t2) basis.
    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_lin_n = wp.dot(n_hat, acc_imp1_world)
    acc_ang_t1 = wp.dot(t1, acc_imp2_world)
    acc_ang_t2 = wp.dot(t2, acc_imp2_world)
    acc_imp1_world = acc_lin_n * n_hat
    acc_imp2_world = acc_ang_t1 * t1 + acc_ang_t2 * t2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world)

    # Apply warm-start. Linear impulse acts at the COM (no torque);
    # angular impulse is a pure couple (no force).
    velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        acc_imp1_world,
        acc_imp2_world,
        acc_imp2_world,
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


# ---------------------------------------------------------------------------
# Cable (soft fixed) mode
# ---------------------------------------------------------------------------
#
# Three block-GS sub-blocks, each with its own pre-computed soft
# inverse; the outer PGS loop closes the cross-coupling.
#   1. Anchor-1: rigid 3-row Box2D-soft point lock.
#   2. Anchor-2 tangent: 2-row PD (k_bend, d_bend).
#   3. Anchor-3 scalar: 1-row PD (k_twist, d_twist).
#
# User gains are in rotational SI units; rescaled by 1/rest_length^2
# to positional springs at the lever-armed anchors. Avoids an
# angular Jacobian / log-map and matches REVOLUTE/PRISMATIC
# convergence. k_bend -> inf yields REVOLUTE; k_twist -> inf on top
# yields FIXED.


@wp.func
def _cable_prepare_at(
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
    """Cable-mode prepare pass.

    Anchor-1 ball-socket 3-row block (Box2D soft via hertz / damping
    ratio); anchor-2 tangent 2-row block with PD softness using user
    gains ``k_bend, d_bend`` rescaled by ``1 / rest_length^2``;
    anchor-3 scalar row with PD softness using ``k_twist, d_twist``
    likewise rescaled. All three blocks are decoupled within one PGS
    sweep (block Gauss-Seidel) -- caches stand-alone effective-mass
    inverses, not Schur complements.
    """
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

    # Slide axis: world direction from body 2's anchor 1 to anchor 2.
    hinge_vec = p2_b2 - p1_b2
    hinge_len2 = wp.dot(hinge_vec, hinge_vec)
    if hinge_len2 > 1.0e-20:
        n_hat = hinge_vec / wp.sqrt(hinge_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # Anchor-3-aligned tangent basis (same convention as PRISMATIC /
    # FIXED so the anchor-3 scalar row is a unit-gain gate for
    # rotation about ``n_hat``).
    t1, t2 = _tangent_basis_from_anchor3(n_hat, r1_b1, r3_b1)
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

    # ---- Anchor-1 3-row Box2D-soft block (hertz, damping_ratio) -----
    a1 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
    a1_inv = wp.inverse(a1)
    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)
    bias1 = (p1_b2 - p1_b1) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)

    # ---- Anchor-2 tangent 2-row PD-soft block (bend, k_bend, d_bend) -
    # Stand-alone 3x3 cross-mass at anchor 2, projected onto (t1, t2).
    b22 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))
    b22_t1 = b22 @ t1
    b22_t2 = b22 @ t2
    k22_00 = wp.dot(t1, b22_t1)
    k22_01 = wp.dot(t1, b22_t2)
    k22_10 = wp.dot(t2, b22_t1)
    k22_11 = wp.dot(t2, b22_t2)

    rest_length = read_float(constraints, base_offset + _OFF_REST_LENGTH, cid)
    rest_len2 = rest_length * rest_length
    if rest_len2 > 1.0e-12:
        inv_rest2 = wp.float32(1.0) / rest_len2
    else:
        inv_rest2 = wp.float32(0.0)

    k_bend_user = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    d_bend_user = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    k_pos_bend = k_bend_user * inv_rest2
    d_pos_bend = d_bend_user * inv_rest2

    # Nyquist-clamp the positional bend stiffness using the average
    # diagonal of K22 as the representative scalar effective mass.
    # Cap is ``N / (M_inv * dt^2)`` where ``N = PHOENX_BOOST_CABLE_BEND``
    # (default ``10``), clamped to ``[1, _PD_NYQUIST_HEADROOM_MAX]``
    # so the global cap can prevent excess headroom requests.
    bend_boost = wp.clamp(PHOENX_BOOST_CABLE_BEND, wp.float32(1.0), _PD_NYQUIST_HEADROOM_MAX)
    eff_inv_bend = wp.float32(0.5) * (k22_00 + k22_11)
    bias_factor_bend = wp.float32(0.0)
    gamma_bend = wp.float32(0.0)
    if (k_pos_bend > wp.float32(0.0)) or (d_pos_bend > wp.float32(0.0)):
        if eff_inv_bend > wp.float32(0.0):
            k_max_bend = bend_boost / (eff_inv_bend * dt * dt)
            k_clamped_bend = wp.min(k_pos_bend, k_max_bend)
        else:
            k_clamped_bend = k_pos_bend
        denom_bend = d_pos_bend + dt * k_clamped_bend
        if denom_bend > wp.float32(0.0):
            softness_bend = wp.float32(1.0) / denom_bend
            bias_factor_bend = dt * k_clamped_bend * softness_bend
            gamma_bend = softness_bend * idt

    # Soften K22 by adding gamma * I_2x2, then invert. K22 is symmetric
    # (b22 is symmetric), so the off-diagonal is shared.
    k22s_00 = k22_00 + gamma_bend
    k22s_11 = k22_11 + gamma_bend
    k22s_01 = k22_01
    k22s_10 = k22_10
    det_b = k22s_00 * k22s_11 - k22s_01 * k22s_10
    if wp.abs(det_b) > wp.float32(1.0e-20):
        inv_det_b = wp.float32(1.0) / det_b
    else:
        inv_det_b = wp.float32(0.0)
    write_float(constraints, base_offset + _OFF_CABLE_K22_INV_00, cid, k22s_11 * inv_det_b)
    write_float(constraints, base_offset + _OFF_CABLE_K22_INV_01, cid, -k22s_01 * inv_det_b)
    write_float(constraints, base_offset + _OFF_CABLE_K22_INV_10, cid, -k22s_10 * inv_det_b)
    write_float(constraints, base_offset + _OFF_CABLE_K22_INV_11, cid, k22s_00 * inv_det_b)
    write_float(constraints, base_offset + _OFF_CABLE_GAMMA_BEND, cid, gamma_bend)

    # PD bias: positional drift at anchor 2 along (t1, t2), scaled by
    # ``bias_factor / dt``. Sign matches the Box2D convention used by
    # REVOLUTE's anchor-2 tangent rows (drift > 0 -> bias > 0 ->
    # ``lambda = -K_inv * (Jv + bias)`` produces lambda < 0, closing
    # the drift). Stored in the existing BIAS2 vec3 slot.
    drift2 = p2_b2 - p2_b1
    bias_bend_t1 = wp.dot(t1, drift2) * bias_factor_bend * idt
    bias_bend_t2 = wp.dot(t2, drift2) * bias_factor_bend * idt
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, wp.vec3f(bias_bend_t1, bias_bend_t2, 0.0))

    # ---- Anchor-3 scalar 1-row PD-soft block (twist, k_twist, d_twist)
    b33 = m_diag + cr3_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr3_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))
    eff_inv_twist = wp.dot(t2, b33 @ t2)

    k_twist_user = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    d_twist_user = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
    k_pos_twist = k_twist_user * inv_rest2
    d_pos_twist = d_twist_user * inv_rest2

    twist_boost = wp.clamp(PHOENX_BOOST_CABLE_TWIST, wp.float32(1.0), _PD_NYQUIST_HEADROOM_MAX)
    bias_factor_twist = wp.float32(0.0)
    gamma_twist = wp.float32(0.0)
    m_twist_soft = wp.float32(0.0)
    if (k_pos_twist > wp.float32(0.0)) or (d_pos_twist > wp.float32(0.0)):
        if eff_inv_twist > wp.float32(0.0):
            # Same headroom rule as the bend clamp.
            k_max_twist = twist_boost / (eff_inv_twist * dt * dt)
            k_clamped_twist = wp.min(k_pos_twist, k_max_twist)
        else:
            k_clamped_twist = k_pos_twist
        denom_twist = d_pos_twist + dt * k_clamped_twist
        if denom_twist > wp.float32(0.0):
            softness_twist = wp.float32(1.0) / denom_twist
            bias_factor_twist = dt * k_clamped_twist * softness_twist
            gamma_twist = softness_twist * idt
            m_twist_soft = wp.float32(1.0) / (eff_inv_twist + gamma_twist)
    write_float(constraints, base_offset + _OFF_CABLE_M_TWIST_SOFT, cid, m_twist_soft)
    write_float(constraints, base_offset + _OFF_CABLE_GAMMA_TWIST, cid, gamma_twist)

    drift3 = p3_b2 - p3_b1
    bias_twist = wp.dot(t2, drift3) * bias_factor_twist * idt
    write_float(constraints, base_offset + _OFF_BIAS3, cid, bias_twist)

    # ---- Positional warm-start --------------------------------------
    # Re-project anchor-2 / anchor-3 accumulated impulses onto the
    # current tangent basis (PRISMATIC / FIXED do the same trick).
    acc_imp1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc2_t1 = wp.dot(t1, acc_imp2_world)
    acc2_t2 = wp.dot(t2, acc_imp2_world)
    acc_imp2_world = acc2_t1 * t1 + acc2_t2 * t2
    acc3_t2 = wp.dot(t2, acc_imp3_world)
    acc_imp3_world = acc3_t2 * t2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc_imp3_world)

    total_linear = acc_imp1 + acc_imp2_world + acc_imp3_world
    velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
        velocity1,
        velocity2,
        angular_velocity1,
        angular_velocity2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        total_linear,
        cr1_b1 @ acc_imp1 + cr2_b1 @ acc_imp2_world + cr3_b1 @ acc_imp3_world,
        cr1_b2 @ acc_imp1 + cr2_b2 @ acc_imp2_world + cr3_b2 @ acc_imp3_world,
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

    # Zero unused axial drive / limit state so wrench helpers and any
    # cross-mode reads see a clean column.
    write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, 0.0)
    write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, 0.0)


# ---------------------------------------------------------------------------
# Fixed (weld) mode
# ---------------------------------------------------------------------------
#
# FIXED = REVOLUTE anchor-1 3-row + REVOLUTE anchor-2 tangent 2-row +
# PRISMATIC anchor-3 scalar 1-row. Anchor-3 is derived at init as
# ``anchor1 + rest_length * t_ref``, so no new state is needed.
# Tangent basis follows the PRISMATIC convention (``t1`` aligned with
# anchor-3 perpendicular to ``n_hat``), making the anchor-3 scalar row
# a unit-gain gate for rotation about ``n_hat``.
#
# Prepare caches ``a1_inv`` + ``ut_ai`` + ``s_inv_packed`` (from
# REVOLUTE) for the 3+2 block and ``s_scalar_inv`` (standalone, no
# anchor-1/2 coupling) for the anchor-3 row. Iterate runs the two
# blocks in Gauss-Seidel; outer PGS iterations couple them.


# ---------------------------------------------------------------------------
# Universal (Hooke) mode math
# ---------------------------------------------------------------------------
#
# Composition: BALL_SOCKET (anchor-1 3-row positional lock) + 1-row
# angular lock about the user-specified axis stored in ``axis_local1``.
# The angular lock reuses the existing axial drive/limit machinery in
# *rigid-limit* mode -- prepare seeds ``min_value = max_value = 0`` and
# ``hertz_limit = DEFAULT_HERTZ_LIMIT`` (rigid Box2D) at construction
# time, so the limit row is always clamped on any non-zero twist drift.
#
# The 3-row positional Schur and the 1-row angular Schur are decoupled
# (the angular impulse along ``n_hat`` produces no positional drift at
# the joint anchor, and the positional impulse produces no torque
# about ``n_hat``). PGS solves them block-Gauss-Seidel without needing
# cross terms.


# ---------------------------------------------------------------------------
# Mode-dispatching entry points
# ---------------------------------------------------------------------------


@wp.func
def _box2d_pivot_slide_prepare_at(
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
    joint_mode: wp.int32,
):
    """Unified prepare for the six Box2D-soft modes: BALL_SOCKET,
    REVOLUTE, FIXED, UNIVERSAL, PRISMATIC, CYLINDRICAL. CABLE has
    PD-soft anchor-2/3 blocks (different math) and PLANAR uses
    COM-frame Jacobians; both keep their own prepare functions.

    Mode-specific behaviour is gated by ``joint_mode`` flags computed
    once at the top. Each shared K-factor / axial-drive helper is
    invoked from exactly this site."""
    has_anchor1_only = joint_mode == JOINT_MODE_BALL_SOCKET or joint_mode == JOINT_MODE_UNIVERSAL
    has_schur_3plus2 = joint_mode == JOINT_MODE_REVOLUTE or joint_mode == JOINT_MODE_FIXED
    has_tangent_4row = joint_mode == JOINT_MODE_PRISMATIC or joint_mode == JOINT_MODE_CYLINDRICAL
    has_anchor3 = joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_PRISMATIC
    has_angular_axial = joint_mode == JOINT_MODE_REVOLUTE or joint_mode == JOINT_MODE_UNIVERSAL
    has_linear_axial = joint_mode == JOINT_MODE_PRISMATIC or joint_mode == JOINT_MODE_CYLINDRICAL
    use_tangent_from_anchor3 = joint_mode == JOINT_MODE_PRISMATIC or joint_mode == JOINT_MODE_FIXED
    use_axis_from_anchors = (
        joint_mode == JOINT_MODE_REVOLUTE
        or joint_mode == JOINT_MODE_FIXED
        or joint_mode == JOINT_MODE_PRISMATIC
        or joint_mode == JOINT_MODE_CYLINDRICAL
    )
    use_axis_from_local1 = joint_mode == JOINT_MODE_UNIVERSAL
    needs_anchor2 = has_schur_3plus2 or has_tangent_4row
    needs_axial_drive_prep = has_angular_axial or has_linear_axial

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

    # ---- Anchor reads + world-frame rotation ------------------------
    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)
    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2

    r2_b1 = wp.vec3f(0.0, 0.0, 0.0)
    r2_b2 = wp.vec3f(0.0, 0.0, 0.0)
    p2_b1 = wp.vec3f(0.0, 0.0, 0.0)
    p2_b2 = wp.vec3f(0.0, 0.0, 0.0)
    if needs_anchor2:
        la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
        r2_b1 = wp.quat_rotate(orientation1, la2_b1)
        r2_b2 = wp.quat_rotate(orientation2, la2_b2)
        write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
        write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)
        p2_b1 = position1 + r2_b1
        p2_b2 = position2 + r2_b2

    r3_b1 = wp.vec3f(0.0, 0.0, 0.0)
    r3_b2 = wp.vec3f(0.0, 0.0, 0.0)
    p3_b1 = wp.vec3f(0.0, 0.0, 0.0)
    p3_b2 = wp.vec3f(0.0, 0.0, 0.0)
    if has_anchor3:
        la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
        la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
        r3_b1 = wp.quat_rotate(orientation1, la3_b1)
        r3_b2 = wp.quat_rotate(orientation2, la3_b2)
        write_vec3(constraints, base_offset + _OFF_R3_B1, cid, r3_b1)
        write_vec3(constraints, base_offset + _OFF_R3_B2, cid, r3_b2)
        p3_b1 = position1 + r3_b1
        p3_b2 = position2 + r3_b2

    # ---- n_hat (joint axis in world) --------------------------------
    n_hat = wp.vec3f(1.0, 0.0, 0.0)
    if use_axis_from_anchors:
        hinge_vec = p2_b2 - p1_b2
        hinge_len2 = wp.dot(hinge_vec, hinge_vec)
        if hinge_len2 > 1.0e-20:
            n_hat = hinge_vec / wp.sqrt(hinge_len2)
        write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)
    elif use_axis_from_local1:
        axis_local1 = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
        n_hat = wp.quat_rotate(orientation1, axis_local1)
        write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # ---- Tangent basis ---------------------------------------------
    # BALL_SOCKET keeps the populate-time defaults; everyone else
    # writes a fresh basis.
    t1 = wp.vec3f(1.0, 0.0, 0.0)
    t2 = wp.vec3f(0.0, 1.0, 0.0)
    if use_tangent_from_anchor3:
        t1, t2 = _tangent_basis_from_anchor3(n_hat, r1_b1, r3_b1)
        write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
        write_vec3(constraints, base_offset + _OFF_T2, cid, t2)
    elif not has_anchor1_only or use_axis_from_local1:
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

    # ---- K-factor block A: anchor-1 standalone / 3+2 Schur / 4-row tangent
    if has_anchor1_only:
        eye3 = wp.identity(3, dtype=wp.float32)
        a1 = inv_mass1 * eye3
        a1 = a1 + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1))
        a1 = a1 + inv_mass2 * eye3
        a1 = a1 + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
        a1_inv = wp.inverse(a1)
        write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    if has_schur_3plus2:
        _pivot_anchor1_anchor2_K_factor_at(
            constraints,
            cid,
            base_offset,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            cr1_b1,
            cr1_b2,
            cr2_b1,
            cr2_b2,
            t1,
            t2,
        )
    if has_tangent_4row:
        b11, _b22, _b12 = _slide_anchor1_anchor2_K4_factor_at(
            constraints,
            cid,
            base_offset,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            cr1_b1,
            cr1_b2,
            cr2_b1,
            cr2_b2,
            t1,
            t2,
        )
    else:
        b11 = wp.identity(3, dtype=wp.float32)

    # ---- K-factor block B: anchor-3 Box2D scalar lock (FIXED, PRISMATIC)
    if has_anchor3:
        eye3_a3 = wp.identity(3, dtype=wp.float32)
        b33 = (inv_mass1 + inv_mass2) * eye3_a3
        b33 = b33 + cr3_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1))
        b33 = b33 + cr3_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))
        d_scalar = wp.dot(t2, b33 @ t2)
        if wp.abs(d_scalar) > 1.0e-20:
            s_scalar_inv = 1.0 / d_scalar
        else:
            s_scalar_inv = 0.0
        write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)
        # PRISMATIC's pre-block-GS schema kept a coupling vector here.
        # Block-GS leaves it dormant; stamp 0 for column-schema stability.
        write_vec4(constraints, base_offset + _OFF_C_PRIS, cid, wp.vec4f(0.0, 0.0, 0.0, 0.0))

    # ---- Soft-constraint coefficients (Box2D, shared) -----------------
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # ---- Biases (drift correction) -----------------------------------
    if has_anchor1_only:
        # 3-vec drift correction at anchor-1.
        bias1 = (p1_b2 - p1_b1) * bias_rate
        write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    if has_schur_3plus2:
        bias1 = (p1_b2 - p1_b1) * bias_rate
        drift2 = p2_b2 - p2_b1
        bias2 = wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0)
        write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
        write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)
    if has_tangent_4row:
        # PRISMATIC / CYLINDRICAL: tangent drift at both anchors.
        drift1 = p1_b2 - p1_b1
        drift2 = p2_b2 - p2_b1
        bias1 = wp.vec3f(wp.dot(t1, drift1) * bias_rate, wp.dot(t2, drift1) * bias_rate, 0.0)
        bias2 = wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0)
        write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
        write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)
    if has_anchor3:
        drift3 = p3_b2 - p3_b1
        bias3 = wp.dot(t2, drift3) * bias_rate
        write_float(constraints, base_offset + _OFF_BIAS3, cid, bias3)

    # ---- Warm-start application + re-projection ----------------------
    if has_anchor1_only:
        acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
        velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
            velocity1,
            velocity2,
            angular_velocity1,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            acc1,
            cr1_b1 @ acc1,
            cr1_b2 @ acc1,
        )
    if has_schur_3plus2:
        acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
        acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
        velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
            velocity1,
            velocity2,
            angular_velocity1,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            acc1 + acc2,
            cr1_b1 @ acc1 + cr2_b1 @ acc2,
            cr1_b2 @ acc1 + cr2_b2 @ acc2,
        )
    if has_tangent_4row:
        # Re-project tangent acc onto the new (t1, t2) basis.
        acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
        acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
        acc1_t1 = wp.dot(t1, acc_imp1_world)
        acc1_t2 = wp.dot(t2, acc_imp1_world)
        acc2_t1 = wp.dot(t1, acc_imp2_world)
        acc2_t2 = wp.dot(t2, acc_imp2_world)
        acc_imp1_world = acc1_t1 * t1 + acc1_t2 * t2
        acc_imp2_world = acc2_t1 * t1 + acc2_t2 * t2
        write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc_imp1_world)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc_imp2_world)
        total_linear = acc_imp1_world + acc_imp2_world
        velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
            velocity1,
            velocity2,
            angular_velocity1,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            total_linear,
            cr1_b1 @ acc_imp1_world + cr2_b1 @ acc_imp2_world,
            cr1_b2 @ acc_imp1_world + cr2_b2 @ acc_imp2_world,
        )
    if has_anchor3:
        # Re-project acc3 (PRISMATIC stores world-frame along t2;
        # FIXED similarly) and apply.
        acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
        acc3_t2 = wp.dot(t2, acc_imp3_world)
        acc_imp3_world = acc3_t2 * t2
        write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc_imp3_world)
        velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
            velocity1,
            velocity2,
            angular_velocity1,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            acc_imp3_world,
            cr3_b1 @ acc_imp3_world,
            cr3_b2 @ acc_imp3_world,
        )

    # ---- Axial drive / limit / friction prepare ----------------------
    # Single call to ``_axial_drive_limit_prepare_at`` with all inputs
    # computed conditionally above the call. The post-call impulse
    # application is conditional too (pure couple for angular, linear-
    # with-anchor-1-lever for linear), but the prepare helper itself
    # is hit exactly once per cid.
    if needs_axial_drive_prep:
        cumulative_value = wp.float32(0.0)
        eff_inv = wp.float32(0.0)
        drive_boost = PHOENX_BOOST_PRISMATIC_DRIVE
        limit_boost = PHOENX_BOOST_PRISMATIC_LIMIT
        if has_angular_axial:
            # Cumulative angle tracker about n_hat (orientation-delta
            # against inv_initial_orientation; revolution counter folds
            # the wrap discontinuity).
            inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
            diff = orientation2 * inv_init * wp.quat_inverse(orientation1)
            new_q_angle = extract_rotation_angle(diff, n_hat)
            old_counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
            old_prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
            new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
            write_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid, new_counter)
            write_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
            cumulative_value = revolution_tracker_angle(new_counter, new_prev)
            eff_inv = wp.dot(n_hat, inv_inertia1 @ n_hat) + wp.dot(n_hat, inv_inertia2 @ n_hat)
            drive_boost = PHOENX_BOOST_REVOLUTE_DRIVE
            limit_boost = PHOENX_BOOST_REVOLUTE_LIMIT
        else:
            # Linear slide along n_hat measured at anchor 1.
            cumulative_value = wp.dot(n_hat, p1_b2 - p1_b1)
            eff_inv = wp.dot(n_hat, b11 @ n_hat)
        axial_imp = _axial_drive_limit_prepare_at(
            constraints,
            cid,
            base_offset,
            cumulative_value,
            eff_inv,
            dt,
            drive_boost,
            limit_boost,
        )
        if has_angular_axial:
            # Pure couple about n_hat.
            angular_velocity1, angular_velocity2 = apply_pair_angular_impulse(
                angular_velocity1, angular_velocity2, inv_inertia1, inv_inertia2, -n_hat * axial_imp, -n_hat * axial_imp
            )
        else:
            # Linear impulse with anchor-1 lever.
            axial_world = n_hat * axial_imp
            velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
                velocity1,
                velocity2,
                angular_velocity1,
                angular_velocity2,
                inv_mass1,
                inv_mass2,
                inv_inertia1,
                inv_inertia2,
                -axial_world,
                -wp.cross(r1_b1, axial_world),
                -wp.cross(r1_b2, axial_world),
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
    """Prepare-pass dispatcher. CABLE and PLANAR have unique math (CABLE
    runs PD-soft anchor-2/3, PLANAR uses COM-frame Jacobians) and have
    their own prepare functions. Every other joint mode -- BALL_SOCKET,
    REVOLUTE, FIXED, UNIVERSAL, PRISMATIC, CYLINDRICAL -- routes through
    the unified :func:`_box2d_pivot_slide_prepare_at` which gates on
    ``joint_mode`` to select the right K-factor block, biases, warm-
    start, and axial-drive prep."""
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)
    if joint_mode == JOINT_MODE_CABLE:
        _cable_prepare_at(
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
        )
    elif joint_mode == JOINT_MODE_PLANAR:
        _planar_prepare_at(
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
        )
    else:
        _box2d_pivot_slide_prepare_at(
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

    Equivalent to calling :func:`_revolute_iterate_at` ``num_sweeps``
    times with the same arguments, but every per-cid constant
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
    # ``_revolute_iterate_at_multi`` is only invoked from the multi-world
    # fast-tail kernel, where ``mass_splitting`` is rejected at PhoenXWorld
    # construction. Use the lean direct-read path so NVRTC doesn't compile
    # the unreachable slot lookup into the kernel binary.
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
    ) = _ms_load_body_pair_lean(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

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

    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    ut_ai = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)
    s_inv_22 = _read_s_inv_22(constraints, base_offset, cid)
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
    acc2_world_initial = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_tan = wp.vec2f(wp.dot(t1, acc2_world_initial), wp.dot(t2, acc2_world_initial))

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
    limit_active = clamp != _CLAMP_NONE

    # ---- Friction constants (hoisted out of the sweep loop) ----------
    # Mirrors the friction row in :func:`_axial_drive_limit_iterate`,
    # inlined here for the register-cached multi-sweep path. The acc /
    # cached gamma + eff_mass come from the same dwords; only the
    # surrounding loop differs.
    friction_eff_mass = read_float(constraints, base_offset + _OFF_FRICTION_EFF_MASS, cid)
    friction_active = friction_eff_mass > wp.float32(0.0)
    friction_coefficient = wp.float32(0.0)
    friction_gamma = wp.float32(0.0)
    acc_friction = wp.float32(0.0)
    max_lambda_friction = wp.float32(0.0)
    if friction_active:
        friction_coefficient = read_float(constraints, base_offset + _OFF_FRICTION_COEFFICIENT, cid)
        friction_gamma = read_float(constraints, base_offset + _OFF_FRICTION_GAMMA, cid)
        acc_friction = read_float(constraints, base_offset + _OFF_ACC_FRICTION, cid)
        max_lambda_friction = friction_coefficient * (wp.float32(1.0) / idt)
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
        # Positional PGS: anchor-1 (3 rows) + anchor-2 tangent (2 rows)
        jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
        jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
        jv2_t1 = wp.dot(t1, jv2_world)
        jv2_t2 = wp.dot(t2, jv2_world)
        jv2 = wp.vec2f(jv2_t1, jv2_t2)

        rhs1 = jv1 + bias1
        rhs2 = jv2 + bias2_tan

        ut_ai_rhs1_3 = ut_ai @ rhs1
        ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

        lam2_us = block_solve_inverse_2(s_inv_22, rhs2 - ut_ai_rhs1)
        update2 = block_project_accumulated_2(lam2_us, acc2_tan, mass_coeff, impulse_coeff, sor_boost)
        lam2 = update2.delta

        lam2_world = lam2[0] * t1 + lam2[1] * t2
        lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

        u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
        u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
        u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

        lam1_us = block_solve_inverse_3(a1_inv, rhs1 + u_lam2_us)
        update1 = block_project_accumulated_3(lam1_us, acc1, mass_coeff, impulse_coeff, sor_boost)
        lam1 = update1.delta

        total_lin = lam1 + lam2_world

        velocity1, velocity2, angular_velocity1, angular_velocity2 = apply_pair_spatial_impulse(
            velocity1,
            velocity2,
            angular_velocity1,
            angular_velocity2,
            inv_mass1,
            inv_mass2,
            inv_inertia1,
            inv_inertia2,
            total_lin,
            cr1_b1 @ lam1 + cr2_b1 @ lam2_world,
            cr1_b2 @ lam1 + cr2_b2 @ lam2_world,
        )

        acc1 = update1.lambda_new
        acc2_tan = update2.lambda_new

        # Axial drive + limit + friction scalar PGS.
        jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)
        axial_update = _axial_project_scalar_rows(
            jv_axial,
            clamp,
            sor_boost,
            drive_active,
            max_force_drive,
            max_lambda_drive,
            bias_drive,
            gamma_drive,
            eff_mass_drive_soft,
            acc_drive,
            limit_active,
            pd_mode_limit,
            pd_mass,
            pd_gamma,
            pd_beta,
            eff_axial,
            bias_box,
            mc_limit,
            ic_limit,
            acc_limit,
            friction_active,
            friction_eff_mass,
            friction_gamma,
            max_lambda_friction,
            acc_friction,
        )
        axial_lam = axial_update.delta
        acc_drive = axial_update.acc_drive
        acc_limit = axial_update.acc_limit
        acc_friction = axial_update.acc_friction
        angular_velocity1, angular_velocity2 = apply_pair_angular_impulse(
            angular_velocity1, angular_velocity2, inv_inertia1, inv_inertia2, -n_hat * axial_lam, -n_hat * axial_lam
        )

        it += 1

    # ---- Writeback ---------------------------------------------------
    # See the matching ``_ms_load_body_pair_lean`` comment at the top of
    # this function for why the lean store is safe here.
    _ms_store_body_pair_lean(
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

    acc2_world = acc2_tan[0] * t1 + acc2_tan[1] * t2
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world)
    if drive_active:
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, acc_drive)
    if limit_active:
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, acc_limit)
    if friction_active:
        write_float(constraints, base_offset + _OFF_ACC_FRICTION, cid, acc_friction)


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
    body_pair = constraint_bodies_make(b1, b2)
    joint_mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if joint_mode == JOINT_MODE_REVOLUTE:
        _revolute_iterate_at_multi(
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
        )
    else:
        it = wp.int32(0)
        while it < num_sweeps:
            actuated_double_ball_socket_iterate_at(
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
            )
            it += 1


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
    """Building-block iterate dispatcher. Reads ``joint_mode`` once and
    treats it as a feature vector that selects which row-block helpers
    fire. Each block helper is called from exactly this site, gated by
    a single ``if`` -- one inlined copy per block in the kernel
    binary, regardless of how many joint modes activate it.

    PLANAR has a unique row structure (COM-frame Jacobians, no anchor
    lever arms) that doesn't fit the shared anchor-lever-arm grammar
    and routes out to its own iterate. Every other joint mode --
    BALL_SOCKET, REVOLUTE, FIXED, UNIVERSAL, PRISMATIC, CYLINDRICAL,
    CABLE -- decomposes into a subset of eight row blocks defined
    above.

    Joint-mode -> block matrix (Y means the block fires for this mode):

        ===========   ====  ====  =====  ====  ====  =====  ====  ====
        Mode          A1    A1A2  Tan4   A3sc  CblA2 CblA3  AngA  LinA
        ===========   ====  ====  =====  ====  ====  =====  ====  ====
        BALL_SOCKET   Y     -     -      -     -     -      -     -
        REVOLUTE      -     Y     -      -     -     -      Y     -
        FIXED         -     Y     -      Y     -     -      -     -
        UNIVERSAL     Y     -     -      -     -     -      Y     -
        PRISMATIC     -     -     Y      Y     -     -      -     Y
        CYLINDRICAL   -     -     Y      -     -     -      -     Y
        CABLE         Y     -     -      -     Y     Y      -     -
        ===========   ====  ====  =====  ====  ====  =====  ====  ====

    A1 = :func:`_anchor1_standalone_block` (Box2D-soft 3-row)
    A1A2 = :func:`_anchor1_anchor2_schur_block` (3+2 Schur)
    Tan4 = :func:`_anchor1_anchor2_tangent_4row_block` (4-row direct)
    A3sc = :func:`_anchor3_scalar_block` (Box2D-soft 1-row)
    CblA2 = :func:`_cable_anchor2_pd_block` (PD-soft 2-row bend)
    CblA3 = :func:`_cable_anchor3_pd_block` (PD-soft 1-row twist)
    AngA = :func:`_angular_axial_block`
    LinA = :func:`_linear_axial_block`

    ``use_bias`` is the Box2D v3 TGS-soft flag (gates positional drift
    biases on Box2D-soft blocks; CABLE PD blocks read their biases
    unconditionally as spring forces, not drift)."""
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)

    # Block-enable flags computed from joint_mode.
    has_anchor1_only = (
        joint_mode == JOINT_MODE_BALL_SOCKET or joint_mode == JOINT_MODE_UNIVERSAL or joint_mode == JOINT_MODE_CABLE
    )
    has_schur_3plus2 = joint_mode == JOINT_MODE_REVOLUTE or joint_mode == JOINT_MODE_FIXED
    has_tangent_4row = joint_mode == JOINT_MODE_PRISMATIC or joint_mode == JOINT_MODE_CYLINDRICAL
    has_anchor3_box2d = joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_PRISMATIC
    has_cable_anchor2 = joint_mode == JOINT_MODE_CABLE
    has_cable_anchor3 = joint_mode == JOINT_MODE_CABLE
    has_angular_axial = joint_mode == JOINT_MODE_REVOLUTE or joint_mode == JOINT_MODE_UNIVERSAL
    has_linear_axial = joint_mode == JOINT_MODE_PRISMATIC or joint_mode == JOINT_MODE_CYLINDRICAL
    has_planar = joint_mode == JOINT_MODE_PLANAR

    b1 = body_pair.b1
    b2 = body_pair.b2
    (
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    r1_b1 = read_vec3(constraints, base_offset + _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    # anchor-2 lever arms / tangent basis used by every block that
    # involves anchor-2 or anchor-3 (BALL_SOCKET / CABLE-anchor1 don't
    # read these but the dispatcher reads are scalar-cheap).
    r2_b1 = read_vec3(constraints, base_offset + _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    # ---- Block A: anchor-1 positional lock ---------------------------
    if has_anchor1_only:
        v1, v2, w1, w2 = _anchor1_standalone_block(
            constraints,
            cid,
            base_offset,
            v1,
            v2,
            w1,
            w2,
            im1,
            im2,
            ii1,
            ii2,
            cr1_b1,
            cr1_b2,
            bias1,
            mass_coeff,
            impulse_coeff,
            sor_boost,
        )
    if has_schur_3plus2:
        v1, v2, w1, w2 = _anchor1_anchor2_schur_block(
            constraints,
            cid,
            base_offset,
            v1,
            v2,
            w1,
            w2,
            im1,
            im2,
            ii1,
            ii2,
            t1,
            t2,
            cr1_b1,
            cr1_b2,
            cr2_b1,
            cr2_b2,
            bias1,
            bias2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
        )
    if has_tangent_4row:
        v1, v2, w1, w2 = _anchor1_anchor2_tangent_4row_block(
            constraints,
            cid,
            base_offset,
            v1,
            v2,
            w1,
            w2,
            im1,
            im2,
            ii1,
            ii2,
            t1,
            t2,
            cr1_b1,
            cr1_b2,
            cr2_b1,
            cr2_b2,
            bias1,
            bias2,
            mass_coeff,
            impulse_coeff,
            sor_boost,
        )

    # ---- Block B: anchor-2 PD-soft (cable bend) ----------------------
    if has_cable_anchor2:
        v1, v2, w1, w2 = _cable_anchor2_pd_block(
            constraints,
            cid,
            base_offset,
            v1,
            v2,
            w1,
            w2,
            im1,
            im2,
            ii1,
            ii2,
            t1,
            t2,
            cr2_b1,
            cr2_b2,
            sor_boost,
        )

    # ---- Block C: anchor-3 scalar lock -------------------------------
    if has_anchor3_box2d or has_cable_anchor3:
        r3_b1 = read_vec3(constraints, base_offset + _OFF_R3_B1, cid)
        r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
        cr3_b1 = wp.skew(r3_b1)
        cr3_b2 = wp.skew(r3_b2)
        if has_anchor3_box2d:
            if use_bias:
                bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
            else:
                bias3 = wp.float32(0.0)
            v1, v2, w1, w2 = _anchor3_scalar_block(
                constraints,
                cid,
                base_offset,
                v1,
                v2,
                w1,
                w2,
                im1,
                im2,
                ii1,
                ii2,
                t2,
                cr3_b1,
                cr3_b2,
                bias3,
                mass_coeff,
                impulse_coeff,
                sor_boost,
            )
        if has_cable_anchor3:
            v1, v2, w1, w2 = _cable_anchor3_pd_block(
                constraints,
                cid,
                base_offset,
                v1,
                v2,
                w1,
                w2,
                im1,
                im2,
                ii1,
                ii2,
                t2,
                cr3_b1,
                cr3_b2,
                sor_boost,
            )

    # ---- Block D: axial drive / limit / friction ---------------------
    if has_angular_axial or has_linear_axial:
        n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
        clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
        if has_angular_axial:
            w1, w2 = _angular_axial_block(
                constraints,
                cid,
                base_offset,
                w1,
                w2,
                ii1,
                ii2,
                n_hat,
                clamp,
                idt,
                sor_boost,
            )
        if has_linear_axial:
            v1, v2, w1, w2 = _linear_axial_block(
                constraints,
                cid,
                base_offset,
                v1,
                v2,
                w1,
                w2,
                im1,
                im2,
                ii1,
                ii2,
                r1_b1,
                r1_b2,
                n_hat,
                clamp,
                idt,
                sor_boost,
            )

    # ---- Block E: PLANAR 3-row COM-frame solve -----------------------
    # PLANAR is the only mode whose row Jacobian acts on relative COM
    # motion (no anchor lever arms). It reuses the ``A1_INV`` /
    # ``ACC_IMP1`` / ``ACC_IMP2`` slots with PLANAR-specific semantics:
    # ``ACC_IMP1`` is the linear-along-n_hat impulse, ``ACC_IMP2`` is
    # the (t1, t2) angular impulse. ``n_hat`` is the plane normal.
    if has_planar:
        n_hat_planar = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
        v1, v2, w1, w2 = _planar_3row_block(
            constraints,
            cid,
            base_offset,
            v1,
            v2,
            w1,
            w2,
            im1,
            im2,
            ii1,
            ii2,
            n_hat_planar,
            t1,
            t2,
            bias1,
            mass_coeff,
            impulse_coeff,
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
        v1,
        w1,
        v2,
        w2,
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

    if joint_mode == JOINT_MODE_REVOLUTE:
        force = (acc1 + acc2) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt)
        # Axial block is a torque about -n_hat.
        torque = torque - n_hat * ((acc_drive + acc_limit) * idt)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
        # Axial block is a linear force along -n_hat.
        axial_force = n_hat * ((acc_drive + acc_limit) * idt)
        force = force - axial_force
        torque = torque - wp.cross(r1_b2, axial_force)
    elif joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_CABLE:
        # Same anchor layout (anchor-1 3-row + anchor-2 tangent 2-row +
        # anchor-3 scalar 1-row); no axial block. CABLE's PD softness
        # is already baked into the accumulated impulses, so the
        # wrench reflects the actual reaction the joint applied this
        # substep.
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    else:
        # Ball-socket: only the anchor-1 impulse.
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt)
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
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_prepare_for_iteration_at(
        constraints, cid, 0, bodies, particles, copy_state, num_bodies, parallel_id, body_pair, idt
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
    (
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        slot1,
        slot2,
    ) = _ms_load_body_pair(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    r1_b1 = read_vec3(constraints, _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, _OFF_R1_B2, cid)
    r2_b1 = read_vec3(constraints, _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, _OFF_R2_B2, cid)
    t1 = read_vec3(constraints, _OFF_T1, cid)
    t2 = read_vec3(constraints, _OFF_T2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    if use_bias:
        bias1 = read_vec3(constraints, _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, _OFF_BIAS2, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
    mass_coeff = read_float(constraints, _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, _OFF_IMPULSE_COEFF, cid)

    v1, v2, w1, w2 = _anchor1_anchor2_schur_block(
        constraints,
        cid,
        0,
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        t1,
        t2,
        cr1_b1,
        cr1_b2,
        cr2_b1,
        cr2_b2,
        bias1,
        bias2,
        mass_coeff,
        impulse_coeff,
        sor_boost,
    )
    n_hat = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
    clamp = read_int(constraints, _OFF_CLAMP, cid)
    w1, w2 = _angular_axial_block(constraints, cid, 0, w1, w2, ii1, ii2, n_hat, clamp, idt, sor_boost)

    _ms_store_body_pair(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        v1,
        w1,
        v2,
        w2,
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
    """Apply cached revolute warm-start impulses without rebuilding J.

    Intended for prepare-refresh reuse in rigid, mass-splitting-free
    worlds where the previous prepare pass already cached r1/r2, the
    tangent basis, soft coefficients, clamp state, and axial row data.
    """
    _ = idt
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    (
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        slot1,
        slot2,
    ) = _ms_load_body_pair_lean(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    r1_b1 = read_vec3(constraints, _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, _OFF_R1_B2, cid)
    r2_b1 = read_vec3(constraints, _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, _OFF_R2_B2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

    acc1 = read_vec3(constraints, _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, _OFF_ACC_IMP2, cid)
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        acc1 + acc2,
        cr1_b1 @ acc1 + cr2_b1 @ acc2,
        cr1_b2 @ acc1 + cr2_b2 @ acc2,
    )

    n_hat = read_vec3(constraints, _OFF_AXIS_WORLD, cid)
    axial_imp = (
        read_float(constraints, _OFF_ACC_DRIVE, cid)
        + read_float(constraints, _OFF_ACC_LIMIT, cid)
        + read_float(constraints, _OFF_ACC_FRICTION, cid)
    )
    w1, w2 = apply_pair_angular_impulse(w1, w2, ii1, ii2, -n_hat * axial_imp, -n_hat * axial_imp)

    _ms_store_body_pair_lean(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        v1,
        w1,
        v2,
        w2,
    )


@wp.func
def _ball_socket_cached_warmstart(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
):
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    (
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        slot1,
        slot2,
    ) = _ms_load_body_pair_lean(bodies, particles, copy_state, b1, b2, parallel_id, num_bodies)

    r1_b1 = read_vec3(constraints, _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, _OFF_R1_B2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    acc1 = read_vec3(constraints, _OFF_ACC_IMP1, cid)
    v1, v2, w1, w2 = apply_pair_spatial_impulse(
        v1,
        v2,
        w1,
        w2,
        im1,
        im2,
        ii1,
        ii2,
        acc1,
        cr1_b1 @ acc1,
        cr1_b2 @ acc1,
    )

    _ms_store_body_pair_lean(
        bodies,
        particles,
        copy_state,
        b1,
        b2,
        slot1,
        slot2,
        num_bodies,
        v1,
        w1,
        v2,
        w2,
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
    """Apply cached joint warm-start when a prepare pass is reused.

    Revolute keeps the existing 5-row cached path. Ball-socket can reuse
    cached anchor-1 lever arms directly. Other ADBS modes still need fresh
    tangent/axis projection, so they intentionally fall back to full prepare
    on skipped-refresh substeps.
    """
    joint_mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if joint_mode == JOINT_MODE_REVOLUTE:
        revolute_cached_warmstart(constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt)
    elif joint_mode == JOINT_MODE_BALL_SOCKET:
        _ball_socket_cached_warmstart(constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id)
    else:
        actuated_double_ball_socket_prepare_for_iteration(
            constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
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
    """Revolute-only prepare entry. Routes into the unified
    :func:`_box2d_pivot_slide_prepare_at` with ``joint_mode`` passed
    as the compile-time-known REVOLUTE tag so the dispatch flags
    fold to (has_schur_3plus2, has_angular_axial) on inlining."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    _box2d_pivot_slide_prepare_at(
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
    body_pair = constraint_bodies_make(b1, b2)
    _revolute_iterate_at_multi(
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
    if joint_mode != JOINT_MODE_BALL_SOCKET:
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
        # as _prismatic_prepare_at). The axial sign matches the
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
