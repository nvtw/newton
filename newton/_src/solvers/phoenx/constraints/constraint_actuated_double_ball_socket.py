# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unified ball-socket / revolute / prismatic joint with optional PD
drive and limit on the free DoF. Runtime ``joint_mode`` picks the
locked DoF set; everything else (soft-constraint plumbing,
warm-starting, Schur blocks) is shared.

Ball-socket (:data:`JOINT_MODE_BALL_SOCKET`)
    3-row point lock at ``anchor1``; all 3 rotations free; no drive
    / limit. ``anchor2`` unused.

Revolute (:data:`JOINT_MODE_REVOLUTE`)
    5-DoF hinge about ``n_hat = anchor2 - anchor1``. Anchor 1: full
    3-row lock. Anchor 2: 2-row tangent-plane lock (axial row is the
    analytical null-space of the 6-row stack). Solved as a 3x3 + 2x2
    Schur. Drive / limit act on the twist about ``n_hat``; ``target``,
    ``min_value``, ``max_value`` are in rad, ``max_force_drive`` in N*m.

Prismatic (:data:`JOINT_MODE_PRISMATIC`)
    5-DoF slider along ``n_hat``. Anchors 1+2 each contribute 2
    tangent rows; a third auto-derived off-axis anchor contributes
    one scalar row along ``t2`` that kills rotation about ``n_hat``.
    Basis ``(t1, t2)`` is rebuilt per substep so the scalar row is a
    unit-gain tangential velocity gate (avoids ``cos(alpha)``
    mismatches in block-GS chains). Solved as a 4x4 + 1x1 Schur.
    Drive / limit act on the slide; units are m and N.

Drive row -- always PD. ``DRIVE_MODE_POSITION`` / ``DRIVE_MODE_VELOCITY``
decide whether the bias folds in ``target`` or ``target_velocity``;
both require caller-supplied ``stiffness_drive`` and/or
``damping_drive`` (``DRIVE_MODE_VELOCITY`` further requires
``damping_drive > 0``). ``max_force_drive > 0`` caps the per-substep
impulse; ``0`` means unlimited (POSITION) or disables the drive
(VELOCITY). Both gains zero disables the drive row entirely.

Limit row -- unilateral ``[min_value, max_value]`` spring-damper
(``min_value > max_value`` disables). Dual softness: both PD gains zero
uses Box2D ``(hertz_limit, damping_ratio_limit)``; either gain
positive selects PD with absolute SI gains. Drive and limit share the
same scalar PGS row; the limit is unilateral and always wins.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    pd_coefficients,
    pd_coefficients_split,
    read_float,
    read_int,
    read_mat33,
    read_mat44,
    read_quat,
    read_vec3,
    read_vec4,
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
    create_orthonormal,
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)

__all__ = [
    "ADBS_DWORDS",
    "DRIVE_MODE_OFF",
    "DRIVE_MODE_POSITION",
    "DRIVE_MODE_VELOCITY",
    "JOINT_MODE_BALL_SOCKET",
    "JOINT_MODE_BEAM",
    "JOINT_MODE_CABLE",
    "JOINT_MODE_FIXED",
    "JOINT_MODE_PRISMATIC",
    "JOINT_MODE_REVOLUTE",
    "ActuatedDoubleBallSocketData",
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
#: Cable (soft ball-socket): BALL_SOCKET's rigid 3-row point lock +
#: three soft angular rows (2x ``k_bend`` perpendicular to
#: ``anchor1->anchor2``, 1x ``k_twist`` along it). Each row uses the
#: same Box2D-style soft PD plumbing as the axial drive/limit, so
#: warm-start, Nyquist clamping, and relax semantics are reused.
#:
#: Column slot reuse (unchanged ``ADBS_DWORDS``):
#:   * ``inv_initial_orientation`` -- rest relative orientation
#:     (REVOLUTE already snapshots this quaternion).
#:   * ``axis_local1`` -- ``anchor2-anchor1`` in body-1 local frame.
#:   * ``stiffness_drive`` / ``damping_drive`` -> ``k_bend`` / ``d_bend``.
#:   * ``stiffness_limit`` / ``damping_limit`` -> ``k_twist`` / ``d_twist``.
#:   * ``ut_ai`` (mat33) -> world-frame row directions
#:     ``(d_bend1, d_bend2, d_twist)``.
#:   * ``s_inv`` (mat33) -> per-row ``(gamma, bias, eff_mass_soft)``.
#:   * ``acc_imp2`` (vec3) -> ``(acc_bend1, acc_bend2, acc_twist)``.
JOINT_MODE_CABLE = wp.constant(wp.int32(4))
#: Beam (soft fixed): rigid 3-row anchor-1 ball-socket + 2-row tangent
#: anchor-2 PD spring-damper (k_bend, d_bend) + 1-row scalar anchor-3
#: PD spring-damper (k_twist, d_twist). Solved as block Gauss-Seidel
#: between the three blocks (same row layout as :data:`JOINT_MODE_FIXED`),
#: but with independent per-block soft coefficients. Converges to
#: REVOLUTE-quality behaviour as ``k_bend -> infinity`` (the anchor-2
#: tangent rows lock both bend axes) and to FIXED as ``k_twist -> infinity``
#: as well. User gains are in **rotational SI units**:
#:
#:   * ``k_bend`` [N*m/rad], ``d_bend`` [N*m*s/rad] -- maps to a positional
#:     spring at anchor 2 with ``k_pos = k_bend / rest_length^2`` (lever
#:     arm ``rest_length`` between anchor 1 and anchor 2 along ``n_hat``).
#:   * ``k_twist`` [N*m/rad], ``d_twist`` [N*m*s/rad] -- maps to a
#:     positional spring along ``t2`` at anchor 3 with the same
#:     ``1 / rest_length^2`` rescale (lever arm ``rest_length`` between
#:     anchor 1 and anchor 3 perpendicular to ``n_hat``).
#:
#: Column slot reuse (no schema growth):
#:   * ``stiffness_drive`` / ``damping_drive`` -> ``k_bend`` / ``d_bend``.
#:   * ``stiffness_limit`` / ``damping_limit`` -> ``k_twist`` / ``d_twist``.
#:   * Anchor-3 snapshot uses the PRISMATIC / FIXED ``mode_extras`` layout
#:     (la3_b1, la3_b2, ...). No revolute twist tracker.
#:   * ``s_inv`` (mat33, 9 dwords) packs the per-substep PD soft cache:
#:     dwords 0..3 = K22 soft inverse (2x2), 4 = gamma_bend,
#:     5 = M_twist_soft, 6 = gamma_twist; 7..8 unused.
#:   * ``bias2`` (vec3) holds (bias_bend_t1, bias_bend_t2, 0).
#:   * ``bias3`` (1 dword) holds bias_twist along t2.
JOINT_MODE_BEAM = wp.constant(wp.int32(5))


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

    Union over the two joint modes: revolute and prismatic use
    mode-specific Schur caches but share every other slot (anchors,
    lever arms, warm-start impulses, drive / limit scalars), so the
    dispatcher treats them as one constraint type.

    Layout groups:

    * **Header** -- ``constraint_type / body1 / body2``.
    * **Shared positional block** -- two user anchors on the joint
      axis, their body-local snapshots, runtime lever arms, cached
      tangent basis, revolute rest-pose quat.
    * **Revolute Schur cache** -- ``a1_inv, ut_ai, s_inv``.
    * **Prismatic Schur cache** -- ``a4_inv`` (mat44f), ``c_pris``
      (vec4f), ``s_scalar_inv``, plus anchor-3 snapshots / lever arms.
    * **Warm-start** -- three ``vec3f`` accumulated impulses covering
      both modes (revolute: ``acc_imp1`` full vec3 + ``acc_imp2``
      tangent-only; prismatic: all three tangent-only).
    * **Actuator block** -- drive / limit setpoints, cached soft-
      constraint coefficients, scalar accumulated impulses.

    Storage: ~80 dwords (~320 B per joint).
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
    # :func:`pd_coefficients` for the implicit-Euler math.
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
# Cable-only fields. Cable shares the revolute ``inv_initial_orientation``
# in dwords 0..3 (the rest pose snapshot uses the same construction); on
# top of that it stores per-row damping effective masses for the
# spring/damping split (relax pass applies them as velocity-only damping
# impulses). dwords 6..15 are still free.
_OFF_CABLE_DAMP_MASS_BEND1 = wp.constant(int(_OFF_MODE_EXTRAS) + 6)
_OFF_CABLE_DAMP_MASS_BEND2 = wp.constant(int(_OFF_MODE_EXTRAS) + 7)
_OFF_CABLE_DAMP_MASS_TWIST = wp.constant(int(_OFF_MODE_EXTRAS) + 8)
# Beam-only PD soft-cache aliases over the existing ``s_inv`` mat33 slot
# (9 dwords). Beam never uses the 3+2 Schur, so the revolute / fixed /
# cable layout for these dwords is free to reinterpret here.
#   dwords 0..3 = K22_soft inverse (2x2 packed: m00, m01, m10, m11)
#   dword 4     = gamma_bend       (PD softness coefficient, anchor-2 PD rows)
#   dword 5     = M_twist_soft     (PD softened effective mass for anchor-3 row)
#   dword 6     = gamma_twist      (PD softness coefficient, anchor-3 PD row)
#   dwords 7..8 = unused
_OFF_BEAM_K22_INV_00 = wp.constant(int(_OFF_S_INV) + 0)
_OFF_BEAM_K22_INV_01 = wp.constant(int(_OFF_S_INV) + 1)
_OFF_BEAM_K22_INV_10 = wp.constant(int(_OFF_S_INV) + 2)
_OFF_BEAM_K22_INV_11 = wp.constant(int(_OFF_S_INV) + 3)
_OFF_BEAM_GAMMA_BEND = wp.constant(int(_OFF_S_INV) + 4)
_OFF_BEAM_M_TWIST_SOFT = wp.constant(int(_OFF_S_INV) + 5)
_OFF_BEAM_GAMMA_TWIST = wp.constant(int(_OFF_S_INV) + 6)

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

#: Total dword count of one unified joint constraint.
ADBS_DWORDS: int = num_dwords(ActuatedDoubleBallSocketData)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
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
            units; both ``0`` disables the drive row.
        min_value, max_value: Limit window [rad or m]; ``min > max``
            disables the limit.
        hertz_limit, damping_ratio_limit: Box2D-style limit knobs;
            used iff ``stiffness_limit == damping_limit == 0``.
        stiffness_limit, damping_limit: PD limit gains (absolute SI).
            If either > 0 the limit uses the Jitter2 spring-damper
            path and the Box2D knobs are ignored.
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

    # ``mode_extras`` block is mode-aliased: REVOLUTE / CABLE store the
    # twist-tracker scratch (inv_initial_orientation, revolution_counter,
    # previous_quaternion_angle); PRISMATIC / FIXED store the anchor-3
    # snapshot + bias3 + acc_imp3. Writing both layouts unconditionally
    # would clobber the alias, so we branch.
    if mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_BEAM:
        write_vec3(constraints, _OFF_LA3_B1, cid, la3_b1)
        write_vec3(constraints, _OFF_LA3_B2, cid, la3_b2)
        write_vec3(constraints, _OFF_R3_B1, cid, zero3)
        write_vec3(constraints, _OFF_R3_B2, cid, zero3)
        write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, 0.0)
    else:
        # REVOLUTE / CABLE / BALL_SOCKET: zero out the anchor-3 slots
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
) -> wp.float32:
    """Scalar drive+limit PGS step for revolute/prismatic mode.

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

    Returns:
        Net per-iteration axial impulse ``lam_drive + lam_limit``.
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

    return lam_drive + lam_limit


# ---------------------------------------------------------------------------
# Shared anchor-1 positional prepare helper
# ---------------------------------------------------------------------------


@wp.func
def _anchor1_positional_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Anchor-1 3-row positional lock prepare. Shared by BALL_SOCKET
    and CABLE modes (which use a standalone anchor-1 lock as their
    only positional constraint).

    Reads the body-local snapshots ``la1_b{1,2}``, rotates them into
    world frame, rebuilds the 3x3 effective mass ``a1`` and its
    inverse, folds Box2D soft-constraint coefficients from
    ``hertz / damping_ratio`` into ``bias1`` + ``mass_coeff`` +
    ``impulse_coeff``, and applies the anchor-1 warm-start to the
    body velocities. Writes ``r1_b{1,2}``, ``a1_inv``, ``bias_rate``,
    ``mass_coeff``, ``impulse_coeff``, ``bias1`` to the column.

    Returns ``(r1_b1, r1_b2, cr1_b1, cr1_b2, velocity1,
    angular_velocity1, velocity2, angular_velocity2)`` so callers can
    continue with additional positional / angular rows (cable's bend
    + twist block, etc.) before committing body velocities.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]
    position1 = bodies.position[b1]
    position2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la1_b2 = read_vec3(constraints, base_offset + _OFF_LA1_B2, cid)
    r1_b1 = wp.quat_rotate(orient1, la1_b1)
    r1_b2 = wp.quat_rotate(orient2, la1_b2)
    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)

    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    eye3 = wp.identity(3, dtype=wp.float32)
    a1 = inv_mass1 * eye3
    a1 = a1 + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1))
    a1 = a1 + inv_mass2 * eye3
    a1 = a1 + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
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

    # 3-DoF positional warm-start (only the anchor-1 impulse).
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    velocity1 = bodies.velocity[b1] - inv_mass1 * acc1
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1_b1 @ acc1)
    velocity2 = bodies.velocity[b2] + inv_mass2 * acc1
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr1_b2 @ acc1)

    return r1_b1, r1_b2, cr1_b1, cr1_b2, velocity1, angular_velocity1, velocity2, angular_velocity2


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
            stiffness_drive, damping_drive, drive_C, eff_inv, dt
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
        pd_gamma_limit, pd_beta_limit, pd_m_soft = pd_coefficients(stiffness_limit, damping_limit, limit_C, eff_inv, dt)
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
    return acc_drive + acc_limit


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


@wp.func
def _ball_socket_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Ball-socket prepare pass.

    Strict subset of :func:`_revolute_prepare_at`: only the 3-row
    anchor-1 lock is built. Delegates to
    :func:`_anchor1_positional_prepare_at` for the shared 3-row
    assemble + warm-start; then commits the warm-started velocities
    to body storage.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    (
        _r1_b1,
        _r1_b2,
        _cr1_b1,
        _cr1_b2,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    ) = _anchor1_positional_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def _ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Ball-socket PGS iterate.

    Single 3-row positional solve: ``lam1_us = -A1^-1 * (J v + bias)``
    followed by the shared soft-constraint softening
    ``lam1 = mass_coeff * lam1_us - impulse_coeff * acc1`` and the
    usual ``acc1 += lam1`` warm-start update. No anchor-2 or anchor-3
    rows, no axial block.

    ``use_bias=False`` zeroes ``bias1`` -- the Box2D v3 TGS-soft
    relax-pass convention that enforces ``Jv = 0`` without
    re-injecting positional drift as velocity.
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

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)

    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)

    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    rhs1 = jv1 + bias1

    lam1_us = -(a1_inv @ rhs1)
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1

    velocity1 = velocity1 - inv_mass1 * lam1
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1)
    velocity2 = velocity2 + inv_mass2 * lam1
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)


# ---------------------------------------------------------------------------
# Revolute (hinge) mode math
# ---------------------------------------------------------------------------


@wp.func
def _revolute_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Revolute-mode prepare pass.

    Identical to the old :mod:`constraint_actuated_double_ball_socket`
    math: a 3+2 Schur complement on the (anchor1 3-row lock, anchor2
    tangent 2-row lock) stack, followed by a scalar drive / limit row
    on the axial twist. See the module docstring for the formulation.
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

    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)
    r2_b1 = wp.quat_rotate(orientation1, la2_b1)
    r2_b2 = wp.quat_rotate(orientation2, la2_b2)

    write_vec3(constraints, base_offset + _OFF_R1_B1, cid, r1_b1)
    write_vec3(constraints, base_offset + _OFF_R1_B2, cid, r1_b2)
    write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
    write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)

    p1_b1 = position1 + r1_b1
    p1_b2 = position2 + r1_b2
    p2_b1 = position1 + r2_b1
    p2_b2 = position2 + r2_b2

    hinge_vec = p2_b2 - p1_b2
    hinge_len2 = wp.dot(hinge_vec, hinge_vec)
    if hinge_len2 > 1.0e-20:
        n_hat = hinge_vec / wp.sqrt(hinge_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)

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

    t_mat = wp.mat33f(
        t1[0],
        t2[0],
        0.0,
        t1[1],
        t2[1],
        0.0,
        t1[2],
        t2[2],
        0.0,
    )
    tt = wp.transpose(t_mat)

    u_mat = b_mat @ t_mat
    d_mat = tt @ (a2 @ t_mat)

    a1_inv = wp.inverse(a1)
    ut_ai = wp.transpose(u_mat) @ a1_inv
    s_mat = d_mat - ut_ai @ u_mat

    s22 = wp.mat22f(
        s_mat[0, 0],
        s_mat[0, 1],
        s_mat[1, 0],
        s_mat[1, 1],
    )
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

    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, ut_ai)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv_packed)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    bias1 = (p1_b2 - p1_b1) * bias_rate
    drift2 = p2_b2 - p2_b1
    bias2_t1 = wp.dot(t1, drift2) * bias_rate
    bias2_t2 = wp.dot(t2, drift2) * bias_rate
    bias2 = wp.vec3f(bias2_t1, bias2_t2, 0.0)
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)

    # 5-DoF positional warm-start.
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)

    velocity1 = bodies.velocity[b1] - inv_mass1 * (acc1 + acc2)
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1_b1 @ acc1 + cr2_b1 @ acc2)
    velocity2 = bodies.velocity[b2] + inv_mass2 * (acc1 + acc2)
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr1_b2 @ acc1 + cr2_b2 @ acc2)

    # ---- Axial drive + limit block (angular) ------------------------
    # Angular effective mass: the axial impulse is a pure torque along
    # ``n_hat``, so ``m_inv = n . (I1^-1 + I2^-1) . n``. Joint armature
    # (rotor / leadscrew inertia) is *baked into* ``inv_inertia_world``
    # at solver construction (see ``SolverPhoenX._bake_joint_armature_into_body_inertia``),
    # so this expression is already armature-aware -- no extra term here.
    eff_inv = wp.dot(n_hat, inv_inertia1 @ n_hat) + wp.dot(n_hat, inv_inertia2 @ n_hat)

    # Revolute twist tracker: ``diff = q2 * inv_init * q1^*`` is the
    # identity at finalize() time; :func:`extract_rotation_angle`
    # projected onto the body-1 axis returns the signed angle in
    # ``(-pi, pi]``; the revolution counter extends that to an
    # unbounded cumulative angle. We use the body-1 local-axis
    # snapshot here (rather than ``n_hat``) because the tracker
    # output must remain well-defined even if anchors briefly
    # coincide (``|a2 - a1| ~ 0`` would zero out ``n_hat``).
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

    # Shared drive / limit prepare writes gamma_drive, bias_drive,
    # eff_mass_drive_soft, max_lambda_drive, clamp, PD-or-Box2D limit
    # coefficients. Returns the warm-start axial impulse (sum of the
    # drive + limit accumulated impulses, with the limit one gated on
    # the clamp state). Pure torque on both bodies along ``n_hat``.
    axial_imp = _axial_drive_limit_prepare_at(constraints, cid, base_offset, cumulative_angle, eff_inv, dt)
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_imp)
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_imp)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def _revolute_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Revolute-mode PGS iterate.

    3+2 Schur-complement positional solve plus the scalar angular
    drive + limit rows. ``use_bias=False`` zeroes the anchor-1 and
    anchor-2 drift biases for the Box2D v3 TGS-soft relax pass; the
    axial drive / limit row is unaffected (its bias is a velocity /
    limit target, not drift).
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
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)
    acc2_tan = wp.vec2f(acc2_t1, acc2_t2)

    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
    jv2_t1 = wp.dot(t1, jv2_world)
    jv2_t2 = wp.dot(t2, jv2_world)
    jv2 = wp.vec2f(jv2_t1, jv2_t2)
    bias2_tan = wp.vec2f(bias2[0], bias2[1])

    rhs1 = jv1 + bias1
    rhs2 = jv2 + bias2_tan

    ut_ai_rhs1_3 = ut_ai @ rhs1
    ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

    s_inv_22 = wp.mat22f(
        s_inv_packed[0, 0],
        s_inv_packed[0, 1],
        s_inv_packed[1, 0],
        s_inv_packed[1, 1],
    )
    lam2_us = -(s_inv_22 @ (rhs2 - ut_ai_rhs1))
    lam2 = mass_coeff * lam2_us - impulse_coeff * acc2_tan

    lam2_world = lam2[0] * t1 + lam2[1] * t2
    lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

    u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
    u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
    u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

    lam1_us = -(a1_inv @ (rhs1 + u_lam2_us))
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1

    total_lin = lam1 + lam2_world

    velocity1 = velocity1 - inv_mass1 * total_lin
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1 + cr2_b1 @ lam2_world)
    velocity2 = velocity2 + inv_mass2 * total_lin
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1 + cr2_b2 @ lam2_world)

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    # ---- Axial drive + limit scalar PGS rows -------------------------
    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)

    # Angular rate: jv_axial = n . (w1 - w2). Sign convention matches
    # the warm-start below: ``+n_hat`` for body 1, ``-n_hat`` for body
    # 2, so positive lambda spins body 1 *forward* and body 2 *back*.
    jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt)
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_lam)
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_lam)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


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


@wp.func
def _prismatic_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Prismatic-mode prepare pass.

    Computes lever arms for all three anchors, rebuilds the tangent
    basis ``(t1, t2)`` from the current slide axis ``n_hat``, assembles
    the 4x4 + 1 Schur block, and caches ``a4_inv / c / s_scalar_inv`` for
    the iterate. Also computes the scalar drive / limit row for the
    linear slide along ``n_hat``.
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

    # Slide axis: world direction from body 2's anchor 1 to body 2's
    # anchor 2 (rides body 2's rotation, same convention as revolute).
    axis_vec = p2_b2 - p1_b2
    axis_len2 = wp.dot(axis_vec, axis_vec)
    if axis_len2 > 1.0e-20:
        n_hat = axis_vec / wp.sqrt(axis_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # Tangent basis aligned with the anchor-1 -> anchor-3 lever arm
    # (projected perpendicular to ``n_hat``). See
    # :func:`_tangent_basis_from_anchor3` for why this specific choice
    # is necessary for Gauss-Seidel convergence in shared-body chains.
    t1, t2 = _tangent_basis_from_anchor3(n_hat, r1_b1, r3_b1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)

    # ---- Build the 3x3 anchor-to-anchor coupling matrices -----------
    # For anchors i, j, the cross-coupling 3x3 is
    #     B_{i,j} = (1/m1 + 1/m2) I + skew(ri_b1) I1^-1 skew(rj_b1)^T
    #                                 + skew(ri_b2) I2^-1 skew(rj_b2)^T
    # The diagonal blocks B_{i,i} are what ball-socket calls A;
    # off-diagonal blocks use mixed lever arms.
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    cr2_b1 = wp.skew(r2_b1)
    cr2_b2 = wp.skew(r2_b2)
    cr3_b1 = wp.skew(r3_b1)
    cr3_b2 = wp.skew(r3_b2)

    eye3 = wp.identity(3, dtype=wp.float32)
    m_diag = (inv_mass1 + inv_mass2) * eye3

    b11 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))
    b22 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))
    b33 = m_diag + cr3_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr3_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))
    b12 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))
    b13 = m_diag + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))
    b23 = m_diag + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1)) + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))

    # ---- Project to tangent basis ------------------------------------
    # T_2x3 = [t1; t2]; each 2x2 block for (anchor i, anchor j) is
    # [[t1.B_ij.t1, t1.B_ij.t2], [t2.B_ij.t1, t2.B_ij.t2]].
    b11_t1 = b11 @ t1
    b11_t2 = b11 @ t2
    b22_t1 = b22 @ t1
    b22_t2 = b22 @ t2
    b12_t1 = b12 @ t1
    b12_t2 = b12 @ t2
    # b21 = b12^T, computed on-the-fly where needed.
    b13_t2 = b13 @ t2  # c-vector is [a1,a3] and [a2,a3] projected onto t2.
    b23_t2 = b23 @ t2

    # 4x4 K4: rows/cols = (a1 t1, a1 t2, a2 t1, a2 t2).
    # K4[i,j] in the 2x2 diagonal block is anchor i tangent-tangent; in
    # the off-diagonal block (a1, a2) it's t_i . B12 . t_j (and its
    # transpose for (a2, a1)).
    k4_00 = wp.dot(t1, b11_t1)
    k4_01 = wp.dot(t1, b11_t2)
    k4_11 = wp.dot(t2, b11_t2)
    k4_02 = wp.dot(t1, b12_t1)  # t1 . B12 . t1
    k4_03 = wp.dot(t1, b12_t2)  # t1 . B12 . t2
    k4_12 = wp.dot(t2, b12_t1)  # t2 . B12 . t1
    k4_13 = wp.dot(t2, b12_t2)  # t2 . B12 . t2
    k4_22 = wp.dot(t1, b22_t1)
    k4_23 = wp.dot(t1, b22_t2)
    k4_33 = wp.dot(t2, b22_t2)

    # K4 is symmetric (K_ij = K_ji because B_ii is symmetric and the
    # (a2,a1) block equals the transpose of the (a1,a2) block).
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

    # c-vector: coupling of the 4 tangent rows with the a3 scalar row
    # (which is along t2 at anchor 3).
    c0 = wp.dot(t1, b13_t2)  # row (a1, t1) with (a3, t2).
    c1 = wp.dot(t2, b13_t2)
    c2 = wp.dot(t1, b23_t2)
    c3 = wp.dot(t2, b23_t2)
    c = wp.vec4f(c0, c1, c2, c3)

    # d scalar: t2 . B33 . t2.
    b33_t2 = b33 @ t2
    d_scalar = wp.dot(t2, b33_t2)

    # ---- Schur --------------------------------------------------------
    a4_inv = wp.inverse(k4)
    a4_inv_c = a4_inv @ c
    s_scalar = d_scalar - wp.dot(c, a4_inv_c)
    if wp.abs(s_scalar) > 1.0e-20:
        s_scalar_inv = 1.0 / s_scalar
    else:
        s_scalar_inv = 0.0

    write_mat44(constraints, base_offset + _OFF_A4_INV, cid, a4_inv)
    write_vec4(constraints, base_offset + _OFF_C_PRIS, cid, c)
    write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Positional bias: tangent drift at each anchor, scalar drift at
    # anchor 3 along t2.
    drift1 = p1_b2 - p1_b1
    drift2 = p2_b2 - p2_b1
    drift3 = p3_b2 - p3_b1
    bias1 = wp.vec3f(wp.dot(t1, drift1) * bias_rate, wp.dot(t2, drift1) * bias_rate, 0.0)
    bias2 = wp.vec3f(wp.dot(t1, drift2) * bias_rate, wp.dot(t2, drift2) * bias_rate, 0.0)
    bias3 = wp.dot(t2, drift3) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)
    write_float(constraints, base_offset + _OFF_BIAS3, cid, bias3)

    # ---- Warm-start the positional impulses --------------------------
    # acc_imp1 / acc_imp2 are world-frame tangent impulses; re-project
    # them onto the current (t1, t2) basis to handle axis drift across
    # substeps. acc_imp3 is a world-frame vector along the cached t2;
    # we re-project it onto the current t2.
    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)

    # Re-project and rebuild world vectors on the new tangent basis.
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

    # ---- Axial drive + limit block (linear) --------------------------
    # Linear inverse effective mass at anchor 1 along n_hat:
    #   m_inv = n . B11 . n  (same quadratic form as the tangent block,
    #                         but projected onto n_hat instead of t).
    # Joint armature (translational rotor inertia referred to the slide
    # axis) is baked into ``inverse_mass`` / ``inverse_inertia_world``
    # at solver construction, so this expression is already
    # armature-aware -- no extra term here.
    eff_inv = wp.dot(n_hat, b11 @ n_hat)

    # Slide along n_hat measured at anchor 1. Starts at 0 at init
    # (anchors coincident), so this is directly the relative slide
    # in meters. Sign convention: positive ``lam`` decreases the
    # slide (see :func:`_prismatic_iterate_at`).
    slide = wp.dot(n_hat, drift1)

    # Shared drive / limit prepare. Returns the warm-start axial
    # impulse (drive + limit acc; limit gated on clamp state).
    # Prismatic applies it as a linear impulse at anchor 1 (with the
    # corresponding lever-arm torque on both bodies), the only
    # per-mode difference from the revolute warm-start.
    axial_imp = _axial_drive_limit_prepare_at(constraints, cid, base_offset, slide, eff_inv, dt)
    velocity1 = velocity1 + inv_mass1 * (n_hat * axial_imp)
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ wp.cross(r1_b1, n_hat * axial_imp)
    velocity2 = velocity2 - inv_mass2 * (n_hat * axial_imp)
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ wp.cross(r1_b2, n_hat * axial_imp)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def _prismatic_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Prismatic-mode PGS iterate.

    4+1 Schur-complement positional solve (eliminate the scalar a3 row
    first, then the 4x4 tangent block) plus the scalar linear drive /
    limit row along ``n_hat``. ``use_bias=False`` zeroes the 5 lock
    biases (anchor1 xy tangent, anchor2 xy tangent, anchor3 scalar)
    for the Box2D v3 TGS-soft relax pass; the axial drive / limit row
    is unaffected (it's a velocity / limit target, not drift).
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
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc_imp1_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc_imp2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_imp3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc1_tan = wp.vec2f(wp.dot(t1, acc_imp1_world), wp.dot(t2, acc_imp1_world))
    acc2_tan = wp.vec2f(wp.dot(t1, acc_imp2_world), wp.dot(t2, acc_imp2_world))
    acc4 = wp.vec4f(acc1_tan[0], acc1_tan[1], acc2_tan[0], acc2_tan[1])
    acc3_scalar = wp.dot(t2, acc_imp3_world)

    # Velocity Jacobian rows:
    #   jv_i = d_i . ( v2 + w2 x r_i_b2 - v1 - w1 x r_i_b1 ).
    # For tangent rows at anchor k, d in {t1, t2}; for a3 scalar row,
    # d = t2 with anchor 3 levers.
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

    # Schur: eliminate the a3 scalar row first.
    a4_inv_rhs4 = a4_inv @ rhs4
    lam3_us = -s_scalar_inv * (rhs3 - wp.dot(c_pris, a4_inv_rhs4))
    lam3 = mass_coeff * lam3_us - impulse_coeff * acc3_scalar

    lam4_us = -(a4_inv @ (rhs4 + c_pris * lam3_us))
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

    # ---- Linear drive + limit scalar rows along n_hat ----------------
    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)

    # Linear relative rate at anchor 1 along n_hat:
    #   jv_axial = n . ((v1 + w1 x r1_b1) - (v2 + w2 x r1_b2)).
    # Sign structure matches the warm-start (`+v1 * n * lam`
    # / `-v2 * n * lam`): positive lam decreases slide.
    v1_anchor = velocity1 + wp.cross(angular_velocity1, r1_b1)
    v2_anchor = velocity2 + wp.cross(angular_velocity2, r1_b2)
    jv_axial = wp.dot(n_hat, v1_anchor - v2_anchor)
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt)

    # Apply the combined linear impulse: lam along n_hat, with body 1
    # getting +n_hat and body 2 getting -n_hat (mirror of revolute's
    # angular convention).
    axial_imp = n_hat * axial_lam
    velocity1 = velocity1 + inv_mass1 * axial_imp
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ wp.cross(r1_b1, axial_imp)
    velocity2 = velocity2 - inv_mass2 * axial_imp
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ wp.cross(r1_b2, axial_imp)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


# ---------------------------------------------------------------------------
# Cable (soft ball-socket) mode
# ---------------------------------------------------------------------------
#
# Rigid 3-row point lock at anchor1 + three independent soft angular
# rows: 2x "bend" perpendicular to ``n_hat = anchor1 -> anchor2`` and
# 1x "twist" along ``n_hat``. Each scalar row reuses
# :func:`pd_coefficients` (substep Nyquist clamping, warm-start,
# bias-off relax pass for free) and iterates Gauss-Seidel-style, same
# as :func:`_axial_drive_limit_iterate`.
#
# Deflection is measured as the Darboux vector, i.e. the log-map of
# ``q_align = q_wc * inv_init * q_wp^{-1}`` with
# ``inv_init = q_wc_rest^{-1} * q_wp_rest`` (same quantity REVOLUTE
# snapshots in ``_OFF_INV_INITIAL_ORIENTATION``). Under small
# deflection the log-map collapses to ``2 * q_align.xyz``; angular
# Jacobian rows are world-frame unit directions in the reference-
# axis basis, keeping the iterate cheap.


@wp.func
def _cable_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Cable-mode prepare pass: anchor-1 3-row point lock
    (ball-socket) plus three soft angular rows (bend x2, twist x1).

    Caches per-row world-frame Jacobian directions in ``ut_ai`` and
    per-row soft-PD coefficients ``(gamma, bias, eff_mass_soft)`` in
    ``s_inv``. The anchor-1 positional block uses the same
    ``a1_inv`` cache ball-socket uses.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]
    inv_inertia_sum = inv_inertia1 + inv_inertia2

    # ---- Anchor-1 positional block (ball-socket) ---------------------
    (
        _r1_b1,
        _r1_b2,
        _cr1_b1,
        _cr1_b2,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    ) = _anchor1_positional_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    dt = 1.0 / idt

    # ---- Angular block: Darboux vector in world frame ----------------
    # ``q_align = q_wc * inv_init * q_wp^{-1}`` -- same construction
    # the REVOLUTE twist tracker uses (diff quaternion is identity at
    # finalize). Map the quaternion to its rotation vector via the
    # full log-map ``kappa = axis * angle`` with
    # ``angle = 2 * atan2(|xyz|, w)`` and ``axis = xyz / |xyz|``. The
    # cheaper ``2 * q.xyz`` small-angle approximation underestimates
    # the bend by ~6 % at 80 deg deflection, which lets the spring
    # force lag the actual deflection on large bends.
    inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
    q_align = orient2 * inv_init * wp.quat_inverse(orient1)
    # Canonicalise to the shortest-path hemisphere so the log-map
    # picks the rotation in :math:`[-\\pi, +\\pi]` around the axis.
    if q_align[3] < 0.0:
        q_align = -q_align
    qa_xyz = wp.vec3f(q_align[0], q_align[1], q_align[2])
    qa_xyz_len = wp.length(qa_xyz)
    if qa_xyz_len > 1.0e-9:
        # Full log-map. ``angle / |xyz|`` is the conversion from the
        # quaternion's xyz (= sin(theta/2) * axis) to the rotation
        # vector (= theta * axis).
        angle = 2.0 * wp.atan2(qa_xyz_len, q_align[3])
        kappa_world = qa_xyz * (angle / qa_xyz_len)
    else:
        # Small-angle: angle ~ 2 * |xyz|, axis ~ xyz / |xyz|; combined,
        # 2 * xyz cancels the |xyz| division.
        kappa_world = qa_xyz * 2.0

    # ---- Reference axis + bend basis in world frame ------------------
    axis_local1 = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
    n_hat_world = wp.quat_rotate(orient1, axis_local1)
    # Normalize defensively -- should already be unit from init.
    n_len2 = wp.dot(n_hat_world, n_hat_world)
    if n_len2 > 1.0e-20:
        n_hat_world = n_hat_world / wp.sqrt(n_len2)
    else:
        n_hat_world = wp.vec3f(1.0, 0.0, 0.0)

    # Two unit vectors perpendicular to n_hat_world. ``create_orthonormal``
    # is deterministic in n_hat, so the basis is stable across substeps
    # as long as the parent body's world rotation evolves smoothly.
    e1_world = create_orthonormal(n_hat_world)
    # Project out any drift along n_hat, then renormalise (cheap).
    e1_world = e1_world - wp.dot(e1_world, n_hat_world) * n_hat_world
    e1_len2 = wp.dot(e1_world, e1_world)
    if e1_len2 > 1.0e-20:
        e1_world = e1_world / wp.sqrt(e1_len2)
    else:
        e1_world = create_orthonormal(n_hat_world)
    e2_world = wp.cross(n_hat_world, e1_world)

    # ---- Per-row stiffness / damping setup ---------------------------
    # Bend params live in the drive slot (currently unused in cable
    # mode); twist params live in the limit slot. Keeping the two
    # pairs distinct lets users set a stiff rope (high twist, low
    # bend) or the opposite.
    k_bend = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    d_bend = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    k_twist = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    d_twist = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)

    # Per-row scalar PD plumbing. Each row i is a scalar constraint
    # C_i(q) = kappa . d_i with Jacobian J_i_world = d_i_world; the
    # unsoftened inverse effective mass is
    # ``d_i . (I1^{-1} + I2^{-1}) . d_i``. ``pd_coefficients`` folds
    # the stiffness / damping / substep into
    # ``(gamma, bias, eff_mass_soft)`` using the same Box2D idiom
    # :func:`_axial_drive_limit_iterate` uses for the drive row.
    kappa_b1 = wp.dot(kappa_world, e1_world)
    kappa_b2 = wp.dot(kappa_world, e2_world)
    kappa_t = wp.dot(kappa_world, n_hat_world)

    eff_inv_b1 = wp.dot(e1_world, inv_inertia_sum @ e1_world)
    eff_inv_b2 = wp.dot(e2_world, inv_inertia_sum @ e2_world)
    eff_inv_t = wp.dot(n_hat_world, inv_inertia_sum @ n_hat_world)

    # Spring / damping split: the spring triple drives the main solve
    # (use_bias = True), and ``damp_mass`` drives the relax pass
    # (use_bias = False) as a pure velocity damping impulse. Decouples
    # the two physical effects; convergence at high ``c`` no longer
    # collapses to a stiff velocity lock the way the combined
    # formulation did. See :func:`pd_coefficients_split`.
    gamma_b1, bias_b1, eff_soft_b1, damp_mass_b1 = pd_coefficients_split(
        k_bend, d_bend, kappa_b1, eff_inv_b1, dt
    )
    gamma_b2, bias_b2, eff_soft_b2, damp_mass_b2 = pd_coefficients_split(
        k_bend, d_bend, kappa_b2, eff_inv_b2, dt
    )
    gamma_t, bias_t, eff_soft_t, damp_mass_t = pd_coefficients_split(
        k_twist, d_twist, kappa_t, eff_inv_t, dt
    )

    # ---- Cache row directions + coefficients -------------------------
    # ``ut_ai`` (mat33) holds the three world-frame row directions,
    # one per row (e1, e2, n_hat).  ``s_inv`` (mat33) holds the three
    # ``(gamma, bias, eff_mass_soft)`` triples.  Neither slot is used
    # by any other joint mode, so stamping them here doesn't collide
    # with REVOLUTE / PRISMATIC / FIXED state.
    dirs_mat = wp.mat33f(
        e1_world[0],
        e1_world[1],
        e1_world[2],
        e2_world[0],
        e2_world[1],
        e2_world[2],
        n_hat_world[0],
        n_hat_world[1],
        n_hat_world[2],
    )
    coeffs_mat = wp.mat33f(
        gamma_b1,
        bias_b1,
        eff_soft_b1,
        gamma_b2,
        bias_b2,
        eff_soft_b2,
        gamma_t,
        bias_t,
        eff_soft_t,
    )
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, dirs_mat)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, coeffs_mat)
    # Per-row damping effective masses (used by the relax pass).
    write_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_BEND1, cid, damp_mass_b1)
    write_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_BEND2, cid, damp_mass_b2)
    write_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_TWIST, cid, damp_mass_t)

    # Reset the cross-substep angular warm-start. ``acc_kappa`` carries
    # the previous substep's accumulated bend / twist impulses in the
    # *old* (e1, e2, n_hat) basis; the basis is recomputed every
    # substep from the parent body's current orientation, so applying
    # those scalar impulses along the new basis directions is only
    # correct when the parent doesn't rotate. The within-substep
    # warm-start (iterate's acc accumulation across the
    # ``solver_iterations`` sweeps) is unaffected.
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, wp.vec3f(0.0, 0.0, 0.0))

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def _cable_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Cable-mode PGS iterate: ball-socket anchor-1 solve + three
    soft angular rows. ``use_bias=False`` zeros the anchor-1 drift
    bias and the three per-row biases for the relax pass; the rows'
    ``gamma * acc`` softness term stays on in both passes (Box2D
    TGS-soft convention)."""
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

    # ---- Anchor-1 3-row positional block -----------------------------
    r1_b1 = read_vec3(constraints, base_offset + _OFF_R1_B1, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    cr1_b1 = wp.skew(r1_b1)
    cr1_b2 = wp.skew(r1_b2)
    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    lam1_us = -(a1_inv @ (jv1 + bias1))
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1

    velocity1 = velocity1 - inv_mass1 * lam1
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1)
    velocity2 = velocity2 + inv_mass2 * lam1
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)

    # ---- 3 soft angular rows -----------------------------------------
    dirs_mat = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)

    d_bend1 = wp.vec3f(dirs_mat[0, 0], dirs_mat[0, 1], dirs_mat[0, 2])
    d_bend2 = wp.vec3f(dirs_mat[1, 0], dirs_mat[1, 1], dirs_mat[1, 2])
    d_twist = wp.vec3f(dirs_mat[2, 0], dirs_mat[2, 1], dirs_mat[2, 2])

    if use_bias:
        # Main solve: spring-only soft constraint per row. Damping
        # is split out into the relax pass -- see
        # :func:`pd_coefficients_split` for the rationale.
        coeffs_mat = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
        acc_kappa = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
        gamma_b1 = coeffs_mat[0, 0]
        bias_b1 = coeffs_mat[0, 1]
        eff_soft_b1 = coeffs_mat[0, 2]
        gamma_b2 = coeffs_mat[1, 0]
        bias_b2 = coeffs_mat[1, 1]
        eff_soft_b2 = coeffs_mat[1, 2]
        gamma_t = coeffs_mat[2, 0]
        bias_t = coeffs_mat[2, 1]
        eff_soft_t = coeffs_mat[2, 2]

        # Row 1: bend about e1.
        # Sign convention matches :func:`_axial_drive_limit_iterate`:
        # ``jv = d . (w1 - w2)`` and positive ``lam`` spins body 1
        # faster about ``d`` -- i.e. restores the child toward the
        # rest pose when ``kappa > 0``.
        jv_b1 = wp.dot(d_bend1, angular_velocity1 - angular_velocity2)
        lam_b1 = -eff_soft_b1 * (jv_b1 - bias_b1 + gamma_b1 * acc_kappa[0])
        torque_b1 = d_bend1 * lam_b1
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_b1
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_b1

        # Row 2: bend about e2 (use updated body velocities -- in-row GS).
        jv_b2 = wp.dot(d_bend2, angular_velocity1 - angular_velocity2)
        lam_b2 = -eff_soft_b2 * (jv_b2 - bias_b2 + gamma_b2 * acc_kappa[1])
        torque_b2 = d_bend2 * lam_b2
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_b2
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_b2

        # Row 3: twist about n_hat.
        jv_t = wp.dot(d_twist, angular_velocity1 - angular_velocity2)
        lam_t = -eff_soft_t * (jv_t - bias_t + gamma_t * acc_kappa[2])
        torque_t = d_twist * lam_t
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_t
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_t

        new_acc = wp.vec3f(acc_kappa[0] + lam_b1, acc_kappa[1] + lam_b2, acc_kappa[2] + lam_t)
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, new_acc)
    else:
        # Relax pass: pure velocity-only damping per row. The
        # implicit-Euler velocity update ``v_new = v / (1 + dt*c*M_inv)``
        # is applied as one impulse ``lam = -damp_mass * Jv`` per
        # row, which is exact for an isolated rigid body in a single
        # PGS sweep. No bias term -- the spring is decoupled and
        # already converged in the main solve.
        damp_mass_b1 = read_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_BEND1, cid)
        damp_mass_b2 = read_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_BEND2, cid)
        damp_mass_t = read_float(constraints, base_offset + _OFF_CABLE_DAMP_MASS_TWIST, cid)

        # Row 1: bend-e1 damping.
        if damp_mass_b1 > 0.0:
            jv_b1 = wp.dot(d_bend1, angular_velocity1 - angular_velocity2)
            lam_b1 = -damp_mass_b1 * jv_b1
            torque_b1 = d_bend1 * lam_b1
            angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_b1
            angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_b1

        # Row 2: bend-e2 damping.
        if damp_mass_b2 > 0.0:
            jv_b2 = wp.dot(d_bend2, angular_velocity1 - angular_velocity2)
            lam_b2 = -damp_mass_b2 * jv_b2
            torque_b2 = d_bend2 * lam_b2
            angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_b2
            angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_b2

        # Row 3: twist damping.
        if damp_mass_t > 0.0:
            jv_t = wp.dot(d_twist, angular_velocity1 - angular_velocity2)
            lam_t = -damp_mass_t * jv_t
            torque_t = d_twist * lam_t
            angular_velocity1 = angular_velocity1 + inv_inertia1 @ torque_t
            angular_velocity2 = angular_velocity2 - inv_inertia2 @ torque_t

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


# ---------------------------------------------------------------------------
# Beam (soft fixed) mode
# ---------------------------------------------------------------------------
#
# BEAM = anchor-1 3-row Box2D-soft point lock (``hertz``,
# ``damping_ratio``) + anchor-2 tangent 2-row PD spring-damper (bend
# rows, ``k_bend``, ``d_bend``) + anchor-3 scalar 1-row PD
# spring-damper (twist row, ``k_twist``, ``d_twist``). Block GS
# between the three blocks: each block is solved with its own
# pre-computed soft inverse, the outer PGS loop closes the cross-
# coupling.
#
# As ``k_bend -> infinity`` the anchor-2 PD rows lock with revolute
# convergence quality (Nyquist-clamped soft -> rigid). As
# ``k_twist -> infinity`` on top of that the anchor-3 row locks too,
# yielding a fixed weld. User-facing gains are in rotational SI units
# (N*m/rad, N*m*s/rad); the implementation rescales by
# ``1 / rest_length^2`` to obtain the equivalent positional spring at
# the lever-armed anchors. Lever-arm amplification gives the correct
# rotational stiffness without needing an angular Jacobian / log-map
# (cf. CABLE), and the well-conditioned positional rows match
# REVOLUTE / PRISMATIC convergence behaviour.

#: Headroom factor on the BEAM substep stiffness clamp. The implicit-
#: Euler PD's "naive" Nyquist bound is ``k <= 1 / (M_inv * dt^2)``
#: (i.e. ``omega_n * dt <= 1``); past that point the soft-PD's
#: effective mass collapses to ``1 / M_inv`` and the bias spikes to
#: ``C / dt``, so user-supplied gains saturate rather than producing
#: a stiffer spring. This factor multiplies the cap to ``N / (M_inv
#: * dt^2)``, allowing super-Nyquist gains for users who want them.
#: ``N = 1`` matches the strict naive bound; ``N = pi^2 ~ 9.87``
#: matches the cable / Box2D ``omega <= pi/dt`` convention; larger
#: values trade more headroom for more risk of high-frequency
#: ringing on undamped chains. Applied uniformly to both the bend
#: (anchor-2 tangent) and twist (anchor-3 scalar) PD blocks.
_BEAM_NYQUIST_HEADROOM = wp.constant(wp.float32(10.0))


@wp.func
def _beam_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Beam-mode prepare pass.

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
    # Cap is ``N / (M_inv * dt^2)`` with the headroom factor
    # ``_BEAM_NYQUIST_HEADROOM`` -- ``N = 1`` is the strict bound,
    # larger values let user gains push past the substep's resolvable
    # spring frequency at the cost of high-frequency ringing risk.
    eff_inv_bend = wp.float32(0.5) * (k22_00 + k22_11)
    bias_factor_bend = wp.float32(0.0)
    gamma_bend = wp.float32(0.0)
    if (k_pos_bend > wp.float32(0.0)) or (d_pos_bend > wp.float32(0.0)):
        if eff_inv_bend > wp.float32(0.0):
            k_max_bend = _BEAM_NYQUIST_HEADROOM / (eff_inv_bend * dt * dt)
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
    write_float(constraints, base_offset + _OFF_BEAM_K22_INV_00, cid, k22s_11 * inv_det_b)
    write_float(constraints, base_offset + _OFF_BEAM_K22_INV_01, cid, -k22s_01 * inv_det_b)
    write_float(constraints, base_offset + _OFF_BEAM_K22_INV_10, cid, -k22s_10 * inv_det_b)
    write_float(constraints, base_offset + _OFF_BEAM_K22_INV_11, cid, k22s_00 * inv_det_b)
    write_float(constraints, base_offset + _OFF_BEAM_GAMMA_BEND, cid, gamma_bend)

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

    bias_factor_twist = wp.float32(0.0)
    gamma_twist = wp.float32(0.0)
    m_twist_soft = wp.float32(0.0)
    if (k_pos_twist > wp.float32(0.0)) or (d_pos_twist > wp.float32(0.0)):
        if eff_inv_twist > wp.float32(0.0):
            # Same headroom factor as the bend clamp; see
            # ``_BEAM_NYQUIST_HEADROOM`` for the rationale.
            k_max_twist = _BEAM_NYQUIST_HEADROOM / (eff_inv_twist * dt * dt)
            k_clamped_twist = wp.min(k_pos_twist, k_max_twist)
        else:
            k_clamped_twist = k_pos_twist
        denom_twist = d_pos_twist + dt * k_clamped_twist
        if denom_twist > wp.float32(0.0):
            softness_twist = wp.float32(1.0) / denom_twist
            bias_factor_twist = dt * k_clamped_twist * softness_twist
            gamma_twist = softness_twist * idt
            m_twist_soft = wp.float32(1.0) / (eff_inv_twist + gamma_twist)
    write_float(constraints, base_offset + _OFF_BEAM_M_TWIST_SOFT, cid, m_twist_soft)
    write_float(constraints, base_offset + _OFF_BEAM_GAMMA_TWIST, cid, gamma_twist)

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
    velocity1 = bodies.velocity[b1] - inv_mass1 * total_linear
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (
        cr1_b1 @ acc_imp1 + cr2_b1 @ acc_imp2_world + cr3_b1 @ acc_imp3_world
    )
    velocity2 = bodies.velocity[b2] + inv_mass2 * total_linear
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (
        cr1_b2 @ acc_imp1 + cr2_b2 @ acc_imp2_world + cr3_b2 @ acc_imp3_world
    )

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2

    # Zero unused axial drive / limit state so wrench helpers and any
    # cross-mode reads see a clean column.
    write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, 0.0)
    write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, 0.0)


@wp.func
def _beam_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Beam-mode PGS iterate.

    Three independent block solves (block Gauss-Seidel within a sweep):
    anchor-1 3-row Box2D-soft point lock, anchor-2 tangent 2-row
    PD-soft (bend), anchor-3 scalar 1-row PD-soft (twist). The PD
    blocks use ``lambda = -M_soft * (Jv + bias + gamma * acc)``.

    ``use_bias`` only gates the *anchor-1* drift bias, matching the
    Box2D v3 TGS-soft relax-pass convention for hard positional
    locks. The anchor-2 / anchor-3 PD biases are NOT gated: those
    biases encode the spring force ``k * theta`` (not a drift
    correction), so zeroing them on the relax pass would cancel the
    spring entirely (the relax iterate would drive
    ``acc -> -Jv/gamma -> 0`` with the bias gone, and at convergence
    ``gamma * acc`` exactly cancels the next main-pass bias). Same
    rule the standalone PD axial-drive iterate follows -- its
    ``bias_drive`` is unconditional in
    :func:`_axial_drive_limit_iterate`.
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

    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    # ---- Block 1: anchor-1 3-row Box2D-soft -------------------------
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)

    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    rhs1 = jv1 + bias1
    lam1_us = -(a1_inv @ rhs1)
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1
    velocity1 = velocity1 - inv_mass1 * lam1
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1)
    velocity2 = velocity2 + inv_mass2 * lam1
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)

    # ---- Block 2: anchor-2 tangent 2-row PD-soft (bend) -------------
    k22_inv_00 = read_float(constraints, base_offset + _OFF_BEAM_K22_INV_00, cid)
    k22_inv_01 = read_float(constraints, base_offset + _OFF_BEAM_K22_INV_01, cid)
    k22_inv_10 = read_float(constraints, base_offset + _OFF_BEAM_K22_INV_10, cid)
    k22_inv_11 = read_float(constraints, base_offset + _OFF_BEAM_K22_INV_11, cid)
    gamma_bend = read_float(constraints, base_offset + _OFF_BEAM_GAMMA_BEND, cid)
    # NB: PD spring bias is unconditional (see docstring); only the
    # anchor-1 drift bias is gated by ``use_bias``.
    bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)

    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)

    jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
    jv2_t1 = wp.dot(t1, jv2_world)
    jv2_t2 = wp.dot(t2, jv2_world)

    rhs2_t1 = jv2_t1 + bias2[0] + gamma_bend * acc2_t1
    rhs2_t2 = jv2_t2 + bias2[1] + gamma_bend * acc2_t2

    lam2_t1 = -(k22_inv_00 * rhs2_t1 + k22_inv_01 * rhs2_t2)
    lam2_t2 = -(k22_inv_10 * rhs2_t1 + k22_inv_11 * rhs2_t2)
    lam2_world = lam2_t1 * t1 + lam2_t2 * t2

    velocity1 = velocity1 - inv_mass1 * lam2_world
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr2_b1 @ lam2_world)
    velocity2 = velocity2 + inv_mass2 * lam2_world
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr2_b2 @ lam2_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    # ---- Block 3: anchor-3 scalar 1-row PD-soft (twist) -------------
    m_twist_soft = read_float(constraints, base_offset + _OFF_BEAM_M_TWIST_SOFT, cid)
    gamma_twist = read_float(constraints, base_offset + _OFF_BEAM_GAMMA_TWIST, cid)
    # NB: PD spring bias is unconditional (see docstring).
    bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)

    acc3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc3_t2 = wp.dot(t2, acc3_world)

    jv3_world = -velocity1 + cr3_b1 @ angular_velocity1 + velocity2 - cr3_b2 @ angular_velocity2
    jv3_t2 = wp.dot(t2, jv3_world)

    lam3 = -m_twist_soft * (jv3_t2 + bias3 + gamma_twist * acc3_t2)
    lam3_world = lam3 * t2

    velocity1 = velocity1 - inv_mass1 * lam3_world
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr3_b1 @ lam3_world)
    velocity2 = velocity2 + inv_mass2 * lam3_world
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr3_b2 @ lam3_world)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3_world + lam3_world)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


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


@wp.func
def _fixed_prepare_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Fixed-mode prepare pass: REVOLUTE anchor-1+2 Schur plus a
    standalone anchor-3 scalar row along ``t2``."""
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

    # n_hat from anchor-1 -> anchor-2 on body 2 (rides body 2's
    # rotation; same convention as REVOLUTE / PRISMATIC).
    hinge_vec = p2_b2 - p1_b2
    hinge_len2 = wp.dot(hinge_vec, hinge_vec)
    if hinge_len2 > 1.0e-20:
        n_hat = hinge_vec / wp.sqrt(hinge_len2)
    else:
        n_hat = wp.vec3f(1.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # PRISMATIC's tangent-basis choice (shared helper):
    # :func:`_tangent_basis_from_anchor3`. Makes the anchor-3 scalar
    # row a unit-gain gate for rotation about ``n_hat``.
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

    # Anchor-1 3x3 block (full point lock).
    a1 = inv_mass1 * eye3
    a1 = a1 + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr1_b1))
    a1 = a1 + inv_mass2 * eye3
    a1 = a1 + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr1_b2))

    # Anchor-2 3x3 coupling (REVOLUTE convention).
    a2 = inv_mass1 * eye3
    a2 = a2 + cr2_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    a2 = a2 + inv_mass2 * eye3
    a2 = a2 + cr2_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    b_mat = (inv_mass1 + inv_mass2) * eye3
    b_mat = b_mat + cr1_b1 @ (inv_inertia1 @ wp.transpose(cr2_b1))
    b_mat = b_mat + cr1_b2 @ (inv_inertia2 @ wp.transpose(cr2_b2))

    t_mat = wp.mat33f(
        t1[0],
        t2[0],
        0.0,
        t1[1],
        t2[1],
        0.0,
        t1[2],
        t2[2],
        0.0,
    )
    tt = wp.transpose(t_mat)

    u_mat = b_mat @ t_mat
    d_mat = tt @ (a2 @ t_mat)

    a1_inv = wp.inverse(a1)
    ut_ai = wp.transpose(u_mat) @ a1_inv
    s_mat = d_mat - ut_ai @ u_mat

    s22 = wp.mat22f(
        s_mat[0, 0],
        s_mat[0, 1],
        s_mat[1, 0],
        s_mat[1, 1],
    )
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

    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, ut_ai)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv_packed)

    # Anchor-3 standalone scalar: d = t2 . B33 . t2. Block Gauss-Seidel
    # treats anchor-3 as decoupled from anchors 1+2 within a PGS step;
    # the outer PGS loop closes the cross-coupling.
    b33 = (inv_mass1 + inv_mass2) * eye3
    b33 = b33 + cr3_b1 @ (inv_inertia1 @ wp.transpose(cr3_b1))
    b33 = b33 + cr3_b2 @ (inv_inertia2 @ wp.transpose(cr3_b2))
    d_scalar = wp.dot(t2, b33 @ t2)
    if wp.abs(d_scalar) > 1.0e-20:
        s_scalar_inv = 1.0 / d_scalar
    else:
        s_scalar_inv = 0.0
    write_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid, s_scalar_inv)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)

    # Biases: anchor-1 is a 3-vec point-lock drift; anchor-2 is a
    # 2-component tangent drift projected onto (t1, t2); anchor-3 is
    # a scalar drift along t2.
    bias1 = (p1_b2 - p1_b1) * bias_rate
    drift2 = p2_b2 - p2_b1
    bias2 = wp.vec3f(
        wp.dot(t1, drift2) * bias_rate,
        wp.dot(t2, drift2) * bias_rate,
        0.0,
    )
    drift3 = p3_b2 - p3_b1
    bias3 = wp.dot(t2, drift3) * bias_rate
    write_vec3(constraints, base_offset + _OFF_BIAS1, cid, bias1)
    write_vec3(constraints, base_offset + _OFF_BIAS2, cid, bias2)
    write_float(constraints, base_offset + _OFF_BIAS3, cid, bias3)

    # Positional warm-start. acc_imp1 is world-frame 3-vec (anchor 1);
    # acc_imp2 is the anchor-2 tangent impulse stored as a world-frame
    # 3-vec (re-project onto current (t1, t2) as REVOLUTE does); acc_imp3
    # is a world-frame vector along the cached t2 (re-project like
    # PRISMATIC).
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
    velocity1 = bodies.velocity[b1] - inv_mass1 * total_linear
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (
        cr1_b1 @ acc_imp1 + cr2_b1 @ acc_imp2_world + cr3_b1 @ acc_imp3_world
    )
    velocity2 = bodies.velocity[b2] + inv_mass2 * total_linear
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (
        cr1_b2 @ acc_imp1 + cr2_b2 @ acc_imp2_world + cr3_b2 @ acc_imp3_world
    )

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2

    # Zero the axial drive / limit state so world_wrench / iterate don't
    # pick up stale values from a previous mode assignment.
    write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, 0.0)
    write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, 0.0)
    write_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, 0.0)


@wp.func
def _fixed_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Fixed-mode PGS iterate: REVOLUTE anchor-1+2 3+2 Schur, then
    anchor-3 scalar row along ``t2`` (block Gauss-Seidel)."""
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

    a1_inv = read_mat33(constraints, base_offset + _OFF_A1_INV, cid)
    ut_ai = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)
    s_inv_packed = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
    s_scalar_inv = read_float(constraints, base_offset + _OFF_S_SCALAR_INV, cid)
    if use_bias:
        bias1 = read_vec3(constraints, base_offset + _OFF_BIAS1, cid)
        bias2 = read_vec3(constraints, base_offset + _OFF_BIAS2, cid)
        bias3 = read_float(constraints, base_offset + _OFF_BIAS3, cid)
    else:
        bias1 = wp.vec3f(0.0, 0.0, 0.0)
        bias2 = wp.vec3f(0.0, 0.0, 0.0)
        bias3 = wp.float32(0.0)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)

    # ---- Block 1: anchor-1 3-row + anchor-2 tangent 2-row ----------
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc2_t1 = wp.dot(t1, acc2_world)
    acc2_t2 = wp.dot(t2, acc2_world)
    acc2_tan = wp.vec2f(acc2_t1, acc2_t2)

    jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
    jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
    jv2 = wp.vec2f(wp.dot(t1, jv2_world), wp.dot(t2, jv2_world))
    bias2_tan = wp.vec2f(bias2[0], bias2[1])

    rhs1 = jv1 + bias1
    rhs2 = jv2 + bias2_tan

    ut_ai_rhs1_3 = ut_ai @ rhs1
    ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

    s_inv_22 = wp.mat22f(
        s_inv_packed[0, 0],
        s_inv_packed[0, 1],
        s_inv_packed[1, 0],
        s_inv_packed[1, 1],
    )
    lam2_us = -(s_inv_22 @ (rhs2 - ut_ai_rhs1))
    lam2 = mass_coeff * lam2_us - impulse_coeff * acc2_tan

    lam2_world = lam2[0] * t1 + lam2[1] * t2
    lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

    u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
    u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
    u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

    lam1_us = -(a1_inv @ (rhs1 + u_lam2_us))
    lam1 = mass_coeff * lam1_us - impulse_coeff * acc1

    total_lin = lam1 + lam2_world

    velocity1 = velocity1 - inv_mass1 * total_lin
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr1_b1 @ lam1 + cr2_b1 @ lam2_world)
    velocity2 = velocity2 + inv_mass2 * total_lin
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr1_b2 @ lam1 + cr2_b2 @ lam2_world)

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1 + lam1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world + lam2_world)

    # ---- Block 2: anchor-3 scalar along t2 (standalone 1x1) --------
    acc3_world = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc3_scalar = wp.dot(t2, acc3_world)

    jv3_world = -velocity1 + cr3_b1 @ angular_velocity1 + velocity2 - cr3_b2 @ angular_velocity2
    jv3 = wp.dot(t2, jv3_world)

    lam3_us = -(s_scalar_inv * (jv3 + bias3))
    lam3 = mass_coeff * lam3_us - impulse_coeff * acc3_scalar

    lam3_world = lam3 * t2

    velocity1 = velocity1 - inv_mass1 * lam3_world
    angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cr3_b1 @ lam3_world)
    velocity2 = velocity2 + inv_mass2 * lam3_world
    angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cr3_b2 @ lam3_world)

    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, acc3_world + lam3_world)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


# ---------------------------------------------------------------------------
# Mode-dispatching entry points
# ---------------------------------------------------------------------------


@wp.func
def actuated_double_ball_socket_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass that dispatches on ``joint_mode``.

    Reads the per-constraint ``joint_mode`` tag and calls the
    ball-socket, revolute, or prismatic pre-iteration helper. See
    :func:`_ball_socket_prepare_at`, :func:`_revolute_prepare_at`,
    and :func:`_prismatic_prepare_at` for the per-mode math.
    """
    joint_mode = read_int(constraints, base_offset + _OFF_JOINT_MODE, cid)
    if joint_mode == JOINT_MODE_REVOLUTE:
        _revolute_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        _prismatic_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    elif joint_mode == JOINT_MODE_FIXED:
        _fixed_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    elif joint_mode == JOINT_MODE_CABLE:
        _cable_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    elif joint_mode == JOINT_MODE_BEAM:
        _beam_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)
    else:
        _ball_socket_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)


@wp.func
def _revolute_iterate_at_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
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
    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

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
    s_inv_packed = read_mat33(constraints, base_offset + _OFF_S_INV, cid)
    s_inv_22 = wp.mat22f(
        s_inv_packed[0, 0],
        s_inv_packed[0, 1],
        s_inv_packed[1, 0],
        s_inv_packed[1, 1],
    )
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
        # Positional PGS: anchor-1 (3 rows) + anchor-2 tangent (2 rows)
        acc2_t1 = wp.dot(t1, acc2_world)
        acc2_t2 = wp.dot(t2, acc2_world)
        acc2_tan = wp.vec2f(acc2_t1, acc2_t2)

        jv1 = -velocity1 + cr1_b1 @ angular_velocity1 + velocity2 - cr1_b2 @ angular_velocity2
        jv2_world = -velocity1 + cr2_b1 @ angular_velocity1 + velocity2 - cr2_b2 @ angular_velocity2
        jv2_t1 = wp.dot(t1, jv2_world)
        jv2_t2 = wp.dot(t2, jv2_world)
        jv2 = wp.vec2f(jv2_t1, jv2_t2)

        rhs1 = jv1 + bias1
        rhs2 = jv2 + bias2_tan

        ut_ai_rhs1_3 = ut_ai @ rhs1
        ut_ai_rhs1 = wp.vec2f(ut_ai_rhs1_3[0], ut_ai_rhs1_3[1])

        lam2_us = -(s_inv_22 @ (rhs2 - ut_ai_rhs1))
        lam2 = mass_coeff * lam2_us - impulse_coeff * acc2_tan

        lam2_world = lam2[0] * t1 + lam2[1] * t2
        lam2_us_world = lam2_us[0] * t1 + lam2_us[1] * t2

        u_lam2_us = (inv_mass1 + inv_mass2) * lam2_us_world
        u_lam2_us = u_lam2_us + cr1_b1 @ (inv_inertia1 @ (wp.transpose(cr2_b1) @ lam2_us_world))
        u_lam2_us = u_lam2_us + cr1_b2 @ (inv_inertia2 @ (wp.transpose(cr2_b2) @ lam2_us_world))

        lam1_us = -(a1_inv @ (rhs1 + u_lam2_us))
        lam1 = mass_coeff * lam1_us - impulse_coeff * acc1

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
            old_acc_l = acc_limit
            acc_limit = acc_limit + lam_limit
            if clamp == _CLAMP_MAX:
                acc_limit = wp.max(wp.float32(0.0), acc_limit)
            else:
                acc_limit = wp.min(wp.float32(0.0), acc_limit)
            lam_limit = acc_limit - old_acc_l

        axial_lam = lam_drive + lam_limit
        angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_lam)
        angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_lam)

        it += 1

    # ---- Writeback ---------------------------------------------------
    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2

    write_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid, acc1)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, acc2_world)
    if drive_active:
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, acc_drive)
    if limit_active:
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, acc_limit)


@wp.func
def actuated_double_ball_socket_iterate_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
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
        _revolute_iterate_at_multi(constraints, cid, 0, bodies, body_pair, idt, use_bias, num_sweeps)
    else:
        it = wp.int32(0)
        while it < num_sweeps:
            actuated_double_ball_socket_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)
            it += 1


@wp.func
def actuated_double_ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
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
    if joint_mode == JOINT_MODE_REVOLUTE:
        _revolute_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        _prismatic_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)
    elif joint_mode == JOINT_MODE_FIXED:
        _fixed_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)
    elif joint_mode == JOINT_MODE_CABLE:
        _cable_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)
    elif joint_mode == JOINT_MODE_BEAM:
        _beam_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)
    else:
        _ball_socket_iterate_at(constraints, cid, base_offset, bodies, body_pair, idt, use_bias)


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
    elif joint_mode == JOINT_MODE_FIXED:
        # All three anchor impulses contribute; no axial block.
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    elif joint_mode == JOINT_MODE_BEAM:
        # Same anchor layout as FIXED; no axial block. The PD softness
        # is already baked into the accumulated impulses, so the
        # wrench reflects the actual reaction the joint applied this
        # substep.
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    elif joint_mode == JOINT_MODE_CABLE:
        # Anchor-1 3-row lock plus three soft angular rows. The
        # angular rows are stored as per-row scalar impulses along
        # the cached world-frame directions in ``ut_ai``; each row
        # contributes a pure torque on body 2.
        dirs_mat = read_mat33(constraints, base_offset + _OFF_UT_AI, cid)
        acc_kappa = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
        d_bend1 = wp.vec3f(dirs_mat[0, 0], dirs_mat[0, 1], dirs_mat[0, 2])
        d_bend2 = wp.vec3f(dirs_mat[1, 0], dirs_mat[1, 1], dirs_mat[1, 2])
        d_twist = wp.vec3f(dirs_mat[2, 0], dirs_mat[2, 1], dirs_mat[2, 2])
        # Anchor-1 linear impulse + lever-arm torque (same as
        # ball-socket), plus the three pure-torque contributions.
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt)
        # Soft-angular sign follows the iterate: positive ``lam``
        # rotates body 1 toward ``+d``, so body 2 sees ``-lam * d``.
        torque = torque - (d_bend1 * acc_kappa[0] + d_bend2 * acc_kappa[1] + d_twist * acc_kappa[2]) * idt
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
    idt: wp.float32,
):
    """Direct prepare entry; see
    :func:`actuated_double_ball_socket_prepare_for_iteration_at`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def actuated_double_ball_socket_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Direct iterate entry; see
    :func:`actuated_double_ball_socket_iterate_at`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def revolute_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
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
    body_pair = constraint_bodies_make(b1, b2)
    _revolute_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def revolute_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Revolute-only prepare entry, skipping the ``joint_mode``
    dispatch. Counterpart of :func:`revolute_iterate`."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    _revolute_prepare_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def revolute_iterate_multi(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
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
    _revolute_iterate_at_multi(constraints, cid, 0, bodies, body_pair, idt, use_bias, num_sweeps)


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
    elif joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_BEAM:
        # Anchor-3 scalar drift along the persisted ``t2`` (the 6th
        # locked DoF). FIXED has no drive / limit; BEAM has no axial
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
    elif joint_mode == JOINT_MODE_CABLE:
        # Decompose the Darboux log-map about the reference axis.
        # ``drift_t1`` = bend-e1 component, ``drift_t2`` = bend-e2,
        # ``actuator_err`` = twist about the reference axis. All
        # three are in radians, computed via the full quaternion
        # log-map (matches :func:`_cable_prepare_at`).
        inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
        q_align = q2 * inv_init * wp.quat_inverse(q1)
        if q_align[3] < 0.0:
            q_align = -q_align
        qa_xyz = wp.vec3f(q_align[0], q_align[1], q_align[2])
        qa_xyz_len = wp.length(qa_xyz)
        if qa_xyz_len > 1.0e-9:
            angle = 2.0 * wp.atan2(qa_xyz_len, q_align[3])
            kappa = qa_xyz * (angle / qa_xyz_len)
        else:
            kappa = qa_xyz * 2.0
        axis_local1 = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
        n_hat_world = wp.quat_rotate(q1, axis_local1)
        n_len2 = wp.dot(n_hat_world, n_hat_world)
        if n_len2 > 1.0e-20:
            n_hat_world = n_hat_world / wp.sqrt(n_len2)
        else:
            n_hat_world = wp.vec3f(1.0, 0.0, 0.0)
        e1_world = create_orthonormal(n_hat_world)
        e1_world = e1_world - wp.dot(e1_world, n_hat_world) * n_hat_world
        e1_len2 = wp.dot(e1_world, e1_world)
        if e1_len2 > 1.0e-20:
            e1_world = e1_world / wp.sqrt(e1_len2)
        e2_world = wp.cross(n_hat_world, e1_world)
        drift_t1 = wp.dot(kappa, e1_world)
        drift_t2 = wp.dot(kappa, e2_world)
        actuator_err = wp.dot(kappa, n_hat_world)

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
