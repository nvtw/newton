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
    # Velocity-only damping effective mass for the PD limit row.
    # XPBD-style spring/damping split: spring stays in ``limit_cache``
    # (the PD layout's first 3 dwords) and runs in the main solve;
    # ``damp_mass_limit`` runs in the relax pass when the limit is
    # clamped. Unused on the Box2D limit path. See
    # :func:`pd_coefficients_split`.
    damp_mass_limit: wp.float32
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
# Cable-only field layout (3-point bending). Cable uses the
# prismatic-style ``mode_extras`` layout for anchor-3 storage
# (``la3_b{1,2}``, ``r3_b{1,2}``, ``acc_imp3``, ``bias3`` -- the
# init kernel branches on ``mode in {PRISMATIC, FIXED, CABLE}`` to
# write that layout). Per-row PD spring/damping coefficients and
# biases pack into the ``s_inv`` mat33 slot (mode_cache[18..26],
# unused by cable's own prepare since it doesn't Schur-couple
# anchor-1 with anchor-2). Bend rows share one coefficient triple
# since transverse inertia is near-isotropic for capsule / cylinder
# primitives that cables are built from.
_OFF_CABLE_BEND_GAMMA = wp.constant(int(_OFF_MODE_CACHE) + 18)
_OFF_CABLE_BEND_EFF_MASS = wp.constant(int(_OFF_MODE_CACHE) + 19)
_OFF_CABLE_BEND_DAMP_MASS = wp.constant(int(_OFF_MODE_CACHE) + 20)
_OFF_CABLE_TWIST_GAMMA = wp.constant(int(_OFF_MODE_CACHE) + 21)
_OFF_CABLE_TWIST_EFF_MASS = wp.constant(int(_OFF_MODE_CACHE) + 22)
_OFF_CABLE_TWIST_DAMP_MASS = wp.constant(int(_OFF_MODE_CACHE) + 23)
_OFF_CABLE_BEND_BIAS_T1 = wp.constant(int(_OFF_MODE_CACHE) + 24)
_OFF_CABLE_BEND_BIAS_T2 = wp.constant(int(_OFF_MODE_CACHE) + 25)
_OFF_CABLE_TWIST_BIAS = wp.constant(int(_OFF_MODE_CACHE) + 26)

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
_OFF_DAMP_MASS_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damp_mass_limit"))
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

    # ``mode_extras`` block is mode-aliased: REVOLUTE stores the
    # twist-tracker scratch (inv_initial_orientation, revolution_counter,
    # previous_quaternion_angle); PRISMATIC / FIXED / CABLE store the
    # anchor-3 snapshot + bias3 + acc_imp3 (cable's 3-point bending
    # uses anchor-3 perpendicular to the cable axis for twist
    # resistance; bend uses anchor-2 along the axis). Writing both
    # layouts unconditionally would clobber the alias, so we branch.
    if mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE:
        write_vec3(constraints, _OFF_LA3_B1, cid, la3_b1)
        write_vec3(constraints, _OFF_LA3_B2, cid, la3_b2)
        write_vec3(constraints, _OFF_R3_B1, cid, zero3)
        write_vec3(constraints, _OFF_R3_B2, cid, zero3)
        write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, 0.0)
    else:
        # REVOLUTE / BALL_SOCKET: zero out the anchor-3 slots via the
        # twist-tracker layout. BALL_SOCKET reads neither side of the
        # alias, so any consistent zero is fine.
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
    use_bias: wp.bool,
) -> wp.float32:
    """Scalar drive+limit PGS step for revolute/prismatic mode.

    Both modes apply a single scalar impulse ``axial_lam`` along the
    free DoF axis. Revolute applies it as an angular impulse about
    ``n_hat``; prismatic as a linear impulse along ``n_hat``. The
    actuator cache (drive PD scalars, limit Box2D *or* PD scalars,
    the warm-started accumulated impulses) is identical across both
    modes, so the iterate math collapses to a shared helper.

    Drive row uses the combined Box2D-v3 PD formulation (stiffness +
    damping baked into one ``eff_mass_drive_soft`` / ``gamma_drive``
    /``bias_drive`` triple); ``target_velocity`` is folded into
    ``bias_drive`` at prepare time. Limit row is split spring /
    damping (``use_bias=True`` runs the spring; ``use_bias=False``
    runs damping only) -- the limit can sit at high ``c`` for a long
    time and the split keeps PGS well-conditioned.

    Args:
        constraints: Shared column-major constraint storage.
        cid: Constraint id.
        base_offset: Dword offset of the constraint within its column.
        jv_axial: Pre-step axial velocity residual. Revolute passes
            ``n . (w1 - w2)``; prismatic passes ``n . (v1_anchor -
            v2_anchor)``. See the per-mode callers for the sign
            conventions.
        clamp: Pre-computed limit clamp state (``_CLAMP_NONE`` /
            ``_CLAMP_MIN`` / ``_CLAMP_MAX``) from prepare.
        use_bias: ``True`` during the main solve, ``False`` during
            the relax pass. Affects the limit row only -- the drive
            row runs every iter.

    Returns:
        Net per-iteration axial impulse ``lam_drive + lam_limit``.
    """
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    # ``max_lambda_drive`` was a stored ``max_force_drive * dt``;
    # recompute inline since both inputs are already in registers.
    max_lambda_drive = max_force_drive * (wp.float32(1.0) / idt)
    drive_active = drive_mode != DRIVE_MODE_OFF
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)

    # ---- Drive row (combined PD) -------------------------------------
    lam_drive = float(0.0)
    if drive_active:
        bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
        gamma_drive = read_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid)
        eff_mass_drive_soft = read_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid)
        if eff_mass_drive_soft > 0.0:
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
            if use_bias:
                # Main solve: PD spring-only.
                pd_mass = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid)
                pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid)
                pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid)
                if pd_mass > 0.0:
                    lam_limit = -pd_mass * (jv_axial - pd_beta + pd_gamma * acc_limit)
            else:
                # Relax: velocity-only damping. No bias term -- the
                # spring already drove the limit error toward zero
                # in the main solve.
                damp_mass_limit = read_float(constraints, base_offset + _OFF_DAMP_MASS_LIMIT, cid)
                if damp_mass_limit > 0.0:
                    lam_limit = -damp_mass_limit * jv_axial
        else:
            # Box2D soft-constraint path stays on the combined
            # formulation -- it has a rigid positional lock at the
            # limit, not a soft PD, so there's no damping term to
            # split out. Relax pass enforces ``Jv = 0`` with the
            # bias retained (matches the contact / joint anchor
            # convention).
            if use_bias:
                eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
                bias_box = read_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid)
                mc_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
                ic_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
                if eff_inv > 0.0:
                    eff_axial = 1.0 / eff_inv
                    lam_unsoft = -eff_axial * (jv_axial + bias_box)
                    lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
            else:
                # Box2D-mode relax: enforce Jv = 0 without the
                # position bias.
                eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid)
                mc_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
                ic_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
                if eff_inv > 0.0:
                    eff_axial = 1.0 / eff_inv
                    lam_unsoft = -eff_axial * jv_axial
                    lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
        old_acc = acc_limit
        acc_limit = acc_limit + lam_limit
        # Unilateral clamp: the limit only pushes back toward the
        # allowed range.
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
    clamp_nyquist: wp.bool,
):
    """Anchor-1 3-row positional lock prepare. Shared by BALL_SOCKET
    and CABLE modes (which use a standalone anchor-1 lock as their
    only positional constraint).

    ``clamp_nyquist`` is forwarded to
    :func:`soft_constraint_coefficients`. Both callers pass ``True``
    today -- only the cable bend / twist soft rows opt out of the
    clamp; every positional anchor lock keeps it on so PGS can
    propagate the lock through long chains in a few iterations.

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
    # Joint anchor lock: skip Nyquist clamp so user-requested "rigid"
    # lock actually closes drift each substep (default ``hertz=1e9``
    # otherwise gets clamped and only ~60% of position drift is
    # corrected per substep, producing visible anchor offset).
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
        hertz, damping_ratio, dt, clamp_nyquist
    )
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
    # Fold ``target_velocity`` into ``bias_drive`` so the combined
    # PD row drives ``Jv -> -target_velocity - gamma * acc``. Required
    # for VELOCITY-mode drives.
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
        # Same spring/damping XPBD split as the drive row. Spring
        # half (PD layout) drives the limit error to zero in the
        # main solve; ``damp_mass_limit`` runs in the relax pass
        # when the limit is clamped.
        # PD limit row keeps Nyquist clamp on -- the bias-amplification
        # / chain-instability concern that bit revolute / prismatic
        # anchor locks at the unclamped omega applies here too.
        pd_gamma_limit, pd_beta_limit, pd_m_soft, damp_mass_limit = pd_coefficients_split(
            stiffness_limit, damping_limit, limit_C, eff_inv, dt, True
        )
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, pd_gamma_limit)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, pd_beta_limit)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, pd_m_soft)
        write_float(constraints, base_offset + _OFF_DAMP_MASS_LIMIT, cid, damp_mass_limit)
    else:
        # Box2D limit row: keep Nyquist clamp on (default).
        br_limit, mc_limit, ic_limit = soft_constraint_coefficients(
            hertz_limit, damping_ratio_limit, dt, True
        )
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, -limit_C * br_limit)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mc_limit)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, ic_limit)
        write_float(constraints, base_offset + _OFF_DAMP_MASS_LIMIT, cid, wp.float32(0.0))

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
    ) = _anchor1_positional_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt, True)

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
    # Revolute anchor lock (3-row anchor-1 + 2-row anchor-2 tangent):
    # keep Nyquist clamp on. Long revolute chains (e.g. the 250-hinge
    # motorized chain example) amplify the unclamped bias the way
    # cable's bend / twist + anchor lock do; PGS can't propagate the
    # hard lock through the chain in a few iterations and the chain
    # explodes. The Box2D-soft formulation's slack is what keeps long
    # chains stable.
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt, True)
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
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, use_bias)
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
    # Prismatic anchor lock: keep Nyquist clamp on -- same chain
    # stability constraint as revolute. Long prismatic chains
    # (linkage trains, articulated arms via prismatic joints) need
    # the soft-formulation slack that the clamped formulation
    # provides.
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt, True)
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
    axial_lam = _axial_drive_limit_iterate(constraints, cid, base_offset, jv_axial, clamp, idt, use_bias)

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
    """Cable-mode prepare: anchor-1 hard 3-row ball-socket lock
    plus 3-point bending springs (anchor-2 along the cable axis for
    bend, anchor-3 perpendicular for twist).

    Bend stiffness (user-supplied as angular ``N*m/rad``) is
    converted internally to a linear spring at lever arm
    ``L_axial = ‖anchor2 - anchor1‖``. Same for twist at lever arm
    ``L_perp = L_axial`` (the init kernel snapshots anchor-3 at
    ``rest_length`` perpendicular to the cable axis, where
    ``rest_length == L_axial``). The bias is angle-proportional
    (see :func:`angle_proportional_chord_bias`): a 90° bend produces
    a torque ``k_bend_ang * (pi/2)``, not the saturating
    ``k_bend_ang * sin(pi/2)`` a chord-bias spring would give.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    orient1 = bodies.orientation[b1]
    orient2 = bodies.orientation[b2]
    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]
    inv_mass_sum = inv_mass1 + inv_mass2

    # ---- Anchor-1: hard 3-row positional lock (shared helper) -------
    (
        _r1_b1,
        _r1_b2,
        _cr1_b1,
        _cr1_b2,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
    ) = _anchor1_positional_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt, True)
    dt = wp.float32(1.0) / idt

    # ---- Anchor-2: world-frame snapshot from body-1 / body-2 frames -
    la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
    la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
    la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
    r1_b1_world = wp.quat_rotate(orient1, la1_b1)
    r2_b1 = wp.quat_rotate(orient1, la2_b1)
    r2_b2 = wp.quat_rotate(orient2, la2_b2)
    write_vec3(constraints, base_offset + _OFF_R2_B1, cid, r2_b1)
    write_vec3(constraints, base_offset + _OFF_R2_B2, cid, r2_b2)

    p1_b1 = pos1 + r1_b1_world
    p2_b1 = pos1 + r2_b1
    p2_b2 = pos2 + r2_b2

    # Cable axis from body-1's anchor-1 to body-1's anchor-2 (rotates
    # rigidly with body 1). This is the *current* axis direction; the
    # rest pose has body-2's anchor-2 coinciding with body-1's anchor-2,
    # so any deviation along the perpendicular plane is bend.
    axis_b1 = p2_b1 - p1_b1
    L_axial2 = wp.dot(axis_b1, axis_b1)
    if L_axial2 > 1.0e-20:
        L_axial = wp.sqrt(L_axial2)
        n_hat = axis_b1 / L_axial
    else:
        L_axial = wp.float32(1.0)
        n_hat = wp.vec3f(1.0, 0.0, 0.0)

    t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
    write_vec3(constraints, base_offset + _OFF_T1, cid, t1)
    write_vec3(constraints, base_offset + _OFF_T2, cid, t2)
    write_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid, n_hat)

    # ---- Bend per-row eff_inv ---------------------------------------
    # Constraint along tangent t_i: C_i = t_i . (p2_b2 - p2_b1).
    # J row contribution: linear ±t_i, angular ±(r2_b{1,2} × t_i).
    # eff_inv_i = (1/m1 + 1/m2) + (r2_b1 × t_i)·I1⁻¹·(r2_b1 × t_i)
    #                            + (r2_b2 × t_i)·I2⁻¹·(r2_b2 × t_i)
    cross1_t1 = wp.cross(r2_b1, t1)
    cross2_t1 = wp.cross(r2_b2, t1)
    cross1_t2 = wp.cross(r2_b1, t2)
    cross2_t2 = wp.cross(r2_b2, t2)
    eff_inv_t1 = inv_mass_sum + wp.dot(cross1_t1, inv_inertia1 @ cross1_t1) + wp.dot(cross2_t1, inv_inertia2 @ cross2_t1)
    eff_inv_t2 = inv_mass_sum + wp.dot(cross1_t2, inv_inertia1 @ cross1_t2) + wp.dot(cross2_t2, inv_inertia2 @ cross2_t2)
    # One coefficient triple shared between the two tangent rows;
    # transverse inertia is near-isotropic on capsule / cylinder
    # primitives, so averaging is a small approximation.
    eff_inv_bend = wp.float32(0.5) * (eff_inv_t1 + eff_inv_t2)

    # ---- Bend stiffness conversion (angular -> linear) --------------
    # User passes ``bend_stiffness`` as angular [N·m/rad]; the point
    # spring at lever arm L produces ``tau = k_lin * L^2 * theta``,
    # so ``k_lin = k_ang / L^2``. Same for damping.
    k_bend_ang = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    c_bend_ang = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    L_axial2_safe = wp.max(L_axial2, wp.float32(1.0e-20))
    k_bend_lin = k_bend_ang / L_axial2_safe
    c_bend_lin = c_bend_ang / L_axial2_safe
    gamma_bend, _bias_bend_unused, eff_mass_bend, damp_mass_bend = pd_coefficients_split(
        k_bend_lin, c_bend_lin, wp.float32(0.0), eff_inv_bend, dt, True
    )

    # ---- Bend angle-proportional bias --------------------------------
    # The 2-row tangent constraint sees the *perpendicular*
    # projection of the drift, whose magnitude is ``L * sin(theta)``
    # (not the full 3D chord ``2L sin(theta/2)``). So the angle
    # recovery is ``theta = asin(|perp| / L)`` and per-component
    # bias is ``(chord_along_t / |perp|) * theta * L * idt`` =
    # ``chord_along_t * (theta / sin(theta)) * idt``. Reduces to
    # ``chord_along_t * idt`` at small angles (``theta / sin(theta)
    # -> 1``); grows the bias by ``~1.4x`` at 80 deg bend so the
    # spring sees the actual deflection angle, not the chord.
    drift2 = p2_b2 - p2_b1
    chord_t1 = wp.dot(t1, drift2)
    chord_t2 = wp.dot(t2, drift2)
    perp_mag2 = chord_t1 * chord_t1 + chord_t2 * chord_t2
    if perp_mag2 > 1.0e-20:
        perp_mag = wp.sqrt(perp_mag2)
        sin_theta = wp.clamp(perp_mag / L_axial, wp.float32(-1.0), wp.float32(1.0))
        theta_bend = wp.asin(sin_theta)
        # Scale = ``theta / sin(theta)`` >= 1, applied to the
        # perp-tangent components to get per-tangent bias.
        bend_scale = theta_bend * L_axial / perp_mag
        bias_bend_t1 = chord_t1 * bend_scale * idt
        bias_bend_t2 = chord_t2 * bend_scale * idt
    else:
        bias_bend_t1 = wp.float32(0.0)
        bias_bend_t2 = wp.float32(0.0)

    # ---- Anchor-3: world-frame snapshot (perpendicular twist anchor) -
    la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
    la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
    r3_b1 = wp.quat_rotate(orient1, la3_b1)
    r3_b2 = wp.quat_rotate(orient2, la3_b2)
    write_vec3(constraints, base_offset + _OFF_R3_B1, cid, r3_b1)
    write_vec3(constraints, base_offset + _OFF_R3_B2, cid, r3_b2)

    p3_b1 = pos1 + r3_b1
    p3_b2 = pos2 + r3_b2

    # Twist row: scalar constraint along ``t2``, the perpendicular
    # tangent direction not aligned with the anchor-3 lever arm.
    # ``L_perp = L_axial`` per the design (init snapshots anchor-3 at
    # ``rest_length`` perpendicular to ``n_hat``, and
    # ``rest_length = L_axial``).
    L_perp = L_axial
    cross3_b1_t2 = wp.cross(r3_b1, t2)
    cross3_b2_t2 = wp.cross(r3_b2, t2)
    eff_inv_twist = inv_mass_sum + wp.dot(cross3_b1_t2, inv_inertia1 @ cross3_b1_t2) + wp.dot(cross3_b2_t2, inv_inertia2 @ cross3_b2_t2)

    k_twist_ang = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    c_twist_ang = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)
    L_perp2_safe = wp.max(L_perp * L_perp, wp.float32(1.0e-20))
    k_twist_lin = k_twist_ang / L_perp2_safe
    c_twist_lin = c_twist_ang / L_perp2_safe
    gamma_twist, _bias_twist_unused, eff_mass_twist, damp_mass_twist = pd_coefficients_split(
        k_twist_lin, c_twist_lin, wp.float32(0.0), eff_inv_twist, dt, True
    )

    # Twist bias: same ``theta = asin(perp/L)`` recovery as bend, but
    # the perpendicular-projection here is 1-D (just the ``t2``
    # component of the anchor-3 drift -- the ``t1`` component is
    # axial w.r.t. the lever arm ``r3 = L * t1`` and would catch
    # axial *stretch* of anchor-3, which is already locked by
    # anchor-1).
    chord_twist = wp.dot(t2, p3_b2 - p3_b1)
    sin_phi = wp.clamp(chord_twist / L_perp, wp.float32(-1.0), wp.float32(1.0))
    theta_twist = wp.asin(sin_phi)
    twist_perp_mag = wp.abs(chord_twist)
    if twist_perp_mag > 1.0e-20:
        twist_scale = theta_twist * L_perp / chord_twist
        bias_twist = chord_twist * twist_scale * idt
    else:
        bias_twist = wp.float32(0.0)

    # ---- Stash coefficients + biases in mode_cache slots ------------
    write_float(constraints, base_offset + _OFF_CABLE_BEND_GAMMA, cid, gamma_bend)
    write_float(constraints, base_offset + _OFF_CABLE_BEND_EFF_MASS, cid, eff_mass_bend)
    write_float(constraints, base_offset + _OFF_CABLE_BEND_DAMP_MASS, cid, damp_mass_bend)
    write_float(constraints, base_offset + _OFF_CABLE_TWIST_GAMMA, cid, gamma_twist)
    write_float(constraints, base_offset + _OFF_CABLE_TWIST_EFF_MASS, cid, eff_mass_twist)
    write_float(constraints, base_offset + _OFF_CABLE_TWIST_DAMP_MASS, cid, damp_mass_twist)
    write_float(constraints, base_offset + _OFF_CABLE_BEND_BIAS_T1, cid, bias_bend_t1)
    write_float(constraints, base_offset + _OFF_CABLE_BEND_BIAS_T2, cid, bias_bend_t2)
    write_float(constraints, base_offset + _OFF_CABLE_TWIST_BIAS, cid, bias_twist)

    # ---- Reset cross-substep accumulators ---------------------------
    # The (t1, t2) basis rotates with body 1 every substep, so scalar
    # impulses warm-started along the prior basis would mis-direct.
    # Within-substep warm-start (across solver_iterations sweeps) is
    # preserved in ``acc_imp2`` / ``acc_imp3`` during iterate.
    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, zero3)
    write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, zero3)

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
    """Cable-mode PGS iterate: anchor-1 hard 3-row + anchor-2 soft
    2-row tangent bend springs + anchor-3 soft 1-row twist spring.

    ``use_bias=True`` runs the spring half (with angle-proportional
    biases set in prepare). ``use_bias=False`` runs the relax pass:
    pure velocity-only damping per row, no bias. The spring's
    ``gamma * acc`` softness term is also dropped on relax (matches
    the cable's pre-redesign convention; bend / twist accumulators
    settle within the main solve and the relax just removes the
    velocity drift the bias injected).
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

    # ---- Anchor-1 hard 3-row positional block -----------------------
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

    # ---- Anchor-2 (bend) + anchor-3 (twist) soft point springs ------
    r2_b1 = read_vec3(constraints, base_offset + _OFF_R2_B1, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    r3_b1 = read_vec3(constraints, base_offset + _OFF_R3_B1, cid)
    r3_b2 = read_vec3(constraints, base_offset + _OFF_R3_B2, cid)
    t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
    t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)

    eff_mass_bend = read_float(constraints, base_offset + _OFF_CABLE_BEND_EFF_MASS, cid)
    gamma_bend = read_float(constraints, base_offset + _OFF_CABLE_BEND_GAMMA, cid)
    damp_mass_bend = read_float(constraints, base_offset + _OFF_CABLE_BEND_DAMP_MASS, cid)
    eff_mass_twist = read_float(constraints, base_offset + _OFF_CABLE_TWIST_EFF_MASS, cid)
    gamma_twist = read_float(constraints, base_offset + _OFF_CABLE_TWIST_GAMMA, cid)
    damp_mass_twist = read_float(constraints, base_offset + _OFF_CABLE_TWIST_DAMP_MASS, cid)

    acc_bend = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    acc_twist_vec = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
    acc_bend_t1 = acc_bend[0]
    acc_bend_t2 = acc_bend[1]
    acc_twist = acc_twist_vec[0]

    if use_bias:
        bias_bend_t1 = read_float(constraints, base_offset + _OFF_CABLE_BEND_BIAS_T1, cid)
        bias_bend_t2 = read_float(constraints, base_offset + _OFF_CABLE_BEND_BIAS_T2, cid)
        bias_twist = read_float(constraints, base_offset + _OFF_CABLE_TWIST_BIAS, cid)
    else:
        bias_bend_t1 = wp.float32(0.0)
        bias_bend_t2 = wp.float32(0.0)
        bias_twist = wp.float32(0.0)

    # Sign convention follows revolute / ball-socket anchor lock:
    # ``rhs = Jv + bias`` (positive Jv means drift is growing, bias
    # is positive for positive drift, ``lam = -eff * rhs`` is then
    # negative and restores drift -> 0). Apply: body 2 receives
    # ``+lam`` along the row direction; body 1 receives ``-lam``.
    cross1_t1 = wp.cross(r2_b1, t1)
    cross2_t1 = wp.cross(r2_b2, t1)
    if eff_mass_bend > 0.0:
        jv_t1 = (
            -wp.dot(t1, velocity1)
            - wp.dot(cross1_t1, angular_velocity1)
            + wp.dot(t1, velocity2)
            + wp.dot(cross2_t1, angular_velocity2)
        )
        if use_bias:
            lam_t1 = -eff_mass_bend * (jv_t1 + bias_bend_t1 + gamma_bend * acc_bend_t1)
        else:
            lam_t1 = -damp_mass_bend * jv_t1
        velocity1 = velocity1 - inv_mass1 * (t1 * lam_t1)
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cross1_t1 * lam_t1)
        velocity2 = velocity2 + inv_mass2 * (t1 * lam_t1)
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cross2_t1 * lam_t1)
        if use_bias:
            acc_bend_t1 = acc_bend_t1 + lam_t1

    # Bend row 2: t2-tangent (in-row GS reads the velocities updated by row 1).
    cross1_t2 = wp.cross(r2_b1, t2)
    cross2_t2 = wp.cross(r2_b2, t2)
    if eff_mass_bend > 0.0:
        jv_t2 = (
            -wp.dot(t2, velocity1)
            - wp.dot(cross1_t2, angular_velocity1)
            + wp.dot(t2, velocity2)
            + wp.dot(cross2_t2, angular_velocity2)
        )
        if use_bias:
            lam_t2 = -eff_mass_bend * (jv_t2 + bias_bend_t2 + gamma_bend * acc_bend_t2)
        else:
            lam_t2 = -damp_mass_bend * jv_t2
        velocity1 = velocity1 - inv_mass1 * (t2 * lam_t2)
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cross1_t2 * lam_t2)
        velocity2 = velocity2 + inv_mass2 * (t2 * lam_t2)
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cross2_t2 * lam_t2)
        if use_bias:
            acc_bend_t2 = acc_bend_t2 + lam_t2

    # Twist row: scalar along t2 with anchor-3 lever arms.
    cross3_b1_t2 = wp.cross(r3_b1, t2)
    cross3_b2_t2 = wp.cross(r3_b2, t2)
    if eff_mass_twist > 0.0:
        jv_tw = (
            -wp.dot(t2, velocity1)
            - wp.dot(cross3_b1_t2, angular_velocity1)
            + wp.dot(t2, velocity2)
            + wp.dot(cross3_b2_t2, angular_velocity2)
        )
        if use_bias:
            lam_tw = -eff_mass_twist * (jv_tw + bias_twist + gamma_twist * acc_twist)
        else:
            lam_tw = -damp_mass_twist * jv_tw
        velocity1 = velocity1 - inv_mass1 * (t2 * lam_tw)
        angular_velocity1 = angular_velocity1 - inv_inertia1 @ (cross3_b1_t2 * lam_tw)
        velocity2 = velocity2 + inv_mass2 * (t2 * lam_tw)
        angular_velocity2 = angular_velocity2 + inv_inertia2 @ (cross3_b2_t2 * lam_tw)
        if use_bias:
            acc_twist = acc_twist + lam_tw

    if use_bias:
        write_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid, wp.vec3f(acc_bend_t1, acc_bend_t2, 0.0))
        write_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid, wp.vec3f(acc_twist, 0.0, 0.0))

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
    # Fixed (6-DoF weld) anchor lock: keep Nyquist clamp on. Same
    # chain stability constraint as revolute / prismatic -- welded
    # multi-body assemblies (typical use case) form long constraint
    # graphs that PGS can't propagate through with a hard lock.
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt, True)
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
    # Drive PD: combined Box2D-v3 formulation (stiffness + damping
    # baked into one ``eff_mass_drive_soft``/``gamma_drive``/``bias_drive``
    # triple); ``target_velocity`` is folded into ``bias_drive`` at
    # prepare time. See :func:`pd_coefficients`.
    bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
    gamma_drive = read_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid)
    eff_mass_drive_soft = read_float(constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)

    drive_active = drive_mode != DRIVE_MODE_OFF
    spring_drive_active = drive_active and eff_mass_drive_soft > wp.float32(0.0)

    acc_limit = wp.float32(0.0)
    pd_mode_limit = False
    pd_mass = wp.float32(0.0)
    pd_gamma = wp.float32(0.0)
    pd_beta = wp.float32(0.0)
    damp_mass_limit = wp.float32(0.0)
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
            damp_mass_limit = read_float(constraints, base_offset + _OFF_DAMP_MASS_LIMIT, cid)
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

        # Axial drive (combined PD, runs every iter) + limit row
        # (spring/damping XPBD split: ``use_bias=True`` runs spring
        # toward clamp boundary; ``use_bias=False`` runs damping
        # ``-damp_mass * Jv``; Box2D limit: rigid lock with bias on
        # main solve, ``Jv = 0`` on relax).
        jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)

        lam_drive = wp.float32(0.0)
        if spring_drive_active:
            lam_drive = -eff_mass_drive_soft * (jv_axial - bias_drive + gamma_drive * acc_drive)
            old_acc = acc_drive
            acc_drive = acc_drive + lam_drive
            if max_force_drive > wp.float32(0.0):
                acc_drive = wp.clamp(acc_drive, -max_lambda_drive, max_lambda_drive)
            lam_drive = acc_drive - old_acc

        lam_limit = wp.float32(0.0)
        if limit_active:
            if pd_mode_limit:
                if use_bias:
                    if pd_mass > wp.float32(0.0):
                        lam_limit = -pd_mass * (jv_axial - pd_beta + pd_gamma * acc_limit)
                else:
                    if damp_mass_limit > wp.float32(0.0):
                        lam_limit = -damp_mass_limit * jv_axial
            else:
                if eff_axial > wp.float32(0.0):
                    if use_bias:
                        lam_unsoft = -eff_axial * (jv_axial + bias_box)
                        lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
                    else:
                        # Box2D-mode relax: enforce Jv = 0 without bias.
                        lam_unsoft = -eff_axial * jv_axial
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
    elif joint_mode == JOINT_MODE_CABLE:
        # 3-point bending: anchor-1 hard 3-row lock + anchor-2 soft
        # 2-row tangent bend springs (along t1, t2 with anchor-2
        # lever arms r2_b2) + anchor-3 soft 1-row twist spring
        # (along t2 with anchor-3 lever arm r3_b2). Each soft row
        # contributes a linear force on body 2 along its tangent
        # plus a lever-arm torque.
        t1 = read_vec3(constraints, base_offset + _OFF_T1, cid)
        t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
        acc_bend = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
        acc_twist_vec = read_vec3(constraints, base_offset + _OFF_ACC_IMP3, cid)
        # Bend tangent forces on body 2 (sign matches the iterate:
        # positive ``lam_t_i`` pushes body 2 along ``+t_i``).
        bend_force = (t1 * acc_bend[0] + t2 * acc_bend[1]) * idt
        twist_force = t2 * (acc_twist_vec[0] * idt)
        force = acc1 * idt + bend_force + twist_force
        # Lever-arm torques: anchor-1 lever arm for the rigid lock,
        # anchor-2 lever arm for bend (force applied at p2_b2 = pos2
        # + r2_b2), anchor-3 lever arm for twist.
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, bend_force) + wp.cross(r3_b2, twist_force)
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
    elif joint_mode == JOINT_MODE_FIXED:
        # Anchor-3 scalar drift along the persisted ``t2`` (the 6th
        # locked DoF). Reported in the "actuator" slot since FIXED
        # has no drive / limit.
        la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
        la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
        p3_b1 = pos1 + wp.quat_rotate(q1, la3_b1)
        p3_b2 = pos2 + wp.quat_rotate(q2, la3_b2)
        t2 = read_vec3(constraints, base_offset + _OFF_T2, cid)
        actuator_err = wp.dot(t2, p3_b2 - p3_b1)
    elif joint_mode == JOINT_MODE_CABLE:
        # 3-point bending error decoder: matches the prepare's
        # angle-proportional bias formula. The 2-row tangent
        # projection sees ``L * sin(theta)`` (not the full 3D chord),
        # so ``theta = asin(perp / L)`` per axis.
        la1_b1 = read_vec3(constraints, base_offset + _OFF_LA1_B1, cid)
        la2_b1 = read_vec3(constraints, base_offset + _OFF_LA2_B1, cid)
        la2_b2 = read_vec3(constraints, base_offset + _OFF_LA2_B2, cid)
        la3_b1 = read_vec3(constraints, base_offset + _OFF_LA3_B1, cid)
        la3_b2 = read_vec3(constraints, base_offset + _OFF_LA3_B2, cid)
        r1_b1 = wp.quat_rotate(q1, la1_b1)
        r2_b1 = wp.quat_rotate(q1, la2_b1)
        r2_b2 = wp.quat_rotate(q2, la2_b2)
        r3_b1 = wp.quat_rotate(q1, la3_b1)
        r3_b2 = wp.quat_rotate(q2, la3_b2)
        p2_b1 = pos1 + r2_b1
        p2_b2 = pos2 + r2_b2
        p3_b1 = pos1 + r3_b1
        p3_b2 = pos2 + r3_b2
        axis_b1 = p2_b1 - (pos1 + r1_b1)
        L_axial2 = wp.dot(axis_b1, axis_b1)
        if L_axial2 > 1.0e-20:
            L_axial = wp.sqrt(L_axial2)
            n_hat_world = axis_b1 / L_axial
        else:
            L_axial = wp.float32(1.0)
            n_hat_world = wp.vec3f(1.0, 0.0, 0.0)
        t1_world = create_orthonormal(n_hat_world)
        t2_world = wp.cross(n_hat_world, t1_world)
        bend_drift = p2_b2 - p2_b1
        chord_t1 = wp.dot(t1_world, bend_drift)
        chord_t2 = wp.dot(t2_world, bend_drift)
        chord_twist = wp.dot(t2_world, p3_b2 - p3_b1)
        sin_t1 = wp.clamp(chord_t1 / L_axial, wp.float32(-1.0), wp.float32(1.0))
        sin_t2 = wp.clamp(chord_t2 / L_axial, wp.float32(-1.0), wp.float32(1.0))
        sin_tw = wp.clamp(chord_twist / L_axial, wp.float32(-1.0), wp.float32(1.0))
        drift_t1 = wp.asin(sin_t1)
        drift_t2 = wp.asin(sin_t2)
        actuator_err = wp.asin(sin_tw)

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
