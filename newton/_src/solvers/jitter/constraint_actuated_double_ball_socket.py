# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Unified ball-socket / revolute / prismatic joint with optional drive + limits.

Despite the legacy module name, this file implements a *single* joint
constraint that covers ball-socket, revolute, and prismatic behaviour
via a runtime ``joint_mode`` tag. The old standalone
``constraint_double_ball_socket`` and ``constraint_prismatic`` modules
were folded in here -- their combined feature set (rank-3/5 lock +
optional scalar drive / limit row) is a superset of any one of them.
Keeping one constraint type means one dispatcher arm, one graph-
colouring tag, and the same body-data loads shared across the
positional lock and the actuator row.

The three modes differ only in which DoF are locked and what "axial
drive / axial limit" means -- everything else (soft-constraint plumbing,
warm-starting, Schur block structure) is shared.

Ball-socket (:data:`JOINT_MODE_BALL_SOCKET`)
--------------------------------------------
Locks only the 3 translational DoF at ``anchor1``; all 3 rotational
DoF are free. Pure point-matching formulation, same anchor-1 math as
the revolute / prismatic modes -- a single 3-row point-match constraint
solved as one 3x3 direct solve (no Schur complement needed because
there are no anchor-2 / anchor-3 rows).

``anchor2`` is not used; the public API does not take it. No drive, no
limit, no axial row -- a pair of bodies pinned together at one world
point, free to rotate independently around it.

See :func:`_ball_socket_prepare_at` for the math.

Revolute (:data:`JOINT_MODE_REVOLUTE`)
--------------------------------------
Locks 3 translational + 2 rotational DoF via a pair of ball-socket
anchors on the hinge axis. Pure point-matching formulation:

* Anchor 1 contributes a full 3-row lock: ``p1_b2 == p1_b1``.
* Anchor 2 contributes a 2-row lock projected onto the tangent basis
  ``(t1, t2)`` perpendicular to the hinge axis -- the axial row of
  anchor 2 is the analytical null-space of the 6-row stack, so
  dropping it yields a clean rank-5 system without rank deficiency.

Solved as a 3x3 + 2x2 Schur complement; see :func:`_revolute_prepare_at`
for the math.

The optional scalar row drives / limits the *relative twist* about the
hinge axis ``n_hat = (anchor2 - anchor1) / |...|``; ``target`` /
``min_value`` / ``max_value`` are in radians, ``max_force_drive`` is in
N*m.

Prismatic (:data:`JOINT_MODE_PRISMATIC`)
----------------------------------------
Locks 3 rotational + 2 translational DoF via three ball-socket anchors,
all with tangent-plane projections. Pure point-matching formulation
(no quaternion math):

* Anchor 1: 2 tangent rows -- the bodies' anchor-1 points can only
  separate along the slide axis ``n_hat``.
* Anchor 2: 2 tangent rows -- same lock at the second on-axis anchor.
* Anchor 3: 1 scalar row along ``t2`` -- anchor 3 is auto-derived at
  init as ``anchor1 + rest_length * t_ref`` with ``t_ref`` an arbitrary
  unit perpendicular to ``n_hat``. Projecting onto ``t2`` kills the
  last rotational DoF (rotation about ``n_hat``).

  The tangent basis ``(t1, t2)`` is rebuilt every substep so that
  ``t1`` points along the projection of the *current* anchor1 -> anchor3
  vector perpendicular to ``n_hat``. That choice makes the anchor-3
  scalar row the exact tangential velocity gate for rotation about
  ``n_hat`` (unit gain) and is essential for block Gauss-Seidel
  convergence in multi-joint chains: an arbitrary perpendicular would
  introduce a ``cos(alpha)`` mismatch between joints sharing a body
  and diverge the outer solver iterations.

This yields a rank-5 system solved as a 4x4 + 1x1 Schur complement; the
4x4 block is the stack of the two anchor-tangent pairs, the 1x1 is the
scalar anchor-3 row. See :func:`_prismatic_prepare_at` for the math.

The optional scalar row now drives / limits the *relative slide* along
``n_hat``; ``target`` / ``min_value`` / ``max_value`` are in meters,
``max_force_drive`` is in N.

Drive + limits (shared plumbing)
--------------------------------
Three independent sub-blocks on the free DoF:

* :data:`DRIVE_MODE_OFF`         -- no actuation, DoF is free.
* :data:`DRIVE_MODE_POSITION`    -- PD spring-damper towards ``target``
                                    (rad or m); gains
                                    ``stiffness_drive`` and
                                    ``damping_drive`` are required to
                                    be set by the caller (both zero
                                    means "drive off").
                                    ``max_force_drive`` caps the
                                    per-substep impulse when > 0,
                                    unlimited when 0.
* :data:`DRIVE_MODE_VELOCITY`    -- PD velocity servo tracking
                                    ``target_velocity`` (rad/s or
                                    m/s). The spring term is disabled
                                    (``stiffness_drive = 0``); the
                                    caller must supply
                                    ``damping_drive > 0``, which acts
                                    as the proportional gain on
                                    velocity error. The iterate
                                    collapses to
                                    ``lam = -(1/(1/eff_inv +
                                    kd/dt))*(jv + v_target)`` --
                                    i.e. the canonical PD velocity
                                    servo with time constant ``tau =
                                    I_eff / damping_drive``. There is
                                    *no* rigid pure-velocity-motor
                                    fallback.

All three modes share the same scalar PD row (:func:`pd_coefficients`
+ :func:`_axial_drive_limit_iterate`); the only difference is which of
``target`` / ``target_velocity`` is folded into the bias. The
standalone :mod:`constraint_angular_motor` / :mod:`constraint_linear_motor`
constraints expose the same contract.

Plus a unilateral ``[min_value, max_value]`` spring-damper limit
(``min_value > max_value`` disables it) with a *dual* softness
parameterisation:

* ``stiffness_limit == damping_limit == 0`` -- Box2D
  ``(hertz_limit, damping_ratio_limit)`` soft constraint.
* either ``stiffness_limit > 0`` or ``damping_limit > 0`` -- Jitter2
  PD spring-damper with absolute SI gains.

Same discriminator as the standalone :mod:`constraint_angular_limit`
and :mod:`constraint_linear_limit` modules. Drive (always PD) and
limit share the same scalar PGS row and can both be active at once;
the limit is unilateral and always wins.

XPBD reference: this is the PGS analogue of Section 3.3.2 / 3.4.1 of
*Detailed Rigid Body Simulation with Extended Position Based Dynamics*
(M\u00fcller et al.). The revolute twist uses
:func:`math_helpers.extract_rotation_angle` plus a revolution counter
for unbounded cumulative angles (matching
:mod:`constraint_angular_motor` / :mod:`constraint_angular_limit`);
the prismatic slide reads directly off the world anchor positions so
no quaternion math is needed there.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
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
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_mat33,
    write_mat44,
    write_quat,
    write_vec3,
    write_vec4,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import (
    create_orthonormal,
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)

__all__ = [
    "ADBS_DWORDS",
    "ActuatedDoubleBallSocketData",
    "DRIVE_MODE_OFF",
    "DRIVE_MODE_POSITION",
    "DRIVE_MODE_VELOCITY",
    "JOINT_MODE_BALL_SOCKET",
    "JOINT_MODE_PRISMATIC",
    "JOINT_MODE_REVOLUTE",
    "actuated_double_ball_socket_initialize_kernel",
    "actuated_double_ball_socket_iterate",
    "actuated_double_ball_socket_iterate_at",
    "actuated_double_ball_socket_prepare_for_iteration",
    "actuated_double_ball_socket_prepare_for_iteration_at",
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

    The struct is a *union* over the two joint modes: revolute uses the
    ``a1_inv / ut_ai / s_inv`` Schur cache, prismatic uses the
    ``a4_inv / c_pris / s_scalar_inv`` Schur cache. Both modes share
    every other slot (anchors, lever arms, warm-start impulses,
    drive / limit scalars), which is what lets the dispatcher treat
    them as one constraint type with no extra branch per iteration.

    Layout groups (in storage order):

    * **Header** -- ``constraint_type / body1 / body2`` (required by
      the dispatcher contract).
    * **Shared positional block** -- two user-supplied anchors on the
      joint axis, their body-local snapshots, runtime lever arms,
      cached tangent basis, and the rest-pose relative orientation
      (for the revolute twist extraction).
    * **Revolute-only Schur cache** -- ``a1_inv, ut_ai, s_inv`` (3x3,
      3x3, 3x3 with a packed 2x2). Written by :func:`_revolute_prepare_at`,
      read by :func:`_revolute_iterate_at`.
    * **Prismatic-only Schur cache** -- ``a4_inv (mat44f)``,
      ``c_pris (vec4f)``, ``s_scalar_inv (float)``, plus the anchor-3
      body-local snapshots / world lever arms. Written by
      :func:`_prismatic_prepare_at`, read by :func:`_prismatic_iterate_at`.
    * **Warm-start** -- per-anchor accumulated impulses in world frame:
      three ``vec3f`` slots (``acc_imp1, acc_imp2, acc_imp3``) are
      enough to cover both modes without dword-stomping (revolute uses
      ``acc_imp1`` as a full vec3 and ``acc_imp2`` as a tangent-only
      vec3 with last component 0; prismatic uses all three as
      tangent-only vec3s).
    * **Actuator block** -- drive / limit setpoints, cached soft-
      constraint coefficients, scalar accumulated impulses.

    Storage cost: ~80 dwords (about 320 B per joint). That's ~45%
    larger than the old ActuatedDoubleBallSocket (~55 dwords) but is a
    net LoC save vs keeping three separate constraint types.
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
    # Anchor 3 body-local snapshots: only meaningful in prismatic mode
    # (ignored by revolute). Auto-derived at init as
    # ``anchor1 + rest_length * t_ref`` with ``t_ref`` an arbitrary
    # unit perpendicular to the slide axis.
    local_anchor3_b1: wp.vec3f
    local_anchor3_b2: wp.vec3f
    # Runtime (per-substep) lever arms for the three anchors.
    r1_b1: wp.vec3f
    r1_b2: wp.vec3f
    r2_b1: wp.vec3f
    r2_b2: wp.vec3f
    r3_b1: wp.vec3f
    r3_b2: wp.vec3f
    # Runtime tangent basis perpendicular to the current world joint axis.
    t1: wp.vec3f
    t2: wp.vec3f
    # Positional soft-constraint knobs + cached per-substep coefficients.
    hertz: wp.float32
    damping_ratio: wp.float32
    bias_rate: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32
    # Positional biases.
    # Revolute: ``bias1`` = 3-vec world drift at anchor 1; ``bias2`` =
    #   tangent drift at anchor 2 packed into ``(t1, t2, 0)``.
    # Prismatic: ``bias1`` = tangent drift at anchor 1 packed into
    #   ``(t1, t2, 0)``; ``bias2`` = tangent drift at anchor 2 packed
    #   into ``(t1, t2, 0)``. ``bias3`` = scalar drift at anchor 3
    #   along ``t2``.
    bias1: wp.vec3f
    bias2: wp.vec3f
    bias3: wp.float32
    # Revolute Schur cache (ignored by prismatic).
    a1_inv: wp.mat33f
    ut_ai: wp.mat33f
    s_inv: wp.mat33f
    # Prismatic Schur cache (ignored by revolute).
    a4_inv: wp.mat44f
    c_pris: wp.vec4f
    s_scalar_inv: wp.float32
    # Warm-start accumulated impulses (world frame).
    accumulated_impulse1: wp.vec3f
    accumulated_impulse2: wp.vec3f
    accumulated_impulse3: wp.vec3f

    # ---- Actuator + limit block --------------------------------------
    # Body-1-local joint axis snapshot. Used by revolute for a
    # single-axis Jacobian (matching the standalone angular motor /
    # angular limit's PD path) and by the world_wrench helper. The
    # companion 5-DoF positional lock keeps body 2's axis parallel, so
    # one axis is both simpler and more numerically stable than the
    # old two-axis projection.
    axis_local1: wp.vec3f
    rest_length: wp.float32
    # Rest relative orientation used by the revolute twist tracker:
    # ``diff = q2 * inv_initial_orientation * q1^*`` is identity at
    # finalize() time, and :func:`extract_rotation_angle` projected
    # onto ``j = quat_rotate(q1, axis_local1)`` gives the in-branch
    # twist in ``(-pi, pi]``. The revolution counter wraps that into
    # an unbounded cumulative angle, matching the standalone angular
    # motor / angular limit.
    inv_initial_orientation: wp.quatf
    revolution_counter: wp.int32
    previous_quaternion_angle: wp.float32
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
    # Box2D-only limit cache (stiffness_limit == damping_limit == 0).
    bias_limit_box2d: wp.float32
    mass_coeff_limit: wp.float32
    impulse_coeff_limit: wp.float32
    # PD-only limit cache (stiffness_limit > 0 or damping_limit > 0).
    pd_gamma_limit: wp.float32
    pd_beta_limit: wp.float32
    pd_mass_coeff_limit: wp.float32
    max_lambda_drive: wp.float32
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
_OFF_LA3_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor3_b1"))
_OFF_LA3_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "local_anchor3_b2"))
_OFF_R1_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r1_b1"))
_OFF_R1_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r1_b2"))
_OFF_R2_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r2_b1"))
_OFF_R2_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r2_b2"))
_OFF_R3_B1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r3_b1"))
_OFF_R3_B2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "r3_b2"))
_OFF_T1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "t2"))
_OFF_HERTZ = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio"))
_OFF_BIAS_RATE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_rate"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff"))
_OFF_BIAS1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias2"))
_OFF_BIAS3 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias3"))
_OFF_A1_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "a1_inv"))
_OFF_UT_AI = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "ut_ai"))
_OFF_S_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "s_inv"))
_OFF_A4_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "a4_inv"))
_OFF_C_PRIS = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "c_pris"))
_OFF_S_SCALAR_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "s_scalar_inv"))
_OFF_ACC_IMP1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse1"))
_OFF_ACC_IMP2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse2"))
_OFF_ACC_IMP3 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse3"))

_OFF_AXIS_LOCAL1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_local1"))
_OFF_REST_LENGTH = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "rest_length"))
_OFF_INV_INITIAL_ORIENTATION = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "inv_initial_orientation")
)
_OFF_REVOLUTION_COUNTER = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "revolution_counter")
)
_OFF_PREVIOUS_QUATERNION_ANGLE = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "previous_quaternion_angle")
)
_OFF_DRIVE_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "drive_mode"))
_OFF_TARGET = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target"))
_OFF_TARGET_VELOCITY = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target_velocity"))
_OFF_MAX_FORCE_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_force_drive"))
_OFF_STIFFNESS_DRIVE = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_drive")
)
_OFF_DAMPING_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_drive"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_value"))
_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz_limit"))
_OFF_DAMPING_RATIO_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio_limit")
)
_OFF_STIFFNESS_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "stiffness_limit")
)
_OFF_DAMPING_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "damping_limit")
)
_OFF_EFF_INV_AXIAL = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "eff_inv_axial")
)
_OFF_BIAS_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_drive"))
_OFF_GAMMA_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "gamma_drive"))
_OFF_EFF_MASS_DRIVE_SOFT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "eff_mass_drive_soft")
)
_OFF_BIAS_LIMIT_BOX2D = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "bias_limit_box2d")
)
_OFF_MASS_COEFF_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff_limit")
)
_OFF_IMPULSE_COEFF_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff_limit")
)
_OFF_PD_GAMMA_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "pd_gamma_limit")
)
_OFF_PD_BETA_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "pd_beta_limit")
)
_OFF_PD_MASS_COEFF_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "pd_mass_coeff_limit")
)
_OFF_MAX_LAMBDA_DRIVE = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "max_lambda_drive")
)
_OFF_CLAMP = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "clamp"))
_OFF_AXIS_WORLD = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_world"))
_OFF_ACC_DRIVE = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_drive")
)
_OFF_ACC_LIMIT = wp.constant(
    dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_limit")
)

#: Total dword count of one unified joint constraint.
ADBS_DWORDS: int = num_dwords(ActuatedDoubleBallSocketData)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@wp.kernel
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
):
    """Pack one batch of unified joint descriptors.

    For both joint modes, ``anchor1`` and ``anchor2`` are two user-
    supplied world-space points on the joint axis. Revolute interprets
    the line through them as the hinge axis; prismatic interprets it as
    the slide axis.

    For prismatic mode the init kernel *auto-derives* a third anchor
    ``a3 = anchor1 + rest_length * t_ref`` where ``t_ref`` is an
    arbitrary unit perpendicular to ``n_hat_init`` and
    ``rest_length = |anchor2 - anchor1|``. This third anchor is
    snapshotted into both bodies' local frames so the runtime math can
    rotate it into world space per substep -- no user-visible API
    change compared with the two-anchor interface.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only ``position`` / ``orientation``
            of the referenced bodies are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor1: World-space first anchor [num_in_batch] [m].
        anchor2: World-space second anchor [num_in_batch] [m];
            ``anchor2 - anchor1`` defines the joint axis.
        hertz: Positional Schur block soft-constraint frequency [Hz].
        damping_ratio: Positional Schur block damping ratio.
        joint_mode: :data:`JOINT_MODE_REVOLUTE` or
            :data:`JOINT_MODE_PRISMATIC`.
        drive_mode: :data:`DRIVE_MODE_OFF` / :data:`DRIVE_MODE_POSITION` /
            :data:`DRIVE_MODE_VELOCITY`.
        target: Position-drive setpoint [rad] (revolute) or [m]
            (prismatic).
        target_velocity: Velocity-drive setpoint [rad/s] or [m/s].
        max_force_drive: Drive impulse cap [N*m] (revolute) or [N]
            (prismatic); 0 disables the drive even if mode != OFF.
        stiffness_drive: Drive PD stiffness ``kp`` [N*m/rad] (revolute)
            or [N/m] (prismatic).
        damping_drive: Drive PD damping ``kd`` [N*m*s/rad] (revolute)
            or [N*s/m] (prismatic). ``stiffness_drive ==
            damping_drive == 0`` disables the drive row.
        min_value / max_value: Limit window [rad] or [m]. ``min_value
            > max_value`` disables the limit (matches the standalone
            angular_limit / linear_limit sentinel).
        hertz_limit / damping_ratio_limit: Box2D limit soft-constraint
            knobs; used iff ``stiffness_limit == damping_limit == 0``.
        stiffness_limit / damping_limit: PD limit soft-constraint
            gains (``kp``, ``kd``) in absolute SI units. When either
            is strictly positive the limit uses the Jitter2
            spring-damper path and ignores ``hertz_limit`` /
            ``damping_ratio_limit``. Matches the standalone
            angular_limit / linear_limit's dual-convention dispatch.
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
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
    write_int(constraints, _OFF_JOINT_MODE, cid, joint_mode[tid])
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
    write_vec3(constraints, _OFF_T1, cid, zero3)
    write_vec3(constraints, _OFF_T2, cid, zero3)
    write_vec3(constraints, _OFF_BIAS1, cid, zero3)
    write_vec3(constraints, _OFF_BIAS2, cid, zero3)
    write_float(constraints, _OFF_BIAS3, cid, 0.0)
    write_vec3(constraints, _OFF_ACC_IMP1, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP2, cid, zero3)
    write_vec3(constraints, _OFF_ACC_IMP3, cid, zero3)

    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_BIAS_RATE, cid, 0.0)
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

    # Actuator block.
    write_vec3(constraints, _OFF_AXIS_LOCAL1, cid, axis_local1)
    write_float(constraints, _OFF_REST_LENGTH, cid, rest_length)
    write_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid, inv_initial_orientation)
    write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, 0)
    write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, 0.0)
    write_int(constraints, _OFF_DRIVE_MODE, cid, drive_mode[tid])
    write_float(constraints, _OFF_TARGET, cid, target[tid])
    write_float(constraints, _OFF_TARGET_VELOCITY, cid, target_velocity[tid])
    write_float(constraints, _OFF_MAX_FORCE_DRIVE, cid, max_force_drive[tid])
    write_float(constraints, _OFF_STIFFNESS_DRIVE, cid, stiffness_drive[tid])
    write_float(constraints, _OFF_DAMPING_DRIVE, cid, damping_drive[tid])
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
    write_float(constraints, _OFF_BIAS_LIMIT_BOX2D, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF_LIMIT, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_PD_GAMMA_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_PD_BETA_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_PD_MASS_COEFF_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_MAX_LAMBDA_DRIVE, cid, 0.0)
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
    max_lambda_drive = read_float(constraints, base_offset + _OFF_MAX_LAMBDA_DRIVE, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
    gamma_drive = read_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid)
    eff_mass_drive_soft = read_float(
        constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid
    )
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
        lam_drive = -eff_mass_drive_soft * (
            jv_axial - bias_drive + gamma_drive * acc_drive
        )
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
    anchor-1 lock is built. No anchor 2, no tangent basis, no axial
    row, no Schur complement -- the effective mass is just the
    anchor-1 3x3 block ``A1`` and the positional solve is a single
    direct inverse. Warm-starts the 3-row positional impulse.
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

    r1_b1 = wp.quat_rotate(orientation1, la1_b1)
    r1_b2 = wp.quat_rotate(orientation2, la1_b2)

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
    write_float(constraints, base_offset + _OFF_BIAS_RATE, cid, bias_rate)
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
        t1[0], t2[0], 0.0,
        t1[1], t2[1], 0.0,
        t1[2], t2[2], 0.0,
    )
    tt = wp.transpose(t_mat)

    u_mat = b_mat @ t_mat
    d_mat = tt @ (a2 @ t_mat)

    a1_inv = wp.inverse(a1)
    ut_ai = wp.transpose(u_mat) @ a1_inv
    s_mat = d_mat - ut_ai @ u_mat

    s22 = wp.mat22f(
        s_mat[0, 0], s_mat[0, 1],
        s_mat[1, 0], s_mat[1, 1],
    )
    s22_inv = wp.inverse(s22)
    s_inv_packed = wp.mat33f(
        s22_inv[0, 0], s22_inv[0, 1], 0.0,
        s22_inv[1, 0], s22_inv[1, 1], 0.0,
        0.0,           0.0,           0.0,
    )

    write_mat33(constraints, base_offset + _OFF_A1_INV, cid, a1_inv)
    write_mat33(constraints, base_offset + _OFF_UT_AI, cid, ut_ai)
    write_mat33(constraints, base_offset + _OFF_S_INV, cid, s_inv_packed)

    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    dt = 1.0 / idt
    bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
    write_float(constraints, base_offset + _OFF_BIAS_RATE, cid, bias_rate)
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
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (
        cr1_b1 @ acc1 + cr2_b1 @ acc2
    )
    velocity2 = bodies.velocity[b2] + inv_mass2 * (acc1 + acc2)
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (
        cr1_b2 @ acc1 + cr2_b2 @ acc2
    )

    # ---- Axial drive + limit block (angular) ------------------------
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    target = read_float(constraints, base_offset + _OFF_TARGET, cid)
    target_velocity = read_float(constraints, base_offset + _OFF_TARGET_VELOCITY, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    stiffness_drive = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    damping_drive = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)

    # Scalar inverse effective mass for the axial row. The axial
    # impulse in the iterate is applied along ``n_hat`` (the anchor-
    # derived axis) on both bodies, so the Jacobian rows are
    # ``(0; +n_hat)`` and ``(0; -n_hat)`` and the unconstrained
    # inverse effective mass is ``n . (I1^-1 + I2^-1) . n``. Identical
    # to the standalone angular motor / angular limit's formula once
    # the 5-DoF positional lock aligns both body axes with n_hat.
    eff_inv = wp.dot(n_hat, inv_inertia1 @ n_hat) + wp.dot(n_hat, inv_inertia2 @ n_hat)
    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, eff_inv)

    # Revolute twist tracker: ``diff = q2 * inv_init * q1^*`` is the
    # identity at finalize() time, and :func:`extract_rotation_angle`
    # projected onto the *body-1* axis returns the signed angle in
    # ``(-pi, pi]``; the revolution counter extends that to an
    # unbounded cumulative angle. Byte-for-byte identical to the
    # standalone angular motor / angular limit prepare. We use the
    # body-1 local-axis snapshot here rather than ``n_hat`` because
    # the tracker output must remain well-defined even if anchors
    # briefly coincide across a substep (`|a2 - a1| ~ 0` would zero
    # out ``n_hat`` for a step).
    j1 = wp.quat_rotate(
        orientation1, read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL1, cid)
    )
    inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
    diff = orientation2 * inv_init * wp.quat_inverse(orientation1)
    new_q_angle = extract_rotation_angle(diff, j1)
    old_counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
    old_prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
    new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
    write_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid, new_counter)
    write_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
    cumulative_angle = revolution_tracker_angle(new_counter, new_prev)

    # ---- Drive (PD only) ---------------------------------------------
    # The unified joint exposes a *single* PD drive model; there is no
    # rigid velocity-servo fallback (the Jitter2-style pure-velocity
    # motor it replaces is a degenerate rigid constraint whose
    # stability depends on substep size, not on any gain the user
    # controls). Velocity tracking is recovered by setting
    # ``stiffness_drive == 0`` and ``damping_drive > 0``: the iterate
    # then reduces to ``lam = -(1 / (1/eff_inv + kd/dt)) * (jv +
    # target_velocity)``, i.e. the canonical PD velocity servo with a
    # well-defined time constant ``tau = I_eff / damping_drive``. See
    # :mod:`constraint_angular_motor` for the same contract on the
    # standalone motor side.
    #
    # ``C_drive = cumulative_angle - target`` for position mode; for
    # velocity mode the spring term vanishes (``stiffness_drive = 0``
    # kills the kp term inside :func:`pd_coefficients`) and only the
    # damping row acts.
    drive_C = float(0.0)
    if drive_mode == DRIVE_MODE_POSITION:
        drive_C = cumulative_angle - target
    if stiffness_drive > 0.0 or damping_drive > 0.0:
        gamma_drive, bias_drive, eff_mass_drive_soft = pd_coefficients(
            stiffness_drive, damping_drive, drive_C, eff_inv, dt
        )
    else:
        # Drive fully off (mode == OFF, or POSITION with zero gains,
        # or VELOCITY with zero damping). The iterate sees
        # ``eff_mass_drive_soft == 0`` and short-circuits.
        gamma_drive = wp.float32(0.0)
        bias_drive = wp.float32(0.0)
        eff_mass_drive_soft = wp.float32(0.0)
    # ``target_velocity`` is the body-2 side's commanded spin rate
    # about ``+n_hat`` (positive = body 2 spins in +n direction
    # relative to body 1). Our ``jv_axial = n . (w1 - w2)`` sign
    # convention is the negative of Jitter2's, so we fold
    # ``-target_velocity`` into the bias; the iterate's
    # ``jv - bias_drive`` then expands to ``n.(w1 - w2) +
    # target_velocity`` -- i.e. exactly zero when body 2 spins at the
    # commanded rate, which is the intuitive target-reached condition.
    bias_drive = bias_drive - target_velocity
    write_float(constraints, base_offset + _OFF_GAMMA_DRIVE, cid, gamma_drive)
    write_float(
        constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, eff_mass_drive_soft
    )
    write_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid, bias_drive)

    max_lambda_drive = max_force_drive * dt
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA_DRIVE, cid, max_lambda_drive)

    # ---- Limit (dual convention) -------------------------------------
    # Determine the limit clamp state. We compare the cumulative angle
    # (from the revolution tracker) directly against the user's
    # ``min_value`` / ``max_value`` to stay monotone for arbitrarily
    # large ``|min|`` / ``|max|`` -- matches the standalone
    # angular_limit's clamp selection exactly. ``min_value >
    # max_value`` disables the limit row.
    clamp = _CLAMP_NONE
    limit_C = float(0.0)
    if min_value <= max_value:
        if cumulative_angle > max_value:
            clamp = _CLAMP_MAX
            limit_C = cumulative_angle - max_value
        elif cumulative_angle < min_value:
            clamp = _CLAMP_MIN
            limit_C = cumulative_angle - min_value
    write_int(constraints, base_offset + _OFF_CLAMP, cid, clamp)

    # Dual-convention dispatch: PD path if either gain is strictly
    # positive, else Box2D. Both branches populate both caches (with
    # the inactive one zeroed) so iterate doesn't have to guard on
    # stale state.
    if stiffness_limit > 0.0 or damping_limit > 0.0:
        pd_gamma_limit, pd_beta_limit, pd_m_soft = pd_coefficients(
            stiffness_limit, damping_limit, limit_C, eff_inv, dt
        )
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, pd_gamma_limit)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, pd_beta_limit)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, pd_m_soft)
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, 0.0)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, 0.0)
    else:
        br_limit, mc_limit, ic_limit = soft_constraint_coefficients(
            hertz_limit, damping_ratio_limit, dt
        )
        # Box2D sign convention: iterate drives ``jv == -bias``, so
        # prefold the positional error with a flipped sign.
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, -limit_C * br_limit)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mc_limit)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, ic_limit)
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, 0.0)

    # Warm-start the axial impulses. Drop any lingering limit impulse
    # when the limit is inactive, matching the standalone modules.
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    if clamp == _CLAMP_NONE:
        acc_limit = 0.0
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    axial_imp = acc_drive + acc_limit
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
        s_inv_packed[0, 0], s_inv_packed[0, 1],
        s_inv_packed[1, 0], s_inv_packed[1, 1],
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
    axial_lam = _axial_drive_limit_iterate(
        constraints, cid, base_offset, jv_axial, clamp
    )
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
# Rank-5 pure-points formulation, 2+2+1 rows:
#
#   Rows 0-1: tangent drift at anchor 1 projected onto (t1, t2).
#   Rows 2-3: tangent drift at anchor 2 projected onto (t1, t2).
#   Row  4 : scalar drift at anchor 3 projected onto t2. This kills
#            the last rotational DoF (rotation about n_hat).
#
# The 5x5 effective-mass matrix is block-structured as
#
#     K = [ K4     c  ]    K4 in R^{4x4}  (the two anchor-tangent pairs)
#         [ c^T    d  ]    c  in R^{4}    (cross-coupling a3 <-> a1/a2)
#                          d  in R        (a3 self-coupling)
#
# Schur eliminate the scalar anchor-3 row first:
#
#   s_inv = 1 / (d - c^T K4^{-1} c)
#   lam3  = -s_inv * (rhs3 - c^T K4^{-1} rhs4)
#   lam4  = -K4^{-1} * (rhs4 + c * lam3)
#
# This uses exactly one wp.inverse(mat44f) per prepare pass and no
# per-iter inverses.


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

    # Tangent basis: t1 must be aligned with the anchor3 lever arm's
    # component perpendicular to n_hat, so that rotation of body 2 around
    # n_hat produces motion at anchor 3 along t2 with full gain. Using an
    # arbitrary perpendicular (``create_orthonormal``) yields a ``cos(alpha)``
    # gain mismatch that breaks Gauss-Seidel convergence in chains with
    # two or more joints sharing a body. We use body 1's view of the
    # anchor1 -> anchor3 direction (projected perpendicular to n_hat) as
    # the t1 reference; this makes the anchor-3 scalar row the exact
    # tangential velocity gate for rotation about n_hat.
    anchor3_offset_b1 = r3_b1 - r1_b1
    t1_raw = anchor3_offset_b1 - wp.dot(anchor3_offset_b1, n_hat) * n_hat
    t1_len2 = wp.dot(t1_raw, t1_raw)
    if t1_len2 > 1.0e-20:
        t1 = t1_raw / wp.sqrt(t1_len2)
    else:
        # Fallback: anchor3 has drifted onto the axis (shouldn't happen
        # since rest_length > 0 and bodies can't collapse).
        t1 = create_orthonormal(n_hat)
    t2 = wp.cross(n_hat, t1)
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
        k4_00, k4_01, k4_02, k4_03,
        k4_01, k4_11, k4_12, k4_13,
        k4_02, k4_12, k4_22, k4_23,
        k4_03, k4_13, k4_23, k4_33,
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
    write_float(constraints, base_offset + _OFF_BIAS_RATE, cid, bias_rate)
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
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    target = read_float(constraints, base_offset + _OFF_TARGET, cid)
    target_velocity = read_float(constraints, base_offset + _OFF_TARGET_VELOCITY, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    stiffness_drive = read_float(constraints, base_offset + _OFF_STIFFNESS_DRIVE, cid)
    damping_drive = read_float(constraints, base_offset + _OFF_DAMPING_DRIVE, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    stiffness_limit = read_float(constraints, base_offset + _OFF_STIFFNESS_LIMIT, cid)
    damping_limit = read_float(constraints, base_offset + _OFF_DAMPING_LIMIT, cid)

    # Linear inverse effective mass at anchor 1 along n_hat:
    #   m_inv = n . B11 . n  (same quadratic form as the tangent block,
    #                         but projected onto n_hat instead of t).
    b11_n = b11 @ n_hat
    eff_inv = wp.dot(n_hat, b11_n)
    write_float(constraints, base_offset + _OFF_EFF_INV_AXIAL, cid, eff_inv)

    # Slide along n_hat measured at anchor 1. Starts at 0 at init (anchors
    # coincident), so this is directly the relative slide in meters.
    # Sign convention: iterate applies ``+v1 * n * lam`` and
    # ``-v2 * n * lam`` (with lever-arm angular terms), so positive
    # lambda *decreases* the slide (body 2 is pushed back along +n,
    # body 1 pushed forward). ``slide = p2 - p1`` along n_hat, so
    # C_drive = slide - target has the correct sign for
    # "lam > 0 drives slide toward target".
    slide = wp.dot(n_hat, drift1)

    # ---- Drive (PD only) ---------------------------------------------
    # Same PD-only contract as the revolute prepare; see that docstring
    # for the ``-target_velocity`` sign-fold explanation. Velocity
    # tracking is recovered by ``stiffness_drive = 0`` +
    # ``damping_drive > 0`` (explicit damping gain); there is no
    # rigid-servo fallback.
    drive_C = float(0.0)
    if drive_mode == DRIVE_MODE_POSITION:
        drive_C = slide - target
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
    write_float(
        constraints, base_offset + _OFF_EFF_MASS_DRIVE_SOFT, cid, eff_mass_drive_soft
    )
    write_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid, bias_drive)

    max_lambda_drive = max_force_drive * dt
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA_DRIVE, cid, max_lambda_drive)

    # ---- Limit (dual convention) -------------------------------------
    # ``min_value > max_value`` disables the limit row (matches the
    # standalone linear_limit sentinel).
    clamp = _CLAMP_NONE
    limit_C = float(0.0)
    if min_value <= max_value:
        if slide > max_value:
            clamp = _CLAMP_MAX
            limit_C = slide - max_value
        elif slide < min_value:
            clamp = _CLAMP_MIN
            limit_C = slide - min_value
    write_int(constraints, base_offset + _OFF_CLAMP, cid, clamp)

    # Dual-convention dispatch identical to the revolute prepare /
    # standalone linear_limit: PD if either gain > 0, else Box2D.
    if stiffness_limit > 0.0 or damping_limit > 0.0:
        pd_gamma_limit, pd_beta_limit, pd_m_soft = pd_coefficients(
            stiffness_limit, damping_limit, limit_C, eff_inv, dt
        )
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, pd_gamma_limit)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, pd_beta_limit)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, pd_m_soft)
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, 0.0)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, 0.0)
    else:
        br_limit, mc_limit, ic_limit = soft_constraint_coefficients(
            hertz_limit, damping_ratio_limit, dt
        )
        write_float(constraints, base_offset + _OFF_BIAS_LIMIT_BOX2D, cid, -limit_C * br_limit)
        write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mc_limit)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, ic_limit)
        write_float(constraints, base_offset + _OFF_PD_GAMMA_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA_LIMIT, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF_LIMIT, cid, 0.0)

    # Warm-start the axial impulses (linear). Drop any lingering
    # limit impulse when the limit is inactive.
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    if clamp == _CLAMP_NONE:
        acc_limit = 0.0
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, 0.0)
    axial_imp = acc_drive + acc_limit
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
    axial_lam = _axial_drive_limit_iterate(
        constraints, cid, base_offset, jv_axial, clamp
    )

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
    else:
        _ball_socket_prepare_at(constraints, cid, base_offset, bodies, body_pair, idt)


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
        torque = (
            wp.cross(r1_b2, acc1 * idt)
            + wp.cross(r2_b2, acc2 * idt)
            + wp.cross(r3_b2, acc3 * idt)
        )
        # Axial block is a linear force along -n_hat.
        axial_force = n_hat * ((acc_drive + acc_limit) * idt)
        force = force - axial_force
        torque = torque - wp.cross(r1_b2, axial_force)
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
    actuated_double_ball_socket_prepare_for_iteration_at(
        constraints, cid, 0, bodies, body_pair, idt
    )


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
