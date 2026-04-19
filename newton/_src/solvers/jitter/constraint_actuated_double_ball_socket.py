# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actuated variant of the fused two-anchor "double ball-socket" hinge.

Layered on top of :mod:`constraint_double_ball_socket`: the same 5-DoF
Schur-complement lock (3 translational + 2 rotational) plus an extra
*scalar* PGS row that controls the only free DoF -- the relative twist
about the implicit hinge axis ``n_hat = (anchor2 - anchor1) / |...|``.
The extra row is a Box2D / Bepu / Nordby soft-constraint just like
every other constraint in the solver, with three independent
sub-blocks:

* **Drive** -- pick one of three modes via :data:`DRIVE_MODE_*`:

  * :data:`DRIVE_MODE_OFF`         -- no torque, axis is free.
  * :data:`DRIVE_MODE_POSITION`    -- soft spring towards a target
                                     ``target_angle`` [rad]; "stiffness"
                                     and "damping" come from the
                                     drive's ``hertz_drive`` /
                                     ``damping_ratio_drive``.
  * :data:`DRIVE_MODE_VELOCITY`    -- soft tracking of a target
                                     ``target_velocity`` [rad/s];
                                     ``max_force_drive`` caps the
                                     per-substep torque ``[N*m]``.

  Both drives share the same scalar PGS row, the only thing that
  changes is the bias / impulse-cap formulation. This mirrors what
  PhysX's joint drives and Bepu's PD drives do and makes runtime
  switching cheap (just rewrite the mode + setpoint, no constraint
  rebuild).

* **Limits** -- a one-sided spring-damper on the axial twist confined
  to ``[min_angle, max_angle]`` [rad]. ``min_angle == max_angle == 0``
  disables the limit (axis is free apart from the drive). Same Hertz /
  damping-ratio knobs as :mod:`constraint_hinge_angle`'s limit row.
  The limit and the drive can be active simultaneously -- the limit
  always wins by being unilateral (clamps its own accumulated impulse
  to one sign), so a position drive past the limit cleanly seats the
  body against the stop.

The 5-DoF positional Schur solve is unchanged from
:mod:`constraint_double_ball_socket`; this module just appends the
axial scalar block to the same constraint column. Storage layout,
header contract and dispatch contract follow the rest of the
constraint modules in this directory.

XPBD reference: this is the PGS analogue of Section 3.3.2 / 3.4.1 of
*Detailed Rigid Body Simulation with Extended Position Based Dynamics*
(M\u00fcller et al.). The angle extraction (signed twist about the common
axis) is the same as :mod:`constraint_hinge_angle`'s third row -- via
the closed-form quaternion-error projection rather than the paper's
``2 * arcsin(...)`` form, which is equivalent for the small angles that
PGS / XPBD operate on between substeps.
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
from newton._src.solvers.jitter.math_helpers import create_orthonormal

__all__ = [
    "ADBS_DWORDS",
    "ActuatedDoubleBallSocketData",
    "DRIVE_MODE_OFF",
    "DRIVE_MODE_POSITION",
    "DRIVE_MODE_VELOCITY",
    "actuated_double_ball_socket_initialize_kernel",
    "actuated_double_ball_socket_iterate",
    "actuated_double_ball_socket_iterate_at",
    "actuated_double_ball_socket_prepare_for_iteration",
    "actuated_double_ball_socket_prepare_for_iteration_at",
    "actuated_double_ball_socket_world_wrench",
    "actuated_double_ball_socket_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Drive-mode tags (host-visible: mirrored as plain Python ints so the
# WorldBuilder descriptor can pass them through ``np.int32`` arrays
# without colliding with Warp's wp.constant typing).
# ---------------------------------------------------------------------------

#: No torque applied along the hinge axis -- the third rotational DoF
#: stays free, which matches the parent
#: :mod:`constraint_double_ball_socket` exactly.
DRIVE_MODE_OFF = wp.constant(wp.int32(0))
#: Soft spring towards :func:`actuated_double_ball_socket_get_target_angle`
#: with stiffness derived from ``hertz_drive`` / ``damping_ratio_drive``.
#: Per-substep torque is uncapped (``max_force_drive`` is ignored).
DRIVE_MODE_POSITION = wp.constant(wp.int32(1))
#: Soft tracking of :func:`actuated_double_ball_socket_get_target_velocity`.
#: ``max_force_drive`` caps the per-substep impulse to
#: ``\u00b1 max_force_drive * dt`` [N*m].
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
    """Per-constraint dword-layout schema for the actuated fused hinge.

    Same conventions as :class:`DoubleBallSocketData`; this struct
    extends it with one extra scalar PGS block for the axial drive +
    limit. Naming convention for the new block:

    * ``axis_local{1,2}``  -- hinge axis snapshotted into each body's
      local frame at init. The runtime hinge axis comes from the
      anchors (same as the parent), but having a body-fixed axis lets
      us define a stable signed-twist sign convention; see the
      ``q0`` / ``axis_local2`` derivation in the prepare-pass.
    * ``q0``               -- ``q2_init^* * q1_init`` (rest relative
      orientation). Used together with ``axis_local2`` for the closed-
      form signed-twist extraction.
    * ``drive_mode``       -- one of :data:`DRIVE_MODE_*`.
    * ``target_angle``     -- [rad]. Used in :data:`DRIVE_MODE_POSITION`.
    * ``target_velocity``  -- [rad/s]. Used in :data:`DRIVE_MODE_VELOCITY`.
    * ``max_force_drive``  -- [N*m]. Caps the per-substep drive impulse;
      0 disables the drive even if mode != OFF.
    * ``hertz_drive`` / ``damping_ratio_drive`` -- soft-constraint knobs
      governing the drive's stiffness / damping (same Box2D / Bepu /
      Nordby formulation as the rest of the solver).
    * ``min_angle`` / ``max_angle`` -- [rad] axial twist limits.
      ``min_angle == max_angle == 0`` -> limit disabled.
    * ``hertz_limit`` / ``damping_ratio_limit`` -- soft-constraint knobs
      for the unilateral spring-damper that enforces the limit.
    * ``effective_mass_axial``      -- ``1 / (n . (I1^{-1} + I2^{-1}) n)``.
      Cached scalar, recomputed each prepare-pass.
    * ``bias_drive``                -- cached drive bias [rad/s].
    * ``bias_limit``                -- cached limit bias [rad/s] (signed
      so the iterate-path clamp picks the correct stop).
    * ``mass_coeff_drive`` / ``impulse_coeff_drive`` -- per-substep
      Box2D / Bepu coefficients for the drive row.
    * ``mass_coeff_limit`` / ``impulse_coeff_limit`` -- ditto for the
      limit row.
    * ``max_lambda_drive``          -- ``max_force_drive * dt`` [N*m*s].
      Symmetric clamp.
    * ``clamp``                     -- :data:`_CLAMP_*` flag for the
      limit, cached from the prepare-pass.
    * ``axis_world``                -- cached world-frame hinge axis at
      the most recent prepare-pass. Used by both iterate and
      world-wrench.
    * ``accumulated_impulse_drive`` -- scalar PGS warm-start for the
      drive row [N*m*s].
    * ``accumulated_impulse_limit`` -- ditto for the limit row.

    All fields not listed above are inherited 1:1 from
    :class:`DoubleBallSocketData`; comments there apply unchanged.
    """

    # ---- Header (mandatory for every constraint schema) --------------
    constraint_type: wp.int32
    body1: wp.int32
    body2: wp.int32

    # ---- Inherited ball-socket-style 5-DoF block (identical layout
    # ---- to DoubleBallSocketData; see that struct's docstring) -------
    local_anchor1_b1: wp.vec3f
    local_anchor1_b2: wp.vec3f
    local_anchor2_b1: wp.vec3f
    local_anchor2_b2: wp.vec3f
    r1_b1: wp.vec3f
    r1_b2: wp.vec3f
    r2_b1: wp.vec3f
    r2_b2: wp.vec3f
    t1: wp.vec3f
    t2: wp.vec3f
    hertz: wp.float32
    damping_ratio: wp.float32
    bias_rate: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32
    bias1: wp.vec3f
    bias2: wp.vec3f
    a1_inv: wp.mat33f
    ut_ai: wp.mat33f
    s_inv: wp.mat33f
    accumulated_impulse1: wp.vec3f
    accumulated_impulse2: wp.vec3f

    # ---- Axial actuator + limit block (new in this constraint) -------
    axis_local1: wp.vec3f
    axis_local2: wp.vec3f
    q0: wp.quatf
    drive_mode: wp.int32
    target_angle: wp.float32
    target_velocity: wp.float32
    max_force_drive: wp.float32
    hertz_drive: wp.float32
    damping_ratio_drive: wp.float32
    min_angle: wp.float32
    max_angle: wp.float32
    hertz_limit: wp.float32
    damping_ratio_limit: wp.float32
    effective_mass_axial: wp.float32
    bias_drive: wp.float32
    bias_limit: wp.float32
    mass_coeff_drive: wp.float32
    impulse_coeff_drive: wp.float32
    mass_coeff_limit: wp.float32
    impulse_coeff_limit: wp.float32
    max_lambda_drive: wp.float32
    clamp: wp.int32
    axis_world: wp.vec3f
    accumulated_impulse_drive: wp.float32
    accumulated_impulse_limit: wp.float32


assert_constraint_header(ActuatedDoubleBallSocketData)


# Dword offsets derived once from the schema.
_OFF_BODY1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "body2"))
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
_OFF_BIAS_RATE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_rate"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff"))
_OFF_BIAS1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias2"))
_OFF_A1_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "a1_inv"))
_OFF_UT_AI = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "ut_ai"))
_OFF_S_INV = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "s_inv"))
_OFF_ACC_IMP1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse1"))
_OFF_ACC_IMP2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse2"))

_OFF_AXIS_LOCAL1 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_local1"))
_OFF_AXIS_LOCAL2 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_local2"))
_OFF_Q0 = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "q0"))
_OFF_DRIVE_MODE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "drive_mode"))
_OFF_TARGET_ANGLE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target_angle"))
_OFF_TARGET_VELOCITY = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "target_velocity"))
_OFF_MAX_FORCE_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_force_drive"))
_OFF_HERTZ_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz_drive"))
_OFF_DAMPING_RATIO_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio_drive"))
_OFF_MIN_ANGLE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "min_angle"))
_OFF_MAX_ANGLE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_angle"))
_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "hertz_limit"))
_OFF_DAMPING_RATIO_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "damping_ratio_limit"))
_OFF_EFF_AXIAL = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "effective_mass_axial"))
_OFF_BIAS_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_drive"))
_OFF_BIAS_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "bias_limit"))
_OFF_MASS_COEFF_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff_drive"))
_OFF_IMPULSE_COEFF_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff_drive"))
_OFF_MASS_COEFF_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "mass_coeff_limit"))
_OFF_IMPULSE_COEFF_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "impulse_coeff_limit"))
_OFF_MAX_LAMBDA_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "max_lambda_drive"))
_OFF_CLAMP = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "clamp"))
_OFF_AXIS_WORLD = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "axis_world"))
_OFF_ACC_DRIVE = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_drive"))
_OFF_ACC_LIMIT = wp.constant(dword_offset_of(ActuatedDoubleBallSocketData, "accumulated_impulse_limit"))

#: Total dword count of one actuated-fused-hinge constraint.
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
    drive_mode: wp.array[wp.int32],
    target_angle: wp.array[wp.float32],
    target_velocity: wp.array[wp.float32],
    max_force_drive: wp.array[wp.float32],
    hertz_drive: wp.array[wp.float32],
    damping_ratio_drive: wp.array[wp.float32],
    min_angle: wp.array[wp.float32],
    max_angle: wp.array[wp.float32],
    hertz_limit: wp.array[wp.float32],
    damping_ratio_limit: wp.array[wp.float32],
):
    """Pack one batch of actuated double-ball-socket descriptors.

    See :func:`double_ball_socket_initialize_kernel` for the inherited
    parameters. The actuator block snapshots the world hinge axis
    (``anchor2 - anchor1``) into each body's local frame so the runtime
    twist extraction has a stable body-fixed reference -- exactly the
    same trick :func:`hinge_angle_initialize_kernel` uses.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only ``position`` / ``orientation``
            of the referenced bodies are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor1: World-space first anchor [num_in_batch] [m].
        anchor2: World-space second anchor [num_in_batch] [m].
        hertz: Positional Schur block soft-constraint frequency
            [num_in_batch] [Hz].
        damping_ratio: Positional Schur block damping ratio
            [num_in_batch].
        drive_mode: One of :data:`DRIVE_MODE_OFF` /
            :data:`DRIVE_MODE_POSITION` / :data:`DRIVE_MODE_VELOCITY`
            [num_in_batch].
        target_angle: Position-drive setpoint [num_in_batch] [rad].
        target_velocity: Velocity-drive setpoint [num_in_batch] [rad/s].
        max_force_drive: Drive torque cap [num_in_batch] [N*m]; 0
            disables the drive even if ``drive_mode != OFF``.
        hertz_drive: Drive soft-constraint frequency [num_in_batch] [Hz].
        damping_ratio_drive: Drive damping ratio [num_in_batch].
        min_angle: Lower angular limit [num_in_batch] [rad].
        max_angle: Upper angular limit [num_in_batch] [rad];
            ``min_angle == max_angle == 0`` disables the limit.
        hertz_limit: Limit soft-constraint frequency [num_in_batch] [Hz].
        damping_ratio_limit: Limit damping ratio [num_in_batch].
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

    # Local anchors -- identical to constraint_double_ball_socket.
    la1_b1 = wp.quat_rotate_inv(orient1, a1_w - pos1)
    la1_b2 = wp.quat_rotate_inv(orient2, a1_w - pos2)
    la2_b1 = wp.quat_rotate_inv(orient1, a2_w - pos1)
    la2_b2 = wp.quat_rotate_inv(orient2, a2_w - pos2)

    # Snapshot the world hinge axis. Empty axis (anchors coincide) ->
    # fall back to body x; the limit / drive will be effectively
    # disabled because there's no usable axis (effective_mass would
    # blow up either way; the runtime guards against div-by-zero).
    hinge_world = a2_w - a1_w
    hinge_len2 = wp.dot(hinge_world, hinge_world)
    if hinge_len2 > 1.0e-20:
        n_hat_init = hinge_world / wp.sqrt(hinge_len2)
    else:
        n_hat_init = wp.vec3f(1.0, 0.0, 0.0)

    axis_local1 = wp.quat_rotate_inv(orient1, n_hat_init)
    axis_local2 = wp.quat_rotate_inv(orient2, n_hat_init)
    # Relative rest orientation: q2_init^* * q1_init -- identical to
    # the hinge-angle convention so the same closed-form twist
    # projection works.
    q0 = wp.quat_inverse(orient2) * orient1

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_ACTUATED_DOUBLE_BALL_SOCKET)
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
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

    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_BIAS_RATE, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, 0.0)

    eye = wp.identity(3, dtype=wp.float32)
    write_mat33(constraints, _OFF_A1_INV, cid, eye)
    write_mat33(constraints, _OFF_UT_AI, cid, eye)
    write_mat33(constraints, _OFF_S_INV, cid, eye)

    # Actuator block.
    write_vec3(constraints, _OFF_AXIS_LOCAL1, cid, axis_local1)
    write_vec3(constraints, _OFF_AXIS_LOCAL2, cid, axis_local2)
    write_quat(constraints, _OFF_Q0, cid, q0)
    write_int(constraints, _OFF_DRIVE_MODE, cid, drive_mode[tid])
    write_float(constraints, _OFF_TARGET_ANGLE, cid, target_angle[tid])
    write_float(constraints, _OFF_TARGET_VELOCITY, cid, target_velocity[tid])
    write_float(constraints, _OFF_MAX_FORCE_DRIVE, cid, max_force_drive[tid])
    write_float(constraints, _OFF_HERTZ_DRIVE, cid, hertz_drive[tid])
    write_float(constraints, _OFF_DAMPING_RATIO_DRIVE, cid, damping_ratio_drive[tid])
    write_float(constraints, _OFF_MIN_ANGLE, cid, min_angle[tid])
    write_float(constraints, _OFF_MAX_ANGLE, cid, max_angle[tid])
    write_float(constraints, _OFF_HERTZ_LIMIT, cid, hertz_limit[tid])
    write_float(constraints, _OFF_DAMPING_RATIO_LIMIT, cid, damping_ratio_limit[tid])
    write_float(constraints, _OFF_EFF_AXIAL, cid, 0.0)
    write_float(constraints, _OFF_BIAS_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_BIAS_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF_DRIVE, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF_LIMIT, cid, 1.0)
    write_float(constraints, _OFF_IMPULSE_COEFF_LIMIT, cid, 0.0)
    write_float(constraints, _OFF_MAX_LAMBDA_DRIVE, cid, 0.0)
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, n_hat_init)
    write_float(constraints, _OFF_ACC_DRIVE, cid, 0.0)
    write_float(constraints, _OFF_ACC_LIMIT, cid, 0.0)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Same two access levels as DoubleBallSocket: ``*_at`` for composability
# (takes an explicit base_offset + ConstraintBodies), and thin direct
# wrappers for the unified dispatcher.
#
# Symbol cheat-sheet (extends DBS):
#   n_hat            : world-frame hinge axis (recomputed each prepare)
#   eff_axial        : 1 / (n_hat . (I1^-1 + I2^-1) . n_hat)  -- scalar
#   twist_signed     : signed axial twist between the bodies [rad/2 ish]
#                      via the same closed-form projection as
#                      hinge_angle. Strictly speaking the projection
#                      onto axis_local2 yields sin(theta/2) for small
#                      angles, but linearises to theta/2 for the soft
#                      bias term -- this is the same approximation
#                      hinge_angle uses for its limit row, and matches
#                      the half-angle convention baked into the
#                      Jacobian.


@wp.func
def _twist_error(
    q1: wp.quatf,
    q2: wp.quatf,
    q0: wp.quatf,
    axis_local2: wp.vec3f,
) -> wp.float32:
    """Closed-form signed axial twist between body 1 and body 2.

    Same projection :func:`hinge_angle_prepare_for_iteration_at` uses
    for its limit row -- ``err = q0 * q1^{-1} * q2``, then dot the xyz
    part with the body-2-local hinge axis (with the standard short-
    rotation sign flip on ``err.w < 0``). Returns approximately
    ``sin(theta/2)`` for small ``theta``, which lines up with the
    half-angle convention of the closed-form Jacobian below.
    """
    q1_inv = wp.quat_inverse(q1)
    quat_e = q0 * q1_inv * q2
    quat_e_xyz = wp.vec3f(quat_e[0], quat_e[1], quat_e[2])
    err = wp.dot(axis_local2, quat_e_xyz)
    if quat_e[3] < 0.0:
        err = -err
    return err


@wp.func
def actuated_double_ball_socket_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable prepare pass for the actuated fused two-anchor hinge.

    Layered on top of the parent DBS prepare-pass: first does the same
    5-DoF Schur-complement bookkeeping (lever arms, tangent basis,
    A1^{-1}, U^T A1^{-1}, S^{-1}, positional bias, warm-start on the
    accumulated linear/angular impulses), then computes the axial drive
    + limit scalar block.

    See :func:`ball_socket_prepare_for_iteration_at` for the
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

    # ---- 5-DoF positional warm-start (identical to parent DBS) -------
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)

    velocity1 = bodies.velocity[b1] - inv_mass1 * (acc1 + acc2)
    angular_velocity1 = bodies.angular_velocity[b1] - inv_inertia1 @ (cr1_b1 @ acc1 + cr2_b1 @ acc2)
    velocity2 = bodies.velocity[b2] + inv_mass2 * (acc1 + acc2)
    angular_velocity2 = bodies.angular_velocity[b2] + inv_inertia2 @ (cr1_b2 @ acc1 + cr2_b2 @ acc2)

    # ---- Axial drive + limit block -----------------------------------
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    target_angle = read_float(constraints, base_offset + _OFF_TARGET_ANGLE, cid)
    target_velocity = read_float(constraints, base_offset + _OFF_TARGET_VELOCITY, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)
    hertz_drive = read_float(constraints, base_offset + _OFF_HERTZ_DRIVE, cid)
    damping_ratio_drive = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_DRIVE, cid)
    min_angle = read_float(constraints, base_offset + _OFF_MIN_ANGLE, cid)
    max_angle = read_float(constraints, base_offset + _OFF_MAX_ANGLE, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)

    axis_local2 = read_vec3(constraints, base_offset + _OFF_AXIS_LOCAL2, cid)
    q0 = read_quat(constraints, base_offset + _OFF_Q0, cid)

    # Scalar effective mass for the axial DoF: m^{-1} = n . (I1^-1 + I2^-1) . n.
    inv_inertia_sum = inv_inertia1 + inv_inertia2
    eff_inv = wp.dot(n_hat, inv_inertia_sum @ n_hat)
    if eff_inv < 1.0e-20:
        eff_axial = 0.0
    else:
        eff_axial = 1.0 / eff_inv
    write_float(constraints, base_offset + _OFF_EFF_AXIAL, cid, eff_axial)

    # Soft-constraint coefficients for the drive row.
    br_drive, mc_drive, ic_drive = soft_constraint_coefficients(hertz_drive, damping_ratio_drive, dt)
    br_limit, mc_limit, ic_limit = soft_constraint_coefficients(hertz_limit, damping_ratio_limit, dt)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_DRIVE, cid, mc_drive)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_DRIVE, cid, ic_drive)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mc_limit)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, ic_limit)

    # Drive impulse cap. Velocity drive: ``max_force * dt``. Position
    # drive ignores the cap (it relies on the soft spring instead --
    # capping a position drive would prevent it from ever reaching the
    # setpoint when fighting external loads).
    max_lambda_drive = max_force_drive * dt
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA_DRIVE, cid, max_lambda_drive)

    # Twist projection: half-angle approximation, same as hinge_angle.
    # ``twist_err`` ~ sin(theta/2) ~ theta/2 for the small-angle range
    # the soft solve operates on between substeps.
    twist_err = _twist_error(orientation1, orientation2, q0, axis_local2)

    # ---- Drive bias --------------------------------------------------
    # Sign convention: ``jv_axial = n_hat . (w1 - w2)`` and the iterate
    # applies ``+inv1 * (n_hat * lam)`` to body 1 and
    # ``-inv2 * (n_hat * lam)`` to body 2, so a *positive* lam increases
    # ``jv_axial`` (spins body 1 forward / body 2 backward about
    # ``+n_hat``). Combined with the kinematic identity
    # ``d(twist) / dt = -jv_axial`` (body 1 fixed -> twist evolves with
    # body 2's axial rate, which is ``-jv_axial``), this means a positive
    # lam *decreases* the relative twist. The PGS row solves
    #     lam = -mass_coeff * eff * (jv + bias) - impulse_coeff * acc
    # so to drive ``jv -> -bias``:
    #   * Position drive: want ``jv = -d(twist)/dt = +(target - twist)*br``,
    #     i.e. ``bias = -(target - twist) * br`` would push the wrong way.
    #     Correct sign: ``bias = +(target - twist) * br``.
    #   * Velocity drive: want body 2 axial rate ``omega = target_velocity``,
    #     i.e. ``jv = -target_velocity``, so ``bias = +target_velocity``.
    bias_drive = float(0.0)
    if drive_mode == DRIVE_MODE_POSITION:
        # Linearised half-angle => multiply by 2 to recover the actual
        # twist angle in [rad].
        twist_angle = 2.0 * twist_err
        bias_drive = (target_angle - twist_angle) * br_drive
    elif drive_mode == DRIVE_MODE_VELOCITY:
        bias_drive = target_velocity
    write_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid, bias_drive)

    # ---- Limit clamp + bias -----------------------------------------
    # Convert user-facing [min_angle, max_angle] in radians to the same
    # half-angle space the projection produces. ``min_angle ==
    # max_angle == 0`` -> limit disabled (no row).
    #
    # _CLAMP_MAX (twist past upper stop): need positive lam to push
    #   twist back down -> bias must drive jv positive -> bias < 0,
    #   accumulated impulse clamped to ``>= 0``.
    # _CLAMP_MIN (twist past lower stop): need negative lam to push
    #   twist back up -> bias > 0, accumulated impulse clamped to
    #   ``<= 0``.
    clamp = _CLAMP_NONE
    bias_limit = float(0.0)
    if min_angle != 0.0 or max_angle != 0.0:
        sin_half_min = wp.sin(min_angle * 0.5)
        sin_half_max = wp.sin(max_angle * 0.5)
        if twist_err > sin_half_max:
            clamp = _CLAMP_MAX
            bias_limit = -(twist_err - sin_half_max) * br_limit * 2.0
        elif twist_err < sin_half_min:
            clamp = _CLAMP_MIN
            bias_limit = -(twist_err - sin_half_min) * br_limit * 2.0
    write_int(constraints, base_offset + _OFF_CLAMP, cid, clamp)
    write_float(constraints, base_offset + _OFF_BIAS_LIMIT, cid, bias_limit)

    # ---- Warm-start the axial impulses --------------------------------
    # Apply the cached scalar accumulated impulses to the body angular
    # velocities (positive sign on body 1, negative on body 2 -- same
    # convention as AngularMotor with j1 = j2 = n_hat).
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    axial_imp = acc_drive + acc_limit
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_imp)
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_imp)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def actuated_double_ball_socket_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable PGS iteration step for the actuated fused hinge.

    First does the parent DBS Schur-complement solve (5-DoF positional),
    then runs the scalar axial drive + limit rows. The two stages share
    the same body-data load (one read of v / w / I^{-1} per body per
    iter) but write twice -- once with the linear+tangent impulse, once
    with the axial torque.
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

    # ---- 5-DoF Schur-complement positional solve (parent DBS) --------
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
    eff_axial = read_float(constraints, base_offset + _OFF_EFF_AXIAL, cid)
    drive_mode = read_int(constraints, base_offset + _OFF_DRIVE_MODE, cid)
    max_lambda_drive = read_float(constraints, base_offset + _OFF_MAX_LAMBDA_DRIVE, cid)
    bias_drive = read_float(constraints, base_offset + _OFF_BIAS_DRIVE, cid)
    bias_limit = read_float(constraints, base_offset + _OFF_BIAS_LIMIT, cid)
    mc_drive = read_float(constraints, base_offset + _OFF_MASS_COEFF_DRIVE, cid)
    ic_drive = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_DRIVE, cid)
    mc_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
    ic_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)
    max_force_drive = read_float(constraints, base_offset + _OFF_MAX_FORCE_DRIVE, cid)

    # Axial relative velocity along n_hat: jv_axial = n . (w1 - w2).
    # Same sign convention as the AngularMotor.
    jv_axial = wp.dot(n_hat, angular_velocity1 - angular_velocity2)

    # ---- Drive row ---------------------------------------------------
    # Active iff the user picked a non-OFF mode AND gave the drive
    # somewhere to push against (eff_axial > 0; eff_axial = 0 means
    # both bodies are infinitely heavy along the hinge axis -- nothing
    # to drive).
    drive_active = drive_mode != DRIVE_MODE_OFF
    if drive_mode == DRIVE_MODE_VELOCITY and max_force_drive <= 0.0:
        # Velocity drive with no torque cap is a no-op; matches
        # AngularMotor's convention. Position drives keep going since
        # they're soft springs whose magnitude is bounded by
        # mass_coeff * eff_axial * bias_rate -- the cap is implicit.
        drive_active = False

    lam_drive = float(0.0)
    if drive_active and eff_axial > 0.0:
        lam_unsoft = -eff_axial * (jv_axial + bias_drive)
        lam_drive = mc_drive * lam_unsoft - ic_drive * acc_drive
        old_acc = acc_drive
        acc_drive = acc_drive + lam_drive
        if drive_mode == DRIVE_MODE_VELOCITY:
            acc_drive = wp.clamp(acc_drive, -max_lambda_drive, max_lambda_drive)
        lam_drive = acc_drive - old_acc
        write_float(constraints, base_offset + _OFF_ACC_DRIVE, cid, acc_drive)

    # ---- Limit row ---------------------------------------------------
    # Already computed clamp tag in prepare; iterate just enforces the
    # one-sided clamp on the accumulated impulse. _CLAMP_MAX -> twist
    # is past the upper stop -> need positive lam to push twist back
    # down (positive lam decreases relative twist; see the prepare-pass
    # sign-convention comment) -> accumulated impulse must be ``>= 0``.
    # _CLAMP_MIN is the symmetric case.
    lam_limit = float(0.0)
    if clamp != _CLAMP_NONE and eff_axial > 0.0:
        lam_unsoft = -eff_axial * (jv_axial + bias_limit)
        lam_limit = mc_limit * lam_unsoft - ic_limit * acc_limit
        old_acc = acc_limit
        acc_limit = acc_limit + lam_limit
        if clamp == _CLAMP_MAX:
            acc_limit = wp.max(0.0, acc_limit)
        else:
            acc_limit = wp.min(0.0, acc_limit)
        lam_limit = acc_limit - old_acc
        write_float(constraints, base_offset + _OFF_ACC_LIMIT, cid, acc_limit)

    # Apply the combined axial impulse.
    axial_lam = lam_drive + lam_limit
    angular_velocity1 = angular_velocity1 + inv_inertia1 @ (n_hat * axial_lam)
    angular_velocity2 = angular_velocity2 - inv_inertia2 @ (n_hat * axial_lam)

    bodies.velocity[b1] = velocity1
    bodies.angular_velocity[b1] = angular_velocity1
    bodies.velocity[b2] = velocity2
    bodies.angular_velocity[b2] = angular_velocity2


@wp.func
def actuated_double_ball_socket_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable wrench on body 2 -- linear from the two anchors plus
    axial torque from drive + limit. See
    :func:`actuated_double_ball_socket_world_wrench` for semantics.
    """
    acc1 = read_vec3(constraints, base_offset + _OFF_ACC_IMP1, cid)
    acc2 = read_vec3(constraints, base_offset + _OFF_ACC_IMP2, cid)
    r1_b2 = read_vec3(constraints, base_offset + _OFF_R1_B2, cid)
    r2_b2 = read_vec3(constraints, base_offset + _OFF_R2_B2, cid)
    n_hat = read_vec3(constraints, base_offset + _OFF_AXIS_WORLD, cid)
    acc_drive = read_float(constraints, base_offset + _OFF_ACC_DRIVE, cid)
    acc_limit = read_float(constraints, base_offset + _OFF_ACC_LIMIT, cid)

    force = (acc1 + acc2) * idt
    torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt)
    torque = torque - n_hat * ((acc_drive + acc_limit) * idt)
    return force, torque


@wp.func
def actuated_double_ball_socket_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct prepare entry: see
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
):
    """Direct iterate entry; see
    :func:`actuated_double_ball_socket_iterate_at`.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    actuated_double_ball_socket_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def actuated_double_ball_socket_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body 2.

    Force is the sum of the two anchor accumulated impulses divided by
    the substep ``dt`` (``idt = 1 / substep_dt``); torque is each
    impulse's moment about body 2's COM (using the cached lever arms
    from the most recent ``prepare_for_iteration``) *minus* the axial
    drive + limit torque acting on body 2 along ``-n_hat``.
    """
    return actuated_double_ball_socket_world_wrench_at(constraints, cid, 0, idt)
