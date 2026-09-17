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
    body_load_vw,
    body_set_access_mode,
    body_store_vw,
    mat33_from_sym6,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    CONSTRAINT_TYPE_JOINT,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_read_multiplier,
    constraint_read_multiplier_vec3,
    constraint_set_type,
    constraint_write_multiplier,
    constraint_write_multiplier_vec3,
    read_float,
    read_int,
    read_vec3,
    write_float,
    write_int,
    write_quat,
    write_vec3,
)
from newton._src.solvers.phoenx.constraints.d6_inequality import prepare_d6_inequalities
from newton._src.solvers.phoenx.constraints.d6_joint_data import D6_AXIS_COUNT
from newton._src.solvers.phoenx.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.phoenx.helpers.math_helpers import (
    create_orthonormal,
    revolution_tracker_angle,
)
from newton._src.solvers.phoenx.mass_splitting.access import (
    read_angular_velocity_unified,
    read_velocity_unified,
    write_angular_velocity_unified,
    write_velocity_unified,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer

__all__ = [
    "DRIVE_MODE_OFF",
    "DRIVE_MODE_POSITION",
    "DRIVE_MODE_VELOCITY",
    "JOINT_CONSTRAINT_DWORDS",
    "JOINT_CONSTRAINT_TIME_US_OFFSET",
    "JOINT_MODE_BALL_SOCKET",
    "JOINT_MODE_CABLE",
    "JOINT_MODE_CARTESIAN",
    "JOINT_MODE_CARTESIAN_PLANE",
    "JOINT_MODE_CYLINDRICAL",
    "JOINT_MODE_DISTANCE",
    "JOINT_MODE_FIXED",
    "JOINT_MODE_GENERIC_D6",
    "JOINT_MODE_PLANAR",
    "JOINT_MODE_PRISMATIC",
    "JOINT_MODE_REVOLUTE",
    "JOINT_MODE_UNIVERSAL",
    "JointConstraintData",
    "joint_constraint_clear_reset_worlds",
    "joint_constraint_initialize_kernel",
    "joint_constraint_prepare_inequality",
    "joint_constraint_world_error",
    "joint_constraint_world_error_at",
    "joint_constraint_world_wrench",
    "joint_constraint_world_wrench_at",
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
#: Fixed (weld) joint: all six structural rows are solved by the direct
#: mechanism system. It has no PGS inequality row.
JOINT_MODE_FIXED = wp.constant(wp.int32(3))
#: Cable joint: its structural spring-damper rows are solved by the direct
#: mechanism system. It has no PGS inequality row.
JOINT_MODE_CABLE = wp.constant(wp.int32(4))
#: Universal (Hooke) joint: locks anchor translation and one angular
#: twist axis. D6-dispatched universal joints may also carry angular
#: limit rows on their two free axes.
JOINT_MODE_UNIVERSAL = wp.constant(wp.int32(5))
JOINT_MODE_CYLINDRICAL = wp.constant(wp.int32(6))
JOINT_MODE_PLANAR = wp.constant(wp.int32(7))
#: Cartesian translation joint with two free in-plane linear axes.
JOINT_MODE_CARTESIAN_PLANE = wp.constant(wp.int32(8))
#: Cartesian translation joint with all three linear axes free.
JOINT_MODE_CARTESIAN = wp.constant(wp.int32(9))
#: Radial distance interval. This mode has no bilateral structural rows.
JOINT_MODE_DISTANCE = wp.constant(wp.int32(10))
#: Generic D6 whose locked row basis is precomputed by the direct solver.
JOINT_MODE_GENERIC_D6 = wp.constant(wp.int32(11))

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
#: pure-velocity-motor fallback when ``damping_drive == 0``.
DRIVE_MODE_VELOCITY = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class JointConstraintData:
    """Store joint inequality state and experimental projector geometry."""

    # ---- Header -------------------------------------------------------
    constraint_type: wp.int32
    body1: wp.int32
    body2: wp.int32

    # ---- Shared geometry ---------------------------------------------
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
    # Runtime bias vectors retained by the experimental tree projector;
    bias1: wp.vec3f
    bias2: wp.vec3f
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

    # ---- Free-coordinate inequality state ----------------------------
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
    # Limit window: rad (revolute) or m (prismatic). ``min_value >
    # max_value`` disables the limit (matches the standalone
    # angular_limit / linear_limit sentinel).
    min_value: wp.float32
    max_value: wp.float32
    # Cached world-frame joint axis from the most recent prepare-pass.
    axis_world: wp.vec3f
    #: Opt-in per-column wall-clock accumulator (microseconds). See
    #: :func:`constraint_accumulate_time_us`.
    time_us: wp.float32


assert_constraint_header(JointConstraintData)


# Dword offsets derived once from the schema. Named per field.
_OFF_BODY1 = wp.constant(dword_offset_of(JointConstraintData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(JointConstraintData, "body2"))
_OFF_JOINT_MODE = wp.constant(dword_offset_of(JointConstraintData, "joint_mode"))
_OFF_LA1_B1 = wp.constant(dword_offset_of(JointConstraintData, "local_anchor1_b1"))
_OFF_LA1_B2 = wp.constant(dword_offset_of(JointConstraintData, "local_anchor1_b2"))
_OFF_LA2_B1 = wp.constant(dword_offset_of(JointConstraintData, "local_anchor2_b1"))
_OFF_LA2_B2 = wp.constant(dword_offset_of(JointConstraintData, "local_anchor2_b2"))
_OFF_R1_B1 = wp.constant(dword_offset_of(JointConstraintData, "r1_b1"))
_OFF_R1_B2 = wp.constant(dword_offset_of(JointConstraintData, "r1_b2"))
_OFF_R2_B1 = wp.constant(dword_offset_of(JointConstraintData, "r2_b1"))
_OFF_R2_B2 = wp.constant(dword_offset_of(JointConstraintData, "r2_b2"))
_OFF_T1 = wp.constant(dword_offset_of(JointConstraintData, "t1"))
_OFF_T2 = wp.constant(dword_offset_of(JointConstraintData, "t2"))
_OFF_BIAS1 = wp.constant(dword_offset_of(JointConstraintData, "bias1"))
_OFF_BIAS2 = wp.constant(dword_offset_of(JointConstraintData, "bias2"))
# Aliased mode-extras block. Prismatic packs anchor-3 / r3 / acc_imp3
# / bias3 (16 dwords); revolute packs the twist-tracker scratch
# (inv_initial_orientation + revolution_counter + previous_quaternion_angle
# = 6 dwords). Mutually exclusive, so we share the 16-dword block.
_OFF_MODE_EXTRAS = wp.constant(dword_offset_of(JointConstraintData, "mode_extras"))
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
_OFF_AXIS_LOCAL1 = wp.constant(dword_offset_of(JointConstraintData, "axis_local1"))
_OFF_REST_LENGTH = wp.constant(dword_offset_of(JointConstraintData, "rest_length"))
_OFF_DRIVE_MODE = wp.constant(dword_offset_of(JointConstraintData, "drive_mode"))
_OFF_TARGET = wp.constant(dword_offset_of(JointConstraintData, "target"))
_OFF_TARGET_VELOCITY = wp.constant(dword_offset_of(JointConstraintData, "target_velocity"))
_OFF_MAX_FORCE_DRIVE = wp.constant(dword_offset_of(JointConstraintData, "max_force_drive"))
_OFF_STIFFNESS_DRIVE = wp.constant(dword_offset_of(JointConstraintData, "stiffness_drive"))
_OFF_DAMPING_DRIVE = wp.constant(dword_offset_of(JointConstraintData, "damping_drive"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(JointConstraintData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(JointConstraintData, "max_value"))
_OFF_AXIS_WORLD = wp.constant(dword_offset_of(JointConstraintData, "axis_world"))
# Family-aliased mutable state in three aligned vec4 groups: impulse.xyz and
# its correlated limit/friction scalar in w.
_MUL_ACC_IMP1 = wp.constant(wp.int32(0))
_MUL_ACC_IMP2 = wp.constant(wp.int32(4))
_MUL_ACC_LIMIT = wp.constant(wp.int32(7))
_MUL_ACC_IMP3 = wp.constant(wp.int32(8))
JOINT_CONSTRAINT_TIME_US_OFFSET = wp.constant(dword_offset_of(JointConstraintData, "time_us"))

#: Total dword count of one unified joint constraint.
JOINT_CONSTRAINT_DWORDS: int = num_dwords(JointConstraintData)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False, module="unique")
def joint_constraint_initialize_kernel(
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

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_JOINT)
    mode = joint_mode[tid]
    if mode == JOINT_MODE_DISTANCE:
        # Distance endpoints are the two independently authored anchors.
        la1_b2 = wp.quat_rotate_inv(orient2, a2_w - pos2)

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
        # via the twist-tracker layout.
        write_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid, inv_initial_orientation)
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, 0)
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, 0.0)

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
    write_float(constraints, _OFF_MIN_VALUE, cid, min_value[tid])
    write_float(constraints, _OFF_MAX_VALUE, cid, max_value[tid])
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, n_hat_init)
    constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, 0.0)


# ---------------------------------------------------------------------------
# Runtime reset
# ---------------------------------------------------------------------------


@wp.func
def _joint_constraint_world(bodies: BodyContainer, b1: wp.int32, b2: wp.int32) -> wp.int32:
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
def _joint_constraint_clear_reset_worlds_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: wp.int32,
    dones: wp.array[wp.float32],
):
    cid = wp.tid()
    if cid >= joint_count:
        return

    world = _joint_constraint_world(
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
    write_vec3(constraints, _OFF_BIAS1, cid, zero3)
    write_vec3(constraints, _OFF_BIAS2, cid, zero3)
    mode = read_int(constraints, _OFF_JOINT_MODE, cid)
    if mode == JOINT_MODE_PRISMATIC or mode == JOINT_MODE_FIXED or mode == JOINT_MODE_CABLE:
        write_vec3(constraints, _OFF_R3_B1, cid, zero3)
        write_vec3(constraints, _OFF_R3_B2, cid, zero3)
        constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP3, cid, zero3)
        write_float(constraints, _OFF_BIAS3, cid, wp.float32(0.0))
    else:
        write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, wp.int32(0))
        write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, wp.float32(0.0))

    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP1, cid, zero3)
    constraint_write_multiplier_vec3(constraints, _MUL_ACC_IMP2, cid, zero3)
    write_vec3(constraints, _OFF_AXIS_WORLD, cid, zero3)
    constraint_write_multiplier(constraints, _MUL_ACC_LIMIT, cid, wp.float32(0.0))
    if constraints.d6.enabled != wp.int32(0):
        for row in range(6):
            constraints.d6.lower_impulse[cid, row] = wp.float32(0.0)
            constraints.d6.upper_impulse[cid, row] = wp.float32(0.0)
            constraints.d6.friction_impulse[cid, row] = wp.float32(0.0)
            constraints.d6.revolution_counter[cid, row] = wp.int32(0)
            constraints.d6.previous_angle[cid, row] = wp.float32(0.0)


def joint_constraint_clear_reset_worlds(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    joint_count: int,
    dones: wp.array[wp.float32],
    device: wp.DeviceLike = None,
) -> None:
    """Clear joint constraint runtime caches and warm starts for reset worlds."""
    count = max(0, min(int(joint_count), int(constraints.data.shape[1])))
    if count == 0:
        return
    wp.launch(
        _joint_constraint_clear_reset_worlds_kernel,
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
def _joint_constraint_prepare_inequality_full(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Prepare common D6 limit, speed-cap, and friction rows."""
    if constraints.d6.enabled == wp.int32(0) or constraints.d6.row_count[cid] == wp.int32(0):
        return

    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_set_access_mode(bodies, b1, ACCESS_MODE_VELOCITY_LEVEL, idt)
    body_set_access_mode(bodies, b2, ACCESS_MODE_VELOCITY_LEVEL, idt)
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

    velocity1, angular_velocity1, velocity2, angular_velocity2 = prepare_d6_inequalities(
        constraints.d6,
        cid,
        bodies,
        b1,
        b2,
        inv_mass1,
        inv_mass2,
        inv_inertia1,
        inv_inertia2,
        velocity1,
        angular_velocity1,
        velocity2,
        angular_velocity2,
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
def joint_constraint_world_wrench_at(
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
    if joint_mode == JOINT_MODE_REVOLUTE:
        force = (acc1 + acc2) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt)
    elif joint_mode == JOINT_MODE_DISTANCE:
        force = wp.vec3f(0.0, 0.0, 0.0)
        torque = wp.vec3f(0.0, 0.0, 0.0)
    elif joint_mode == JOINT_MODE_PRISMATIC:
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    elif joint_mode == JOINT_MODE_UNIVERSAL:
        acc_limit = constraint_read_multiplier(constraints, _MUL_ACC_LIMIT, cid)
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt) - n_hat * (acc_limit * idt) - acc2 * idt
    elif joint_mode == JOINT_MODE_FIXED or joint_mode == JOINT_MODE_CABLE:
        # Same anchor layout (anchor-1 3-row + anchor-2 tangent 2-row +
        # anchor-3 scalar 1-row); no axial block. CABLE's PD softness
        # is already baked into the accumulated impulses, so the
        # wrench reflects the actual reaction the joint applied this
        # substep.
        force = (acc1 + acc2 + acc3) * idt
        torque = wp.cross(r1_b2, acc1 * idt) + wp.cross(r2_b2, acc2 * idt) + wp.cross(r3_b2, acc3 * idt)
    else:
        # Ball-socket: only the anchor-1 impulse contributes here.
        force = acc1 * idt
        torque = wp.cross(r1_b2, acc1 * idt)

    # Common D6 inequalities keep their warm-start impulses outside the
    # legacy joint column. Include the exact wrench applied to body 2 so
    # diagnostics report limits, speed caps, and friction on every D6 axis.
    if constraints.d6.enabled != wp.int32(0):
        row_count = constraints.d6.row_count[cid]
        for row in range(D6_AXIS_COUNT):
            if wp.int32(row) < row_count:
                impulse = (
                    constraints.d6.lower_impulse[cid, row]
                    + constraints.d6.upper_impulse[cid, row]
                    + constraints.d6.friction_impulse[cid, row]
                )
                wrench = constraints.d6.wrench1[cid, row]
                force += wp.spatial_top(wrench) * (impulse * idt)
                torque += wp.spatial_bottom(wrench) * (impulse * idt)
    return force, torque


@wp.func
def joint_constraint_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame (force, torque) this constraint exerts on body 2.

    Units: [N], [N*m]. See
    :func:`joint_constraint_world_wrench_at` for details.
    """
    return joint_constraint_world_wrench_at(constraints, cid, 0, idt)


@wp.func
def joint_constraint_world_error_at(
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
def joint_constraint_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`joint_constraint_world_error_at`."""
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return joint_constraint_world_error_at(constraints, cid, 0, bodies, body_pair)


@wp.func
def joint_constraint_prepare_inequality(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    particles: ParticleContainer,
    copy_state: CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
):
    """Prepare common D6 limit, speed-cap, and friction rows."""
    _joint_constraint_prepare_inequality_full(
        constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt
    )
