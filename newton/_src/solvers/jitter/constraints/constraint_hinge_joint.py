# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused single-column hinge joint (HingeAngle + BallSocket + AngularMotor).

Mirrors the C# composition in ``Jitter2.Dynamics.Joints.HingeJoint``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Joints/HingeJoint.cs``):
a motorised revolute joint is a hinge-angle (locks the two angular DoFs
perpendicular to the hinge axis), a ball-socket (locks all three
translational DoFs at the anchor), and an optional angular motor
(drives the remaining axial angular DoF toward a target relative
velocity, capped by ``max_force``).

In the standalone solver these three pieces would each occupy their
own column in the :class:`ConstraintContainer` and each get partitioned
into its own colour (because all three touch the same ``(body1, body2)``
pair, the partitioner *must* place each in a separate launch). That
costs us:

  * three partitioned launches per PGS sweep instead of one,
  * three reloads of body 1 / body 2 state from
    :class:`~newton._src.solvers.jitter.body.BodyContainer`,
  * three storage columns per joint in the container.

The fused layout collapses all three into a single column owned by a
single PGS thread. The thread reads the body pair *once* from the
shared header, runs the three sub-iterations sequentially with the
body data hot in registers, and writes back once. Convergence
typically improves substantially on heavily-loaded chains because
Gauss-Seidel propagates corrections immediately within the fused
constraint instead of across launches.

Layout (column-major-by-cid, dword offsets inside one column)::

    0..2:                 shared header (constraint_type, body1, body2)
                          BallSocket sub-base = 0 -- BS's own "header"
                          coincides with the shared header.
    3..BS_DWORDS-1:       BallSocket data fields (la1, la2, r1, r2, u,
                          bias_factor, softness, effective_mass,
                          accumulated_impulse, bias)
    BS_DWORDS..+2:        unused (HingeAngle's 'wasted header' --
                          *_at variants never touch the header dwords
                          of their own sub-block, the fused header is
                          authoritative for the body pair)
    BS_DWORDS+3..:        HingeAngle data fields
                          HingeAngle sub-base = BS_DWORDS
    +HA_DWORDS..+2:       unused (AngularMotor's 'wasted header')
    BS_DWORDS+HA_DWORDS+3..:  AngularMotor data fields
                          AngularMotor sub-base = BS_DWORDS + HA_DWORDS

Total ``HJ_DWORDS = BS_DWORDS + HA_DWORDS + AM_DWORDS``. The 6 wasted
header dwords (HingeAngle's + AngularMotor's) are a flat tax for being
able to reuse the existing ``*_at`` sub-funcs unmodified -- they
expect their own per-sub-struct dword layout, where the first three
dwords are constraint_type/body1/body2.

Per-sub-block iteration order matches Jitter2 ``HingeJoint.cs``:
HingeAngle -> BallSocket -> AngularMotor.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_angular_motor import (
    AM_DWORDS,
    AngularMotorData,
    angular_motor_iterate_at,
    angular_motor_prepare_for_iteration_at,
    angular_motor_world_error_at,
    angular_motor_world_wrench_at,
)
from newton._src.solvers.jitter.constraints.constraint_ball_socket import (
    BS_DWORDS,
    BallSocketData,
    ball_socket_iterate_at,
    ball_socket_prepare_for_iteration_at,
    ball_socket_world_error_at,
    ball_socket_world_wrench_at,
)
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_HINGE_JOINT,
    ConstraintContainer,
    constraint_bodies_make,
    constraint_get_body1,
    constraint_get_body2,
    constraint_set_body1,
    constraint_set_body2,
    constraint_set_type,
    write_float,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.constraints.constraint_hinge_angle import (
    HA_DWORDS,
    HingeAngleData,
    hinge_angle_iterate_at,
    hinge_angle_prepare_for_iteration_at,
    hinge_angle_world_error_at,
    hinge_angle_world_wrench_at,
)
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of

__all__ = [
    "HJ_AM_BASE",
    "HJ_BS_BASE",
    "HJ_DWORDS",
    "HJ_HA_BASE",
    "hinge_joint_initialize_kernel",
    "hinge_joint_iterate",
    "hinge_joint_prepare_for_iteration",
    "hinge_joint_world_error",
    "hinge_joint_world_wrench",
]


# ---------------------------------------------------------------------------
# Sub-block bases inside one fused column
# ---------------------------------------------------------------------------
#
# See the module docstring for the layout reasoning. Each sub-base is the
# dword at which the corresponding sub-struct *thinks* it starts; the
# first three dwords of every sub-struct are the constraint header (which
# the ``*_at`` variants ignore -- they take body indices via the
# ``ConstraintBodies`` argument). For the BallSocket sub the header
# coincides with the shared dispatcher header, so a single set of header
# dwords serves both purposes.

HJ_BS_BASE: int = 0
HJ_HA_BASE: int = BS_DWORDS
HJ_AM_BASE: int = BS_DWORDS + HA_DWORDS

#: Total dword count of one fused hinge-joint constraint. Used by the
#: host-side container allocator to size ``ConstraintContainer.data``'s
#: row count.
HJ_DWORDS: int = BS_DWORDS + HA_DWORDS + AM_DWORDS

# Wrap the bases as wp.constant so kernels can use them as compile-time
# literals in arithmetic with ``cid``.
_HJ_BS_BASE_C = wp.constant(wp.int32(HJ_BS_BASE))
_HJ_HA_BASE_C = wp.constant(wp.int32(HJ_HA_BASE))
_HJ_AM_BASE_C = wp.constant(wp.int32(HJ_AM_BASE))


# ---------------------------------------------------------------------------
# Sub-block field offsets within each sub-struct's own dword layout
# ---------------------------------------------------------------------------
#
# Re-derived from the schema here (rather than imported as ``_OFF_*``
# from the per-type modules) to keep the dependency direction strictly
# one-way: this module only uses the public ``*_at`` entry points and
# the sub-struct schemas. The numerical values are identical to the
# private offsets inside each per-type module.
_BS_OFF_LA1 = wp.constant(dword_offset_of(BallSocketData, "local_anchor1"))
_BS_OFF_LA2 = wp.constant(dword_offset_of(BallSocketData, "local_anchor2"))
_BS_OFF_R1 = wp.constant(dword_offset_of(BallSocketData, "r1"))
_BS_OFF_R2 = wp.constant(dword_offset_of(BallSocketData, "r2"))
_BS_OFF_HERTZ = wp.constant(dword_offset_of(BallSocketData, "hertz"))
_BS_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(BallSocketData, "damping_ratio"))
_BS_OFF_BIAS_RATE = wp.constant(dword_offset_of(BallSocketData, "bias_rate"))
_BS_OFF_MASS_COEFF = wp.constant(dword_offset_of(BallSocketData, "mass_coeff"))
_BS_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(BallSocketData, "impulse_coeff"))
_BS_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(BallSocketData, "effective_mass"))
_BS_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(BallSocketData, "accumulated_impulse"))
_BS_OFF_BIAS = wp.constant(dword_offset_of(BallSocketData, "bias"))

_HA_OFF_MIN_ANGLE = wp.constant(dword_offset_of(HingeAngleData, "min_angle"))
_HA_OFF_MAX_ANGLE = wp.constant(dword_offset_of(HingeAngleData, "max_angle"))
_HA_OFF_HERTZ_LOCK = wp.constant(dword_offset_of(HingeAngleData, "hertz_lock"))
_HA_OFF_DAMPING_RATIO_LOCK = wp.constant(
    dword_offset_of(HingeAngleData, "damping_ratio_lock")
)
_HA_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "hertz_limit"))
_HA_OFF_DAMPING_RATIO_LIMIT = wp.constant(
    dword_offset_of(HingeAngleData, "damping_ratio_limit")
)
_HA_OFF_BIAS_RATE_LOCK = wp.constant(dword_offset_of(HingeAngleData, "bias_rate_lock"))
_HA_OFF_MASS_COEFF_LOCK = wp.constant(dword_offset_of(HingeAngleData, "mass_coeff_lock"))
_HA_OFF_IMPULSE_COEFF_LOCK = wp.constant(
    dword_offset_of(HingeAngleData, "impulse_coeff_lock")
)
_HA_OFF_BIAS_RATE_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "bias_rate_limit"))
_HA_OFF_MASS_COEFF_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "mass_coeff_limit"))
_HA_OFF_IMPULSE_COEFF_LIMIT = wp.constant(
    dword_offset_of(HingeAngleData, "impulse_coeff_limit")
)
_HA_OFF_AXIS = wp.constant(dword_offset_of(HingeAngleData, "axis"))
_HA_OFF_Q0 = wp.constant(dword_offset_of(HingeAngleData, "q0"))
_HA_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(HingeAngleData, "accumulated_impulse"))
_HA_OFF_BIAS = wp.constant(dword_offset_of(HingeAngleData, "bias"))
_HA_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(HingeAngleData, "effective_mass"))
_HA_OFF_JACOBIAN = wp.constant(dword_offset_of(HingeAngleData, "jacobian"))
_HA_OFF_CLAMP = wp.constant(dword_offset_of(HingeAngleData, "clamp"))

_AM_OFF_LOCAL_AXIS1 = wp.constant(dword_offset_of(AngularMotorData, "local_axis1"))
_AM_OFF_LOCAL_AXIS2 = wp.constant(dword_offset_of(AngularMotorData, "local_axis2"))
_AM_OFF_VELOCITY = wp.constant(dword_offset_of(AngularMotorData, "velocity"))
_AM_OFF_MAX_FORCE = wp.constant(dword_offset_of(AngularMotorData, "max_force"))
_AM_OFF_HERTZ = wp.constant(dword_offset_of(AngularMotorData, "hertz"))
_AM_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(AngularMotorData, "damping_ratio"))
_AM_OFF_MAX_LAMBDA = wp.constant(dword_offset_of(AngularMotorData, "max_lambda"))
_AM_OFF_MASS_COEFF = wp.constant(dword_offset_of(AngularMotorData, "mass_coeff"))
_AM_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(AngularMotorData, "impulse_coeff"))
_AM_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(AngularMotorData, "effective_mass"))
_AM_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(AngularMotorData, "accumulated_impulse"))

# HingeAngle's clamp-state sentinels (kept private to that module). We
# only need the "no clamp active" value here for initialisation.
_CLAMP_NONE = wp.constant(wp.int32(0))


# ---------------------------------------------------------------------------
# Initialisation kernel
# ---------------------------------------------------------------------------


@wp.kernel
def hinge_joint_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    anchor: wp.array[wp.vec3f],
    world_axis: wp.array[wp.vec3f],
    min_angle_rad: wp.array[wp.float32],
    max_angle_rad: wp.array[wp.float32],
    target_velocity: wp.array[wp.float32],
    max_force: wp.array[wp.float32],
    hertz_linear: wp.array[wp.float32],
    damping_ratio_linear: wp.array[wp.float32],
    hertz_lock: wp.array[wp.float32],
    damping_ratio_lock: wp.array[wp.float32],
    hertz_limit: wp.array[wp.float32],
    damping_ratio_limit: wp.array[wp.float32],
    hertz_motor: wp.array[wp.float32],
    damping_ratio_motor: wp.array[wp.float32],
):
    """Pack one batch of fused hinge-joint descriptors into ``constraints``.

    Performs the same per-sub-block initialisation as
    :func:`~newton._src.solvers.jitter.constraints.constraint_ball_socket.ball_socket_initialize_kernel`,
    :func:`~newton._src.solvers.jitter.constraints.constraint_hinge_angle.hinge_angle_initialize_kernel`,
    and
    :func:`~newton._src.solvers.jitter.constraints.constraint_angular_motor.angular_motor_initialize_kernel`,
    but writes everything into the *single* fused column at the three
    sub-bases (``HJ_BS_BASE`` / ``HJ_HA_BASE`` / ``HJ_AM_BASE``).

    The shared header (constraint_type, body1, body2) is written once
    -- the BallSocket sub's own header dwords coincide with it.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; this kernel only reads
            ``position`` / ``orientation`` of the two referenced bodies.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        anchor: World-space anchor points for the ball-socket lock
            [num_in_batch] [m].
        world_axis: Hinge axes in *world* space [num_in_batch]. Used by
            both the HingeAngle (to derive ``axis`` in body 2's local
            frame) and the AngularMotor (as the world axis on both
            bodies). Must be unit length; the kernel does not renormalise.
        min_angle_rad: Lower angular limits [num_in_batch] [rad].
        max_angle_rad: Upper angular limits [num_in_batch] [rad].
        target_velocity: Target relative axial angular velocity for the
            motor [num_in_batch] [rad/s].
        max_force: Maximum motor torque [num_in_batch] [N*m]. Pass 0 for
            a passive (unmotorised) joint -- the AngularMotor sub then
            applies zero corrective impulse and acts as a no-op.
        hertz_linear: Soft-constraint frequency for the BallSocket sub
            [num_in_batch] [Hz].
        damping_ratio_linear: Soft-constraint damping ratio for the
            BallSocket sub [num_in_batch].
        hertz_lock: Soft-constraint frequency for the HingeAngle's
            perpendicular angular lock [num_in_batch] [Hz].
        damping_ratio_lock: Soft-constraint damping ratio for the
            HingeAngle lock [num_in_batch].
        hertz_limit: Soft-constraint frequency for the HingeAngle's
            axial min/max limit [num_in_batch] [Hz].
        damping_ratio_limit: Soft-constraint damping ratio for the
            HingeAngle limit [num_in_batch].
        hertz_motor: Soft-constraint frequency for the AngularMotor
            sub [num_in_batch] [Hz].
        damping_ratio_motor: Soft-constraint damping ratio for the
            AngularMotor [num_in_batch].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a = anchor[tid]
    w_axis = world_axis[tid]
    min_a = min_angle_rad[tid]
    max_a = max_angle_rad[tid]
    target_vel = target_velocity[tid]
    mf = max_force[tid]

    pos1 = bodies.position[b1]
    pos2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # --- Shared header --------------------------------------------------
    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_HINGE_JOINT)
    constraint_set_body1(constraints, cid, b1)
    constraint_set_body2(constraints, cid, b2)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    eye3 = wp.identity(3, dtype=wp.float32)

    # --- BallSocket sub (base = 0) -------------------------------------
    la1 = wp.quat_rotate_inv(q1, a - pos1)
    la2 = wp.quat_rotate_inv(q2, a - pos2)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_LA1, cid, la1)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_LA2, cid, la2)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_R1, cid, zero3)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_R2, cid, zero3)
    write_float(constraints, _HJ_BS_BASE_C + _BS_OFF_HERTZ, cid, hertz_linear[tid])
    write_float(
        constraints, _HJ_BS_BASE_C + _BS_OFF_DAMPING_RATIO, cid, damping_ratio_linear[tid]
    )
    write_float(constraints, _HJ_BS_BASE_C + _BS_OFF_BIAS_RATE, cid, 0.0)
    write_float(constraints, _HJ_BS_BASE_C + _BS_OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _HJ_BS_BASE_C + _BS_OFF_IMPULSE_COEFF, cid, 0.0)
    write_mat33(constraints, _HJ_BS_BASE_C + _BS_OFF_EFFECTIVE_MASS, cid, eye3)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_ACCUMULATED_IMPULSE, cid, zero3)
    write_vec3(constraints, _HJ_BS_BASE_C + _BS_OFF_BIAS, cid, zero3)

    # --- HingeAngle sub (base = BS_DWORDS) -----------------------------
    axis_local = wp.quat_rotate_inv(q2, w_axis)
    q0 = wp.quat_inverse(q2) * q1
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_HERTZ_LOCK, cid, hertz_lock[tid])
    write_float(
        constraints, _HJ_HA_BASE_C + _HA_OFF_DAMPING_RATIO_LOCK, cid, damping_ratio_lock[tid]
    )
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_HERTZ_LIMIT, cid, hertz_limit[tid])
    write_float(
        constraints, _HJ_HA_BASE_C + _HA_OFF_DAMPING_RATIO_LIMIT, cid, damping_ratio_limit[tid]
    )
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_BIAS_RATE_LOCK, cid, 0.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_MASS_COEFF_LOCK, cid, 1.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_IMPULSE_COEFF_LOCK, cid, 0.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_BIAS_RATE_LIMIT, cid, 0.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_MASS_COEFF_LIMIT, cid, 1.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_IMPULSE_COEFF_LIMIT, cid, 0.0)
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_MIN_ANGLE, cid, wp.sin(min_a * 0.5))
    write_float(constraints, _HJ_HA_BASE_C + _HA_OFF_MAX_ANGLE, cid, wp.sin(max_a * 0.5))
    write_vec3(constraints, _HJ_HA_BASE_C + _HA_OFF_AXIS, cid, axis_local)
    write_quat(constraints, _HJ_HA_BASE_C + _HA_OFF_Q0, cid, q0)
    write_vec3(constraints, _HJ_HA_BASE_C + _HA_OFF_ACCUMULATED_IMPULSE, cid, zero3)
    write_vec3(constraints, _HJ_HA_BASE_C + _HA_OFF_BIAS, cid, zero3)
    write_mat33(constraints, _HJ_HA_BASE_C + _HA_OFF_EFFECTIVE_MASS, cid, eye3)
    write_mat33(constraints, _HJ_HA_BASE_C + _HA_OFF_JACOBIAN, cid, eye3)
    write_int(constraints, _HJ_HA_BASE_C + _HA_OFF_CLAMP, cid, _CLAMP_NONE)

    # --- AngularMotor sub (base = BS_DWORDS + HA_DWORDS) ----------------
    a1 = wp.normalize(w_axis)
    a2 = a1
    motor_local_axis1 = wp.quat_rotate_inv(q1, a1)
    motor_local_axis2 = wp.quat_rotate_inv(q2, a2)
    write_vec3(constraints, _HJ_AM_BASE_C + _AM_OFF_LOCAL_AXIS1, cid, motor_local_axis1)
    write_vec3(constraints, _HJ_AM_BASE_C + _AM_OFF_LOCAL_AXIS2, cid, motor_local_axis2)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_VELOCITY, cid, target_vel)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_MAX_FORCE, cid, mf)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_HERTZ, cid, hertz_motor[tid])
    write_float(
        constraints, _HJ_AM_BASE_C + _AM_OFF_DAMPING_RATIO, cid, damping_ratio_motor[tid]
    )
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_MAX_LAMBDA, cid, 0.0)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_MASS_COEFF, cid, 1.0)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_IMPULSE_COEFF, cid, 0.0)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_EFFECTIVE_MASS, cid, 0.0)
    write_float(constraints, _HJ_AM_BASE_C + _AM_OFF_ACCUMULATED_IMPULSE, cid, 0.0)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# All three entry points read the body pair *once* from the shared
# header, build a :class:`ConstraintBodies` carrier, and forward the
# same instance to each of the three sub ``*_at`` funcs in sequence.
# The body indices stay in registers across the sub-calls -- the
# compiler also typically keeps the body-data loads from the first
# sub-call hot for the second and third.


@wp.func
def hinge_joint_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Run the three sub-prepare passes back-to-back on a fused column.

    Order matches ``Jitter2.HingeJoint`` (HingeAngle -> BallSocket ->
    AngularMotor). Each sub-pass reads/writes only its own sub-block
    via the corresponding ``*_at`` func, and writes back velocity /
    angular_velocity to the same body pair via the shared header.
    """
    b1 = constraint_get_body1(constraints, cid)
    b2 = constraint_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)

    hinge_angle_prepare_for_iteration_at(
        constraints, cid, _HJ_HA_BASE_C, bodies, body_pair, idt
    )
    ball_socket_prepare_for_iteration_at(
        constraints, cid, _HJ_BS_BASE_C, bodies, body_pair, idt
    )
    angular_motor_prepare_for_iteration_at(
        constraints, cid, _HJ_AM_BASE_C, bodies, body_pair, idt
    )


@wp.func
def hinge_joint_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Run one PGS iteration of all three sub-constraints sequentially.

    Gauss-Seidel propagation: each sub-iteration sees the body
    velocities updated by the previous one, so the three corrections
    converge inside a single PGS sweep instead of needing three
    partition launches. ``use_bias`` is forwarded to each sub-iterate
    so the Box2D v3 TGS-soft relax pass zeroes the rigid-lock bias.
    """
    b1 = constraint_get_body1(constraints, cid)
    b2 = constraint_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)

    hinge_angle_iterate_at(constraints, cid, _HJ_HA_BASE_C, bodies, body_pair, idt, use_bias)
    ball_socket_iterate_at(constraints, cid, _HJ_BS_BASE_C, bodies, body_pair, idt, use_bias)
    angular_motor_iterate_at(constraints, cid, _HJ_AM_BASE_C, bodies, body_pair, idt, use_bias)


@wp.func
def hinge_joint_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Combined world-frame wrench (force, torque) on body2.

    Sum of the three sub-constraints' contributions:

      * BallSocket: linear constraint force + its moment arm torque.
      * HingeAngle: pure angular reaction (limit + lock torques).
      * AngularMotor: pure angular drive torque.

    Pure-angular sub-constraints contribute zero force; the sum is the
    total joint reaction wrench felt by body 2 over the most recent
    substep (impulse / substep_dt).
    """
    b1 = constraint_get_body1(constraints, cid)
    b2 = constraint_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)

    f_bs, t_bs = ball_socket_world_wrench_at(constraints, cid, _HJ_BS_BASE_C, idt)
    f_ha, t_ha = hinge_angle_world_wrench_at(constraints, cid, _HJ_HA_BASE_C, idt)
    f_am, t_am = angular_motor_world_wrench_at(
        constraints, cid, _HJ_AM_BASE_C, bodies, body_pair, idt
    )

    return f_bs + f_ha + f_am, t_bs + t_ha + t_am


@wp.func
def hinge_joint_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
) -> wp.spatial_vector:
    """Combined position-level constraint residual for a fused hinge joint.

    Sums the three sub-constraints into one :class:`wp.spatial_vector`:

      * ``spatial_top`` = BallSocket anchor error ``p2 - p1`` [m].
      * ``spatial_bottom`` = HingeAngle (err_x, err_y, err_z_limit)
        with the angular-motor's PD position error folded into the
        z component (motor and limit share the axial axis; reporting
        the sum gives the full axial deviation from target when both
        are active).

    Residuals are computed from the persisted sub-block state in the
    same way as the stand-alone :func:`*_world_error_at` funcs, so the
    values match what each sub-constraint would report in isolation.
    """
    b1 = constraint_get_body1(constraints, cid)
    b2 = constraint_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    e_bs = ball_socket_world_error_at(constraints, cid, _HJ_BS_BASE_C, bodies, body_pair)
    e_ha = hinge_angle_world_error_at(constraints, cid, _HJ_HA_BASE_C, bodies, body_pair)
    e_am = angular_motor_world_error_at(constraints, cid, _HJ_AM_BASE_C)
    lin = wp.spatial_top(e_bs) + wp.spatial_top(e_ha) + wp.spatial_top(e_am)
    ang = wp.spatial_bottom(e_bs) + wp.spatial_bottom(e_ha) + wp.spatial_bottom(e_am)
    return wp.spatial_vector(lin, ang)
