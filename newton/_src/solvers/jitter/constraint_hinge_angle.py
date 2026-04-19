# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of Jitter2's HingeAngle constraint.

Direct translation of ``Jitter2.Dynamics.Constraints.HingeAngle``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/HingeAngle.cs``).
HingeAngle is the angular half of a revolute joint: it locks the two
rotational DoF *perpendicular* to a hinge axis (so only rotation around
that axis remains free) and optionally clamps that remaining axis to a
``[min_angle, max_angle]`` range. The translational lock is the job of
a co-located :class:`BallSocketData`; together they form a hinge.

Storage: same conventions as :mod:`constraint_ball_socket` -- the
``@wp.struct HingeAngleData`` is a *schema only*, used at module load
to derive dword offsets into the shared :class:`ConstraintContainer`
(column-major-by-cid). All runtime kernels read/write fields via the
typed ``hinge_angle_get_* / hinge_angle_set_*`` accessors.

Initialisation: identical pattern to ball-socket -- a kernel
(:func:`hinge_angle_initialize_kernel`) launched once by
:meth:`WorldBuilder.finalize` snapshots the body orientations to derive
``axis`` (in body 2's local frame) and ``q0 = q2^* * q1``. The user-
facing API stays plain Python and just carries
``(body1, body2, world_axis, min_angle_rad, max_angle_rad)``.

Mapping summary (mirrors the BallSocket header for consistency):

* ``JVector``                            -> ``wp.vec3f``
* ``JQuaternion``                        -> ``wp.quatf``
* ``JMatrix``                            -> ``wp.mat33f``    (row-major)
* ``JVector.Transform(v, M)``            -> ``M @ v``        (column-vec)
* ``JVector.Transform(v, q)``            -> ``wp.quat_rotate(q, v)``
* ``JVector.ConjugatedTransform(v, q)``  -> ``wp.quat_rotate_inv(q, v)``
* ``JVector.TransposedTransform(v, M)``  -> ``wp.transpose(M) @ v``
* ``JMatrix.TransposedMultiply(A, B)``   -> ``wp.transpose(A) @ B``
* ``JMatrix.Multiply(A, B)``             -> ``A @ B``
* ``JMatrix.Inverse(M, out M)``          -> ``wp.inverse(M)``
* ``JQuaternion.Conjugate(q)``           -> ``wp.quat_inverse(q)``  (unit-quat conjugate == inverse)
* ``QMatrix.ProjectMultiplyLeftRight(L, R)`` ->
  :func:`newton._src.solvers.jitter.math_helpers.qmatrix_project_multiply_left_right`
* ``MathHelper.CreateOrthonormal(v)``        ->
  :func:`newton._src.solvers.jitter.math_helpers.create_orthonormal`
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_HINGE_ANGLE,
    ConstraintContainer,
    assert_constraint_header,
    constraint_set_type,
    read_float,
    read_int,
    read_mat33,
    read_quat,
    read_vec3,
    write_float,
    write_int,
    write_mat33,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import (
    create_orthonormal,
    qmatrix_project_multiply_left_right,
)

__all__ = [
    "DEFAULT_ANGULAR_BIAS",
    "DEFAULT_ANGULAR_LIMIT_BIAS",
    "DEFAULT_ANGULAR_LIMIT_SOFTNESS",
    "DEFAULT_ANGULAR_SOFTNESS",
    "HA_DWORDS",
    "HingeAngleData",
    "hinge_angle_get_accumulated_impulse",
    "hinge_angle_get_axis",
    "hinge_angle_get_bias",
    "hinge_angle_get_bias_factor",
    "hinge_angle_get_body1",
    "hinge_angle_get_body2",
    "hinge_angle_get_clamp",
    "hinge_angle_get_effective_mass",
    "hinge_angle_get_jacobian",
    "hinge_angle_get_limit_bias",
    "hinge_angle_get_limit_softness",
    "hinge_angle_get_max_angle",
    "hinge_angle_get_min_angle",
    "hinge_angle_get_q0",
    "hinge_angle_get_softness",
    "hinge_angle_initialize_kernel",
    "hinge_angle_iterate",
    "hinge_angle_prepare_for_iteration",
    "hinge_angle_set_accumulated_impulse",
    "hinge_angle_set_axis",
    "hinge_angle_set_bias",
    "hinge_angle_set_bias_factor",
    "hinge_angle_set_body1",
    "hinge_angle_set_body2",
    "hinge_angle_set_clamp",
    "hinge_angle_set_effective_mass",
    "hinge_angle_set_jacobian",
    "hinge_angle_set_limit_bias",
    "hinge_angle_set_limit_softness",
    "hinge_angle_set_max_angle",
    "hinge_angle_set_min_angle",
    "hinge_angle_set_q0",
    "hinge_angle_set_softness",
    "hinge_angle_world_wrench",
]


# ---------------------------------------------------------------------------
# Constants (mirrors Jitter2.Dynamics.Constraints.Constraint)
# ---------------------------------------------------------------------------

# Defaults from Constraint.cs:192-201. Exposed at module scope so callers
# (e.g. the world builder) can override per-constraint if they want.
DEFAULT_ANGULAR_SOFTNESS = wp.constant(wp.float32(0.001))
DEFAULT_ANGULAR_BIAS = wp.constant(wp.float32(0.2))
DEFAULT_ANGULAR_LIMIT_SOFTNESS = wp.constant(wp.float32(0.001))
DEFAULT_ANGULAR_LIMIT_BIAS = wp.constant(wp.float32(0.1))

# Clamp state values for the limit branch in PrepareForIteration /
# Iterate. Match the C# meaning: 0 = within limits, 1 = clamped at max,
# 2 = clamped at min.
_CLAMP_NONE = wp.constant(wp.int32(0))
_CLAMP_MAX = wp.constant(wp.int32(1))
_CLAMP_MIN = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class HingeAngleData:
    """Per-constraint dword-layout schema for a hinge-angle constraint.

    *Schema only.* Field order fixes dword offsets; runtime kernels
    operate on the shared :class:`ConstraintContainer` via the typed
    accessors below. Layout intentionally mirrors C#'s ``HingeAngleData``
    field-for-field (minus the dispatch fields) so ports of related
    constraints (TwistAngle, ConeLimit, ...) line up.

    Fields are arranged in struct-natural order with no manual padding;
    everything is 32-bit so the struct is dword-aligned end-to-end.

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    # Body indices (replaces JHandle<RigidBodyData>).
    body1: wp.int32
    body2: wp.int32

    # Angular limits, stored as ``sin(limit/2)`` (matches C# Initialize)
    # so the limit comparison is a plain float compare against the
    # quaternion's xyz component projected onto the hinge axis.
    min_angle: wp.float32
    max_angle: wp.float32

    bias_factor: wp.float32
    limit_bias: wp.float32

    limit_softness: wp.float32
    softness: wp.float32

    # Hinge axis in body 2's local frame and the relative rest
    # orientation q0 = q2^* * q1 (set by Initialize from current body
    # orientations).
    axis: wp.vec3f
    q0: wp.quatf

    accumulated_impulse: wp.vec3f
    bias: wp.vec3f

    effective_mass: wp.mat33f
    jacobian: wp.mat33f

    # Limit clamp state for the third axis: 0 / 1 / 2 (see _CLAMP_*).
    clamp: wp.int32


# Enforce the global constraint header contract (constraint_type / body1
# / body2 at dwords 0 / 1 / 2) at import time so a future field reorder
# fails loudly here instead of silently mis-tagging columns or
# scrambling body indices at runtime.
assert_constraint_header(HingeAngleData)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(HingeAngleData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(HingeAngleData, "body2"))
_OFF_MIN_ANGLE = wp.constant(dword_offset_of(HingeAngleData, "min_angle"))
_OFF_MAX_ANGLE = wp.constant(dword_offset_of(HingeAngleData, "max_angle"))
_OFF_BIAS_FACTOR = wp.constant(dword_offset_of(HingeAngleData, "bias_factor"))
_OFF_LIMIT_BIAS = wp.constant(dword_offset_of(HingeAngleData, "limit_bias"))
_OFF_LIMIT_SOFTNESS = wp.constant(dword_offset_of(HingeAngleData, "limit_softness"))
_OFF_SOFTNESS = wp.constant(dword_offset_of(HingeAngleData, "softness"))
_OFF_AXIS = wp.constant(dword_offset_of(HingeAngleData, "axis"))
_OFF_Q0 = wp.constant(dword_offset_of(HingeAngleData, "q0"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(HingeAngleData, "accumulated_impulse"))
_OFF_BIAS = wp.constant(dword_offset_of(HingeAngleData, "bias"))
_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(HingeAngleData, "effective_mass"))
_OFF_JACOBIAN = wp.constant(dword_offset_of(HingeAngleData, "jacobian"))
_OFF_CLAMP = wp.constant(dword_offset_of(HingeAngleData, "clamp"))

#: Total dword count of one hinge-angle constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
HA_DWORDS: int = num_dwords(HingeAngleData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def hinge_angle_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def hinge_angle_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def hinge_angle_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def hinge_angle_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def hinge_angle_get_min_angle(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MIN_ANGLE, cid)


@wp.func
def hinge_angle_set_min_angle(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MIN_ANGLE, cid, v)


@wp.func
def hinge_angle_get_max_angle(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_ANGLE, cid)


@wp.func
def hinge_angle_set_max_angle(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_ANGLE, cid, v)


@wp.func
def hinge_angle_get_bias_factor(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_FACTOR, cid)


@wp.func
def hinge_angle_set_bias_factor(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_FACTOR, cid, v)


@wp.func
def hinge_angle_get_limit_bias(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_LIMIT_BIAS, cid)


@wp.func
def hinge_angle_set_limit_bias(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_LIMIT_BIAS, cid, v)


@wp.func
def hinge_angle_get_limit_softness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_LIMIT_SOFTNESS, cid)


@wp.func
def hinge_angle_set_limit_softness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_LIMIT_SOFTNESS, cid, v)


@wp.func
def hinge_angle_get_softness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_SOFTNESS, cid)


@wp.func
def hinge_angle_set_softness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_SOFTNESS, cid, v)


@wp.func
def hinge_angle_get_axis(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_AXIS, cid)


@wp.func
def hinge_angle_set_axis(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_AXIS, cid, v)


@wp.func
def hinge_angle_get_q0(c: ConstraintContainer, cid: wp.int32) -> wp.quatf:
    return read_quat(c, _OFF_Q0, cid)


@wp.func
def hinge_angle_set_q0(c: ConstraintContainer, cid: wp.int32, v: wp.quatf):
    write_quat(c, _OFF_Q0, cid, v)


@wp.func
def hinge_angle_get_accumulated_impulse(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_ACCUMULATED_IMPULSE, cid)


@wp.func
def hinge_angle_set_accumulated_impulse(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_ACCUMULATED_IMPULSE, cid, v)


@wp.func
def hinge_angle_get_bias(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_BIAS, cid)


@wp.func
def hinge_angle_set_bias(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_BIAS, cid, v)


@wp.func
def hinge_angle_get_effective_mass(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_EFFECTIVE_MASS, cid)


@wp.func
def hinge_angle_set_effective_mass(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_EFFECTIVE_MASS, cid, v)


@wp.func
def hinge_angle_get_jacobian(c: ConstraintContainer, cid: wp.int32) -> wp.mat33f:
    return read_mat33(c, _OFF_JACOBIAN, cid)


@wp.func
def hinge_angle_set_jacobian(c: ConstraintContainer, cid: wp.int32, v: wp.mat33f):
    write_mat33(c, _OFF_JACOBIAN, cid, v)


@wp.func
def hinge_angle_get_clamp(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_CLAMP, cid)


@wp.func
def hinge_angle_set_clamp(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_CLAMP, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel; mirrors host-side BallSocket init pattern)
# ---------------------------------------------------------------------------


@wp.kernel
def hinge_angle_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    world_axis: wp.array[wp.vec3f],
    min_angle_rad: wp.array[wp.float32],
    max_angle_rad: wp.array[wp.float32],
):
    """Pack one batch of hinge-angle descriptors into ``constraints``.

    Direct port of ``HingeAngle.Initialize`` (HingeAngle.cs:73-92).
    Snapshots the current body 1 / body 2 orientations to compute:

      * ``axis = q2^* * world_axis`` (hinge axis in body 2's local frame),
      * ``q0   = q2^* * q1``         (relative rest orientation).

    The angular limits are stored as ``sin(angle/2)`` so the limit
    comparison in :func:`_hinge_angle_prepare_for_iteration` is a plain
    float compare against the projected quaternion xyz component.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only body orientations are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        world_axis: Hinge axes in *world* space [num_in_batch].
            Must be unit length; the kernel does not renormalise.
        min_angle_rad: Lower angular limits [num_in_batch] [rad].
        max_angle_rad: Upper angular limits [num_in_batch] [rad].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    w_axis = world_axis[tid]
    min_a = min_angle_rad[tid]
    max_a = max_angle_rad[tid]

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    # Axis in body-2 local frame: data.Axis = q2^* * axis (Jitter uses
    # ConjugatedTransform here, which is the inverse rotation; for a
    # unit quaternion that's quat_rotate_inv).
    axis_local = wp.quat_rotate_inv(q2, w_axis)

    # Relative rest orientation: data.Q0 = q2.Conjugate() * q1
    # (== q2^{-1} * q1 for unit quaternions).
    q0 = wp.quat_inverse(q2) * q1

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_HINGE_ANGLE)
    hinge_angle_set_body1(constraints, cid, b1)
    hinge_angle_set_body2(constraints, cid, b2)

    hinge_angle_set_softness(constraints, cid, DEFAULT_ANGULAR_SOFTNESS)
    hinge_angle_set_limit_softness(constraints, cid, DEFAULT_ANGULAR_LIMIT_SOFTNESS)
    hinge_angle_set_bias_factor(constraints, cid, DEFAULT_ANGULAR_BIAS)
    hinge_angle_set_limit_bias(constraints, cid, DEFAULT_ANGULAR_LIMIT_BIAS)

    hinge_angle_set_min_angle(constraints, cid, wp.sin(min_a * 0.5))
    hinge_angle_set_max_angle(constraints, cid, wp.sin(max_a * 0.5))

    hinge_angle_set_axis(constraints, cid, axis_local)
    hinge_angle_set_q0(constraints, cid, q0)

    zero3 = wp.vec3f(0.0, 0.0, 0.0)
    hinge_angle_set_accumulated_impulse(constraints, cid, zero3)
    hinge_angle_set_bias(constraints, cid, zero3)

    eye3 = wp.identity(3, dtype=wp.float32)
    hinge_angle_set_effective_mass(constraints, cid, eye3)
    hinge_angle_set_jacobian(constraints, cid, eye3)

    hinge_angle_set_clamp(constraints, cid, _CLAMP_NONE)


# ---------------------------------------------------------------------------
# Per-iteration math (wp.func helpers + dispatch kernels)
# ---------------------------------------------------------------------------


@wp.func
def hinge_angle_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationHingeAngle`` (HingeAngle.cs:109).

    Builds the 3x3 angular Jacobian (rows: lock-axis-1, lock-axis-2,
    hinge-axis), computes the effective mass with softness, picks up
    the limit clamp state for the third row, and warm-starts the body
    angular velocities with the cached accumulated impulse.

    The translational part of a revolute joint is handled separately by
    a co-located :class:`BallSocketData` -- this kernel only writes
    angular velocities.
    """
    b1 = hinge_angle_get_body1(constraints, cid)
    b2 = hinge_angle_get_body2(constraints, cid)

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    axis = hinge_angle_get_axis(constraints, cid)
    q0 = hinge_angle_get_q0(constraints, cid)
    softness = hinge_angle_get_softness(constraints, cid)
    limit_softness = hinge_angle_get_limit_softness(constraints, cid)
    min_angle = hinge_angle_get_min_angle(constraints, cid)
    max_angle = hinge_angle_get_max_angle(constraints, cid)
    acc = hinge_angle_get_accumulated_impulse(constraints, cid)

    # Two unit vectors perpendicular to the hinge axis. ``axis`` is
    # already unit length (initialise enforces this), and CreateOrthonormal
    # returns a unit vector orthogonal to it; the cross of those two unit
    # vectors is therefore unit length too.
    p0 = create_orthonormal(axis)
    p1 = wp.cross(axis, p0)

    # quat0 = q0 * q1^* * q2  -- the "error quaternion" between the
    # current and rest relative orientations.
    q1_inv = wp.quat_inverse(q1)
    quat0 = q0 * q1_inv * q2

    # error.{x,y,z} = dot({p0,p1,axis}, quat0.xyz)
    quat0_xyz = wp.vec3f(quat0[0], quat0[1], quat0[2])
    err_x = wp.dot(p0, quat0_xyz)
    err_y = wp.dot(p1, quat0_xyz)
    err_z = wp.dot(axis, quat0_xyz)

    # m0 = -1/2 * QMatrix.ProjectMultiplyLeftRight(q0 * q1^*, q2). The
    # 1/2 here pairs with the Jacobian's geometric meaning: for a unit
    # quat, infinitesimal rotation by omega gives dq = 1/2 * (omega, 0)
    # * q, so the angular-velocity-to-quaternion-error Jacobian carries
    # a 1/2.
    qq = q0 * q1_inv
    m0 = qmatrix_project_multiply_left_right(qq, q2) * (-0.5)

    # quat0.W < 0 -> flip both error and m0 so we always pick the
    # short-rotation branch (avoids the 2pi wrap-around).
    if quat0[3] < 0.0:
        err_x = -err_x
        err_y = -err_y
        err_z = -err_z
        m0 = m0 * (-1.0)

    # Jacobian rows (column-vector convention used everywhere here):
    #   row k = (m0^T * row_k_axis)^T  -- C# stores the rows directly via
    #   UnsafeGet(k) = TransposedTransform(p, m0), and TransposedTransform
    #   is M^T * v in JVector's docstring (LinearMath/JVector.cs:233).
    m0_t = wp.transpose(m0)
    j_row0 = m0_t @ p0
    j_row1 = m0_t @ p1
    j_row2 = m0_t @ axis

    jacobian = wp.mat33f(
        j_row0[0], j_row0[1], j_row0[2],
        j_row1[0], j_row1[1], j_row1[2],
        j_row2[0], j_row2[1], j_row2[2],
    )

    # EffectiveMass = J^T * (InvI1 + InvI2) * J  (TransposedMultiply(J, M*J)).
    inv_inertia_sum = inv_inertia1 + inv_inertia2
    eff = wp.transpose(jacobian) @ (inv_inertia_sum @ jacobian)

    # Softness contribution: per-row softness on the diagonal.
    eff[0, 0] = eff[0, 0] + softness * idt
    eff[1, 1] = eff[1, 1] + softness * idt
    eff[2, 2] = eff[2, 2] + limit_softness * idt

    clamp = _CLAMP_NONE
    if err_z > max_angle:
        clamp = _CLAMP_MAX
        err_z = err_z - max_angle
    elif err_z < min_angle:
        clamp = _CLAMP_MIN
        err_z = err_z - min_angle
    else:
        # Within limits: drop the third row of the Jacobian so it no
        # longer affects the solve, and rebuild the third row/col of
        # EffectiveMass to identity-on-diagonal so wp.inverse stays
        # well-conditioned. Mirrors HingeAngle.cs:108-118.
        acc = wp.vec3f(acc[0], acc[1], 0.0)
        eff[2, 2] = 1.0
        eff[0, 2] = 0.0
        eff[2, 0] = 0.0
        eff[1, 2] = 0.0
        eff[2, 1] = 0.0
        # Clear the third Jacobian row -- in our row-major mat33 that's
        # the row indexed [2,:].
        jacobian[2, 0] = 0.0
        jacobian[2, 1] = 0.0
        jacobian[2, 2] = 0.0

    # Final per-axis bias: x,y use bias_factor; z uses limit_bias.
    bias_factor = hinge_angle_get_bias_factor(constraints, cid)
    limit_bias = hinge_angle_get_limit_bias(constraints, cid)

    bias = wp.vec3f(
        err_x * idt * bias_factor,
        err_y * idt * bias_factor,
        err_z * idt * limit_bias,
    )

    # Persist all the prepared state.
    hinge_angle_set_jacobian(constraints, cid, jacobian)
    hinge_angle_set_effective_mass(constraints, cid, wp.inverse(eff))
    hinge_angle_set_clamp(constraints, cid, clamp)
    hinge_angle_set_bias(constraints, cid, bias)
    hinge_angle_set_accumulated_impulse(constraints, cid, acc)

    # Warm start: apply +- J * (InvI * acc) to the two body angular
    # velocities. C# writes the inner Transform first (Transform(acc, J)
    # = J * acc), then outer Transform by InvI (= InvI * (J * acc)).
    j_acc = jacobian @ acc
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] + inv_inertia1 @ j_acc
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] - inv_inertia2 @ j_acc


@wp.func
def hinge_angle_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``IterateHingeAngle`` (HingeAngle.cs:132).

    One PGS-style iteration on the angular Jacobian: project the current
    velocity through the Jacobian, build the corrective impulse, clamp
    the third axis against the limit (if active), and apply the velocity
    delta to both body angular velocities.
    """
    b1 = hinge_angle_get_body1(constraints, cid)
    b2 = hinge_angle_get_body2(constraints, cid)

    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    jacobian = hinge_angle_get_jacobian(constraints, cid)
    eff = hinge_angle_get_effective_mass(constraints, cid)
    bias = hinge_angle_get_bias(constraints, cid)
    acc = hinge_angle_get_accumulated_impulse(constraints, cid)
    softness = hinge_angle_get_softness(constraints, cid)
    limit_softness = hinge_angle_get_limit_softness(constraints, cid)
    clamp = hinge_angle_get_clamp(constraints, cid)

    # jv = J^T * (w1 - w2)  (TransposedTransform(w, J) = J^T * w).
    jv = wp.transpose(jacobian) @ (angular_velocity1 - angular_velocity2)

    softness_vec = wp.vec3f(
        acc[0] * idt * softness,
        acc[1] * idt * softness,
        acc[2] * idt * limit_softness,
    )

    lam = -(eff @ (jv + bias + softness_vec))

    orig_acc = acc
    acc = acc + lam

    if clamp == _CLAMP_MAX:
        # Limit pushes downward only (negative z impulse).
        acc = wp.vec3f(acc[0], acc[1], wp.min(0.0, acc[2]))
    elif clamp == _CLAMP_MIN:
        acc = wp.vec3f(acc[0], acc[1], wp.max(0.0, acc[2]))
    else:
        # No limit active -> drop the z component everywhere so the
        # third row of the solve is a no-op.
        orig_acc = wp.vec3f(orig_acc[0], orig_acc[1], 0.0)
        acc = wp.vec3f(acc[0], acc[1], 0.0)

    lam = acc - orig_acc

    hinge_angle_set_accumulated_impulse(constraints, cid, acc)

    # Apply +-J * InvI * lambda. C# does Transform(lam, J) = J * lam,
    # then Transform(_, InvI) = InvI * _.
    j_lam = jacobian @ lam
    bodies.angular_velocity[b1] = angular_velocity1 + inv_inertia1 @ j_lam
    bodies.angular_velocity[b2] = angular_velocity2 - inv_inertia2 @ j_lam


@wp.func
def hinge_angle_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this constraint exerts on body2.

    Hinge constraints carry no positional component, so ``force`` is
    zero. The torque is ``-jacobian @ accumulated_impulse / substep_dt``;
    sign matches ``iterate``'s ``angular_velocity[b2] -= invI @ (J @ lam)``.
    The cached ``jacobian`` from ``prepare_for_iteration`` is in world
    frame, so no extra rotation is required.
    """
    acc = hinge_angle_get_accumulated_impulse(constraints, cid)
    jacobian = hinge_angle_get_jacobian(constraints, cid)
    torque = -(jacobian @ acc) * idt
    return wp.vec3f(0.0, 0.0, 0.0), torque
