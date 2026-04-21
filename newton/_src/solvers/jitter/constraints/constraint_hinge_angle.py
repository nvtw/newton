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
  :func:`newton._src.solvers.jitter.helpers.math_helpers.qmatrix_project_multiply_left_right`
* ``MathHelper.CreateOrthonormal(v)``        ->
  :func:`newton._src.solvers.jitter.helpers.math_helpers.create_orthonormal`
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_HINGE_ANGLE,
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_ANGULAR,
    DEFAULT_HERTZ_LIMIT,
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
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.helpers.math_helpers import (
    create_orthonormal,
    qmatrix_project_multiply_left_right,
)

__all__ = [
    "HA_DWORDS",
    "HingeAngleData",
    "hinge_angle_get_accumulated_impulse",
    "hinge_angle_get_axis",
    "hinge_angle_get_bias",
    "hinge_angle_get_bias_rate_lock",
    "hinge_angle_get_bias_rate_limit",
    "hinge_angle_get_body1",
    "hinge_angle_get_body2",
    "hinge_angle_get_clamp",
    "hinge_angle_get_damping_ratio_limit",
    "hinge_angle_get_damping_ratio_lock",
    "hinge_angle_get_effective_mass",
    "hinge_angle_get_hertz_limit",
    "hinge_angle_get_hertz_lock",
    "hinge_angle_get_impulse_coeff_lock",
    "hinge_angle_get_impulse_coeff_limit",
    "hinge_angle_get_jacobian",
    "hinge_angle_get_mass_coeff_limit",
    "hinge_angle_get_mass_coeff_lock",
    "hinge_angle_get_max_angle",
    "hinge_angle_get_min_angle",
    "hinge_angle_get_q0",
    "hinge_angle_initialize_kernel",
    "hinge_angle_iterate",
    "hinge_angle_iterate_at",
    "hinge_angle_prepare_for_iteration",
    "hinge_angle_prepare_for_iteration_at",
    "hinge_angle_set_accumulated_impulse",
    "hinge_angle_set_axis",
    "hinge_angle_set_bias",
    "hinge_angle_set_bias_rate_limit",
    "hinge_angle_set_bias_rate_lock",
    "hinge_angle_set_body1",
    "hinge_angle_set_body2",
    "hinge_angle_set_clamp",
    "hinge_angle_set_damping_ratio_limit",
    "hinge_angle_set_damping_ratio_lock",
    "hinge_angle_set_effective_mass",
    "hinge_angle_set_hertz_limit",
    "hinge_angle_set_hertz_lock",
    "hinge_angle_set_impulse_coeff_limit",
    "hinge_angle_set_impulse_coeff_lock",
    "hinge_angle_set_jacobian",
    "hinge_angle_set_mass_coeff_limit",
    "hinge_angle_set_mass_coeff_lock",
    "hinge_angle_set_max_angle",
    "hinge_angle_set_min_angle",
    "hinge_angle_set_q0",
    "hinge_angle_world_wrench",
    "hinge_angle_world_wrench_at",
]

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

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation; see :func:`soft_constraint_coefficients`). Two
    # independent (hertz, damping_ratio) pairs:
    #
    #   * ``*_lock``  governs the two perpendicular-to-hinge angular
    #     DoFs (rows 0/1 of the Jacobian) -- the "axis lock".
    #   * ``*_limit`` governs the third (axial) DoF when the joint hits
    #     a min/max angle limit (row 2 of the Jacobian, only active when
    #     ``clamp != _CLAMP_NONE``).
    #
    # Persisted across substeps; the six derived per-substep
    # coefficients below are recomputed every prepare from the current
    # substep dt.
    hertz_lock: wp.float32
    damping_ratio_lock: wp.float32
    hertz_limit: wp.float32
    damping_ratio_limit: wp.float32

    # Cached per-substep coefficients for the lock pair (rows 0/1).
    bias_rate_lock: wp.float32
    mass_coeff_lock: wp.float32
    impulse_coeff_lock: wp.float32

    # Cached per-substep coefficients for the limit pair (row 2).
    bias_rate_limit: wp.float32
    mass_coeff_limit: wp.float32
    impulse_coeff_limit: wp.float32

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
_OFF_HERTZ_LOCK = wp.constant(dword_offset_of(HingeAngleData, "hertz_lock"))
_OFF_DAMPING_RATIO_LOCK = wp.constant(dword_offset_of(HingeAngleData, "damping_ratio_lock"))
_OFF_HERTZ_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "hertz_limit"))
_OFF_DAMPING_RATIO_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "damping_ratio_limit"))
_OFF_BIAS_RATE_LOCK = wp.constant(dword_offset_of(HingeAngleData, "bias_rate_lock"))
_OFF_MASS_COEFF_LOCK = wp.constant(dword_offset_of(HingeAngleData, "mass_coeff_lock"))
_OFF_IMPULSE_COEFF_LOCK = wp.constant(dword_offset_of(HingeAngleData, "impulse_coeff_lock"))
_OFF_BIAS_RATE_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "bias_rate_limit"))
_OFF_MASS_COEFF_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "mass_coeff_limit"))
_OFF_IMPULSE_COEFF_LIMIT = wp.constant(dword_offset_of(HingeAngleData, "impulse_coeff_limit"))
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
def hinge_angle_get_hertz_lock(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ_LOCK, cid)


@wp.func
def hinge_angle_set_hertz_lock(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ_LOCK, cid, v)


@wp.func
def hinge_angle_get_damping_ratio_lock(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_RATIO_LOCK, cid)


@wp.func
def hinge_angle_set_damping_ratio_lock(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO_LOCK, cid, v)


@wp.func
def hinge_angle_get_hertz_limit(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ_LIMIT, cid)


@wp.func
def hinge_angle_set_hertz_limit(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ_LIMIT, cid, v)


@wp.func
def hinge_angle_get_damping_ratio_limit(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_RATIO_LIMIT, cid)


@wp.func
def hinge_angle_set_damping_ratio_limit(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO_LIMIT, cid, v)


@wp.func
def hinge_angle_get_bias_rate_lock(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_RATE_LOCK, cid)


@wp.func
def hinge_angle_set_bias_rate_lock(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_RATE_LOCK, cid, v)


@wp.func
def hinge_angle_get_mass_coeff_lock(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF_LOCK, cid)


@wp.func
def hinge_angle_set_mass_coeff_lock(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF_LOCK, cid, v)


@wp.func
def hinge_angle_get_impulse_coeff_lock(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF_LOCK, cid)


@wp.func
def hinge_angle_set_impulse_coeff_lock(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF_LOCK, cid, v)


@wp.func
def hinge_angle_get_bias_rate_limit(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_BIAS_RATE_LIMIT, cid)


@wp.func
def hinge_angle_set_bias_rate_limit(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_BIAS_RATE_LIMIT, cid, v)


@wp.func
def hinge_angle_get_mass_coeff_limit(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF_LIMIT, cid)


@wp.func
def hinge_angle_set_mass_coeff_limit(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF_LIMIT, cid, v)


@wp.func
def hinge_angle_get_impulse_coeff_limit(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF_LIMIT, cid)


@wp.func
def hinge_angle_set_impulse_coeff_limit(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF_LIMIT, cid, v)


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
    hertz_lock: wp.array[wp.float32],
    damping_ratio_lock: wp.array[wp.float32],
    hertz_limit: wp.array[wp.float32],
    damping_ratio_limit: wp.array[wp.float32],
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

    hinge_angle_set_hertz_lock(constraints, cid, hertz_lock[tid])
    hinge_angle_set_damping_ratio_lock(constraints, cid, damping_ratio_lock[tid])
    hinge_angle_set_hertz_limit(constraints, cid, hertz_limit[tid])
    hinge_angle_set_damping_ratio_limit(constraints, cid, damping_ratio_limit[tid])
    hinge_angle_set_bias_rate_lock(constraints, cid, 0.0)
    hinge_angle_set_mass_coeff_lock(constraints, cid, 1.0)
    hinge_angle_set_impulse_coeff_lock(constraints, cid, 0.0)
    hinge_angle_set_bias_rate_limit(constraints, cid, 0.0)
    hinge_angle_set_mass_coeff_limit(constraints, cid, 1.0)
    hinge_angle_set_impulse_coeff_limit(constraints, cid, 0.0)

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
#
# Two access levels mirror the BallSocket pattern:
#
# * ``*_at`` variants take an explicit ``base_offset`` (dword offset of
#   the hinge-angle sub-block within its column) and a
#   :class:`ConstraintBodies` carrier with the body indices. The fused
#   :data:`CONSTRAINT_TYPE_HINGE_JOINT` constraint calls these with
#   ``base_offset = HJ_HA_BASE`` and the body pair read once from the
#   fused header.
# * The plain ``hinge_angle_prepare_for_iteration`` /
#   ``hinge_angle_iterate`` / ``hinge_angle_world_wrench`` are thin
#   wrappers used by the unified dispatcher; they read body1/body2 from
#   their own column header and forward to the ``*_at`` variant with
#   ``base_offset = 0``.


@wp.func
def hinge_angle_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable ``PrepareForIterationHingeAngle`` (HingeAngle.cs:109).

    See :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    axis = read_vec3(constraints, base_offset + _OFF_AXIS, cid)
    q0 = read_quat(constraints, base_offset + _OFF_Q0, cid)
    min_angle = read_float(constraints, base_offset + _OFF_MIN_ANGLE, cid)
    max_angle = read_float(constraints, base_offset + _OFF_MAX_ANGLE, cid)
    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    # Recompute soft-constraint coefficients for both pairs from the
    # current substep dt. Rows 0/1 (perpendicular-to-hinge lock) use the
    # ``*_lock`` knobs, row 2 (axial limit) uses the ``*_limit`` knobs.
    dt = 1.0 / idt
    hertz_lock = read_float(constraints, base_offset + _OFF_HERTZ_LOCK, cid)
    damping_ratio_lock = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LOCK, cid)
    hertz_limit = read_float(constraints, base_offset + _OFF_HERTZ_LIMIT, cid)
    damping_ratio_limit = read_float(constraints, base_offset + _OFF_DAMPING_RATIO_LIMIT, cid)
    bias_rate_lock, mass_coeff_lock, impulse_coeff_lock = soft_constraint_coefficients(
        hertz_lock, damping_ratio_lock, dt
    )
    bias_rate_limit, mass_coeff_limit, impulse_coeff_limit = soft_constraint_coefficients(
        hertz_limit, damping_ratio_limit, dt
    )
    write_float(constraints, base_offset + _OFF_BIAS_RATE_LOCK, cid, bias_rate_lock)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_LOCK, cid, mass_coeff_lock)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LOCK, cid, impulse_coeff_lock)
    write_float(constraints, base_offset + _OFF_BIAS_RATE_LIMIT, cid, bias_rate_limit)
    write_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid, mass_coeff_limit)
    write_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid, impulse_coeff_limit)

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

    # quat0.W < 0 -> flip error, m0, **and the warm-start impulse** so we
    # always pick the short-rotation branch (avoids the 2pi wrap-around).
    #
    # The accumulated impulse ``acc`` is persisted across substeps and is
    # expressed in the sign convention of the Jacobian at the time it
    # was written. Because the sign of ``m0`` (and hence of every row of
    # the Jacobian ``J = m0^T @ {p0, p1, axis}``) flips on this branch,
    # the previously-stored ``acc`` must flip alongside it; otherwise the
    # warm-start ``J @ acc`` applies the impulse in the *wrong* direction
    # on the first substep after ``quat0.W`` crosses zero, kicking the
    # bodies with an un-physical pulse that the remaining iterations
    # can't fully absorb (visible as a sudden angular-velocity burst on
    # every half-revolution). Flipping ``acc`` keeps the stored impulse
    # consistent with the flipped Jacobian, so the warm-start is
    # direction-correct and the only residual is the sub-step numerical
    # noise the iterate step already handles.
    #
    # NB: this is a genuine bug in upstream Jitter2's ``HingeAngle.cs``
    # too (the reference also omits the ``acc`` flip). Covered by
    # ``TestHingeAngle.test_wrap_around_does_not_inject_energy``.
    if quat0[3] < 0.0:
        err_x = -err_x
        err_y = -err_y
        err_z = -err_z
        m0 = m0 * (-1.0)
        acc = -acc

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
    # Box2D / Bepu soft-constraint formulation: keep this *unsoftened*;
    # softness lives in the per-row mass_coeff / impulse_coeff scaling
    # in the iterate path.
    inv_inertia_sum = inv_inertia1 + inv_inertia2
    eff = wp.transpose(jacobian) @ (inv_inertia_sum @ jacobian)

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

    # Per-axis velocity-correction bias: rows 0/1 use the lock bias_rate,
    # row 2 uses the limit bias_rate. Both are already in 1/s units.
    bias = wp.vec3f(
        err_x * bias_rate_lock,
        err_y * bias_rate_lock,
        err_z * bias_rate_limit,
    )

    # Persist all the prepared state.
    write_mat33(constraints, base_offset + _OFF_JACOBIAN, cid, jacobian)
    write_mat33(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid, wp.inverse(eff))
    write_int(constraints, base_offset + _OFF_CLAMP, cid, clamp)
    write_vec3(constraints, base_offset + _OFF_BIAS, cid, bias)
    write_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    # Warm start: apply +- J * (InvI * acc) to the two body angular
    # velocities. C# writes the inner Transform first (Transform(acc, J)
    # = J * acc), then outer Transform by InvI (= InvI * (J * acc)).
    j_acc = jacobian @ acc
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] + inv_inertia1 @ j_acc
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] - inv_inertia2 @ j_acc


@wp.func
def hinge_angle_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Composable ``IterateHingeAngle`` (HingeAngle.cs:132).

    See :func:`ball_socket_iterate_at` for the ``base_offset`` /
    ``body_pair`` contract. ``use_bias`` is the Box2D v3 TGS-soft
    ``useBias`` flag: ``True`` for the main solve, ``False`` for the
    relax pass.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    jacobian = read_mat33(constraints, base_offset + _OFF_JACOBIAN, cid)
    eff = read_mat33(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid)
    if use_bias:
        bias = read_vec3(constraints, base_offset + _OFF_BIAS, cid)
    else:
        bias = wp.vec3f(0.0, 0.0, 0.0)
    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    mass_coeff_lock = read_float(constraints, base_offset + _OFF_MASS_COEFF_LOCK, cid)
    impulse_coeff_lock = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LOCK, cid)
    mass_coeff_limit = read_float(constraints, base_offset + _OFF_MASS_COEFF_LIMIT, cid)
    impulse_coeff_limit = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF_LIMIT, cid)
    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)

    # jv = J^T * (w1 - w2)  (TransposedTransform(w, J) = J^T * w).
    jv = wp.transpose(jacobian) @ (angular_velocity1 - angular_velocity2)

    # Box2D / Bepu soft-constraint PGS update:
    #   lambda = -mass_coeff * EffectiveMass * (jv + bias)
    #            - impulse_coeff * accumulated_impulse
    # Each row uses its own (mass_coeff, impulse_coeff) pair: rows 0/1
    # are the perpendicular-to-hinge "lock" coefficients, row 2 is the
    # axial "limit" coefficient.
    lam_full = -(eff @ (jv + bias))
    lam = wp.vec3f(
        mass_coeff_lock * lam_full[0] - impulse_coeff_lock * acc[0],
        mass_coeff_lock * lam_full[1] - impulse_coeff_lock * acc[1],
        mass_coeff_limit * lam_full[2] - impulse_coeff_limit * acc[2],
    )

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

    write_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    # Apply +-J * InvI * lambda. C# does Transform(lam, J) = J * lam,
    # then Transform(_, InvI) = InvI * _.
    j_lam = jacobian @ lam
    bodies.angular_velocity[b1] = angular_velocity1 + inv_inertia1 @ j_lam
    bodies.angular_velocity[b2] = angular_velocity2 - inv_inertia2 @ j_lam


@wp.func
def hinge_angle_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    idt: wp.float32,
):
    """Composable hinge-angle wrench on body2; see
    :func:`hinge_angle_world_wrench` for semantics."""
    acc = read_vec3(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    jacobian = read_mat33(constraints, base_offset + _OFF_JACOBIAN, cid)
    torque = -(jacobian @ acc) * idt
    return wp.vec3f(0.0, 0.0, 0.0), torque


@wp.func
def hinge_angle_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationHingeAngle`` (HingeAngle.cs:109).

    Thin wrapper: see :func:`hinge_angle_prepare_for_iteration_at`.
    """
    b1 = hinge_angle_get_body1(constraints, cid)
    b2 = hinge_angle_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    hinge_angle_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def hinge_angle_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Direct port of ``IterateHingeAngle`` (HingeAngle.cs:132).

    Thin wrapper: see :func:`hinge_angle_iterate_at`.
    """
    b1 = hinge_angle_get_body1(constraints, cid)
    b2 = hinge_angle_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    hinge_angle_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


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
    return hinge_angle_world_wrench_at(constraints, cid, 0, idt)
