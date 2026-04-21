# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of Jitter2's AngularMotor constraint, extended with a
PD position-target mode.

Base translation of ``Jitter2.Dynamics.Constraints.AngularMotor``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/AngularMotor.cs``).

AngularMotor is a 1-DoF angular constraint that acts between body 1
and body 2 along a hinge axis. It carries **two operating modes**
selected by the gain inputs on :meth:`WorldBuilder.add_angular_motor`
and chosen automatically by :func:`angular_motor_prepare_for_iteration_at`:

* **Velocity target (Jitter2 / Box2D legacy).** When both
  ``stiffness`` and ``damping`` are zero, the constraint drives the
  relative axial angular velocity toward ``velocity`` (rad/s), capped
  by an impulse budget ``max_force * dt`` per substep. Softness comes
  from the Box2D v3 / Bepu ``(hertz, damping_ratio)`` knobs and feeds
  through the ``mass_coeff`` / ``impulse_coeff`` scalars (see
  :func:`soft_constraint_coefficients`). This is a verbatim port of
  the original Jitter2 kernel and keeps the historical behaviour for
  every consumer that was already using it.

* **PD position target (PhoenX / Jolt style).** When ``stiffness > 0``
  or ``damping > 0``, the constraint becomes an *angular*
  spring-damper that holds the relative axial angle at ``target_angle``
  (rad). Gains are in absolute SI units:
  ``stiffness`` in N·m/rad, ``damping`` in N·m·s/rad, matching the
  Jitter2 ``SpringConstraint`` convention of
  :func:`constraint_container.pd_coefficients`. The angle is tracked
  across 2*pi wraps using :func:`math_helpers.revolution_tracker_update`
  so arbitrarily large limit/target angles are well-defined even when
  the joint spins through many full turns.

Both modes share the same constraint header, Jacobian, and
accumulated-impulse storage. Only the per-substep bias / effective-mass
pipeline differs, and there is no allocation/dispatch branching at the
solver level.

Composition: a motorised revolute joint in Jitter is the triple
``BallSocket + HingeAngle + AngularMotor`` -- the ball-socket locks
the linear DoFs, the hinge locks the two angular DoFs perpendicular to
the axis, and the motor drives the remaining axial DoF. Each piece is
its own constraint in our solver too.

Axis convention for the PD path
-------------------------------
PhoenX/Jitter2's ``AngularMotor.PrepareForIteration`` uses the
**body-1** axis only when building the Jacobian and when extracting
the relative angle from the quaternion. The body-2 axis is not read:
the companion HingeAngle constraint already keeps ``axis1`` and
``axis2`` colinear in world space, so using a single axis for the PD
row is both simpler and more accurate -- it avoids the small
misalignment signal from ``q2 * axis_local2`` that would leak into
the drive impulse during a substep. The **velocity path** keeps the
pre-existing per-body Jacobian form because every test in
``test_angular_motor.py`` is phrased in that geometry and the
formulation is momentum-conserving; both are PGS-valid for a 1-DoF
row and differ only in how they behave when the two axes are
momentarily non-parallel.

Storage: same conventions as the other constraint files -- the
``@wp.struct AngularMotorData`` is a *schema only*, used at module
load to derive dword offsets into the shared
:class:`ConstraintContainer` (column-major-by-cid). All runtime kernels
read/write fields via the typed
``angular_motor_get_* / angular_motor_set_*`` accessors.

Initialisation: identical pattern to ball-socket / hinge-angle -- a
kernel (:func:`angular_motor_initialize_kernel`) launched once by
:meth:`WorldBuilder.finalize` snapshots the body orientations to derive
``local_axis1 = q1^* * world_axis1`` and ``local_axis2 = q2^* * world_axis2``.
The user-facing descriptor stays plain Python and just carries
``(body1, body2, world_axis1, world_axis2, target_velocity, max_force)``.
The two world axes are usually the same vector (a hinge axis), but the
constraint allows different per-body axes for completeness.

Note on ``EffectiveMass``: this is a *scalar* (1-DoF constraint), not a
3x3 matrix. The PGS update collapses to a one-line scalar projection.

Mapping summary:

* ``JVector``                            -> ``wp.vec3f``
* ``JQuaternion``                        -> ``wp.quatf``
* ``JVector.Transform(v, q)``            -> ``wp.quat_rotate(q, v)``
* ``JVector.ConjugatedTransform(v, q)``  -> ``wp.quat_rotate_inv(q, v)``
* ``JVector.Transform(v, M)``            -> ``M @ v`` (column-vector)
* ``JVector operator *(a, b)``           -> ``wp.dot(a, b)`` (Jitter overloads ``*`` as dot)
* ``Math.Clamp(x, lo, hi)``              -> ``wp.clamp(x, lo, hi)``
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraint_container import (
    CONSTRAINT_TYPE_ANGULAR_MOTOR,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    pd_coefficients,
    read_float,
    read_int,
    read_quat,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_quat,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.math_helpers import (
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)

__all__ = [
    "AM_DWORDS",
    "AngularMotorData",
    "angular_motor_get_accumulated_impulse",
    "angular_motor_get_body1",
    "angular_motor_get_body2",
    "angular_motor_get_damping",
    "angular_motor_get_damping_ratio",
    "angular_motor_get_effective_mass",
    "angular_motor_get_hertz",
    "angular_motor_get_impulse_coeff",
    "angular_motor_get_inv_initial_orientation",
    "angular_motor_get_local_axis1",
    "angular_motor_get_local_axis2",
    "angular_motor_get_mass_coeff",
    "angular_motor_get_max_force",
    "angular_motor_get_max_lambda",
    "angular_motor_get_pd_beta",
    "angular_motor_get_pd_gamma",
    "angular_motor_get_pd_mass_coeff",
    "angular_motor_get_position_error",
    "angular_motor_get_previous_quaternion_angle",
    "angular_motor_get_revolution_counter",
    "angular_motor_get_stiffness",
    "angular_motor_get_target_angle",
    "angular_motor_get_velocity",
    "angular_motor_initialize_kernel",
    "angular_motor_iterate",
    "angular_motor_iterate_at",
    "angular_motor_prepare_for_iteration",
    "angular_motor_prepare_for_iteration_at",
    "angular_motor_set_accumulated_impulse",
    "angular_motor_set_body1",
    "angular_motor_set_body2",
    "angular_motor_set_damping",
    "angular_motor_set_damping_ratio",
    "angular_motor_set_effective_mass",
    "angular_motor_set_hertz",
    "angular_motor_set_impulse_coeff",
    "angular_motor_set_inv_initial_orientation",
    "angular_motor_set_local_axis1",
    "angular_motor_set_local_axis2",
    "angular_motor_set_mass_coeff",
    "angular_motor_set_max_force",
    "angular_motor_set_max_lambda",
    "angular_motor_set_pd_beta",
    "angular_motor_set_pd_gamma",
    "angular_motor_set_pd_mass_coeff",
    "angular_motor_set_position_error",
    "angular_motor_set_previous_quaternion_angle",
    "angular_motor_set_revolution_counter",
    "angular_motor_set_stiffness",
    "angular_motor_set_target_angle",
    "angular_motor_set_velocity",
    "angular_motor_world_wrench",
    "angular_motor_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class AngularMotorData:
    """Per-constraint dword-layout schema for an angular motor.

    *Schema only.* Field order fixes dword offsets; runtime kernels
    operate on the shared :class:`ConstraintContainer` via the typed
    accessors below. Layout mirrors C#'s ``AngularMotorData``
    field-for-field (minus the dispatch fields).

    Fields are arranged in struct-natural order with no manual padding;
    everything is 32-bit so the struct is dword-aligned end-to-end.

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    local_axis1: wp.vec3f
    local_axis2: wp.vec3f

    # Target relative angular velocity along the (current world) axis
    # [rad/s]. The motor drives ``-w1.axis + w2.axis`` toward this value
    # subject to the per-step impulse cap.
    velocity: wp.float32

    # Maximum motor torque [N*m]. The Iterate kernel converts this to a
    # per-step impulse cap ``max_lambda = max_force * dt = max_force / idt``
    # and clamps the accumulated impulse to ``[-max_lambda, max_lambda]``.
    # ``max_force = 0`` -> disabled motor (max_lambda = 0 clamps the
    # impulse to zero on every PGS pass).
    max_force: wp.float32

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation; see :func:`soft_constraint_coefficients`). The motor
    # carries no positional bias so ``hertz`` here only governs how fast
    # the accumulated impulse decays toward the velocity setpoint -- in
    # spring terms it's a velocity-target spring with this stiffness.
    # Used only on the **velocity path** (``stiffness == 0 and damping == 0``).
    hertz: wp.float32
    damping_ratio: wp.float32

    # PD position-target mode (Jitter2 ``SpringConstraint`` convention).
    # Active whenever ``stiffness > 0`` or ``damping > 0``. When active,
    # the motor holds the axial angle at ``target_angle`` (rad, unbounded
    # across 2*pi wraps) with spring gain ``stiffness`` (N*m/rad) and
    # damping gain ``damping`` (N*m*s/rad); the ``velocity`` field is
    # then interpreted as a feed-forward velocity target and is usually
    # zero. ``max_force`` still caps the per-substep impulse in both
    # modes. See :func:`constraint_container.pd_coefficients`.
    target_angle: wp.float32
    stiffness: wp.float32
    damping: wp.float32

    # Conjugate of the relative orientation at initialisation time
    # (PhoenX ``mInvInitialOrientation``). Defined as
    # ``inv_initial_orientation = q2_init^* * q1_init`` so the
    # prepare-time twist is
    # ``diff = q2 * inv_initial_orientation * q1^*`` which evaluates to
    # identity at ``t=0``. Extracting ``diff``'s angle around the
    # (world) hinge axis gives the in-branch relative twist that the
    # revolution tracker unwraps. A quaternion occupies 4 dwords.
    inv_initial_orientation: wp.quatf

    # Unbounded-angle tracker state (PhoenX ``FullRevolutionTracker``).
    # Seeded to zero at :func:`angular_motor_initialize_kernel` time so
    # steady-state ``target_angle = 0`` means "hold the initial
    # relative pose". Updated every substep in
    # :func:`angular_motor_prepare_for_iteration_at`.
    revolution_counter: wp.int32
    previous_quaternion_angle: wp.float32

    # Cached per-substep PD coefficients (Jitter2 ``SpringConstraint``
    # ``gamma`` / ``beta`` / ``massCoeff``). Populated in Prepare, read
    # in Iterate. Unused (zero) on the velocity path. See
    # :func:`constraint_container.pd_coefficients`.
    pd_gamma: wp.float32
    pd_beta: wp.float32
    pd_mass_coeff: wp.float32

    # Cached positional error ``theta - target_angle`` at Prepare time.
    # On the velocity path this stays zero; on the PD path the Iterate
    # step consumes it as the positional bias term.
    position_error: wp.float32

    # Cached per-substep value of ``max_force / idt`` so Iterate doesn't
    # need to redo the divide every PGS pass.
    max_lambda: wp.float32

    # Cached per-substep soft-constraint coefficients. ``mass_coeff``
    # scales the velocity-error impulse; ``impulse_coeff`` damps the
    # accumulated impulse. ``bias_rate`` is unused by this constraint
    # (no positional separation) but kept for layout symmetry / debugging.
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    # Scalar effective mass (1-DoF constraint).
    effective_mass: wp.float32

    accumulated_impulse: wp.float32


# Enforce the global constraint header contract (constraint_type / body1
# / body2 at dwords 0 / 1 / 2) at import time so a future field reorder
# fails loudly here instead of silently mis-tagging columns or
# scrambling body indices at runtime.
assert_constraint_header(AngularMotorData)

# Dword offsets derived once from the schema. Each is a Python int;
# wrapped in wp.constant so kernels can use them as compile-time literals.
_OFF_BODY1 = wp.constant(dword_offset_of(AngularMotorData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(AngularMotorData, "body2"))
_OFF_LOCAL_AXIS1 = wp.constant(dword_offset_of(AngularMotorData, "local_axis1"))
_OFF_LOCAL_AXIS2 = wp.constant(dword_offset_of(AngularMotorData, "local_axis2"))
_OFF_VELOCITY = wp.constant(dword_offset_of(AngularMotorData, "velocity"))
_OFF_MAX_FORCE = wp.constant(dword_offset_of(AngularMotorData, "max_force"))
_OFF_HERTZ = wp.constant(dword_offset_of(AngularMotorData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(AngularMotorData, "damping_ratio"))
_OFF_TARGET_ANGLE = wp.constant(dword_offset_of(AngularMotorData, "target_angle"))
_OFF_STIFFNESS = wp.constant(dword_offset_of(AngularMotorData, "stiffness"))
_OFF_DAMPING = wp.constant(dword_offset_of(AngularMotorData, "damping"))
_OFF_INV_INITIAL_ORIENTATION = wp.constant(dword_offset_of(AngularMotorData, "inv_initial_orientation"))
_OFF_REVOLUTION_COUNTER = wp.constant(dword_offset_of(AngularMotorData, "revolution_counter"))
_OFF_PREVIOUS_QUATERNION_ANGLE = wp.constant(dword_offset_of(AngularMotorData, "previous_quaternion_angle"))
_OFF_PD_GAMMA = wp.constant(dword_offset_of(AngularMotorData, "pd_gamma"))
_OFF_PD_BETA = wp.constant(dword_offset_of(AngularMotorData, "pd_beta"))
_OFF_PD_MASS_COEFF = wp.constant(dword_offset_of(AngularMotorData, "pd_mass_coeff"))
_OFF_POSITION_ERROR = wp.constant(dword_offset_of(AngularMotorData, "position_error"))
_OFF_MAX_LAMBDA = wp.constant(dword_offset_of(AngularMotorData, "max_lambda"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(AngularMotorData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(AngularMotorData, "impulse_coeff"))
_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(AngularMotorData, "effective_mass"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(AngularMotorData, "accumulated_impulse"))

#: Total dword count of one angular-motor constraint. Used by the host-side
#: container allocator to size ``ConstraintContainer.data``'s row count.
AM_DWORDS: int = num_dwords(AngularMotorData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def angular_motor_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def angular_motor_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def angular_motor_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def angular_motor_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def angular_motor_get_local_axis1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_AXIS1, cid)


@wp.func
def angular_motor_set_local_axis1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_AXIS1, cid, v)


@wp.func
def angular_motor_get_local_axis2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_AXIS2, cid)


@wp.func
def angular_motor_set_local_axis2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_AXIS2, cid, v)


@wp.func
def angular_motor_get_velocity(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_VELOCITY, cid)


@wp.func
def angular_motor_set_velocity(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_VELOCITY, cid, v)


@wp.func
def angular_motor_get_max_force(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_FORCE, cid)


@wp.func
def angular_motor_set_max_force(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_FORCE, cid, v)


@wp.func
def angular_motor_get_hertz(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ, cid)


@wp.func
def angular_motor_set_hertz(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ, cid, v)


@wp.func
def angular_motor_get_damping_ratio(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_RATIO, cid)


@wp.func
def angular_motor_set_damping_ratio(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO, cid, v)


@wp.func
def angular_motor_get_max_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_LAMBDA, cid)


@wp.func
def angular_motor_set_max_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_LAMBDA, cid, v)


@wp.func
def angular_motor_get_mass_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF, cid)


@wp.func
def angular_motor_set_mass_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF, cid, v)


@wp.func
def angular_motor_get_impulse_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF, cid)


@wp.func
def angular_motor_set_impulse_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF, cid, v)


@wp.func
def angular_motor_get_effective_mass(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_EFFECTIVE_MASS, cid)


@wp.func
def angular_motor_set_effective_mass(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_EFFECTIVE_MASS, cid, v)


@wp.func
def angular_motor_get_accumulated_impulse(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_ACCUMULATED_IMPULSE, cid)


@wp.func
def angular_motor_set_accumulated_impulse(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ACCUMULATED_IMPULSE, cid, v)


@wp.func
def angular_motor_get_target_angle(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_TARGET_ANGLE, cid)


@wp.func
def angular_motor_set_target_angle(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_TARGET_ANGLE, cid, v)


@wp.func
def angular_motor_get_stiffness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_STIFFNESS, cid)


@wp.func
def angular_motor_set_stiffness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_STIFFNESS, cid, v)


@wp.func
def angular_motor_get_damping(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING, cid)


@wp.func
def angular_motor_set_damping(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING, cid, v)


@wp.func
def angular_motor_get_inv_initial_orientation(c: ConstraintContainer, cid: wp.int32) -> wp.quatf:
    return read_quat(c, _OFF_INV_INITIAL_ORIENTATION, cid)


@wp.func
def angular_motor_set_inv_initial_orientation(c: ConstraintContainer, cid: wp.int32, v: wp.quatf):
    write_quat(c, _OFF_INV_INITIAL_ORIENTATION, cid, v)


@wp.func
def angular_motor_get_revolution_counter(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_REVOLUTION_COUNTER, cid)


@wp.func
def angular_motor_set_revolution_counter(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_REVOLUTION_COUNTER, cid, v)


@wp.func
def angular_motor_get_previous_quaternion_angle(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PREVIOUS_QUATERNION_ANGLE, cid)


@wp.func
def angular_motor_set_previous_quaternion_angle(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, v)


@wp.func
def angular_motor_get_pd_gamma(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_GAMMA, cid)


@wp.func
def angular_motor_set_pd_gamma(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_GAMMA, cid, v)


@wp.func
def angular_motor_get_pd_beta(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_BETA, cid)


@wp.func
def angular_motor_set_pd_beta(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_BETA, cid, v)


@wp.func
def angular_motor_get_pd_mass_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_MASS_COEFF, cid)


@wp.func
def angular_motor_set_pd_mass_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_MASS_COEFF, cid, v)


@wp.func
def angular_motor_get_position_error(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_POSITION_ERROR, cid)


@wp.func
def angular_motor_set_position_error(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_POSITION_ERROR, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel; mirrors host-side BallSocket init pattern)
# ---------------------------------------------------------------------------


@wp.kernel
def angular_motor_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    world_axis1: wp.array[wp.vec3f],
    world_axis2: wp.array[wp.vec3f],
    target_velocity: wp.array[wp.float32],
    max_force: wp.array[wp.float32],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
    target_angle: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
):
    """Pack one batch of angular-motor descriptors into ``constraints``.

    Direct port of ``AngularMotor.Initialize`` (AngularMotor.cs:22-37)
    plus zero-initialisation of the cached per-substep fields.

    Snapshots the current body 1 / body 2 orientations to compute:

      * ``local_axis1 = q1^* * world_axis1``
      * ``local_axis2 = q2^* * world_axis2``

    Both world axes are normalised before being moved into local frame
    (matches Jitter's ``JVector.NormalizeInPlace``), so callers can
    pass un-normalised vectors and still get a unit-length local axis.
    For a typical hinge motor, ``world_axis1 == world_axis2`` -- the
    same world-space direction expressed in each body's local frame.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only orientations are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        world_axis1: Motor axis on body 1 in *world* space [num_in_batch].
        world_axis2: Motor axis on body 2 in *world* space [num_in_batch].
        target_velocity: Target relative angular velocity [num_in_batch] [rad/s].
        max_force: Maximum motor torque [num_in_batch] [N*m]. Pass 0 for a
            disabled motor; a non-zero value enables it.
        hertz: Soft-constraint natural frequency [num_in_batch] [Hz]. Set
            to 0 for a perfectly stiff (rigid) motor; small positive
            values yield a "soft" motor that gives in to large external
            torques. See :func:`soft_constraint_coefficients`.
        damping_ratio: Soft-constraint damping ratio [num_in_batch].
            Typical critical-damping value is 1.0. See
            :func:`soft_constraint_coefficients`.
        target_angle: PD-path target relative twist angle [num_in_batch]
            [rad]. Only consumed when ``stiffness > 0`` or
            ``damping > 0``; ignored on the velocity path. Measured
            relative to the *initial* relative pose so ``0.0`` means
            "hold the current relative orientation of the bodies".
        stiffness: PD-path rotational spring gain [num_in_batch]
            [N*m/rad]. Zero on both ``stiffness`` and ``damping``
            selects the legacy velocity-target path.
        damping: PD-path rotational damping gain [num_in_batch]
            [N*m*s/rad].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a1 = wp.normalize(world_axis1[tid])
    a2 = wp.normalize(world_axis2[tid])

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    local_axis1 = wp.quat_rotate_inv(q1, a1)
    local_axis2 = wp.quat_rotate_inv(q2, a2)

    # PhoenX ``invInitialOrientation`` = (q1^* * q2)^* = q2^* * q1. With
    # this definition, ``diff = q2 * invInit * q1^*`` evaluates to
    # identity at t=0, so the extracted rotation angle (and therefore
    # the revolution tracker) starts cleanly at zero regardless of how
    # the two bodies happen to be oriented in world space.
    inv_init = wp.quat_inverse(wp.quat_inverse(q1) * q2)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_ANGULAR_MOTOR)
    angular_motor_set_body1(constraints, cid, b1)
    angular_motor_set_body2(constraints, cid, b2)
    angular_motor_set_local_axis1(constraints, cid, local_axis1)
    angular_motor_set_local_axis2(constraints, cid, local_axis2)

    angular_motor_set_velocity(constraints, cid, target_velocity[tid])
    angular_motor_set_max_force(constraints, cid, max_force[tid])
    angular_motor_set_hertz(constraints, cid, hertz[tid])
    angular_motor_set_damping_ratio(constraints, cid, damping_ratio[tid])
    angular_motor_set_target_angle(constraints, cid, target_angle[tid])
    angular_motor_set_stiffness(constraints, cid, stiffness[tid])
    angular_motor_set_damping(constraints, cid, damping[tid])
    angular_motor_set_inv_initial_orientation(constraints, cid, inv_init)
    angular_motor_set_revolution_counter(constraints, cid, 0)
    angular_motor_set_previous_quaternion_angle(constraints, cid, 0.0)
    angular_motor_set_pd_gamma(constraints, cid, 0.0)
    angular_motor_set_pd_beta(constraints, cid, 0.0)
    angular_motor_set_pd_mass_coeff(constraints, cid, 0.0)
    angular_motor_set_position_error(constraints, cid, 0.0)
    angular_motor_set_max_lambda(constraints, cid, 0.0)
    angular_motor_set_mass_coeff(constraints, cid, 1.0)
    angular_motor_set_impulse_coeff(constraints, cid, 0.0)
    angular_motor_set_effective_mass(constraints, cid, 0.0)
    angular_motor_set_accumulated_impulse(constraints, cid, 0.0)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Two access levels mirror the BallSocket / HingeAngle pattern:
#
# * ``*_at`` variants take an explicit ``base_offset`` (dword offset of
#   the angular-motor sub-block within its column) and a
#   :class:`ConstraintBodies` carrier with the body indices. Used by
#   the fused :data:`CONSTRAINT_TYPE_HINGE_JOINT` constraint.
# * The plain ``angular_motor_prepare_for_iteration`` /
#   ``angular_motor_iterate`` / ``angular_motor_world_wrench`` are thin
#   wrappers used by the unified dispatcher.


@wp.func
def angular_motor_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable ``PrepareForIterationAngularMotor`` (AngularMotor.cs:39).

    Dual-mode port. Branches on ``(stiffness, damping)`` to select
    either the Jitter2 velocity-target path or the PhoenX-style PD
    position-target path; both share the same warm-start, effective-
    mass cache, and impulse cap so consumers never have to know which
    mode is active.

    See :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    max_force = read_float(constraints, base_offset + _OFF_MAX_FORCE, cid)
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    target_angle = read_float(constraints, base_offset + _OFF_TARGET_ANGLE, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    dt = 1.0 / idt
    pd_mode = stiffness > 0.0 or damping > 0.0

    # -- Velocity path: two-axis Jacobian (Jitter2 legacy) -------------
    # -- PD path:       single-axis (body-1) Jacobian (PhoenX/Jolt) ----
    # Both share the warm-start/impulse-cap bookkeeping below.
    if pd_mode:
        # PhoenX uses ``worldSpaceHingeAxis1`` for the motor row; the
        # companion hinge-rotation constraint keeps axis2 parallel to it
        # anyway, and a single axis avoids the small non-orthogonal
        # projection error while integrating.
        j1 = wp.quat_rotate(q1, local_axis1)
        j2 = j1
    else:
        j1 = wp.quat_rotate(q1, local_axis1)
        j2 = wp.quat_rotate(q2, local_axis2)

    # Scalar effective mass for the 1-DoF constraint:
    #   m^-1 = j1 . (InvI1 * j1) + j2 . (InvI2 * j2)
    # NOTE: kept *unsoftened* here -- softness is applied in iterate via
    # the cached coefficients below (mass_coeff/impulse_coeff on the
    # velocity path, or pd_gamma/pd_beta/pd_mass_coeff on the PD path).
    eff_inv = wp.dot(inv_inertia1 @ j1, j1) + wp.dot(inv_inertia2 @ j2, j2)
    write_float(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid, 1.0 / eff_inv)

    # Per-step impulse cap. C# writes ``1.0 / idt * MaxForce`` which is
    # just ``MaxForce * dt``; we use the same form for byte-for-byte
    # parity.
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA, cid, max_force * dt)

    if pd_mode:
        # Update the revolution tracker from the current relative pose.
        # ``diff = q2 * invInit * q1^*`` is identity at t=0 (see
        # initialize_kernel), so the extracted angle stays in-branch
        # until the joint spins close to +/- pi; the tracker then wraps
        # it into the unbounded ``cumulative_angle`` below.
        inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
        diff = q2 * inv_init * wp.quat_inverse(q1)
        new_q_angle = extract_rotation_angle(diff, j1)
        old_counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
        old_prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
        new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
        write_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid, new_counter)
        write_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
        cumulative_angle = revolution_tracker_angle(new_counter, new_prev)

        # Position error in the Jitter2 spring-constraint sign
        # convention: ``C = actual - target``. The iterate step applies
        # ``-M_eff_soft * (jv - bias + gamma * acc)`` (see
        # :func:`pd_coefficients`).
        C = cumulative_angle - target_angle
        gamma, beta, m_soft = pd_coefficients(stiffness, damping, C, eff_inv, dt)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, gamma)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, beta)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, m_soft)
        write_float(constraints, base_offset + _OFF_POSITION_ERROR, cid, C)
        # Zero out the Box2D-soft scalars so Iterate's velocity path
        # branch is inert; keeps the storage in a well-defined state
        # for diagnostics.
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, 0.0)
    else:
        # Time-step-independent soft-constraint coefficients. The motor
        # has no positional separation on this path so ``bias_rate``
        # is unused; we only cache ``mass_coeff`` / ``impulse_coeff``.
        _, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_POSITION_ERROR, cid, 0.0)

    # Warm start: re-apply the previous solve's accumulated impulse.
    # On the PD path ``j1 == j2``, so the two writes still produce an
    # equal-and-opposite angular impulse pair that conserves momentum.
    bodies.angular_velocity[b1] = bodies.angular_velocity[b1] - inv_inertia1 @ (j1 * acc)
    bodies.angular_velocity[b2] = bodies.angular_velocity[b2] + inv_inertia2 @ (j2 * acc)


@wp.func
def angular_motor_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Composable ``IterateAngularMotor`` (AngularMotor.cs:59).

    ``use_bias`` is the Box2D v3 TGS-soft ``useBias`` flag. Pure
    velocity / PD motors have no *positional* drift bias to gate (the
    "bias" in a motor is a velocity target, not a drift correction),
    so this flag is accepted for dispatcher-signature uniformity but
    ignored here.

    Dual-mode PGS update. The ``(stiffness, damping)`` check picks the
    Jitter2 ``(mass_coeff, impulse_coeff)`` velocity path or the
    PhoenX/Jitter2-``SpringConstraint`` PD path. Both paths emit a
    single scalar impulse ``lam`` that is then applied to ``acc``
    (clamped to ``max_lambda``) and pushed into both bodies'
    angular velocities; the only per-mode difference is how ``lam`` is
    computed from the prepared coefficients.

    See :func:`ball_socket_iterate_at` for the ``base_offset`` /
    ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    velocity = read_float(constraints, base_offset + _OFF_VELOCITY, cid)
    eff = read_float(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid)
    max_lambda = read_float(constraints, base_offset + _OFF_MAX_LAMBDA, cid)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    pd_mass_coeff = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid)
    pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA, cid)
    pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    pd_mode = stiffness > 0.0 or damping > 0.0

    if pd_mode:
        # PhoenX/Jitter2 PD path: single-axis Jacobian, matches
        # ``motorConstraintPart.CalculateConstraintProperties`` + apply.
        j1 = wp.quat_rotate(q1, local_axis1)
        j2 = j1
    else:
        j1 = wp.quat_rotate(q1, local_axis1)
        j2 = wp.quat_rotate(q2, local_axis2)

    # jv = -j1 . w1 + j2 . w2  (with j1 == j2 on the PD path)
    jv = -wp.dot(j1, angular_velocity1) + wp.dot(j2, angular_velocity2)

    if pd_mode:
        # Jitter2 SpringConstraint iterate, reduced to one row
        # (see :func:`pd_coefficients`):
        #   lam = -M_eff_soft * (jv - v_ff + bias + gamma * acc)
        # where ``v_ff`` is the optional feed-forward angular velocity
        # (usually zero for a position-target motor; kept on board so
        # users can still stack a velocity bias on top of the spring).
        lam = -pd_mass_coeff * (jv - velocity + pd_beta + pd_gamma * acc)
    else:
        # Box2D/Bepu soft PGS step (no positional bias for a velocity
        # motor):
        #   lam = -mass_coeff * eff * (jv - velocity) - impulse_coeff * acc
        lam = -mass_coeff * eff * (jv - velocity) - impulse_coeff * acc

    old_acc = acc
    acc = wp.clamp(acc + lam, -max_lambda, max_lambda)
    lam = acc - old_acc

    write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    bodies.angular_velocity[b1] = angular_velocity1 - inv_inertia1 @ (j1 * lam)
    bodies.angular_velocity[b2] = angular_velocity2 + inv_inertia2 @ (j2 * lam)


@wp.func
def angular_motor_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable angular-motor wrench on body2; see
    :func:`angular_motor_world_wrench` for semantics."""
    b2 = body_pair.b2
    q2 = bodies.orientation[b2]
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    j2 = wp.quat_rotate(q2, local_axis2)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    torque = j2 * (acc * idt)
    return wp.vec3f(0.0, 0.0, 0.0), torque


@wp.func
def angular_motor_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationAngularMotor`` (AngularMotor.cs:39).

    Thin wrapper: see :func:`angular_motor_prepare_for_iteration_at`.
    """
    b1 = angular_motor_get_body1(constraints, cid)
    b2 = angular_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    angular_motor_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def angular_motor_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Direct port of ``IterateAngularMotor`` (AngularMotor.cs:59).

    Thin wrapper: see :func:`angular_motor_iterate_at`.
    """
    b1 = angular_motor_get_body1(constraints, cid)
    b2 = angular_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    angular_motor_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def angular_motor_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this motor exerts on body2.

    Pure-angular constraint, so ``force`` is zero. Per ``iterate``,
    body2 receives ``+j2 * acc`` of angular impulse, where ``j2`` is the
    body2 motor axis transformed into world frame. Dividing by
    ``substep_dt`` yields the torque applied during the most recent
    substep.
    """
    b1 = angular_motor_get_body1(constraints, cid)
    b2 = angular_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return angular_motor_world_wrench_at(constraints, cid, 0, bodies, body_pair, idt)
