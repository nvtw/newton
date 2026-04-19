# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp port of Jitter2's AngularMotor constraint.

Direct translation of ``Jitter2.Dynamics.Constraints.AngularMotor``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/AngularMotor.cs``).

AngularMotor is a 1-DoF *velocity* constraint that drives the relative
angular velocity of body 1 and body 2 along a shared axis to a target
``velocity``, capped by a per-step impulse derived from ``max_force``.
It does not lock the axis (that's the job of HingeAngle); it merely
applies torque toward the velocity setpoint as long as the resulting
accumulated impulse stays within ``[-max_force * dt, +max_force * dt]``.

Composition: a motorised revolute joint in Jitter is the triple
``BallSocket + HingeAngle + AngularMotor`` -- the ball-socket locks
the linear DoFs, the hinge locks the two angular DoFs perpendicular to
the axis, and the motor drives the remaining axial DoF. Each piece is
its own constraint in our solver too.

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
    read_float,
    read_int,
    read_vec3,
    write_float,
    write_int,
    write_vec3,
)
from newton._src.solvers.jitter.data_packing import dword_offset_of, num_dwords

__all__ = [
    "AM_DWORDS",
    "AngularMotorData",
    "angular_motor_get_accumulated_impulse",
    "angular_motor_get_body1",
    "angular_motor_get_body2",
    "angular_motor_get_effective_mass",
    "angular_motor_get_local_axis1",
    "angular_motor_get_local_axis2",
    "angular_motor_get_max_force",
    "angular_motor_get_max_lambda",
    "angular_motor_get_velocity",
    "angular_motor_initialize_kernel",
    "angular_motor_iterate",
    "angular_motor_iterate_at",
    "angular_motor_prepare_for_iteration",
    "angular_motor_prepare_for_iteration_at",
    "angular_motor_set_accumulated_impulse",
    "angular_motor_set_body1",
    "angular_motor_set_body2",
    "angular_motor_set_effective_mass",
    "angular_motor_set_local_axis1",
    "angular_motor_set_local_axis2",
    "angular_motor_set_max_force",
    "angular_motor_set_max_lambda",
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
    # Set to 0 by Initialize -- callers must override post-init for the
    # motor to actually apply torque.
    max_force: wp.float32

    # Cached per-substep value of ``max_force / idt`` so Iterate doesn't
    # need to redo the divide every PGS pass.
    max_lambda: wp.float32

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
_OFF_MAX_LAMBDA = wp.constant(dword_offset_of(AngularMotorData, "max_lambda"))
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
def angular_motor_get_max_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_LAMBDA, cid)


@wp.func
def angular_motor_set_max_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_LAMBDA, cid, v)


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

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_ANGULAR_MOTOR)
    angular_motor_set_body1(constraints, cid, b1)
    angular_motor_set_body2(constraints, cid, b2)
    angular_motor_set_local_axis1(constraints, cid, local_axis1)
    angular_motor_set_local_axis2(constraints, cid, local_axis2)

    angular_motor_set_velocity(constraints, cid, target_velocity[tid])
    angular_motor_set_max_force(constraints, cid, max_force[tid])
    angular_motor_set_max_lambda(constraints, cid, 0.0)
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
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    # World-space Jacobian rows (the per-DoF axes).
    j1 = wp.quat_rotate(q1, local_axis1)
    j2 = wp.quat_rotate(q2, local_axis2)

    # Scalar effective mass for the 1-DoF constraint:
    #   m^-1 = j1 . (InvI1 * j1) + j2 . (InvI2 * j2)
    eff_inv = wp.dot(inv_inertia1 @ j1, j1) + wp.dot(inv_inertia2 @ j2, j2)
    write_float(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid, 1.0 / eff_inv)

    # Per-step impulse cap. C# writes ``1.0 / idt * MaxForce`` which is
    # just ``MaxForce * dt``; we use the same form for byte-for-byte
    # parity.
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA, cid, max_force / idt)

    # Warm start: re-apply the previous solve's accumulated impulse.
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
):
    """Composable ``IterateAngularMotor`` (AngularMotor.cs:59).

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
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    j1 = wp.quat_rotate(q1, local_axis1)
    j2 = wp.quat_rotate(q2, local_axis2)

    # jv = -j1 . w1 + j2 . w2
    jv = -wp.dot(j1, angular_velocity1) + wp.dot(j2, angular_velocity2)

    lam = -(jv - velocity) * eff

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
):
    """Direct port of ``IterateAngularMotor`` (AngularMotor.cs:59).

    Thin wrapper: see :func:`angular_motor_iterate_at`.
    """
    b1 = angular_motor_get_body1(constraints, cid)
    b2 = angular_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    angular_motor_iterate_at(constraints, cid, 0, bodies, body_pair, idt)


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
