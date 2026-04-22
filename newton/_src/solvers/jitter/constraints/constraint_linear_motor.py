#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""1-DoF linear motor: velocity-target and PD position-target modes.

Translational twin of :mod:`constraint_angular_motor`. Drives the
relative linear velocity (or linear position) between two bodies along
a world axis that is fixed in *each* body's local frame at
initialisation time. The free DoF is a single linear axis; the two
locked perpendicular DoFs are assumed to be handled by a companion
rotation-locking / slide-locking constraint (e.g.
:mod:`constraint_double_ball_socket_prismatic`), just as the companion
relationship between :mod:`constraint_angular_motor` and
:mod:`constraint_double_ball_socket` makes up a motorised revolute
joint.

Reference
---------
Base translation of ``Jitter2.Dynamics.Constraints.LinearMotor``
(``C:/git3/jitterphysics2/src/Jitter2/Dynamics/Constraints/LinearMotor.cs``),
extended with a PD position-target branch in the same spirit as the
PD extension on :mod:`constraint_angular_motor`.

Jitter2's original ``LinearMotor.PrepareForIteration`` uses
**body-COM velocities only** (``jv = -j1 . v1 + j2 . v2``) and a pure
mass-based effective scalar (``1 / (1/m1 + 1/m2)``) -- no lever arms,
no anchor offsets. This is mathematically correct when a companion
constraint already locks the two perpendicular translational DoFs
(prismatic slide lock + rotation lock), because then the relative
velocity at every point of the joint equals the COM relative velocity
projected onto the slide axis. We keep that simpler Jitter2
formulation on the **velocity path** so the port stays byte-for-byte
compatible with the upstream reference (and with the
:mod:`constraint_prismatic` legacy module that also uses the
COM-only Jacobian).

The **PD path** additionally needs an unbounded relative slide
``s(t)`` so the spring term has a position error to act on. We
reconstruct it from anchor points: the user supplies one anchor per
body (``world_anchor1``, ``world_anchor2``), which is snapshotted at
initialise time in each body's local frame. Each substep's prepare
recomputes the world-frame anchors and reads

.. math::
    s = \\hat{n} \\cdot (p_{a2} - p_{a1})

where ``hat_n = j1 = q1 * local_axis1`` is the body-1 world axis. We
track ``s`` directly -- no 2*pi wrap to worry about, so no revolution
tracker -- and the PD ``position_error`` fed into
:func:`constraint_container.pd_coefficients` is ``s - target_position``.
The Jacobian used in the iterate stays the COM-based
``-j1 . v1 + j2 . v2`` so both paths share the same row shape and
effective-mass scalar; only the bias term differs.

Axis convention
---------------
Both paths use ``j1 = q1 * local_axis1`` for body 1 and
``j2 = q2 * local_axis2`` for body 2. For a typical prismatic joint
the two world axes are identical at finalize time, and the companion
prismatic-lock constraint keeps them parallel through the simulation;
the motor doesn't enforce the alignment on its own.

The scalar impulse ``lambda`` is applied as
``-j1 * lambda * inv_m1`` to body 1 and ``+j2 * lambda * inv_m2`` to
body 2, again matching Jitter2's ``LinearMotor.Iterate`` conventions.

Mapping summary
---------------
* ``JVector``                            -> ``wp.vec3f``
* ``JVector.Transform(v, q)``            -> ``wp.quat_rotate(q, v)``
* ``JVector.ConjugatedTransform(v, q)``  -> ``wp.quat_rotate_inv(q, v)``
* ``JVector.operator *(a, b)``           -> ``wp.dot(a, b)`` (dot as ``*``)
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_LINEAR_MOTOR,
    ConstraintBodies,
    ConstraintContainer,
    assert_constraint_header,
    constraint_bodies_make,
    constraint_set_type,
    pd_coefficients,
    read_float,
    read_int,
    read_vec3,
    soft_constraint_coefficients,
    write_float,
    write_int,
    write_vec3,
)
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords

__all__ = [
    "LM_DWORDS",
    "LinearMotorData",
    "linear_motor_get_accumulated_impulse",
    "linear_motor_get_body1",
    "linear_motor_get_body2",
    "linear_motor_get_damping",
    "linear_motor_get_damping_ratio",
    "linear_motor_get_effective_mass",
    "linear_motor_get_hertz",
    "linear_motor_get_impulse_coeff",
    "linear_motor_get_local_anchor1",
    "linear_motor_get_local_anchor2",
    "linear_motor_get_local_axis1",
    "linear_motor_get_local_axis2",
    "linear_motor_get_mass_coeff",
    "linear_motor_get_max_force",
    "linear_motor_get_max_lambda",
    "linear_motor_get_pd_beta",
    "linear_motor_get_pd_gamma",
    "linear_motor_get_pd_mass_coeff",
    "linear_motor_get_position_error",
    "linear_motor_get_rest_offset",
    "linear_motor_get_stiffness",
    "linear_motor_get_target_position",
    "linear_motor_get_velocity",
    "linear_motor_initialize_kernel",
    "linear_motor_iterate",
    "linear_motor_iterate_at",
    "linear_motor_prepare_for_iteration",
    "linear_motor_prepare_for_iteration_at",
    "linear_motor_set_accumulated_impulse",
    "linear_motor_set_body1",
    "linear_motor_set_body2",
    "linear_motor_set_damping",
    "linear_motor_set_damping_ratio",
    "linear_motor_set_effective_mass",
    "linear_motor_set_hertz",
    "linear_motor_set_impulse_coeff",
    "linear_motor_set_local_anchor1",
    "linear_motor_set_local_anchor2",
    "linear_motor_set_local_axis1",
    "linear_motor_set_local_axis2",
    "linear_motor_set_mass_coeff",
    "linear_motor_set_max_force",
    "linear_motor_set_max_lambda",
    "linear_motor_set_pd_beta",
    "linear_motor_set_pd_gamma",
    "linear_motor_set_pd_mass_coeff",
    "linear_motor_set_position_error",
    "linear_motor_set_rest_offset",
    "linear_motor_set_stiffness",
    "linear_motor_set_target_position",
    "linear_motor_set_velocity",
    "linear_motor_world_error",
    "linear_motor_world_error_at",
    "linear_motor_world_wrench",
    "linear_motor_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class LinearMotorData:
    """Per-constraint dword-layout schema for a linear motor.

    *Schema only.* Field order fixes dword offsets; runtime kernels
    operate on the shared :class:`ConstraintContainer` via the typed
    accessors below. Layout mirrors C#'s ``LinearMotorData`` plus the
    PD-extension fields (same naming / roles as
    :class:`AngularMotorData`).

    Fields are arranged in struct-natural order with no manual padding;
    everything is 32-bit so the struct is dword-aligned end-to-end.

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Motor direction on each body in the body's local frame (captured
    # from ``world_axis1/2`` at initialise time, after normalisation).
    # The world-frame Jacobian rows are recomputed every substep as
    # ``j1 = q1 * local_axis1`` and ``j2 = q2 * local_axis2``.
    local_axis1: wp.vec3f
    local_axis2: wp.vec3f

    # Anchor offsets from each body's centre of mass in the body's
    # local frame. Used only by the **PD path** to reconstruct the
    # relative slide ``s = hat_n . (p_a2 - p_a1)`` each substep; the
    # velocity path leaves them unread. Captured from the user's
    # world-space anchors at initialise time (``local_anchor{k} =
    # q{k}^* * (world_anchor{k} - p{k})``).
    local_anchor1: wp.vec3f
    local_anchor2: wp.vec3f

    # Target relative velocity along the (current world) motor axis
    # [m/s]. The motor drives ``-v1 . j1 + v2 . j2`` toward this value
    # subject to the per-substep impulse cap. On the PD path this
    # doubles as a feed-forward velocity target (usually zero).
    velocity: wp.float32

    # Maximum motor force [N]. The iterate kernel converts this to a
    # per-substep impulse cap ``max_lambda = max_force / idt`` and
    # clamps the accumulated impulse to ``[-max_lambda, max_lambda]``.
    # ``max_force = 0`` disables the velocity path entirely (zero cap);
    # the PD path guards on ``(stiffness, damping)`` instead, so a PD
    # drive with ``max_force = 0`` remains active and is only bounded
    # by its softness.
    max_force: wp.float32

    # User-facing soft-constraint knobs (Box2D v3 / Bepu / Nordby
    # formulation; see :func:`soft_constraint_coefficients`). Active
    # on the **velocity path** (``stiffness == 0 and damping == 0``).
    # The motor carries no positional bias on that path, so ``hertz``
    # only governs how fast the accumulated impulse decays toward the
    # velocity setpoint.
    hertz: wp.float32
    damping_ratio: wp.float32

    # PD position-target mode (Jitter2 ``SpringConstraint`` convention
    # rescaled into absolute SI units). Active whenever
    # ``stiffness > 0`` or ``damping > 0``. When active, the motor
    # holds the relative slide along the axis at ``target_position``
    # (m, measured relative to the rest slide computed at initialise)
    # with spring gain ``stiffness`` [N/m] and damping gain ``damping``
    # [N*s/m]. ``max_force`` still caps the per-substep impulse in
    # both modes.
    target_position: wp.float32
    stiffness: wp.float32
    damping: wp.float32

    # Rest slide at initialise time (``hat_n . (p_a2 - p_a1)`` evaluated
    # at t=0). Stored so the PD ``position_error = (s_now - rest) -
    # target_position`` evaluates to zero when the user asks for
    # "hold the initial pose" (``target_position = 0``), regardless of
    # where the bodies happen to be in world space.
    rest_offset: wp.float32

    # Cached per-substep PD coefficients (Jitter2 ``SpringConstraint``
    # ``gamma`` / ``beta`` / ``massCoeff``). Populated in prepare,
    # read in iterate. Unused (zero) on the velocity path. See
    # :func:`constraint_container.pd_coefficients`.
    pd_gamma: wp.float32
    pd_beta: wp.float32
    pd_mass_coeff: wp.float32

    # Cached positional error ``s - rest - target_position`` at the
    # most recent prepare. Zero on the velocity path.
    position_error: wp.float32

    # Cached per-substep value of ``max_force / idt`` so the iterate
    # doesn't redo the divide every PGS pass.
    max_lambda: wp.float32

    # Cached per-substep soft-constraint coefficients
    # (``bias_rate`` left unused -- linear motor has no positional
    # separation on the velocity path). ``mass_coeff`` scales the
    # velocity-error impulse; ``impulse_coeff`` damps the accumulated
    # impulse.
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    # Scalar effective mass (1-DoF row) = ``1 / (1/m1 + 1/m2)``.
    # Constant across substeps (inverse masses don't move), but cached
    # here anyway so iterate loads it locally.
    effective_mass: wp.float32

    accumulated_impulse: wp.float32


assert_constraint_header(LinearMotorData)

_OFF_BODY1 = wp.constant(dword_offset_of(LinearMotorData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(LinearMotorData, "body2"))
_OFF_LOCAL_AXIS1 = wp.constant(dword_offset_of(LinearMotorData, "local_axis1"))
_OFF_LOCAL_AXIS2 = wp.constant(dword_offset_of(LinearMotorData, "local_axis2"))
_OFF_LOCAL_ANCHOR1 = wp.constant(dword_offset_of(LinearMotorData, "local_anchor1"))
_OFF_LOCAL_ANCHOR2 = wp.constant(dword_offset_of(LinearMotorData, "local_anchor2"))
_OFF_VELOCITY = wp.constant(dword_offset_of(LinearMotorData, "velocity"))
_OFF_MAX_FORCE = wp.constant(dword_offset_of(LinearMotorData, "max_force"))
_OFF_HERTZ = wp.constant(dword_offset_of(LinearMotorData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(LinearMotorData, "damping_ratio"))
_OFF_TARGET_POSITION = wp.constant(dword_offset_of(LinearMotorData, "target_position"))
_OFF_STIFFNESS = wp.constant(dword_offset_of(LinearMotorData, "stiffness"))
_OFF_DAMPING = wp.constant(dword_offset_of(LinearMotorData, "damping"))
_OFF_REST_OFFSET = wp.constant(dword_offset_of(LinearMotorData, "rest_offset"))
_OFF_PD_GAMMA = wp.constant(dword_offset_of(LinearMotorData, "pd_gamma"))
_OFF_PD_BETA = wp.constant(dword_offset_of(LinearMotorData, "pd_beta"))
_OFF_PD_MASS_COEFF = wp.constant(dword_offset_of(LinearMotorData, "pd_mass_coeff"))
_OFF_POSITION_ERROR = wp.constant(dword_offset_of(LinearMotorData, "position_error"))
_OFF_MAX_LAMBDA = wp.constant(dword_offset_of(LinearMotorData, "max_lambda"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(LinearMotorData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(LinearMotorData, "impulse_coeff"))
_OFF_EFFECTIVE_MASS = wp.constant(dword_offset_of(LinearMotorData, "effective_mass"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(dword_offset_of(LinearMotorData, "accumulated_impulse"))

#: Total dword count of one linear-motor constraint. Used by the
#: host-side container allocator to size
#: :attr:`ConstraintContainer.data`'s row count.
LM_DWORDS: int = num_dwords(LinearMotorData)


# ---------------------------------------------------------------------------
# Typed accessors -- thin wrappers over column-major dword get/set
# ---------------------------------------------------------------------------


@wp.func
def linear_motor_get_body1(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY1, cid)


@wp.func
def linear_motor_set_body1(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY1, cid, v)


@wp.func
def linear_motor_get_body2(c: ConstraintContainer, cid: wp.int32) -> wp.int32:
    return read_int(c, _OFF_BODY2, cid)


@wp.func
def linear_motor_set_body2(c: ConstraintContainer, cid: wp.int32, v: wp.int32):
    write_int(c, _OFF_BODY2, cid, v)


@wp.func
def linear_motor_get_local_axis1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_AXIS1, cid)


@wp.func
def linear_motor_set_local_axis1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_AXIS1, cid, v)


@wp.func
def linear_motor_get_local_axis2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_AXIS2, cid)


@wp.func
def linear_motor_set_local_axis2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_AXIS2, cid, v)


@wp.func
def linear_motor_get_local_anchor1(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_ANCHOR1, cid)


@wp.func
def linear_motor_set_local_anchor1(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_ANCHOR1, cid, v)


@wp.func
def linear_motor_get_local_anchor2(c: ConstraintContainer, cid: wp.int32) -> wp.vec3f:
    return read_vec3(c, _OFF_LOCAL_ANCHOR2, cid)


@wp.func
def linear_motor_set_local_anchor2(c: ConstraintContainer, cid: wp.int32, v: wp.vec3f):
    write_vec3(c, _OFF_LOCAL_ANCHOR2, cid, v)


@wp.func
def linear_motor_get_velocity(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_VELOCITY, cid)


@wp.func
def linear_motor_set_velocity(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_VELOCITY, cid, v)


@wp.func
def linear_motor_get_max_force(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_FORCE, cid)


@wp.func
def linear_motor_set_max_force(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_FORCE, cid, v)


@wp.func
def linear_motor_get_hertz(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_HERTZ, cid)


@wp.func
def linear_motor_set_hertz(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_HERTZ, cid, v)


@wp.func
def linear_motor_get_damping_ratio(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING_RATIO, cid)


@wp.func
def linear_motor_set_damping_ratio(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING_RATIO, cid, v)


@wp.func
def linear_motor_get_target_position(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_TARGET_POSITION, cid)


@wp.func
def linear_motor_set_target_position(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_TARGET_POSITION, cid, v)


@wp.func
def linear_motor_get_stiffness(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_STIFFNESS, cid)


@wp.func
def linear_motor_set_stiffness(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_STIFFNESS, cid, v)


@wp.func
def linear_motor_get_damping(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_DAMPING, cid)


@wp.func
def linear_motor_set_damping(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_DAMPING, cid, v)


@wp.func
def linear_motor_get_rest_offset(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_REST_OFFSET, cid)


@wp.func
def linear_motor_set_rest_offset(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_REST_OFFSET, cid, v)


@wp.func
def linear_motor_get_pd_gamma(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_GAMMA, cid)


@wp.func
def linear_motor_set_pd_gamma(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_GAMMA, cid, v)


@wp.func
def linear_motor_get_pd_beta(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_BETA, cid)


@wp.func
def linear_motor_set_pd_beta(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_BETA, cid, v)


@wp.func
def linear_motor_get_pd_mass_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_PD_MASS_COEFF, cid)


@wp.func
def linear_motor_set_pd_mass_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_PD_MASS_COEFF, cid, v)


@wp.func
def linear_motor_get_position_error(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_POSITION_ERROR, cid)


@wp.func
def linear_motor_set_position_error(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_POSITION_ERROR, cid, v)


@wp.func
def linear_motor_get_max_lambda(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MAX_LAMBDA, cid)


@wp.func
def linear_motor_set_max_lambda(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MAX_LAMBDA, cid, v)


@wp.func
def linear_motor_get_mass_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_MASS_COEFF, cid)


@wp.func
def linear_motor_set_mass_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_MASS_COEFF, cid, v)


@wp.func
def linear_motor_get_impulse_coeff(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_IMPULSE_COEFF, cid)


@wp.func
def linear_motor_set_impulse_coeff(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_IMPULSE_COEFF, cid, v)


@wp.func
def linear_motor_get_effective_mass(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_EFFECTIVE_MASS, cid)


@wp.func
def linear_motor_set_effective_mass(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_EFFECTIVE_MASS, cid, v)


@wp.func
def linear_motor_get_accumulated_impulse(c: ConstraintContainer, cid: wp.int32) -> wp.float32:
    return read_float(c, _OFF_ACCUMULATED_IMPULSE, cid)


@wp.func
def linear_motor_set_accumulated_impulse(c: ConstraintContainer, cid: wp.int32, v: wp.float32):
    write_float(c, _OFF_ACCUMULATED_IMPULSE, cid, v)


# ---------------------------------------------------------------------------
# Initialization (kernel; mirrors angular_motor_initialize_kernel)
# ---------------------------------------------------------------------------


@wp.kernel
def linear_motor_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    world_axis1: wp.array[wp.vec3f],
    world_axis2: wp.array[wp.vec3f],
    world_anchor1: wp.array[wp.vec3f],
    world_anchor2: wp.array[wp.vec3f],
    target_velocity: wp.array[wp.float32],
    max_force: wp.array[wp.float32],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
    target_position: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
):
    """Pack one batch of linear-motor descriptors into ``constraints``.

    Direct port of ``LinearMotor.Initialize`` (LinearMotor.cs:82-97)
    plus zero-initialisation of the PD-path fields.

    Snapshots the current body 1 / body 2 orientations and positions to
    compute:

      * ``local_axis{k} = q{k}^* * world_axis{k}``
      * ``local_anchor{k} = q{k}^* * (world_anchor{k} - p{k})``
      * ``rest_offset = hat_n . (world_anchor2 - world_anchor1)``

    with ``hat_n`` the world-space body-1 axis (both world axes are
    assumed colinear at init; the dot product picks up the signed
    separation along that direction).

    Both world axes are normalised before being moved into local frame
    (matches Jitter's ``JVector.NormalizeInPlace``), so callers can
    pass un-normalised vectors and still get a unit-length local axis.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only positions and orientations
            are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        world_axis1: Motor direction on body 1 in *world* space
            [num_in_batch]. Normalised internally.
        world_axis2: Motor direction on body 2 in *world* space
            [num_in_batch]. Normalised internally. For a prismatic
            joint this is usually equal to ``world_axis1``.
        world_anchor1: Anchor point on body 1 in *world* space
            [num_in_batch] [m]. Only consumed by the PD path; velocity-
            only consumers may pass any point on body 1 (the body's
            current position is a fine default).
        world_anchor2: Anchor point on body 2 in *world* space
            [num_in_batch] [m]. Typically coincident with
            ``world_anchor1`` at finalise time so the initial relative
            slide is zero.
        target_velocity: Target relative linear velocity [num_in_batch]
            [m/s].
        max_force: Maximum motor force [num_in_batch] [N]. Pass 0 to
            disable the velocity path; a non-zero value enables it.
            The PD path ignores this in principle but still uses it as
            an impulse cap when > 0.
        hertz: Soft-constraint natural frequency [num_in_batch] [Hz].
            Set to 0 for a perfectly stiff (rigid) velocity motor.
        damping_ratio: Soft-constraint damping ratio [num_in_batch].
            Typical critical-damping value is 1.0.
        target_position: PD-path target relative slide [num_in_batch]
            [m]. Measured relative to the *initial* relative slide so
            ``0.0`` means "hold the current relative pose".
        stiffness: PD-path linear spring gain [num_in_batch] [N/m].
            Zero on both ``stiffness`` and ``damping`` selects the
            legacy velocity-target path (Jitter2 compatibility).
        damping: PD-path linear damping gain [num_in_batch] [N*s/m].
    """
    tid = wp.tid()
    cid = cid_offset + tid

    b1 = body1[tid]
    b2 = body2[tid]
    a1 = wp.normalize(world_axis1[tid])
    a2 = wp.normalize(world_axis2[tid])
    w_anc1 = world_anchor1[tid]
    w_anc2 = world_anchor2[tid]

    p1 = bodies.position[b1]
    p2 = bodies.position[b2]
    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]

    local_axis1 = wp.quat_rotate_inv(q1, a1)
    local_axis2 = wp.quat_rotate_inv(q2, a2)
    local_anchor1 = wp.quat_rotate_inv(q1, w_anc1 - p1)
    local_anchor2 = wp.quat_rotate_inv(q2, w_anc2 - p2)

    # Rest slide along the body-1 world axis; stored so the PD
    # ``position_error`` expression is
    # ``(current_slide - rest_offset) - target_position``, i.e. the
    # same "target is a *delta* from the initial pose" convention used
    # by :mod:`constraint_angular_motor`.
    rest_offset = wp.dot(a1, w_anc2 - w_anc1)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_LINEAR_MOTOR)
    linear_motor_set_body1(constraints, cid, b1)
    linear_motor_set_body2(constraints, cid, b2)
    linear_motor_set_local_axis1(constraints, cid, local_axis1)
    linear_motor_set_local_axis2(constraints, cid, local_axis2)
    linear_motor_set_local_anchor1(constraints, cid, local_anchor1)
    linear_motor_set_local_anchor2(constraints, cid, local_anchor2)

    linear_motor_set_velocity(constraints, cid, target_velocity[tid])
    linear_motor_set_max_force(constraints, cid, max_force[tid])
    linear_motor_set_hertz(constraints, cid, hertz[tid])
    linear_motor_set_damping_ratio(constraints, cid, damping_ratio[tid])
    linear_motor_set_target_position(constraints, cid, target_position[tid])
    linear_motor_set_stiffness(constraints, cid, stiffness[tid])
    linear_motor_set_damping(constraints, cid, damping[tid])
    linear_motor_set_rest_offset(constraints, cid, rest_offset)

    linear_motor_set_pd_gamma(constraints, cid, 0.0)
    linear_motor_set_pd_beta(constraints, cid, 0.0)
    linear_motor_set_pd_mass_coeff(constraints, cid, 0.0)
    linear_motor_set_position_error(constraints, cid, 0.0)
    linear_motor_set_max_lambda(constraints, cid, 0.0)
    linear_motor_set_mass_coeff(constraints, cid, 1.0)
    linear_motor_set_impulse_coeff(constraints, cid, 0.0)
    linear_motor_set_effective_mass(constraints, cid, 0.0)
    linear_motor_set_accumulated_impulse(constraints, cid, 0.0)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------
#
# Same two access levels as the other per-type modules:
#
# * ``*_at`` variants take an explicit ``base_offset`` + a
#   :class:`ConstraintBodies` carrier. Reserved for a future fused
#   prismatic joint (translational analogue of
#   :data:`CONSTRAINT_TYPE_HINGE_JOINT`).
# * Direct wrappers are the entry points used by the dispatcher.


@wp.func
def linear_motor_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable ``PrepareForIterationLinearMotor`` (LinearMotor.cs:131).

    Dual-mode port. Branches on ``(stiffness, damping)`` to select the
    Jitter2 velocity-target path or the PD position-target path; both
    share the same warm-start, effective-mass cache, and impulse cap so
    consumers don't need to know which mode is active.

    See :func:`ball_socket_prepare_for_iteration_at` for the
    ``base_offset`` / ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    p1 = bodies.position[b1]
    p2 = bodies.position[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]

    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    local_anchor1 = read_vec3(constraints, base_offset + _OFF_LOCAL_ANCHOR1, cid)
    local_anchor2 = read_vec3(constraints, base_offset + _OFF_LOCAL_ANCHOR2, cid)
    max_force = read_float(constraints, base_offset + _OFF_MAX_FORCE, cid)
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    target_position = read_float(constraints, base_offset + _OFF_TARGET_POSITION, cid)
    rest_offset = read_float(constraints, base_offset + _OFF_REST_OFFSET, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    dt = 1.0 / idt
    pd_mode = stiffness > 0.0 or damping > 0.0

    j1 = wp.quat_rotate(q1, local_axis1)
    j2 = wp.quat_rotate(q2, local_axis2)

    # Jitter2 LinearMotor.cs:141-142: 1-DoF linear row, no lever arm
    # contributions. ``eff_inv = 1/m1 + 1/m2`` is mathematically
    # correct when the companion prismatic lock cancels the
    # perpendicular DoFs at the joint anchor; for a free-floating
    # motor it would under-predict the effective mass, but that use
    # case has no physical meaning (bodies would just translate
    # rigidly). Kept as-is for parity with the C# reference.
    eff_inv = inv_mass1 + inv_mass2
    if eff_inv > 1.0e-20:
        eff = 1.0 / eff_inv
    else:
        eff = 0.0
    write_float(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid, eff)

    # Per-step impulse cap (LinearMotor.cs:143).
    write_float(constraints, base_offset + _OFF_MAX_LAMBDA, cid, max_force * dt)

    if pd_mode:
        # Reconstruct the current relative slide directly from the
        # world-frame anchors; no integration / drift-tracking needed
        # because positions are already the ground truth.
        world_anchor1 = p1 + wp.quat_rotate(q1, local_anchor1)
        world_anchor2 = p2 + wp.quat_rotate(q2, local_anchor2)
        slide_now = wp.dot(j1, world_anchor2 - world_anchor1)

        # Jitter2 SpringConstraint sign: ``C = actual - target`` so
        # the iterate below applies ``-M_eff_soft * (jv + C_bias + ...)``.
        # ``slide_now - rest_offset`` is the *signed* change from the
        # rest configuration; subtract ``target_position`` to get the
        # error the spring should eliminate.
        C = (slide_now - rest_offset) - target_position
        gamma, beta, m_soft = pd_coefficients(stiffness, damping, C, eff_inv, dt)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, gamma)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, beta)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, m_soft)
        write_float(constraints, base_offset + _OFF_POSITION_ERROR, cid, C)
        # Zero out the Box2D-soft scalars so iterate's velocity path
        # branch would be inert if entered by mistake.
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, 0.0)
    else:
        # Velocity path: Box2D / Bepu (hertz, damping_ratio). No
        # positional bias -- ``bias_rate`` is discarded, matching
        # :mod:`constraint_angular_motor`'s velocity-only formulation.
        _, mass_coeff, impulse_coeff = soft_constraint_coefficients(hertz, damping_ratio, dt)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_POSITION_ERROR, cid, 0.0)

    # Warm start: re-apply the previous solve's accumulated impulse.
    # Sign convention matches LinearMotor.cs:145-146 (body 1 gets
    # ``-j1 * acc * inv_m1``, body 2 gets ``+j2 * acc * inv_m2``).
    bodies.velocity[b1] = bodies.velocity[b1] - j1 * (acc * inv_mass1)
    bodies.velocity[b2] = bodies.velocity[b2] + j2 * (acc * inv_mass2)


@wp.func
def linear_motor_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Composable ``IterateLinearMotor`` (LinearMotor.cs:163).

    ``use_bias`` is the Box2D v3 TGS-soft ``useBias`` flag; a pure
    velocity / PD linear motor has no positional drift bias to gate,
    so the flag is accepted for dispatcher-signature uniformity but
    ignored here.

    Dual-mode PGS update. The ``(stiffness, damping)`` check picks the
    Jitter2 ``(mass_coeff, impulse_coeff)`` velocity path or the
    PhoenX / Jitter2-``SpringConstraint`` PD path. Both paths emit a
    single scalar impulse ``lam`` that is then accumulated, clamped to
    ``max_lambda`` (when ``max_force > 0``), and pushed into both
    bodies' linear velocities.

    See :func:`ball_socket_iterate_at` for the ``base_offset`` /
    ``body_pair`` contract.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]

    q1 = bodies.orientation[b1]
    q2 = bodies.orientation[b2]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    velocity = read_float(constraints, base_offset + _OFF_VELOCITY, cid)
    eff = read_float(constraints, base_offset + _OFF_EFFECTIVE_MASS, cid)
    max_lambda = read_float(constraints, base_offset + _OFF_MAX_LAMBDA, cid)
    max_force = read_float(constraints, base_offset + _OFF_MAX_FORCE, cid)
    mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
    impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    pd_mass_coeff = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid)
    pd_gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA, cid)
    pd_beta = read_float(constraints, base_offset + _OFF_PD_BETA, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    pd_mode = stiffness > 0.0 or damping > 0.0

    j1 = wp.quat_rotate(q1, local_axis1)
    j2 = wp.quat_rotate(q2, local_axis2)

    # LinearMotor.cs:172 exactly (body-COM velocities only, no lever
    # contribution): ``jv = -j1 . v1 + j2 . v2``.
    jv = -wp.dot(j1, velocity1) + wp.dot(j2, velocity2)

    if pd_mode:
        # See :mod:`constraint_angular_motor` for the derivation of
        # this form from Jitter2's SpringConstraint iterate:
        #   lam = -M_eff_soft * (jv - v_ff + bias + gamma * acc)
        lam = -pd_mass_coeff * (jv - velocity + pd_beta + pd_gamma * acc)
    else:
        # Box2D / Bepu soft PGS step (no positional bias):
        #   lam = -mass_coeff * eff * (jv - velocity) - impulse_coeff * acc
        lam = -mass_coeff * eff * (jv - velocity) - impulse_coeff * acc

    old_acc = acc
    acc = acc + lam
    # ``max_force > 0`` caps the per-substep impulse in both modes.
    # ``max_force == 0`` leaves the PD drive self-bounded by its
    # softness and disables the velocity drive (max_lambda is also 0
    # in that case).
    if max_force > 0.0:
        acc = wp.clamp(acc, -max_lambda, max_lambda)
    elif not pd_mode:
        acc = wp.clamp(acc, -max_lambda, max_lambda)
    lam = acc - old_acc

    write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    bodies.velocity[b1] = velocity1 - j1 * (lam * inv_mass1)
    bodies.velocity[b2] = velocity2 + j2 * (lam * inv_mass2)


@wp.func
def linear_motor_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable linear-motor wrench on body 2.

    Pure-linear constraint -- ``torque`` is zero. Per the iterate,
    body 2 receives ``+j2 * acc`` of linear impulse; dividing by the
    substep ``dt`` yields the force applied during the most recent
    substep.
    """
    b2 = body_pair.b2
    q2 = bodies.orientation[b2]
    local_axis2 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS2, cid)
    j2 = wp.quat_rotate(q2, local_axis2)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    force = j2 * (acc * idt)
    return force, wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def linear_motor_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """Direct port of ``PrepareForIterationLinearMotor`` (LinearMotor.cs:131).

    Thin wrapper: see :func:`linear_motor_prepare_for_iteration_at`.
    """
    b1 = linear_motor_get_body1(constraints, cid)
    b2 = linear_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    linear_motor_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def linear_motor_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """Direct port of ``IterateLinearMotor`` (LinearMotor.cs:163).

    Thin wrapper: see :func:`linear_motor_iterate_at`.
    """
    b1 = linear_motor_get_body1(constraints, cid)
    b2 = linear_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    linear_motor_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def linear_motor_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this motor exerts on body 2.

    Pure-linear constraint, so ``torque`` is zero. Per ``iterate``,
    body 2 receives ``+j2 * acc`` of linear impulse, where ``j2`` is
    the body-2 motor axis in world space. Dividing by ``substep_dt``
    yields the force applied during the most recent substep.
    """
    b1 = linear_motor_get_body1(constraints, cid)
    b2 = linear_motor_get_body2(constraints, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return linear_motor_world_wrench_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def linear_motor_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
) -> wp.spatial_vector:
    """Position-level constraint residual for a linear motor.

    PD mode: returns the cached ``position_error = (slide_now -
    rest_offset) - target_position`` [m] in the x component of the
    linear slot. Velocity mode: zero. See
    :func:`angular_motor_world_error_at` for the rationale.

    Output: :class:`wp.spatial_vector` with ``spatial_top`` =
    ``(position_error_or_0, 0, 0)`` and ``spatial_bottom`` = zero.
    """
    err = read_float(constraints, base_offset + _OFF_POSITION_ERROR, cid)
    return wp.spatial_vector(wp.vec3f(err, 0.0, 0.0), wp.vec3f(0.0, 0.0, 0.0))


@wp.func
def linear_motor_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`linear_motor_world_error_at`."""
    return linear_motor_world_error_at(constraints, cid, 0)
