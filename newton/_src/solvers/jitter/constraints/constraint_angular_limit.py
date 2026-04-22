# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""1-DoF angular limit: one- or two-sided soft rotational stop.

Standalone angular analogue of the ``limit`` portion of the actuated
double-ball-socket, extracted into its own constraint so a passive
revolute joint composes as ``DoubleBallSocket + AngularLimit``
(positional lock + rotation lock + one-sided limit) the same way a
motorised joint composes as
``DoubleBallSocket + AngularMotor + (optional) AngularLimit``.

The constraint clamps the relative twist angle between body 1 and
body 2 along a hinge axis to ``[min_value, max_value]`` (rad).
Exactly one of three states is active per substep:

* **Inactive** (``min_value <= angle <= max_value``): the row is
  non-contributing -- ``lam = 0`` every PGS pass.
* **Max clamp** (``angle > max_value``): only non-negative ``lam`` is
  accumulated. The impulse pushes the joint *back* toward
  ``max_value``.
* **Min clamp** (``angle < min_value``): only non-positive ``lam`` is
  accumulated; impulse pushes the joint back toward ``min_value``.

One-sided limits are expressed by pushing the unused bound well
outside the physical range (e.g. ``min_value = -1e9``).  An explicit
*disabled* row is expressed by ``min_value > max_value`` (``0.0, 0.0``
still reads as "lock at exactly zero" and is a valid two-sided limit).

Softness
--------
Same dual-convention plumbing as :mod:`constraint_angular_motor`:

* **Box2D / Bepu** (``stiffness == 0 and damping == 0``): the spring
  is described by ``hertz`` [Hz] + ``damping_ratio`` [dimensionless],
  which feed through
  :func:`constraint_container.soft_constraint_coefficients`. The
  positional error ``limit_C`` enters as a velocity bias
  ``-limit_C * bias_rate``, and ``mass_coeff`` / ``impulse_coeff``
  attenuate the PGS step. This matches the legacy behaviour inside
  :mod:`constraint_actuated_double_ball_socket`.
* **PD spring-damper** (``stiffness > 0`` or ``damping > 0``): the
  gains are in absolute SI units (``stiffness`` in N*m/rad,
  ``damping`` in N*m*s/rad) and are plugged into
  :func:`constraint_container.pd_coefficients` the exact same way the
  PD path of :mod:`constraint_angular_motor` does. When this path is
  used ``hertz`` / ``damping_ratio`` are ignored.

In both cases the iterate applies a **unilateral** PGS clamp, so the
limit only exerts torque pushing the joint back into range -- never
pulling it onto either stop.

Axis & Jacobian
---------------
Like :mod:`constraint_angular_motor`, the limit reads **body-1's**
world axis for its Jacobian and its angle extraction. The companion
rotation-lock constraint (e.g. the 5-DoF
:mod:`constraint_double_ball_socket`) keeps ``axis1 == axis2`` to
within solver tolerance, so a single-axis row is both simpler and
slightly more accurate than the two-axis Jitter2 formulation.

Unbounded angle tracking uses the same PhoenX
``FullRevolutionTracker`` (see :mod:`math_helpers`) as the angular
motor so ``min_value`` / ``max_value`` can be arbitrarily large
multiples of pi. The relative twist is extracted from
``diff = q2 * inv_initial_orientation * q1^*`` which evaluates to the
identity quaternion at finalize time; ``0.0`` therefore means "hold
the initial relative orientation".

Storage matches the other scalar constraints: a ``@wp.struct``
schema defines dword offsets into the shared
:class:`ConstraintContainer` and typed accessors are used at
runtime.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_ANGULAR_LIMIT,
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
from newton._src.solvers.jitter.helpers.data_packing import dword_offset_of, num_dwords
from newton._src.solvers.jitter.helpers.math_helpers import (
    extract_rotation_angle,
    revolution_tracker_angle,
    revolution_tracker_update,
)

__all__ = [
    "AL_DWORDS",
    "AngularLimitData",
    "angular_limit_initialize_kernel",
    "angular_limit_iterate",
    "angular_limit_iterate_at",
    "angular_limit_prepare_for_iteration",
    "angular_limit_prepare_for_iteration_at",
    "angular_limit_world_error",
    "angular_limit_world_error_at",
    "angular_limit_world_wrench",
    "angular_limit_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Clamp state constants (match constraint_actuated_double_ball_socket)
# ---------------------------------------------------------------------------

_CLAMP_NONE = wp.constant(wp.int32(0))
_CLAMP_MAX = wp.constant(wp.int32(1))
_CLAMP_MIN = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class AngularLimitData:
    """Per-constraint dword-layout schema for an angular limit.

    *Schema only.* Field order fixes dword offsets; runtime kernels
    operate on the shared :class:`ConstraintContainer` via the typed
    accessors below.

    The first field is the global ``constraint_type`` tag (mandatory
    contract for every constraint schema -- see
    :func:`assert_constraint_header`).
    """

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Hinge axis on each body in the body's local frame (captured from
    # ``world_axis1/2`` at initialise time, after normalisation). The
    # world-frame Jacobian is rebuilt every substep as
    # ``j1 = q1 * local_axis1`` and ``j2 = q2 * local_axis2``.
    local_axis1: wp.vec3f
    local_axis2: wp.vec3f

    # Allowed twist-angle range [rad], measured from the initial
    # relative pose. The limit is disabled when ``min_value > max_value``;
    # ``0.0, 0.0`` is a valid (degenerate) two-sided lock at zero.
    min_value: wp.float32
    max_value: wp.float32

    # Box2D / Bepu soft-constraint knobs. Active when ``stiffness == 0``
    # and ``damping == 0``. ``hertz = 0`` is a rigid (un-softened)
    # limit and works fine because ``limit_C`` only feeds in when the
    # joint has already crossed the stop.
    hertz: wp.float32
    damping_ratio: wp.float32

    # PD spring-damper gains. Active when ``stiffness > 0`` or
    # ``damping > 0``; units are N*m/rad and N*m*s/rad respectively.
    # Ignored otherwise.
    stiffness: wp.float32
    damping: wp.float32

    # PhoenX ``invInitialOrientation`` = q2_init^* * q1_init; identity
    # relative rotation at t=0 (see :mod:`constraint_angular_motor`).
    inv_initial_orientation: wp.quatf

    # Unbounded-angle tracker state (see
    # :func:`math_helpers.revolution_tracker_update`).
    revolution_counter: wp.int32
    previous_quaternion_angle: wp.float32

    # Current clamp side (``_CLAMP_NONE`` / ``_CLAMP_MAX`` /
    # ``_CLAMP_MIN``). Rewritten every substep in prepare.
    clamp: wp.int32

    # Cached scalar inverse effective mass, ``eff_inv = j . (InvI1 *
    # j) + j . (InvI2 * j)``. Needed by both paths in iterate; also
    # feeds :func:`pd_coefficients` on the PD path.
    eff_inv: wp.float32

    # Cached per-substep Box2D soft-constraint coefficients.
    # ``bias`` already absorbs the ``-limit_C * bias_rate`` term so
    # iterate drives ``jv`` toward ``-bias``. Unused (zero) on the
    # PD path.
    bias: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    # Cached per-substep PD coefficients (Jitter2 ``SpringConstraint``
    # gamma / beta / massCoeff). Unused (zero) on the Box2D path.
    pd_gamma: wp.float32
    pd_beta: wp.float32
    pd_mass_coeff: wp.float32

    # Accumulated impulse [N*m*s] across the substep's PGS passes.
    # Unilateral clamp is applied in iterate: acc >= 0 when
    # clamp == _CLAMP_MAX, acc <= 0 when clamp == _CLAMP_MIN.
    accumulated_impulse: wp.float32


assert_constraint_header(AngularLimitData)

_OFF_BODY1 = wp.constant(dword_offset_of(AngularLimitData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(AngularLimitData, "body2"))
_OFF_LOCAL_AXIS1 = wp.constant(dword_offset_of(AngularLimitData, "local_axis1"))
_OFF_LOCAL_AXIS2 = wp.constant(dword_offset_of(AngularLimitData, "local_axis2"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(AngularLimitData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(AngularLimitData, "max_value"))
_OFF_HERTZ = wp.constant(dword_offset_of(AngularLimitData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(AngularLimitData, "damping_ratio"))
_OFF_STIFFNESS = wp.constant(dword_offset_of(AngularLimitData, "stiffness"))
_OFF_DAMPING = wp.constant(dword_offset_of(AngularLimitData, "damping"))
_OFF_INV_INITIAL_ORIENTATION = wp.constant(
    dword_offset_of(AngularLimitData, "inv_initial_orientation")
)
_OFF_REVOLUTION_COUNTER = wp.constant(dword_offset_of(AngularLimitData, "revolution_counter"))
_OFF_PREVIOUS_QUATERNION_ANGLE = wp.constant(
    dword_offset_of(AngularLimitData, "previous_quaternion_angle")
)
_OFF_CLAMP = wp.constant(dword_offset_of(AngularLimitData, "clamp"))
_OFF_EFF_INV = wp.constant(dword_offset_of(AngularLimitData, "eff_inv"))
_OFF_BIAS = wp.constant(dword_offset_of(AngularLimitData, "bias"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(AngularLimitData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(AngularLimitData, "impulse_coeff"))
_OFF_PD_GAMMA = wp.constant(dword_offset_of(AngularLimitData, "pd_gamma"))
_OFF_PD_BETA = wp.constant(dword_offset_of(AngularLimitData, "pd_beta"))
_OFF_PD_MASS_COEFF = wp.constant(dword_offset_of(AngularLimitData, "pd_mass_coeff"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(
    dword_offset_of(AngularLimitData, "accumulated_impulse")
)

#: Total dword count of one angular-limit constraint. Used by the
#: host-side container allocator to size
#: :attr:`ConstraintContainer.data`'s row count.
AL_DWORDS: int = num_dwords(AngularLimitData)


# ---------------------------------------------------------------------------
# Initialization kernel
# ---------------------------------------------------------------------------


@wp.kernel
def angular_limit_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    world_axis1: wp.array[wp.vec3f],
    world_axis2: wp.array[wp.vec3f],
    min_value: wp.array[wp.float32],
    max_value: wp.array[wp.float32],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
):
    """Pack one batch of angular-limit descriptors into ``constraints``.

    Mirrors the angular-motor initialise path: snapshots current body
    1 / body 2 orientations to derive the per-body local axes and the
    ``invInitialOrientation`` quaternion that makes the relative
    twist angle start at zero.

    Args:
        constraints: Shared column-major constraint storage.
        bodies: Solver body container; only orientations are read.
        cid_offset: Global cid of the first constraint in this batch.
        body1: Body indices for body 1 [num_in_batch].
        body2: Body indices for body 2 [num_in_batch].
        world_axis1: Hinge axis on body 1 in *world* space [num_in_batch].
        world_axis2: Hinge axis on body 2 in *world* space [num_in_batch].
        min_value: Lower bound on relative twist angle
            [num_in_batch] [rad]. Pass something like ``-1e9`` for a
            one-sided upper limit.
        max_value: Upper bound on relative twist angle
            [num_in_batch] [rad]. Pass something like ``1e9`` for a
            one-sided lower limit. The limit is disabled when
            ``min_value > max_value``.
        hertz: Box2D soft-constraint natural frequency
            [num_in_batch] [Hz]. Active when
            ``stiffness == 0 and damping == 0``.
        damping_ratio: Box2D soft-constraint damping ratio
            [num_in_batch]. Typical critical-damping value is 1.0.
        stiffness: PD spring gain [num_in_batch] [N*m/rad]. Nonzero
            selects the PD path.
        damping: PD damping gain [num_in_batch] [N*m*s/rad]. Nonzero
            selects the PD path.
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

    inv_init = wp.quat_inverse(wp.quat_inverse(q1) * q2)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_ANGULAR_LIMIT)
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
    write_vec3(constraints, _OFF_LOCAL_AXIS1, cid, local_axis1)
    write_vec3(constraints, _OFF_LOCAL_AXIS2, cid, local_axis2)
    write_float(constraints, _OFF_MIN_VALUE, cid, min_value[tid])
    write_float(constraints, _OFF_MAX_VALUE, cid, max_value[tid])
    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_STIFFNESS, cid, stiffness[tid])
    write_float(constraints, _OFF_DAMPING, cid, damping[tid])
    write_quat(constraints, _OFF_INV_INITIAL_ORIENTATION, cid, inv_init)
    write_int(constraints, _OFF_REVOLUTION_COUNTER, cid, 0)
    write_float(constraints, _OFF_PREVIOUS_QUATERNION_ANGLE, cid, 0.0)
    write_int(constraints, _OFF_CLAMP, cid, _CLAMP_NONE)
    write_float(constraints, _OFF_EFF_INV, cid, 0.0)
    write_float(constraints, _OFF_BIAS, cid, 0.0)
    write_float(constraints, _OFF_MASS_COEFF, cid, 0.0)
    write_float(constraints, _OFF_IMPULSE_COEFF, cid, 0.0)
    write_float(constraints, _OFF_PD_GAMMA, cid, 0.0)
    write_float(constraints, _OFF_PD_BETA, cid, 0.0)
    write_float(constraints, _OFF_PD_MASS_COEFF, cid, 0.0)
    write_float(constraints, _OFF_ACCUMULATED_IMPULSE, cid, 0.0)


# ---------------------------------------------------------------------------
# Per-iteration math
# ---------------------------------------------------------------------------


@wp.func
def angular_limit_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Per-substep prepare.

    Updates the revolution tracker, decides which side of the stop
    (if any) is active, caches the soft-constraint / PD coefficients
    and (importantly) warm-starts the accumulated limit impulse.

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
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    inv_init = read_quat(constraints, base_offset + _OFF_INV_INITIAL_ORIENTATION, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    dt = 1.0 / idt
    pd_mode = stiffness > 0.0 or damping > 0.0

    # Single-axis Jacobian (body-1 world axis). Matches the angular
    # motor's PD path: a companion rotation lock keeps the two body
    # axes parallel, so one axis is both simpler and more accurate.
    j = wp.quat_rotate(q1, local_axis1)

    # Scalar inverse effective mass for the 1-DoF row.
    eff_inv = wp.dot(inv_inertia1 @ j, j) + wp.dot(inv_inertia2 @ j, j)
    write_float(constraints, base_offset + _OFF_EFF_INV, cid, eff_inv)

    # Unbounded-angle tracker. ``diff`` evaluates to identity at the
    # initialise-time pose so the extracted angle starts at zero.
    diff = q2 * inv_init * wp.quat_inverse(q1)
    new_q_angle = extract_rotation_angle(diff, j)
    old_counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
    old_prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
    new_counter, new_prev = revolution_tracker_update(new_q_angle, old_counter, old_prev)
    write_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid, new_counter)
    write_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid, new_prev)
    cumulative_angle = revolution_tracker_angle(new_counter, new_prev)

    # Determine which side of the stop (if any) is active. We compare
    # the cumulative angle directly against the user bounds to stay
    # monotone for arbitrarily large ``|min|`` / ``|max|`` -- see the
    # analogous block in :mod:`constraint_actuated_double_ball_socket`.
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

    # Cache per-substep coefficients. Both branches write every slot
    # (even when the limit is inactive) so iterate doesn't have to
    # guard on stale values.
    if pd_mode:
        gamma, beta, m_soft = pd_coefficients(stiffness, damping, limit_C, eff_inv, dt)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, gamma)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, beta)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, m_soft)
        # Zero the Box2D slots to keep the storage in a defined state.
        write_float(constraints, base_offset + _OFF_BIAS, cid, 0.0)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, 0.0)
    else:
        bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
            hertz, damping_ratio, dt
        )
        # Legacy / Box2D sign convention: iterate drives ``jv ==
        # -bias``, so prefold the positional error.
        write_float(constraints, base_offset + _OFF_BIAS, cid, -limit_C * bias_rate)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, 0.0)

    # Warm start. ``acc`` retains sign from the previous substep --
    # still unilateral for the newly-selected clamp side but that's
    # re-enforced in iterate on the first PGS pass.
    if clamp != _CLAMP_NONE:
        bodies.angular_velocity[b1] = bodies.angular_velocity[b1] + inv_inertia1 @ (j * acc)
        bodies.angular_velocity[b2] = bodies.angular_velocity[b2] - inv_inertia2 @ (j * acc)
    else:
        # No active clamp -> drop any lingering impulse so the row
        # contributes nothing this substep.
        write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, 0.0)


@wp.func
def angular_limit_iterate_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
    use_bias: wp.bool,
):
    """PGS iterate.

    Dual-mode (Box2D or PD) unilateral scalar row. ``clamp``
    decides which side of the stop is active -- ``_CLAMP_NONE``
    leaves everything untouched.

    ``use_bias`` is the Box2D v3 TGS-soft ``useBias`` flag. Limits
    keep their bias in both passes (Box2D v3 only gates the
    *compliant* soft-bias branch: when the limit is actively violated
    the ``C > 0`` speculative bias stays on regardless, since letting
    the relax pass skip penetration correction would leave the body
    stuck inside the stop). For the in-range compliant branch we
    currently keep the bias on too; flip to the gated behaviour if
    limit ringing shows up in practice.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
    if clamp == _CLAMP_NONE:
        return

    angular_velocity1 = bodies.angular_velocity[b1]
    angular_velocity2 = bodies.angular_velocity[b2]
    inv_inertia1 = bodies.inverse_inertia_world[b1]
    inv_inertia2 = bodies.inverse_inertia_world[b2]

    q1 = bodies.orientation[b1]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    j = wp.quat_rotate(q1, local_axis1)

    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    pd_mode = stiffness > 0.0 or damping > 0.0

    # Sign convention matches :mod:`constraint_actuated_double_ball_socket`'s
    # axial limit row so positive ``acc`` corresponds to a torque that
    # *decreases* the cumulative twist angle: ``+j`` spins body 1 forward
    # and body 2 back, and ``angle ~ w2.j - w1.j``.
    jv_axial = wp.dot(j, angular_velocity1 - angular_velocity2)

    lam = float(0.0)
    if pd_mode:
        m_soft = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid)
        gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA, cid)
        beta = read_float(constraints, base_offset + _OFF_PD_BETA, cid)
        if m_soft > 0.0:
            # Same form as the actuated DBS limit iterate: no feed-forward
            # velocity term (a limit has no velocity setpoint).
            lam = -m_soft * (jv_axial - beta + gamma * acc)
    else:
        mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
        impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)
        bias = read_float(constraints, base_offset + _OFF_BIAS, cid)
        if eff_inv > 0.0:
            eff = 1.0 / eff_inv
            lam_unsoft = -eff * (jv_axial + bias)
            lam = mass_coeff * lam_unsoft - impulse_coeff * acc

    old_acc = acc
    acc = acc + lam
    # Unilateral clamp: the limit only pushes back toward the
    # allowed range. With the sign convention above, ``acc > 0``
    # reduces the twist angle (right thing when clamped at max) and
    # ``acc < 0`` increases it (right thing when clamped at min).
    if clamp == _CLAMP_MAX:
        acc = wp.max(0.0, acc)
    else:
        acc = wp.min(0.0, acc)
    lam = acc - old_acc
    write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    bodies.angular_velocity[b1] = angular_velocity1 + inv_inertia1 @ (j * lam)
    bodies.angular_velocity[b2] = angular_velocity2 - inv_inertia2 @ (j * lam)


@wp.func
def angular_limit_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable angular-limit wrench on body 2."""
    b1 = body_pair.b1
    q1 = bodies.orientation[b1]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    j = wp.quat_rotate(q1, local_axis1)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    # Body 2 receives ``-j * acc`` of angular impulse (see iterate).
    # Divide by the substep's dt to get the torque.
    torque = -j * (acc * idt)
    return wp.vec3f(0.0, 0.0, 0.0), torque


@wp.func
def angular_limit_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    angular_limit_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def angular_limit_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    angular_limit_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def angular_limit_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this angular limit exerts on body 2.

    Pure-angular constraint -- ``force`` is always zero. See
    :func:`angular_limit_world_wrench_at` for the torque math.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return angular_limit_world_wrench_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def angular_limit_world_error_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
) -> wp.spatial_vector:
    """Position-level constraint residual for an angular limit.

    Reads the persisted revolution-tracker state (updated every
    ``prepare_for_iteration``) to recover the unbounded relative
    twist angle, then returns ``cumulative_angle - max_value`` when
    clamped at the upper stop, ``cumulative_angle - min_value``
    when clamped at the lower stop, and zero when strictly within
    ``[min_value, max_value]``. This matches the ``limit_C``
    expression the prepare kernel feeds into the bias / PD gains.

    Output: :class:`wp.spatial_vector` with ``spatial_top`` = zero,
    ``spatial_bottom`` = ``(0, 0, limit_C)``.
    """
    counter = read_int(constraints, base_offset + _OFF_REVOLUTION_COUNTER, cid)
    prev = read_float(constraints, base_offset + _OFF_PREVIOUS_QUATERNION_ANGLE, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    cumulative_angle = revolution_tracker_angle(counter, prev)
    limit_c = wp.float32(0.0)
    if min_value <= max_value:
        if cumulative_angle > max_value:
            limit_c = cumulative_angle - max_value
        elif cumulative_angle < min_value:
            limit_c = cumulative_angle - min_value
    return wp.spatial_vector(wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(0.0, 0.0, limit_c))


@wp.func
def angular_limit_world_error(
    constraints: ConstraintContainer,
    cid: wp.int32,
) -> wp.spatial_vector:
    """Direct wrapper around :func:`angular_limit_world_error_at`."""
    return angular_limit_world_error_at(constraints, cid, 0)
