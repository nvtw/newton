# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""1-DoF linear limit: one- or two-sided soft translational stop.

Translational analogue of :mod:`constraint_angular_limit`. Clamps the
relative slide between two bodies along a world axis to
``[min_value, max_value]`` (m) with a unilateral soft PGS row.
Designed to compose with :mod:`constraint_double_ball_socket_prismatic`
(positional lock + rotation lock) and optionally
:mod:`constraint_linear_motor` (drive) to build a limited prismatic
joint without baking the limit into either piece.

Like :mod:`constraint_linear_motor`, the velocity Jacobian is the
body-COM form ``jv = n . (v1 - v2)`` (no lever arms). The companion
prismatic lock already cancels the perpendicular translations at the
joint anchor, so the centre-of-mass relative velocity equals the
anchor-point relative velocity projected onto the slide axis --
keeping the row cheap and matching the upstream Jitter2 formulation
used by :mod:`constraint_linear_motor`.

The slide *position* for the bias term is reconstructed from anchor
points in exactly the same way as the linear-motor PD path: each
body carries one anchor in its local frame, the current world
anchors are ``p_k + q_k * local_anchor_k``, and

.. math::
    s = \\hat{n} \\cdot (p_{a2} - p_{a1})

The positional error fed to :func:`constraint_container.pd_coefficients`
/ :func:`constraint_container.soft_constraint_coefficients` is
``s - rest_offset - (min/max)_value``, where ``rest_offset`` is the
initial slide at finalize time (so ``min_value = max_value = 0``
locks the joint at the initial relative pose, matching the angular
limit's "0 = hold initial" convention).

Convention table (same as :mod:`constraint_angular_limit`):

* **Box2D / Bepu** when ``stiffness == 0 and damping == 0`` -- uses
  ``hertz`` [Hz] and ``damping_ratio`` [dimensionless].
* **PD spring-damper** when ``stiffness > 0`` or ``damping > 0`` --
  gains are ``stiffness`` [N/m] and ``damping`` [N*s/m].

In both cases the clamp is unilateral (``acc >= 0`` when at
``max``, ``acc <= 0`` when at ``min``) so the limit only ever pushes
the joint back into the allowed range. ``min_value > max_value``
expresses a disabled row; one-sided limits are spelled by a large
sentinel on the unused side (e.g. ``min_value = -1e9``).
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.jitter.body import BodyContainer
from newton._src.solvers.jitter.constraints.constraint_container import (
    CONSTRAINT_TYPE_LINEAR_LIMIT,
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
    "LL_DWORDS",
    "LinearLimitData",
    "linear_limit_initialize_kernel",
    "linear_limit_iterate",
    "linear_limit_iterate_at",
    "linear_limit_prepare_for_iteration",
    "linear_limit_prepare_for_iteration_at",
    "linear_limit_world_wrench",
    "linear_limit_world_wrench_at",
]


# ---------------------------------------------------------------------------
# Clamp state constants (match angular_limit / actuated_dbs)
# ---------------------------------------------------------------------------

_CLAMP_NONE = wp.constant(wp.int32(0))
_CLAMP_MAX = wp.constant(wp.int32(1))
_CLAMP_MIN = wp.constant(wp.int32(2))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@wp.struct
class LinearLimitData:
    """Per-constraint dword-layout schema for a linear limit."""

    constraint_type: wp.int32

    body1: wp.int32
    body2: wp.int32

    # Slide axis on each body in the body's local frame (captured from
    # ``world_axis1/2`` at initialise time, after normalisation).
    local_axis1: wp.vec3f
    local_axis2: wp.vec3f

    # Anchor offsets from each body's COM in local frame. Used to
    # reconstruct the current slide ``s = hat_n . (p_a2 - p_a1)`` each
    # substep. See the file docstring.
    local_anchor1: wp.vec3f
    local_anchor2: wp.vec3f

    # Allowed slide range [m], measured as (current_slide -
    # rest_offset). Disabled when ``min_value > max_value``.
    min_value: wp.float32
    max_value: wp.float32

    # Box2D / Bepu soft-constraint knobs. Active when ``stiffness == 0``
    # and ``damping == 0``.
    hertz: wp.float32
    damping_ratio: wp.float32

    # PD spring-damper gains (N/m, N*s/m). Active when either is > 0.
    stiffness: wp.float32
    damping: wp.float32

    # Initial slide ``hat_n . (world_anchor2 - world_anchor1)`` at
    # finalize. Subtracted from the current slide so user bounds are
    # always relative to the starting pose.
    rest_offset: wp.float32

    # Current clamp state (_CLAMP_NONE / _CLAMP_MAX / _CLAMP_MIN),
    # rewritten each substep.
    clamp: wp.int32

    # Cached scalar inverse effective mass ``1/m1 + 1/m2`` (constant
    # across substeps for rigid bodies but cached here so iterate
    # doesn't need to reload the two inverse masses).
    eff_inv: wp.float32

    # Cached Box2D soft-constraint coefficients. ``bias`` already
    # absorbs ``-limit_C * bias_rate``.
    bias: wp.float32
    mass_coeff: wp.float32
    impulse_coeff: wp.float32

    # Cached PD coefficients (gamma, beta, effective-mass softened).
    pd_gamma: wp.float32
    pd_beta: wp.float32
    pd_mass_coeff: wp.float32

    # Accumulated linear impulse [N*s] across the substep's PGS
    # passes. Unilateral clamp applied in iterate.
    accumulated_impulse: wp.float32


assert_constraint_header(LinearLimitData)

_OFF_BODY1 = wp.constant(dword_offset_of(LinearLimitData, "body1"))
_OFF_BODY2 = wp.constant(dword_offset_of(LinearLimitData, "body2"))
_OFF_LOCAL_AXIS1 = wp.constant(dword_offset_of(LinearLimitData, "local_axis1"))
_OFF_LOCAL_AXIS2 = wp.constant(dword_offset_of(LinearLimitData, "local_axis2"))
_OFF_LOCAL_ANCHOR1 = wp.constant(dword_offset_of(LinearLimitData, "local_anchor1"))
_OFF_LOCAL_ANCHOR2 = wp.constant(dword_offset_of(LinearLimitData, "local_anchor2"))
_OFF_MIN_VALUE = wp.constant(dword_offset_of(LinearLimitData, "min_value"))
_OFF_MAX_VALUE = wp.constant(dword_offset_of(LinearLimitData, "max_value"))
_OFF_HERTZ = wp.constant(dword_offset_of(LinearLimitData, "hertz"))
_OFF_DAMPING_RATIO = wp.constant(dword_offset_of(LinearLimitData, "damping_ratio"))
_OFF_STIFFNESS = wp.constant(dword_offset_of(LinearLimitData, "stiffness"))
_OFF_DAMPING = wp.constant(dword_offset_of(LinearLimitData, "damping"))
_OFF_REST_OFFSET = wp.constant(dword_offset_of(LinearLimitData, "rest_offset"))
_OFF_CLAMP = wp.constant(dword_offset_of(LinearLimitData, "clamp"))
_OFF_EFF_INV = wp.constant(dword_offset_of(LinearLimitData, "eff_inv"))
_OFF_BIAS = wp.constant(dword_offset_of(LinearLimitData, "bias"))
_OFF_MASS_COEFF = wp.constant(dword_offset_of(LinearLimitData, "mass_coeff"))
_OFF_IMPULSE_COEFF = wp.constant(dword_offset_of(LinearLimitData, "impulse_coeff"))
_OFF_PD_GAMMA = wp.constant(dword_offset_of(LinearLimitData, "pd_gamma"))
_OFF_PD_BETA = wp.constant(dword_offset_of(LinearLimitData, "pd_beta"))
_OFF_PD_MASS_COEFF = wp.constant(dword_offset_of(LinearLimitData, "pd_mass_coeff"))
_OFF_ACCUMULATED_IMPULSE = wp.constant(
    dword_offset_of(LinearLimitData, "accumulated_impulse")
)

#: Total dword count of one linear-limit constraint.
LL_DWORDS: int = num_dwords(LinearLimitData)


# ---------------------------------------------------------------------------
# Initialization kernel
# ---------------------------------------------------------------------------


@wp.kernel
def linear_limit_initialize_kernel(
    constraints: ConstraintContainer,
    bodies: BodyContainer,
    cid_offset: wp.int32,
    body1: wp.array[wp.int32],
    body2: wp.array[wp.int32],
    world_axis1: wp.array[wp.vec3f],
    world_axis2: wp.array[wp.vec3f],
    world_anchor1: wp.array[wp.vec3f],
    world_anchor2: wp.array[wp.vec3f],
    min_value: wp.array[wp.float32],
    max_value: wp.array[wp.float32],
    hertz: wp.array[wp.float32],
    damping_ratio: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
):
    """Pack one batch of linear-limit descriptors into ``constraints``.

    Mirrors the linear-motor initialise path. See the field-level
    docstrings on :class:`LinearLimitData` for the meaning of each
    parameter.
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

    rest_offset = wp.dot(a1, w_anc2 - w_anc1)

    constraint_set_type(constraints, cid, CONSTRAINT_TYPE_LINEAR_LIMIT)
    write_int(constraints, _OFF_BODY1, cid, b1)
    write_int(constraints, _OFF_BODY2, cid, b2)
    write_vec3(constraints, _OFF_LOCAL_AXIS1, cid, local_axis1)
    write_vec3(constraints, _OFF_LOCAL_AXIS2, cid, local_axis2)
    write_vec3(constraints, _OFF_LOCAL_ANCHOR1, cid, local_anchor1)
    write_vec3(constraints, _OFF_LOCAL_ANCHOR2, cid, local_anchor2)
    write_float(constraints, _OFF_MIN_VALUE, cid, min_value[tid])
    write_float(constraints, _OFF_MAX_VALUE, cid, max_value[tid])
    write_float(constraints, _OFF_HERTZ, cid, hertz[tid])
    write_float(constraints, _OFF_DAMPING_RATIO, cid, damping_ratio[tid])
    write_float(constraints, _OFF_STIFFNESS, cid, stiffness[tid])
    write_float(constraints, _OFF_DAMPING, cid, damping[tid])
    write_float(constraints, _OFF_REST_OFFSET, cid, rest_offset)
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
def linear_limit_prepare_for_iteration_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Per-substep prepare.

    Recomputes the current world-space slide, decides the clamp side,
    caches soft/PD coefficients, and warm-starts.
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
    local_anchor1 = read_vec3(constraints, base_offset + _OFF_LOCAL_ANCHOR1, cid)
    local_anchor2 = read_vec3(constraints, base_offset + _OFF_LOCAL_ANCHOR2, cid)
    min_value = read_float(constraints, base_offset + _OFF_MIN_VALUE, cid)
    max_value = read_float(constraints, base_offset + _OFF_MAX_VALUE, cid)
    hertz = read_float(constraints, base_offset + _OFF_HERTZ, cid)
    damping_ratio = read_float(constraints, base_offset + _OFF_DAMPING_RATIO, cid)
    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    rest_offset = read_float(constraints, base_offset + _OFF_REST_OFFSET, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    dt = 1.0 / idt
    pd_mode = stiffness > 0.0 or damping > 0.0

    # Body-1 axis drives the Jacobian (matches linear_motor).
    j = wp.quat_rotate(q1, local_axis1)

    # Scalar effective mass (body-COM form).
    eff_inv = inv_mass1 + inv_mass2
    write_float(constraints, base_offset + _OFF_EFF_INV, cid, eff_inv)

    # Current slide from world anchors.
    world_anchor1 = p1 + wp.quat_rotate(q1, local_anchor1)
    world_anchor2 = p2 + wp.quat_rotate(q2, local_anchor2)
    slide = wp.dot(j, world_anchor2 - world_anchor1) - rest_offset

    # Clamp side + positional error.
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

    # Cache both coefficient sets every substep regardless of branch,
    # matching :mod:`constraint_angular_limit`.
    if pd_mode:
        gamma, beta, m_soft = pd_coefficients(stiffness, damping, limit_C, eff_inv, dt)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, gamma)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, beta)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, m_soft)
        write_float(constraints, base_offset + _OFF_BIAS, cid, 0.0)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, 0.0)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, 0.0)
    else:
        bias_rate, mass_coeff, impulse_coeff = soft_constraint_coefficients(
            hertz, damping_ratio, dt
        )
        write_float(constraints, base_offset + _OFF_BIAS, cid, -limit_C * bias_rate)
        write_float(constraints, base_offset + _OFF_MASS_COEFF, cid, mass_coeff)
        write_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid, impulse_coeff)
        write_float(constraints, base_offset + _OFF_PD_GAMMA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_BETA, cid, 0.0)
        write_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid, 0.0)

    # Warm start: apply previous-substep impulse if clamped (and match
    # the sign convention used in iterate -- body 1 gets +j, body 2
    # gets -j).
    if clamp != _CLAMP_NONE:
        bodies.velocity[b1] = bodies.velocity[b1] + j * (acc * inv_mass1)
        bodies.velocity[b2] = bodies.velocity[b2] - j * (acc * inv_mass2)
    else:
        write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, 0.0)


@wp.func
def linear_limit_iterate_at(
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
    determines whether the row contributes. ``use_bias`` is the
    Box2D v3 TGS-soft ``useBias`` flag; see
    :func:`angular_limit_iterate_at` for why limits keep their bias
    on in both passes.
    """
    b1 = body_pair.b1
    b2 = body_pair.b2

    clamp = read_int(constraints, base_offset + _OFF_CLAMP, cid)
    if clamp == _CLAMP_NONE:
        return

    velocity1 = bodies.velocity[b1]
    velocity2 = bodies.velocity[b2]
    inv_mass1 = bodies.inverse_mass[b1]
    inv_mass2 = bodies.inverse_mass[b2]

    q1 = bodies.orientation[b1]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    j = wp.quat_rotate(q1, local_axis1)

    stiffness = read_float(constraints, base_offset + _OFF_STIFFNESS, cid)
    damping = read_float(constraints, base_offset + _OFF_DAMPING, cid)
    eff_inv = read_float(constraints, base_offset + _OFF_EFF_INV, cid)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)

    pd_mode = stiffness > 0.0 or damping > 0.0

    # Match actuated-DBS prismatic sign: ``jv = n . (v1 - v2)``;
    # positive ``lam`` applied as ``+j`` on body 1 / ``-j`` on body 2
    # decreases the slide (body 2 moves in -j direction relative to 1).
    jv = wp.dot(j, velocity1 - velocity2)

    lam = float(0.0)
    if pd_mode:
        m_soft = read_float(constraints, base_offset + _OFF_PD_MASS_COEFF, cid)
        gamma = read_float(constraints, base_offset + _OFF_PD_GAMMA, cid)
        beta = read_float(constraints, base_offset + _OFF_PD_BETA, cid)
        if m_soft > 0.0:
            lam = -m_soft * (jv - beta + gamma * acc)
    else:
        mass_coeff = read_float(constraints, base_offset + _OFF_MASS_COEFF, cid)
        impulse_coeff = read_float(constraints, base_offset + _OFF_IMPULSE_COEFF, cid)
        bias = read_float(constraints, base_offset + _OFF_BIAS, cid)
        if eff_inv > 0.0:
            eff = 1.0 / eff_inv
            lam_unsoft = -eff * (jv + bias)
            lam = mass_coeff * lam_unsoft - impulse_coeff * acc

    old_acc = acc
    acc = acc + lam
    # Unilateral clamp: ``acc > 0`` decreases slide (correct at max),
    # ``acc < 0`` increases slide (correct at min).
    if clamp == _CLAMP_MAX:
        acc = wp.max(0.0, acc)
    else:
        acc = wp.min(0.0, acc)
    lam = acc - old_acc
    write_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid, acc)

    bodies.velocity[b1] = velocity1 + j * (lam * inv_mass1)
    bodies.velocity[b2] = velocity2 - j * (lam * inv_mass2)


@wp.func
def linear_limit_world_wrench_at(
    constraints: ConstraintContainer,
    cid: wp.int32,
    base_offset: wp.int32,
    bodies: BodyContainer,
    body_pair: ConstraintBodies,
    idt: wp.float32,
):
    """Composable linear-limit wrench on body 2."""
    b1 = body_pair.b1
    q1 = bodies.orientation[b1]
    local_axis1 = read_vec3(constraints, base_offset + _OFF_LOCAL_AXIS1, cid)
    j = wp.quat_rotate(q1, local_axis1)
    acc = read_float(constraints, base_offset + _OFF_ACCUMULATED_IMPULSE, cid)
    # Body 2 gets ``-j * acc`` of linear impulse -> divide by dt for
    # the force applied during the last substep.
    force = -j * (acc * idt)
    return force, wp.vec3f(0.0, 0.0, 0.0)


@wp.func
def linear_limit_prepare_for_iteration(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    linear_limit_prepare_for_iteration_at(constraints, cid, 0, bodies, body_pair, idt)


@wp.func
def linear_limit_iterate(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
    use_bias: wp.bool,
):
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    linear_limit_iterate_at(constraints, cid, 0, bodies, body_pair, idt, use_bias)


@wp.func
def linear_limit_world_wrench(
    constraints: ConstraintContainer,
    cid: wp.int32,
    bodies: BodyContainer,
    idt: wp.float32,
):
    """World-frame wrench (force, torque) this linear limit exerts on body 2.

    Pure-linear constraint -- ``torque`` is always zero. See
    :func:`linear_limit_world_wrench_at` for the force math.
    """
    b1 = read_int(constraints, _OFF_BODY1, cid)
    b2 = read_int(constraints, _OFF_BODY2, cid)
    body_pair = constraint_bodies_make(b1, b2)
    return linear_limit_world_wrench_at(constraints, cid, 0, bodies, body_pair, idt)
