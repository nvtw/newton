# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Testable joint-constraint math for the Box3D TGS-Soft solver.

Every function is a pure ``@wp.func`` operating on scalars and vectors
so it can be unit-tested from a trivial ``@wp.kernel`` wrapper.  The
production fused-solve kernel reimplements the same math inline inside
``@wp.func_native`` CUDA snippets with shared memory.

Port of Box2D v3 joint formulations to 3-D with full ``mat33`` inertia.

Supported joint types
---------------------
* **Revolute** -- 3-D hinge: point-to-point (3 DOF) + angular lock
  perpendicular to hinge axis (2 DOF) + optional motor (1 DOF).
* **Fixed** (weld) -- full rigid lock: point-to-point (3 DOF) + full
  angular lock (3 DOF).
"""

from __future__ import annotations

import warp as wp


# ---------------------------------------------------------------------------
# 3x3 symmetric solve via Cramer's rule
# ---------------------------------------------------------------------------


@wp.func
def _solve_symmetric_3x3(
    K00: float,
    K01: float,
    K02: float,
    K11: float,
    K12: float,
    K22: float,
    rhs_x: float,
    rhs_y: float,
    rhs_z: float,
) -> wp.vec3:
    """Solve a 3x3 symmetric positive-definite system via Cramer's rule.

    The matrix is::

        K = [K00  K01  K02]
            [K01  K11  K12]
            [K02  K12  K22]

    Returns the solution vector ``x`` such that ``K @ x = rhs``, or zero
    if the determinant is too small.

    Args:
        K00: Element (0,0).
        K01: Element (0,1) = (1,0).
        K02: Element (0,2) = (2,0).
        K11: Element (1,1).
        K12: Element (1,2) = (2,1).
        K22: Element (2,2).
        rhs_x: Right-hand side x component.
        rhs_y: Right-hand side y component.
        rhs_z: Right-hand side z component.
    """
    det = K00 * (K11 * K22 - K12 * K12) - K01 * (K01 * K22 - K12 * K02) + K02 * (K01 * K12 - K11 * K02)
    if wp.abs(det) < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    inv_det = 1.0 / det
    # Cofactor / adjugate matrix (symmetric)
    i00 = (K11 * K22 - K12 * K12) * inv_det
    i01 = (K02 * K12 - K01 * K22) * inv_det
    i02 = (K01 * K12 - K02 * K11) * inv_det
    i11 = (K00 * K22 - K02 * K02) * inv_det
    i12 = (K01 * K02 - K00 * K12) * inv_det
    i22 = (K00 * K11 - K01 * K01) * inv_det
    x = i00 * rhs_x + i01 * rhs_y + i02 * rhs_z
    y = i01 * rhs_x + i11 * rhs_y + i12 * rhs_z
    z = i02 * rhs_x + i12 * rhs_y + i22 * rhs_z
    return wp.vec3(x, y, z)


# ---------------------------------------------------------------------------
# Skew-symmetric helpers
# ---------------------------------------------------------------------------


@wp.func
def _skew_outer_diag(r: wp.vec3, inv_I: wp.mat33) -> wp.mat33:
    """Compute ``-[r]_x @ I^-1 @ [r]_x`` (symmetric) using explicit expansion.

    This gives the rotational contribution to the 3x3 effective-mass
    matrix for a point-to-point constraint::

        K_rot = [r]_x^T @ I^-1 @ [r]_x   (note: [r]_x^T = -[r]_x)

    so ``K_rot = -[r]_x @ I^-1 @ [r]_x``.

    Returns the 3x3 symmetric matrix as ``wp.mat33``.

    Args:
        r: Anchor offset vector [m].
        inv_I: World-frame inverse inertia [1/(kg m^2)].
    """
    # Columns of [r]_x (skew-symmetric of r):
    # col0 = (0, r[2], -r[1]), col1 = (-r[2], 0, r[0]), col2 = (r[1], -r[0], 0)
    # We need M = -[r]_x @ inv_I @ [r]_x = [r]_x^T @ inv_I @ [r]_x
    # where [r]_x v = r x v
    #
    # For efficiency, compute inv_I @ [r]_x first (3 columns), then dot
    # with rows of [r]_x^T.
    #
    # Let c_i = inv_I @ (r x e_i)  for standard basis e_i
    # Then K_{ij} = (r x e_i) . c_j = (r x e_i) . (inv_I @ (r x e_j))
    # But that's K = [r]_x^T @ inv_I @ [r]_x -- exactly what we want.

    # r x e_0 = (0, r[2], -r[1])
    s0 = inv_I * wp.vec3(0.0, r[2], -r[1])
    # r x e_1 = (-r[2], 0, r[0])
    s1 = inv_I * wp.vec3(-r[2], 0.0, r[0])
    # r x e_2 = (r[1], -r[0], 0)
    s2 = inv_I * wp.vec3(r[1], -r[0], 0.0)

    # K_{ij} = dot(r x e_i, s_j)
    re0 = wp.vec3(0.0, r[2], -r[1])
    re1 = wp.vec3(-r[2], 0.0, r[0])
    re2 = wp.vec3(r[1], -r[0], 0.0)

    return wp.mat33(
        wp.dot(re0, s0),
        wp.dot(re0, s1),
        wp.dot(re0, s2),
        wp.dot(re1, s0),
        wp.dot(re1, s1),
        wp.dot(re1, s2),
        wp.dot(re2, s0),
        wp.dot(re2, s1),
        wp.dot(re2, s2),
    )


# ---------------------------------------------------------------------------
# Revolute joint -- point-to-point (3 DOF)
# ---------------------------------------------------------------------------


@wp.func
def solve_revolute_point_to_point(
    vel_a: wp.vec3,
    ang_vel_a: wp.vec3,
    vel_b: wp.vec3,
    ang_vel_b: wp.vec3,
    r_a: wp.vec3,
    r_b: wp.vec3,
    separation: wp.vec3,
    inv_mass_a: float,
    inv_inertia_a: wp.mat33,
    inv_mass_b: float,
    inv_inertia_b: wp.mat33,
    linear_impulse: wp.vec3,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    use_bias: int,
) -> wp.vec3:
    """Solve the 3-DOF point-to-point position constraint of a revolute joint.

    Builds the 3x3 effective-mass matrix::

        K = (m_A^-1 + m_B^-1) I_3 + [r_A]_x^T I_A^-1 [r_A]_x
                                    + [r_B]_x^T I_B^-1 [r_B]_x

    and solves ``K @ impulse = -(mass_scale * Cdot + bias) - impulse_scale * accumulated``.

    Returns the impulse delta ``vec3`` (to be added to accumulated impulse).
    The caller is responsible for applying the impulse to body velocities.

    Args:
        vel_a: Linear velocity of body A [m/s].
        ang_vel_a: Angular velocity of body A [rad/s].
        vel_b: Linear velocity of body B [m/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        r_a: World-space anchor offset from COM of body A [m].
        r_b: World-space anchor offset from COM of body B [m].
        separation: Position error ``(pos_B + r_B) - (pos_A + r_A)`` [m].
        inv_mass_a: Inverse mass of body A [1/kg].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_mass_b: Inverse mass of body B [1/kg].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        linear_impulse: Accumulated linear impulse from previous iterations [N s].
        bias_rate: Softness bias rate [1/s].
        mass_scale: Softness mass scale [dimensionless].
        impulse_scale: Softness impulse scale [dimensionless].
        use_bias: 1 for biased pass, 0 for relaxation.
    """
    # Velocity error: Cdot = (vB + wB x rB) - (vA + wA x rA)
    cdot = (vel_b + wp.cross(ang_vel_b, r_b)) - (vel_a + wp.cross(ang_vel_a, r_a))

    # Bias (position correction)
    bias = wp.vec3(0.0, 0.0, 0.0)
    ms = 1.0
    isv = 0.0
    if use_bias != 0:
        bias = bias_rate * separation
        ms = mass_scale
        isv = impulse_scale

    # Build 3x3 K matrix
    m_total = inv_mass_a + inv_mass_b
    Ka = _skew_outer_diag(r_a, inv_inertia_a)
    Kb = _skew_outer_diag(r_b, inv_inertia_b)

    K00 = m_total + Ka[0, 0] + Kb[0, 0]
    K01 = Ka[0, 1] + Kb[0, 1]
    K02 = Ka[0, 2] + Kb[0, 2]
    K11 = m_total + Ka[1, 1] + Kb[1, 1]
    K12 = Ka[1, 2] + Kb[1, 2]
    K22 = m_total + Ka[2, 2] + Kb[2, 2]

    rhs = cdot + bias
    sol = _solve_symmetric_3x3(K00, K01, K02, K11, K12, K22, rhs[0], rhs[1], rhs[2])

    impulse = wp.vec3(
        -ms * sol[0] - isv * linear_impulse[0],
        -ms * sol[1] - isv * linear_impulse[1],
        -ms * sol[2] - isv * linear_impulse[2],
    )
    return impulse


# ---------------------------------------------------------------------------
# Revolute joint -- angular constraint perpendicular to hinge (2 DOF)
# ---------------------------------------------------------------------------


@wp.func
def solve_revolute_angular(
    ang_vel_a: wp.vec3,
    ang_vel_b: wp.vec3,
    hinge_axis_world: wp.vec3,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    angular_impulse: wp.vec2,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    use_bias: int,
) -> wp.vec3:
    """Solve the 2-DOF angular constraint perpendicular to the hinge axis.

    Constrains the relative angular velocity to lie along the hinge axis
    only, removing 2 rotational DOFs.

    Returns the 3-D angular impulse vector to apply (caller adds to angular
    velocity). The accumulated impulse is 2-D (along two basis vectors
    perpendicular to the hinge axis); the caller must update those.

    The two perpendicular basis vectors ``b1, b2`` are computed from
    *hinge_axis_world* (the hinge axis as seen by body A).

    Args:
        ang_vel_a: Angular velocity of body A [rad/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        hinge_axis_world: Unit hinge axis in world frame [dimensionless].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        angular_impulse: Accumulated 2-D angular impulse ``(b1, b2)`` [N m s].
        bias_rate: Softness bias rate [1/s] (unused for angular -- no position error tracked).
        mass_scale: Softness mass scale [dimensionless].
        impulse_scale: Softness impulse scale [dimensionless].
        use_bias: 1 for biased pass, 0 for relaxation.
    """
    h = hinge_axis_world

    # Build perpendicular basis (b1, b2) from hinge axis
    # Use the axis with the smallest component for numerical stability
    if wp.abs(h[1]) < 0.9:
        b1 = wp.vec3(h[2], 0.0, -h[0])
    else:
        b1 = wp.vec3(0.0, -h[2], h[1])
    b1_len = wp.length(b1)
    if b1_len > 1.0e-8:
        b1 = b1 / b1_len
    b2 = wp.cross(h, b1)

    # Relative angular velocity
    dw = ang_vel_b - ang_vel_a

    # Effective angular mass along each basis direction
    # k = b . (IA^-1 + IB^-1) . b  for each basis vector
    k_ang = inv_inertia_a + inv_inertia_b

    ms = 1.0
    isv = 0.0
    if use_bias != 0:
        ms = mass_scale
        isv = impulse_scale

    # Solve each DOF independently (diagonal approximation -- the two
    # perpendicular directions are decoupled for symmetric inertia)
    k1 = wp.dot(b1, k_ang * b1)
    k2 = wp.dot(b2, k_ang * b2)

    am1 = 0.0
    am2 = 0.0
    if k1 > 0.0:
        am1 = 1.0 / k1
    if k2 > 0.0:
        am2 = 1.0 / k2

    cdot1 = wp.dot(dw, b1)
    cdot2 = wp.dot(dw, b2)

    i1 = -ms * am1 * cdot1 - isv * angular_impulse[0]
    i2 = -ms * am2 * cdot2 - isv * angular_impulse[1]

    return i1 * b1 + i2 * b2


# ---------------------------------------------------------------------------
# Revolute joint -- motor (1 DOF)
# ---------------------------------------------------------------------------


@wp.func
def solve_revolute_motor(
    ang_vel_a: wp.vec3,
    ang_vel_b: wp.vec3,
    hinge_axis_world: wp.vec3,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    motor_impulse: float,
    motor_speed: float,
    max_motor_torque: float,
    sub_dt: float,
) -> float:
    """Solve the 1-DOF motor constraint on the hinge axis.

    Drives the relative angular velocity along the hinge axis toward
    *motor_speed*.  The accumulated motor impulse is clamped to
    ``[-max_motor_torque * sub_dt, +max_motor_torque * sub_dt]``.

    Returns the impulse delta (scalar) to apply along the hinge axis.

    Args:
        ang_vel_a: Angular velocity of body A [rad/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        hinge_axis_world: Unit hinge axis in world frame [dimensionless].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        motor_impulse: Accumulated motor impulse [N m s].
        motor_speed: Target relative angular speed on hinge axis [rad/s].
        max_motor_torque: Maximum motor torque [N m].
        sub_dt: Sub-step duration [s].
    """
    h = hinge_axis_world
    k_ang = inv_inertia_a + inv_inertia_b
    k = wp.dot(h, k_ang * h)
    if k <= 0.0:
        return 0.0
    am = 1.0 / k

    # Current relative angular velocity on hinge axis
    cdot = wp.dot(ang_vel_b - ang_vel_a, h) - motor_speed
    impulse = -am * cdot
    max_impulse = max_motor_torque * sub_dt
    new_motor_impulse = wp.clamp(motor_impulse + impulse, -max_impulse, max_impulse)
    return new_motor_impulse - motor_impulse


# ---------------------------------------------------------------------------
# Revolute joint -- hinge-axis limit (1 DOF)
# ---------------------------------------------------------------------------


@wp.func
def solve_revolute_limit(
    ang_vel_a: wp.vec3,
    ang_vel_b: wp.vec3,
    hinge_axis_world: wp.vec3,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    limit_impulse: float,
    angle_error: float,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    inv_sub_dt: float,
    use_bias: int,
    is_upper: int,
) -> float:
    """Solve one side of a revolute joint angle limit.

    For the **lower** limit, ``angle_error = joint_angle - lower_angle``.
    For the **upper** limit, ``angle_error = upper_angle - joint_angle``
    (and ``is_upper = 1`` to flip the velocity sign).

    The limit is unilateral: the accumulated impulse is clamped >= 0.

    Returns the impulse delta (scalar) along the hinge axis.

    Args:
        ang_vel_a: Angular velocity of body A [rad/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        hinge_axis_world: Unit hinge axis in world frame [dimensionless].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        limit_impulse: Accumulated limit impulse [N m s].
        angle_error: Signed distance from the limit [rad].
        bias_rate: Softness bias rate [1/s].
        mass_scale: Softness mass scale [dimensionless].
        impulse_scale: Softness impulse scale [dimensionless].
        inv_sub_dt: Inverse sub-step duration [1/s].
        use_bias: 1 for biased pass, 0 for relaxation.
        is_upper: 1 for upper limit (flip velocity sign), 0 for lower.
    """
    h = hinge_axis_world
    k_ang = inv_inertia_a + inv_inertia_b
    k = wp.dot(h, k_ang * h)
    if k <= 0.0:
        return 0.0
    am = 1.0 / k

    bias = 0.0
    ms = 1.0
    isv = 0.0

    if angle_error > 0.0:
        # Speculative -- limit not yet reached
        bias = angle_error * inv_sub_dt
    elif use_bias != 0:
        bias = mass_scale * bias_rate * angle_error
        ms = mass_scale
        isv = impulse_scale

    # Velocity error (sign flipped for upper limit)
    cdot = wp.dot(ang_vel_b - ang_vel_a, h)
    if is_upper != 0:
        cdot = -cdot

    impulse = -ms * am * (cdot + bias) - isv * limit_impulse
    new_impulse = wp.max(limit_impulse + impulse, 0.0)
    return new_impulse - limit_impulse


# ---------------------------------------------------------------------------
# Fixed (weld) joint -- angular lock (3 DOF)
# ---------------------------------------------------------------------------


@wp.func
def solve_fixed_angular(
    ang_vel_a: wp.vec3,
    ang_vel_b: wp.vec3,
    inv_inertia_a: wp.mat33,
    inv_inertia_b: wp.mat33,
    angular_impulse: wp.vec3,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    use_bias: int,
) -> wp.vec3:
    """Solve the 3-DOF angular lock of a fixed (weld) joint.

    Constrains the relative angular velocity to zero in all three axes.

    Returns the angular impulse delta ``vec3`` to apply.  The caller adds
    this to the accumulated angular impulse and applies it to body
    angular velocities.

    Args:
        ang_vel_a: Angular velocity of body A [rad/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        angular_impulse: Accumulated 3-D angular impulse [N m s].
        bias_rate: Softness bias rate [1/s] (angular position error not tracked here).
        mass_scale: Softness mass scale [dimensionless].
        impulse_scale: Softness impulse scale [dimensionless].
        use_bias: 1 for biased pass, 0 for relaxation.
    """
    dw = ang_vel_b - ang_vel_a

    ms = 1.0
    isv = 0.0
    if use_bias != 0:
        ms = mass_scale
        isv = impulse_scale

    # Effective angular mass: k_ang = I_A^-1 + I_B^-1
    k_ang = inv_inertia_a + inv_inertia_b

    # Solve k_ang @ impulse = -(ms * dw) - isv * accumulated
    # Invert k_ang (3x3 symmetric) via Cramer's rule
    rhs = ms * dw
    sol = _solve_symmetric_3x3(
        k_ang[0, 0],
        k_ang[0, 1],
        k_ang[0, 2],
        k_ang[1, 1],
        k_ang[1, 2],
        k_ang[2, 2],
        rhs[0],
        rhs[1],
        rhs[2],
    )

    return wp.vec3(
        -sol[0] - isv * angular_impulse[0],
        -sol[1] - isv * angular_impulse[1],
        -sol[2] - isv * angular_impulse[2],
    )
