# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Testable contact-constraint math for the Box3D TGS-Soft solver.

Every function in this module is a pure ``@wp.func`` operating on scalars
and vectors — no array indexing, no shared memory.  This makes them
callable from a trivial ``@wp.kernel`` wrapper for unit testing while
the production fused-solve kernel reimplements the same math inline
inside a ``@wp.func_native`` CUDA snippet with shared memory.

The formulations follow Box2D v3 (Erin Catto, 2024) extended to 3-D
with full 3x3 inverse-inertia tensors.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Softness parameters (Box2D v3 formulation)
# ---------------------------------------------------------------------------

PI = wp.constant(3.14159265358979323846)


@wp.func
def compute_softness(hertz: float, damping_ratio: float, h: float) -> wp.vec3:
    """Compute TGS-Soft parameters from spring frequency and damping ratio.

    Returns a ``vec3(bias_rate, mass_scale, impulse_scale)``.

    When *hertz* is zero the returned vector is ``(0, 0, 0)`` (fully rigid).

    Args:
        hertz: Spring natural frequency [Hz].
        damping_ratio: Damping ratio (1.0 = critically damped).
        h: Sub-step duration [s].
    """
    if hertz == 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    omega = 2.0 * PI * hertz
    zeta = damping_ratio
    a1 = 2.0 * zeta + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return wp.vec3(omega / a1, a2 * a3, a3)


# ---------------------------------------------------------------------------
# Tangent basis
# ---------------------------------------------------------------------------


@wp.func
def compute_tangent_basis(normal: wp.vec3) -> wp.mat33:
    """Build an orthonormal frame from a contact normal.

    Returns a ``mat33`` whose **rows** are ``[normal, tangent1, tangent2]``
    (i.e. ``result[0]`` is the normal, ``result[1]`` and ``result[2]`` are
    the two tangent directions).

    Uses the method from *Duff et al. 2017* for a branch-free, stable
    tangent-pair construction.

    Args:
        normal: Unit contact normal (A → B) [dimensionless].
    """
    n = normal
    # Duff et al. 2017 — pick the axis with the smaller component
    sign = 1.0
    if n[2] < 0.0:
        sign = -1.0
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    t1 = wp.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0])
    t2 = wp.vec3(b, sign + n[1] * n[1] * a, -n[1])
    return wp.mat33(n[0], n[1], n[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])


# ---------------------------------------------------------------------------
# Effective mass
# ---------------------------------------------------------------------------


@wp.func
def compute_contact_effective_mass(
    inv_mass_a: float,
    inv_inertia_a: wp.mat33,
    inv_mass_b: float,
    inv_inertia_b: wp.mat33,
    r_a: wp.vec3,
    r_b: wp.vec3,
    direction: wp.vec3,
) -> float:
    """Compute the effective mass for a 1-DOF constraint along *direction*.

    The effective mass is ``1 / K`` where::

        K = m_A^-1 + m_B^-1
            + dot(d, cross(I_A^-1 * cross(r_A, d), r_A))
            + dot(d, cross(I_B^-1 * cross(r_B, d), r_B))

    Returns zero if K <= 0 (degenerate configuration).

    Args:
        inv_mass_a: Inverse mass of body A [1/kg].
        inv_inertia_a: World-frame inverse inertia of body A [1/(kg m^2)].
        inv_mass_b: Inverse mass of body B [1/kg].
        inv_inertia_b: World-frame inverse inertia of body B [1/(kg m^2)].
        r_a: Anchor offset from COM of body A [m].
        r_b: Anchor offset from COM of body B [m].
        direction: Unit direction for the constraint [dimensionless].
    """
    rn_a = wp.cross(r_a, direction)
    rn_b = wp.cross(r_b, direction)
    k = inv_mass_a + inv_mass_b
    k += wp.dot(direction, wp.cross(inv_inertia_a * rn_a, r_a))
    k += wp.dot(direction, wp.cross(inv_inertia_b * rn_b, r_b))
    if k > 0.0:
        return 1.0 / k
    return 0.0


# ---------------------------------------------------------------------------
# Relative velocity
# ---------------------------------------------------------------------------


@wp.func
def compute_relative_velocity(
    vel_a: wp.vec3,
    ang_vel_a: wp.vec3,
    r_a: wp.vec3,
    vel_b: wp.vec3,
    ang_vel_b: wp.vec3,
    r_b: wp.vec3,
) -> wp.vec3:
    """Compute world-space relative velocity at the contact point (B - A).

    ``v_rel = (v_B + w_B x r_B) - (v_A + w_A x r_A)``

    Args:
        vel_a: Linear velocity of body A [m/s].
        ang_vel_a: Angular velocity of body A [rad/s].
        r_a: Anchor offset from COM of body A [m].
        vel_b: Linear velocity of body B [m/s].
        ang_vel_b: Angular velocity of body B [rad/s].
        r_b: Anchor offset from COM of body B [m].
    """
    vp_a = vel_a + wp.cross(ang_vel_a, r_a)
    vp_b = vel_b + wp.cross(ang_vel_b, r_b)
    return vp_b - vp_a


# ---------------------------------------------------------------------------
# Normal impulse (Box2D v3 soft-step)
# ---------------------------------------------------------------------------


@wp.func
def solve_contact_normal(
    vn: float,
    separation: float,
    lambda_n: float,
    normal_mass: float,
    bias_rate: float,
    mass_scale: float,
    impulse_scale: float,
    inv_sub_dt: float,
    contact_speed: float,
    use_bias: int,
) -> float:
    """Compute the normal-impulse delta for one contact point.

    Implements Box2D v3's soft-step normal constraint:

    - **Speculative** (``separation > 0``): ``bias = separation * inv_sub_dt``
    - **Penetrating** (``separation <= 0``, biased):
      ``bias = max(mass_scale * bias_rate * separation, -contact_speed)``
    - **Relaxation** (``use_bias == 0``): ``bias = 0``, ``mass_scale = 1``,
      ``impulse_scale = 0``

    The accumulated impulse is clamped >= 0 (contacts push, never pull).

    Returns the impulse delta to apply (``new_lambda - old_lambda``).

    Args:
        vn: Relative normal velocity (B - A) along normal [m/s].
        separation: Signed distance (negative = penetrating) [m].
        lambda_n: Accumulated normal impulse from previous iterations [N s].
        normal_mass: Effective mass along normal (1/K) [kg].
        bias_rate: Softness bias rate [1/s].
        mass_scale: Softness mass scale [dimensionless].
        impulse_scale: Softness impulse scale [dimensionless].
        inv_sub_dt: Inverse sub-step duration [1/s].
        contact_speed: Maximum penetration recovery speed [m/s].
        use_bias: 1 for biased pass, 0 for relaxation.
    """
    velocity_bias = 0.0
    ms = 1.0
    isv = 0.0

    if separation > 0.005:
        # Speculative contact — always active (even during relaxation).
        # Small slop (5mm) prevents jitter from floating-point separation noise.
        velocity_bias = separation * inv_sub_dt
    elif use_bias != 0:
        # Penetrating — soft position correction (only during biased pass)
        velocity_bias = wp.max(mass_scale * bias_rate * separation, -contact_speed)
        ms = mass_scale
        isv = impulse_scale
    # else: relaxation pass with penetrating contact — no bias, ms=1, isv=0

    impulse = -normal_mass * (ms * vn + velocity_bias) - isv * lambda_n
    new_lambda = wp.max(lambda_n + impulse, 0.0)
    return new_lambda - lambda_n


# ---------------------------------------------------------------------------
# Friction impulse (Coulomb cone)
# ---------------------------------------------------------------------------


@wp.func
def solve_contact_friction(
    vt: float,
    lambda_f: float,
    max_friction: float,
    tangent_mass: float,
) -> float:
    """Compute the friction-impulse delta for one tangent direction.

    Uses Coulomb friction: the accumulated impulse is clamped to
    ``[-max_friction, max_friction]`` where ``max_friction = mu * lambda_n``.

    Returns the impulse delta to apply.

    Args:
        vt: Relative tangential velocity along this tangent direction [m/s].
        lambda_f: Accumulated friction impulse [N s].
        max_friction: Friction limit (``mu * normal_impulse``) [N s].
        tangent_mass: Effective mass along tangent (1/K) [kg].
    """
    impulse = -tangent_mass * vt
    new_lambda = wp.clamp(lambda_f + impulse, -max_friction, max_friction)
    return new_lambda - lambda_f


# ---------------------------------------------------------------------------
# Apply impulse to a single body
# ---------------------------------------------------------------------------


@wp.func
def apply_body_impulse(
    vel: wp.vec3,
    ang_vel: wp.vec3,
    inv_mass: float,
    inv_inertia: wp.mat33,
    r: wp.vec3,
    impulse: wp.vec3,
    sign: float,
) -> wp.vec3:
    """Apply a linear + angular impulse to a body and return the new velocity pair.

    ``vel += sign * inv_mass * impulse``
    ``ang_vel += sign * inv_inertia * cross(r, impulse)``

    The caller passes ``sign = -1`` for body A and ``sign = +1`` for body B.

    Returns ``(new_vel, new_ang_vel)`` packed as a single ``vec3`` for the
    linear part; the angular part is returned separately via the second
    call convention (caller calls this twice, once for linear, once for angular,
    or uses :func:`apply_body_impulse_full`).

    Args:
        vel: Current linear velocity [m/s].
        ang_vel: Current angular velocity [rad/s].
        inv_mass: Inverse mass [1/kg].
        inv_inertia: World-frame inverse inertia [1/(kg m^2)].
        r: Anchor offset from COM [m].
        impulse: World-space impulse vector [N s].
        sign: -1.0 for body A, +1.0 for body B.
    """
    return vel + sign * inv_mass * impulse


@wp.func
def apply_body_angular_impulse(
    ang_vel: wp.vec3,
    inv_inertia: wp.mat33,
    r: wp.vec3,
    impulse: wp.vec3,
    sign: float,
) -> wp.vec3:
    """Apply the angular component of a contact/joint impulse.

    ``ang_vel += sign * inv_inertia * cross(r, impulse)``

    Args:
        ang_vel: Current angular velocity [rad/s].
        inv_inertia: World-frame inverse inertia [1/(kg m^2)].
        r: Anchor offset from COM [m].
        impulse: World-space impulse vector [N s].
        sign: -1.0 for body A, +1.0 for body B.
    """
    return ang_vel + sign * (inv_inertia * wp.cross(r, impulse))


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------


@wp.func
def solve_restitution(
    vn: float,
    pre_solve_rel_vel: float,
    lambda_n: float,
    total_normal_impulse: float,
    normal_mass: float,
    restitution: float,
    threshold: float,
) -> float:
    """Compute a restitution impulse delta.

    Restitution is only applied when:

    1. The pre-solve relative velocity exceeded *threshold* (bodies were
       approaching fast enough to bounce).
    2. The contact accumulated a positive total normal impulse during the
       substep loop (the contact was actually active).

    Returns the impulse delta to apply along the normal.

    Args:
        vn: Current relative normal velocity (B - A) [m/s].
        pre_solve_rel_vel: Relative normal velocity recorded before the
            substep loop [m/s].  Negative means approaching.
        lambda_n: Current accumulated normal impulse [N s].
        total_normal_impulse: Running total of normal impulse across
            all substeps [N s].
        normal_mass: Effective mass along normal [kg].
        restitution: Coefficient of restitution [dimensionless].
        threshold: Minimum approach speed for bounce [m/s].
    """
    if restitution == 0.0:
        return 0.0
    if pre_solve_rel_vel > -threshold:
        return 0.0
    if total_normal_impulse == 0.0:
        return 0.0
    impulse = -normal_mass * (vn + restitution * pre_solve_rel_vel)
    new_lambda = wp.max(lambda_n + impulse, 0.0)
    return new_lambda - lambda_n
