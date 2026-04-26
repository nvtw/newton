# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Math helpers shared across the Jitter port's constraint kernels.

These ``@wp.func`` helpers are direct ports of routines from
``Jitter2.LinearMath`` / ``Jitter2.Dynamics.Constraints.Internal`` that
don't have a one-line equivalent in Warp's stdlib. Each is exposed as
its own ``@wp.func`` so it inlines into caller kernels without any
host-side overhead.

Naming follows the corresponding C# routine in snake_case and drops the
leading underscore that earlier versions used inside individual
constraint files (these are now part of the shared API).
"""

from __future__ import annotations

import math

import warp as wp

__all__ = [
    "apply_pair_velocity_impulse",
    "create_orthonormal",
    "effective_mass_scalar",
    "extract_rotation_angle",
    "qmatrix_project_multiply_left_right",
    "revolution_tracker_angle",
    "revolution_tracker_update",
    "rotate_inertia",
]


PI = wp.constant(wp.float32(math.pi))
TWO_PI = wp.constant(wp.float32(2.0 * math.pi))


@wp.func
def create_orthonormal(v: wp.vec3f) -> wp.vec3f:
    """Direct port of ``MathHelper.CreateOrthonormal`` (MathHelper.cs:202).

    Returns a unit vector orthogonal to ``v``. Picks the axis with the
    smallest absolute component to avoid the near-degenerate case where
    the chosen perpendicular would shrink to zero.

    Used wherever a constraint needs a basis perpendicular to a fixed
    direction (hinge axis, cone axis, contact normal, ...). The choice
    of perpendicular is arbitrary but consistent across calls with the
    same input, which is all the constraint solvers actually require.
    """
    ax = wp.abs(v[0])
    ay = wp.abs(v[1])
    az = wp.abs(v[2])

    if ax <= ay and ax <= az:
        # (0, z, -y)
        y = v[2]
        z = -v[1]
        inv_len = 1.0 / wp.sqrt(y * y + z * z)
        return wp.vec3f(0.0, y * inv_len, z * inv_len)
    elif ay <= az:
        # (-z, 0, x)
        x = -v[2]
        z = v[0]
        inv_len = 1.0 / wp.sqrt(x * x + z * z)
        return wp.vec3f(x * inv_len, 0.0, z * inv_len)
    else:
        # (y, -x, 0)
        x = v[1]
        y = -v[0]
        inv_len = 1.0 / wp.sqrt(x * x + y * y)
        return wp.vec3f(x * inv_len, y * inv_len, 0.0)


@wp.func
def qmatrix_project_multiply_left_right(left: wp.quatf, right: wp.quatf) -> wp.mat33f:
    """Direct port of ``QMatrix.ProjectMultiplyLeftRight`` (QMatrix.cs:113).

    Returns the 3x3 projection of the 4x4 product ``L_left * L_right^T``
    where ``L_*`` are the (left/right) quaternion-multiplication
    matrices. The closed form below is what Jitter actually uses; we
    don't pay for the general 4x4 build.

    Quaternion convention: ``wp.quatf(x, y, z, w)`` -- same as Jitter's
    ``JQuaternion`` field order, so the indices line up directly.

    Used by every angular constraint that needs to map angular velocity
    to quaternion-error (HingeAngle, TwistAngle, ConeLimit, FixedAngle).
    """
    lx = left[0]
    ly = left[1]
    lz = left[2]
    lw = left[3]
    rx = right[0]
    ry = right[1]
    rz = right[2]
    rw = right[3]

    m00 = -lx * rx + lw * rw + lz * rz + ly * ry
    m01 = -lx * ry + lw * rz - lz * rw - ly * rx
    m02 = -lx * rz - lw * ry - lz * rx + ly * rw
    m10 = -ly * rx + lz * rw - lw * rz - lx * ry
    m11 = -ly * ry + lz * rz + lw * rw + lx * rx
    m12 = -ly * rz - lz * ry + lw * rx - lx * rw
    m20 = -lz * rx - ly * rw - lx * rz + lw * ry
    m21 = -lz * ry - ly * rz + lx * rw - lw * rx
    m22 = -lz * rz + ly * ry + lx * rx + lw * rw

    return wp.mat33f(
        m00,
        m01,
        m02,
        m10,
        m11,
        m12,
        m20,
        m21,
        m22,
    )


# ---------------------------------------------------------------------------
# Full-revolution angle tracker
#
# Ported from PhoenX (Jolt) ``FullRevolutionTracker`` +
# ``quat::ExtractRotationAngle``. Unwraps the quaternion's [-pi, pi]
# branch into an unbounded cumulative angle so hinge limits can use
# arbitrarily large ``min``/``max`` (e.g. ``min = -1e2`` for a one-
# sided clamp) without ``sin(theta/2)`` wrap-around at theta > pi.
#
# Stateless at this layer -- the caller owns two persistent scalars
# per joint: ``revolution_counter`` (int32, init 0) and
# ``previous_quaternion_angle`` (float32, init 0). Each prepare step:
# read them, call :func:`revolution_tracker_update`, write back, then
# read total angle via :func:`revolution_tracker_angle`.
# ---------------------------------------------------------------------------


@wp.func
def extract_rotation_angle(q: wp.quatf, rotation_axis: wp.vec3f) -> wp.float32:
    """Signed angle of ``q`` about ``rotation_axis``, in ``(-pi, pi]``.

    Direct port of PhoenX's ``quat::ExtractRotationAngle``
    (MiniMath/quat.cuh:703):

    .. code-block:: text

        theta = (q.W == 0) ? pi : 2 * atan(dot(q.xyz, axis) / q.W)

    The ``q.W == 0`` branch pins the output to ``+pi`` at the half-
    rotation singularity (``dot(q.xyz, axis)`` is ``+/- 1`` there, and
    ``atan2`` would otherwise return a discontinuous ``+/- pi/2``). The
    ``atan(x / y)`` form matches the C# port bit-for-bit; swapping in
    ``atan2`` would wrap at a different branch and break the
    :func:`revolution_tracker_update` delta test.

    ``rotation_axis`` is expected to be unit-length and colinear with
    the hinge's rotation axis in the same coordinate frame as ``q``
    (both world-space, per the PhoenX convention).
    """
    if q[3] == 0.0:
        return PI
    xyz = wp.vec3f(q[0], q[1], q[2])
    return 2.0 * wp.atan(wp.dot(xyz, rotation_axis) / q[3])


@wp.func
def revolution_tracker_update(
    new_quaternion_angle: wp.float32,
    revolution_counter: wp.int32,
    previous_quaternion_angle: wp.float32,
):
    """Absorb a new in-branch angle into the cumulative revolution count.

    Port of PhoenX's ``FullRevolutionTracker.Update``. Both angles are
    in ``(-pi, pi]``; a per-substep |delta| > pi must be a wrap, so:

    * ``delta > +pi``  -> wrapped from ``~-pi`` up past ``+pi``
      (rotated backwards): counter ``- 1``.
    * ``delta < -pi``  -> wrapped from ``~+pi`` down past ``-pi``
      (rotated forwards past a full turn): counter ``+ 1``.

    Returns ``(new_counter, new_prev_angle)``; the caller writes
    both back before the next prepare. ``new_prev_angle`` is just
    ``new_quaternion_angle`` (raw delta is never stored).
    """
    delta = new_quaternion_angle - previous_quaternion_angle
    new_counter = revolution_counter
    if delta > PI:
        new_counter = revolution_counter - 1
    elif delta < -PI:
        new_counter = revolution_counter + 1
    return new_counter, new_quaternion_angle


# ---------------------------------------------------------------------------
# Velocity / impulse helpers shared by every PGS row solver.
# ---------------------------------------------------------------------------


@wp.func
def apply_pair_velocity_impulse(
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1_world: wp.mat33f,
    inv_inertia2_world: wp.mat33f,
    r1: wp.vec3f,
    r2: wp.vec3f,
    imp: wp.vec3f,
):
    """Antisymmetric body-pair velocity update for a point-applied impulse.

    The impulse ``imp`` acts from body 1 onto body 2 at world-frame
    lever arms ``r1`` (in body 1) / ``r2`` (in body 2). Returns the
    updated ``(v1, v2, w1, w2)`` tuple. Inlined into PGS iterate /
    warm-start scatter / position-iterate paths.
    """
    v1_new = v1 - inv_mass1 * imp
    v2_new = v2 + inv_mass2 * imp
    w1_new = w1 - inv_inertia1_world @ wp.cross(r1, imp)
    w2_new = w2 + inv_inertia2_world @ wp.cross(r2, imp)
    return v1_new, v2_new, w1_new, w2_new


@wp.func
def effective_mass_scalar(
    axis: wp.vec3f,
    r1: wp.vec3f,
    r2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1_world: wp.mat33f,
    inv_inertia2_world: wp.mat33f,
) -> wp.float32:
    """Scalar ``1 / (J M^-1 J^T)`` for a point-axis row.

    ``J = [-axis, axis, -cross(r1, axis)^T, cross(r2, axis)^T]``; the
    quadratic form ``J M^-1 J^T`` reduces to
    ``inv_m1 + inv_m2 + dot(rc1, I1 @ rc1) + dot(rc2, I2 @ rc2)``.
    Returns 0 when the denominator collapses (both bodies static or
    kinematically pinned) so callers can short-circuit without a
    divide-by-zero.
    """
    rc1 = wp.cross(r1, axis)
    rc2 = wp.cross(r2, axis)
    w = inv_mass1 + inv_mass2 + wp.dot(rc1, inv_inertia1_world @ rc1) + wp.dot(rc2, inv_inertia2_world @ rc2)
    if w > 1.0e-12:
        return 1.0 / w
    return 0.0


@wp.func
def rotate_inertia(rotation: wp.mat33f, inertia_body: wp.mat33f) -> wp.mat33f:
    """Conjugation ``R * I_body * R^T`` for inertia / inverse-inertia.

    Used wherever the world-frame inertia is refreshed from a body's
    current orientation: PhoenX's once-per-step
    ``_phoenx_update_inertia_kernel`` and the example helpers that
    seed the body container. Plumbing this through one helper keeps
    the math identical across callers and gives us a single place to
    drop in a ``Mat3Sym``-aware path later (Phase C).
    """
    return rotation * inertia_body * wp.transpose(rotation)


@wp.func
def revolution_tracker_angle(
    revolution_counter: wp.int32,
    previous_quaternion_angle: wp.float32,
) -> wp.float32:
    """Assemble the unbounded cumulative angle from the tracker state.

    Direct port of PhoenX's ``FullRevolutionTracker.GetAngle``
    (RevoluteJoint.cuh:54): ``2*pi * counter + previous``.

    Intended to be called immediately after
    :func:`revolution_tracker_update` so ``previous_quaternion_angle``
    already holds the latest in-branch angle. The output has no
    wrap-around -- comparisons against ``min_value`` / ``max_value``
    are straightforward even for limits that span multiple full
    turns.
    """
    return TWO_PI * wp.float32(revolution_counter) + previous_quaternion_angle
