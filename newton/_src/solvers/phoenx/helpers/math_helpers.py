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
    "create_orthonormal",
    "qmatrix_project_multiply_left_right",
    "extract_rotation_angle",
    "revolution_tracker_update",
    "revolution_tracker_angle",
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
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22,
    )


# ---------------------------------------------------------------------------
# Full-revolution angle tracker
#
# Ported from PhoenX (Jolt) ``FullRevolutionTracker`` (RevoluteJoint.cuh:27)
# plus ``quat::ExtractRotationAngle`` (MiniMath/quat.cuh:703). The tracker
# unwraps the quaternion's [-pi, pi] branch into an unbounded cumulative
# angle so hinge-style limits can be specified with arbitrarily large
# ``min_value`` / ``max_value`` (e.g. the "wide-open" idiom
# ``min = -1e2`` for a one-sided clamp) without the ``sin(theta/2)``
# wrap-around that bites the half-angle projection at theta > pi.
#
# The tracker is stateless at this layer: the caller owns two persistent
# scalar fields per joint instance,
#
#     revolution_counter:       int32  (initially 0)
#     previous_quaternion_angle: float32  (initially 0)
#
# and on every ``prepare`` step reads them, calls
# :func:`revolution_tracker_update` to produce the new ``(counter',
# previous')`` pair, writes them back, then reads the total angle via
# :func:`revolution_tracker_angle`. No allocation, no host/device sync,
# no per-joint book-keeping outside what the constraint already stores.
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

    Direct port of PhoenX's ``FullRevolutionTracker.Update``
    (RevoluteJoint.cuh:40). Given the previous and current values of
    ``extract_rotation_angle`` (both in ``(-pi, pi]``), detect a
    branch-crossing and update the revolution counter so the cumulative
    angle is continuous.

    Sign convention: the raw quaternion angle is limited to one full
    principal branch. A step's true angular velocity can only move the
    angle by a fraction of a turn per substep (assuming sane timesteps
    and no rigid-body teleport), so if the in-branch delta exceeds
    ``+/- pi`` it has to be a wrap:

      * ``delta > +pi``  -> we wrapped from ``~-pi`` back up past
        ``+pi``; physically the joint rotated *backwards* a little, so
        the cumulative turn count decreases by one.
      * ``delta < -pi``  -> we wrapped from ``~+pi`` back down past
        ``-pi``; the joint rotated *forwards* past a full turn, so the
        count increases.

    Args:
        new_quaternion_angle: Latest in-branch angle from
            :func:`extract_rotation_angle`, in radians, in
            ``(-pi, pi]``.
        revolution_counter: Persistent counter read from the
            constraint's scratch storage.
        previous_quaternion_angle: Persistent previous in-branch
            angle read from the constraint's scratch storage.

    Returns:
        ``(new_revolution_counter, new_previous_quaternion_angle)``
        -- the two scalars the caller must write back before the
        next prepare step. The new previous angle is simply the
        input ``new_quaternion_angle`` (the caller never stores the
        raw delta).
    """
    delta = new_quaternion_angle - previous_quaternion_angle
    new_counter = revolution_counter
    if delta > PI:
        new_counter = revolution_counter - 1
    elif delta < -PI:
        new_counter = revolution_counter + 1
    return new_counter, new_quaternion_angle


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
