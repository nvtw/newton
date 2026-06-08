# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Math helpers shared across the constraint kernels (ports from Jitter2 /
PhoenX). Each ``@wp.func`` inlines into callers."""

from __future__ import annotations

import math

import warp as wp

__all__ = [
    "apply_pair_spatial_impulse",
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
    """Unit vector orthogonal to ``v``. Picks the axis with the smallest
    absolute component to avoid near-degenerate perpendiculars."""
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
    """3x3 projection of ``L_left * L_right^T`` (quaternion-multiplication
    matrices). Maps angular velocity to quaternion-error in angular constraints."""
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


# Full-revolution angle tracker. Unwraps quaternion's (-pi, pi] branch into an
# unbounded cumulative angle so hinge limits can span multiple turns. Caller
# owns two persistent scalars per joint: revolution_counter, previous_quaternion_angle.


@wp.func
def extract_rotation_angle(q: wp.quatf, rotation_axis: wp.vec3f) -> wp.float32:
    """Signed angle of ``q`` about unit ``rotation_axis``, in ``(-pi, pi]``.

    ``theta = 2 * atan(dot(q.xyz, axis) / q.w)``; the ``q.w == 0`` branch
    pins output to +pi (atan2 would discontinuously return +/-pi/2 there).
    The atan(x/y) form matches the C# port; atan2 would wrap differently
    and break :func:`revolution_tracker_update`.
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
    """Absorb new in-branch angle into the cumulative revolution count.
    delta > +pi -> wrapped backwards (counter - 1); delta < -pi -> wrapped
    forwards (counter + 1). Returns (new_counter, new_prev_angle)."""
    delta = new_quaternion_angle - previous_quaternion_angle
    new_counter = revolution_counter
    if delta > PI:
        new_counter = revolution_counter - 1
    elif delta < -PI:
        new_counter = revolution_counter + 1
    return new_counter, new_quaternion_angle


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
    ``imp`` acts from body 1 onto body 2 at world-frame lever arms r1, r2.
    Returns updated (v1, v2, w1, w2)."""
    v1_new = v1 - inv_mass1 * imp
    v2_new = v2 + inv_mass2 * imp
    w1_new = w1 - inv_inertia1_world @ wp.cross(r1, imp)
    w2_new = w2 + inv_inertia2_world @ wp.cross(r2, imp)
    return v1_new, v2_new, w1_new, w2_new


@wp.func
def apply_pair_spatial_impulse(
    v1: wp.vec3f,
    v2: wp.vec3f,
    w1: wp.vec3f,
    w2: wp.vec3f,
    inv_mass1: wp.float32,
    inv_mass2: wp.float32,
    inv_inertia1_world: wp.mat33f,
    inv_inertia2_world: wp.mat33f,
    linear_impulse: wp.vec3f,
    angular_impulse1: wp.vec3f,
    angular_impulse2: wp.vec3f,
):
    """Antisymmetric body-pair update with precomputed angular impulses.

    The linear impulse acts from body 1 onto body 2. The two angular
    impulses are world-space r-cross-impulse terms, or pure couples,
    with matching signs before multiplying by inverse inertia.
    """
    v1_new = v1 - inv_mass1 * linear_impulse
    v2_new = v2 + inv_mass2 * linear_impulse
    w1_new = w1 - inv_inertia1_world @ angular_impulse1
    w2_new = w2 + inv_inertia2_world @ angular_impulse2
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
    """Scalar ``1 / (J M^-1 J^T)`` for a point-axis row. Returns 0 when the
    denominator collapses (both bodies static/pinned)."""
    rc1 = wp.cross(r1, axis)
    rc2 = wp.cross(r2, axis)
    w = inv_mass1 + inv_mass2 + wp.dot(rc1, inv_inertia1_world @ rc1) + wp.dot(rc2, inv_inertia2_world @ rc2)
    if w > 1.0e-12:
        return 1.0 / w
    return 0.0


@wp.func
def rotate_inertia(rotation: wp.mat33f, inertia_body: wp.mat33f) -> wp.mat33f:
    """Conjugation ``R * I_body * R^T``."""
    return rotation * inertia_body * wp.transpose(rotation)


@wp.func
def revolution_tracker_angle(
    revolution_counter: wp.int32,
    previous_quaternion_angle: wp.float32,
) -> wp.float32:
    """Cumulative unbounded angle: 2*pi * counter + previous_quaternion_angle.
    Call after :func:`revolution_tracker_update`."""
    return TWO_PI * wp.float32(revolution_counter) + previous_quaternion_angle
