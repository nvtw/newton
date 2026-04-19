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

import warp as wp

__all__ = [
    "create_orthonormal",
    "qmatrix_project_multiply_left_right",
]


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
