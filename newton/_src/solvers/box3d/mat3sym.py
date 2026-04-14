# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Symmetric 3x3 matrix type for inertia tensors.

A symmetric 3x3 matrix has only 6 unique values instead of 9.
Storing as ``mat3sym`` saves 33% memory bandwidth on reads/writes,
which matters for the contact/joint solve kernels that read inertia
for every constraint.

Layout: ``(m00, m01, m02, m11, m12, m22)`` — upper triangle, row-major.

    [[m00, m01, m02],
     [m01, m11, m12],
     [m02, m12, m22]]
"""

from __future__ import annotations

import warp as wp


@wp.struct
class mat3sym:
    """Symmetric 3x3 matrix stored as 6 floats (upper triangle)."""

    m00: float
    m01: float
    m02: float
    m11: float
    m12: float
    m22: float


@wp.func
def mat3sym_from_mat33(m: wp.mat33) -> mat3sym:
    """Convert a full mat33 to symmetric (takes upper triangle)."""
    s = mat3sym()
    s.m00 = m[0, 0]
    s.m01 = m[0, 1]
    s.m02 = m[0, 2]
    s.m11 = m[1, 1]
    s.m12 = m[1, 2]
    s.m22 = m[2, 2]
    return s


@wp.func
def mat3sym_to_mat33(s: mat3sym) -> wp.mat33:
    """Convert symmetric to full mat33."""
    return wp.mat33(
        s.m00, s.m01, s.m02,
        s.m01, s.m11, s.m12,
        s.m02, s.m12, s.m22,
    )


@wp.func
def mat3sym_mul_vec(s: mat3sym, v: wp.vec3) -> wp.vec3:
    """Multiply symmetric matrix by vector: s @ v."""
    return wp.vec3(
        s.m00 * v[0] + s.m01 * v[1] + s.m02 * v[2],
        s.m01 * v[0] + s.m11 * v[1] + s.m12 * v[2],
        s.m02 * v[0] + s.m12 * v[1] + s.m22 * v[2],
    )


@wp.func
def mat3sym_add(a: mat3sym, b: mat3sym) -> mat3sym:
    """Add two symmetric matrices."""
    s = mat3sym()
    s.m00 = a.m00 + b.m00
    s.m01 = a.m01 + b.m01
    s.m02 = a.m02 + b.m02
    s.m11 = a.m11 + b.m11
    s.m12 = a.m12 + b.m12
    s.m22 = a.m22 + b.m22
    return s


@wp.func
def mat3sym_zero() -> mat3sym:
    """Return the zero symmetric matrix."""
    s = mat3sym()
    s.m00 = 0.0
    s.m01 = 0.0
    s.m02 = 0.0
    s.m11 = 0.0
    s.m12 = 0.0
    s.m22 = 0.0
    return s


@wp.func
def mat3sym_rotate(R: wp.mat33, s: mat3sym) -> mat3sym:
    """Rotate a symmetric matrix: R @ S @ R^T (result is symmetric)."""
    # Expand R @ S @ R^T using the 6 unique elements
    # This is more efficient than converting to mat33 and back
    M = mat3sym_to_mat33(s)
    result = R * M * wp.transpose(R)
    return mat3sym_from_mat33(result)


@wp.func
def mat3sym_det(s: mat3sym) -> float:
    """Determinant of a symmetric 3x3 matrix."""
    return (s.m00 * (s.m11 * s.m22 - s.m12 * s.m12)
            - s.m01 * (s.m01 * s.m22 - s.m12 * s.m02)
            + s.m02 * (s.m01 * s.m12 - s.m11 * s.m02))


@wp.func
def mat3sym_solve(s: mat3sym, rhs: wp.vec3) -> wp.vec3:
    """Solve s @ x = rhs via Cramer's rule for a symmetric matrix."""
    det = mat3sym_det(s)
    if wp.abs(det) < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    inv_det = 1.0 / det
    # Cofactors (symmetric adjugate)
    i00 = (s.m11 * s.m22 - s.m12 * s.m12) * inv_det
    i01 = (s.m02 * s.m12 - s.m01 * s.m22) * inv_det
    i02 = (s.m01 * s.m12 - s.m02 * s.m11) * inv_det
    i11 = (s.m00 * s.m22 - s.m02 * s.m02) * inv_det
    i12 = (s.m01 * s.m02 - s.m00 * s.m12) * inv_det
    i22 = (s.m00 * s.m11 - s.m01 * s.m01) * inv_det
    return wp.vec3(
        i00 * rhs[0] + i01 * rhs[1] + i02 * rhs[2],
        i01 * rhs[0] + i11 * rhs[1] + i12 * rhs[2],
        i02 * rhs[0] + i12 * rhs[1] + i22 * rhs[2],
    )
