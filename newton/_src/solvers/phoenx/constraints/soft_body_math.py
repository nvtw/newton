# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp math for PhoenX soft-body constraints."""

from __future__ import annotations

import warp as wp

__all__ = [
    "neohookean_constraints_from_F",
    "tet_chain_rule_gradients",
    "tet_deformation_gradient",
]


@wp.func
def tet_deformation_gradient(
    x_a: wp.vec3f,
    x_b: wp.vec3f,
    x_c: wp.vec3f,
    x_d: wp.vec3f,
    inv_rest: wp.mat33f,
) -> wp.mat33f:
    """Deformation gradient ``F = D * inv_rest`` for one tetrahedron."""
    e_ab = x_b - x_a
    e_ac = x_c - x_a
    e_ad = x_d - x_a
    f00 = inv_rest[0, 0] * e_ab[0] + inv_rest[1, 0] * e_ac[0] + inv_rest[2, 0] * e_ad[0]
    f01 = inv_rest[0, 1] * e_ab[0] + inv_rest[1, 1] * e_ac[0] + inv_rest[2, 1] * e_ad[0]
    f02 = inv_rest[0, 2] * e_ab[0] + inv_rest[1, 2] * e_ac[0] + inv_rest[2, 2] * e_ad[0]
    f10 = inv_rest[0, 0] * e_ab[1] + inv_rest[1, 0] * e_ac[1] + inv_rest[2, 0] * e_ad[1]
    f11 = inv_rest[0, 1] * e_ab[1] + inv_rest[1, 1] * e_ac[1] + inv_rest[2, 1] * e_ad[1]
    f12 = inv_rest[0, 2] * e_ab[1] + inv_rest[1, 2] * e_ac[1] + inv_rest[2, 2] * e_ad[1]
    f20 = inv_rest[0, 0] * e_ab[2] + inv_rest[1, 0] * e_ac[2] + inv_rest[2, 0] * e_ad[2]
    f21 = inv_rest[0, 1] * e_ab[2] + inv_rest[1, 1] * e_ac[2] + inv_rest[2, 1] * e_ad[2]
    f22 = inv_rest[0, 2] * e_ab[2] + inv_rest[1, 2] * e_ac[2] + inv_rest[2, 2] * e_ad[2]
    return wp.mat33f(f00, f01, f02, f10, f11, f12, f20, f21, f22)


@wp.func
def tet_chain_rule_gradients(
    dCdF: wp.mat33f,
    inv_rest: wp.mat33f,
):
    """Return ``(g_a, g_b, g_c, g_d)`` from a tetrahedral ``dC/dF``."""
    inv_row0 = wp.vec3f(inv_rest[0, 0], inv_rest[0, 1], inv_rest[0, 2])
    inv_row1 = wp.vec3f(inv_rest[1, 0], inv_rest[1, 1], inv_rest[1, 2])
    inv_row2 = wp.vec3f(inv_rest[2, 0], inv_rest[2, 1], inv_rest[2, 2])
    g_b = dCdF * inv_row0
    g_c = dCdF * inv_row1
    g_d = dCdF * inv_row2
    g_a = -(g_b + g_c + g_d)
    return g_a, g_b, g_c, g_d


@wp.func
def neohookean_constraints_from_F(
    F: wp.mat33f,
    gamma_offset: wp.float32,
    dev_eps: wp.float32,
):
    """Return block Neo-Hookean constraints and ``dC/dF`` matrices."""
    f0 = wp.vec3f(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3f(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3f(F[0, 2], F[1, 2], F[2, 2])

    cof0 = wp.cross(f1, f2)
    cof1 = wp.cross(f2, f0)
    cof2 = wp.cross(f0, f1)
    c_h = wp.dot(f0, cof0) - gamma_offset
    dCH_dF = wp.mat33f(
        cof0[0],
        cof1[0],
        cof2[0],
        cof0[1],
        cof1[1],
        cof2[1],
        cof0[2],
        cof1[2],
        cof2[2],
    )

    i_c = (
        f0[0] * f0[0]
        + f0[1] * f0[1]
        + f0[2] * f0[2]
        + f1[0] * f1[0]
        + f1[1] * f1[1]
        + f1[2] * f1[2]
        + f2[0] * f2[0]
        + f2[1] * f2[1]
        + f2[2] * f2[2]
    )
    c_d = wp.sqrt(i_c + dev_eps)
    dCD_dF = F * (wp.float32(1.0) / c_d)
    return c_h, c_d, dCH_dF, dCD_dF
