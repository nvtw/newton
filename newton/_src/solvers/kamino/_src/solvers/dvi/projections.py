# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact projection functions for dense and sparse DVI kernels."""

import warp as wp

from ...core.math import FLOAT32_EPS

float32 = wp.float32
vec3f = wp.vec3f


@wp.func
def project_contact_split_update(
    lambda_old: vec3f,
    v_c: vec3f,
    D_diag: vec3f,
    regularization: float32,
    omega: float32,
    mu: float32,
) -> vec3f:
    """Solve Signorini normal contact, then project friction onto its Coulomb disk.

    The split keeps the normal load out of the tangential effective-mass step,
    while the final disk projection enforces ``norm(lambda_t) <= mu * lambda_n``.
    """
    lambda_n = lambda_old.z
    if D_diag.z > FLOAT32_EPS:
        lambda_n = wp.max(float32(0.0), lambda_old.z - omega * v_c.z / (D_diag.z + regularization))

    lambda_t = vec3f(lambda_old.x, lambda_old.y, float32(0.0))
    if D_diag.x > FLOAT32_EPS:
        lambda_t.x = lambda_old.x - omega * v_c.x / (D_diag.x + regularization)
    if D_diag.y > FLOAT32_EPS:
        lambda_t.y = lambda_old.y - omega * v_c.y / (D_diag.y + regularization)
    lambda_t_norm = wp.sqrt(lambda_t.x * lambda_t.x + lambda_t.y * lambda_t.y)
    lambda_t_max = mu * lambda_n
    if lambda_t_norm > lambda_t_max and lambda_t_norm > FLOAT32_EPS:
        lambda_t *= lambda_t_max / lambda_t_norm
    return vec3f(lambda_t.x, lambda_t.y, lambda_n)
