# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def fr_operators(order: int, correction: str = "g2") -> tuple[np.ndarray, ...]:
    """Build the KPM-FR one-dimensional operators."""
    if order not in (3, 4, 5, 6):
        raise ValueError(f"order must be 3, 4, 5, or 6, got {order}")
    if correction not in ("g2", "gdg"):
        raise ValueError(f"unknown correction {correction}")

    legendre = np.polynomial.legendre.Legendre
    poly = np.polynomial.Polynomial
    inner = np.real_if_close(legendre.basis(order - 1).deriv().roots())
    points = np.concatenate(([-1.0], inner, [1.0]))

    bary = np.ones(order)
    for j in range(order):
        bary[j] = 1.0 / np.prod(points[j] - np.delete(points, j))

    derivative = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            derivative[i, j] = 0.0 if i == j else bary[j] / (bary[i] * (points[i] - points[j]))
        derivative[i, i] = -np.sum(derivative[i])

    restriction = np.zeros((2, order))
    restriction[0, 0] = 1.0
    restriction[1, -1] = 1.0

    def radau(degree: int) -> np.polynomial.Polynomial:
        p = legendre.basis(degree).convert(kind=poly)
        q = legendre.basis(degree - 1).convert(kind=poly)
        return 0.5 * (-1) ** degree * (p - q)

    g = radau(order)
    if correction == "g2":
        g = (order - 1) / (2 * order - 1) * g + order / (2 * order - 1) * radau(order - 1)
    dg = g.deriv()
    correction_matrix = np.column_stack((dg(points), -dg(-points)))
    cr = correction_matrix @ restriction
    return tuple(x.astype(np.float32) for x in (points, derivative, correction_matrix, cr, derivative - cr))
