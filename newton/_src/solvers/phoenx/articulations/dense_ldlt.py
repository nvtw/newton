# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Small dense LDLT factorization used to validate articulation systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def regularize_diagonal(matrix: np.ndarray, diagonal_scale: float, *, min_diagonal: float = 1.0) -> np.ndarray:
    """Return ``matrix`` with scaled positive diagonal regularization.

    Args:
        matrix: Square matrix to regularize.
        diagonal_scale: Scale applied to ``max(abs(A_ii), min_diagonal)``.
        min_diagonal: Fallback diagonal magnitude for zero rows.

    Returns:
        A copy with regularization added to the diagonal.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got shape {a.shape}")
    out = a.copy()
    if out.size == 0:
        return out
    scale = float(diagonal_scale)
    if scale < 0.0:
        raise ValueError(f"diagonal_scale must be non-negative, got {scale}")
    diag = np.diag(out)
    reg = scale * np.maximum(np.abs(diag), float(min_diagonal))
    out[np.diag_indices_from(out)] += reg
    return out


@dataclass(frozen=True)
class DenseLDLTFactorization:
    """Dense ``A = L D L.T`` factorization with unit diagonal ``L``."""

    l: np.ndarray
    d: np.ndarray

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the factored system for one or more RHS columns."""
        b = np.asarray(rhs, dtype=np.float64)
        squeeze = b.ndim == 1
        if squeeze:
            b = b[:, None]
        if b.shape[0] != self.d.size:
            raise ValueError(f"rhs has incompatible row count {b.shape[0]}, expected {self.d.size}")

        n = self.d.size
        y = b.copy()
        for i in range(n):
            if i > 0:
                y[i] -= self.l[i, :i] @ y[:i]

        z = y / self.d[:, None]
        x = z.copy()
        for i in range(n - 1, -1, -1):
            if i + 1 < n:
                x[i] -= self.l[i + 1 :, i] @ x[i + 1 :]

        return x[:, 0] if squeeze else x


def factorize_ldlt(matrix: np.ndarray, *, min_pivot: float = 1.0e-12) -> DenseLDLTFactorization:
    """Factorize a symmetric positive definite dense matrix with LDLT.

    Args:
        matrix: Square SPD matrix.
        min_pivot: Minimum accepted diagonal pivot.

    Returns:
        Dense LDLT factorization.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got shape {a.shape}")

    n = int(a.shape[0])
    l = np.eye(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    pivot_floor = float(min_pivot)

    for i in range(n):
        for j in range(i):
            accum = 0.0
            if j > 0:
                accum = float(np.dot(l[i, :j] * d[:j], l[j, :j]))
            if abs(d[j]) <= pivot_floor:
                raise np.linalg.LinAlgError(f"zero LDLT pivot at row {j}: {d[j]}")
            l[i, j] = (a[i, j] - accum) / d[j]

        diag_accum = 0.0
        if i > 0:
            diag_accum = float(np.dot(l[i, :i] * l[i, :i], d[:i]))
        d[i] = a[i, i] - diag_accum
        if d[i] <= pivot_floor:
            raise np.linalg.LinAlgError(f"non-positive LDLT pivot at row {i}: {d[i]}")

    return DenseLDLTFactorization(l=l, d=d)
