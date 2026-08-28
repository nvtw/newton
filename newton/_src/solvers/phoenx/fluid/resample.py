# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def interpolation_matrix(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Evaluate a nodal Lagrange basis at sample positions."""
    points = np.asarray(points)
    weights = np.array([1.0 / np.prod(points[j] - np.delete(points, j)) for j in range(len(points))])
    matrix = np.empty((len(samples), len(points)))
    for i, sample in enumerate(samples):
        exact = np.flatnonzero(np.isclose(sample, points))
        if len(exact):
            matrix[i].fill(0.0)
            matrix[i, exact[0]] = 1.0
        else:
            row = weights / (sample - points)
            matrix[i] = row / np.sum(row)
    return matrix.astype(np.float32)


def resample_state_3d(state: np.ndarray, resolution: tuple[int, int, int], points: np.ndarray) -> np.ndarray:
    """Resample element-local LGL fields to uniform subcell centers."""
    nx, ny, nz = resolution
    order = len(points)
    samples = 2.0 * (np.arange(order) + 0.5) / order - 1.0
    matrix = interpolation_matrix(points, samples)
    local = state.reshape(state.shape[0], nx, ny, nz, order, order, order)
    uniform = np.einsum("az,by,cx,vijkzyx->vijkabc", matrix, matrix, matrix, local, optimize=True)
    return uniform.transpose(1, 6, 2, 5, 3, 4, 0).reshape(nx * order, ny * order, nz * order, state.shape[0])
