# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Host block-sparse LDLT factorization for articulation validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dense_ldlt import DenseLDLTFactorization, factorize_ldlt
from .symbolic import BlockSparseSymbolic


@dataclass(frozen=True)
class BlockSparseLDLTFactorization:
    """Block ``A = L D L.T`` factorization in symbolic pivot order."""

    symbolic: BlockSparseSymbolic
    d_blocks: tuple[np.ndarray, ...]
    d_factors: tuple[DenseLDLTFactorization, ...]
    l_blocks: dict[tuple[int, int], np.ndarray]
    original_offsets: np.ndarray
    pivot_offsets: np.ndarray

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the factored system and return rows in original block order."""
        b_original = np.asarray(rhs, dtype=np.float64)
        squeeze = b_original.ndim == 1
        if squeeze:
            b_original = b_original[:, None]
        if b_original.shape[0] != int(self.original_offsets[-1]):
            raise ValueError(f"rhs has {b_original.shape[0]} rows, expected {int(self.original_offsets[-1])}")

        b = _permute_vector_to_pivot(b_original, self.symbolic, self.original_offsets, self.pivot_offsets)
        n = self.symbolic.num_blocks

        y = np.zeros_like(b)
        for i in range(n):
            islice = _block_slice(self.pivot_offsets, i)
            value = b[islice].copy()
            for ridx in range(int(self.symbolic.l_row_ptr[i]), int(self.symbolic.l_row_ptr[i + 1])):
                k = int(self.symbolic.l_col_idx[ridx])
                kslice = _block_slice(self.pivot_offsets, k)
                value -= self.l_blocks[(i, k)] @ y[kslice]
            y[islice] = value

        z = np.zeros_like(y)
        for i in range(n):
            islice = _block_slice(self.pivot_offsets, i)
            z[islice] = self.d_factors[i].solve(y[islice])

        x = np.zeros_like(z)
        for i in range(n - 1, -1, -1):
            islice = _block_slice(self.pivot_offsets, i)
            value = z[islice].copy()
            for ptr in range(int(self.symbolic.l_col_ptr[i]), int(self.symbolic.l_col_ptr[i + 1])):
                row = int(self.symbolic.l_row_idx[ptr])
                rslice = _block_slice(self.pivot_offsets, row)
                value -= self.l_blocks[(row, i)].T @ x[rslice]
            x[islice] = value

        out = _permute_vector_to_original(x, self.symbolic, self.original_offsets, self.pivot_offsets)
        return out[:, 0] if squeeze else out


def factorize_block_sparse_ldlt(
    matrix: np.ndarray,
    symbolic: BlockSparseSymbolic,
) -> BlockSparseLDLTFactorization:
    """Factorize ``matrix`` with the supplied block-sparse symbolic pattern.

    Args:
        matrix: Dense matrix in original active-block row order.
        symbolic: Topology-only block symbolic data.

    Returns:
        Host block-sparse LDLT factorization.
    """
    a_original = np.asarray(matrix, dtype=np.float64)
    if a_original.ndim != 2 or a_original.shape[0] != a_original.shape[1]:
        raise ValueError(f"matrix must be square, got {a_original.shape}")

    original_offsets = _original_offsets(symbolic)
    expected_rows = int(original_offsets[-1])
    if a_original.shape != (expected_rows, expected_rows):
        raise ValueError(f"matrix has shape {a_original.shape}, expected {(expected_rows, expected_rows)}")

    pivot_offsets = np.zeros(symbolic.num_blocks + 1, dtype=np.int32)
    np.cumsum(symbolic.block_sizes, out=pivot_offsets[1:])
    a = _permute_matrix_to_pivot(a_original, symbolic, original_offsets, pivot_offsets)

    n = symbolic.num_blocks
    d_blocks: list[np.ndarray] = []
    d_factors: list[DenseLDLTFactorization] = []
    l_blocks: dict[tuple[int, int], np.ndarray] = {}

    for j in range(n):
        jslice = _block_slice(pivot_offsets, j)
        diag = a[jslice, jslice].copy()
        for ridx in range(int(symbolic.l_row_ptr[j]), int(symbolic.l_row_ptr[j + 1])):
            k = int(symbolic.l_col_idx[ridx])
            ljk = l_blocks[(j, k)]
            diag -= ljk @ d_blocks[k] @ ljk.T

        d_factor = factorize_ldlt(diag)
        d_blocks.append(diag)
        d_factors.append(d_factor)

        for ptr in range(int(symbolic.l_col_ptr[j]), int(symbolic.l_col_ptr[j + 1])):
            i = int(symbolic.l_row_idx[ptr])
            islice = _block_slice(pivot_offsets, i)
            block = a[islice, jslice].copy()
            row_i_cols = _row_predecessors(symbolic, i)
            row_j_cols = _row_predecessors(symbolic, j)
            for k in sorted(set(row_i_cols).intersection(row_j_cols)):
                l_ik = l_blocks[(i, k)]
                l_jk = l_blocks[(j, k)]
                block -= l_ik @ d_blocks[k] @ l_jk.T
            # L_ij D_j = block, so L_ij = block D_j^-1.
            l_blocks[(i, j)] = d_factor.solve(block.T).T

    return BlockSparseLDLTFactorization(
        symbolic=symbolic,
        d_blocks=tuple(d_blocks),
        d_factors=tuple(d_factors),
        l_blocks=l_blocks,
        original_offsets=original_offsets,
        pivot_offsets=pivot_offsets,
    )


def _row_predecessors(symbolic: BlockSparseSymbolic, row: int) -> set[int]:
    return {
        int(symbolic.l_col_idx[ridx]) for ridx in range(int(symbolic.l_row_ptr[row]), int(symbolic.l_row_ptr[row + 1]))
    }


def _original_offsets(symbolic: BlockSparseSymbolic) -> np.ndarray:
    block_count = int(symbolic.inv_pivot_order.size)
    sizes = np.zeros(block_count, dtype=np.int32)
    for pivot, original in enumerate(symbolic.pivot_order):
        sizes[int(original)] = int(symbolic.block_sizes[pivot])
    offsets = np.zeros(block_count + 1, dtype=np.int32)
    np.cumsum(sizes, out=offsets[1:])
    return offsets


def _block_slice(offsets: np.ndarray, block: int) -> slice:
    return slice(int(offsets[block]), int(offsets[block + 1]))


def _permute_matrix_to_pivot(
    matrix: np.ndarray,
    symbolic: BlockSparseSymbolic,
    original_offsets: np.ndarray,
    pivot_offsets: np.ndarray,
) -> np.ndarray:
    total = int(pivot_offsets[-1])
    out = np.zeros((total, total), dtype=np.float64)
    for pi, oi in enumerate(symbolic.pivot_order):
        src_i = _block_slice(original_offsets, int(oi))
        dst_i = _block_slice(pivot_offsets, pi)
        for pj, oj in enumerate(symbolic.pivot_order):
            src_j = _block_slice(original_offsets, int(oj))
            dst_j = _block_slice(pivot_offsets, pj)
            out[dst_i, dst_j] = matrix[src_i, src_j]
    return out


def _permute_vector_to_pivot(
    vector: np.ndarray,
    symbolic: BlockSparseSymbolic,
    original_offsets: np.ndarray,
    pivot_offsets: np.ndarray,
) -> np.ndarray:
    out = np.zeros((int(pivot_offsets[-1]), vector.shape[1]), dtype=np.float64)
    for pivot, original in enumerate(symbolic.pivot_order):
        out[_block_slice(pivot_offsets, pivot)] = vector[_block_slice(original_offsets, int(original))]
    return out


def _permute_vector_to_original(
    vector: np.ndarray,
    symbolic: BlockSparseSymbolic,
    original_offsets: np.ndarray,
    pivot_offsets: np.ndarray,
) -> np.ndarray:
    out = np.zeros((int(original_offsets[-1]), vector.shape[1]), dtype=np.float64)
    for pivot, original in enumerate(symbolic.pivot_order):
        out[_block_slice(original_offsets, int(original))] = vector[_block_slice(pivot_offsets, pivot)]
    return out
