# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Host validation system for prefactorized PhoenX articulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block_sparse_ldlt import BlockSparseLDLTFactorization, factorize_block_sparse_ldlt
from .dense_ldlt import DenseLDLTFactorization, factorize_ldlt, regularize_diagonal
from .symbolic import BlockSparseSymbolic, compute_block_sparse_symbolic
from .topology import ArticulationTopology


@dataclass
class PrefactorizedArticulationSystem:
    """Prefactorized full-coordinate equality system for one topology.

    The symbolic block factorization is topology-only and can be reused across
    frames. The dense numeric factorization is a host-side validation path; the
    production numeric path will use the same topology and symbolic metadata in
    Warp kernels.
    """

    topology: ArticulationTopology
    symbolic: BlockSparseSymbolic
    diagonal_regularization: float = 1.0e-8
    dense_matrix: np.ndarray | None = None
    factorization: DenseLDLTFactorization | None = None
    block_sparse_factorization: BlockSparseLDLTFactorization | None = None

    @classmethod
    def from_topology(
        cls,
        topology: ArticulationTopology,
        *,
        use_meca: bool = True,
        diagonal_regularization: float = 1.0e-8,
    ) -> PrefactorizedArticulationSystem:
        """Create a system with topology-only symbolic data."""
        symbolic = compute_block_sparse_symbolic(
            topology.active_body1,
            topology.active_body2,
            topology.active_row_counts,
            use_meca=use_meca,
        )
        return cls(
            topology=topology,
            symbolic=symbolic,
            diagonal_regularization=float(diagonal_regularization),
        )

    def assemble_dense_matrix(
        self,
        jacobian: np.ndarray,
        inverse_mass: np.ndarray,
        inverse_inertia_world: np.ndarray,
    ) -> np.ndarray:
        """Assemble ``H = J W J.T`` for compact full-coordinate rows.

        Args:
            jacobian: Compact row Jacobian, shape ``[total_rows, 12]``.
                Columns ``0:6`` are body1 linear/angular terms; ``6:12`` are
                body2 linear/angular terms.
            inverse_mass: Inverse mass per body.
            inverse_inertia_world: World inverse inertia per body, shape
                ``[body_count, 3, 3]``.

        Returns:
            Dense unregularized holonomic matrix.
        """
        jac = np.asarray(jacobian, dtype=np.float64)
        if jac.shape != (self.topology.total_rows, 12):
            raise ValueError(f"jacobian must have shape {(self.topology.total_rows, 12)}, got {jac.shape}")

        inv_mass = np.asarray(inverse_mass, dtype=np.float64)
        inv_inertia = np.asarray(inverse_inertia_world, dtype=np.float64)
        if inv_inertia.ndim != 3 or inv_inertia.shape[1:] != (3, 3):
            raise ValueError(f"inverse_inertia_world must have shape [body_count, 3, 3], got {inv_inertia.shape}")

        total_rows = self.topology.total_rows
        h = np.zeros((total_rows, total_rows), dtype=np.float64)
        row_blocks = self.topology.row_to_active_block
        active_body1 = self.topology.active_body1
        active_body2 = self.topology.active_body2

        for r0 in range(total_rows):
            block0 = int(row_blocks[r0])
            bodies0 = ((int(active_body1[block0]), jac[r0, 0:6]), (int(active_body2[block0]), jac[r0, 6:12]))
            for r1 in range(r0, total_rows):
                block1 = int(row_blocks[r1])
                bodies1 = ((int(active_body1[block1]), jac[r1, 0:6]), (int(active_body2[block1]), jac[r1, 6:12]))
                value = 0.0
                for body0, j0 in bodies0:
                    if body0 < 0:
                        continue
                    for body1, j1 in bodies1:
                        if body0 == body1:
                            value += _body_metric_dot(body0, j0, j1, inv_mass, inv_inertia)
                h[r0, r1] = value
                h[r1, r0] = value

        return h

    def factorize_from_jacobian(
        self,
        jacobian: np.ndarray,
        inverse_mass: np.ndarray,
        inverse_inertia_world: np.ndarray,
    ) -> DenseLDLTFactorization:
        """Assemble and factorize the regularized dense validation system."""
        h = self.assemble_dense_matrix(jacobian, inverse_mass, inverse_inertia_world)
        h_reg = regularize_diagonal(h, self.diagonal_regularization)
        self.dense_matrix = h_reg
        self.factorization = factorize_ldlt(h_reg)
        self.block_sparse_factorization = factorize_block_sparse_ldlt(h_reg, self.symbolic)
        return self.factorization

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve with the most recent dense numeric factorization."""
        if self.factorization is None:
            raise RuntimeError("factorize_from_jacobian() must be called before solve()")
        return self.factorization.solve(rhs)

    def solve_block_sparse(self, rhs: np.ndarray) -> np.ndarray:
        """Solve with the most recent block-sparse validation factorization."""
        if self.block_sparse_factorization is None:
            raise RuntimeError("factorize_from_jacobian() must be called before solve_block_sparse()")
        return self.block_sparse_factorization.solve(rhs)


def _body_metric_dot(
    body: int,
    j0: np.ndarray,
    j1: np.ndarray,
    inverse_mass: np.ndarray,
    inverse_inertia_world: np.ndarray,
) -> float:
    lin0 = j0[:3]
    ang0 = j0[3:]
    lin1 = j1[:3]
    ang1 = j1[3:]
    value = float(inverse_mass[body]) * float(np.dot(lin0, lin1))
    value += float(ang0 @ inverse_inertia_world[body] @ ang1)
    return value
