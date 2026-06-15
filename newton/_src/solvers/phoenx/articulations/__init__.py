# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Full-coordinate articulation support for PhoenX.

This package contains the topology, symbolic factorization, and matrix
assembly pieces used by the PhoenX DVI articulation path. The current PGS
joint solver remains the production path while these pieces are integrated and
validated incrementally.
"""

from .block_sparse_ldlt import BlockSparseLDLTFactorization, factorize_block_sparse_ldlt
from .dense_ldlt import DenseLDLTFactorization, factorize_ldlt, regularize_diagonal
from .device import ArticulationDeviceSystem
from .joint_rows import (
    JointRowBlock,
    ball_socket_rows,
    cylindrical_rows,
    fixed_rows,
    orthonormal_pair,
    prismatic_rows,
    revolute_rows,
)
from .symbolic import BlockSparseSymbolic, ConstraintGraph, compute_block_sparse_symbolic
from .system import PrefactorizedArticulationSystem
from .topology import (
    ArticulationTopology,
    d6_constraint_row_count,
    joint_axial_row_count,
    joint_constraint_row_count,
    joint_drive_row_count,
)

__all__ = [
    "ArticulationDeviceSystem",
    "ArticulationTopology",
    "BlockSparseLDLTFactorization",
    "BlockSparseSymbolic",
    "ConstraintGraph",
    "DenseLDLTFactorization",
    "JointRowBlock",
    "PrefactorizedArticulationSystem",
    "ball_socket_rows",
    "compute_block_sparse_symbolic",
    "cylindrical_rows",
    "d6_constraint_row_count",
    "factorize_block_sparse_ldlt",
    "factorize_ldlt",
    "fixed_rows",
    "joint_axial_row_count",
    "joint_constraint_row_count",
    "joint_drive_row_count",
    "orthonormal_pair",
    "prismatic_rows",
    "regularize_diagonal",
    "revolute_rows",
]
