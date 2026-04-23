# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kamino Linear Algebra Module"""

from . import utils
from .core import (
    DenseLinearOperatorData,
    DenseRectangularMultiLinearInfo,
    DenseSquareMultiLinearInfo,
)
from .linear import (
    ConjugateGradientSolver,
    ConjugateResidualSolver,
    DirectSolver,
    IterativeSolver,
    LinearSolver,
    LinearSolverNameToType,
    LinearSolverType,
    LinearSolverTypeToName,
    LLTBlockedSolver,
    LLTSequentialSolver,
)

# Import the RCM-reordered semi-sparse blocked LLT solver here (rather than
# from .linear) to avoid a circular import: .factorize.llt_blocked_nd_solver
# imports DirectSolver from .linear, so .linear cannot import it back.
# At this point .linear has been fully resolved, so the downstream import is safe.
from .factorize.llt_blocked_nd_solver import LLTBlockedNDSolver

# Register the reordering solver in the name<->type maps so it can be selected
# via the string "LLTBND" in ConstrainedDynamicsConfig.linear_solver_type.
# (The class name keeps the historical "ND" suffix for backwards-compatibility;
# internally the reordering is now Reverse Cuthill-McKee.)
LinearSolverNameToType["LLTBND"] = LLTBlockedNDSolver
LinearSolverTypeToName[LLTBlockedNDSolver] = "LLTBND"

# Widen the LinearSolverType alias to include the reordering solver. This
# matters because `delassus.py` performs a runtime
# `issubclass(solver, LinearSolverType)` check and would otherwise reject it.
LinearSolverType = (
    LLTSequentialSolver | LLTBlockedSolver | LLTBlockedNDSolver
    | ConjugateGradientSolver | ConjugateResidualSolver
)

###
# Module interface
###

__all__ = [
    "ConjugateGradientSolver",
    "ConjugateResidualSolver",
    "DenseLinearOperatorData",
    "DenseRectangularMultiLinearInfo",
    "DenseSquareMultiLinearInfo",
    "DirectSolver",
    "IterativeSolver",
    "LLTBlockedNDSolver",
    "LLTBlockedSolver",
    "LLTSequentialSolver",
    "LinearSolver",
    "LinearSolverNameToType",
    "LinearSolverType",
    "LinearSolverTypeToName",
    "utils",
]
