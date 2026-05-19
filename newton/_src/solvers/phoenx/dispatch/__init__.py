# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX solver dispatchers.

A :class:`SolverDispatcher` owns the per-substep PGS sweep choreography
for one ``(step_layout, mass_splitting)`` combination. Implementations
write straight-line code -- no runtime capability checks inside the hot
path -- and own any path-private scratch (mass-splitting copy state,
multi-world per-world buffers, etc.). Kernels and shared physics state
(bodies, constraints, partitioner) are read from the parent
:class:`PhoenXWorld` via a backref.

See :mod:`base` for the Protocol surface. Concrete dispatchers live in
sibling modules named after their configuration.
"""

from newton._src.solvers.phoenx.dispatch.base import SolverDispatcher

__all__ = ["SolverDispatcher"]
