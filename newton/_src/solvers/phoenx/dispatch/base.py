# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-dispatcher Protocol.

Each PhoenX step combination (single-world / multi-world × mass-splitting
on/off, in the future per-world MS, multi-stream, etc.) is implemented
by a concrete dispatcher. :class:`PhoenXWorld` picks one at construction
and routes every per-substep PGS sweep through it; no runtime branching
on capability flags in the hot path.

Dispatchers read shared physics state (bodies, constraints, partitioner,
contact pipeline) from their parent ``world`` backref but own all
path-private scratch buffers themselves -- so mass-splitting state only
exists when the mass-splitting dispatcher is active, and per-world
buffers only when the multi-world dispatcher is active.

Implementations are expected to be capture-safe (called from inside a
captured CUDA graph by :meth:`PhoenXWorld.step`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import warp as wp


@runtime_checkable
class SolverDispatcher(Protocol):
    """Per-(step_layout, mass_splitting) PGS sweep choreography.

    The hot-path lifecycle is:

    1. :meth:`begin_step` -- once per outer step, after the partitioner
       has produced this step's CSR. Mass-splitting dispatchers rebuild
       their ``(body, partition_key)`` interaction graph here.
    2. Per substep (called from :meth:`PhoenXWorld.step`):

       a. :meth:`solve` -- bias-on PGS phase. Includes any pre-solve
          setup (e.g. mass-splitting broadcast), the prepare sweep, the
          ``solver_iterations`` iterate loop, and any post-solve
          writeback. **Does not** call ``_integrate_positions`` -- that
          stays in :meth:`PhoenXWorld.step` because it's identical
          across dispatchers.
       b. :meth:`relax` -- bias-off TGS-soft relax sweeps and any
          post-relax writeback. No-op when ``velocity_iterations ==
          0``.
    """

    def begin_step(self) -> None:
        """Per-outer-step setup that depends on the just-built CSR.
        Default-implementations are no-ops."""
        ...

    def solve(self, idt: wp.float32) -> None:
        """Run one substep's bias-on PGS phase.

        ``idt`` is the inverse substep dt (``1.0 / world.substep_dt``).
        """
        ...

    def relax(self, idt: wp.float32) -> None:
        """Run one substep's bias-off relax sweeps + post-relax bookkeeping."""
        ...


__all__ = ["SolverDispatcher"]
