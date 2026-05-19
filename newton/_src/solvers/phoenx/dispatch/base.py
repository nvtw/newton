# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-dispatcher Protocol. Each (step_layout, mass_splitting)
combination is a concrete dispatcher; :class:`PhoenXWorld` picks one
at construction so the hot path has no capability branches.
Implementations must be capture-safe (called from inside the
captured graph of :meth:`PhoenXWorld.step`)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import warp as wp


@runtime_checkable
class SolverDispatcher(Protocol):
    """Per-step PGS sweep choreography.

    Lifecycle: ``begin_step`` once per outer step (after CSR build) then
    ``solve`` + ``relax`` per substep. ``_integrate_positions`` stays in
    :meth:`PhoenXWorld.step` (identical across dispatchers).
    """

    def begin_step(self) -> None:
        """Per-step setup that depends on the just-built CSR (no-op
        by default; mass-splitting rebuilds its interaction graph)."""
        ...

    def solve(self, idt: wp.float32) -> None:
        """Bias-on PGS phase for one substep (prepare + iterate loop +
        any pre/post bookkeeping). ``idt = 1.0 / world.substep_dt``."""
        ...

    def relax(self, idt: wp.float32) -> None:
        """Bias-off relax sweeps for one substep. No-op when
        ``velocity_iterations == 0``."""
        ...


__all__ = ["SolverDispatcher"]
