# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dispatcher for ``step_layout='multi_world'`` without mass splitting.

Pattern: one block per world, "fast-tail" PGS kernel that runs
prepare + ``solver_iterations`` iterate sweeps in a single launch
(the world-local colour streams are short and fit in shared memory).
Per-world bucketing + per-world coloring (JP or greedy MIS) happens
upstream in :meth:`PhoenXWorld._build_per_world_coloring`.

Mass splitting is not supported in multi-world today (the per-world
CSR layout differs from the global one the interaction-graph emit
kernel expects) -- :class:`PhoenXWorld` raises ``NotImplementedError``
at construction in that case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class MultiWorldFastTailDispatcher:
    """Per-world fast-tail PGS dispatcher (mass splitting OFF)."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        # No per-step setup beyond what PhoenXWorld.step() already does
        # (per-world coloring is built via _build_per_world_coloring).
        pass

    def solve(self, idt: wp.float32) -> None:
        # _solve_main bundles prepare + every iterate into a single fast-tail
        # launch; ``idt`` is recomputed inside from substep_dt for byte-
        # identical behaviour with the pre-refactor path.
        self._world._solve_main()

    def relax(self, idt: wp.float32) -> None:
        self._world._relax_velocities()


__all__ = ["MultiWorldFastTailDispatcher"]
