# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture_while-free mass-splitting dispatcher.

Differences from :class:`SingleWorldMassSplittingDispatcher`:

* No ``wp.capture_while``: mass splitting bounds colours at K+1, so the
  colour drain becomes a fixed ``for _ in range(K+1)`` host loop.
* No fused tail: the head kernel processes every partition size via the
  persistent grid (the fused tail's single-block serial pass is a
  pessimisation on dense contact partitions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class SingleWorldMassSplittingUnrolledDispatcher:
    """Single-world + mass-splitting PGS with no ``wp.capture_while``."""

    __slots__ = ("_launch_bound", "_world")

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world
        # Exactly K+1 head launches per sweep (K coloured + 1 overflow).
        # A smaller bound would skip the overflow at index K (cursor
        # model processes partitions in order) and cause physics drift.
        self._launch_bound = int(world.max_colored_partitions) + 1

    def begin_step(self) -> None:
        self._world._rebuild_mass_splitting_graph()

    def _unrolled_sweep(self, head_kernel, idt: wp.float32, contact_container=None) -> None:
        """Head-only fixed-count colour drain."""
        w = self._world
        fuse_threshold = wp.int32(-1)
        for _ in range(self._launch_bound):
            w._launch_singleworld_head(head_kernel, idt, fuse_threshold, contact_container)

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        w._mass_splitting_broadcast()
        if w._constraint_capacity == 0:
            w._mass_splitting_writeback()
            return

        inv_dt = 1.0 / w.substep_dt
        prepare_head, _, iterate_head, _, _, _ = w._singleworld_kernels()
        if w._refresh_prepare_this_substep():
            w._partitioner.begin_sweep()
            self._unrolled_sweep(
                prepare_head,
                idt,
                w._contact_container_solve if w._colored_contact_rows else None,
            )
            w._mass_splitting_average_and_broadcast(inv_dt)
        else:
            w._run_cached_prepare_bookkeeping(idt)
        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(iterate_head, idt, w._contact_container_solve)
            w._mass_splitting_average_and_broadcast(inv_dt)

        w._mass_splitting_writeback(already_averaged=True)

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w.velocity_iterations <= 0:
            return

        inv_dt = 1.0 / w.substep_dt
        _, _, _, _, relax_head, _ = w._singleworld_kernels()
        for _ in range(w.velocity_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(relax_head, idt, w._contact_container_solve)
            w._mass_splitting_average_and_broadcast(inv_dt)

        w._mass_splitting_writeback(already_averaged=True)


__all__ = ["SingleWorldMassSplittingUnrolledDispatcher"]
