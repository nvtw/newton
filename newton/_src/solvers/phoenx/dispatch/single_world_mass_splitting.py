# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-world dispatcher with Tonge mass splitting.

Same PGS shape as :mod:`single_world` (head + fused tail under
``capture_while``) but with:

* ``begin_step`` rebuilds the ``(body, partition_key)`` interaction graph.
* ``solve`` broadcasts body/particle state into copy slots, averages
  slots between iterations (required for Jacobi convergence on the
  overflow bucket), and writes ``slot[0] -> body`` before integrate.
* ``relax`` runs the bias-off pass, averages between iters, writes back.

Unrolled variant: :mod:`single_world_mass_splitting_unrolled` (drops
``capture_while`` since mass-splitting bounds the colour count).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class SingleWorldMassSplittingDispatcher:
    """Single-world PGS dispatcher with Tonge mass splitting."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        self._world._rebuild_mass_splitting_graph()

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        # Fan body / particle state into every owned copy-state slot
        # before the constraint kernels read them.
        w._mass_splitting_broadcast()
        if w._constraint_capacity == 0:
            # Still need to writeback to keep slot[0] -> body in sync
            # for the next substep.
            w._mass_splitting_writeback()
            return

        inv_dt = 1.0 / w.substep_dt
        prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = w._singleworld_kernels()
        # Prepare applies the warm-start impulse to each body's slots;
        # average so the iterate phase starts from converged slot values.
        if w._refresh_prepare_this_substep():
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(prepare_head, prepare_fused, idt)
            w._mass_splitting_average_and_broadcast(inv_dt)
        else:
            w._run_cached_prepare_bookkeeping(idt)

        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(iterate_head, iterate_fused, idt)
            w._mass_splitting_average_and_broadcast(inv_dt)

        # Writeback slot[0].velocity -> body.velocity. step()'s
        # integrate_positions then advances bodies with the post-PGS
        # velocity.
        w._mass_splitting_writeback(already_averaged=True)

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w.velocity_iterations <= 0:
            return

        inv_dt = 1.0 / w.substep_dt
        _, _, _, _, relax_head, relax_fused = w._singleworld_kernels()
        for _ in range(w.velocity_iterations):
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(relax_head, relax_fused, idt)
            w._mass_splitting_average_and_broadcast(inv_dt)

        # Second writeback after relax: relax also routes through slots,
        # so the next substep would see stale body.velocity otherwise.
        w._mass_splitting_writeback(already_averaged=True)


__all__ = ["SingleWorldMassSplittingDispatcher"]
