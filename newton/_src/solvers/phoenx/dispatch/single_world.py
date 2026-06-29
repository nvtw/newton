# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-world dispatcher (mass splitting OFF): per-substep PGS via
persistent-grid head + single-block fused tail, with ``wp.capture_while``
draining all colours. Mass-splitting variant in
:mod:`single_world_mass_splitting`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class SingleWorldDispatcher:
    """Single-world PGS dispatcher (mass splitting OFF)."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        # No-op: only mass splitting rebuilds an interaction graph.
        pass

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0:
            return
        if not w._regular_pgs_active_this_step:
            if w._reduced_contacts_active_this_step:
                w._reduced_articulation.solve_contacts(w, idt, relax=False)
            return
        prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = w._singleworld_kernels()
        if w._refresh_prepare_this_substep():
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(prepare_head, prepare_fused, idt)
        else:
            w._run_cached_prepare_bookkeeping(idt)
        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(iterate_head, iterate_fused, idt)
        if w._reduced_contacts_active_this_step:
            w._reduced_articulation.solve_contacts(w, idt, relax=False)

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w.velocity_iterations <= 0:
            return
        if not w._regular_pgs_active_this_step:
            if w._reduced_contacts_active_this_step:
                w._reduced_articulation.solve_contacts(w, idt, relax=True)
            return
        _, _, _, _, relax_head, relax_fused = w._singleworld_kernels()
        for _ in range(w.velocity_iterations):
            w._partitioner.begin_sweep()
            w._singleworld_head_plus_tail_sweep(relax_head, relax_fused, idt)
        if w._reduced_contacts_active_this_step:
            w._reduced_articulation.solve_contacts(w, idt, relax=True)


__all__ = ["SingleWorldDispatcher"]
