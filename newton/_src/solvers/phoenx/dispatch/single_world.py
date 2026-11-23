# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-world dispatcher (mass splitting OFF): per-substep PGS via
persistent-grid head + single-block fused tail, with ``wp.capture_while``
draining all colours. Mass-splitting variant in
:mod:`single_world_mass_splitting`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from newton._src.solvers.phoenx.islands.iteration_schedule import IslandIterationSchedule

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.simulation import PhoenXWorld


class SingleWorldDispatcher:
    """Single-world PGS dispatcher (mass splitting OFF)."""

    __slots__ = ("_island_schedule", "_world")

    def __init__(self, world: PhoenXWorld, *, island_parallel_iterations: bool = False) -> None:
        self._world = world
        self._island_schedule = (
            IslandIterationSchedule(world) if island_parallel_iterations and world._constraint_capacity else None
        )

    def begin_step(self) -> None:
        # No-op: only mass splitting rebuilds an interaction graph.
        pass

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0:
            if w.temporal_iterations:
                w._temporal_apply_forces(w.solver_iterations - 1)
            return
        if w.temporal_iterations and not w._regular_pgs_active_this_step:
            # No contact substeps run: apply the substep's remaining forces before any projection.
            w._temporal_apply_forces(w.solver_iterations - 1)
        direct = getattr(w, "_direct_equality_system", None)
        if direct is not None and direct.enabled:
            direct.prepare_and_factor(idt)
        if w._regular_pgs_active_this_step:
            if direct is not None and direct.enabled and not w._combine_direct_prepare_projection:
                direct.solve(use_bias=False)
            prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = w._singleworld_kernels()
            # Contact columns, joint ownership, and sleep states span the native substeps.
            if self._island_schedule is not None and w._current_substep_index == 0:
                self._island_schedule.build(w)
            if w._refresh_prepare_this_substep():
                prepare_schedule = self._island_schedule if w.parallel_contact_prepare else None
                if prepare_schedule is None:
                    w._partitioner.begin_sweep()
                w._singleworld_head_plus_tail_sweep(
                    prepare_head,
                    prepare_fused,
                    idt,
                    island_schedule=prepare_schedule,
                )
            else:
                w._run_cached_prepare_bookkeeping(idt)
            if direct is not None and direct.enabled:
                w._warm_start_owned_contacts()
                # Project contact/limit warm starts before the first inequality
                # sweep. The factor is fixed for the entire substep.
                direct.solve(use_bias=False)
                # Test drive bounds only once contact warm starts load the mechanisms.
                direct.resolve_bounded_drives(idt, use_bias=False)
            temporal = w.temporal_iterations
            if temporal:
                w._temporal_begin_sweeps()
            for iteration in range(w.solver_iterations):
                last = iteration == w.solver_iterations - 1
                if temporal and iteration > 0:
                    w._temporal_next_sweep(idt)
                if self._island_schedule is None:
                    w._partitioner.begin_sweep()
                w._singleworld_head_plus_tail_sweep(
                    iterate_head, iterate_fused, idt, island_schedule=self._island_schedule
                )
                if direct is not None and direct.enabled:
                    # Every temporal sweep is a contact substep whose motion enters the
                    # integrated positions, so each one carries the joint recovery bias.
                    direct.solve(use_bias=last or temporal)
                    # Limits and contacts must react to the capped motor during
                    # iteration, not retain an uncapped reaction at integration.
                    direct.resolve_bounded_drives(idt, use_bias=last or temporal)
                if temporal:
                    w._temporal_end_sweep(last)
        elif direct is not None and direct.enabled:
            w._warm_start_owned_contacts()
            direct.solve(use_bias=True)
            direct.resolve_bounded_drives(idt, use_bias=True)
        w._solve_direct_contacts(use_bias=True, refresh_mobility=True)
        if w._maximal_tree_projector is not None:
            if w._direct_tree_contacts:
                w._maximal_tree_projector.factor_contact_response()
            else:
                w._maximal_tree_projector.project(use_bias=True, idt=idt)
            w._solve_maximal_articulated_contacts(use_bias=True, refresh_mobility=True)
        if w._reduced_constraints_active_this_step:
            w._reduced_articulation.solve_constraints(w, idt, relax=False)

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w._active_velocity_iterations <= 0:
            return
        direct = getattr(w, "_direct_equality_system", None)
        if w._regular_pgs_active_this_step and w._active_velocity_iterations > 0:
            _, _, _, _, relax_head, relax_fused = w._singleworld_kernels()
            for _ in range(w._active_velocity_iterations):
                if self._island_schedule is None:
                    w._partitioner.begin_sweep()
                w._singleworld_head_plus_tail_sweep(relax_head, relax_fused, idt, island_schedule=self._island_schedule)
                if direct is not None and direct.enabled:
                    direct.solve(use_bias=False)
                    direct.resolve_bounded_drives(idt, use_bias=False)
        elif direct is not None and direct.enabled and w._active_velocity_iterations > 0:
            direct.solve(use_bias=False)
            direct.resolve_bounded_drives(idt, use_bias=False)
        if w._active_velocity_iterations > 0:
            w._solve_direct_contacts(use_bias=False, refresh_mobility=False)
        if w._maximal_tree_projector is not None:
            if not w._direct_tree_contacts:
                w._maximal_tree_projector.project(use_bias=False, idt=idt)
            if w._active_velocity_iterations > 0:
                w._solve_maximal_articulated_contacts(use_bias=False, refresh_mobility=False)
        if w._reduced_constraints_active_this_step:
            w._reduced_articulation.solve_constraints(w, idt, relax=True)


__all__ = ["SingleWorldDispatcher"]
