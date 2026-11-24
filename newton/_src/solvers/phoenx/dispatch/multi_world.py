# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world dispatcher (mass splitting OFF).

Per-world coloring is built upstream in
:meth:`PhoenXWorld._build_per_world_coloring`. The selected scheduler may be
fast-tail or block-per-world, but it is fixed before CUDA graph capture. Mass
splitting is not supported on this path (per-world CSR layout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.simulation import PhoenXWorld


class MultiWorldDispatcher:
    """Per-world multi-world PGS dispatcher (mass splitting OFF)."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        # No-op; per-world coloring is built upstream.
        pass

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        if not w._regular_pgs_active_this_step:
            # Apply the substep's remaining forces before any projection.
            w._temporal_skip_sweeps()
        direct = getattr(w, "_direct_equality_system", None)
        overlap_factor = bool(
            direct is not None
            and direct.enabled
            and w._regular_pgs_active_this_step
            and w._combine_direct_prepare_projection
            and direct.supports_async_factor
        )
        if direct is not None and direct.enabled:
            if overlap_factor:
                direct.prepare_matrix(idt)
                direct.factor_async()
            else:
                direct.prepare_and_factor(idt)
        if w._regular_pgs_active_this_step:
            block_world = w._multi_world_scheduler == "block_world" and w._block_world_supported()
            direct_enabled = direct is not None and direct.enabled
            if direct_enabled or w.temporal_iterations:
                # Prepare once, then run the biased sweeps one at a time: exact
                # mechanisms project after each, and each temporal sweep is a
                # contact substep.
                if direct_enabled and not w._combine_direct_prepare_projection:
                    direct.solve(use_bias=False)
                if block_world:
                    w._solve_main_block_world(num_iterations=0, solve_direct=False)
                else:
                    w._solve_main(num_iterations=0, solve_direct=False)
                if direct_enabled:
                    if overlap_factor:
                        direct.wait_factor()
                    w._warm_start_owned_contacts()
                    direct.solve(use_bias=False)
                    # Test drive bounds only once contact warm starts load the mechanisms.
                    direct.resolve_bounded_drives(idt, use_bias=False)
                for iteration in range(w.solver_iterations):
                    w._temporal_enter_sweep(iteration, idt)
                    if block_world:
                        w._iterate_main_block_world(w._sweep_order(iteration))
                    else:
                        w._iterate_main(w._sweep_order(iteration))
                    if direct_enabled:
                        w._project_direct_after_sweep(direct, iteration, idt)
                    w._temporal_close_sweep(iteration)
            elif block_world:
                w._solve_main_block_world()
            else:
                w._solve_main()
        elif direct is not None and direct.enabled:
            w._warm_start_owned_contacts()
            direct.solve(use_bias=True)
            direct.resolve_bounded_drives(idt, use_bias=True)
        w._solve_owned_contacts_after_sweeps()
        if w._maximal_tree_projector is not None:
            if w._direct_tree_contacts:
                w._maximal_tree_projector.factor_contact_response()
            else:
                w._maximal_tree_projector.project(use_bias=True, idt=idt)
            w._solve_maximal_articulated_contacts(use_bias=True, refresh_mobility=True)
        if w._reduced_constraints_active_this_step:
            w._reduced_articulation.solve_constraints(w, idt, relax=False)

    def relax(self, idt: wp.float32) -> None:
        if self._world._active_velocity_iterations <= 0:
            return
        direct = getattr(self._world, "_direct_equality_system", None)
        if self._world._regular_pgs_active_this_step:
            block_world = self._world._multi_world_scheduler == "block_world" and self._world._block_world_supported()
            if direct is not None and direct.enabled:
                for iteration in range(self._world._active_velocity_iterations):
                    if block_world:
                        self._world._relax_velocities_block_world(
                            num_iterations=1, solve_direct=False, iteration_offset=iteration
                        )
                    else:
                        self._world._relax_velocities(num_iterations=1, solve_direct=False, iteration_offset=iteration)
                    direct.solve(use_bias=False)
                    direct.resolve_bounded_drives(idt, use_bias=False)
            elif block_world:
                self._world._relax_velocities_block_world()
            else:
                self._world._relax_velocities()
        elif direct is not None and direct.enabled and self._world._active_velocity_iterations > 0:
            direct.solve(use_bias=False)
            direct.resolve_bounded_drives(idt, use_bias=False)
        if self._world._active_velocity_iterations > 0:
            self._world._solve_direct_contacts(use_bias=False, refresh_mobility=False)
        if self._world._maximal_tree_projector is not None:
            if not self._world._direct_tree_contacts:
                self._world._maximal_tree_projector.project(use_bias=False, idt=idt)
            if self._world._active_velocity_iterations > 0:
                self._world._solve_maximal_articulated_contacts(use_bias=False, refresh_mobility=False)
        if self._world._reduced_constraints_active_this_step:
            self._world._reduced_articulation.solve_constraints(self._world, idt, relax=True)


__all__ = ["MultiWorldDispatcher"]
