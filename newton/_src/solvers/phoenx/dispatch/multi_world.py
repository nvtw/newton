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
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class MultiWorldDispatcher:
    """Per-world multi-world PGS dispatcher (mass splitting OFF)."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        # No-op; per-world coloring is built upstream.
        pass

    def solve(self, idt: wp.float32) -> None:
        direct = getattr(self._world, "_direct_equality_system", None)
        overlap_factor = bool(
            direct is not None
            and direct.enabled
            and self._world._regular_pgs_active_this_step
            and self._world._combine_direct_prepare_projection
            and direct.factor_stream is not None
        )
        if direct is not None and direct.enabled:
            if overlap_factor:
                direct.prepare_matrix(idt)
                direct.factor_async()
            else:
                direct.prepare_and_factor(idt)
        if self._world._regular_pgs_active_this_step:
            block_world = self._world._multi_world_scheduler == "block_world" and self._world._block_world_supported()
            if direct is not None and direct.enabled:
                # Factor once, then alternate inequality sweeps with triangular
                # solves so contacts see mechanism-level mobility.
                if not self._world._combine_direct_prepare_projection:
                    direct.solve(use_bias=False)
                    direct.resolve_bounded_drives(idt, use_bias=False)
                if block_world:
                    self._world._solve_main_block_world(num_iterations=0, solve_direct=False)
                else:
                    self._world._solve_main(num_iterations=0, solve_direct=False)
                if overlap_factor:
                    direct.wait_factor()
                direct.solve(use_bias=False)
                if self._world._combine_direct_prepare_projection:
                    direct.resolve_bounded_drives(idt, use_bias=False)
                for iteration in range(self._world.solver_iterations):
                    if block_world:
                        self._world._iterate_main_block_world(iteration)
                    else:
                        self._world._iterate_main(iteration)
                    direct.solve(use_bias=iteration == self._world.solver_iterations - 1)
                direct.resolve_bounded_drives(idt, use_bias=True)
            elif block_world:
                self._world._solve_main_block_world()
            else:
                self._world._solve_main()
        elif direct is not None and direct.enabled:
            direct.solve(use_bias=True)
            direct.resolve_bounded_drives(idt, use_bias=True)
        self._world._solve_direct_contacts(use_bias=True, refresh_mobility=True)
        if self._world._maximal_tree_projector is not None:
            if self._world._direct_tree_contacts:
                self._world._maximal_tree_projector.factor_contact_response()
            else:
                self._world._maximal_tree_projector.project(use_bias=True, dt=self._world.substep_dt)
            self._world._solve_maximal_articulated_contacts(use_bias=True, refresh_mobility=True)
        if self._world._reduced_constraints_active_this_step:
            self._world._reduced_articulation.solve_constraints(self._world, idt, relax=False)

    def relax(self, idt: wp.float32) -> None:
        direct = getattr(self._world, "_direct_equality_system", None)
        if self._world._regular_pgs_active_this_step:
            block_world = self._world._multi_world_scheduler == "block_world" and self._world._block_world_supported()
            if direct is not None and direct.enabled:
                for iteration in range(self._world.velocity_iterations):
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
        elif direct is not None and direct.enabled and self._world.velocity_iterations > 0:
            direct.solve(use_bias=False)
            direct.resolve_bounded_drives(idt, use_bias=False)
        if self._world.velocity_iterations > 0:
            self._world._solve_direct_contacts(use_bias=False, refresh_mobility=False)
        if self._world._maximal_tree_projector is not None:
            if not self._world._direct_tree_contacts:
                self._world._maximal_tree_projector.project(use_bias=False, dt=self._world.substep_dt)
            if self._world.velocity_iterations > 0:
                self._world._solve_maximal_articulated_contacts(use_bias=False, refresh_mobility=False)
        if self._world._reduced_constraints_active_this_step:
            self._world._reduced_articulation.solve_constraints(self._world, idt, relax=True)


__all__ = ["MultiWorldDispatcher"]
