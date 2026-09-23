# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world PGS dispatcher with Tonge mass splitting.

Rigid contact-only worlds update regular independent-set colors directly and
reserve private copy state for the overflow color. Other worlds retain split
state for every color. A mass-weighted overflow average preserves linear and
angular momentum before the next PGS iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.simulation import PhoenXWorld


class MultiWorldMassSplittingDispatcher:
    """Per-world PGS dispatcher with Tonge mass splitting."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        self._world._rebuild_multiworld_mass_splitting_graph()

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        # Fully split worlds read copy state during regular colors and need an
        # initial broadcast. Overflow-only worlds read physical state first;
        # their persistent kernel broadcasts immediately before overflow.
        if not w._overflow_only_mass_splitting:
            w._mass_splitting_broadcast()
        if w._constraint_capacity == 0:
            # Fully split worlds still need slot[0] -> body writeback.
            if not w._overflow_only_mass_splitting:
                w._mass_splitting_writeback()
            return
        direct = getattr(w, "_direct_equality_system", None)
        overlap_factor = bool(
            direct is not None
            and direct.enabled
            and w._regular_pgs_active_this_step
            and w._combine_direct_prepare_projection
            and direct.supports_async_factor
            and not direct.has_bounded_drives
            # The ordinary-color dispatcher uses capture_while(), which
            # pauses graph capture between its tail and persistent head.
            # CUDA cannot pause while factor_stream is an unjoined fork.
            # Color groups use one fixed launch and can overlap safely.
            and w._color_group_data is not None
        )
        if direct is not None and direct.enabled:
            if overlap_factor:
                direct.prepare_matrix(idt)
                direct.factor_async()
            else:
                direct.prepare_and_factor(idt)
            # Unbounded local blocks are solved by the colored callbacks.
            # Keep their preparation, but avoid copy round trips around a
            # global solve that cannot change body velocities.
            if (
                not getattr(direct, "requires_global_projection", True)
                and w._regular_pgs_active_this_step
                and w._direct_contact_response is None
                and w._maximal_tree_projector is None
                and not w._reduced_constraints_active_this_step
            ):
                direct = None

        if not w._regular_pgs_active_this_step:
            w._mass_splitting_writeback()
            if direct is not None and direct.enabled:
                w._warm_start_owned_contacts()
                direct.solve(use_bias=True)
                direct.resolve_bounded_drives(idt, use_bias=True)
            w._solve_direct_contacts(use_bias=True, refresh_mobility=True)
            if w._maximal_tree_projector is not None:
                w._maximal_tree_projector.project(use_bias=True)
                w._solve_maximal_articulated_contacts(use_bias=True, refresh_mobility=True)
            if w._reduced_constraints_active_this_step:
                w._reduced_articulation.solve_constraints(w, idt, relax=False)
            return

        inv_dt = 1.0 / w.substep_dt
        direct_regular_colors = bool(
            w._overflow_only_mass_splitting
            and (direct is None or not direct.enabled)
            and not w._reduced_constraints_active_this_step
            and w.joint_refinement_iterations == 0
        )
        if direct is not None and direct.enabled:
            # The post-warm-start solve below supersedes an exact projection
            # here. Refresh the copy slots because direct pre-solve operations
            # may still update physical body velocities.
            direct.resolve_bounded_drives(idt, use_bias=False)
            w._mass_splitting_broadcast()
        # Prepare applies the warm-start impulse to each body's slots;
        # average so the iterate phase starts from converged slot values.
        if direct_regular_colors:
            phase = "prepare" if w._refresh_prepare_this_substep() else "cached_prepare"
            w._multiworld_mass_splitting_direct_regular(phase, idt)
        elif w._refresh_prepare_this_substep():
            w._multiworld_mass_splitting_sweep("prepare", idt)
            w._mass_splitting_average_and_broadcast(inv_dt)
        else:
            w._run_cached_prepare_bookkeeping(idt)
        if direct is not None and direct.enabled:
            w._mass_splitting_writeback(already_averaged=True)
            if overlap_factor:
                direct.wait_factor()
            w._warm_start_owned_contacts()
            direct.solve(use_bias=False)
            w._mass_splitting_broadcast()
        fuse_iterations = bool(
            w._fused_multiworld_mass_splitting
            and (direct is None or not direct.enabled)
            and not w._reduced_constraints_active_this_step
            and w.joint_refinement_iterations == 0
        )
        if direct_regular_colors:
            w._multiworld_mass_splitting_direct_regular(
                "iterate",
                idt,
                num_iterations=w.solver_iterations,
            )
        elif fuse_iterations:
            w._multiworld_mass_splitting_iterate_fused(idt)
        else:
            for iteration in range(w.solver_iterations):
                w._multiworld_mass_splitting_sweep("iterate", idt, reverse_colors=bool(iteration % 2))
                w._mass_splitting_average_and_broadcast(inv_dt)
                # Divide contact sweeps into temporal blocks when requested. A
                # complete PCR pass transfers intermediate contact impulses through
                # the joint graph. Final position recovery and the velocity pass
                # retain residual refinement.
                direct_projection = iteration in w._direct_joint_projection_iterations
                if direct is not None and direct.enabled and direct_projection:
                    w._mass_splitting_writeback(already_averaged=True)
                    final_projection = iteration == w.solver_iterations - 1
                    direct.solve(use_bias=final_projection, refine=final_projection)
                    if iteration + 1 < w.solver_iterations:
                        w._mass_splitting_broadcast()

        # Keep each joint's original copy ownership and reconcile its paired
        # impulse before the next refinement or body writeback.
        for _ in range(w.joint_refinement_iterations):
            w._multiworld_mass_splitting_sweep("iterate", idt, joint_only=True)
            w._mass_splitting_average_and_broadcast(inv_dt)

        # Writeback slot[0].velocity -> body.velocity. step()'s
        # integrate_positions then advances bodies with the post-PGS
        # velocity.
        if (direct is None or not direct.enabled) and not direct_regular_colors:
            w._mass_splitting_writeback(already_averaged=True)
        if direct is not None and direct.enabled:
            direct.resolve_bounded_drives(idt, use_bias=True)
        w._solve_direct_contacts(use_bias=True, refresh_mobility=True)
        if w._maximal_tree_projector is not None:
            w._maximal_tree_projector.project(use_bias=True)
            w._solve_maximal_articulated_contacts(use_bias=True, refresh_mobility=True)
        if w._reduced_constraints_active_this_step:
            w._reduced_articulation.solve_constraints(w, idt, relax=False)

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w._active_velocity_iterations <= 0:
            return

        direct = getattr(w, "_direct_equality_system", None)
        if not w._regular_pgs_active_this_step:
            if direct is not None and direct.enabled:
                direct.solve(use_bias=False)
                direct.resolve_bounded_drives(idt, use_bias=False)
            w._solve_direct_contacts(use_bias=False, refresh_mobility=False)
            if w._maximal_tree_projector is not None:
                w._maximal_tree_projector.project(use_bias=False)
                w._solve_maximal_articulated_contacts(use_bias=False, refresh_mobility=False)
            if w._reduced_constraints_active_this_step:
                w._reduced_articulation.solve_constraints(w, idt, relax=True)
            return

        direct_regular_colors = bool(
            w._overflow_only_mass_splitting
            and (direct is None or not direct.enabled)
            and not w._reduced_constraints_active_this_step
        )

        # Pose integration updates anisotropic angular velocity and world
        # inertia on the body state. Refresh every copy before relaxation so
        # the split rows cannot overwrite that torque-free update with stale
        # pre-integrate velocities.
        if not direct_regular_colors:
            w._mass_splitting_broadcast()
        inv_dt = 1.0 / w.substep_dt
        for iteration in range(w._active_velocity_iterations):
            if direct_regular_colors:
                w._multiworld_mass_splitting_direct_regular("relax", idt)
            else:
                w._multiworld_mass_splitting_sweep("relax", idt)
                w._mass_splitting_average_and_broadcast(inv_dt)
            if direct is not None and direct.enabled:
                w._mass_splitting_writeback(already_averaged=True)
                w._wait_direct_factor()
                direct.solve(use_bias=False)
                if iteration + 1 < w._active_velocity_iterations:
                    w._mass_splitting_broadcast()

        # Second writeback after relax: relax also routes through slots,
        # so the next substep would see stale body.velocity otherwise.
        if (direct is None or not direct.enabled) and not direct_regular_colors:
            w._mass_splitting_writeback(already_averaged=True)
        if direct is not None and direct.enabled:
            w._wait_direct_factor()
            direct.resolve_bounded_drives(idt, use_bias=False)

        w._solve_direct_contacts(use_bias=False, refresh_mobility=False)
        if w._maximal_tree_projector is not None:
            w._maximal_tree_projector.project(use_bias=False)
            w._solve_maximal_articulated_contacts(use_bias=False, refresh_mobility=False)
        if w._reduced_constraints_active_this_step:
            w._reduced_articulation.solve_constraints(w, idt, relax=True)


__all__ = ["MultiWorldMassSplittingDispatcher"]
