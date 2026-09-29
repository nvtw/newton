# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Auxiliary constraint solves shared by mass-splitting dispatchers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.simulation import PhoenXWorld


def solve_auxiliary_constraints(world: PhoenXWorld, idt: wp.float32, *, relax: bool) -> None:
    """Finish a mass-splitting phase after its copy-state reconciliation."""
    use_bias = not relax
    world._solve_direct_contacts(use_bias=use_bias, refresh_mobility=use_bias)
    if world._maximal_tree_projector is not None:
        world._maximal_tree_projector.project(use_bias=use_bias)
        world._solve_maximal_articulated_contacts(use_bias=use_bias, refresh_mobility=use_bias)
    if world._reduced_constraints_active_this_step:
        world._reduced_articulation.solve_constraints(world, idt, relax=relax)


__all__ = ["solve_auxiliary_constraints"]
