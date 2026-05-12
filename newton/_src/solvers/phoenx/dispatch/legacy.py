# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transitional dispatcher that forwards every method to the parent
:class:`PhoenXWorld`.

This exists only during the dispatch-refactor staging. It lets
:meth:`PhoenXWorld.step` route through the dispatcher API today
without moving any logic yet -- subsequent commits move the
single-world / multi-world / mass-splitting paths into dedicated
dispatchers and drop this fallback.

Behavior is byte-identical to the pre-refactor inline ``if/else`` in
``step()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class LegacyDispatcher:
    """Forward every call to the legacy in-line methods on
    :class:`PhoenXWorld`.

    Encapsulates exactly the ``if self.step_layout == "single_world":``
    cascade that lived in :meth:`PhoenXWorld.step` so future commits can
    pull each branch out into its own dispatcher class incrementally.
    """

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        w = self._world
        if w.mass_splitting_enabled and w.step_layout == "single_world":
            w._rebuild_mass_splitting_graph()

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        if w.mass_splitting_enabled and w.step_layout == "single_world":
            w._mass_splitting_broadcast()
        if w.step_layout == "single_world":
            w._solve_main_singleworld()
            if w.mass_splitting_enabled:
                w._mass_splitting_writeback()
        else:
            w._solve_main()

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w.step_layout == "single_world":
            w._relax_velocities_singleworld()
            if w.mass_splitting_enabled and w.velocity_iterations > 0:
                w._mass_splitting_writeback()
        else:
            w._relax_velocities()


__all__ = ["LegacyDispatcher"]
