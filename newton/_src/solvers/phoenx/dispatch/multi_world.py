# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-world dispatcher (mass splitting OFF). One block per world;
the fast-tail kernel runs prepare + every iterate sweep in a single
launch. Per-world coloring is built upstream in
:meth:`PhoenXWorld._build_per_world_coloring`. Mass splitting is not
supported on this path (per-world CSR layout)."""

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
        # No-op; per-world coloring is built upstream.
        pass

    def solve(self, idt: wp.float32) -> None:
        # Bundles prepare + every iterate into one scheduler launch.
        if self._world._multi_world_scheduler == "block_world" and self._world._block_world_supported():
            self._world._solve_main_block_world()
        else:
            self._world._solve_main()

    def relax(self, idt: wp.float32) -> None:
        if self._world._multi_world_scheduler == "block_world" and self._world._block_world_supported():
            self._world._relax_velocities_block_world()
        else:
            self._world._relax_velocities()


__all__ = ["MultiWorldFastTailDispatcher"]
