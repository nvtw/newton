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

from newton._src.solvers.phoenx.mass_splitting import (
    launch_average_and_broadcast,
)

# Mirror of ``solver_phoenx._SINGLEWORLD_BLOCK_DIM``; redeclared to
# avoid circular import. Keep in sync.
_SINGLEWORLD_BLOCK_DIM: int = 32

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

    def _unrolled_sweep(self, head_kernel, idt: wp.float32) -> None:
        """Head-only fixed-count colour drain. ``fuse_threshold=-1``
        keeps head from bailing on small partitions. Empty colour
        slots are cheap no-op launches; overflow at index K is the
        heavy work."""
        w = self._world
        contact_views = w._contact_views if w._contact_views is not None else w._contact_views_placeholder
        ms_cap = wp.int32(int(w.max_colored_partitions))
        ms_batch = wp.int32(int(w.mass_splitting_batch_size))
        for _ in range(self._launch_bound):
            wp.launch(
                head_kernel,
                dim=w._singleworld_total_threads,
                inputs=[
                    w.constraints,
                    w._contact_cols,
                    w.bodies,
                    w._particles_or_sentinel(),
                    idt,
                    wp.float32(w.sor_boost),
                    w._partitioner.element_ids_by_color,
                    w._partitioner.color_starts,
                    w._partitioner.num_colors,
                    w._partitioner.color_cursor,
                    w._contact_container,
                    contact_views,
                    wp.int32(w.num_joints),
                    wp.int32(w.num_cloth_triangles),
                    wp.int32(w.num_cloth_bending),
                    wp.int32(w.num_soft_tetrahedra),
                    wp.int32(w.num_bodies),
                    wp.int32(w._singleworld_total_threads),
                    wp.int32(-1),  # fuse_threshold disabled -> head handles all sizes
                    w._head_active,
                    w._copy_state,
                    ms_cap,
                    ms_batch,
                    w._partitioner.sweep_direction,
                ],
                block_dim=_SINGLEWORLD_BLOCK_DIM,
                device=w.device,
            )

    def solve(self, idt: wp.float32) -> None:
        w = self._world
        w._mass_splitting_broadcast()
        if w._constraint_capacity == 0:
            w._mass_splitting_writeback()
            return

        inv_dt = 1.0 / w.substep_dt
        prepare_head, _, iterate_head, _, _, _ = w._singleworld_kernels()
        particles_or_sentinel = w._particles_or_sentinel()
        num_bodies = w.num_bodies
        copy_state = w._copy_state
        bodies = w.bodies

        w._partitioner.begin_sweep()
        self._unrolled_sweep(prepare_head, idt)
        launch_average_and_broadcast(
            copy_state,
            bodies,
            particles_or_sentinel,
            num_bodies=num_bodies,
            inv_dt=inv_dt,
        )

        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(iterate_head, idt)
            launch_average_and_broadcast(
                copy_state,
                bodies,
                particles_or_sentinel,
                num_bodies=num_bodies,
                inv_dt=inv_dt,
            )

        w._mass_splitting_writeback()

    def relax(self, idt: wp.float32) -> None:
        w = self._world
        if w._constraint_capacity == 0 or w.velocity_iterations <= 0:
            return

        inv_dt = 1.0 / w.substep_dt
        _, _, _, _, relax_head, _ = w._singleworld_kernels()
        particles_or_sentinel = w._particles_or_sentinel()
        num_bodies = w.num_bodies
        copy_state = w._copy_state
        bodies = w.bodies

        for _ in range(w.velocity_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(relax_head, idt)
            launch_average_and_broadcast(
                copy_state,
                bodies,
                particles_or_sentinel,
                num_bodies=num_bodies,
                inv_dt=inv_dt,
            )

        w._mass_splitting_writeback()


__all__ = ["SingleWorldMassSplittingUnrolledDispatcher"]
