# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture_while-free mass-splitting dispatcher.

Two structural changes vs :class:`SingleWorldMassSplittingDispatcher`:

1. **No ``wp.capture_while``.** The partitioner caps colour count at
   ``max_colored_partitions + 1`` (K coloured + 1 overflow) under mass
   splitting, so the colour-drain loop has a host-known upper bound
   and becomes a fixed ``for`` range. Mirrors the C# PhoenX
   ``RunMethodParallelIterate`` pattern.
2. **No fused tail kernel.** The single-block tail was designed to
   amortise launch overhead across many trailing small colours, but on
   the MS path the head kernel can handle every partition size (small
   and overflow alike -- it already does ``ms_batch_size`` batching
   internally) using the persistent grid (multiple SMs in parallel).
   The tail's single-block serial pass is a *pessimisation* on dense
   contact scenes (cloth-vs-rigid impacts have ~70-element partitions
   that the tail processes serially when they could run parallel
   across 16+ blocks).

Per sweep: exactly ``K + 1`` head launches, no tail, no capture_while.
Rounds past ``cursor == 0`` early-exit cheaply in the head kernel.
``fuse_threshold = -1`` disables the head's "bail to tail" check so
head processes every partition itself.

Profile motivation: on example_cloth_hanging at the cube-settled phase,
the fused tail was 78% of GPU time (0.31 ms/launch × 360
launches/5steps). Switching to head-only routes the same work through
parallel persistent-grid launches at a small fraction of the per-call
cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from newton._src.solvers.phoenx.mass_splitting import (
    launch_average_and_broadcast,
)

# Mirror of :data:`solver_phoenx._SINGLEWORLD_BLOCK_DIM`. Re-declared
# here to avoid a circular import from the dispatch module back into
# the solver module.
_SINGLEWORLD_BLOCK_DIM: int = 256

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class SingleWorldMassSplittingUnrolledDispatcher:
    """Single-world + mass-splitting PGS with no ``wp.capture_while``."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        self._world._rebuild_mass_splitting_graph()

    def _unrolled_sweep(self, head_kernel, idt: wp.float32) -> None:
        """Host-side fixed-count colour-drain, head-only.

        Loop iterates exactly ``max_colored_partitions + 1`` times --
        the partitioner's upper bound on colour count under mass
        splitting. Each iteration launches the head kernel ONCE with
        ``fuse_threshold = -1`` (so head doesn't bail on small
        partitions -- it processes every size itself via persistent
        grid). After cursor hits 0, remaining launches early-exit
        cheaply.

        No tail kernel, no capture_while. ``K + 1`` head launches per
        sweep, period.
        """
        w = self._world
        contact_views = w._contact_views if w._contact_views is not None else w._contact_views_placeholder
        ms_cap = wp.int32(int(w.max_colored_partitions))
        ms_batch = wp.int32(int(w.mass_splitting_batch_size))
        k_plus_one = int(w.max_colored_partitions) + 1
        for _ in range(k_plus_one):
            wp.launch(
                head_kernel,
                dim=w._singleworld_total_threads,
                inputs=[
                    w.constraints,
                    w._contact_cols,
                    w.bodies,
                    w._particles_or_sentinel(),
                    idt,
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
        prepare_head, prepare_fused, iterate_head, iterate_fused, _, _ = w._singleworld_kernels()
        particles_or_sentinel = w._particles_or_sentinel()
        num_bodies = w.num_bodies
        copy_state = w._copy_state
        bodies = w.bodies

        w._partitioner.begin_sweep()
        self._unrolled_sweep(prepare_head, idt)
        launch_average_and_broadcast(
            copy_state, bodies, particles_or_sentinel,
            num_bodies=num_bodies, inv_dt=inv_dt,
        )

        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(iterate_head, idt)
            launch_average_and_broadcast(
                copy_state, bodies, particles_or_sentinel,
                num_bodies=num_bodies, inv_dt=inv_dt,
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
                copy_state, bodies, particles_or_sentinel,
                num_bodies=num_bodies, inv_dt=inv_dt,
            )

        w._mass_splitting_writeback()


__all__ = ["SingleWorldMassSplittingUnrolledDispatcher"]
