# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture_while-free mass-splitting dispatcher (experimental).

Same physics as :class:`SingleWorldMassSplittingDispatcher` but
replaces both ``wp.capture_while`` calls with host-side fixed loops.
Trip counts are bounded under mass splitting:

* Outer (colour drain): ``num_colors <= max_colored_partitions + 1``.
* Inner (head re-launch within a colour-round): also bounded by
  ``max_colored_partitions + 1`` -- worst case the head processes
  every partition in one round.

The bound is the partitioner's contract under mass splitting: it caps
at ``K`` regular colours plus 1 overflow bucket. Rounds past the
actual colour count, and head launches after ``head_active`` would
have flipped, are kernel-side no-ops (single int read + early
return). The trade-off versus :class:`SingleWorldMassSplittingDispatcher`:

* Save the per-round CUDA conditional-graph predicate evaluations
  (each ~600 ns × thousands per step).
* Pay for the extra no-op kernel launches (~1-3 µs each in graph
  capture).

The break-even depends on how many real rounds the capture_while body
runs naturally. For scenes where the tail kernel drains every colour
in one launch (e.g. the soft-body drop with all partitions <=
fuse_threshold), the capture_while wins because it bails after one
iteration. For scenes that genuinely need many rounds, the unrolled
variant catches up. Use the ``solver_config`` flag to A/B per scene.

Mirrors the C# PhoenX ``RunMethodParallelIterate`` pattern (host-side
``for partitionId in 0..MaxNumPartitions:`` loop, no GPU conditional
nodes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from newton._src.solvers.phoenx.mass_splitting import (
    launch_average_and_broadcast,
)
from newton._src.solvers.phoenx.solver_phoenx_kernels import _reset_head_active_kernel

if TYPE_CHECKING:
    from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


class SingleWorldMassSplittingUnrolledDispatcher:
    """Single-world + mass-splitting PGS with no ``wp.capture_while``."""

    __slots__ = ("_world",)

    def __init__(self, world: PhoenXWorld) -> None:
        self._world = world

    def begin_step(self) -> None:
        self._world._rebuild_mass_splitting_graph()

    def _unrolled_sweep(self, head_kernel, tail_kernel, idt: wp.float32) -> None:
        """Host-side fixed-count colour-drain.

        Outer loop iterates ``max_colored_partitions + 1`` times -- the
        partitioner's upper bound on colour count under mass splitting
        (K coloured + 1 overflow). Each iteration: one head pass
        (which itself unrolls ``NUM_INNER_WHILE_ITERATIONS=8`` head
        launches via :meth:`PhoenXWorld._capture_singleworld_sweep`)
        plus one fused-tail launch. After cursor hits 0, subsequent
        iters' head + tail launches early-exit cheaply (single int
        read + return).

        No ``wp.capture_while`` on either axis.
        """
        w = self._world
        k_plus_one = int(w.max_colored_partitions) + 1
        for _ in range(k_plus_one):
            wp.launch(
                _reset_head_active_kernel,
                dim=1,
                inputs=[w._head_active],
                device=w.device,
            )
            # One head pass per outer iter -- internally an unrolled
            # NUM_INNER_WHILE_ITERATIONS=8 head launches that bail
            # cheaply once head_active flips to 0 or cursor hits 0.
            # Replaces the inner wp.capture_while(head_active, ...).
            w._capture_singleworld_sweep(kernel=head_kernel, idt=idt)
            w._capture_singleworld_tail_sweep(kernel=tail_kernel, idt=idt)

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
        self._unrolled_sweep(prepare_head, prepare_fused, idt)
        launch_average_and_broadcast(
            copy_state, bodies, particles_or_sentinel,
            num_bodies=num_bodies, inv_dt=inv_dt,
        )

        for _ in range(w.solver_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(iterate_head, iterate_fused, idt)
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
        _, _, _, _, relax_head, relax_fused = w._singleworld_kernels()
        particles_or_sentinel = w._particles_or_sentinel()
        num_bodies = w.num_bodies
        copy_state = w._copy_state
        bodies = w.bodies

        for _ in range(w.velocity_iterations):
            w._partitioner.begin_sweep()
            self._unrolled_sweep(relax_head, relax_fused, idt)
            launch_average_and_broadcast(
                copy_state, bodies, particles_or_sentinel,
                num_bodies=num_bodies, inv_dt=inv_dt,
            )

        w._mass_splitting_writeback()


__all__ = ["SingleWorldMassSplittingUnrolledDispatcher"]
