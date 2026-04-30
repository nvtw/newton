# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central knobs for the PhoenX solver.

All PhoenX examples, tests, and internal solver code read their
tuning constants from this module.

Knobs
-----

``PHOENX_CONTACT_MATCHING`` (``Literal["sticky", "latest"]``)
    Contact-matching mode for :class:`newton.CollisionPipeline`.
    Both modes populate ``Contacts.rigid_contact_match_index`` (what
    the warm-start gather consumes); they differ in what happens to
    matched contacts' geometry:

    * ``"sticky"`` (required for good stacking): after matching, the
      anchors (``point0/1``, ``offset0/1``) and world-frame ``normal``
      are overwritten with the previous-frame values. No
      frame-to-frame geometry jitter leaks into the Jacobian.
    * ``"latest"`` (fallback / diagnostic): match index still
      populated but this frame's narrow-phase geometry is kept. Used
      to isolate whether a regression is caused by the anchor pinning.

    Do NOT set to ``"disabled"`` -- the ingest path raises without
    a match index.

``NUM_INNER_WHILE_ITERATIONS`` (``int``)
    Fixed-count inner unroll inside every ``wp.capture_while`` body.
    Each outer iteration pays a graph-edge-traversal cost that
    dominates once the body is short (per-colour PGS sweep, a
    tail JP round); bundling N copies amortises the cost by N.
    Safe because the kernels early-exit once their termination
    predicate (``color_cursor``, ``num_remaining``) hits zero --
    tail iterations are no-ops. Shipped default ``8``.

``FUSE_TAIL_MAX_COLOR_SIZE`` (``int``)
    Max colour size (in cids) eligible for the single-block fused-
    tail path of the single-world PGS sweep. Once a sweep's
    remaining colours are all <= this value, control hands off from
    the persistent multi-block grid to a single 1D-block fused
    kernel that walks the rest back-to-back using ``__syncthreads``
    between colours instead of kernel boundaries; saves
    O(remaining-colours) kernel launches per sweep.

    Correctness: a "small" colour fits inside one block of width
    :data:`FUSE_TAIL_BLOCK_DIM`, so each cid owns a distinct lane
    and ``__syncthreads`` orders writes between colours. The
    persistent-grid kernels hand off by early-exit (without
    decrementing ``color_cursor``) when the current colour's size is
    already <= this threshold.

    ``0`` disables the path. Default :data:`FUSE_TAIL_BLOCK_DIM`
    (256); parity asserted to 1e-6 by
    :mod:`newton._src.solvers.phoenx.tests.test_tail_fuse`.

``FUSE_TAIL_BLOCK_DIM`` (``int``)
    Block width of the fused tail kernel. Must be
    >= :data:`FUSE_TAIL_MAX_COLOR_SIZE`.

Usage::

    from newton._src.solvers.phoenx.solver_config import (
        FUSE_TAIL_BLOCK_DIM,
        FUSE_TAIL_MAX_COLOR_SIZE,
        NUM_INNER_WHILE_ITERATIONS,
        PHOENX_CONTACT_MATCHING,
    )

    pipeline = newton.CollisionPipeline(model, contact_matching=PHOENX_CONTACT_MATCHING)
"""

from __future__ import annotations

from typing import Literal

#: Contact-matching mode fed into :class:`newton.CollisionPipeline`.
#: See the module docstring; ``"sticky"`` is the production default.
PHOENX_CONTACT_MATCHING: Literal["sticky", "latest"] = "sticky"

#: Unroll count for every ``wp.capture_while`` body in PhoenX. See
#: the module docstring for the early-exit contract.
NUM_INNER_WHILE_ITERATIONS: int = 8

#: Max colour size (in cids) eligible for the single-block fused-tail
#: path of the single-world PGS sweep. ``0`` disables; default
#: :data:`FUSE_TAIL_BLOCK_DIM`. See the module docstring for the
#: hand-off mechanism and correctness contract.
FUSE_TAIL_MAX_COLOR_SIZE: int = 256

#: Block width of the fused-tail kernel.
#: Must be >= :data:`FUSE_TAIL_MAX_COLOR_SIZE`.
FUSE_TAIL_BLOCK_DIM: int = 256

#: Use the JP-MIS + greedy-colour partitioner
#: (:meth:`IncrementalContactPartitioner.build_csr_greedy`) instead of
#: the round-equals-colour JP path. Greedy gives 2-3x fewer colours on
#: dense contact graphs (Kapla tower: 78 → 28 = lower bound) at
#: comparable build time, in exchange for a max-colour-size that's
#: typically ~2x larger -- the colour sweep parallelism is already
#: GPU-saturated on big colours, so the trade favours fewer rounds.
#: Bounded at 64 colours total (single-int64 forbidden mask); falls
#: back to a descriptive error if a graph wants more, in which case
#: flip this off to recover the round-based JP.
PHOENX_USE_GREEDY_COLORING: bool = True

__all__ = [
    "FUSE_TAIL_BLOCK_DIM",
    "FUSE_TAIL_MAX_COLOR_SIZE",
    "NUM_INNER_WHILE_ITERATIONS",
    "PHOENX_CONTACT_MATCHING",
    "PHOENX_USE_GREEDY_COLORING",
]
