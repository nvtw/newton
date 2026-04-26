# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central knobs for the PhoenX solver.

All PhoenX examples, tests, and internal solver code read their
solver-wide tuning constants from this module.  Edit one value here to
sweep every in-tree scene + every unit test in a single commit.

Knobs
-----

``PHOENX_CONTACT_MATCHING`` (``Literal["sticky", "latest"]``)
    Frame-to-frame contact matching mode fed into
    :class:`newton.CollisionPipeline`.

    Both modes populate ``Contacts.rigid_contact_match_index`` (the
    per-contact "which previous-frame contact am I the continuation
    of?" lookup the PhoenX warm-start gather kernel consumes), so
    either satisfies the solver's validation.  The difference is what
    happens to the matched contacts' geometry each frame:

    - ``"sticky"`` (required for good stacking): after matching, the
      contact's body-frame anchors
      (``point0`` / ``point1`` / ``offset0`` / ``offset1``) and its
      world-frame ``normal`` are **overwritten** with the saved
      previous-frame values.  Matched contacts therefore keep the
      exact same geometry across frames until they break, which
      eliminates the frame-to-frame jitter that otherwise leaks into
      the constraint Jacobian and wobbles stacks apart.
    - ``"latest"`` (fallback / diagnostic): matching still runs and
      the match index is still populated, but the current frame's
      fresh narrow-phase geometry is kept.  Useful when you want to
      isolate whether a regression is caused by the anchor-pinning
      (flip to ``"latest"`` and see if the symptom survives) or by
      matching itself.

    Do **not** set this to ``"disabled"`` -- the solver's ingest path
    raises at step-time without a match index.

``NUM_INNER_WHILE_ITERATIONS`` (``int``)
    Fixed-count inner loop wrapped inside every ``wp.capture_while``
    body in the PhoenX solver.  Each outer capture-while iteration
    evaluates its device-side termination flag and traverses one edge
    of the captured CUDA graph; that fixed cost dominates once the
    body itself is short (e.g. the per-colour single-world sweep, or
    one Jones-Plassmann round near the end of coloring).  Bundling
    ``NUM_INNER_WHILE_ITERATIONS`` copies of the body inside a single
    outer iteration amortises the overhead by the same factor.

    The solver + partitioner kernels inside the capture-while bodies
    early-exit (no memory work, no counter updates) when their
    device-side termination predicate -- ``color_cursor`` for the PGS
    sweep kernels, ``num_remaining`` for the JP coloring loop -- has
    already hit zero within the *same* outer iteration.  That makes
    it safe to unconditionally run the body ``NUM_INNER_WHILE_ITERATIONS``
    times; the tail iterations past convergence are no-ops and the
    capture-while exits on the following outer check.

    Tuning: larger values reduce overhead but increase wasted tail
    work on short loops (e.g. tiny scenes with only a handful of
    colours).  8 is the shipped default -- measured to be a strong
    sweet spot on the single-world PGS path where the per-colour
    launch overhead was the dominant cost.

``FUSE_TAIL_MAX_COLOR_SIZE`` (``int``)
    Maximum colour size (in cids) eligible for the **single-block
    fused tail** path of the single-world PGS sweep.  Once a sweep's
    remaining colours are all this small, control is handed from the
    persistent multi-block grid kernel to a single 1D-block fused
    kernel that walks the rest of the colours back-to-back, using
    ``wp.sync_block`` (i.e. ``__syncthreads``) in place of the
    kernel-launch boundary that would otherwise enforce write
    visibility between colours.  This eliminates O(remaining-colours)
    kernel-launch latencies on the tail of every sweep.

    Correctness contract: a "small" colour (size <= this knob) fits
    entirely inside a single block of width
    :data:`FUSE_TAIL_BLOCK_DIM`, so every cid of the colour is owned
    by a distinct lane and ``wp.sync_block`` suffices to order the
    colour's body-velocity writes before the next colour's reads --
    no cross-block coordination is needed.  The persistent-grid
    kernels hand off automatically: they early-exit (without
    decrementing ``color_cursor``) when the *current* colour's size
    is already <= this threshold, leaving the cursor pointing at the
    first colour the fused tail kernel will sweep.

    Set to ``0`` to disable the fused-tail path entirely; every
    colour then flows through the persistent-grid kernel as before.
    The shipped default is ``0`` until the path has been measured on
    the target workloads; bump to ``FUSE_TAIL_BLOCK_DIM`` (256) to
    enable.

``FUSE_TAIL_BLOCK_DIM`` (``int``)
    Block width of the single-block fused tail kernel.  Must be
    >= :data:`FUSE_TAIL_MAX_COLOR_SIZE` so a small colour's cids are
    each owned by a distinct lane within the one block.  Held equal
    to the ``FUSE_TAIL_MAX_COLOR_SIZE`` default (256) for the same
    register-budget reasons as :data:`FUSE_TAIL_BLOCK_DIM`'s
    grid-mode sibling ``_SINGLEWORLD_BLOCK_DIM``.

Usage
-----
::

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

#: Contact-matching mode every PhoenX example and test feeds into
#: :class:`newton.CollisionPipeline`.  See the module docstring above
#: for what each mode does and why ``"sticky"`` is the production
#: default.
PHOENX_CONTACT_MATCHING: Literal["sticky", "latest"] = "latest"

#: Number of times each ``wp.capture_while`` body is unrolled inside a
#: single outer capture-while iteration.  See the module docstring for
#: the correctness + performance rationale and the early-exit
#: contract every body kernel must honour.
NUM_INNER_WHILE_ITERATIONS: int = 8

#: Maximum colour size (in cids) eligible for the single-block fused
#: tail path of the single-world PGS sweep.  ``0`` disables the path
#: (every colour stays on the persistent-grid kernel).  See the module
#: docstring for the correctness contract and the hand-off mechanism.
FUSE_TAIL_MAX_COLOR_SIZE: int = 0

#: Block width of the fused tail kernel.  Must be
#: >= :data:`FUSE_TAIL_MAX_COLOR_SIZE` so every cid of a "small"
#: colour is owned by a distinct lane within the one block.
FUSE_TAIL_BLOCK_DIM: int = 256

__all__ = [
    "FUSE_TAIL_BLOCK_DIM",
    "FUSE_TAIL_MAX_COLOR_SIZE",
    "NUM_INNER_WHILE_ITERATIONS",
    "PHOENX_CONTACT_MATCHING",
]
