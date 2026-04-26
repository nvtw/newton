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

Usage
-----
::

    from newton._src.solvers.phoenx.solver_config import (
        NUM_INNER_WHILE_ITERATIONS,
        PHOENX_CONTACT_MATCHING,
    )

    pipeline = newton.CollisionPipeline(
        model, contact_matching=PHOENX_CONTACT_MATCHING
    )
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

__all__ = [
    "NUM_INNER_WHILE_ITERATIONS",
    "PHOENX_CONTACT_MATCHING",
]
