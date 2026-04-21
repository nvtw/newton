# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Central knob for the Jitter solver's frame-to-frame contact matching.

All Jitter examples and tests build their :class:`newton.CollisionPipeline`
through this module instead of hard-coding a ``contact_matching=...``
literal.  That gives us exactly one place to flip the mode for the whole
solver when we want to experiment with the non-sticky fallback.

Modes
-----
Both modes populate ``Contacts.rigid_contact_match_index`` (the
per-contact "which previous-frame contact am I the continuation of?"
lookup the Jitter warm-start gather kernel consumes), so either one
satisfies the solver's validation.  The difference is what happens to
the matched contacts' geometry each frame:

- ``"sticky"`` (default, required for good stacking):
  After matching, the contact's body-frame anchors
  (``point0`` / ``point1`` / ``offset0`` / ``offset1``) and its
  world-frame ``normal`` are **overwritten** with the saved
  previous-frame values.  Matched contacts therefore keep the exact
  same geometry across frames until they break, which eliminates the
  frame-to-frame jitter that otherwise leaks into the constraint
  Jacobian and wobbles stacks apart.
- ``"latest"`` (fallback / diagnostic):
  Matching still runs and the match index is still populated, but
  the current frame's fresh narrow-phase geometry is kept.  Useful
  when you want to isolate whether a regression is caused by the
  anchor-pinning (flip to ``"latest"`` and see if the symptom
  survives) or by matching itself.

Usage
-----
::

    from newton._src.solvers.jitter.contact_matching_config import (
        JITTER_CONTACT_MATCHING,
    )

    pipeline = newton.CollisionPipeline(
        model, contact_matching=JITTER_CONTACT_MATCHING
    )

Flip the constant below to switch every Jitter example + test in one
edit.  Do **not** set it to ``"disabled"`` -- the solver's ingest path
raises at step-time without a match index.
"""

from __future__ import annotations

from typing import Literal

#: Contact-matching mode every Jitter example and test feeds into
#: :class:`newton.CollisionPipeline`.  Change this single constant to
#: sweep every in-tree Jitter scene; see the module docstring above for
#: what each mode does and why ``"sticky"`` is the default.
JITTER_CONTACT_MATCHING: Literal["sticky", "latest"] = "latest"

__all__ = ["JITTER_CONTACT_MATCHING"]
