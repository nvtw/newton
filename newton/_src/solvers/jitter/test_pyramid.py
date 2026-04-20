# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-stability regression test: settle an N-layer box pyramid.

This is the first end-to-end test for the Jitter solver's persistent
contact path. It exercises:

* :class:`CollisionPipeline` ingest with ``contact_matching=True``,
* the ``(contact_first, contact_count)`` column packing in
  :class:`ConstraintContainer`,
* frame-to-frame warm-starting via ``rigid_contact_match_index`` and
  the double-buffered :class:`ContactContainer`,
* the unified PGS dispatcher (contacts and joints share the loop),
* and the Baumgarte / speculative bias that keeps resting contacts
  stable without hopping or sinking.

Any regression in the contact prepare/iterate path, ingest ordering,
warm-start gather, or bias formulation collapses the stack and the
test's per-cube position / velocity budget catches it.

The scene itself is driven by :class:`Example` from
:mod:`newton._src.solvers.jitter.example_pyramid` so the test and
the interactive example stay bit-for-bit in sync.
"""

from __future__ import annotations

import types
import unittest

import warp as wp

from newton._src.solvers.jitter.example_pyramid import Example


class _HeadlessViewer:
    """Minimal viewer stub so :class:`Example` can be instantiated in tests.

    Only the methods :class:`Example.__init__` calls (``set_model`` /
    ``set_camera``) and the render-path no-ops need to exist; the
    test never calls ``render()``.
    """

    def set_model(self, *_a, **_kw) -> None: ...
    def set_camera(self, *_a, **_kw) -> None: ...
    def begin_frame(self, *_a, **_kw) -> None: ...
    def end_frame(self, *_a, **_kw) -> None: ...
    def log_state(self, *_a, **_kw) -> None: ...
    def log_contacts(self, *_a, **_kw) -> None: ...


def _run_pyramid(layers: int, frames: int) -> Example:
    """Build and settle an ``layers``-tall pyramid for ``frames`` frames."""
    viewer = _HeadlessViewer()
    args = types.SimpleNamespace(layers=layers)
    ex = Example(viewer, args)
    for _ in range(frames):
        ex.step()
    return ex


class TestPyramidSettle(unittest.TestCase):
    """End-to-end settle test for the Jitter contact solver.

    A ``layers``-row pyramid starts ~1 mm above its resting height in
    each slot. After ``SETTLE_FRAMES`` simulation frames at 60 fps /
    4 substeps, every cube's :meth:`Example.test_final` budget
    (10 cm position slack, 0.5 m/s velocity slack) must pass. Tested
    for a tiny (3-layer) and a realistic (10-layer) tower.
    """

    # Big frame counts on purpose: the warm-start path needs several
    # seconds of simulated time to damp out the initial transient
    # when the whole stack is freshly dropped.
    SETTLE_FRAMES = 180  # 3 s @ 60 fps

    @classmethod
    def setUpClass(cls):
        # Warp picks the default device; keep it explicit so CI logs
        # show us which backend was actually exercised.
        cls.device = wp.get_preferred_device()

    def test_3_layer_pyramid(self):
        """A 3-layer pyramid (6 cubes) must come to rest."""
        ex = _run_pyramid(layers=3, frames=self.SETTLE_FRAMES)
        ex.test_final()

    def test_10_layer_pyramid(self):
        """A 10-layer pyramid (55 cubes) must come to rest.

        This is the user-facing stability bar: if a 10-tall stack
        holds together, the contact solver's warm-start and friction
        paths are doing their job.
        """
        ex = _run_pyramid(layers=10, frames=self.SETTLE_FRAMES)
        ex.test_final()


if __name__ == "__main__":
    unittest.main(verbosity=2)
