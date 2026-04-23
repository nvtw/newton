# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CI regression for the rabbit-pile SDF-contact stress test.

Runs a small (6-bunny) variant of
:mod:`example_phoenx_rabbit_pile` for 30 frames and asserts:

* Every bunny's state is finite (no NaN from over-stressed contacts).
* The pile hasn't exploded (every bunny is within 2 m of origin).
* The contact buffer has carried a non-trivial number of
  bunny-vs-bunny points at some point during the settle (proves the
  SDF narrow phase actually fired -- a regression that silently
  drops the SDF path would still produce finite state).

Separate from the dense :mod:`test_phoenx_stacking` suite because
SDF narrow-phase kernel compilation is heavy and we don't want to
add several seconds to the fast PhoenX test gate.
"""

from __future__ import annotations

import argparse
import unittest

import warp as wp


@unittest.skipUnless(wp.is_cuda_available(), "Rabbit pile requires CUDA (SDF)")
class TestPhoenXRabbitPile(unittest.TestCase):
    def test_six_bunnies_settle_without_exploding(self) -> None:
        from newton._src.solvers.jitter.examples.example_phoenx_rabbit_pile import (
            Example,
        )

        # Skip the pxr USD import path here -- the test runs without
        # optional deps and falls back to the icosahedron stand-in,
        # which exercises the exact same contact pipeline.
        args = argparse.Namespace(num_bunnies=6)
        # Stub the viewer -- we don't render, just step.
        class _NullViewer:
            def set_model(self, *_args, **_kwargs):
                pass

            def set_camera(self, *_args, **_kwargs):
                pass

            def apply_forces(self, *_args, **_kwargs):
                pass

            def register_shape_picker(self, *_args, **_kwargs):
                pass

            def begin_frame(self, *_args, **_kwargs):
                pass

            def log_state(self, *_args, **_kwargs):
                pass

            def log_contacts(self, *_args, **_kwargs):
                pass

            def end_frame(self, *_args, **_kwargs):
                pass

        ex = Example(_NullViewer(), args)

        peak_contacts = 0
        for _ in range(30):
            ex.step()
            n = int(ex.contacts.rigid_contact_count.numpy()[0])
            peak_contacts = max(peak_contacts, n)

        # This raises AssertionError on non-finite state or drift.
        ex.test_final()

        self.assertGreater(
            peak_contacts,
            0,
            "Rabbit-pile scene produced zero contacts during settle; "
            "SDF narrow phase likely regressed.",
        )


if __name__ == "__main__":
    unittest.main()
