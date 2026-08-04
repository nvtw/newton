# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the PhoenX wrecking-ball example."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples.example_wrecking_ball import Example
from newton.viewer import ViewerNull


class TestWreckingBall(unittest.TestCase):
    def test_chain_remains_stable_during_release(self):
        """Keep the released chain bounded during its initial swing."""
        if not wp.get_device().is_cuda:
            self.skipTest("PhoenX requires CUDA")

        args = SimpleNamespace()
        example = Example(ViewerNull(), args)

        for _ in range(80):
            example.step()

        body_q = example.state.body_q.numpy()
        body_qd = example.state.body_qd.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        self.assertTrue(np.isfinite(body_qd).all())
        self.assertLess(float(np.abs(body_qd).max()), 100.0)


if __name__ == "__main__":
    unittest.main()
