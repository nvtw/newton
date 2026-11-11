# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The mixed warehouse of ``newton.examples.phoenx.example_phoenx_warehouse`` as one solver scene.

A small fleet of every robot type shares one world with shelved stock, loose packages,
packages shared by two robots, robots that touch each other, and hinged and sliding
mechanisms: the combination a large single world steps through together.
"""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.phoenx import example_phoenx_warehouse as warehouse


def _run(frames: int) -> warehouse.Example:
    args = argparse.Namespace(robots=64, storage_scale=1.0, seed=0, substeps=4, iterations=4)
    example = warehouse.Example(newton.viewer.ViewerNull(), args)
    for _ in range(frames):
        example.step()
        example.test_post_step()
    return example


@unittest.skipUnless(wp.is_cuda_available(), "the warehouse scene runs under CUDA graph capture")
class TestWarehouseScene(unittest.TestCase):
    def test_two_runs_repeat_bit_for_bit(self):
        """Robots, stock, shared packages, touching robots and mechanisms step deterministically."""
        first = _run(90)
        second = _run(90)
        self.assertIsNotNone(first.solver._reduced_articulation.fused_substep)
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(getattr(first.state, name).numpy(), getattr(second.state, name).numpy())
        # The run exercised contacts between robots, not only robots apart.
        cross = first.solver._reduced_articulation.contact_block_system.cross_column_count
        self.assertGreater(int(cross.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()
