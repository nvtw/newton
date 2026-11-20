# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU stepping for the authored PhoenX Kapla example."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.examples import example_kapla_tower
from newton.viewer import ViewerNull


class TestPhoenXKaplaCPU(unittest.TestCase):
    def test_cpu_example_advances_bricks_and_camera(self) -> None:
        with wp.ScopedDevice("cpu"), patch.object(example_kapla_tower, "NUM_BRICKS", 8):
            example = example_kapla_tower.Example(ViewerNull(), SimpleNamespace(solver="classic", max_colors=10))
            camera_target = np.add(example._camera_collider_initial_pos, (0.02, 0.0, 0.0))
            example.viewer.camera = SimpleNamespace(pos=wp.vec3(*camera_target))
            example.step()
            self.assertEqual(example.frame_index, 1)
            self.assertEqual(example.model.body_count, 9)
            positions = example.bodies.position.numpy()
            self.assertTrue(np.isfinite(positions).all())
            np.testing.assert_allclose(positions[example._camera_phoenx_slot], camera_target, atol=1.0e-6)
