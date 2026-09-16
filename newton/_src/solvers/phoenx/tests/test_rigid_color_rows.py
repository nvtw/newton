# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify exact rigid specialization of deterministic color grouping."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups


class TestRigidColorRows(unittest.TestCase):
    def test_generic_multibody_default(self):
        """Keep all eight endpoints active in the generic default on both devices."""
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            elements = wp.zeros(2, dtype=ElementInteractionData, device=device)
            host = elements.numpy()
            host["bodies"].fill(-1)
            host["bodies"][0] = np.arange(8)
            host["bodies"][1, :2] = [2, 3]
            elements.assign(host)
            active = wp.array([2], dtype=wp.int32, device=device)
            data = color_groups.allocate(2, 8, device)
            color_groups.build(data, elements, active, 4, device)
            np.testing.assert_array_equal(data["row_color"].numpy()[:2], [0, 1])

    def test_exact_buffers(self):
        """Preserve graph buffers through growth, shrinkage and multiple mask words."""
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            elements = wp.zeros(300, dtype=ElementInteractionData, device=device)
            host = elements.numpy()
            host["bodies"].fill(-1)
            host["bodies"][:96, :2] = [0, 1]
            rng = np.random.default_rng(1986)
            host["bodies"][96:, :2] = rng.integers(-1, 19, (204, 2))
            elements.assign(host)
            active = wp.zeros(1, dtype=wp.int32, device=device)
            expected = color_groups.allocate(300, 19, device)
            actual = color_groups.allocate(300, 19, device)
            for count in [300, 32, 0, 1, 96, 300]:
                for width in [1, 4, 8, 33]:
                    with self.subTest(device=device, count=count, width=width):
                        active.assign(np.array([count], dtype=np.int32))
                        color_groups.build(expected, elements, active, width, device)
                        color_groups.build(actual, elements, active, width, device, rigid_only=True)
                        for key in expected:
                            np.testing.assert_array_equal(actual[key].numpy(), expected[key].numpy())


if __name__ == "__main__":
    unittest.main()
