# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare every graph buffer against generic first-fit for rigid rows."""

import unittest

import numpy as np
import warp as wp

from local_studies.colibri.rigid_color_rows import color_rows
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups


class TestRigidColorRows(unittest.TestCase):
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
                        actual["masks"].zero_()
                        actual["counts"].zero_()
                        wp.launch(
                            color_rows,
                            1,
                            [
                                elements,
                                active,
                                width,
                                *[
                                    actual[key]
                                    for key in (
                                        "masks",
                                        "row_color",
                                        "row_partition",
                                        "counts",
                                        "starts",
                                        "cursors",
                                        "ids",
                                        "num_colors",
                                    )
                                ],
                            ],
                            device=device,
                        )
                        for key in expected:
                            np.testing.assert_array_equal(actual[key].numpy(), expected[key].numpy())


if __name__ == "__main__":
    unittest.main()
