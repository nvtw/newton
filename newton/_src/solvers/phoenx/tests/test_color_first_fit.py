# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise arbitrary mask occupancy through exact first-fit graph schedules."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups as reference


class TestBinaryColorSearch(unittest.TestCase):
    def test_random_graphs_match_greedy_reference(self):
        """Preserve all greedy colors for ragged multibody rows and mask boundaries."""
        rng = np.random.default_rng(1942)
        for seed in range(12):
            bodies = np.full((300, 8), -1, dtype=np.int32)
            bodies[:96, :2] = [0, 1]
            for row in range(96, 300):
                count = int(rng.integers(0, 9))
                bodies[row, :count] = rng.choice(19, count, replace=False)
            elements = wp.zeros(300, dtype=ElementInteractionData, device="cpu")
            host = elements.numpy()
            host["bodies"] = bodies
            elements.assign(host)
            active = wp.array([300], dtype=wp.int32, device="cpu")
            data = reference.allocate(300, 19, "cpu")
            reference.build(data, elements, active, seed + 1, "cpu")
            used = []
            expected = []
            for row in bodies:
                nodes = set(row[row >= 0])
                color = next((i for i, busy in enumerate(used) if not nodes & busy), len(used))
                if color == len(used):
                    used.append(set())
                used[color].update(nodes)
                expected.append(color)
            np.testing.assert_array_equal(data["row_color"].numpy()[:300], expected)


if __name__ == "__main__":
    unittest.main()
