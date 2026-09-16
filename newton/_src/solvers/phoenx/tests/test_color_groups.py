# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check deterministic coloring and partition coverage independently."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting import color_groups


class TestColorGroups(unittest.TestCase):
    def test_rebuild_covers_rows_and_preserves_independent_colors(self):
        """Match greedy set coloring across word boundaries and active-count changes."""
        endpoints = [[0, 1]] * 70 + [[2, 3], [-1, 4], [1, 3], [0, 2, 4], [-1, -1]]
        capacity = len(endpoints) + 7
        elements = wp.zeros(capacity, dtype=ElementInteractionData, device="cpu")
        host = elements.numpy()
        host["bodies"].fill(-1)
        for i, row in enumerate(endpoints):
            host["bodies"][i, : len(row)] = row
        elements.assign(host)
        active = wp.zeros(1, dtype=wp.int32, device="cpu")
        for width in (1, 2, 8, 33):
            data = color_groups.allocate(capacity, 5, "cpu")
            for count in (len(endpoints), 3, 0, len(endpoints)):
                active.assign(np.asarray([count], dtype=np.int32))
                color_groups.build(data, elements, active, width, "cpu")
                occupied, members, expected = [], [], []
                for i, row in enumerate(endpoints[:count]):
                    nodes = {v for v in row if v >= 0}
                    color = next((c for c, used in enumerate(occupied) if not used & nodes), len(occupied))
                    if color == len(occupied):
                        occupied.append(set())
                        members.append([])
                    occupied[color].update(nodes)
                    members[color].append(i)
                    expected.append(color)
                np.testing.assert_array_equal(data["row_color"].numpy()[:count], expected)
                np.testing.assert_array_equal(
                    data["row_partition"].numpy()[:count], np.asarray(expected, dtype=int) // width
                )
                np.testing.assert_array_equal(data["ids"].numpy()[:count], [row for group in members for row in group])
                np.testing.assert_array_equal(
                    data["starts"].numpy()[: len(members) + 1], np.r_[0, np.cumsum([len(group) for group in members])]
                )
                self.assertEqual(int(data["num_colors"].numpy()[0]), len(members))

    def test_invalid_dimensions_are_rejected(self):
        """Reject dimensions that would divide by zero or allocate no mask words."""
        for capacity, nodes in ((0, 1), (1, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                color_groups.allocate(capacity, nodes, "cpu")
        with self.assertRaises(ValueError):
            color_groups.build({}, None, None, 0, "cpu")


if __name__ == "__main__":
    unittest.main()
