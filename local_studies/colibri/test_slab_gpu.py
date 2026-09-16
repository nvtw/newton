# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Captured slab topology must reproduce the CPU schedule exactly."""

import unittest
from pathlib import Path

import numpy as np
import warp as wp

from local_studies.colibri import slab_gpu
from local_studies.colibri.slab_schedule import build_schedule
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import ElementInteractionData
from newton._src.solvers.phoenx.mass_splitting.copy_state import copy_state_container_zeros
from newton._src.solvers.phoenx.mass_splitting.interaction_graph import (
    build_interaction_graph,
    interaction_graph_scratch_zeros,
)


class TestGpuSlabs(unittest.TestCase):
    def check_graph(self, endpoints, num_bodies):
        """Replay a captured build after changing active rows and clearing history."""
        capacity = len(endpoints) + 7
        elements = wp.zeros(capacity, dtype=ElementInteractionData, device="cuda:0")
        host = elements.numpy()
        host["bodies"].fill(-1)
        host["bodies"][: len(endpoints), : endpoints.shape[1]] = endpoints
        elements.assign(host)
        active = wp.array([len(endpoints)], dtype=wp.int32, device="cuda:0")
        for width in (1, 2, 8):
            data = slab_gpu.allocate(capacity, num_bodies, "cuda:0")
            scratch = interaction_graph_scratch_zeros(capacity * 8, "cuda:0")
            copies = copy_state_container_zeros(capacity * 8, num_bodies, "cuda:0")

            def build(data=data, width=width, scratch=scratch, copies=copies):
                slab_gpu.build(data, elements, active, width, "cuda:0")
                wp.launch(
                    slab_gpu.emit_slab_pairs, capacity, [elements, active, data["row_slab"], scratch], device="cuda:0"
                )
                build_interaction_graph(scratch, copies)

            build()
            with wp.ScopedCapture(device="cuda:0") as capture:
                build()
            for n in (len(endpoints), 3, len(endpoints), 0):
                active.assign(np.array([n], np.int32))
                wp.capture_launch(capture.graph)
                expected = build_schedule(endpoints[:n], num_bodies, width)
                expected_counts = np.array([len(s) for s in expected.body_slabs])
                np.testing.assert_array_equal(copies.count_per_node.numpy(), expected_counts)
                np.testing.assert_array_equal(copies.section_end.numpy(), np.cumsum(expected_counts))
                slots = int(copies.highest_index_in_use.numpy()[0])
                np.testing.assert_array_equal(
                    copies.partition_list.numpy()[:slots], [slab for slabs in expected.body_slabs for slab in slabs]
                )
                count = int(data["num_colors"].numpy()[0])
                self.assertEqual(count, len(expected.colors))
                np.testing.assert_array_equal(data["row_color"].numpy()[:n], expected.row_color)
                np.testing.assert_array_equal(data["row_slab"].numpy()[:n], expected.row_slab)
                np.testing.assert_array_equal(data["ids"].numpy()[:n], [r for c in expected.colors for r in c])
                np.testing.assert_array_equal(
                    data["starts"].numpy()[: count + 1], np.r_[0, np.cumsum([len(c) for c in expected.colors])]
                )

    def test_repeated_edges_cross_color_word_boundary(self):
        """Bit masks retain repeated rows beyond 32 and 64 colors."""
        edges = np.array([[0, 1]] * 70 + [[2, 3], [-1, 4], [1, 3]], np.int32)
        self.check_graph(edges, 5)

    def test_saved_full_colibri_graph(self):
        """Match every active full-scene endpoint row from the saved audit."""
        path = Path("/tmp/colibri_splitprep_trace_reference.npz")
        if not path.exists():
            self.skipTest("Local full-scene graph snapshot is unavailable")
        snapshot = np.load(path)
        n = int(snapshot["0_before_active_count"][0])
        self.check_graph(snapshot["0_before_constraint_bodies"][:n], int(snapshot["0_before_dimensions"][0]))


if __name__ == "__main__":
    unittest.main()
