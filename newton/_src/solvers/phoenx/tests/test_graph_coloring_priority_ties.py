# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Equal packed priorities must not create simultaneous endpoint writers."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import IncrementalContactPartitioner
from newton._src.solvers.phoenx.tests.test_graph_coloring_speculative import _make_elements


class TestGraphColoringPriorityTies(unittest.TestCase):
    def test_repeated_pair_priority_ties(self):
        """Give repeated pair edges distinct colors even with identical priorities."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        device = "cuda:0"
        size = 16
        bodies = np.full((size, 8), -1, dtype=np.int32)
        bodies[:, :2] = [0, 1]
        elements = _make_elements(bodies, device)
        count = wp.array([size], dtype=wp.int32, device=device)
        for algorithm in ("jp", "greedy", "speculative", "endpoint_owner"):
            with self.subTest(algorithm=algorithm):
                partitioner = IncrementalContactPartitioner(
                    max_num_interactions=size,
                    max_num_nodes=2,
                    device=device,
                    enable_warm_start=False,
                    max_colored_partitions=32,
                    endpoint_owner_coloring=algorithm == "endpoint_owner",
                )
                partitioner._packed_priorities.fill_(123)
                partitioner.set_speculative_coloring(algorithm == "speculative")

                def build(partitioner=partitioner, algorithm=algorithm):
                    partitioner.reset(elements, count)
                    if algorithm == "jp":
                        partitioner.build_csr()
                    elif algorithm == "endpoint_owner":
                        partitioner.build_csr_endpoint_owner()
                    else:
                        partitioner.build_csr_greedy_with_jp_fallback()

                build()
                with wp.ScopedCapture(device=device) as capture:
                    build()
                previous = None
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    starts = partitioner.color_starts.numpy()
                    colors = int(partitioner.num_colors.numpy()[0])
                    ids = partitioner.element_ids_by_color.numpy()
                    for color in range(colors):
                        self.assertLessEqual(
                            int(starts[color + 1] - starts[color]),
                            1,
                            f"{algorithm}: repeated pair edges share color {color}",
                        )
                    self.assertEqual(int(starts[colors]), size)
                    np.testing.assert_array_equal(np.sort(ids[:size]), np.arange(size))
                    if previous is not None:
                        np.testing.assert_array_equal(ids[:size], previous)
                    previous = ids[:size].copy()


if __name__ == "__main__":
    unittest.main()
