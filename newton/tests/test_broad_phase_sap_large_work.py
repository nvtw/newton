# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_sap import _sap_select_sweep_kernel
from newton.geometry import BroadPhaseSAP


class TestSAPLargeWork(unittest.TestCase):
    def test_conditional_capture_retains_overlapping_pairs(self):
        """Capture both 64-bit SAP scans without allocating inside a conditional body."""
        device = wp.get_device()
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional graph capture requires a supported CUDA device")
        if not hasattr(wp.utils, "array_scan_temp_storage_size"):
            self.skipTest("Conditional SAP capture requires Warp scan storage support")

        count = 4096  # Exercise the regular and adaptive chunk-count scans.
        centers = np.zeros((count, 3), dtype=np.float32)
        centers[:, 0] = np.arange(count) // 2
        lower = wp.array(centers - 0.1, dtype=wp.vec3, device=device)
        upper = wp.array(centers + 0.1, dtype=wp.vec3, device=device)
        groups = wp.ones(count, dtype=wp.int32, device=device)
        worlds = wp.zeros(count, dtype=wp.int32, device=device)
        pairs = wp.empty(count // 2, dtype=wp.vec2i, device=device)
        pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        enabled = wp.ones(1, dtype=wp.int32, device=device)
        broad_phase = BroadPhaseSAP(worlds, direction_search=False, device=device)

        def launch():
            broad_phase.launch(lower, upper, None, groups, worlds, count, pairs, pair_count, device=device)

        launch()
        pair_count.zero_()
        with wp.ScopedCapture(device=device) as capture:
            wp.capture_if(enabled, on_true=launch)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            actual_count = int(pair_count.numpy()[0])
            self.assertEqual(actual_count, count // 2)
            actual_pairs = {tuple(pair) for pair in pairs.numpy()[:actual_count]}
            self.assertEqual(actual_pairs, {(i, i + 1) for i in range(0, count, 2)})
        enabled.zero_()
        pair_count.zero_()
        wp.capture_launch(capture.graph)
        self.assertEqual(int(pair_count.numpy()[0]), 0)

    def test_dense_projection_keeps_more_than_two_billion_candidates(self):
        """A dense projection must not wrap its prefix sum and skip all work."""
        count = 65_537
        broad_phase = BroadPhaseSAP(np.full(count, -1, dtype=np.int32), device="cpu")
        ranges = np.arange(count - 1, -1, -1, dtype=np.int64)
        broad_phase.sap_range.assign(ranges)
        wp.utils.array_scan(broad_phase.sap_range, broad_phase.sap_cumulative_sum, inclusive=True)
        self.assertEqual(int(broad_phase.sap_cumulative_sum.numpy()[-1]), count * (count - 1) // 2)
        wp.launch(
            _sap_select_sweep_kernel,
            dim=1,
            inputs=[broad_phase.sap_cumulative_sum, count, broad_phase.sap_sweep_mode],
            device="cpu",
        )
        self.assertEqual(int(broad_phase.sap_sweep_mode.numpy()[0]), 1)


if __name__ == "__main__":
    unittest.main()
