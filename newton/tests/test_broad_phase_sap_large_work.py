# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.broad_phase_sap import _sap_select_sweep_kernel
from newton.geometry import BroadPhaseSAP


class TestSAPLargeWork(unittest.TestCase):
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
