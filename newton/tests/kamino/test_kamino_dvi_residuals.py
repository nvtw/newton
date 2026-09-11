# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cooperative terminal DVI residual reductions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.kernels import _compute_dvi_status_residuals
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus


class TestDVIResiduals(unittest.TestCase):
    def test_cooperative_matches_serial(self):
        """Preserve all status fields across ragged worlds and constraint types."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA warp reductions")
        device = wp.get_cuda_devices()[0]
        rng = np.random.default_rng(916)
        counts = np.array(
            [(137, 53, 9, 7), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 3, 5), (65, 33, 0, 0), (3, 1, 1, 1)], dtype=np.int32
        )
        njc, nbc, nl, nc = counts.T
        dimensions = njc + nbc + nl + 3 * nc
        vector_offsets = np.concatenate(([0], np.cumsum(dimensions)[:-1]))
        bounded_offsets = np.concatenate(([0], np.cumsum(nbc)[:-1]))
        contact_offsets = np.concatenate(([0], np.cumsum(nc)[:-1]))
        size = int(dimensions.sum())

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(values, dtype=wp.float32, device=device)

        configs = []
        statuses = []
        for index in range(len(counts)):
            config = DVIConfigStruct()
            config.tolerance = 1.0e-5
            configs.append(config)
            status = DVIStatus()
            status.iterations = index
            statuses.append(status)
        initial_status = wp.array(statuses, dtype=DVIStatus, device=device)
        inputs = [
            ints(dimensions),
            ints(vector_offsets),
            ints(njc),
            ints(nbc),
            ints(nl),
            ints(nc),
            ints(njc),
            ints(njc + nbc),
            ints(njc + nbc + nl),
            ints(bounded_offsets),
            ints(contact_offsets),
            floats(rng.uniform(0, 1, int(nc.sum()))),
            floats(-rng.uniform(0, 2, int(nbc.sum()))),
            floats(rng.uniform(0, 2, int(nbc.sum()))),
            wp.array(configs, dtype=DVIConfigStruct, device=device),
        ]
        for converged in (False, True):
            velocities = np.zeros(size, dtype=np.float32) if converged else rng.normal(size=size).astype(np.float32)
            impulses = np.zeros(size, dtype=np.float32) if converged else rng.normal(size=size).astype(np.float32)
            outputs = []
            for workers in (1, 32):
                status = wp.clone(initial_status)
                wp.launch(
                    _compute_dvi_status_residuals,
                    dim=len(counts) * workers,
                    inputs=[*inputs, floats(velocities), floats(impulses), status, workers],
                    block_dim=128,
                    device=device,
                )
                outputs.append(status.numpy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
            if converged:
                np.testing.assert_array_equal(outputs[1]["converged"], 1)
                np.testing.assert_array_equal(outputs[1]["r_b"], 0)
                np.testing.assert_array_equal(outputs[1]["r_p"], 0)
                np.testing.assert_array_equal(outputs[1]["r_d"], 0)
                np.testing.assert_array_equal(outputs[1]["r_c"], 0)
            else:
                # The single bilateral row has an analytic residual, including
                # when its world shares a CUDA block with empty/contact worlds.
                residual = abs(velocities[vector_offsets[2]])
                self.assertEqual(outputs[1]["r_b"][2], residual)
                self.assertEqual(outputs[1]["r_d"][2], residual)
