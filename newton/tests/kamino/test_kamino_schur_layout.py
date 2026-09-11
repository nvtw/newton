# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for packed row-major Schur responses."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import _assemble_compact_unilateral_schur_tiled


class TestSchurLayout(unittest.TestCase):
    def test_dynamic_rhs_graph_replay(self):
        """Preserve world boundaries as compact RHS counts change in a captured graph."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Tiled Schur construction requires CUDA")
        joints = np.array([3, 0, 129], dtype=np.int32)
        capacity = np.array([8, 2, 160], dtype=np.int32)
        sizes = joints * capacity
        offsets = np.array([0, sizes[0], sizes[0]], dtype=np.int32)
        vector_offsets = np.array([0, 11, 13], dtype=np.int32)
        total_vectors = 13 + 129 + 160

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        dimensions = ints(joints)
        response = wp.zeros(int(sizes.sum()), dtype=wp.float32, device=device)
        schur = wp.empty_like(response)
        correction = wp.empty(total_vectors, dtype=wp.float32, device=device)
        inputs = [
            dimensions,
            ints(joints),
            ints(vector_offsets),
            ints(offsets),
            ints(capacity),
            response,
            schur,
            correction,
        ]
        graphs = []
        for groups in (8, 16, 128):
            wp.launch(
                _assemble_compact_unilateral_schur_tiled,
                dim=(3, groups, 128),
                inputs=[*inputs, groups],
                block_dim=128,
                device=device,
            )
            with wp.ScopedCapture(device=device) as capture:
                wp.launch(
                    _assemble_compact_unilateral_schur_tiled,
                    dim=(3, groups, 128),
                    inputs=[*inputs, groups],
                    block_dim=128,
                    device=device,
                )
            graphs.append(capture.graph)

        rng = np.random.default_rng(941)
        for counts in ([4, 0, 129], [0, 0, 0], [1, 1, 130], [5, 0, 160], [3, 0, 7], [4, 0, 129]):
            dimensions.assign(joints + np.asarray(counts, dtype=np.int32))
            source = np.full(response.size, -111.0, dtype=np.float32)
            expected = np.full(response.size, -222.0, dtype=np.float32)
            expected_correction = np.full(total_vectors, -333.0, dtype=np.float32)
            for world, (n, nu, size) in enumerate(zip(joints, counts, sizes, strict=True)):
                offset = offsets[world]
                white = rng.normal(0.0, 0.01, (n, nu)).astype(np.float32)
                source[offset : offset + n * nu] = white.ravel()
                if nu * nu <= size:
                    expected[offset : offset + nu * nu] = (white.astype(np.float64).T @ white).ravel()
                    start = vector_offsets[world] + n
                    expected_correction[start : start + nu] = 0.0
            response.assign(source)
            reference = None
            for graph in graphs:
                schur.fill_(-222.0)
                correction.fill_(-333.0)
                wp.capture_launch(graph)
                result = schur.numpy()
                np.testing.assert_allclose(result, expected, atol=1.0e-8, rtol=2.0e-6)
                np.testing.assert_array_equal(correction.numpy(), expected_correction)
                if reference is not None:
                    np.testing.assert_array_equal(result, reference)
                reference = result


if __name__ == "__main__":
    unittest.main()
