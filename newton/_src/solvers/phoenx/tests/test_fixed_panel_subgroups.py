# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check compact mechanism solves against independent dense references."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT


@unittest.skipUnless(wp.is_cuda_available(), "CUDA subgroups required")
class TestFixedPanelSubgroups(unittest.TestCase):
    def _check_batch(self, width, dimensions):
        rng = np.random.default_rng(42)
        starts = np.concatenate(([0], np.cumsum(dimensions))).astype(np.int32)
        permutation = np.concatenate([rng.permutation(d) for d in dimensions]).astype(np.int32)
        rows = tuple(frozenset((0, 1)) for _ in range(starts[-1]))
        solvers = [
            FixedPatternPanelLLT(
                dimensions,
                starts,
                permutation,
                rows,
                block_size=width,
                device="cuda:0",
                use_subgroup_solve=enabled,
            )
            for enabled in (False, True)
        ]
        factors = []
        for d in dimensions:
            a = rng.normal(size=(d, d))
            factors.append(np.linalg.cholesky(a @ a.T + 0.5 * np.eye(d)).astype(np.float32))
        for solver in solvers:
            packed = np.zeros(solver.factor.shape, dtype=np.float32)
            tables = solver.panel_index.numpy()
            offsets = solver.panel_table_offset.numpy()
            for m, (d, lower) in enumerate(zip(dimensions, factors, strict=True)):
                count = (d + width - 1) // width
                padded = np.eye(count * width, dtype=np.float32)
                padded[:d, :d] = lower
                for i in range(count):
                    for j in range(i + 1):
                        panel = tables[offsets[m] + i * count + j]
                        self.assertGreaterEqual(panel, 0)
                        block = padded[i * width : (i + 1) * width, j * width : (j + 1) * width]
                        packed[panel * width * width : (panel + 1) * width * width] = block.ravel()
            solver.factor.assign(packed)
        rhs_host = rng.normal(size=starts[-1]).astype(np.float32)
        rhs = wp.array(rhs_host, device="cuda:0")
        results = []
        for solver in solvers:
            result = wp.zeros_like(rhs)
            solver.solve(rhs, result)
            host = result.numpy()
            with wp.ScopedCapture(device="cuda:0") as capture:
                solver.solve(rhs, result)
            for _ in range(4):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(result.numpy(), host)
            results.append(host)
        np.testing.assert_array_equal(results[1], results[0])
        for m, lower in enumerate(factors):
            begin, end = starts[m : m + 2]
            order = permutation[begin:end]
            matrix = lower.astype(float) @ lower.astype(float).T
            expected = np.linalg.solve(matrix, rhs_host[begin:end][order])
            np.testing.assert_allclose(results[1][begin:end][order], expected, rtol=3e-5, atol=3e-6)

    def test_variable_partial_subgroups(self):
        for width, sizes in ((16, (3, 5, 6, 9, 12, 15, 16)), (32, (3, 15, 16, 17, 21, 31, 32))):
            with self.subTest(width=width):
                # 35 systems cover several blocks and a partial last block.
                self._check_batch(width, sizes * 5)

    def test_mixed_panel_sizes(self):
        for width in (16, 32):
            with self.subTest(width=width):
                self._check_batch(width, (3, width - 1, width, width + 1, 2 * width + 3))


if __name__ == "__main__":
    unittest.main()
