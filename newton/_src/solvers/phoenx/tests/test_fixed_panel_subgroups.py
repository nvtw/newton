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
            if solver.subgroup_packed:
                # Packed subgroup solves read task-ordered lower triangles.
                triangle = width * (width + 1) // 2
                packed_lower = np.zeros(solver.packed_factor.shape, dtype=np.float32)
                for task, m in enumerate(solver.subgroup_mechanism.numpy()):
                    padded = np.eye(width, dtype=np.float32)
                    padded[: dimensions[m], : dimensions[m]] = factors[m]
                    rows, columns = np.tril_indices(width)
                    packed_lower[task * triangle : (task + 1) * triangle] = padded[rows, columns]
                solver.packed_factor.assign(packed_lower)
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

    def _multi_panel_errors(self, width, dimensions, chain):
        """Return the warp and tile solves' worst (backward, forward) errors over random right-hand sides."""
        rng = np.random.default_rng(width + len(dimensions))
        starts = np.concatenate(([0], np.cumsum(dimensions))).astype(np.int32)
        permutation = np.concatenate([rng.permutation(d) for d in dimensions]).astype(np.int32)
        # A chain of bodies leaves panels outside the band structurally absent.
        rows = tuple(frozenset((r, r + 1) if chain else (0, 1)) for r in range(starts[-1]))
        solver = FixedPatternPanelLLT(dimensions, starts, permutation, rows, block_size=width, device="cuda:0")
        multi_panel = [m for m, d in enumerate(dimensions) if d > width]
        np.testing.assert_array_equal(np.sort(solver._warp_solve_mechanism.numpy()), multi_panel)
        tables = solver.panel_index.numpy()
        offsets = solver.panel_table_offset.numpy()
        packed = np.zeros(solver.factor.shape, dtype=np.float32)
        factors = []
        for m, d in enumerate(dimensions):
            count = (d + width - 1) // width
            table = tables[offsets[m] : offsets[m] + count * count].reshape(count, count)
            # A random lower factor with the panel pattern; the pivots spread
            # over two decades so that the triangular solves amplify rounding.
            lower = np.tril(rng.normal(scale=0.3, size=(count * width, count * width)), -1)
            lower[np.diag_indices(count * width)] = np.geomspace(0.1, 10.0, count * width)[
                rng.permutation(count * width)
            ]
            for i in range(count):
                for j in range(i + 1):
                    block = lower[i * width : (i + 1) * width, j * width : (j + 1) * width]
                    if table[i, j] < 0:
                        block[:] = 0.0
                    else:
                        packed[table[i, j] * width * width : (table[i, j] + 1) * width * width] = block.ravel()
            factors.append(lower[:d, :d].astype(np.float32).astype(np.float64))
        solver.factor.assign(packed)
        rhs = wp.zeros(int(starts[-1]), dtype=wp.float32, device="cuda:0")
        result = wp.zeros_like(rhs)
        # Normwise backward error |b - L L^T x| / (|L| |L^T| |x| + |b|) and
        # forward error |x - x64| / |x64| against float64 with the same factor.
        errors = np.zeros((2, 2))
        for path, warp_path in enumerate((True, False)):
            if not warp_path:
                solver._tiled_solve_mechanism = solver.cooperative_mechanism
                solver._warp_solve_mechanism = wp.empty(0, dtype=wp.int32, device="cuda:0")
            with wp.ScopedCapture(device="cuda:0") as capture:
                solver.solve(rhs, result)
            sample_rng = np.random.default_rng(7)
            for _ in range(8):
                rhs_host = sample_rng.normal(size=starts[-1]).astype(np.float32)
                rhs.assign(rhs_host)
                wp.capture_launch(capture.graph)
                host = result.numpy()
                for m in multi_panel:
                    begin, end = starts[m : m + 2]
                    order = permutation[begin:end]
                    lower = factors[m]
                    b = rhs_host[begin:end][order].astype(float)
                    x = host[begin:end][order].astype(float)
                    expected = np.linalg.solve(lower.T, np.linalg.solve(lower, b))
                    norm = np.linalg.norm(lower, np.inf) * np.linalg.norm(lower.T, np.inf)
                    backward = np.max(np.abs(b - lower @ (lower.T @ x))) / (
                        norm * np.max(np.abs(x)) + np.max(np.abs(b))
                    )
                    forward = np.max(np.abs(x - expected)) / np.max(np.abs(expected))
                    errors[path] = np.maximum(errors[path], (backward, forward))
        return errors

    def test_warp_multi_panel_solves(self):
        """Multi-panel mechanisms solve on one warp as accurately as the tile solve."""
        cases = (
            (16, (17, 40, 54, 66, 81, 96, 128), False),
            (16, (20, 47, 64, 99, 112), True),
            (32, (33, 64, 75, 128), False),
            (32, (40, 96, 127), True),
        )
        for width, dimensions, chain in cases:
            with self.subTest(width=width, chain=chain):
                (warp_backward, warp_forward), (tile_backward, tile_forward) = self._multi_panel_errors(
                    width, dimensions, chain
                )
                # Both are backward stable far below float32 epsilon (about
                # 1e-9 here); forward errors of these ill-conditioned factors
                # (a few 1e-6) differ by rounding order only.
                self.assertLess(warp_backward, 1.0e-8)
                self.assertLess(tile_backward, 1.0e-8)
                self.assertLess(warp_forward, 2.0 * tile_forward)


if __name__ == "__main__":
    unittest.main()
