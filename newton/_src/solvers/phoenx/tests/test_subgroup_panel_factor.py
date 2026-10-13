# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The register subgroup factor must reproduce the tile factor's panels."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_llt import FixedPatternPanelLLT


def _factor_storage(panel: FixedPatternPanelLLT, subgroup: bool, rhs: wp.array) -> tuple[np.ndarray, np.ndarray]:
    device = panel.device
    if not subgroup:
        panel._tile_factor_mechanism = panel.cooperative_factor_mechanism
        panel.subgroup_factor_mechanism = wp.zeros(0, dtype=wp.int32, device=device)
    panel.factor.zero_()
    solution = wp.zeros_like(rhs)
    with wp.ScopedCapture(device) as capture:
        panel.compute()
        panel.solve(rhs, solution)
    wp.capture_launch(capture.graph)
    return panel.factor.numpy(), solution.numpy()


@unittest.skipUnless(wp.is_cuda_available(), "CUDA subgroup factor")
class TestSubgroupPanelFactor(unittest.TestCase):
    def test_matches_tile_factor_with_padding(self):
        """Full and padded panels match the tile factor, zero upper triangle and identity padding."""
        for block_size, pattern in ((16, (15, 16, 7, 1, 40)), (32, (31, 32, 9, 70))):
            with self.subTest(block_size=block_size):
                # Enough mechanisms for one block per mechanism instead of task queues.
                dimensions = pattern * 50
                starts = np.cumsum(np.asarray((0, *dimensions), dtype=np.int32))
                permutation = np.concatenate([np.arange(dimension, dtype=np.int32) for dimension in dimensions])
                row_bodies = tuple(
                    frozenset((1000 + mechanism,)) for mechanism, d in enumerate(dimensions) for _ in range(d)
                )
                rng = np.random.default_rng(2026 + block_size)
                matrices = []
                for dimension in dimensions:
                    lower = np.tril(rng.normal(scale=0.3, size=(dimension, dimension)), -1) + np.diag(
                        rng.uniform(0.5, 2.0, size=dimension)
                    )
                    matrices.append(lower @ lower.T)
                rhs_np = rng.normal(size=int(starts[-1])).astype(np.float32)
                results = []
                for subgroup in (True, False):
                    panel = FixedPatternPanelLLT(
                        dimensions, starts, permutation, row_bodies, block_size=block_size, device="cuda:0"
                    )
                    small = [m for m, d in enumerate(dimensions) if d <= block_size]
                    np.testing.assert_array_equal(np.sort(panel.subgroup_factor_mechanism.numpy()), small)
                    storage = np.zeros(panel.matrix.size, dtype=np.float32)
                    for row, column, address in zip(
                        panel.symbolic.matrix_row,
                        panel.symbolic.matrix_column,
                        panel.symbolic.matrix_storage,
                        strict=True,
                    ):
                        mechanism = int(np.searchsorted(starts[1:], row, side="right"))
                        storage[address] = matrices[mechanism][row - starts[mechanism], column - starts[mechanism]]
                    panel.matrix.assign(storage)
                    rhs = wp.array(rhs_np, dtype=wp.float32, device="cuda:0")
                    results.append(_factor_storage(panel, subgroup, rhs))
                (register_factor, register_solution), (tile_factor, tile_solution) = results
                np.testing.assert_allclose(register_factor, tile_factor, rtol=2.0e-4, atol=4.0e-6)
                np.testing.assert_array_equal(register_factor == 0.0, tile_factor == 0.0)
                for mechanism, matrix in enumerate(matrices):
                    begin, end = int(starts[mechanism]), int(starts[mechanism + 1])
                    rhs_slice = rhs_np[begin:end]
                    register = np.linalg.norm(matrix @ register_solution[begin:end] - rhs_slice)
                    tile = np.linalg.norm(matrix @ tile_solution[begin:end] - rhs_slice)
                    # Both float32 factors carry rounding; neither may be markedly worse.
                    self.assertLess(register, 2.0 * tile + 1.0e-5 * np.linalg.norm(rhs_slice), msg=f"{mechanism}")


if __name__ == "__main__":
    unittest.main()
