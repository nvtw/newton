# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check reusable RCM factors consumed as dense triangular matrices by DVI."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver


class TestKaminoFactorTileReuse(unittest.TestCase):
    def test_disappearing_tiles_are_zero(self):
        """Clear old off-diagonal values when a reused factor's sparsity shrinks."""
        if not wp.is_cuda_available():
            self.skipTest("Blocked factorization requires CUDA")
        device = wp.get_device("cuda:0")
        for parallel in (False, True):
            for n in (64, 65):
                with self.subTest(parallel=parallel, dimension=n):
                    rng = np.random.default_rng(32)
                    dense = rng.normal(size=(n, n)).astype(np.float32)
                    first = dense @ dense.T / n + np.eye(n, dtype=np.float32)
                    second = np.eye(n, dtype=np.float32) * 4.0
                    info = DenseSquareMultiLinearInfo()
                    info.finalize(dimensions=[n], dtype=wp.float32, device=device)
                    matrix = wp.array(first.ravel(), dtype=wp.float32, device=device)
                    operator = DenseLinearOperatorData(info=info, mat=matrix)
                    solver = LLTBlockedRCMSolver(
                        operator=operator,
                        block_size=32,
                        reuse_permutation=True,
                        parallel_factorization=parallel,
                        device=device,
                    )
                    solver.compute(matrix)
                    initial = np.tril(solver.L.numpy().reshape(n, n))
                    self.assertGreater(np.count_nonzero(initial[32:, :32]), 0)
                    matrix.assign(second.ravel())
                    if n == 65:
                        with wp.ScopedCapture(device=device) as capture:
                            solver.compute(matrix)
                        wp.capture_launch(capture.graph)
                    else:
                        solver.compute(matrix)
                    factor = np.tril(solver.L.numpy().reshape(n, n))
                    np.testing.assert_array_equal(factor, np.eye(n, dtype=np.float32) * 2.0)


if __name__ == "__main__":
    unittest.main()
