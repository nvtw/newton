# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise fixed structural masks without changing numerical RCM ordering."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver


@wp.kernel
def _assemble(
    source: wp.array[wp.float32],
    inverse: wp.array[wp.int32],
    permuted: bool,
    n: int,
    target: wp.array[wp.float32],
):
    row, col = wp.tid()
    target_row = row
    target_col = col
    if permuted:
        target_row = inverse[row]
        target_col = inverse[col]
    target[target_row * n + target_col] = source[row * n + col]


class TestKaminoSparseAssemblyLifecycle(unittest.TestCase):
    def test_capture_reset_and_changing_values(self):
        """Preserve numeric ordering through first use, resets, and changing values."""
        if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional CUDA graphs are required")
        device = wp.get_device("cuda:0")
        n = 65
        first = np.eye(n, dtype=np.float32) * 4.0
        for row in range(n - 1):
            first[row, row + 1] = first[row + 1, row] = -0.25
        second = first.copy()
        second[0, -1] = second[-1, 0] = 0.5
        second[1, 40] = second[40, 1] = -0.125
        rows, cols = np.nonzero(np.triu(second, 1))
        rhs_np = np.random.default_rng(78).normal(size=n).astype(np.float32)
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                info = DenseSquareMultiLinearInfo()
                info.finalize(dimensions=[n], dtype=wp.float32, device=device)
                matrix = wp.zeros(n * n, dtype=wp.float32, device=device)
                source = wp.array(first.ravel(), dtype=wp.float32, device=device)
                solver = LLTBlockedRCMSolver(
                    operator=DenseLinearOperatorData(info=info, mat=matrix),
                    parallel_factorization=parallel,
                    device=device,
                )
                solver.configure_sparse_assembly(
                    wp.zeros(len(rows), dtype=wp.int32, device=device),
                    wp.array(rows, dtype=wp.int32, device=device),
                    wp.array(cols, dtype=wp.int32, device=device),
                )
                rhs = wp.array(rhs_np, dtype=wp.float32, device=device)
                solution = wp.zeros_like(rhs)

                def assemble(target, inverse, source=source, info=info):
                    wp.launch(
                        _assemble,
                        dim=(n, n),
                        inputs=[source, info.vio if inverse is None else inverse, inverse is not None, n, target],
                        device=device,
                    )

                def capture(solver=solver, assemble=assemble, rhs=rhs, solution=solution):
                    with wp.ScopedCapture(device=device) as captured:
                        for _ in range(2):
                            solver.compute_sparse(assemble)
                            solver.solve(rhs, solution)
                    return captured.graph

                def check(expected, fresh_order=False, solution=solution, solver=solver, info=info, source=source):
                    actual = solution.numpy()
                    np.testing.assert_allclose(actual, np.linalg.solve(expected.astype(np.float64), rhs_np), atol=1e-6)
                    p = solver.P.numpy()
                    np.testing.assert_array_equal(np.sort(p), np.arange(n))
                    np.testing.assert_array_equal(solver.inv_P.numpy()[p], np.arange(n))
                    if fresh_order:
                        reference = LLTBlockedRCMSolver(
                            operator=DenseLinearOperatorData(info=info, mat=source), device=device
                        )
                        reference.compute(source)
                        np.testing.assert_array_equal(p, reference.P.numpy())

                discarded = capture()
                del discarded
                graph = capture()
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                # Values may change after capture but before its first replay.
                source.assign(second.ravel())
                wp.capture_launch(graph)
                check(second, fresh_order=True)
                source.assign(first.ravel())
                wp.capture_launch(graph)
                check(first)
                solver.reset()
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                wp.capture_launch(graph)
                check(first, fresh_order=True)
                source.assign(second.ravel())
                solver.compute_sparse(assemble)
                self.assertEqual(int(solver._structural_ready.numpy()[0]), 0)
                wp.capture_launch(graph)
                check(second)
                # Recapture after initialization, then reset and replay it.
                replacement = capture()
                solver.reset()
                wp.capture_launch(replacement)
                check(second, fresh_order=True)


if __name__ == "__main__":
    unittest.main()
