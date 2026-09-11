# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise packed storage through the existing RCM solver lifecycle and API."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.core import DenseLinearOperatorData, DenseSquareMultiLinearInfo
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from newton._src.solvers.kamino._src.linalg.factorize.llt_packed import _PackedLLT, packed_element_offset


@wp.kernel
def _assemble_permuted(
    dimensions: wp.array[wp.int32],
    matrix_offsets: wp.array[wp.int32],
    vector_offsets: wp.array[wp.int32],
    slots: wp.array[wp.int64],
    inverse: wp.array[wp.int32],
    source: wp.array[wp.float32],
    target: wp.array[wp.float32],
):
    world, element = wp.tid()
    n = dimensions[world]
    if element >= n * n:
        return
    row = inverse[vector_offsets[world] + element // n]
    col = inverse[vector_offsets[world] + element % n]
    if row // 32 >= col // 32:
        target[packed_element_offset(slots[world], row, col)] = source[matrix_offsets[world] + element]


class TestPackedSolver(unittest.TestCase):
    def _fixture(self):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Packed sparse assembly lifecycle requires CUDA conditional graphs")
        dimensions = [33, 65]
        info = DenseSquareMultiLinearInfo()
        info.finalize(dimensions=dimensions, dtype=wp.float32, device=device)
        matrix = wp.zeros(info.total_mat_size, dtype=wp.float32, device=device)
        solver = LLTBlockedRCMSolver(operator=DenseLinearOperatorData(info=info, mat=matrix), device=device)
        pairs = [(w, row, col) for w, n in enumerate(dimensions) for row in range(n) for col in range(row)]
        metadata = [wp.array(values, dtype=wp.int32, device=device) for values in zip(*pairs, strict=True)]
        solver.configure_sparse_assembly(*metadata)
        # Force only storage selection in this small fixture; retain the real solver dispatch and lifecycle.
        solver._packed = _PackedLLT(
            dimensions, device, solver.tile_pattern, solver.tile_pattern_offsets, solver.P, info.vio
        )
        solver._packed_matrix = wp.zeros(solver._packed.factor_size, dtype=wp.float32, device=device)
        solver._packed_factor = wp.zeros_like(solver._packed_matrix)
        return solver, info, matrix

    @staticmethod
    def _inputs(info, phase):
        rng = np.random.default_rng(274 + phase)
        matrices = []
        for n in info.dimensions:
            basis = rng.normal(size=(n, n)) / np.sqrt(n)
            if phase % 2:
                basis = np.diag(np.diag(basis))
            matrices.append((basis @ basis.T + 0.5 * np.eye(n)).astype(np.float32))
        rhs = rng.normal(size=info.total_vec_size).astype(np.float32)
        return matrices, rhs

    def _assert_solution(self, info, matrices, rhs, solution):
        offsets = info.vio.numpy()
        for world, matrix in enumerate(matrices):
            segment = slice(offsets[world], offsets[world] + len(matrix))
            expected = np.linalg.solve(matrix.astype(np.float64), rhs[segment].astype(np.float64))
            np.testing.assert_allclose(solution[segment], expected, atol=2e-5, rtol=2e-5)

    def _assert_factor(self, solver, info, matrices):
        factors, orders = solver.L.numpy(), solver.P.numpy()
        for world, matrix in enumerate(matrices):
            n = len(matrix)
            mio, vio = info.mio.numpy()[world], info.vio.numpy()[world]
            order = orders[vio : vio + n]
            lower = np.tril(factors[mio : mio + n * n].reshape(n, n)).astype(np.float64)
            reference = matrix[np.ix_(order, order)].astype(np.float64)
            np.testing.assert_allclose(lower @ lower.T, reference, atol=2e-6, rtol=2e-5)

    def test_intermediate_getter_after_solve(self):
        """Expose the current unpadded forward-substitution result through solver.y."""
        solver, info, matrix = self._fixture()
        rhs = wp.zeros(info.total_vec_size, dtype=wp.float32, device=solver.device)
        result = wp.zeros_like(rhs)
        for phase in range(2):
            matrices, values = self._inputs(info, phase)
            matrix.assign(np.concatenate([a.ravel() for a in matrices]))
            rhs.assign(values)
            solver.compute(matrix)
            solver.solve(rhs, result)
            self._assert_solution(info, matrices, values, result.numpy())
            self._assert_factor(solver, info, matrices)
            intermediate, orders = solver.y.numpy(), solver.P.numpy()
            for world, a in enumerate(matrices):
                offset = info.vio.numpy()[world]
                segment = slice(offset, offset + len(a))
                order = orders[segment]
                lower = np.linalg.cholesky(a[np.ix_(order, order)].astype(np.float64))
                expected = np.linalg.solve(lower, values[segment][order].astype(np.float64))
                with self.subTest(phase=phase, world=world):
                    np.testing.assert_allclose(intermediate[segment], expected, atol=2e-5, rtol=2e-5)

    def test_captured_sparse_reset_and_reuse(self):
        """Reinitialize packed sparse assembly after reset and discarded initialization captures."""
        solver, info, matrix = self._fixture()
        source = wp.zeros_like(matrix)
        rhs = wp.zeros(info.total_vec_size, dtype=wp.float32, device=solver.device)
        result = wp.zeros_like(rhs)

        def assemble(target, inverse):
            if inverse is None:
                wp.copy(target, source)
            else:
                target.zero_()
                wp.launch(
                    _assemble_permuted,
                    dim=(info.num_blocks, info.max_dimension * info.max_dimension),
                    inputs=[info.dim, info.mio, info.vio, solver._packed.slot_offsets, inverse, source, target],
                    device=solver.device,
                )

        def run():
            solver.compute_sparse(assemble)
            solver.solve(rhs, result)

        matrices, values = self._inputs(info, 0)
        source.assign(np.concatenate([a.ravel() for a in matrices]))
        rhs.assign(values)
        run()
        solver.reset()
        with wp.ScopedCapture(device=solver.device):
            run()
        np.testing.assert_array_equal(solver._structural_ready.numpy(), [0])
        with wp.ScopedCapture(device=solver.device) as capture:
            run()
        for phase in range(4):
            if phase == 2:
                solver.reset()
                np.testing.assert_array_equal(solver._structural_ready.numpy(), [0])
            matrices, values = self._inputs(info, phase)
            source.assign(np.concatenate([a.ravel() for a in matrices]))
            rhs.assign(values)
            wp.capture_launch(capture.graph)
            self._assert_solution(info, matrices, values, result.numpy())
            self._assert_factor(solver, info, matrices)
            np.testing.assert_array_equal(solver._structural_ready.numpy(), [1])
            np.testing.assert_array_equal(solver._packed.errors.numpy(), [0])
        # Eager stepping must still consume the original-order callback input after graph reuse.
        matrices, values = self._inputs(info, 4)
        source.assign(np.concatenate([a.ravel() for a in matrices]))
        rhs.assign(values)
        run()
        self._assert_solution(info, matrices, values, result.numpy())
        self._assert_factor(solver, info, matrices)


if __name__ == "__main__":
    unittest.main()
