# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mas_pcg import BatchedBSRMatrix, BatchedMASPCG


def make_block_laplacian(node_count: int, diagonal_shift: float = 0.25):
    """Build a float32 1D block Laplacian in BSR and dense forms."""
    rows = [0]
    cols = []
    values = []
    identity = np.eye(3, dtype=np.float32)
    for row in range(node_count):
        if row > 0:
            cols.append(row - 1)
            values.append(-identity)
        cols.append(row)
        values.append((2.0 + diagonal_shift) * identity)
        if row + 1 < node_count:
            cols.append(row + 1)
            values.append(-identity)
        rows.append(len(cols))

    dense = np.zeros((3 * node_count, 3 * node_count), dtype=np.float32)
    for row in range(node_count):
        for block in range(rows[row], rows[row + 1]):
            col = cols[block]
            dense[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] = values[block]
    return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32), np.asarray(values), dense


def extract_symmetric_upper(rows, cols, values):
    """Extract upper-triangle blocks from a symmetric BSR matrix."""
    upper_rows = [0]
    upper_cols = []
    upper_values = []
    for row in range(rows.size - 1):
        for block in range(int(rows[row]), int(rows[row + 1])):
            if int(cols[block]) >= row:
                upper_cols.append(int(cols[block]))
                upper_values.append(values[block])
        upper_rows.append(len(upper_cols))
    return np.asarray(upper_rows, dtype=np.int32), np.asarray(upper_cols), np.asarray(upper_values)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for tile Cholesky")
class TestMASPCG(unittest.TestCase):
    def test_symmetric_upper_storage_matches_dense_and_captures(self):
        """Mirror nonsymmetric off-diagonal blocks in SpMV, MAS, and PCG."""
        diagonal_0 = np.diag(np.asarray([2.0, 2.5, 3.0], dtype=np.float32))
        diagonal_1 = np.diag(np.asarray([2.25, 2.75, 3.25], dtype=np.float32))
        coupling = np.asarray(
            [[-0.2, 0.05, 0.0], [0.01, -0.1, 0.03], [0.0, 0.02, -0.15]],
            dtype=np.float32,
        )
        rows = np.asarray([0, 2, 3], dtype=np.int32)
        cols = np.asarray([0, 1, 1], dtype=np.int32)
        values = np.asarray([diagonal_0, coupling, diagonal_1], dtype=np.float32)
        dense = np.block([[diagonal_0, coupling], [coupling.T, diagonal_1]])
        matrix = BatchedBSRMatrix.from_host(
            [rows],
            [cols],
            [values],
            symmetric_upper=True,
            device="cuda:0",
        )
        rhs_host = np.linspace(-0.75, 1.0, 6, dtype=np.float32)
        rhs = wp.array(rhs_host, device=matrix.device)
        product = wp.zeros_like(rhs)
        active = wp.full(1, True, dtype=wp.bool, device=matrix.device)

        matrix.gemv(rhs, product, active)
        np.testing.assert_allclose(product.numpy(), dense @ rhs_host, rtol=2.0e-6, atol=2.0e-6)

        x = wp.zeros_like(rhs)
        solver = BatchedMASPCG(
            matrix,
            rtol=1.0e-6,
            max_iterations=30,
            use_cuda_graph=True,
            fused_spmv_min_rows=0,
        )
        graph = solver.capture(rhs, x)
        wp.capture_launch(graph)
        np.testing.assert_allclose(x.numpy(), np.linalg.solve(dense, rhs_host), rtol=2.0e-5, atol=2.0e-5)

    def test_symmetric_upper_dynamic_assembly_captures(self):
        """Rebuild upper-triangle contact-like blocks inside the solve graph."""
        full_rows, full_cols, full_values, dense = make_block_laplacian(9)
        rows, cols, values = extract_symmetric_upper(full_rows, full_cols, full_values)
        capacities = np.diff(rows) + 2
        matrix = BatchedBSRMatrix.from_host(
            [rows],
            [cols],
            [values],
            row_capacities=[capacities],
            symmetric_upper=True,
            device="cuda:0",
        )
        input_rows = np.repeat(np.arange(9, dtype=np.int32), np.diff(rows))
        rows_gpu = wp.array(input_rows, device=matrix.device)
        cols_gpu = wp.array(cols, dtype=wp.int32, device=matrix.device)
        values_gpu = wp.array(values, dtype=wp.mat33f, device=matrix.device)
        count_gpu = wp.array([len(cols)], dtype=wp.int32, device=matrix.device)
        rhs_host = np.linspace(-0.5, 0.75, 27, dtype=np.float32)
        rhs = wp.array(rhs_host, device=matrix.device)
        x = wp.zeros_like(rhs)
        solver = BatchedMASPCG(matrix, rtol=2.0e-5, max_iterations=80, use_cuda_graph=True)

        with wp.ScopedCapture(matrix.device) as capture:
            matrix.begin_assembly()
            matrix.insert_blocks(rows_gpu, cols_gpu, values_gpu, count_gpu)
            x.zero_()
            solver.solve(rhs, x)
        wp.capture_launch(capture.graph)

        np.testing.assert_allclose(x.numpy(), np.linalg.solve(dense, rhs_host), rtol=3.0e-4, atol=3.0e-4)
        self.assertEqual(int(matrix.overflow.numpy()[0]), 0)

        matrix.begin_assembly()
        matrix.insert_blocks(
            wp.array([1], dtype=wp.int32, device=matrix.device),
            wp.array([0], dtype=wp.int32, device=matrix.device),
            wp.array([np.eye(3, dtype=np.float32)], dtype=wp.mat33f, device=matrix.device),
            wp.array([1], dtype=wp.int32, device=matrix.device),
        )
        self.assertEqual(int(matrix.overflow.numpy()[0]), 1)

    def test_batched_solve_matches_dense_reference(self):
        """Solve varied world sizes to the requested float32 tolerance."""
        systems = [make_block_laplacian(size) for size in (7, 19, 65)]
        matrix = BatchedBSRMatrix.from_host(
            [system[0] for system in systems],
            [system[1] for system in systems],
            [system[2] for system in systems],
            device="cuda:0",
        )
        rng = np.random.default_rng(123)
        rhs_worlds = [rng.normal(size=system[3].shape[0]).astype(np.float32) for system in systems]
        rhs_host = np.concatenate(rhs_worlds)
        rhs = wp.array(rhs_host, device=matrix.device)
        x = wp.zeros_like(rhs)
        solver = BatchedMASPCG(matrix, rtol=2.0e-5, atol=1.0e-6, max_iterations=150, use_cuda_graph=False)

        solver.solve(rhs, x, refit=False)
        result = x.numpy()

        offset = 0
        for system, world_rhs in zip(systems, rhs_worlds, strict=True):
            expected = np.linalg.solve(system[3], world_rhs)
            actual = result[offset : offset + expected.size]
            np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-4)
            offset += expected.size

    def test_capacity_storage_accepts_dynamic_pattern(self):
        """Reassemble a changed sparse pattern without reallocating matrix arrays."""
        rows, cols, values, dense = make_block_laplacian(11)
        capacities = np.diff(rows) + 2
        matrix = BatchedBSRMatrix.from_host([rows], [cols], [values], row_capacities=[capacities], device="cuda:0")
        input_rows = np.repeat(np.arange(11, dtype=np.int32), np.diff(rows))
        input_count = wp.array([len(cols)], dtype=wp.int32, device=matrix.device)
        matrix.begin_assembly()
        matrix.insert_blocks(
            wp.array(input_rows, device=matrix.device),
            wp.array(cols, device=matrix.device),
            wp.array(values, dtype=wp.mat33f, device=matrix.device),
            input_count,
        )
        x_host = np.arange(33, dtype=np.float32) / 17.0
        x = wp.array(x_host, device=matrix.device)
        y = wp.zeros_like(x)
        active = wp.full(1, True, dtype=wp.bool, device=matrix.device)

        matrix.gemv(x, y, active)

        np.testing.assert_allclose(y.numpy(), dense @ x_host, rtol=2.0e-6, atol=2.0e-6)
        self.assertEqual(int(matrix.overflow.numpy()[0]), 0)

    def test_solve_captures_as_one_cuda_graph(self):
        """Capture numeric MAS refit and device-conditional PCG together."""
        rows, cols, values, dense = make_block_laplacian(13)
        matrix = BatchedBSRMatrix.from_host([rows], [cols], [values], device="cuda:0")
        rhs_host = np.linspace(-1.0, 1.0, 39, dtype=np.float32)
        rhs = wp.array(rhs_host, device=matrix.device)
        x = wp.zeros_like(rhs)
        solver = BatchedMASPCG(matrix, rtol=2.0e-5, max_iterations=80, use_cuda_graph=True)

        graph = solver.capture(rhs, x)
        wp.capture_launch(graph)
        result = x.numpy()

        np.testing.assert_allclose(result, np.linalg.solve(dense, rhs_host), rtol=3.0e-4, atol=3.0e-4)

    def test_dynamic_assembly_and_solve_capture_together(self):
        """Capture changing BSR assembly, MAS refit, and PCG without host synchronization."""
        rows, cols, values, dense = make_block_laplacian(9)
        matrix = BatchedBSRMatrix.from_host(
            [rows],
            [cols],
            [values],
            row_capacities=[np.diff(rows) + 2],
            device="cuda:0",
        )
        input_rows = np.repeat(np.arange(9, dtype=np.int32), np.diff(rows))
        rows_gpu = wp.array(input_rows, device=matrix.device)
        cols_gpu = wp.array(cols, device=matrix.device)
        values_gpu = wp.array(values, dtype=wp.mat33f, device=matrix.device)
        count_gpu = wp.array([len(cols)], dtype=wp.int32, device=matrix.device)
        rhs_host = np.linspace(0.25, 1.25, 27, dtype=np.float32)
        rhs = wp.array(rhs_host, device=matrix.device)
        x = wp.zeros_like(rhs)
        solver = BatchedMASPCG(matrix, rtol=2.0e-5, max_iterations=80, use_cuda_graph=True)

        with wp.ScopedCapture(matrix.device) as capture:
            matrix.begin_assembly()
            matrix.insert_blocks(rows_gpu, cols_gpu, values_gpu, count_gpu)
            x.zero_()
            solver.solve(rhs, x)
        wp.capture_launch(capture.graph)
        result = x.numpy()

        np.testing.assert_allclose(result, np.linalg.solve(dense, rhs_host), rtol=3.0e-4, atol=3.0e-4)
        self.assertEqual(int(matrix.overflow.numpy()[0]), 0)

    def test_tensor_core_preconditioners_converge_in_capture(self):
        """Keep experimental local apply modes accurate under captured FP32 PCG."""
        rows, cols, values, dense = make_block_laplacian(19, diagonal_shift=0.05)
        rhs_host = np.linspace(-1.0, 1.0, 57, dtype=np.float32)
        expected = np.linalg.solve(dense, rhs_host)
        for apply_mode in ("fp32_async", "fp32_tile", "bf16_tile", "bf16_vector"):
            with self.subTest(apply_mode=apply_mode):
                matrix = BatchedBSRMatrix.from_host([rows], [cols], [values], device="cuda:0")
                rhs = wp.array(rhs_host, device=matrix.device)
                x = wp.zeros_like(rhs)
                solver = BatchedMASPCG(
                    matrix,
                    rtol=2.0e-5,
                    max_iterations=100,
                    use_cuda_graph=True,
                    mas_apply_mode=apply_mode,
                )

                graph = solver.capture(rhs, x)
                wp.capture_launch(graph)

                np.testing.assert_allclose(x.numpy(), expected, rtol=5.0e-4, atol=5.0e-4)


if __name__ == "__main__":
    unittest.main()
