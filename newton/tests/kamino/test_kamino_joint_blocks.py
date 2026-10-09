# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for joint-block solves and conditional Schur dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi import sparse
from newton._src.solvers.kamino._src.solvers.dvi.joint_blocks import JointBlockSolver, _build_topology


def _fixture(device):
    """Construct two padded worlds with interleaved five/six/five-row groups."""
    rng = np.random.default_rng(94017)
    worlds, n, ld, stride = 2, 16, 20, 9
    groups = [[0, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15]]

    def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def floats(values):
        return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=device)

    mio = np.array([3, 3 + ld * ld])
    vio = np.array([2, 2 + n + 3])
    rio = np.array([5, 5 + n * stride + 3])
    nus = np.array([3, 7])
    matrix = np.full(int(mio[-1] + ld * ld + 5), 71.0, dtype=np.float32)
    dense = []
    pair_world, pair_row, pair_col = [], [], []
    for world in range(worlds):
        a = np.zeros((n, n))
        for i, rows in enumerate(groups):
            for j, cols in enumerate(groups):
                if abs(i - j) <= 1:
                    a[np.ix_(rows, cols)] = rng.normal(0, 0.03, (len(rows), len(cols)))
        a = (a + a.T) * 0.5 + np.eye(n) * (2 + world)
        dense.append(a)
        for row in range(n):
            matrix[mio[world] + row * ld : mio[world] + row * ld + row + 1] = a[row, : row + 1]
        for i in range(n):
            for j in range(i + 1, n):
                if a[i, j] != 0:
                    pair_world.append(world)
                    pair_row.append(i)
                    pair_col.append(j)
    info = SimpleNamespace(dim=ints([n, n]), mio=ints(mio), vio=ints(vio), maxdim=ints([ld, ld]))
    operator = SimpleNamespace(info=info, mat=floats(matrix))
    scale = rng.uniform(0.5, 2.0, int(vio[-1] + n + 5)).astype(np.float32)
    coupling = np.full(int(rio[-1] + n * stride + 5), 83.0, dtype=np.float32)
    for world, nu in enumerate(nus):
        coupling[rio[world] : rio[world] + n * nu] = rng.normal(size=n * nu)
    state = SimpleNamespace(
        bilateral_response_stride=ints([stride, stride]),
        bilateral_response_mio=ints(rio),
        bilateral_preconditioner=floats(scale),
        bilateral_coupling=floats(coupling),
        bilateral_response=wp.full(len(coupling), -123.0, dtype=wp.float32, device=device),
    )
    joints = SimpleNamespace(
        wid=ints([0, 0, 0, 1, 1, 1]),
        num_dynamic_cts=ints([1] * 6),
        num_kinematic_cts=ints([4, 5, 4] * 2),
        dynamic_cts_offset_total_cts=ints([0, 1, 2, 64, 65, 66]),
        kinematic_cts_offset_total_cts=ints([3, 7, 12, 67, 71, 76]),
    )
    path = SimpleNamespace(
        device=device,
        size=SimpleNamespace(num_worlds=worlds),
        model=SimpleNamespace(joints=joints, info=SimpleNamespace(total_cts_offset=ints([0, 64]))),
        data=SimpleNamespace(bilateral_dim=info.dim, bilateral_operator=operator, state=state),
        bilateral_nzb_pairs=(ints(pair_world), ints(pair_row), ints(pair_col)),
    )
    problem = SimpleNamespace(data=SimpleNamespace(dim=ints(n + nus), njc=info.dim))
    return path, problem, dense, matrix, mio, vio, rio, nus, scale, coupling


class TestKaminoJointBlocks(unittest.TestCase):
    def _device(self, conditional=False):
        """Select a CUDA device with the capabilities required by the test."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA tile solves")
        device = wp.get_cuda_devices()[0]
        if conditional and not wp.is_conditional_graph_supported():
            self.skipTest("Requires CUDA conditional graphs")
        return device

    def test_mixed_group_dense_reference(self):
        """Match dense solves and response Grams with padding and inactive worlds."""
        device = self._device()
        path, problem, matrices, _, _, vio, rio, nus, scale, coupling = _fixture(device)
        solver = JointBlockSolver.create(path)
        self.assertIsNotNone(solver)
        solver.prepare(path, problem)
        self.assertEqual(int(solver.failure.numpy()[0]), 0)
        rng = np.random.default_rng(419)
        rhs = rng.normal(size=len(scale)).astype(np.float32)
        rhs_wp = wp.array(rhs, device=device)
        out = wp.full(len(rhs), -123.0, dtype=wp.float32, device=device)
        active = wp.array([16, 0], dtype=wp.int32, device=device)
        solver.solve(rhs_wp, out, active)
        result = out.numpy()
        np.testing.assert_allclose(
            result[vio[0] : vio[0] + 16], np.linalg.solve(matrices[0], rhs[vio[0] : vio[0] + 16]), atol=2e-6, rtol=2e-6
        )
        untouched = np.ones(len(rhs), dtype=bool)
        untouched[vio[0] : vio[0] + 16] = False
        np.testing.assert_array_equal(result[untouched], -123.0)
        solver.solve(rhs_wp, out)
        for world in range(2):
            np.testing.assert_allclose(
                out.numpy()[vio[world] : vio[world] + 16],
                np.linalg.solve(matrices[world], rhs[vio[world] : vio[world] + 16]),
                atol=2e-6,
                rtol=2e-6,
            )
        solver.response(path, problem)
        response = path.data.state.bilateral_response.numpy()
        untouched = np.ones(len(response), dtype=bool)
        for world, nu in enumerate(nus):
            start = rio[world]
            y = response[start : start + 16 * nu].reshape(16, nu).astype(np.float64)
            b = coupling[start : start + 16 * nu].reshape(16, nu) * scale[vio[world] : vio[world] + 16, None]
            expected = b.T @ np.linalg.solve(matrices[world], b)
            np.testing.assert_allclose(y.T @ y, expected, atol=2e-5, rtol=3e-6)
            untouched[start : start + 16 * nu] = False
        np.testing.assert_array_equal(response[untouched], -123.0)

    def test_reject_nonchordal_topology(self):
        """Reject a four-cycle instead of silently dropping symbolic fill."""
        device = self._device()
        path, *_ = _fixture(device)

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        path.model.joints = SimpleNamespace(
            wid=ints([0] * 4 + [1] * 4),
            num_dynamic_cts=ints([0] * 8),
            num_kinematic_cts=ints([4] * 8),
            dynamic_cts_offset_total_cts=ints([0] * 4 + [64] * 4),
            kinematic_cts_offset_total_cts=ints([0, 4, 8, 12, 64, 68, 72, 76]),
        )
        pairs = (ints([0] * 4 + [1] * 4), ints([0, 4, 8, 0] * 2), ints([4, 8, 12, 12] * 2))
        metadata, reason = _build_topology(path, *pairs)
        self.assertIsNone(metadata)
        self.assertIn("nonchordal", reason)

    def test_captured_failure_and_recovery(self):
        """Reset fallback selection across invalid factors, dimensions, and recovery."""
        device = self._device(conditional=True)
        path, problem, matrices, matrix, mio, vio, *_ = _fixture(device)
        solver = JointBlockSolver.create(path)
        rhs = wp.ones(40, dtype=wp.float32, device=device)
        out = wp.zeros_like(rhs)
        solver.prepare(path, problem)
        solver.solve(rhs, out)
        with wp.ScopedCapture(device=device) as capture:
            solver.prepare(path, problem)
            wp.capture_if(solver.failure, on_true=lambda: out.fill_(-777.0), on_false=lambda: solver.solve(rhs, out))
        original_dim = problem.data.dim.numpy()
        for case in ("valid", "dimension", "valid", "factor", "valid"):
            current = matrix.copy()
            dims = original_dim.copy()
            if case == "dimension":
                dims[1] = 33
            elif case == "factor":
                current[mio[1]] = -1.0
            path.data.bilateral_operator.mat.assign(current)
            problem.data.dim.assign(dims)
            wp.capture_launch(capture.graph)
            expected_failure = case != "valid"
            self.assertEqual(bool(solver.failure.numpy()[0]), expected_failure)
            result = out.numpy()
            if expected_failure:
                np.testing.assert_array_equal(result, -777.0)
            else:
                for world in range(2):
                    np.testing.assert_allclose(
                        result[vio[world] : vio[world] + 16],
                        np.linalg.solve(matrices[world], np.ones(16)),
                        atol=2e-6,
                        rtol=2e-6,
                    )

    def test_dispatch_refreshes_cached_values_before_conditional(self):
        """Refresh changing Jacobian values on both branches of every graph replay."""
        device = self._device(conditional=True)
        source = wp.array([1.0], dtype=wp.float32, device=device)
        cached = wp.zeros_like(source)
        output = wp.zeros_like(source)
        flag = wp.zeros(1, dtype=wp.int32, device=device)
        operator = SimpleNamespace(_needs_update=True)

        def update():
            wp.copy(cached, source)
            operator._needs_update = False

        operator.update = update
        blocks = SimpleNamespace(failure=flag, prepare=lambda path, problem: None)
        path = SimpleNamespace(
            use_schur_complement=True,
            joint_block_solver=blocks,
            device=device,
            data=SimpleNamespace(
                bilateral_dim=None, bilateral_operator=SimpleNamespace(info=SimpleNamespace(dim=None), mat=None)
            ),
        )

        def continuation(path, problem, joint_blocks=None):
            if operator._needs_update:
                operator.update()
            wp.copy(output, cached)

        with (
            patch.object(sparse, "_get_sparse_delassus", return_value=operator),
            patch.object(sparse, "_assemble_sparse_bilateral_block"),
            patch.object(sparse, "_solve_sparse_with_bilateral_schur_complement", side_effect=continuation),
        ):
            with wp.ScopedCapture(device=device) as capture:
                sparse._solve_sparse_with_bilateral_direct_block(path, SimpleNamespace())
        for failed, value in ((0, 2.0), (1, 3.0), (0, 5.0)):
            flag.fill_(failed)
            source.fill_(value)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(output.numpy(), [value])


if __name__ == "__main__":
    unittest.main()
