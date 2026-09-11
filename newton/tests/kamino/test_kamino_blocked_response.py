# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check blocked compact responses against dense triangular solves."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm import make_llt_blocked_rcm_solve_kernel
from newton._src.solvers.kamino._src.solvers.dvi.response import (
    _add_forward_bilateral_gradient,
    _update_forward_bilateral_rhs,
    make_response_kernel,
)


class TestKaminoBlockedResponse(unittest.TestCase):
    def test_dense_reference(self):
        """Verify partial tiles, permuted scaling, zero tiles, and guarded output."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA tile solves")
        device = wp.get_cuda_devices()[0]
        rng = np.random.default_rng(813)
        for n, nu in ((17, 9), (65, 13), (129, 37)):
            with self.subTest(n=n, nu=nu):
                lower = np.tril(rng.normal(0, 0.02, (n, n))).astype(np.float32)
                np.fill_diagonal(lower, 2.0)
                lower[32:, :32] = 0
                permutation = rng.permutation(n).astype(np.int32)
                scale = rng.uniform(0.5, 2, n).astype(np.float32)
                stride = nu + 5
                coupling = rng.normal(size=(n, stride)).astype(np.float32)
                expected = np.linalg.solve(lower.astype(np.float64), (scale[:, None] * coupling[:, :nu])[permutation])
                # Compact worlds store the coupling densely with row stride nu.
                coupling_input = np.zeros(n * stride, dtype=np.float32)
                coupling_input[: n * nu] = coupling[:, :nu].ravel()
                tiles = (n + 31) // 32
                pattern = np.ones((tiles, tiles), dtype=np.int32)
                pattern[1:, 0] = 0

                def ints(values):
                    return wp.array(values, dtype=wp.int32, device=device)

                def floats(values):
                    return wp.array(np.asarray(values).ravel(), dtype=wp.float32, device=device)

                output = wp.full(n * stride + 7, -123.0, device=device)
                wp.launch(
                    make_response_kernel(),
                    dim=((nu + 3) // 4, 128),
                    inputs=[
                        ints([n + nu]),
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        floats(scale),
                        floats(lower),
                        ints(permutation),
                        ints([0]),
                        ints([stride]),
                        floats(coupling_input),
                        output,
                        ints([0]),
                        ints(pattern.ravel()),
                        nu,
                    ],
                    block_dim=128,
                    device=device,
                )
                result = output.numpy()
                np.testing.assert_allclose(result[: n * nu].reshape(n, nu), expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(result[n * nu :], -123.0)

                # Reuse these whitened columns to update a forward-solved RHS,
                # then recover the final original-coordinate solution.
                rhs = rng.normal(size=n).astype(np.float32)
                initial_y = np.linalg.solve(lower.astype(np.float64), rhs[permutation].astype(np.float64))
                y = wp.empty(n, dtype=wp.float32, device=device)
                untouched_x = wp.full(n, -123.0, device=device)
                untouched_x_hat = wp.full(n, -123.0, device=device)
                wp.launch(
                    make_llt_blocked_rcm_solve_kernel(32, True, False),
                    dim=(1, 128),
                    inputs=[
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        ints(permutation),
                        floats(lower),
                        ints(pattern.ravel()),
                        floats(rhs),
                        y,
                        untouched_x_hat,
                        untouched_x,
                    ],
                    block_dim=128,
                    device=device,
                )
                np.testing.assert_allclose(y.numpy(), initial_y, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(untouched_x.numpy(), -123.0)
                np.testing.assert_array_equal(untouched_x_hat.numpy(), -123.0)
                initial_gradient = rng.normal(size=n + nu).astype(np.float32)
                gradient = floats(initial_gradient)
                wp.launch(
                    _add_forward_bilateral_gradient,
                    dim=(1, nu + 3, 32),
                    inputs=[ints([n + nu]), ints([n]), ints([0]), ints([0]), ints([0]), output, y, gradient],
                    block_dim=128,
                    device=device,
                )
                expected_gradient = initial_gradient.astype(np.float64)
                expected_gradient[n:] += expected.T @ initial_y
                np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=2e-6, atol=2e-6)
                delta = rng.normal(0, 0.01, nu).astype(np.float32)
                wp.launch(
                    _update_forward_bilateral_rhs,
                    dim=(1, n, 32),
                    inputs=[
                        ints([n + nu]),
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        output,
                        floats(np.zeros(n + nu)),
                        floats(np.concatenate((np.zeros(n), delta))),
                        y,
                    ],
                    block_dim=128,
                    device=device,
                )
                x = wp.empty(n, dtype=wp.float32, device=device)
                wp.launch(
                    make_llt_blocked_rcm_solve_kernel(32, False),
                    dim=(1, 128),
                    inputs=[
                        ints([n]),
                        ints([0]),
                        ints([0]),
                        ints([0]),
                        ints(permutation),
                        floats(lower),
                        ints(pattern.ravel()),
                        floats(np.full(n, np.nan)),
                        y,
                        wp.empty_like(y),
                        x,
                    ],
                    block_dim=128,
                    device=device,
                )
                expected_y = initial_y.astype(np.float64) - expected @ delta.astype(np.float64)
                expected_x = np.empty(n)
                expected_x[permutation] = np.linalg.solve(lower.astype(np.float64).T, expected_y)
                np.testing.assert_allclose(y.numpy(), expected_y, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(x.numpy(), expected_x, rtol=2e-6, atol=2e-6)

    def test_heterogeneous_worlds(self):
        """Keep skipped worlds and padding intact with independent offsets."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA tile solves")
        rng = np.random.default_rng(817)
        sizes = ((65, 13), (33, 17), (17, 37), (0, 0), (32, 0), (129, 37))
        factors, permutations, scales, couplings, patterns = [], [], [], [], []
        matrix_offsets, vector_offsets, response_offsets, pattern_offsets = [], [], [], []
        expected = []
        for n, nu in sizes:
            matrix_offsets.append(len(factors))
            vector_offsets.append(len(scales))
            response_offsets.append(len(couplings))
            pattern_offsets.append(len(patterns))
            stride = nu + 5
            lower = np.tril(rng.normal(0, 0.02, (n, n))).astype(np.float32)
            np.fill_diagonal(lower, 2.0)
            permutation = rng.permutation(n).astype(np.int32)
            scale = rng.uniform(0.5, 2, n).astype(np.float32)
            coupling = rng.normal(size=(n, stride)).astype(np.float32)
            tiles = (n + 31) // 32
            factors.extend(lower.ravel())
            permutations.extend(permutation)
            scales.extend(scale)
            coupling_input = coupling.ravel().copy()
            if nu and nu * nu <= n * stride:
                # Compact worlds store the coupling densely with row stride nu.
                coupling_input[:] = 0.0
                coupling_input[: n * nu] = coupling[:, :nu].ravel()
            couplings.extend(coupling_input)
            # Exercise offsets which are independent of matrix dimensions.
            factors.extend([0.0] * 3)
            permutations.extend([0] * 2)
            scales.extend([0.0] * 2)
            couplings.extend([0.0] * 7)
            patterns.extend(np.ones(tiles * tiles + 1, dtype=np.int32))
            target = np.full(n * stride + 7, -123.0, dtype=np.float32)
            if nu and nu * nu <= n * stride:
                target[: n * nu] = np.linalg.solve(
                    lower.astype(np.float64), (scale[:, None] * coupling[:, :nu])[permutation]
                ).ravel()
            expected.extend(target)
        device = wp.get_cuda_devices()[0]

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(values, dtype=wp.float32, device=device)

        output = wp.full(len(couplings), -123.0, device=device)
        max_columns = max(nu for _, nu in sizes)
        wp.launch(
            make_response_kernel(),
            dim=(len(sizes) * ((max_columns + 3) // 4), 128),
            inputs=[
                ints([n + nu for n, nu in sizes]),
                ints([n for n, _ in sizes]),
                ints(matrix_offsets),
                ints(vector_offsets),
                floats(scales),
                floats(factors),
                ints(permutations),
                ints(response_offsets),
                ints([nu + 5 for _, nu in sizes]),
                floats(couplings),
                output,
                ints(pattern_offsets),
                ints(patterns),
                max_columns,
            ],
            block_dim=128,
            device=device,
        )
        np.testing.assert_allclose(output.numpy(), expected, rtol=2e-6, atol=2e-6)
