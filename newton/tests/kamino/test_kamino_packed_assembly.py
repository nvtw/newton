# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check production packed bilateral assembly against dense storage and an FP64 oracle."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    make_build_sparse_bilateral_block_kernel,
    make_set_sparse_bilateral_diagonal_kernel,
)


class TestPackedAssembly(unittest.TestCase):
    def test_ragged_permuted_changing_contributions(self):
        """Preserve scaled assembly, diagonal-tile symmetry, and double-add semantics."""
        device = wp.get_device()
        rng = np.random.default_rng(34123)
        dimensions = np.array([0, 1, 17, 33, 65], dtype=np.int32)
        counts = (dimensions + 31) // 32
        guard = 7
        sizes = np.maximum(dimensions, 1)
        vio = (guard + np.cumsum(np.r_[0, (sizes + guard)[:-1]])).astype(np.int32)
        pvio = (guard + np.cumsum(np.r_[0, (sizes + guard + 3)[:-1]])).astype(np.int32)
        mio = (guard + np.cumsum(np.r_[0, (np.maximum(dimensions * dimensions, 1) + guard)[:-1]])).astype(np.int32)
        slots = np.cumsum(np.r_[0, (counts * (counts + 1) // 2)[:-1]]).astype(np.int64)
        packed_size = int(np.sum(counts * (counts + 1) // 2) * 1024)
        dense_size = int(mio[-1] + dimensions[-1] ** 2 + guard)
        vector_size = int(vio[-1] + dimensions[-1] + guard)
        problem_size = int(pvio[-1] + dimensions[-1] + guard)
        pairs = []
        for world, n in enumerate(dimensions):
            for row in range(n):
                for col in range(row):
                    pairs.append((world, row, col, world, vio[world] + row, vio[world] + col))
            if n:
                # One literal diagonal pair must add its contribution twice, not once.
                pairs.append((world, 0, 0, world, vio[world], vio[world]))
            if n > 1:
                pairs.append((world, 1, 0, world, vio[world] + 1, vio[world]))

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        def floats(values, dtype=wp.float32):
            return wp.array(np.asarray(values, dtype=np.float32), dtype=dtype, device=device)

        dims, vector_offsets, problem_offsets, dense_offsets = map(ints, (dimensions, vio, pvio, mio))
        packed_offsets = wp.array(slots, dtype=wp.int64, device=device)
        pair_arrays = [ints(values) for values in zip(*pairs, strict=True)]
        masses = rng.uniform(0.1, 2.0, len(dimensions)).astype(np.float32)
        basis = rng.normal(size=(len(dimensions), 3, 3)).astype(np.float32)
        inertias = basis @ basis.transpose(0, 2, 1)
        mass, inertia = floats(masses), floats(inertias, wp.mat33f)
        sentinel = np.float32(-1729.25)
        dense_active = np.zeros(dense_size, dtype=bool)
        vector_active = np.zeros(vector_size, dtype=bool)
        for world, n in enumerate(dimensions):
            dense_active[mio[world] : mio[world] + max(int(n * n), 1)] = True
            vector_active[vio[world] : vio[world] + max(int(n), 1)] = True

        for phase in range(4):
            use_permutation = phase != 0
            inverse_values = np.full(vector_size, -1, dtype=np.int32)
            diagonal = np.full(problem_size, sentinel, dtype=np.float32)
            jacobian = rng.normal(size=(vector_size, 6)).astype(np.float32)
            if phase == 1:
                jacobian.fill(0.0)
            elif phase == 2:
                jacobian[::2] = 0.0
            for world, n in enumerate(dimensions):
                inverse_values[vio[world] : vio[world] + n] = rng.permutation(n)
                diagonal[pvio[world] : pvio[world] + n] = rng.uniform(-2.0, 3.0, n)
                if n > 4:
                    diagonal[pvio[world] + 4] = 0.0
                    jacobian[vio[world] + 4] = 0.0
            inverse = ints(inverse_values)
            diag, jac = floats(diagonal), floats(jacobian, vec6f)
            initial_dense = np.full(dense_size, sentinel, dtype=np.float32)
            initial_dense[dense_active] = 0.0
            initial_packed = np.full(packed_size + guard, sentinel, dtype=np.float32)
            initial_packed[:packed_size] = 0.0
            matrices = [floats(initial_dense), floats(initial_packed)]
            scales = [floats(np.full(vector_size, sentinel, dtype=np.float32)) for _ in range(2)]
            for index, offsets in enumerate((dense_offsets, packed_offsets)):
                wp.launch(
                    make_set_sparse_bilateral_diagonal_kernel(bool(index)),
                    dim=(len(dimensions), int(dimensions.max())),
                    inputs=[
                        dims,
                        problem_offsets,
                        offsets,
                        vector_offsets,
                        diag,
                        matrices[index],
                        scales[index],
                        inverse,
                        use_permutation,
                    ],
                    device=device,
                )
                wp.launch(
                    make_build_sparse_bilateral_block_kernel(bool(index)),
                    dim=len(pairs),
                    inputs=[
                        mass,
                        inertia,
                        *pair_arrays,
                        jac,
                        dims,
                        offsets,
                        vector_offsets,
                        scales[index],
                        matrices[index],
                        inverse,
                        use_permutation,
                    ],
                    device=device,
                )
            dense, packed = [array.numpy() for array in matrices]
            scale = scales[0].numpy()
            np.testing.assert_array_equal(scale.view(np.uint32), scales[1].numpy().view(np.uint32))
            np.testing.assert_array_equal(scale[~vector_active], sentinel)
            np.testing.assert_array_equal(dense[~dense_active], sentinel)
            np.testing.assert_array_equal(packed[packed_size:], sentinel)
            self.assertEqual(dense[mio[0]], 1.0)
            self.assertEqual(scale[vio[0]], 1.0)
            oracle = [np.zeros((n, n), dtype=np.float64) for n in dimensions]
            orders = []
            for world, n in enumerate(dimensions):
                order = inverse_values[vio[world] : vio[world] + n] if use_permutation else np.arange(n)
                orders.append(order)
                for row in range(n):
                    p = float(scale[vio[world] + row])
                    oracle[world][order[row], order[row]] = p * abs(float(diagonal[pvio[world] + row])) * p + 7e-7
            for world, row, col, body, block_i, block_j in pairs:
                a, b = jacobian[block_i].astype(np.float64), jacobian[block_j].astype(np.float64)
                value = float(masses[body]) * np.dot(a[:3], b[:3]) + np.dot(
                    a[3:], inertias[body].astype(np.float64) @ b[3:]
                )
                value *= float(scale[vio[world] + row]) * float(scale[vio[world] + col])
                r, c = orders[world][row], orders[world][col]
                oracle[world][r, c] += value
                oracle[world][c, r] += value
            for world, (n, count) in enumerate(zip(dimensions, counts, strict=True)):
                expected = dense[mio[world] : mio[world] + n * n].reshape(n, n)
                for row in range(count):
                    for col in range(row + 1):
                        slot = int(slots[world] + row * (row + 1) // 2 + col)
                        panel = packed[slot * 1024 : (slot + 1) * 1024].reshape(32, 32)
                        nr, nc = min(32, n - row * 32), min(32, n - col * 32)
                        actual = panel[:nr, :nc]
                        reference = expected[row * 32 : row * 32 + nr, col * 32 : col * 32 + nc]
                        independent = oracle[world][row * 32 : row * 32 + nr, col * 32 : col * 32 + nc]
                        with self.subTest(phase=phase, world=world, tile=(row, col)):
                            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
                            np.testing.assert_allclose(actual, independent, rtol=3e-6, atol=3e-6)
                            # All but the repeated pair have one writer (or two ordered writes in one thread).
                            deterministic = np.ones((nr, nc), dtype=bool)
                            if n > 1:
                                a, b = orders[world][:2]
                                for r, c in ((a, b), (b, a)):
                                    if r // 32 == row and c // 32 == col:
                                        deterministic[r % 32, c % 32] = False
                            np.testing.assert_array_equal(
                                actual[deterministic].view(np.uint32), reference[deterministic].view(np.uint32)
                            )
                            np.testing.assert_array_equal(panel[nr:, :], 0.0)
                            np.testing.assert_array_equal(panel[:, nc:], 0.0)


if __name__ == "__main__":
    unittest.main()
