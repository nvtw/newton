# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical and storage lifecycle coverage for packed bilateral LLT kernels."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_packed import (
    _get_float_array_offset_ptr,
    _PackedLLT,
    packed_element_offset,
)


@wp.kernel
def _probe_offsets(storage: wp.array[wp.float32], offsets: wp.array[wp.int64], addresses: wp.array[wp.uint64]):
    offsets[0] = packed_element_offset(wp.int64(524289), 64, 33)
    offsets[1] = packed_element_offset(wp.int64(2147483648), 31, 31)
    addresses[0] = _get_float_array_offset_ptr(storage, offsets[0]) - _get_float_array_offset_ptr(storage, wp.int64(0))
    addresses[1] = _get_float_array_offset_ptr(storage, offsets[1]) - _get_float_array_offset_ptr(storage, wp.int64(0))


class TestPackedLLT(unittest.TestCase):
    def test_ragged_replay_and_active_solves(self):
        """Preserve factor and solve accuracy across mask changes and world reactivation."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Packed tiled factorization requires CUDA")
        rng = np.random.default_rng(71412)
        dimensions = np.array([0, 1, 17, 32, 33, 65, 258], dtype=np.int32)
        tiles = (dimensions + 31) // 32
        pattern_offsets = np.cumsum(np.r_[0, (tiles * tiles)[:-1]]).astype(np.int32)
        slots = tiles * (tiles + 1) // 2
        slot_offsets = np.cumsum(np.r_[0, slots[:-1]])
        guard = 7
        vector_offsets = (guard + np.cumsum(np.r_[0, (dimensions + guard)[:-1]])).astype(np.int32)
        vector_size = int(np.sum(dimensions + guard) + guard)
        sentinel = np.float32(-871.25)
        permutations = [rng.permutation(n).astype(np.int32) for n in dimensions]
        permutation_values = np.full(vector_size, -1, dtype=np.int32)
        active_vector = np.zeros(vector_size, dtype=bool)
        for n, offset, order in zip(dimensions, vector_offsets, permutations, strict=True):
            permutation_values[offset : offset + n] = order
            active_vector[offset : offset + n] = True

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        pattern = ints(np.ones(int(np.sum(tiles * tiles)), dtype=np.int32))
        active_dimensions = ints(dimensions)
        ops = _PackedLLT(
            dimensions, device, pattern, ints(pattern_offsets), ints(permutation_values), ints(vector_offsets)
        )
        matrix = wp.full(ops.factor_size + guard, sentinel, dtype=wp.float32, device=device)
        factor = wp.full_like(matrix, sentinel)
        rhs = wp.full(vector_size, sentinel, dtype=wp.float32, device=device)
        solution = wp.full_like(rhs, sentinel)
        inplace = wp.full_like(rhs, sentinel)

        def launch():
            ops.factor(matrix, factor)
            ops.solve(factor, rhs, solution, active_dimensions)
            ops.solve(factor, inplace, inplace, active_dimensions)

        # Compile before capture; real SPD inputs replace this storage before replay.
        launch()
        with wp.ScopedCapture(device=device) as capture:
            launch()

        for phase in range(4):
            packed_values = np.full(matrix.size, sentinel, dtype=np.float32)
            pattern_values = np.zeros(pattern.size, dtype=np.int32)
            rhs_values = np.full(vector_size, sentinel, dtype=np.float32)
            reference = np.full(vector_size, sentinel, dtype=np.float64)
            matrices = []
            for world, (n, count, offset, order) in enumerate(
                zip(dimensions, tiles, vector_offsets, permutations, strict=True)
            ):
                if phase == 1:
                    a = np.diag(rng.uniform(0.5, 2.0, n))
                else:
                    basis = rng.normal(size=(n, n)) / np.sqrt(max(int(n), 1))
                    if phase == 2:
                        groups = np.arange(n) // 32
                        basis[groups[:, None] != groups[None, :]] = 0.0
                    a = basis @ basis.T + np.eye(n) * 0.5
                a = a.astype(np.float32)
                matrices.append(a)
                vector = rng.normal(size=n).astype(np.float32)
                rhs_values[offset : offset + n] = vector
                if n:
                    reference[offset + order] = np.linalg.solve(a.astype(np.float64), vector[order].astype(np.float64))
                padded = np.zeros((count * 32, count * 32), dtype=np.float32)
                padded[:n, :n] = a
                mask = np.eye(count, dtype=np.int32)
                for row in range(count):
                    for col in range(row + 1):
                        panel = padded[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
                        slot = int(slot_offsets[world] + row * (row + 1) // 2 + col)
                        packed_values[slot * 1024 : (slot + 1) * 1024] = panel.ravel()
                        if np.any(panel):
                            mask[row, col] = mask[col, row] = 1
                start = pattern_offsets[world]
                pattern_values[start : start + count * count] = mask.ravel()
            matrix.assign(packed_values)
            pattern.assign(pattern_values)

            # Do not clear factors or solve workspaces: shrinking masks must remove stale values.
            for mode in ("active", "inactive", "reactivated"):
                active = dimensions.copy()
                if mode == "inactive":
                    active[::2] = 0
                active_dimensions.assign(active)
                rhs.assign(rhs_values)
                inplace.assign(rhs_values)
                solution.fill_(sentinel)
                wp.capture_launch(capture.graph)
                actual, actual_inplace = solution.numpy(), inplace.numpy()
                np.testing.assert_array_equal(ops.errors.numpy(), [0])
                np.testing.assert_array_equal(rhs.numpy(), rhs_values)
                np.testing.assert_array_equal(actual[~active_vector], sentinel)
                np.testing.assert_array_equal(actual_inplace[~active_vector], sentinel)
                for world, (n, offset) in enumerate(zip(dimensions, vector_offsets, strict=True)):
                    segment = slice(offset, offset + n)
                    with self.subTest(phase=phase, mode=mode, dimension=int(n)):
                        if active[world]:
                            np.testing.assert_allclose(actual[segment], reference[segment], rtol=2e-5, atol=2e-5)
                            np.testing.assert_array_equal(actual_inplace[segment], actual[segment])
                        else:
                            np.testing.assert_array_equal(actual[segment], sentinel)
                            np.testing.assert_array_equal(actual_inplace[segment], rhs_values[segment])

            packed_factor = factor.numpy()
            np.testing.assert_array_equal(packed_factor[ops.factor_size :], sentinel)
            np.testing.assert_array_equal(matrix.numpy(), packed_values)
            for world, (n, count) in enumerate(zip(dimensions, tiles, strict=True)):
                if n == 0:
                    continue
                reconstructed = np.zeros((count * 32, count * 32), dtype=np.float64)
                for row in range(count):
                    for col in range(row + 1):
                        slot = int(slot_offsets[world] + row * (row + 1) // 2 + col)
                        panel = packed_factor[slot * 1024 : (slot + 1) * 1024].reshape(32, 32)
                        reconstructed[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] = panel
                        if not pattern_values[pattern_offsets[world] + row * count + col]:
                            np.testing.assert_array_equal(panel, 0.0)
                lower = np.tril(reconstructed)
                expected_padded = np.eye(count * 32, dtype=np.float64)
                expected_padded[:n, :n] = matrices[world]
                with self.subTest(phase=phase, dimension=int(n), check="factor"):
                    relative_error = np.linalg.norm(lower @ lower.T - expected_padded) / np.linalg.norm(expected_padded)
                    self.assertLess(relative_error, 2e-6)
                    np.testing.assert_allclose(
                        lower[:n, :n], np.linalg.cholesky(matrices[world].astype(np.float64)), rtol=2e-5, atol=2e-6
                    )
                    np.testing.assert_array_equal(lower[n:, :n], 0.0)
                    np.testing.assert_array_equal(lower[n:, n:], np.eye(count * 32 - n))

    def test_large_packed_address_arithmetic(self):
        """Widen packed slot and byte offsets before arithmetic exceeds signed int32."""
        device = wp.get_device()
        storage = wp.zeros(1, dtype=wp.float32, device=device)
        offsets = wp.zeros(2, dtype=wp.int64, device=device)
        addresses = wp.zeros(2, dtype=wp.uint64, device=device)
        # Only compute pointers; do not allocate or dereference multi-gigabyte storage.
        wp.launch(_probe_offsets, dim=1, inputs=[storage, offsets, addresses], device=device)
        expected = np.array([(524289 + 4) * 1024 + 1, 2147483648 * 1024 + 1023], dtype=np.int64)
        np.testing.assert_array_equal(offsets.numpy(), expected)
        np.testing.assert_array_equal(addresses.numpy(), expected.astype(np.uint64) * 4)


if __name__ == "__main__":
    unittest.main()
