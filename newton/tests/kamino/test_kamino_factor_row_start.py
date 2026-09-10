# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify exact factor prefixes from conservative RCM tile masks."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _find_bilateral_factor_row_start,
    _find_bilateral_factor_row_start_rcm,
)


class TestKaminoFactorRowStart(unittest.TestCase):
    def test_ragged_tile_masks(self):
        """Preserve exact prefixes across ragged tiles, signed zeros, and NaNs."""
        for block_size in (8, 16, 32):
            rng = np.random.default_rng(23)
            dimensions = [0, 1, 15, 16, 17, 31, 32, 33, 64, 65, 97]
            matrices, patterns, mio, vio, tpo, expected = [], [], [], [], [], []
            for n in dimensions:
                tiles = (n + block_size - 1) // block_size
                mask = rng.integers(0, 2, size=(tiles, tiles), dtype=np.int32)
                np.fill_diagonal(mask, 1)
                factor = np.tril(rng.normal(size=(n, n)).astype(np.float32))
                for i in range(tiles):
                    for j in range(i):
                        if mask[i, j] == 0:
                            factor[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 0
                # Conservative symbolic masks may include rows that are numerically zero.
                for row in range(n):
                    factor[row, : min(row, row // 2)] = -0.0
                if n > 32:
                    factor[-1, block_size] = np.nan
                    mask[(n - 1) // block_size, 1] = 1
                    factor[-2, 3] = np.float32(1e-20)
                    mask[(n - 2) // block_size, 0] = 1
                mio.append(len(matrices))
                tpo.append(len(patterns))
                vio.append(len(expected) + 1)
                matrices.extend(factor.ravel())
                patterns.extend(mask.ravel())
                expected.append(-77)
                for row in range(n):
                    nonzero = np.flatnonzero(factor[row, :row] != 0)
                    first = int(nonzero[0]) if len(nonzero) else row
                    expected.append(first // 16 * 16)
                expected.append(-77)
            for device in wp.get_devices():
                dims = wp.array(dimensions, dtype=wp.int32, device=device)
                matrices_wp = wp.array(matrices, dtype=wp.float32, device=device)
                matrix_offsets = wp.array(mio, dtype=wp.int32, device=device)
                vector_offsets = wp.array(vio, dtype=wp.int32, device=device)
                pattern_offsets = wp.array(tpo, dtype=wp.int32, device=device)
                pattern = wp.array(patterns, dtype=wp.int32, device=device)
                reference = wp.full(len(expected), -77, dtype=wp.int32, device=device)
                candidate = wp.full(len(expected), -77, dtype=wp.int32, device=device)
                inputs = [dims, matrix_offsets, vector_offsets, matrices_wp]
                wp.launch(
                    _find_bilateral_factor_row_start,
                    dim=(len(dimensions), max(dimensions)),
                    inputs=[*inputs, reference],
                    device=device,
                )
                wp.launch(
                    _find_bilateral_factor_row_start_rcm,
                    dim=(len(dimensions), max(dimensions)),
                    inputs=[*inputs, candidate, pattern_offsets, pattern, block_size],
                    device=device,
                )
                np.testing.assert_array_equal(reference.numpy(), expected)
                np.testing.assert_array_equal(candidate.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
