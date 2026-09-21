# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical tests for the PhoenX block-tridiagonal PCR solver."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.articulations.fixed_pattern_pcr import _subtract_compensated_product


@wp.kernel(enable_backward=False)
def _subtract_products(
    source: wp.float32,
    left: wp.array[wp.float32],
    right: wp.array[wp.float32],
    count: wp.int32,
    result: wp.array[wp.float32],
):
    compensated = wp.vec2f(source, wp.float32(0.0))
    naive = source
    index = wp.int32(0)
    while index < count:
        compensated = _subtract_compensated_product(compensated, left[index], right[index])
        naive -= left[index] * right[index]
        index += wp.int32(1)
    result[0] = compensated[0] + compensated[1]
    result[1] = naive


class TestPhoenXFixedPatternPCR(unittest.TestCase):
    def test_compensated_products_preserve_cancelled_residual(self):
        """Recover residuals hidden while subtracting small products from a large value."""
        device = wp.get_device()
        left = np.ones(48, dtype=np.float32)
        right = np.ones(48, dtype=np.float32)
        left[-1] = np.float32(1.0e8)
        result = wp.zeros(2, dtype=wp.float32, device=device)

        wp.launch(
            _subtract_products,
            dim=1,
            inputs=[
                wp.float32(1.0e8),
                wp.array(left, dtype=wp.float32, device=device),
                wp.array(right, dtype=wp.float32, device=device),
                len(left),
                result,
            ],
            device=device,
        )

        compensated, naive = result.numpy()
        self.assertAlmostEqual(float(compensated), -47.0, places=5)
        self.assertGreater(abs(float(naive) + 47.0), 40.0)
