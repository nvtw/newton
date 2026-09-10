# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check byte offsets used by Kamino's tiled factorization and Schur kernels."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_rcm import (
    get_float32_array_offset_ptr,
    get_int32_array_offset_ptr,
)


@wp.kernel
def _compute_pointer_offsets(
    floats: wp.array[wp.float32],
    integers: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    offsets: wp.array2d[wp.uint64],
):
    i = wp.tid()
    offsets[i, 0] = get_float32_array_offset_ptr(floats, indices[i]) - get_float32_array_offset_ptr(floats, 0)
    offsets[i, 1] = get_int32_array_offset_ptr(integers, indices[i]) - get_int32_array_offset_ptr(integers, 0)


class TestKaminoPointerOffsets(unittest.TestCase):
    def test_large_byte_offsets(self):
        """Preserve 64-bit byte offsets across the signed 32-bit boundary."""
        indices = np.array([0, 1, 2**29 - 1, 2**29, 2**29 + 1, 2**30, 2**31 - 1], dtype=np.int32)
        expected = np.repeat((indices.astype(np.uint64) * 4)[:, None], 2, axis=1)
        for device in wp.get_devices():
            with self.subTest(device=device):
                # Only compute addresses: do not allocate or dereference multi-GiB buffers.
                floats = wp.zeros(1, dtype=wp.float32, device=device)
                integers = wp.zeros(1, dtype=wp.int32, device=device)
                indices_device = wp.array(indices, dtype=wp.int32, device=device)
                offsets = wp.empty((len(indices), 2), dtype=wp.uint64, device=device)
                wp.launch(
                    _compute_pointer_offsets,
                    dim=len(indices),
                    inputs=[floats, integers, indices_device],
                    outputs=[offsets],
                    device=device,
                )
                np.testing.assert_array_equal(offsets.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
