# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare all membership/history indices, including cache overflow and rebuilds."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_tgs_partition import allocate, partition
from newton._src.solvers.phoenx.constraints.contact_tgs_partition_cuda import BLOCK_SIZE, partition_shared


class TestSharedPartition(unittest.TestCase):
    def test_exact_partition_with_overflow_and_rebuilds(self):
        """Match serial partitions across cache overflow and generation changes."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        rng = np.random.default_rng(914)
        counts = np.array([0, 1, 90, 513, 1024, 1025], dtype=np.int32)
        first = np.cumsum(np.r_[0, counts[:-1]]).astype(np.int32)
        normals = rng.normal(size=(int(counts.sum()), 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[::3] = [0, 0, 1]
        device = "cuda:0"
        normal_array = wp.array(normals, dtype=wp.vec3f, device=device)
        first_array = wp.array(first, dtype=int, device=device)
        count_array = wp.array(counts, dtype=int, device=device)
        active = wp.array([len(counts)], dtype=int, device=device)
        reference = allocate(len(normals), len(counts), device)
        actual = allocate(len(normals), len(counts), device)
        for sizes, groups in ((counts, len(counts)), (counts // 2, len(counts) - 1), (counts, len(counts))):
            count_array.assign(sizes)
            active.assign([groups])
            args = [normal_array, first_array, count_array, active, 0.999]
            wp.launch(partition, len(counts), [*args, reference], device=device)
            wp.launch(partition_shared, (len(counts), BLOCK_SIZE), [*args, actual], block_dim=BLOCK_SIZE, device=device)
            for field in (
                "point_patch",
                "point_next",
                "patch_first",
                "patch_last",
                "patch_count",
                "patch_next",
                "patch_normal",
                "group_first",
                "group_count",
            ):
                np.testing.assert_array_equal(
                    getattr(reference, field).numpy(), getattr(actual, field).numpy(), err_msg=field
                )


if __name__ == "__main__":
    unittest.main()
