# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check bounded contact preparation against the original full-width grid."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.tests.test_rigid_split_prepare import run_scene


@unittest.skipUnless(wp.is_cuda_available(), "Contact preparation requires CUDA")
class TestContactPrepareGrid(unittest.TestCase):
    def test_chunk_bound_reduces_grid_without_changing_state(self):
        """Preserve contact state and momentum with fewer idle prepare lanes."""
        launch = wp.launch

        def run(split, chunk, full_grid):
            dimensions = []

            def record(kernel, *args, **kwargs):
                if "_get_parallel_contact_prepare_kernel" in kernel.func.__qualname__:
                    if full_grid:
                        kwargs["dim"] = (kwargs["dim"][0], 128)
                    dimensions.append(kwargs["dim"])
                return launch(kernel, *args, **kwargs)

            with patch.object(wp, "launch", record):
                states = run_scene(True, mass_splitting=split, chunk_size=chunk)
            self.assertTrue(dimensions)
            expected = 128 if full_grid else min(128, chunk or 128)
            self.assertTrue(all(dim[1] == expected for dim in dimensions))
            return states

        for split in (False, True):
            for chunk in (0, 1, 3, 6, 128, 129):
                with self.subTest(split=split, chunk=chunk):
                    before = run(split, chunk, True)
                    after = run(split, chunk, False)
                    for reference, actual in zip(before, after, strict=True):
                        for name in reference:
                            np.testing.assert_array_equal(
                                reference[name].view(np.uint8), actual[name].view(np.uint8), err_msg=name
                            )


if __name__ == "__main__":
    unittest.main()
