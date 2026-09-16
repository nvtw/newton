# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Accuracy controls for the bounded FP32 joint/normal proposal."""

import unittest
from pathlib import Path

import numpy as np

from .colored_joint_normal_incremental import compensated_matvec, solve_incremental
from .coupled_support_online import assemble_snapshot


class TestIncrementalJointNormal(unittest.TestCase):
    def test_cancelled_products(self):
        """Retain the residual of cancelling rounded products using FP32 operations."""
        matrix = np.array([[1.000001, -1]], dtype=np.float32)
        vector = np.array([1.000001, np.float32(1.000001) ** 2], dtype=np.float32)
        oracle = matrix.astype(float) @ vector.astype(float)
        plain = matrix @ vector
        compensated = compensated_matvec(matrix, vector)
        self.assertEqual(compensated.dtype, np.float32)
        self.assertGreater(float(np.max(abs(plain - oracle))), 1e-14)
        np.testing.assert_allclose(compensated, oracle, atol=1e-18, rtol=1e-6)

    @unittest.skipUnless(
        Path("/tmp/colibri_colored_block_dispatch_trace600.phases.npz").exists(), "Captured CPU fixture unavailable"
    )
    def test_actual_scatter_and_equations(self):
        """Reject plain scatter and certify compensated increments against original equations."""
        z = np.load("/tmp/colibri_colored_block_dispatch_trace600.phases.npz")
        d = {k.split(".", 1)[1]: z[k] for k in z.files if k.startswith("biased_after.")}
        d["copy_section_end"] = np.arange(len(d["velocity"]))
        d["copy_velocity"] = d["velocity"][1:]
        d["copy_angular_velocity"] = d["angular_velocity"][1:]
        a = assemble_snapshot(d, "biased", float(z["dt"][0]), int(z["num_joints"][0]))
        plain = solve_incremental(a, compensate=False)
        fixed = solve_incremental(a)
        self.assertFalse(plain["accepted"])
        self.assertGreater(plain["records"][-1]["applied_scatter"], 1e-8)
        self.assertTrue(fixed["accepted"], fixed["records"])
        self.assertLess(fixed["records"][-1]["work_error"], 1e-10)


if __name__ == "__main__":
    unittest.main()
