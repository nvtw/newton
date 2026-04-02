# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
import warp as wp

POLICY_PATH = "C:/tmp/policies/anymal_c_walking_policy.onnx"


class TestOnnxRuntime(unittest.TestCase):
    """Tests for the lightweight Warp-based ONNX runtime."""

    def setUp(self):
        if not os.path.exists(POLICY_PATH):
            self.skipTest(f"ONNX policy not found at {POLICY_PATH}")

    def _run_on_device(self, device_str: str):
        from newton._src.onnx_runtime import OnnxRuntime

        rt = OnnxRuntime(POLICY_PATH, device=device_str)

        self.assertEqual(rt.input_names, ["observation"])
        self.assertEqual(rt.output_names, ["action"])

        rng = np.random.default_rng(42)
        obs = rng.standard_normal((1, 48)).astype(np.float32)
        obs_wp = wp.array(obs, dtype=wp.float32, device=device_str)

        result = rt({"observation": obs_wp})
        action = result["action"]

        self.assertEqual(action.shape, (1, 12))
        action_np = action.numpy()

        # Reference values computed with onnxruntime (default_rng seed=42).
        expected = np.array(
            [
                [
                    0.47301683,
                    1.0230064,
                    0.6639304,
                    1.1513045,
                    0.73161083,
                    -1.572881,
                    0.6339237,
                    -0.7481932,
                    -0.22896627,
                    1.26267,
                    -0.9770715,
                    -2.3447196,
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(action_np, expected, atol=1e-4)

    def test_cpu(self):
        self._run_on_device("cpu")

    def test_cuda(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")
        self._run_on_device("cuda:0")

    def test_deterministic(self):
        """Two consecutive calls with the same input produce identical output."""
        from newton._src.onnx_runtime import OnnxRuntime

        rt = OnnxRuntime(POLICY_PATH, device="cpu")
        obs = np.ones((1, 48), dtype=np.float32) * 0.5
        obs_wp = wp.array(obs, dtype=wp.float32, device="cpu")

        out1 = rt({"observation": obs_wp})["action"].numpy()
        out2 = rt({"observation": obs_wp})["action"].numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_batch(self):
        """Batch size > 1 works correctly."""
        from newton._src.onnx_runtime import OnnxRuntime

        rt = OnnxRuntime(POLICY_PATH, device="cpu", batch_size=4)
        rng = np.random.default_rng(123)
        obs = rng.standard_normal((4, 48)).astype(np.float32)
        obs_wp = wp.array(obs, dtype=wp.float32, device="cpu")

        result = rt({"observation": obs_wp})
        action = result["action"]
        self.assertEqual(action.shape, (4, 12))

        # Verify each row matches single-batch inference
        rt_single = OnnxRuntime(POLICY_PATH, device="cpu", batch_size=1)
        for i in range(4):
            single_obs = wp.array(obs[i : i + 1], dtype=wp.float32, device="cpu")
            single_out = rt_single({"observation": single_obs})["action"].numpy()
            np.testing.assert_allclose(action.numpy()[i : i + 1], single_out, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
