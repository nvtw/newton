# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import tempfile
import unittest

import numpy as np

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_WARP_NN = importlib.util.find_spec("warp_nn") is not None

if _HAS_ONNX and _HAS_TORCH and _HAS_WARP_NN:
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    from newton._src.solvers.kamino.examples.rl.onnx_policy import WarpOnnxPolicy


@unittest.skipUnless(_HAS_ONNX and _HAS_TORCH and _HAS_WARP_NN, "onnx, torch, or warp-nn not installed")
class TestKaminoRlOnnx(unittest.TestCase):
    """Test Warp-NN policy inference used by the Kamino RL example."""

    def _save_policy(self, directory, *, input_count=1, output_count=1, output_width=2):
        """Create a small ONNX policy with configurable inputs and outputs."""
        weights = np.arange(output_width * 2, dtype=np.float32).reshape(output_width, 2)
        bias = np.arange(output_width, dtype=np.float32)
        offset = np.zeros(2, dtype=np.float32)
        scale = np.ones(2, dtype=np.float32)
        inputs = [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, 2])]
        inputs.extend(
            helper.make_tensor_value_info(f"extra_input_{i}", TensorProto.FLOAT, [None, 2])
            for i in range(input_count - 1)
        )
        outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, output_width])]
        outputs.extend(
            helper.make_tensor_value_info(f"extra_output_{i}", TensorProto.FLOAT, [None, output_width])
            for i in range(output_count - 1)
        )
        nodes = [
            helper.make_node("Sub", ["observation", "offset"], ["centered"]),
            helper.make_node("Div", ["centered", "scale"], ["normalized"]),
            helper.make_node("Gemm", ["normalized", "weight", "bias"], ["action"], transB=1),
        ]
        nodes.extend(
            helper.make_node("Gemm", ["normalized", "weight", "bias"], [f"extra_output_{i}"], transB=1)
            for i in range(output_count - 1)
        )
        graph = helper.make_graph(
            nodes,
            "policy",
            inputs,
            outputs,
            [
                numpy_helper.from_array(weights, "weight"),
                numpy_helper.from_array(bias, "bias"),
                numpy_helper.from_array(offset, "offset"),
                numpy_helper.from_array(scale, "scale"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = os.path.join(directory, "policy.onnx")
        onnx.save(model, path)
        return path, weights, bias

    def test_policy_accepts_torch_tensor(self):
        """Evaluate a DR Legs-style ONNX policy from a zero-copy Torch input."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, weights, bias = self._save_policy(tmp_dir)
            policy = WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)
            observation = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
            actual = policy(observation)

        expected = observation @ torch.from_numpy(weights).T + torch.from_numpy(bias)
        torch.testing.assert_close(actual, expected)

    def test_policy_rejects_invalid_torch_tensor(self):
        """Reject observations that cannot use zero-copy Warp inference."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir)
            policy = WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

            with self.assertRaisesRegex(TypeError, "torch.float32"):
                policy(torch.ones((2, 2), dtype=torch.float64))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                policy(torch.ones((2, 4), dtype=torch.float32)[:, ::2])

    def test_policy_rejects_multiple_inputs_or_outputs(self):
        """Reject policy models that do not have one input and one output."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir, input_count=2)
            with self.assertRaisesRegex(ValueError, "exactly one input and one output"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

            path, _, _ = self._save_policy(tmp_dir, output_count=2)
            with self.assertRaisesRegex(ValueError, "exactly one input and one output"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

    def test_policy_rejects_invalid_action_width(self):
        """Reject a policy whose output width does not match the actions."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir, output_width=3)
            with self.assertRaisesRegex(ValueError, "output shape"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)


if __name__ == "__main__":
    unittest.main()
