# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlashSAC ONNX export and replay."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton.rl as public_rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class TestFlashSACONNX(unittest.TestCase):
    def test_export_load_and_capture_replay(self) -> None:
        """Round-trip deterministic actor inference through ONNX and CUDA graphs."""

        try:
            onnx = importlib.import_module("onnx")
            ReferenceEvaluator = importlib.import_module("onnx.reference").ReferenceEvaluator
        except ImportError:
            self.skipTest("ONNX extra is unavailable")
        device = require_cuda_graph_capture("FlashSAC ONNX replay")
        config = public_rl.ConfigFlashSAC(
            actor_hidden_dim=16,
            actor_num_blocks=1,
            critic_hidden_dim=16,
            critic_num_blocks=1,
            distributional_atoms=11,
        )
        trainer = public_rl.TrainerFlashSAC(obs_dim=7, action_dim=3, config=config, device=device, seed=17)
        network = trainer.actor.net
        network.input_norm.running_mean.assign(np.linspace(-0.2, 0.3, 7, dtype=np.float32))
        network.input_norm.running_variance.assign(np.linspace(0.6, 1.4, 7, dtype=np.float32))
        for index, (first, second) in enumerate(network.block_norms):
            first.running_mean.assign(np.linspace(-0.1, 0.1, first.width, dtype=np.float32) + index)
            first.running_variance.assign(np.linspace(0.7, 1.3, first.width, dtype=np.float32))
            second.running_mean.assign(np.linspace(-0.05, 0.05, second.width, dtype=np.float32) - index)
            second.running_variance.assign(np.linspace(0.8, 1.2, second.width, dtype=np.float32))

        rng = np.random.default_rng(23)
        observations_np = rng.standard_normal((65, 7)).astype(np.float32)
        observations = wp.array(observations_np, dtype=wp.float32, device=device)
        expected = trainer.actor.sample_reuse(observations, seed=0, deterministic=True)[0].numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.onnx"
            public_rl.export_flash_sac_actor_onnx(trainer, path)
            model = onnx.load(path)
            onnx.checker.check_model(model)
            external = ReferenceEvaluator(model).run(None, {"observations": observations_np})
            np.testing.assert_allclose(external[0], expected, rtol=2.0e-5, atol=2.0e-6)

            policy = public_rl.load_flash_sac_actor_onnx(path, batch_size=65, device=device)
            actual = policy.act(observations)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2.0e-5, atol=2.0e-6)
            action_ptr = int(actual.ptr)
            graph = policy.capture(observations)
            replay_input = rng.standard_normal((65, 7)).astype(np.float32)
            observations.assign(replay_input)
            first = graph.launch().numpy().copy()
            second = graph.launch().numpy().copy()
            self.assertEqual(int(graph.actions.ptr), action_ptr)
            np.testing.assert_array_equal(first, second)
            external_replay = ReferenceEvaluator(model).run(None, {"observations": replay_input})[0]
            np.testing.assert_allclose(first, external_replay, rtol=2.0e-5, atol=2.0e-6)

    def test_rejects_non_flash_model(self) -> None:
        """Reject an ONNX artifact without the FlashSAC schema metadata."""

        try:
            onnx = importlib.import_module("onnx")
        except ImportError:
            self.skipTest("ONNX extra is unavailable")
        device = require_cuda_graph_capture("FlashSAC ONNX validation")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.onnx"
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Relu", ["input"], ["output"])],
                "invalid",
                [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
                [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
            model.ir_version = 8
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "not a PhoenX FlashSAC actor"):
                public_rl.load_flash_sac_actor_onnx(path, batch_size=1, device=device)


if __name__ == "__main__":
    unittest.main()
