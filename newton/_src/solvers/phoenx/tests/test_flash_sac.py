# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-Warp FlashSAC implementation."""

from __future__ import annotations

import ast
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton.rl as public_rl
from newton._src.solvers.phoenx.rl_training import g1_recipe
from newton._src.solvers.phoenx.rl_training.flash_sac import (
    BufferReplayFlashSAC,
    ConfigFlashSAC,
    RewardNormalizerFlashSAC,
    TrainerFlashSAC,
    _flash_sac_alpha_loss_kernel,
)
from newton._src.solvers.phoenx.rl_training.flash_sac_networks import (
    _TILE_REDUCTION_BLOCK_DIM,
    NetworkFlashSAC,
    _batch_moments_tile_kernel,
    _batch_moments_transposed_tile_kernel,
    _batch_norm_amp_dual_kernel,
    _batch_norm_backward_amp_tile_kernel,
    _batch_norm_backward_amp_transposed_tile_kernel,
    _batch_norm_input_grad_amp_kernel,
    _batch_norm_inv_std_amp_dual_kernel,
    _batch_norm_inv_std_amp_kernel,
    _batch_norm_kernel,
    _head_bias_amp_kernel,
    _head_bias_log_std_amp_kernel,
    _launch_rms_backward_stats,
    _launch_rms_inv,
    _launch_rms_scale_grad,
    _residual_add_mixed_dual_kernel,
    _residual_add_mixed_f32_kernel,
    _rms_norm_dual_kernel,
    _rms_norm_f16_kernel,
    _rms_norm_input_grad_f16_kernel,
    _rms_norm_kernel,
    _round_batch_moments_f16_kernel,
    _transpose_2d_tile_kernel,
    _UnitBatchNorm,
)
from newton._src.solvers.phoenx.rl_training.g1 import (
    ACTION_DIM_G1,
    OBS_DIM_G1,
    ConfigEnvG1PhoenX,
    EnvG1PhoenX,
)
from newton._src.solvers.phoenx.rl_training.kernels import (
    amp_update_scale_kernel,
    sac_distributional_min_projection_device_alpha_kernel,
)
from newton._src.solvers.phoenx.rl_training.networks import WarpMLP
from newton._src.solvers.phoenx.rl_training.ppo import BufferRollout, ConfigPPO, TrainerPPO
from newton._src.solvers.phoenx.rl_training.sac import BatchSAC
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


class _G1SmokeEnv:
    def __init__(self, obs: wp.array2d[wp.float32], next_obs: wp.array2d[wp.float32]):
        self.world_count = int(obs.shape[0])
        self.obs_dim = int(obs.shape[1])
        self.action_dim = ACTION_DIM_G1
        self.device = obs.device
        self._initial_obs = obs
        self._obs = obs
        self.obs = obs
        self._next_obs = next_obs
        self._rewards = wp.array(np.linspace(-0.2, 0.3, self.world_count, dtype=np.float32), device=self.device)
        self._dones = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_next_obs = next_obs
        self.step_terminateds = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.step_truncateds = wp.ones(self.world_count, dtype=wp.float32, device=self.device)

    def reset(self) -> wp.array2d[wp.float32]:
        self._obs = self._initial_obs
        self.obs = self._obs
        return self._obs

    def observe(self) -> wp.array2d[wp.float32]:
        return self._obs

    def step(self, actions: wp.array2d[wp.float32]) -> tuple[wp.array, wp.array, wp.array]:
        self._obs = self._initial_obs
        self.obs = self._obs
        return self._obs, self._rewards, wp.ones_like(self._dones)


class TestTrainerFlashSAC(unittest.TestCase):
    def test_authoritative_pytorch_actor_fixture(self) -> None:
        """Match a fixed actor forward/backward fixture from upstream PyTorch."""

        # Generated from FlashSAC 87edc9061150ae9e962dd84e6544e27a1554b3ab
        # FlashSACActor with the explicit parameters below, training=False, and
        # loss=sum(mean + 0.3 * log_std). Runtime validation uses only NumPy/Warp.
        device = require_cuda_graph_capture("authoritative FlashSAC actor fixture")
        network = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=1,
            output_dim=2,
            actor_heads=True,
            device=device,
            seed=5,
        )
        network.input_norm.running_mean.assign(np.asarray([0.2, -0.4], dtype=np.float32))
        network.input_norm.running_variance.assign(np.asarray([1.5, 0.7], dtype=np.float32))
        network.input_norm.scale.assign(np.asarray([1.1, 0.8], dtype=np.float32))
        network.input_norm.bias.assign(np.asarray([-0.1, 0.3], dtype=np.float32))
        network.embed_weight.assign(np.asarray([[0.6, 0.4], [-0.2, 0.7]], dtype=np.float32))
        w1, w2 = network.block_weights[0]
        w1.assign(
            np.asarray(
                [
                    [0.2, 0.3, -0.5, 0.1, 0.7, -0.2, 0.4, -0.6],
                    [-0.1, 0.4, 0.2, 0.6, -0.3, 0.5, 0.1, -0.2],
                ],
                dtype=np.float32,
            )
        )
        norm1, norm2 = network.block_norms[0]
        norm1.running_mean.assign(np.asarray([0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.05, -0.15], dtype=np.float32))
        norm1.running_variance.assign(np.asarray([0.8, 1.2, 0.9, 1.4, 0.7, 1.1, 1.3, 0.6], dtype=np.float32))
        norm1.scale.assign(np.asarray([0.9, 1.1, 0.8, 1.2, 0.7, 1.0, 1.3, 0.6], dtype=np.float32))
        norm1.bias.assign(np.asarray([0.0, 0.1, -0.1, 0.2, 0.05, -0.05, 0.15, -0.2], dtype=np.float32))
        w2.assign(
            np.asarray(
                [
                    [0.2, -0.4],
                    [-0.1, 0.2],
                    [0.3, 0.1],
                    [0.4, -0.2],
                    [-0.2, 0.3],
                    [0.1, 0.6],
                    [0.5, -0.1],
                    [-0.3, 0.4],
                ],
                dtype=np.float32,
            )
        )
        norm2.running_mean.assign(np.asarray([0.25, -0.35], dtype=np.float32))
        norm2.running_variance.assign(np.asarray([1.25, 0.85], dtype=np.float32))
        norm2.scale.assign(np.asarray([1.2, 0.75], dtype=np.float32))
        norm2.bias.assign(np.asarray([-0.05, 0.2], dtype=np.float32))
        network.rms_scale.assign(np.asarray([0.9, 1.1], dtype=np.float32))
        network.head_weights[0].assign(np.asarray([[0.3], [-0.7]], dtype=np.float32))
        network.head_biases[0].assign(np.asarray([0.15], dtype=np.float32))
        network.head_weights[1].assign(np.asarray([[-0.4], [0.5]], dtype=np.float32))
        network.head_biases[1].assign(np.asarray([-0.2], dtype=np.float32))
        x = wp.array(np.asarray([[0.5, -1.2], [1.7, 0.3]], dtype=np.float32), device=device)
        output = network.forward_manual(x, training=False)
        expected_output = np.asarray([[-0.5968491, -2.73498], [-0.7396883, -1.9618626]], dtype=np.float32)
        np.testing.assert_allclose(output.numpy(), expected_output, rtol=2.0e-6, atol=2.0e-6)

        output_grad = wp.array(np.asarray([[1.0, 0.3], [1.0, 0.3]], dtype=np.float32), device=device)
        input_grad = wp.empty_like(x)
        network.backward_manual(output_grad, input_grad=input_grad)
        np.testing.assert_allclose(
            input_grad.numpy(),
            np.asarray([[-0.34714878, 0.6968381], [-0.12745747, 0.050784227]], dtype=np.float32),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            network.embed_weight.grad.numpy(),
            np.asarray([[-0.49126515, 0.13067824], [0.28889334, -0.3279666]], dtype=np.float32),
            rtol=2.0e-5,
            atol=2.0e-5,
        )

    def test_graph_replays_reusable_reference_actor(self) -> None:
        """Replay reference actor sampling with persistent buffers and device seeds."""

        device = require_cuda_graph_capture("FlashSAC reusable actor graph replay")
        trainer = TrainerFlashSAC(
            obs_dim=3,
            action_dim=2,
            config=ConfigFlashSAC(actor_hidden_dim=4, actor_num_blocks=1, critic_hidden_dim=4, critic_num_blocks=1),
            device=device,
            seed=17,
        )
        obs = wp.array(
            np.asarray([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9]], dtype=np.float32),
            device=device,
        )
        seed_counter = wp.array(np.asarray([11], dtype=np.int32), dtype=wp.int32, device=device)
        trainer.reserve_buffers(3)
        trainer.act_reuse_seed_counter(obs, seed_counter=seed_counter)
        with wp.ScopedCapture(device=device) as capture:
            actions, log_probs = trainer.act_reuse_seed_counter(obs, seed_counter=seed_counter)

        seed_counter.assign(np.asarray([31], dtype=np.int32))
        wp.capture_launch(capture.graph)
        first_actions = actions.numpy().copy()
        first_log_probs = log_probs.numpy().copy()
        seed_counter.assign(np.asarray([47], dtype=np.int32))
        wp.capture_launch(capture.graph)
        second_actions = actions.numpy().copy()
        seed_counter.assign(np.asarray([31], dtype=np.int32))
        wp.capture_launch(capture.graph)
        repeated_actions = actions.numpy().copy()
        repeated_log_probs = log_probs.numpy().copy()

        self.assertEqual(actions.shape, (3, 2))
        self.assertEqual(log_probs.shape, (3,))
        self.assertTrue(np.isfinite(first_actions).all())
        self.assertTrue(np.isfinite(first_log_probs).all())
        self.assertFalse(np.array_equal(first_actions, second_actions))
        np.testing.assert_array_equal(repeated_actions, first_actions)
        np.testing.assert_array_equal(repeated_log_probs, first_log_probs)

    def test_transposed_batch_moments_match_reference_reduction(self) -> None:
        """Preserve two-pass BatchNorm moments across tails and input dtypes."""

        device = require_cuda_graph_capture("FlashSAC transposed BatchNorm moments")
        rng = np.random.default_rng(401)
        rows = 65
        for dtype in (wp.float32, wp.float16):
            for width in (99, 128, 256):
                values = rng.normal(size=(rows, width)).astype(np.float32)
                source = wp.array(values, dtype=dtype, device=device)
                transposed = wp.empty((width, rows), dtype=dtype, device=device)
                old_mean = wp.empty(width, dtype=wp.float32, device=device)
                old_variance = wp.empty(width, dtype=wp.float32, device=device)
                old_inv_std = wp.empty(width, dtype=wp.float32, device=device)
                new_mean = wp.empty(width, dtype=wp.float32, device=device)
                new_variance = wp.empty(width, dtype=wp.float32, device=device)
                new_inv_std = wp.empty(width, dtype=wp.float32, device=device)
                wp.launch(
                    _batch_moments_tile_kernel,
                    dim=(width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[source, rows, 1.0e-5],
                    outputs=[old_mean, old_variance, old_inv_std],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                wp.launch(
                    _transpose_2d_tile_kernel,
                    dim=((rows + 31) // 32, (width + 31) // 32, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[source],
                    outputs=[transposed],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                wp.launch(
                    _batch_moments_transposed_tile_kernel,
                    dim=(width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[transposed, rows, 1.0e-5],
                    outputs=[new_mean, new_variance, new_inv_std],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                np.testing.assert_array_equal(new_mean.numpy(), old_mean.numpy())
                np.testing.assert_array_equal(new_variance.numpy(), old_variance.numpy())
                np.testing.assert_array_equal(new_inv_std.numpy(), old_inv_std.numpy())
                output_grad = wp.array(rng.normal(size=(rows, width)).astype(np.float32), device=device)
                transposed_grad = wp.empty((width, rows), dtype=wp.float32, device=device)
                scale = wp.array(rng.normal(size=width).astype(np.float32), device=device)
                old_grads = [wp.empty(width, dtype=wp.float32, device=device) for _ in range(4)]
                new_grads = [wp.empty(width, dtype=wp.float32, device=device) for _ in range(4)]
                wp.launch(
                    _batch_norm_backward_amp_tile_kernel,
                    dim=(width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[source, output_grad, old_mean, old_inv_std, scale],
                    outputs=old_grads,
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                wp.launch(
                    _transpose_2d_tile_kernel,
                    dim=((rows + 31) // 32, (width + 31) // 32, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[output_grad],
                    outputs=[transposed_grad],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                wp.launch(
                    _batch_norm_backward_amp_transposed_tile_kernel,
                    dim=(width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[transposed, transposed_grad, new_mean, new_inv_std, scale],
                    outputs=new_grads,
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
                for new_grad, old_grad in zip(new_grads, old_grads, strict=True):
                    np.testing.assert_array_equal(new_grad.numpy(), old_grad.numpy())

    def test_reference_backbone_forward_equations_and_initialization(self) -> None:
        """Match reference normalization, residual, head, and orthogonal equations."""

        device = require_cuda_graph_capture("FlashSAC reference backbone equations")
        network = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=1,
            output_dim=2,
            actor_heads=False,
            device=device,
            seed=11,
        )
        network.input_norm.running_mean.assign(np.asarray([1.0, -1.0], dtype=np.float32))
        network.input_norm.running_variance.assign(np.asarray([4.0, 9.0], dtype=np.float32))
        network.input_norm.scale.assign(np.asarray([2.0, 3.0], dtype=np.float32))
        network.input_norm.bias.assign(np.asarray([0.5, -0.5], dtype=np.float32))
        network.embed_weight.assign(np.eye(2, dtype=np.float32))
        for weight in network.block_weights[0]:
            weight.zero_()
        network.rms_scale.assign(np.asarray([2.0, 0.5], dtype=np.float32))
        network.head_weights[0].assign(np.eye(2, dtype=np.float32))
        network.head_biases[0].assign(np.asarray([0.25, -0.25], dtype=np.float32))
        x_np = np.asarray([[3.0, 2.0]], dtype=np.float32)
        output = network.forward(wp.array(x_np, device=device), requires_grad=False).numpy()

        normalized = (x_np - np.asarray([1.0, -1.0], dtype=np.float32)) / np.sqrt(
            np.asarray([4.0, 9.0], dtype=np.float32) + 1.0e-5
        )
        normalized = normalized * np.asarray([2.0, 3.0], dtype=np.float32) + np.asarray([0.5, -0.5], dtype=np.float32)
        rms = np.sqrt(np.mean(normalized * normalized, axis=-1, keepdims=True) + 1.0e-6)
        expected = normalized / rms * np.asarray([2.0, 0.5], dtype=np.float32)
        expected += np.asarray([0.25, -0.25], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=1.0e-6, atol=1.0e-6)

        square = NetworkFlashSAC(
            input_dim=4,
            hidden_dim=4,
            num_blocks=0,
            output_dim=2,
            actor_heads=False,
            device=device,
            seed=13,
        ).embed_weight.numpy()
        np.testing.assert_allclose(square.T @ square, np.eye(4), rtol=0.0, atol=2.0e-6)

    def test_reference_batch_norm_running_state_and_parameter_ema(self) -> None:
        """Use unbiased running variance and exclude running state from target EMA."""

        device = require_cuda_graph_capture("FlashSAC reference BatchNorm state")
        online = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=17,
        )
        target = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=19,
        )
        target.default_training = True
        values = wp.array(np.asarray([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32), device=device)
        target.forward(values, requires_grad=False)
        np.testing.assert_allclose(target.input_norm.running_mean.numpy(), [0.02, 0.05], atol=1.0e-7)
        np.testing.assert_allclose(target.input_norm.running_variance.numpy(), [1.01, 1.07], atol=1.0e-7)

        running_before = target.input_norm.running_mean.numpy().copy()
        online.input_norm.running_mean.assign(np.asarray([9.0, 8.0], dtype=np.float32))
        target.head_biases[0].zero_()
        online.head_biases[0].fill_(2.0)
        target.soft_update_from(online, 0.5)
        np.testing.assert_array_equal(target.input_norm.running_mean.numpy(), running_before)
        np.testing.assert_allclose(target.head_biases[0].numpy(), [1.0])

    def test_tile_batch_norm_moments_across_shapes_and_graph_replays(self) -> None:
        """Match centered NumPy moments across shapes with stable captured storage."""

        device = require_cuda_graph_capture("FlashSAC tiled BatchNorm reductions")
        rng = np.random.default_rng(271)
        for rows, width in ((512, 128), (2048, 256), (512, 1024)):
            values = rng.normal(loc=5.0, scale=2.0, size=(rows, width)).astype(np.float32)
            x = wp.array(values, device=device)
            norm = _UnitBatchNorm(width, device)
            scratch = norm.scratch(rows)
            self.assertIs(scratch, norm.scratch(rows))
            with wp.ScopedCapture(device=device) as capture:
                wp.launch(
                    _batch_moments_tile_kernel,
                    dim=(width, _TILE_REDUCTION_BLOCK_DIM),
                    inputs=[x, rows, norm.eps],
                    outputs=[scratch.mean, scratch.variance, scratch.inv_std],
                    block_dim=_TILE_REDUCTION_BLOCK_DIM,
                    device=device,
                )
            wp.capture_launch(capture.graph)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(scratch.mean.numpy(), values.mean(axis=0), rtol=2.0e-6, atol=2.0e-6)
            np.testing.assert_allclose(scratch.variance.numpy(), values.var(axis=0), rtol=3.0e-6, atol=3.0e-6)
            np.testing.assert_allclose(
                scratch.inv_std.numpy(), 1.0 / np.sqrt(values.var(axis=0) + norm.eps), rtol=3.0e-6, atol=3.0e-6
            )
            capture.graph = None

    def test_packed_rms_reductions_and_shared_mlp_projection(self) -> None:
        """Match multi-shape RMS reductions and shared MLP unit projection."""

        device = require_cuda_graph_capture("FlashSAC packed RMS reductions")
        rng = np.random.default_rng(277)
        for rows, width in ((512, 128), (2048, 256), (512, 1024)):
            values = rng.normal(size=(rows, width)).astype(np.float32)
            grads = rng.normal(size=(rows, width)).astype(np.float32)
            scale = rng.uniform(0.5, 1.5, size=width).astype(np.float32)
            x = wp.array(values, device=device)
            output_grad = wp.array(grads, device=device)
            scale_array = wp.array(scale, device=device)
            inv_rms = wp.empty(rows, dtype=wp.float32, device=device)
            projection = wp.empty(rows, dtype=wp.float32, device=device)
            scale_grad = wp.empty(width, dtype=wp.float32, device=device)
            _launch_rms_inv(x, 1.0e-6, inv_rms)
            _launch_rms_backward_stats(x, output_grad, scale_array, 1.0e-6, inv_rms, projection)
            _launch_rms_scale_grad(x, output_grad, inv_rms, scale_grad)
            expected_inv = 1.0 / np.sqrt(np.mean(values * values, axis=1) + 1.0e-6)
            expected_projection = np.sum(grads * scale[None, :] * values, axis=1)
            expected_scale_grad = np.sum(grads * values * expected_inv[:, None], axis=0)
            np.testing.assert_allclose(inv_rms.numpy(), expected_inv, rtol=3.0e-6, atol=3.0e-6)
            np.testing.assert_allclose(projection.numpy(), expected_projection, rtol=5.0e-6, atol=2.0e-4)
            np.testing.assert_allclose(scale_grad.numpy(), expected_scale_grad, rtol=8.0e-6, atol=2.0e-4)

        network = WarpMLP((127, 256), device=device, seed=281)
        weights = rng.normal(size=(127, 256)).astype(np.float32)
        network.weights[0].assign(weights)
        with wp.ScopedCapture(device=device) as capture:
            network.normalize_weights()
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        normalized = network.weights[0].numpy()
        np.testing.assert_allclose(np.sum(normalized * normalized, axis=0), 1.0, rtol=2.0e-6, atol=2.0e-6)
        capture.graph = None

    def test_reference_actor_log_std_smooth_mapping(self) -> None:
        """Map raw actor standard deviations smoothly into upstream bounds."""

        device = require_cuda_graph_capture("FlashSAC smooth log-std mapping")
        network = NetworkFlashSAC(
            input_dim=1,
            hidden_dim=1,
            num_blocks=0,
            output_dim=2,
            actor_heads=True,
            device=device,
            seed=23,
        )
        network.embed_weight.fill_(1.0)
        network.head_weights[1].zero_()
        x = wp.ones((1, 1), dtype=wp.float32, device=device)
        for raw in (-20.0, 0.0, 20.0):
            network.head_biases[1].fill_(raw)
            mapped = float(network.forward(x, requires_grad=False).numpy()[0, 1])
            expected = -10.0 + 12.0 * 0.5 * (1.0 + math.tanh(raw))
            self.assertAlmostEqual(mapped, expected, places=5)

    def test_reference_backbone_backward_and_temperature_gradient(self) -> None:
        """Match a finite-difference network gradient and exact temperature gradient."""

        device = require_cuda_graph_capture("FlashSAC reference gradient equations")
        network = NetworkFlashSAC(
            input_dim=2,
            hidden_dim=2,
            num_blocks=0,
            output_dim=1,
            actor_heads=False,
            device=device,
            seed=29,
        )
        x_np = np.asarray([[-1.0, 0.5], [0.2, 1.3], [1.4, -0.7]], dtype=np.float32)
        x = wp.array(x_np, device=device)
        output = network.forward_manual(x)
        network.backward_manual(wp.ones(output.shape, dtype=wp.float32, device=device))
        analytic = float(network.embed_weight.grad.numpy()[0, 0])
        weights = network.embed_weight.numpy().copy()
        epsilon = 1.0e-3
        losses = []
        for delta in (-epsilon, epsilon):
            perturbed = weights.copy()
            perturbed[0, 0] += delta
            network.embed_weight.assign(perturbed)
            losses.append(float(network.forward(x, requires_grad=False, training=True).numpy().sum()))
        network.embed_weight.assign(weights)
        finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, finite_difference, delta=2.0e-2)

        log_probs = wp.array(np.asarray([-2.0, -1.0], dtype=np.float32), device=device)
        log_alpha = wp.array(np.asarray([math.log(0.25)], dtype=np.float32), device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(
                _flash_sac_alpha_loss_kernel,
                dim=2,
                inputs=[log_probs, log_alpha, 2, -0.5],
                outputs=[loss],
                device=device,
            )
        tape.backward(loss)
        expected = 0.25 * ((2.0 + 0.5) + (1.0 + 0.5)) / 2.0
        self.assertAlmostEqual(float(loss.numpy()[0]), expected, places=6)
        self.assertAlmostEqual(float(log_alpha.grad.numpy()[0]), expected, places=6)

    def test_flash_sac_defaults_and_replay_warmup(self) -> None:
        """Match upstream defaults and enforce replay warmup."""

        device = require_cuda_graph_capture("FlashSAC replay tests")
        config = ConfigFlashSAC()
        self.assertTrue(config.distributional_critic)
        self.assertEqual(config.distributional_atoms, 101)
        self.assertEqual(config.policy_frequency, 2)
        self.assertAlmostEqual(config.initial_alpha, 0.01)
        self.assertEqual(config.buffer_max_length, 1_000_000)
        self.assertEqual(config.buffer_min_length, 10_000)
        self.assertEqual(config.sample_batch_size, 2048)
        replay = BufferReplayFlashSAC(
            minimum_size=2,
            capacity=4,
            obs_dim=2,
            action_dim=1,
            batch_size=2,
            device=device,
        )
        self.assertFalse(replay.can_sample())
        replay.add_batch(
            wp.zeros((2, 2), dtype=wp.float32, device=device),
            wp.zeros((2, 1), dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.zeros((2, 2), dtype=wp.float32, device=device),
        )
        self.assertTrue(replay.can_sample())

    def test_fused_ensemble_matches_separate_reference_update(self) -> None:
        """Match separate twin critics across one exact fused forward and backward update."""

        device = require_cuda_graph_capture("FlashSAC fused ensemble equivalence")
        config = ConfigFlashSAC(
            actor_hidden_dim=8,
            actor_num_blocks=1,
            critic_hidden_dim=8,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_rewards=False,
            policy_frequency=1,
        )
        fused = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=901)
        separate = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=901)
        del separate._critic_ensemble, separate._target_critic_ensemble
        rng = np.random.default_rng(903)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
            actions=wp.array(rng.normal(size=(16, 2)).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=16).astype(np.float32), device=device),
            dones=wp.array((rng.random(16) < 0.2).astype(np.float32), device=device),
            next_obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
        )
        fused_stats = fused.update(batch, seed=907)
        separate_stats = separate.update(batch, seed=907)
        np.testing.assert_allclose(
            tuple(fused_stats.__dict__.values()),
            tuple(separate_stats.__dict__.values()),
            rtol=1.0e-6,
            atol=2.0e-6,
        )
        for fused_network, separate_network in (
            (fused.actor.net, separate.actor.net),
            (fused.critic1, separate.critic1),
            (fused.critic2, separate.critic2),
            (fused.target_critic1, separate.target_critic1),
            (fused.target_critic2, separate.target_critic2),
        ):
            for fused_array, separate_array in zip(
                fused_network.state_arrays(), separate_network.state_arrays(), strict=True
            ):
                np.testing.assert_allclose(fused_array.numpy(), separate_array.numpy(), rtol=1.0e-6, atol=2.0e-6)

    def test_reference_backbone_integrated_update(self) -> None:
        """Run a finite update through residual reference actor and critics."""

        device = require_cuda_graph_capture("FlashSAC integrated reference update")
        rng = np.random.default_rng(31)
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_observations=True,
            normalize_rewards=False,
            policy_frequency=1,
        )
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=37)
        self.assertIsInstance(trainer.actor.net, NetworkFlashSAC)
        self.assertFalse(trainer.config.normalize_observations)
        actor_running_before = trainer.actor.net.input_norm.running_mean.numpy().copy()
        target_running_before = trainer.target_critic1.input_norm.running_mean.numpy().copy()
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(8, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=8).astype(np.float32), device=device),
            dones=wp.zeros(8, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
        )
        stats = trainer.update(batch, seed=41)

        self.assertTrue(all(math.isfinite(value) for value in stats.__dict__.values()))
        expected_actor_running = 0.01 * np.concatenate((batch.obs.numpy(), batch.next_obs.numpy()), axis=0).mean(axis=0)
        np.testing.assert_allclose(
            trainer.actor.net.input_norm.running_mean.numpy(), expected_actor_running, atol=1.0e-7
        )
        self.assertFalse(np.array_equal(trainer.actor.net.input_norm.running_mean.numpy(), actor_running_before))
        self.assertFalse(np.array_equal(trainer.target_critic1.input_norm.running_mean.numpy(), target_running_before))
        for network in (trainer.actor.net, trainer.critic1, trainer.critic2):
            for weight in network.weights:
                np.testing.assert_allclose(np.linalg.norm(weight.numpy(), axis=0), 1.0, rtol=0.0, atol=3.0e-6)
            norms = [network.input_norm]
            for norm1, norm2 in network.block_norms:
                norms.extend((norm1, norm2))
            for norm in norms:
                affine_squared = np.sum(norm.scale.numpy() ** 2) + np.sum(norm.bias.numpy() ** 2)
                self.assertAlmostEqual(float(affine_squared), float(norm.width), delta=3.0e-5)
            self.assertAlmostEqual(
                float(np.sum(network.rms_scale.numpy() ** 2)), float(network.hidden_dim), delta=3.0e-5
            )

    def test_flash_sac_constraints_survive_update(self) -> None:
        """Keep unit incoming weights after a full FlashSAC update."""

        device = require_cuda_graph_capture("FlashSAC update tests")
        rng = np.random.default_rng(41)
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, hidden_layers=(8,), device=device, seed=43)
        expected_entropy = 0.5 * 2 * math.log(2.0 * math.pi * math.e * 0.15**2)
        self.assertAlmostEqual(trainer.target_entropy, expected_entropy)
        self.assertAlmostEqual(trainer._scheduled_learning_rate(0), 3.0e-4)
        self.assertAlmostEqual(trainer._scheduled_learning_rate(1_000_000), 1.5e-4)

        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(16, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=16).astype(np.float32), device=device),
            dones=wp.zeros(16, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(16, 3)).astype(np.float32), device=device),
        )
        stats = trainer.update(batch, seed=47)
        self.assertTrue(all(math.isfinite(value) for value in stats.__dict__.values()))
        for network in (trainer.actor.net, trainer.critic1, trainer.critic2):
            for weight in network.weights:
                np.testing.assert_allclose(np.linalg.norm(weight.numpy(), axis=0), 1.0, rtol=0.0, atol=2.0e-6)

    def test_flash_sac_checkpoint_round_trip(self) -> None:
        """Restore networks, targets, temperature, optimizers, counters, and config."""

        device = require_cuda_graph_capture("FlashSAC checkpoint tests")
        rng = np.random.default_rng(131)
        config = ConfigFlashSAC(normalize_observations=False, normalize_rewards=False)
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, hidden_layers=(8,), config=config, device=device, seed=137)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(8, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=8).astype(np.float32), device=device),
            dones=wp.zeros(8, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
        )
        trainer.update(batch, seed=139)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/flash_sac.npz"
            public_rl.save_flash_sac_checkpoint(trainer, path)
            restored = public_rl.load_flash_sac_checkpoint(path, device=device)

        self.assertEqual(restored._update_count, trainer._update_count)
        self.assertEqual(restored._gradient_update_count, trainer._gradient_update_count)
        self.assertEqual(restored.config, trainer.config)
        self.assertIsNone(restored.replay_buffer)
        np.testing.assert_array_equal(restored.log_alpha.numpy(), trainer.log_alpha.numpy())
        np.testing.assert_array_equal(restored._alpha.numpy(), trainer._alpha.numpy())

        for expected_network, actual_network in (
            (trainer.actor.net, restored.actor.net),
            (trainer.critic1, restored.critic1),
            (trainer.critic2, restored.critic2),
            (trainer.target_critic1, restored.target_critic1),
            (trainer.target_critic2, restored.target_critic2),
        ):
            for expected, actual in zip(expected_network.parameters(), actual_network.parameters(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        for expected_optimizer, actual_optimizer in (
            (trainer.actor_optimizer, restored.actor_optimizer),
            (trainer.critic1_optimizer, restored.critic1_optimizer),
            (trainer.critic2_optimizer, restored.critic2_optimizer),
            (trainer.alpha_optimizer, restored.alpha_optimizer),
        ):
            self.assertEqual(actual_optimizer.step_count, expected_optimizer.step_count)
            for expected, actual in zip(expected_optimizer.m, actual_optimizer.m, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            for expected, actual in zip(expected_optimizer.v, actual_optimizer.v, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_reference_backbone_checkpoint_round_trip(self) -> None:
        """Restore reference affine, running-statistic, RMS, and head state."""

        device = require_cuda_graph_capture("FlashSAC reference checkpoint tests")
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_rewards=False,
        )
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=149)
        values = wp.array(
            np.asarray([[-1.0, 0.5, 2.0], [0.0, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=np.float32),
            device=device,
        )
        trainer.actor.net.forward(values, training=True, requires_grad=False)
        trainer.actor.net.rms_scale.assign(np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/flash_sac_reference.npz"
            trainer.save_checkpoint(path)
            restored = TrainerFlashSAC.load_checkpoint(path, device=device)

        self.assertIsInstance(restored.actor.net, NetworkFlashSAC)
        for expected_network, actual_network in (
            (trainer.actor.net, restored.actor.net),
            (trainer.critic1, restored.critic1),
            (trainer.critic2, restored.critic2),
            (trainer.target_critic1, restored.target_critic1),
            (trainer.target_critic2, restored.target_critic2),
        ):
            for expected, actual in zip(expected_network.state_arrays(), actual_network.state_arrays(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_reference_checkpoint_continues_exact_update(self) -> None:
        """Continue a reference trainer with identical update and action results."""

        device = require_cuda_graph_capture("FlashSAC exact checkpoint continuation")
        rng = np.random.default_rng(157)
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_rewards=False,
            policy_frequency=1,
            use_amp=True,
        )
        trainer = TrainerFlashSAC(obs_dim=3, action_dim=2, config=config, device=device, seed=163)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
            actions=wp.array(np.tanh(rng.normal(size=(8, 2))).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=8).astype(np.float32), device=device),
            dones=wp.zeros(8, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(8, 3)).astype(np.float32), device=device),
        )
        trainer.update(batch, seed=167)
        for seed in (168, 169, 170):
            trainer.act(batch.obs, seed=seed)
        trainer._device_noise_repeat_count.assign(np.asarray([2], dtype=np.int32))
        trainer._device_noise_repeat_steps.assign(np.asarray([5], dtype=np.int32))
        trainer._device_exploration_seed.assign(np.asarray([181], dtype=np.int32))
        trainer._device_interaction_seed.assign(np.asarray([191], dtype=np.int32))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/continuation.npz"
            trainer.save_checkpoint(path)
            restored = TrainerFlashSAC.load_checkpoint(path, device=device)

        self.assertTrue(restored.config.use_amp)
        for expected, actual in (
            (trainer._device_noise_repeat_count, restored._device_noise_repeat_count),
            (trainer._device_noise_repeat_steps, restored._device_noise_repeat_steps),
            (trainer._device_exploration_seed, restored._device_exploration_seed),
            (trainer._device_interaction_seed, restored._device_interaction_seed),
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        for seed in range(171, 187):
            for actual, expected in zip(
                restored.act(batch.obs, seed=seed), trainer.act(batch.obs, seed=seed), strict=True
            ):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        expected_stats = trainer.update(batch, seed=173)
        actual_stats = restored.update(batch, seed=173)
        self.assertEqual(actual_stats, expected_stats)
        deterministic_expected = trainer.act(batch.obs, seed=179, deterministic=True)
        deterministic_actual = restored.act(batch.obs, seed=181, deterministic=True)
        for actual, expected in zip(deterministic_actual, deterministic_expected, strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        for expected_network, actual_network in (
            (trainer.actor.net, restored.actor.net),
            (trainer.critic1, restored.critic1),
            (trainer.critic2, restored.critic2),
            (trainer.target_critic1, restored.target_critic1),
            (trainer.target_critic2, restored.target_critic2),
        ):
            for expected, actual in zip(expected_network.state_arrays(), actual_network.state_arrays(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_same_seed_produces_identical_actions_and_updates(self) -> None:
        """Produce bitwise-identical stochastic actions and updates from one seed."""

        device = require_cuda_graph_capture("FlashSAC deterministic seed behavior")
        config = ConfigFlashSAC(
            actor_hidden_dim=4,
            actor_num_blocks=1,
            critic_hidden_dim=4,
            critic_num_blocks=1,
            distributional_atoms=7,
            normalize_rewards=False,
            policy_frequency=1,
        )
        first = TrainerFlashSAC(obs_dim=2, action_dim=1, config=config, device=device, seed=191)
        second = TrainerFlashSAC(obs_dim=2, action_dim=1, config=config, device=device, seed=191)
        obs = wp.array(
            np.asarray([[-1.0, 0.25], [0.5, 1.0], [1.5, -0.75], [0.0, 0.5]], dtype=np.float32), device=device
        )
        for actual, expected in zip(first.act(obs, seed=193), second.act(obs, seed=193), strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        batch = BatchSAC(
            obs=obs,
            actions=wp.array(np.asarray([[-0.5], [0.25], [0.75], [0.0]], dtype=np.float32), device=device),
            rewards=wp.array(np.asarray([-1.0, 0.5, 1.0, 0.25], dtype=np.float32), device=device),
            dones=wp.zeros(4, dtype=wp.float32, device=device),
            next_obs=wp.array(obs.numpy() * 0.9, device=device),
        )
        self.assertEqual(first.update(batch, seed=197), second.update(batch, seed=197))
        for expected_network, actual_network in (
            (first.actor.net, second.actor.net),
            (first.critic1, second.critic1),
            (first.critic2, second.critic2),
        ):
            for expected, actual in zip(expected_network.state_arrays(), actual_network.state_arrays(), strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_replay_wrap_determinism_n_step_and_round_trip(self) -> None:
        """Preserve circular replay, deterministic samples, and pending multi-world n-step state."""

        device = require_cuda_graph_capture("FlashSAC replay persistence")
        replay = BufferReplayFlashSAC(
            minimum_size=0,
            capacity=4,
            obs_dim=1,
            action_dim=1,
            batch_size=4,
            n_step=2,
            gamma=0.5,
            normalize_rewards=True,
            device=device,
        )

        def add_step(
            target: BufferReplayFlashSAC, step: int, *, terminated: bool = False, truncated: bool = False
        ) -> None:
            target.add_batch(
                wp.array([[float(step)], [100.0 + step]], dtype=wp.float32, device=device),
                wp.array([[step * 0.1], [-step * 0.1]], dtype=wp.float32, device=device),
                wp.array([float(step + 1), float((step + 1) * 10)], dtype=wp.float32, device=device),
                wp.array([float(terminated), 0.0], dtype=wp.float32, device=device),
                wp.array([[float(step + 1)], [101.0 + step]], dtype=wp.float32, device=device),
                truncateds=wp.array([0.0, float(truncated)], dtype=wp.float32, device=device),
            )

        add_step(replay, 0, truncated=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/replay.npz"
            replay.save(path)
            restored = BufferReplayFlashSAC.load(path, device=device)
        add_step(replay, 1, terminated=True)
        add_step(restored, 1, terminated=True)
        np.testing.assert_allclose(replay.rewards.numpy()[:2], [2.0, 10.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(replay.dones.numpy()[:2], [1.0, 0.0])
        np.testing.assert_array_equal(replay.next_obs.numpy()[:2, 0], [2.0, 101.0])
        for step in (2, 3):
            add_step(replay, step)
            add_step(restored, step)
        self.assertEqual((replay.size, replay.position), (4, 2))
        self.assertEqual((restored.size, restored.position), (4, 2))
        for expected, actual in (
            (replay.obs, restored.obs),
            (replay.actions, restored.actions),
            (replay.rewards, restored.rewards),
            (replay.dones, restored.dones),
            (replay.next_obs, restored.next_obs),
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        expected_sample = replay.sample(seed=211)
        actual_sample = restored.sample(seed=211)
        for actual, expected in zip(actual_sample.__dict__.values(), expected_sample.__dict__.values(), strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_amp_matches_authoritative_autocast_boundaries(self) -> None:
        """Match traced FlashSAC autocast dtypes without requiring PyTorch."""

        device = require_cuda_graph_capture("FlashSAC autocast dtype boundaries")
        trainer = TrainerFlashSAC(
            obs_dim=3,
            action_dim=2,
            config=ConfigFlashSAC(
                actor_hidden_dim=4,
                actor_num_blocks=1,
                critic_hidden_dim=4,
                critic_num_blocks=1,
                distributional_atoms=5,
                normalize_rewards=False,
                use_amp=True,
            ),
            device=device,
            seed=313,
        )
        obs = wp.zeros((4, 3), dtype=wp.float32, device=device)
        trainer.actor.net.forward_manual(obs, training=True)
        actor_cache = trainer.actor.net._manual_cache
        self.assertIsNotNone(actor_cache)
        self.assertEqual(actor_cache["input_normalized"].dtype, wp.float32)
        actor_block = actor_cache["blocks"][0]
        for value in actor_block:
            self.assertEqual(value.dtype, wp.float16)
        self.assertEqual(actor_cache["rms_input"].dtype, wp.float16)
        self.assertEqual(actor_cache["normalized"].dtype, wp.float16)
        self.assertEqual(actor_cache["heads"][0].dtype, wp.float16)

        trainer.actor.net.forward_reuse(obs)
        interaction_buffers = trainer.actor.net._forward_buffers
        self.assertEqual(interaction_buffers["input_normalized"].dtype, wp.float32)
        self.assertEqual(interaction_buffers["embed"].dtype, wp.float32)
        self.assertEqual(interaction_buffers["normalized"].dtype, wp.float32)
        for block in interaction_buffers["blocks"]:
            for value in block:
                self.assertEqual(value.dtype, wp.float32)

        critic_input = wp.zeros((4, 5), dtype=wp.float32, device=device)
        trainer._critic_ensemble.forward_manual(critic_input, training=True)
        critic_cache = trainer._critic_ensemble._manual_cache
        self.assertEqual(critic_cache["input_normalized"].dtype, wp.float32)
        residual, pre1, _normed1, activated1, pre2, _normed2, activated2 = critic_cache["blocks"][0]
        self.assertEqual(residual.dtype, wp.float16)
        self.assertEqual(pre1.dtype, wp.float16)
        self.assertEqual(pre2.dtype, wp.float16)
        self.assertEqual(activated1.dtype, wp.float32)
        self.assertEqual(activated2.dtype, wp.float32)
        self.assertEqual(critic_cache["rms_input"].dtype, wp.float32)
        self.assertEqual(critic_cache["normalized"].dtype, wp.float32)
        self.assertEqual(critic_cache["heads"].dtype, wp.float16)

    def test_amp_dual_output_producers_match_separate_casts(self) -> None:
        """Match FP32 producers and their separately rounded FP16 mirrors bit for bit."""

        device = require_cuda_graph_capture("FlashSAC AMP dual-output producers")
        rng = np.random.default_rng(727)
        rows, width = 5, 7
        source = wp.array(rng.normal(size=(rows, width)).astype(np.float16), device=device)
        mean = wp.array(rng.normal(size=width).astype(np.float32), device=device)
        variance = wp.array((rng.random(width) + 0.25).astype(np.float32), device=device)
        inv_std = wp.array((1.0 / np.sqrt(variance.numpy() + 1.0e-5)).astype(np.float32), device=device)
        scale = wp.array(rng.normal(size=width).astype(np.float32), device=device)
        bias = wp.array(rng.normal(size=width).astype(np.float32), device=device)
        old = wp.empty((rows, width), dtype=wp.float32, device=device)
        dual = wp.empty_like(old)
        mirror = wp.empty((rows, width), dtype=wp.float16, device=device)

        wp.launch(
            _batch_norm_inv_std_amp_kernel,
            dim=source.shape,
            inputs=[source, mean, inv_std, scale, bias],
            outputs=[old],
            device=device,
        )
        wp.launch(
            _batch_norm_inv_std_amp_dual_kernel,
            dim=source.shape,
            inputs=[source, mean, inv_std, scale, bias, False],
            outputs=[dual, mirror],
            device=device,
        )
        np.testing.assert_array_equal(dual.numpy(), old.numpy())
        np.testing.assert_array_equal(mirror.numpy(), old.numpy().astype(np.float16))

        wp.launch(
            _batch_norm_kernel,
            dim=source.shape,
            inputs=[source, mean, variance, scale, bias, 1.0e-5],
            outputs=[old],
            device=device,
        )
        wp.launch(
            _batch_norm_amp_dual_kernel,
            dim=source.shape,
            inputs=[source, mean, variance, scale, bias, 1.0e-5, True],
            outputs=[dual, mirror],
            device=device,
        )
        expected = np.maximum(old.numpy(), np.float32(0.0))
        np.testing.assert_array_equal(dual.numpy(), expected)
        np.testing.assert_array_equal(mirror.numpy(), expected.astype(np.float16))

        residual = wp.array(rng.normal(size=(rows, width)).astype(np.float32), device=device)
        wp.launch(
            _residual_add_mixed_f32_kernel,
            dim=source.shape,
            inputs=[source, residual],
            outputs=[old],
            device=device,
        )
        wp.launch(
            _residual_add_mixed_dual_kernel,
            dim=source.shape,
            inputs=[source, residual],
            outputs=[dual, mirror],
            device=device,
        )
        np.testing.assert_array_equal(dual.numpy(), old.numpy())
        np.testing.assert_array_equal(mirror.numpy(), old.numpy().astype(np.float16))

        inv_rms = wp.array((rng.random(rows) + 0.5).astype(np.float32), device=device)
        wp.launch(
            _rms_norm_kernel,
            dim=source.shape,
            inputs=[residual, scale, inv_rms],
            outputs=[old],
            device=device,
        )
        wp.launch(
            _rms_norm_dual_kernel,
            dim=source.shape,
            inputs=[residual, scale, inv_rms],
            outputs=[dual, mirror],
            device=device,
        )
        np.testing.assert_array_equal(dual.numpy(), old.numpy())
        np.testing.assert_array_equal(mirror.numpy(), old.numpy().astype(np.float16))

    def test_amp_matches_authoritative_pytorch_stage_fixture(self) -> None:
        """Match fixed autocast values generated from FlashSAC commit 87edc906."""

        device = require_cuda_graph_capture("FlashSAC authoritative autocast values")
        source = np.asarray(
            [
                [-1.2, 0.3, 2.1],
                [0.7, -0.8, 1.3],
                [2.4, 0.1, -0.5],
                [-0.9, 1.7, 0.2],
                [0.4, -1.1, 1.8],
                [1.2, 0.6, -2.0],
                [-0.3, 2.2, 0.9],
                [1.6, -0.4, 0.5],
            ],
            dtype=np.float16,
        )
        values = wp.array(source, dtype=wp.float16, device=device)
        mean = wp.empty(3, dtype=wp.float32, device=device)
        variance = wp.empty(3, dtype=wp.float32, device=device)
        inv_std = wp.empty(3, dtype=wp.float32, device=device)
        wp.launch(
            _batch_moments_tile_kernel,
            dim=(3, _TILE_REDUCTION_BLOCK_DIM),
            inputs=[values, 8, 1.0e-5],
            outputs=[mean, variance, inv_std],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=device,
        )
        wp.launch(
            _round_batch_moments_f16_kernel,
            dim=3,
            inputs=[mean, variance, 1.0e-5],
            outputs=[inv_std],
            device=device,
        )
        np.testing.assert_array_equal(mean.numpy(), np.asarray([0.48754883, 0.32495117, 0.53759766], dtype=np.float32))
        np.testing.assert_array_equal(variance.numpy(), np.asarray([1.3564453, 1.1689453, 1.546875], dtype=np.float32))
        np.testing.assert_allclose(inv_std.numpy(), [0.85861576, 0.9249173, 0.80403024], rtol=1.0e-6)
        normalized = wp.empty(values.shape, dtype=wp.float32, device=device)
        wp.launch(
            _batch_norm_inv_std_amp_kernel,
            dim=values.shape,
            inputs=[
                values,
                mean,
                inv_std,
                wp.ones(3, dtype=wp.float32, device=device),
                wp.zeros(3, dtype=wp.float32, device=device),
            ],
            outputs=[normalized],
            device=device,
        )
        np.testing.assert_allclose(normalized.numpy()[0], [-1.448914, -0.02303261, 1.2562972], rtol=2.0e-6, atol=2.0e-6)

        output_grad = wp.array(np.linspace(-0.8, 1.3, 24, dtype=np.float32).reshape(8, 3), device=device)
        scale = wp.array(np.asarray([0.7, 1.1, -0.4], dtype=np.float32), device=device)
        mean_grad = wp.empty(3, dtype=wp.float32, device=device)
        variance_grad = wp.empty(3, dtype=wp.float32, device=device)
        scale_grad = wp.empty(3, dtype=wp.float32, device=device)
        bias_grad = wp.empty(3, dtype=wp.float32, device=device)
        input_grad = wp.empty(values.shape, dtype=wp.float16, device=device)
        wp.launch(
            _batch_norm_backward_amp_tile_kernel,
            dim=(3, _TILE_REDUCTION_BLOCK_DIM),
            inputs=[values, output_grad, mean, inv_std, scale],
            outputs=[mean_grad, variance_grad, scale_grad, bias_grad],
            block_dim=_TILE_REDUCTION_BLOCK_DIM,
            device=device,
        )
        wp.launch(
            _batch_norm_input_grad_amp_kernel,
            dim=values.shape,
            inputs=[values, output_grad, scale, mean, inv_std, mean_grad, variance_grad],
            outputs=[input_grad],
            device=device,
        )
        np.testing.assert_array_equal(
            input_grad.numpy()[[0, 7]],
            np.asarray(
                [[-0.41845703, -0.97216797, 0.21887207], [0.47216797, 1.0703125, -0.30615234]],
                dtype=np.float16,
            ),
        )
        np.testing.assert_allclose(scale_grad.numpy(), [1.4460932, 1.1144105, -1.7731792], rtol=1.0e-6)
        np.testing.assert_allclose(bias_grad.numpy(), [1.269565, 2.0, 2.7304347], rtol=1.0e-6)

        head = wp.array([[-8.0703125, 9.5390625, -3.498046875]], dtype=wp.float16, device=device)
        bias = wp.zeros(3, dtype=wp.float32, device=device)
        output = wp.empty((1, 6), dtype=wp.float32, device=device)
        wp.launch(_head_bias_amp_kernel, dim=head.shape, inputs=[head, bias, 0], outputs=[output], device=device)
        wp.launch(
            _head_bias_log_std_amp_kernel,
            dim=head.shape,
            inputs=[head, bias, 3, -10.0, 2.0],
            outputs=[output],
            device=device,
        )
        np.testing.assert_array_equal(output.numpy()[0], [-8.0703125, 9.5390625, -3.498046875, -10.0, 2.0, -9.984375])

    def test_amp_rms_norm_matches_authoritative_gradient_fixture(self) -> None:
        """Match authoritative autocast RMSNorm forward and backward values."""

        device = require_cuda_graph_capture("FlashSAC authoritative autocast RMSNorm")
        values = wp.array(
            np.asarray([[-1.2, 0.3, 2.1, 0.7], [0.7, -0.8, 1.3, -0.4]], dtype=np.float16),
            dtype=wp.float16,
            device=device,
        )
        scale = wp.array([0.7, 1.1, -0.4, 0.9], dtype=wp.float32, device=device)
        output_grad = wp.array(
            np.asarray([[0.2, -0.6, 0.8, 1.1], [-0.7, 0.3, 0.4, -0.2]], dtype=np.float16),
            dtype=wp.float16,
            device=device,
        )
        inv_rms = wp.empty(2, dtype=wp.float32, device=device)
        output = wp.empty(values.shape, dtype=wp.float16, device=device)
        _launch_rms_inv(values, 1.0e-6, inv_rms)
        wp.launch(
            _rms_norm_f16_kernel,
            dim=values.shape,
            inputs=[values, scale, inv_rms],
            outputs=[output],
            device=device,
        )

        projection = wp.empty(2, dtype=wp.float32, device=device)
        input_grad = wp.empty(values.shape, dtype=wp.float16, device=device)
        scale_grad = wp.empty(4, dtype=wp.float32, device=device)
        _launch_rms_backward_stats(values, output_grad, scale, 1.0e-6, inv_rms, projection)
        wp.launch(
            _rms_norm_input_grad_f16_kernel,
            dim=values.shape,
            inputs=[values, output_grad, scale, inv_rms, projection],
            outputs=[input_grad],
            device=device,
        )
        _launch_rms_scale_grad(values, output_grad, inv_rms, scale_grad)

        np.testing.assert_array_equal(
            output.numpy(),
            np.asarray(
                [
                    [-0.66259765625, 0.26025390625, -0.66259765625, 0.4970703125],
                    [0.56787109375, -1.01953125, -0.6025390625, -0.4169921875],
                ],
                dtype=np.float16,
            ),
        )
        np.testing.assert_array_equal(
            input_grad.numpy(),
            np.asarray(
                [
                    [0.05963134765625, -0.5078125, -0.16357421875, 0.81005859375],
                    [-0.365478515625, 0.1512451171875, 0.1903076171875, -0.323974609375],
                ],
                dtype=np.float16,
            ),
        )
        np.testing.assert_allclose(
            scale_grad.numpy(), [-0.7573656, -0.42008877, 1.9268548, 0.69996125], rtol=2.0e-6, atol=2.0e-6
        )

    def test_config_and_shape_failures_are_explicit(self) -> None:
        """Reject invalid FlashSAC configuration, replay, and update shapes."""

        device = require_cuda_graph_capture("FlashSAC failure cases")
        with self.assertRaisesRegex(ValueError, "target_sigma"):
            TrainerFlashSAC(obs_dim=2, action_dim=1, config=ConfigFlashSAC(target_sigma=0.0), device=device)
        with self.assertRaisesRegex(ValueError, "block"):
            TrainerFlashSAC(obs_dim=2, action_dim=1, config=ConfigFlashSAC(actor_num_blocks=0), device=device)
        with self.assertRaisesRegex(ValueError, "distributional_atoms"):
            TrainerFlashSAC(obs_dim=2, action_dim=1, config=ConfigFlashSAC(distributional_atoms=1), device=device)
        with self.assertRaisesRegex(ValueError, "use_amp requires"):
            TrainerFlashSAC(
                obs_dim=2,
                action_dim=1,
                hidden_layers=(4,),
                config=ConfigFlashSAC(use_amp=True),
                device=device,
            )
        replay = BufferReplayFlashSAC(minimum_size=0, capacity=2, obs_dim=2, action_dim=1, batch_size=1, device=device)
        with self.assertRaisesRegex(ValueError, "empty"):
            replay.sample(seed=1)
        with self.assertRaisesRegex(ValueError, "Observation dimensions"):
            replay.add_batch(
                wp.zeros((1, 3), dtype=wp.float32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
                wp.zeros(1, dtype=wp.float32, device=device),
                wp.zeros(1, dtype=wp.float32, device=device),
                wp.zeros((1, 3), dtype=wp.float32, device=device),
            )

    def test_flash_sac_runtime_has_no_torch_imports(self) -> None:
        """Keep the pure-Warp FlashSAC runtime free of PyTorch imports."""

        runtime = Path(__file__).parents[1] / "rl_training"
        for filename in ("flash_sac.py", "flash_sac_networks.py", "sac.py", "kernels.py", "optim.py"):
            tree = ast.parse((runtime / filename).read_text(encoding="utf-8"), filename=filename)
            imported = {
                alias.name.split(".", maxsplit=1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import | ast.ImportFrom)
                for alias in node.names
            }
            self.assertNotIn("torch", imported, filename)

    def test_flash_sac_reuses_exploration_noise(self) -> None:
        """Reuse sampled Gaussian noise for the configured repeat duration."""

        device = require_cuda_graph_capture("FlashSAC exploration tests")
        trainer = TrainerFlashSAC(obs_dim=2, action_dim=1, hidden_layers=(8,), device=device, seed=53)
        trainer._noise_repeat_steps = 3
        trainer._noise_repeat_count = 0
        obs = wp.zeros((4, 2), dtype=wp.float32, device=device)
        first = trainer.act(obs, seed=101)[0].numpy()
        second = trainer.act(obs, seed=202)[0].numpy()
        np.testing.assert_array_equal(first, second)

    def test_min_expected_q_selects_one_target_distribution(self) -> None:
        """Select the lower-Q critic distribution for both targets."""

        device = require_cuda_graph_capture("FlashSAC categorical target tests")
        logits1 = wp.array([[8.0, 0.0, 0.0]], dtype=wp.float32, device=device)
        logits2 = wp.array([[0.0, 0.0, 8.0]], dtype=wp.float32, device=device)
        targets1 = wp.zeros_like(logits1)
        targets2 = wp.zeros_like(logits2)
        wp.launch(
            sac_distributional_min_projection_device_alpha_kernel,
            dim=(1, 3),
            inputs=[
                wp.zeros(1, dtype=wp.float32, device=device),
                wp.zeros(1, dtype=wp.float32, device=device),
                logits1,
                logits2,
                wp.zeros(1, dtype=wp.float32, device=device),
                1.0,
                wp.zeros(1, dtype=wp.float32, device=device),
                3,
                -1.0,
                1.0,
            ],
            outputs=[targets1, targets2],
            device=device,
        )
        np.testing.assert_allclose(targets1.numpy(), targets2.numpy(), rtol=0.0, atol=0.0)
        self.assertEqual(int(np.argmax(targets1.numpy()[0])), 0)

    def test_n_step_replay_stops_at_truncation(self) -> None:
        """Accumulate n-step rewards while preserving truncation bootstrap semantics."""

        device = require_cuda_graph_capture("FlashSAC n-step replay tests")
        replay = BufferReplayFlashSAC(
            minimum_size=0,
            capacity=8,
            obs_dim=1,
            action_dim=1,
            batch_size=2,
            n_step=3,
            gamma=0.5,
            normalize_rewards=False,
            device=device,
        )
        for step, rewards in enumerate(([1.0, 10.0], [2.0, 20.0], [3.0, 30.0])):
            replay.add_batch(
                wp.array([[float(step)], [float(step)]], dtype=wp.float32, device=device),
                wp.zeros((2, 1), dtype=wp.float32, device=device),
                wp.array(rewards, dtype=wp.float32, device=device),
                wp.zeros(2, dtype=wp.float32, device=device),
                wp.array([[float(step + 1)], [float(step + 1)]], dtype=wp.float32, device=device),
                truncateds=wp.array([0.0, float(step == 0)], dtype=wp.float32, device=device),
            )
        np.testing.assert_allclose(replay.rewards.numpy()[:2], [2.75, 10.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(replay.dones.numpy()[:2], [0.0, 0.0])
        np.testing.assert_allclose(replay.next_obs.numpy()[:2, 0], [3.0, 1.0], rtol=0.0, atol=0.0)

    def test_reference_distributional_policy_learns_continuous_optimum(self) -> None:
        """Reduce deterministic policy error on a fixed continuous-control optimum."""

        device = require_cuda_graph_capture("FlashSAC deterministic learning regression")
        seed = 3
        rng = np.random.default_rng(seed)
        world_count = 512
        trainer = TrainerFlashSAC(
            obs_dim=2,
            action_dim=1,
            config=ConfigFlashSAC(
                gamma=0.0,
                initial_alpha=0.01,
                target_entropy=0.0,
                actor_lr=1.0e-3,
                critic_lr=1.0e-3,
                alpha_lr=1.0e-3,
                learning_rate_end=1.0e-3,
                learning_rate_decay_steps=1000,
                actor_hidden_dim=32,
                critic_hidden_dim=32,
                actor_num_blocks=1,
                critic_num_blocks=1,
                distributional_atoms=51,
                normalize_rewards=False,
                buffer_min_length=1,
                sample_batch_size=1024,
            ),
            device=device,
            seed=seed,
        )
        replay = BufferReplayFlashSAC(
            minimum_size=1,
            capacity=65536,
            obs_dim=2,
            action_dim=1,
            batch_size=1024,
            normalize_rewards=False,
            device=device,
        )
        eval_obs_np = rng.uniform(-1.0, 1.0, (1024, 2)).astype(np.float32)
        eval_target = np.tanh(1.2 * eval_obs_np[:, 0] - 0.7 * eval_obs_np[:, 1])
        eval_obs = wp.array(eval_obs_np, device=device)
        initial_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        initial_mse = float(np.mean((initial_actions - eval_target) ** 2))

        for update in range(200):
            obs_np = rng.uniform(-1.0, 1.0, (world_count, 2)).astype(np.float32)
            obs = wp.array(obs_np, device=device)
            if update < 4:
                actions_np = rng.uniform(-1.0, 1.0, (world_count, 1)).astype(np.float32)
            else:
                actions_np = trainer.act(obs, seed=seed * 1000 + update)[0].numpy()
            target = np.tanh(1.2 * obs_np[:, 0] - 0.7 * obs_np[:, 1])
            rewards_np = -((actions_np[:, 0] - target) ** 2)
            replay.add_batch(
                obs,
                wp.array(actions_np, device=device),
                wp.array(rewards_np.astype(np.float32), device=device),
                wp.ones(world_count, dtype=wp.float32, device=device),
                obs,
            )
            trainer.update(replay.sample(seed=update), seed=10000 + update, read_stats=False)

        learned_actions = trainer.act(eval_obs, seed=0, deterministic=True)[0].numpy()[:, 0]
        learned_mse = float(np.mean((learned_actions - eval_target) ** 2))
        self.assertGreater(initial_mse, 0.2)
        self.assertLess(learned_mse, 0.05)

    def test_captured_update_matches_eager_state_progression(self) -> None:
        """Match eager state across delayed-actor CUDA graph replays."""

        device = require_cuda_graph_capture("FlashSAC full update graph replay")

        def make_trainer() -> TrainerFlashSAC:
            return TrainerFlashSAC(
                obs_dim=2,
                action_dim=1,
                config=ConfigFlashSAC(
                    actor_hidden_dim=4,
                    critic_hidden_dim=4,
                    actor_num_blocks=1,
                    critic_num_blocks=1,
                    distributional_atoms=5,
                    normalize_rewards=False,
                    learning_rate_warmup_steps=1,
                    learning_rate_decay_steps=8,
                    use_amp=True,
                ),
                device=device,
                seed=241,
            )

        rng = np.random.default_rng(239)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(4, 2)).astype(np.float32), device=device),
            actions=wp.array(rng.uniform(-1.0, 1.0, size=(4, 1)).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=4).astype(np.float32), device=device),
            dones=wp.zeros(4, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(4, 2)).astype(np.float32), device=device),
        )
        eager = make_trainer()
        captured = make_trainer()
        update_graph = captured.capture_update_graph(batch)
        for _ in range(4):
            eager.update(batch, read_stats=False)
            update_graph.launch()

        eager_arrays: list[wp.array] = []
        captured_arrays: list[wp.array] = []
        for name in ("actor", "critic1", "critic2", "target_critic1", "target_critic2"):
            eager_network = getattr(eager, name)
            captured_network = getattr(captured, name)
            if name == "actor":
                eager_network = eager_network.net
                captured_network = captured_network.net
            eager_arrays.extend(eager_network.state_arrays())
            captured_arrays.extend(captured_network.state_arrays())
        eager_arrays.extend((eager.log_alpha, eager._alpha, eager._amp_scale, eager._amp_growth_tracker))
        captured_arrays.extend((captured.log_alpha, captured._alpha, captured._amp_scale, captured._amp_growth_tracker))
        for eager_optimizer, captured_optimizer in zip(
            (eager.actor_optimizer, eager.critic1_optimizer, eager.critic2_optimizer, eager.alpha_optimizer),
            (
                captured.actor_optimizer,
                captured.critic1_optimizer,
                captured.critic2_optimizer,
                captured.alpha_optimizer,
            ),
            strict=True,
        ):
            eager_arrays.extend((*eager_optimizer.m, *eager_optimizer.v, eager_optimizer._step_count))
            captured_arrays.extend((*captured_optimizer.m, *captured_optimizer.v, captured_optimizer._step_count))
        for eager_array, captured_array in zip(eager_arrays, captured_arrays, strict=True):
            np.testing.assert_allclose(captured_array.numpy(), eager_array.numpy(), rtol=2.0e-6, atol=2.0e-7)
        self.assertEqual(captured._update_count, 4)
        self.assertEqual(captured._gradient_update_count, 4)
        np.testing.assert_array_equal(captured._device_update_count.numpy(), [4])
        np.testing.assert_array_equal(captured._device_gradient_update_count.numpy(), [4])
        self.assertEqual(captured.actor_optimizer.step_count, eager.actor_optimizer.step_count)
        self.assertEqual(captured.critic1_optimizer.step_count, eager.critic1_optimizer.step_count)
        self.assertLessEqual(captured.actor_optimizer.step_count, 2)
        self.assertLessEqual(captured.critic1_optimizer.step_count, 4)

    def test_amp_overflow_skips_complete_optimizer_step_in_graph(self) -> None:
        """Skip parameters, moments, and counters globally on AMP overflow."""

        device = require_cuda_graph_capture("FlashSAC GradScaler overflow skip")

        def make_trainer() -> TrainerFlashSAC:
            trainer = TrainerFlashSAC(
                obs_dim=2,
                action_dim=1,
                config=ConfigFlashSAC(
                    actor_hidden_dim=4,
                    critic_hidden_dim=4,
                    actor_num_blocks=1,
                    critic_num_blocks=1,
                    distributional_atoms=5,
                    normalize_rewards=False,
                    use_amp=True,
                ),
                device=device,
                seed=271,
            )
            trainer._amp_scale.assign(np.asarray([np.finfo(np.float32).max], dtype=np.float32))
            return trainer

        rng = np.random.default_rng(269)
        batch = BatchSAC(
            obs=wp.array(rng.normal(size=(4, 2)).astype(np.float32), device=device),
            actions=wp.array(rng.normal(size=(4, 1)).astype(np.float32), device=device),
            rewards=wp.array(rng.normal(size=4).astype(np.float32), device=device),
            dones=wp.zeros(4, dtype=wp.float32, device=device),
            next_obs=wp.array(rng.normal(size=(4, 2)).astype(np.float32), device=device),
        )
        eager = make_trainer()
        captured = make_trainer()
        actor_m_before = [value.numpy().copy() for value in captured.actor_optimizer.m]
        graph = captured.capture_update_graph(batch)
        eager.update(batch, read_stats=False)
        graph.launch()

        self.assertEqual(captured.actor_optimizer.step_count, 0)
        self.assertEqual(captured.critic1_optimizer.step_count, 0)
        self.assertEqual(captured.critic2_optimizer.step_count, 0)
        for eager_parameter, captured_parameter in zip(
            eager.actor.net.parameters(), captured.actor.net.parameters(), strict=True
        ):
            np.testing.assert_array_equal(captured_parameter.numpy(), eager_parameter.numpy())
            self.assertTrue(np.isfinite(captured_parameter.numpy()).all())
        for before, moment in zip(actor_m_before, captured.actor_optimizer.m, strict=True):
            np.testing.assert_array_equal(moment.numpy(), before)
        np.testing.assert_array_equal(captured._amp_scale.numpy(), eager._amp_scale.numpy())
        np.testing.assert_array_equal(captured._amp_growth_tracker.numpy(), eager._amp_growth_tracker.numpy())
        self.assertEqual(int(captured._amp_growth_tracker.numpy()[0]), 0)

    def test_amp_scaler_growth_and_backoff_match_upstream(self) -> None:
        """Match upstream GradScaler growth and overflow backoff equations."""

        device = require_cuda_graph_capture("FlashSAC GradScaler progression")
        scale = wp.array([65536.0], dtype=wp.float32, device=device)
        tracker = wp.array([1999], dtype=wp.int32, device=device)
        found_inf = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            amp_update_scale_kernel,
            dim=1,
            inputs=[found_inf, scale, tracker, 2.0, 0.5, 2000],
            device=device,
        )
        np.testing.assert_array_equal(scale.numpy(), [131072.0])
        np.testing.assert_array_equal(tracker.numpy(), [0])

        found_inf.assign(np.asarray([1], dtype=np.int32))
        wp.launch(
            amp_update_scale_kernel,
            dim=1,
            inputs=[found_inf, scale, tracker, 2.0, 0.5, 2000],
            device=device,
        )
        np.testing.assert_array_equal(scale.numpy(), [65536.0])
        np.testing.assert_array_equal(tracker.numpy(), [0])

    def test_reward_normalizer_matches_upstream_initial_epsilon(self) -> None:
        """Include upstream RunningMeanStd epsilon in the initial variance merge."""

        device = require_cuda_graph_capture("FlashSAC reward normalizer epsilon")
        normalizer = RewardNormalizerFlashSAC(gamma=0.0, normalized_return_max=5.0, device=device)
        normalizer.update(
            wp.array([1.0, 3.0], dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
        )
        np.testing.assert_allclose(normalizer.running_mean.numpy(), [2.0], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(normalizer.running_var.numpy(), [1.00005], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(normalizer.running_count.numpy(), [2.0], rtol=0.0, atol=0.0)

    def test_graph_replay_matches_eager_n_step_wrap_and_sampling(self) -> None:
        """Match eager n-step replay across truncation, termination, wrap, and sampling."""

        device = require_cuda_graph_capture("FlashSAC device n-step replay")
        kwargs = {
            "minimum_size": 0,
            "capacity": 5,
            "obs_dim": 1,
            "action_dim": 1,
            "batch_size": 3,
            "n_step": 3,
            "gamma": 0.5,
            "normalize_rewards": False,
            "device": device,
        }
        eager = BufferReplayFlashSAC(**kwargs)
        captured = BufferReplayFlashSAC(**kwargs)
        captured.reserve_graph_buffers(2)
        obs = wp.zeros((2, 1), dtype=wp.float32, device=device)
        actions = wp.zeros((2, 1), dtype=wp.float32, device=device)
        rewards = wp.zeros(2, dtype=wp.float32, device=device)
        terminateds = wp.zeros(2, dtype=wp.float32, device=device)
        truncateds = wp.zeros(2, dtype=wp.float32, device=device)
        next_obs = wp.zeros((2, 1), dtype=wp.float32, device=device)
        with wp.ScopedCapture(device=device) as store_capture:
            captured.add_batch_graph(
                obs,
                actions,
                rewards,
                terminateds,
                next_obs,
                truncateds=truncateds,
            )
        for step in range(5):
            obs.assign(np.asarray([[step], [step + 0.25]], dtype=np.float32))
            actions.assign(np.asarray([[step * 0.1], [-step * 0.1]], dtype=np.float32))
            rewards.assign(np.asarray([step + 1.0, (step + 1.0) * 10.0], dtype=np.float32))
            terminateds.assign(np.asarray([float(step == 1), 0.0], dtype=np.float32))
            truncateds.assign(np.asarray([0.0, float(step == 0)], dtype=np.float32))
            next_obs.assign(np.asarray([[step + 1.0], [step + 1.25]], dtype=np.float32))
            eager.add_batch(obs, actions, rewards, terminateds, next_obs, truncateds=truncateds)
            wp.capture_launch(store_capture.graph)
            captured.advance_graph_host_state()

        self.assertEqual(captured.size, eager.size)
        self.assertEqual(captured.position, eager.position)
        for captured_array, eager_array in zip(
            (captured.obs, captured.actions, captured.rewards, captured.dones, captured.next_obs),
            (eager.obs, eager.actions, eager.rewards, eager.dones, eager.next_obs),
            strict=True,
        ):
            np.testing.assert_allclose(captured_array.numpy(), eager_array.numpy(), rtol=0.0, atol=1.0e-6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/graph_replay.npz"
            captured.save(path)
            restored = BufferReplayFlashSAC.load(path, device=device)
        self.assertEqual((restored.size, restored.position), (captured.size, captured.position))
        self.assertEqual(restored._graph_pending_count_host, captured._graph_pending_count_host)
        for expected, actual in (
            (captured._graph_pending_obs, restored._graph_pending_obs),
            (captured._graph_pending_actions, restored._graph_pending_actions),
            (captured._graph_pending_rewards, restored._graph_pending_rewards),
            (captured._graph_pending_terminateds, restored._graph_pending_terminateds),
            (captured._graph_pending_truncateds, restored._graph_pending_truncateds),
            (captured._graph_pending_next_obs, restored._graph_pending_next_obs),
            (captured._graph_pending_cursor, restored._graph_pending_cursor),
            (captured._graph_pending_count, restored._graph_pending_count),
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        seed_counter = wp.array([17], dtype=wp.int32, device=device)
        with wp.ScopedCapture(device=device) as sample_capture:
            graph_batch = captured.sample_graph_seed_counter(seed_counter)
        wp.capture_launch(sample_capture.graph)
        eager_batch = eager.sample(seed=17)
        for graph_array, eager_array in zip(
            (graph_batch.obs, graph_batch.actions, graph_batch.rewards, graph_batch.dones, graph_batch.next_obs),
            (eager_batch.obs, eager_batch.actions, eager_batch.rewards, eager_batch.dones, eager_batch.next_obs),
            strict=True,
        ):
            np.testing.assert_array_equal(graph_array.numpy(), eager_array.numpy())

    def test_prepare_training_graph_warms_n_step_replay_and_checkpoints(self) -> None:
        """Warm n-step replay graph-natively and preserve it through a checkpoint."""

        device = require_cuda_graph_capture("FlashSAC graph-native warmup lifecycle")
        obs = wp.zeros((2, 3), dtype=wp.float32, device=device)
        next_obs = wp.ones((2, 3), dtype=wp.float32, device=device)
        env = _G1SmokeEnv(obs, next_obs)
        trainer = TrainerFlashSAC(
            obs_dim=3,
            action_dim=ACTION_DIM_G1,
            config=ConfigFlashSAC(
                buffer_max_length=16,
                buffer_min_length=2,
                sample_batch_size=2,
                n_step=3,
                normalize_rewards=False,
                actor_hidden_dim=4,
                critic_hidden_dim=4,
                actor_num_blocks=1,
                critic_num_blocks=1,
                distributional_atoms=5,
            ),
            device=device,
            seed=251,
        )
        training_graph = trainer.prepare_training_graph(
            env,
            updates_per_step=2,
            interactions_per_graph=2,
            seed=257,
        )
        replay = trainer.replay_buffer
        self.assertEqual(replay.n_step, 3)
        self.assertEqual(replay.size, 2)
        self.assertEqual(len(replay._n_step_transitions), 0)
        self.assertEqual(replay._graph_pending_count_host, 3)
        np.testing.assert_allclose(replay.rewards.numpy()[:2], [-0.2, 0.3], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(replay.dones.numpy()[:2], [0.0, 0.0])
        np.testing.assert_array_equal(replay.next_obs.numpy()[:2], next_obs.numpy())

        training_graph.launch()
        self.assertEqual(replay.size, 6)
        self.assertEqual(trainer._update_count, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/graph_warmup_replay.npz"
            replay.save(path)
            restored = BufferReplayFlashSAC.load(path, device=device)
        self.assertEqual((restored.size, restored.position), (replay.size, replay.position))
        self.assertEqual(restored._graph_pending_count_host, replay._graph_pending_count_host)
        for expected, actual in (
            (replay.obs, restored.obs),
            (replay.actions, restored.actions),
            (replay.rewards, restored.rewards),
            (replay.dones, restored.dones),
            (replay.next_obs, restored.next_obs),
            (replay._graph_pending_obs, restored._graph_pending_obs),
            (replay._graph_pending_next_obs, restored._graph_pending_next_obs),
        ):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_overlapped_training_graph_orders_replay_policy_and_checkpoint(self) -> None:
        """Order fixed replay phases, policy snapshots, counters, and checkpoint state."""

        device = require_cuda_graph_capture("FlashSAC overlapped training graph")
        obs = wp.zeros((2, 3), dtype=wp.float32, device=device)
        next_obs = wp.ones((2, 3), dtype=wp.float32, device=device)
        env = _G1SmokeEnv(obs, next_obs)
        config = ConfigFlashSAC(
            buffer_max_length=16,
            buffer_min_length=2,
            sample_batch_size=2,
            n_step=3,
            normalize_rewards=False,
            actor_hidden_dim=4,
            critic_hidden_dim=4,
            actor_num_blocks=1,
            critic_num_blocks=1,
            distributional_atoms=5,
            use_amp=True,
        )
        trainer = TrainerFlashSAC(
            obs_dim=3,
            action_dim=ACTION_DIM_G1,
            config=config,
            device=device,
            seed=271,
        )
        training_graph = trainer.prepare_training_graph(
            env,
            updates_per_step=2,
            interactions_per_graph=2,
            seed=277,
            overlap=True,
        )
        self.assertTrue(training_graph.overlaps_rollout_and_updates)
        self.assertEqual(training_graph.phase, 0)
        self.assertIsNotNone(training_graph.phase_batches)
        phase_batches = training_graph.phase_batches
        if phase_batches is None:
            self.fail("overlapped graph did not retain phase batches")
        phase_zero_before = tuple(
            tuple(
                array.numpy().copy() for array in (batch.obs, batch.actions, batch.rewards, batch.dones, batch.next_obs)
            )
            for batch in phase_batches[0]
        )
        replay_size_before = trainer.replay_buffer.size
        training_graph.launch()
        training_graph.synchronize()

        self.assertEqual(training_graph.phase, 1)
        self.assertEqual(trainer.replay_buffer.size, replay_size_before + 4)
        self.assertEqual(trainer._update_count, 4)
        self.assertEqual(trainer._gradient_update_count, 4)
        for batch, expected_arrays in zip(phase_batches[0], phase_zero_before, strict=True):
            for actual, expected in zip(
                (batch.obs, batch.actions, batch.rewards, batch.dones, batch.next_obs), expected_arrays, strict=True
            ):
                np.testing.assert_array_equal(actual.numpy(), expected)

        rollout_actor = training_graph.rollout_actor
        self.assertIsNotNone(rollout_actor)
        for expected, actual in zip(trainer.actor.net.state_arrays(), rollout_actor.net.state_arrays(), strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/overlap_checkpoint.npz"
            trainer.save_checkpoint(path)
            restored = TrainerFlashSAC.load_checkpoint(path, device=device)
        self.assertEqual(restored._update_count, trainer._update_count)
        self.assertEqual(restored._gradient_update_count, trainer._gradient_update_count)
        for expected, actual in zip(trainer.actor.net.state_arrays(), restored.actor.net.state_arrays(), strict=True):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        training_graph.close()
        self.assertIsNone(training_graph.rollout_graph)
        self.assertIsNone(training_graph.update_stream)
        self.assertIsNone(training_graph.prepare_stream)
        self.assertIsNone(training_graph.phase_batches)
        self.assertEqual(training_graph.retained_arrays, ())

    def test_real_g1_end_to_end_training_graph(self) -> None:
        """Capture real G1 interaction, pre-reset replay, sampling, and learner updates."""

        device = require_cuda_graph_capture("real G1 end-to-end FlashSAC graph")
        env = EnvG1PhoenX(
            ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                max_episode_steps=1,
                auto_reset=True,
                randomize_commands_on_reset=False,
                command_resample_steps=0,
                parse_visuals=False,
            ),
            device=device,
        )
        obs = env.reset_noisy(seed=251)
        trainer = TrainerFlashSAC(
            obs_dim=OBS_DIM_G1,
            action_dim=env.policy_action_dim,
            config=ConfigFlashSAC(
                buffer_max_length=8,
                buffer_min_length=1,
                sample_batch_size=1,
                n_step=1,
                normalize_rewards=False,
                actor_hidden_dim=4,
                critic_hidden_dim=4,
                actor_num_blocks=1,
                critic_num_blocks=1,
                distributional_atoms=5,
            ),
            device=device,
            seed=257,
        )
        replay = BufferReplayFlashSAC(
            minimum_size=1,
            capacity=8,
            obs_dim=OBS_DIM_G1,
            action_dim=env.policy_action_dim,
            batch_size=1,
            n_step=1,
            normalize_rewards=False,
            device=device,
        )
        replay.reserve_graph_buffers(1)
        warm_actions, _ = trainer.act(obs, seed=263)
        pre_step_obs = wp.clone(obs)
        env.step(warm_actions)
        replay.add_batch_graph(
            pre_step_obs,
            warm_actions,
            env.step_rewards,
            env.step_terminateds,
            env.step_next_obs,
            truncateds=env.step_truncateds,
        )
        training_graph = trainer.capture_training_graph(
            env,
            replay,
            updates_per_step=2,
            interactions_per_graph=2,
            seed=269,
            overlap=True,
        )
        self.assertTrue(training_graph.overlaps_rollout_and_updates)
        batch_ptr = replay.reserve_graph_buffers(1).obs.ptr
        sim_time_before = env.sim_time
        stats = training_graph.run(1, stats_interval=1)

        self.assertEqual(len(stats), 1)
        self.assertTrue(all(np.isfinite(value) for value in stats[0].__dict__.values()))
        self.assertEqual(replay.size, 3)
        self.assertEqual(trainer._update_count, 4)
        self.assertEqual(trainer._gradient_update_count, 4)
        self.assertEqual(replay.reserve_graph_buffers(1).obs.ptr, batch_ptr)
        self.assertAlmostEqual(env.sim_time, sim_time_before + 2.0 * env.config.frame_dt)
        np.testing.assert_array_equal(env.step_terminateds.numpy(), [0.0])
        np.testing.assert_array_equal(env.step_truncateds.numpy(), [1.0])
        np.testing.assert_array_equal(replay.dones.numpy()[:3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(replay.next_obs.numpy()[2:3], env.step_next_obs.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(trainer._device_update_count.numpy(), [4])
        np.testing.assert_array_equal(trainer._device_interaction_seed.numpy(), [271])
        repeat_count = int(trainer._device_noise_repeat_count.numpy()[0])
        repeat_steps = int(trainer._device_noise_repeat_steps.numpy()[0])
        self.assertGreaterEqual(repeat_count, 1)
        self.assertLessEqual(repeat_count, repeat_steps)

    def test_reward_normalizer_bounds_discounted_return(self) -> None:
        """Bound normalized rewards by the configured return range."""

        device = require_cuda_graph_capture("FlashSAC reward normalization tests")
        replay = BufferReplayFlashSAC(
            minimum_size=0,
            capacity=2,
            obs_dim=1,
            action_dim=1,
            batch_size=1,
            normalized_return_max=5.0,
            device=device,
        )
        replay.add_batch(
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.array([10.0], dtype=wp.float32, device=device),
            wp.zeros(1, dtype=wp.float32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
        )
        np.testing.assert_allclose(replay.sample(seed=1).rewards.numpy(), [5.0], rtol=0.0, atol=1.0e-6)

    def test_training_stats_interval_avoids_host_readback(self) -> None:
        """Skip synchronized diagnostics on the configured fast training path."""

        device = require_cuda_graph_capture("FlashSAC stats interval")
        obs = wp.array(np.asarray([[-0.5, 0.25], [0.75, -0.25]], dtype=np.float32), device=device)
        next_obs = wp.array(obs.numpy() * 0.9, device=device)
        trainer = TrainerFlashSAC(
            obs_dim=2,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(4,),
            config=ConfigFlashSAC(
                buffer_max_length=4,
                buffer_min_length=2,
                sample_batch_size=2,
                normalize_observations=False,
                normalize_rewards=False,
            ),
            device=device,
            seed=223,
        )

        def fail_readback() -> StatsSACUpdate:
            self.fail("fast FlashSAC path synchronized update diagnostics")

        trainer._read_update_stats = fail_readback
        stats = public_rl.train_flash_sac(
            _G1SmokeEnv(obs, next_obs),
            trainer,
            interaction_steps=1,
            stats_interval=0,
            seed=227,
        )
        self.assertEqual(stats, [])
        self.assertEqual(trainer._update_count, 1)
        with self.assertRaisesRegex(ValueError, "stats_interval"):
            public_rl.train_flash_sac(
                _G1SmokeEnv(obs, next_obs),
                trainer,
                interaction_steps=1,
                updates_per_step=0,
                stats_interval=-1,
            )

    def test_training_prefers_compact_policy_action_dimension(self) -> None:
        """Validate and store actions against an environment policy interface."""

        device = require_cuda_graph_capture("FlashSAC compact policy action interface")
        obs = wp.zeros((2, 3), dtype=wp.float32, device=device)
        next_obs = wp.ones((2, 3), dtype=wp.float32, device=device)
        env = _G1SmokeEnv(obs, next_obs)
        env.policy_action_dim = 2
        trainer = TrainerFlashSAC(
            obs_dim=3,
            action_dim=2,
            hidden_layers=(4,),
            config=ConfigFlashSAC(
                buffer_max_length=4,
                buffer_min_length=4,
                sample_batch_size=2,
                normalize_observations=False,
            ),
            device=device,
            seed=229,
        )
        stats = public_rl.train_flash_sac(env, trainer, interaction_steps=1, updates_per_step=0, seed=233)
        self.assertEqual(stats, [])
        if trainer.replay_buffer is None:
            self.fail("compact-action collection did not initialize replay")
        self.assertEqual(trainer.replay_buffer.actions.shape, (4, 2))
        with self.assertRaisesRegex(ValueError, "policy interface"):
            incompatible = TrainerFlashSAC(obs_dim=3, action_dim=3, hidden_layers=(4,), device=device)
            public_rl.train_flash_sac(env, incompatible, interaction_steps=1, updates_per_step=0)

    def test_update_uses_upstream_network_order(self) -> None:
        """Update actor and temperature before critic and target networks."""

        device = require_cuda_graph_capture("FlashSAC update-order tests")
        trainer = TrainerFlashSAC(
            obs_dim=2,
            action_dim=1,
            hidden_layers=(4,),
            config=ConfigFlashSAC(normalize_observations=False),
            device=device,
        )
        events: list[str] = []
        trainer._update_actor = lambda batch, seed: events.append("actor")
        trainer._update_alpha = lambda batch, seed: events.append("temperature")
        trainer._update_critics = lambda batch, seed: events.append("critic")
        trainer.target_critic1.soft_update_from = lambda source, tau: events.append("target1")
        trainer.target_critic2.soft_update_from = lambda source, tau: events.append("target2")
        batch = BatchSAC(
            obs=wp.zeros((2, 2), dtype=wp.float32, device=device),
            actions=wp.zeros((2, 1), dtype=wp.float32, device=device),
            rewards=wp.zeros(2, dtype=wp.float32, device=device),
            dones=wp.zeros(2, dtype=wp.float32, device=device),
            next_obs=wp.zeros((2, 2), dtype=wp.float32, device=device),
        )
        trainer.update(batch, read_stats=False)
        self.assertEqual(events, ["actor", "temperature", "critic", "target1", "target2"])

    def test_expanded_layers_match_upstream_capacities(self) -> None:
        """Match upstream block expansion depths and distinct feature widths."""

        self.assertEqual(TrainerFlashSAC._expanded_block_layers(128, 2), (128, 512, 128, 512, 128))
        self.assertEqual(TrainerFlashSAC._expanded_block_layers(256, 2), (256, 1024, 256, 1024, 256))

    def test_g1_timeout_preserves_flash_sac_transition(self) -> None:
        """Expose timeout truncation and pre-reset next observations for FlashSAC."""

        device = require_cuda_graph_capture("G1 FlashSAC timeout transition test")
        env = EnvG1PhoenX(
            ConfigEnvG1PhoenX(
                world_count=1,
                sim_substeps=1,
                solver_iterations=1,
                max_episode_steps=1,
                auto_reset=True,
                randomize_commands_on_reset=False,
                command_resample_steps=0,
                parse_visuals=False,
            ),
            device=device,
        )
        actions = wp.zeros((1, ACTION_DIM_G1), dtype=wp.float32, device=device)
        returned_obs, _rewards, dones = env.step(actions)
        np.testing.assert_array_equal(dones.numpy(), [1.0])
        np.testing.assert_array_equal(env.step_truncateds.numpy(), [1.0])
        np.testing.assert_array_equal(env.step_terminateds.numpy(), [0.0])
        self.assertEqual(env.step_next_obs.shape, returned_obs.shape)
        self.assertTrue(np.isfinite(env.step_next_obs.numpy()).all())

    def test_g1_flash_sac_and_ppo_trainer_workflows_smoke(self) -> None:
        """Exercise one PPO and FlashSAC update with G1-sized synthetic transitions."""

        device = require_cuda_graph_capture("G1 FlashSAC and PPO trainer smoke test")
        world_count = 4
        obs_np = np.linspace(-0.5, 0.5, world_count * OBS_DIM_G1, dtype=np.float32).reshape(world_count, OBS_DIM_G1)
        obs = wp.array(obs_np, dtype=wp.float32, device=device)
        next_obs = wp.array(obs_np * 0.9, dtype=wp.float32, device=device)

        flash_config = ConfigFlashSAC(
            buffer_max_length=8,
            buffer_min_length=world_count,
            sample_batch_size=world_count,
            normalize_observations=False,
            normalize_rewards=False,
        )
        flash = TrainerFlashSAC(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(8,),
            config=flash_config,
            device=device,
            seed=71,
        )
        ppo = TrainerPPO(
            obs_dim=OBS_DIM_G1,
            action_dim=ACTION_DIM_G1,
            hidden_layers=(8,),
            config=ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                normalize_observations=False,
            ),
            device=device,
            seed=73,
        )

        flash_actions, _flash_log_probs = flash.act(obs, seed=79)
        ppo_actions, ppo_log_probs, _ppo_values = ppo.act(obs, seed=83)
        self.assertEqual(flash_actions.shape, (world_count, ACTION_DIM_G1))
        self.assertEqual(ppo_actions.shape, (world_count, ACTION_DIM_G1))
        for actions in (flash_actions.numpy(), ppo_actions.numpy()):
            self.assertTrue(np.isfinite(actions).all())
            self.assertLessEqual(float(np.max(np.abs(actions))), 1.0)

        flash_updates = public_rl.train_flash_sac(_G1SmokeEnv(obs, next_obs), flash, interaction_steps=1, seed=97)
        self.assertTrue(flash.can_start_training())
        if flash.replay_buffer is None:
            self.fail("public FlashSAC workflow did not initialize replay")
        np.testing.assert_allclose(flash.replay_buffer.next_obs.numpy()[:world_count], next_obs.numpy())
        np.testing.assert_array_equal(flash.replay_buffer.dones.numpy()[:world_count], 0.0)

        buffer = BufferRollout(
            num_steps=1, num_envs=world_count, obs_dim=OBS_DIM_G1, action_dim=ACTION_DIM_G1, device=device
        )
        buffer.obs.assign(obs)
        buffer.actions.assign(ppo_actions)
        buffer.old_log_probs.assign(ppo_log_probs)
        buffer.advantages.assign(np.linspace(-0.2, 0.2, world_count, dtype=np.float32))
        buffer.returns.assign(np.linspace(-0.1, 0.3, world_count, dtype=np.float32))
        buffer.old_values.zero_()

        ppo_stats = ppo.update(buffer)
        flash_stats = flash_updates[0]
        self.assertTrue(all(math.isfinite(value) for value in ppo_stats.__dict__.values()))
        self.assertTrue(all(math.isfinite(value) for value in flash_stats.__dict__.values()))

    def test_high_level_g1_ppo_uses_compact_policy_actions(self) -> None:
        """Use only controlled G1 joints in the high-level PPO lifecycle."""

        device = require_cuda_graph_capture("compact G1 PPO lifecycle")
        env_config = ConfigEnvG1PhoenX(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            max_episode_steps=2,
            randomize_commands_on_reset=False,
            command_resample_steps=0,
            parse_visuals=False,
        )
        result = public_rl.train_g1_ppo(
            public_rl.ConfigTrainG1PPO(
                iterations=1,
                rollout_steps=2,
                hidden_layers=(4,),
                env_config=env_config,
                ppo_config=ConfigPPO(
                    train_epochs=1,
                    minibatch_size=2,
                    normalize_advantages=False,
                    mirror_loss_coeff=0.25,
                ),
                device=device,
                seed=307,
                randomize_commands=False,
                readback_diagnostics=False,
            )
        )
        self.assertEqual(result.env.policy_action_dim, 12)
        self.assertEqual(result.trainer.action_dim, result.env.policy_action_dim)
        self.assertEqual(result.buffer.actions.shape[1], result.env.policy_action_dim)
        self.assertEqual(len(result.trainer.mirror_map.action_src), result.env.policy_action_dim)

    def test_high_level_g1_ppo_uses_isaaclab_flat_mirror_map(self) -> None:
        """Train one PPO step with the full-action IsaacLab-flat symmetry map."""
        device = require_cuda_graph_capture("IsaacLab-flat G1 PPO lifecycle")
        env_config = g1_recipe.isaaclab_flat_g1_env_config(
            world_count=1,
            sim_substeps=1,
            solver_iterations=1,
            max_episode_steps=1,
            randomize_commands_on_reset=False,
            command_resample_steps=0,
            parse_visuals=False,
        )
        result = public_rl.train_g1_ppo(
            public_rl.ConfigTrainG1PPO(
                iterations=1,
                rollout_steps=1,
                hidden_layers=(4,),
                env_config=env_config,
                ppo_config=ConfigPPO(
                    train_epochs=1,
                    minibatch_size=1,
                    normalize_advantages=False,
                    mirror_loss_coeff=0.25,
                ),
                device=device,
                seed=353,
                randomize_commands=False,
                readback_diagnostics=False,
            )
        )

        self.assertEqual(result.trainer.obs_dim, result.env.obs_dim)
        self.assertEqual(result.trainer.action_dim, ACTION_DIM_G1)
        self.assertEqual(len(result.trainer.mirror_map.obs_src), result.env.obs_dim)
        self.assertEqual(len(result.trainer.mirror_map.action_src), ACTION_DIM_G1)

    def test_real_g1_flash_sac_and_ppo_workflow_smoke(self) -> None:
        """Exercise one real G1 collection and update for each trainer workflow."""

        device = require_cuda_graph_capture("real G1 FlashSAC and PPO workflow smoke")
        env_kwargs = {
            "world_count": 1,
            "sim_substeps": 1,
            "solver_iterations": 1,
            "max_episode_steps": 1,
            "auto_reset": True,
            "randomize_commands_on_reset": False,
            "command_resample_steps": 0,
            "parse_visuals": False,
        }
        flash_env = EnvG1PhoenX(ConfigEnvG1PhoenX(**env_kwargs), device=device)
        ppo_env = EnvG1PhoenX(ConfigEnvG1PhoenX(**env_kwargs), device=device)
        flash_obs = flash_env.reset_noisy(seed=101)
        ppo_obs = ppo_env.reset_noisy(seed=211)
        self.assertEqual(flash_obs.shape, (1, OBS_DIM_G1))
        self.assertEqual(ppo_obs.shape, (1, OBS_DIM_G1))

        flash = TrainerFlashSAC(
            obs_dim=OBS_DIM_G1,
            action_dim=flash_env.policy_action_dim,
            hidden_layers=(4,),
            config=ConfigFlashSAC(
                buffer_max_length=2,
                buffer_min_length=1,
                sample_batch_size=1,
                normalize_observations=False,
                normalize_rewards=False,
            ),
            device=device,
            seed=103,
        )
        ppo = TrainerPPO(
            obs_dim=OBS_DIM_G1,
            action_dim=ppo_env.policy_action_dim,
            hidden_layers=(4,),
            config=ConfigPPO(
                train_epochs=1,
                normalize_advantages=False,
                normalize_observations=False,
            ),
            device=device,
            seed=223,
        )

        flash_updates = public_rl.train_flash_sac(
            flash_env,
            flash,
            interaction_steps=1,
            seed=107,
            reset_at_start=False,
        )
        self.assertEqual(len(flash_updates), 1)
        if flash.replay_buffer is None:
            self.fail("real G1 FlashSAC collection did not initialize replay")
        self.assertEqual(flash.replay_buffer.actions.shape, (2, flash_env.policy_action_dim))
        self.assertEqual(flash_env.step_next_obs.shape, (1, OBS_DIM_G1))
        np.testing.assert_allclose(flash.replay_buffer.next_obs.numpy()[:1], flash_env.step_next_obs.numpy())
        np.testing.assert_array_equal(flash_env.step_terminateds.numpy(), [0.0])
        np.testing.assert_array_equal(flash_env.step_truncateds.numpy(), [1.0])
        np.testing.assert_array_equal(flash.replay_buffer.dones.numpy()[:1], [0.0])

        rollout = BufferRollout(
            num_steps=1,
            num_envs=1,
            obs_dim=OBS_DIM_G1,
            action_dim=ppo_env.policy_action_dim,
            device=device,
        )
        ppo_env.collect_ppo_rollout(ppo, rollout, seed=227)
        self.assertEqual(rollout.obs.shape, (1, OBS_DIM_G1))
        self.assertEqual(rollout.actions.shape, (1, ppo_env.policy_action_dim))
        ppo_stats = ppo.update(rollout)
        self.assertTrue(all(math.isfinite(value) for value in flash_updates[0].__dict__.values()))
        self.assertTrue(all(math.isfinite(value) for value in ppo_stats.__dict__.values()))
        self.assertTrue(np.isfinite(rollout.obs.numpy()).all())
        self.assertTrue(np.isfinite(rollout.actions.numpy()).all())

    def test_isaaclab_flat_full_action_g1_workflows_smoke(self) -> None:
        """Exercise PPO and FlashSAC on the shared full-action G1 recipe."""
        device = require_cuda_graph_capture("full-action G1 PPO and FlashSAC workflow smoke")
        env_kwargs = {
            "world_count": 1,
            "sim_substeps": 1,
            "solver_iterations": 1,
            "max_episode_steps": 1,
            "auto_reset": True,
            "randomize_commands_on_reset": False,
            "command_resample_steps": 0,
            "parse_visuals": False,
        }
        flash_env = EnvG1PhoenX(g1_recipe.isaaclab_flat_g1_env_config(**env_kwargs), device=device)
        ppo_env = EnvG1PhoenX(g1_recipe.isaaclab_flat_g1_env_config(**env_kwargs), device=device)
        flash = TrainerFlashSAC(
            obs_dim=flash_env.obs_dim,
            action_dim=flash_env.policy_action_dim,
            hidden_layers=(4,),
            config=ConfigFlashSAC(
                buffer_max_length=2,
                buffer_min_length=1,
                sample_batch_size=1,
                normalize_observations=False,
                normalize_rewards=False,
            ),
            device=device,
            seed=331,
        )
        ppo = TrainerPPO(
            obs_dim=ppo_env.obs_dim,
            action_dim=ppo_env.policy_action_dim,
            hidden_layers=(4,),
            config=ConfigPPO(train_epochs=1, normalize_advantages=False, normalize_observations=False),
            device=device,
            seed=337,
        )

        flash_updates = public_rl.train_flash_sac(flash_env, flash, interaction_steps=1, seed=347)
        rollout = BufferRollout(
            num_steps=1,
            num_envs=1,
            obs_dim=ppo_env.obs_dim,
            action_dim=ppo_env.policy_action_dim,
            device=device,
        )
        ppo_env.collect_ppo_rollout(ppo, rollout, seed=349)
        ppo_stats = ppo.update(rollout)

        self.assertEqual(flash_env.policy_action_dim, ACTION_DIM_G1)
        self.assertEqual(ppo_env.policy_action_dim, ACTION_DIM_G1)
        self.assertEqual(flash_env.obs_dim, ppo_env.obs_dim)
        self.assertEqual(rollout.actions.shape, (1, ACTION_DIM_G1))
        self.assertEqual(flash.replay_buffer.actions.shape, (2, ACTION_DIM_G1))
        self.assertTrue(all(math.isfinite(value) for value in flash_updates[0].__dict__.values()))
        self.assertTrue(all(math.isfinite(value) for value in ppo_stats.__dict__.values()))


if __name__ == "__main__":
    wp.init()
    unittest.main()
