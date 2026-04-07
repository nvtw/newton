# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic tests for ANYmal C training environment.

Validates that the physics pipeline, actuators, observations, and rewards
behave correctly before committing to a long training run.  Designed to
be readable by AI assistants for automated verification.

Run::

    uv run --extra dev python -m newton.tests.test_anymal_actuators
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton.examples.robot.example_robot_anymal_c_train_onnx import AnymalEnv


class TestAnymalActuators(unittest.TestCase):
    """Validate that joint actuation actually moves the robot."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"
        cls.env = AnymalEnv(num_envs=4, device=cls.device)

    def test_env_creation(self):
        """Environment builds without errors and has correct dimensions."""
        self.assertEqual(self.env.q_stride, 19)  # 7 free + 12 revolute
        self.assertEqual(self.env.qd_stride, 18)  # 6 free + 12 revolute
        self.assertEqual(self.env.obs_dim, 48)
        self.assertEqual(self.env.act_dim, 12)

    def test_initial_height(self):
        """Robot starts at a reasonable height above ground."""
        obs = self.env.reset()
        q = self.env.state.joint_q.numpy()
        heights = q[2::19]  # z-position of base for each env
        for i, h in enumerate(heights):
            self.assertGreater(h, 0.3, f"Env {i}: initial height {h:.3f} too low")
            self.assertLess(h, 0.8, f"Env {i}: initial height {h:.3f} too high")

    def test_observation_shape_and_finiteness(self):
        """Observations are the right shape and contain no NaN/Inf."""
        obs = self.env.reset()
        obs_np = obs.numpy()
        self.assertEqual(obs_np.shape, (4, 48))
        self.assertTrue(np.all(np.isfinite(obs_np)), f"Non-finite obs: {obs_np}")

    def test_zero_action_robot_stands(self):
        """With zero actions the robot should remain standing for several steps."""
        self.env.reset()
        actions = wp.zeros((4, 12), dtype=wp.float32, device=self.device)
        for t in range(50):
            obs, rewards, dones = self.env.step(actions)
        q = self.env.state.joint_q.numpy()
        heights = q[2::19]
        for i, h in enumerate(heights):
            self.assertGreater(h, 0.25, f"Env {i}: fell over with zero actions (h={h:.3f})")

    def test_constant_action_changes_joint_positions(self):
        """Applying a constant nonzero action should move joints from their initial positions."""
        self.env.reset()
        q_before = self.env.state.joint_q.numpy()[:19].copy()

        # Push all joints in one direction
        actions = wp.array(np.full((4, 12), 0.5, dtype=np.float32), device=self.device)
        for _ in range(20):
            self.env.step(actions)

        q_after = self.env.state.joint_q.numpy()[:19]
        joint_delta = np.abs(q_after[7:] - q_before[7:])
        self.assertGreater(
            joint_delta.max(),
            0.01,
            f"No joint movement after 20 steps with action=0.5: max_delta={joint_delta.max():.6f}",
        )

    def test_opposite_actions_produce_different_states(self):
        """Positive vs negative actions should result in different joint states."""
        self.env.reset()

        act_pos = wp.array(np.full((4, 12), 0.5, dtype=np.float32), device=self.device)
        for _ in range(20):
            self.env.step(act_pos)
        q_pos = self.env.state.joint_q.numpy()[:19].copy()

        self.env.reset()
        act_neg = wp.array(np.full((4, 12), -0.5, dtype=np.float32), device=self.device)
        for _ in range(20):
            self.env.step(act_neg)
        q_neg = self.env.state.joint_q.numpy()[:19]

        joint_diff = np.abs(q_pos[7:] - q_neg[7:])
        self.assertGreater(
            joint_diff.max(),
            0.01,
            f"Positive and negative actions produce same state: max_diff={joint_diff.max():.6f}",
        )

    def test_observation_changes_with_action(self):
        """Observations should change when actions are applied."""
        self.env.reset()
        obs0 = self.env.obs.numpy().copy()

        actions = wp.array(np.full((4, 12), 0.3, dtype=np.float32), device=self.device)
        for _ in range(10):
            obs, _, _ = self.env.step(actions)

        obs1 = obs.numpy()
        diff = np.abs(obs1 - obs0).max()
        self.assertGreater(diff, 0.01, f"Observations unchanged after 10 steps: max_diff={diff:.6f}")

    def test_rewards_finite_and_reasonable(self):
        """Rewards should be finite and within a reasonable range."""
        self.env.reset()
        actions = wp.zeros((4, 12), dtype=wp.float32, device=self.device)
        all_rewards = []
        for _ in range(50):
            _, rewards, _ = self.env.step(actions)
            all_rewards.append(rewards.numpy().copy())

        all_rewards = np.array(all_rewards)
        self.assertTrue(np.all(np.isfinite(all_rewards)), "Non-finite rewards detected")
        # IsaacLab rewards are multiplied by frame_dt=0.02, so they're small
        self.assertLess(
            np.abs(all_rewards).max(),
            1.0,
            f"Reward magnitude suspiciously large: {np.abs(all_rewards).max():.4f}",
        )

    def test_termination_on_fall(self):
        """Robot should terminate if it falls below height threshold."""
        self.env.reset()
        # Apply wild random actions to make the robot fall
        rng = np.random.default_rng(42)
        fell = False
        for _ in range(200):
            act = wp.array(rng.standard_normal((4, 12)).astype(np.float32) * 2.0, device=self.device)
            _, _, dones = self.env.step(act)
            if dones.numpy().sum() > 0:
                fell = True
                break
        self.assertTrue(fell, "Robot never fell despite wild random actions for 200 steps")

    def test_reset_restores_initial_state(self):
        """After reset, the robot should be back near initial height."""
        self.env.reset()
        # Make robot fall
        rng = np.random.default_rng(0)
        for _ in range(100):
            act = wp.array(rng.standard_normal((4, 12)).astype(np.float32) * 2.0, device=self.device)
            self.env.step(act)

        # Reset
        self.env.reset()
        q = self.env.state.joint_q.numpy()
        heights = q[2::19]
        for i, h in enumerate(heights):
            self.assertGreater(h, 0.4, f"Env {i}: height {h:.3f} too low after reset")

    def test_memory_stability(self):
        """Training loop should not leak memory over 50 iterations."""
        import resource

        self.env.reset()
        actions = wp.zeros((4, 12), dtype=wp.float32, device=self.device)
        # Warm up
        for _ in range(10):
            self.env.step(actions)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _ in range(50):
            self.env.step(actions)
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        growth_mb = (rss1 - rss0) / 1024
        self.assertLess(growth_mb, 100, f"Memory grew by {growth_mb:.0f} MB over 50 env steps")

    def test_body_name_indices(self):
        """Foot and thigh body indices should match expected names."""
        body_names = [lbl.split("/")[-1] for lbl in self.env.model.body_label[:13]]
        for idx in self.env._foot_body_ids:
            self.assertIn("SHANK", body_names[idx], f"Foot body {idx} is '{body_names[idx]}', expected SHANK")
        for idx in self.env._thigh_body_ids:
            self.assertIn("THIGH", body_names[idx], f"Thigh body {idx} is '{body_names[idx]}', expected THIGH")


if __name__ == "__main__":
    unittest.main()
