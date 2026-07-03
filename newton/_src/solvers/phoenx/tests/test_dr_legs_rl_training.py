# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton.rl as rl
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture


def _quat_rotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    vector = q[:3]
    return value + 2.0 * q[3] * np.cross(vector, value) + 2.0 * np.cross(vector, np.cross(vector, value))


def _max_anchor_residual(env: rl.EnvDrLegsPhoenX) -> float:
    body_q = env.state_0.body_q.numpy()
    joint_parent = env.model.joint_parent.numpy()
    joint_child = env.model.joint_child.numpy()
    joint_xform_parent = env.model.joint_X_p.numpy()
    joint_xform_child = env.model.joint_X_c.numpy()
    residual = 0.0
    for joint in range(int(env.model.joint_count)):
        parent = int(joint_parent[joint])
        child = int(joint_child[joint])
        parent_anchor = body_q[parent, :3] + _quat_rotate(body_q[parent, 3:], joint_xform_parent[joint, :3])
        child_anchor = body_q[child, :3] + _quat_rotate(body_q[child, 3:], joint_xform_child[joint, :3])
        residual = max(residual, float(np.linalg.norm(child_anchor - parent_anchor)))
    return residual


class TestDrLegsPhoenXRL(unittest.TestCase):
    def test_hold_pose_preserves_loops_inside_cuda_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX DR Legs RL tests")
        env = rl.EnvDrLegsPhoenX(
            rl.ConfigEnvDrLegsPhoenX(task="hold", world_count=1, max_episode_steps=0, auto_reset=False),
            device=device,
        )
        actions = wp.zeros((env.world_count, env.action_dim), dtype=wp.float32, device=device)

        env.step(actions)
        env.reset()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(10):
                env.step(actions)
        wp.capture_launch(capture.graph)

        body_q = env.state_0.body_q.numpy().reshape(env.world_count, env.body_stride, 7)
        joint_q = env.state_0.joint_q.numpy()
        self.assertEqual(env.obs.shape, (1, rl.OBS_DIM_DR_LEGS_HOLD))
        self.assertTrue(np.all(np.isfinite(env.obs.numpy())))
        self.assertTrue(np.all(np.isfinite(body_q)))
        np.testing.assert_allclose(env.step_dones.numpy(), 0.0, rtol=0.0, atol=0.0)
        self.assertGreater(float(body_q[0, 0, 2]), 0.2)
        self.assertGreater(float(np.max(np.abs(joint_q))), 1.0e-4)
        self.assertLess(_max_anchor_residual(env), 1.0e-3)

    def test_walk_observation_and_targets_inside_cuda_graph(self) -> None:
        device = require_cuda_graph_capture("PhoenX DR Legs RL tests")
        config = rl.ConfigEnvDrLegsPhoenX(
            task="walk",
            world_count=2,
            sim_substeps=4,
            collision_refresh_interval=2,
            solver_iterations=4,
            command=(0.2, -0.1, 0.3),
            max_episode_steps=0,
            auto_reset=False,
        )
        env = rl.EnvDrLegsPhoenX(config, device=device)
        action_row = np.linspace(-0.1, 0.1, env.action_dim, dtype=np.float32)
        actions = wp.array(np.tile(action_row, (env.world_count, 1)), dtype=wp.float32, device=device)

        env.step(actions)
        env.reset()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(3):
                env.step(actions)
        wp.capture_launch(capture.graph)

        obs = env.obs.numpy()
        targets = env.control.joint_target_q.numpy().reshape(env.world_count, env.joint_stride)
        actuated = env.actuated_joint.numpy()
        expected = config.action_scale * action_row
        self.assertEqual(obs.shape, (env.world_count, rl.OBS_DIM_DR_LEGS_WALK))
        self.assertTrue(np.all(np.isfinite(obs)))
        np.testing.assert_allclose(
            obs[:, 42:45],
            np.tile(np.asarray(config.command, dtype=np.float32), (env.world_count, 1)),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(targets[:, actuated], np.tile(expected, (env.world_count, 1)), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(env.step_dones.numpy(), 0.0, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
