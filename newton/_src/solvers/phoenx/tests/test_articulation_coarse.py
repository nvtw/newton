# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CUDA graph tests for PhoenX articulation coarse correction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.constraint_joint import (
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_REVOLUTE,
)
from newton._src.solvers.phoenx.tests.test_articulation_dvi import _make_adbs_world


def _momentum(world) -> tuple[np.ndarray, np.ndarray]:
    position = world.bodies.position.numpy()[1:]
    velocity = world.bodies.velocity.numpy()[1:]
    angular_velocity = world.bodies.angular_velocity.numpy()[1:]
    linear = velocity.sum(axis=0)
    angular = (angular_velocity + np.cross(position, velocity)).sum(axis=0)
    return linear, angular


class TestPhoenXArticulationCoarse(unittest.TestCase):
    def test_auto_mixed_joint_path_graph_and_momentum(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("articulation coarse tests require CUDA graph capture")

        joint_count = 8
        body1 = np.arange(1, joint_count + 1, dtype=np.int32)
        body2 = body1 + 1
        modes = np.where(
            np.arange(joint_count) % 2 == 0,
            int(JOINT_MODE_REVOLUTE),
            int(JOINT_MODE_BALL_SOCKET),
        ).astype(np.int32)
        positions = np.zeros((joint_count + 2, 3), dtype=np.float32)
        positions[1:, 0] = np.arange(joint_count + 1)
        world = _make_adbs_world(
            device,
            body1,
            body2,
            modes,
            positions_np=positions,
            world_kwargs={
                "substeps": 4,
                "solver_iterations": 1,
                "velocity_iterations": 0,
                "gravity": (0.0, 0.0, 0.0),
                "articulation_coarse_mode": "auto",
                "articulation_coarse_stride": 2,
            },
        )
        self.assertEqual(world.articulation_coarse_setup.mode, "path")
        self.assertEqual(world.articulation_coarse_setup.solver.rows, 3)

        rng = np.random.default_rng(41)
        velocity = np.zeros_like(positions)
        angular_velocity = np.zeros_like(positions)
        velocity[1:] = rng.normal(size=velocity[1:].shape)
        angular_velocity[1:] = rng.normal(size=angular_velocity[1:].shape)
        world.bodies.velocity.assign(velocity)
        world.bodies.angular_velocity.assign(angular_velocity)
        linear_before, angular_before = _momentum(world)
        with wp.ScopedCapture(device=device) as capture:
            world.solve_articulations_dvi_host(dt=1.0e-4, recovery_speed=0.0)
        wp.capture_launch(capture.graph)
        linear_after, angular_after = _momentum(world)

        np.testing.assert_allclose(linear_after, linear_before, rtol=4.0e-6, atol=4.0e-6)
        np.testing.assert_allclose(angular_after, angular_before, rtol=4.0e-6, atol=4.0e-6)

    def test_auto_selects_tree_and_replays_graph(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("articulation coarse tests require CUDA graph capture")

        body1 = np.array([1, 2, 3, 3, 5, 3, 7], dtype=np.int32)
        body2 = np.array([2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        modes = np.where(
            np.arange(body1.size) % 2 == 0,
            int(JOINT_MODE_BALL_SOCKET),
            int(JOINT_MODE_REVOLUTE),
        ).astype(np.int32)
        positions = np.zeros((9, 3), dtype=np.float32)
        positions[1:, 0] = np.arange(8)
        world = _make_adbs_world(
            device,
            body1,
            body2,
            modes,
            positions_np=positions,
            world_kwargs={
                "substeps": 4,
                "solver_iterations": 1,
                "velocity_iterations": 0,
                "gravity": (0.0, 0.0, 0.0),
                "articulation_coarse_mode": "auto",
                "articulation_coarse_stride": 2,
            },
        )
        self.assertEqual(world.articulation_coarse_setup.mode, "tree")
        self.assertEqual(world.articulation_coarse_setup.solver.rows, 3)

        world.step(1.0e-4)
        with wp.ScopedCapture(device=device) as capture:
            world.step(1.0e-4)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(world.bodies.position.numpy()).all())
        self.assertTrue(np.isfinite(world.bodies.velocity.numpy()).all())

    def test_auto_selects_general_graph_for_cycle(self):
        device = wp.get_preferred_device()
        if not device.is_cuda:
            self.skipTest("articulation coarse tests require CUDA graph capture")

        body1 = np.array([1, 2, 3, 4], dtype=np.int32)
        body2 = np.array([2, 3, 4, 1], dtype=np.int32)
        positions = np.zeros((5, 3), dtype=np.float32)
        positions[1:] = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        world = _make_adbs_world(
            device,
            body1,
            body2,
            np.full(body1.size, int(JOINT_MODE_BALL_SOCKET), dtype=np.int32),
            positions_np=positions,
            world_kwargs={
                "gravity": (0.0, 0.0, 0.0),
                "articulation_coarse_mode": "auto",
            },
        )
        self.assertEqual(world.articulation_coarse_setup.mode, "graph")

        rng = np.random.default_rng(53)
        velocity = np.zeros_like(positions)
        angular_velocity = np.zeros_like(positions)
        velocity[1:] = rng.normal(size=velocity[1:].shape)
        angular_velocity[1:] = rng.normal(size=angular_velocity[1:].shape)
        world.bodies.velocity.assign(velocity)
        world.bodies.angular_velocity.assign(angular_velocity)
        linear_before, angular_before = _momentum(world)
        with wp.ScopedCapture(device=device) as capture:
            world.solve_articulations_dvi_host(dt=1.0e-4, recovery_speed=0.0)
        wp.capture_launch(capture.graph)
        linear_after, angular_after = _momentum(world)

        np.testing.assert_allclose(linear_after, linear_before, rtol=4.0e-6, atol=4.0e-6)
        np.testing.assert_allclose(angular_after, angular_before, rtol=4.0e-6, atol=4.0e-6)


if __name__ == "__main__":
    unittest.main()
