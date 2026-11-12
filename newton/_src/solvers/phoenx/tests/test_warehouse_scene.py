# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The mixed warehouse of ``newton.examples.phoenx.example_phoenx_warehouse`` as one solver scene.

A small fleet of every robot type shares one world with shelved stock, loose packages,
packages shared by two robots, robots that touch each other, and hinged and sliding
mechanisms: the combination a large single world steps through together.
"""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.examples.phoenx import example_phoenx_warehouse as warehouse


def _run(frames: int, runtime_robots: int = 0) -> warehouse.Example:
    args = argparse.Namespace(
        robots=64, storage_scale=1.0, seed=0, substeps=4, iterations=4, runtime_robots=runtime_robots
    )
    example = warehouse.Example(newton.viewer.ViewerNull(), args)
    for _ in range(frames):
        example.step()
        example.test_post_step()
    return example


@unittest.skipUnless(wp.is_cuda_available(), "the warehouse scene runs under CUDA graph capture")
class TestWarehouseScene(unittest.TestCase):
    def test_two_runs_repeat_bit_for_bit(self):
        """Robots, stock, shared packages, touching robots and mechanisms step deterministically."""
        first = _run(90)
        second = _run(90)
        self.assertIsNotNone(first.solver._reduced_articulation.fused_substep)
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(getattr(first.state, name).numpy(), getattr(second.state, name).numpy())
        # The run exercised contacts between robots, not only robots apart.
        cross = first.solver._reduced_articulation.contact_block_system.cross_column_count
        self.assertGreater(int(cross.numpy()[0]), 0)

    def test_humanoids_arrive_and_leave_at_runtime(self):
        """Humanoids created on the device join the fused robots, stand, and leave again."""
        frames = 4 * warehouse.DOCK_PERIOD + 10  # three arrivals, then two departures
        first = _run(frames, runtime_robots=3)
        self.assertIsNotNone(first.solver._reduced_articulation.fused_substep)
        dynamics = first.solver.dynamics
        self.assertEqual(int(dynamics.error_count.numpy()[0]), 0)
        np.testing.assert_array_equal(first.dock_occupant.numpy(), [-1, -1, 2])
        start = int(dynamics.data.instance_body_start.numpy()[2])
        self.assertAlmostEqual(float(first.state.body_q.numpy()[start, 2]), warehouse.G1_ROOT_HEIGHT, delta=0.01)
        first.test_final()
        second = _run(frames, runtime_robots=3)
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(getattr(first.state, name).numpy(), getattr(second.state, name).numpy())

    def test_a_robot_leaves_and_rejoins_the_running_warehouse(self):
        """A robot of the pool is parked far away between steps and placed back later."""
        example = _run(60)
        model, state = example.model, example.state
        root = int(example.scene.robot_roots["g1"][0])
        joint = int(np.flatnonzero(model.joint_child.numpy() == root)[0])
        coordinate = int(model.joint_q_start.numpy()[joint])
        home = state.joint_q.numpy()[coordinate : coordinate + 3].copy()
        others = np.setdiff1d(np.concatenate(list(example.scene.robot_roots.values())), [root])
        others_before = state.body_q.numpy()[others, :3].copy()

        def place(position: np.ndarray) -> None:
            joint_q = state.joint_q.numpy()
            joint_q[coordinate : coordinate + 3] = position
            state.joint_q.assign(joint_q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        def run(frames: int) -> None:
            for _ in range(frames):
                example.step()
                example.test_post_step()

        # Parked two kilometers away it keeps standing, and nobody else notices.
        place(home + np.array([2000.0, 0.0, 0.0], dtype=np.float32))
        run(60)
        self.assertAlmostEqual(float(state.body_q.numpy()[root, 0]), float(home[0]) + 2000.0, delta=0.05)
        np.testing.assert_allclose(state.body_q.numpy()[others, :3], others_before, atol=0.02)

        # Back at its place it stands where it stood.
        place(home)
        run(60)
        np.testing.assert_allclose(state.body_q.numpy()[root, :3], home, atol=0.05)
        np.testing.assert_allclose(state.body_q.numpy()[others, :3], others_before, atol=0.02)
        example.test_final()


if __name__ == "__main__":
    unittest.main()
