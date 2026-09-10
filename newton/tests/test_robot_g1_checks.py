# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the G1 example's final-state predicates."""

import unittest
from types import SimpleNamespace

import warp as wp

from newton.examples.robot.example_robot_g1 import Example


class TestRobotG1Checks(unittest.TestCase):
    def test_velocity_threshold(self):
        """Accept rest and reject excessive speed in every twist component."""
        with wp.ScopedDevice("cpu"):
            model = SimpleNamespace(body_count=1, body_label=["body"])
            state = SimpleNamespace(
                body_q=wp.array([wp.transform((0.0, 0.0, 1.0), wp.quat_identity())], dtype=wp.transform),
                body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            )
            example = SimpleNamespace(model=model, state_0=state)
            for speed in (0.0, 0.01, -0.01, 0.02, -0.02):
                for component in range(6):
                    with self.subTest(speed=speed, component=component):
                        velocity = [0.0] * 6
                        velocity[component] = speed
                        state.body_qd.assign([velocity])
                        if abs(speed) < 0.015:
                            Example.test_final(example)
                        else:
                            with self.assertRaisesRegex(ValueError, "all body velocities are small"):
                                Example.test_final(example)


if __name__ == "__main__":
    unittest.main()
