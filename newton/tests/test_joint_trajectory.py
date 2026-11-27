# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for sampling authored joint trajectories between their samples."""

import unittest

import numpy as np
import warp as wp

from newton._src.utils.joint_trajectory import sample_joint_trajectories


class TestJointTrajectory(unittest.TestCase):
    def _sample(self, rows, phase, q, qd, sample_dt):
        device = wp.get_preferred_device()
        items, axes = len(rows), q.shape[1]
        coordinates = np.arange(items * axes, dtype=np.int32).reshape(items, axes)
        target_q = wp.zeros(items * axes, dtype=wp.float32, device=device)
        target_qd = wp.zeros(items * axes, dtype=wp.float32, device=device)
        sample_joint_trajectories(
            wp.array(np.asarray(rows, dtype=np.int32), dtype=wp.int32, device=device),
            phase,
            sample_dt,
            wp.array(q, dtype=wp.float32, device=device),
            wp.array(qd, dtype=wp.float32, device=device),
            wp.array(coordinates, dtype=wp.int32, device=device),
            wp.array(coordinates, dtype=wp.int32, device=device),
            target_q,
            target_qd,
        )
        return target_q.numpy().reshape(items, axes), target_qd.numpy().reshape(items, axes)

    def test_hermite_matches_samples_and_is_c1(self):
        """Hit both samples with their velocities; the velocity is the position's derivative."""
        rng = np.random.default_rng(3)
        q = rng.normal(size=(3, 4)).astype(np.float32)
        qd = rng.normal(size=(3, 4)).astype(np.float32)
        h = 1.0 / 60.0
        rows = [[0, 1], [1, 2]]
        for phase, current, following in ((0.0, [0, 1], [0, 1]), (1.0, [1, 2], [1, 2])):
            position, velocity = self._sample(rows, phase, q, qd, h)
            np.testing.assert_allclose(position, q[current if phase == 0.0 else following], atol=1e-6)
            np.testing.assert_allclose(velocity, qd[current if phase == 0.0 else following], atol=1e-4)
        # The end of one interval and the start of the next agree in position and velocity.
        end_q, end_qd = self._sample([[0, 1]], 1.0, q, qd, h)
        start_q, start_qd = self._sample([[1, 2]], 0.0, q, qd, h)
        np.testing.assert_allclose(end_q, start_q, atol=1e-6)
        np.testing.assert_allclose(end_qd, start_qd, atol=1e-4)
        # Central differences of the position match the sampled velocity.
        for phase in (0.2, 0.5, 0.8):
            step = 1.0e-2
            before, _ = self._sample(rows, phase - step, q, qd, h)
            after, _ = self._sample(rows, phase + step, q, qd, h)
            _, velocity = self._sample(rows, phase, q, qd, h)
            np.testing.assert_allclose((after - before) / (2.0 * step * h), velocity, rtol=2e-3, atol=2e-2)

    def test_held_row_keeps_its_sample(self):
        """A group naming one row twice holds that sample at every phase."""
        q = np.asarray([[0.3, -0.2], [1.0, 2.0]], dtype=np.float32)
        qd = np.asarray([[0.5, 0.1], [3.0, 4.0]], dtype=np.float32)
        position, velocity = self._sample([[0, 0]], 0.7, q, qd, 1.0 / 60.0)
        np.testing.assert_array_equal(position, q[:1])
        np.testing.assert_array_equal(velocity, qd[:1])


if __name__ == "__main__":
    unittest.main()
