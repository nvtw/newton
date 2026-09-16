# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the candidate material-anchor geometry before solver integration."""

import unittest

import numpy as np


def reset_anchors(positions, rotations, point):
    return np.einsum("bji,bj->bi", rotations, point - positions)


def tangent_drift(anchors, positions, rotations, normal):
    points = positions + np.einsum("bij,bj->bi", rotations, anchors)
    difference = points[1] - points[0]
    return difference - normal * np.dot(normal, difference)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


class TestStickyAnchorGeometry(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[1.2, 0.3, -0.7], [0.2, -0.6, 0.8]])
        self.rotations = np.array([rotation([1, 2, 3], 0.7), rotation([3, -1, 2], -1.3)])
        self.point = np.array([0.7, -0.15, 0.1])
        self.normal = np.array([0.0, 0.0, 1.0])
        self.anchors = reset_anchors(self.positions, self.rotations, self.point)

    def test_reset_has_zero_drift_for_rotated_bodies(self):
        """Remove stored elastic displacement at any body orientations."""
        np.testing.assert_allclose(
            tangent_drift(self.anchors, self.positions, self.rotations, self.normal), 0.0, atol=1e-15
        )

    def test_reference_is_objective_under_rigid_frame_change(self):
        """Rotate and translate the experiment without changing its response."""
        moved = self.positions + np.array([[0.001, -0.002, 0.0003], [0.0, 0.0, 0.0]])
        drift = tangent_drift(self.anchors, moved, self.rotations, self.normal)
        frame = rotation([2, -3, 1], 1.9)
        shifted = moved @ frame.T + np.array([-0.4, 0.7, 1.3])
        transformed = tangent_drift(
            self.anchors, shifted, np.einsum("ij,bjk->bik", frame, self.rotations), frame @ self.normal
        )
        np.testing.assert_allclose(transformed, frame @ drift, atol=1e-15)

    def test_endpoint_swap_reverses_drift(self):
        """Keep contact response symmetric under exchanging the two bodies."""
        moved = self.positions + np.array([[0.001, -0.002, 0.0003], [0.0, 0.0, 0.0]])
        forward = tangent_drift(self.anchors, moved, self.rotations, self.normal)
        reverse = tangent_drift(self.anchors[::-1], moved[::-1], self.rotations[::-1], -self.normal)
        np.testing.assert_allclose(reverse, -forward, atol=1e-15)

    def test_reset_preserves_only_motion_after_break(self):
        """Recover subsequent stick displacement once without restoring old drift."""
        moved = self.positions + np.array([[0.001, -0.002, 0.0003], [0.0, 0.0, 0.0]])
        reset = reset_anchors(moved, self.rotations, self.point)
        np.testing.assert_allclose(tangent_drift(reset, moved, self.rotations, self.normal), 0.0, atol=1e-15)
        advanced = moved + np.array([[0.00001, 0.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            tangent_drift(reset, advanced, self.rotations, self.normal), [-0.00001, 0.0, 0.0], atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
