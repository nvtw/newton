# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify common rigid references without extrapolating to new material contacts."""

import unittest

import numpy as np

from .rigid_patch_reference import PlanarPatchReference


def reference():
    """Build a certified square face on a fixed plane."""
    return PlanarPatchReference(
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        [0, 0, 0.3],
        [[-1, -1, -0.3], [1, -1, -0.3], [1, 1, -0.3], [-1, 1, -0.3]],
        [0, 0, 1],
        face_token="face1",
        normal_dot_min=0.995,
    )


class TestRigidPatchReference(unittest.TestCase):
    def test_resampling_preserves_material_displacement(self):
        """Give unsampled interior witnesses the same rigid translation without rebirth."""
        patch = reference()
        for p in ([0, 0, -0.3], [0.7, 0.4, -0.3], [-0.3, 0.2, -0.3]):
            value = patch.displacement(
                p, [0, 0, 1], np.eye(3), np.zeros(3), np.eye(3), [0.02, -0.01, 0.3], face_token="face1"
            )
            self.assertEqual(value.dtype, np.float32)
            np.testing.assert_allclose(value, [0.02, -0.01, 0], atol=3e-8, rtol=0)

    def test_rotation_retains_spatially_varying_original_targets(self):
        """Retain the rigid rotational displacement at every original point."""
        patch = reference()
        angle = 0.03
        r = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32
        )
        for p in ([-0.7, -0.2, -0.3], [0.8, 0.5, -0.3]):
            actual = patch.displacement(p, [0, 0, 1], np.eye(3), np.zeros(3), r, [0, 0, 0.3], face_token="face1")
            expected = (r - np.eye(3)) @ p
            np.testing.assert_allclose(actual, expected, atol=5e-8, rtol=0)

    def test_common_world_transform_covariance(self):
        """Rotate and translate both bodies without changing relative material history."""
        theta = np.float32(0.4)
        rotation = np.array(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=np.float32
        )
        translation = np.array([0.7, -0.2, 0.5], dtype=np.float32)
        patch = reference()
        transformed = PlanarPatchReference(
            rotation,
            translation,
            rotation,
            rotation @ np.array([0, 0, 0.3]) + translation,
            patch.polygon,
            patch.normal,
            face_token="face1",
            normal_dot_min=0.995,
        )
        p = [0.2, 0.3, -0.3]
        original = patch.displacement(
            p, [0, 0, 1], np.eye(3), np.zeros(3), np.eye(3), [0.02, -0.01, 0.3], face_token="face1"
        )
        actual = transformed.displacement(
            p,
            [0, 0, 1],
            rotation,
            translation,
            rotation,
            rotation @ np.array([0.02, -0.01, 0.3]) + translation,
            face_token="face1",
        )
        np.testing.assert_allclose(actual, rotation @ original, atol=1e-7, rtol=0)

    def test_new_domain_face_normal_and_loss_reject(self):
        """Reject extrapolation, new material faces, rolling normals and ended contact."""
        patch = reference()
        self.assertFalse(patch.contains([1.1, 0, -0.3], [0, 0, 1], face_token="face1"))
        self.assertFalse(patch.contains([0, 0, -0.3], [0, 0, 1], face_token="other_connected_face"))
        self.assertFalse(patch.contains([0, 0, -0.29], [0, 0, 1], face_token="face1"))
        self.assertFalse(patch.contains([0, 0, -0.3], [0, 1, 0], face_token="face1"))
        patch.invalidate()
        self.assertFalse(patch.contains([0, 0, -0.3], [0, 0, 1], face_token="face1"))
        fresh = reference()
        np.testing.assert_array_equal(
            fresh.displacement(
                [0, 0, -0.3], [0, 0, 1], np.eye(3), np.zeros(3), np.eye(3), [0, 0, 0.3], face_token="face1"
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
