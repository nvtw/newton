# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.viewer.viewer_gl import ViewerGL


class TestViewerGLPointColors(unittest.TestCase):
    def setUp(self):
        self.viewer = ViewerGL.__new__(ViewerGL)
        self.viewer.device = wp.get_device("cpu")

    def test_normalize_constant_tuple(self):
        colors = self.viewer._normalize_point_colors((0.2, 0.4, 0.6), num_points=3)

        np.testing.assert_allclose(colors.numpy(), np.tile((0.2, 0.4, 0.6), (3, 1)))

    def test_normalize_per_point_numpy_array(self):
        expected = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
        colors = self.viewer._normalize_point_colors(expected, num_points=2)

        np.testing.assert_array_equal(colors.numpy(), expected)

    def test_normalize_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "RGB triplet"):
            self.viewer._normalize_point_colors((0.2, 0.4), num_points=3)


if __name__ == "__main__":
    unittest.main()
