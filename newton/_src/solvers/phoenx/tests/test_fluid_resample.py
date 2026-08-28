# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton._src.solvers.phoenx.fluid import KPMFR3D, KPMFR3DConfig
from newton._src.solvers.phoenx.fluid.export import export_state_3d
from newton._src.solvers.phoenx.fluid.resample import interpolation_matrix


class TestFluidResample(unittest.TestCase):
    def test_interpolation_preserves_polynomials(self):
        """Interpolate every polynomial represented by the nodal basis."""
        points = np.array([-1.0, -0.25, 0.5, 1.0])
        samples = np.linspace(-0.9, 0.9, 7)
        matrix = interpolation_matrix(points, samples)
        for degree in range(len(points)):
            np.testing.assert_allclose(matrix @ points**degree, samples**degree, atol=2.0e-6)

    def test_export_preserves_uniform_state(self):
        """Export a uniform state on the regular subcell grid."""
        solver = KPMFR3D(KPMFR3DConfig((2, 2, 2), order=3), device="cpu")
        solver.initialize(1.0, (0.1, -0.02, 0.03))
        dense = export_state_3d(solver).dense()
        np.testing.assert_allclose(dense[..., 0], 0.0, atol=1.0e-3)
        np.testing.assert_allclose(dense[..., 1], 0.0, atol=1.0e-3)


if __name__ == "__main__":
    unittest.main()
