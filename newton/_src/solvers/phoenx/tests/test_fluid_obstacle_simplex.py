# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.solvers.phoenx.fluid import KPMFR3D, KPMFR3DConfig
from newton._src.solvers.phoenx.fluid.obstacles import rasterize_obstacles


class TestFluidObstacleSimplex(unittest.TestCase):
    def _count(self, builder: newton.ModelBuilder) -> int:
        model = builder.finalize(device="cpu")
        solver = KPMFR3D(KPMFR3DConfig((6, 6, 6), order=3), device="cpu")
        rasterize_obstacles(solver, model, smoothing=0.0)
        return int(np.count_nonzero(solver.volume_fraction.numpy()))

    def test_triangle(self):
        """Rasterize a triangle as a resolution-scaled obstacle shell."""
        builder = newton.ModelBuilder()
        builder.add_shape_triangle(-1, (0.2, 0.2, 0.5), (0.8, 0.2, 0.5), (0.2, 0.8, 0.5))
        self.assertGreater(self._count(builder), 0)

    def test_tetrahedron(self):
        """Rasterize the solid interior of a tetrahedron."""
        builder = newton.ModelBuilder()
        builder.add_shape_tetrahedron(
            -1,
            (0.2, 0.2, 0.2),
            (0.8, 0.2, 0.2),
            (0.2, 0.8, 0.2),
            (0.2, 0.2, 0.8),
        )
        self.assertGreater(self._count(builder), 0)


if __name__ == "__main__":
    unittest.main()
