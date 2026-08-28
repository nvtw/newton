# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import Heightfield
from newton._src.solvers.phoenx.fluid import KPMFR3D, KPMFR3DConfig
from newton._src.solvers.phoenx.fluid.obstacles import rasterize_obstacles


class TestFluidObstacles(unittest.TestCase):
    def _occupied(self, add_shape) -> int:
        builder = newton.ModelBuilder()
        add_shape(builder)
        model = builder.finalize(device="cpu")
        solver = KPMFR3D(KPMFR3DConfig((4, 4, 4), order=3), device="cpu")
        rasterize_obstacles(solver, model, smoothing=0.0)
        return int(np.count_nonzero(solver.volume_fraction.numpy()))

    def test_analytic_shapes(self):
        """Rasterize every analytic Newton collision shape."""
        xform = wp.transform((0.5, 0.5, 0.5), wp.quat_identity())
        shapes = (
            lambda b: b.add_shape_sphere(-1, xform=xform, radius=0.2),
            lambda b: b.add_shape_box(-1, xform=xform, hx=0.2, hy=0.2, hz=0.2),
            lambda b: b.add_shape_capsule(-1, xform=xform, radius=0.12, half_height=0.2),
            lambda b: b.add_shape_cylinder(-1, xform=xform, radius=0.18, half_height=0.2),
            lambda b: b.add_shape_cone(-1, xform=xform, radius=0.2, half_height=0.2),
            lambda b: b.add_shape_ellipsoid(-1, xform=xform, rx=0.2, ry=0.16, rz=0.12),
            lambda b: b.add_shape_plane(-1, xform=xform, width=0.4, length=0.4),
        )
        for add_shape in shapes:
            with self.subTest(add_shape=add_shape):
                self.assertGreater(self._occupied(add_shape), 0)

    def test_mesh_and_convex_shapes(self):
        """Rasterize mesh and convex-hull Newton shapes."""
        xform = wp.transform((0.5, 0.5, 0.5), wp.quat_identity())
        for convex in (False, True):

            def add_shape(builder, convex=convex):
                mesh = newton.Mesh.create_box(0.2, 0.2, 0.2, duplicate_vertices=False)
                if convex:
                    builder.add_shape_convex_hull(-1, xform=xform, mesh=mesh)
                else:
                    builder.add_shape_mesh(-1, xform=xform, mesh=mesh)

            with self.subTest(convex=convex):
                self.assertGreater(self._occupied(add_shape), 0)

    def test_heightfield_shape(self):
        """Rasterize a Newton heightfield as a solid half-space."""

        def add_shape(builder):
            data = np.zeros((3, 3), dtype=np.float32)
            heightfield = Heightfield(data=data, nrow=3, ncol=3, hx=0.5, hy=0.5)
            builder.add_shape_heightfield(
                xform=wp.transform((0.5, 0.5, 0.5), wp.quat_identity()),
                heightfield=heightfield,
            )

        self.assertGreater(self._occupied(add_shape), 0)


if __name__ == "__main__":
    unittest.main()
