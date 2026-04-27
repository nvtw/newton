# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Contact / Confined``
#
# A grid of spheres confined inside a closed-box cage of static walls.
# No gravity. The 2D demo packs 25x25 = 625 circles inside a 22x21
# cage; here we use 8x8x4 = 256 spheres in a 6x6x4 cage.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_confined
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

R = 0.3
GRID_X = 8
GRID_Y = 8
GRID_Z = 4
WALL_HE = 0.2
HALF_X = GRID_X * R + 0.5
HALF_Y = GRID_Y * R + 0.5
HALF_Z = GRID_Z * R + 0.5


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 16
    gravity = (0.0, 0.0, 0.0)

    def build_scene(self, builder: newton.ModelBuilder):
        # Six static wall boxes forming the cage.
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        for sign in (-1.0, +1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(p=wp.vec3(sign * (HALF_X + WALL_HE), 0.0, 0.0), q=wp.quat_identity()),
                hx=WALL_HE,
                hy=HALF_Y + 2 * WALL_HE,
                hz=HALF_Z + 2 * WALL_HE,
                cfg=wall_cfg,
            )
            builder.add_shape_box(
                -1,
                xform=wp.transform(p=wp.vec3(0.0, sign * (HALF_Y + WALL_HE), 0.0), q=wp.quat_identity()),
                hx=HALF_X,
                hy=WALL_HE,
                hz=HALF_Z + 2 * WALL_HE,
                cfg=wall_cfg,
            )
            builder.add_shape_box(
                -1,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, sign * (HALF_Z + WALL_HE)), q=wp.quat_identity()),
                hx=HALF_X,
                hy=HALF_Y,
                hz=WALL_HE,
                cfg=wall_cfg,
            )

        extents: list = []
        for k in range(GRID_Z):
            for j in range(GRID_Y):
                for i in range(GRID_X):
                    x = (i - (GRID_X - 1) * 0.5) * (2.1 * R)
                    y = (j - (GRID_Y - 1) * 0.5) * (2.1 * R)
                    z = (k - (GRID_Z - 1) * 0.5) * (2.1 * R)
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                    )
                    builder.add_shape_sphere(body, radius=R)
                    extents.append(default_sphere_half_extents(R))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 4.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
