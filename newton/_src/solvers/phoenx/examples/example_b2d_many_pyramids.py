# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Many Pyramids``
#
# A grid of small pyramids -- canonical solver-stress benchmark. Each
# pyramid is base 5 cubes, scaled down vs ``Large Pyramid`` so the grid
# fits into a few-second settle.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_many_pyramids
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.25
GAP = 0.005
BASE = 5
GRID = 4  # GRID x GRID pyramids


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        spacing_x = (BASE + 1) * (2 * HE + GAP)
        spacing_y = (BASE + 1) * (2 * HE + GAP)

        for gy in range(GRID):
            for gx in range(GRID):
                ox = (gx - GRID * 0.5 + 0.5) * spacing_x
                oy = (gy - GRID * 0.5 + 0.5) * spacing_y
                for row in range(BASE):
                    count = BASE - row
                    x0 = -((count - 1) * (2 * HE + GAP)) * 0.5
                    z = HE + GAP + row * (2 * HE + GAP)
                    for c in range(count):
                        x = ox + x0 + c * (2 * HE + GAP)
                        body = builder.add_body(
                            xform=wp.transform(p=wp.vec3(x, oy, z), q=wp.quat_identity()),
                        )
                        builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                        extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(15.0, -15.0, 5.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
