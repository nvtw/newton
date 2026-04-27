# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Large Pyramid``
#
# Triangular 2D pyramid -> 2D pyramid extruded into 3D so the cross-section
# is a stacked-row triangle. Each layer drops one cube from the row below.
# The Box2D ``baseCount`` defaults to 20; we scale to 15 to keep the demo
# settle-time short.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_large_pyramid
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.5
GAP = 0.01
BASE = 15


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        for row in range(BASE):
            count = BASE - row
            x0 = -((count - 1) * (2 * HE + GAP)) * 0.5
            z = HE + GAP + row * (2 * HE + GAP)
            for c in range(count):
                x = x0 + c * (2 * HE + GAP)
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, 0.0, z), q=wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(15.0, 0.0, 8.0), pitch=-12.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
