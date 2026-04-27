# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Vertical Stack``
#
# 12 cubes (~0.45 m half-extent) stacked vertically; the Box2D demo also
# has an optional bullet-firing mechanism we omit here. The 2D rounded
# polygon becomes a plain 3D box.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_vertical_stack
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.45
GAP = 0.01
ROWS = 12


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        z = HE + GAP
        for _ in range(ROWS):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()))
            builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
            extents.append(default_box_half_extents(HE, HE, HE))
            z += 2 * HE + GAP
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, 0.0, 6.0), pitch=-15.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
