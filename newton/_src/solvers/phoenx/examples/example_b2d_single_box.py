# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Single Box``
#
# A single 1 m half-extent cube spawned 1 m above a ground segment with an
# initial linear velocity of 5 m/s along +x. The Box2D demo's 2D ground
# segment becomes a 3D ground plane; the cube's 2D extent ``(1, 1)``
# becomes a 3D ``(1, 1, 1)`` cube.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_single_box
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

EXTENT = 1.0


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, EXTENT + 1.0), q=wp.quat_identity()),
            linear_velocity=(5.0, 0.0, 0.0),
        )
        builder.add_shape_box(body, hx=EXTENT, hy=EXTENT, hz=EXTENT)
        return [default_box_half_extents(EXTENT, EXTENT, EXTENT)]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, 0.0, 3.0), pitch=-15.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
