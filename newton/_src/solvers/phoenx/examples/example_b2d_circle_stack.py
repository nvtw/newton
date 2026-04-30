# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Circle Stack``
#
# A column of spheres stacked on top of each other. Spheres are notably
# harder to settle than boxes (single-point manifold + rolling) -- this
# is a stress test for restitution / friction balance.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_circle_stack
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

R = 0.5
N = 8


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 24
    default_friction = 0.6

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        z = R + 0.05
        for _ in range(N):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()))
            builder.add_shape_sphere(body, radius=R)
            extents.append(default_sphere_half_extents(R))
            z += 2 * R + 0.01
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, 0.0, 5.0), pitch=-15.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
