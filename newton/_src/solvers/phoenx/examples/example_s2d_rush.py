# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Contact / Rush``
#
# 400 spheres arranged in an outward Archimedean spiral around a static
# central sphere. No gravity, no initial velocity -- the spheres start
# in place and then a sweep would drive them inward in the original demo.
# Here we drop the inward force and just watch them settle from the
# overlapping spiral arrangement (tests broad-phase + warm-start on a
# very dense, near-overlapping cluster).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_rush
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

R = 0.5
N = 200  # half the original count to keep settle time reasonable


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 16
    gravity = (0.0, 0.0, 0.0)

    def build_scene(self, builder: newton.ModelBuilder):
        # No ground -- this is a free-floating cluster.
        extents: list = []
        # Static central sphere (anchor).
        builder.add_shape_sphere(
            -1,
            xform=wp.transform_identity(),
            radius=R,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        distance = 5.0
        delta_angle = 1.0 / distance
        delta_distance = 0.05
        angle = 0.0
        for _ in range(N):
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, y, 0.0), q=wp.quat_identity()),
            )
            cfg = newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.2)
            builder.add_shape_sphere(body, radius=R, cfg=cfg)
            extents.append(default_sphere_half_extents(R))
            angle += delta_angle
            distance += delta_distance
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -40.0, 30.0), pitch=-30.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
