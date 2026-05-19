# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Forces / Restitution``
#
# A row of spheres dropped from the same height, each with a different
# restitution coefficient -- visualises bouncy vs dead-stop behaviour.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_restitution
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

RADIUS = 0.4
DROP_Z = 5.0
RESTITUTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        spacing = 1.2
        x0 = -((len(RESTITUTIONS) - 1) * spacing) * 0.5
        for i, e in enumerate(RESTITUTIONS):
            x = x0 + i * spacing
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, 0.0, DROP_Z), q=wp.quat_identity()),
            )
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, restitution=e)
            builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
            extents.append(default_sphere_half_extents(RADIUS))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -6.0, 4.0), pitch=-20.0, yaw=140.0)


if __name__ == "__main__":
    run_ported_example(Example)
