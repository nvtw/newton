# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / RestitutionAndFriction``
#
# A grid of cubes dropped onto a static plane, with friction varying
# along one axis and restitution varying along the other -- visualises
# the full 2D parameter sweep at one frame time.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_restitution_and_friction
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.3
SPACING = 1.2
FRICTIONS = (0.0, 0.2, 0.4, 0.7)
RESTITUTIONS = (0.0, 0.3, 0.6, 0.9)


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        x0 = -((len(FRICTIONS) - 1) * SPACING) * 0.5
        y0 = -((len(RESTITUTIONS) - 1) * SPACING) * 0.5
        for i, mu in enumerate(FRICTIONS):
            for j, e in enumerate(RESTITUTIONS):
                cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu, restitution=e)
                body = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(x0 + i * SPACING, y0 + j * SPACING, 4.0),
                        q=wp.quat_identity(),
                    ),
                )
                builder.add_shape_box(body, hx=HE, hy=HE, hz=HE, cfg=cfg)
                extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 5.0), pitch=-25.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
