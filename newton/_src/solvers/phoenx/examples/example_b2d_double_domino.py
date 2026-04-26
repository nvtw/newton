# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Double Domino``
#
# A row of dominoes -- each thin tall plank -- triggered by a sphere
# rolled into the first one. Every plank topples its neighbour.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_double_domino
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

HX = 0.04
HY = 0.4
HZ = 0.6
N_DOMINOES = 25
SPACING = 0.6
BALL_R = 0.25


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16
    default_friction = 0.6

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        for i in range(N_DOMINOES):
            x = i * SPACING
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, 0.0, HZ + 0.02), q=wp.quat_identity()),
            )
            builder.add_shape_box(body, hx=HX, hy=HY, hz=HZ)
            extents.append(default_box_half_extents(HX, HY, HZ))
        # Trigger sphere with initial +x velocity into the first domino.
        ball = builder.add_body(
            xform=wp.transform(p=wp.vec3(-1.5, 0.0, BALL_R + 0.02), q=wp.quat_identity()),
        )
        builder.body_qd[ball] = (0.0, 0.0, 0.0, 6.0, 0.0, 0.0)
        builder.add_shape_sphere(ball, radius=BALL_R)
        extents.append(default_sphere_half_extents(BALL_R))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(N_DOMINOES * SPACING * 0.5, -8.0, 1.5), pitch=-10.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
