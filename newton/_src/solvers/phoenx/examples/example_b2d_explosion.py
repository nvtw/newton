# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Forces / Explosion``
#
# A pile of cubes is given a single radial impulse at frame 0 -- mimics
# the blast wave of an explosion. After that the scene settles back to
# rest under gravity. Done with body initial velocity (instead of an
# applied force kernel) so the demo stays self-contained.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_explosion
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.25
N_RING = 24
RADIUS = 1.5


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        impulse_speed = 8.0
        for i in range(N_RING):
            theta = 2.0 * math.pi * i / N_RING
            x = RADIUS * math.cos(theta)
            y = RADIUS * math.sin(theta)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, y, HE + 0.05), q=wp.quat_identity()),
            )
            # Radial outward velocity in the X-Y plane.
            builder.body_qd[body] = (
                0.0,
                0.0,
                0.0,
                float(impulse_speed * math.cos(theta)),
                float(impulse_speed * math.sin(theta)),
                4.0,
            )
            builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
            extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -10.0, 5.0), pitch=-25.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
