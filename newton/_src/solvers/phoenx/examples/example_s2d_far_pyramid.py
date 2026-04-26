# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Far / Pyramid``
#
# A 10-base pyramid spawned at world coordinate origin = (1e5, -8e4, 0).
# The original solver2d sample tests how far from the world origin the
# solver can hold a stack together: at ~65 km a float32 ULP is
# ~5 mm, which is comparable to the desired contact accuracy. PhoenX
# stores positions in float32 like Box2D; this scene exercises the
# same precision boundary in 3D.
#
# **Known limitation (TODO):** PhoenX has no local-frame substepping,
# so the float32 ULP shows up as ~2 m/s of jitter per frame on a
# settled stack at this distance. The settle test classifies this
# scene as "swing" so the test only catches outright blow-up, not the
# residual jitter. Solving this properly needs a local-frame
# substepping solver pass (Box2D-v3 has one; PhoenX doesn't yet).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_far_pyramid
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
GAP = 0.005
BASE = 10
ORIGIN = (100_000.0, -80_000.0, 0.0)


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 24

    def build_scene(self, builder: newton.ModelBuilder):
        ox, oy, oz = ORIGIN
        # Static ground at the far origin.
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(ox, oy, oz - 1.0), q=wp.quat_identity()),
            hx=100.0, hy=100.0, hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )

        extents: list = []
        for row in range(BASE):
            count = BASE - row
            x_start = -((count - 1) * (2 * HE + GAP)) * 0.5
            z = oz + HE + row * (2 * HE + GAP)
            for c in range(count):
                x = ox + x_start + c * (2 * HE + GAP)
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, oy, z), q=wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        ox, oy, oz = ORIGIN
        viewer.set_camera(pos=wp.vec3(ox + 12.0, oy - 12.0, oz + 5.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
