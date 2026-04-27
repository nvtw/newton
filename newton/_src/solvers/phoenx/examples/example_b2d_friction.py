# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Forces / Friction``
#
# A row of cubes sliding down inclined ramps with progressively increasing
# friction. Each ramp tilts about the world +y axis; the cubes start at
# rest at the top of each ramp.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_friction
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

RAMP_HX = 4.0
RAMP_HY = 1.0
RAMP_HZ = 0.05
TILT_DEG = 30.0
CUBE_HE = 0.4
FRICTIONS = (0.0, 0.1, 0.2, 0.4, 0.7)


def _quat_y(angle_rad: float):
    half = 0.5 * angle_rad
    return wp.quat(0.0, math.sin(half), 0.0, math.cos(half))


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        tilt = math.radians(TILT_DEG)
        # Lay ramps along +y; tilt about the +y axis so they fall toward -x.
        spacing = 2.5 * RAMP_HY
        y0 = -((len(FRICTIONS) - 1) * spacing) * 0.5
        for i, mu in enumerate(FRICTIONS):
            yc = y0 + i * spacing
            ramp_z = RAMP_HX * math.sin(tilt) + RAMP_HZ + 0.5
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
            builder.add_shape_box(
                -1,
                xform=wp.transform(p=wp.vec3(0.0, yc, ramp_z), q=_quat_y(-tilt)),
                hx=RAMP_HX,
                hy=RAMP_HY,
                hz=RAMP_HZ,
                cfg=cfg,
            )
            # Cube on top, near upper end of ramp.
            top_x = RAMP_HX * 0.7 * math.cos(tilt)
            top_z = ramp_z + RAMP_HX * 0.7 * math.sin(tilt) + RAMP_HZ + CUBE_HE
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(top_x, yc, top_z), q=wp.quat_identity()),
            )
            cube_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu)
            builder.add_shape_box(body, hx=CUBE_HE, hy=CUBE_HE, hz=CUBE_HE, cfg=cube_cfg)
            extents.append(default_box_half_extents(CUBE_HE, CUBE_HE, CUBE_HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(10.0, 0.0, 5.0), pitch=-15.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
