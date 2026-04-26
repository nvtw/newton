# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / Colosseum``
#
# Concentric rings of bricks forming a coliseum-like wall around a pile
# of cubes inside. Smaller scale than the original (5 rings * 30 bricks
# instead of 8 * 60) to keep settle time short.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_colosseum
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

BRICK_HX = 0.4
BRICK_HY = 0.15
BRICK_HZ = 0.2
N_RINGS = 5
BRICKS_PER_RING = 30
RADIUS = 4.0
INTERIOR_HE = 0.3


def _quat_z(angle_rad: float):
    half = 0.5 * angle_rad
    return wp.quat(0.0, 0.0, math.sin(half), math.cos(half))


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 16
    default_friction = 0.6

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        for ring in range(N_RINGS):
            z = BRICK_HZ + 0.02 + ring * (2 * BRICK_HZ + 0.01)
            # Stagger every other ring so seams don't align vertically.
            angle_offset = (math.pi / BRICKS_PER_RING) if (ring % 2) else 0.0
            for k in range(BRICKS_PER_RING):
                theta = 2.0 * math.pi * k / BRICKS_PER_RING + angle_offset
                cx = RADIUS * math.cos(theta)
                cy = RADIUS * math.sin(theta)
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(cx, cy, z), q=_quat_z(theta + math.pi * 0.5)),
                )
                builder.add_shape_box(body, hx=BRICK_HX, hy=BRICK_HY, hz=BRICK_HZ)
                extents.append(default_box_half_extents(BRICK_HX, BRICK_HY, BRICK_HZ))

        # Interior pile of cubes.
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    x = (i - 1) * (2.5 * INTERIOR_HE)
                    y = (j - 1) * (2.5 * INTERIOR_HE)
                    z = INTERIOR_HE + 0.02 + k * (2.2 * INTERIOR_HE)
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=INTERIOR_HE, hy=INTERIOR_HE, hz=INTERIOR_HE)
                    extents.append(default_box_half_extents(INTERIOR_HE, INTERIOR_HE, INTERIOR_HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -12.0, 5.0), pitch=-15.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
