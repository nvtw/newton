# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Arch``
#
# Stone-block arch held up only by friction + gravity: 13 wedge-shaped
# blocks arranged along a half-circle. The 2D wedges become 3D wedges
# (boxes whose top face is angled to fit the arch curve).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_arch
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

ARCH_RADIUS = 4.0
N_BLOCKS = 13
BLOCK_HY = 0.6
BLOCK_HZ = 0.4
BLOCK_HX = (math.pi / N_BLOCKS) * ARCH_RADIUS * 0.5  # tangential half-extent


def _quat_y(angle_rad: float):
    half = 0.5 * angle_rad
    return wp.quat(0.0, math.sin(half), 0.0, math.cos(half))


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 32
    default_friction = 0.7

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        # Arch lies in the X-Z plane with apex along +z. Block i sits at
        # angle theta_i = pi * (i + 0.5) / N from -x to +x.
        for i in range(N_BLOCKS):
            theta = math.pi * (i + 0.5) / N_BLOCKS
            cx = -ARCH_RADIUS * math.cos(theta)
            cz = ARCH_RADIUS * math.sin(theta) + BLOCK_HZ * 0.5
            # Block is rotated so its tangential direction follows the arc.
            rot_y = -(theta - math.pi * 0.5)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(cx, 0.0, cz), q=_quat_y(rot_y)),
            )
            builder.add_shape_box(body, hx=BLOCK_HX, hy=BLOCK_HY, hz=BLOCK_HZ)
            extents.append(default_box_half_extents(BLOCK_HX, BLOCK_HY, BLOCK_HZ))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 3.0), pitch=-10.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
