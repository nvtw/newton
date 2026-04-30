# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / TowerOfJitter``
#
# Single tall tower: 30 layers of 4 planks each, alternating orientation
# 90 degrees per layer (Jenga-style). Stress test for tall stack
# stability with offset contacts.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_tower_of_jitter
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

PLANK_HX = 0.6
PLANK_HY = 0.2
PLANK_HZ = 0.1
N_LAYERS = 30
PLANKS_PER_LAYER = 3


def _quat_z(angle_rad: float):
    half = 0.5 * angle_rad
    return wp.quat(0.0, 0.0, math.sin(half), math.cos(half))


class Example(PortedExample):
    sim_substeps = 20
    solver_iterations = 16
    default_friction = 0.6

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        layer_h = 2 * PLANK_HZ + 0.005
        for layer in range(N_LAYERS):
            theta = (math.pi * 0.5) if (layer % 2) else 0.0
            q = _quat_z(theta)
            z = PLANK_HZ + 0.02 + layer * layer_h
            # Lay 3 planks side by side along the layer's tangential axis.
            for j in range(PLANKS_PER_LAYER):
                offset = (j - (PLANKS_PER_LAYER - 1) * 0.5) * (2 * PLANK_HY + 0.01)
                if layer % 2 == 0:
                    p = wp.vec3(0.0, offset, z)
                else:
                    p = wp.vec3(offset, 0.0, z)
                body = builder.add_body(xform=wp.transform(p=p, q=q))
                builder.add_shape_box(body, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
                extents.append(default_box_half_extents(PLANK_HX, PLANK_HY, PLANK_HZ))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 4.0), pitch=-10.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
