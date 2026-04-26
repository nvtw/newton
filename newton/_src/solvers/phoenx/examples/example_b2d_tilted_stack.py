# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Tilted Stack``
#
# A grid of cubes spawned with random initial rotations -- forces the
# solver to settle a leaning, partially-overlapping pile rather than
# a clean axis-aligned stack.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_tilted_stack
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.4
ROWS = 6
COLS = 6


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []
        rng = np.random.default_rng(seed=0)
        for r in range(ROWS):
            for c in range(COLS):
                x = (c - COLS * 0.5) * (2.5 * HE)
                z = HE + 0.05 + r * (2.2 * HE)
                # Random small rotation about a random axis.
                axis = rng.normal(size=3).astype(np.float32)
                axis /= max(np.linalg.norm(axis), 1e-6)
                angle = float(rng.uniform(-0.4, 0.4))
                half = 0.5 * angle
                s = math.sin(half)
                q = wp.quat(float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s, math.cos(half))
                body = builder.add_body(xform=wp.transform(p=wp.vec3(x, 0.0, z), q=q))
                builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 4.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
