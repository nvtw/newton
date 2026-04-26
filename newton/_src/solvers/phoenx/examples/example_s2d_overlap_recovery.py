# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# solver2d port: ``Contact / Overlap Recovery``
#
# 10 cubes spawned with 25% mutual overlap in a triangular arrangement.
# Tests the solver's penetration-recovery (Baumgarte / soft-projection)
# when the initial state is geometrically infeasible.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_s2d_overlap_recovery
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
OVERLAP = 0.25  # fraction of half-extent
BASE = 4


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 24

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        # Spacing collapses by (1 - overlap) so cubes interpenetrate.
        fraction = 1.0 - OVERLAP
        z = HE
        for i in range(BASE):
            x0 = fraction * HE * (i - BASE)
            for j in range(i, BASE):
                x = x0 + 2.0 * fraction * HE * (j - i)
                body = builder.add_body(
                    xform=wp.transform(p=wp.vec3(x, 0.0, z + HE), q=wp.quat_identity()),
                )
                builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                extents.append(default_box_half_extents(HE, HE, HE))
            z += 2.0 * fraction * HE
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -6.0, 3.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
