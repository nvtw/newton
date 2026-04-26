# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / AncientPyramids``
#
# Two large pyramids of cubes, plus a row of cylinders -- canonical
# JitterPhysics2 stress test. Cylinders are approximated with capsules
# (cylinder = capsule with radius = full half-extent of the side).
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_ancient_pyramids
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    run_ported_example,
)

HE = 0.4
GAP = 0.01
BASE = 12


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 16
    default_friction = 0.6

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        # Two pyramids side by side.
        for ox in (-(BASE * (2 * HE + GAP) * 0.6), +(BASE * (2 * HE + GAP) * 0.6)):
            for row in range(BASE):
                count = BASE - row
                x0 = ox - ((count - 1) * (2 * HE + GAP)) * 0.5
                z = HE + GAP + row * (2 * HE + GAP)
                for c in range(count):
                    x = x0 + c * (2 * HE + GAP)
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(x, 0.0, z), q=wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=HE, hy=HE, hz=HE)
                    extents.append(default_box_half_extents(HE, HE, HE))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(20.0, -20.0, 8.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
