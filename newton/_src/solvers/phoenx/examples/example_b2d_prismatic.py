# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Prismatic``
#
# A box constrained to a slider (prismatic) joint along world +z, with
# limits and a position-PD drive that pushes the box toward the upper
# limit. Box2D's 2D vertical slider becomes our +z slider.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_prismatic
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

BOX_HE = 0.5
LIMIT_LO = -2.0
LIMIT_HI = 2.0


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16
    gravity = (0.0, 0.0, -9.81)

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        anchor_z = 4.0
        link = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(link, hx=BOX_HE, hy=BOX_HE, hz=BOX_HE)

        joint = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
            limit_lower=LIMIT_LO,
            limit_upper=LIMIT_HI,
            target_pos=LIMIT_HI,
            target_ke=200.0,
            target_kd=10.0,
        )
        builder.add_articulation([joint])
        return [default_box_half_extents(BOX_HE, BOX_HE, BOX_HE)]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(6.0, 0.0, 4.0), pitch=-5.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
