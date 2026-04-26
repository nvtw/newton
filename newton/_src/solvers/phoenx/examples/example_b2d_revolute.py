# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Revolute``
#
# Two-link pendulum hung from world via a revolute joint and a second
# revolute joint mid-chain. The 2D hinge axis becomes the world +y axis;
# links are thin rods oriented along +x.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_revolute
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

LINK_LEN = 1.0
LINK_HE = 0.05
LINK_HX = LINK_LEN * 0.5


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16
    gravity = (0.0, 0.0, -9.81)

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        # Anchor body 1 m above ground; revolute joints rotate about +y.
        anchor_z = 4.0
        link_a = builder.add_link(
            xform=wp.transform(p=wp.vec3(LINK_HX, 0.0, anchor_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(link_a, hx=LINK_HX, hy=LINK_HE, hz=LINK_HE)

        link_b = builder.add_link(
            xform=wp.transform(p=wp.vec3(LINK_LEN + LINK_HX, 0.0, anchor_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(link_b, hx=LINK_HX, hy=LINK_HE, hz=LINK_HE)

        # World -> link_a hinge at link_a's left end.
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_a,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-LINK_HX, 0.0, 0.0), q=wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
        )
        # link_a -> link_b hinge at link_a's right end.
        j1 = builder.add_joint_revolute(
            parent=link_a,
            child=link_b,
            parent_xform=wp.transform(p=wp.vec3(LINK_HX, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-LINK_HX, 0.0, 0.0), q=wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_articulation([j0, j1])

        return [
            default_box_half_extents(LINK_HX, LINK_HE, LINK_HE),
            default_box_half_extents(LINK_HX, LINK_HE, LINK_HE),
        ]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(5.0, 0.0, 3.0), pitch=-10.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
