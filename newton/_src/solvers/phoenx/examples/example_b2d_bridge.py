# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Bridge``
#
# 30-plank rope bridge: each plank pinned to its neighbour by a revolute
# joint; the two end planks pin to static anchors. Drop a ball on the
# bridge to see it bend.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_bridge
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

PLANK_HX = 0.4
PLANK_HY = 0.6
PLANK_HZ = 0.05
N_PLANKS = 30
GAP = 0.02


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-5.0)
        joints: list[int] = []
        extents: list = []
        z = 4.0
        x0 = -N_PLANKS * (PLANK_HX + GAP * 0.5)

        # Ball first -- the chain articulation is finalised by
        # ``add_articulation`` and adding a free dynamic body afterward
        # confuses the implicit articulation bookkeeping.
        ball = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z + 3.0), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(ball, radius=0.5)
        extents_ball = default_sphere_half_extents(0.5)

        prev = -1
        prev_xform = wp.transform(p=wp.vec3(x0, 0.0, z), q=wp.quat_identity())
        for i in range(N_PLANKS):
            cx = x0 + (i + 0.5) * (2 * PLANK_HX + GAP)
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(cx, 0.0, z), q=wp.quat_identity()),
            )
            builder.add_shape_box(link, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
            j = builder.add_joint_revolute(
                parent=prev,
                child=link,
                parent_xform=prev_xform,
                child_xform=wp.transform(p=wp.vec3(-PLANK_HX, 0.0, 0.0), q=wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
            )
            joints.append(j)
            extents.append(default_box_half_extents(PLANK_HX, PLANK_HY, PLANK_HZ))
            prev = link
            prev_xform = wp.transform(p=wp.vec3(PLANK_HX + GAP, 0.0, 0.0), q=wp.quat_identity())

        builder.add_articulation(joints)
        # ``extents`` has the chain links in body-id order; the ball was
        # body 0, so prepend its OBB.
        return [extents_ball, *extents]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -25.0, 4.0), pitch=-10.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
