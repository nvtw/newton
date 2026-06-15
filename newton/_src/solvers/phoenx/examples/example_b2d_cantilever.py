# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Cantilever``
#
# 8-plank cantilever beam: every joint is a weld (fixed) at the
# connecting edge, matching Box2D's ``sample_joints.cpp::Cantilever``
# (``b2WeldJoint`` for all 8 joints, including the world anchor). The
# beam sags slightly under gravity.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_cantilever
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

PLANK_HX = 0.5
PLANK_HY = 0.3
PLANK_HZ = 0.06
N_PLANKS = 8


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 24

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-5.0)
        joints: list[int] = []
        extents: list = []
        z = 4.0

        # Plank 0 welded to the world.
        first = builder.add_link(
            xform=wp.transform(p=wp.vec3(PLANK_HX, 0.0, z), q=wp.quat_identity()),
        )
        builder.add_shape_box(first, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
        joints.append(
            builder.add_joint_fixed(
                parent=-1,
                child=first,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(-PLANK_HX, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        extents.append(default_box_half_extents(PLANK_HX, PLANK_HY, PLANK_HZ))

        prev = first
        for i in range(1, N_PLANKS):
            cx = (2 * i + 1) * PLANK_HX
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(cx, 0.0, z), q=wp.quat_identity()),
            )
            builder.add_shape_box(link, hx=PLANK_HX, hy=PLANK_HY, hz=PLANK_HZ)
            # Weld (fixed) joint -- matches Box2D's b2WeldJoint. A revolute
            # would let the beam articulate into a rope, not a beam.
            joints.append(
                builder.add_joint_fixed(
                    parent=prev,
                    child=link,
                    parent_xform=wp.transform(p=wp.vec3(PLANK_HX, 0.0, 0.0), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(-PLANK_HX, 0.0, 0.0), q=wp.quat_identity()),
                )
            )
            extents.append(default_box_half_extents(PLANK_HX, PLANK_HY, PLANK_HZ))
            prev = link

        builder.add_articulation(joints)
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(8.0, -8.0, 3.0), pitch=-10.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
