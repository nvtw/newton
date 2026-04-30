# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Door``
#
# A door panel pinned to a static frame via a revolute joint at one edge.
# Drop a ball at the door to push it open. Hinge axis is world +z.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_door
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

DOOR_HX = 0.04
DOOR_HY = 0.5
DOOR_HZ = 1.0
BALL_RADIUS = 0.3


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        # Static door frame markers (visual only -- the hinge is at the
        # joint, not the frame).
        # Door panel (the dynamic body).
        door = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, DOOR_HY, DOOR_HZ + 0.05), q=wp.quat_identity()),
        )
        builder.add_shape_box(door, hx=DOOR_HX, hy=DOOR_HY, hz=DOOR_HZ)

        joint = builder.add_joint_revolute(
            parent=-1,
            child=door,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, DOOR_HZ + 0.05), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, -DOOR_HY, 0.0), q=wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            # Matches Box2D's Door: lowerAngle=-pi/2, upperAngle=pi/2.
            limit_lower=-0.5 * math.pi,
            limit_upper=0.5 * math.pi,
        )
        builder.add_articulation([joint])

        # Ball thrown at the door.
        ball = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 0.5, DOOR_HZ + 0.05), q=wp.quat_identity()),
            linear_velocity=(-8.0, 0.0, 0.0),
        )
        builder.add_shape_sphere(ball, radius=BALL_RADIUS)

        return [
            default_box_half_extents(DOOR_HX, DOOR_HY, DOOR_HZ),
            default_sphere_half_extents(BALL_RADIUS),
        ]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(4.0, 4.0, 2.0), pitch=-15.0, yaw=210.0)


if __name__ == "__main__":
    run_ported_example(Example)
