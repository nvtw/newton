# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Ball & Chain``
#
# 30-link chain hanging from a fixed anchor, with a heavy ball at the
# end. 2D revolute joints between adjacent capsules become 3D revolute
# joints about world +y. The end ball is a 3D sphere.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_ball_and_chain
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

LINK_RADIUS = 0.05
LINK_HALF_LEN = 0.25
LINK_LEN = 2.0 * (LINK_RADIUS + LINK_HALF_LEN)
N_LINKS = 30
BALL_RADIUS = 0.4

# Quaternion rotating body-frame +z (the capsule's default axis) to +x.
_QUAT_ZTOX = wp.quat(0.0, 0.7071067811865476, 0.0, 0.7071067811865476)


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16
    gravity = (0.0, 0.0, -9.81)

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-10.0)

        anchor_z = 0.0
        joints: list[int] = []
        prev_link = -1
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity())
        extents: list = []

        for i in range(N_LINKS):
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(LINK_LEN * (i + 1), 0.0, anchor_z), q=wp.quat_identity()),
                mass=1.0,
            )
            # Capsule oriented along +x; half_height is the cylindrical
            # mid-section, total length is 2*(radius + half_height).
            builder.add_shape_capsule(
                link,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_QUAT_ZTOX),
                radius=LINK_RADIUS,
                half_height=LINK_HALF_LEN,
            )
            # Joint friction (D-only) so the chain settles instead of
            # swinging indefinitely under the heavy ball.
            j = builder.add_joint_revolute(
                parent=prev_link,
                child=link,
                parent_xform=prev_xform,
                child_xform=wp.transform(p=wp.vec3(-LINK_LEN * 0.5, 0.0, 0.0), q=wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                target_vel=0.0,
                target_kd=0.2,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            joints.append(j)
            extents.append(default_capsule_half_extents(LINK_RADIUS, LINK_HALF_LEN))
            prev_link = link
            prev_xform = wp.transform(p=wp.vec3(LINK_LEN * 0.5, 0.0, 0.0), q=wp.quat_identity())

        ball = builder.add_link(
            xform=wp.transform(p=wp.vec3(LINK_LEN * (N_LINKS + 1), 0.0, anchor_z), q=wp.quat_identity()),
            mass=20.0,
        )
        builder.add_shape_sphere(ball, radius=BALL_RADIUS)
        j = builder.add_joint_revolute(
            parent=prev_link,
            child=ball,
            parent_xform=prev_xform,
            child_xform=wp.transform(p=wp.vec3(-BALL_RADIUS, 0.0, 0.0), q=wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
        )
        joints.append(j)
        extents.append(default_sphere_half_extents(BALL_RADIUS))
        builder.add_articulation(joints)
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(LINK_LEN * (N_LINKS + 1) + 6.0, 0.0, -2.0), pitch=-5.0, yaw=180.0)


if __name__ == "__main__":
    run_ported_example(Example)
