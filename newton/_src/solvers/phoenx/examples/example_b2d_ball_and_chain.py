# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Ball & Chain``
#
# 30-link chain hanging from a fixed anchor, with a heavy ball at the
# end. 2D revolute joints between adjacent capsules become 3D revolute
# joints about world +y. The end ball is a 3D sphere.
#
# Geometry and mass ratio mirror Box2D's ``samples/sample_joints.cpp``
# ``BallAndChain``: capsule half-length ``hx = 0.5`` with radius
# ``0.125`` (so each link is 1.25 m end-to-end), end ball radius
# ``4.0``, and a density of 20 on every shape. With those shapes the
# 2D ball/link area ratio is ~168, which is what makes this scene a
# classic mass-ratio stress test for iterative solvers.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_ball_and_chain
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

LINK_RADIUS = 0.125
LINK_HALF_LEN = 0.5
LINK_LEN = 2.0 * (LINK_RADIUS + LINK_HALF_LEN)
N_LINKS = 30
BALL_RADIUS = 4.0

# Box2D uses ``density = 20`` for both shapes and lets the engine
# integrate area * density to get mass; we keep the same ratio by
# integrating those 2D areas analytically. Capsule = rectangle of size
# (2 * LINK_HALF_LEN, 2 * LINK_RADIUS) plus a full disc of radius
# LINK_RADIUS (the two end caps glued together). Ball = disc of radius
# BALL_RADIUS. Density cancels in the ratio so the absolute mass units
# are arbitrary -- only the ratio matters for solver stress; we pick
# LINK_MASS = 1 kg and scale the ball by the area ratio (~168x).
_LINK_AREA = (2.0 * LINK_HALF_LEN) * (2.0 * LINK_RADIUS) + math.pi * LINK_RADIUS * LINK_RADIUS
_BALL_AREA = math.pi * BALL_RADIUS * BALL_RADIUS
LINK_MASS = 1.0
BALL_MASS = LINK_MASS * (_BALL_AREA / _LINK_AREA)  # ~168 kg

# Quaternion rotating body-frame +z (the capsule's default axis) to +x.
_QUAT_ZTOX = wp.quat(0.0, 0.7071067811865476, 0.0, 0.7071067811865476)


class Example(PortedExample):
    sim_substeps = 20
    solver_iterations = 4
    gravity = (0.0, 0.0, -9.81)

    def build_scene(self, builder: newton.ModelBuilder):
        # Ground sits well below the fully-extended chain (~30 * 1.25 m
        # of links + 2 * BALL_RADIUS) so the swing never clips it.
        builder.add_ground_plane(height=-50.0)

        anchor_z = 0.0
        joints: list[int] = []
        prev_link = -1
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity())
        extents: list = []

        for i in range(N_LINKS):
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(LINK_LEN * (i + 1), 0.0, anchor_z), q=wp.quat_identity()),
                mass=LINK_MASS,
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
            mass=BALL_MASS,
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
        # Box2D's sample frames the scene at center (0, -8) zoom 27.5 in
        # 2D screen units; in 3D we centre on the chain's mid-span and
        # pull the camera back along -y by ~50 m so the whole chain plus
        # the ball fit on screen.
        chain_mid_x = 0.5 * LINK_LEN * (N_LINKS + 1)
        viewer.set_camera(pos=wp.vec3(chain_mid_x, -55.0, -8.0), pitch=-5.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
