# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / MotorAndLimit``
#
# A box pinned to the world via a revolute joint that has both a
# velocity-mode motor and angle limits. The motor drives the box to spin
# at +omega; the limits bracket the rotation so the box swings inside
# the limit window instead of spinning freely.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_motor_and_limit
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

BOX_HX = 0.4
BOX_HY = 0.4
BOX_HZ = 0.05


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16
    gravity = (0.0, 0.0, -9.81)

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-3.0)

        anchor_z = 1.5
        link = builder.add_link(
            xform=wp.transform(p=wp.vec3(BOX_HX, 0.0, anchor_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(link, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ)

        # Velocity-mode actuator with angle limits.
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, anchor_z), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-BOX_HX, 0.0, 0.0), q=wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            target_vel=2.0,
            target_kd=2.0,
            limit_lower=-math.pi / 4.0,
            limit_upper=+math.pi / 4.0,
            actuator_mode=newton.JointTargetMode.VELOCITY,
        )
        builder.add_articulation([joint])
        return [default_box_half_extents(BOX_HX, BOX_HY, BOX_HZ)]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(4.0, 4.0, 1.5), pitch=-5.0, yaw=210.0)


if __name__ == "__main__":
    run_ported_example(Example)
