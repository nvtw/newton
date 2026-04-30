# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Bridge``
#
# 30-plank rope bridge. Each plank pinned to its neighbour by a revolute
# joint; both ends pinned to static world anchors (matches Box2D's
# ``sample_joints.cpp::Bridge``). A ball drops onto the bridge.
#
# PGS solver tuning: 20 substeps x 4 iterations -- finer dt beats more
# inner iterations for hinge chains. Per-hinge D-only velocity drive
# acts as joint friction so the bridge settles after impact.
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

HINGE_DAMPING = 100.0


class Example(PortedExample):
    sim_substeps = 20
    solver_iterations = 4

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-5.0)
        joints: list[int] = []
        extents: list = []
        z = 4.0
        x0 = -N_PLANKS * (PLANK_HX + GAP * 0.5)

        # Ball must come before the chain articulation; adding free
        # dynamic bodies after ``add_articulation`` is unsupported.
        ball = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z + 3.0), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(ball, radius=0.5)
        extents_ball = default_sphere_half_extents(0.5)

        # Right-end static anchor: zero-mass ``add_link`` body. PhoenX
        # treats ``inverse_mass == 0`` as MOTION_STATIC, so this acts as
        # a fixed world peg without dropping into the PhoenX API.
        last_plank_cx = x0 + (N_PLANKS - 0.5) * (2 * PLANK_HX + GAP)
        right_anchor_x = last_plank_cx + PLANK_HX
        right_anchor = builder.add_link(
            xform=wp.transform(p=wp.vec3(right_anchor_x, 0.0, z), q=wp.quat_identity()),
        )
        extents_right_anchor = None

        prev = -1
        prev_xform = wp.transform(p=wp.vec3(x0, 0.0, z), q=wp.quat_identity())
        last_link = -1
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
                target_vel=0.0,
                target_kd=HINGE_DAMPING,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            joints.append(j)
            extents.append(default_box_half_extents(PLANK_HX, PLANK_HY, PLANK_HZ))
            prev = link
            last_link = link
            prev_xform = wp.transform(p=wp.vec3(PLANK_HX + GAP, 0.0, 0.0), q=wp.quat_identity())

        builder.add_articulation(joints)

        # Loop-closing hinge: must sit outside any articulation, and
        # ``_validate_joints`` requires its child to be articulated --
        # hence parent=right_anchor (free), child=last_link (articulated).
        builder.add_joint_revolute(
            parent=right_anchor,
            child=last_link,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(PLANK_HX, 0.0, 0.0), q=wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            target_vel=0.0,
            target_kd=HINGE_DAMPING,
            actuator_mode=newton.JointTargetMode.VELOCITY,
        )

        # Body order: 0=ball, 1=right_anchor, 2..N_PLANKS+1=planks.
        return [extents_ball, extents_right_anchor, *extents]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -25.0, 4.0), pitch=-10.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
