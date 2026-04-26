# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# JitterPhysics2 port: ``Demo / DoublePendulum``
#
# Two arms each made of 8 capsules connected by revolute joints, hung
# from a fixed pivot. The whole assembly is allowed to swing freely
# under gravity.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_jitter_double_pendulum
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    run_ported_example,
)

LINK_R = 0.05
LINK_H = 0.25
LINK_LEN = 2.0 * (LINK_R + LINK_H)
N_PER_ARM = 8

# Quats rotating body-frame +z to +x and +y respectively (the capsule's
# default axis is +z; we want the per-arm capsules aligned along the arm
# direction).
_S = 0.7071067811865476
_QUAT_ZTOX = wp.quat(0.0, _S, 0.0, _S)
_QUAT_ZTOY = wp.quat(_S, 0.0, 0.0, _S)


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-15.0)
        joints: list[int] = []
        extents: list = []

        # Build one arm hanging along -x from world anchor at origin.
        # Arms span (0, 0, 0) -> (-N*LINK_LEN, 0, 0).
        prev = -1
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
        for i in range(N_PER_ARM):
            x = -(i + 0.5) * LINK_LEN
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(x, 0.0, 0.0), q=wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_capsule(
                link,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_QUAT_ZTOX),
                radius=LINK_R,
                half_height=LINK_H,
            )
            j = builder.add_joint_revolute(
                parent=prev,
                child=link,
                parent_xform=prev_xform,
                child_xform=wp.transform(p=wp.vec3(LINK_LEN * 0.5, 0.0, 0.0), q=wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
            )
            joints.append(j)
            extents.append(default_capsule_half_extents(LINK_R, LINK_H))
            prev = link
            prev_xform = wp.transform(p=wp.vec3(-LINK_LEN * 0.5, 0.0, 0.0), q=wp.quat_identity())

        # Second arm: branch off from the first link's far end so the
        # double pendulum has two distinct arms swinging from the
        # midpoint.
        # Drive the second arm along -y instead so the two arms aren't
        # collinear and we get the chaotic double-pendulum motion.
        prev = -1
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
        for i in range(N_PER_ARM):
            y = -(i + 0.5) * LINK_LEN
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(0.0, y, 0.0), q=wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_capsule(
                link,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_QUAT_ZTOY),
                radius=LINK_R,
                half_height=LINK_H,
            )
            j = builder.add_joint_revolute(
                parent=prev,
                child=link,
                parent_xform=prev_xform,
                child_xform=wp.transform(p=wp.vec3(0.0, LINK_LEN * 0.5, 0.0), q=wp.quat_identity()),
                axis=(1.0, 0.0, 0.0),
            )
            joints.append(j)
            extents.append(default_capsule_half_extents(LINK_R, LINK_H))
            prev = link
            prev_xform = wp.transform(p=wp.vec3(0.0, -LINK_LEN * 0.5, 0.0), q=wp.quat_identity())

        builder.add_articulation(joints)
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(6.0, 6.0, 0.0), pitch=-5.0, yaw=225.0)


if __name__ == "__main__":
    run_ported_example(Example)
