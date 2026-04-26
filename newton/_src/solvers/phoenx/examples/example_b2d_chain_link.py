# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Chain Link``
#
# A horizontal chain of cubes pinned to a fixed left anchor by a long
# series of revolute joints. Lifting the right end (initial offset
# upward) tightens the chain under gravity into a curve.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_chain_link
###########################################################################

from __future__ import annotations

import warp as wp

import newton

from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

HE = 0.2
GAP = 0.02
N_LINKS = 24


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-5.0)
        joints: list[int] = []
        extents: list = []

        z = 4.0
        prev = -1
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity())
        for i in range(N_LINKS):
            cx = (i + 0.5) * (2 * HE + GAP)
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(cx, 0.0, z), q=wp.quat_identity()),
            )
            builder.add_shape_box(link, hx=HE, hy=HE, hz=HE)
            joints.append(
                builder.add_joint_revolute(
                    parent=prev,
                    child=link,
                    parent_xform=prev_xform,
                    child_xform=wp.transform(p=wp.vec3(-HE, 0.0, 0.0), q=wp.quat_identity()),
                    axis=(0.0, 1.0, 0.0),
                )
            )
            extents.append(default_box_half_extents(HE, HE, HE))
            prev = link
            prev_xform = wp.transform(p=wp.vec3(HE + GAP, 0.0, 0.0), q=wp.quat_identity())

        builder.add_articulation(joints)
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(N_LINKS * HE, -8.0, 1.0), pitch=-5.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
