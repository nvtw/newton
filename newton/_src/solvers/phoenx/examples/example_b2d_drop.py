# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Forces / Drop``
#
# A handful of small objects (cube, sphere, capsule) dropped from various
# heights. The 2D demo's polygons collapse to 3D primitives.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_drop
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    default_capsule_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

_S = 0.7071067811865476
_QUAT_ZTOX = wp.quat(0.0, _S, 0.0, _S)


class Example(PortedExample):
    sim_substeps = 8
    solver_iterations = 16
    default_restitution = 0.3

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        b = builder.add_body(xform=wp.transform(p=wp.vec3(-1.0, 0.0, 4.0), q=wp.quat_identity()))
        builder.add_shape_box(b, hx=0.4, hy=0.4, hz=0.4)
        extents.append(default_box_half_extents(0.4, 0.4, 0.4))

        b = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()))
        builder.add_shape_sphere(b, radius=0.4)
        extents.append(default_sphere_half_extents(0.4))

        b = builder.add_body(xform=wp.transform(p=wp.vec3(1.0, 0.0, 6.0), q=wp.quat_identity()))
        builder.add_shape_capsule(
            b,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_QUAT_ZTOX),
            radius=0.2,
            half_height=0.4,
        )
        extents.append(default_capsule_half_extents(0.2, 0.4))

        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(5.0, -5.0, 3.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
