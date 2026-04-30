# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Shapes / Compound Shapes``
#
# Bodies with multiple collision shapes attached (compound rigid bodies).
# Each body here has a box + sphere + capsule rigidly welded together
# at different local offsets.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_compound_shapes
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    run_ported_example,
)

_S = 0.7071067811865476
_QUAT_ZTOX = wp.quat(0.0, _S, 0.0, _S)


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        # Three identical compound bodies dropped at different heights.
        for _i, (x, z) in enumerate([(-2.0, 4.0), (0.0, 6.0), (2.0, 8.0)]):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(x, 0.0, z), q=wp.quat_identity()))
            # Compound: a horizontal capsule + a sphere on top + a small box on each side.
            builder.add_shape_capsule(
                body,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_QUAT_ZTOX),
                radius=0.2,
                half_height=0.5,
            )
            builder.add_shape_sphere(
                body,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.4), q=wp.quat_identity()),
                radius=0.3,
            )
            builder.add_shape_box(
                body,
                xform=wp.transform(p=wp.vec3(-0.6, 0.0, 0.0), q=wp.quat_identity()),
                hx=0.15,
                hy=0.15,
                hz=0.15,
            )
            builder.add_shape_box(
                body,
                xform=wp.transform(p=wp.vec3(0.6, 0.0, 0.0), q=wp.quat_identity()),
                hx=0.15,
                hy=0.15,
                hz=0.15,
            )
            # Pick OBB chosen to enclose the longest axis (capsule).
            extents.append((0.7, 0.3, 0.7))
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(6.0, -6.0, 4.0), pitch=-15.0, yaw=135.0)


if __name__ == "__main__":
    run_ported_example(Example)
