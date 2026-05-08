# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Static-vs-static triangle contact debug scene for PhoenX.
#
# Scene contents:
#   * Ground plane (static, ``shape_body == -1``).
#   * Two single-triangle meshes intentionally placed so their geometry
#     interpenetrates. Both are static (``body=-1``). Used to inspect
#     whether the collision pipeline emits contacts between two
#     ``shape_body == -1`` shapes.
#   * One ordinary dynamic sphere dropped onto the ground plane. Its
#     sphere-vs-plane contact is the control case that confirms the
#     viewer's contact arrows are being rendered at all.
#
# Starts paused so the initial overlap is visible. Press SPACE in the
# ViewerGL window to step.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_static_triangle_contacts
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

#: Half edge length of each debug triangle. The two triangles share the
#: world origin in XY and are offset along Z by less than their height
#: so their volumes interpenetrate.
TRI_HALF_EDGE = 1.0
TRI_Z_A = 1.0
TRI_Z_B = 1.3

#: Falling sphere parameters (control case for contact rendering).
SPHERE_RADIUS = 0.3
SPHERE_START = (3.0, 0.0, 2.0)


def _equilateral_triangle_mesh(half_edge: float) -> newton.Mesh:
    """Single CCW-wound triangle in the XY plane, centred at the origin.

    The triangle is a flat sheet (one face). It's added as a regular
    ``GeoType.MESH``; PhoenX's narrow phase treats it as a thin
    triangle-soup collider.
    """
    h = float(half_edge)
    vertices = np.array(
        [
            [-h, -h * 0.577, 0.0],
            [h, -h * 0.577, 0.0],
            [0.0, h * 1.155, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.int32)
    return newton.Mesh(vertices, indices)


class Example(PortedExample):
    """Debug scene for static-vs-static triangle contact emission."""

    fps = 60
    sim_substeps = 4
    solver_iterations = 8
    broad_phase = "sap"
    show_contacts = True
    start_paused = True

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        # ---- Two overlapping static triangles ------------------------
        # Both attach to the world (body=-1). The builder auto-adds a
        # collision-filter pair between shapes that share the same body
        # bucket -- including the ``-1`` (no-body) bucket -- so by
        # default these two will be filtered out. We undo that filter
        # below so we can actually observe whether the pipeline would
        # have emitted contacts between them.
        tri_mesh = _equilateral_triangle_mesh(TRI_HALF_EDGE)
        cfg_static = newton.ModelBuilder.ShapeConfig(density=0.0)

        tri_a = builder.add_shape_mesh(
            body=-1,
            mesh=tri_mesh,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, TRI_Z_A), q=wp.quat_identity()),
            cfg=cfg_static,
            color=(0.85, 0.25, 0.25),
            label="static_tri_a",
        )
        tri_b = builder.add_shape_mesh(
            body=-1,
            mesh=tri_mesh,
            xform=wp.transform(
                p=wp.vec3(0.2, 0.0, TRI_Z_B),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4),
            ),
            cfg=cfg_static,
            color=(0.25, 0.55, 0.85),
            label="static_tri_b",
        )

        # Strip any auto-added (tri_a, tri_b) filter pair so the broad
        # phase is free to report the overlap.
        pair = (min(tri_a, tri_b), max(tri_a, tri_b))
        if hasattr(builder, "shape_collision_filter_pairs"):
            builder.shape_collision_filter_pairs = [
                p for p in builder.shape_collision_filter_pairs if tuple(p) != pair
            ]

        # ---- Falling sphere (sanity check for contact arrows) --------
        sphere_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(*SPHERE_START), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(sphere_body, radius=SPHERE_RADIUS)

        # ``PortedExample`` expects one entry per body. Static shapes
        # don't add bodies, so we only report the sphere here.
        return [default_sphere_half_extents(SPHERE_RADIUS)]

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(5.0, -4.0, 2.5), pitch=-15.0, yaw=140.0)


if __name__ == "__main__":
    run_ported_example(Example)
