# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Static-vs-static triangle contact debug scene for PhoenX.
#
# Scene contents:
#   * Ground plane (static, ``shape_body == -1``).
#   * Two :data:`newton.GeoType.TRIANGLE` primitives stacked parallel
#     to each other and separated along world +Z by 0.01 m. Both are
#     static (``body=-1``). Each has ``margin=0.005`` and ``gap=0.03``
#     so the broad / narrow phase pair lookups should fire even though
#     the geometric surfaces don't intersect. Used to inspect whether
#     the collision pipeline emits contacts between two
#     ``shape_body == -1`` shapes.
#
#     ``GeoType.TRIANGLE`` is a first-class primitive: the three
#     vertices are packed into ``shape_scale`` (12 B) plus the rigid
#     ``shape_transform`` (28 B). No ``wp.Mesh``, no BVH, no per-vertex
#     buffers -- the cheapest possible triangle representation, which
#     is exactly what we want for a debug scene.
#   * One ordinary dynamic sphere dropped onto the ground plane. Its
#     sphere-vs-plane contact is the control case that confirms the
#     viewer's contact arrows are being rendered at all.
#
# Starts paused so the initial layout is visible. Press SPACE in the
# ViewerGL window to step.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_static_triangle_contacts
###########################################################################

from __future__ import annotations

import math

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_sphere_half_extents,
    run_ported_example,
)

#: Equilateral debug triangle. Side length ``s``; the three corners
#: lie in a horizontal plane and are arranged around the origin.
TRI_SIDE = 1.5
_HALF = 0.5 * TRI_SIDE
_HEIGHT = TRI_SIDE * math.sqrt(3.0) / 2.0
#: Centroid offset from vertex C in the canonical equilateral layout.
_C_OFFSET_Y = (1.0 / 3.0) * _HEIGHT


def _equilateral_corners(z: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Three corners of a horizontal equilateral triangle at height z."""
    return (
        wp.vec3(-_HALF, -_C_OFFSET_Y, z),
        wp.vec3(_HALF, -_C_OFFSET_Y, z),
        wp.vec3(0.0, _HEIGHT - _C_OFFSET_Y, z),
    )


#: Two parallel triangles, identical XY footprint, separated along world
#: +Z by ``TRI_SEPARATION``. With margin=0.005 per triangle the combined
#: contact corridor is 0.01 m, exactly the geometric gap, so the broad
#: phase should still flag the pair if it considers the margins (and
#: produce a near-zero distance contact).
TRI_Z_A = 1.0
TRI_SEPARATION = 0.01
TRI_Z_B = TRI_Z_A + TRI_SEPARATION

#: Per-shape collision parameters for the two debug triangles. ``margin``
#: extrudes the triangle's contact surface symmetrically along its face
#: normal; ``gap`` is the additional broad-phase / contact-report
#: distance band around the shape.
TRI_MARGIN = 0.005
TRI_GAP = 0.03

#: Falling sphere parameters (control case for contact rendering).
SPHERE_RADIUS = 0.3
SPHERE_START = (3.0, 0.0, 2.0)


class Example(PortedExample):
    """Debug scene for static-vs-static triangle contact emission."""

    fps = 60
    sim_substeps = 4
    solver_iterations = 8
    broad_phase = "sap"
    show_contacts = True
    start_paused = True

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane(height=-5)

        # ---- Two parallel static triangles ---------------------------
        # Both attach to the world (body=-1). The builder auto-adds a
        # collision-filter pair between shapes that share the same body
        # bucket -- including the ``-1`` (no-body) bucket -- so by
        # default these two will be filtered out. We undo that filter
        # below so we can actually observe whether the pipeline would
        # have emitted contacts between them.
        cfg_static = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=TRI_MARGIN,
            gap=TRI_GAP,
        )

        a_a, a_b, a_c = _equilateral_corners(TRI_Z_A)
        tri_a = builder.add_shape_triangle(
            body=-1,
            point_a=a_a,
            point_b=a_b,
            point_c=a_c,
            cfg=cfg_static,
            color=(0.85, 0.25, 0.25),
            label="static_tri_a",
        )

        # Second triangle: same XY footprint, lifted by ``TRI_SEPARATION``
        # along world +Z so the two triangles are parallel and stacked.
        b_a, b_b, b_c = _equilateral_corners(TRI_Z_B)
        tri_b = builder.add_shape_triangle(
            body=-1,
            point_a=b_a,
            point_b=b_b,
            point_c=b_c,
            cfg=cfg_static,
            color=(0.25, 0.55, 0.85),
            label="static_tri_b",
        )

        # Strip any auto-added (tri_a, tri_b) filter pair so the broad
        # phase is free to report the overlap.
        pair = (min(tri_a, tri_b), max(tri_a, tri_b))
        if hasattr(builder, "shape_collision_filter_pairs"):
            builder.shape_collision_filter_pairs = [p for p in builder.shape_collision_filter_pairs if tuple(p) != pair]

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
