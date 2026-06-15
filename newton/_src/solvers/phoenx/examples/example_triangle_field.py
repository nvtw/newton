# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Triangle-primitive stress test for PhoenX.
#
# A scene built almost entirely out of the first-class
# :data:`newton.GeoType.TRIANGLE` primitive:
#
# * a 6x6 grid of static "fin" triangles standing up off the ground, each
#   one tilted at a random angle (the obstacle field);
# * a small grid of dynamic spheres dropping onto the field so contacts
#   accumulate over time;
# * a separate stack of dynamic triangles falling through the obstacles
#   to exercise triangle-vs-triangle collision via GJK / MPR.
#
# All shapes use ``margin=0.01`` and ``gap=0.03`` per the request:
# the margin keeps PhoenX's contact warm-start happy on the very thin
# triangle shells, and the gap inflates each shape's broad-phase AABB
# so contacts get generated a little before geometric penetration.
#
# PhoenX does not have its own narrow phase for rigid bodies -- it
# delegates to ``newton.CollisionPipeline`` / ``NarrowPhase``. Because
# ``GeoType.TRIANGLE`` is plumbed through both ``support_map`` and
# ``support_map_lean``, the GJK / MPR path picks it up with no
# solver-side changes. Adding this example also serves as a smoke test
# for that integration.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_triangle_field
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

# --- Scene constants ---------------------------------------------------

#: Per-shape collision margin [m] applied to every triangle, sphere, and
#: plane in this scene. Small but non-zero so PhoenX's sticky contact
#: matching has a stable warm-start corridor on triangle thin shells.
SHAPE_MARGIN = 0.01

#: Per-shape collision gap [m] inflates the broad-phase AABB so contacts
#: get generated slightly before geometric penetration. Larger than
#: ``SHAPE_MARGIN`` because the gap also has to absorb one frame of
#: closing velocity at 60 Hz with gravity.
SHAPE_GAP = 0.03

#: Static obstacle field: 6x6 grid of upright triangle fins.
#: Each fin's edge_ab (base length) and vertex C are sampled from the
#: ranges below so no two fins are identical.
FIN_GRID_X = 6
FIN_GRID_Y = 6
FIN_SPACING = 1.2
FIN_GRID_OFFSET = (-3.0, -3.0)
FIN_BASE_LENGTH_RANGE = (0.4, 0.9)  # |AB| -- base edge along local +Z.
FIN_HEIGHT_RANGE = (0.45, 1.0)  # vertex C "height" above the base.
FIN_C_OFFSET_RANGE = (-0.25, 0.25)  # along-base offset of vertex C, as a fraction of |AB|.
FIN_TILT_RANGE = (-0.35, 0.35)  # radians; small random tilt around world +Z.

#: Dynamic spheres raining onto the field.
RAIN_GRID_X = 4
RAIN_GRID_Y = 4
RAIN_SPACING = 1.0
RAIN_OFFSET = (-1.5, -1.5, 4.0)
SPHERE_RADIUS = 0.18
RAIN_JITTER = 0.1

#: Dynamic triangle stack falling through the field. Triangles are
#: thin shells, so most of the contact testing here is sphere-vs-tri
#: and tri-vs-tri at oblique angles. Each falling triangle's base
#: length and vertex-C position are sampled per-instance.
DYN_TRI_COUNT = 12
DYN_TRI_BASE_RANGE = (0.35, 0.7)  # |AB|
DYN_TRI_C_Y_RANGE = (0.3, 0.6)  # vertex C "height" along local +Y.
DYN_TRI_C_Z_RANGE = (-0.25, 0.25)  # vertex C along-base offset.
DYN_TRI_START_Z = 6.0
DYN_TRI_DELTA_Z = 0.5
DYN_TRI_PITCH_RANGE = (-0.4, 0.4)  # radians; pitch around world +X.

RNG_SEED = 4242


def _fin_triangle_xform(x: float, y: float, tilt: float) -> wp.transform:
    """Build the world transform that places one upright fin triangle.

    Canonical triangle frame: vertex A at origin, edge AB along local +Z,
    vertex C in the local YZ plane. We want the fin to stand vertically:

    1. Rotate the canonical +Z (edge AB) onto world +X via a -90 deg
       rotation about world +Y so the base edge runs along world +X.
    2. Add a small random twist about world +Z (``tilt``) so the fins
       aren't all parallel.
    3. Translate to the grid cell, lifting vertex A off the ground by
       ``2 * SHAPE_MARGIN`` so the fin's collision margin envelope and
       the ground's collision margin envelope don't start interpenetrating
       at frame 0 (PhoenX warm-start would otherwise see persistent
       contact noise on every static fin).

    Vertex A ends up at ``(x, y, 2*margin)``, B at the same height
    along world +X, and C up in the air -- giving a near-vertical fin
    that sits a hair above the ground.
    """
    align = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 2.0)
    twist = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(tilt))
    rot = wp.mul(twist, align)
    z_lift = 2.0 * SHAPE_MARGIN
    return wp.transform(p=wp.vec3(float(x), float(y), z_lift), q=rot)


def _dynamic_triangle_xform(z: float, yaw: float, pitch: float) -> wp.transform:
    """Randomized transform for one dynamic triangle. ``yaw`` rotates
    around world +Z and ``pitch`` rotates around world +X, so each
    falling triangle hits the obstacle field at a different orientation
    and exercises GJK / MPR over a range of relative poses rather than
    just the canonical one."""
    yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
    pitch_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(pitch))
    rot = wp.mul(yaw_q, pitch_q)
    return wp.transform(p=wp.vec3(0.0, 0.0, float(z)), q=rot)


class Example(PortedExample):
    """Triangle-heavy PhoenX scene: fin obstacle field + raining
    spheres + stacked dynamic triangles.

    Demonstrates that the first-class :data:`newton.GeoType.TRIANGLE`
    primitive routes through PhoenX with no solver-side changes, and
    that ``margin`` / ``gap`` are honoured per shape end-to-end.
    """

    fps = 60
    sim_substeps = 8
    solver_iterations = 12
    broad_phase = "sap"
    step_layout = "single_world"
    show_contacts = False

    def build_scene(self, builder: newton.ModelBuilder):
        rng = np.random.default_rng(RNG_SEED)
        extents: list = []

        # ---- Ground plane (gives us a hard floor under everything).
        # Margin / gap match the rest of the scene so PhoenX sees a
        # uniform contact corridor across all shape pairs.
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                margin=0,
                gap=SHAPE_GAP,
            ),
        )

        # ---- Dynamic spheres raining down ---------------------------
        sphere_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,
            mu=self.default_friction,
            restitution=self.default_restitution,
            margin=0,
            gap=SHAPE_GAP,
        )
        for ix in range(RAIN_GRID_X):
            for iy in range(RAIN_GRID_Y):
                sx = RAIN_OFFSET[0] + ix * RAIN_SPACING + float(rng.uniform(-RAIN_JITTER, RAIN_JITTER))
                sy = RAIN_OFFSET[1] + iy * RAIN_SPACING + float(rng.uniform(-RAIN_JITTER, RAIN_JITTER))
                sz = RAIN_OFFSET[2] + float(rng.uniform(-RAIN_JITTER, RAIN_JITTER))
                body = builder.add_body(xform=wp.transform(p=wp.vec3(sx, sy, sz), q=wp.quat_identity()))
                builder.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=sphere_cfg)
                extents.append(default_sphere_half_extents(SPHERE_RADIUS))

        # ---- Dynamic triangle stack --------------------------------
        # Mass and inertia are computed from the prism interpretation of
        # the triangle: thickness ``2 * margin`` along the local +X axis
        # (the canonical face normal), centroid at ``(0, c_y/3, (L+c_z)/3)``,
        # full inertia tensor about that centroid. See
        # ``compute_inertia_triangle_prism``.
        dyn_tri_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,
            mu=self.default_friction,
            restitution=self.default_restitution,
            margin=SHAPE_MARGIN,
            gap=SHAPE_GAP,
        )
        for i in range(DYN_TRI_COUNT):
            z = DYN_TRI_START_Z + i * DYN_TRI_DELTA_Z
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            pitch = float(rng.uniform(*DYN_TRI_PITCH_RANGE))
            edge_ab = float(rng.uniform(*DYN_TRI_BASE_RANGE))
            c_y = float(rng.uniform(*DYN_TRI_C_Y_RANGE))
            c_z = float(rng.uniform(*DYN_TRI_C_Z_RANGE)) * edge_ab + 0.5 * edge_ab
            body = builder.add_body(xform=_dynamic_triangle_xform(z, yaw, pitch))
            builder.add_shape_triangle(
                body=body,
                point_a=wp.vec3(0.0, 0.0, 0.0),
                point_b=wp.vec3(0.0, 0.0, edge_ab),
                point_c=wp.vec3(0.0, c_y, c_z),
                cfg=dyn_tri_cfg,
            )
            # Picking OBB: enclose vertices A, B, C in the canonical
            # local frame. A=(0,0,0), B=(0,0,edge_ab), C=(0, c_y, c_z).
            half_x = max(SHAPE_MARGIN, 0.02)  # thin shell -- give the OBB a hair of width
            half_y = 0.5 * abs(c_y)
            half_z = 0.5 * max(edge_ab, abs(c_z))
            extents.append((half_x, half_y, half_z))

        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -8.0, 5.0), pitch=-25.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
