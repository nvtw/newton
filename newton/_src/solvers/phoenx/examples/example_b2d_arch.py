# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Stacking / Arch``
#
# Direct port of ``samples/sample_stacking.cpp:Arch`` from Box2D v3:
# 17 trapezoidal voussoirs (8 right + 8 left + 1 keystone) tracing a
# half-circle. The arch is held up by friction + compression alone;
# replacing the voussoirs with uniform boxes (as the previous port did)
# loses the wedge geometry that locks the stones together.
#
# Each Box2D 2D quad becomes a 3D convex hull by extrusion: the original
# (x, y) polygon lies in Newton's (x, z) plane (z up, matching the
# (0, 0, -9.81) gravity), and we extrude along +/- ``BLOCK_DEPTH / 2`` in
# y to give the bodies a non-zero thickness for collision and rendering.
#
# **Known limitation.** Box2D's arch is the canonical 2D arch demo:
# adjacent voussoirs share *edges* exactly, and Box2D's polygon-vs-
# polygon SAT narrow phase clips contacts cleanly along those shared
# edges. Translating to 3D extrudes those shared edges into shared
# *faces*, which Newton's convex-hull GJK / MPR resolves with
# degenerate closest-feature output (zero-volume overlap, zero-distance
# separation) -- contact normals can flip and the arch sinks softly
# through itself. We side-step the GJK degeneracy with a sub-millimetre
# cumulative vertical offset between adjacent voussoirs, which is
# enough to disambiguate the contact normals (verified by inspecting
# ``rigid_contact_normal`` at frame 0). However, even with correct
# normals the arch is not stable in 3D under Newton's contact pipeline
# at any tested combination of friction (0.6 -> 5.0), substeps
# (16 -> 40), iterations (6 -> 64), block depth (1 m -> 8 m), or
# density (10 -> 1000 kg/m^3). The same scene also collapses under
# ``SolverMuJoCo``, so this isn't PhoenX-specific. We believe the
# root cause is that ``b2MakePolygon`` produces face-distributed
# contact patches in 2D, while Newton's narrow phase produces only
# 4 corner contacts per voussoir-pair in 3D -- enough for parallel-
# face stacking (see ``example_b2d_large_pyramid``) but not enough
# friction-distribution for the slanted-face wedge lock the arch
# relies on.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_arch
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

# Inner / outer radius polygon vertices straight from Box2D's sample
# (samples/sample_stacking.cpp:739-757). Pre-scale so the rest of the
# code matches the C++ literal layout.
_SCALE = 0.25
_PS1 = [
    (16.0, 0.0),
    (14.93803712795643, 5.133601056842984),
    (13.79871746027416, 10.24928069555078),
    (12.56252963284711, 15.34107019122473),
    (11.20040987372525, 20.39856541571217),
    (9.66521217819836, 25.40369899225096),
    (7.87179930638133, 30.3179337000085),
    (5.635199558196225, 35.03820717801641),
    (2.405937953536585, 39.09554102558315),
]
_PS2 = [
    (24.0, 0.0),
    (22.33619528222415, 6.02299846205841),
    (20.54936888969905, 12.00964361211476),
    (18.60854610798073, 17.9470321677465),
    (16.46769273811807, 23.81367936585418),
    (14.05325025774858, 29.57079353071012),
    (11.23551045834022, 35.13775818285372),
    (7.752568160730571, 40.30450679009583),
    (3.016931552701656, 44.28891593799322),
]
PS1 = [(_SCALE * x, _SCALE * z) for (x, z) in _PS1]
PS2 = [(_SCALE * x, _SCALE * z) for (x, z) in _PS2]

#: Extrusion thickness along the y-axis for the 2D->3D lift.
BLOCK_DEPTH = 1.0

#: Box2D's load-box half-extents (sample_stacking.cpp:810).
LOAD_BOX_HX = 2.0
LOAD_BOX_HZ = 0.5


#: Cumulative vertical offset (along +z, the fall direction) added to
#: each voussoir as we walk up the arch. Box2D's original sample ships
#: polygons whose adjacent edges *exactly* coincide -- the 2D
#: polygon-vs-polygon SAT narrow phase handles that case cleanly by
#: clipping contacts along the shared edge. Newton's 3D convex-hull
#: GJK / MPR can't: zero-volume overlap + zero-distance separation
#: make the closest-feature output ambiguous, so contact normals flip
#: and PhoenX (correctly) reacts to a perceived deep penetration along
#: the wrong axis -- the 17-voussoir arch sinks softly through the
#: ground.
#:
#: Lifting voussoir ``i`` by ``i * _VOUSSOIR_GAP`` along +z preserves
#: every face's slope and orientation (the wedge lock that makes the
#: arch self-supporting still works) and only separates adjacent
#: voussoirs along the gravity axis by a visible gap. Bricks fall
#: ``_VOUSSOIR_GAP`` (default 2 cm) to seat against their lower
#: neighbour with cleanly-disambiguated normals.
_VOUSSOIR_GAP: float = 0.02

#: Per-rank taper applied to each voussoir's depth (along y) and radial
#: thickness (R_out - R_in span). Foot voussoirs (rank 0) are full
#: size; each step up toward the keystone shaves another
#: ``_VOUSSOIR_TAPER`` off both dimensions. A 1 % per-rank taper makes
#: the keystone (rank 8 on a 17-voussoir arch) ~92 % the linear size
#: of the feet -- visibly tapered without distorting the wedge angles
#: that drive the friction lock. Pairs with ``_VOUSSOIR_GAP``: the
#: smaller upper bricks no longer share corners with their bigger
#: neighbours, which is the second half of the GJK-degeneracy fix.
_VOUSSOIR_TAPER: float = 0.01


def _prism_mesh_from_quad_xz(
    quad_xz: list[tuple[float, float]],
    depth: float,
) -> newton.Mesh:
    """Build a closed 3D convex prism mesh by extruding a 2D quad.

    Args:
        quad_xz: Four (x, z) points listed CCW in the x-z plane (i.e. the
            same winding Box2D produces for its 2D polygons).
        depth: Total thickness along the y-axis; the prism spans
            ``y ∈ [-depth/2, +depth/2]``.

    Returns:
        A :class:`newton.Mesh` with 8 vertices and 12 triangles. Triangle
        winding is CCW from outside on every face so the mesh is
        watertight and ``compute_inertia=True`` produces a positive
        volume.
    """
    hd = 0.5 * float(depth)
    # vertices[0..3]: front face at y=-hd, original CCW order in (x, z).
    # vertices[4..7]: back face at y=+hd, same (x, z).
    front = [(x, -hd, z) for (x, z) in quad_xz]
    back = [(x, +hd, z) for (x, z) in quad_xz]
    vertices = np.asarray(front + back, dtype=np.float32)

    # Triangle indices wound CCW from outside on all six faces.
    #   Front  (normal -y): (0,1,2)(0,2,3) — original 2D order at y=-hd.
    #   Back   (normal +y): (4,6,5)(4,7,6) — reversed for the +y side.
    #   Sides  (4 quads):   (i, i+4, j+4)(i, j+4, j) with j = (i+1) % 4.
    # fmt: off
    triangles = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    # fmt: on
    indices = np.asarray(triangles, dtype=np.int32).flatten()
    return newton.Mesh(vertices=vertices, indices=indices)


def _quad_pick_extents(quad_xz: list[tuple[float, float]], depth: float) -> tuple[float, float, float]:
    """AABB half-extents of a prism for picking-OBB sizing."""
    xs = [x for (x, _) in quad_xz]
    zs = [z for (_, z) in quad_xz]
    return (
        0.5 * (max(xs) - min(xs)),
        0.5 * float(depth),
        0.5 * (max(zs) - min(zs)),
    )


def _scale_quad_about_centroid(quad_xz: list[tuple[float, float]], factor: float) -> list[tuple[float, float]]:
    """Shrink ``quad_xz`` toward its (x, z) centroid by ``factor``. Used to
    apply the per-rank radial taper without reshaping the wedge angles."""
    cx = sum(p[0] for p in quad_xz) / 4.0
    cz = sum(p[1] for p in quad_xz) / 4.0
    return [(cx + (p[0] - cx) * factor, cz + (p[1] - cz) * factor) for p in quad_xz]


class Example(PortedExample):
    sim_substeps = 20
    solver_iterations = 20
    default_friction = 0.6  # b2ShapeDef.material.friction in the sample
    start_paused = True
    pause_after_step = True  # SPACE advances one frame at a time

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        # ``rank`` = position along the arch from foot toward keystone.
        # Rank 0 = foot (bottom), rank 8 = keystone (top). Both the
        # vertical offset and the per-rank size taper accumulate with
        # rank so the upper bricks are clearly above and clearly
        # smaller than the lower ones.
        def _scale_for_rank(rank: int) -> float:
            return (1.0 - _VOUSSOIR_TAPER) ** rank

        # Right-side voussoirs: quad (ps1[i], ps2[i], ps2[i+1], ps1[i+1])
        # — inner-low, outer-low, outer-high, inner-high (CCW in the
        # x-z plane). Each voussoir is lifted by ``rank *
        # _VOUSSOIR_GAP`` along +z and scaled by ``_scale_for_rank``
        # so adjacent pieces don't share faces (Newton's convex-hull
        # narrow phase resolves shared faces with degenerate normals
        # -- see the constants' docstrings).
        for i in range(8):
            rank = i
            scale = _scale_for_rank(rank)
            quad = _scale_quad_about_centroid([PS1[i], PS2[i], PS2[i + 1], PS1[i + 1]], scale)
            mesh = _prism_mesh_from_quad_xz(quad, BLOCK_DEPTH * scale)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, rank * _VOUSSOIR_GAP), q=wp.quat_identity()),
            )
            builder.add_shape_convex_hull(body=body, mesh=mesh)
            extents.append(_quad_pick_extents(quad, BLOCK_DEPTH * scale))

        # Left-side voussoirs: mirror x. Box2D's order
        # (-ps2[i], -ps1[i], -ps1[i+1], -ps2[i+1]) is CCW after the
        # mirror flip; verified by signed-area check.
        for i in range(8):
            rank = i
            scale = _scale_for_rank(rank)
            quad = _scale_quad_about_centroid(
                [
                    (-PS2[i][0], PS2[i][1]),
                    (-PS1[i][0], PS1[i][1]),
                    (-PS1[i + 1][0], PS1[i + 1][1]),
                    (-PS2[i + 1][0], PS2[i + 1][1]),
                ],
                scale,
            )
            mesh = _prism_mesh_from_quad_xz(quad, BLOCK_DEPTH * scale)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, rank * _VOUSSOIR_GAP), q=wp.quat_identity()),
            )
            builder.add_shape_convex_hull(body=body, mesh=mesh)
            extents.append(_quad_pick_extents(quad, BLOCK_DEPTH * scale))

        # Keystone: bridges the right and left top voussoirs. Bottom edge
        # rests on the inner-radius corners (ps1[8]); top edge runs along
        # the outer-radius arc (ps2[8]).
        rank = 8
        scale = _scale_for_rank(rank)
        keystone_quad = _scale_quad_about_centroid(
            [
                PS1[8],
                PS2[8],
                (-PS2[8][0], PS2[8][1]),
                (-PS1[8][0], PS1[8][1]),
            ],
            scale,
        )
        mesh = _prism_mesh_from_quad_xz(keystone_quad, BLOCK_DEPTH * scale)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, rank * _VOUSSOIR_GAP), q=wp.quat_identity()),
        )
        builder.add_shape_convex_hull(body=body, mesh=mesh)
        extents.append(_quad_pick_extents(keystone_quad, BLOCK_DEPTH * scale))

        # 4 load boxes stacked on top of the keystone (Box2D
        # sample_stacking.cpp:808-814).
        keystone_top_z = PS2[8][1]
        num_load_boxes = 0
        for i in range(num_load_boxes):
            cz = LOAD_BOX_HZ + keystone_top_z + 1.0 * i
            body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, cz)))
            builder.add_shape_box(body, hx=LOAD_BOX_HX, hy=0.5 * BLOCK_DEPTH, hz=LOAD_BOX_HZ)
            extents.append(default_box_half_extents(LOAD_BOX_HX, 0.5 * BLOCK_DEPTH, LOAD_BOX_HZ))

        return extents

    def configure_camera(self, viewer):
        # Arch outer feet at x=+/-6, keystone top at z~11, load-box stack
        # peaks near z=15. Frame the whole pile from in front (-y),
        # ``yaw=90`` looks back along +y.
        viewer.set_camera(pos=wp.vec3(0.0, -22.0, 7.0), pitch=-10.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
