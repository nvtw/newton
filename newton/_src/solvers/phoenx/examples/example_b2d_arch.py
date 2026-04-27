# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Half-circle masonry arch (extruded 3D voussoirs)
#
# Self-supporting masonry arch built from N symmetric trapezoidal
# voussoirs traced along a half-circle guide curve, then extruded
# along the y-axis to give the bodies a non-zero thickness for
# collision and rendering.
#
# Replaces an earlier port of Box2D's ``samples/sample_stacking.cpp:
# Arch``. That port translated Box2D's polygon-vs-polygon SAT-clipped
# shared edges into Newton's 3D convex-hull GJK / MPR, which can't
# disambiguate shared faces (zero-volume overlap, zero-distance
# separation): contact normals flip and the arch sinks softly through
# itself. Even after re-sampling neighbour faces with a small
# cumulative offset to fix the normals, the original 8-voussoir / 22.5
# degree wedge geometry is too sensitive to corner-only contact
# distributions to settle in 3D (verified across friction, substeps,
# iterations, depth, and density sweeps; ``SolverMuJoCo`` collapses
# the same scene). The fix is a parametric design:
#
#   * 5 voussoirs spanning the half-circle -- 36 deg wedge per
#     voussoir, comfortably above the corner-contact friction-cone
#     threshold. Arches with more (narrower) voussoirs are stable in
#     2D / Box2D but slip out of equilibrium in 3D under Newton's
#     contact pipeline.
#   * Sub-degree angular gap between adjacent voussoirs so neither
#     GJK nor MPR sees coincident faces. Gaps close within ~1 frame
#     of simulation as the wedge geometry pre-loads each voussoir
#     against its neighbours.
#   * Foot voussoirs keep their grounded edges flush (no inset on the
#     ground side) so the arch sits flat on the ground plane.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_arch
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_box_half_extents,
    run_ported_example,
)

#: Number of voussoirs spanning the half-circle. 5 gives a 36-degree
#: wedge angle per voussoir -- the empirical sweet spot under
#: Newton's 3D corner-contact distribution at ``mu = 0.6`` (tested 5,
#: 7, 9, 12, 17; only 5 and 7 settle, 5 has the largest stability
#: margin).
N_VOUSSOIRS: int = 5

#: Inner / outer radius of the arch's half-circle guide curve [m].
R_INNER: float = 4.0
R_OUTER: float = 6.0

#: Angular gap [rad] between adjacent voussoirs. ~0.3 deg at the
#: typical arch radius corresponds to ~3 cm of perpendicular gap,
#: large enough to disambiguate GJK / MPR closest-feature output and
#: small enough that voussoirs seat against each other within one
#: simulation frame as the wedge geometry pre-loads.
GAP_RAD: float = 0.005

#: Extrusion thickness along the y-axis for the 2D->3D lift.
BLOCK_DEPTH: float = 1.0


def _voussoir_quads(
    n: int = N_VOUSSOIRS,
    r_in: float = R_INNER,
    r_out: float = R_OUTER,
    gap: float = GAP_RAD,
) -> list[list[tuple[float, float]]]:
    """Build N symmetric trapezoidal voussoirs along a half-circle.

    Each voussoir spans angular range ``[theta_lo, theta_hi]`` with a
    half-gap inset on both shared edges, except the foot voussoirs
    (``i == 0`` and ``i == n - 1``) which keep their grounded edges
    flush with theta=0 and theta=pi so they rest flat on the ground
    plane.

    Returns a list of N quads, each four ``(x, z)`` tuples in CCW
    order: inner-low, outer-low, outer-high, inner-high.
    """
    voussoirs: list[list[tuple[float, float]]] = []
    dtheta = math.pi / n
    for i in range(n):
        lo_inset = gap / 2.0 if i > 0 else 0.0
        hi_inset = gap / 2.0 if i < n - 1 else 0.0
        theta_lo = i * dtheta + lo_inset
        theta_hi = (i + 1) * dtheta - hi_inset
        c_lo, s_lo = math.cos(theta_lo), math.sin(theta_lo)
        c_hi, s_hi = math.cos(theta_hi), math.sin(theta_hi)
        voussoirs.append(
            [
                (r_in * c_lo, r_in * s_lo),
                (r_out * c_lo, r_out * s_lo),
                (r_out * c_hi, r_out * s_hi),
                (r_in * c_hi, r_in * s_hi),
            ]
        )
    return voussoirs


def _prism_mesh_from_quad_xz(
    quad_xz: list[tuple[float, float]],
    depth: float,
) -> newton.Mesh:
    """Build a closed 3D convex prism mesh by extruding a 2D quad.

    Args:
        quad_xz: Four (x, z) points listed CCW in the x-z plane.
        depth: Total thickness along the y-axis; the prism spans
            ``y ∈ [-depth/2, +depth/2]``.

    Returns:
        A :class:`newton.Mesh` with 8 vertices and 12 triangles. Triangle
        winding is CCW from outside on every face so the mesh is
        watertight and ``compute_inertia=True`` produces a positive
        volume.
    """
    hd = 0.5 * float(depth)
    front = [(x, -hd, z) for (x, z) in quad_xz]
    back = [(x, +hd, z) for (x, z) in quad_xz]
    vertices = np.asarray(front + back, dtype=np.float32)
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


class Example(PortedExample):
    sim_substeps = 16
    solver_iterations = 16
    default_friction = 0.6
    start_paused = True

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()
        extents: list = []

        for quad in _voussoir_quads():
            mesh = _prism_mesh_from_quad_xz(quad, BLOCK_DEPTH)
            body = builder.add_body()
            builder.add_shape_convex_hull(body=body, mesh=mesh)
            extents.append(_quad_pick_extents(quad, BLOCK_DEPTH))

        # Optional load boxes on top of the keystone, demonstrating the
        # arch supports vertical compression. The stack height is
        # bounded by the friction-cone margin at the arch crown; 4
        # boxes at this size is comfortable.
        keystone_top_z = R_OUTER * math.sin(math.pi / 2)  # = R_OUTER
        load_hx = 1.5
        load_hz = 0.3
        for i in range(4):
            cz = keystone_top_z + 0.6 + load_hz + 2 * load_hz * i
            body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, cz), q=wp.quat_identity()))
            builder.add_shape_box(body, hx=load_hx, hy=0.5 * BLOCK_DEPTH, hz=load_hz)
            extents.append(default_box_half_extents(load_hx, 0.5 * BLOCK_DEPTH, load_hz))

        return extents

    def configure_camera(self, viewer):
        # Arch outer feet at x=+/-R_OUTER, keystone top at z=R_OUTER.
        # Frame the whole structure from in front (-y) at yaw=90.
        viewer.set_camera(pos=wp.vec3(0.0, -16.0, 4.0), pitch=-5.0, yaw=90.0)


if __name__ == "__main__":
    run_ported_example(Example)
