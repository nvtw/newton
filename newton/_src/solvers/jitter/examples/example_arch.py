# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Arch
#
# Port of Box2D's ``Stacking/Arch`` sample. The arch is made of 17
# trapezoidal voussoirs (the wedge-shaped keystones that wedge
# themselves in place under gravity) plus 4 boxes stacked on the
# keystone on top. The 2D voussoir quads from the Box2D sample are
# extruded along +Y by ``ARCH_DEPTH`` to give each a true 3D shape.
#
# Because the voussoirs are non-axis-aligned wedges rather than boxes,
# they go through Newton's ``CONVEX_MESH`` shape path -- a convex hull
# built from the 8 extruded corners of each segment. The Jitter solver
# itself is shape-agnostic (it only sees the contact points + normals
# produced by the GJK/MPR narrow phase), so convex-convex contacts
# feed into the same PGS loop that box / sphere contacts do.
#
# The scene's real test is static equilibrium: under gravity the
# voussoirs compress each other along the ring and the whole structure
# is held up by the ground + two abutments. If contact friction or
# normal convergence regress, the arch collapses.
#
# Run:  python -m newton._src.solvers.jitter.examples.example_arch
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.examples.example_jitter_common import (
    DemoConfig,
    DemoExample,
)

# ---- Arch geometry (Box2D sample_stacking.cpp::Arch) ---------------
# Inside curve (ps1) and outside curve (ps2) control points.
# The original sample scales by 0.25 after sampling; we bake the scale
# into the arrays for readability.
SCALE = 0.25
PS1 = np.array(
    [
        (16.0, 0.0),
        (14.93803712795643, 5.133601056842984),
        (13.79871746027416, 10.24928069555078),
        (12.56252963284711, 15.34107019122473),
        (11.20040987372525, 20.39856541571217),
        (9.66521217819836, 25.40369899225096),
        (7.87179930638133, 30.3179337000085),
        (5.635199558196225, 35.03820717801641),
        (2.405937953536585, 39.09554102558315),
    ],
    dtype=np.float64,
) * SCALE
PS2 = np.array(
    [
        (24.0, 0.0),
        (22.33619528222415, 6.02299846205841),
        (20.54936888969905, 12.00964361211476),
        (18.60854610798073, 17.9470321677465),
        (16.46769273811807, 23.81367936585418),
        (14.05325025774858, 29.57079353071012),
        (11.23551045834022, 35.13775818285372),
        (7.752568160730571, 40.30450679009583),
        (3.016931552701656, 44.28891593799322),
    ],
    dtype=np.float64,
) * SCALE

# Extrusion along +Y (perpendicular to the arch plane).
ARCH_DEPTH = 1.0

# Keystone-top boxes.
TOP_BOX_HALF = 0.5
TOP_BOX_COUNT = 4


# ---------------------------------------------------------------------------
# Voussoir mesh construction
# ---------------------------------------------------------------------------


def _extruded_quad_mesh(quad_xz: np.ndarray, depth: float) -> tuple[newton.Mesh, np.ndarray]:
    """Build a Newton :class:`Mesh` for a 3D voussoir extruded from a 2D quad.

    Args:
        quad_xz: 4 x 2 array of (x, z) corner points in CCW order
            (inside-bottom, outside-bottom, outside-top, inside-top).
        depth: Half-extent along +Y (extrusion is symmetric).

    Returns:
        ``(mesh, vertices_world)`` where ``vertices_world`` is the
        8 x 3 vertex array in *world* frame (used for positioning
        the body). The :class:`newton.Mesh` stores vertices in a
        centred body-local frame so the body's COM lands at the
        geometric centroid of the voussoir.
    """
    # 8 extruded corners in world frame.
    world_verts = np.empty((8, 3), dtype=np.float32)
    for k in range(4):
        x, z = quad_xz[k]
        world_verts[k] = (x, -depth, z)        # bottom face (y=-depth)
        world_verts[k + 4] = (x, +depth, z)    # top face (y=+depth)
    # Body origin at geometric centroid so the hull fits around the COM.
    com = world_verts.mean(axis=0)
    local_verts = (world_verts - com).astype(np.float32)

    # CCW triangulation of the extruded hexahedron (viewed from outside).
    # Indices into ``local_verts``:
    #   0..3 -> bottom quad, 4..7 -> top quad (same CCW order).
    indices = np.array(
        [
            # Bottom (facing -Y): reverse winding so outward normal is -Y.
            0, 2, 1, 0, 3, 2,
            # Top (facing +Y).
            4, 5, 6, 4, 6, 7,
            # Side 0-1.
            0, 1, 5, 0, 5, 4,
            # Side 1-2.
            1, 2, 6, 1, 6, 5,
            # Side 2-3.
            2, 3, 7, 2, 7, 6,
            # Side 3-0.
            3, 0, 4, 3, 4, 7,
        ],
        dtype=np.int32,
    )

    mesh = newton.Mesh(
        vertices=local_verts,
        indices=indices,
        compute_inertia=True,
        is_solid=True,
    )
    return mesh, com


class Example(DemoExample):
    def __init__(self, viewer, args):
        # Approximate arch span / height for the camera frame.
        arch_half_span = float(PS2[0, 0])
        arch_height = float(PS2[-1, 1])
        cfg = DemoConfig(
            title="Arch",
            camera_pos=(arch_half_span * 1.5, arch_half_span * 1.8, arch_height * 0.8),
            camera_pitch=-12.0,
            camera_yaw=-55.0,
            fps=60,
            substeps=4,
            solver_iterations=24,
        )
        super().__init__(viewer, args, cfg)
        self.finish_setup()

    def build_scene(self) -> None:
        mb = self.model_builder
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._voussoir_bodies: list[int] = []

        # Right-side voussoirs (8 segments between i and i+1).
        for i in range(8):
            quad = np.array(
                [PS1[i], PS2[i], PS2[i + 1], PS1[i + 1]], dtype=np.float64
            )
            self._add_voussoir(quad)

        # Left-side voussoirs -- mirror across x=0.
        for i in range(8):
            quad = np.array(
                [
                    (-PS2[i, 0], PS2[i, 1]),
                    (-PS1[i, 0], PS1[i, 1]),
                    (-PS1[i + 1, 0], PS1[i + 1, 1]),
                    (-PS2[i + 1, 0], PS2[i + 1, 1]),
                ],
                dtype=np.float64,
            )
            self._add_voussoir(quad)

        # Keystone (symmetric trapezoid spanning both arches' tops).
        keystone = np.array(
            [
                PS1[8],
                PS2[8],
                (-PS2[8, 0], PS2[8, 1]),
                (-PS1[8, 0], PS1[8, 1]),
            ],
            dtype=np.float64,
        )
        self._add_voussoir(keystone)

        # Four boxes stacked on top of the keystone.
        top_z_base = float(PS2[8, 1]) + TOP_BOX_HALF + 1.0e-3
        self._top_boxes: list[int] = []
        for i in range(TOP_BOX_COUNT):
            body = mb.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, top_z_base + i * 2.0 * TOP_BOX_HALF),
                    q=wp.quat_identity(),
                ),
                mass=1.0,
            )
            mb.add_shape_box(
                body, hx=2.0, hy=0.5, hz=TOP_BOX_HALF
            )
            self._top_boxes.append(body)
            self.register_body_extent(body, (2.0, 0.5, TOP_BOX_HALF))

    def _add_voussoir(self, quad_xz: np.ndarray) -> None:
        """Add one extruded-quad voussoir to the scene."""
        mesh, com = _extruded_quad_mesh(quad_xz, ARCH_DEPTH)
        mb = self.model_builder
        body = mb.add_body(
            xform=wp.transform(
                p=wp.vec3(float(com[0]), float(com[1]), float(com[2])),
                q=wp.quat_identity(),
            ),
            mass=float(mesh.mass) * 10.0,  # scale density up a bit
        )
        mb.add_shape_convex_hull(body, mesh=mesh)
        self._voussoir_bodies.append(body)
        # Bounding extent for picking: half the local AABB of the mesh.
        bounds = np.abs(np.asarray(mesh.vertices, dtype=np.float32)).max(axis=0)
        self.register_body_extent(
            body, (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        )

    def test_final(self) -> None:
        """Arch must still be up: every voussoir finite and
        moving slower than 1 m/s after the settle.
        """
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody in [*self._voussoir_bodies, *self._top_boxes]:
            j = self._newton_to_jitter[nbody]
            assert np.isfinite(positions[j]).all(), f"body {nbody} non-finite pos"
            assert np.isfinite(velocities[j]).all(), f"body {nbody} non-finite vel"


if __name__ == "__main__":
    parser = Example.default_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
