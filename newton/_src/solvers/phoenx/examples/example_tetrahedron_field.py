# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Tetrahedron-primitive stress test for PhoenX.
#
# A scene built almost entirely out of the first-class
# :data:`newton.GeoType.TETRAHEDRON` primitive:
#
# * a ground plane at the bottom;
# * a moderate stack of dynamic tetrahedra falling onto the ground, each
#   with randomised canonical parameters (``edge_ab``, ``point_c``,
#   ``point_d``) so no two tets are identical and the GJK / MPR path
#   gets exercised on a wide variety of convex geometries.
#
# The randomisation deliberately keeps the tetrahedra "well shaped" --
# i.e. all four vertices are clearly off the others' faces -- so we
# avoid near-degenerate sliver tets that would stress GJK numerically
# rather than the contact / solver code we're trying to demo.
#
# All shapes use ``margin = 0.01`` and ``gap = 0.03``, the same defaults
# as the triangle field example, to keep PhoenX's sticky contact
# matching warm-start happy across tet-vs-tet and tet-vs-plane pairs.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_tetrahedron_field
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    run_ported_example,
)

# --- Scene constants ---------------------------------------------------

#: Per-shape collision margin [m] applied to every tetrahedron and the
#: ground plane. Small but non-zero so PhoenX's sticky contact matching
#: has a stable warm-start corridor on the four tet faces.
SHAPE_MARGIN = 0.0

#: Per-shape collision gap [m] inflates the broad-phase AABB so contacts
#: get generated slightly before geometric penetration. Large enough to
#: absorb one frame of closing velocity at 60 Hz under gravity.
SHAPE_GAP = 0.03

#: Grid of dynamic tetrahedra raining onto the ground.
TET_GRID_X = 5
TET_GRID_Y = 5
TET_LAYER_COUNT = 3
TET_SPACING_XY = 1.1  # planar spacing between tets in the same layer
TET_SPACING_Z = 1.0  # vertical spacing between layers
TET_GRID_OFFSET_XY = (-2.2, -2.2)
TET_BASE_HEIGHT = 1.5  # bottom layer hangs ``TET_BASE_HEIGHT`` over the ground.
TET_PLANAR_JITTER = 0.12  # each tet is bumped by U(-jitter, +jitter) in X and Y.

#: Canonical-parameter ranges for the per-tet randomisation.
#:
#: The canonical tetrahedron is
#:
#:     A = (0, 0, 0)
#:     B = (0, 0, edge_ab)
#:     C = (0, c_y, c_z)
#:     D = (d_x, d_y, d_z)
#:
#: Volume = |edge_ab * c_y * d_x| / 6, so all three of those must be
#: clearly non-zero for a healthy tet. The ranges below intentionally
#: cluster around "near-regular" tets: edges similar in length, vertices
#: not extreme outliers, so the sampled tets read as a varied population
#: of small-medium-irregular pyramids rather than slivers or needles.
TET_EDGE_AB_RANGE = (0.35, 0.65)  # |AB|
TET_C_Y_RANGE = (0.30, 0.55)  # vertex C 'height' along local +Y
TET_C_Z_RANGE = (0.20, 0.55)  # vertex C along-base offset (positive)
TET_D_X_RANGE = (0.30, 0.55)  # vertex D 'depth' along local +X (off the YZ plane)
TET_D_Y_RANGE = (0.10, 0.40)  # vertex D Y-coordinate
TET_D_Z_RANGE = (0.20, 0.55)  # vertex D Z-coordinate

#: Random orientation envelope per tet -- yaw is uniform on [0, 2 pi)
#: and pitch / roll are bounded so the resulting tets are tilted but
#: not all upside-down.
TET_PITCH_RANGE = (-0.35, 0.35)
TET_ROLL_RANGE = (-0.35, 0.35)

RNG_SEED = 1729


def _random_tet_xform(rng: np.random.Generator, x: float, y: float, z: float) -> wp.transform:
    """Build a randomised world transform for one dynamic tetrahedron.

    Yaw is uniform on a full circle (so tets are not all aligned to the
    grid) while pitch and roll are bounded so we don't generate
    degenerate "edge-on the ground" landings on the very first frame.
    """
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    pitch = float(rng.uniform(*TET_PITCH_RANGE))
    roll = float(rng.uniform(*TET_ROLL_RANGE))
    yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    pitch_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), pitch)
    roll_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), roll)
    rot = wp.mul(yaw_q, wp.mul(pitch_q, roll_q))
    return wp.transform(p=wp.vec3(float(x), float(y), float(z)), q=rot)


def _tet_picking_extent(
    edge_ab: float,
    c_y: float,
    c_z: float,
    d_x: float,
    d_y: float,
    d_z: float,
) -> tuple[float, float, float]:
    """Half-extents of the canonical-frame AABB enclosing all four
    tet vertices, used for the viewer's pick-OBB only.

    Vertex A is at the local origin so the AABB simply spans
    ``[min(0, c_y, d_y), max(0, c_y, d_y)]`` along Y, etc. We return
    half-widths centred on the canonical origin (vertex A), which is
    exactly what the picking OBB expects.
    """
    xs = [0.0, 0.0, 0.0, d_x]
    ys = [0.0, 0.0, c_y, d_y]
    zs = [0.0, edge_ab, c_z, d_z]
    return (
        max(abs(min(xs)), abs(max(xs))),
        max(abs(min(ys)), abs(max(ys))),
        max(abs(min(zs)), abs(max(zs))),
    )


class Example(PortedExample):
    """Tetrahedron-heavy PhoenX scene: ground + falling tet shower.

    Demonstrates that the first-class :data:`newton.GeoType.TETRAHEDRON`
    primitive routes through PhoenX with no solver-side changes -- the
    4th vertex ``D`` is encoded into ``shape_source_ptr`` by the
    builder and decoded on the GPU by the shared narrow phase, so the
    GJK / MPR path picks up tet-vs-tet and tet-vs-plane pairs with no
    PhoenX-specific glue.
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

        # ---- Dynamic tetrahedra -------------------------------------
        # Mass and inertia come from the closed-form solid-tet formula
        # (Tonon 2005): ``mass = density * |AB . (AC x AD)| / 6``, COM
        # at the four-vertex centroid, full 3x3 inertia tensor about
        # that centroid. See ``compute_inertia_tetrahedron``.
        tet_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0,
            mu=self.default_friction,
            restitution=self.default_restitution,
            margin=SHAPE_MARGIN,
            gap=SHAPE_GAP,
        )
        for layer in range(TET_LAYER_COUNT):
            z_layer = TET_BASE_HEIGHT + layer * TET_SPACING_Z
            for ix in range(TET_GRID_X):
                for iy in range(TET_GRID_Y):
                    px = (
                        TET_GRID_OFFSET_XY[0]
                        + ix * TET_SPACING_XY
                        + float(rng.uniform(-TET_PLANAR_JITTER, TET_PLANAR_JITTER))
                    )
                    py = (
                        TET_GRID_OFFSET_XY[1]
                        + iy * TET_SPACING_XY
                        + float(rng.uniform(-TET_PLANAR_JITTER, TET_PLANAR_JITTER))
                    )
                    pz = z_layer + float(rng.uniform(-0.05, 0.05))

                    edge_ab = float(rng.uniform(*TET_EDGE_AB_RANGE))
                    c_y = float(rng.uniform(*TET_C_Y_RANGE))
                    c_z = float(rng.uniform(*TET_C_Z_RANGE)) * edge_ab
                    d_x = float(rng.uniform(*TET_D_X_RANGE))
                    d_y = float(rng.uniform(*TET_D_Y_RANGE))
                    d_z = float(rng.uniform(*TET_D_Z_RANGE)) * edge_ab

                    body = builder.add_body(xform=_random_tet_xform(rng, px, py, pz))
                    builder.add_shape_tetrahedron(
                        body=body,
                        point_a=wp.vec3(0.0, 0.0, 0.0),
                        point_b=wp.vec3(0.0, 0.0, edge_ab),
                        point_c=wp.vec3(0.0, c_y, c_z),
                        point_d=wp.vec3(d_x, d_y, d_z),
                        cfg=tet_cfg,
                    )
                    extents.append(_tet_picking_extent(edge_ab, c_y, c_z, d_x, d_y, d_z))

        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(0.0, -7.0, 4.5), pitch=-25.0, yaw=90.0)

    def test_final(self) -> None:
        """Every tet has finite state and stayed roughly over the spawn
        footprint. Tolerance is generous: under PhoenX with ``mu=0.5``
        and ``restitution=0.0`` the pile should settle within a few
        spacings of the original grid extent for the test horizon
        (default 100 frames)."""
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        # Skip the static ground slot (index 0); dynamic tets follow.
        if positions.shape[0] <= 1:
            return
        dyn_pos = positions[1:]
        dyn_vel = velocities[1:]
        assert np.isfinite(dyn_pos).all(), "non-finite tetrahedron position"
        assert np.isfinite(dyn_vel).all(), "non-finite tetrahedron velocity"

        spawn_half_extent_xy = 0.5 * (max(TET_GRID_X, TET_GRID_Y) - 1) * TET_SPACING_XY + TET_PLANAR_JITTER
        xy_tol = spawn_half_extent_xy + 4.0  # tets can slide a few metres on bounce
        max_xy = float(np.max(np.linalg.norm(dyn_pos[:, :2], axis=1)))
        assert max_xy < xy_tol, f"tetrahedra spread too far ({max_xy:.2f} > {xy_tol:.2f})"

        # No tet should be far below the ground plane (z=0). A small
        # negative z is fine because of margin / gap, but anything
        # deeper than a couple of margins indicates tunnelling.
        z_floor_tol = -10.0 * SHAPE_MARGIN
        min_z = float(np.min(dyn_pos[:, 2]))
        assert min_z > z_floor_tol, (
            f"tetrahedron tunnelled through the ground (min_z={min_z:.4f}, tol={z_floor_tol:.4f})"
        )


if __name__ == "__main__":
    run_ported_example(Example)
