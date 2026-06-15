# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Triangle cloth: rigid-body cloth made of triangle pairs joined at corners.
#
# Each cloth quad is split into two triangles along its diagonal so the
# cloth surface is a triangulated mesh. Every triangle is a separate rigid
# body whose mass / inertia comes from the prism interpretation of
# :data:`newton.GeoType.TRIANGLE` (thickness ``2 * margin``); neighbouring
# triangles are tied together with ball-socket joints at the cloth-grid
# corners they share, so each corner anchors a "fan" of incident triangle
# bodies in the same way the capsule net anchors a star of capsule
# segments.
#
# Comparison with :mod:`example_capsule_net`:
#
# * Capsule net: cells are linked by capsule *edges* and joined at corners.
# * Triangle cloth: cells are filled by triangle *faces* (two per quad)
#   and joined at corners as well.
#
# Both share the same chain-of-incident-bodies pattern at each corner --
# one ball-socket joint per consecutive pair, no redundant links.
#
# Collision filtering. Two triangles that share even a single vertex sit
# at zero (or near-zero) separation at rest and would generate spurious
# contacts; with ``shape_pairs_max=None`` and the default broad phase
# we'd also burn a huge number of NarrowPhase candidates on these. We
# therefore call :meth:`~newton.ModelBuilder.add_shape_collision_filter_pair`
# for every triangle pair that shares any cloth-grid vertex (within a
# net), which is O(corners * incident_pairs) and bounded above by
# ~6^2 / 2 per corner for a regular grid. Non-adjacent triangles still
# collide normally (e.g. two cloths swinging into each other).
#
# Articulation. PhoenX's ADBS treats every enabled joint as an
# independent constraint column, so the cloth's joint graph is allowed
# to have cycles -- but ``ModelBuilder.finalize`` insists every dynamic
# body be reachable from some articulated joint. ``add_body`` already
# does that for us: each call wraps the new body in a single-joint
# articulation backed by a FREE joint (which the PhoenX model adapter
# skips, see ``newton._src.solvers.phoenx.model_adapter``). Every
# ball-socket joint we add between triangles is therefore a pure loop
# closure and lives outside any articulation.
#
# Run::
#
#     python -m newton._src.solvers.phoenx.examples.example_triangle_cloth
###########################################################################

from __future__ import annotations

import colorsys

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    run_ported_example,
)

# --- Scene constants ---------------------------------------------------

#: Per-shape collision margin [m]. Doubles as the cloth's half-thickness:
#: ``compute_inertia_shape`` reads ``cfg.margin`` as the prism half-extent
#: along the triangle's local +X normal, so the cloth has an effective
#: physical thickness of ``2 * SHAPE_MARGIN`` for both inertia and contact.
SHAPE_MARGIN = 0.01

#: Per-shape collision gap [m]. Inflates the broad-phase AABB so contacts
#: are generated slightly before geometric penetration.
SHAPE_GAP = 0.03

#: Cloth grid: NUM_QUADS_X x NUM_QUADS_Y quads of cloth, with one extra
#: row + column of corner positions, so the cloth has
#: ``(NUM_QUADS_X + 1) x (NUM_QUADS_Y + 1)`` corners.
NUM_QUADS_X = 10
NUM_QUADS_Y = 10
CELL_PITCH = 0.12  # corner-to-corner spacing along world X / Y [m]

#: Material density [kg/m^3]. Combined with the triangle's prism volume
#: (area * 2 * SHAPE_MARGIN) this gives a per-triangle mass of order
#: ``rho * 0.5 * pitch^2 * 2 * margin`` -- a few grams for the defaults.
CLOTH_DENSITY = 600.0

#: Cloth spawn pose. The cloth lies flat in the world XY plane at this
#: height; pinned corners (when enabled) attach the top row to the world
#: at z = HEIGHT.
HEIGHT = 1.5

#: Multi-cloth setup. Each cloth gets a unique hue shift so neighbours
#: read as visually distinct as they swing into each other.
NUM_NETS = 2
NET_SPACING_CELLS = 1.5  # extra Y-gap between cloths, in cell pitches

#: Pin behaviour: which top-row corners are tacked to the world via a
#: world-anchored ball-socket joint. ``"corners"`` pins (0, NY) and (NX, NY);
#: ``"edge"`` pins the entire top row.
PIN_MODE = "corners"

#: Hue offsets applied to the cloth's base RGB so each net stands out.
_NET_HUE_OFFSETS = (0.0, 0.27, 0.54, 0.13, 0.40, 0.67)
#: Two RGB tones for the two triangles in each quad. The "lower-left"
#: triangle (vertices A-B-C, see ``_quad_triangles``) gets the warm tone;
#: the "upper-right" triangle (A-C-D) gets the cool tone.
_TRI_BASE_COLOR_LO = (0.95, 0.55, 0.20)  # warm orange
_TRI_BASE_COLOR_UP = (0.20, 0.65, 0.95)  # cool blue


def _shift_hue(rgb: tuple[float, float, float], hue_offset: float) -> tuple[float, float, float]:
    """Rotate the hue of an RGB triple by ``hue_offset`` (in [0, 1])."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + hue_offset) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)


def _triangle_canonical_from_corners(
    p_a: tuple[float, float, float],
    p_b: tuple[float, float, float],
    p_c: tuple[float, float, float],
) -> tuple[wp.transform, float, float, float]:
    """Map three world-space corners to the canonical triangle params + xform.

    The :data:`newton.GeoType.TRIANGLE` primitive is parameterised in a
    canonical local frame:

    * vertex A at the local origin,
    * vertex B at ``(0, 0, edge_ab)`` along the local +Z axis,
    * vertex C at ``(0, c_y, c_z)`` in the local YZ plane (``c_y > 0``).

    Given any three non-collinear world-space corners ``p_a, p_b, p_c``,
    this returns the rigid transform that places the canonical triangle
    so its three vertices land on those corners, plus the canonical
    ``(edge_ab, c_y, c_z)``. The triangle's local +X axis is the face
    normal pointing from A toward the half-plane that does *not* contain
    C reflected across AB.
    """
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    c = np.asarray(p_c, dtype=np.float64)

    ab = b - a
    edge_ab = float(np.linalg.norm(ab))
    if edge_ab <= 1.0e-12:
        raise ValueError(f"Degenerate triangle: |AB| = 0 (a={p_a}, b={p_b})")
    local_z = ab / edge_ab

    # Local +Y is the component of (C - A) perpendicular to AB,
    # normalised. Local +X is local_y x local_z (right-handed).
    ac = c - a
    c_z = float(np.dot(ac, local_z))
    perp = ac - c_z * local_z
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm <= 1.0e-12:
        raise ValueError(f"Degenerate triangle: A, B, C collinear (a={p_a}, b={p_b}, c={p_c})")
    local_y = perp / perp_norm
    c_y = perp_norm
    local_x = np.cross(local_y, local_z)

    # Build the world rotation matrix whose columns are the local axes
    # expressed in world coords; convert to a quaternion. Using a 3x3
    # construction + numpy's quaternion-from-matrix avoids a separate
    # scipy dependency.
    rot_mat = np.column_stack((local_x, local_y, local_z)).astype(np.float64)
    quat = _quat_from_matrix(rot_mat)
    xform = wp.transform(p=wp.vec3(float(a[0]), float(a[1]), float(a[2])), q=wp.quat(*quat))
    return xform, edge_ab, c_y, c_z


def _quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to ``(x, y, z, w)``.

    Standard branch-on-trace algorithm; numerically stable for any
    proper rotation. Inlined here so we can avoid pulling in scipy.
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _world_to_local(
    xform: wp.transform,
    point_world: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Transform a world-space point into the body's local frame.

    Used to express each ball-socket joint's anchor in the local frame
    of the triangle bodies it ties together; ``add_joint_ball`` takes
    a body-local ``parent_xform`` / ``child_xform``.
    """
    p_local = wp.transform_point(wp.transform_inverse(xform), wp.vec3(*point_world))
    return float(p_local[0]), float(p_local[1]), float(p_local[2])


def _quad_triangles(
    corners: dict[tuple[int, int], tuple[float, float, float]],
    i: int,
    j: int,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
]:
    """Two triangle vertex-index triples for the quad at grid cell (i, j).

    The quad has corners ``(i, j), (i+1, j), (i+1, j+1), (i, j+1)``;
    we split along the diagonal from ``(i, j)`` to ``(i+1, j+1)``:

    * Lower-right triangle (``T1``): ``(i, j) -> (i+1, j) -> (i+1, j+1)``.
    * Upper-left triangle (``T2``):  ``(i, j) -> (i+1, j+1) -> (i, j+1)``.

    Each tuple is returned in the order ``(A, B, C)`` that we feed into
    :func:`_triangle_canonical_from_corners`. The shared diagonal is the
    edge ``A -> C`` of both triangles, so ball-socket joints at the two
    diagonal corners (``(i, j)`` and ``(i+1, j+1)``) are sufficient to
    keep the quad coherent before any inter-quad joints fire.
    """
    a = (i, j)
    b = (i + 1, j)
    c = (i + 1, j + 1)
    d = (i, j + 1)
    # Sanity check: caller must supply all four corner positions.
    for v in (a, b, c, d):
        if v not in corners:
            raise KeyError(f"_quad_triangles: missing corner {v}")
    return (a, b, c), (a, c, d)


class Example(PortedExample):
    """Triangulated rigid-body cloth held together by ball-socket joints.

    Each cloth quad is two :data:`newton.GeoType.TRIANGLE` bodies
    sharing the diagonal ``A-C``; corners shared between neighbouring
    quads / triangles are tied with ball-socket joints. The top row
    is optionally pinned to the world so the cloth hangs.
    """

    fps = 60
    sim_substeps = 16
    solver_iterations = 12
    velocity_iterations = 1
    broad_phase = "sap"
    step_layout = "single_world"
    show_contacts = False
    default_friction = 0.5

    def build_scene(self, builder: newton.ModelBuilder):
        nx = int(NUM_QUADS_X)
        ny = int(NUM_QUADS_Y)
        if nx < 1 or ny < 1:
            raise ValueError("NUM_QUADS_X and NUM_QUADS_Y must be >= 1")
        pitch = float(CELL_PITCH)
        height = float(HEIGHT)
        net_spacing = float((ny + NET_SPACING_CELLS) * pitch)

        # ---- Ground plane (catches falling cloth). ----------------------
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                margin=SHAPE_MARGIN,
                gap=SHAPE_GAP,
            ),
            height=1.0,
        )

        extents: list[tuple[float, float, float] | None] = []
        # Slot 0 of ``extents`` corresponds to a Newton body if/when one
        # is added; ``add_ground_plane`` only adds a shape on body=-1
        # (the world), so it consumes no body slot.

        all_pinned = 0
        all_loop_joints = 0
        all_triangles = 0

        for net_idx in range(int(NUM_NETS)):
            center_y = (net_idx - 0.5 * (NUM_NETS - 1)) * net_spacing
            hue_offset = _NET_HUE_OFFSETS[net_idx % len(_NET_HUE_OFFSETS)] if NUM_NETS > 1 else 0.0
            color_lo = _shift_hue(_TRI_BASE_COLOR_LO, hue_offset)
            color_up = _shift_hue(_TRI_BASE_COLOR_UP, hue_offset)

            n_loop, n_pin, n_tri = self._build_one_cloth(
                builder=builder,
                nx=nx,
                ny=ny,
                pitch=pitch,
                center_y=center_y,
                height=height,
                color_lo=color_lo,
                color_up=color_up,
                extents=extents,
            )
            all_loop_joints += n_loop
            all_pinned += n_pin
            all_triangles += n_tri

        print(
            f"[PhoenX TriangleCloth] nets={NUM_NETS} grid={nx}x{ny} quads/net "
            f"triangles={all_triangles} bodies={builder.body_count} "
            f"joints=loop:{all_loop_joints} pin:{all_pinned} "
            f"pitch={pitch:.3f}m margin={SHAPE_MARGIN:.3f}m gap={SHAPE_GAP:.3f}m"
        )
        return extents

    # ------------------------------------------------------------------
    # Per-cloth build
    # ------------------------------------------------------------------

    def _build_one_cloth(
        self,
        *,
        builder: newton.ModelBuilder,
        nx: int,
        ny: int,
        pitch: float,
        center_y: float,
        height: float,
        color_lo: tuple[float, float, float],
        color_up: tuple[float, float, float],
        extents: list[tuple[float, float, float] | None],
    ) -> tuple[int, int, int]:
        """Build one cloth and return ``(loop_joints, pin_joints, articulation_joints)``."""
        # World-space corner positions, indexed by grid coords ``(i, j)``.
        corners: dict[tuple[int, int], tuple[float, float, float]] = {}
        cx = 0.5 * nx
        cy = 0.5 * ny
        for j in range(ny + 1):
            for i in range(nx + 1):
                corners[(i, j)] = (
                    (i - cx) * pitch,
                    (j - cy) * pitch + center_y,
                    height,
                )

        # Per-quad: two triangle body ids + their canonical xforms (so we
        # can convert world-space anchors into body-local later).
        # Map grid corner -> list of (body_id, world_xform) of every
        # triangle that has a vertex at that corner. The chain we build
        # at each corner walks this list in registration order, which
        # mirrors the capsule-net pattern.
        incident: dict[tuple[int, int], list[tuple[int, wp.transform]]] = {}

        # Track which (sorted) body pairs have already been collision-
        # filtered. Two triangles that share any vertex must not collide
        # (they're glued at that point and otherwise generate non-stop
        # contacts at zero distance).
        filtered_pairs: set[tuple[int, int]] = set()

        # Per-net config. Density drives the prism mass / inertia via
        # ``compute_inertia_shape`` for ``GeoType.TRIANGLE``.
        tri_cfg = newton.ModelBuilder.ShapeConfig(
            density=CLOTH_DENSITY,
            mu=self.default_friction,
            restitution=self.default_restitution,
            margin=SHAPE_MARGIN,
            gap=SHAPE_GAP,
        )

        # ---- Pass 1: spawn triangles --------------------------------
        # Each quad emits two triangles sharing the diagonal A-C.
        # ``add_body`` auto-creates the per-body FREE joint +
        # articulation needed to satisfy ``finalize``'s reachability
        # check; the corner ball joints below are loop closures.
        triangle_count = 0
        for j in range(ny):
            for i in range(nx):
                t1_corners, t2_corners = _quad_triangles(corners, i, j)
                for tri_idx, tri_verts in enumerate((t1_corners, t2_corners)):
                    color = color_lo if tri_idx == 0 else color_up
                    p_a, p_b, p_c = (corners[v] for v in tri_verts)
                    xform, edge_ab, c_y, c_z = _triangle_canonical_from_corners(p_a, p_b, p_c)

                    body = builder.add_body(xform=xform)
                    builder.add_shape_triangle(
                        body=body,
                        point_a=wp.vec3(0.0, 0.0, 0.0),
                        point_b=wp.vec3(0.0, 0.0, edge_ab),
                        point_c=wp.vec3(0.0, c_y, c_z),
                        cfg=tri_cfg,
                        color=color,
                    )
                    triangle_count += 1

                    for v in tri_verts:
                        incident.setdefault(v, []).append((body, xform))

                    # Picking OBB: enclose all three vertices in the
                    # triangle's local frame. A is at the local origin,
                    # B = (0, 0, edge_ab), C = (0, c_y, c_z). Inflate a
                    # hair along local +X so the OBB has nonzero volume
                    # for the picking ray test.
                    half_x = max(SHAPE_MARGIN, 0.01)
                    half_y = 0.5 * abs(c_y) + 1.0e-3
                    half_z = 0.5 * max(edge_ab, abs(c_z)) + 1.0e-3
                    extents.append((half_x, half_y, half_z))

        # ---- Pass 2: ball-socket joints at every shared corner ------
        # Chain consecutive incident bodies via the same anchor so the
        # constraint count is O(corners * (incident_count - 1)) rather
        # than O(incident_count^2). All triangles meeting at a corner
        # are still rigidly tied because each link enforces coincidence
        # at the shared point and the chain transitively connects them.
        loop_joint_count = 0
        for corner, world_pos in corners.items():
            inc = incident.get(corner, [])
            if len(inc) < 2:
                continue
            # First filter every pair of triangles incident here so the
            # collision detector never tries to resolve their (zero or
            # near-zero) penetration at the shared vertex.
            for ka in range(len(inc)):
                for kb in range(ka + 1, len(inc)):
                    pair = (inc[ka][0], inc[kb][0])
                    pair = (min(pair), max(pair))
                    if pair in filtered_pairs or pair[0] == pair[1]:
                        continue
                    filtered_pairs.add(pair)
                    self._filter_collision_between_bodies(builder, pair[0], pair[1])

            # Then chain-link them with one ball-socket joint per pair.
            for k in range(len(inc) - 1):
                body_a, xf_a = inc[k]
                body_b, xf_b = inc[k + 1]
                anchor_a = _world_to_local(xf_a, world_pos)
                anchor_b = _world_to_local(xf_b, world_pos)
                builder.add_joint_ball(
                    parent=body_a,
                    child=body_b,
                    parent_xform=wp.transform(p=wp.vec3(*anchor_a), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(*anchor_b), q=wp.quat_identity()),
                    # ``collision_filter_parent=False`` because we've
                    # already added the explicit shape filter pair above
                    # (and we don't want this single joint to suppress
                    # the rest of the body pair's collisions implicitly).
                    collision_filter_parent=False,
                )
                loop_joint_count += 1

        # ---- Pass 3: pin top-row corners to the world ---------------
        pin_count = 0
        if PIN_MODE == "corners":
            pin_corners = [(0, ny), (nx, ny)]
        elif PIN_MODE == "edge":
            pin_corners = [(i, ny) for i in range(nx + 1)]
        else:
            pin_corners = []
        for corner in pin_corners:
            inc = incident.get(corner, [])
            if not inc:
                continue
            # Pinning any one incident triangle is enough; the local
            # corner chain tied above carries the lock to its peers.
            body, xf = inc[0]
            anchor_local = _world_to_local(xf, corners[corner])
            anchor_world = corners[corner]
            builder.add_joint_ball(
                parent=-1,
                child=body,
                parent_xform=wp.transform(p=wp.vec3(*anchor_world), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(*anchor_local), q=wp.quat_identity()),
                collision_filter_parent=False,
            )
            pin_count += 1

        # Each ``add_body`` above already wrapped its FREE joint in a
        # one-joint articulation, so ``finalize`` is satisfied without
        # any explicit ``add_articulation`` call here. Loop-closure
        # ball joints (corner chain + pins) live outside any
        # articulation, which is allowed because their child bodies
        # are reachable through their own per-body FREE joint.

        return loop_joint_count, pin_count, triangle_count

    @staticmethod
    def _filter_collision_between_bodies(builder: newton.ModelBuilder, body_a: int, body_b: int) -> None:
        """Mark every shape on ``body_a`` as not-colliding with every
        shape on ``body_b``.

        ``ModelBuilder.add_shape_collision_filter_pair`` works on shape
        indices; for the cloth each triangle body has exactly one
        shape, but we walk ``body_shapes`` so this stays correct if a
        future variant attaches multiple shapes per triangle (e.g. a
        sphere at each vertex)."""
        shapes_a = builder.body_shapes[body_a]
        shapes_b = builder.body_shapes[body_b]
        for sa in shapes_a:
            for sb in shapes_b:
                lo, hi = (sa, sb) if sa < sb else (sb, sa)
                builder.add_shape_collision_filter_pair(lo, hi)

    # ------------------------------------------------------------------
    # Camera / sanity
    # ------------------------------------------------------------------

    def configure_camera(self, viewer):
        # Pull back so several cloths fit in frame; height eyes the
        # middle of the hanging cloth.
        cloth_width = NUM_QUADS_X * CELL_PITCH
        net_spacing = (NUM_QUADS_Y + NET_SPACING_CELLS) * CELL_PITCH
        scene_y_extent = max(cloth_width, (NUM_NETS - 1) * net_spacing + NUM_QUADS_Y * CELL_PITCH)
        viewer.set_camera(
            pos=wp.vec3(0.0, -2.4 * max(scene_y_extent, cloth_width), HEIGHT + 0.3 * cloth_width),
            pitch=-15.0,
            yaw=90.0,
        )

    def test_final(self) -> None:
        positions = self.world.bodies.position.numpy()
        assert np.isfinite(positions).all(), "non-finite body position in triangle cloth"
        # Cloth shouldn't fly apart: the corner pins (when enabled) keep
        # it bounded; even unpinned, free-fall keeps it within a few
        # cloth-widths of spawn for the first few frames the test runs.
        cloth_width = (NUM_QUADS_X + 1) * CELL_PITCH
        net_spacing = (NUM_QUADS_Y + NET_SPACING_CELLS) * CELL_PITCH
        envelope = max(cloth_width, (NUM_NETS - 1) * net_spacing + cloth_width)
        # Skip the static slot (index 0) when measuring lateral travel.
        max_lateral = float(np.max(np.abs(positions[1:, :2]))) if positions.shape[0] > 1 else 0.0
        assert max_lateral < 8.0 * envelope, (
            f"triangle cloth escaped its envelope (max_lateral={max_lateral:.3f}, envelope={envelope:.3f})"
        )


if __name__ == "__main__":
    run_ported_example(Example)
