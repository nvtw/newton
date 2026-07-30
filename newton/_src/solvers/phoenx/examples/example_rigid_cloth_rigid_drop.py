# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX rigid-cloth + rigid-cube drop demo.

A rigid-body "cloth" -- a triangulated grid where every triangle is its
own rigid body and shared corners are tied together with ball-socket
joints -- catches a small rigid cube falling from above. This is the
articulated cousin of :mod:`example_cloth_rigid_drop`: instead of a
deformable cloth iterate, the cloth is a network of
:data:`newton.GeoType.TRIANGLE` rigid bodies whose mass and inertia
come from the prism interpretation (thickness ``2 * margin``), connected
by ball joints solved as one direct equality mechanism.

The scene uses :class:`newton.solvers.SolverPhoenX`, mirroring the structure
of :mod:`example_cloth_rigid_drop`:

* Triangulated cloth is built in :class:`newton.ModelBuilder`, two
  triangles per quad sharing the diagonal A-C, joined at every shared
  corner via :meth:`~newton.ModelBuilder.add_joint_ball` chained between
  consecutive incident triangles. Adjacent triangles' collisions are
  filtered so they don't generate spurious zero-distance contacts at
  their shared vertices.
* The four corner triangles are pinned to the world with
  :meth:`~newton.ModelBuilder.add_joint_ball` against ``parent=-1``.
* A free-floating rigid cube is added with density-derived mass /
  inertia and spawns above the cloth centre.
* PhoenX detects the connected rigid-cloth joint mechanism automatically and
  solves its bilateral rows directly; contact inequalities remain in PGS.

Run::

    python -m newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------------------------------------------------------------
# Triangle construction helpers
# ---------------------------------------------------------------------


def _world_to_local(
    xform: wp.transform,
    point_world: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Transform a world-space point into the body's local frame."""
    p_local = wp.transform_point(wp.transform_inverse(xform), wp.vec3(*point_world))
    return float(p_local[0]), float(p_local[1]), float(p_local[2])


def _filter_collision_between_bodies(builder: newton.ModelBuilder, body_a: int, body_b: int) -> None:
    """Mark every shape on ``body_a`` as not-colliding with every shape on ``body_b``."""
    shapes_a = builder.body_shapes[body_a]
    shapes_b = builder.body_shapes[body_b]
    for sa in shapes_a:
        for sb in shapes_b:
            lo, hi = (sa, sb) if sa < sb else (sb, sa)
            builder.add_shape_collision_filter_pair(lo, hi)


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------


class Example:
    """A rigid-body triangulated cloth pinned at four corners catching a falling rigid cube."""

    def __init__(
        self,
        viewer,
        args=None,
        width: int = 12,
        height: int = 12,
        cloth_density: float = 600.0,
        mass_splitting: bool = False,
        mass_splitting_unrolled: bool = False,
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Bilateral cloth rows are solved directly; PGS work is limited to
        # cube/cloth contact inequalities.
        self.sim_substeps = 5
        self.solver_iterations = 2
        self.velocity_iterations = 1

        self.dim_x = int(width)
        self.dim_y = int(height)
        self.cell = 0.05
        self.cloth_z = 1.0
        self.cloth_margin = 0.005
        # Per-shape contact gap [m]. The broad phase inflates each
        # triangle's AABB by ``gap`` and pairs ``(a, b)`` whose
        # inflated boxes overlap; for a planar triangle grid that
        # means ``gap_a + gap_b`` must stay strictly below the cell
        # pitch, otherwise the inflated AABB of one triangle reaches
        # past its immediate neighbours into the next-but-one row /
        # column and we generate spurious "non-touching" contacts
        # between coplanar triangles that don't share a vertex (so
        # aren't suppressed by the share-vertex filter). A safe
        # bound is ``gap < 0.5 * cell`` (gap-sum < ``cell``); we use
        # a tighter ``0.1 * cell`` to keep the contact pre-roll
        # narrow without sacrificing the cube's contact envelope.
        self.cloth_gap = 0.1 * self.cell
        self.cloth_density = float(cloth_density)

        # ---- Build the Newton model -----------------------------------
        # Cloth lies flat in the XY plane, centred on the origin; the
        # rigid cube spawns above the centre and falls onto it.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

        self._build_rigid_cloth(builder)

        self.cube_he = 0.05
        self.cube_spawn_z = self.cloth_z + 0.4
        self.cube_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.cube_spawn_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(
            self.cube_body,
            hx=self.cube_he,
            hy=self.cube_he,
            hz=self.cube_he,
            cfg=newton.ModelBuilder.ShapeConfig(density=600.0, mu=0.6),
        )
        self.model = builder.finalize(device=self.device)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            broad_phase="nxn",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            collision_pipeline=self.collision_pipeline,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            step_layout="single_world",
            articulation_mode="maximal",
            mass_splitting=mass_splitting,
            mass_splitting_unrolled=mass_splitting_unrolled,
        )
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.0, 1.6), pitch=-15.0, yaw=135.0)

        self._capture()

    # ------------------------------------------------------------------
    # Cloth construction
    # ------------------------------------------------------------------

    def _build_rigid_cloth(self, builder: newton.ModelBuilder) -> None:
        """Build a triangulated rigid-body cloth, pinned at its four corners.

        The cloth is laid out in the world XY plane, centred on the
        origin, at z = ``self.cloth_z``. Each quad ``(i, j) -> (i+1,
        j+1)`` is split along its diagonal into two triangle bodies;
        every shared corner is tied with a chain of ball-socket
        joints, and the four extreme corners are anchored to the
        world.
        """
        nx = self.dim_x
        ny = self.dim_y
        pitch = self.cell

        cx = 0.5 * nx
        cy = 0.5 * ny

        # World-space corner positions, indexed by grid coords (i, j).
        corners: dict[tuple[int, int], tuple[float, float, float]] = {}
        for j in range(ny + 1):
            for i in range(nx + 1):
                corners[(i, j)] = (
                    (i - cx) * pitch,
                    (j - cy) * pitch,
                    self.cloth_z,
                )

        # Map grid corner -> list of (body_id, world_xform) of every
        # triangle incident at that corner. The chain we build at each
        # corner walks this list in registration order.
        incident: dict[tuple[int, int], list[tuple[int, wp.transform]]] = {}

        # Track which (sorted) body pairs have already been collision-
        # filtered. Triangles sharing any vertex sit at zero separation
        # at rest and would otherwise generate non-stop contacts.
        filtered_pairs: set[tuple[int, int]] = set()

        tri_cfg = newton.ModelBuilder.ShapeConfig(
            density=self.cloth_density,
            mu=0.6,
            margin=self.cloth_margin,
            gap=self.cloth_gap,
        )

        # ---- Pass 1: spawn triangles --------------------------------
        # Two triangles per quad, sharing the diagonal A-C. ``add_body``
        # auto-creates the per-body FREE joint + articulation needed
        # to satisfy ``finalize``'s reachability check; the corner
        # ball joints below are pure loop closures.
        tri_color_lo = (0.95, 0.55, 0.20)
        tri_color_up = (0.20, 0.65, 0.95)
        for j in range(ny):
            for i in range(nx):
                a = (i, j)
                b = (i + 1, j)
                c = (i + 1, j + 1)
                d = (i, j + 1)
                tri_specs = (
                    ((a, b, c), tri_color_lo),  # lower-right
                    ((a, c, d), tri_color_up),  # upper-left
                )
                for tri_verts, color in tri_specs:
                    p_a, p_b, p_c = (corners[v] for v in tri_verts)

                    # Body frame == world frame; ``add_shape_triangle``
                    # rebases the three vertices onto its canonical local
                    # frame internally and folds the offset into the
                    # shape's transform.
                    body_xform = wp.transform_identity()
                    body = builder.add_body(xform=body_xform)
                    builder.add_shape_triangle(
                        body=body,
                        point_a=wp.vec3(*p_a),
                        point_b=wp.vec3(*p_b),
                        point_c=wp.vec3(*p_c),
                        cfg=tri_cfg,
                        color=color,
                    )
                    for v in tri_verts:
                        incident.setdefault(v, []).append((body, body_xform))

        # ---- Pass 2: ball-socket joints at every shared corner ------
        # Chain consecutive incident bodies via the same anchor so the
        # constraint count is O(corners * (incident_count - 1)) rather
        # than O(incident_count^2). All triangles meeting at a corner
        # are still rigidly tied: each link enforces coincidence at
        # the shared point, and the chain transitively connects them.
        loop_joints = 0
        for corner, world_pos in corners.items():
            inc = incident.get(corner, [])
            if len(inc) < 2:
                continue
            # Filter every pair of triangles incident here so the
            # collision detector never tries to resolve their (zero
            # or near-zero) penetration at the shared vertex.
            for ka in range(len(inc)):
                for kb in range(ka + 1, len(inc)):
                    pair = (inc[ka][0], inc[kb][0])
                    pair = (min(pair), max(pair))
                    if pair in filtered_pairs or pair[0] == pair[1]:
                        continue
                    filtered_pairs.add(pair)
                    _filter_collision_between_bodies(builder, pair[0], pair[1])

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
                    # Already added the explicit shape filter pair above;
                    # don't let the joint suppress the rest of the body
                    # pair's collisions implicitly.
                    collision_filter_parent=False,
                )
                loop_joints += 1

        # ---- Pass 3: pin the four cloth corners to the world --------
        self.corner_grid = (
            (0, 0),
            (nx, 0),
            (0, ny),
            (nx, ny),
        )
        self.pinned_bodies: list[int] = []
        self.pinned_anchors: list[tuple[int, np.ndarray, np.ndarray]] = []
        for corner in self.corner_grid:
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
            self.pinned_bodies.append(body)
            self.pinned_anchors.append(
                (
                    body,
                    np.asarray(anchor_local, dtype=np.float32),
                    np.asarray(anchor_world, dtype=np.float32),
                )
            )

        triangle_count = sum(1 for inc in incident.values() for _ in inc) // 3
        print(
            f"[PhoenX RigidClothRigidDrop] grid={nx}x{ny} quads "
            f"triangles={triangle_count} bodies={builder.body_count} "
            f"loop_joints={loop_joints} pins={len(self.pinned_bodies)} "
            f"pitch={pitch:.3f}m margin={self.cloth_margin:.3f}m "
            f"density={self.cloth_density:.0f}"
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_one_frame(self) -> None:
        """Advance the rigid-cloth mechanism and contact inequalities."""
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.collision_pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.state, self.control, self.contacts, self.frame_dt)

    def _capture(self) -> None:
        """Capture one complete frame into a CUDA graph."""
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_one_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_one_frame()
        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Viewer + tests
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After the example finishes:

        * All body positions / orientations are finite.
        * The four pinned-corner triangles haven't drifted far from
          their spawn pose (the world-anchored ball joints hold the
          corner anchor in place; the rest of the triangle can still
          rotate around it).
        * The cube has fallen below its spawn height.
        * The cube hasn't tunnelled through the cloth (a few cube
          heights below the pinned cloth plane is acceptable to allow
          for the cloth sagging under load).
        """
        body_q = self.state.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise RuntimeError("non-finite body transform in final state")

        def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            xyz = np.asarray(q[:3], dtype=np.float32)
            t = 2.0 * np.cross(xyz, v)
            return v + float(q[3]) * t + np.cross(xyz, t)

        for pinned, anchor_local, anchor_world in self.pinned_anchors:
            q = body_q[pinned]
            anchor_now = q[:3] + rotate(q[3:7], anchor_local)
            allowed = 2.0 * self.cell + 1.0e-3
            drift = float(np.linalg.norm(anchor_now - anchor_world))
            if drift > allowed:
                raise RuntimeError(
                    f"pinned cloth corner anchor drifted: body={pinned} drift={drift:.4f} m allowed={allowed:.4f} m"
                )

        cube_z = float(body_q[self.cube_body, 2])
        if cube_z >= self.cube_spawn_z - 1.0e-3:
            raise RuntimeError(f"cube did not fall (z={cube_z:.4f} m, spawn={self.cube_spawn_z:.4f} m)")

        floor = self.cloth_z - 4.0 * self.cube_he
        if cube_z < floor:
            raise RuntimeError(f"cube fell through cloth (z={cube_z:.4f} m, floor={floor:.4f} m)")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=12, help="Cloth resolution along x (quads)")
    parser.add_argument("--height", type=int, default=12, help="Cloth resolution along y (quads)")
    parser.add_argument(
        "--cloth-density",
        type=float,
        default=600.0,
        help="Triangle prism density [kg/m^3]; drives per-triangle mass via the prism volume.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        args,
        width=args.width,
        height=args.height,
        cloth_density=args.cloth_density,
    )
    newton.examples.run(example, args)
