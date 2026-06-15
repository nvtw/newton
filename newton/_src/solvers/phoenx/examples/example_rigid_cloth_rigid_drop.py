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
by ADBS ball joints.

The scene is hand-rolled directly against :class:`PhoenXWorld` (no
:class:`SolverPhoenX` adapter), mirroring the structure of
:mod:`example_cloth_rigid_drop`:

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
* :func:`build_adbs_init_arrays` translates the Newton joint graph into
  ADBS columns; PhoenX advances the rigid bodies, contact pipeline, and
  joints in a single PGS sweep.

Run::

    python -m newton._src.solvers.phoenx.examples.example_rigid_cloth_rigid_drop
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.model_adapter import build_adbs_init_arrays
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

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
    ):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Articulated cloth + cube benefits from a similar substep
        # budget to the deformable variant: many ball-joint loops plus
        # rigid contact resolution share the PGS sweep.
        self.sim_substeps = 8
        self.solver_iterations = 12
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
        builder = newton.ModelBuilder()

        self._build_rigid_cloth(builder)

        # Free-floating rigid cube spawned above the cloth centre.
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
        builder.gravity = -9.81

        self.model = builder.finalize(device=self.device)

        # ``add_body`` queues each free-joint pose in ``joint_q``; FK
        # propagates them into ``body_q`` so the PhoenX init kernel
        # seeds every triangle / the cube at its spawn pose instead of
        # the URDF-rest origin.
        if int(self.model.body_count) > 0 and int(self.model.joint_count) > 0:
            tmp_state = self.model.state()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, tmp_state)
            self.model.body_q.assign(tmp_state.body_q)
            self.model.body_qd.assign(tmp_state.body_qd)

        # ---- Newton collision pipeline -------------------------------
        # Pure rigid-body scene -> standard ``newton.CollisionPipeline``;
        # no cloth-aware suffix, no virtual cloth-triangle shapes. NXN
        # is fine for the modest body count of these grids.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="nxn",
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        # ---- Build the PhoenX body container -------------------------
        # Slot 0 is the static world anchor; Newton bodies occupy
        # slots ``[1, body_count + 1)``.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.model.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=self.device,
        )
        self.bodies = bodies

        # ---- Translate Newton joints into ADBS columns ---------------
        # Each ball-socket joint (corner chain links + four world pins)
        # becomes a single ADBS constraint column. ``add_body`` also
        # creates a per-body FREE joint to satisfy ``finalize``'s
        # reachability check; the model adapter's joint translator
        # ignores those.
        self._adbs = build_adbs_init_arrays(self.model, device=self.device)
        num_joints = self._adbs.num_joint_columns

        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            device=self.device,
        )

        # Newton's ``shape_body`` uses -1 for the world anchor; PhoenX
        # uses slot 0. Shift accordingly so contact ingest finds the
        # right body slot per shape.
        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # ---- Build the PhoenXWorld -----------------------------------
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            num_joints=num_joints,
            num_worlds=1,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout="single_world",
            device=self.device,
        )

        # Joint columns must be initialised after the body container is
        # seeded (the ADBS init kernel reads PhoenX body positions to
        # snapshot the body-local anchor offsets).
        if num_joints > 0:
            self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())

        # ---- Newton state for the renderer ---------------------------
        # We mirror PhoenX body state -> ``state.body_q`` / ``body_qd``
        # once per frame; a fresh ``model.collide`` pass refreshes the
        # contact buffer before each PhoenX step.
        self.state = self.model.state()

        # Snapshot the rigid-cloth pinned-corner triangle slots so we
        # can assert they didn't drift in ``test_final``.
        self._initial_body_q = self.model.body_q.numpy().copy()

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

    def _sync_phoenx_to_newton(self) -> None:
        """Mirror PhoenX body state into the Newton ``State``.

        The state is what the viewer renders and what
        ``model.collide`` reads from when it refreshes the contact
        buffer. ``_phoenx_to_newton_kernel`` reconstructs Newton's
        body-origin transform convention from the COM-centred PhoenX
        layout.
        """
        n = self.model.body_count
        if n == 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def _simulate_one_frame(self) -> None:
        """One physics frame: mirror state, refresh contacts, then step.

        Order matters:

        1. Mirror PhoenX bodies into ``state.body_q`` / ``body_qd`` so
           the collision pipeline reads the freshest pose.
        2. Run ``model.collide`` to populate the rigid-body contact
           buffer for this frame.
        3. ``world.step`` advances the bodies + joints, ingesting the
           refreshed contacts. The next frame's mirror picks up the
           result.
        """
        self._sync_phoenx_to_newton()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )

    def _capture(self) -> None:
        """Capture per-frame ``simulate`` into a CUDA graph."""
        if self.device.is_cuda:
            self._simulate_one_frame()  # warm-up
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

    def _final_state(self) -> None:
        """Mirror PhoenX state into Newton state once for assertions /
        rendering after a captured-graph step. The graph captures the
        sync inside ``_simulate_one_frame`` so this is a redundant
        no-op except on the very first frame; we run it
        unconditionally for clarity."""
        self._sync_phoenx_to_newton()

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
        self._final_state()
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
        self._final_state()
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
