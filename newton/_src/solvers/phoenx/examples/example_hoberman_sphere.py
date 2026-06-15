# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Hoberman Sphere
#
# PhoenX variant of the kamino articulation demo in
# ``example_sim_hoberman_sphere.py``: same Hoberman sphere USD
# (``models/hoberman_sphere_articulation.usda``, 240 rigid struts
# linked by revolute joints), same zero-gravity rigid-body spin,
# same authored configuration -- driven directly by
# :class:`PhoenXWorld` (no :class:`SolverPhoenX` adapter, no
# Newton state ping-pong inside the step) so the example doubles as
# a minimal walkthrough of the raw PhoenX API.
#
# Joint plumbing:
#   * The USD's revolute joints are translated into ADBS columns by
#     :func:`build_adbs_init_arrays` and uploaded once via
#     :meth:`PhoenXWorld.initialize_actuated_double_ball_socket_joints`.
#     With the joints active the cluster stays hinged when picked --
#     pulling on one strut now drags its neighbours along the chain
#     instead of breaking it free of the sphere.
#   * No collision pipeline -- the struts overlap by design in the
#     authored pose (they're meant to be hinged), so any contact
#     detection would explode the sphere on frame 1. PhoenX runs
#     with ``rigid_contact_max=0`` and ``world.step(contacts=None)``.
#   * No gravity -- matches the kamino demo's
#     ``set_gravity([0, 0, 0])`` so the cluster just spins.
#
# Each strut is seeded with a rigid-body spin about +Z so the whole
# sphere rotates as one cluster: ``v = omega x r``, ``omega = 0.1 z``.
# Matches the kamino demo's initial-velocity setup verbatim.
#
# Rendering: the USD authors every strut as its own :class:`newton.Mesh`
# prim (16 verts / 24 tris each, all bit-different vertex floats),
# which Newton's viewer hashes to 240 single-instance batches and
# 240 GL draws per frame -- enough to drag the framerate to ~12 fps
# even when paused. Following the kapla-tower fast path, we replace
# every USD mesh shape's source with a single shared :class:`Mesh`
# stub (1 invisible viewer batch instead of 240) and re-add each
# strut as two :func:`add_shape_box` tiles. Snap-to-grid quantising
# the tile half-extents collapses ~480 box shapes into a handful of
# unique sizes -- one viewer batch per size, just like kapla.
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_hoberman_sphere
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.model_adapter import build_adbs_init_arrays
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_phoenx import (
    PhoenXWorld,
    pack_body_xforms_kernel,
)

# Strut tile count: every Hoberman strut mesh is authored as two
# rotated rectangular tiles glued at the hinge end (8 verts each
# inside the 16-vert mesh). Rendering each as its own oriented box
# preserves the USD silhouette.
_TILES_PER_STRUT = 2

# Quantisation step (m) for tile half-extents. The USD authors
# nominally-equal tile sizes with sub-millimetre float-noise
# differences which would otherwise produce one viewer batch per
# tile. Snapping to a 1 mm grid collapses the ~480 tiles down to
# ~12 unique sizes (the actual authored variants) -> ~12 viewer
# batches, while keeping the rendered geometry within 1 mm of the
# joint-anchor placement -- well below the visible threshold for
# a strut spanning ~50 cm.
_TILE_SIZE_QUANTUM_M = 0.001

# Authored Hoberman tile cross-section (m). The USD ships every
# strut tile as a 5 cm-wide / 5 cm-thick slab; we hard-code the
# halves rather than measuring per-strut so that the rendered
# boxes share a single width-and-thickness pair across all 480
# tiles (collapses one more axis of viewer-batch variation).
_TILE_HALF_WIDTH_M = 0.025
_TILE_HALF_THICKNESS_M = 0.025

# Path to the Hoberman sphere USDA -- shared with the kamino
# articulation demo.
USDA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "models",
    "hoberman_sphere_articulation.usda",
)

# Initial rigid-body spin applied to the whole cluster [rad/s].
# Matches the kamino demo: 0.1 rad/s about +Z, which rotates the full
# sphere about once every ~63 s.
SPIN_RATE_RAD_S: float = 0.1


class Example:
    """Hoberman sphere rotating in zero-g, driven by :class:`PhoenXWorld`.

    Pipeline per frame:
        1. Sync Newton state -> PhoenX body container.
        2. Call :meth:`PhoenXWorld.step` with ``contacts=None``.
        3. Sync PhoenX body container -> Newton state.

    The USD's revolute joints are ingested through the ADBS
    constraint columns so the sphere stays articulated under
    picking. There are no contacts (``rigid_contact_max=0``) and no
    gravity, so once seeded with the rigid-body spin the cluster
    coasts forever.
    """

    def __init__(self, viewer, args):
        # Frame pacing. 50 Hz / 1 ms substep matches the kamino demo.
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        # 1 ms substep matches the kamino demo. With ~240 hinged
        # struts a few PGS iterations per substep keep the chain
        # rigid under picking forces; 1 iteration was fine when the
        # joints were ignored but lets the chain stretch visibly now.
        self.sim_substeps = 4
        self.sim_time = 0.0
        self.solver_iterations = 4

        self.viewer = viewer
        self.device = wp.get_device()

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder()

        # USD loader options match the kamino demo verbatim so the two
        # examples build out of the same Newton model. The articulation
        # comes along for the ride (the USD has revolute joints) but
        # PhoenX never touches it.
        builder.add_usd(
            USDA_PATH,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            collapse_fixed_joints=False,
            apply_up_axis_from_stage=True,
        )

        # Replace USD-imported per-strut Mesh shapes with shared
        # box-shaped tiles -- see :meth:`_swap_struts_for_box_tiles`
        # for the rationale (kapla-style viewer batching).
        self._swap_struts_for_box_tiles(builder)

        builder.color()
        self.model = builder.finalize(skip_validation_joints=True)
        body_inv_mass_np = self.model.body_inv_mass.numpy()
        self._strut_body_ids = [int(i) for i in range(self.model.body_count) if body_inv_mass_np[i] > 0.0]
        print(
            f"[PhoenX Hoberman] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"struts={len(self._strut_body_ids)}"
        )

        # State: ``add_usd`` populates ``model.body_q`` with each
        # body's absolute world transform. We skip ``newton.eval_fk``
        # because it would walk the USD's revolute-joint chain and
        # collapse body_q to the joint-derived pose -- we want the
        # authored absolute positions.
        self.state = self.model.state()
        self.state.body_q.assign(self.model.body_q)

        # Seed every body with a rigid-body spin about +Z so the
        # cluster rotates as one rigid body: ``v = omega x r``.
        body_q_np = self.model.body_q.numpy()
        ang_vel = np.array([0.0, 0.0, SPIN_RATE_RAD_S], dtype=np.float32)
        body_qd_np = np.zeros((self.model.body_count, 6), dtype=np.float32)
        for i in range(self.model.body_count):
            pos = body_q_np[i, 0:3]
            body_qd_np[i, 0:3] = np.cross(ang_vel, pos)
            body_qd_np[i, 3:6] = ang_vel
        self.state.body_qd.assign(body_qd_np)

        # ---- PhoenX body container (slot 0 = static world anchor) ----
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Seed every slot's orientation to identity so the rotation-
        # to-matrix call in the inertia-refresh kernel doesn't blow up
        # on the zero-quaternion default.
        bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32))
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.state.body_qd,
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

        # ---- Joint translation ---------------------------------------
        # ``build_adbs_init_arrays`` walks ``model.joint_*`` and packs
        # one actuated-double-ball-socket descriptor per joint. The
        # USD ships ~240 revolute hinges; we keep them all and let
        # PhoenX hold them rigid (drive_mode=OFF, no PD gains, no
        # axial limit) so the sphere behaves as one articulated
        # body when picked.
        self._adbs = build_adbs_init_arrays(self.model, device=self.device)
        num_joints = self._adbs.num_joint_columns

        # Constraint container sized for every USD joint plus zero
        # contact / cloth / soft-tet capacity.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            device=self.device,
        )

        # ---- Solver ---------------------------------------------------
        # ``rigid_contact_max=0`` disables PhoenX's contact-column
        # pipeline; ``gravity=(0, 0, 0)`` matches the kamino demo's
        # ``set_gravity([0, 0, 0])``. With the joints active, picking
        # a single strut drags the rest of the chain along -- the
        # cluster stays hinged instead of breaking apart.
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,
            gravity=(0.0, 0.0, 0.0),
            rigid_contact_max=0,
            num_joints=num_joints,
            device=self.device,
        )

        # The ADBS init kernel reads PhoenX body positions to
        # snapshot body-local anchor offsets, so it must run *after*
        # the body container has been seeded above.
        if num_joints > 0:
            self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())

        # ---- Viewer ---------------------------------------------------
        self._xforms = wp.zeros(num_phoenx_bodies, dtype=wp.transform, device=self.device)
        self.viewer.set_model(self.model)
        # Authored sphere radius is ~2 m; pull the camera back to ~3x
        # so the whole cluster stays in frame.
        self.viewer.set_camera(
            pos=wp.vec3(6.0, -6.0, 2.0),
            pitch=-15.0,
            yaw=135.0,
        )

        # ---- Picking --------------------------------------------------
        # Uniform conservative AABB per strut (~0.6 m cube). Slot 0
        # (static world anchor) stays at zero so rays ignore it.
        half_extents_np = np.zeros((num_phoenx_bodies, 3), dtype=np.float32)
        for newton_idx in self._strut_body_ids:
            half_extents_np[newton_idx + 1] = (0.3, 0.3, 0.3)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # CUDA graph capture for the per-frame step pipeline. Falls
        # back to direct :meth:`simulate` on CPU.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.picking.apply_force()
        self.world.step(dt=self.frame_dt)
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            phoenx_to_newton_kernel,
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

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def _swap_struts_for_box_tiles(self, builder: newton.ModelBuilder) -> None:
        """Replace per-strut Mesh shapes with shared box tiles.

        Mirrors the kapla-tower viewer fast path: every visible
        shape is a :class:`newton.GeoType.BOX` whose batch key is
        ``(GeoType.BOX, (hx, hy, hz), ...)``. Tiles with the same
        quantised half-extents fall into the same batch and the
        whole sphere renders in a handful of GL draws.

        Tile placement is derived directly from the USD's revolute
        joints -- no PCA, no OBB fit. Every Hoberman strut body
        carries exactly three revolute hinges in its body frame:
        one in the middle (the strut-strut hinge between the two
        tiles) and one at each end (where the strut hinges to a
        neighbour). All three hinges share the same axis -- the
        pin direction the strut rotates about -- and that axis is
        the strut tile's thickness axis (the mesh is a thin slab
        normal to it). The middle anchor is the one closest to the
        midpoint of the other two; each tile then spans from the
        middle anchor to its end anchor in the strut plane. Width
        and thickness are the USD's authored 5 cm.

        For every USD-imported strut shape we
            * place two ``add_shape_box`` tiles whose length, centre
              and orientation come straight from the joint anchor
              triple plus the hinge axis, and whose width and
              thickness are ``_TILE_HALF_WIDTH_M`` and
              ``_TILE_HALF_THICKNESS_M``,
            * snap the half-extents to ``_TILE_SIZE_QUANTUM_M`` so
              float-noise length differences collapse,
            * neutralise the original mesh shape: source -> ``None``,
              type -> zero-extent ``BOX``, ``VISIBLE`` /
              ``COLLIDE_SHAPES`` flags cleared. All 240 stripped
              shapes then share the same viewer batch key and
              ``builder.finalize`` skips per-mesh BVH allocation
              (a degenerate ``Mesh`` placeholder would crash the
              Warp BVH builder with a CUDA OOM-shaped error).
        """
        n_shapes = len(builder.shape_source)
        if n_shapes == 0:
            return

        visible_bit = int(newton.ShapeFlags.VISIBLE)
        collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)

        # Mass-less ShapeConfig keeps the new tile boxes from
        # rewriting the USD-authored body inertia: ``add_shape``
        # only calls ``_update_body_mass`` when ``cfg.density > 0``.
        massless_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)

        # Body-local revolute joint info: anchor position (3D, in
        # body frame) + hinge axis (3D, in body frame). The hinge
        # axis is the joint's local +x rotated by the joint frame
        # quaternion. Across the three revolute joints attached to
        # one strut body all three axes are parallel (the pin
        # direction); we average them later for numerical stability.
        qd_starts = [*builder.joint_qd_start, builder.joint_dof_count]
        body_joints: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(builder.body_count)]
        for j in range(len(builder.joint_parent)):
            if builder.joint_type[j] != newton.JointType.REVOLUTE:
                continue
            p = builder.joint_parent[j]
            c = builder.joint_child[j]
            axis_local = np.array(list(builder.joint_axis[qd_starts[j]]), dtype=np.float64)
            if 0 <= p < builder.body_count:
                xf = builder.joint_X_p[j]
                anchor = np.array(list(xf.p), dtype=np.float64)
                axis_body = Example._quat_rotate_xyzw(np.array(list(xf.q), dtype=np.float64), axis_local)
                body_joints[p].append((anchor, axis_body))
            if 0 <= c < builder.body_count:
                xf = builder.joint_X_c[j]
                anchor = np.array(list(xf.p), dtype=np.float64)
                axis_body = Example._quat_rotate_xyzw(np.array(list(xf.q), dtype=np.float64), axis_local)
                body_joints[c].append((anchor, axis_body))

        # Snapshot the original strut shape data before we mutate
        # ``builder.shape_*``; ``add_shape_box`` below appends to the
        # same lists and would otherwise feed the new entries back
        # into the loop.
        strut_data: list[tuple[int, np.ndarray]] = []
        for shape_idx in range(n_shapes):
            body_idx = builder.shape_body[shape_idx]
            mesh_src = builder.shape_source[shape_idx]
            if body_idx < 0 or mesh_src is None or not hasattr(mesh_src, "vertices"):
                continue
            verts = np.asarray(mesh_src.vertices, dtype=np.float32)
            if verts.shape[0] != _TILES_PER_STRUT * 8:
                continue
            if len(body_joints[body_idx]) != 3:
                continue
            color = (
                np.asarray(builder.shape_color[shape_idx], dtype=np.float32)
                if shape_idx < len(builder.shape_color)
                else np.array([0.5, 0.5, 0.5], dtype=np.float32)
            )
            strut_data.append((int(body_idx), color))
            # Strip the original mesh shape from the viewer / collision
            # pipelines without removing it from the ~25 parallel
            # ``builder.shape_*`` lists. Setting source=None tells
            # ``builder.finalize`` to skip the per-mesh ``wp.Mesh``
            # BVH allocation (line 10213: ``if geo and not Heightfield``)
            # which is critical -- a degenerate ``Mesh()`` placeholder
            # would otherwise trigger an OOM-shaped CUDA error inside
            # the BVH builder. Switching the type to a zero-extent BOX
            # gives every stripped shape the same ``(BOX, None, 0,0,0)``
            # batch key -> one cheap hidden viewer batch instead of 240.
            builder.shape_source[shape_idx] = None
            builder.shape_type[shape_idx] = int(newton.GeoType.BOX)
            builder.shape_scale[shape_idx] = (0.0, 0.0, 0.0)
            builder.shape_flags[shape_idx] = builder.shape_flags[shape_idx] & ~visible_bit & ~collide_bit

        # Add two box tiles per strut from the joint anchor triple.
        for body_idx, color in strut_data:
            tiles = self._tiles_from_anchors(body_joints[body_idx])
            for centre, quat_xyzw, half_extents in tiles:
                snapped = np.maximum(
                    _TILE_SIZE_QUANTUM_M,
                    np.round(half_extents / _TILE_SIZE_QUANTUM_M) * _TILE_SIZE_QUANTUM_M,
                )
                tile_xform = wp.transform(
                    p=wp.vec3(float(centre[0]), float(centre[1]), float(centre[2])),
                    q=wp.quat(
                        float(quat_xyzw[0]),
                        float(quat_xyzw[1]),
                        float(quat_xyzw[2]),
                        float(quat_xyzw[3]),
                    ),
                )
                builder.add_shape_box(
                    body=body_idx,
                    xform=tile_xform,
                    hx=float(snapped[0]),
                    hy=float(snapped[1]),
                    hz=float(snapped[2]),
                    cfg=massless_cfg,
                    color=tuple(float(c) for c in color),
                )

    @staticmethod
    def _tiles_from_anchors(
        joints: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Derive both tile OBBs of a strut from its three revolute
        joints in body frame.

        Each tuple in ``joints`` is ``(anchor_pos, hinge_axis)`` --
        the joint frame's body-local origin and the hinge direction
        (joint local +x rotated by the joint frame quaternion). All
        three hinges of a single strut share the same axis (the pin
        the strut spins around); we average them to suppress the
        sub-degree numerical scatter that comes from float-noise
        joint-frame quaternions.

        That averaged hinge axis is the strut tile's **thickness
        axis** -- the mesh slab is normal to it. Anchor positions
        are projected onto the plane perpendicular to the hinge to
        recover the in-plane tile geometry: the middle anchor
        (closest to the midpoint of the other two within that
        plane) is the strut-strut hinge, the other two are the
        tile-end hinges. Each tile spans middle -> end along the
        in-plane length axis, with width
        ``_TILE_HALF_WIDTH_M`` perpendicular in the strut plane.

        Args:
            joints: Three ``(anchor_pos, hinge_axis)`` tuples in
                body frame.

        Returns:
            ``[(centre, quat_xyzw, half_extents), ...]`` -- one
            entry per tile. ``centre`` is the body-local 3D OBB
            centre, ``quat_xyzw`` rotates the unit-cube axes onto
            (length, width, hinge), ``half_extents`` is
            ``(half_length, half_width, half_thickness)``.
        """
        # Average the three hinge axes (flipping antiparallel ones
        # back into agreement with the first axis so the mean is
        # not a near-zero vector).
        axis_sum = joints[0][1].copy()
        for _, a in joints[1:]:
            axis_sum = axis_sum + (a if axis_sum @ a >= 0.0 else -a)
        hinge_axis = axis_sum / (np.linalg.norm(axis_sum) + 1e-12)

        # Build an in-plane orthonormal basis (u, v) perpendicular
        # to hinge_axis. Pick the world-axis least aligned with
        # hinge_axis as the seed to keep the cross product well
        # conditioned.
        seed_idx = int(np.argmin(np.abs(hinge_axis)))
        seed = np.zeros(3, dtype=np.float64)
        seed[seed_idx] = 1.0
        u = np.cross(hinge_axis, seed)
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(hinge_axis, u)

        # Project each anchor onto the (u, v) plane through origin.
        # The plane offset along hinge_axis (the slab top/bottom
        # face) is constant per body and irrelevant to the tile
        # in-plane geometry; we restore it via ``hinge_offset`` so
        # the tile centre lies on the slab midplane.
        anchors_3d = np.stack([j[0] for j in joints])
        hinge_offsets = anchors_3d @ hinge_axis
        plane_coords = np.stack([anchors_3d @ u, anchors_3d @ v], axis=1)

        # Pick the middle hinge: minimise distance to the midpoint
        # of the other two within the in-plane (u, v) frame.
        order = (
            (0, 1, 2),
            (1, 0, 2),
            (2, 0, 1),
        )
        i_mid, i_a, i_b = min(
            order,
            key=lambda t: float(np.linalg.norm(plane_coords[t[0]] - 0.5 * (plane_coords[t[1]] + plane_coords[t[2]]))),
        )
        middle_uv = plane_coords[i_mid]
        # The strut midplane sits halfway between the two slab
        # faces. Anchors of one body land on one face (their hinge
        # offsets are equal); the slab centre is one half-thickness
        # *toward* the body interior, which is whichever side has
        # less mass. With the USD authoring all three anchors on
        # the same face we just shift by -sign * half_thickness;
        # numerically the offset alternates ±half_thickness per
        # body and this keeps the slab on the right side.
        slab_face_offset = float(hinge_offsets[i_mid])
        slab_centre_offset = slab_face_offset - np.sign(slab_face_offset or 1.0) * _TILE_HALF_THICKNESS_M

        tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i_end in (i_a, i_b):
            end_uv = plane_coords[i_end]
            length_vec_uv = end_uv - middle_uv
            anchor_distance = float(np.linalg.norm(length_vec_uv))
            if anchor_distance < 1e-6:
                continue
            length_dir_uv = length_vec_uv / anchor_distance
            length_axis = length_dir_uv[0] * u + length_dir_uv[1] * v
            width_axis = np.cross(hinge_axis, length_axis)
            width_axis /= np.linalg.norm(width_axis) + 1e-12

            # Tile is flush with the middle anchor (short edge at
            # L=0) and extends one half-width past the end anchor
            # (the hinge pin sits one half-width inset from the
            # rounded tip). Matches the USD mesh extents to within
            # 0.5 mm.
            half_length = 0.5 * (anchor_distance + _TILE_HALF_WIDTH_M)
            centre_uv = middle_uv + half_length * length_dir_uv
            centre = centre_uv[0] * u + centre_uv[1] * v + slab_centre_offset * hinge_axis

            axes = np.column_stack((length_axis, width_axis, hinge_axis))
            half_extents = np.array(
                [half_length, _TILE_HALF_WIDTH_M, _TILE_HALF_THICKNESS_M],
                dtype=np.float64,
            )
            quat_xyzw = Example._rotation_matrix_to_quat_xyzw(axes)
            tiles.append(
                (
                    centre.astype(np.float32),
                    quat_xyzw.astype(np.float32),
                    half_extents.astype(np.float32),
                )
            )
        return tiles

    @staticmethod
    def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector ``v`` by quaternion ``q = (x, y, z, w)``.

        Uses ``v + 2 * cross(q_xyz, cross(q_xyz, v) + w*v)`` -- the
        standard branch-free form, no trig.
        """
        qv = np.array([q[0], q[1], q[2]], dtype=np.float64)
        return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[3] * v)

    @staticmethod
    def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
        """Convert a 3x3 right-handed rotation matrix to (x, y, z, w).

        Uses the numerically stable branch picked by the largest of
        ``trace``, ``R[0,0]``, ``R[1,1]``, ``R[2,2]``.
        """
        m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif m00 > m11 and m00 > m22:
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float32)

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Post-step validation
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """Every strut must have finite state and stay within a
        generous envelope around the authored sphere centre.

        With no contacts and no gravity, the only motion is the
        seeded ``omega x r`` rigid-body spin. Joint constraints
        keep the cluster hinged (no separation), so the bounding
        sphere is invariant -- any drift past a wide tolerance
        indicates an integrator blow-up.
        """
        positions = self.bodies.position.numpy()
        velocities = self.bodies.velocity.numpy()
        radius_tolerance = 8.0
        for newton_idx in self._strut_body_ids:
            slot = newton_idx + 1
            pos = positions[slot]
            vel = velocities[slot]
            assert np.isfinite(pos).all(), f"strut {newton_idx} pos non-finite ({pos})"
            assert np.isfinite(vel).all(), f"strut {newton_idx} vel non-finite ({vel})"
            r = float(np.linalg.norm(pos))
            assert r < radius_tolerance, (
                f"strut {newton_idx} flew off the sphere: r={r:.3f} m, tol={radius_tolerance:.3f}, pos={pos}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Start paused so the viewer shows the authored sphere
    # configuration before the spin starts (matches the kamino demo).
    example.viewer._paused = True
    newton.examples.run(example, args)
