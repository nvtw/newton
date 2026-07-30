# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Hoberman Sphere
#
# A 240-strut, full-coordinate looped mechanism loaded from the Hoberman
# sphere USD. SolverPhoenX automatically discovers its connected revolute
# graph, precomputes one RCM-ordered sparse direct system, and leaves no
# bilateral rows in PGS. The authored overlapping visual struts remain
# non-colliding, and zero gravity lets the sphere coast with a rigid-body spin.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_hoberman_sphere
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples

_TILES_PER_STRUT = 2
_TILE_SIZE_QUANTUM_M = 0.001
_TILE_HALF_WIDTH_M = 0.025
_TILE_HALF_THICKNESS_M = 0.025

USDA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "models",
    "hoberman_sphere_articulation.usda",
)
SPIN_RATE_RAD_S = 0.1


class Example:
    """Simulate a looped Hoberman mechanism with direct equalities."""

    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
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
        self._swap_struts_for_box_tiles(builder)
        builder.color()
        self.model = builder.finalize(skip_validation_joints=True)

        self.state = self.model.state()
        self.state.body_q.assign(self.model.body_q)
        body_q = self.model.body_q.numpy()
        angular_velocity = np.array([0.0, 0.0, SPIN_RATE_RAD_S], dtype=np.float32)
        body_qd = np.zeros((self.model.body_count, 6), dtype=np.float32)
        body_qd[:, :3] = np.cross(angular_velocity, body_q[:, :3])
        body_qd[:, 3:] = angular_velocity
        self.state.body_qd.assign(body_qd)
        self.control = self.model.control()

        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=5,
            solver_iterations=2,
            velocity_iterations=1,
            articulation_mode="maximal",
        )
        direct = self.solver._direct_equality_system
        if direct is None or not direct.enabled:
            raise RuntimeError("Hoberman joints were not assigned to the direct equality solver")
        if len(direct.topology.dimensions) != 1:
            raise RuntimeError(f"expected one connected Hoberman mechanism, got {direct.topology.dimensions}")
        if not self.solver.world._joint_pgs_all_disabled:
            raise RuntimeError("bilateral Hoberman rows unexpectedly remain in PGS")
        print(
            f"[PhoenX Hoberman] bodies={self.model.body_count} "
            f"joints={self.model.joint_count} direct_rows={direct.topology.dimensions[0]}"
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(6.0, -6.0, 2.0), pitch=-15.0, yaw=135.0)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        """Advance one rendered frame."""
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.solver.step(self.state, self.state, self.control, None, self.frame_dt)

    def step(self) -> None:
        """Advance the captured or eager simulation."""
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
        massless_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )

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
        """Render the current Newton body state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify the looped direct mechanism remains finite and bounded."""
        body_q = self.state.body_q.numpy()
        body_qd = self.state.body_qd.numpy()
        assert np.isfinite(body_q).all(), "Hoberman sphere produced non-finite poses"
        assert np.isfinite(body_qd).all(), "Hoberman sphere produced non-finite velocities"

        maximum_radius = float(np.linalg.norm(body_q[:, :3], axis=1).max())
        print(f"[direct_hoberman] maximum_radius={maximum_radius:.4f} m")
        assert maximum_radius < 8.0, f"Hoberman sphere escaped its stability envelope: {maximum_radius:.3f} m"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
