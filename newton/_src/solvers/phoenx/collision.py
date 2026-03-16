# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone collision pipeline for the PhoenX solver.

Uses Newton's :class:`~newton.geometry.BroadPhaseAllPairs` and
:class:`~newton._src.geometry.narrow_phase.NarrowPhase` without
requiring a full :class:`~newton.sim.Model` / :class:`~newton.sim.State`.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.narrow_phase import NarrowPhase
from newton.geometry import BroadPhaseAllPairs

from .schemas import BODY_FLAG_STATIC

GEO_TYPE_PLANE = 1
GEO_TYPE_SPHERE = 3
GEO_TYPE_CAPSULE = 4
GEO_TYPE_CYLINDER = 6
GEO_TYPE_BOX = 7
GEO_TYPE_MESH = 8

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _update_shape_transforms_kernel(
    shape_body_row: wp.array(dtype=wp.int32),
    shape_local_transform: wp.array(dtype=wp.transform),
    body_position: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    body_flags: wp.array(dtype=wp.int32),
    shape_transform_out: wp.array(dtype=wp.transform),
    count: wp.array(dtype=wp.int32),
):
    """Compose body pose with shape-local transform to get world-space shape transform."""
    tid = wp.tid()
    if tid >= count[0]:
        return

    br = shape_body_row[tid]
    local_xf = shape_local_transform[tid]

    if br < 0 or (body_flags[br] & BODY_FLAG_STATIC) != 0:
        shape_transform_out[tid] = local_xf
        return

    bp = body_position[br]
    bq = body_orientation[br]
    local_p = wp.transform_get_translation(local_xf)
    local_q = wp.transform_get_rotation(local_xf)
    world_p = bp + wp.quat_rotate(bq, local_p)
    world_q = bq * local_q
    shape_transform_out[tid] = wp.transform(world_p, world_q)


@wp.kernel
def _compute_shape_aabbs_kernel(
    shape_type: wp.array(dtype=wp.int32),
    shape_data: wp.array(dtype=wp.vec4),
    shape_transform: wp.array(dtype=wp.transform),
    shape_collision_aabb_lower: wp.array(dtype=wp.vec3),
    shape_collision_aabb_upper: wp.array(dtype=wp.vec3),
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
):
    """Compute world-space AABBs from shape transform and geometry data."""
    tid = wp.tid()
    if tid >= count[0]:
        return

    xf = shape_transform[tid]
    pos = wp.transform_get_translation(xf)
    q = wp.transform_get_rotation(xf)
    data = shape_data[tid]
    geo = shape_type[tid]

    if geo == 7:  # BOX
        hx = data[0]
        hy = data[1]
        hz = data[2]
        margin = data[3]

        r = wp.quat_to_matrix(q)
        ex = wp.abs(r[0, 0]) * hx + wp.abs(r[0, 1]) * hy + wp.abs(r[0, 2]) * hz + margin
        ey = wp.abs(r[1, 0]) * hx + wp.abs(r[1, 1]) * hy + wp.abs(r[1, 2]) * hz + margin
        ez = wp.abs(r[2, 0]) * hx + wp.abs(r[2, 1]) * hy + wp.abs(r[2, 2]) * hz + margin

        aabb_lower[tid] = pos - wp.vec3(ex, ey, ez)
        aabb_upper[tid] = pos + wp.vec3(ex, ey, ez)
    elif geo == 3:  # SPHERE
        radius = data[0] + data[3]
        aabb_lower[tid] = pos - wp.vec3(radius, radius, radius)
        aabb_upper[tid] = pos + wp.vec3(radius, radius, radius)
    elif geo == 4:  # CAPSULE: data = (radius, half_length, 0, margin)
        radius = data[0] + data[3]
        half_len = data[1]
        # Capsule axis is local Z
        r = wp.quat_to_matrix(q)
        ax = r[0, 2] * half_len
        ay = r[1, 2] * half_len
        az = r[2, 2] * half_len
        ex = wp.abs(ax) + radius
        ey = wp.abs(ay) + radius
        ez = wp.abs(az) + radius
        aabb_lower[tid] = pos - wp.vec3(ex, ey, ez)
        aabb_upper[tid] = pos + wp.vec3(ex, ey, ez)
    elif geo == 6:  # CYLINDER: data = (radius, half_height, 0, margin)
        radius = data[0] + data[3]
        half_h = data[1] + data[3]
        # Cylinder axis is local Z
        r = wp.quat_to_matrix(q)
        # Conservative AABB: max of (rotated cylinder endpoint, rotated disc radius)
        ex = wp.abs(r[0, 2]) * half_h + wp.sqrt(r[0, 0] * r[0, 0] + r[0, 1] * r[0, 1]) * radius
        ey = wp.abs(r[1, 2]) * half_h + wp.sqrt(r[1, 0] * r[1, 0] + r[1, 1] * r[1, 1]) * radius
        ez = wp.abs(r[2, 2]) * half_h + wp.sqrt(r[2, 0] * r[2, 0] + r[2, 1] * r[2, 1]) * radius
        aabb_lower[tid] = pos - wp.vec3(ex, ey, ez)
        aabb_upper[tid] = pos + wp.vec3(ex, ey, ez)
    elif geo == 8:  # MESH
        # Transform pre-computed local-space AABB to world space
        local_lo = shape_collision_aabb_lower[tid]
        local_hi = shape_collision_aabb_upper[tid]
        r = wp.quat_to_matrix(q)
        # Compute rotated AABB extents
        half = (local_hi - local_lo) * 0.5
        center_local = (local_lo + local_hi) * 0.5
        center_world = pos + wp.quat_rotate(q, center_local)
        ex = wp.abs(r[0, 0]) * half[0] + wp.abs(r[0, 1]) * half[1] + wp.abs(r[0, 2]) * half[2]
        ey = wp.abs(r[1, 0]) * half[0] + wp.abs(r[1, 1]) * half[1] + wp.abs(r[1, 2]) * half[2]
        ez = wp.abs(r[2, 0]) * half[0] + wp.abs(r[2, 1]) * half[1] + wp.abs(r[2, 2]) * half[2]
        aabb_lower[tid] = center_world - wp.vec3(ex, ey, ez)
        aabb_upper[tid] = center_world + wp.vec3(ex, ey, ez)
    else:
        # Planes and unknown types: large AABB
        big = float(1.0e6)
        aabb_lower[tid] = wp.vec3(-big, -big, -big)
        aabb_upper[tid] = wp.vec3(big, big, big)


@wp.kernel
def _convert_contacts_kernel(
    np_pair: wp.array(dtype=wp.vec2i),
    np_position: wp.array(dtype=wp.vec3),
    np_normal: wp.array(dtype=wp.vec3),
    np_penetration: wp.array(dtype=wp.float32),
    np_count: wp.array(dtype=wp.int32),
    shape_body_row: wp.array(dtype=wp.int32),
    body_position: wp.array(dtype=wp.vec3),
    body_orientation: wp.array(dtype=wp.quat),
    default_friction: float,
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_body0: wp.array(dtype=wp.int32),
    out_body1: wp.array(dtype=wp.int32),
    out_normal: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    out_acc_n: wp.array(dtype=wp.float32),
    out_acc_t1: wp.array(dtype=wp.float32),
    out_acc_t2: wp.array(dtype=wp.float32),
):
    """Convert NarrowPhase contacts into PhoenX contact-store columns.

    The narrow phase reports the midpoint between the two surface points.
    We reconstruct the per-shape surface points from the penetration
    (negative when overlapping) and the normal (A -> B) so that the PGS
    solver can compute a correct gap.
    """
    tid = wp.tid()
    if tid >= np_count[0]:
        return

    pair = np_pair[tid]
    s0 = pair[0]
    s1 = pair[1]
    b0 = shape_body_row[s0]
    b1 = shape_body_row[s1]

    midpoint = np_position[tid]
    n = np_normal[tid]
    pen = np_penetration[tid]

    surface_a = midpoint - 0.5 * pen * n
    surface_b = midpoint + 0.5 * pen * n

    p0 = body_position[b0]
    q0 = body_orientation[b0]
    p1 = body_position[b1]
    q1 = body_orientation[b1]

    offset0 = wp.quat_rotate_inv(q0, surface_a - p0)
    offset1 = wp.quat_rotate_inv(q1, surface_b - p1)

    out_shape0[tid] = s0
    out_shape1[tid] = s1
    out_body0[tid] = b0
    out_body1[tid] = b1
    out_normal[tid] = n
    out_offset0[tid] = offset0
    out_offset1[tid] = offset1
    # Newton convention: margin = radius_eff + shape_margin.
    # For box/plane primitives with no rounding, margins are 0.
    out_margin0[tid] = 0.0
    out_margin1[tid] = 0.0
    out_friction[tid] = default_friction
    out_acc_n[tid] = 0.0
    out_acc_t1[tid] = 0.0
    out_acc_t2[tid] = 0.0


# ---------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------


class PhoenXCollisionPipeline:
    """Standalone broad + narrow phase collision pipeline for PhoenX.

    Shapes are registered on the host side via :meth:`add_shape_box` /
    :meth:`add_shape_plane`, then :meth:`finalize` copies everything to the
    GPU and creates the broad-phase data structures. Each frame,
    :meth:`collide` reads body transforms from a :class:`SolverState`,
    runs broad phase, narrow phase, and writes contacts into the solver's
    :class:`DataStore`.

    Args:
        max_shapes: upper bound on the number of collision shapes.
        max_contacts: upper bound on the number of contacts per frame.
        device: Warp device string or object.
    """

    def __init__(
        self,
        max_shapes: int,
        max_contacts: int,
        device: wp.context.Device | str | None = None,
    ):
        self.device = wp.get_device(device)
        self.max_shapes = max_shapes
        self.max_contacts = max_contacts

        self._shape_type_list: list[int] = []
        self._shape_data_list: list[tuple[float, float, float, float]] = []
        self._shape_local_xf_list: list[tuple] = []
        self._shape_body_row_list: list[int] = []
        self._shape_collision_radius_list: list[float] = []
        self._shape_mesh_ids: dict[int, int] = {}  # shape index → wp.Mesh.id (uint64)
        self._shape_local_aabb: dict[int, tuple] = {}  # shape index → (lower, upper, voxel_res) tuples
        self._shape_gap_list: list[float] = []  # per-shape contact gap
        self._finalized = False

    # -- shape registration (host-side, before finalize) --------------------

    def add_shape_box(
        self,
        body_row: int,
        local_transform: tuple | None = None,
        half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5),
        margin: float = 0.0,
    ) -> int:
        """Register a box collision shape.

        Args:
            body_row: storage row in the solver's body store (from
                ``body_store.handle_to_index[handle]``).
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` shape-to-body
                transform. ``None`` for identity.
            half_extents: box half-extents ``(hx, hy, hz)`` [m].
            margin: collision margin [m].

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_BOX)
        hx, hy, hz = half_extents
        self._shape_data_list.append((hx, hy, hz, margin))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(math.sqrt(hx * hx + hy * hy + hz * hz))
        self._shape_gap_list.append(0.0)
        return idx

    def add_shape_sphere(
        self,
        body_row: int,
        local_transform: tuple | None = None,
        radius: float = 0.5,
        margin: float = 0.0,
    ) -> int:
        """Register a sphere collision shape.

        Args:
            body_row: storage row in the solver's body store.
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` or ``None``.
            radius: sphere radius [m].
            margin: collision margin [m].

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_SPHERE)
        self._shape_data_list.append((radius, 0.0, 0.0, margin))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(radius + margin)
        self._shape_gap_list.append(0.0)
        return idx

    def add_shape_capsule(
        self,
        body_row: int,
        local_transform: tuple | None = None,
        radius: float = 0.25,
        half_length: float = 0.5,
        margin: float = 0.0,
    ) -> int:
        """Register a capsule collision shape (axis along local Z).

        Args:
            body_row: storage row in the solver's body store.
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` or ``None``.
            radius: capsule radius [m].
            half_length: half-length of the cylindrical section [m].
            margin: collision margin [m].

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_CAPSULE)
        self._shape_data_list.append((radius, half_length, 0.0, margin))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(radius + half_length + margin)
        self._shape_gap_list.append(0.0)
        return idx

    def add_shape_cylinder(
        self,
        body_row: int,
        local_transform: tuple | None = None,
        radius: float = 0.25,
        half_height: float = 0.5,
        margin: float = 0.0,
    ) -> int:
        """Register a cylinder collision shape (axis along local Z).

        Args:
            body_row: storage row in the solver's body store.
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` or ``None``.
            radius: cylinder radius [m].
            half_height: half-height along local Z [m].
            margin: collision margin [m].

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_CYLINDER)
        self._shape_data_list.append((radius, half_height, 0.0, margin))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(math.sqrt(radius * radius + half_height * half_height) + margin)
        self._shape_gap_list.append(0.0)
        return idx

    def add_shape_plane(
        self,
        body_row: int,
        local_transform: tuple | None = None,
    ) -> int:
        """Register an infinite ground-plane shape.

        The narrow phase plane normal is local +Z, matching Newton's
        Z-up convention.  ``None`` gives an identity transform (ground
        at the origin with normal pointing up).

        Args:
            body_row: storage row of the owning body.
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` or ``None``
                for identity.

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_PLANE)
        self._shape_data_list.append((0.0, 0.0, 0.0, 0.0))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(1.0e6)
        self._shape_gap_list.append(0.0)
        return idx

    def add_shape_mesh(
        self,
        body_row: int,
        mesh_id: int,
        local_transform: tuple | None = None,
        collision_radius: float = 1.0,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        margin: float = 0.0,
        gap: float = 0.0,
        aabb_lower: tuple[float, float, float] = (0.0, 0.0, 0.0),
        aabb_upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
        voxel_resolution: tuple[int, int, int] = (4, 4, 4),
    ) -> int:
        """Register a triangle mesh collision shape.

        Args:
            body_row: storage row in the solver's body store.
            mesh_id: ``wp.Mesh.id`` (uint64) for the mesh BVH.
            local_transform: ``(px, py, pz, qx, qy, qz, qw)`` or ``None``.
            collision_radius: bounding sphere radius [m] for broad phase.
            scale: mesh scale ``(sx, sy, sz)``.
            margin: collision margin [m].
            gap: contact detection gap [m].
            aabb_lower: local-space AABB lower bound [m].
            aabb_upper: local-space AABB upper bound [m].
            voxel_resolution: voxel grid resolution for contact binning.

        Returns:
            Shape index.
        """
        idx = len(self._shape_type_list)
        self._shape_type_list.append(GEO_TYPE_MESH)
        # shape_data stores (scale_x, scale_y, scale_z, margin) for narrow phase
        self._shape_data_list.append((scale[0], scale[1], scale[2], margin))
        if local_transform is None:
            local_transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._shape_local_xf_list.append(local_transform)
        self._shape_body_row_list.append(body_row)
        self._shape_collision_radius_list.append(collision_radius)
        self._shape_gap_list.append(gap)
        self._shape_mesh_ids[idx] = mesh_id
        self._shape_local_aabb[idx] = (aabb_lower, aabb_upper, voxel_resolution)
        return idx

    # -- finalize -----------------------------------------------------------

    def finalize(self):
        """Copy host-side shape data to GPU and create broad-phase structures."""
        d = self.device
        n = len(self._shape_type_list)

        self.shape_count_val = n
        self.shape_count = wp.array([n], dtype=wp.int32, device=d)

        self.shape_type = wp.array(self._shape_type_list, dtype=wp.int32, device=d)
        self.shape_data = wp.array(
            [wp.vec4(*v) for v in self._shape_data_list],
            dtype=wp.vec4,
            device=d,
        )

        xf_np = []
        for xf in self._shape_local_xf_list:
            px, py, pz, qx, qy, qz, qw = xf
            xf_np.append(wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw)))
        self.shape_local_transform = wp.array(xf_np, dtype=wp.transform, device=d)

        self.shape_body_row = wp.array(self._shape_body_row_list, dtype=wp.int32, device=d)

        # Populate shape_source with mesh IDs (0 for non-mesh shapes)
        source_list = [0] * n
        for idx, mesh_id in self._shape_mesh_ids.items():
            source_list[idx] = mesh_id
        self.shape_source = wp.array(source_list, dtype=wp.uint64, device=d)

        self.shape_gap = wp.array(self._shape_gap_list, dtype=wp.float32, device=d)
        self.shape_collision_radius = wp.array(self._shape_collision_radius_list, dtype=wp.float32, device=d)
        flags_val = int(ShapeFlags.COLLIDE_SHAPES)
        self.shape_flags = wp.full(n, flags_val, dtype=wp.int32, device=d)
        self.shape_collision_group = wp.full(n, 1, dtype=wp.int32, device=d)
        self.shape_world = wp.zeros(n, dtype=wp.int32, device=d)
        if not hasattr(self, "_shape_sdf_index"):
            self.shape_sdf_index = wp.full(n, -1, dtype=wp.int32, device=d)
        if not hasattr(self, "texture_sdf_data"):
            self.texture_sdf_data = None

        # Populate per-shape local AABBs and voxel resolutions (meshes have real values)
        aabb_lower_list = []
        aabb_upper_list = []
        voxel_res_list = []
        for i in range(n):
            if i in self._shape_local_aabb:
                lo, hi, vr = self._shape_local_aabb[i]
                aabb_lower_list.append(lo)
                aabb_upper_list.append(hi)
                voxel_res_list.append(vr)
            else:
                aabb_lower_list.append((0.0, 0.0, 0.0))
                aabb_upper_list.append((1.0, 1.0, 1.0))
                voxel_res_list.append((4, 4, 4))

        self.shape_voxel_resolution = wp.array([wp.vec3i(*v) for v in voxel_res_list], dtype=wp.vec3i, device=d)
        self.shape_collision_aabb_lower = wp.array([wp.vec3(*v) for v in aabb_lower_list], dtype=wp.vec3, device=d)
        self.shape_collision_aabb_upper = wp.array([wp.vec3(*v) for v in aabb_upper_list], dtype=wp.vec3, device=d)

        self.shape_transform = wp.zeros(n, dtype=wp.transform, device=d)
        self.shape_aabb_lower = wp.zeros(n, dtype=wp.vec3, device=d)
        self.shape_aabb_upper = wp.zeros(n, dtype=wp.vec3, device=d)

        max_candidates = n * (n - 1) // 2 + 1
        self.candidate_pair = wp.zeros(max_candidates, dtype=wp.vec2i, device=d)
        self.candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=d)

        # Pre-allocate a dummy filter_pairs array to avoid per-frame
        # wp.empty() in BroadPhaseAllPairs.launch(), which would break
        # CUDA graph capture.  Size 1 so shape[0] > 0 skips the branch.
        self._filter_pairs = wp.zeros(1, dtype=wp.vec2i, device=d)

        self.contact_pair = wp.zeros(self.max_contacts, dtype=wp.vec2i, device=d)
        self.contact_position = wp.zeros(self.max_contacts, dtype=wp.vec3, device=d)
        self.contact_normal = wp.zeros(self.max_contacts, dtype=wp.vec3, device=d)
        self.contact_penetration = wp.zeros(self.max_contacts, dtype=wp.float32, device=d)
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=d)

        self.broad_phase = BroadPhaseAllPairs(
            shape_world=self.shape_world,
            shape_flags=self.shape_flags,
            device=d,
        )
        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=max_candidates,
            device=d,
        )

        self._finalized = True

    # -- per-frame collision ------------------------------------------------

    def collide(self, solver_state):
        """Run the full collision pipeline and write contacts into the solver.

        Args:
            solver_state: a :class:`SolverState` whose ``body_store`` and
                ``contact_store`` will be read / written.
        """
        d = self.device
        bs = solver_state.body_store
        cs = solver_state.contact_store
        n = self.shape_count_val

        # 1. Update world-space shape transforms
        wp.launch(
            _update_shape_transforms_kernel,
            dim=n,
            inputs=[
                self.shape_body_row,
                self.shape_local_transform,
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("flags"),
                self.shape_transform,
                self.shape_count,
            ],
            device=d,
        )

        # 2. Compute AABBs
        wp.launch(
            _compute_shape_aabbs_kernel,
            dim=n,
            inputs=[
                self.shape_type,
                self.shape_data,
                self.shape_transform,
                self.shape_collision_aabb_lower,
                self.shape_collision_aabb_upper,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                self.shape_count,
            ],
            device=d,
        )

        # 3. Broad phase
        self.candidate_pair_count.zero_()
        self.broad_phase.launch(
            shape_lower=self.shape_aabb_lower,
            shape_upper=self.shape_aabb_upper,
            shape_gap=self.shape_gap,
            shape_collision_group=self.shape_collision_group,
            shape_world=self.shape_world,
            shape_count=n,
            candidate_pair=self.candidate_pair,
            candidate_pair_count=self.candidate_pair_count,
            device=d,
            filter_pairs=self._filter_pairs,
            num_filter_pairs=0,
        )

        # 4. Narrow phase
        self.contact_count.zero_()
        self.narrow_phase.launch(
            candidate_pair=self.candidate_pair,
            candidate_pair_count=self.candidate_pair_count,
            shape_types=self.shape_type,
            shape_data=self.shape_data,
            shape_transform=self.shape_transform,
            shape_source=self.shape_source,
            shape_sdf_index=self.shape_sdf_index,
            texture_sdf_data=self.texture_sdf_data,
            shape_gap=self.shape_gap,
            shape_collision_radius=self.shape_collision_radius,
            shape_flags=self.shape_flags,
            shape_collision_aabb_lower=self.shape_collision_aabb_lower,
            shape_collision_aabb_upper=self.shape_collision_aabb_upper,
            shape_voxel_resolution=self.shape_voxel_resolution,
            contact_pair=self.contact_pair,
            contact_position=self.contact_position,
            contact_normal=self.contact_normal,
            contact_penetration=self.contact_penetration,
            contact_count=self.contact_count,
            device=d,
        )

        # 5. Copy contact count into solver's contact store
        wp.copy(cs.count, self.contact_count)

        # 6. Convert NarrowPhase output → PhoenX contact-store columns
        wp.launch(
            _convert_contacts_kernel,
            dim=self.max_contacts,
            inputs=[
                self.contact_pair,
                self.contact_position,
                self.contact_normal,
                self.contact_penetration,
                self.contact_count,
                self.shape_body_row,
                bs.column_of("position"),
                bs.column_of("orientation"),
                solver_state.default_friction,
                cs.column_of("shape0"),
                cs.column_of("shape1"),
                cs.column_of("body0"),
                cs.column_of("body1"),
                cs.column_of("normal"),
                cs.column_of("offset0"),
                cs.column_of("offset1"),
                cs.column_of("margin0"),
                cs.column_of("margin1"),
                cs.column_of("friction"),
                cs.column_of("accumulated_normal_impulse"),
                cs.column_of("accumulated_tangent_impulse1"),
                cs.column_of("accumulated_tangent_impulse2"),
            ],
            device=d,
        )

        # 7. Warm-starter pipeline
        solver_state.warm_starter.import_keys(
            cs.column_of("shape0"),
            cs.column_of("shape1"),
            cs.count,
            offset0=cs.column_of("offset0"),
        )
        solver_state.warm_starter.sort()
        solver_state.warm_starter.build_bundles()
        solver_state.warm_starter.transfer_impulses(
            cs.column_of("accumulated_normal_impulse"),
            cs.column_of("accumulated_tangent_impulse1"),
            cs.column_of("accumulated_tangent_impulse2"),
        )
