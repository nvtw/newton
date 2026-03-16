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

"""Drop-in Newton solver adapter for the PhoenX PGS rigid-body engine.

Provides :class:`SolverPhoenX` which implements Newton's
:class:`~newton._src.solvers.solver.SolverBase` interface, allowing
existing Newton examples to run with the PhoenX solver by changing
one line: ``solver = SolverPhoenX(model)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, State
from ...sim.enums import BodyFlags, JointType
from ..solver import SolverBase
from .collision import (
    GEO_TYPE_BOX,
    GEO_TYPE_CAPSULE,
    GEO_TYPE_CYLINDER,
    GEO_TYPE_MESH,
    GEO_TYPE_PLANE,
    GEO_TYPE_SPHERE,
    PhoenXCollisionPipeline,
)
from .solver_phoenx import SolverState

# ---------------------------------------------------------------------------
# Sync kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _sync_newton_to_phoenx_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    px_position: wp.array(dtype=wp.vec3),
    px_orientation: wp.array(dtype=wp.quat),
    px_velocity: wp.array(dtype=wp.vec3),
    px_angular_velocity: wp.array(dtype=wp.vec3),
    count: int,
):
    """Copy Newton body state into PhoenX body-store columns."""
    tid = wp.tid()
    if tid >= count:
        return

    q = body_q[tid]
    qd = body_qd[tid]

    px_position[tid] = wp.transform_get_translation(q)
    px_orientation[tid] = wp.transform_get_rotation(q)
    # Newton spatial_vector: top = linear velocity, bottom = angular velocity
    px_velocity[tid] = wp.spatial_top(qd)
    px_angular_velocity[tid] = wp.spatial_bottom(qd)


@wp.kernel
def _sync_phoenx_to_newton_kernel(
    px_position: wp.array(dtype=wp.vec3),
    px_orientation: wp.array(dtype=wp.quat),
    px_velocity: wp.array(dtype=wp.vec3),
    px_angular_velocity: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    count: int,
):
    """Copy PhoenX body-store columns back into Newton body state."""
    tid = wp.tid()
    if tid >= count:
        return

    body_q[tid] = wp.transform(px_position[tid], px_orientation[tid])
    body_qd[tid] = wp.spatial_vector(px_velocity[tid], px_angular_velocity[tid])


# ---------------------------------------------------------------------------
# Newton adapter
# ---------------------------------------------------------------------------

# Map Newton GeoType integer values to collision pipeline methods.
_GEO_TYPE_MAP = {
    GEO_TYPE_PLANE: "plane",
    GEO_TYPE_SPHERE: "sphere",
    GEO_TYPE_CAPSULE: "capsule",
    GEO_TYPE_CYLINDER: "cylinder",
    GEO_TYPE_BOX: "box",
    GEO_TYPE_MESH: "mesh",
}


class SolverPhoenX(SolverBase):
    """Newton :class:`SolverBase` adapter for the PhoenX PGS rigid-body solver.

    Translates Newton's :class:`Model` bodies, shapes, and joints into
    PhoenX's internal representation at construction time, then syncs
    state back and forth every :meth:`step`.

    Args:
        model: Newton simulation model.
        num_iterations: PGS position iterations (with Baumgarte bias).
        num_velocity_iterations: PGS velocity iterations (no bias).
        default_friction: Coulomb friction coefficient for contacts.
        max_colors: maximum graph-colouring partitions for parallel PGS.
        max_contacts: upper bound on contacts per frame.
        num_substeps: number of PhoenX substeps per :meth:`step` call.
    """

    def __init__(
        self,
        model: Model,
        num_iterations: int = 8,
        num_velocity_iterations: int = 0,
        default_friction: float = 0.5,
        max_colors: int = 12,
        max_contacts: int = 4096,
        num_substeps: int = 8,
    ):
        super().__init__(model)

        self._num_iterations = num_iterations
        self._num_velocity_iterations = num_velocity_iterations
        self._num_substeps = num_substeps

        # Read gravity from model (world 0)
        if model.gravity is not None:
            g = model.gravity.numpy()[0]
            self._gravity = (float(g[0]), float(g[1]), float(g[2]))
        else:
            self._gravity = (0.0, -9.81, 0.0)

        d = model.device
        n_bodies = model.body_count
        n_shapes = model.shape_count
        n_joints = model.joint_count

        # Determine whether we need a static world body for joints with
        # parent == -1 or shapes with body == -1 (ground plane, etc.).
        self._has_world_body = False
        if n_joints > 0 and model.joint_parent is not None:
            jp = model.joint_parent.numpy()
            if np.any(jp < 0):
                self._has_world_body = True
        if n_shapes > 0 and model.shape_body is not None:
            sb = model.shape_body.numpy()
            if np.any(sb < 0):
                self._has_world_body = True

        total_bodies = n_bodies + (1 if self._has_world_body else 0)

        # Create SolverState
        self.ss = SolverState(
            body_capacity=max(total_bodies, 1),
            contact_capacity=max_contacts,
            shape_count=n_shapes,
            device=d,
            default_friction=default_friction,
            max_colors=max_colors,
            joint_capacity=n_joints,
        )

        # Mapping from Newton body index → PhoenX handle
        self._newton_to_phoenx: dict[int, int] = {}

        # Create collision pipeline
        self.pipeline = PhoenXCollisionPipeline(
            max_shapes=max(n_shapes, 1),
            max_contacts=max_contacts,
            device=d,
        )

        # Populate
        self._init_bodies(model)
        self._init_shapes(model)
        self._init_joints(model)
        self.pipeline.finalize()

        # Pass through SDF data from Newton model for mesh-mesh contacts
        if hasattr(model, "texture_sdf_data") and model.texture_sdf_data is not None:
            self.pipeline.texture_sdf_data = model.texture_sdf_data
        if hasattr(model, "shape_sdf_index") and model.shape_sdf_index is not None:
            self.pipeline.shape_sdf_index = model.shape_sdf_index

        # Wire up mesh data for voxel-bucketed warm starting
        if any(t == GEO_TYPE_MESH for t in self.pipeline._shape_type_list):
            self.ss.warm_starter.set_mesh_data(
                shape_type=self.pipeline.shape_type,
                shape_collision_aabb_lower=self.pipeline.shape_collision_aabb_lower,
                shape_collision_aabb_upper=self.pipeline.shape_collision_aabb_upper,
                shape_voxel_resolution=self.pipeline.shape_voxel_resolution,
            )

    # -- initialisation helpers ---------------------------------------------

    def _init_bodies(self, model: Model) -> None:
        """Add all Newton bodies to the PhoenX solver state."""
        n = model.body_count
        if n == 0:
            return

        body_q_np = model.body_q.numpy()  # (N, 7) — px,py,pz,qx,qy,qz,qw
        body_inv_mass_np = model.body_inv_mass.numpy()  # (N,)
        body_inv_inertia_np = model.body_inv_inertia.numpy()  # (N, 3, 3)
        body_flags_np = model.body_flags.numpy()  # (N,)

        for i in range(n):
            xf = body_q_np[i]
            pos = (float(xf[0]), float(xf[1]), float(xf[2]))
            orient = (float(xf[3]), float(xf[4]), float(xf[5]), float(xf[6]))

            inv_m = float(body_inv_mass_np[i])
            inv_I = body_inv_inertia_np[i]
            is_static = not bool(body_flags_np[i] & int(BodyFlags.DYNAMIC))

            handle = self.ss.add_body(
                position=pos,
                orientation=orient,
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                inverse_mass=inv_m,
                inverse_inertia_local=inv_I,
                linear_damping=1.0,
                angular_damping=1.0,
                is_static=is_static,
            )
            self._newton_to_phoenx[i] = handle

        # If needed, add a static world body for world-anchored joints.
        if self._has_world_body:
            h = self.ss.add_body(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                inverse_mass=0.0,
                is_static=True,
            )
            self._world_body_handle = h

    def _init_shapes(self, model: Model) -> None:
        """Register Newton collision shapes with the PhoenX pipeline."""
        n = model.shape_count
        if n == 0:
            return

        shape_type_np = model.shape_type.numpy()
        shape_scale_np = model.shape_scale.numpy()
        shape_body_np = model.shape_body.numpy()
        shape_xf_np = model.shape_transform.numpy()
        shape_mu_np = model.shape_material_mu.numpy()

        # Pre-load mesh-related arrays if any mesh shapes exist
        has_meshes = any(int(t) == GEO_TYPE_MESH for t in shape_type_np)
        if has_meshes:
            source_ptr_np = model.shape_source_ptr.numpy()
            collision_radius_np = model.shape_collision_radius.numpy()
            collision_aabb_lo_np = model.shape_collision_aabb_lower.numpy()
            collision_aabb_hi_np = model.shape_collision_aabb_upper.numpy()
            voxel_res_np = model._shape_voxel_resolution.numpy()
            shape_margin_np = model.shape_margin.numpy()
            shape_gap_np = model.shape_gap.numpy()

        h2i = self.ss.body_store.handle_to_index.numpy()

        for i in range(n):
            geo = int(shape_type_np[i])
            scale = shape_scale_np[i]
            body_idx = int(shape_body_np[i])

            # Resolve PhoenX body row
            if body_idx < 0:
                # Shape on a static/world body — use world body if available
                if self._has_world_body:
                    body_row = int(h2i[self._world_body_handle])
                else:
                    body_row = 0
            else:
                handle = self._newton_to_phoenx.get(body_idx)
                if handle is None:
                    warnings.warn(
                        f"Shape {i}: body index {body_idx} not found in PhoenX mapping, skipping.",
                        stacklevel=2,
                    )
                    continue
                body_row = int(h2i[handle])

            # Always map shape → body in PhoenX (needed for contact import)
            phoenx_handle = self._newton_to_phoenx.get(body_idx)
            if phoenx_handle is not None:
                self.ss.set_shape_body(i, phoenx_handle)
            elif self._has_world_body:
                self.ss.set_shape_body(i, self._world_body_handle)

            # Build local transform tuple (px, py, pz, qx, qy, qz, qw)
            xf = shape_xf_np[i]
            local_xf = (
                float(xf[0]),
                float(xf[1]),
                float(xf[2]),
                float(xf[3]),
                float(xf[4]),
                float(xf[5]),
                float(xf[6]),
            )

            mu = float(shape_mu_np[i])

            if geo == GEO_TYPE_BOX:
                self.pipeline.add_shape_box(
                    body_row=body_row,
                    local_transform=local_xf,
                    half_extents=(float(scale[0]), float(scale[1]), float(scale[2])),
                    friction=mu,
                )
            elif geo == GEO_TYPE_SPHERE:
                self.pipeline.add_shape_sphere(
                    body_row=body_row,
                    local_transform=local_xf,
                    radius=float(scale[0]),
                    friction=mu,
                )
            elif geo == GEO_TYPE_CAPSULE:
                self.pipeline.add_shape_capsule(
                    body_row=body_row,
                    local_transform=local_xf,
                    radius=float(scale[0]),
                    half_length=float(scale[1]),
                    friction=mu,
                )
            elif geo == GEO_TYPE_CYLINDER:
                self.pipeline.add_shape_cylinder(
                    body_row=body_row,
                    local_transform=local_xf,
                    radius=float(scale[0]),
                    half_height=float(scale[1]),
                    friction=mu,
                )
            elif geo == GEO_TYPE_PLANE:
                self.pipeline.add_shape_plane(
                    body_row=body_row,
                    local_transform=local_xf,
                    friction=mu,
                )
            elif geo == GEO_TYPE_MESH:
                mesh_id = int(source_ptr_np[i])
                coll_radius = float(collision_radius_np[i])
                aabb_lo = collision_aabb_lo_np[i]
                aabb_hi = collision_aabb_hi_np[i]
                vr = voxel_res_np[i]
                mesh_margin = float(shape_margin_np[i])
                mesh_gap = float(shape_gap_np[i])
                self.pipeline.add_shape_mesh(
                    body_row=body_row,
                    mesh_id=mesh_id,
                    local_transform=local_xf,
                    collision_radius=coll_radius,
                    scale=(float(scale[0]), float(scale[1]), float(scale[2])),
                    margin=mesh_margin,
                    gap=mesh_gap,
                    friction=mu,
                    aabb_lower=(float(aabb_lo[0]), float(aabb_lo[1]), float(aabb_lo[2])),
                    aabb_upper=(float(aabb_hi[0]), float(aabb_hi[1]), float(aabb_hi[2])),
                    voxel_resolution=(int(vr[0]), int(vr[1]), int(vr[2])),
                )
            else:
                geo_name = _GEO_TYPE_MAP.get(geo, str(geo))
                warnings.warn(
                    f"Shape {i}: unsupported geometry type {geo_name} ({geo}), contacts from Newton will still work.",
                    stacklevel=2,
                )

    def _init_joints(self, model: Model) -> None:
        """Translate Newton joints into PhoenX constraints."""
        n = model.joint_count
        if n == 0:
            return

        jt_np = model.joint_type.numpy()
        jp_np = model.joint_parent.numpy()
        jc_np = model.joint_child.numpy()
        jXp_np = model.joint_X_p.numpy()  # (N, 7)
        jXc_np = model.joint_X_c.numpy()  # (N, 7)

        # Joint axis (per-DOF), and qd_start to find the first axis per joint
        has_axis = model.joint_axis is not None and model.joint_qd_start is not None
        if has_axis:
            axis_np = model.joint_axis.numpy()
            qd_start_np = model.joint_qd_start.numpy()

        # Joint limits and drives (per-DOF arrays)
        has_limits = model.joint_limit_lower is not None and model.joint_limit_upper is not None
        if has_limits:
            lim_lo_np = model.joint_limit_lower.numpy()
            lim_hi_np = model.joint_limit_upper.numpy()
        has_drives = (
            model.joint_target_ke is not None
            and model.joint_target_kd is not None
            and model.joint_target_pos is not None
        )
        if has_drives:
            drive_ke_np = model.joint_target_ke.numpy()
            drive_kd_np = model.joint_target_kd.numpy()
            drive_pos_np = model.joint_target_pos.numpy()
            drive_mode_np = model.joint_target_mode.numpy() if model.joint_target_mode is not None else None

        # Read body poses for computing world-space anchors
        body_q_np = model.body_q.numpy()

        def _body_pose(body_idx):
            """Return (position, orientation) for a body, or origin for world."""
            if body_idx < 0:
                return np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            xf = body_q_np[body_idx]
            return xf[:3].astype(np.float32), xf[3:7].astype(np.float32)

        def _quat_rotate(q, v):
            """Rotate vector v by quaternion q (xyzw layout)."""
            qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)

            def _qmul(a, b):
                return np.array(
                    [
                        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
                    ],
                    dtype=np.float32,
                )

            qc = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
            return _qmul(_qmul(q, qv), qc)[:3]

        # First pass: identify bodies that are fixed-jointed to world.
        # These should be static in PhoenX (no gravity, no dynamics).
        from .schemas import BODY_FLAG_STATIC

        for i in range(n):
            jtype = int(jt_np[i])
            parent_body = int(jp_np[i])
            child_body = int(jc_np[i])
            if jtype == JointType.FIXED and parent_body < 0:
                # Mark child body as static in PhoenX
                handle = self._newton_to_phoenx.get(child_body)
                if handle is not None:
                    row = int(self.ss.body_store.handle_to_index.numpy()[handle])
                    inv_m = self.ss.body_store.column_of("inverse_mass").numpy()
                    inv_m[row] = 0.0
                    self.ss.body_store.column_of("inverse_mass").assign(
                        wp.array(inv_m, dtype=wp.float32, device=self.ss.device)
                    )
                    inv_i = self.ss.body_store.column_of("inverse_inertia_local").numpy()
                    inv_i[row] = np.zeros((3, 3), dtype=np.float32)
                    self.ss.body_store.column_of("inverse_inertia_local").assign(
                        wp.array(inv_i, dtype=wp.mat33, device=self.ss.device)
                    )
                    flags_col = self.ss.body_store.column_of("flags").numpy()
                    flags_col[row] = BODY_FLAG_STATIC
                    self.ss.body_store.column_of("flags").assign(
                        wp.array(flags_col, dtype=wp.int32, device=self.ss.device)
                    )

        # Second pass: create PhoenX joints (skip fixed-to-world, already handled)
        for i in range(n):
            jtype = int(jt_np[i])
            parent_body = int(jp_np[i])
            child_body = int(jc_np[i])

            # Skip fixed-to-world joints (child is now static)
            if jtype == JointType.FIXED and parent_body < 0:
                continue

            # Resolve PhoenX handles
            if parent_body < 0:
                if not self._has_world_body:
                    warnings.warn(
                        f"Joint {i}: parent is world but no world body available, skipping.",
                        stacklevel=2,
                    )
                    continue
                parent_handle = self._world_body_handle
            else:
                parent_handle = self._newton_to_phoenx.get(parent_body)
                if parent_handle is None:
                    warnings.warn(f"Joint {i}: parent body {parent_body} not mapped, skipping.", stacklevel=2)
                    continue

            child_handle = self._newton_to_phoenx.get(child_body)
            if child_handle is None:
                warnings.warn(f"Joint {i}: child body {child_body} not mapped, skipping.", stacklevel=2)
                continue

            # Compute world-space anchor from parent body pose + joint_X_p
            p_pos, p_orient = _body_pose(parent_body)
            xp = jXp_np[i]
            local_anchor = xp[:3].astype(np.float32)
            anchor_world = p_pos + _quat_rotate(p_orient, local_anchor)
            anchor = (float(anchor_world[0]), float(anchor_world[1]), float(anchor_world[2]))

            # Get joint axis in child frame, then rotate to world
            if has_axis and jtype in (JointType.REVOLUTE, JointType.PRISMATIC):
                dof_start = int(qd_start_np[i])
                local_axis = axis_np[dof_start].astype(np.float32)
                c_pos, c_orient = _body_pose(child_body)
                world_axis = _quat_rotate(c_orient, local_axis)
                axis = (float(world_axis[0]), float(world_axis[1]), float(world_axis[2]))
            else:
                axis = (0.0, 0.0, 1.0)

            # Read per-DOF limits for this joint
            angle_min = -1.0e7
            angle_max = 1.0e7
            if has_limits and has_axis:
                dof_start = int(qd_start_np[i])
                lo = float(lim_lo_np[dof_start])
                hi = float(lim_hi_np[dof_start])
                if lo > -1.0e6 or hi < 1.0e6:
                    angle_min = lo
                    angle_max = hi

            ji = -1
            if jtype == JointType.REVOLUTE:
                ji = self.ss.add_joint_revolute(
                    parent_handle,
                    child_handle,
                    anchor,
                    axis,
                    angle_min=angle_min,
                    angle_max=angle_max,
                )
            elif jtype == JointType.BALL:
                ji = self.ss.add_joint_ball_socket(parent_handle, child_handle, anchor)
            elif jtype == JointType.FIXED:
                ji = self.ss.add_joint_fixed(parent_handle, child_handle, anchor)
            elif jtype == JointType.PRISMATIC:
                ji = self.ss.add_joint_prismatic(
                    parent_handle,
                    child_handle,
                    anchor,
                    axis,
                    slide_min=angle_min,
                    slide_max=angle_max,
                )
            elif jtype == JointType.FREE:
                continue  # No constraint needed
            elif jtype == JointType.DISTANCE:
                ji = self.ss.add_joint_ball_socket(parent_handle, child_handle, anchor)
            else:
                warnings.warn(
                    f"Joint {i}: unsupported joint type {jtype}, skipping.",
                    stacklevel=2,
                )
                continue

            # Apply joint drives (position drive with stiffness/damping)
            if ji >= 0 and has_drives and has_axis:
                dof_start = int(qd_start_np[i])
                ke = float(drive_ke_np[dof_start])
                kd = float(drive_kd_np[dof_start])
                target = float(drive_pos_np[dof_start])
                if ke > 0.0 or kd > 0.0:
                    self.ss.set_joint_drive(
                        ji,
                        mode=self.ss.DRIVE_POSITION,
                        target=target,
                        stiffness=ke,
                        damping=kd,
                    )

    # -- step ---------------------------------------------------------------

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by *dt* using PhoenX PGS solving.

        Args:
            state_in: input Newton state (body poses and velocities).
            state_out: output Newton state (written with updated poses/velocities).
            control: ignored (PhoenX does not consume Newton controls).
            contacts: ignored (PhoenX runs its own collision pipeline).
            dt: total time step [s].
        """
        d = self.model.device
        n = self.model.body_count
        if n == 0:
            return

        bs = self.ss.body_store

        # 1. Sync Newton state_in → PhoenX body store
        wp.launch(
            _sync_newton_to_phoenx_kernel,
            dim=n,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                n,
            ],
            device=d,
        )

        # 2. PhoenX pipeline (matching C# World.Step: detect once, substep N times)
        self.ss.update_world_inertia()

        # Collision detection ONCE per frame (C# lines 211-304)
        self.ss.warm_starter.begin_frame()
        self.pipeline.collide(self.ss)

        sub_dt = dt / float(self._num_substeps)
        for _ in range(self._num_substeps):
            self.ss.step(
                sub_dt,
                gravity=self._gravity,
                num_iterations=self._num_iterations,
                num_velocity_iterations=self._num_velocity_iterations,
            )

        self.ss.export_impulses()

        # 3. Sync PhoenX body store → Newton state_out
        wp.launch(
            _sync_phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                bs.column_of("position"),
                bs.column_of("orientation"),
                bs.column_of("velocity"),
                bs.column_of("angular_velocity"),
                state_out.body_q,
                state_out.body_qd,
                n,
            ],
            device=d,
        )
