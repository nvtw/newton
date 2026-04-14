# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverBox3D — Box2D v3 TGS-Soft solver ported to 3-D for Newton.

A GPU-parallel rigid-body solver designed for AI training, using one
CUDA thread block per world and colored Gauss-Seidel for race-free
constraint solving within each block.  Uses Newton's standard
:class:`~newton.CollisionPipeline` for contact detection and
:class:`~newton._src.geometry.contact_match.ContactMatcher` for
warm starting impulses between frames.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .buffers import SolverBuffers
from .coloring import color_joints_cpu, coloring_prepare_kernel, prepare_contact_masses_2d
from .config import Box3DConfig, Softness, compute_softness
from .convert import (
    convert_bodies_from_box3d,
    convert_bodies_to_box3d,
    convert_contacts_to_box3d,
    convert_joints_to_box3d,
)
from .kernels_integrate import integrate_positions_2d, integrate_velocities_2d, update_world_inertia_2d
from .kernels_solve import contact_solve_kernel
from .kernels_store import store_impulses_2d


class SolverBox3D(SolverBase):
    """Box2D v3 TGS-Soft solver ported to 3-D with colored Gauss-Seidel.

    Designed for GPU-parallel simulation of many independent worlds,
    using one CUDA thread block per world.  Uses Newton's standard
    :class:`~newton.CollisionPipeline` for contact detection and
    :class:`~newton._src.geometry.contact_match.ContactMatcher` for
    warm starting impulses between frames.

    The solver implements the following per-step pipeline:

    1. Convert Newton body state to internal 2-D ``[world, body]`` layout.
    2. Convert Newton contacts to raw per-world contacts with warm-started
       impulses via ``ContactMatcher``.
    3. GPU graph-color contacts and compute tangent bases / effective masses.
    4. Sub-step loop (*num_substeps* times):
       a. Integrate velocities (gravity + damping).
       b. Fused contact solve (biased, warm start on first sub-step).
       c. Integrate positions (accumulate delta-position).
       d. Fused contact solve (relaxation, restitution on last sub-step).
    5. Store impulses for next-frame warm starting.
    6. Convert body state back to Newton format.

    Args:
        model: The Newton model.
        config: Solver configuration.  Defaults to :class:`Box3DConfig` defaults.
    """

    def __init__(self, model: Model, config: Box3DConfig | None = None):
        super().__init__(model=model)
        self._config = config or Box3DConfig()
        self._init_kinematic_state()

        self._num_worlds = max(model.world_count, 1)

        # Allocate solver buffers
        self._buf = SolverBuffers(self._num_worlds, self._config, model.device)

        # Pre-color joints and convert static joint data
        self._color_joints()
        self._convert_joints()

        # CUDA graph state
        self._graph: wp.Graph | None = None
        self._graph_dt: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Joint coloring (CPU, once at construction / model change)
    # ──────────────────────────────────────────────────────────────────

    def _get_joint_ranges(self):
        """Get (start, end) ranges per world, including global joints in world 0."""
        model = self.model
        jws = model.joint_world_start.numpy()
        # Global joints: indices 0..jws[0]-1 (before first world)
        # and jws[-2]..jws[-1]-1 (after last world).
        # We merge all global joints into world 0.
        ranges = []
        for w in range(self._num_worlds):
            j_start = int(jws[w])
            j_end = int(jws[w + 1])
            ranges.append((j_start, j_end))
        # Add global prefix and suffix to world 0
        global_prefix_end = int(jws[0])
        global_suffix_start = int(jws[-2])
        global_suffix_end = int(jws[-1])
        # Collect all joint indices for world 0
        w0_indices = list(range(0, global_prefix_end))
        w0_start, w0_end = ranges[0] if ranges else (0, 0)
        w0_indices.extend(range(w0_start, w0_end))
        w0_indices.extend(range(global_suffix_start, global_suffix_end))
        return w0_indices, ranges

    def _color_joints(self):
        """Run greedy graph coloring on joints and upload color order."""
        model = self.model
        if model.joint_count == 0:
            self._num_joint_colors = 0
            return

        j_parent_np = model.joint_parent.numpy()
        j_child_np = model.joint_child.numpy()
        body_world_np = model.body_world.numpy()
        bws = model.body_world_start.numpy()

        w0_indices, _ = self._get_joint_ranges()

        # For now, process all joints as world 0 (single-world case)
        # Multi-world support can be added later
        nj = len(w0_indices)
        if nj == 0:
            self._num_joint_colors = 0
            return

        nb = model.body_count

        # Compute local body indices
        parent_local = np.zeros(nj, dtype=np.int32)
        child_local = np.zeros(nj, dtype=np.int32)
        for i, gj in enumerate(w0_indices):
            pg = j_parent_np[gj]
            cg = j_child_np[gj]
            # Global bodies use their global index as local index
            parent_local[i] = pg if pg >= 0 else -1
            child_local[i] = cg if cg >= 0 else -1

        order, offsets, nc = color_joints_cpu(parent_local, child_local, nj, nb)

        # Upload to buffers
        buf = self._buf
        jc_off_np = buf.joint_color_offsets.numpy()
        jc_off_np[0, : nc + 1] = offsets
        for k in range(nc + 1, self._config.max_colors + 1):
            jc_off_np[0, k] = int(offsets[-1])
        buf.joint_color_offsets.assign(jc_off_np)
        jcount_np = buf.joint_count.numpy()
        jcount_np[0] = nj
        buf.joint_count.assign(jcount_np)

        self._num_joint_colors = nc
        self._joint_color_order = order
        self._joint_global_indices = np.array(w0_indices, dtype=np.int32)

    def _convert_joints(self):
        """Convert Newton joint data to Box3D 2-D format (CPU, once)."""
        model = self.model
        buf = self._buf
        if model.joint_count == 0:
            return

        j_type_np = model.joint_type.numpy()
        j_parent_np = model.joint_parent.numpy()
        j_child_np = model.joint_child.numpy()
        j_X_p_np = model.joint_X_p.numpy()
        j_X_c_np = model.joint_X_c.numpy()
        j_axis_np = model.joint_axis.numpy()
        j_qd_start_np = model.joint_qd_start.numpy()
        body_com_np = model.body_com.numpy()

        order = self._joint_color_order
        global_indices = self._joint_global_indices
        nj = len(global_indices)

        b_body_a = buf.j_body_a.numpy()
        b_body_b = buf.j_body_b.numpy()
        b_type = buf.j_type.numpy()
        b_la = buf.j_local_anchor_a.numpy()
        b_lb = buf.j_local_anchor_b.numpy()
        b_ha = buf.j_hinge_axis_local.numpy()

        for src_j in range(nj):
            dst_j = int(order[src_j])
            gj = int(global_indices[src_j])

            jtype = int(j_type_np[gj])
            # Skip FREE joints (type 4) — they don't constrain anything
            if jtype == 4:
                b_type[0, dst_j] = 0  # mark as inactive (type 0 = PRISMATIC, unused)
                b_body_a[0, dst_j] = -1
                b_body_b[0, dst_j] = -1
                continue

            pg = int(j_parent_np[gj])
            cg = int(j_child_np[gj])
            b_body_a[0, dst_j] = pg if pg >= 0 else -1
            b_body_b[0, dst_j] = cg if cg >= 0 else -1
            b_type[0, dst_j] = jtype

            # Anchor translations from joint transforms (first 3 elements)
            # Adjust for COM offset: anchor_in_com_frame = anchor_in_body_frame - body_com
            anchor_a = j_X_p_np[gj, :3].copy()
            anchor_b = j_X_c_np[gj, :3].copy()
            if pg >= 0:
                anchor_a -= body_com_np[pg]
            if cg >= 0:
                anchor_b -= body_com_np[cg]
            b_la[0, dst_j] = anchor_a
            b_lb[0, dst_j] = anchor_b

            # Hinge axis from joint_axis array
            qd_start = int(j_qd_start_np[gj])
            if jtype == 1:  # REVOLUTE — single axis
                b_ha[0, dst_j] = j_axis_np[qd_start, :3]
            else:
                b_ha[0, dst_j] = [0, 0, 1]  # default Z for non-revolute

        buf.j_body_a.assign(b_body_a)
        buf.j_body_b.assign(b_body_b)
        buf.j_type.assign(b_type)
        buf.j_local_anchor_a.assign(b_la)
        buf.j_local_anchor_b.assign(b_lb)
        buf.j_hinge_axis_local.assign(b_ha)

    # ──────────────────────────────────────────────────────────────────
    # Main step
    # ──────────────────────────────────────────────────────────────────

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        model = self.model
        cfg = self._config
        buf = self._buf
        device = model.device
        W = self._num_worlds

        sub_dt = dt / float(cfg.num_substeps)
        inv_sub_dt = 1.0 / sub_dt if sub_dt > 0.0 else 0.0

        # Softness parameters (computed per step, constant across substeps)
        soft = compute_softness(cfg.contact_hertz, cfg.contact_damping_ratio, sub_dt)
        soft_static = compute_softness(
            cfg.contact_hertz * cfg.static_hertz_scale,
            cfg.contact_damping_ratio,
            sub_dt,
        )
        soft_joint = compute_softness(cfg.joint_hertz, cfg.joint_damping_ratio, sub_dt)

        num_joints = model.joint_count
        num_joint_colors = self._num_joint_colors

        gravity_np = model.gravity.numpy().flatten()[:3]
        gravity_vec = wp.vec3(float(gravity_np[0]), float(gravity_np[1]), float(gravity_np[2]))

        # ── 1. Convert bodies Newton → Box3D ────────────────────────
        buf.bodies_per_world.zero_()
        if model.body_count > 0:
            wp.launch(
                convert_bodies_to_box3d,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_com,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.body_flags,
                    model.body_world,
                    model.body_world_start,
                    W,
                ],
                outputs=[
                    buf.body_pos, buf.body_ori, buf.body_vel, buf.body_ang_vel,
                    buf.body_inv_mass, buf.body_inv_inertia, buf.body_com,
                    buf.body_delta_pos, buf.body_inv_inertia_body,
                    buf.bodies_per_world,
                ],
                device=device,
            )

        # ── 2. Convert contacts Newton → Box3D ─────────────────────
        has_contacts = contacts is not None and model.shape_count > 0

        # Zero per-world contact counts
        buf.contact_count.zero_()

        if has_contacts:
            has_matching = contacts.contact_matching and contacts.rigid_contact_match_index is not None
            match_index = contacts.rigid_contact_match_index if has_matching else buf.contact_count  # dummy

            wp.launch(
                convert_contacts_to_box3d,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    match_index,
                    model.shape_body,
                    model.shape_material_mu,
                    model.shape_material_restitution,
                    model.body_world,
                    model.body_world_start,
                    state_in.body_q,
                    model.body_com,
                    model.body_flags,
                    buf.prev_normal_impulse,
                    buf.prev_friction1_impulse,
                    buf.prev_friction2_impulse,
                    buf.prev_contact_count,
                    cfg.max_contacts_per_world,
                ],
                outputs=[
                    buf.raw_body_a, buf.raw_body_b, buf.raw_normal,
                    buf.raw_r_a, buf.raw_r_b, buf.raw_base_sep,
                    buf.raw_friction, buf.raw_restitution,
                    buf.raw_normal_impulse, buf.raw_friction1_impulse,
                    buf.raw_friction2_impulse,
                    buf.contact_count,
                ],
                device=device,
            )

        # ── 3-5. Solve (optionally graph-captured) ──────────────────
        if cfg.enable_graph:
            if self._graph is None or self._graph_dt != dt:
                self._graph_dt = dt
                wp.capture_begin(device=device, force_module_load=True)
                try:
                    self._launch_solve_kernels(
                        W, cfg, buf, device, sub_dt, inv_sub_dt,
                        gravity_vec, soft, soft_static, soft_joint,
                        num_joints, num_joint_colors, has_contacts,
                    )
                finally:
                    self._graph = wp.capture_end(device=device)
            wp.capture_launch(self._graph)
        else:
            self._launch_solve_kernels(
                W, cfg, buf, device, sub_dt, inv_sub_dt,
                gravity_vec, soft, soft_static, soft_joint,
                num_joints, num_joint_colors, has_contacts,
            )

        # ── 6. Store impulses for next-frame warm starting ──────────
        wp.launch(
            store_impulses_2d,
            dim=(W, cfg.max_contacts_per_world),
            inputs=[
                buf.c_normal_impulse, buf.c_friction1_impulse,
                buf.c_friction2_impulse, buf.contact_count,
            ],
            outputs=[
                buf.prev_normal_impulse, buf.prev_friction1_impulse,
                buf.prev_friction2_impulse, buf.prev_contact_count,
            ],
            device=device,
        )

        # ── 7. Convert bodies Box3D → Newton ────────────────────────
        # First, copy all state from input to output (kinematic bodies
        # and any other untouched data). The convert-back kernel will
        # overwrite dynamic bodies.
        if model.body_count > 0:
            state_out.body_q.assign(state_in.body_q)
            state_out.body_qd.assign(state_in.body_qd)
        if model.body_count > 0:
            wp.launch(
                convert_bodies_from_box3d,
                dim=model.body_count,
                inputs=[
                    buf.body_pos, buf.body_ori, buf.body_vel, buf.body_ang_vel,
                    buf.body_com,
                    model.body_world, model.body_world_start,
                    model.body_flags, model.body_com, W,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=device,
            )

        # Copy particle state through unchanged
        if model.particle_count > 0:
            state_out.particle_q.assign(state_in.particle_q)
            state_out.particle_qd.assign(state_in.particle_qd)

    def _launch_solve_kernels(
        self,
        W: int,
        cfg: Box3DConfig,
        buf: SolverBuffers,
        device: wp.Device,
        sub_dt: float,
        inv_sub_dt: float,
        gravity_vec: wp.vec3,
        soft: Softness,
        soft_static: Softness,
        soft_joint: Softness,
        num_joints: int,
        num_joint_colors: int,
        has_contacts: bool,
    ) -> None:
        """Launch the graph-capturable solve kernels (steps 3-5)."""
        # ── 3. Graph-color contacts + prepare masses ────────────────
        if has_contacts:
            wp.launch_tiled(
                coloring_prepare_kernel,
                dim=[W],
                inputs=[
                    buf.raw_body_a, buf.raw_body_b, buf.raw_normal,
                    buf.raw_r_a, buf.raw_r_b, buf.raw_base_sep,
                    buf.raw_friction, buf.raw_restitution,
                    buf.raw_normal_impulse, buf.raw_friction1_impulse,
                    buf.raw_friction2_impulse,
                    buf.c_body_a, buf.c_body_b, buf.c_normal,
                    buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                    buf.c_friction, buf.c_restitution,
                    buf.c_normal_impulse, buf.c_friction1_impulse,
                    buf.c_friction2_impulse,
                    buf.c_total_normal_impulse, buf.c_is_static,
                    buf.contact_count, buf.color_offsets,
                    buf.color_body_mask, buf.color_to_raw,
                    buf.bodies_per_world, buf.body_inv_mass,
                    cfg.max_colors,
                ],
                block_dim=cfg.block_dim,
                device=device,
            )

            # Compute tangent basis + effective masses
            wp.launch(
                prepare_contact_masses_2d,
                dim=(W, cfg.max_contacts_per_world),
                inputs=[
                    buf.c_body_a, buf.c_body_b, buf.c_normal,
                    buf.c_r_a, buf.c_r_b,
                    buf.body_vel, buf.body_ang_vel,
                    buf.body_inv_mass, buf.body_inv_inertia,
                    buf.contact_count, cfg.max_contacts_per_world,
                ],
                outputs=[
                    buf.c_tangent1, buf.c_tangent2,
                    buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                    buf.c_rel_vel_normal,
                ],
                device=device,
            )

        # ── 4. Substep loop ─────────────────────────────────────────
        max_bodies = cfg.max_bodies_per_world

        for sub in range(cfg.num_substeps):
            is_first = sub == 0
            is_last = sub == cfg.num_substeps - 1

            # 4a. Update world-frame inertia from current orientation
            if sub > 0:
                wp.launch(
                    update_world_inertia_2d,
                    dim=(W, max_bodies),
                    inputs=[
                        buf.body_ori, buf.body_inv_mass,
                        buf.body_inv_inertia_body, buf.body_inv_inertia,
                        buf.bodies_per_world,
                    ],
                    device=device,
                )

            # 4b. Integrate velocities
            wp.launch(
                integrate_velocities_2d,
                dim=(W, max_bodies),
                inputs=[
                    buf.body_vel, buf.body_ang_vel, buf.body_inv_mass,
                    buf.bodies_per_world, gravity_vec,
                    cfg.linear_damping, cfg.angular_damping, sub_dt,
                ],
                device=device,
            )

            # 4b. Biased contact+joint solve (+ warm start on first substep)
            for _ in range(cfg.num_velocity_iters):
                wp.launch_tiled(
                    contact_solve_kernel,
                    dim=[W],
                    inputs=[
                        buf.body_vel, buf.body_ang_vel,
                        buf.body_inv_mass, buf.body_inv_inertia,
                        buf.body_delta_pos,
                        buf.c_body_a, buf.c_body_b,
                        buf.c_normal, buf.c_tangent1, buf.c_tangent2,
                        buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                        buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                        buf.c_friction, buf.c_restitution, buf.c_rel_vel_normal,
                        buf.c_normal_impulse, buf.c_friction1_impulse,
                        buf.c_friction2_impulse, buf.c_total_normal_impulse,
                        buf.c_is_static,
                        buf.color_offsets,
                        max_bodies, cfg.max_colors,
                        1,  # use_bias
                        1 if is_first else 0,  # warm_start
                        0,  # no restitution yet
                        inv_sub_dt,
                        soft.bias_rate, soft.mass_scale, soft.impulse_scale,
                        soft_static.bias_rate, soft_static.mass_scale,
                        soft_static.impulse_scale,
                        cfg.contact_speed, cfg.restitution_threshold,
                        # Joint parameters
                        buf.body_pos, buf.body_ori,
                        buf.j_body_a, buf.j_body_b, buf.j_type,
                        buf.j_local_anchor_a, buf.j_local_anchor_b,
                        buf.j_hinge_axis_local,
                        buf.j_linear_impulse, buf.j_angular_impulse,
                        buf.j_motor_speed, buf.j_max_motor_torque,
                        buf.joint_color_offsets,
                        num_joints, num_joint_colors,
                        soft_joint.bias_rate, soft_joint.mass_scale,
                        soft_joint.impulse_scale, sub_dt,
                    ],
                    block_dim=cfg.block_dim,
                    device=device,
                )

            # 4c. Integrate positions
            wp.launch(
                integrate_positions_2d,
                dim=(W, max_bodies),
                inputs=[
                    buf.body_pos, buf.body_ori, buf.body_vel,
                    buf.body_ang_vel, buf.body_inv_mass,
                    buf.body_delta_pos, buf.bodies_per_world, sub_dt,
                ],
                device=device,
            )

            # 4d. Relaxation contact+joint solve (+ restitution on last substep)
            for _ in range(cfg.num_relaxation_iters):
                wp.launch_tiled(
                    contact_solve_kernel,
                    dim=[W],
                    inputs=[
                        buf.body_vel, buf.body_ang_vel,
                        buf.body_inv_mass, buf.body_inv_inertia,
                        buf.body_delta_pos,
                        buf.c_body_a, buf.c_body_b,
                        buf.c_normal, buf.c_tangent1, buf.c_tangent2,
                        buf.c_r_a, buf.c_r_b, buf.c_base_sep,
                        buf.c_normal_mass, buf.c_tangent1_mass, buf.c_tangent2_mass,
                        buf.c_friction, buf.c_restitution, buf.c_rel_vel_normal,
                        buf.c_normal_impulse, buf.c_friction1_impulse,
                        buf.c_friction2_impulse, buf.c_total_normal_impulse,
                        buf.c_is_static,
                        buf.color_offsets,
                        max_bodies, cfg.max_colors,
                        0,  # no bias (relaxation)
                        0,  # no warm start
                        1 if is_last else 0,  # restitution on last substep
                        inv_sub_dt,
                        soft.bias_rate, soft.mass_scale, soft.impulse_scale,
                        soft_static.bias_rate, soft_static.mass_scale,
                        soft_static.impulse_scale,
                        cfg.contact_speed, cfg.restitution_threshold,
                        # Joint parameters
                        buf.body_pos, buf.body_ori,
                        buf.j_body_a, buf.j_body_b, buf.j_type,
                        buf.j_local_anchor_a, buf.j_local_anchor_b,
                        buf.j_hinge_axis_local,
                        buf.j_linear_impulse, buf.j_angular_impulse,
                        buf.j_motor_speed, buf.j_max_motor_torque,
                        buf.joint_color_offsets,
                        num_joints, num_joint_colors,
                        soft_joint.bias_rate, soft_joint.mass_scale,
                        soft_joint.impulse_scale, sub_dt,
                    ],
                    block_dim=cfg.block_dim,
                    device=device,
                )

    # ──────────────────────────────────────────────────────────────────
    # Model change notification
    # ──────────────────────────────────────────────────────────────────

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._color_joints()
            self._convert_joints()
        self._graph = None  # invalidate CUDA graph

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        # TODO: Convert Box3D solver impulses to Newton contact forces
        pass
