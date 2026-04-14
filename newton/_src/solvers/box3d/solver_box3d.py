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
from .kernels_integrate import integrate_positions_2d, integrate_velocities_2d
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

        # Pre-color joints (static topology)
        self._color_joints()

        # Convert static joint data
        self._joints_converted = False

    # ──────────────────────────────────────────────────────────────────
    # Joint coloring (CPU, once at construction / model change)
    # ──────────────────────────────────────────────────────────────────

    def _color_joints(self):
        """Run greedy graph coloring on joints and upload color order."""
        model = self.model
        if model.joint_count == 0:
            self._num_joint_colors = 0
            return

        jws = model.joint_world_start.numpy()
        bws = model.body_world_start.numpy()

        j_parent_np = model.joint_parent.numpy()
        j_child_np = model.joint_child.numpy()

        # Color each world's joints independently
        max_colors = 0
        for w in range(self._num_worlds):
            j_start = int(jws[w])
            j_end = int(jws[w + 1])
            nj = j_end - j_start
            if nj == 0:
                continue

            b_start = int(bws[w])
            b_end = int(bws[w + 1])
            nb = b_end - b_start

            # Local body indices
            parent_local = np.where(
                j_parent_np[j_start:j_end] >= 0,
                j_parent_np[j_start:j_end] - b_start,
                -1,
            )
            child_local = np.where(
                j_child_np[j_start:j_end] >= 0,
                j_child_np[j_start:j_end] - b_start,
                -1,
            )

            order, offsets, nc = color_joints_cpu(parent_local, child_local, nj, nb)
            if nc > max_colors:
                max_colors = nc

            # Upload color order for this world
            order_2d = np.zeros(self._config.max_joints_per_world, dtype=np.int32)
            order_2d[:nj] = order
            self._buf.joint_color_offsets.numpy()[w, : nc + 1] = offsets
            # Pad remaining offsets
            for k in range(nc + 1, self._config.max_colors + 1):
                self._buf.joint_color_offsets.numpy()[w, k] = int(offsets[-1])

            self._buf.joint_count.numpy()[w] = nj

        self._num_joint_colors = max_colors

        # Sync to device
        wp.synchronize_device(self.device)

    def _convert_joints(self):
        """Convert Newton joint data to Box3D format (once, or after topology change)."""
        model = self.model
        if model.joint_count == 0:
            self._joints_converted = True
            return

        # For now, joints are converted via a kernel.
        # The color_order mapping is already in joint_color_offsets.
        # We need to build the color_order 2D array from the CPU coloring.
        # This is a simplification — we store identity order for now since
        # joint_color_offsets already provides the color structure.

        # TODO: launch convert_joints_to_box3d kernel
        self._joints_converted = True

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
                    buf.body_delta_pos, buf.bodies_per_world,
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

            # 4a. Integrate velocities
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

            # 4b. Biased contact solve (+ warm start on first substep)
            if has_contacts:
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

            # 4d. Relaxation contact solve (+ restitution on last substep)
            if has_contacts:
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
                        ],
                        block_dim=cfg.block_dim,
                        device=device,
                    )

        # ── 5. Store impulses for next-frame warm starting ──────────
        if has_contacts:
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

        # ── 6. Convert bodies Box3D → Newton ────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # Model change notification
    # ──────────────────────────────────────────────────────────────────

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._color_joints()
            self._joints_converted = False

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        # TODO: Convert Box3D solver impulses to Newton contact forces
        pass
