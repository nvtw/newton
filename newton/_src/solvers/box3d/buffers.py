# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pre-allocated GPU buffers for the Box3D solver.

All arrays are allocated once at solver construction and reused every
step.  No dynamic allocation happens during :meth:`SolverBox3D.step`.
This is critical for CUDA-graph capture and for eliminating allocator
overhead in the training hot path.

Arrays use a 2-D ``[world, local_index]`` layout so that each CUDA
thread block can index its own world's data directly.
"""

from __future__ import annotations

import warp as wp

from .config import Box3DConfig


class SolverBuffers:
    """Pre-allocated GPU buffer pool for SolverBox3D.

    Args:
        num_worlds: Number of independent simulation worlds.
        cfg: Solver configuration (provides max sizes).
        device: Target device.
    """

    def __init__(self, num_worlds: int, cfg: Box3DConfig, device: wp.Device):
        W = num_worlds
        B = cfg.max_bodies_per_world
        C = cfg.max_contacts_per_world
        J = cfg.max_joints_per_world
        K = cfg.max_colors

        # ── Body state (2-D: [world, body]) ──────────────────────────
        self.body_pos = wp.zeros((W, B), dtype=wp.vec3, device=device)
        self.body_ori = wp.zeros((W, B), dtype=wp.quat, device=device)
        self.body_vel = wp.zeros((W, B), dtype=wp.vec3, device=device)
        self.body_ang_vel = wp.zeros((W, B), dtype=wp.vec3, device=device)
        self.body_inv_mass = wp.zeros((W, B), dtype=float, device=device)
        self.body_inv_inertia = wp.zeros((W, B), dtype=wp.mat33, device=device)
        self.body_com = wp.zeros((W, B), dtype=wp.vec3, device=device)
        self.body_delta_pos = wp.zeros((W, B), dtype=wp.vec3, device=device)
        self.body_inv_inertia_body = wp.zeros((W, B), dtype=wp.mat33, device=device)
        """Body-frame inverse inertia (constant). Used to recompute world-frame each substep."""

        # ── Raw contacts (before coloring, 2-D: [world, contact]) ────
        self.raw_body_a = wp.zeros((W, C), dtype=wp.int32, device=device)
        self.raw_body_b = wp.zeros((W, C), dtype=wp.int32, device=device)
        self.raw_normal = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.raw_r_a = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.raw_r_b = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.raw_base_sep = wp.zeros((W, C), dtype=float, device=device)
        self.raw_friction = wp.zeros((W, C), dtype=float, device=device)
        self.raw_restitution = wp.zeros((W, C), dtype=float, device=device)
        self.raw_normal_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.raw_friction1_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.raw_friction2_impulse = wp.zeros((W, C), dtype=float, device=device)

        # ── Colored contact solver arrays (ordered by color) ─────────
        self.c_body_a = wp.zeros((W, C), dtype=wp.int32, device=device)
        self.c_body_b = wp.zeros((W, C), dtype=wp.int32, device=device)
        self.c_normal = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.c_tangent1 = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.c_tangent2 = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.c_r_a = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.c_r_b = wp.zeros((W, C), dtype=wp.vec3, device=device)
        self.c_base_sep = wp.zeros((W, C), dtype=float, device=device)
        self.c_normal_mass = wp.zeros((W, C), dtype=float, device=device)
        self.c_tangent1_mass = wp.zeros((W, C), dtype=float, device=device)
        self.c_tangent2_mass = wp.zeros((W, C), dtype=float, device=device)
        self.c_friction = wp.zeros((W, C), dtype=float, device=device)
        self.c_restitution = wp.zeros((W, C), dtype=float, device=device)
        self.c_normal_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.c_friction1_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.c_friction2_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.c_total_normal_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.c_rel_vel_normal = wp.zeros((W, C), dtype=float, device=device)
        self.c_is_static = wp.zeros((W, C), dtype=wp.int32, device=device)

        # ── Contact coloring ─────────────────────────────────────────
        self.contact_count = wp.zeros(W, dtype=wp.int32, device=device)
        self.color_offsets = wp.zeros((W, K + 1), dtype=wp.int32, device=device)
        # Scratch for coloring (per-body bitmask)
        self.color_body_mask = wp.zeros((W, B), dtype=wp.int64, device=device)

        # ── Warm starting (previous-frame impulses, in sort order) ───
        self.prev_normal_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.prev_friction1_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.prev_friction2_impulse = wp.zeros((W, C), dtype=float, device=device)
        self.prev_contact_count = wp.zeros(W, dtype=wp.int32, device=device)
        # Mapping from color-ordered to raw order (for storing impulses back)
        self.color_to_raw = wp.zeros((W, C), dtype=wp.int32, device=device)

        # ── Joint solver arrays (pre-colored, 2-D: [world, joint]) ───
        self.j_body_a = wp.zeros((W, J), dtype=wp.int32, device=device)
        self.j_body_b = wp.zeros((W, J), dtype=wp.int32, device=device)
        self.j_type = wp.zeros((W, J), dtype=wp.int32, device=device)
        self.j_local_anchor_a = wp.zeros((W, J), dtype=wp.vec3, device=device)
        self.j_local_anchor_b = wp.zeros((W, J), dtype=wp.vec3, device=device)
        self.j_hinge_axis_local = wp.zeros((W, J), dtype=wp.vec3, device=device)
        self.j_linear_impulse = wp.zeros((W, J), dtype=wp.vec3, device=device)
        self.j_angular_impulse = wp.zeros((W, J), dtype=wp.vec3, device=device)
        self.j_motor_speed = wp.zeros((W, J), dtype=float, device=device)
        self.j_max_motor_torque = wp.zeros((W, J), dtype=float, device=device)
        self.j_limit_lower = wp.zeros((W, J), dtype=float, device=device)
        self.j_limit_upper = wp.zeros((W, J), dtype=float, device=device)
        self.j_limit_enabled = wp.zeros((W, J), dtype=wp.int32, device=device)
        self.j_motor_enabled = wp.zeros((W, J), dtype=wp.int32, device=device)
        self.j_lower_impulse = wp.zeros((W, J), dtype=float, device=device)
        self.j_upper_impulse = wp.zeros((W, J), dtype=float, device=device)

        # ── Joint coloring ───────────────────────────────────────────
        self.joint_count = wp.zeros(W, dtype=wp.int32, device=device)
        self.joint_color_offsets = wp.zeros((W, K + 1), dtype=wp.int32, device=device)

        # ── Per-world body counts ────────────────────────────────────
        self.bodies_per_world = wp.zeros(W, dtype=wp.int32, device=device)
