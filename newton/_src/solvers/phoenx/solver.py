# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PhoenX solver wrapped in Newton's :class:`SolverBase` interface.

Drives :class:`PhoenXWorld` from Newton's Model/State/Control/Contacts. Per step:
import (Newton -> PhoenX body fields + joint_f), joint-control writeback (Control
+ Model gains -> drive dwords), export (PhoenX -> body_q / body_qd).
PhoenX slot 0 is the static world anchor; pass ``substeps=1`` to substep outside.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.sim import BodyFlags, Contacts, Control, Model, State
from newton._src.solvers.flags import SolverNotifyFlags
from newton._src.solvers.phoenx.body import (
    BodyContainer,
    body_container_zeros,
)
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    _OFF_DAMPING_DRIVE,
    _OFF_DRIVE_MODE,
    _OFF_MAX_FORCE_DRIVE,
    _OFF_STIFFNESS_DRIVE,
    _OFF_TARGET,
    _OFF_TARGET_VELOCITY,
)
from newton._src.solvers.phoenx.model_adapter import (
    AdbsInitArrays,
    build_adbs_init_arrays,
)
from newton._src.solvers.phoenx.solver_kernels import (
    _apply_joint_drive_control_kernel,
    _apply_joint_forces_kernel,
    _contact_impulse_to_force_wrapper_kernel,
    _export_body_qdd_kernel,
    _export_body_state_avg_kernel,
    _export_body_state_fd_kernel,
    _export_body_state_kernel,
    _import_body_state_kernel,
    _init_phoenx_body_container_kernel,
    _seed_kinematic_initial_pose_kernel,
    _snapshot_pre_step_pose_kernel,
    _snapshot_pre_step_velocity_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.solver import SolverBase

__all__ = ["SolverPhoenX"]


# Host-side quaternion / rotation helpers (used by the armature bake).


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by quaternion ``q = (x, y, z, w)`` (numpy, host)."""
    qx, qy, qz, qw = (float(x) for x in q)
    vx, vy, vz = (float(x) for x in v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.array(
        [
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        ],
        dtype=np.float64,
    )


def _quat_to_rot_np(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from quaternion ``q = (x, y, z, w)``."""
    qx, qy, qz, qw = (float(x) for x in q)
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-20:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1.0 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1.0 - s * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _estimate_rigid_contact_max_phoenx(model) -> int | None:
    """Tight rigid_contact_max from shape_contact_pair_count * 5 (CPP) * 2 (safety).
    None when pair count is unavailable (caller falls back to Newton's default)."""
    pair_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if pair_count <= 0:
        return None

    # GJK/MPR primitive contacts cap at ~5/pair; Newton's default 40 is for
    # opt-in hydroelastic and overshoots us.
    PRIMITIVE_CPP = 5
    SAFETY = 2
    return max(1000, pair_count * PRIMITIVE_CPP * SAFETY)


class _PhoenXCollisionPipelineAdapter:
    """Route model.collide() through PhoenX deformable geometry refresh."""

    def __init__(self, solver: SolverPhoenX, pipeline):
        self._solver = solver
        self._pipeline = pipeline

    def __getattr__(self, name: str):
        return getattr(self._pipeline, name)

    def contacts(self):
        return self._pipeline.contacts()

    def collide(self, state: State, contacts: Contacts, *, soft_contact_margin: float | None = None) -> None:
        self._solver.collide(state, contacts)


class SolverPhoenX(SolverBase):
    """Newton :class:`SolverBase` wrapper around :class:`PhoenXWorld`.

    Supports REVOLUTE / PRISMATIC (PD drive + limit), BALL, FIXED,
    CABLE (soft fixed with PD bend/twist; stretch DoF is rigid), and
    FREE (no column). DISTANCE and D6 raise at construction.

    Newton :class:`Picking` works out of the box: pick force/torque is
    added to ``state.body_f``, which :meth:`step` imports into PhoenX's
    force accumulators before integrating.
    """

    def __init__(
        self,
        model: Model,
        *,
        substeps: int = 1,
        solver_iterations: int = 8,
        velocity_iterations: int = 1,
        default_friction: float = 0.5,
        step_layout: str = "multi_world",
        threads_per_world: int | str = "auto",
        multi_world_scheduler: str = "auto",
        max_thread_blocks: int | None = None,
        velocity_readout: str = "substep_end",
        mass_splitting: bool = False,
        max_colored_partitions: int = 12,
        mass_splitting_batch_size: int = 8,
        partitioner_algorithm: str = "greedy",
        enable_warm_start_coloring: bool = True,
        sor_boost: float = 1.0,
        sleeping_velocity_threshold: float = 0.0,
        sleeping_frames_required: int = 30,
        prepare_refresh_stride: int | str = "auto",
    ):
        """Build the PhoenX solver from ``model``.

        Args:
            substeps: PhoenX internal substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep.
            velocity_iterations: TGS-soft relax sweeps per substep.
            prepare_refresh_stride: Refresh cached rigid contact/joint
                prepare data every N substeps. ``"auto"`` chooses a
                conservative stride from the substep count and falls back
                to ``1`` when cached prepare is unsupported. Pass ``1``
                to force exact per-substep rebuilds.
            default_friction: Fallback when Contacts/shapes carry no material.
            step_layout: ``"multi_world"`` (many small worlds) or
                ``"single_world"`` (a few big worlds).
            threads_per_world: ``"auto"`` / 32 / 16 / 8 (multi-world).
            multi_world_scheduler: Static multi-world scheduler policy.
                ``"auto"`` is the default performance policy and resolves
                before graph capture; ``"fast_tail"`` and
                ``"block_world[_32|_64|_128]"`` force a path for
                benchmarking.
            max_thread_blocks: Optional cap on the single-world PGS grid.
            velocity_readout: ``"substep_end"`` (default, bit-faithful),
                ``"finite_difference"``, or ``"substep_average"``.
            partitioner_algorithm: ``"greedy"`` (default) or
                ``"luby_fixed"`` (single-world only).
            enable_warm_start_coloring: Reuse previous-frame colour
                assignments. No-op on multi-world.
            sor_boost: Per-impulse SOR factor. 1.0 = vanilla PGS;
                1.1-1.5 typical; ``>= 2.0`` diverges.
            sleeping_velocity_threshold: Per-island sleep cutoff
                ``[m/s + rad/s * 0.5 * aabb_diag]``. ``0.0`` disables
                sleeping (no island build, no extra allocations).
                Sleeping bodies are dropped from coloring + the
                overflow partition and skip gravity / forces; a
                sleeping-aware broad-phase filter is auto-installed.
            sleeping_frames_required: Frames an island must stay below
                threshold before being flagged sleeping. Default 30
                (~0.5 s @ 60 Hz). Wake-up is always single-frame.
                ``0`` recovers single-frame sleep.
        """
        super().__init__(model)
        valid_readouts = ("substep_end", "finite_difference", "substep_average")
        if velocity_readout not in valid_readouts:
            raise ValueError(f"velocity_readout must be one of {valid_readouts}, got {velocity_readout!r}")
        self._velocity_readout = velocity_readout

        num_bodies_phoenx = int(model.body_count) + 1
        self.bodies: BodyContainer = body_container_zeros(num_bodies_phoenx, device=self.device)

        # FD/substep-avg readout buffers — always allocated so graph capture
        # has stable refs; only written when the corresponding readout fires.
        n_newton_bodies = int(model.body_count)
        self._fd_pos_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._fd_orient_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.quatf, device=self.device)
        self._substep_vel_accum = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._substep_omega_accum = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        # body_qdd snapshot buffers (linear + angular velocity, pre-step).
        # Always allocated; the FD launch is gated on state.body_qdd being live.
        self._qdd_vel_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)
        self._qdd_omega_prev = wp.zeros(max(1, n_newton_bodies), dtype=wp.vec3f, device=self.device)

        # Identity orientation everywhere so the first _update_inertia is well-defined.
        self.bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32))

        if model.body_count:
            self._launch_init_phoenx_bodies(model)
            # PhoenX is maximal-coordinate; bake reduced-coord armature into both
            # bodies' inertia along the joint axis so eff_inv = J M^-1 J^T sees
            # the augmented mass. Skinny links (<0.1 kg) need this for stable PD.
            self._bake_joint_armature_into_body_inertia(model)

        # FK so model.body_q reflects model.joint_q (URDF rigs may set joint_q
        # before finalize without running FK).
        if int(model.body_count) > 0 and int(model.joint_count) > 0:
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        self._adbs: AdbsInitArrays = build_adbs_init_arrays(model, device=self.device)
        num_joints = self._adbs.num_joint_columns
        num_particles = int(getattr(model, "particle_count", 0) or 0)
        num_cloth_triangles = int(getattr(model, "tri_count", 0) or 0)
        num_cloth_bending = int(getattr(model, "edge_count", 0) or 0)
        num_soft_tetrahedra = int(getattr(model, "tet_count", 0) or 0)
        self._has_particles = num_particles > 0
        self._has_deformable_collision = num_cloth_triangles > 0 or num_soft_tetrahedra > 0
        self._particle_state_imported: State | None = None
        self._default_joint_gear = wp.ones(max(1, int(model.joint_dof_count)), dtype=wp.float32, device=self.device)

        # PhoenX's warm-start path needs contact_matching != "disabled".
        # Auto-attach a sticky pipeline so users don't have to size Contacts.
        self._sleeping_enabled: bool = float(sleeping_velocity_threshold) > 0.0
        if int(model.shape_count) > 0 and not self._has_deformable_collision:
            existing_cp = getattr(model, "_collision_pipeline", None)
            needs_new_cp = existing_cp is None or not getattr(existing_cp, "contact_matching", False)
            # When sleeping is on we need a broad-phase filter func wired
            # into the pipeline at construction time; force rebuild if
            # the existing pipeline doesn't carry one.
            if self._sleeping_enabled and not needs_new_cp:
                existing_filter = getattr(existing_cp, "_broad_phase_filter_func", None)
                if existing_filter is None:
                    needs_new_cp = True
            if needs_new_cp:
                import newton as _newton  # noqa: PLC0415

                # PhoenX-tight rigid_contact_max from shape_contact_pair_count;
                # Newton's default ignores COLLIDE_SHAPES filter and overshoots
                # ~15x with visual-only meshes.
                tight_rcm = _estimate_rigid_contact_max_phoenx(model)
                if tight_rcm is not None:
                    model.rigid_contact_max = 0  # bypass "already sized" short-circuit
                from newton._src.solvers.phoenx.cloth_collision import (  # noqa: PLC0415
                    PhoenXClothShareVertexFilterData,
                    phoenx_cloth_share_vertex_filter,
                )
                from newton._src.solvers.phoenx.solver_config import (  # noqa: PLC0415
                    PHOENX_CONTACT_MATCHING,
                )

                cp_kwargs = {
                    "contact_matching": PHOENX_CONTACT_MATCHING,
                    "rigid_contact_max": tight_rcm,
                }
                if self._sleeping_enabled:
                    cp_kwargs["broad_phase_filter"] = (
                        phoenx_cloth_share_vertex_filter,
                        PhoenXClothShareVertexFilterData,
                    )
                model._collision_pipeline = _newton.CollisionPipeline(model, **cp_kwargs)
                model._collision_pipeline.contacts()  # forces buffer sizing
        if self._has_deformable_collision and int(model.rigid_contact_max) <= 0:
            deformable_shapes = num_cloth_triangles + num_soft_tetrahedra
            model.rigid_contact_max = max(1000, 8 * (int(model.shape_count) + deformable_shapes))
        rigid_contact_max = int(model.rigid_contact_max)

        gravity_np = self._read_model_gravity_np(model)
        num_worlds = max(1, int(gravity_np.shape[0]))
        gravity_tuples = [tuple(float(x) for x in row) for row in gravity_np]
        if len(gravity_tuples) == 1:
            gravity_arg = gravity_tuples[0]
        else:
            gravity_arg = gravity_tuples

        self._constraints: ConstraintContainer = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            num_cloth_triangles=num_cloth_triangles,
            num_cloth_bending=num_cloth_bending,
            num_soft_tetrahedra=num_soft_tetrahedra,
            device=self.device,
        )

        # Body-pair grouping pays a sort/gather cost. Use it for
        # single-scene compound contact graphs, not multi-world fleets.
        has_compound_bodies = False
        if model.shape_body is not None and model.shape_count > 0 and model.body_count > 0:
            sb = model.shape_body.numpy()
            sb = sb[sb >= 0]
            if sb.size > 0:
                counts = np.bincount(sb, minlength=int(model.body_count))
                has_compound_bodies = bool((counts > 1).any())

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self._constraints,
            substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=int(velocity_iterations),
            gravity=gravity_arg,
            rigid_contact_max=rigid_contact_max,
            num_joints=num_joints,
            num_particles=num_particles,
            num_cloth_triangles=num_cloth_triangles,
            num_cloth_bending=num_cloth_bending,
            num_soft_tetrahedra=num_soft_tetrahedra,
            default_friction=float(default_friction),
            num_worlds=num_worlds,
            step_layout=step_layout,
            threads_per_world=threads_per_world,
            multi_world_scheduler=multi_world_scheduler,
            max_thread_blocks=max_thread_blocks,
            enable_body_pair_grouping=has_compound_bodies and (step_layout == "single_world" or num_worlds == 1),
            mass_splitting=mass_splitting,
            max_colored_partitions=max_colored_partitions,
            mass_splitting_batch_size=mass_splitting_batch_size,
            partitioner_algorithm=partitioner_algorithm,
            enable_warm_start_coloring=enable_warm_start_coloring,
            sor_boost=sor_boost,
            sleeping_velocity_threshold=float(sleeping_velocity_threshold),
            sleeping_frames_required=int(sleeping_frames_required),
            prepare_refresh_stride=prepare_refresh_stride,
            device=self.device,
        )

        # When sleeping is on (and not already wired by a downstream
        # ``setup_cloth_collision_pipeline``), bind the share-vertex
        # filter data with sleeping fields populated. The pipeline's
        # filter func is shared with cloth setups; cloth setup paths
        # call ``build_phoenx_share_vertex_filter_data`` themselves and
        # overwrite this binding without losing the sleeping fields.
        if self._sleeping_enabled and int(model.shape_count) > 0 and not self._has_deformable_collision:
            from newton._src.solvers.phoenx.cloth_collision import (  # noqa: PLC0415
                build_phoenx_share_vertex_filter_data,
            )

            tri_sentinel = wp.zeros((1, 3), dtype=wp.int32, device=self.device)
            tet_sentinel = wp.zeros((1, 4), dtype=wp.int32, device=self.device)
            filter_data = build_phoenx_share_vertex_filter_data(
                num_rigid_shapes=int(model.shape_count),
                num_cloth_triangles=0,
                tri_indices=tri_sentinel,
                tet_indices=tet_sentinel,
                sleeping_enabled=True,
                phoenx_body_offset=1,
                shape_body=model.shape_body,
                body_island_root=self.bodies.island_root,
                body_motion_type=self.bodies.motion_type,
                device=self.device,
            )
            model._collision_pipeline.set_broad_phase_filter_data(filter_data)
            self._share_vertex_filter_data = filter_data
            self.world._share_vertex_filter_data = filter_data

        # Seed body pose BEFORE joint init — ADBS init reads body positions to
        # snapshot body-local anchors. Without this, welds pull child to origin.
        if int(model.body_count) > 0:
            zero_wrench = wp.zeros(int(model.body_count), dtype=wp.spatial_vector, device=self.device)
            wp.launch(
                _import_body_state_kernel,
                dim=int(model.body_count),
                inputs=[
                    model.body_q,
                    model.body_qd,
                    zero_wrench,
                    model.body_com,
                    self.bodies,
                ],
                device=self.device,
            )
            wp.launch(
                _seed_kinematic_initial_pose_kernel,
                dim=int(model.body_count) + 1,  # +1 for slot 0 (world anchor)
                inputs=[self.bodies],
                device=self.device,
            )

        if num_joints > 0:
            self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())

        if num_cloth_triangles > 0:
            self.world.populate_cloth_triangles_from_model(model)
        if num_cloth_bending > 0:
            self.world.populate_cloth_bending_from_model(model)
        if num_soft_tetrahedra > 0:
            self.world.populate_soft_tetrahedra_from_model(model)
        if self._has_deformable_collision:
            pipeline = self.world.setup_cloth_collision_pipeline(model, rigid_contact_max=rigid_contact_max)
            model._collision_pipeline = _PhoenXCollisionPipelineAdapter(self, pipeline)

        if model.shape_material_mu is not None and model.shape_count > 0:
            self._install_shape_materials()

        # Newton shape_body uses -1 for world; PhoenX slot 0 is the world anchor.
        if self._has_deformable_collision and self.world._shape_body_internal is not None:
            self._shape_body = self.world._shape_body_internal
        elif model.shape_body is not None and model.shape_count > 0:
            shape_body_np = model.shape_body.numpy()
            shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
            self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)
        else:
            self._shape_body = None

        self._has_joint_forces = model.joint_dof_count > 0
        self._last_dt: float = 0.0

        # Placeholder for _contact_impulse_to_force_wrapper_kernel when grouping
        # is off (has_perm=0 makes the kernel ignore it).
        self._sort_perm_placeholder = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _bake_joint_armature_into_body_inertia(self, model: Model) -> None:
        """Add per-joint axial armature into both attached bodies' inertia so the
        constraint solve sees ``M_chain + armature`` along the joint axis.

        Bake amount per body (``alpha``):
        * Fixed-anchor: ``alpha = a`` on child only.
        * Two-body: solve ``alpha^2 + alpha*(S - 2a - 2*M_chain) - a*S = 0`` for
          the positive root (S = I_A + I_B, M_chain = I_A*I_B/S). Yields
          ``(I_A + alpha)*(I_B + alpha) / (I_A + I_B + 2*alpha) == M_chain + a``,
          giving exact-MuJoCo equivalence for any mass ratio.
        """
        if model.joint_count == 0 or model.joint_armature is None:
            return

        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_X_p = model.joint_X_p.numpy()
        joint_X_c = model.joint_X_c.numpy()
        armature = model.joint_armature.numpy()

        body_inv_mass = self.bodies.inverse_mass.numpy().copy()
        body_inv_inertia = self.bodies.inverse_inertia.numpy().copy()
        body_inv_inertia_world = self.bodies.inverse_inertia_world.numpy().copy()
        body_orientation = self.bodies.orientation.numpy()

        n_phoenx = body_inv_inertia.shape[0]
        body_inertia = np.zeros_like(body_inv_inertia)
        body_mass = np.zeros(n_phoenx, dtype=np.float32)
        for i in range(n_phoenx):
            invI = body_inv_inertia[i]
            if abs(np.linalg.det(invI)) > 1e-30:
                body_inertia[i] = np.linalg.inv(invI)
            body_mass[i] = (1.0 / body_inv_mass[i]) if body_inv_mass[i] > 0.0 else 0.0

        rev_t = int(newton.JointType.REVOLUTE)
        pri_t = int(newton.JointType.PRISMATIC)

        any_baked = False
        for j in range(model.joint_count):
            jt = int(joint_type[j])
            if jt != rev_t and jt != pri_t:
                continue
            qd = int(joint_qd_start[j])
            if qd >= len(armature):
                continue
            a = float(armature[qd])
            if a <= 0.0:
                continue
            any_baked = True

            # Newton body indices; PhoenX slots are body+1 with -1->0 (world).
            n_p = int(joint_parent[j])
            n_c = int(joint_child[j])
            slot_p = 0 if n_p < 0 else n_p + 1
            slot_c = 0 if n_c < 0 else n_c + 1

            if jt == rev_t:
                axis_jf = joint_axis[qd] if qd < len(joint_axis) else np.array([1.0, 0.0, 0.0])
                # Rotate joint axis into each body's local frame.
                Xp_q = joint_X_p[j][3:7]
                Xc_q = joint_X_c[j][3:7]
                axis_in_p = _quat_rotate_np(Xp_q, axis_jf)
                axis_in_c = _quat_rotate_np(Xc_q, axis_jf)
                np_p = float(np.linalg.norm(axis_in_p))
                np_c = float(np.linalg.norm(axis_in_c))
                if np_p > 1e-12:
                    axis_in_p = axis_in_p / np_p
                if np_c > 1e-12:
                    axis_in_c = axis_in_c / np_c

                if slot_p == 0:
                    alpha_p = 0.0
                    alpha_c = a
                elif slot_c == 0:
                    alpha_p = a
                    alpha_c = 0.0
                else:
                    i_a = float(axis_in_p @ body_inertia[slot_p] @ axis_in_p)
                    i_b = float(axis_in_c @ body_inertia[slot_c] @ axis_in_c)
                    if i_a > 0.0 and i_b > 0.0:
                        s_sum = i_a + i_b
                        m_chain = i_a * i_b / s_sum
                        b_coef = s_sum - 2.0 * a - 2.0 * m_chain
                        c_coef = -a * s_sum
                        disc = b_coef * b_coef - 4.0 * c_coef
                        # c_coef <= 0 so disc >= b_coef^2 (positive root real).
                        alpha = 0.5 * (-b_coef + float(np.sqrt(max(disc, 0.0))))
                    else:
                        alpha = a
                    alpha_p = alpha
                    alpha_c = alpha

                if slot_p > 0 and alpha_p > 0.0:
                    body_inertia[slot_p] = body_inertia[slot_p] + alpha_p * np.outer(axis_in_p, axis_in_p)
                if slot_c > 0 and alpha_c > 0.0:
                    body_inertia[slot_c] = body_inertia[slot_c] + alpha_c * np.outer(axis_in_c, axis_in_c)
            else:  # PRISMATIC: scalar add to mass (same quadratic as REVOLUTE).
                if slot_p == 0:
                    alpha_p = 0.0
                    alpha_c = a
                elif slot_c == 0:
                    alpha_p = a
                    alpha_c = 0.0
                else:
                    m_a = body_mass[slot_p]
                    m_b = body_mass[slot_c]
                    if m_a > 0.0 and m_b > 0.0:
                        s_sum = m_a + m_b
                        m_chain = m_a * m_b / s_sum
                        b_coef = s_sum - 2.0 * a - 2.0 * m_chain
                        c_coef = -a * s_sum
                        disc = b_coef * b_coef - 4.0 * c_coef
                        alpha = 0.5 * (-b_coef + float(np.sqrt(max(disc, 0.0))))
                    else:
                        alpha = a
                    alpha_p = alpha
                    alpha_c = alpha

                if slot_p > 0 and alpha_p > 0.0:
                    body_mass[slot_p] = body_mass[slot_p] + alpha_p
                if slot_c > 0 and alpha_c > 0.0:
                    body_mass[slot_c] = body_mass[slot_c] + alpha_c

        if not any_baked:
            return

        # Recompute inverses, then rotate to world for the rest pose.
        for i in range(1, n_phoenx):
            I = body_inertia[i]
            if abs(np.linalg.det(I)) > 1e-30:
                body_inv_inertia[i] = np.linalg.inv(I).astype(np.float32)
            m = body_mass[i]
            body_inv_mass[i] = (1.0 / m) if m > 0.0 else 0.0

            # Rotate to world. Quaternion is (x, y, z, w).
            q = body_orientation[i]
            R = _quat_to_rot_np(q)
            body_inv_inertia_world[i] = (R @ body_inv_inertia[i] @ R.T).astype(np.float32)

        self.bodies.inverse_mass.assign(body_inv_mass)
        self.bodies.inverse_inertia.assign(body_inv_inertia)
        self.bodies.inverse_inertia_world.assign(body_inv_inertia_world)

    def _install_shape_materials(self) -> None:
        """Stream Model's per-shape (mu_static, mu_dynamic, restitution) into
        PhoenX's material table; each shape gets its own material index."""
        from newton._src.solvers.phoenx.materials import (  # noqa: PLC0415
            CombineMode,
            Material,
            material_table_from_list,
        )

        mu_np = self.model.shape_material_mu.numpy()
        restitution = (
            self.model.shape_material_restitution.numpy()
            if self.model.shape_material_restitution is not None
            else np.zeros_like(mu_np)
        )
        materials = [
            Material(
                static_friction=float(mu_np[i]),
                dynamic_friction=float(mu_np[i]),
                restitution=float(restitution[i]),
                friction_combine_mode=CombineMode.AVERAGE,
                restitution_combine_mode=CombineMode.AVERAGE,
            )
            for i in range(self.model.shape_count)
        ]
        material_data = material_table_from_list(materials, device=self.device)
        shape_material_idx = wp.array(
            np.arange(self.model.shape_count, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.world.set_materials(material_data, shape_material_idx)

    def _apply_joint_control(self, control: Control) -> None:
        """Rewrite ADBS drive dwords from control + model. Falls back to Model
        targets if Control doesn't supply them."""
        if self._adbs.num_drive_columns == 0:
            return
        model = self.model
        target_pos = (
            control.joint_target_q
            if control is not None and control.joint_target_q is not None
            else model.joint_target_q
        )
        target_vel = (
            control.joint_target_qd
            if control is not None and control.joint_target_qd is not None
            else model.joint_target_qd
        )
        if target_pos is None or target_vel is None or model.joint_target_mode is None:
            return  # no per-DOF drive configured
        wp.launch(
            _apply_joint_drive_control_kernel,
            dim=int(self._adbs.num_drive_columns),
            inputs=[
                self._adbs.drive_cid,
                self._adbs.drive_dof_start,
                self._adbs.drive_target_q_index,
                self._adbs.drive_q_at_init,
                model.joint_target_mode,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_effort_limit,
                self._joint_gear_array(),
                target_pos,
                target_vel,
                wp.int32(0),  # DRIVE_MODE_OFF
                wp.int32(1),  # DRIVE_MODE_POSITION
                wp.int32(2),  # DRIVE_MODE_VELOCITY
                wp.int32(int(newton.JointTargetMode.POSITION)),
                wp.int32(int(newton.JointTargetMode.VELOCITY)),
                wp.int32(int(newton.JointTargetMode.POSITION_VELOCITY)),
                wp.int32(int(_OFF_DRIVE_MODE)),
                wp.int32(int(_OFF_TARGET)),
                wp.int32(int(_OFF_TARGET_VELOCITY)),
                wp.int32(int(_OFF_STIFFNESS_DRIVE)),
                wp.int32(int(_OFF_DAMPING_DRIVE)),
                wp.int32(int(_OFF_MAX_FORCE_DRIVE)),
                self._constraints,
            ],
            device=self.device,
        )

    def _joint_gear_array(self) -> wp.array[wp.float32]:
        n = max(1, int(self.model.joint_dof_count))
        if self._default_joint_gear.shape[0] != n:
            self._default_joint_gear = wp.ones(n, dtype=wp.float32, device=self.device)
        return self._default_joint_gear

    def _accumulate_joint_forces(self, state_in: State, control: Control, dt: float) -> None:
        """Fold ``control.joint_f`` into ``state_in.body_f`` (Newton's EFFORT path)."""
        if control is None or control.joint_f is None:
            return
        if not self._has_joint_forces:
            return
        model = self.model
        if model.joint_count == 0:
            return
        wp.launch(
            _apply_joint_forces_kernel,
            dim=int(model.joint_count),
            inputs=[
                state_in.body_q,
                model.body_com,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                control.joint_f,
                state_in.body_f,
            ],
            device=self.device,
        )

    def _import_body_state(self, state_in: State) -> None:
        """Pull state_in into the PhoenX body container (slot i+1).
        Kinematic bodies go to kinematic_target_*; dynamic/static go direct."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _import_body_state_kernel,
            dim=n,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_in.body_f,
                self.model.body_com,
                self.bodies,
            ],
            device=self.device,
        )

    def _import_particle_state(self, state_in: State, *, force: bool = False) -> None:
        """Pull Newton particle state into PhoenX."""
        if not self._has_particles:
            return
        particles = self.world.particles
        if particles is None:
            return
        if not force and state_in is self._particle_state_imported:
            return
        if state_in.particle_q is None or state_in.particle_qd is None:
            raise ValueError("SolverPhoenX requires particle_q and particle_qd for particle models.")
        wp.copy(particles.position, state_in.particle_q)
        wp.copy(particles.velocity, state_in.particle_qd)
        self._particle_state_imported = state_in

    def _export_particle_state(self, state_out: State) -> None:
        """Push PhoenX particle state to Newton."""
        if not self._has_particles:
            return
        particles = self.world.particles
        if particles is None:
            return
        if state_out.particle_q is None or state_out.particle_qd is None:
            raise ValueError("SolverPhoenX requires particle_q and particle_qd for particle models.")
        wp.copy(state_out.particle_q, particles.position)
        wp.copy(state_out.particle_qd, particles.velocity)

    def collide(self, state: State, contacts: Contacts) -> None:
        """Run PhoenX deformable-aware collision."""
        if not self._has_deformable_collision:
            self.model.collide(state, contacts)
            return
        self._import_particle_state(state, force=True)
        self.world.collide(state, contacts)

    def _snapshot_pre_step_pose(self) -> None:
        """Snapshot pre-step COM-in-world pose for the FD readout."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _snapshot_pre_step_pose_kernel,
            dim=n,
            inputs=[self.bodies.position, self.bodies.orientation],
            outputs=[self._fd_pos_prev, self._fd_orient_prev],
            device=self.device,
        )

    def _snapshot_pre_step_velocity(self) -> None:
        """Snapshot pre-step linear + angular velocity for the body_qdd readout.
        Captured after :meth:`_import_body_state` so the FD covers the outer dt."""
        n = int(self.model.body_count)
        if n == 0:
            return
        wp.launch(
            _snapshot_pre_step_velocity_kernel,
            dim=n,
            inputs=[self.bodies.velocity, self.bodies.angular_velocity],
            outputs=[self._qdd_vel_prev, self._qdd_omega_prev],
            device=self.device,
        )

    def _export_body_qdd(self, state_out: State, dt: float) -> None:
        """Pack post-step ``body_qdd`` into ``state_out`` as a finite-difference
        of (post-step - pre-step) velocity over the outer dt. Newton convention:
        ``spatial_top`` is linear acceleration (world frame, includes
        gravity-induced terms), ``spatial_bottom`` is angular acceleration
        (world frame). Matches what :class:`~newton.sensors.SensorIMU` consumes."""
        n = int(self.model.body_count)
        if n == 0:
            return
        inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
        wp.launch(
            _export_body_qdd_kernel,
            dim=n,
            inputs=[
                self.bodies.velocity,
                self.bodies.angular_velocity,
                self._qdd_vel_prev,
                self._qdd_omega_prev,
                wp.float32(inv_dt),
            ],
            outputs=[state_out.body_qdd],
            device=self.device,
        )

    def _export_body_state(self, state_out: State, dt: float) -> None:
        """Pack PhoenX body state back into state_out.body_q / body_qd, switching
        on ``self._velocity_readout``."""
        n = int(self.model.body_count)
        if n == 0:
            return
        if self._velocity_readout == "finite_difference":
            inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
            wp.launch(
                _export_body_state_fd_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.model.body_com,
                    self._fd_pos_prev,
                    self._fd_orient_prev,
                    wp.float32(inv_dt),
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=self.device,
            )
        elif self._velocity_readout == "substep_average":
            inv_dt = 1.0 / float(dt) if dt > 0.0 else 0.0
            wp.launch(
                _export_body_state_avg_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.model.body_com,
                    self._substep_vel_accum,
                    self._substep_omega_accum,
                    wp.float32(inv_dt),
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=self.device,
            )
        else:
            wp.launch(
                _export_body_state_kernel,
                dim=n,
                inputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.bodies.velocity,
                    self.bodies.angular_velocity,
                    self.model.body_com,
                ],
                outputs=[
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=self.device,
            )

    def wake_on_external_input(self, state_in: State) -> None:
        """Wake every sleeping island whose bodies carry an external
        force or torque set in ``state_in.body_f``, *before* the host
        calls ``model.collide(...)``.

        The per-step sleeping pass inside :meth:`step` cannot drive
        broad-phase decisions on the wake frame: by the time it clears
        ``island_root`` for a body that picking just pushed, the
        sleep-aware broad-phase filter has already dropped that body's
        contact pairs and the substep solve sees an empty stack. Call
        sequence on the host side::

            state.body_f.assign(...)  # picking / wrenches
            solver.wake_on_external_input(state)  # propagate wake
            model.collide(state, contacts)  # broad-phase keeps pairs
            solver.step(state, state_out, ...)

        Imports ``state_in.body_f`` into PhoenX's force accumulators
        first so the wake pass reads the user-applied wrench rather
        than the post-clear zeroes that :meth:`step` would otherwise
        re-load it from. A no-op when the sleeping pipeline is
        disabled.
        """
        if not self._sleeping_enabled:
            return
        self._import_body_state(state_in)
        self.world.wake_on_external_input()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance ``state_in`` to ``state_out`` by ``dt``: control writeback +
        effort forces -> import -> PhoenXWorld.step -> export."""
        if control is None:
            # Alias Model per-DOF arrays (no clone). Matches XPBD/Featherstone.
            control = self.model.control(clone_variables=False)

        self._apply_joint_control(control)
        self._accumulate_joint_forces(state_in, control, dt)
        self._import_body_state(state_in)
        self._import_particle_state(state_in)

        # FD readout snapshots the imported (state_in-aligned) pose so the
        # post-step delta covers the full outer dt.
        if self._velocity_readout == "finite_difference":
            self._snapshot_pre_step_pose()

        # body_qdd readout: snapshot pre-step velocity so FD covers the outer dt.
        # Gated on state_out.body_qdd allocation (stable across graph capture
        # since the user requests the attribute on the Model before allocating
        # State).
        want_body_qdd = state_out.body_qdd is not None
        if want_body_qdd:
            self._snapshot_pre_step_velocity()

        if self._velocity_readout == "substep_average":
            self._substep_vel_accum.zero_()
            self._substep_omega_accum.zero_()
            world_vel_accum = self._substep_vel_accum
            world_omega_accum = self._substep_omega_accum
        else:
            world_vel_accum = None
            world_omega_accum = None

        # When sleeping is enabled, hand the broad-phase per-shape AABB
        # arrays to the world so it can compute body diagonals for the
        # spin-velocity term of the sleep score. ``narrow_phase`` is
        # populated by ``model.collide(...)`` -- which the caller runs
        # before ``solver.step(...)``.
        shape_aabb_lower = None
        shape_aabb_upper = None
        if self._sleeping_enabled:
            cp = getattr(self.model, "_collision_pipeline", None)
            np_ = getattr(cp, "narrow_phase", None) if cp is not None else None
            shape_aabb_lower = getattr(np_, "shape_aabb_lower", None)
            shape_aabb_upper = getattr(np_, "shape_aabb_upper", None)

        self.world.step(
            dt=float(dt),
            contacts=contacts,
            shape_body=self._shape_body,
            vel_accum=world_vel_accum,
            omega_accum=world_omega_accum,
            shape_aabb_lower=shape_aabb_lower,
            shape_aabb_upper=shape_aabb_upper,
        )
        self._last_dt = float(dt) / max(1, self.world.substeps)

        self._export_body_state(state_out, dt=float(dt))
        self._export_particle_state(state_out)
        self._particle_state_imported = None
        if want_body_qdd:
            self._export_body_qdd(state_out, dt=float(dt))
        # Sync joint_q/joint_qd via eval_ik for policies that read them.
        if state_out.joint_q is not None and state_out.joint_qd is not None and int(self.model.joint_count) > 0:
            newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)

    @staticmethod
    def _read_model_gravity_np(model) -> np.ndarray:
        """Host-side ``model.gravity`` array, defaulting to single-world
        ``[0, 0, -9.81]``. Shared by ``__init__`` (consumed as tuples for
        the PhoenXWorld ctor) and :meth:`notify_model_changed` (consumed
        as a ``wp.array``)."""
        if model.gravity is not None:
            return model.gravity.numpy()
        return np.asarray([[0.0, 0.0, -9.81]], dtype=np.float32)

    def _launch_init_phoenx_bodies(self, model) -> None:
        """Refresh the PhoenX :class:`BodyContainer` from a Newton
        :class:`Model`. Same kernel + inputs/outputs at ``__init__`` and
        :meth:`notify_model_changed`'s body-property refresh path."""
        wp.launch(
            _init_phoenx_body_container_kernel,
            dim=int(model.body_count) + 1,
            inputs=[
                model.body_inv_mass,
                model.body_inv_inertia,
                model.body_com,
                model.body_flags,
                model.body_world,
                wp.int32(int(BodyFlags.KINEMATIC)),
            ],
            outputs=[
                self.bodies.inverse_mass,
                self.bodies.inverse_inertia,
                self.bodies.inverse_inertia_world,
                self.bodies.body_com,
                self.bodies.affected_by_gravity,
                self.bodies.motion_type,
                self.bodies.world_id,
                self.bodies.linear_damping,
                self.bodies.angular_damping,
            ],
            device=self.device,
        )

    def notify_model_changed(self, flags: int) -> None:
        """Refresh state on Model edits. Joint-property changes rebuild the ADBS
        init arrays from scratch; gravity is reread from ``model.gravity``."""
        joint_props_changed = bool(
            flags & (int(SolverNotifyFlags.JOINT_PROPERTIES) | int(SolverNotifyFlags.JOINT_DOF_PROPERTIES))
        )
        if joint_props_changed:
            self._adbs = build_adbs_init_arrays(self.model, device=self.device)
            if self._adbs.num_joint_columns > 0:
                self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())
        if flags & int(SolverNotifyFlags.MODEL_PROPERTIES):
            self.world.gravity = wp.array(self._read_model_gravity_np(self.model), dtype=wp.vec3f, device=self.device)
        # Single body refresh kernel covers both BODY_PROPERTIES and BODY_INERTIAL_PROPERTIES.
        # Joint-DOF changes also force a refresh: armature is baked into body
        # inertia at construction, so any change to model.joint_armature
        # requires resetting the inertia back to Model.body_inv_inertia and
        # re-baking with the new armature. Without this path, domain
        # randomization of armature silently keeps the original values.
        body_refresh_mask = int(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES | SolverNotifyFlags.BODY_PROPERTIES)
        if (flags & body_refresh_mask) or joint_props_changed:
            if self.model.body_count > 0:
                self._launch_init_phoenx_bodies(self.model)
                # Re-bake armature: the kernel just overwrote inertia.
                self._bake_joint_armature_into_body_inertia(self.model)
        if flags & int(SolverNotifyFlags.SHAPE_PROPERTIES):
            if self.model.shape_material_mu is not None and self.model.shape_count > 0:
                self._install_shape_materials()

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Write per-contact wrenches back to
        :attr:`Contacts.force` if the user opted in via
        :meth:`Model.request_contact_attributes('force')`.

        Forces are reported at the contact point in world frame;
        torque is always zero (a per-point force has no torque about
        its own application point). When the compound-body grouping
        optimization is active, the writeback honors the ingest sort
        permutation so ``contacts.force[k]`` aligns with
        ``contacts.rigid_contact_shape0[k]`` and
        ``contacts.rigid_contact_normal[k]`` -- matching the layout
        consumed by :class:`~newton.sensors.SensorContact`.
        """
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if self._last_dt <= 0.0:
            contacts.force.zero_()
            return

        cc = self.world._contact_container
        # Body-pair grouping keys cc by sorted_k; sort_perm maps to newton_k.
        scratch = self.world._ingest_scratch
        if scratch is not None and scratch.sort_perm is not None:
            sort_perm = scratch.sort_perm
            has_perm = wp.int32(1)
        else:
            sort_perm = self._sort_perm_placeholder
            has_perm = wp.int32(0)
        contacts.force.zero_()
        wp.launch(
            _contact_impulse_to_force_wrapper_kernel,
            dim=int(contacts.rigid_contact_max),
            inputs=[
                contacts.rigid_contact_count,
                cc,
                wp.float32(1.0 / self._last_dt),
                sort_perm,
                has_perm,
            ],
            outputs=[contacts.force],
            device=self.device,
        )

    def step_report(self) -> PhoenXWorld.StepReport:
        """Diagnostic snapshot. Forwards to :meth:`PhoenXWorld.step_report`.
        Triggers D2H copies — not graph-capture safe."""
        return self.world.step_report()
