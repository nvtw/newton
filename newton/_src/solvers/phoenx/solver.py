# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PhoenX solver wrapped in Newton's :class:`SolverBase` interface.

Drives :class:`PhoenXWorld` from Newton's standard
``Model`` / ``State`` / ``Control`` / ``Contacts`` surface. Constraint
and contact storage stays column-major inside PhoenX; only per-step
body state (pose + twist + wrench) round-trips through Newton SoA.

Construction: allocate ``body_count + 1`` body buffers (slot 0 =
static world anchor), copy mass / inertia / com / world_id / flags
from ``Model``, walk ``model.joint_*`` to stamp one ADBS column per
supported joint (REVOLUTE, PRISMATIC, BALL, FIXED; FREE gets none).

Per step (on top of PhoenX's own step): one import kernel (Newton ->
PhoenX body fields + ``joint_f`` -> wrenches), one joint-control
writeback kernel (``Control`` + ``Model`` gains -> drive dwords),
one export kernel (PhoenX -> ``State.body_q`` / ``body_qd``).
PhoenX's internal substep loop is preserved; pass ``substeps=1`` to
substep outside.
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
    _apply_joint_control_kernel,
    _contact_impulse_to_force_wrapper_kernel,
    _export_body_state_kernel,
    _import_body_state_kernel,
    _init_phoenx_body_container_kernel,
    _seed_kinematic_initial_pose_kernel,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.xpbd.kernels import apply_joint_forces

__all__ = ["SolverPhoenX"]


# ---------------------------------------------------------------------------
# Contact-buffer sizing for the PhoenX path
# ---------------------------------------------------------------------------


def _estimate_rigid_contact_max_phoenx(model) -> int | None:
    """Tight ``rigid_contact_max`` estimate for the PhoenX solver.

    Uses the precomputed ``model.shape_contact_pair_count`` (already
    ``COLLIDE_SHAPES``-filtered) * primitive-CPP narrow-phase cap * 2
    safety, rather than Newton's default ``num_meshes * 20 * 40``
    which over-counts non-colliding visual meshes by ~7x in humanoid
    scenes. Returns ``None`` when the pair count isn't available so
    the caller falls back to Newton's default.
    """
    pair_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if pair_count <= 0:
        return None

    # PhoenX's narrow phase uses the primitive-contacts path for
    # shape-vs-shape collisions by default; the MESH=40 contacts-per-
    # pair penalty in Newton's default is an opt-in hydroelastic
    # upper bound, not what our GJK / MPR pipeline actually produces.
    PRIMITIVE_CPP = 5
    SAFETY = 2
    return max(1000, pair_count * PRIMITIVE_CPP * SAFETY)


# ---------------------------------------------------------------------------
# SolverPhoenX
# ---------------------------------------------------------------------------


class SolverPhoenX(SolverBase):
    """Newton :class:`SolverBase` wrapper around :class:`PhoenXWorld`.

    Build once from a finalised :class:`Model`; call :meth:`step`
    each frame with :class:`State` / :class:`Control` / :class:`Contacts`.
    PhoenX's internal substep loop is preserved (``substeps``
    controls the count); callers that prefer outer substepping should
    pass ``substeps=1``.

    Supported joint types (automatic, based on
    :attr:`Model.joint_type`):

    * :data:`JointType.REVOLUTE` -- hinge with optional PD drive +
      limit;
    * :data:`JointType.PRISMATIC` -- slider with optional PD drive +
      limit;
    * :data:`JointType.BALL` -- 3-DoF point lock (no drive / limit);
    * :data:`JointType.FIXED` -- 6-DoF weld (no drive / limit);
    * :data:`JointType.FREE` -- free-floating base (no constraint
      column; integration alone handles it).

    :data:`JointType.DISTANCE`, :data:`JointType.D6`,
    :data:`JointType.CABLE` are not supported and raise at construction.
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
    ):
        """Build the PhoenX solver from ``model``.

        Args:
            model: The finalised Newton :class:`Model`. Must be built;
                ``model.body_q`` / ``body_com`` / ``body_inv_mass`` /
                ``body_inv_inertia`` / ``joint_*`` arrays are read
                once here.
            substeps: PhoenX internal substeps per :meth:`step` call.
            solver_iterations: PGS iterations per substep.
            velocity_iterations: TGS-soft relax sweeps per substep.
            default_friction: Fallback friction when the Contacts
                buffer carries no per-contact or per-shape material.
            step_layout: ``"multi_world"`` (default) -- per-world
                fast-tail kernels, one warp per world (scales to
                thousands of small worlds). ``"single_world"`` --
                per-colour grid launches via ``wp.capture_while``
                over the global JP colouring; wins for one or a few
                very big worlds.
            threads_per_world: Effective threads-per-world for the
                multi-world fast-tail kernels. ``"auto"`` (default)
                picks per-step from the colour-size histogram;
                ``32`` = one warp per world (legacy), ``16`` = two,
                ``8`` = four (rarely wins). Graph-capture safe.
        """
        super().__init__(model)

        # ---- Build the PhoenX body container ---------------------------
        num_bodies_phoenx = int(model.body_count) + 1
        self.bodies: BodyContainer = body_container_zeros(num_bodies_phoenx, device=self.device)

        # Identity orientation for every slot (including slot 0) so the
        # first _update_inertia doesn't see a zero quaternion.
        self.bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32))

        # Static property copy. Launches over N + 1 threads (slot 0 is
        # the static anchor).
        if model.body_count:
            wp.launch(
                _init_phoenx_body_container_kernel,
                dim=num_bodies_phoenx,
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

        # ---- Sync model.body_q with model.joint_q ---------------------
        # The adapter snapshots body-local joint anchors from
        # ``model.body_q``; if the caller set non-zero ``builder.joint_q``
        # before ``finalize()`` (as URDF rigs and the Anymal walking
        # example do), ``model.body_q`` is still at the URDF rest pose
        # and FK hasn't run yet. Without this FK pass every joint
        # anchor would be baked at the wrong configuration and the
        # constraint columns would pull bodies back to the rest pose
        # on every step.
        if int(model.body_count) > 0 and int(model.joint_count) > 0:
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)

        # ---- Build the joint (ADBS) column layout ----------------------
        self._adbs: AdbsInitArrays = build_adbs_init_arrays(model, device=self.device)
        num_joints = self._adbs.num_joint_columns

        # ---- Collision-related sizing ---------------------------------
        # PhoenX's warm-start path requires ``contact_matching`` to be
        # non-disabled. If the model doesn't already have a pipeline
        # (or has one with matching off), attach a sticky pipeline so
        # ``model.contacts()`` produces the right buffer. This makes
        # SolverPhoenX self-sized: the user never has to allocate
        # Contacts up-front.
        if int(model.shape_count) > 0:
            existing_cp = getattr(model, "_collision_pipeline", None)
            needs_new_cp = existing_cp is None or not getattr(existing_cp, "contact_matching", False)
            if needs_new_cp:
                import newton as _newton  # local to keep the import cycle tight

                # Override Newton's ``rigid_contact_max`` estimator
                # with a PhoenX-tight one from
                # ``shape_contact_pair_count`` (COLLIDE_SHAPES-filtered).
                # Newton's default ignores the flag when counting
                # meshes, which inflates the budget ~15x in models
                # with visual-only mesh shapes (1.3 GB unused at
                # h1_flat/4096). Clear any stale value first -- the
                # builder may have set one from an earlier
                # ``model.contacts()``.
                tight_rcm = _estimate_rigid_contact_max_phoenx(model)
                if tight_rcm is not None:
                    model.rigid_contact_max = 0  # bypass "already sized" short-circuit
                from newton._src.solvers.phoenx.solver_config import (
                    PHOENX_CONTACT_MATCHING,
                )

                model._collision_pipeline = _newton.CollisionPipeline(
                    model,
                    contact_matching=PHOENX_CONTACT_MATCHING,
                    rigid_contact_max=tight_rcm,
                )
                model._collision_pipeline.contacts()  # forces buffer sizing
        rigid_contact_max = int(model.rigid_contact_max)

        # ---- PhoenX gravity: aggregate Model gravity ------------------
        # Model stores gravity per world in ``model.gravity`` with
        # shape (num_worlds, 3). PhoenX takes the same.
        gravity_np = (
            model.gravity.numpy() if model.gravity is not None else np.asarray([[0.0, 0.0, -9.81]], dtype=np.float32)
        )
        num_worlds = max(1, int(gravity_np.shape[0]))
        gravity_tuples = [tuple(float(x) for x in row) for row in gravity_np]
        if len(gravity_tuples) == 1:
            gravity_arg = gravity_tuples[0]
        else:
            gravity_arg = gravity_tuples

        # ---- Make the constraint container + world --------------------
        # One constraint column per ``(shape_a, shape_b)`` pair covers
        # an arbitrary contact count per pair; size the column buffer
        # 1:1 against ``rigid_contact_max`` (the same number
        # ``PhoenXWorld.__init__`` derives internally).
        self._constraints: ConstraintContainer = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            device=self.device,
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self._constraints,
            substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=int(velocity_iterations),
            gravity=gravity_arg,
            rigid_contact_max=rigid_contact_max,
            num_joints=num_joints,
            default_friction=float(default_friction),
            num_worlds=num_worlds,
            step_layout=step_layout,
            threads_per_world=threads_per_world,
            device=self.device,
        )

        # Seed the PhoenX body container with the model's initial pose
        # (``model.body_q`` / ``body_qd``) BEFORE joint initialization --
        # the ADBS init kernel reads body positions to snapshot
        # body-local anchors. Without this seed every body appears at
        # the origin and welds pull the child body to slot 0's world
        # anchor instead of its intended rest pose.
        if int(model.body_count) > 0:
            # Zero body_f / particle_f on the temp state; only pose/
            # twist matter for joint init.
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
            # Seed kinematic bodies' ``position`` / ``orientation``
            # from the target slots the import kernel just filled.
            # See :func:`_seed_kinematic_initial_pose_kernel` for the
            # why.
            wp.launch(
                _seed_kinematic_initial_pose_kernel,
                dim=int(model.body_count) + 1,  # +1 for slot 0 (world anchor)
                inputs=[self.bodies],
                device=self.device,
            )

        if num_joints > 0:
            self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())

        # Install per-shape materials (friction only for now; Newton's
        # shape_material_mu maps directly).
        if model.shape_material_mu is not None and model.shape_count > 0:
            self._install_shape_materials()

        # ---- Shape -> PhoenX-slot map for contact ingest --------------
        # Newton's shape_body uses -1 for world; PhoenX's slot 0 is
        # the static world anchor.
        if model.shape_body is not None and model.shape_count > 0:
            shape_body_np = model.shape_body.numpy()
            shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
            self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)
        else:
            self._shape_body = None

        # ---- Scratch for apply_joint_forces ---------------------------
        self._has_joint_forces = model.joint_dof_count > 0

        # ---- Cached time step (for contact force reconstruction) ------
        self._last_dt: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _install_shape_materials(self) -> None:
        """Stream Model's per-shape friction into PhoenX's material
        table. Each shape gets a unique material index; the table
        carries ``(mu_static, mu_dynamic, restitution)`` via the
        existing :mod:`materials` plumbing."""
        from newton._src.solvers.phoenx.materials import (
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
        """Rewrite the ADBS column drive dwords from ``control`` +
        ``model``. Joint-count threads; out-of-scope joints early-out."""
        if self._adbs.num_joint_columns == 0:
            return
        model = self.model
        # Fall back to Model-held targets / gains if Control doesn't
        # supply them.
        target_pos = (
            control.joint_target_pos
            if control is not None and control.joint_target_pos is not None
            else model.joint_target_pos
        )
        target_vel = (
            control.joint_target_vel
            if control is not None and control.joint_target_vel is not None
            else model.joint_target_vel
        )
        if target_pos is None or target_vel is None or model.joint_target_mode is None:
            return  # no per-DOF drive configured
        wp.launch(
            _apply_joint_control_kernel,
            dim=int(model.joint_count),
            inputs=[
                self._adbs.joint_idx_to_cid,
                self._adbs.joint_idx_to_dof_start,
                self._adbs.joint_q_at_init,
                model.joint_target_mode,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_effort_limit,
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

    def _accumulate_joint_forces(self, state_in: State, control: Control) -> None:
        """Fold ``control.joint_f`` (generalized joint forces) into
        ``state_in.body_f`` via the stock Newton
        :func:`apply_joint_forces` kernel. This is the EFFORT path
        (joints that pass scalar torque / linear force directly rather
        than via PD drive)."""
        if control is None or control.joint_f is None:
            return
        if not self._has_joint_forces:
            return
        model = self.model
        if model.joint_count == 0:
            return
        wp.launch(
            apply_joint_forces,
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
            ],
            outputs=[state_in.body_f],
            device=self.device,
        )

    def _import_body_state(self, state_in: State) -> None:
        """Pull ``state_in.body_q`` / ``body_qd`` / ``body_f`` into
        the PhoenX body container (slot ``i + 1``).

        Kinematic bodies route their pose to the
        ``kinematic_target_{pos,orient}`` slots with ``valid=1`` so
        the solver infers velocity from the per-step pose delta and
        interpolates between substeps; dynamic / static bodies get
        their pose and velocity written directly. See
        :func:`_import_body_state_kernel` for the branch.
        """
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

    def _export_body_state(self, state_out: State) -> None:
        """Pack PhoenX's body state back into ``state_out.body_q`` /
        ``body_qd``."""
        n = int(self.model.body_count)
        if n == 0:
            return
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance ``state_in`` to ``state_out`` by ``dt``.

        Order:
            1. Fold Control into State: write joint drive targets into
               the ADBS columns; accumulate joint effort forces into
               ``state_in.body_f``.
            2. Import body state (Newton State -> PhoenX body container).
            3. :meth:`PhoenXWorld.step`.
            4. Export body state (PhoenX -> ``state_out``).
        """
        if control is None:
            control = self.model.control()

        self._apply_joint_control(control)
        self._accumulate_joint_forces(state_in, control)
        self._import_body_state(state_in)

        self.world.step(
            dt=float(dt),
            contacts=contacts,
            shape_body=self._shape_body,
        )
        self._last_dt = float(dt) / max(1, self.world.substeps)

        self._export_body_state(state_out)
        # Sync the canonical joint coordinates. Policies that read
        # ``state.joint_q`` / ``state.joint_qd`` (e.g. the Anymal PyTorch
        # rig) need these kept current; eval_ik is the inverse of the
        # FK that produced ``body_q`` / ``body_qd`` in the first place.
        if state_out.joint_q is not None and state_out.joint_qd is not None and int(self.model.joint_count) > 0:
            newton.eval_ik(self.model, state_out, state_out.joint_q, state_out.joint_qd)

    def notify_model_changed(self, flags: int) -> None:
        """Refresh internal state when the caller edits ``Model``.

        We alias most Model arrays directly (``body_inv_mass`` /
        ``body_inv_inertia`` are *copied* once at construction so
        slot 0 can hold the world anchor; all other arrays are read
        per-step). Joint-property changes rebuild the ADBS init arrays
        from scratch. Gravity is reread from ``model.gravity``.
        """
        if flags & (int(SolverNotifyFlags.JOINT_PROPERTIES) | int(SolverNotifyFlags.JOINT_DOF_PROPERTIES)):
            self._adbs = build_adbs_init_arrays(self.model, device=self.device)
            if self._adbs.num_joint_columns > 0:
                self.world.initialize_actuated_double_ball_socket_joints(**self._adbs.to_initialize_kwargs())
        if flags & int(SolverNotifyFlags.MODEL_PROPERTIES):
            gravity_np = (
                self.model.gravity.numpy()
                if self.model.gravity is not None
                else np.asarray([[0.0, 0.0, -9.81]], dtype=np.float32)
            )
            self.world.gravity = wp.array(gravity_np, dtype=wp.vec3f, device=self.device)
        if flags & int(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            # Refresh the copied inv_mass / inv_inertia slots.
            if self.model.body_count > 0:
                wp.launch(
                    _init_phoenx_body_container_kernel,
                    dim=self.model.body_count + 1,
                    inputs=[
                        self.model.body_inv_mass,
                        self.model.body_inv_inertia,
                        self.model.body_com,
                        self.model.body_flags,
                        self.model.body_world,
                        wp.int32(int(BodyFlags.NO_GRAVITY)),
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

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Write per-contact wrenches back to
        :attr:`Contacts.force` if the user opted in via
        :meth:`Model.request_contact_attributes('force')`.

        Forces are reported at the contact point in world frame;
        torque is always zero (a per-point force has no torque about
        its own application point).
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
        # PhoenX stores lambda_n / lambda_t1 / lambda_t2 as 1D float
        # arrays inside the lambdas 2D buffer; extract them via the
        # existing cc_get_* helpers. We launch one thread per contact
        # slot and decompose the impulse into (n, t1, t2) world axes.
        contacts.force.zero_()
        wp.launch(
            _contact_impulse_to_force_wrapper_kernel,
            dim=int(contacts.rigid_contact_max),
            inputs=[
                contacts.rigid_contact_count,
                cc,
                wp.float32(1.0 / self._last_dt),
            ],
            outputs=[contacts.force],
            device=self.device,
        )
