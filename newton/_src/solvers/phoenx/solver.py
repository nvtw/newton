# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PhoenX solver wrapped in Newton's :class:`SolverBase` interface.

Drives :class:`PhoenXWorld` from Newton's standard
:class:`Model` / :class:`State` / :class:`Control` / :class:`Contacts`
surface.  Constraint and contact storage stays column-major inside
PhoenX; only the per-step body state (pose + twist + wrench) is
round-tripped through Newton's SoA.

One-time cost at construction:
    * allocate ``body_count + 1`` PhoenX body buffers (slot 0 is the
      static world anchor); copy ``inv_mass`` / ``inv_inertia`` /
      ``body_com`` / ``world_id`` / flags from Model;
    * walk ``model.joint_*`` to build the 19 ADBS init arrays and
      stamp one constraint column per supported joint (REVOLUTE,
      PRISMATIC, BALL, FIXED). FREE joints get no column.

Per-step cost on top of PhoenX's own step:
    * one import kernel (``State.body_q`` / ``body_qd`` / ``body_f``
      -> PhoenX fields; and a one-shot wrench accumulation from
      ``control.joint_f`` via the stock Newton ``apply_joint_forces``
      kernel);
    * one joint-control writeback kernel that rewrites the per-joint
      drive dwords (``target`` / ``target_velocity`` / ``stiffness``
      / ``damping`` / ``max_force_drive``) from ``Control`` +
      ``Model`` gains;
    * one export kernel (PhoenX -> ``State.body_q`` / ``body_qd``).

PhoenX's internal substep loop is preserved; the outer caller can
still substep by setting ``substeps=1``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.sim import BodyFlags, Contacts, Control, Model, State
from newton._src.solvers.flags import SolverNotifyFlags
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
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
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    write_float,
    write_int,
)
from newton._src.solvers.phoenx.model_adapter import (
    AdbsInitArrays,
    build_adbs_init_arrays,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld
from newton._src.solvers.solver import SolverBase
from newton._src.solvers.xpbd.kernels import apply_joint_forces

__all__ = ["SolverPhoenX"]


# ---------------------------------------------------------------------------
# One-time model-to-PhoenX property copy kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _init_phoenx_body_container_kernel(
    # Model inputs (length N = model.body_count).
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33f],
    body_com: wp.array[wp.vec3f],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    # Flag bit Newton uses to mark kinematic bodies.
    kinematic_flag: wp.int32,
    # Outputs (length N + 1; slot 0 is the static world anchor).
    inv_mass_out: wp.array[wp.float32],
    inv_inertia_out: wp.array[wp.mat33f],
    inv_inertia_world_out: wp.array[wp.mat33f],
    body_com_out: wp.array[wp.vec3f],
    affected_by_gravity_out: wp.array[wp.int32],
    motion_type_out: wp.array[wp.int32],
    world_id_out: wp.array[wp.int32],
    linear_damping_out: wp.array[wp.float32],
    angular_damping_out: wp.array[wp.float32],
):
    """One-shot copy of static body properties from Model to PhoenX
    slots. Slot 0 is the static world anchor (mass 0, no gravity);
    slots [1, N+1) mirror Newton body ``tid - 1``."""
    tid = wp.tid()
    if tid == 0:
        zero_mat = wp.mat33f(0.0)
        inv_mass_out[0] = 0.0
        inv_inertia_out[0] = zero_mat
        inv_inertia_world_out[0] = zero_mat
        body_com_out[0] = wp.vec3f(0.0, 0.0, 0.0)
        affected_by_gravity_out[0] = 0
        motion_type_out[0] = MOTION_STATIC
        world_id_out[0] = 0
        linear_damping_out[0] = 1.0
        angular_damping_out[0] = 1.0
        return

    i = tid - 1
    inv_mass_out[tid] = body_inv_mass[i]
    inv_inertia_out[tid] = body_inv_inertia[i]
    # Seed inverse_inertia_world with the body-frame matrix; the first
    # _update_inertia launch rotates it by the current orientation.
    inv_inertia_world_out[tid] = body_inv_inertia[i]
    body_com_out[tid] = body_com[i]
    flags = body_flags[i]
    if (flags & kinematic_flag) != 0:
        motion_type_out[tid] = MOTION_KINEMATIC
        affected_by_gravity_out[tid] = 0
    elif body_inv_mass[i] == 0.0:
        motion_type_out[tid] = MOTION_STATIC
        affected_by_gravity_out[tid] = 0
    else:
        motion_type_out[tid] = MOTION_DYNAMIC
        affected_by_gravity_out[tid] = 1

    linear_damping_out[tid] = 1.0
    angular_damping_out[tid] = 1.0

    w = body_world[i]
    if w < 0:
        world_id_out[tid] = 0
    else:
        world_id_out[tid] = w


# ---------------------------------------------------------------------------
# Per-step import / export of body state
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _import_body_state_kernel(
    # Newton State inputs (length N = model.body_count).
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3f],
    # PhoenX body container outputs (slot index = tid + 1).
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
    force: wp.array[wp.vec3f],
    torque: wp.array[wp.vec3f],
):
    """Unpack Newton's ``body_q`` / ``body_qd`` / ``body_f`` into
    PhoenX's SoA body slots.

    Newton conventions:
        * ``body_q.translation`` is the body origin in world frame;
          PhoenX stores COM-in-world, so we add ``R * body_com``.
        * ``body_qd`` = ``(v_com_world, omega_world)`` -- already in
          PhoenX's convention (COM-linear, world-angular).
        * ``body_f`` = wrench at the COM in world frame -- matches
          PhoenX's ``force`` / ``torque`` semantics.

    Writes go to slot ``tid + 1`` because slot 0 is the static world
    anchor (never touched by this kernel).
    """
    tid = wp.tid()
    dst = tid + 1
    q = body_q[tid]
    rot = wp.transform_get_rotation(q)
    origin = wp.transform_get_translation(q)
    position[dst] = origin + wp.quat_rotate(rot, body_com[tid])
    orientation[dst] = rot
    qd = body_qd[tid]
    velocity[dst] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[dst] = wp.vec3f(qd[3], qd[4], qd[5])
    wrench = body_f[tid]
    force[dst] = wp.vec3f(wrench[0], wrench[1], wrench[2])
    torque[dst] = wp.vec3f(wrench[3], wrench[4], wrench[5])


@wp.kernel(enable_backward=False)
def _export_body_state_kernel(
    # PhoenX body container (slot index = tid + 1).
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
    body_com: wp.array[wp.vec3f],
    # Newton State outputs (length N = model.body_count).
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pack PhoenX's SoA body slots back into Newton's ``body_q`` /
    ``body_qd``. Reverses :func:`_import_body_state_kernel`."""
    tid = wp.tid()
    src = tid + 1
    rot = orientation[src]
    com_world = position[src]
    origin = com_world - wp.quat_rotate(rot, body_com[tid])
    body_q[tid] = wp.transform(origin, rot)
    body_qd[tid] = wp.spatial_vector(velocity[src], angular_velocity[src])


# ---------------------------------------------------------------------------
# Per-step control -> ADBS column writeback
# ---------------------------------------------------------------------------
#
# The ADBS column already carries per-joint ``target`` / ``target_vel``
# / ``stiffness_drive`` / ``damping_drive`` / ``max_force_drive`` /
# ``drive_mode`` fields. Rather than pass them through an extra array,
# we rewrite those column dwords in place every step from the user's
# ``Control`` and ``Model`` drive-gain arrays. Joints whose
# ``target_mode == EFFORT`` (or ``NONE``) end up with
# ``DRIVE_MODE_OFF`` and their torque is applied via
# :func:`apply_joint_forces` -> ``state.body_f`` instead.


@wp.kernel(enable_backward=False)
def _apply_joint_control_kernel(
    # Per-joint lookup tables (length = model.joint_count).
    joint_idx_to_cid: wp.array[wp.int32],
    joint_idx_to_dof_start: wp.array[wp.int32],
    # Newton Model + Control (per-DOF).
    joint_target_mode: wp.array[wp.int32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    control_target_pos: wp.array[wp.float32],
    control_target_vel: wp.array[wp.float32],
    # Drive-mode constants.
    mode_off: wp.int32,
    mode_position: wp.int32,
    mode_velocity: wp.int32,
    target_mode_position: wp.int32,
    target_mode_velocity: wp.int32,
    target_mode_position_velocity: wp.int32,
    # ADBS column offsets.
    off_drive_mode: wp.int32,
    off_target: wp.int32,
    off_target_velocity: wp.int32,
    off_stiffness_drive: wp.int32,
    off_damping_drive: wp.int32,
    off_max_force_drive: wp.int32,
    # Constraint container to rewrite.
    constraints: ConstraintContainer,
):
    """Per-joint writeback of drive knobs into the ADBS column.

    One thread per Newton joint; joints with ``cid == -1`` (FREE base,
    disabled, unsupported) skip out immediately. For supported joints
    this copies the current-frame Control targets and Model gains into
    the ADBS column's ``target`` / ``target_velocity`` /
    ``stiffness_drive`` / ``damping_drive`` / ``max_force_drive`` /
    ``drive_mode`` dwords.
    """
    j = wp.tid()
    cid = joint_idx_to_cid[j]
    if cid < 0:
        return
    dof = joint_idx_to_dof_start[j]
    if dof < 0:
        # FIXED / BALL joints have no 1-axis DOF mapping; nothing to write.
        return

    tm = joint_target_mode[dof]
    stiffness = joint_target_ke[dof]
    damping = joint_target_kd[dof]
    target = control_target_pos[dof]
    target_vel = control_target_vel[dof]
    effort = joint_effort_limit[dof]

    # Mode mapping:
    # POSITION / POSITION_VELOCITY with ke > 0 -> PhoenX POSITION.
    # VELOCITY with kd > 0 -> PhoenX VELOCITY. Anything else -> OFF.
    drive = mode_off
    if tm == target_mode_position or tm == target_mode_position_velocity:
        if stiffness > 0.0:
            drive = mode_position
    elif tm == target_mode_velocity:
        if damping > 0.0:
            drive = mode_velocity

    # Clamp non-finite effort (e.g. inf) to 0 == "unlimited" for
    # PhoenX POSITION drives.
    max_force = effort
    if (max_force != max_force) or (max_force > 1.0e18) or (max_force < -1.0e18):
        max_force = 0.0

    write_int(constraints, off_drive_mode, cid, drive)
    write_float(constraints, off_target, cid, target)
    write_float(constraints, off_target_velocity, cid, target_vel)
    write_float(constraints, off_stiffness_drive, cid, stiffness)
    write_float(constraints, off_damping_drive, cid, damping)
    write_float(constraints, off_max_force_drive, cid, max_force)


# ---------------------------------------------------------------------------
# Contact wrench writeback (mirrors XPBD's pattern)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _contact_impulse_to_force_kernel(
    rigid_contact_count: wp.array[wp.int32],
    # PhoenX accumulated per-contact impulses (read from the
    # ContactContainer).
    lam_n: wp.array[wp.float32],
    lam_t1: wp.array[wp.float32],
    lam_t2: wp.array[wp.float32],
    normal: wp.array[wp.vec3f],
    tangent1: wp.array[wp.vec3f],
    idt: wp.float32,
    # Output: Contacts.force[k] as spatial_vector (top=force, bot=torque).
    # Torque is zero here (per-point force at the contact point itself).
    force_out: wp.array[wp.spatial_vector],
):
    """Convert a PhoenX per-contact impulse triple into a Newton
    spatial-vector force.

    ``lambda_n`` is accumulated along the contact normal; the two
    tangents span the friction plane with ``tangent2 = cross(normal,
    tangent1)``. Output force is ``(lam_n * n + lam_t1 * t1 +
    lam_t2 * t2) / dt``; torque is zero since the force is reported
    at the contact point.
    """
    k = wp.tid()
    n_active = rigid_contact_count[0]
    if k >= n_active:
        return
    n = normal[k]
    t1 = tangent1[k]
    t2 = wp.cross(n, t1)
    f = (lam_n[k] * n + lam_t1[k] * t1 + lam_t2[k] * t2) * idt
    force_out[k] = wp.spatial_vector(f, wp.vec3f(0.0, 0.0, 0.0))


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
        position_iterations: int = 0,
        default_friction: float = 0.5,
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
            position_iterations: XPBD contact tangent-drift sweeps per
                substep.
            default_friction: Fallback friction when the Contacts
                buffer carries no per-contact or per-shape material.
        """
        super().__init__(model)

        # ---- Build the PhoenX body container ---------------------------
        num_bodies_phoenx = int(model.body_count) + 1
        self.bodies: BodyContainer = body_container_zeros(
            num_bodies_phoenx, device=self.device
        )

        # Identity orientation for every slot (including slot 0) so the
        # first _update_inertia doesn't see a zero quaternion.
        self.bodies.orientation.assign(
            np.tile([0.0, 0.0, 0.0, 1.0], (num_bodies_phoenx, 1)).astype(np.float32)
        )

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
                model._collision_pipeline = _newton.CollisionPipeline(
                    model, contact_matching="sticky"
                )
                model._collision_pipeline.contacts()  # forces buffer sizing
        rigid_contact_max = int(model.rigid_contact_max)
        # One constraint column per shape pair covers arbitrary contact
        # counts per pair; guarded by ``max(1, ...)`` for the contact-
        # free case.
        max_contact_columns = max(1, rigid_contact_max)

        # ---- PhoenX gravity: aggregate Model gravity ------------------
        # Model stores gravity per world in ``model.gravity`` with
        # shape (num_worlds, 3). PhoenX takes the same.
        gravity_np = model.gravity.numpy() if model.gravity is not None else np.asarray(
            [[0.0, 0.0, -9.81]], dtype=np.float32
        )
        num_worlds = max(1, int(gravity_np.shape[0]))
        gravity_tuples = [tuple(float(x) for x in row) for row in gravity_np]
        if len(gravity_tuples) == 1:
            gravity_arg = gravity_tuples[0]
        else:
            gravity_arg = gravity_tuples

        # ---- Make the constraint container + world --------------------
        self._constraints: ConstraintContainer = PhoenXWorld.make_constraint_container(
            num_joints=num_joints,
            max_contact_columns=max_contact_columns,
            device=self.device,
        )

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self._constraints,
            substeps=int(substeps),
            solver_iterations=int(solver_iterations),
            velocity_iterations=int(velocity_iterations),
            position_iterations=int(position_iterations),
            gravity=gravity_arg,
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(model.shape_count),
            num_joints=num_joints,
            default_friction=float(default_friction),
            num_worlds=num_worlds,
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
            zero_wrench = wp.zeros(
                int(model.body_count), dtype=wp.spatial_vector, device=self.device
            )
            wp.launch(
                _import_body_state_kernel,
                dim=int(model.body_count),
                inputs=[
                    model.body_q,
                    model.body_qd,
                    zero_wrench,
                    model.body_com,
                ],
                outputs=[
                    self.bodies.position,
                    self.bodies.orientation,
                    self.bodies.velocity,
                    self.bodies.angular_velocity,
                    self.bodies.force,
                    self.bodies.torque,
                ],
                device=self.device,
            )

        if num_joints > 0:
            self.world.initialize_actuated_double_ball_socket_joints(
                **self._adbs.to_initialize_kwargs()
            )

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
            self._shape_body = wp.array(
                shape_body_phoenx, dtype=wp.int32, device=self.device
            )
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
        restitution = self.model.shape_material_restitution.numpy() if self.model.shape_material_restitution is not None else np.zeros_like(mu_np)
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
        target_pos = control.joint_target_pos if control is not None and control.joint_target_pos is not None else model.joint_target_pos
        target_vel = control.joint_target_vel if control is not None and control.joint_target_vel is not None else model.joint_target_vel
        if target_pos is None or target_vel is None or model.joint_target_mode is None:
            return  # no per-DOF drive configured
        wp.launch(
            _apply_joint_control_kernel,
            dim=int(model.joint_count),
            inputs=[
                self._adbs.joint_idx_to_cid,
                self._adbs.joint_idx_to_dof_start,
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
        the PhoenX body container (slot ``i + 1``)."""
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
            ],
            outputs=[
                self.bodies.position,
                self.bodies.orientation,
                self.bodies.velocity,
                self.bodies.angular_velocity,
                self.bodies.force,
                self.bodies.torque,
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
                self.world.initialize_actuated_double_ball_socket_joints(
                    **self._adbs.to_initialize_kwargs()
                )
        if flags & int(SolverNotifyFlags.MODEL_PROPERTIES):
            gravity_np = self.model.gravity.numpy() if self.model.gravity is not None else np.asarray([[0.0, 0.0, -9.81]], dtype=np.float32)
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


# ---------------------------------------------------------------------------
# Contact impulse -> force kernel using the live ContactContainer struct
# ---------------------------------------------------------------------------


from newton._src.solvers.phoenx.constraints.contact_container import (  # noqa: E402
    ContactContainer,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)


@wp.kernel(enable_backward=False)
def _contact_impulse_to_force_wrapper_kernel(
    rigid_contact_count: wp.array[wp.int32],
    cc: ContactContainer,
    idt: wp.float32,
    # out
    force_out: wp.array[wp.spatial_vector],
):
    """Pack ``ContactContainer`` lambdas into ``Contacts.force``.

    ``tangent2 = cross(normal, tangent1)`` is recomputed since the
    container only stores ``normal`` and ``tangent1``.
    """
    k = wp.tid()
    n_active = rigid_contact_count[0]
    if k >= n_active:
        return
    n = cc_get_normal(cc, k)
    t1 = cc_get_tangent1(cc, k)
    t2 = wp.cross(n, t1)
    lam_n = cc_get_normal_lambda(cc, k)
    lam_t1 = cc_get_tangent1_lambda(cc, k)
    lam_t2 = cc_get_tangent2_lambda(cc, k)
    f = (lam_n * n + lam_t1 * t1 + lam_t2 * t2) * idt
    force_out[k] = wp.spatial_vector(f, wp.vec3f(0.0, 0.0, 0.0))
