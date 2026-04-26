# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for the :class:`SolverPhoenX` Newton-API adapter.

Split out of :mod:`solver` so the host-flow file stays focused on
orchestration. These kernels round-trip per-step body state and joint
control between Newton's :class:`Model` / :class:`State` /
:class:`Control` SoA and the column-major PhoenX containers, plus a
contact-impulse readback kernel for :meth:`SolverPhoenX.update_contacts`.
"""

from __future__ import annotations

import warp as wp

from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    ConstraintContainer,
    write_float,
    write_int,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal,
    cc_get_normal_lambda,
    cc_get_tangent1,
    cc_get_tangent1_lambda,
    cc_get_tangent2_lambda,
)

__all__ = [
    "_apply_joint_control_kernel",
    "_contact_impulse_to_force_wrapper_kernel",
    "_export_body_state_kernel",
    "_import_body_state_kernel",
    "_init_phoenx_body_container_kernel",
    "_seed_kinematic_initial_pose_kernel",
]


# ---------------------------------------------------------------------------
# One-time model-to-PhoenX property copy
# ---------------------------------------------------------------------------
#
# Slot 0 of the PhoenX body container is the static world anchor (mass
# 0, no gravity); slots [1, N+1) mirror Newton body ``tid - 1``. The
# motion-type tag is derived from ``body_inv_mass == 0`` (static) and
# the ``KINEMATIC`` flag bit; everything else falls through to dynamic.


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
    # PhoenX body container (slot index = tid + 1).
    bodies: BodyContainer,
):
    """Unpack Newton ``body_q`` / ``body_qd`` / ``body_f`` into PhoenX SoA slots.

    Conventions: PhoenX stores COM-in-world so we add ``R * body_com``
    to Newton's body-origin ``body_q.translation``. ``body_qd =
    (v_com_world, omega_world)`` and ``body_f`` (wrench at COM, world
    frame) already match PhoenX. Writes go to ``tid + 1`` since
    slot 0 is the static anchor.

    Per-body routing by motion type:

    * **Dynamic / static**: pose + velocity go straight into
      ``bodies.position / orientation / velocity / angular_velocity``.
    * **Kinematic**: pose goes to ``bodies.kinematic_target_pos /
      kinematic_target_orient`` with ``kinematic_target_valid = 1``;
      next step's :func:`_kinematic_prepare_step_kernel` infers
      velocity from ``(prev pose, target)`` delta (``body_qd``
      ignored). Lets Newton-API callers script kinematic motion by
      updating ``state.body_q`` without computing velocities.
    """
    tid = wp.tid()
    dst = tid + 1
    q = body_q[tid]
    rot = wp.transform_get_rotation(q)
    origin = wp.transform_get_translation(q)
    pose_origin = origin + wp.quat_rotate(rot, body_com[tid])

    wrench = body_f[tid]
    bodies.force[dst] = wp.vec3f(wrench[0], wrench[1], wrench[2])
    bodies.torque[dst] = wp.vec3f(wrench[3], wrench[4], wrench[5])

    if bodies.motion_type[dst] == MOTION_KINEMATIC:
        bodies.kinematic_target_pos[dst] = pose_origin
        bodies.kinematic_target_orient[dst] = rot
        bodies.kinematic_target_valid[dst] = 1
    else:
        bodies.position[dst] = pose_origin
        bodies.orientation[dst] = rot
        qd = body_qd[tid]
        bodies.velocity[dst] = wp.vec3f(qd[0], qd[1], qd[2])
        bodies.angular_velocity[dst] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def _seed_kinematic_initial_pose_kernel(
    bodies: BodyContainer,
):
    """One-time seed for kinematic bodies during solver setup.

    :func:`_import_body_state_kernel` routes kinematic body poses to
    the ``kinematic_target_*`` slots (so per-step imports don't
    clobber ``position``, which the solver uses as the lerp / slerp
    origin). On the very first import that leaves ``bodies.position``
    unset for kinematic bodies, which would cause the first step's
    velocity inference to teleport the body from ``pos = 0`` to the
    target. This kernel copies the just-imported target into
    ``position`` / ``orientation`` so the invariant holds:
    ``position_prev == position == target`` at step 0 for a
    kinematic body the user hasn't yet scripted.
    """
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return
    bodies.position[i] = bodies.kinematic_target_pos[i]
    bodies.orientation[i] = bodies.kinematic_target_orient[i]
    # No need to reset ``kinematic_target_valid`` -- if the user
    # doesn't call set_kinematic_pose / re-import between setup and
    # the first step, the prepare kernel reads target == position and
    # infers zero velocity; the redundant valid=1 is harmless.


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
# Rewrites per-joint ``target`` / ``target_vel`` / ``stiffness_drive``
# / ``damping_drive`` / ``max_force_drive`` / ``drive_mode`` directly
# into the ADBS column from ``Control`` + ``Model`` drive gains each
# step. ``target_mode == EFFORT`` / ``NONE`` map to ``DRIVE_MODE_OFF``
# and route torque through :func:`apply_joint_forces` -> ``body_f``.


@wp.kernel(enable_backward=False)
def _apply_joint_control_kernel(
    # Per-joint lookup tables (length = model.joint_count).
    joint_idx_to_cid: wp.array[wp.int32],
    joint_idx_to_dof_start: wp.array[wp.int32],
    # Per-ADBS-cid init offset; indexed by the cid each joint maps to.
    # PhoenX's revolution tracker is init-relative, so Newton's
    # absolute ``target_pos`` must be shifted by -joint_q_at_init
    # before being written into the ADBS column.
    joint_q_at_init_per_cid: wp.array[wp.float32],
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
    # ``target`` is absolute in Newton but PhoenX's cumulative angle
    # tracks displacement from init -- offset by the per-cid init
    # coordinate so targets line up.
    target = control_target_pos[dof] - joint_q_at_init_per_cid[cid]
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
# Contact impulse -> Newton Contacts.force readback
# ---------------------------------------------------------------------------


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
