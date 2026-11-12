# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for the :class:`SolverPhoenX` Newton-API adapter.

Round-trips body state and joint control between Newton's :class:`Model` /
:class:`State` / :class:`Control` SoA and the column-major PhoenX containers,
plus contact-impulse readback for :meth:`SolverPhoenX.update_contacts`.
"""

from __future__ import annotations

import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.phoenx.body import (
    MOTION_DYNAMIC,
    MOTION_KINEMATIC,
    MOTION_STATIC,
    BodyContainer,
    sym6_from_mat33,
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
from newton._src.solvers.phoenx.helpers.math_helpers import rotate_inertia

__all__ = [
    "_accumulate_contact_impulse_kernel",
    "_apply_joint_drive_control_kernel",
    "_apply_joint_forces_kernel",
    "_contact_impulse_to_force_wrapper_kernel",
    "_export_body_qdd_kernel",
    "_export_body_state_kernel",
    "_import_body_state_kernel",
    "_init_phoenx_body_container_kernel",
    "_seed_kinematic_initial_pose_kernel",
    "_snapshot_pre_step_velocity_kernel",
]


@wp.func
def write_body_properties(
    bodies: BodyContainer,
    slot: wp.int32,
    inv_mass: wp.float32,
    inv_inertia: wp.mat33f,
    com: wp.vec3f,
    kinematic: wp.bool,
    world: wp.int32,
):
    """Write one slot's static properties; shared by model import and runtime body creation."""
    bodies.inverse_mass[slot] = inv_mass
    bodies.inverse_inertia[slot] = inv_inertia
    # First _update_inertia launch rotates this by current orientation.
    bodies.inverse_inertia_world[slot] = sym6_from_mat33(inv_inertia)
    bodies.body_com[slot] = com
    if kinematic:
        # Prescribed motion has no impulse response, even when the model
        # retains physical mass and inertia for this body.
        bodies.inverse_mass[slot] = 0.0
        bodies.inverse_inertia[slot] = wp.mat33f(0.0)
        bodies.inverse_inertia_world[slot] = sym6_from_mat33(wp.mat33f(0.0))
        bodies.motion_type[slot] = MOTION_KINEMATIC
        bodies.affected_by_gravity[slot] = 0
    elif inv_mass == 0.0:
        bodies.motion_type[slot] = MOTION_STATIC
        bodies.affected_by_gravity[slot] = 0
    else:
        bodies.motion_type[slot] = MOTION_DYNAMIC
        bodies.affected_by_gravity[slot] = 1
    bodies.linear_damping[slot] = 1.0
    bodies.angular_damping[slot] = 1.0
    bodies.world_id[slot] = wp.max(world, 0)


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
    # Output (length N + 1; slot 0 is the static world anchor).
    bodies: BodyContainer,
):
    """One-shot copy of static body properties from Model to PhoenX slots.
    Slot 0 is the static world anchor; slots [1, N+1) mirror Newton body tid-1."""
    tid = wp.tid()
    if tid == 0:
        write_body_properties(bodies, 0, 0.0, wp.mat33f(0.0), wp.vec3f(0.0), False, 0)
        return
    i = tid - 1
    write_body_properties(
        bodies,
        tid,
        body_inv_mass[i],
        body_inv_inertia[i],
        body_com[i],
        (body_flags[i] & kinematic_flag) != 0,
        body_world[i],
    )


@wp.kernel(enable_backward=False)
def _import_body_state_kernel(
    # Newton State inputs (length N = model.body_count).
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3f],
    # PhoenX body container (slot index = tid + 1).
    bodies: BodyContainer,
    # Per-slot flag of bodies created at runtime since the last import (may be empty).
    fresh: wp.array[wp.int32],
    consume_fresh: wp.int32,
):
    """Unpack Newton ``body_q`` / ``body_qd`` / ``body_f`` into PhoenX SoA slots.

    PhoenX stores COM-in-world (adds ``R * body_com`` to Newton's body-origin).
    Kinematic bodies route pose to ``kinematic_target_*`` with valid=1; the next
    step's prepare kernel infers velocity from the (prev, target) delta.
    Bodies created at runtime keep the pose and velocity they were created with,
    since ``body_q`` and ``body_qd`` still hold the previous occupant's state.
    """
    tid = wp.tid()
    dst = tid + 1
    wrench = body_f[tid]
    bodies.force[dst] = wp.vec3f(wrench[0], wrench[1], wrench[2])
    bodies.torque[dst] = wp.vec3f(wrench[3], wrench[4], wrench[5])
    if fresh.shape[0] > 0:
        if fresh[dst] != 0:
            if consume_fresh != 0:
                fresh[dst] = 0
            return

    q = body_q[tid]
    rot = wp.transform_get_rotation(q)
    origin = wp.transform_get_translation(q)
    pose_origin = origin + wp.quat_rotate(rot, body_com[tid])

    if bodies.motion_type[dst] == MOTION_KINEMATIC:
        bodies.kinematic_target_pos[dst] = pose_origin
        bodies.kinematic_target_orient[dst] = rot
        bodies.kinematic_target_valid[dst] = 1
    else:
        bodies.position[dst] = pose_origin
        bodies.orientation[dst] = rot
        rotation = wp.quat_to_matrix(rot)
        bodies.inverse_inertia_world[dst] = sym6_from_mat33(rotate_inertia(rotation, bodies.inverse_inertia[dst]))
        qd = body_qd[tid]
        bodies.velocity[dst] = wp.vec3f(qd[0], qd[1], qd[2])
        bodies.angular_velocity[dst] = wp.vec3f(qd[3], qd[4], qd[5])


# A continued step already owns pose and velocity; only external wrenches change.
@wp.kernel(enable_backward=False)
def _import_body_forces_kernel(
    body_f: wp.array[wp.spatial_vector],
    bodies: BodyContainer,
):
    body = wp.tid()
    wrench = body_f[body]
    slot = body + wp.int32(1)
    bodies.force[slot] = wp.vec3f(wrench[0], wrench[1], wrench[2])
    bodies.torque[slot] = wp.vec3f(wrench[3], wrench[4], wrench[5])


@wp.kernel(enable_backward=False)
def _seed_kinematic_initial_pose_kernel(
    bodies: BodyContainer,
):
    """One-time seed for kinematic bodies during solver setup. Copies the just-
    imported target into position/orientation so the first step doesn't teleport
    from pos=0 to target."""
    i = wp.tid()
    if bodies.motion_type[i] != MOTION_KINEMATIC:
        return
    bodies.position[i] = bodies.kinematic_target_pos[i]
    bodies.orientation[i] = bodies.kinematic_target_orient[i]


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


@wp.kernel(enable_backward=False)
def _export_body_state_avg_kernel(
    # PhoenX body container (slot index = tid + 1).
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    body_com: wp.array[wp.vec3f],
    # Time-integrated velocity accumulators (length = N).
    vel_accum: wp.array[wp.vec3f],
    omega_accum: wp.array[wp.vec3f],
    inv_dt: wp.float32,
    # Newton State outputs.
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Substep-averaged velocity readout. Pose is substep-end; velocity is
    accumulator * inv_dt = time-averaged over the outer step."""
    tid = wp.tid()
    src = tid + 1
    rot = orientation[src]
    com_world = position[src]
    origin = com_world - wp.quat_rotate(rot, body_com[tid])
    body_q[tid] = wp.transform(origin, rot)
    v_avg = vel_accum[tid] * inv_dt
    w_avg = omega_accum[tid] * inv_dt
    body_qd[tid] = wp.spatial_vector(v_avg, w_avg)


@wp.kernel(enable_backward=False)
def _accumulate_substep_velocity_kernel(
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
    substep_dt: wp.float32,
    # Accumulators; length = N (one slot per Newton body, slot 0 of
    # PhoenX is the static anchor and isn't reflected back).
    vel_accum: wp.array[wp.vec3f],
    omega_accum: wp.array[wp.vec3f],
):
    """Accumulate ``velocity * dt`` per substep for time-averaged readout
    (matches MuJoCo Warp's post-integration qvel convention)."""
    tid = wp.tid()
    src = tid + 1  # PhoenX slot 0 is the static world anchor.
    vel_accum[tid] = vel_accum[tid] + velocity[src] * substep_dt
    omega_accum[tid] = omega_accum[tid] + angular_velocity[src] * substep_dt


@wp.kernel(enable_backward=False)
def _snapshot_pre_step_velocity_kernel(
    # PhoenX body container (length N + 1 with slot 0 = anchor).
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
    # Snapshot buffers (length N -- one per Newton body, slot 0 omitted).
    vel_prev_out: wp.array[wp.vec3f],
    omega_prev_out: wp.array[wp.vec3f],
):
    """Snapshot pre-step COM-velocity for the body_qdd readout. Captured right
    after :func:`_import_body_state_kernel` so the FD covers the full outer dt."""
    tid = wp.tid()
    src = tid + 1
    vel_prev_out[tid] = velocity[src]
    omega_prev_out[tid] = angular_velocity[src]


@wp.kernel(enable_backward=False)
def _export_body_qdd_kernel(
    # PhoenX body container (slot index = tid + 1).
    velocity: wp.array[wp.vec3f],
    angular_velocity: wp.array[wp.vec3f],
    # Pre-step snapshot from :func:`_snapshot_pre_step_velocity_kernel`.
    vel_prev: wp.array[wp.vec3f],
    omega_prev: wp.array[wp.vec3f],
    inv_dt: wp.float32,
    # Newton State output.
    body_qdd: wp.array[wp.spatial_vector],
):
    """FD-derived ``body_qdd`` from the outer-step velocity delta:
    ``a = (qd_now - qd_prev) / dt``. Newton convention: ``spatial_top`` is
    linear acceleration (world frame, includes gravity-induced terms),
    ``spatial_bottom`` is angular acceleration (world frame). Matches the
    sign convention consumed by :class:`~newton.sensors.SensorIMU`."""
    tid = wp.tid()
    src = tid + 1
    lin_acc = (velocity[src] - vel_prev[tid]) * inv_dt
    ang_acc = (angular_velocity[src] - omega_prev[tid]) * inv_dt
    body_qdd[tid] = wp.spatial_vector(lin_acc, ang_acc)


@wp.kernel(enable_backward=False)
def _snapshot_pre_step_pose_kernel(
    # PhoenX body container (length N + 1 with slot 0 = anchor).
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    # Snapshot buffers (length N -- one per Newton body, slot 0 omitted).
    pos_prev_out: wp.array[wp.vec3f],
    orient_prev_out: wp.array[wp.quatf],
):
    """Snapshot pre-step pose for the FD velocity readout
    (``velocity_readout="finite_difference"``)."""
    tid = wp.tid()
    src = tid + 1
    pos_prev_out[tid] = position[src]
    orient_prev_out[tid] = orientation[src]


@wp.func
def _quat_log_axis_angle(q: wp.quatf) -> wp.vec3f:
    """Quaternion log -> ``axis * angle`` rotation vector. Picks the shorter
    rotation; collapses to ``2 * q.xyz`` when |xyz| is small."""
    qw = q[3]
    if qw < wp.float32(0.0):
        q = wp.quatf(-q[0], -q[1], -q[2], -q[3])
        qw = q[3]
    xyz = wp.vec3f(q[0], q[1], q[2])
    s = wp.length(xyz)
    if s < wp.float32(1.0e-8):
        return xyz * wp.float32(2.0)
    half_angle = wp.atan2(s, qw)
    return xyz * (wp.float32(2.0) * half_angle / s)


@wp.kernel(enable_backward=False)
def _export_body_state_fd_kernel(
    # PhoenX body container (slot index = tid + 1).
    position: wp.array[wp.vec3f],
    orientation: wp.array[wp.quatf],
    body_com: wp.array[wp.vec3f],
    # Pre-step snapshot from :func:`_snapshot_pre_step_pose_kernel`.
    pos_prev: wp.array[wp.vec3f],
    orient_prev: wp.array[wp.quatf],
    inv_dt: wp.float32,
    # Newton State outputs.
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """FD-derived ``body_qd`` from the outer-step pose delta:
    v_com_fd = (com_now - com_prev) / dt; omega_fd = log(q_now * inv(q_prev)) / dt.
    ``body_q`` is the post-substep pose."""
    tid = wp.tid()
    src = tid + 1
    rot = orientation[src]
    com_now = position[src]
    com_prev = pos_prev[tid]
    rot_prev = orient_prev[tid]

    origin = com_now - wp.quat_rotate(rot, body_com[tid])
    body_q[tid] = wp.transform(origin, rot)

    v_com_fd = (com_now - com_prev) * inv_dt
    delta_q = rot * wp.quat_inverse(rot_prev)
    omega_fd = _quat_log_axis_angle(delta_q) * inv_dt
    body_qd[tid] = wp.spatial_vector(v_com_fd, omega_fd)


# Per-step control -> joint constraint column writeback. EFFORT/NONE map to DRIVE_MODE_OFF
# and route torque through _apply_joint_forces_kernel -> body_f.


@wp.kernel(enable_backward=False)
def _apply_joint_forces_kernel(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_axis: wp.array[wp.vec3],
    joint_f: wp.array[wp.float32],
    body_f: wp.array[wp.spatial_vector],
):
    """Fold generalized joint forces into Newton body wrenches.

    This mirrors XPBD's ``apply_joint_forces`` math, but keeps PhoenX's
    per-step adapter path independent from XPBD's larger kernel module.
    """
    tid = wp.tid()
    type = joint_type[tid]
    if not joint_enabled[tid]:
        return
    if type == JointType.FIXED or type == JointType.ROD:
        return

    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    qd_start = joint_qd_start[tid]
    lin_axis_count = joint_dof_dim[tid, 0]
    ang_axis_count = joint_dof_dim[tid, 1]

    t_total = wp.vec3()
    f_total = wp.vec3()

    if type == JointType.FREE or type == JointType.DISTANCE:
        f_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
        t_total = wp.vec3(joint_f[qd_start + 3], joint_f[qd_start + 4], joint_f[qd_start + 5])
        wp.atomic_add(body_f, id_c, wp.spatial_vector(f_total, t_total))
        if id_p >= 0:
            wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total))
        return
    elif type == JointType.BALL:
        t_total = wp.vec3(joint_f[qd_start + 0], joint_f[qd_start + 1], joint_f[qd_start + 2])
    elif type == JointType.REVOLUTE or type == JointType.PRISMATIC or type == JointType.D6:
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            f = joint_f[qd_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            f = joint_f[qd_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            f = joint_f[qd_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += f * a_p

        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            f = joint_f[qd_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            f = joint_f[qd_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            f = joint_f[qd_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += f * a_p
    else:
        return

    child_wrench_at_com = wp.spatial_vector(f_total, t_total + wp.cross(r_c, f_total))
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(f_total, t_total + wp.cross(r_p, f_total)))
    wp.atomic_add(body_f, id_c, child_wrench_at_com)


@wp.kernel(enable_backward=False)
def _apply_joint_drive_control_kernel(
    drive_cid: wp.array[wp.int32],
    drive_dof_start: wp.array[wp.int32],
    drive_target_q_index: wp.array[wp.int32],
    drive_q_at_init: wp.array[wp.float32],
    # Newton Model + Control. Position targets may use coord layout.
    joint_target_mode: wp.array[wp.int32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_effort_limit: wp.array[wp.float32],
    joint_gear: wp.array[wp.float32],
    control_target_pos: wp.array[wp.float32],
    control_target_vel: wp.array[wp.float32],
    # Drive-mode constants.
    mode_off: wp.int32,
    mode_position: wp.int32,
    mode_velocity: wp.int32,
    target_mode_position: wp.int32,
    target_mode_velocity: wp.int32,
    target_mode_position_velocity: wp.int32,
    # joint constraint column offsets.
    off_drive_mode: wp.int32,
    off_target: wp.int32,
    off_target_velocity: wp.int32,
    off_stiffness_drive: wp.int32,
    off_damping_drive: wp.int32,
    off_max_force_drive: wp.int32,
    # Constraint container to rewrite.
    constraints: ConstraintContainer,
):
    """Uniform per-drive writeback into joint constraint columns.

    The compact lookup arrays contain only active scalar drive rows, so every
    thread follows the same path and reads contiguous drive metadata.
    """
    i = wp.tid()
    cid = drive_cid[i]
    dof = drive_dof_start[i]
    target_q = drive_target_q_index[i]

    tm = joint_target_mode[dof]
    stiffness = joint_target_ke[dof]
    damping = joint_target_kd[dof]
    target = control_target_pos[target_q] - drive_q_at_init[i]
    target_vel = control_target_vel[dof]
    effort = joint_effort_limit[dof]

    gear = joint_gear[dof]
    if (gear != gear) or gear <= 0.0:
        gear = 1.0

    drive = mode_off
    if tm == target_mode_position or tm == target_mode_position_velocity:
        if stiffness > 0.0:
            drive = mode_position
    elif tm == target_mode_velocity:
        if damping > 0.0:
            drive = mode_velocity

    max_force = gear * effort
    if (max_force != max_force) or (max_force > 1.0e18) or (max_force < -1.0e18):
        max_force = 0.0

    write_int(constraints, off_drive_mode, cid, drive)
    write_float(constraints, off_target, cid, target)
    write_float(constraints, off_target_velocity, cid, target_vel)
    write_float(constraints, off_stiffness_drive, cid, stiffness)
    write_float(constraints, off_damping_drive, cid, damping)
    write_float(constraints, off_max_force_drive, cid, max_force)


@wp.func
def _active_contact_count(rigid_contact_count: wp.array[wp.int32], capacity: wp.int32) -> wp.int32:
    # On narrow-phase overflow the raw counter keeps climbing past the
    # contact buffer, so clamp it to the buffer capacity.
    return wp.min(rigid_contact_count[0], capacity)


@wp.func
def _contact_impulse(cc: ContactContainer, k: wp.int32) -> wp.vec3f:
    """World-frame impulse of contact ``k`` on shape1 in the current substep."""
    n = cc_get_normal(cc, k)
    t1 = cc_get_tangent1(cc, k)
    t2 = wp.cross(n, t1)
    return cc_get_normal_lambda(cc, k) * n + cc_get_tangent1_lambda(cc, k) * t1 + cc_get_tangent2_lambda(cc, k) * t2


@wp.kernel(enable_backward=False)
def _accumulate_contact_impulse_kernel(
    rigid_contact_count: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
    cc: ContactContainer,
    impulse_sum: wp.array[wp.vec3f],
):
    """Add each contact's substep impulse to its total over the step."""
    k = wp.tid()
    if k >= _active_contact_count(rigid_contact_count, impulse_sum.shape[0]) or cid_of_contact[k] < wp.int32(0):
        return
    impulse_sum[k] = impulse_sum[k] + _contact_impulse(cc, k)


@wp.kernel(enable_backward=False)
def _contact_impulse_to_force_wrapper_kernel(
    rigid_contact_count: wp.array[wp.int32],
    cid_of_contact: wp.array[wp.int32],
    cc: ContactContainer,
    impulse_sum: wp.array[wp.vec3f],
    idt: wp.float32,
    sort_perm: wp.array[wp.int32],
    has_perm: wp.int32,
    shape0: wp.array[wp.int32],
    sorted_shape0: wp.array[wp.int32],
    # out
    force_out: wp.array[wp.spatial_vector],
):
    """Pack contact impulses into ``Contacts.force``.

    ``impulse_sum`` holds each contact's impulse summed over the step's
    substeps when the step recorded it, and ``idt`` is then the inverse
    step ``dt``; otherwise it is empty and the last substep's lambdas are
    divided by its ``dt``. Re-permutes from sorted_k (compound-body
    grouping) to newton narrow-phase order via ``sort_perm``. Sign:
    container lambdas are impulse on shape1; Newton stores force on
    shape0, so we negate before writing and restore the narrow-phase
    endpoint orientation when compound grouping reversed it.
    """
    k = wp.tid()
    # ``sort_perm`` may be a size-1 placeholder when ``has_perm == 0``, so
    # size against ``force_out`` (always ``rigid_contact_max`` long).
    if k >= _active_contact_count(rigid_contact_count, force_out.shape[0]) or cid_of_contact[k] < wp.int32(0):
        return
    if impulse_sum.shape[0] > 0:
        f = -impulse_sum[k] * idt
    else:
        f = -_contact_impulse(cc, k) * idt
    if has_perm != 0:
        out_k = sort_perm[k]
        if sorted_shape0[k] != shape0[out_k]:
            f = -f
    else:
        out_k = k
    force_out[out_k] = wp.spatial_vector(f, wp.vec3f(0.0, 0.0, 0.0))
