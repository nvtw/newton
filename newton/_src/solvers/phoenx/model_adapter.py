# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side conversion from Newton :class:`Model` joints to ADBS init arrays.

Mapping: REVOLUTE/PRISMATIC/BALL/FIXED/CABLE -> ADBS joint modes; FREE -> no
column; DISTANCE/D6 unsupported. PhoenX slot 0 is the static world anchor, so
Newton body ``i`` maps to PhoenX slot ``i + 1`` and ``joint_parent == -1`` maps
to slot 0.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_actuated_double_ball_socket import (
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_FIXED,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
)
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
)

__all__ = [
    "AdbsInitArrays",
    "build_adbs_init_arrays",
]


def _transform_translation(t: np.ndarray) -> np.ndarray:
    """Translation of a ``wp.transform`` stored as ``[px, py, pz, qx, qy, qz, qw]``."""
    return np.asarray(t[:3], dtype=np.float32)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q = [qx, qy, qz, qw]``."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # t = 2 * cross(q.xyz, v); v' = v + qw*t + cross(q.xyz, t).
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    vx = v[0] + qw * tx + (qy * tz - qz * ty)
    vy = v[1] + qw * ty + (qz * tx - qx * tz)
    vz = v[2] + qw * tz + (qx * ty - qy * tx)
    return np.asarray([vx, vy, vz], dtype=np.float32)


def _transform_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two ``wp.transform`` represented as 7-float arrays."""
    out = np.zeros(7, dtype=np.float32)
    # Translation: a.rotation @ b.translation + a.translation.
    out[:3] = _quat_rotate_np(a[3:], b[:3]) + a[:3]
    # Quaternion multiply: q_out = a.rotation * b.rotation.
    ax, ay, az, aw = a[3], a[4], a[5], a[6]
    bx, by, bz, bw = b[3], b[4], b[5], b[6]
    out[3] = aw * bx + ax * bw + ay * bz - az * by
    out[4] = aw * by - ax * bz + ay * bw + az * bx
    out[5] = aw * bz + ax * by - ay * bx + az * bw
    out[6] = aw * bw - ax * bx - ay * by - az * bz
    return out


class AdbsInitArrays:
    """ADBS init kwargs plus joint-index -> cid map for per-step control writeback.
    ``joint_idx_to_cid`` is ``-1`` for joints without a constraint column."""

    def __init__(
        self,
        *,
        body1: wp.array,
        body2: wp.array,
        anchor1: wp.array,
        anchor2: wp.array,
        hertz: wp.array,
        damping_ratio: wp.array,
        joint_mode: wp.array,
        drive_mode: wp.array,
        target: wp.array,
        target_velocity: wp.array,
        max_force_drive: wp.array,
        stiffness_drive: wp.array,
        damping_drive: wp.array,
        min_value: wp.array,
        max_value: wp.array,
        hertz_limit: wp.array,
        damping_ratio_limit: wp.array,
        stiffness_limit: wp.array,
        damping_limit: wp.array,
        armature: wp.array,
        joint_idx_to_cid: wp.array,
        joint_idx_to_dof_start: wp.array,
        joint_q_at_init: wp.array,
        drive_cid: wp.array,
        drive_dof_start: wp.array,
        drive_target_q_index: wp.array,
        drive_q_at_init: wp.array,
        num_joint_columns: int,
        num_drive_columns: int,
    ):
        self.body1 = body1
        self.body2 = body2
        self.anchor1 = anchor1
        self.anchor2 = anchor2
        self.hertz = hertz
        self.damping_ratio = damping_ratio
        self.joint_mode = joint_mode
        self.drive_mode = drive_mode
        self.target = target
        self.target_velocity = target_velocity
        self.max_force_drive = max_force_drive
        self.stiffness_drive = stiffness_drive
        self.damping_drive = damping_drive
        self.min_value = min_value
        self.max_value = max_value
        self.hertz_limit = hertz_limit
        self.damping_ratio_limit = damping_ratio_limit
        self.stiffness_limit = stiffness_limit
        self.damping_limit = damping_limit
        self.armature = armature
        self.joint_idx_to_cid = joint_idx_to_cid
        self.joint_idx_to_dof_start = joint_idx_to_dof_start
        #: Per-ADBS-column initial Newton joint coordinate. PhoenX measures
        #: displacement from init, so Newton's absolute target/limit values
        #: must be offset by this before being written into the ADBS column.
        self.joint_q_at_init = joint_q_at_init
        self.drive_cid = drive_cid
        self.drive_dof_start = drive_dof_start
        self.drive_target_q_index = drive_target_q_index
        self.drive_q_at_init = drive_q_at_init
        self.num_joint_columns = num_joint_columns
        self.num_drive_columns = num_drive_columns

    def to_initialize_kwargs(self) -> dict:
        """Kwargs for
        :meth:`PhoenXWorld.initialize_actuated_double_ball_socket_joints`."""
        return {
            "body1": self.body1,
            "body2": self.body2,
            "anchor1": self.anchor1,
            "anchor2": self.anchor2,
            "hertz": self.hertz,
            "damping_ratio": self.damping_ratio,
            "joint_mode": self.joint_mode,
            "drive_mode": self.drive_mode,
            "target": self.target,
            "target_velocity": self.target_velocity,
            "max_force_drive": self.max_force_drive,
            "stiffness_drive": self.stiffness_drive,
            "damping_drive": self.damping_drive,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "hertz_limit": self.hertz_limit,
            "damping_ratio_limit": self.damping_ratio_limit,
            "stiffness_limit": self.stiffness_limit,
            "damping_limit": self.damping_limit,
            "armature": self.armature,
        }


def _newton_target_mode_to_adbs_drive_mode(target_mode: int, stiffness: float, damping: float) -> int:
    """Map Newton :class:`JointTargetMode` to PhoenX :class:`DriveMode`. POSITION/
    VELOCITY drive modes require positive stiffness/damping respectively, else OFF."""
    mode = newton.JointTargetMode(int(target_mode))
    if mode in (
        newton.JointTargetMode.POSITION,
        newton.JointTargetMode.POSITION_VELOCITY,
    ):
        if stiffness > 0.0:
            return int(DRIVE_MODE_POSITION)
        return int(DRIVE_MODE_OFF)
    if mode is newton.JointTargetMode.VELOCITY:
        if damping > 0.0:
            return int(DRIVE_MODE_VELOCITY)
        return int(DRIVE_MODE_OFF)
    # NONE / EFFORT
    return int(DRIVE_MODE_OFF)


def build_adbs_init_arrays(
    model: newton.Model,
    device: wp.context.Devicelike | None = None,
) -> AdbsInitArrays:
    """Convert ``model``'s joints to ADBS init arrays on ``device``.

    Raises:
        NotImplementedError: If any joint is DISTANCE or D6.
    """
    if device is None:
        device = model.device

    n_joints = int(model.joint_count)
    if n_joints == 0:
        empty_i = wp.zeros(0, dtype=wp.int32, device=device)
        empty_v = wp.zeros(0, dtype=wp.vec3f, device=device)
        empty_f = wp.zeros(0, dtype=wp.float32, device=device)
        joint_idx_to_cid = wp.zeros(0, dtype=wp.int32, device=device)
        joint_idx_to_dof_start = wp.zeros(0, dtype=wp.int32, device=device)
        return AdbsInitArrays(
            body1=empty_i,
            body2=empty_i,
            anchor1=empty_v,
            anchor2=empty_v,
            hertz=empty_f,
            damping_ratio=empty_f,
            joint_mode=empty_i,
            drive_mode=empty_i,
            target=empty_f,
            target_velocity=empty_f,
            max_force_drive=empty_f,
            stiffness_drive=empty_f,
            damping_drive=empty_f,
            min_value=empty_f,
            max_value=empty_f,
            hertz_limit=empty_f,
            damping_ratio_limit=empty_f,
            stiffness_limit=empty_f,
            damping_limit=empty_f,
            armature=empty_f,
            joint_idx_to_cid=joint_idx_to_cid,
            joint_idx_to_dof_start=joint_idx_to_dof_start,
            joint_q_at_init=empty_f,
            drive_cid=empty_i,
            drive_dof_start=empty_i,
            drive_target_q_index=empty_i,
            drive_q_at_init=empty_f,
            num_joint_columns=0,
            num_drive_columns=0,
        )

    # ---- Pull every relevant joint array back to host ----------------
    joint_type = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_X_p = model.joint_X_p.numpy()  # (N, 7) float32
    joint_X_c = model.joint_X_c.numpy()  # (N, 7) float32 -- child attachment
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_target_q_start = model.joint_target_q_start.numpy()
    joint_axis = model.joint_axis.numpy() if model.joint_axis is not None else np.zeros((0, 3), dtype=np.float32)
    body_q = model.body_q.numpy()  # (body_count, 7)
    joint_q_arr = model.joint_q.numpy() if model.joint_q is not None else np.zeros(0, dtype=np.float32)

    # Per-DOF arrays (may be None on minimal models).
    def _pull_dof_f(arr):
        return arr.numpy() if arr is not None else None

    def _pull_dof_i(arr):
        return arr.numpy() if arr is not None else None

    target_mode = _pull_dof_i(model.joint_target_mode)
    target_pos = _pull_dof_f(model.joint_target_q)
    target_vel = _pull_dof_f(model.joint_target_qd)
    target_ke = _pull_dof_f(model.joint_target_ke)
    target_kd = _pull_dof_f(model.joint_target_kd)
    joint_armature = _pull_dof_f(model.joint_armature)
    effort_limit = _pull_dof_f(model.joint_effort_limit)
    limit_lower = _pull_dof_f(model.joint_limit_lower)
    limit_upper = _pull_dof_f(model.joint_limit_upper)
    joint_enabled = model.joint_enabled.numpy() if model.joint_enabled is not None else np.ones(n_joints, dtype=bool)

    # ---- Walk joints --------------------------------------------------
    descriptors: list[dict] = []
    joint_idx_to_cid_np = np.full(n_joints, -1, dtype=np.int32)
    joint_idx_to_dof_start_np = np.full(n_joints, -1, dtype=np.int32)
    joint_idx_to_target_q_index_np = np.full(n_joints, -1, dtype=np.int32)

    for j in range(n_joints):
        jtype = newton.JointType(int(joint_type[j]))

        # FREE and disabled joints don't get a constraint column.
        if not bool(joint_enabled[j]):
            continue
        if jtype is newton.JointType.FREE:
            continue
        if jtype in (
            newton.JointType.DISTANCE,
            newton.JointType.D6,
        ):
            raise NotImplementedError(
                f"PhoenX does not support JointType.{jtype.name} (joint {j}). "
                "Supported: REVOLUTE, PRISMATIC, BALL, FIXED, CABLE, FREE (no column)."
            )

        parent_idx = int(joint_parent[j])
        child_idx = int(joint_child[j])
        # Shift Newton body indices to PhoenX slots (slot 0 = static world).
        phoenx_parent = 0 if parent_idx < 0 else parent_idx + 1
        phoenx_child = 0 if child_idx < 0 else child_idx + 1

        # Joint world transform at init: pose_p * joint_X_p[j].
        if parent_idx < 0:
            X_w_p = np.asarray(joint_X_p[j], dtype=np.float32)
        else:
            X_w_p = _transform_multiply(
                np.asarray(body_q[parent_idx], dtype=np.float32), np.asarray(joint_X_p[j], dtype=np.float32)
            )

        anchor1_world = _transform_translation(X_w_p)
        qd_start = int(joint_qd_start[j])
        # FIXED/BALL have no 1-axis DoF; -1 lets the control kernel skip them.
        dof_start_for_control = qd_start if jtype in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC) else -1
        target_q_index_for_control = -1
        if dof_start_for_control >= 0:
            target_q_index_for_control = int(joint_target_q_start[j])
        joint_idx_to_dof_start_np[j] = dof_start_for_control
        joint_idx_to_target_q_index_np[j] = target_q_index_for_control

        # Per-mode anchor2 and drive/limit defaults.
        anchor2_world = anchor1_world.copy()
        drive_mode = int(DRIVE_MODE_OFF)
        target_val = 0.0
        target_vel_val = 0.0
        stiff_drive = 0.0
        damp_drive = 0.0
        max_force = 0.0
        min_val = 1.0  # disabled: min > max
        max_val = -1.0
        stiff_limit = 0.0
        damp_limit = 0.0
        hertz_limit_val = float(DEFAULT_HERTZ_LIMIT)
        damping_ratio_limit_val = float(DEFAULT_DAMPING_RATIO)
        # Armature only applies to REVOLUTE/PRISMATIC axial rows; 0 elsewhere.
        armature_val = 0.0

        if jtype is newton.JointType.BALL:
            phoenx_mode = int(JOINT_MODE_BALL_SOCKET)
        elif jtype is newton.JointType.CABLE:
            phoenx_mode = int(JOINT_MODE_CABLE)
            # Newton CABLE has 2 DoFs (linear stretch + isotropic angular bend/twist).
            # PhoenX cable is a soft fixed joint (3+2+1 rows) with PD bend/twist; the
            # axial bond is treated as rigid (PhoenX has no axial compliance) and
            # Newton's isotropic angular gain feeds both bend AND twist slots.
            # If anchor1 and anchor2 coincide, synthesize a 1 m offset along the
            # joint X axis so the bend basis stays well-defined.
            if child_idx >= 0:
                X_w_c = _transform_multiply(
                    np.asarray(body_q[child_idx], dtype=np.float32),
                    np.asarray(joint_X_c[j], dtype=np.float32),
                )
            else:  # pragma: no cover
                X_w_c = np.asarray(joint_X_c[j], dtype=np.float32)
            anchor2_world = _transform_translation(X_w_c)
            if float(np.linalg.norm(anchor2_world - anchor1_world)) < 1e-6:
                axis_world = _quat_rotate_np(X_w_p[3:], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
                anchor2_world = anchor1_world + axis_world

            # Bend/twist gains live on the angular DoF (qd_start + 1).
            bend_qd = qd_start + 1
            bend_ke = float(target_ke[bend_qd]) if (target_ke is not None and bend_qd < len(target_ke)) else 0.0
            bend_kd = float(target_kd[bend_qd]) if (target_kd is not None and bend_qd < len(target_kd)) else 0.0
            stiff_drive = bend_ke
            damp_drive = bend_kd
            stiff_limit = bend_ke
            damp_limit = bend_kd
        elif jtype is newton.JointType.FIXED:
            phoenx_mode = int(JOINT_MODE_FIXED)
            # Pick joint-frame X axis so the anchor-3 basis is well-defined.
            axis_world = _quat_rotate_np(X_w_p[3:], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            anchor2_world = anchor1_world + axis_world
        elif jtype is newton.JointType.REVOLUTE or jtype is newton.JointType.PRISMATIC:
            phoenx_mode = int(JOINT_MODE_REVOLUTE) if jtype is newton.JointType.REVOLUTE else int(JOINT_MODE_PRISMATIC)
            axis_local = (
                np.asarray(joint_axis[qd_start], dtype=np.float32)
                if len(joint_axis)
                else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            )
            axis_len = float(np.linalg.norm(axis_local))
            if axis_len > 1e-12:
                axis_local = axis_local / axis_len
            else:
                axis_local = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            axis_world = _quat_rotate_np(X_w_p[3:], axis_local)
            anchor2_world = anchor1_world + axis_world

            # Drive / limit from per-DOF arrays (first DoF only for the
            # supported 1-DoF joints).
            if target_ke is not None:
                stiff_drive = float(target_ke[qd_start])
            if target_kd is not None:
                damp_drive = float(target_kd[qd_start])
            if target_pos is not None and target_q_index_for_control < len(target_pos):
                target_val = float(target_pos[target_q_index_for_control])
            if target_vel is not None:
                target_vel_val = float(target_vel[qd_start])
            if effort_limit is not None:
                # PhoenX reads 0 as "unlimited" for POSITION drives, so clamp inf/NaN to 0.
                raw = float(effort_limit[qd_start])
                max_force = raw if np.isfinite(raw) else 0.0
            if target_mode is not None:
                drive_mode = _newton_target_mode_to_adbs_drive_mode(int(target_mode[qd_start]), stiff_drive, damp_drive)
            # Limits are hard stops via DEFAULT_HERTZ_LIMIT (matches SolverXPBD's
            # rigid-limit contract; Newton's limit_ke/limit_kd are XPBD-only soft
            # penalties that don't map to PhoenX's absolute SI PD path). Users who
            # want soft PD limits should drive ADBS init directly.
            if limit_lower is not None and limit_upper is not None:
                lo = float(limit_lower[qd_start])
                hi = float(limit_upper[qd_start])
                if lo <= hi:
                    min_val = lo
                    max_val = hi
            if joint_armature is not None and qd_start < len(joint_armature):
                armature_val = float(joint_armature[qd_start])
        else:  # pragma: no cover -- defensive
            raise NotImplementedError(f"joint {j}: unhandled joint type {jtype}")

        # Init joint coord for this joint's first DOF. BALL/FIXED publish 0 to
        # keep the per-joint array length aligned with the ADBS column array.
        q_start_idx = int(joint_q_start[j])
        if jtype in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC) and len(joint_q_arr) > q_start_idx:
            init_q = float(joint_q_arr[q_start_idx])
        else:
            init_q = 0.0

        # Offset limit window into PhoenX's cumulative-from-init coordinate.
        if min_val <= max_val:
            min_val -= init_q
            max_val -= init_q

        descriptors.append(
            {
                "body1": phoenx_parent,
                "body2": phoenx_child,
                "anchor1": anchor1_world,
                "anchor2": anchor2_world,
                "hertz": float(DEFAULT_HERTZ_LINEAR),
                "damping_ratio": float(DEFAULT_DAMPING_RATIO),
                "joint_mode": phoenx_mode,
                "drive_mode": drive_mode,
                # Init-offset target for first step; per-step control kernel
                # re-applies the offset on each control update.
                "target": target_val - init_q,
                "target_velocity": target_vel_val,
                "max_force_drive": max_force,
                "stiffness_drive": stiff_drive,
                "damping_drive": damp_drive,
                "min_value": min_val,
                "max_value": max_val,
                "hertz_limit": hertz_limit_val,
                "damping_ratio_limit": damping_ratio_limit_val,
                "stiffness_limit": stiff_limit,
                "damping_limit": damp_limit,
                "armature": armature_val,
                "joint_q_at_init": init_q,
            }
        )
        joint_idx_to_cid_np[j] = len(descriptors) - 1

    # ---- Upload --------------------------------------------------------
    num_cols = len(descriptors)
    drive_mask = (joint_idx_to_cid_np >= 0) & (joint_idx_to_dof_start_np >= 0)
    drive_cid_np = joint_idx_to_cid_np[drive_mask].astype(np.int32, copy=False)
    drive_dof_start_np = joint_idx_to_dof_start_np[drive_mask].astype(np.int32, copy=False)
    drive_target_q_index_np = joint_idx_to_target_q_index_np[drive_mask].astype(np.int32, copy=False)
    if drive_cid_np.size:
        drive_q_at_init_np = np.asarray(
            [descriptors[int(cid)]["joint_q_at_init"] for cid in drive_cid_np], dtype=np.float32
        )
    else:
        drive_q_at_init_np = np.zeros(0, dtype=np.float32)

    def _stack_i(key: str) -> wp.array:
        a = np.asarray([d[key] for d in descriptors], dtype=np.int32) if num_cols else np.zeros(0, dtype=np.int32)
        return wp.array(a, dtype=wp.int32, device=device)

    def _stack_f(key: str) -> wp.array:
        a = np.asarray([d[key] for d in descriptors], dtype=np.float32) if num_cols else np.zeros(0, dtype=np.float32)
        return wp.array(a, dtype=wp.float32, device=device)

    def _stack_v(key: str) -> wp.array:
        a = (
            np.asarray([d[key] for d in descriptors], dtype=np.float32).reshape(-1, 3)
            if num_cols
            else np.zeros((0, 3), dtype=np.float32)
        )
        return wp.array(a, dtype=wp.vec3f, device=device)

    return AdbsInitArrays(
        body1=_stack_i("body1"),
        body2=_stack_i("body2"),
        anchor1=_stack_v("anchor1"),
        anchor2=_stack_v("anchor2"),
        hertz=_stack_f("hertz"),
        damping_ratio=_stack_f("damping_ratio"),
        joint_mode=_stack_i("joint_mode"),
        drive_mode=_stack_i("drive_mode"),
        target=_stack_f("target"),
        target_velocity=_stack_f("target_velocity"),
        max_force_drive=_stack_f("max_force_drive"),
        stiffness_drive=_stack_f("stiffness_drive"),
        damping_drive=_stack_f("damping_drive"),
        min_value=_stack_f("min_value"),
        max_value=_stack_f("max_value"),
        hertz_limit=_stack_f("hertz_limit"),
        damping_ratio_limit=_stack_f("damping_ratio_limit"),
        stiffness_limit=_stack_f("stiffness_limit"),
        damping_limit=_stack_f("damping_limit"),
        armature=_stack_f("armature"),
        joint_idx_to_cid=wp.array(joint_idx_to_cid_np, dtype=wp.int32, device=device),
        joint_idx_to_dof_start=wp.array(joint_idx_to_dof_start_np, dtype=wp.int32, device=device),
        joint_q_at_init=_stack_f("joint_q_at_init"),
        drive_cid=wp.array(drive_cid_np, dtype=wp.int32, device=device),
        drive_dof_start=wp.array(drive_dof_start_np, dtype=wp.int32, device=device),
        drive_target_q_index=wp.array(drive_target_q_index_np, dtype=wp.int32, device=device),
        drive_q_at_init=wp.array(drive_q_at_init_np, dtype=wp.float32, device=device),
        num_joint_columns=num_cols,
        num_drive_columns=int(drive_cid_np.size),
    )
