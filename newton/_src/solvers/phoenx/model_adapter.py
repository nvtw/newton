# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side conversion from Newton :class:`Model` joints to joint constraint init arrays.

Mapping: REVOLUTE/PRISMATIC/BALL/FIXED/CABLE/DISTANCE -> joint modes;
FREE -> no column. D6 joints use exact restored modes where possible,
including two- and three-axis Cartesian translation modes. PhoenX
slot 0 is the static world anchor, so Newton body ``i`` maps to PhoenX slot
``i + 1`` and ``joint_parent == -1`` maps to slot 0.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.constraints.constraint_container import (
    DEFAULT_DAMPING_RATIO,
    DEFAULT_HERTZ_LIMIT,
    DEFAULT_HERTZ_LINEAR,
)
from newton._src.solvers.phoenx.constraints.constraint_joint import (
    DRIVE_MODE_OFF,
    DRIVE_MODE_POSITION,
    DRIVE_MODE_VELOCITY,
    JOINT_MODE_BALL_SOCKET,
    JOINT_MODE_CABLE,
    JOINT_MODE_CARTESIAN,
    JOINT_MODE_CARTESIAN_PLANE,
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_DISTANCE,
    JOINT_MODE_FIXED,
    JOINT_MODE_GENERIC_D6,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
)

__all__ = [
    "JointInitArrays",
    "build_joint_init_arrays",
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


def _cross3_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cross product of two host-side 3-vectors."""
    return np.asarray(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float32,
    )


def _norm3_np(v: np.ndarray) -> float:
    """Return the Euclidean norm of a host-side 3-vector."""
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


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


def _transform_multiply_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose batches of ``wp.transform`` arrays using the scalar arithmetic contract."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError("transform batches must have matching shape (N, 7)")

    out = np.empty_like(b)
    # Match :func:`_quat_rotate_np`: it promotes scalar components to Python
    # floats, then stores the rotated vector in FP32 before adding translation.
    q = a[:, 3:].astype(np.float64)
    v = b[:, :3].astype(np.float64)
    t = np.empty_like(v)
    t[:, 0] = 2.0 * (q[:, 1] * v[:, 2] - q[:, 2] * v[:, 1])
    t[:, 1] = 2.0 * (q[:, 2] * v[:, 0] - q[:, 0] * v[:, 2])
    t[:, 2] = 2.0 * (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0])
    rotated = np.empty_like(v)
    rotated[:, 0] = v[:, 0] + q[:, 3] * t[:, 0] + (q[:, 1] * t[:, 2] - q[:, 2] * t[:, 1])
    rotated[:, 1] = v[:, 1] + q[:, 3] * t[:, 1] + (q[:, 2] * t[:, 0] - q[:, 0] * t[:, 2])
    rotated[:, 2] = v[:, 2] + q[:, 3] * t[:, 2] + (q[:, 0] * t[:, 1] - q[:, 1] * t[:, 0])
    out[:, :3] = rotated.astype(np.float32) + a[:, :3]

    ax, ay, az, aw = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    bx, by, bz, bw = b[:, 3], b[:, 4], b[:, 5], b[:, 6]
    out[:, 3] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 4] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 5] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 6] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _is_locked_dof(limit_lower: np.ndarray | None, limit_upper: np.ndarray | None, qd: int) -> bool:
    """Return whether a scalar joint DoF uses Newton's locked sentinel."""
    if limit_lower is None or limit_upper is None or qd >= len(limit_lower) or qd >= len(limit_upper):
        return False
    return float(limit_lower[qd]) > float(limit_upper[qd])


def _classify_d6_legacy_mode(
    n_lin: int,
    n_ang: int,
    locked_lin: list[bool],
    locked_ang: list[bool],
) -> tuple[str | None, int]:
    """Map D6 axes to the supported joint constraint mode set.

    Missing axes are locked by construction in Newton's D6 kinematics.
    ``dof_offset`` is the scalar DoF within the D6 joint used for the
    reduced REVOLUTE/PRISMATIC axis. For UNIVERSAL, ``dof_offset`` is
    the locked angular axis when it is explicitly authored, or ``-1``
    for MJCF-style two-axis angular-only D6 joints.
    """
    lin_free = [i for i, locked in enumerate(locked_lin) if not locked]
    ang_free = [i for i, locked in enumerate(locked_ang) if not locked]

    if not lin_free and not ang_free:
        return "FIXED", -1

    if not lin_free and len(ang_free) == 3:
        return "BALL", -1

    if n_lin == 0 and n_ang == 2 and len(ang_free) == 2:
        return "UNIVERSAL", -1

    if n_lin == 3 and n_ang == 3 and not lin_free and len(ang_free) == 2:
        locked_ang_idx = next(i for i, locked in enumerate(locked_ang) if locked)
        return "UNIVERSAL", n_lin + locked_ang_idx

    if not lin_free and len(ang_free) == 1:
        return "REVOLUTE", n_lin + ang_free[0]

    if len(lin_free) == 1 and not ang_free:
        return "PRISMATIC", lin_free[0]

    if len(lin_free) == 1 and len(ang_free) == 1:
        return "CYLINDRICAL", -1

    if len(lin_free) == 2 and len(ang_free) == 1:
        return "PLANAR", -1

    if len(lin_free) == 2 and not ang_free:
        return "CARTESIAN_PLANE", -1

    if len(lin_free) == 3 and not ang_free:
        return "CARTESIAN", -1

    return None, -1


def _friction_slip_scale_from_mujoco(solref: np.ndarray | None, solimp: np.ndarray | None) -> float:
    """Return MuJoCo friction slip scale from ``solreffriction/solimpfriction``.

    MuJoCo frictionloss rows use ``R = (1 - impedance) / impedance * dA`` and
    ``B = 2 / (dmax * timeconst)`` for positive-format ``solref``. PhoenX does
    not know the row inverse effective mass until prepare, so the adapter stores
    ``R / (B * dA)`` and the device row later multiplies by the current axial
    inverse effective mass and friction limit.
    """

    if solref is None or solimp is None:
        return -1.0
    solref = np.asarray(solref, dtype=np.float32).reshape(-1)
    solimp = np.asarray(solimp, dtype=np.float32).reshape(-1)
    if len(solref) < 2 or len(solimp) < 2:
        return -1.0
    imp = float(np.clip(float(solimp[0]), 0.0001, 0.9999))
    dmax = float(np.clip(float(solimp[1]), 0.0001, 0.9999))
    timeconst = float(solref[0])
    direct_damping = float(solref[1])
    if timeconst > 0.0:
        damping = 2.0 / max(1.0e-15, dmax * timeconst)
    elif direct_damping < 0.0:
        damping = -direct_damping / max(1.0e-15, dmax)
    else:
        return -1.0
    return float(((1.0 - imp) / max(1.0e-15, imp)) / max(1.0e-15, damping))


def _append_d6_angular_limit(
    qd: int,
    coord_offset: int,
    *,
    limit_lower: np.ndarray | None,
    limit_upper: np.ndarray | None,
    joint_axis: np.ndarray,
    joint_q_arr: np.ndarray,
    joint_q_start: np.ndarray,
    joint_index: int,
    joint_world_xform: np.ndarray,
    d6_limit_axes: list[np.ndarray],
    d6_limit_lower: np.ndarray,
    d6_limit_upper: np.ndarray,
    d6_limit_count: int,
) -> int:
    """Pack one finite D6 angular limit and return the next write slot."""
    if d6_limit_count >= 3 or limit_lower is None or limit_upper is None:
        return d6_limit_count
    if qd < 0 or qd >= len(limit_lower) or qd >= len(limit_upper):
        return d6_limit_count
    lo = float(limit_lower[qd])
    hi = float(limit_upper[qd])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        return d6_limit_count
    if lo <= -2.0 * np.pi and hi >= 2.0 * np.pi:
        return d6_limit_count
    axis_local = (
        np.asarray(joint_axis[qd], dtype=np.float32)
        if len(joint_axis) and qd < len(joint_axis)
        else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    )
    axis_len = _norm3_np(axis_local)
    if axis_len <= 1.0e-12:
        return d6_limit_count
    axis_world = _quat_rotate_np(joint_world_xform[3:], axis_local / axis_len)
    q_idx = int(joint_q_start[joint_index]) + coord_offset
    init_axis_q = float(joint_q_arr[q_idx]) if len(joint_q_arr) > q_idx else 0.0
    d6_limit_axes[d6_limit_count] = axis_world
    d6_limit_lower[d6_limit_count] = lo - init_axis_q
    d6_limit_upper[d6_limit_count] = hi - init_axis_q
    return d6_limit_count + 1


class JointInitArrays:
    """joint constraint init kwargs plus joint-index -> cid map for per-step control writeback.
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
        velocity_limit: wp.array,
        max_force_drive: wp.array,
        stiffness_drive: wp.array,
        damping_drive: wp.array,
        min_value: wp.array,
        max_value: wp.array,
        hertz_limit: wp.array,
        damping_ratio_limit: wp.array,
        stiffness_limit: wp.array,
        damping_limit: wp.array,
        friction_coefficient: wp.array,
        friction_slip_scale: wp.array,
        d6_limit_axis0: wp.array,
        d6_limit_axis1: wp.array,
        d6_limit_axis2: wp.array,
        d6_limit_lower: wp.array,
        d6_limit_upper: wp.array,
        d6_limit_count: wp.array,
        joint_idx_to_cid: wp.array,
        joint_idx_to_dof_start: wp.array,
        joint_q_at_init: wp.array,
        drive_cid: wp.array,
        drive_dof_start: wp.array,
        drive_target_q_index: wp.array,
        drive_q_at_init: wp.array,
        num_joint_columns: int,
        num_drive_columns: int,
        has_velocity_limits: bool,
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
        self.velocity_limit = velocity_limit
        self.max_force_drive = max_force_drive
        self.stiffness_drive = stiffness_drive
        self.damping_drive = damping_drive
        self.min_value = min_value
        self.max_value = max_value
        self.hertz_limit = hertz_limit
        self.damping_ratio_limit = damping_ratio_limit
        self.stiffness_limit = stiffness_limit
        self.damping_limit = damping_limit
        self.friction_coefficient = friction_coefficient
        self.friction_slip_scale = friction_slip_scale
        self.d6_limit_axis0 = d6_limit_axis0
        self.d6_limit_axis1 = d6_limit_axis1
        self.d6_limit_axis2 = d6_limit_axis2
        self.d6_limit_lower = d6_limit_lower
        self.d6_limit_upper = d6_limit_upper
        self.d6_limit_count = d6_limit_count
        self.joint_idx_to_cid = joint_idx_to_cid
        self.joint_idx_to_dof_start = joint_idx_to_dof_start
        #: Per-joint-column initial Newton joint coordinate. PhoenX measures
        #: displacement from init, so Newton's absolute target/limit values
        #: must be offset by this before being written into the joint constraint column.
        self.joint_q_at_init = joint_q_at_init
        self.drive_cid = drive_cid
        self.drive_dof_start = drive_dof_start
        self.drive_target_q_index = drive_target_q_index
        self.drive_q_at_init = drive_q_at_init
        self.num_joint_columns = num_joint_columns
        self.num_drive_columns = num_drive_columns
        self.has_velocity_limits = has_velocity_limits

    def to_initialize_kwargs(self) -> dict:
        """Kwargs for
        :meth:`PhoenXWorld.initialize_joint_constraints`."""
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
            "friction_coefficient": self.friction_coefficient,
            "friction_slip_scale": self.friction_slip_scale,
            "d6_limit_axis0": self.d6_limit_axis0,
            "d6_limit_axis1": self.d6_limit_axis1,
            "d6_limit_axis2": self.d6_limit_axis2,
            "d6_limit_lower": self.d6_limit_lower,
            "d6_limit_upper": self.d6_limit_upper,
            "d6_limit_count": self.d6_limit_count,
            "velocity_limit": self.velocity_limit,
        }


def _newton_target_mode_to_joint_drive_mode(target_mode: int, stiffness: float, damping: float) -> int:
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


def build_joint_init_arrays(
    model: newton.Model,
    device: wp.context.Devicelike | None = None,
    *,
    joint_friction_model: Literal["hard", "mujoco"] = "hard",
    reduced_articulations: bool = False,
) -> JointInitArrays:
    """Convert ``model``'s joints to joint constraint init arrays on ``device``.

    Args:
        model: Newton model to convert.
        device: Device for the generated Warp arrays.
        joint_friction_model: ``"hard"`` keeps PhoenX Coulomb friction;
            ``"mujoco"`` maps MuJoCo solref/solimp friction metadata.
        reduced_articulations: Whether tree joints are owned by the reduced
            articulation solver instead of maximal-coordinate joint constraint columns.

    Raises:
        NotImplementedError: If a non-reduced D6 configuration cannot be
            reduced to the supported joint constraint mode set.
    """
    if device is None:
        device = model.device
    if joint_friction_model not in ("hard", "mujoco"):
        raise ValueError('joint_friction_model must be "hard" or "mujoco"')

    n_joints = int(model.joint_count)
    if n_joints == 0:
        empty_i = wp.zeros(0, dtype=wp.int32, device=device)
        empty_v = wp.zeros(0, dtype=wp.vec3f, device=device)
        empty_f = wp.zeros(0, dtype=wp.float32, device=device)
        joint_idx_to_cid = wp.zeros(0, dtype=wp.int32, device=device)
        joint_idx_to_dof_start = wp.zeros(0, dtype=wp.int32, device=device)
        return JointInitArrays(
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
            velocity_limit=empty_f,
            max_force_drive=empty_f,
            stiffness_drive=empty_f,
            damping_drive=empty_f,
            min_value=empty_f,
            max_value=empty_f,
            hertz_limit=empty_f,
            damping_ratio_limit=empty_f,
            stiffness_limit=empty_f,
            damping_limit=empty_f,
            friction_coefficient=empty_f,
            friction_slip_scale=empty_f,
            d6_limit_axis0=empty_v,
            d6_limit_axis1=empty_v,
            d6_limit_axis2=empty_v,
            d6_limit_lower=empty_v,
            d6_limit_upper=empty_v,
            d6_limit_count=empty_i,
            joint_idx_to_cid=joint_idx_to_cid,
            joint_idx_to_dof_start=joint_idx_to_dof_start,
            joint_q_at_init=empty_f,
            drive_cid=empty_i,
            drive_dof_start=empty_i,
            drive_target_q_index=empty_i,
            drive_q_at_init=empty_f,
            num_joint_columns=0,
            num_drive_columns=0,
            has_velocity_limits=False,
        )

    # ---- Pull every relevant joint array back to host ----------------
    joint_type = model.joint_type.numpy()
    joint_articulation = model.joint_articulation.numpy() if reduced_articulations else None
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_X_p = model.joint_X_p.numpy()  # (N, 7) float32
    joint_X_c = model.joint_X_c.numpy()  # (N, 7) float32 -- child attachment
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_target_q_start = model.joint_target_q_start.numpy()
    joint_axis = model.joint_axis.numpy() if model.joint_axis is not None else np.zeros((0, 3), dtype=np.float32)
    joint_dof_dim = (
        model.joint_dof_dim.numpy() if model.joint_dof_dim is not None else np.zeros((n_joints, 2), dtype=np.int32)
    )
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
    velocity_limit = _pull_dof_f(model.joint_velocity_limit)
    target_ke = _pull_dof_f(model.joint_target_ke)
    target_kd = _pull_dof_f(model.joint_target_kd)
    joint_friction = _pull_dof_f(model.joint_friction)
    mujoco_attrs = getattr(model, "mujoco", None) if joint_friction_model == "mujoco" else None
    friction_solref = _pull_dof_f(getattr(mujoco_attrs, "solreffriction", None)) if mujoco_attrs is not None else None
    friction_solimp = _pull_dof_f(getattr(mujoco_attrs, "solimpfriction", None)) if mujoco_attrs is not None else None
    effort_limit = _pull_dof_f(model.joint_effort_limit)
    limit_lower = _pull_dof_f(model.joint_limit_lower)
    limit_upper = _pull_dof_f(model.joint_limit_upper)
    joint_enabled = model.joint_enabled.numpy() if model.joint_enabled is not None else np.ones(n_joints, dtype=bool)

    joint_world_xform = np.asarray(joint_X_p, dtype=np.float32).copy()
    parented = joint_parent >= 0
    joint_world_xform[parented] = _transform_multiply_batch(
        body_q[joint_parent[parented]],
        joint_X_p[parented],
    )

    # ---- Walk joints --------------------------------------------------
    descriptors: list[dict] = []
    joint_idx_to_cid_np = np.full(n_joints, -1, dtype=np.int32)
    joint_idx_to_dof_start_np = np.full(n_joints, -1, dtype=np.int32)
    joint_idx_to_target_q_index_np = np.full(n_joints, -1, dtype=np.int32)

    for j in range(n_joints):
        jtype = newton.JointType(int(joint_type[j]))

        # Reduced coordinates own articulation-tree joints directly; only
        # out-of-tree closure rows need maximal-coordinate joint constraint columns.
        if reduced_articulations and int(joint_articulation[j]) >= 0:
            continue

        # FREE and disabled joints don't get a constraint column.
        if not bool(joint_enabled[j]):
            continue
        if jtype is newton.JointType.FREE:
            continue
        parent_idx = int(joint_parent[j])
        child_idx = int(joint_child[j])
        # Shift Newton body indices to PhoenX slots (slot 0 = static world).
        phoenx_parent = 0 if parent_idx < 0 else parent_idx + 1
        phoenx_child = 0 if child_idx < 0 else child_idx + 1

        # Joint world transform at init: pose_p * joint_X_p[j].
        X_w_p = joint_world_xform[j]

        anchor1_world = _transform_translation(X_w_p)
        qd_start = int(joint_qd_start[j])
        effective_jtype = jtype
        effective_dof_offset = 0
        effective_qd = qd_start
        if jtype is newton.JointType.D6:
            n_lin = int(joint_dof_dim[j, 0])
            n_ang = int(joint_dof_dim[j, 1])
            locked_lin = [_is_locked_dof(limit_lower, limit_upper, qd_start + i) for i in range(n_lin)]
            locked_ang = [_is_locked_dof(limit_lower, limit_upper, qd_start + n_lin + i) for i in range(n_ang)]
            classified_tag, classified_offset = _classify_d6_legacy_mode(n_lin, n_ang, locked_lin, locked_ang)
            if classified_tag is None:
                if reduced_articulations and int(joint_articulation[j]) >= 0:
                    continue
                classified_tag = "GENERIC"
            if classified_tag == "BALL":
                effective_jtype = newton.JointType.BALL
            elif classified_tag == "FIXED":
                effective_jtype = newton.JointType.FIXED
            elif classified_tag == "REVOLUTE":
                effective_jtype = newton.JointType.REVOLUTE
            elif classified_tag == "PRISMATIC":
                effective_jtype = newton.JointType.PRISMATIC
            elif classified_tag == "UNIVERSAL":
                effective_jtype = newton.JointType.D6
            effective_dof_offset = classified_offset if classified_offset >= 0 else 0
            if classified_offset >= 0 and classified_tag in ("REVOLUTE", "PRISMATIC"):
                effective_qd = qd_start + classified_offset

        # FIXED/BALL have no 1-axis DoF; -1 lets the control kernel skip them.
        dof_start_for_control = (
            effective_qd if effective_jtype in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC) else -1
        )
        target_q_index_for_control = -1
        if dof_start_for_control >= 0:
            target_q_index_for_control = int(joint_target_q_start[j]) + effective_dof_offset
        joint_idx_to_dof_start_np[j] = dof_start_for_control
        joint_idx_to_target_q_index_np[j] = target_q_index_for_control

        d6_mode_tag: str | None = None
        d6_locked_axis_offset = -1
        if jtype is newton.JointType.D6:
            d6_mode_tag = classified_tag
            d6_locked_axis_offset = classified_offset

        # Per-mode anchor2 and drive/limit defaults.
        anchor2_world = anchor1_world.copy()
        drive_mode = int(DRIVE_MODE_OFF)
        target_val = 0.0
        target_vel_val = 0.0
        velocity_limit_val = 0.0
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
        friction_val = 0.0
        friction_slip_scale_val = -1.0
        d6_limit_axes = [np.zeros(3, dtype=np.float32) for _ in range(3)]
        d6_limit_lower = np.zeros(3, dtype=np.float32)
        d6_limit_upper = np.zeros(3, dtype=np.float32)
        d6_limit_count = 0

        d6_limit_kwargs = {
            "limit_lower": limit_lower,
            "limit_upper": limit_upper,
            "joint_axis": joint_axis,
            "joint_q_arr": joint_q_arr,
            "joint_q_start": joint_q_start,
            "joint_index": j,
            "joint_world_xform": X_w_p,
            "d6_limit_axes": d6_limit_axes,
            "d6_limit_lower": d6_limit_lower,
            "d6_limit_upper": d6_limit_upper,
        }

        if d6_mode_tag == "GENERIC":
            phoenx_mode = int(JOINT_MODE_GENERIC_D6)
            for dof in range(qd_start, qd_start + n_lin + n_ang):
                if _is_locked_dof(limit_lower, limit_upper, dof):
                    continue
                lo = float(limit_lower[dof]) if limit_lower is not None else -1.0e10
                hi = float(limit_upper[dof]) if limit_upper is not None else 1.0e10
                if lo > -1.0e5 or hi < 1.0e5:
                    raise NotImplementedError(
                        f"Generic D6 joint {j} has a finite free-axis limit; "
                        "generic D6 inequalities are not implemented yet."
                    )
                if joint_friction is not None and float(joint_friction[dof]) > 0.0:
                    raise NotImplementedError(
                        f"Generic D6 joint {j} has Coulomb friction; generic D6 inequalities are not implemented yet."
                    )
                if velocity_limit is not None:
                    raw_velocity_limit = float(velocity_limit[dof])
                    if np.isfinite(raw_velocity_limit) and 0.0 < raw_velocity_limit < 1.0e5:
                        raise NotImplementedError(
                            f"Generic D6 joint {j} has a velocity limit; "
                            "generic D6 inequalities are not implemented yet."
                        )
        elif effective_jtype is newton.JointType.DISTANCE:
            phoenx_mode = int(JOINT_MODE_DISTANCE)
            if child_idx >= 0:
                X_w_c = _transform_multiply(
                    np.asarray(body_q[child_idx], dtype=np.float32),
                    np.asarray(joint_X_c[j], dtype=np.float32),
                )
            else:
                X_w_c = np.asarray(joint_X_c[j], dtype=np.float32)
            anchor2_world = _transform_translation(X_w_c)

            # Newton uses negative distance bounds as independent disabled
            # sentinels. Distance itself is nonnegative, so an omitted lower
            # bound maps to zero and an omitted upper bound maps to infinity.
            distance_dof = qd_start
            lo = float(limit_lower[distance_dof]) if limit_lower is not None else -1.0
            hi = float(limit_upper[distance_dof]) if limit_upper is not None else -1.0
            if lo >= 0.0 or hi >= 0.0:
                min_val = max(0.0, lo)
                max_val = hi if hi >= 0.0 else 1.0e10
        elif effective_jtype is newton.JointType.BALL:
            phoenx_mode = int(JOINT_MODE_BALL_SOCKET)
            if d6_mode_tag == "BALL":
                for ai, locked in enumerate(locked_ang):
                    if not locked:
                        d6_limit_count = _append_d6_angular_limit(
                            qd_start + n_lin + ai,
                            n_lin + ai,
                            d6_limit_count=d6_limit_count,
                            **d6_limit_kwargs,
                        )
        elif effective_jtype is newton.JointType.CABLE:
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
            if _norm3_np(anchor2_world - anchor1_world) < 1e-6:
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
        elif d6_mode_tag in ("CARTESIAN_PLANE", "CARTESIAN"):
            phoenx_mode = (
                int(JOINT_MODE_CARTESIAN_PLANE) if d6_mode_tag == "CARTESIAN_PLANE" else int(JOINT_MODE_CARTESIAN)
            )
            lin_free = [i for i, locked in enumerate(locked_lin) if not locked]
            linear_axes = np.asarray([joint_axis[qd_start + i] for i in lin_free], dtype=np.float32)
            linear_rank = int(np.linalg.matrix_rank(linear_axes, tol=1.0e-6))
            if linear_rank != len(lin_free):
                raise NotImplementedError(f"Cartesian D6 joint {j} has linearly dependent translation axes.")
            for linear_index in lin_free:
                dof = qd_start + linear_index
                lower = float(limit_lower[dof]) if limit_lower is not None else -1.0e10
                upper = float(limit_upper[dof]) if limit_upper is not None else 1.0e10
                if lower > -5.0e9 or upper < 5.0e9:
                    raise NotImplementedError(
                        f"Cartesian D6 joint {j} has a finite linear limit; "
                        "Cartesian limit inequalities are not implemented yet."
                    )
                if joint_friction is not None and float(joint_friction[dof]) > 0.0:
                    raise NotImplementedError(
                        f"Cartesian D6 joint {j} has linear Coulomb friction; "
                        "Cartesian friction inequalities are not implemented yet."
                    )
        elif d6_mode_tag in ("CYLINDRICAL", "PLANAR"):
            phoenx_mode = int(JOINT_MODE_CYLINDRICAL) if d6_mode_tag == "CYLINDRICAL" else int(JOINT_MODE_PLANAR)
            lin_free = [i for i, locked in enumerate(locked_lin) if not locked]
            ang_free = [i for i, locked in enumerate(locked_ang) if not locked]
            angular_axis = np.asarray(joint_axis[qd_start + n_lin + ang_free[0]], dtype=np.float32)
            angular_length = _norm3_np(angular_axis)
            if angular_length <= 1.0e-12:
                raise NotImplementedError(f"D6 joint {j} has a zero-length free angular axis.")
            angular_axis /= angular_length

            if d6_mode_tag == "CYLINDRICAL":
                linear_axis = np.asarray(joint_axis[qd_start + lin_free[0]], dtype=np.float32)
                linear_length = _norm3_np(linear_axis)
                if linear_length <= 1.0e-12:
                    raise NotImplementedError(f"D6 joint {j} has a zero-length free linear axis.")
                linear_axis /= linear_length
                if abs(float(np.dot(linear_axis, angular_axis))) < 1.0 - 1.0e-4:
                    raise NotImplementedError(
                        f"D6 joint {j} has non-parallel free linear and angular axes; it is not a cylindrical joint."
                    )
                axis_local = linear_axis
            else:
                axis_local = angular_axis
                locked_linear = [i for i, locked in enumerate(locked_lin) if locked]
                if locked_linear:
                    linear_axis = np.asarray(joint_axis[qd_start + locked_linear[0]], dtype=np.float32)
                    linear_length = _norm3_np(linear_axis)
                    if linear_length <= 1.0e-12:
                        raise NotImplementedError(f"D6 joint {j} has a zero-length locked plane-normal axis.")
                    linear_axis /= linear_length
                    if abs(float(np.dot(linear_axis, angular_axis))) < 1.0 - 1.0e-4:
                        raise NotImplementedError(
                            f"D6 joint {j} has non-parallel locked-linear and free-angular axes; "
                            "it is not a planar joint."
                        )
                    axis_local = linear_axis
                else:
                    for linear_index in lin_free:
                        linear_axis = np.asarray(joint_axis[qd_start + linear_index], dtype=np.float32)
                        linear_length = _norm3_np(linear_axis)
                        if (
                            linear_length <= 1.0e-12
                            or abs(float(np.dot(linear_axis / linear_length, axis_local))) > 1.0e-4
                        ):
                            raise NotImplementedError(
                                f"D6 joint {j} has an in-plane axis that is not perpendicular to its normal."
                            )

            axis_world = _quat_rotate_np(X_w_p[3:], axis_local)
            anchor2_world = anchor1_world + axis_world
            for ai in ang_free:
                d6_limit_count = _append_d6_angular_limit(
                    qd_start + n_lin + ai,
                    n_lin + ai,
                    d6_limit_count=d6_limit_count,
                    **d6_limit_kwargs,
                )
        elif d6_mode_tag == "UNIVERSAL":
            phoenx_mode = int(JOINT_MODE_UNIVERSAL)
            if d6_locked_axis_offset >= 0:
                locked_qd = qd_start + d6_locked_axis_offset
                axis_local = (
                    np.asarray(joint_axis[locked_qd], dtype=np.float32)
                    if len(joint_axis) and locked_qd < len(joint_axis)
                    else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                )
            else:
                axis_a = np.asarray(joint_axis[qd_start], dtype=np.float32)
                axis_b = np.asarray(joint_axis[qd_start + 1], dtype=np.float32)
                axis_local = _cross3_np(axis_a, axis_b)
            axis_len = _norm3_np(axis_local)
            if axis_len <= 1.0e-12:
                raise NotImplementedError(
                    f"D6 joint {j} has two angular axes that cannot define a universal locked twist axis."
                )
            axis_world = _quat_rotate_np(X_w_p[3:], axis_local / axis_len)
            anchor2_world = anchor1_world + axis_world
            min_val = 0.0
            max_val = 0.0
            if n_ang == 2 and d6_locked_axis_offset < 0:
                for ai in range(2):
                    d6_limit_count = _append_d6_angular_limit(
                        qd_start + ai,
                        ai,
                        d6_limit_count=d6_limit_count,
                        **d6_limit_kwargs,
                    )
            else:
                for ai, locked in enumerate(locked_ang):
                    if not locked:
                        d6_limit_count = _append_d6_angular_limit(
                            qd_start + n_lin + ai,
                            n_lin + ai,
                            d6_limit_count=d6_limit_count,
                            **d6_limit_kwargs,
                        )
        elif effective_jtype is newton.JointType.FIXED:
            phoenx_mode = int(JOINT_MODE_FIXED)
            # Pick joint-frame X axis so the anchor-3 basis is well-defined.
            axis_world = _quat_rotate_np(X_w_p[3:], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            anchor2_world = anchor1_world + axis_world
        elif effective_jtype is newton.JointType.REVOLUTE or effective_jtype is newton.JointType.PRISMATIC:
            phoenx_mode = (
                int(JOINT_MODE_REVOLUTE) if effective_jtype is newton.JointType.REVOLUTE else int(JOINT_MODE_PRISMATIC)
            )
            axis_local = (
                np.asarray(joint_axis[effective_qd], dtype=np.float32)
                if len(joint_axis) and effective_qd < len(joint_axis)
                else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            )
            axis_len = _norm3_np(axis_local)
            if axis_len > 1e-12:
                axis_local = axis_local / axis_len
            else:
                axis_local = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            axis_world = _quat_rotate_np(X_w_p[3:], axis_local)
            anchor2_world = anchor1_world + axis_world

            # Drive / limit from per-DOF arrays (first DoF only for the
            # supported 1-DoF joints).
            if target_ke is not None:
                stiff_drive = float(target_ke[effective_qd])
            if target_kd is not None:
                damp_drive = float(target_kd[effective_qd])
            if target_pos is not None and target_q_index_for_control < len(target_pos):
                target_val = float(target_pos[target_q_index_for_control])
            if target_vel is not None:
                target_vel_val = float(target_vel[effective_qd])
            if velocity_limit is not None:
                raw_velocity_limit = float(velocity_limit[effective_qd])
                if np.isfinite(raw_velocity_limit) and 0.0 < raw_velocity_limit < 1.0e5:
                    velocity_limit_val = raw_velocity_limit
            if effort_limit is not None:
                # PhoenX reads 0 as "unlimited" for POSITION drives, so clamp inf/NaN to 0.
                raw = float(effort_limit[effective_qd])
                max_force = raw if np.isfinite(raw) else 0.0
            if target_mode is not None:
                drive_mode = _newton_target_mode_to_joint_drive_mode(
                    int(target_mode[effective_qd]), stiff_drive, damp_drive
                )
            # Limits are hard stops via DEFAULT_HERTZ_LIMIT (matches SolverXPBD's
            # rigid-limit contract; Newton's limit_ke/limit_kd are XPBD-only soft
            # penalties that don't map to PhoenX's absolute SI PD path). Users who
            # want soft PD limits should drive joint constraint init directly.
            if limit_lower is not None and limit_upper is not None:
                lo = float(limit_lower[effective_qd])
                hi = float(limit_upper[effective_qd])
                if lo <= hi:
                    min_val = lo
                    max_val = hi
            if joint_friction is not None and effective_qd < len(joint_friction):
                friction_val = float(joint_friction[effective_qd])
            if friction_solref is not None and friction_solimp is not None and effective_qd < len(friction_solref):
                friction_slip_scale_val = _friction_slip_scale_from_mujoco(
                    friction_solref[effective_qd], friction_solimp[effective_qd]
                )
        else:  # pragma: no cover -- defensive
            raise NotImplementedError(f"joint {j}: unhandled joint type {jtype}")

        # Init joint coord for this joint's first DOF. BALL/FIXED publish 0 to
        # keep the per-joint array length aligned with the joint constraint column array.
        q_start_idx = int(joint_q_start[j])
        q_coord_idx = q_start_idx + effective_dof_offset
        if (
            effective_jtype in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
            and len(joint_q_arr) > q_coord_idx
        ):
            init_q = float(joint_q_arr[q_coord_idx])
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
                "velocity_limit": velocity_limit_val,
                "max_force_drive": max_force,
                "stiffness_drive": stiff_drive,
                "damping_drive": damp_drive,
                "min_value": min_val,
                "max_value": max_val,
                "hertz_limit": hertz_limit_val,
                "damping_ratio_limit": damping_ratio_limit_val,
                "stiffness_limit": stiff_limit,
                "damping_limit": damp_limit,
                "friction_coefficient": friction_val,
                "friction_slip_scale": friction_slip_scale_val,
                "d6_limit_axis0": d6_limit_axes[0],
                "d6_limit_axis1": d6_limit_axes[1],
                "d6_limit_axis2": d6_limit_axes[2],
                "d6_limit_lower": d6_limit_lower,
                "d6_limit_upper": d6_limit_upper,
                "d6_limit_count": d6_limit_count,
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

    return JointInitArrays(
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
        velocity_limit=_stack_f("velocity_limit"),
        max_force_drive=_stack_f("max_force_drive"),
        stiffness_drive=_stack_f("stiffness_drive"),
        damping_drive=_stack_f("damping_drive"),
        min_value=_stack_f("min_value"),
        max_value=_stack_f("max_value"),
        hertz_limit=_stack_f("hertz_limit"),
        damping_ratio_limit=_stack_f("damping_ratio_limit"),
        stiffness_limit=_stack_f("stiffness_limit"),
        damping_limit=_stack_f("damping_limit"),
        friction_coefficient=_stack_f("friction_coefficient"),
        friction_slip_scale=_stack_f("friction_slip_scale"),
        d6_limit_axis0=_stack_v("d6_limit_axis0"),
        d6_limit_axis1=_stack_v("d6_limit_axis1"),
        d6_limit_axis2=_stack_v("d6_limit_axis2"),
        d6_limit_lower=_stack_v("d6_limit_lower"),
        d6_limit_upper=_stack_v("d6_limit_upper"),
        d6_limit_count=_stack_i("d6_limit_count"),
        joint_idx_to_cid=wp.array(joint_idx_to_cid_np, dtype=wp.int32, device=device),
        joint_idx_to_dof_start=wp.array(joint_idx_to_dof_start_np, dtype=wp.int32, device=device),
        joint_q_at_init=_stack_f("joint_q_at_init"),
        drive_cid=wp.array(drive_cid_np, dtype=wp.int32, device=device),
        drive_dof_start=wp.array(drive_dof_start_np, dtype=wp.int32, device=device),
        drive_target_q_index=wp.array(drive_target_q_index_np, dtype=wp.int32, device=device),
        drive_q_at_init=wp.array(drive_q_at_init_np, dtype=wp.float32, device=device),
        num_joint_columns=num_cols,
        num_drive_columns=int(drive_cid_np.size),
        has_velocity_limits=any(float(d["velocity_limit"]) > 0.0 for d in descriptors),
    )
