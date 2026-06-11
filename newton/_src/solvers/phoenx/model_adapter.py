# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Host-side conversion from Newton :class:`Model` joints to ADBS init arrays.

Mapping: REVOLUTE/PRISMATIC/BALL/FIXED/CABLE -> ADBS joint modes;
FREE -> no column; DISTANCE unsupported. D6 joints are auto-dispatched
to specialized modes when the per-DoF lock pattern matches (see
:func:`_classify_d6_pattern`); generic D6 configurations raise. PhoenX
slot 0 is the static world anchor, so Newton body ``i`` maps to PhoenX
slot ``i + 1`` and ``joint_parent == -1`` maps to slot 0.
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
    JOINT_MODE_CYLINDRICAL,
    JOINT_MODE_FIXED,
    JOINT_MODE_PLANAR,
    JOINT_MODE_PRISMATIC,
    JOINT_MODE_REVOLUTE,
    JOINT_MODE_UNIVERSAL,
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
        friction_coefficient: wp.array,
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
        self.friction_coefficient = friction_coefficient
        self.d6_limit_axis0 = d6_limit_axis0
        self.d6_limit_axis1 = d6_limit_axis1
        self.d6_limit_axis2 = d6_limit_axis2
        self.d6_limit_lower = d6_limit_lower
        self.d6_limit_upper = d6_limit_upper
        self.d6_limit_count = d6_limit_count
        self.joint_idx_to_cid = joint_idx_to_cid
        self.joint_idx_to_dof_start = joint_idx_to_dof_start
        #: Per-ADBS-column initial Newton joint coordinate. PhoenX measures
        #: displacement from init, so Newton's absolute target/limit values
        #: must be offset by this before being written into the ADBS column.
        self.joint_q_at_init = joint_q_at_init
        #: Compact per-drive lookup tables. Every entry is an active scalar
        #: REVOLUTE/PRISMATIC-like drive row, so the control writeback kernel
        #: can run one uniform thread per drive without per-joint skip branches.
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
            "friction_coefficient": self.friction_coefficient,
            "d6_limit_axis0": self.d6_limit_axis0,
            "d6_limit_axis1": self.d6_limit_axis1,
            "d6_limit_axis2": self.d6_limit_axis2,
            "d6_limit_lower": self.d6_limit_lower,
            "d6_limit_upper": self.d6_limit_upper,
            "d6_limit_count": self.d6_limit_count,
        }


def _classify_d6_pattern(
    n_lin: int,
    n_ang: int,
    locked_lin: list[bool],
    locked_ang: list[bool],
) -> tuple[str | None, int]:
    """Classify a D6 joint by per-DoF lock pattern, mapping to an existing
    PhoenX mode where possible. Returns ``(mode_tag, free_dof_offset)``:

    * ``mode_tag``: ``"REVOLUTE"`` / ``"PRISMATIC"`` / ``"BALL"`` / ``"FIXED"``
      if the pattern matches a specialized mode, a ``*_CANDIDATE`` tag if
      the caller still needs an axis-parallelism check, or ``None`` if the
      joint needs generic D6 support.
    * ``free_dof_offset``: index (within the joint's DoF range starting at
      ``qd_start``) of the single free DoF for REVOLUTE / PRISMATIC. Set to
      ``-1`` for BALL / FIXED (no single free DoF to drive).

    The "locked" condition matches Newton's convention: ``limit_lower >
    limit_upper`` is the sentinel for a permanently-zeroed DoF. Free or
    limited DoFs are both treated as "not locked" here; the existing modes
    handle limits via ``min_value`` / ``max_value`` downstream.
    """
    n_lin_locked = sum(locked_lin)
    n_ang_locked = sum(locked_ang)
    n_lin_free = n_lin - n_lin_locked
    n_ang_free = n_ang - n_ang_locked

    # FIXED: all DoFs locked (requires the full 6 axes; partial coverage
    # would leave unconstrained DoFs that FIXED doesn't model).
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 3 and n_ang_locked == 3:
        return ("FIXED", -1)

    # BALL: 3 linear locked, 3 angular free. Some MJCF-imported D6s omit
    # explicit linear lock axes; for angular-only D6s, absent linear axes
    # still imply locked translation.
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 3 and n_ang_free == 3:
        return ("BALL", -1)
    if n_lin == 0 and n_ang == 3 and n_ang_free == 3:
        return ("BALL", -1)

    # REVOLUTE: 3 linear locked, 2 angular locked, 1 angular free/limited.
    # The free DoF (qd offset == n_lin + index_of_unlocked_angular) becomes
    # the joint's drive / limit / friction axis.
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 3 and n_ang_locked == 2 and n_ang_free == 1:
        ang_free_idx = next(i for i, lk in enumerate(locked_ang) if not lk)
        return ("REVOLUTE", n_lin + ang_free_idx)

    # PRISMATIC: 2 linear locked + 1 linear free, 3 angular locked.
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 2 and n_lin_free == 1 and n_ang_locked == 3:
        lin_free_idx = next(i for i, lk in enumerate(locked_lin) if not lk)
        return ("PRISMATIC", lin_free_idx)

    # UNIVERSAL: 3 linear locked + 1 angular locked + 2 angular free.
    # The single locked angular DoF (the "no-twist" axis) becomes the
    # constraint row; the two perpendicular angular DoFs are
    # unconstrained. ``free_dof_offset`` returns the locked-axis qd
    # offset so the adapter can pull that DoF's axis vector.
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 3 and n_ang_locked == 1 and n_ang_free == 2:
        ang_locked_idx = next(i for i, lk in enumerate(locked_ang) if lk)
        return ("UNIVERSAL", n_lin + ang_locked_idx)
    if n_lin == 0 and n_ang == 2 and n_ang_free == 2:
        # MJCF often represents a universal joint as two hinge axes on
        # one body. Translation is implicitly locked; the missing twist
        # axis is derived from the two free angular axes by the caller.
        return ("UNIVERSAL", -1)

    # CYLINDRICAL candidate: 2 linear locked + 1 linear free + 2 angular
    # locked + 1 angular free. The free axes must be parallel for this to
    # be a true cylindrical joint -- that parallelism check is done by
    # the caller (it needs ``joint_axis`` array access). ``free_dof_offset``
    # returns the linear-free qd offset; the angular-free offset is
    # implicit (the single unlocked angular slot).
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 2 and n_lin_free == 1 and n_ang_locked == 2 and n_ang_free == 1:
        lin_free_idx = next(i for i, lk in enumerate(locked_lin) if not lk)
        return ("CYLINDRICAL_CANDIDATE", lin_free_idx)

    # PLANAR candidate: 1 linear locked + 2 linear free + 2 angular
    # locked + 1 angular free. The locked linear axis must be PARALLEL
    # to the free angular axis (both = plane normal). Parallelism check
    # is done by the caller. ``free_dof_offset`` returns the locked-lin
    # qd offset (= the plane normal axis qd).
    if n_lin == 3 and n_ang == 3 and n_lin_locked == 1 and n_lin_free == 2 and n_ang_locked == 2 and n_ang_free == 1:
        lin_locked_idx = next(i for i, lk in enumerate(locked_lin) if lk)
        return ("PLANAR_CANDIDATE", lin_locked_idx)

    return (None, -1)


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
        NotImplementedError: If any joint is DISTANCE or an unsupported
            D6 configuration.
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
            friction_coefficient=empty_f,
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
    # joint_dof_dim shape [joint_count, 2]: (n_linear, n_angular) per joint.
    # Used by D6 dispatch to count locked / free DOFs per axis kind.
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
    target_ke = _pull_dof_f(model.joint_target_ke)
    target_kd = _pull_dof_f(model.joint_target_kd)
    joint_armature = _pull_dof_f(model.joint_armature)
    joint_damping = _pull_dof_f(model.joint_damping) if hasattr(model, "joint_damping") else None
    joint_friction = _pull_dof_f(model.joint_friction) if hasattr(model, "joint_friction") else None
    joint_gear = _pull_dof_f(model.joint_gear) if hasattr(model, "joint_gear") else None
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
        if jtype is newton.JointType.DISTANCE:
            raise NotImplementedError(
                f"PhoenX does not support JointType.{jtype.name} (joint {j}). "
                "Supported: REVOLUTE, PRISMATIC, BALL, FIXED, CABLE, D6, FREE (no column)."
            )

        # D6 auto-dispatch: route supported lock patterns to specialized
        # PhoenX modes; reject generic D6.
        d6_mode_tag: str | None = None
        d6_free_dof_offset = -1
        if jtype is newton.JointType.D6:
            n_lin = int(joint_dof_dim[j, 0])
            n_ang = int(joint_dof_dim[j, 1])
            qd_start_d6 = int(joint_qd_start[j])
            locked_lin: list[bool] = []
            locked_ang: list[bool] = []
            for dof_offset in range(n_lin + n_ang):
                qd = qd_start_d6 + dof_offset
                lo = float(limit_lower[qd]) if (limit_lower is not None and qd < len(limit_lower)) else 0.0
                hi = float(limit_upper[qd]) if (limit_upper is not None and qd < len(limit_upper)) else 0.0
                is_locked = lo > hi
                if dof_offset < n_lin:
                    locked_lin.append(is_locked)
                else:
                    locked_ang.append(is_locked)

            d6_mode_tag, d6_free_dof_offset = _classify_d6_pattern(n_lin, n_ang, locked_lin, locked_ang)

            # CYLINDRICAL is a candidate after the lock-count test; the
            # true cylindrical requires the free linear axis to be
            # parallel to the free angular axis (so the two free DoFs
            # share a single ``n_hat``). Otherwise it's a more general
            # D6 we don't yet support.
            if d6_mode_tag == "CYLINDRICAL_CANDIDATE":
                lin_free_idx = d6_free_dof_offset
                ang_free_idx = next(i for i, lk in enumerate(locked_ang) if not lk)
                lin_axis = np.asarray(joint_axis[qd_start_d6 + lin_free_idx], dtype=np.float64)
                ang_axis = np.asarray(joint_axis[qd_start_d6 + n_lin + ang_free_idx], dtype=np.float64)
                lin_norm = float(np.linalg.norm(lin_axis))
                ang_norm = float(np.linalg.norm(ang_axis))
                if lin_norm > 1e-12 and ang_norm > 1e-12:
                    cosine = abs(float(np.dot(lin_axis / lin_norm, ang_axis / ang_norm)))
                else:
                    cosine = 0.0
                if cosine > 0.999:  # within ~2.5 degrees of parallel
                    d6_mode_tag = "CYLINDRICAL"
                else:
                    d6_mode_tag = None  # fall through to the unsupported branch

            if d6_mode_tag == "PLANAR_CANDIDATE":
                lin_locked_idx = d6_free_dof_offset
                ang_free_idx = next(i for i, lk in enumerate(locked_ang) if not lk)
                lin_axis = np.asarray(joint_axis[qd_start_d6 + lin_locked_idx], dtype=np.float64)
                ang_axis = np.asarray(joint_axis[qd_start_d6 + n_lin + ang_free_idx], dtype=np.float64)
                lin_norm = float(np.linalg.norm(lin_axis))
                ang_norm = float(np.linalg.norm(ang_axis))
                if lin_norm > 1e-12 and ang_norm > 1e-12:
                    cosine = abs(float(np.dot(lin_axis / lin_norm, ang_axis / ang_norm)))
                else:
                    cosine = 0.0
                if cosine > 0.999:
                    d6_mode_tag = "PLANAR"
                else:
                    d6_mode_tag = None

            if d6_mode_tag is None:
                raise NotImplementedError(
                    f"D6 joint {j} has a configuration that PhoenX cannot yet auto-dispatch "
                    f"({n_lin} linear axes / {n_ang} angular axes; "
                    f"locked={tuple(locked_lin)} linear, {tuple(locked_ang)} angular). "
                    "Currently supported patterns: FIXED (all locked), BALL (3 lin locked + 3 ang free), "
                    "REVOLUTE (3 lin locked + 2 ang locked + 1 ang free), "
                    "PRISMATIC (2 lin locked + 1 lin free + 3 ang locked), "
                    "UNIVERSAL (3 lin locked + 1 ang locked + 2 ang free), "
                    "CYLINDRICAL (2 lin locked + 1 lin free + 2 ang locked + 1 ang free, axes parallel), "
                    "PLANAR (1 lin locked + 2 lin free + 2 ang locked + 1 ang free, locked-lin and free-ang axes parallel). "
                    "Generic D6 is tracked as Phase 3 work."
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
        # For D6 dispatched to REVOLUTE/PRISMATIC, the "effective" DoF is
        # the single free axis (located via d6_free_dof_offset), not the
        # joint's first DoF. All per-DoF lookups (axis, gains, limits, etc.)
        # use this offset. Native single-DoF joints set offset == qd_start.
        effective_qd = qd_start
        if jtype is newton.JointType.D6 and d6_mode_tag in ("REVOLUTE", "PRISMATIC"):
            effective_qd = qd_start + d6_free_dof_offset

        # FIXED/BALL have no 1-axis DoF; -1 lets the control kernel skip them.
        is_axis_joint = jtype in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC) or (
            jtype is newton.JointType.D6 and d6_mode_tag in ("REVOLUTE", "PRISMATIC")
        )
        dof_start_for_control = effective_qd if is_axis_joint else -1
        target_q_index_for_control = -1
        if is_axis_joint:
            target_q_index_for_control = int(joint_target_q_start[j]) + (effective_qd - qd_start)
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
        # Coulomb friction on the axial DoF (revolute / prismatic only).
        friction_val = 0.0
        d6_limit_axes = [np.zeros(3, dtype=np.float32) for _ in range(3)]
        d6_limit_lower = np.zeros(3, dtype=np.float32)
        d6_limit_upper = np.zeros(3, dtype=np.float32)
        d6_limit_count = 0

        def _max_joint_damping(dof_indices):
            if joint_damping is None:
                return 0.0
            damping = 0.0
            for qd in dof_indices:
                if 0 <= qd < len(joint_damping):
                    raw = float(joint_damping[qd])
                    if raw > damping and np.isfinite(raw):
                        damping = raw
            return damping

        def _append_d6_angular_limit(
            qd: int,
            coord_offset: int,
            *,
            x_w_p=X_w_p,
            joint_index=j,
            axes=d6_limit_axes,
            lower_out=d6_limit_lower,
            upper_out=d6_limit_upper,
        ) -> None:
            nonlocal d6_limit_count
            if d6_limit_count >= 3 or limit_lower is None or limit_upper is None:
                return
            if qd < 0 or qd >= len(limit_lower) or qd >= len(limit_upper):
                return
            lo = float(limit_lower[qd])
            hi = float(limit_upper[qd])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
                return
            if lo <= -2.0 * np.pi and hi >= 2.0 * np.pi:
                return
            axis_local = (
                np.asarray(joint_axis[qd], dtype=np.float32)
                if len(joint_axis)
                else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            )
            axis_len = float(np.linalg.norm(axis_local))
            if axis_len <= 1.0e-12:
                return
            axis_world = _quat_rotate_np(x_w_p[3:], axis_local / axis_len)
            q_idx = int(joint_q_start[joint_index]) + coord_offset
            init_axis_q = float(joint_q_arr[q_idx]) if len(joint_q_arr) > q_idx else 0.0
            axes[d6_limit_count] = axis_world
            lower_out[d6_limit_count] = lo - init_axis_q
            upper_out[d6_limit_count] = hi - init_axis_q
            d6_limit_count += 1

        # D6 dispatched modes share the per-mode setup with their native
        # counterparts -- the existing branches now fire for both. Routing
        # is determined by ``d6_mode_tag`` for D6 joints, ``jtype`` otherwise.
        is_ball = jtype is newton.JointType.BALL or (jtype is newton.JointType.D6 and d6_mode_tag == "BALL")
        is_fixed = jtype is newton.JointType.FIXED or (jtype is newton.JointType.D6 and d6_mode_tag == "FIXED")
        is_universal = jtype is newton.JointType.D6 and d6_mode_tag == "UNIVERSAL"
        is_cylindrical = jtype is newton.JointType.D6 and d6_mode_tag == "CYLINDRICAL"
        is_planar = jtype is newton.JointType.D6 and d6_mode_tag == "PLANAR"
        is_revolute = jtype is newton.JointType.REVOLUTE or (jtype is newton.JointType.D6 and d6_mode_tag == "REVOLUTE")
        is_prismatic = jtype is newton.JointType.PRISMATIC or (
            jtype is newton.JointType.D6 and d6_mode_tag == "PRISMATIC"
        )

        if is_ball:
            phoenx_mode = int(JOINT_MODE_BALL_SOCKET)
            if jtype is newton.JointType.D6:
                damp_drive = _max_joint_damping(range(qd_start + n_lin, qd_start + n_lin + n_ang))
                for ai in range(n_ang):
                    if not locked_ang[ai]:
                        _append_d6_angular_limit(qd_start + n_lin + ai, n_lin + ai)
            else:
                n_ang_native = int(joint_dof_dim[j, 1]) if joint_dof_dim.shape[0] > j else 3
                damp_drive = _max_joint_damping(range(qd_start, qd_start + max(1, n_ang_native)))
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
        elif is_fixed:
            phoenx_mode = int(JOINT_MODE_FIXED)
            # Pick joint-frame X axis so the anchor-3 basis is well-defined.
            axis_world = _quat_rotate_np(X_w_p[3:], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            anchor2_world = anchor1_world + axis_world
        elif is_universal:
            # Universal: 3 lin locked + 1 ang locked + 2 ang free. The
            # locked angular axis defines the constraint; we set
            # ``anchor2_world = anchor1_world + locked_axis_world`` so
            # the init kernel snapshots ``axis_local1`` along the locked
            # direction (same convention as REVOLUTE). The axial row is
            # configured as a rigid Box2D limit (``min == max == 0``,
            # rigid ``hertz_limit``) so it always clamps the cumulative
            # twist back to zero.
            phoenx_mode = int(JOINT_MODE_UNIVERSAL)
            if d6_free_dof_offset >= 0:
                locked_qd = qd_start + d6_free_dof_offset  # offset points at the LOCKED axis
                axis_local = (
                    np.asarray(joint_axis[locked_qd], dtype=np.float32)
                    if len(joint_axis)
                    else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                )
            else:
                # Angular-only MJCF D6: derive the missing locked twist
                # axis from the two free angular axes.
                axis_a = np.asarray(joint_axis[qd_start], dtype=np.float32)
                axis_b = np.asarray(joint_axis[qd_start + 1], dtype=np.float32)
                axis_local = np.cross(axis_a, axis_b).astype(np.float32)
            axis_len = float(np.linalg.norm(axis_local))
            if axis_len > 1e-12:
                axis_local = axis_local / axis_len
            else:
                raise NotImplementedError(
                    f"D6 joint {j} has two angular axes that cannot define a universal locked twist axis."
                )
            axis_world = _quat_rotate_np(X_w_p[3:], axis_local)
            anchor2_world = anchor1_world + axis_world
            # Rigid hard lock at zero twist. The shared axial prepare /
            # iterate picks the Box2D path (because ``stiffness_limit ==
            # damping_limit == 0``) and engages the limit clamp on any
            # cumulative-angle drift.
            min_val = 0.0
            max_val = 0.0
            hertz_limit_val = float(DEFAULT_HERTZ_LIMIT)
            if n_ang == 2 and d6_free_dof_offset < 0:
                for ai in range(2):
                    _append_d6_angular_limit(qd_start + ai, ai)
                damp_drive = _max_joint_damping(range(qd_start, qd_start + 2))
            else:
                for ai, locked in enumerate(locked_ang):
                    if not locked:
                        _append_d6_angular_limit(qd_start + n_lin + ai, n_lin + ai)
                damp_drive = _max_joint_damping(
                    qd_start + n_lin + i for i, locked in enumerate(locked_ang) if not locked
                )
        elif is_cylindrical:
            # Cylindrical: 2 lin locked + 1 lin free + 2 ang locked + 1 ang
            # free, with the two free axes parallel (verified above). The
            # shared axis defines ``n_hat``; anchor2 = anchor1 + axis so
            # the prepare can recover it from ``axis_local1``. MVP is
            # kinematic-only -- all axial drive / limit / friction
            # parameters stay zero, the axial row short-circuits in the
            # iterate. Future work can promote one or both free DoFs to
            # driven by adding paired axial rows.
            phoenx_mode = int(JOINT_MODE_CYLINDRICAL)
            lin_free_qd = qd_start + d6_free_dof_offset
            axis_local = (
                np.asarray(joint_axis[lin_free_qd], dtype=np.float32)
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
        elif is_planar:
            # Planar: 1 lin locked + 2 lin free + 2 ang locked + 1 ang
            # free, with the locked-lin and free-ang axes parallel (the
            # plane normal). ``d6_free_dof_offset`` points at the locked
            # linear DoF; its axis is the plane normal.
            phoenx_mode = int(JOINT_MODE_PLANAR)
            lin_locked_qd = qd_start + d6_free_dof_offset
            axis_local = (
                np.asarray(joint_axis[lin_locked_qd], dtype=np.float32)
                if len(joint_axis)
                else np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            )
            axis_len = float(np.linalg.norm(axis_local))
            if axis_len > 1e-12:
                axis_local = axis_local / axis_len
            else:
                axis_local = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            axis_world = _quat_rotate_np(X_w_p[3:], axis_local)
            anchor2_world = anchor1_world + axis_world
        elif is_revolute or is_prismatic:
            phoenx_mode = int(JOINT_MODE_REVOLUTE) if is_revolute else int(JOINT_MODE_PRISMATIC)
            axis_local = (
                np.asarray(joint_axis[effective_qd], dtype=np.float32)
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

            # Drive / limit from per-DOF arrays at ``effective_qd`` (qd_start
            # for native single-DoF joints; qd_start + d6_free_dof_offset for
            # D6 dispatched to REVOLUTE / PRISMATIC).
            if target_ke is not None:
                stiff_drive = float(target_ke[effective_qd])
            if target_kd is not None:
                damp_drive = float(target_kd[effective_qd])
            if target_pos is not None and target_q_index_for_control < len(target_pos):
                target_val = float(target_pos[target_q_index_for_control])
            if target_vel is not None:
                target_vel_val = float(target_vel[effective_qd])
            # Gear-ratio scaling: motor-side effort and rotor inertia are
            # converted to joint-frame quantities via ``gear`` and
            # ``gear**2`` respectively. ``gear == 1`` (default) is a no-op
            # so models built before the gear field landed are unaffected.
            gear = 1.0
            if joint_gear is not None and effective_qd < len(joint_gear):
                raw_gear = float(joint_gear[effective_qd])
                # Defensive: clamp non-positive / non-finite to 1 (gear must
                # be a positive scalar; the adapter is not the place to
                # validate the URDF / MJCF importer's output).
                gear = raw_gear if (raw_gear > 0.0 and np.isfinite(raw_gear)) else 1.0
            if effort_limit is not None:
                # PhoenX reads 0 as "unlimited" for POSITION drives, so clamp inf/NaN to 0.
                raw = float(effort_limit[effective_qd])
                max_force = (gear * raw) if np.isfinite(raw) else 0.0
            if target_mode is not None:
                drive_mode = _newton_target_mode_to_adbs_drive_mode(
                    int(target_mode[effective_qd]), stiff_drive, damp_drive
                )
            if drive_mode == int(DRIVE_MODE_OFF) and joint_damping is not None and effective_qd < len(joint_damping):
                raw_damping = float(joint_damping[effective_qd])
                if raw_damping > 0.0 and np.isfinite(raw_damping):
                    damp_drive = raw_damping
                    target_vel_val = 0.0
                    drive_mode = int(DRIVE_MODE_VELOCITY)
            # Limits are hard stops via DEFAULT_HERTZ_LIMIT (matches SolverXPBD's
            # rigid-limit contract; Newton's limit_ke/limit_kd are XPBD-only soft
            # penalties that don't map to PhoenX's absolute SI PD path). Users who
            # want soft PD limits should drive ADBS init directly.
            if limit_lower is not None and limit_upper is not None:
                lo = float(limit_lower[effective_qd])
                hi = float(limit_upper[effective_qd])
                if lo <= hi:
                    min_val = lo
                    max_val = hi
            if joint_armature is not None and effective_qd < len(joint_armature):
                # Gear**2 scaling: motor rotor inertia reflected through a
                # gearbox of ratio ``r`` appears at the joint as ``r**2 * I_rotor``.
                # See JointDofConfig.gear_ratio for the convention.
                armature_val = gear * gear * float(joint_armature[effective_qd])
            if joint_friction is not None and effective_qd < len(joint_friction):
                # Newton allows negative / non-finite values in some edge cases.
                # Clamp to 0 ("disabled") so the iterate's short-circuit fires.
                raw_fric = float(joint_friction[effective_qd])
                # Friction (like effort) is on the motor side: a motor-side
                # Coulomb torque of ``μ_motor`` produces ``gear * μ_motor`` at
                # the joint.
                friction_val = (gear * raw_fric) if (raw_fric >= 0.0 and np.isfinite(raw_fric)) else 0.0
        else:  # pragma: no cover -- defensive
            raise NotImplementedError(f"joint {j}: unhandled joint type {jtype}")

        # Init joint coord for this joint's free DoF. BALL/FIXED publish 0
        # to keep the per-joint array length aligned with the ADBS column
        # array. For D6 dispatched to REVOLUTE/PRISMATIC, the free DoF lives
        # at qd_start + d6_free_dof_offset in joint_qd_*, which for
        # joint_q_* corresponds to q_start + (offset translated to coord
        # space). Newton's REVOLUTE/PRISMATIC use scalar coords so qd offset
        # == q offset; D6 only contains REVOLUTE/PRISMATIC-like DoFs (no
        # quaternion sub-coords) so the same equality holds.
        q_start_idx = int(joint_q_start[j])
        coord_offset = 0
        if jtype is newton.JointType.D6 and d6_mode_tag in ("REVOLUTE", "PRISMATIC"):
            coord_offset = d6_free_dof_offset
        if is_revolute or is_prismatic:
            q_idx = q_start_idx + coord_offset
            if len(joint_q_arr) > q_idx:
                init_q = float(joint_q_arr[q_idx])
            else:
                init_q = 0.0
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
                "friction_coefficient": friction_val,
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
        friction_coefficient=_stack_f("friction_coefficient"),
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
    )
