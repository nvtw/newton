# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional per-axis state for maximal-coordinate D6 inequalities."""

from typing import Literal

import numpy as np
import warp as wp

from newton._src.sim import JointType, Model

D6_AXIS_COUNT = 6
D6_ROW_AXIS = 0
D6_ROW_DISTANCE = 1


@wp.struct
class D6JointData:
    """Compact common state for limited, frictional, or speed-capped D6 axes.

    Authored axes stay separate from bilateral factor rows because a drive and
    its unilateral bounds may coexist on the same degree of freedom.
    """

    enabled: wp.int32
    row_count: wp.array[wp.int32]
    row_axis: wp.array2d[wp.int32]
    row_kind: wp.array2d[wp.int32]
    linear_count: wp.array[wp.int32]
    angular_count: wp.array[wp.int32]
    axis: wp.array2d[wp.vec3f]
    joint_x_p: wp.array[wp.transform]
    joint_x_c: wp.array[wp.transform]
    lower: wp.array2d[wp.float32]
    upper: wp.array2d[wp.float32]
    velocity_limit: wp.array2d[wp.float32]
    friction: wp.array2d[wp.float32]
    friction_slip_scale: wp.array2d[wp.float32]
    unwrap_angle: wp.array2d[wp.int32]
    condense_translation: wp.array2d[wp.int32]
    revolution_counter: wp.array2d[wp.int32]
    previous_angle: wp.array2d[wp.float32]
    wrench0: wp.array2d[wp.spatial_vector]
    wrench1: wp.array2d[wp.spatial_vector]
    coordinate: wp.array2d[wp.float32]
    effective_mass_inverse: wp.array2d[wp.float32]
    lower_impulse: wp.array2d[wp.float32]
    upper_impulse: wp.array2d[wp.float32]
    friction_impulse: wp.array2d[wp.float32]


def friction_slip_scale_from_mujoco(solref: np.ndarray | None, solimp: np.ndarray | None) -> float:
    """Return MuJoCo friction slip scale from ``solreffriction/solimpfriction``.

    MuJoCo friction-loss rows use ``R = (1 - impedance) / impedance * dA``
    and ``B = 2 / (dmax * timeconst)`` for positive-format ``solref``.
    The row inverse effective mass is only available during device prepare,
    so this stores ``R / (B * dA)`` for conversion to slip velocity there.
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


def build_d6_inequality_data(
    model: Model,
    joint_idx_to_cid: np.ndarray,
    joint_friction_model: Literal["hard", "mujoco"] = "hard",
) -> tuple[D6JointData, np.ndarray]:
    """Pack active D6 inequalities without inflating every joint column."""
    if joint_friction_model not in ("hard", "mujoco"):
        raise ValueError('joint_friction_model must be "hard" or "mujoco"')
    cid_count = max(int(joint_idx_to_cid.max(initial=-1)) + 1, 1)
    counts = np.zeros(cid_count, dtype=np.int32)
    row_axis = np.full((cid_count, D6_AXIS_COUNT), -1, dtype=np.int32)
    row_kind = np.full((cid_count, D6_AXIS_COUNT), D6_ROW_AXIS, dtype=np.int32)
    linear_count = np.zeros(cid_count, dtype=np.int32)
    angular_count = np.zeros(cid_count, dtype=np.int32)
    axes = np.zeros((cid_count, D6_AXIS_COUNT, 3), dtype=np.float32)
    joint_x_p = np.zeros((cid_count, 7), dtype=np.float32)
    joint_x_c = np.zeros((cid_count, 7), dtype=np.float32)
    joint_x_p[:, 6] = 1.0
    joint_x_c[:, 6] = 1.0
    lower_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.float32)
    upper_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.float32)
    velocity_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.float32)
    friction_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.float32)
    friction_slip_rows = np.full((cid_count, D6_AXIS_COUNT), -1.0, dtype=np.float32)
    unwrap_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.int32)
    condense_translation_rows = np.zeros((cid_count, D6_AXIS_COUNT), dtype=np.int32)

    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32)
    qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int32)
    dof_dim = np.asarray(model.joint_dof_dim.numpy(), dtype=np.int32)
    model_axis = np.asarray(model.joint_axis.numpy(), dtype=np.float32)
    lower = np.asarray(model.joint_limit_lower.numpy(), dtype=np.float32)
    upper = np.asarray(model.joint_limit_upper.numpy(), dtype=np.float32)
    velocity = np.asarray(model.joint_velocity_limit.numpy(), dtype=np.float32)
    friction = np.asarray(model.joint_friction.numpy(), dtype=np.float32)
    friction_solref = None
    friction_solimp = None
    if joint_friction_model == "mujoco":
        mujoco = getattr(model, "mujoco", None)
        if mujoco is not None:
            solref = getattr(mujoco, "solreffriction", None)
            solimp = getattr(mujoco, "solimpfriction", None)
            friction_solref = None if solref is None else np.asarray(solref.numpy(), dtype=np.float32)
            friction_solimp = None if solimp is None else np.asarray(solimp.numpy(), dtype=np.float32)
    x_p = np.asarray(model.joint_X_p.numpy(), dtype=np.float32)
    x_c = np.asarray(model.joint_X_c.numpy(), dtype=np.float32)

    common_joint = np.isin(
        joint_type,
        (
            int(JointType.BALL),
            int(JointType.D6),
            int(JointType.PRISMATIC),
            int(JointType.REVOLUTE),
            int(JointType.DISTANCE),
        ),
    )
    for joint in np.flatnonzero(common_joint):
        cid = int(joint_idx_to_cid[joint])
        if cid < 0:
            continue
        start = int(qd_start[joint])
        n_linear = int(dof_dim[joint, 0])
        n_angular = int(dof_dim[joint, 1])
        linear_count[cid] = n_linear
        angular_count[cid] = n_angular
        joint_x_p[cid] = x_p[joint]
        joint_x_c[cid] = x_c[joint]
        total = min(n_linear + n_angular, D6_AXIS_COUNT)
        axes[cid, :total] = model_axis[start : start + total]
        if joint_type[joint] == int(JointType.DISTANCE):
            linear_count[cid] = 0
            angular_count[cid] = 0
            lo = float(lower[start])
            hi = float(upper[start])
            if lo >= 0.0 or hi >= 0.0:
                row_axis[cid, 0] = 0
                row_kind[cid, 0] = D6_ROW_DISTANCE
                lower_rows[cid, 0] = max(lo, 0.0)
                upper_rows[cid, 0] = hi if hi >= 0.0 else 1.0e10
                speed_limit = float(velocity[start])
                if np.isfinite(speed_limit) and 0.0 < speed_limit < 1.0e5:
                    velocity_rows[cid, 0] = speed_limit
                friction_rows[cid, 0] = max(float(friction[start]), 0.0)
                if friction_solref is not None and friction_solimp is not None:
                    friction_slip_rows[cid, 0] = friction_slip_scale_from_mujoco(
                        friction_solref[start], friction_solimp[start]
                    )
                counts[cid] = 1
            continue
        for local in range(total):
            dof = start + local
            if float(lower[dof]) > float(upper[dof]):
                continue
            finite_limit = float(lower[dof]) > -1.0e5 or float(upper[dof]) < 1.0e5
            speed_limit = float(velocity[dof]) if np.isfinite(velocity[dof]) and 0.0 < velocity[dof] < 1.0e5 else 0.0
            axis_friction = max(float(friction[dof]), 0.0)
            if not finite_limit and speed_limit == 0.0 and axis_friction == 0.0:
                continue
            row = int(counts[cid])
            row_axis[cid, row] = local
            lower_rows[cid, row] = lower[dof]
            upper_rows[cid, row] = upper[dof]
            velocity_rows[cid, row] = speed_limit
            friction_rows[cid, row] = axis_friction
            if friction_solref is not None and friction_solimp is not None:
                friction_slip_rows[cid, row] = friction_slip_scale_from_mujoco(
                    friction_solref[dof], friction_solimp[dof]
                )
            unwrap_rows[cid, row] = int(n_angular == 1 and local >= n_linear)
            condense_translation_rows[cid, row] = int(
                joint_type[joint] == int(JointType.REVOLUTE) and local >= n_linear
            )
            counts[cid] += 1

    device = model.device
    data = D6JointData()
    data.enabled = int(np.any(counts))
    data.row_count = wp.array(counts, dtype=wp.int32, device=device)
    data.row_axis = wp.array(row_axis, dtype=wp.int32, device=device)
    data.row_kind = wp.array(row_kind, dtype=wp.int32, device=device)
    data.linear_count = wp.array(linear_count, dtype=wp.int32, device=device)
    data.angular_count = wp.array(angular_count, dtype=wp.int32, device=device)
    data.axis = wp.array(axes, dtype=wp.vec3f, device=device)
    data.joint_x_p = wp.array(joint_x_p, dtype=wp.transform, device=device)
    data.joint_x_c = wp.array(joint_x_c, dtype=wp.transform, device=device)
    data.lower = wp.array(lower_rows, dtype=wp.float32, device=device)
    data.upper = wp.array(upper_rows, dtype=wp.float32, device=device)
    data.velocity_limit = wp.array(velocity_rows, dtype=wp.float32, device=device)
    data.friction = wp.array(friction_rows, dtype=wp.float32, device=device)
    data.friction_slip_scale = wp.array(friction_slip_rows, dtype=wp.float32, device=device)
    data.unwrap_angle = wp.array(unwrap_rows, dtype=wp.int32, device=device)
    data.condense_translation = wp.array(condense_translation_rows, dtype=wp.int32, device=device)
    data.revolution_counter = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.int32, device=device)
    data.previous_angle = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    data.wrench0 = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.spatial_vector, device=device)
    data.wrench1 = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.spatial_vector, device=device)
    data.coordinate = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    data.effective_mass_inverse = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    data.lower_impulse = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    data.upper_impulse = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    data.friction_impulse = wp.zeros((cid_count, D6_AXIS_COUNT), dtype=wp.float32, device=device)
    return data, counts
