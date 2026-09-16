# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU final-state joint equation, world impulse, and saved trajectory audit."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation  # noqa: TID253


def audit(stem):
    state = np.load(stem.with_suffix(".native_state.npz"))
    history = np.load(stem.with_suffix(".npz"))
    p = state["body_position"].astype(float)
    twist = np.concatenate((state["body_velocity"], state["body_angular_velocity"]), axis=1).astype(float)
    j0 = state["direct_row_wrench0"][0].astype(float)
    j1 = state["direct_row_wrench1"][0].astype(float)
    rows = state["direct_row_local"]
    impulse = state["direct_accumulated_impulse"].astype(float)
    dynamic = state["direct_row_dynamic"]
    residual = j0[rows] @ twist[1] + j1[rows] @ twist[2]
    residual[dynamic] += (
        impulse[dynamic] / state["direct_dynamic_mass"][dynamic] - state["direct_velocity_reference"][dynamic]
    )
    linear = j0[:, :3] + j1[:, :3]
    angular = j0[:, 3:] + np.cross(p[1], j0[:, :3]) + j1[:, 3:] + np.cross(p[2], j1[:, :3])
    q = history["q_history"].astype(float)
    angles = np.rad2deg((Rotation.from_quat(q[:, 0, 3:]).inv() * Rotation.from_quat(q[:, 1, 3:])).as_rotvec()[:, 2])
    times = history["history_times"]
    start = int(np.argmin(abs(times - times[-1] / 2)))
    delta = q[-1, 0, :3] - q[start, 0, :3]
    return {
        "scope": "Saved final native joint rows and COM velocities; unbiased final-row residual, not a per-phase proof.",
        "final_joint_residual": residual.tolist(),
        "hard_residual_max": float(np.max(abs(residual[~dynamic]))),
        "drive_residual_max": float(np.max(abs(residual[dynamic]))),
        "per_unit_impulse_linear_defect": linear.tolist(),
        "per_unit_impulse_world_angular_defect_m": angular.tolist(),
        "accumulated_joint_linear_defect_Ns": (impulse @ linear[rows]).tolist(),
        "accumulated_joint_world_angular_defect_Nms": (impulse @ angular[rows]).tolist(),
        "final_hinge_angle_deg": float(angles[-1]),
        "window_start_s": float(times[start]),
        "window_end_s": float(times[-1]),
        "window_base_delta_m": delta.tolist(),
        "window_horizontal_speed_um_s": float(np.linalg.norm(delta[:2]) / (times[-1] - times[start]) * 1e6),
        "window_angle_change_deg": float(angles[-1] - angles[start]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", type=Path)
    args = parser.parse_args()
    result = audit(args.stem)
    args.stem.with_suffix(".physical_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
