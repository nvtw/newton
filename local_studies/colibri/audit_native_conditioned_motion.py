"""Read-only motion and native drive-constitutive audit for the two-body control."""

import argparse
import json
from pathlib import Path

import numpy as np


def relative(a, b):
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return np.r_[a[3] * b[:3] - b[3] * a[:3] - np.cross(a[:3], b[:3]), a[3] * b[3] + a[:3] @ b[:3]]


def world_relative(a, b):
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return np.r_[a[3] * b[:3] - b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] + a[:3] @ b[:3]]


def angles(q):
    x, y, z, w = q
    return np.degrees(
        [
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ]
    ).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    a = np.load(args.snapshot)
    q = a["q_history"].astype(float)
    times = a["history_times"]
    report = {"snapshot": str(args.snapshot), "windows": {}}
    for start in (5.0, 10.0, 20.0, 30.0, 50.0):
        if times[-1] <= start:
            continue
        i = int(np.argmin(abs(times - start)))
        if i == len(times) - 1:
            continue
        delta = q[-1, 0, :3] - q[i, 0, :3]
        rotation = relative(q[i, 0, 3:], q[-1, 0, 3:])
        report["windows"][str(start)] = {
            "duration_s": float(times[-1] - times[i]),
            "delta_m": delta.tolist(),
            "xy_um": float(np.linalg.norm(delta[:2]) * 1e6),
            "xy_um_s": float(np.linalg.norm(delta[:2]) * 1e6 / (times[-1] - times[i])),
            "body_local_roll_pitch_yaw_deg": angles(rotation),
            "world_roll_pitch_yaw_deg": angles(world_relative(q[i, 0, 3:], q[-1, 0, 3:])),
            "max_xy_excursion_um": float(np.max(np.linalg.norm(q[i:, 0, :2] - q[i, 0, :2], axis=1)) * 1e6),
        }
    report["final_base_world_roll_pitch_yaw_deg"] = angles(world_relative(a["initial_q"][0, 3:], q[-1, 0, 3:]))
    hinge = relative(q[-1, 0, 3:], q[-1, 1, 3:])
    report["final_hinge_deg"] = float(np.degrees(2 * np.arctan2(hinge[2], hinge[3])))
    state_path = args.snapshot.with_suffix(".native_state.npz")
    if state_path.exists():
        s = np.load(state_path)
        if "direct_row_dynamic" in s:
            velocity = np.c_[s["body_velocity"], s["body_angular_velocity"]].astype(float)
            rows = []
            for row in np.flatnonzero(s["direct_row_dynamic"]):
                joint = int(s["direct_row_joint"][row])
                assert joint == 1, "Audit expects the sole authored two-body hinge"
                structural = int(s["direct_joint_to_structural"][joint])
                local = int(s["direct_row_local"][row])
                jv = float(
                    s["direct_row_wrench0"][structural, local] @ velocity[1]
                    + s["direct_row_wrench1"][structural, local] @ velocity[2]
                )
                impulse = float(s["direct_accumulated_impulse"][row])
                mass = float(s["direct_dynamic_mass"][row])
                reference = float(s["direct_velocity_reference"][row])
                rows.append(
                    {
                        "row": int(row),
                        "jv_rad_s": jv,
                        "generalized_impulse_N_m_s": impulse,
                        "dynamic_mass_kg_m2": mass,
                        "reference_rad_s": reference,
                        "constitutive_residual_rad_s": jv + impulse / mass - reference,
                    }
                )
            report["final_drive_rows"] = rows
    output = args.snapshot.with_suffix(".motion.json")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
