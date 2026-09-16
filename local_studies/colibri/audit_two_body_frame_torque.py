"""Physical hinge impulse balance across every microstep of captured frame330."""

import json
from pathlib import Path

import numpy as np

a = np.load("/tmp/colibri_base_frame_totalnormal_frame330.npz")
assert a["last_substep_counter"][0] == 19799
dt = float(a["dt"][0])
records = []


def phase(name, i):
    return {k.split(".", 1)[1]: a[k][i] for k in a.files if k.startswith(name + ".")}


def velocity(s):
    v = np.c_[s["velocity"], s["angular_velocity"]].astype(float)
    for b in (1, 2):
        lo, hi = s["copy_section_end"][b - 1 : b + 1]
        if hi > lo:
            v[b] = np.r_[
                s["copy_velocity"][lo:hi].astype(float).mean(0), s["copy_angular_velocity"][lo:hi].astype(float).mean(0)
            ]
    return v


for i in range(60):
    for name in ("warm", "biased", "relax"):
        if name == "relax" and i not in (29, 59):
            continue
        before, s, after = [phase(name + "_" + suffix, i) for suffix in ("before", "solved", "averaged")]
        u, v = velocity(before), velocity(after)
        parts = {
            k: np.zeros((4, 6))
            for k in ("overlap_normal", "overlap_tangent", "spec_normal", "spec_tangent", "joint_hard", "joint_drive")
        }
        h = s["headers"].view(np.int32)
        for col in range(int(s["column_count"][0])):
            b0, b1 = h[1:3, col]
            for p in range(h[5, col], h[5, col] + h[6, col]):
                n, t = s["lambdas"][:3, p].astype(float), s["lambdas"][3:6, p].astype(float)
                axes = np.array([n, t, np.cross(n, t)])
                delta = s["impulses"][:, p].astype(float)
                if name != "warm":
                    delta -= before["impulses"][:, p]
                prefix = "spec" if s["derived"][3, p] > 0 else "overlap"
                for b, sign, r in ((b0, -1, s["derived"][9:12, p]), (b1, 1, s["derived"][12:15, p])):
                    j = sign * np.c_[axes, np.cross(r.astype(float), axes)]
                    parts[prefix + "_normal"][b] += j[0] * delta[0]
                    parts[prefix + "_tangent"][b] += j[1:].T @ delta[1:]
        rows = s["joint_row_indices"][0, : s["joint_row_count"][0]]
        structural = int(s["joint_structural_index"][0])
        drive = int(rows[s["joint_row_dynamic"][rows] > 0][0])
        for row in rows:
            local = int(s["joint_row_local"][row])
            delta = float(s["joint_accumulated"][row]) - float(before["joint_accumulated"][row])
            key = "joint_drive" if s["joint_row_dynamic"][row] else "joint_hard"
            for b, keyw in ((1, "joint_wrench0"), (2, "joint_wrench1")):
                parts[key][b] += s[keyw][structural, local].astype(float) * delta
        w = s["joint_wrench1"][structural].astype(float)
        local = int(s["joint_row_local"][drive])
        axis = w[local, 3:]
        pivot = s["position"][2] + np.array([w[1, 5], -w[0, 5], w[0, 4]])

        def torque(z, b=2, axis=axis, s=s, pivot=pivot):
            return float(axis @ (z[3:] + np.cross(s["position"][b] - pivot, z[:3])))

        actual = np.zeros((4, 6))
        for b in (1, 2):
            xx, yy, zz, xy, xz, yz = s["inverse_inertia"][b].astype(float)
            actual[b, :3] = (v[b, :3] - u[b, :3]) / s["inverse_mass"][b]
            actual[b, 3:] = np.linalg.solve([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], v[b, 3:] - u[b, 3:])
        bv = float(s["joint_wrench0"][structural, local] @ v[1] + w[local] @ v[2])
        dm = float(s["joint_dynamic_mass"][drive])
        target = float(s["joint_reference"][drive])
        lam = float(s["joint_accumulated"][drive])
        gravity = np.array([0.0, 0.0, -a["body_mass"][1] * dt, 0.0, 0.0, 0.0])
        records.append(
            {
                "slot": i,
                "phase": name,
                "torque_impulse_Nms": {k: torque(z[2]) for k, z in parts.items()},
                "actual_torque_impulse_Nms": torque(actual[2]),
                "gravity_torque_impulse_Nms": torque(gravity),
                "ledger_error": float(abs(actual[1:3] - sum(parts.values())[1:3]).max()),
                "drive_actual_rest_speed_expected_Nms": [lam, dm * target, dm * (target - bv)],
                "drive_residual_rad_s": bv + lam / dm - target,
            }
        )
assert max(r["ledger_error"] for r in records) < 1e-9
summary = {}
for name in ("warm", "biased", "relax"):
    subset = [r for r in records if r["phase"] == name]
    summary[name] = {
        "count": len(subset),
        "mean_torque_Nm": {k: sum(r["torque_impulse_Nms"][k] for r in subset) / (60 * dt) for k in parts},
        "actual_mean_torque_Nm": sum(r["actual_torque_impulse_Nms"] for r in subset) / (60 * dt),
        "max_ledger_error": max(r["ledger_error"] for r in subset),
        "drive_residual_range": [
            min(r["drive_residual_rad_s"] for r in subset),
            max(r["drive_residual_rad_s"] for r in subset),
        ],
    }
subset = [r for r in records if r["phase"] == "biased"]
summary["gravity_mean_torque_Nm"] = sum(r["gravity_torque_impulse_Nms"] for r in subset) / (60 * dt)
summary["mean_biased_actual_rest_speed_expected_drive_Nm"] = (
    np.mean([r["drive_actual_rest_speed_expected_Nms"] for r in subset], axis=0) / dt
).tolist()
report = {
    "scope": "All60microsteps; relax slots29/59 only. Actual physical copy means, full warm impulses, subsequent impulse increments. Per-phase instantaneous hinge pivot/axis; no claim of moving-origin angular-momentum transport closure.",
    "summary": summary,
    "records": records,
}
Path("/tmp/colibri_two_body_frame_torque.json").write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
