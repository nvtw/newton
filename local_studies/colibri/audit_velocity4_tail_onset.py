"""Audit saved tail-failure onset without resimulating or changing physics."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.audit_assembly_momentum import rotate
from local_studies.colibri.joint_accuracy import measure
from newton.examples.kamino.example_kamino_colibri import JOINTS


def multiply(a, b):
    """Multiply XYZW quaternions without changing saved input poses."""
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])]


def main():
    """Measure final successful frames and the separately saved rejected pose."""
    prefix = "/tmp/colibri_native_velocity4_clean18000"
    d = np.load(prefix + ".npz")
    result = json.loads(Path(prefix + ".json").read_text())
    labels = d["labels"].astype(str)
    q = np.concatenate((d["q_history"][-60:], d["q"][None])).astype(float)
    qd = np.concatenate((d["qd_history"][-60:], d["qd"][None])).astype(float)
    times = np.r_[d["history_times"][-60:], result["sim_time"]]
    mass, inertia = d["body_mass"].astype(float), d["body_inertia"].astype(float)
    inverse = q[..., 3:].copy()
    inverse[..., :3] *= -1
    angular_local = rotate(inverse, qd[..., 3:])
    kinetic = 0.5 * (
        mass * np.sum(qd[..., :3] ** 2, axis=-1) + np.einsum("tbi,bij,tbj->tb", angular_local, inertia, angular_local)
    )
    kinetic[:, labels == "Flower"] = 0
    tail = np.array(["Tail" in label for label in labels])
    rows = []
    for index, (pose, velocity, time) in enumerate(zip(q, qd, times, strict=True)):
        accuracy = measure(pose, labels)
        selected = [r for r in accuracy["joints"] if "Tail" in r["joint"]]
        bodies = []
        for body in np.flatnonzero(tail):
            bodies.append(
                {
                    "body": labels[body],
                    "velocity": velocity[body, :3].tolist(),
                    "angular_velocity": velocity[body, 3:].tolist(),
                    "kinetic_energy_J": float(kinetic[index, body]),
                }
            )
        angles = {}
        for parent, child, kind, axis, fp, fc, _ in JOINTS:
            if kind != "revolute" or "Tail" not in parent + child:
                continue
            a = multiply(pose[np.where(labels == parent)[0][0], 3:], np.array(fp[3:]))
            b = multiply(pose[np.where(labels == child)[0][0], 3:], np.array(fc[3:]))
            a[:3] *= -1
            relative = multiply(a, b)
            angles[parent + "/" + child] = float(2 * np.arctan2(relative["XYZ".index(axis)], relative[3]))
        rows.append(
            {
                "time": float(time),
                "joints": selected,
                "relative_twist_angle_rad": angles,
                "bodies": bodies,
                "total_kinetic_energy_J": float(kinetic[index].sum()),
                "tail_kinetic_energy_J": float(kinetic[index, tail].sum()),
            }
        )
    jumps = []
    for body in range(len(labels)):
        jumps.append(
            {
                "body": labels[body],
                "delta_linear_speed_m_s": float(np.linalg.norm(qd[-1, body, :3] - qd[-2, body, :3])),
                "delta_angular_velocity_rad_s": float(np.linalg.norm(qd[-1, body, 3:] - qd[-2, body, 3:])),
                "kinetic_before_J": float(kinetic[-2, body]),
                "kinetic_after_J": float(kinetic[-1, body]),
            }
        )
    out = {
        "scope": "60Hz saved trajectory, not per120Hz contact or solver-phase trace; external work absent",
        "rows": rows,
        "final_velocity_jumps": sorted(jumps, key=lambda x: x["delta_angular_velocity_rad_s"], reverse=True),
    }
    Path("/tmp/colibri_velocity4_tail_onset.json").write_text(json.dumps(out, indent=2))
    for row in rows[-8:]:
        feather = next(r for r in row["joints"] if r["joint"] == "TailMount/Tail_Feather_A__2x__mirrored")
        print(row["time"], feather, "KE", row["total_kinetic_energy_J"], "tailKE", row["tail_kinetic_energy_J"])
    print(json.dumps(out["final_velocity_jumps"][:6], indent=2))


if __name__ == "__main__":
    main()
